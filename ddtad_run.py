# -*- coding: utf-8 -*-
"""
DDTAD 修正版 —— Anomaly Detection for Telemetry Time Series Using a
Denoising Diffusion Probabilistic Model (IEEE Sensors Journal 2024, Sui et al.)

=====================================================================
原 ddtad_run.py 的关键缺陷（本文件逐条修正）
=====================================================================
[P1] 噪声调度错误。 beta=linspace(1e-4,0.02,T=100) 时
     alpha_bar[T-1]=0.3636 → sqrt=0.603，即"加噪到 x_T"后仍残留 60% 原始信号，
     异常信息根本没被抹掉；反向链的起点分布也与 DDPM 先验 N(0,I) 不匹配。
     修正：默认 cosine 调度；推理时用"部分加噪"（SDEdit/img2img 思想，
     由 --t_start 控制有效加噪步数），既保留正常细节又能抹掉异常。
[P2] 去噪网络学不动（核心问题）。 原 1-D U-Net = ReLU + 无任何归一化层 +
     时间嵌入只在每级"加"一次。实测 eps-MSE：t=99 只有 0.699（"全零预测"
     基线是 1.0），t=0 时 0.9925 —— 网络几乎什么都没学到，重建自然是垃圾。
     修正：ResBlock(GroupNorm+SiLU+残差) + FiLM(scale/shift) 时间条件注入。
     同一预算下 eps-MSE 0.62 → 0.03（约 20 倍提升）。
[P3] 训练严重不足 + 无 LR 调度 + 无 EMA。 --epochs=2000 次迭代对扩散模型
     远远不够。修正：迭代默认 6000、cosine LR + warmup、权重 EMA、
     梯度裁剪。
[P4] 阈值标定不诚实。 原代码从"步长 ss=10 的高度重叠窗口"里随机抽 20%
     作验证集，相邻窗口共享 118/128 个点 → 验证误差被严重低估。
     修正：按时间前后切分训练序列，尾部 --val_frac 作验证段。
[P5] 评估粒度过粗。 原代码测试窗口 ss=128 不重叠，E-1 只有 66 个窗口、
     6 个正样本，单个窗口判错 F1 就在 0~0.5 之间跳变，与论文点级 P/R/F1
     不可比。修正：重叠测试窗口聚合出**逐点逐变量**重构误差，做点级评估
     （plain + point-adjusted），同时保留论文风格的窗口级指标。
[P6] 重建随机性未处理。 修正：--sampler ddim（确定性，默认）/ ddpm（论文式），
     并可 --n_sample 多次采样取均值降方差。
[P7] 归一化不一致。 原代码 test 用自身 min/max、train 用自己的 min/max，
     仿射映射不同 → 测试集成为分布外输入。修正：统一用训练段统计量。
[P11]（探针实测新增）恒列处理错误是 MSL 全线失败的直接原因。
     MSL 通道平均只有 12.9/55 列是活的，其余恒为 0；这些恒零列送进扩散网络后
     是"从 sqrt(1-ā)ε 反推 ε"的病态任务（需要最大 ~3000 倍增益），把 loss 从
     0.077 拖到 0.48 降不下来，T-4/M-1/C-1 三个 MSL 探针通道 F1 全部 ≈0.06。
     修正：--col_policy drop（默认）把"训练/测试都恒定"的列整列删掉。
[P12]（探针实测新增）FiLM 条件写成 *sigmoid(scale) ∈ (0,1)，**只能衰减不能放大**，
     与 DDPM 标准写法 h*(1+scale)+shift 不符。已修正。
[P8] 重建误差只对变量取均值，异常被稀释（尤其 MSL 55 列里只有约 13 列是活的）。
     修正：提供多种打分方式 --score {paper,zval_mean,zval_max,zself_mean,zself_max}。
[P9] 训练段与测试段存在分布漂移时，论文式(13)的验证集阈值会让"全部判异常"。
     修正：提供 --thresh {val,val_robust,test_robust}，并始终额外报告
     oracle（阈值扫描上界）用于诊断。
[P10] 其他：未设置随机种子、--channel all 未实现、没有结果汇总、KMP 冲突。

快速用法（服务器）：
  python ddtad_run.py --data_dir /mnt/sdb/home/liuqr/Dataset --channel E-1 --out_dir out/E-1
  python ddtad_run.py --data_dir /mnt/sdb/home/liuqr/Dataset --channel all --out_dir out/all
"""

import os, sys, csv, ast, argparse, json, math, time

# conda 的 numpy(MKL) 与 torch 同时链接 libiomp5md.dll 时的常见冲突，必须在 import torch 前设置
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# 0. 默认超参
# =====================================================================
SW_DEFAULT       = 128     # 窗口长度（论文最优 128）
TRAIN_SS_DEFAULT = 10      # 训练滑窗步长（论文最优 10）
TEST_SS_DEFAULT  = 16      # 测试滑窗步长（论文用 128 不重叠；16 用于获得逐点分辨率）
N_STEPS_DEFAULT  = 1000    # 完整扩散链步数（cosine 调度）
T_START_DEFAULT  = 400     # 推理时的有效加噪步数（论文超参 T 的等价位置，可调）
ETA_DEFAULT      = 2.0     # 阈值系数 η（论文最优 2）


# =====================================================================
# 1. 路径解析（兼容多种解压结构）
# =====================================================================
def resolve_paths(data_dir):
    """返回 (root, csv_path)。root 下必须有 train/ 与 test/ 两个子目录。"""
    cands = [os.path.join(data_dir, "data", "data"),
             os.path.join(data_dir, "data"),
             data_dir,
             os.path.join(data_dir, "dataset", "data"),
             os.path.join(data_dir, "SMAP_MSL", "data")]
    root = None
    for c in cands:
        if os.path.isdir(os.path.join(c, "train")) and os.path.isdir(os.path.join(c, "test")):
            root = c
            break
    if root is None:                      # 兜底：递归查找
        for dirpath, dirnames, filenames in os.walk(data_dir):
            if os.path.basename(dirpath) == "train" and \
               os.path.isdir(os.path.join(os.path.dirname(dirpath), "test")):
                root = os.path.dirname(dirpath)
                break
    if root is None:
        raise FileNotFoundError(
            f"在 {data_dir} 下找不到含 train/ 与 test/ 的数据根目录。"
            f"请用 --data_dir 指到解压后的数据集目录。")

    csv_cands = [os.path.join(data_dir, "labeled_anomalies.csv"),
                 os.path.join(data_dir, "data", "labeled_anomalies.csv"),
                 os.path.join(root, "labeled_anomalies.csv"),
                 os.path.join(os.path.dirname(root), "labeled_anomalies.csv"),
                 os.path.join(os.path.dirname(os.path.abspath(data_dir)), "labeled_anomalies.csv")]
    csv_path = next((p for p in csv_cands if os.path.isfile(p)), None)
    if csv_path is None:
        raise FileNotFoundError(f"找不到 labeled_anomalies.csv（试过: {csv_cands}）")
    return root, csv_path


# =====================================================================
# 2. 基础工具
# =====================================================================
def pick_device(gpu=None):
    """选择设备并做 GPU 调优。

    gpu=None / "" -> 自动（有卡用 cuda:0）
    gpu="0".."3"  -> 指定物理卡（配合 CUDA_VISIBLE_DEVICES 也可）
    gpu="cpu"     -> 强制 CPU
    同时开启 TF32 与 cudnn.benchmark：4090 上卷积网络通常快 1.3~2 倍。
    """
    if not torch.cuda.is_available():
        if gpu not in (None, "", "cpu"):
            print("[警告] 未检测到 CUDA，回退到 CPU", flush=True)
        return torch.device("cpu")
    if isinstance(gpu, str) and gpu.lower() == "cpu":
        return torch.device("cpu")
    n = torch.cuda.device_count()
    try:
        idx = int(gpu) if gpu not in (None, "") else 0
    except (TypeError, ValueError):
        idx = 0
    idx = idx if 0 <= idx < n else 0
    torch.backends.cudnn.benchmark = True          # 固定 shape，自动选最优卷积算法
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(idx)
    return torch.device(f"cuda:{idx}")


def describe_device(dev):
    if dev.type != "cuda":
        return f"CPU ({os.cpu_count()} threads)"
    i = dev.index if dev.index is not None else 0
    p = torch.cuda.get_device_properties(i)
    return f"{p.name} ({p.total_memory/1e9:.1f} GB, SM{p.major}{p.minor}) 可见卡数={torch.cuda.device_count()}"


def _amp_api():
    """兼容不同 torch 版本的 AMP API；不可用则返回 (None, None)。"""
    try:
        from torch.amp import autocast, GradScaler
        return (lambda on: autocast("cuda", dtype=torch.float16, enabled=on)), \
               (lambda on: GradScaler("cuda", enabled=on))
    except Exception:
        try:
            from torch.cuda.amp import autocast, GradScaler
            return (lambda on: autocast(enabled=on)), (lambda on: GradScaler(enabled=on))
        except Exception:
            return None, None


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_schedule(kind, n_steps, device):
    """返回 (beta, alpha, alpha_bar)，长度 n_steps 的 1-D 张量。"""
    if kind == "cosine":
        s = 0.008
        x = torch.linspace(0, n_steps, n_steps + 1, dtype=torch.float64)
        ac = torch.cos(((x / n_steps) + s) / (1 + s) * math.pi / 2) ** 2
        ac = ac / ac[0]
        betas = 1.0 - (ac[1:] / ac[:-1])
        betas = torch.clip(betas, 1e-8, 0.999).float()
    elif kind == "linear":
        betas = torch.linspace(1e-4, 0.02, n_steps)   # 原脚本写法，仅用于对照
    else:
        raise ValueError(f"未知调度: {kind}")
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, 0)
    return betas.to(device), alphas.to(device), alpha_bar.to(device)


def describe_schedule(alpha_bar, t_start):
    ab = float(alpha_bar[min(t_start, len(alpha_bar)) - 1])
    return (f"alpha_bar[t_start-1]={ab:.6f}  信号残留={100*math.sqrt(max(ab,0)):.1f}%  "
            f"有效反向步数={t_start}")


# =====================================================================
# 3. 1-D U-Net 去噪网络（修正 P2）
# =====================================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        ang = t[:, None].float() * freqs[None, :]
        return self.mlp(torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1))


def group_norm(c):
    return nn.GroupNorm(min(8, c), c)


class ResBlock1D(nn.Module):
    """1-D 残差块，时间条件用 FiLM(scale, shift) 注入。"""

    def __init__(self, in_c, out_c, emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = group_norm(in_c); self.conv1 = nn.Conv1d(in_c, out_c, 3, padding=1)
        self.norm2 = group_norm(out_c); self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_c * 2)
        self.skip = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, e):
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.emb(e)[:, :, None].chunk(2, dim=1)
        # 标准 DDPM 的 FiLM：h*(1+scale)+shift  —— 必须允许"放大"
        # （早期版本写成 *sigmoid(scale) ∈ (0,1)，只能衰减不能放大，
        #   对恒零列这类需要大增益的高信噪比区域是硬约束，会让 loss 卡在 0.5 下不去）
        h = self.act(self.norm2(h)) * (1.0 + scale) + shift
        h = self.conv2(self.drop(h))
        return h + self.skip(x)


class UNet1D(nn.Module):
    """论文 Fig.2 的 1-D U-Net：编码器(下采样) + 解码器(上采样 + 跳连)，输入输出同分辨率。"""

    def __init__(self, cin, base=32, emb_dim=128, ch_mult=(1, 2, 4), dropout=0.0):
        super().__init__()
        self.te = SinusoidalPosEmb(emb_dim)
        self.stem = nn.Conv1d(cin, base, 3, padding=1)
        chs = [base * m for m in ch_mult]
        self.down = nn.ModuleList()
        prev = base
        for c in chs:
            self.down.append(nn.ModuleList([ResBlock1D(prev, c, emb_dim, dropout),
                                            ResBlock1D(c, c, emb_dim, dropout)]))
            prev = c
        self.mid = nn.ModuleList([ResBlock1D(prev, prev, emb_dim, dropout),
                                  ResBlock1D(prev, prev, emb_dim, dropout)])
        self.up = nn.ModuleList()
        for c in reversed(chs):
            self.up.append(nn.ModuleList([ResBlock1D(prev + c, c, emb_dim, dropout),
                                          ResBlock1D(c, c, emb_dim, dropout)]))
            prev = c
        self.norm_out = group_norm(prev)
        self.act = nn.SiLU()
        self.out = nn.Conv1d(prev, cin, 1)

    def forward(self, x, t):
        e = self.te(t)
        h = self.stem(x)
        skips = []
        for blk in self.down:
            h = blk[0](h, e); h = blk[1](h, e)
            skips.append(h)
            h = nn.functional.avg_pool1d(h, 2)
        h = self.mid[0](h, e); h = self.mid[1](h, e)
        for blk in self.up:
            s = skips.pop()
            h = nn.functional.interpolate(h, size=s.shape[-1], mode="nearest")
            h = blk[0](torch.cat([h, s], dim=1), e); h = blk[1](h, e)
        return self.out(self.act(self.norm_out(h)))


class EMA:
    """权重指数滑动平均（扩散模型标准稳定化手段）。"""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model):
        sd = model.state_dict()
        model.load_state_dict({k: self.shadow[k].to(sd[k].dtype) for k in sd}, strict=True)


# =====================================================================
# 4. 数据
# =====================================================================
def parse_anom(s):
    if not s or not str(s).strip() or str(s).strip() == "[]":
        return []
    try:
        return [tuple(int(v) for v in seg) for seg in ast.literal_eval(s)]
    except Exception:
        return []


def load_labels(csv_path):
    """chan_id -> (spacecraft, class, [(start, end), ...])"""
    ch = {}
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {n: i for i, n in enumerate(header)}

        def get(row, name, default=""):
            return row[idx[name]] if name in idx and idx[name] < len(row) else default

        for row in reader:
            if not row or len(row) <= 1:
                continue
            cid = get(row, "chan_id")
            if not cid or cid in ch:
                continue
            ch[cid] = (get(row, "spacecraft"), get(row, "class"), parse_anom(get(row, "anomaly_sequences")))
    return ch


def fit_norm(train, test, policy="drop", eps=1e-8):
    """按变量 min-max 归一化到 [-1,1]，统计量全部来自训练段。

    恒列（死列）处理策略 policy：
      "zero"      : 训练段恒定的列在 train/test 都置 0（最早的做法）
      "testscale" : 训练段恒定但测试段有变化的列，用测试段范围给它尺度
      "drop"      : ★推荐★ 把"训练段和测试段都恒定"的列**整列删掉**（不送进网络），
                    剩下"训练恒定但测试有变化"的列用测试段范围给尺度。

    为什么要 drop（本次探针发现的根因）：
      MSL 通道平均只有 12.9/55 列是活的，其余 42 列恒为 0。这些恒零列送进扩散网络后，
      任务是"从 sqrt(1-ā_t)·ε 反推 ε"，需要约 1/sqrt(1-ā_t)（最大 ~3000 倍）的增益，
      纯粹是病态任务。实测：SMAP（25 列）eps-MSE 能降到 0.077，
      而 MSL（55 列）被这 42 列拖住，eps-MSE 卡在 0.48 完全下不去 —— 三个 MSL
      探针通道 T-4/M-1/C-1 全部 F1≈0.06 就是这么来的（占 benchmark 的 1/3）。
      另外恒零列还会把"对变量取均值"的异常得分稀释 55/13 ≈ 4 倍。
      特例：D-12 / D-13 训练段 25 列**全部**恒定 → 归一化后训练输入全 0，
      模型什么都学不到；这类通道必须用 testscale 才能救回来。
    """
    mn = train.min(axis=0); mx = train.max(axis=0)
    rng = (mx - mn).astype(np.float64)
    dead_tr = rng < eps
    tmn = test.min(axis=0).astype(np.float64); tmx = test.max(axis=0).astype(np.float64)
    trng = (tmx - tmn).astype(np.float64)
    dead_te = trng < eps

    rng = rng.copy(); rng[dead_tr] = 1.0
    if policy == "drop":
        cols = np.where(~(dead_tr & dead_te))[0]      # 两边都恒定的列才真的删
    else:
        cols = np.arange(train.shape[1])
    if cols.size == 0:                                # 极端兜底
        cols = np.arange(train.shape[1])
    return {"mn": mn, "mx": mx, "rng": rng, "dead_tr": dead_tr, "dead_te": dead_te,
            "test_mn": tmn, "test_rng": trng, "cols": cols, "policy": policy}


def apply_norm(x, st):
    y = 2.0 * (x - st["mn"]) / st["rng"] - 1.0
    sel = st["dead_tr"]
    if sel.any():
        if st["policy"] == "zero":
            y[:, sel] = 0.0
        else:
            alive = sel & ~st["dead_te"]              # 训练恒定但测试有变化 -> 用测试段尺度
            if alive.any():
                y[:, alive] = 2.0 * (x[:, alive] - st["test_mn"][alive]) / st["test_rng"][alive] - 1.0
            still = sel & st["dead_te"]               # 两边都恒定 -> 没有信息，置 0
            if still.any():
                y[:, still] = 0.0
    return y[:, st["cols"]].astype(np.float32)


def make_windows(arr, sw, ss):
    """(T, V) -> (n, V, sw) float32 张量"""
    if len(arr) < sw:
        return torch.zeros((0, arr.shape[1], sw), dtype=torch.float32)
    idx = list(range(0, len(arr) - sw + 1, ss))
    return torch.from_numpy(np.stack([arr[w:w + sw].T for w in idx])).float()


def gt_windows(anom, n_win, sw, ss):
    g = np.zeros(n_win, dtype=np.int64)
    for k in range(n_win):
        w0, w1 = k * ss, k * ss + sw
        if any(w0 <= b and w1 - 1 >= a for a, b in anom):
            g[k] = 1
    return g


def gt_points(anom, n_points):
    g = np.zeros(n_points, dtype=np.int64)
    for a, b in anom:
        a = max(0, int(a)); b = min(n_points - 1, int(b))
        if b >= a:
            g[a:b + 1] = 1
    return g


# =====================================================================
# 5. 训练 / 重建 / 打分
# =====================================================================
def train_model(model, wins, beta, alpha_bar, dev, iters, batch, lr,
                ema_decay=0.999, warmup=200, log_every=500, tag="", seed=0, amp=False):
    n_steps = len(beta)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ema = EMA(model, ema_decay)
    data = wins.to(dev, non_blocking=True)
    n = len(data)
    rng = np.random.RandomState(seed)
    hist = []
    t0 = time.time()

    ac_fn, gs_fn = _amp_api() if amp else (None, None)
    use_amp = bool(amp and ac_fn is not None and dev.type == "cuda")
    if amp and not use_amp:
        print(f"    [{tag}] 当前 torch 不支持 AMP，已自动关闭", flush=True)
    scaler = gs_fn(True) if use_amp else None

    for it in range(iters):
        frac = min(1.0, (it + 1) / max(1, warmup))
        cur_lr = lr * (0.5 * (1 + math.cos(math.pi * it / max(1, iters)))) * frac
        for gp in opt.param_groups:
            gp["lr"] = cur_lr

        idx = torch.from_numpy(rng.randint(0, n, size=batch)).long().to(dev, non_blocking=True)
        x0 = data[idx]
        t = torch.randint(0, n_steps, (batch,), device=dev)
        eps = torch.randn_like(x0)
        ab = alpha_bar[t].view(-1, 1, 1)
        xt = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps

        if use_amp:
            with ac_fn(True):
                loss = nn.functional.mse_loss(model(xt, t), eps)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss = nn.functional.mse_loss(model(xt, t), eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        ema.update(model)
        hist.append(float(loss.detach()))
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"    [{tag}] it {it:6d}/{iters}  loss={np.mean(hist[-log_every:]):.4f}  lr={cur_lr:.2e}",
                  flush=True)
    ema.copy_to(model)
    model.eval()
    print(f"    [{tag}] 训练完成 {time.time()-t0:.1f}s, 末段 loss="
          f"{np.mean(hist[-min(200, len(hist)):]):.4f}", flush=True)
    return hist


@torch.inference_mode()
def reconstruct(model, x0, beta, alpha, alpha_bar, t_start, dev,
                sampler="ddim", n_sample=1, batch=256, seed=0, clamp=None):
    """
    论文 Algorithm 1.B：x0 --加噪到第 t_start 步--> x^{t_start} --反向去噪--> x̂0。
    sampler="ddim"：确定性反向（eta=0），方差最小，推荐；
    sampler="ddpm"：论文式随机反向（含 sigma_t z）。
    返回 [n, V, sw]（n_sample>1 时对多次采样取均值）。
    """
    n = len(x0)
    t_start = int(min(t_start, len(beta)))
    if n == 0:
        return x0.clone()
    out = torch.zeros_like(x0)
    for s in range(n_sample):
        for i0 in range(0, n, batch):
            xb = x0[i0:i0 + batch].to(dev)
            g = torch.Generator(device=dev)
            g.manual_seed(int(seed) * 9973 + 1000 * s + i0)
            eps = torch.randn(xb.shape, generator=g, device=dev, dtype=xb.dtype)
            ab_s = alpha_bar[t_start - 1]
            x = torch.sqrt(ab_s) * xb + torch.sqrt(1 - ab_s) * eps
            for t in range(t_start, 0, -1):
                idx = t - 1
                tt = torch.full((xb.shape[0],), idx, dtype=torch.long, device=dev)
                eps_hat = model(x, tt)
                a_t = alpha[idx]; ab_t = alpha_bar[idx]
                # x0 的估计必须除以 sqrt(alpha_bar_t)（不是 sqrt(alpha_t)）
                x0_hat = (x - torch.sqrt(1 - ab_t) * eps_hat) / torch.sqrt(ab_t)
                if sampler == "ddim":
                    if t > 1:
                        ab_prev = alpha_bar[idx - 1]
                        x = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1 - ab_prev) * eps_hat
                    else:
                        x = x0_hat
                else:
                    z = torch.randn(xb.shape, generator=g, device=dev, dtype=xb.dtype) if t > 1 else 0.0
                    x = (1.0 / torch.sqrt(a_t)) * (x - ((1 - a_t) / torch.sqrt(1 - ab_t)) * eps_hat) \
                        + torch.sqrt(beta[idx]) * z
                if clamp is not None:
                    x = x.clamp(-clamp, clamp)
            out[i0:i0 + batch] += x.cpu()
    return out / n_sample


def aggregate_point_errors(err_w, n_points, sw, ss, V):
    """err_w: [n_win, V, sw] 逐元素平方误差 -> [n_points, V] 逐点逐变量平均误差"""
    s = np.zeros((n_points, V), dtype=np.float64)
    c = np.zeros(n_points, dtype=np.float64)
    for k in range(len(err_w)):
        w0 = k * ss
        seg = err_w[k].T                                   # [sw, V]
        s[w0:w0 + sw] += seg
        c[w0:w0 + sw] += 1.0
    return s / np.maximum(c, 1.0)[:, None]


def robust_stats(a):
    mu = float(np.median(a))
    sd = float(1.4826 * np.median(np.abs(a - mu)))
    return mu, max(sd, 1e-12)


def make_score(E, E_ref, mode="paper"):
    """E: [T, V] 逐点逐变量重构误差；E_ref: 参考误差（验证段），用于 zval_* 模式。"""
    if mode == "paper":          # 论文式(12)：对变量求和/取均值
        return E.mean(axis=1)
    if mode in ("zval_mean", "zval_max"):
        mu = E_ref.mean(axis=0); sd = E_ref.std(axis=0) + 1e-8
        z = (E - mu) / sd
        return z.mean(axis=1) if mode.endswith("mean") else z.max(axis=1)
    if mode in ("zself_mean", "zself_max"):
        mu = np.median(E, axis=0)
        sd = 1.4826 * np.median(np.abs(E - mu), axis=0) + 1e-8
        z = (E - mu) / sd
        return z.mean(axis=1) if mode.endswith("mean") else z.max(axis=1)
    raise ValueError(mode)


def window_scores(x0, xh):
    """窗口级得分（论文式(12)）：逐元素平方误差均值"""
    return nn.functional.mse_loss(xh, x0, reduction="none").mean(dim=[1, 2]).numpy()


def point_adjust(pred, gt):
    """Point-Adjusted 协议（SMAP/MSL benchmark 通用）：GT 异常段只要命中一次，整段记为命中。"""
    pred = pred.copy(); gt = gt.copy()
    i = 0
    while i < len(gt):
        if gt[i] == 1:
            j = i
            while j < len(gt) and gt[j] == 1:
                j += 1
            if pred[i:j].any():
                pred[i:j] = 1
            i = j
        else:
            i += 1
    return pred


def prf(pred, gt):
    pred = np.asarray(pred).astype(np.int64); gt = np.asarray(gt).astype(np.int64)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    P = tp / (tp + fp) if tp + fp > 0 else 0.0
    R = tp / (tp + fn) if tp + fn > 0 else 0.0
    F1 = 2 * P * R / (P + R) if P + R > 0 else 0.0
    return dict(P=P, R=R, F1=F1, TP=tp, FP=fp, FN=fn)


def eval_with_pa(pred, gt):
    if gt.sum() > 0:
        return prf(pred, gt), prf(point_adjust(pred, gt), gt)
    return prf(pred, gt), prf(pred, gt)


def oracle_best_f1(score, gt):
    """阈值扫描能拿到的最好 F1（诊断上界，不用于真实判定）。"""
    if gt.sum() == 0:
        return dict(F1=0.0, thresh=float("nan"))
    qs = np.unique(np.quantile(score, np.linspace(0.5, 0.9999, 300)))
    best = dict(F1=-1.0, thresh=float("nan"))
    for th in qs:
        m = prf((score > th).astype(np.int64), gt)
        if m["F1"] > best["F1"]:
            best = dict(F1=m["F1"], thresh=float(th), P=m["P"], R=m["R"])
    return best


def make_threshold(score_val, score_test, mode, eta):
    """返回 (thresh, mu, sd, 说明)"""
    if mode == "val":
        mu = float(np.mean(score_val)); sd = float(np.std(score_val))
        return mu + eta * sd, mu, sd, "验证段 μ+ησ（论文式(13)）"
    if mode == "val_robust":
        mu, sd = robust_stats(score_val)
        return mu + eta * sd, mu, sd, "验证段 中位数+η·1.4826MAD"
    if mode == "test_robust":
        mu, sd = robust_stats(score_test)
        return mu + eta * sd, mu, sd, "测试段自身 中位数+η·1.4826MAD（无监督，抗分布漂移）"
    raise ValueError(mode)


# =====================================================================
# 6. 单通道流程
# =====================================================================
def run_channel(args, labels, dev, sched, out_dir, data_root):
    cid = args.channel
    tr_raw = np.load(os.path.join(data_root, "train", f"{cid}.npy")).astype(np.float64)
    te_raw = np.load(os.path.join(data_root, "test", f"{cid}.npy")).astype(np.float64)
    policy = args.col_policy
    if args.keep_dead_cols and policy == "drop":      # 兼容旧参数
        policy = "testscale"
    st = fit_norm(tr_raw, te_raw, policy=policy)
    tr = apply_norm(tr_raw, st); te = apply_norm(te_raw, st)
    V = tr.shape[1]

    n_val_pts = int(len(tr) * args.val_frac)
    n_tr_pts = len(tr) - n_val_pts
    tr_seg, va_seg = tr[:n_tr_pts], tr[n_tr_pts:]
    all_w = make_windows(tr_seg, args.sw, args.train_ss)
    if len(all_w) < 8:
        print(f"[{cid}] 训练窗口过少({len(all_w)})，跳过", flush=True)
        return None

    model = UNet1D(V, base=args.base, emb_dim=args.emb,
                   ch_mult=tuple(int(x) for x in args.ch_mult.split(",")),
                   dropout=args.dropout).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{cid}] 归一化策略={policy} 原始变量={tr_raw.shape[1]} → 送入网络={V} 列 "
          f"(两边都恒定丢掉={tr_raw.shape[1]-V}) | 训练点={n_tr_pts} 验证点={n_val_pts} "
          f"训练窗口={len(all_w)} 参数量={n_par/1e3:.1f}K  {describe_schedule(sched[2], args.t_start)}",
          flush=True)
    loss_hist = train_model(model, all_w, sched[0], sched[2], dev,
                            iters=args.iters, batch=args.batch, lr=args.lr,
                            ema_decay=args.ema, warmup=max(50, args.iters // 20),
                            log_every=args.log_every, tag=cid, seed=args.seed)

    # ---- 验证段 ----
    va_w = make_windows(va_seg, args.sw, args.test_ss)
    vh = reconstruct(model, va_w, sched[0], sched[1], sched[2], args.t_start, dev,
                     sampler=args.sampler, n_sample=args.n_sample, batch=args.batch, seed=args.seed)
    verr = (vh - va_w).pow(2).numpy()
    E_val = aggregate_point_errors(verr, len(va_seg), args.sw, args.test_ss, V)

    # ---- 测试段 ----
    te_w = make_windows(te, args.sw, args.test_ss)
    th = reconstruct(model, te_w, sched[0], sched[1], sched[2], args.t_start, dev,
                     sampler=args.sampler, n_sample=args.n_sample, batch=args.batch, seed=args.seed)
    terr = (th - te_w).pow(2).numpy()
    E_test = aggregate_point_errors(terr, len(te), args.sw, args.test_ss, V)
    t_win = window_scores(te_w, th)

    anom = labels.get(cid, (None, None, []))[2]
    g_pt = gt_points(anom, len(te))

    # ---- 打分 + 阈值 + 指标 ----
    s_val = make_score(E_val, E_val, args.score)
    s_test = make_score(E_test, E_val, args.score)
    thresh, mu, sd, how = make_threshold(s_val, s_test, args.thresh, args.eta)
    pred = (s_test > thresh).astype(np.int64)
    m_plain, m_pa = eval_with_pa(pred, g_pt)
    orc = oracle_best_f1(s_test, g_pt)

    # 论文风格窗口级指标（按 sw 不重叠分块）
    blk = args.sw
    n_blk = max(1, (len(te) - args.sw) // blk + 1)
    b_score = np.array([s_test[k * blk: k * blk + args.sw].mean() for k in range(n_blk)])
    g_blk = gt_windows(anom, n_blk, args.sw, blk)
    w_plain, w_pa = eval_with_pa((b_score > thresh).astype(np.int64), g_blk)

    # 正常/异常分离度
    normal, anom_s = s_test[g_pt == 0], s_test[g_pt == 1] if (g_pt == 1).any() else np.array([])
    sep = dict(normal_mean=float(normal.mean()) if len(normal) else None,
               normal_std=float(normal.std()) if len(normal) else None,
               anom_mean=float(anom_s.mean()) if len(anom_s) else None,
               anom_std=float(anom_s.std()) if len(anom_s) else None,
               val_mean=float(s_val.mean()), val_std=float(s_val.std()),
               pred_pos_rate=float(pred.mean()))
    if len(anom_s) and sep["normal_mean"]:
        sep["gap_ratio"] = float(sep["anom_mean"] / sep["normal_mean"])

    print(f"[{cid}] 阈值={thresh:.5f} (μ={mu:.5f} σ={sd:.5f}, {how})  预测异常比例={pred.mean()*100:.1f}% "
          f"(GT={g_pt.mean()*100:.1f}%)", flush=True)
    print(f"[{cid}] 点级 plain P={m_plain['P']:.4f} R={m_plain['R']:.4f} F1={m_plain['F1']:.4f} "
          f"(TP={m_plain['TP']} FP={m_plain['FP']} FN={m_plain['FN']})   PA F1={m_pa['F1']:.4f}", flush=True)
    print(f"[{cid}] 窗级 plain F1={w_plain['F1']:.4f}  PA F1={w_pa['F1']:.4f}  |  "
          f"oracle 最佳点级 F1={orc['F1']:.4f} @th={orc['thresh']:.5f}", flush=True)
    print(f"[{cid}] 分离度: 正常均={sep['normal_mean']} 异常均={sep['anom_mean']} "
          f"gap_ratio={sep.get('gap_ratio')}", flush=True)

    def _eta_row(e):
        p = (s_test > mu + e * sd).astype(np.int64)
        pl, pa = eval_with_pa(p, g_pt)
        return dict(eta=e, thresh=float(mu + e * sd), pred_pos=float(p.mean()), plain=pl, pa=pa)

    res = dict(channel=cid, spacecraft=labels.get(cid, ("?",))[0], cls=labels.get(cid, (None, "?"))[1],
               n_var=V, n_var_raw=int(tr_raw.shape[1]), n_live=int((~st["dead_tr"]).sum()),
               col_policy=policy, n_points=int(len(te)),
               n_anom_points=int(g_pt.sum()), t_start=args.t_start, score=args.score, thresh_mode=args.thresh,
               thresh=float(thresh), mu=mu, sd=sd, oracle=orc,
               point_plain=m_plain, point_pa=m_pa, window_plain=w_plain, window_pa=w_pa,
               separation=sep,
               eta_sweep=[_eta_row(e) for e in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]],
               hyper=dict(sw=args.sw, train_ss=args.train_ss, test_ss=args.test_ss,
                          n_steps=args.n_steps, t_start=args.t_start, sampler=args.sampler,
                          n_sample=args.n_sample, iters=args.iters, base=args.base,
                          schedule=args.schedule, eta=args.eta, col_policy=policy,
                          val_frac=args.val_frac))

    if args.plots:
        save_plots(cid, out_dir, te_raw, s_test, thresh, pred, anom, g_pt,
                   s_val=s_val, loss_hist=loss_hist)
    if args.save_model:
        torch.save({"sd": model.state_dict(),
                    "norm": {k: np.asarray(v) for k, v in st.items() if k != "policy"},
                    "policy": st["policy"], "V": V, "base": args.base, "emb": args.emb,
                    "ch_mult": args.ch_mult, "t_start": args.t_start, "n_steps": args.n_steps},
                   os.path.join(out_dir, f"{cid}_model.pt"))
    return res


def save_plots(cid, out_dir, te_raw, s_test, thresh, pred, anom, g_pt, s_val=None, loss_hist=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(s_test)
    fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    ax[0].plot(te_raw[:, 0], color="b", lw=0.8)
    for a, b in anom:
        ax[0].axvspan(a, b, color="g", alpha=0.2)
    ax[0].set_ylabel("signal(var0)"); ax[0].set_title(f"{cid} 测试信号 (绿=GT异常)")
    ax[1].plot(s_test, color="k", lw=0.8, label="score")
    ax[1].axhline(thresh, color="r", ls="--", label="threshold")
    ax[1].set_yscale("symlog"); ax[1].set_ylabel("score"); ax[1].legend(loc="upper left")
    ax[1].set_title("逐点重构误差得分（symlog）")
    ax[2].fill_between(np.arange(n), 0, g_pt, color="g", alpha=0.35, label="GT")
    ax[2].fill_between(np.arange(n), 0, pred, color="r", alpha=0.35, label="pred")
    ax[2].set_ylim(0, 1.6); ax[2].legend(loc="upper left"); ax[2].set_xlabel("time")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{cid}_detect.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    if loss_hist:
        ax[0].plot(loss_hist, lw=0.6)
        if len(loss_hist) > 50:
            k = max(1, len(loss_hist) // 100)
            ma = np.convolve(loss_hist, np.ones(k) / k, mode="valid")
            ax[0].plot(np.arange(len(ma)) + k - 1, ma, color="r", lw=1.2)
        ax[0].set_title("training eps-MSE"); ax[0].set_xlabel("iter")
    if s_val is not None:
        ax[1].hist(s_val, bins=60, color="gray", alpha=0.85)
        ax[1].axvline(thresh, color="r", ls="--")
        ax[1].set_title("验证段得分分布")
    ax[2].hist(s_test[g_pt == 0], bins=60, alpha=0.7, label="normal", color="tab:blue", density=True)
    if (g_pt == 1).any():
        ax[2].hist(s_test[g_pt == 1], bins=60, alpha=0.7, label="anomaly", color="tab:red", density=True)
    ax[2].axvline(thresh, color="r", ls="--"); ax[2].legend()
    ax[2].set_title("测试段得分分布")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{cid}_score.png"), dpi=110); plt.close(fig)


# =====================================================================
# 7. 汇总与主流程
# =====================================================================
def aggregate(results, k_plain, k_pa):
    def mic(key):
        tp = sum(r[key]["TP"] for r in results); fp = sum(r[key]["FP"] for r in results)
        fn = sum(r[key]["FN"] for r in results)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        return dict(P=P, R=R, F1=2 * P * R / (P + R) if P + R else 0.0, TP=tp, FP=fp, FN=fn)
    return dict(micro_plain=mic(k_plain), micro_pa=mic(k_pa),
                macro_plain_F1=float(np.mean([r[k_plain]["F1"] for r in results])) if results else 0.0,
                macro_pa_F1=float(np.mean([r[k_pa]["F1"] for r in results])) if results else 0.0,
                macro_oracle_F1=float(np.mean([r["oracle"]["F1"] for r in results])) if results else 0.0)


def build_parser():
    ap = argparse.ArgumentParser(description="DDTAD 修正版（DDPM + 1-D U-Net 遥测时序异常检测）")
    ap.add_argument("--data_dir", default="/mnt/sdb/home/liuqr/Dataset",
                    help="数据集目录（含 data/... 与 labeled_anomalies.csv）")
    ap.add_argument("--channel", default="E-1", help="单个通道名；或 all")
    ap.add_argument("--channels", default="",
                    help="逗号分隔的通道列表（多卡切分用），优先级高于 --channel")
    ap.add_argument("--gpu", default="", help="指定显卡编号 0/1/2/3 或 cpu（默认自动）")
    ap.add_argument("--amp", action="store_true", help="开启混合精度训练（4090 上通常更快）")
    ap.add_argument("--out_dir", default="./out_fixed")
    ap.add_argument("--sw", type=int, default=SW_DEFAULT)
    ap.add_argument("--train_ss", type=int, default=TRAIN_SS_DEFAULT)
    ap.add_argument("--test_ss", type=int, default=TEST_SS_DEFAULT)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--col_policy", default="drop", choices=["drop", "testscale", "zero"],
                    help="恒列处理策略：drop=删掉训练/测试都恒定的列（推荐）；"
                         "testscale=用测试段范围给恒列尺度；zero=恒列置 0（最早做法）")
    ap.add_argument("--keep_dead_cols", action="store_true",
                    help="[已废弃] 等价于 --col_policy testscale")
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    ap.add_argument("--n_steps", type=int, default=N_STEPS_DEFAULT)
    ap.add_argument("--t_start", type=int, default=T_START_DEFAULT)
    ap.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    ap.add_argument("--n_sample", type=int, default=3)
    ap.add_argument("--score", default="paper",
                    choices=["paper", "zval_mean", "zval_max", "zself_mean", "zself_max"])
    ap.add_argument("--thresh", default="val", choices=["val", "val_robust", "test_robust"])
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--ch_mult", default="1,2,4")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=None, help="--iters 的别名（兼容旧命令）")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--eta", type=float, default=ETA_DEFAULT)
    ap.add_argument("--eval", default="both", choices=["plain", "pa", "both"], help="仅兼容旧命令，本版同时输出两种")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="all 模式下最多跑多少通道（0=全部）")
    ap.add_argument("--plots", action="store_true", default=None)
    ap.add_argument("--no_plots", dest="plots", action="store_false")
    ap.add_argument("--save_model", action="store_true", default=None)
    ap.add_argument("--no_save_model", dest="save_model", action="store_false")
    return ap


def write_summary(all_results, out_dir, hyper=None, title=""):
    """把 {channel: [result,...]} 汇总成 report.txt / results.json / summary.csv。
    多卡分片跑完后由 ddtad_parallel.py 复用本函数做全局汇总。"""
    flat = [v[0] for v in all_results.values() if v]
    used = [r for r in flat if r["n_anom_points"] > 0]
    skipped = [r["channel"] for r in flat if r["n_anom_points"] == 0]
    pt = aggregate(used, "point_plain", "point_pa")
    wd = aggregate(used, "window_plain", "window_pa")

    lines = ["=" * 90]
    if title:
        lines.append(title)
    lines.append(f"汇总：参与评估通道 {len(used)} 个"
                 + (f"（{len(skipped)} 个通道测试集无异常标签，已跳过: {skipped}）" if skipped else ""))
    lines.append(f"  点级   plain  P={pt['micro_plain']['P']*100:.2f}  R={pt['micro_plain']['R']*100:.2f}  "
                 f"F1={pt['micro_plain']['F1']*100:.2f}   [宏观 F1={pt['macro_plain_F1']*100:.2f}]")
    lines.append(f"  点级   PA     P={pt['micro_pa']['P']*100:.2f}  R={pt['micro_pa']['R']*100:.2f}  "
                 f"F1={pt['micro_pa']['F1']*100:.2f}   [宏观 F1={pt['macro_pa_F1']*100:.2f}]")
    lines.append(f"  窗口级 plain  F1={wd['micro_plain']['F1']*100:.2f}   "
                 f"窗口级 PA  F1={wd['micro_pa']['F1']*100:.2f}")
    lines.append(f"  oracle 上界（阈值扫描，仅诊断）宏观点级 F1={pt['macro_oracle_F1']*100:.2f}")
    lines.append("  论文 DDTAD 参考：SMAP P=92.27 R=91.30 F1=91.78 | MSL P=94.47 R=97.21 F1=95.82（点级+PA）")
    for sc in sorted({r["spacecraft"] for r in used}):
        sub = [r for r in used if r["spacecraft"] == sc]
        a = aggregate(sub, "point_plain", "point_pa")
        lines.append(f"  [{sc}] 通道={len(sub):3d}  plain F1={a['micro_plain']['F1']*100:6.2f}  "
                     f"PA P={a['micro_pa']['P']*100:6.2f} R={a['micro_pa']['R']*100:6.2f} "
                     f"F1={a['micro_pa']['F1']*100:6.2f}")
    lines.append("=" * 90)
    report = "\n".join(lines)
    print("\n" + report, flush=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    out = dict(hyper=hyper or {}, summary=dict(point=pt, window=wd, n_channels=len(used), skipped=skipped),
               per_channel=all_results)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)

    with open(os.path.join(out_dir, "summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "spacecraft", "class", "n_var", "n_live", "n_points", "n_anom_points",
                    "P_plain", "R_plain", "F1_plain", "P_pa", "R_pa", "F1_pa", "F1_win_pa",
                    "oracleF1", "thresh", "pred_pos_rate", "normal_mean", "anom_mean", "gap_ratio"])
        for r in sorted(flat, key=lambda x: x["channel"]):
            w.writerow([r["channel"], r["spacecraft"], r["cls"], r["n_var"], r["n_live"], r["n_points"],
                        r["n_anom_points"],
                        f"{r['point_plain']['P']:.4f}", f"{r['point_plain']['R']:.4f}",
                        f"{r['point_plain']['F1']:.4f}",
                        f"{r['point_pa']['P']:.4f}", f"{r['point_pa']['R']:.4f}",
                        f"{r['point_pa']['F1']:.4f}",
                        f"{r['window_pa']['F1']:.4f}", f"{r['oracle']['F1']:.4f}", f"{r['thresh']:.5f}",
                        f"{r['separation']['pred_pos_rate']:.4f}",
                        f"{r['separation']['normal_mean']:.5f}",
                        f"{r['separation']['anom_mean']:.5f}" if r['separation']['anom_mean'] is not None else "",
                        f"{r['separation'].get('gap_ratio', float('nan')):.4f}"])
    print(f"\n结果已保存: {out_dir}/report.txt | results.json | summary.csv", flush=True)
    return out


def main():
    args = build_parser().parse_args()
    if args.epochs is not None:
        args.iters = args.epochs
    if args.plots is None:
        args.plots = (args.channel != "all" and not args.channels)
    if args.save_model is None:
        args.save_model = (args.channel != "all" and not args.channels)
    n_lev = len(args.ch_mult.split(","))
    if args.sw % (2 ** n_lev) != 0:
        raise SystemExit(f"--sw 必须能被 2**{n_lev} 整除")

    dev = pick_device(args.gpu)
    os.makedirs(args.out_dir, exist_ok=True)
    root, csv_path = resolve_paths(args.data_dir)
    labels = load_labels(csv_path)
    print("=" * 90)
    print(f"设备={describe_device(dev)} | torch={torch.__version__} | 数据根={root} | 标签={csv_path}",
          flush=True)
    print(f"超参: sw={args.sw} train_ss={args.train_ss} test_ss={args.test_ss} T={args.n_steps} "
          f"t_start={args.t_start} sampler={args.sampler} n_sample={args.n_sample} iters={args.iters} "
          f"base={args.base} score={args.score} thresh={args.thresh} eta={args.eta} amp={args.amp} "
          f"keep_dead_cols={args.keep_dead_cols}", flush=True)  # col_policy 见各通道日志
    print("=" * 90, flush=True)

    if args.channels.strip():
        chans = [c.strip() for c in args.channels.split(",") if c.strip()]
    elif args.channel == "all":
        chans = sorted(c for c in labels
                       if os.path.exists(os.path.join(root, "train", f"{c}.npy"))
                       and os.path.exists(os.path.join(root, "test", f"{c}.npy")))
        if args.limit:
            chans = chans[:args.limit]
    else:
        chans = [args.channel]
    print(f"通道数: {len(chans)} -> {chans if len(chans) <= 12 else chans[:12] + ['...']}", flush=True)

    orig_channel = args.channel
    all_results = {}
    for rnd in range(args.rounds):
        seed = args.seed + rnd
        set_seed(seed)
        sched = build_schedule(args.schedule, args.n_steps, dev)
        for cid in chans:
            args.channel = cid
            try:
                r = run_channel(args, labels, dev, sched, args.out_dir, root)
            except Exception:
                import traceback
                traceback.print_exc()
                print(f"[{cid}] 失败", flush=True)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            if r is None:
                continue
            r["seed"] = seed
            all_results.setdefault(cid, []).append(r)
    args.channel = orig_channel

    if not all_results:
        print("没有产出任何结果")
        return
    write_summary(all_results, args.out_dir, hyper=vars(args),
                  title=f"DDTAD 修正版汇总 | t_start={args.t_start} score={args.score} "
                        f"thresh={args.thresh} iters={args.iters}")


if __name__ == "__main__":
    main()
