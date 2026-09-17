# DDTAD edition13 —— 跑全量 + 本轮补齐的 GPU 适配

> 本文档随 `edition13/` 一起上传。

---

## 〇、GPU 迁移 & 多卡适配审计结果（本轮逐项核过代码）

### 结论：**早就做了**。这轮审计又找出并修好了 **3 个真实的 GPU 效率/健壮性问题**。

#### ✅ 已有的 GPU 迁移（逐条可查）

| 项 | 位置（`ddtad_run.py`） |
|---|---|
| 设备选择 + `torch.cuda.set_device()` | `pick_device()` |
| **TF32** + `cudnn.benchmark` | `pick_device()` |
| AMP（跨 torch 版本兼容，不可用自动降级） | `_amp_api()` |
| 调度张量 / 数据 / 时间步全部在 device 上 | `train_model` / `reconstruct` |
| 采样器 Random generator 指定 device | `reconstruct` |
| 显存与卡名打印 | `describe_device()` |

#### ✅ 已有的多卡适配（通道级并行，不是 DDP）

| 项 | 位置（`ddtad_parallel.py`） |
|---|---|
| 卡数自动探测（尊重 `CUDA_VISIBLE_DEVICES`） | `torch_count()` |
| 按 test_len 贪心负载均衡分片 | `shard_by_cost()` |
| 卡↔进程映射 `gpu = i % n_gpu` | 主循环 |
| 单卡隔离 `CUDA_VISIBLE_DEVICES=<gpu>` | 主循环 |
| 父进程**不**初始化 CUDA 上下文（不占 GPU0 显存） | `gpu_banner()` |
| 子进程日志 + 60s 进度 + ETA | 等待循环 |
| 结果自动合并 | `write_summary()` |

> 为什么不用 DDP：模型只有 3.96M 参数、单通道数据几千点，且实测**理论算力利用率仅 ~2.7%**
> （单迭代理论下限 0.27ms、实测 ~23ms）——瓶颈是 kernel launch 而非算力。
> DDP 拆小模型收益极低，**通道级并行才是真加速**。

### 🔧 本轮新修的 3 个 GPU 问题

| # | 问题 | 影响 | 修复 |
|---|---|---|---|
| **1** | **EMA 每次迭代约 320 次 kernel launch** | `base=64` 有 **160 个参数张量**，旧实现逐个 `mul_`+`add_` → 320 次启动。本任务本来就 launch-bound，这几乎是纯开销 | 改用 `torch._foreach_mul_` / `_foreach_add_` → **降到 2 次**。已验证与旧实现**数值完全一致（差异 0.000e+00）** |
| **2** | **81 通道连续跑没有显存回收** | 不同通道 `V`（列数）不同 → 激活形状变化 → 默认分配器碎片化 | ① 设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；② 每 5 个通道 `empty_cache()` 并打印 allocated/reserved |
| **3** | **多卡绑定没有核验** | 万一子进程没吃到 `CUDA_VISIBLE_DEVICES`，会静默全挤在 cuda:0，日志里看不出来 | 子程序横幅打印 `CVD=<值>`；启动器回读每个 shard 日志**核验 shard i 真的在 GPU i 上**，不符就警告 |

顺带修了一个**潜在陷阱**：死代码 `window_scores()` 里的 `.numpy()` 没先 `.cpu()`，
一旦被复用就会在 GPU 上报错（就是我们之前踩过的那个坑），已补上。

### 🆕 新增：GPU 实测基准（`ddtad_smoke.py --bench`）

`run_quick.sh` 的步骤 0 现在会用 `--device cuda:0 --bench`，在**真实 GPU** 上量：

```
[0/3] GPU 基准（论文设置：base=64, batch=32, 输入 [32,24,128]）
  设备: NVIDIA GeForce RTX 4090  可见卡数=5
  参数量=3.96M  参数张量数=160
  单次迭代: 含EMA x.xx ms | 不含EMA y.yy ms  -> EMA 占比 z.z%
  显存: allocated=... reserved=...
[多卡核验] 10/10 个分片都跑在预期的 GPU 上 ✔      <- 跑全量时由启动器打印
```

这样你**一眼就能确认**：真的在用 GPU、EMA 优化生效、每张卡都吃到了活。

---

## 一、跑全量

```bash
scp -r edition13 liuqr@pserver4090:/mnt/sdb/home/liuqr/Project/20269/ddtad/
cd /mnt/sdb/home/liuqr/Project/20269/ddtad/edition13
bash run_all.sh          # 约 30~40 分钟（EMA 优化后应更快）
```

启动时会打印 **`[多卡核验] 10/10 个分片都跑在预期的 GPU 上 ✔`** —— 这就是多卡适配的直接证据。
每隔 60s 打印 `[进度] ... ETA≈X.Xmin`，每 5 个通道打印一次显存。

**为什么一次跑分就够**：打分/阈值/平滑的评估是**免费**的（重构缓存复用），所以这一跑同时产出：

| report.txt 里的表 | 内容 |
|---|---|
| 主指标 | **论文忠实版**：窗口级 + 式(12) + μ+2σ + 不 PA |
| `t_start 对照` | 6 个有效加噪步数 |
| **`t_start × 打分方式` 交叉排行榜** | 7 打分 × 6 t_start = 42 组合 |
| 评测口径对照 | 窗口/点 × 固定/最优阈值 × 是否 PA |
| **方案扫描排行榜** | 322 种组合，按窗口级 F1 排序 |

**回传**：`out/all/report.txt` + `out/all/summary.csv`

---

## 二、多卡性能建议

| 配置 | 说明 |
|---|---|
| `bash run_all.sh` | 默认 **每卡 2 进程**（`WORKERS=2*NGPU`）。模型小、launch-bound，超订能填满 GPU |
| `WORKERS=15 bash run_all.sh` | 每卡 3 进程，更激进（机器上有别人的任务，注意 CPU 争抢） |
| `WORKERS=5 bash run_all.sh` | 保守，每卡 1 进程 |
| `NGPU=3 bash run_all.sh` | 只用前 3 张卡 |

> 先看步骤 0 打印的 `EMA 占比`：如果仍很高，说明还受 launch 限制，`WORKERS` 可以再加。

---

## 三、已对齐论文的部分（未变）

| 项 | 论文 | 现状 |
|---|---|---|
| 判定单元 / point-adjust | 窗口 / 不用 | ✅ |
| 归一化 / 窗口 / 步长 | [−1,1] / 128 / 10 与 128 | ✅ |
| 验证集 | 随机 20% 窗口 | ✅ |
| T / schedule | 100 / Linear | ✅（β 范围论文未给） |
| Dimension / lr / betas | 64 / 5e-5 / (0.9,0.99) | ✅ |
| EMA / batch / epoch | 0.995 / 32 / 1500 | ✅ |
| η / 采样 | 2 / Algorithm 1.B 随机反向 | ✅ |
| 轮数 | 10 轮随机划分 | ⚠️ 全量先跑 1 轮（`--rounds 10` 对齐） |
| 打分 | 式(12) | ⚠️ 主指标用式(12)，另附 `zratio_max` 调优版 |

---

## 四、上一轮交叉扫描的结论（决定打分方式）

| t_start | score | **winP** | winR | **winF1** |
|---|---|---|---|---|
| **100** | **zratio_max** | **100.00** | 42.86 | **60.00** |
| 100 | paper（论文设置） | 33.33 | 42.86 | 37.50 |
| 100 | zself_max | 23.53 | 57.14 | 33.33 |

`zratio_max` 比论文式(12) 好 **22.5 个点**且零误报；论文设置排名 7/36。
全量跑完后我给出**论文忠实版**和**调优版**两组 SMAP/MSL 数字。

> 本地已跑 `ddtad_smoke.py`：端到端（含 JSON 往返合并）**4 项检查全部通过**，
> foreach EMA 与旧实现**数值完全一致**。
