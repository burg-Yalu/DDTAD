# DDTAD edition2 —— 第一轮探针结论 + 修正 + 下一轮诊断

> 本文档随 `edition2/` 一起上传。**先看第一节，再执行第三节的命令。**

---

## 一、第一轮探针（edition1）发现了什么

### 🎯 根因：MSL 全部训练失败，恒零列把损失函数毁掉了

| 通道 | 数据集 | 变量数 | 活列 | **eps-MSE(t=800)** | 结果 |
|---|---|---|---|---|---|
| E-1 | SMAP | 25 | 18 | **0.077** ✔ | PA 0.96 |
| P-1 | SMAP | 25 | 16 | **0.077** ✔ | PA 0.75~0.96 |
| D-1 | SMAP | 25 | 18 | **0.077** ✔ | **plain F1=0.914 / PA=0.968** |
| T-4 | MSL | 55 | 11 | **0.480** ✘ | F1 ≈ 0.06 |
| M-1 | MSL | 55 | 13 | **0.485** ✘ | F1 ≈ 0.01 |
| C-1 | MSL | 55 | 15 | **0.486** ✘ | F1 ≈ 0.02 |

分界线极其干净：**SMAP（25 列）→ loss 0.077；MSL（55 列）→ 卡在 0.48 下不去。**

原因：MSL 平均只有 **12.9 / 55** 列是活的（`data_check_report.txt`），其余约 42 列恒为 0。
恒零列加噪后 `x_t = √(1-ā_t)·ε`，网络要从它反推 ε，需要 `1/√(1-ā_t)` 的增益
（t 小时高达 **3000 倍**）—— 纯病态任务，还吃掉了 **77% 的 loss 权重**。
MSL 占 benchmark 的 1/3（27/81 通道），全部损坏在这里。

### 其它三个次要发现

| # | 现象 | 证据 |
|---|---|---|
| ② | `zero` 策略把 **D-12 / D-13** 变成全零输入 | D-12: `liveTr=0, liveTe=8` → 训练输入全 0，模型学不到任何东西 |
| ③ | FiLM 只能衰减不能放大 | 写成 `*sigmoid(scale) ∈ (0,1)`，标准 DDPM 是 `h*(1+scale)+shift` |
| ④ | `t_start` **越小越好**（与直觉相反） | E-1 gap: t=200→**1.88**，t=800→1.05（单调降）；D-1: 12.6→2.2 |
| ⑤ | 验证段 `val_sd` 被离群撑大 | E-1 t=200: `val_sd=0.0155` 是 `tstNorm=0.0075` 的 2 倍 → μ+2σ 偏高 → R 仅 0.030 |

结论 ④ 说明：真正起作用的是**低噪声下的"去噪自编码器"机制**（模型只在正常流形上准确），
而不是论文描述的"把异常抹掉再生成"。所以 `t_start` 要往 50~150 探。

### 已经确认可用的部分

- **D-1**（SMAP point，38.3% 异常）：η=2 时 plain **F1 = 0.914**、PA = 0.968、oracle = 0.936
  —— **已达论文水平**
- E-1 的 PA@η=4 也有 0.958~0.967
- M-1 的 oracle 只有 0.505，而"全判异常"就有 0.668 → 该通道得分反相关，属退化通道

---

## 二、edition2 相对 edition1 的改动（共 5 处）

| # | 文件 | 改动 |
|---|---|---|
| 1 | `ddtad_run.py` | 新增 **`--col_policy {drop,testscale,zero}`，默认 `drop`**：删掉"训练+测试都恒定"的列；"训练恒定但测试有变化"的列用测试段尺度（这一条同时救回 D-12/D-13） |
| 2 | `ddtad_run.py` | FiLM 改为 `h*(1+scale)+shift`（允许放大） |
| 3 | `ddtad_probe.py` | 新增 `--col_policies`，可对同一通道做多种策略对照 |
| 4 | `ddtad_probe.py` | `t_start` 网格改为 **50,100,200,300,400**（下探到 50） |
| 5 | `ddtad_probe.py` | 表格新增 `VR2PA / VR3PA` 列：验证段**中位数+η·1.4826MAD** 稳健阈值 |

`ddtad_check_data.py` 新增 `keepDrop`（drop 策略下保留的列数）与 `allDead`（训练段全恒定告警）两列。

---

## 三、上传清单与执行

### 上传

把 `edition2/` 整个目录 scp 到 `/mnt/sdb/home/liuqr/Project/20269/ddtad/`：

| 文件 | 作用 |
|---|---|
| `ddtad_run.py` | 主程序（恒列策略 drop + FiLM 修正） |
| `ddtad_probe.py` | 诊断探针（新增策略对照 + t_start 下探 + 稳健阈值） |
| `ddtad_check_data.py` | 全通道数据体检 |
| `ddtad_parallel.py` | 4 卡并行调度器 |
| `run_probe.sh` | 一键诊断 |
| `run_all.sh` | 全量跑分 |
| `RUN_ON_SERVER.md` | 本文档 |

```bash
scp -r edition2 liuqr@pserver4090:/mnt/sdb/home/liuqr/Project/20269/ddtad/
```

### 执行

```bash
cd /mnt/sdb/home/liuqr/Project/20269/ddtad/edition2
bash run_probe.sh          # 约 35~45 分钟（含自检）
```

`run_probe.sh` 的 7 个阶段（**前 3 步是新增的多卡自检，几分钟内就能暴露问题**）：

| 阶段 | 内容 | 目的 |
|---|---|---|
| [0/7] | 环境自检：逐张卡做 TF32 矩阵乘 + AMP API 探测 | 确认 **4 张卡全部可用**，不是只有 cuda:0 |
| [1/7] | 单卡冒烟（E-1，50 迭代） | 确认单卡路径通 |
| [2/7] | **多卡自检（探针路径）**：4 卡各 1 通道，校验 `probe_report_all.txt` 含 4 个通道 | 验证 `ddtad_parallel.py` 分片 + 报告合并 |
| [3/7] | **多卡自检（主程序路径）**：4 卡各 1 通道，校验 `summary.csv` 恰好 5 行 | 验证主程序分片 + `write_summary` 全局汇总 |
| [4/7] | 全通道数据体检 | 81 通道死列/异常可见度 |
| [5/7] | **正式诊断探针**（4 卡并行） | 本轮主要目的 |
| [6/7] | 打包 `ddtad_probe_results.tgz` | 回传 |
| [7/7] | 扫描分片日志里的 OOM / Traceback | 避免"静默失败" |

> 第 2、3 步是我特意加上的：上一版脚本只冒烟测试了单卡，多卡路径从未被验证过，
> 万一有 bug 要等 40 分钟的正式探针开始时才发现。现在几分钟就能确认。

等价的手动命令：

```bash
python ddtad_parallel.py --prog ddtad_probe.py --ngpu 4 --workers 4 \
  --data_dir /mnt/sdb/home/liuqr/Dataset --out_dir probe_out \
  --channels E-1,P-1,D-1,D-12,T-4,M-1,C-1 \
  --extra "--iters_list 6000 --col_policies drop,zero \
           --t_starts 50,100,200,300,400 \
           --scores paper,zval_max,zself_max --n_sample 2"

# 只看分片方案、不执行：
python ddtad_parallel.py --prog ddtad_run.py --ngpu 4 --dry_run
```

---

## 四、这一轮要看的判据

| 看哪里 | 期望 | 含义 |
|---|---|---|
| T-4 / M-1 / C-1 在 `策略=drop` 下的 **eps-MSE** | 从 0.48 掉到 **≤0.15** | 恒零列根因确认修好 |
| 同上的 `gap` 与 `oracle` | gap 明显 >1，oracle ≫ 0.2 | 得分重新有判别力 |
| `[Q1 归一化] 策略=drop: 55 列 -> 送入网络 N 列` | N ≈ 13~25 | 列裁剪生效 |
| D-12 的 `送入网络` 列数 | **8 列**（不是 25，也不是 0） | 全恒定通道被救回 |
| `策略=zero` 那几行 | 应与上一轮一致（0.48） | 对照组，确认差异来自策略本身 |
| SMAP 三通道在 `t_start=50~100` | gap 是否比 200 更大 | 确认低噪声方向 |
| `VR2PA` vs `PA`（η=2 列） | VR 更高 | 稳健阈值解决 val_sd 被撑大的问题 |

**回传**：`ddtad_probe_results.tgz`（在 edition2 目录下），或直接贴 `probe_out/probe_report_all.txt`。

---

## 五、时间预算（4×4090）

| 任务 | 单卡 | 4 卡并行 |
|---|---|---|
| 环境自检 + 冒烟测试 | — | ~2 分钟 |
| 数据体检（81 通道） | 数秒 | — |
| 本轮探针（7 通道 × 2 策略 × 5 个 t_start） | ~2 小时 | **~30~40 分钟** |
| 全量跑分（81 通道，`run_all.sh`） | ~3.5 小时 | **~55 分钟** |

---

## 六、背景：原始 `ddtad_run.py` 的完整问题清单

| 编号 | 问题 | 后果 | 修正 |
|---|---|---|---|
| P1 | `beta=linspace(1e-4,0.02,T=100)` → `alpha_bar[T-1]=0.3636`，加噪后残留 **60%** 信号 | 异常没被抹掉；反向链还把信号放大 1.66 倍 → 重建 MSE **2.257**，比"什么都不做"(0.793) 还差 3 倍 | cosine 调度 + 部分加噪 `--t_start` |
| P2 | 原 1-D U-Net = 纯 ReLU + **无任何归一化层** | 实测 ε-MSE：t=99 → 0.699（全零基线 1.0），网络几乎没学到 | ResBlock(GroupNorm+SiLU+残差) + FiLM；实测 0.62 → 0.03 |
| P3 | `--epochs=2000` 且无 LR 调度 / 无 EMA | 欠训练 | cosine LR + warmup + EMA，默认 6000 迭代 |
| P4 | 验证集从 `ss=10` 的重叠窗口随机抽 20%（相邻窗口共享 118/128 点） | 阈值偏低 → 误报爆炸 | 按时间前后切分 |
| P5 | 测试窗口 `ss=128` 不重叠，E-1 只有 66 窗口/6 正样本 | 与论文点级指标不可比 | 重叠窗口聚合的**逐点逐变量**误差 |
| P6 | 反向采样随机且无降方差 | 得分方差大 | `--sampler ddim`（确定性）+ `--n_sample` |
| P7 | test 用自身 min/max 归一化 | 分布外输入 | 统一用训练段统计量 |
| P8 | 得分只对变量取均值 | 异常被稀释 | 5 种打分方式 `--score` |
| P9 | 训练/测试分布漂移时论文式(13)阈值会"全判异常" | 阈值失效 | `--thresh {val,val_robust,test_robust}` + 始终报告 oracle 上界 |
| P10 | GPU 迁移只做了一半 | 4 张卡用 1 张 | `--gpu` / TF32 / `--amp` / `ddtad_parallel.py` 通道级并行 |
| **P11** | **恒零列污染损失（本轮探针发现）** | **MSL 27 个通道全废** | **`--col_policy drop`** |
| **P12** | **FiLM `sigmoid(scale)` 只能衰减** | 高信噪比区域是硬约束 | **`h*(1+scale)+shift`** |
| P13 | `--channel all` 未实现、无随机种子、KMP 冲突 | — | 已修 |
