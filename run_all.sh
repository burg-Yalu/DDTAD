#!/usr/bin/env bash
# =====================================================================
# DDTAD 全量实验（SMAP + MSL，全部通道）—— 多卡并行（卡数自动探测）
#   用法:  bash run_all.sh
#   可选:  DATA=/path/to/Dataset bash run_all.sh          # 自动用全部可见卡
#          NGPU=5  bash run_all.sh                        # 指定用前 5 张
#          WORKERS=10 bash run_all.sh                     # 每卡 2 进程（小模型可能更快）
#          EXTRA="--t_start 100" bash run_all.sh          # 覆盖超参
#
# 说明：每个通道训练一个独立模型，通道之间完全独立 → 通道级并行 ≈ 线性加速。
#       带宽/显存都不是瓶颈，模型只有 ~1.2M 参数，所以也可以用 WORKERS=2*NGPU 超订。
# =====================================================================
set -euo pipefail

DATA=${DATA:-/mnt/sdb/home/liuqr/Dataset}
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
OUT=${OUT:-$ROOT/out/all}    # 绝对路径，避免"报告说保存了但找不到文件"

# ---- 自动探测可用 GPU 数量（尊重 CUDA_VISIBLE_DEVICES） ----
if [ -z "${NGPU:-}" ] || [ "${NGPU}" = "0" ]; then
  NGPU=$(python -c 'import torch;print(max(1,torch.cuda.device_count()))' 2>/dev/null || echo 1)
fi
# 默认每卡开 2 个进程（超订）。
# 理由：本任务模型只有 ~1.2M 参数，单次迭代理论下限 0.265ms、实测约 10ms，
#       只用了约 2.7% 的理论算力 —— 瓶颈是 Python / kernel launch 而不是算力。
#       实测 nvidia-smi 只有 ~28% 利用率、~90W/450W，说明 1 进程/卡远喂不饱 GPU。
#       模型显存占用 <1GB/进程（卡有 24GB），超订没有 OOM 风险。
# 想回到保守模式：WORKERS=$NGPU bash run_all.sh
WORKERS=${WORKERS:-$((NGPU * 2))}

# ★ 主指标仍是论文口径（窗口级 + 验证段固定阈值 + 不 PA），打分用论文式(12) `paper`，
#   这样 headline 是"忠实复现"。同时 report.txt 会附上：
#     · t_start 对照表
#     · t_start × 打分方式 交叉排行榜（7 打分 × 6 个 t_start）
#     · 322 种「打分 × 阈值 × η × 平滑」方案扫描排行榜（按窗口级 F1 排序）
#   所以一次全量跑分就能同时拿到"论文忠实版"和"调优版"两个数字。
#
#     --col_policy drop : 删掉训练/测试都恒定的列（MSL 55→13~18 列，eps-MSE 0.49→0.02）
#     --rounds 1        : 论文是 10 轮随机划分取平均；先用 1 轮拿数字（每轮约 30 分钟）
#                         想要完全对齐论文：EXTRA="--col_policy drop --rounds 10"
EXTRA=${EXTRA:---col_policy drop --rounds 1}

echo "代码目录 : $ROOT"
echo "数据目录 : $DATA"
echo "输出目录 : $OUT"
echo "GPU 数量 : $NGPU (自动探测)   并行进程: $WORKERS ($(( WORKERS / NGPU )) 进程/卡)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
  | sed 's/^/  GPU /' || echo "  (nvidia-smi 不可用)"
echo "超参     : $EXTRA"
echo

# 先看看分片方案（负载均衡是否合理）
python ddtad_parallel.py --prog ddtad_run.py --ngpu "$NGPU" --workers "$WORKERS" \
  --data_dir "$DATA" --out_dir "$OUT" --extra "$EXTRA" --dry_run

read -r -p "确认开始？(y/N) " ans </dev/tty || ans=y
case "$ans" in [yY]*) ;; *) echo "已取消"; exit 0 ;; esac

mkdir -p "$OUT"
set +e
python ddtad_parallel.py --prog ddtad_run.py --ngpu "$NGPU" --workers "$WORKERS" \
  --data_dir "$DATA" --out_dir "$OUT" --extra "$EXTRA" 2>&1 | tee "$OUT.launch.log"
RC=${PIPESTATUS[0]}
set -e

echo
echo "======================================================================"
if [ "$RC" -ne 0 ]; then
  echo "⚠ 启动器退出码 = $RC，请检查 $OUT.launch.log 与 $OUT/shard*.log"
fi
echo "全局报告: $OUT/report.txt   （末尾有「方案扫描」表，挑最终超参用）"
echo "逐通道表: $OUT/summary.csv"
echo "明细JSON: $OUT/results.json"
echo "分片日志: $OUT/shard*.log"
echo "启动日志: $OUT.launch.log"
echo "======================================================================"
echo "请回传: $OUT/report.txt 与 $OUT/summary.csv"
