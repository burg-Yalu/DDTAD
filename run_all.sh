#!/usr/bin/env bash
# =====================================================================
# DDTAD 全量实验（SMAP + MSL，全部通道）—— 4×4090 多卡并行
#   用法:  bash run_all.sh
#   可选:  DATA=/path/to/Dataset NGPU=4 EXTRA="--t_start 400 --score paper" bash run_all.sh
#
# 说明：每个通道训练一个独立模型，通道之间完全独立 → 通道级并行 = 真·4 倍加速。
#       81 个通道单卡约 3.5 小时，4 卡约 55 分钟。
# =====================================================================
set -euo pipefail

DATA=${DATA:-/mnt/sdb/home/liuqr/Dataset}
NGPU=${NGPU:-4}
WORKERS=${WORKERS:-$NGPU}
OUT=${OUT:-out/all}
EXTRA=${EXTRA:---iters 6000 --t_start 200 --col_policy drop --score paper --thresh val --n_sample 3}
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

echo "数据目录 : $DATA"
echo "输出目录 : $OUT"
echo "并行进程 : $WORKERS  (可见卡: $(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo '?'))"
echo "超参     : $EXTRA"
echo

# 先看看分片方案（负载均衡是否合理）
python ddtad_parallel.py --prog ddtad_run.py --ngpu "$NGPU" --workers "$WORKERS" \
  --data_dir "$DATA" --out_dir "$OUT" --extra "$EXTRA" --dry_run

read -r -p "确认开始？(y/N) " ans </dev/tty || ans=y
case "$ans" in [yY]*) ;; *) echo "已取消"; exit 0 ;; esac

python ddtad_parallel.py --prog ddtad_run.py --ngpu "$NGPU" --workers "$WORKERS" \
  --data_dir "$DATA" --out_dir "$OUT" --extra "$EXTRA"

echo
echo "======================================================================"
echo "全局报告: $OUT/report.txt"
echo "逐通道表: $OUT/summary.csv"
echo "明细JSON: $OUT/results.json"
echo "分片日志: $OUT/shard*.log"
echo "======================================================================"
