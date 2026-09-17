#!/usr/bin/env bash
# =====================================================================
# 【第 1 步】单通道快速验证 —— 确认「对齐论文 TABLE II 后」能训得动
#
# 只跑 E-1(SMAP) 和 T-4(MSL) 两个通道，1 张卡，约 4~6 分钟。
# 目的：在花 20 分钟跑全量之前，先确认
#   1) Dimension=64 的模型能收敛（看 loss 是否下降、窗口级 gap 是否 > 1）
#   2) 论文口径（窗口级/固定阈值/不PA）能跑出像样的 P/R/F1
#   3) 没有 OOM / 报错
#
# 用法:  bash run_quick.sh
# =====================================================================
set -euo pipefail

DATA=${DATA:-/mnt/sdb/home/liuqr/Dataset}
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

echo "代码目录 : $ROOT"
echo "数据目录 : $DATA"
echo "配置     : 全部对齐论文 TABLE II / §IV.C"
echo "  T=100 linear | Dimension=64 | Adam lr=5e-5 betas=(0.9,0.99) | batch=32 | EMA=0.995"
echo "  Epoch=1500 | 测试窗口 ss=128 不重叠 | 随机20%窗口做验证 | η=2"
echo "  主指标 = 窗口级 P/R/F1 + 不 point-adjust（论文口径）"
echo

# ---- 步骤 0：CPU 端到端冒烟 + GPU 基准（约 1~2 分钟）----
# (a) 用合成数据把 ddtad_run.py 全流程跑一遍，先抓出 tensor/numpy、shape、
#     KeyError、JSON 键类型这类低级错误；
# (b) 顺带测 GPU 上的迭代耗时与 EMA 占比，确认真的跑在卡上、多卡适配有效。
echo "---- 步骤 0: 端到端冒烟 + GPU 基准 ----"
python ddtad_smoke.py --device cuda:0 --bench || { echo "冒烟测试未通过，先别继续"; exit 1; }
echo

OUT="$ROOT/quick_out"          # 用绝对路径，避免"报告说保存了但文件找不到"
rm -rf "$OUT"
python ddtad_run.py \
  --data_dir "$DATA" \
  --channels E-1,T-4 \
  --col_policy drop \
  --rounds 1 \
  --gpu 0 \
  --out_dir "$OUT" \
  2>&1 | tee "$ROOT/quick_out.log"

echo
echo "======================================================================"
if [ -f "$OUT/report.txt" ] && [ -f "$OUT/summary.csv" ] && [ -f "$OUT/results.json" ]; then
  echo "输出文件校验通过，目录内容："
  ls -la "$OUT"
else
  echo "!! 没有生成输出文件，实际目录情况："
  ls -la "$ROOT" | head -30
  echo "（把上面这段发我）"
  exit 1
fi
echo
echo "看这几行判断是否 OK："
echo "  1) [E-1] 参数量≈3.96M"
echo "  2) [E-1] ★论文口径(窗口级/固定阈值/不PA) P=.. R=.. F1=.."
echo "  3) t_start 对照表 + ★t_start x 打分方式 交叉排行榜"
echo "  4) loss 是否从 ~1.1 降到 0.11 左右"
echo
echo "结果目录: $OUT"
echo "请回传: $OUT/report.txt"
echo "======================================================================"
