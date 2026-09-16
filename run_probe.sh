#!/usr/bin/env bash
# =====================================================================
# DDTAD 第二轮机跑脚本（4×4090 多卡并行版）
#   用法:  bash run_probe.sh
#   可选:  DATA=/path/to/Dataset NGPU=4 CHANS=E-1,D-1 bash run_probe.sh
#
# 流程：环境自检 → 单卡冒烟 → 多卡并行自检(探针路径) → 多卡并行自检(主程序+合并)
#       → 全通道数据体检 → 正式诊断探针(4卡) → 打包
# =====================================================================
set -euo pipefail

DATA=${DATA:-/mnt/sdb/home/liuqr/Dataset}
NGPU=${NGPU:-4}
CHANS=${CHANS:-E-1,P-1,D-1,D-12,T-4,M-1,C-1}
POLICIES=${POLICIES:-drop,zero}
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

echo "代码目录 : $ROOT"
echo "数据目录 : $DATA"
echo "GPU      :"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>/dev/null || echo "  未检测到 nvidia-smi"
echo

# ---------------------------------------------------------------
echo "==== [0/7] 环境自检（torch / CUDA / TF32 / AMP） ===="
python - <<'PY'
import os, time, torch
print("  torch:", torch.__version__, "| CUDA 可用:", torch.cuda.is_available(),
      "| 可见卡数:", torch.cuda.device_count())
n = torch.cuda.device_count()
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} = {p.name}  {p.total_memory/1e9:.1f} GB  SM{p.major}{p.minor}")
assert n >= 1, "没有可见的 CUDA 设备！"
if n < 4:
    print(f"  [警告] 只看到 {n} 张卡，多卡分片会只用这么多张")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
for i in range(n):                      # 逐卡各做一次矩阵乘，确认每张卡都能算
    with torch.cuda.device(i):
        a = torch.randn(2048, 2048, device=f"cuda:{i}")
        b = torch.randn(2048, 2048, device=f"cuda:{i}")
        torch.cuda.synchronize(i); t0 = time.time()
        for _ in range(20): a @ b
        torch.cuda.synchronize(i)
        dt = time.time() - t0
        print(f"  cuda:{i} TF32 2048^3 x20 = {dt:.3f}s" + ("  (TF32 生效)" if dt < 0.6 else "  (偏慢，TF32 可能没生效)"))
        del a, b
try:
    from torch.amp import autocast, GradScaler
    print("  AMP API: torch.amp 可用 -> 支持 --amp")
except Exception:
    try:
        from torch.cuda.amp import autocast, GradScaler
        print("  AMP API: torch.cuda.amp 可用 -> 支持 --amp")
    except Exception:
        print("  AMP API: 不可用，--amp 会自动降级关闭")
PY
echo

# ---------------------------------------------------------------
echo "==== [1/7] 单卡冒烟测试 (E-1, 50 迭代) ===="
python ddtad_probe.py --data_dir "$DATA" --channels E-1 --iters_list 50 \
  --col_policies drop --t_starts 100 --scores paper --n_sample 1 --test_ss 128 \
  --out_dir smoke_single
echo "单卡路径通过 ✔"
echo

# ---------------------------------------------------------------
echo "==== [2/7] 多卡并行自检 —— 探针路径（${NGPU} 卡各 1 通道） ===="
python ddtad_parallel.py --prog ddtad_probe.py --ngpu "$NGPU" --workers "$NGPU" \
  --data_dir "$DATA" --out_dir smoke_par_probe --channels E-1,P-1,D-1,T-4 \
  --extra "--iters_list 50 --col_policies drop --t_starts 100 --scores paper --n_sample 1 --test_ss 128" \
  2>&1 | tee smoke_par_probe_launch.log
NGPU_USED=$(grep -c -- '-> GPU' smoke_par_probe_launch.log || true)
NCH=$(grep -c '^### ' smoke_par_probe/probe_report_all.txt 2>/dev/null || true)
echo "  实际用到 $NGPU_USED 个进程/GPU，报告里含 $NCH 个通道"
if [ -f smoke_par_probe/probe_report_all.txt ] && [ "$NCH" -eq 4 ]; then
  echo "多卡探针路径 + 报告合并通过 ✔"
else
  echo "多卡探针路径失败 ✘ 请查看 smoke_par_probe/shard*.log"; exit 1
fi
echo

# ---------------------------------------------------------------
echo "==== [3/7] 多卡并行自检 —— 主程序路径 + 全局汇总合并 ===="
python ddtad_parallel.py --prog ddtad_run.py --ngpu "$NGPU" --workers "$NGPU" \
  --data_dir "$DATA" --out_dir smoke_par_run --channels E-1,P-1,D-1,T-4 \
  --extra "--iters 50 --col_policy drop --t_start 100 --score paper --thresh val --n_sample 1 --test_ss 128" \
  2>&1 | tee smoke_par_run_launch.log
NROW=$(wc -l < smoke_par_run/summary.csv 2>/dev/null || echo 0)
echo "  summary.csv 行数 = $NROW （期望 5 = 表头 + 4 个通道）"
if [ -f smoke_par_run/report.txt ] && [ "$NROW" -eq 5 ]; then
  echo "多卡主程序路径 + write_summary 合并通过 ✔"
  echo "--- smoke_par_run/report.txt 摘要 ---"
  sed -n '1,10p' smoke_par_run/report.txt
else
  echo "多卡主程序路径失败 ✘ 请查看 smoke_par_run/shard*.log"; exit 1
fi
echo

# ---------------------------------------------------------------
echo "==== [4/7] 全通道数据体检（无 GPU，几秒） ===="
python ddtad_check_data.py --data_dir "$DATA" --out_dir check_out
echo

# ---------------------------------------------------------------
echo "==== [5/7] 正式诊断探针：${NGPU} 卡并行，通道=${CHANS}，恒列策略=${POLICIES} ===="
echo "  说明: drop = 删掉训练/测试都恒定的列（新默认）；zero = 上一轮的配置（对照组）"
python ddtad_parallel.py --prog ddtad_probe.py --ngpu "$NGPU" --workers "$NGPU" \
  --data_dir "$DATA" --out_dir probe_out --channels "$CHANS" \
  --extra "--iters_list 6000 --col_policies $POLICIES --t_starts 50,100,200,300,400 --scores paper,zval_max,zself_max --n_sample 2" \
  2>&1 | tee probe_out_launch.log

# ---------------------------------------------------------------
echo "==== [6/7] 打包 ===="
rm -f ddtad_probe_results.tgz
FILES="check_out/data_check_report.txt check_out/data_check.csv"
if [ -f probe_out/probe_report_all.txt ]; then
  FILES="$FILES probe_out/probe_report_all.txt"
else
  FILES="$FILES $(ls probe_out/shard*/probe_report.txt 2>/dev/null || true)"
fi
tar czf ddtad_probe_results.tgz $FILES
echo "已打包: $ROOT/ddtad_probe_results.tgz"

# ---------------------------------------------------------------
echo "==== [7/7] 分片日志体检（找 OOM / 报错） ===="
grep -l -E "CUDA out of memory|RuntimeError|Traceback" probe_out/shard*.log 2>/dev/null \
  && echo "  ↑ 上面这些分片日志里有报错，请一并回传" \
  || echo "  所有分片日志干净，没有 OOM / 报错 ✔"

echo
echo "======================================================================"
echo "全部完成。请回传以下任一项："
echo "  1) $ROOT/ddtad_probe_results.tgz   （推荐）"
echo "  2) 或把  probe_out/probe_report_all.txt  内容贴给我"
echo "  3) 若有报错，附上  probe_out/shard*.log"
echo "======================================================================"
