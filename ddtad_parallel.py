# -*- coding: utf-8 -*-
"""
ddtad_parallel.py —— 4×4090 多卡并行调度器（通道级并行）

为什么是"通道级并行"而不是 DDP：
    本任务的模型很小（约 1~3 M 参数）、单通道训练数据只有几千个点，
    一个通道就是一个独立模型。用 DDP 把一个小模型拆到 4 张卡上收益极低，
    而 **不同通道之间完全独立** —— 每个通道绑一张卡并行跑，才是真正的 4 倍加速。
    （81 个通道：单卡约 3.5 小时 → 4 卡约 55 分钟）

它会：
  1. 读取 labeled_anomalies.csv，列出全部通道，按"测试序列长度"做负载均衡分片
     （重构耗时 ∝ 测试点数 × t_start，所以按 test_len 贪心分配，避免某张卡提前空闲）
  2. 每个分片起一个子进程，设置 CUDA_VISIBLE_DEVICES=<gpu>，日志写入 shardN.log
  3. 全部结束后合并 results.json，复用 ddtad_run.write_summary 产出全局报告

用法：
  # 全量 SMAP+MSL，4 卡
  python ddtad_parallel.py --prog ddtad_run.py --ngpu 4 --out_dir out/all \
      --extra "--iters 6000 --t_start 400 --score paper --thresh val --n_sample 3"

  # 诊断探针 4 卡并行
  python ddtad_parallel.py --prog ddtad_probe.py --ngpu 4 --out_dir probe_out \
      --channels E-1,P-1,D-1,T-4,M-1,C-1 --extra "--iters_list 6000 --t_starts 200,300,400,500,600,700,800"

  # 只想先看看分片方案，不真的跑
  python ddtad_parallel.py --ngpu 4 --dry_run
"""

import os, sys, json, time, argparse, subprocess, glob

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddtad_run as D


def list_channels(root, labels, only=None):
    chans = sorted(c for c in labels
                   if os.path.exists(os.path.join(root, "train", f"{c}.npy"))
                   and os.path.exists(os.path.join(root, "test", f"{c}.npy")))
    if only:
        want = [c.strip() for c in only.split(",") if c.strip()]
        chans = [c for c in want if c in chans]
    return chans


def shard_by_cost(chans, root, n_shard):
    """按测试序列长度做贪心均衡分片，返回 list[list[chan]]"""
    cost = {}
    for c in chans:
        try:
            cost[c] = int(np.load(os.path.join(root, "test", f"{c}.npy"), mmap_mode="r").shape[0])
        except Exception:
            cost[c] = 1000
    order = sorted(chans, key=lambda c: -cost[c])
    shards = [[] for _ in range(n_shard)]
    load = [0] * n_shard
    for c in order:
        i = int(np.argmin(load))
        shards[i].append(c)
        load[i] += cost[c]
    for s in shards:
        s.sort()
    return shards, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prog", default="ddtad_run.py", help="ddtad_run.py 或 ddtad_probe.py")
    ap.add_argument("--data_dir", default="/mnt/sdb/home/liuqr/Dataset")
    ap.add_argument("--out_dir", default="out/all")
    ap.add_argument("--ngpu", type=int, default=0, help="使用前几张卡（0=自动取全部可见卡）")
    ap.add_argument("--workers", type=int, default=0,
                    help="总进程数（默认=ngpu；设为 2*ngpu 可在小模型上超订，可能更快）")
    ap.add_argument("--channels", default="", help="限定通道（逗号分隔），默认全部")
    ap.add_argument("--extra", default="", help="透传给子程序的额外参数（字符串）")
    ap.add_argument("--amp", action="store_true", help="给子程序加 --amp")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    dev = D.pick_device()
    ngpu = args.ngpu or (torch_count())
    workers = args.workers or ngpu
    args.out_dir = os.path.abspath(args.out_dir)
    args.data_dir = os.path.abspath(args.data_dir)
    prog = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.prog)
    if not os.path.isfile(prog):
        raise SystemExit(f"找不到子程序: {prog}")

    root, csv_path = D.resolve_paths(args.data_dir)
    labels = D.load_labels(csv_path)
    chans = list_channels(root, labels, args.channels or None)
    if not chans:
        raise SystemExit("没有可用通道")
    shards, load = shard_by_cost(chans, root, workers)

    print("=" * 90)
    print(f"程序       : {args.prog}")
    print(f"数据根     : {root}")
    print(f"设备       : {D.describe_device(dev)}")
    print(f"通道数     : {len(chans)}   进程数: {workers}   可见卡数: {torch_count()}")
    print(f"透传参数   : {args.extra}  {'--amp' if args.amp else ''}")
    print("-" * 90)
    for i, s in enumerate(shards):
        print(f"  shard{i} (GPU{i % max(1, torch_count())}, {len(s):2d} 通道, 负载 {load[i]:7d} 点): "
              f"{','.join(s) if len(s) <= 14 else ','.join(s[:14]) + ',...'}")
    print("=" * 90)
    if args.dry_run:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    extra = args.extra.split() if args.extra.strip() else []
    if args.amp and "--amp" not in extra:
        extra.append("--amp")

    procs = []
    t0 = time.time()
    for i, s in enumerate(shards):
        if not s:
            continue
        gpu = i % max(1, torch_count())
        shard_dir = os.path.join(args.out_dir, f"shard{i}")
        os.makedirs(shard_dir, exist_ok=True)
        # --channels / --out_dir 放在最后，保证分片参数一定生效（argparse 后出现的同名参数优先）
        cmd = [sys.executable, "-u", prog, "--data_dir", args.data_dir] + extra + \
              ["--channels", ",".join(s), "--out_dir", shard_dir]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        log = open(os.path.join(args.out_dir, f"shard{i}.log"), "w", encoding="utf-8")
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=os.path.dirname(prog))
        procs.append((i, gpu, p, log, shard_dir, s))
        print(f"[启动] shard{i} -> GPU{gpu}  日志: {args.out_dir}/shard{i}.log", flush=True)

    print(f"\n全部启动，等待完成（可另开终端 tail -f {args.out_dir}/shard0.log 看进度）...\n", flush=True)
    fail = []
    while procs:
        time.sleep(10)
        still = []
        for item in procs:
            i, gpu, p, log, shard_dir, s = item
            if p.poll() is None:
                still.append(item)
            else:
                log.close()
                flag = "OK " if p.returncode == 0 else f"退出码={p.returncode}"
                print(f"[完成] shard{i} (GPU{gpu}) {flag}  用时 {time.time()-t0:.0f}s 累计", flush=True)
                if p.returncode != 0:
                    fail.append(i)
        procs = still

    # ---------------- 合并 ----------------
    print("\n" + "=" * 90)
    if args.prog.endswith("ddtad_run.py"):
        merged = {}
        for i in range(workers):
            rj = os.path.join(args.out_dir, f"shard{i}", "results.json")
            if not os.path.isfile(rj):
                continue
            with open(rj, encoding="utf-8") as f:
                d = json.load(f)
            for cid, v in d.get("per_channel", {}).items():
                merged.setdefault(cid, []).extend(v)
        if merged:
            D.write_summary(merged, args.out_dir, hyper=dict(extra=args.extra, ngpu=ngpu, workers=workers),
                            title=f"DDTAD 多卡汇总（{workers} 进程 / {len(merged)} 通道）")
        else:
            print("没有找到任何 shard 的 results.json，无法合并")
    else:
        # 探针：把各分片的 probe_report.txt 拼起来
        outp = os.path.join(args.out_dir, "probe_report_all.txt")
        n = 0
        with open(outp, "w", encoding="utf-8") as w:
            for i in range(workers):
                f = os.path.join(args.out_dir, f"shard{i}", "probe_report.txt")
                if os.path.isfile(f):
                    w.write(f"\n\n########## shard{i} ##########\n")
                    w.write(open(f, encoding="utf-8").read())
                    n += 1
        print(f"已合并 {n} 个分片报告 -> {outp}")
    if fail:
        print(f"[注意] shard {fail} 非零退出，请查看对应 shardN.log")
    print(f"总耗时 {time.time()-t0:.0f}s")


def torch_count():
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


if __name__ == "__main__":
    main()
