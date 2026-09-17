# -*- coding: utf-8 -*-
"""
ddtad_parallel.py —— 多卡并行调度器（通道级并行，卡数自动探测）

为什么是"通道级并行"而不是 DDP：
    本任务的模型很小（约 1~3 M 参数）、单通道训练数据只有几千个点，
    一个通道就是一个独立模型。用 DDP 把一个小模型拆到 N 张卡上收益极低，
    而 **不同通道之间完全独立** —— 每个通道绑一张卡并行跑，才是真正的 N 倍加速。
    （81 个通道：单卡约 3.5 小时 → 4 卡约 55 分钟 → 5 卡约 45 分钟）

卡数不需要手动指定：
    --ngpu 0（默认）= 自动取 torch.cuda.device_count()，会尊重 CUDA_VISIBLE_DEVICES。
    所以 4 卡、5 卡、8 卡都直接能跑，换机器不用改代码。

它会：
  1. 读取 labeled_anomalies.csv，列出全部通道，按"测试序列长度"做负载均衡分片
     （重构耗时 ∝ 测试点数 × t_start，所以按 test_len 贪心分配，避免某张卡提前空闲）
  2. 每个分片起一个子进程，设置 CUDA_VISIBLE_DEVICES=<gpu>，日志写入 shardN.log
  3. 全部结束后合并 results.json，复用 ddtad_run.write_summary 产出全局报告

用法：
  # 全量 SMAP+MSL，自动用全部可见卡
  python ddtad_parallel.py --prog ddtad_run.py --out_dir out/all \
      --extra "--iters 6000 --t_start 50 --col_policy drop --score paper"

  # 诊断探针并行
  python ddtad_parallel.py --prog ddtad_probe.py --out_dir probe_out \
      --channels E-1,P-1,D-1,T-4,M-1,C-1 --extra "--iters_list 6000 --t_starts 50,100,200"

  # 只看分片方案，不真的跑
  python ddtad_parallel.py --ngpu 0 --dry_run

  # 每卡开 2 个进程（小模型常受 kernel launch 限制，超订有时更快）
  python ddtad_parallel.py --workers 10 --prog ddtad_run.py --out_dir out/all
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

    # 注意：父进程这里刻意**不**初始化 CUDA 上下文（不用 pick_device），
    # 否则父进程会在 GPU0 上占一份显存，而 GPU0 同时还要跑 shard0。
    n_gpu = torch_count()
    ngpu = args.ngpu or n_gpu
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
    print(f"GPU        : {gpu_banner()}")
    print(f"通道数     : {len(chans)}   进程数: {workers}   使用卡数: {min(workers, n_gpu)}/{n_gpu}")
    print(f"透传参数   : {args.extra}  {'--amp' if args.amp else ''}")
    print("-" * 90)
    for i, s in enumerate(shards):
        print(f"  shard{i} (GPU{i % max(1, n_gpu)}, {len(s):2d} 通道, 负载 {load[i]:7d} 点): "
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
        gpu = i % max(1, n_gpu)
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
    # 进度标记：每完成一个通道，子程序会打印这一串
    done_token = "✔全部完成" if args.prog.endswith("ddtad_probe.py") else "分离度:"
    todo = {i: len(s) for (i, gpu, p, log, shard_dir, s) in procs}
    last_report = 0.0

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

        # 每 60s 打一次进度：各分片已完成 / 总数，方便判断是否卡住
        if procs and time.time() - last_report >= 60:
            last_report = time.time()
            parts = []
            tot_done = 0
            for i, n in sorted(todo.items()):
                d = _count_in(os.path.join(args.out_dir, f"shard{i}.log"), done_token)
                d = min(d, n)
                tot_done += d
                parts.append(f"s{i}:{d}/{n}")
            el = time.time() - t0
            eta = (el / tot_done * (sum(todo.values()) - tot_done)) if tot_done else float("nan")
            print(f"[进度] {el:5.0f}s  已完成 {tot_done}/{sum(todo.values())}   "
                  f"{'  '.join(parts)}   ETA≈{eta/60:.1f}min", flush=True)

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


def _count_in(path, token):
    """统计日志里出现 token 的行数（子进程正在写也能安全读）"""
    try:
        n = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if token in line:
                    n += 1
        return n
    except Exception:
        return 0


def torch_count():
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def gpu_banner():
    """枚举可见 GPU（不创建 CUDA 上下文、不占用显存）"""
    try:
        import torch
        n = torch.cuda.device_count()
        if n == 0:
            return "未检测到 CUDA 设备（将回退 CPU，速度会慢很多）"
        names = []
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            names.append(f"cuda:{i}={p.name}({p.total_memory/1e9:.0f}G)")
        return f"{n} 张: " + "  ".join(names)
    except Exception as e:
        return f"枚举失败: {e}"


if __name__ == "__main__":
    main()
