# -*- coding: utf-8 -*-
"""
ddtad_smoke.py —— 本地端到端冒烟测试（纯 CPU，约 1 分钟，不需要数据集）

为什么需要它：
    我们已经连续踩过 3 个"只在真正跑起来才暴露"的低级错误：
      · 窗口标签摊回点级时长度不匹配（8516 vs 8448）
      · np.sqrt() 直接作用在 CUDA tensor 上
      · oracle_best() 返回里缺 TP/FP/FN，汇总表 KeyError
      · t_start_sweep 用了 int 键，而多卡合并走 JSON 后键变成 str，表格空白
    这些在"改完直接打包上传"的流程里发现不了，代价是用户每次等 5 分钟才报错。
    本脚本用合成数据把 run_channel + write_summary + JSON 往返合并全部走一遍。

用法：
    python ddtad_smoke.py            # 默认 CPU
    python ddtad_smoke.py --device cuda:0
"""

import os, sys, csv, json, time, shutil, subprocess, argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np


def make_fake_dataset(root, seed=0):
    rng = np.random.RandomState(seed)
    os.makedirs(os.path.join(root, "data", "data", "train"), exist_ok=True)
    os.makedirs(os.path.join(root, "data", "data", "test"), exist_ok=True)
    chans = {"X-1": (5, 600, 1200), "X-2": (7, 500, 1150)}
    for cid, (V, Ttr, Tte) in chans.items():
        t = np.linspace(0, 20 * np.pi, Ttr)
        tr = np.stack([np.sin(t + i) + 0.05 * rng.randn(Ttr) for i in range(V)], 1)
        t2 = np.linspace(0, 20 * np.pi, Tte)
        te = np.stack([np.sin(t2 + i) + 0.05 * rng.randn(Tte) for i in range(V)], 1)
        te[500:560, 0] += 3.0                       # 注入异常段
        for arr, sub in ((tr, "train"), (te, "test")):
            arr = arr.astype(np.float64)
            arr[:, 1] = 0.7                         # "两边都恒定"的列 -> 测 drop 策略
            arr[:, 2] = 0.0
            np.save(os.path.join(root, "data", "data", sub, f"{cid}.npy"), arr)
    with open(os.path.join(root, "labeled_anomalies.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chan_id", "spacecraft", "class", "num_values", "anomaly_sequences"])
        w.writerow(["X-1", "SMAP", "[point]", 1200, "[[500, 559]]"])
        w.writerow(["X-2", "MSL", "[contextual]", 1150, "[[300, 420]]"])
    return list(chans)


def gpu_bench(device, V=24, base=64, batch=32, iters=40):
    """GPU 实测基准：确认真的在 GPU 上跑，并量化 EMA 优化的收益。
    论文设置 T=100 时模型输入是 [32, 24, 128]，这里用同样形状。"""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ddtad_run as D

    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        print("  (跳过：未指定 CUDA 设备)")
        return
    print(f"  设备: {torch.cuda.get_device_name(dev)}  可见卡数={torch.cuda.device_count()}")
    model = D.UNet1D(V, base=base, emb_dim=128, ch_mult=(1, 2, 4)).to(dev)
    n_t = len(list(model.state_dict().keys()))
    print(f"  参数量={sum(p.numel() for p in model.parameters())/1e6:.2f}M  参数张量数={n_t}")
    opt = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.99))
    ema = D.EMA(model, 0.995)
    x0 = torch.randn(batch, V, 128, device=dev)
    t = torch.randint(0, 100, (batch,), device=dev)
    eps = torch.randn_like(x0)

    def one_step(with_ema=True):
        loss = torch.nn.functional.mse_loss(model(x0, t), eps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if with_ema:
            ema.update(model)

    def timeit(fn, n=iters):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / n * 1000

    full = timeit(lambda: one_step(True))
    no_ema = timeit(lambda: one_step(False))
    print(f"  单次迭代: 含EMA {full:.2f} ms | 不含EMA {no_ema:.2f} ms  "
          f"-> EMA 占比 {100*(full-no_ema)/max(full,1e-9):.1f}%")
    print(f"  显存: allocated={torch.cuda.memory_allocated(dev)/1e9:.2f}G "
          f"reserved={torch.cuda.memory_reserved(dev)/1e9:.2f}G")
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--keep", action="store_true", help="保留临时文件便于排查")
    ap.add_argument("--bench", action="store_true", help="额外跑 GPU 基准（测迭代耗时与 EMA 占比）")
    args = ap.parse_args()

    if args.bench:
        print("[0/3] GPU 基准（论文设置：base=64, batch=32, 输入 [32,24,128]）")
        gpu_bench(args.device)
        print()

    here = os.path.dirname(os.path.abspath(__file__))
    work = os.path.join(here, "_smoke")
    out = os.path.join(work, "out")
    shutil.rmtree(work, ignore_errors=True)
    chans = make_fake_dataset(os.path.join(work, "ds"))
    print(f"[1/3] 合成数据已生成: {os.path.join(work,'ds')}  通道={chans}")

    cmd = [sys.executable, os.path.join(here, "ddtad_run.py"),
           "--data_dir", os.path.join(work, "ds"),
           "--channels", ",".join(chans),
           "--out_dir", out, "--gpu", args.device,
           "--epochs", "3", "--batch", "8", "--n_sample", "1",
           "--t_start_list", "2,5,10", "--t_start", "10",
           "--rounds", "2", "--log_every", "5", "--no_plots"]
    print(f"[2/3] 跑 ddtad_run.py（CPU 端到端）...")
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print("  [FAIL] 失败！stderr 尾部：")
        print(r.stderr[-3000:])
        print("  stdout 尾部：")
        print(r.stdout[-3000:])
        return 1
    print("  [OK] 单进程路径通过（训练 + t_start 扫描 + 变体扫描 + 口径表 + report/csv）")

    print("[3/3] 验证多卡合并路径（从 results.json 还原后重新汇总，键会变成 str）...")
    sys.path.insert(0, here)
    import ddtad_run as D
    d = json.load(open(os.path.join(out, "results.json"), encoding="utf-8"))
    D.write_summary(d["per_channel"], os.path.join(work, "merged"),
                    hyper=d.get("hyper", {}), title="SMOKE-MERGE")

    rep = open(os.path.join(work, "merged", "report.txt"), encoding="utf-8").read()
    checks = [("主指标行", "窗口级 不PA" in rep),
              ("t_start 对照表有数据行", rep.count("  ← 论文") >= 0 and
               sum(1 for L in rep.splitlines() if L.startswith("        ") and "0." in L) > 0),
              ("口径对照表", "评测口径对照" in rep),
              ("方案扫描排行榜", "按【点级" in rep or "排行榜" in rep)]
    ok = True
    for name, passed in checks:
        print(f"  {'[OK]' if passed else '[FAIL]'} {name}")
        ok = ok and passed

    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    print("\n" + ("全部通过 [OK]  可以打包上传了" if ok else "有检查项未通过 [FAIL]  先别上传"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
