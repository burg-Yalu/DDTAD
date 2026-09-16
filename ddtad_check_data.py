# -*- coding: utf-8 -*-
"""
ddtad_check_data.py —— 纯数据侧体检（不需要 GPU，几秒钟跑完全部通道）

作用：先确认"数据/标签这一层"有没有问题，再谈模型。输出每通道：
  - 变量数 / 训练段活列数 / 测试段活列数 / "训练恒定但测试有变化"的列数
  - 异常点数量与占比、异常区间
  - 异常可见度：用训练段 μ,σ 对测试段做 z 标准化后，
        ratio_v = mean|z|(异常区) / mean|z|(正常区)
    top1 变量的 ratio、以及异常总偏差中 top1 变量占的比重
    （若 top1 ratio 只有 1~2 倍，说明异常在数据上就几乎不可见，任何方法都难做）
  - 标记：异常是否落在"训练段恒定、测试段才有变化"的列上（这类列若被置 0 就丢信息了）

输出： check_out/data_check.csv 与 data_check_report.txt

用法：
  python ddtad_check_data.py --data_dir /mnt/sdb/home/liuqr/Dataset --out_dir check_out
"""

import os, sys, csv, json, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddtad_run as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/mnt/sdb/home/liuqr/Dataset")
    ap.add_argument("--out_dir", default="check_out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    root, csv_path = D.resolve_paths(args.data_dir)
    labels = D.load_labels(csv_path)
    rows = []
    for cid, (sc, cls, anom) in sorted(labels.items()):
        ftr = os.path.join(root, "train", f"{cid}.npy")
        fte = os.path.join(root, "test", f"{cid}.npy")
        if not (os.path.isfile(ftr) and os.path.isfile(fte)):
            continue
        tr = np.load(ftr).astype(np.float64)
        te = np.load(fte).astype(np.float64)
        mu, sd = tr.mean(0), tr.std(0)
        live_tr = sd > 1e-9
        live_te = te.std(0) > 1e-9
        dead_but_alive = (~live_tr) & live_te
        g = D.gt_points(anom, len(te))
        z = np.zeros_like(te)
        if live_tr.any():
            z[:, live_tr] = (te[:, live_tr] - mu[live_tr]) / sd[live_tr]
        az = np.abs(z)
        inside = az[g == 1].mean(0) if (g == 1).any() else np.zeros(az.shape[1])
        outside = az[g == 0].mean(0) if (g == 0).any() else np.ones(az.shape[1])
        ratio = inside / np.maximum(outside, 1e-6)
        ratio[~live_tr] = np.nan                      # 训练恒定列无定义
        order = np.argsort(-np.nan_to_num(ratio, nan=-1))
        top1 = int(order[0]) if len(order) else -1
        inside_sum = inside.sum()
        rows.append(dict(
            channel=cid, spacecraft=sc, cls=cls, n_var=tr.shape[1],
            train_len=len(tr), test_len=len(te),
            live_train=int(live_tr.sum()), live_test=int(live_te.sum()),
            dead_but_alive=int(dead_but_alive.sum()),
            keep_drop=int((live_tr | dead_but_alive).sum()),   # --col_policy drop 下保留的列数
            all_dead_train=bool(live_tr.sum() == 0),           # 训练段全恒定 -> 必须 testscale
            n_anom=int(g.sum()), anom_frac=float(g.mean()),
            n_seg=len(anom), segs=str(anom),
            top1_var=top1,
            top1_ratio=float(ratio[top1]) if top1 >= 0 else float("nan"),
            top1_share=float(inside[top1] / inside_sum) if inside_sum > 0 else float("nan"),
            mean_ratio_live=float(np.nanmean(ratio[live_tr])) if live_tr.any() else float("nan"),
            anom_on_dead_col=bool(dead_but_alive.any() and
                                  np.nan_to_num(ratio, nan=0)[dead_but_alive].max() >
                                  np.nanmax(np.nan_to_num(ratio, nan=0)) * 0.5),
        ))

    rows.sort(key=lambda r: -np.nan_to_num(r["top1_ratio"], nan=0))
    cols = list(rows[0].keys())
    with open(os.path.join(args.out_dir, "data_check.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = []
    lines.append(f"数据根: {root}")
    lines.append(f"标签:   {csv_path}")
    lines.append(f"通道数: {len(rows)}")
    lines.append("")
    lines.append("按'异常可见度 top1_ratio'降序（top1_ratio<1.5 基本可认为该通道对重建类方法不可解）")
    lines.append(f"{'chan':>6} {'sc':>5} {'cls':>11} {'nVar':>5} {'liveTr':>6} {'liveTe':>6} "
                 f"{'deadAlive':>9} {'keepDrop':>8} {'allDead':>7} {'anom%':>6} {'nSeg':>4} {'top1v':>5} "
                 f"{'top1ratio':>9} {'top1share':>9} {'meanRatio':>9} {'onDead':>6}")
    for r in rows:
        lines.append(f"{r['channel']:>6} {r['spacecraft']:>5} {str(r['cls'])[:11]:>11} {r['n_var']:>5} "
                     f"{r['live_train']:>6} {r['live_test']:>6} {r['dead_but_alive']:>9} "
                     f"{r['keep_drop']:>8} {str(r['all_dead_train']):>7} "
                     f"{r['anom_frac']*100:>6.2f} {r['n_seg']:>4} {r['top1_var']:>5} "
                     f"{r['top1_ratio']:>9.3f} {r['top1_share']:>9.3f} {r['mean_ratio_live']:>9.3f} "
                     f"{str(r['anom_on_dead_col']):>6}")
    # 汇总统计
    tr_ = np.array([r["top1_ratio"] for r in rows], dtype=float)
    lines.append("")
    lines.append(f"top1_ratio 分布: min={np.nanmin(tr_):.2f} 25%={np.nanpercentile(tr_,25):.2f} "
                 f"中位={np.nanmedian(tr_):.2f} 75%={np.nanpercentile(tr_,75):.2f} max={np.nanmax(tr_):.2f}")
    for th in [1.2, 1.5, 2.0, 3.0, 5.0]:
        lines.append(f"  top1_ratio > {th}: {int((tr_ > th).sum())}/{len(tr_)} 个通道")
    lines.append("")
    lines.append("MSL 平均活列数: %.1f / %d   （--col_policy drop 下平均保留 %.1f 列）" % (
        np.mean([r["live_train"] for r in rows if r["spacecraft"] == "MSL"]),
        int(np.mean([r["n_var"] for r in rows if r["spacecraft"] == "MSL"])),
        np.mean([r["keep_drop"] for r in rows if r["spacecraft"] == "MSL"])))
    lines.append("SMAP 平均活列数: %.1f / %d   （--col_policy drop 下平均保留 %.1f 列）" % (
        np.mean([r["live_train"] for r in rows if r["spacecraft"] == "SMAP"]),
        int(np.mean([r["n_var"] for r in rows if r["spacecraft"] == "SMAP"])),
        np.mean([r["keep_drop"] for r in rows if r["spacecraft"] == "SMAP"])))
    warn = [r["channel"] for r in rows if r["all_dead_train"]]
    lines.append(f"训练段全部列恒定（归一化后会变成全 0 输入，必须用 testscale）: {warn}")
    txt = "\n".join(lines)
    with open(os.path.join(args.out_dir, "data_check_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"\n已保存: {args.out_dir}/data_check.csv | data_check_report.txt")


if __name__ == "__main__":
    main()
