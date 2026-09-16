# -*- coding: utf-8 -*-
"""
ddtad_probe.py —— DDTAD 诊断探针（一次性回答所有关键问题，输出可直接回传）

它回答的问题：
  Q1 数据侧：训练段/测试段各有多少"恒定的死列"？异常到底体现在哪些变量上？
             （决定"对变量取均值"的打分方式会不会把异常稀释掉）
  Q2 模型侧：去噪网络到底学会了吗？eps-MSE 随 t 的变化？重建相对误差多大？
  Q3 噪声侧：t_start（有效加噪步数）取多少时，"正常重建误差"和"异常重建误差"分得最开？
  Q4 打分侧：paper(对变量取均值) / zval_mean / zval_max / zself_mean / zself_max 哪种最好？
  Q5 阈值侧：论文式(13)的验证集阈值 vs 测试集自稳健阈值，哪个不会"全判异常"？
  Q6 天花板：阈值扫描能拿到的 oracle F1 上界是多少？（判断这条路是否还值得走）

输出（全部在 --out_dir 下）：
  probe_report.txt   可读报告（把整个文件回传即可）
  probe.json         结构化结果
  arrays/<chan>_it<iters>.npz   逐点逐变量重构误差 [T,V]，供后续离线分析
  plots/<chan>_it<iters>_*.png  图

服务器用法（建议先跑这一条）：
  python ddtad_probe.py --data_dir /mnt/sdb/home/liuqr/Dataset \
      --channels E-1,P-1,D-1,T-4,M-1,C-1 --iters_list 6000 --out_dir probe_out
"""

import os, sys, json, time, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddtad_run as D


# ---------------------------------------------------------------------
def data_diagnostics(tr_raw, te_raw, anom, sw):
    """纯数据侧诊断：死列情况 + 异常体现在哪些变量上（用训练段 z 标准化）"""
    mu = tr_raw.mean(0); sd = tr_raw.std(0)
    live = sd > 1e-9
    dead_tr = int((~live).sum())
    dead_te = int((te_raw.std(0) <= 1e-9).sum())
    dead_but_alive = int(((~live) & (te_raw.std(0) > 1e-9)).sum())
    g = D.gt_points(anom, len(te_raw))
    z = np.zeros_like(te_raw, dtype=np.float64)
    if live.any():
        z[:, live] = (te_raw[:, live] - mu[live]) / sd[live]
    az = np.abs(z)
    inside = az[g == 1].mean(0) if (g == 1).any() else np.zeros(az.shape[1])
    outside = az[g == 0].mean(0) if (g == 0).any() else np.ones(az.shape[1])
    ratio = inside / np.maximum(outside, 1e-6)
    order = np.argsort(-ratio)[:5]
    top = [(int(v), float(ratio[v]), float(inside[v]), float(outside[v])) for v in order]
    return dict(dead_train=dead_tr, dead_test=dead_te, dead_but_alive=dead_but_alive,
                n_live=int(live.sum()), top_vars=top,
                anom_frac=float(g.mean()), n_anom=int(g.sum()), n_points=int(len(te_raw)))


@torch.inference_mode()
def eps_mse_by_t(model, wins, ts, alpha_bar, dev, batch=256, nrep=2):
    out = []
    for tl in ts:
        tot = 0.0
        for i in range(0, len(wins), batch):
            x0 = wins[i:i + batch].to(dev)
            acc = 0.0
            for _ in range(nrep):
                eps = torch.randn_like(x0)
                ab = alpha_bar[tl].view(-1, 1, 1)
                xt = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
                tt = torch.full((len(x0),), tl, dtype=torch.long, device=dev)
                acc += nn.functional.mse_loss(model(xt, tt), eps).item()
            tot += acc / nrep * len(x0)
        out.append(tot / max(1, len(wins)))
    return out


# ---------------------------------------------------------------------
def run_one(cid, iters, policy, args, labels, root, dev, sched, rep):
    beta, alpha, abar = sched
    tr_raw = np.load(os.path.join(root, "train", f"{cid}.npy")).astype(np.float64)
    te_raw = np.load(os.path.join(root, "test", f"{cid}.npy")).astype(np.float64)
    anom = labels.get(cid, (None, None, []))[2]
    Vraw = tr_raw.shape[1]
    tag = f"{cid}_it{iters}_{policy}"

    ddat = data_diagnostics(tr_raw, te_raw, anom, args.sw)
    rep(f"\n{'='*130}")
    rep(f"### {cid} | 策略={policy} | 变量={Vraw} 活列={ddat['n_live']} | 测试点={ddat['n_points']} "
        f"异常点={ddat['n_anom']} ({ddat['anom_frac']*100:.2f}%) | 类={labels.get(cid,(None,'?'))[1]}")
    rep(f"[Q1 数据] 训练段恒定列={ddat['dead_train']}  测试段恒定列={ddat['dead_test']}  "
        f"训练恒定但测试有变化={ddat['dead_but_alive']}")
    rep("[Q1 数据] 异常最敏感变量 top5 (变量号, 异常区|z|/正常区|z|, 异常区|z|, 正常区|z|): " +
        "  ".join(f"v{v}:{r:.2f}x({i:.2f}/{o:.2f})" for v, r, i, o in ddat["top_vars"]))

    st = D.fit_norm(tr_raw, te_raw, policy=policy)
    tr, te = D.apply_norm(tr_raw, st), D.apply_norm(te_raw, st)
    V = tr.shape[1]
    rep(f"[Q1 归一化] 策略={policy}: {Vraw} 列 -> 送入网络 {V} 列（两边都恒定删掉 {Vraw-V} 列）")
    n_val = int(len(tr) * args.val_frac); n_tr = len(tr) - n_val
    all_w = D.make_windows(tr[:n_tr], args.sw, args.train_ss)
    va_w = D.make_windows(tr[n_tr:], args.sw, args.test_ss)
    te_w = D.make_windows(te, args.sw, args.test_ss)
    g_pt = D.gt_points(anom, len(te))

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = D.UNet1D(V, base=args.base, emb_dim=args.emb,
                     ch_mult=tuple(int(x) for x in args.ch_mult.split(","))).to(dev)
    rep(f"[进度] {cid}: ★开始训练★ {len(all_w)} 个训练窗口 × {iters} 次迭代 "
        f"(base={args.base}, 参数量={sum(p.numel() for p in model.parameters())/1e3:.1f}K) ...")
    D.train_model(model, all_w, beta, abar, dev, iters=iters, batch=args.batch, lr=args.lr,
                  ema_decay=args.ema, warmup=max(50, iters // 20),
                  log_every=max(500, iters // 6), tag=tag, seed=args.seed, amp=args.amp)

    # ---- Q2 模型是否学会 ----
    ts_probe = [0, 50, 100, 200, 400, 600, 800, min(999, len(beta) - 1)]
    em = eps_mse_by_t(model, all_w[:256], ts_probe, abar, dev)
    rep("[Q2 模型] eps-MSE by t: " + "  ".join(f"t{t}={v:.4f}" for t, v in zip(ts_probe, em)) +
        "   （1.0 = 完全没学到；<0.1 = 学得不错）")

    # 验证段/测试段的重构（每个 t_start）
    res = dict(channel=cid, iters=iters, col_policy=policy, V_raw=Vraw, V=V,
               n_live=ddat["n_live"], data=ddat,
               eps_mse={str(t): v for t, v in zip(ts_probe, em)}, t_starts={})
    rep("")
    hdr = (f"{'t_start':>7} {'abar':>7} {'残留%':>6} | {'score':>11} | "
           f"{'val_mu':>9} {'val_sd':>9} | {'tstNorm':>9} {'tstAnom':>9} {'gap':>7} | "
           f"{'η=2 P':>7} {'η=2 R':>7} {'η=2 F1':>7} {'PA':>7} | "
           f"{'η=3 F1':>7} {'PA':>7} | {'η=4 F1':>7} {'PA':>7} | "
           f"{'VR2PA':>6} {'VR3PA':>6} | {'oracle':>7} {'predPos%':>8}")
    rep(hdr); rep("-" * len(hdr))

    E_store = {}
    for t_start in args.t_starts:
        t_start = int(min(t_start, len(beta)))
        vh = D.reconstruct(model, va_w, beta, alpha, abar, t_start, dev, args.sampler,
                           args.n_sample, args.batch, args.seed)
        E_val = D.aggregate_point_errors((vh - va_w).pow(2).numpy(), len(tr) - n_tr, args.sw, args.test_ss, V)
        th = D.reconstruct(model, te_w, beta, alpha, abar, t_start, dev, args.sampler,
                           args.n_sample, args.batch, args.seed)
        E_test = D.aggregate_point_errors((th - te_w).pow(2).numpy(), len(te), args.sw, args.test_ss, V)
        E_store[t_start] = (E_val.astype(np.float32), E_test.astype(np.float32))

        # 重建相对误差（正常点）—— 衡量"重建器到底有多准"
        rel = float(np.sqrt(E_test[g_pt == 0].mean() / max(te[g_pt == 0].var(), 1e-12))) if (g_pt == 0).any() else float("nan")

        ab = float(abar[t_start - 1])
        entry = dict(abar=ab, rel_rmse_normal=rel, scores={})
        for smode in args.scores:
            s_val = D.make_score(E_val, E_val, smode)
            s_test = D.make_score(E_test, E_val, smode)
            normal = s_test[g_pt == 0]; anom_s = s_test[g_pt == 1] if (g_pt == 1).any() else np.array([0.0])
            orc = D.oracle_best_f1(s_test, g_pt)
            e = dict(val_mu=float(s_val.mean()), val_sd=float(s_val.std()),
                     tst_normal=float(normal.mean()), tst_anom=float(anom_s.mean()),
                     gap=float(anom_s.mean() / max(normal.mean(), 1e-12)),
                     oracle=orc, eta={}, eta_robust={})
            # 论文式(13) 验证集阈值 μ+ησ
            for et in [2.0, 3.0, 4.0]:
                thr = s_val.mean() + et * s_val.std()
                pred = (s_test > thr).astype(np.int64)
                mp, mpa = D.eval_with_pa(pred, g_pt)
                e["eta"][str(et)] = dict(thresh=float(thr), pred_pos=float(pred.mean()),
                                         plain=mp, pa=mpa)
            # 验证段"稳健"阈值 中位数+η·1.4826MAD（抗离群，修正 val_sd 被极端值撑大的问题）
            mu_vr, sd_vr = D.robust_stats(s_val)
            for et in [2.0, 3.0, 4.0]:
                thr = mu_vr + et * sd_vr
                pred = (s_test > thr).astype(np.int64)
                mp, mpa = D.eval_with_pa(pred, g_pt)
                e["eta_robust"][str(et)] = dict(thresh=float(thr), pred_pos=float(pred.mean()),
                                                plain=mp, pa=mpa)
            # 测试集自稳健阈值
            mu_r, sd_r = D.robust_stats(s_test)
            pred = (s_test > mu_r + args.eta * sd_r).astype(np.int64)
            mp, mpa = D.eval_with_pa(pred, g_pt)
            e["test_robust"] = dict(thresh=float(mu_r + args.eta * sd_r), pred_pos=float(pred.mean()),
                                    plain=mp, pa=mpa)
            entry["scores"][smode] = e

            a2 = e["eta"]["2.0"]; a3 = e["eta"]["3.0"]; a4 = e["eta"]["4.0"]
            v2 = e["eta_robust"]["2.0"]; v3 = e["eta_robust"]["3.0"]
            rep(f"{t_start:>7} {ab:>7.4f} {100*np.sqrt(max(ab,0)):>6.1f} | {smode:>11} | "
                f"{e['val_mu']:>9.5f} {e['val_sd']:>9.5f} | {e['tst_normal']:>9.5f} {e['tst_anom']:>9.5f} "
                f"{e['gap']:>7.3f} | {a2['plain']['P']:>7.3f} {a2['plain']['R']:>7.3f} {a2['plain']['F1']:>7.3f} "
                f"{a2['pa']['F1']:>7.3f} | {a3['plain']['F1']:>7.3f} {a3['pa']['F1']:>7.3f} | "
                f"{a4['plain']['F1']:>7.3f} {a4['pa']['F1']:>7.3f} | "
                f"{v2['pa']['F1']:>6.3f} {v3['pa']['F1']:>6.3f} | {orc['F1']:>7.3f} "
                f"{100*e['test_robust']['pred_pos']:>7.1f}%")
        entry["rel_rmse_normal"] = rel
        res["t_starts"][str(t_start)] = entry

    # 逐变量：异常点 vs 正常点的重构误差（判断"对变量取均值"稀释了多少）
    rep("")
    rep("[Q4 逐变量] 各 t_start 下按 (异常区误差 / 正常区误差) 排序的 top5 变量 "
        "—— 若 top 变量的比值很高但 gap 接近 1，说明是'对变量取均值'把异常稀释了")
    for t_start in args.t_starts:
        t_start = int(min(t_start, len(beta)))
        if t_start not in E_store:
            continue
        E_test = E_store[t_start][1]
        nm, am = E_test[g_pt == 0].mean(0), (E_test[g_pt == 1].mean(0) if (g_pt == 1).any()
                                            else np.zeros(E_test.shape[1]))
        ratio = am / np.maximum(nm, 1e-12)
        order = np.argsort(-ratio)[:5]
        res["t_starts"][str(t_start)]["var_ratio_top5"] = \
            [[int(v), float(ratio[v])] for v in order]
        rep(f"  t_start={t_start:>4}: " + "  ".join(f"v{v}:{ratio[v]:.2f}x" for v in order))

    if args.save_arrays:
        os.makedirs(os.path.join(args.out_dir, "arrays"), exist_ok=True)
        payload = {"g_pt": g_pt}
        for ts_key, (ev, et) in E_store.items():
            payload[f"Eval_{ts_key}"] = ev
            payload[f"Etest_{ts_key}"] = et
        np.savez_compressed(os.path.join(args.out_dir, "arrays", f"{tag}.npz"), **payload)
    rep(f"[进度] {cid} 策略={policy} iters={iters}: ✔全部完成（训练 + {len(args.t_starts)} 个 t_start × "
        f"{len(args.scores)} 种打分的评估）")
    return res


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/mnt/sdb/home/liuqr/Dataset")
    ap.add_argument("--channels", default="E-1,P-1,D-1,T-4,M-1,C-1")
    ap.add_argument("--gpu", default="", help="指定显卡编号 0/1/2/3 或 cpu（默认自动）")
    ap.add_argument("--amp", action="store_true", help="开启混合精度训练")
    ap.add_argument("--iters_list", default="6000", help="逗号分隔，多个值=多组训练预算（诊断欠训练）")
    ap.add_argument("--col_policies", default="drop,zero,testscale",
                    help="恒列处理策略的组合：drop=删掉两边都恒定的列（新默认）；"
                         "zero=恒列置0（对照，即上次探针的配置）；testscale=恒列用测试段尺度")
    ap.add_argument("--t_starts", default="50,100,150,200,300,400,600")
    ap.add_argument("--scores", default="paper,zval_mean,zval_max,zself_mean,zself_max")
    ap.add_argument("--out_dir", default="probe_out")
    ap.add_argument("--sw", type=int, default=128)
    ap.add_argument("--train_ss", type=int, default=10)
    ap.add_argument("--test_ss", type=int, default=16)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--n_steps", type=int, default=1000)
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    ap.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    ap.add_argument("--n_sample", type=int, default=2)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--ch_mult", default="1,2,4")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eta", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_dead_cols", action="store_true", help="[已废弃] 等价于把 testscale 放进策略列表")
    ap.add_argument("--save_arrays", action="store_true", default=True)
    ap.add_argument("--no_save_arrays", dest="save_arrays", action="store_false")
    args = ap.parse_args()
    args.t_starts = [int(x) for x in str(args.t_starts).split(",") if x.strip()]
    args.scores = [x for x in str(args.scores).split(",") if x.strip()]
    policies = [x for x in str(args.col_policies).split(",") if x.strip()]
    if args.keep_dead_cols and "testscale" not in policies:
        policies.append("testscale")
    iters_list = [int(x) for x in str(args.iters_list).split(",") if x.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    LOG = open(os.path.join(args.out_dir, "probe_report.txt"), "w", encoding="utf-8")

    def rep(s=""):
        print(s, flush=True)
        LOG.write(s + "\n"); LOG.flush()

    dev = D.pick_device(args.gpu)
    root, csv_path = D.resolve_paths(args.data_dir)
    labels = D.load_labels(csv_path)
    rep(f"设备={D.describe_device(dev)} torch={torch.__version__} 数据根={root}")
    rep(f"标签={csv_path} 通道数={len(labels)}")
    rep(f"配置: iters_list={iters_list} col_policies={policies} t_starts={args.t_starts} "
        f"scores={args.scores} sampler={args.sampler} n_sample={args.n_sample} base={args.base} "
        f"调度={args.schedule} T={args.n_steps}")

    sched = D.build_schedule(args.schedule, args.n_steps, dev)
    all_res = {}
    t00 = time.time()
    for cid in [c for c in args.channels.split(",") if c.strip()]:
        if not (os.path.exists(os.path.join(root, "train", f"{cid}.npy"))
                and os.path.exists(os.path.join(root, "test", f"{cid}.npy"))):
            rep(f"[跳过] {cid} 数据不存在"); continue
        for it in iters_list:
            for pol in policies:
                key = f"{cid}_it{it}_{pol}"
                try:
                    all_res[key] = run_one(cid, it, pol, args, labels, root, dev, sched, rep)
                except Exception:
                    import traceback
                    rep(f"[失败] {cid} iters={it} policy={pol}"); rep(traceback.format_exc())
    rep(f"\n总耗时 {time.time()-t00:.0f}s")

    with open(os.path.join(args.out_dir, "probe.json"), "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False, default=float)
    rep("已保存: probe_report.txt / probe.json")
    LOG.close()


if __name__ == "__main__":
    main()
