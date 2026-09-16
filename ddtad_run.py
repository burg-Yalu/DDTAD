# -*- coding: utf-8 -*-
"""
DDTAD 服务器版（GPU + Linux 路径友好 + 自动出图表）
- 路径：--data_dir 指定 archive 目录（内含 data/data/{train,test} 与 labeled_anomalies.csv）
- GPU：自动检测 cuda（可用则用 GPU）
- 输出：训练 loss 曲线、重建对比、得分分布、检测结果图 + 结果汇总(目录内)
用法（Linux 服务器）：
  python ddtad_run.py --data_dir /path/to/archive --channel E-1 --epochs 2000 --rounds 1 --out_dir ./out
  python ddtad_run.py --data_dir /path/to/archive --channel all --epochs 2000 --rounds 1 --out_dir ./out
依赖：torch, numpy, matplotlib（conda 里通常有）
"""
import os, sys, csv, ast, argparse, json
import numpy as np
import torch
import torch.nn as nn

# 服务器无显示，用 Agg 后端，图片直接保存
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SW, TRAIN_SS, TEST_SS, T = 128, 10, 128, 100
ETA = 2.0
BATCH = 128

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv_block(i, o):
    return nn.Sequential(nn.Conv1d(i,o,3,padding=1), nn.ReLU(True), nn.Conv1d(o,o,3,padding=1), nn.ReLU(True))

class TimeEmbed(nn.Module):
    def __init__(self, d):
        super().__init__(); self.mlp=nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.Linear(d,d)); self.d=d
    def forward(self,t):
        h=self.d//2; dev=t.device
        f=torch.exp(-torch.log(torch.tensor(10000.0,device=dev))*torch.arange(h,dtype=torch.float32,device=dev)/h)
        a=torch.cat([torch.cos(t[:,None].float()*f[None,:]),torch.sin(t[:,None].float()*f[None,:])],-1)
        return self.mlp(a)

class UNet1D(nn.Module):
    def __init__(self, cin, base=16, emb=64):
        super().__init__(); self.te=TimeEmbed(emb)
        self.e1p=nn.Linear(emb,base); self.e2p=nn.Linear(emb,base*2); self.e3p=nn.Linear(emb,base*4)
        self.bp=nn.Linear(emb,base*8); self.d3p=nn.Linear(emb,base*4); self.d2p=nn.Linear(emb,base*2); self.d1p=nn.Linear(emb,base)
        self.e1=conv_block(cin,base); self.dn1=nn.MaxPool1d(2); self.e2=conv_block(base,base*2); self.dn2=nn.MaxPool1d(2)
        self.e3=conv_block(base*2,base*4); self.dn3=nn.MaxPool1d(2); self.bn=conv_block(base*4,base*8)
        self.up3=nn.Upsample(scale_factor=2,mode="nearest"); self.d3=conv_block(base*4+base*8,base*4)
        self.up2=nn.Upsample(scale_factor=2,mode="nearest"); self.d2=conv_block(base*4+base*2,base*2)
        self.up1=nn.Upsample(scale_factor=2,mode="nearest"); self.d1=conv_block(base*2+base,base)
        self.out=nn.Conv1d(base,cin,1)
    def forward(self,x,t):
        te=self.te(t)
        e1=self.e1(x)+self.e1p(te).unsqueeze(-1); d1=self.dn1(e1)
        e2=self.e2(d1)+self.e2p(te).unsqueeze(-1); d2=self.dn2(e2)
        e3=self.e3(d2)+self.e3p(te).unsqueeze(-1); d3=self.dn3(e3)
        b=self.bn(d3)+self.bp(te).unsqueeze(-1)
        u3=self.up3(b); c3=torch.cat([u3,e3],1); x3=self.d3(c3)+self.d3p(te).unsqueeze(-1)
        u2=self.up2(x3); c2=torch.cat([u2,e2],1); x2=self.d2(c2)+self.d2p(te).unsqueeze(-1)
        u1=self.up1(x2); c1=torch.cat([u1,e1],1); x1=self.d1(c1)+self.d1p(te).unsqueeze(-1)
        return self.out(x1)

beta = torch.linspace(1e-4, 0.02, T); alpha = 1-beta; alpha_bar = torch.cumprod(alpha, 0)

def parse_anom(s):
    if not s or s.strip()=="[]": return []
    try: return [tuple(v) for v in ast.literal_eval(s)]
    except: return []

def load_labels(csvp):
    ch={}
    with open(csvp, encoding="utf-8", errors="replace") as f:
        r=csv.reader(f); h=next(r); i={n:x for x,n in enumerate(h)}
        for row in r:
            if row and len(row)>1:
                cid=row[i["chan_id"]]
                if cid not in ch: ch[cid]=(row[i["spacecraft"]], parse_anom(row[i["anomaly_sequences"]]))
    return ch

def norm(a):
    mn=a.min(0); mx=a.max(0); rng=mx-mn; rng[rng==0]=1.0
    return ((2*(a-mn)/rng-1)).astype(np.float32)

def windows(a, sw, ss):
    return torch.from_numpy(np.stack([a[w0:w0+sw].T for w0 in range(0, len(a)-sw+1, ss)])).float()

def denoise(model, x0, dev):
    """Algorithm 1.B：加噪到xT再反向去噪T步 → 重建"""
    eps=torch.randn_like(x0)
    x=torch.sqrt(alpha_bar[-1])*x0+torch.sqrt(1-alpha_bar[-1])*eps
    for tt in reversed(range(1,T+1)):
        t_t=torch.full((x0.shape[0],), tt-1, dtype=torch.long, device=dev)
        z=torch.randn_like(x) if tt>1 else torch.zeros_like(x)
        a=alpha[tt-1]; ab=alpha_bar[tt-1]; root=torch.sqrt(1-ab)
        eh=model(x,t_t)
        x=(1/torch.sqrt(a))*(x-((1-a)/root)*eh)+torch.sqrt(beta[tt-1])*z
    return x

def score(x0,xh):
    return torch.nn.functional.mse_loss(xh,x0,reduction="none").mean(dim=[1,2])

def pa_adjust(pred, gt):
    """point-adjusted：GT 异常段只要被任一预测命中，整段都算判对（TP）"""
    pred=pred.copy(); gt=gt.copy(); segs=[]; i=0
    while i<len(gt):
        if gt[i]==1:
            j=i
            while j<len(gt) and gt[j]==1: j+=1
            segs.append((i,j)); i=j
        else: i+=1
    for s,e in segs:
        if pred[s:e].any(): pred[s:e]=1
    return pred

def run_channel(args, labels, dev):
    cid=args.channel
    base=os.path.join(args.data_dir,"data","data")
    tr=np.load(os.path.join(base,"train",f"{cid}.npy")); te=np.load(os.path.join(base,"test",f"{cid}.npy"))
    trn=norm(tr); all_w=windows(trn,SW,TRAIN_SS); V=trn.shape[1]
    tew=windows(norm(te),SW,TEST_SS)
    sc,(anom)=labels.get(cid,(None,[]))
    gt=[]
    for k in range(len(tew)):
        w0=k*TEST_SS; w1=w0+SW
        gt.append(1 if any(w0<b+1 and w1>a for a,b in anom) else 0)
    gt=torch.tensor(gt)

    os.makedirs(args.out_dir,exist_ok=True)
    n=len(all_w); perm=torch.randperm(n); nval=int(0.2*n)
    val_idx=perm[:nval]; tr_idx=perm[nval:]
    model=UNet1D(V, base=args.base).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=2e-4)
    loss_hist=[]
    for it in range(args.epochs):
        idx=torch.randint(0,len(tr_idx),(BATCH,))
        x0=all_w[tr_idx[idx]].to(dev); t=torch.randint(0,T,(BATCH,),device=dev)
        eps=torch.randn_like(x0); ab=alpha_bar[t].view(-1,1,1)
        xt=torch.sqrt(ab)*x0+torch.sqrt(1-ab)*eps
        loss=nn.functional.mse_loss(model(xt,t),eps)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_hist.append(loss.item())
    # 验证集定阈值
    model.eval()
    with torch.no_grad():
        vx=all_w[val_idx].to(dev); vh=denoise(model,vx,dev); vs=score(vx,vh)
        mu=vs.mean().item(); sd=vs.std().item(); thresh=mu+args.eta*sd
        tx=tew.to(dev); th=denoise(model,tx,dev); ts=score(tx,th)
        pred=(ts>thresh).long().cpu().numpy()
    gtn=gt.numpy()
    if args.eval=="pa": pred=pa_adjust(pred, gtn)
    tp=int(((pred==1)&(gtn==1)).sum()); fp=int(((pred==1)&(gtn==0)).sum()); fn=int(((pred==0)&(gtn==1)).sum())
    P=tp/(tp+fp) if tp+fp>0 else 0.0; R=tp/(tp+fn) if tp+fn>0 else 0.0
    F1=2*P*R/(P+R) if P+R>0 else 0.0
    print(f"[{cid}] P={P:.3f} R={R:.3f} F1={F1:.3f} (TP={tp} FP={fp} FN={fn}, thr={thresh:.3f})")

    # ---- 图表 ----
    # 1) loss 曲线
    plt.figure(); plt.plot(loss_hist); plt.xlabel("iter"); plt.ylabel("loss"); plt.title(f"{cid} training loss")
    plt.savefig(os.path.join(args.out_dir,f"{cid}_loss.png")); plt.close()
    # 2) 重建对比（取一个正常窗口 + 一个异常窗口，画变量0）
    with torch.no_grad():
        rec=denoise(model,tx,dev).cpu()
    nidx=int(torch.nonzero(gt==0)[0][0]) if (gt==0).any() else 0
    aidx=int(torch.nonzero(gt==1)[0][0]) if (gt==1).any() else nidx
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(tx[nidx,0,:].cpu().numpy(),label="orig(正常)"); plt.plot(rec[nidx,0,:].numpy(),label="recon"); plt.title("正常窗口") ; plt.legend()
    plt.subplot(1,2,2); plt.plot(tx[aidx,0,:].cpu().numpy(),label="orig(异常)"); plt.plot(rec[aidx,0,:].numpy(),label="recon"); plt.title("异常窗口"); plt.legend()
    plt.savefig(os.path.join(args.out_dir,f"{cid}_recon.png")); plt.close()
    # 3) 得分分布 + 阈值
    plt.figure(); plt.hist(ts.cpu().numpy(),bins=40); plt.axvline(thresh,color="r",ls="--",label="threshold"); plt.xlabel("score"); plt.title(f"{cid} anomaly score"); plt.legend()
    plt.savefig(os.path.join(args.out_dir,f"{cid}_score.png")); plt.close()
    # 4) 检测结果时间线（变量0，标真值异常区 + 预测异常窗口）
    plt.figure(figsize=(14,4))
    plt.plot(te[:,0],color="b",lw=1,label="signal(var0)")
    for a,b in anom: plt.axvspan(a,b,color="g",alpha=0.25,label="GT anomaly" if a==anom[0][0] else "")
    for k in range(len(pred)):
        if pred[k]==1: plt.axvspan(k*TEST_SS,k*TEST_SS+SW,color="r",alpha=0.15)
    plt.xlabel("time"); plt.title(f"{cid} detection (green=GT, red=pred)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,f"{cid}_detect.png")); plt.close()
    torch.save(model.state_dict(), os.path.join(args.out_dir,f"{cid}_model.pt"))
    return {"channel":cid,"P":P,"R":R,"F1":F1,"TP":tp,"FP":fp,"FN":fn,"thresh":thresh}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir",required=True,help="archive 目录(含 data/data 和 labeled_anomalies.csv)")
    ap.add_argument("--channel",default="E-1",help="某通道；或 all 跑全部")
    ap.add_argument("--epochs",type=int,default=2000)
    ap.add_argument("--rounds",type=int,default=1)
    ap.add_argument("--out_dir",default="./out")
    ap.add_argument("--batch",type=int,default=128)
    ap.add_argument("--base",type=int,default=16,help="1-D U-Net 模型宽度(16/32/64)")
    ap.add_argument("--eta",type=float,default=2.0,help="阈值系数 η")
    ap.add_argument("--eval",default="plain",choices=["plain","pa"],help="plain=普通P/R/F1; pa=point-adjusted")
    args=ap.parse_args(); global BATCH; BATCH=args.batch
    dev=device(); print("设备:", dev)
    # 把噪声调度张量也移到 GPU，避免 alpha_bar[t] 的 CPU/GPU 不一致
    global beta, alpha, alpha_bar
    beta=beta.to(dev); alpha=alpha.to(dev); alpha_bar=alpha_bar.to(dev)
    labels=load_labels(os.path.join(args.data_dir,"labeled_anomalies.csv"))
    os.makedirs(args.out_dir,exist_ok=True)
    results={"rounds":args.rounds}
    for rnd in range(args.rounds):
        r=run_channel(args,labels,dev); results.setdefault(r["channel"],[]).append(r)
        print(f"轮 {rnd+1} 完成")
    avg={ch:float(np.mean([x["F1"] for x in v])) for ch,v in results.items() if isinstance(v,list)}
    with open(os.path.join(args.out_dir,"results.json"),"w") as f: json.dump(results,f,indent=2)
    print("平均F1:",avg,"(结果已存 results.json, 图表已存 out/)")

if __name__=="__main__":
    main()
