"""Targeted fix for the diagnosed MODE-COLLAPSE: the stock loss weight
`w = 1 + 10^Y` (linear energy) UP-weights the dominant peak and DOWN-weights the
weak 2nd cross-Tp lobe (energy 15-50%) — which is exactly why the net ignores it.
Fix: compress the dynamic range so the 2nd lobe gets weight comparable to the
primary: `w = 1 + K*(10^Y)^p` (p<1). Everything else identical to d3_prior_net_train.
"""
import argparse, time
import numpy as np, torch, torch.nn as nn
from d3_prior_net_train import PriorNet, band_mask, load_data, DEVICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--data', default='tmp/prior_train/multimodal_big')
    ap.add_argument('--ckpt', default='results/direction3/prior_net_2ndlobe.pt')
    ap.add_argument('--p', type=float, default=0.3)   # dynamic-range compression
    ap.add_argument('--K', type=float, default=4.0)   # 2nd-lobe boost
    a = ap.parse_args()
    X, Y = load_data(a.data)
    n_val = max(200, len(X) // 10)
    Xv, Yv = X[:n_val], Y[:n_val]; Xt, Yt = X[n_val:], Y[n_val:]
    print(f"data {a.data}: train {len(Xt)} val {len(Xv)}  weight=1+{a.K}*(10^Y)^{a.p}")
    mask = torch.tensor(band_mask(), device=DEVICE)
    net = PriorNet().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    lossf = nn.SmoothL1Loss(reduction='none')
    rng = np.random.default_rng(0)
    Xv_t = torch.tensor(Xv, device=DEVICE); Yv_t = torch.tensor(Yv, device=DEVICE)

    def wfn(y):
        return 1.0 + a.K * (10.0 ** y).clamp(min=0) ** a.p   # up-weight 2nd lobe

    best = (np.inf, 0)
    for step in range(1, a.steps + 1):
        ii = rng.choice(len(Xt), a.bs)
        xb = torch.tensor(Xt[ii], device=DEVICE); yb = torch.tensor(Yt[ii], device=DEVICE)
        pred = net(xb); w = wfn(yb)
        loss = (lossf(pred, yb) * w * mask[None]).sum() / (w * mask[None]).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 2000 == 0 or step == a.steps:
            net.eval()
            with torch.no_grad():
                vp = torch.cat([net(Xv_t[i:i+256]) for i in range(0, len(Xv_t), 256)])
                wv = wfn(Yv_t)
                vl = float((lossf(vp, Yv_t) * wv * mask[None]).sum() / (wv * mask[None]).sum())
            net.train()
            if vl < best[0]:
                best = (vl, step); torch.save(dict(model=net.state_dict(), step=step, val=vl), a.ckpt)
            print(f"step {step} train {float(loss):.4f} val {vl:.4f}", flush=True)
    print(f"best val {best[0]:.4f} @ {best[1]} -> {a.ckpt}")


if __name__ == '__main__':
    main()
