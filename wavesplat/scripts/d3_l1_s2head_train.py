"""L1-s2 -- scene-adaptive noise floor, LEARNED (the challenge's first shot).

Today's chain: H5 proved the binding hyperparameter is the noise floor s2
(hand-set smult=10 rescued encounter on sim; P4-1 showed nowcast prefers it
small and the pathological record LJ18g prefers it large -> s2 is SCENE- and
TASK-dependent). The v3 task-loss stack already learns gamma(k) through the
differentiable MMSE with a single GLOBAL log-smult. This run upgrades that
scalar to a HEAD: log_smult_i = b + MLP(pooled X_i features), trained jointly
with the prior net on the fan shards (forecast-lead queries included in
their task loss -- the deployment task).

Init: prior_net_v7fan.pt (fan-era net). Data: tmp/prior_train/fan0 (4000
samples, obs profile ~ uniform-in-r 20-100 m = close to the deployed
stratified selection). Judgment (next session, d3_prior_net_eval machinery):
G-REAL -- net+head vs net+global-smult vs hand recipe on real records,
forecast scoring. This script reports train/val task loss and the head's
smult spread across scenes (a constant head = nothing learned = honest null).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_prior_net_train import PriorNet, band_mask, NGRID, KREL
from d3_prior_net_taskft import load_all, task_loss_one

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SmultHead(nn.Module):
    """pooled scene features -> log noise multiplier (init ~ log 3)."""

    def __init__(self, c=6, h=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 * c, h), nn.SiLU(),
                               nn.Linear(h, 1))
        with torch.no_grad():
            self.f[-1].weight.mul_(0.1)
            self.f[-1].bias.zero_()
        self.b = nn.Parameter(torch.tensor(float(np.log(3.0))))

    def forward(self, X):                     # X (B, 6, 51, 51)
        p = torch.cat([X.mean(dim=(2, 3)), X.std(dim=(2, 3))], 1)
        return self.b + self.f(p).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--data', default='tmp/prior_train/fan0')
    ap.add_argument('--init', default='results/direction3/prior_net_v7fan.pt')
    ap.add_argument('--ckpt', default='results/direction3/prior_net_l1s2.pt')
    a = ap.parse_args()

    D = load_all(a.data)
    n_all = len(D['X'])
    n_val = max(200, n_all // 12)
    print(f"data {a.data}: {n_all} samples ({n_val} val)")
    net = PriorNet().to(DEVICE)
    ck = torch.load(a.init, map_location=DEVICE)
    net.load_state_dict(ck['model'])
    head = SmultHead().to(DEVICE)
    opt = torch.optim.AdamW([{'params': net.parameters(), 'lr': a.lr},
                             {'params': head.parameters(), 'lr': 1e-3}],
                            weight_decay=1e-5)
    mask_t = torch.tensor(band_mask(), device=DEVICE)
    axn = torch.linspace(-KREL, KREL, NGRID, device=DEVICE)
    KXn, KYn = torch.meshgrid(axn, axn, indexing='xy')
    kxn, kyn = KXn[mask_t], KYn[mask_t]

    def sample_loss(i, out, ls):
        P = torch.tensor(D['P'][i], device=DEVICE)
        z = torch.tensor(D['z'][i], device=DEVICE)
        Q = torch.tensor(D['Q'][i], device=DEVICE)
        gt = torch.tensor(D['gt_q'][i], device=DEVICE)
        kp_e, Hs, _, _, nfrac = D['meta'][i]
        sig0 = float(nfrac * Hs)
        return task_loss_one(out, P, z, Q, gt, float(kp_e), sig0,
                             ls, mask_t, kxn, kyn) / (Hs / 4) ** 2

    rng = np.random.default_rng(0)
    vi = np.arange(n_val)
    ti = np.arange(n_val, n_all)
    best = (np.inf, 0)
    t0 = time.time()

    def val_loss():
        net.eval(); head.eval()
        with torch.no_grad():
            vls, sms = [], []
            for i0 in range(0, 160, a.bs):
                jj = vi[i0:i0 + a.bs]
                Xv = torch.tensor(D['X'][jj], device=DEVICE)
                ov = net(Xv)
                lsv = head(Xv)
                vls += [float(sample_loss(int(i), ov[j], lsv[j]))
                        for j, i in enumerate(jj)]
                sms += [float(x) for x in torch.exp(lsv)]
        net.train(); head.train()
        return float(np.mean(vls)), float(np.mean(sms)), float(np.std(sms))

    v0, sm0, ss0 = val_loss()
    print(f"step     0 val {v0:.4f}  smult {sm0:.2f}±{ss0:.2f}", flush=True)
    for step in range(1, a.steps + 1):
        ii = rng.choice(ti, a.bs, replace=False)
        X = torch.tensor(D['X'][ii], device=DEVICE)
        outs = net(X)
        ls = head(X)
        loss = sum(sample_loss(int(i), outs[j], ls[j])
                   for j, i in enumerate(ii)) / a.bs
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.parameters())
                                       + list(head.parameters()), 1.0)
        opt.step()
        if step % 200 == 0 or step == a.steps:
            vl, sm, ss = val_loss()
            tag = ''
            if vl < best[0]:
                best = (vl, step)
                torch.save({'model': net.state_dict(),
                            'head': head.state_dict(),
                            'val': vl, 'step': step}, a.ckpt)
                tag = ' *saved'
            print(f"step {step:5d} [{time.time()-t0:5.0f}s] "
                  f"train {float(loss):.4f} val {vl:.4f} "
                  f"smult {sm:.2f}±{ss:.2f}{tag}", flush=True)
    print(f"best val {best[0]:.4f} @ step {best[1]} (step-0 ref {v0:.4f})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
