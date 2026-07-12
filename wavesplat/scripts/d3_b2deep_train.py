"""B-2 DEEP -- objective head-to-head (task-loss vs likelihood/residual).

Pre-registered in docs/B1B2_DEEP_prereg_2026-07-12.md ("B-2 심화"). Two arms,
everything else identical (arch PriorNet+SmultHead, init v7fan, data bwtask,
lr/bs/val-period/best-ckpt), 12,000 steps, seeds {0,1}:

  --objective task   arm T: the shipped task_loss_one (query-point field MSE).
  --objective resid  arm L: hold out 20% of each scene's observations
                     (rng(seed+i), reproducible), build the differentiable
                     posterior mean from the remaining 80% via the SAME Woodbury
                     path, query it at the held-out observation points, and score
                     held-out z prediction MSE / (Hs/4)^2. This TRAINS the
                     amortizer on the very residual signal the n-poison
                     SELECTION used -- the method claim's decisive test.

Convergence: val every 500 steps; if the last-2k improvement >= 0.002 at the
budget boundary, auto-extend by 4000 steps once (--auto_extend, default on).

Usage:
  python3 scripts/d3_b2deep_train.py --objective task  --seed 0 \
      --ckpt results/direction3/b2deep_T_s0.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_prior_net_train import PriorNet, band_mask, NGRID, KREL
from d3_prior_net_taskft import load_all, task_loss_one
from d3_l1_s2head_train import SmultHead

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--objective', choices=['task', 'resid'], required=True)
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--val_every', type=int, default=500)
    ap.add_argument('--holdout', type=float, default=0.20)
    ap.add_argument('--data', default='tmp/prior_train/bwtask')
    ap.add_argument('--init', default='results/direction3/prior_net_v7fan.pt')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--auto_extend', type=int, default=1)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    D = load_all(a.data)
    n_all = len(D['X'])
    n_val = max(200, n_all // 12)
    print(f"[{a.objective}|s{a.seed}] data {a.data}: {n_all} samples "
          f"({n_val} val), ckpt {a.ckpt}", flush=True)

    net = PriorNet().to(DEVICE)
    net.load_state_dict(torch.load(a.init, map_location=DEVICE)['model'])
    head = SmultHead().to(DEVICE)
    opt = torch.optim.AdamW([{'params': net.parameters(), 'lr': a.lr},
                             {'params': head.parameters(), 'lr': 1e-3}],
                            weight_decay=1e-5)
    mask_t = torch.tensor(band_mask(), device=DEVICE)
    axn = torch.linspace(-KREL, KREL, NGRID, device=DEVICE)
    KXn, KYn = torch.meshgrid(axn, axn, indexing='xy')
    kxn, kyn = KXn[mask_t], KYn[mask_t]

    # arm-L held-out masks: reproducible per-sample via rng(seed + i).
    ho = {}
    if a.objective == 'resid':
        for i in range(n_all):
            n_obs = int(D['z'][i].shape[0])
            k = max(1, int(round(a.holdout * n_obs)))
            hi = np.random.default_rng(a.seed + i).choice(n_obs, k, False)
            m = np.zeros(n_obs, bool)
            m[hi] = True
            ho[i] = m

    def sample_loss(i, out, ls):
        P = torch.tensor(D['P'][i], device=DEVICE)
        z = torch.tensor(D['z'][i], device=DEVICE)
        kp_e, Hs, _, _, nfrac = D['meta'][i]
        sig0 = float(nfrac * Hs)
        if a.objective == 'task':
            Q = torch.tensor(D['Q'][i], device=DEVICE)
            gt = torch.tensor(D['gt_q'][i], device=DEVICE)
            Ptr, ztr = P, z
        else:  # resid -- predict held-out observations from the rest
            m = ho[i]
            Ptr, ztr = P[~m], z[~m]
            Q, gt = P[m], z[m]
        return task_loss_one(out, Ptr, ztr, Q, gt, float(kp_e), sig0,
                             ls, mask_t, kxn, kyn) / (Hs / 4) ** 2

    rng = np.random.default_rng(a.seed)
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
                ov, lsv = net(Xv), head(Xv)
                vls += [float(sample_loss(int(i), ov[j], lsv[j]))
                        for j, i in enumerate(jj)]
                sms += [float(x) for x in torch.exp(lsv)]
        net.train(); head.train()
        return float(np.mean(vls)), float(np.mean(sms)), float(np.std(sms))

    v0, sm0, ss0 = val_loss()
    print(f"step     0 val {v0:.4f}  smult {sm0:.2f}+-{ss0:.2f}", flush=True)
    val_hist = [(0, v0)]
    total, extended = a.steps, False
    step = 0
    while step < total:
        step += 1
        ii = rng.choice(ti, a.bs, replace=False)
        X = torch.tensor(D['X'][ii], device=DEVICE)
        outs, ls = net(X), head(X)
        loss = sum(sample_loss(int(i), outs[j], ls[j])
                   for j, i in enumerate(ii)) / a.bs
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.parameters())
                                       + list(head.parameters()), 1.0)
        opt.step()
        if step % a.val_every == 0 or step == total:
            vl, sm, ss = val_loss()
            val_hist.append((step, vl))
            tag = ''
            if vl < best[0]:
                best = (vl, step)
                torch.save({'model': net.state_dict(),
                            'head': head.state_dict(), 'val': vl, 'step': step,
                            'objective': a.objective, 'seed': a.seed}, a.ckpt)
                tag = ' *saved'
            print(f"step {step:5d} [{time.time()-t0:5.0f}s] "
                  f"train {float(loss.detach()):.4f} val {vl:.4f} "
                  f"smult {sm:.2f}+-{ss:.2f}{tag}", flush=True)
            if step == total and not extended and a.auto_extend:
                vh = dict(val_hist)
                v_ref = vh.get(total - 2000)
                last = [v for s, v in val_hist if total - 2000 < s <= total]
                imp = (v_ref - min(last)) if (v_ref is not None and last) else 0.0
                print(f"  [conv] last-2k improvement {imp:+.4f} "
                      f"(ref@{total-2000} {v_ref} -> best {min(last):.4f})",
                      flush=True)
                if imp >= 0.002:
                    total += 4000
                    extended = True
                    print(f"  [conv] NOT converged -> extend to {total}",
                          flush=True)
                else:
                    print(f"  [conv] converged (plateau)", flush=True)
    print(f"best val {best[0]:.4f} @ step {best[1]} (step-0 ref {v0:.4f}) "
          f"total {total} extended {extended} -> {a.ckpt}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
