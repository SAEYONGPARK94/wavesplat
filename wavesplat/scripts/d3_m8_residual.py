"""M8 -- residual boosting on the encounter track (layer-2 quick win).

Accounting first (design law 1): how much of the recipe's encounter-track
error is SYSTEMATIC (lead-dependent scale/bias -- the 07-08 shrinkage story)
vs irreducible? Ladder, judged on held-out head-sea episodes:

  L0  recipe raw               (fix3-hand candidate: band .35, smult 3)
  L1  per-lead scale recalib   beta(lead) fit on train episodes (NO learning)
  L2  residual MLP             feat = [pred, lead, kp*r?, Hs, env] -> d-eta,
                               zero-init last layer (= L0 at start)

Metric: track corr + nRMSE per episode; paired vs L0. If L1 grabs whatever
exists and L2 adds ~0, M8 closes honestly ("systematic part is a scale
curve"); if L2 > L1 the net found structure beyond scale. Real-window
zero-shot only if the synthetic gate (paired d >= 0.02, t >= 2.5) passes.
"""
import glob
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu
from d3_g0b_selection_aware import DEV

OUT = 'results/direction3/m8_residual.json'
LEADS = np.linspace(0.5, 8.0, 76)


def episode_track(path, rng):
    with h5py.File(path, 'r') as f:
        comp = f['config/wave_components'][:]
        ts = f['frames/timestamps'][:]
        pts = f['frames/lidar_points']
        obs_i = list(range(50, 249, 18))
        t_end = ts[obs_i[-1]]
        P, zv = [], []
        for i in obs_i:
            p = pts[i]
            P.append(np.column_stack([p[:, 0], p[:, 1],
                                      np.full(len(p), ts[i])]))
            zv.append(p[:, 2])
        P, zv = np.vstack(P), np.concatenate(zv)
        kp = float(np.hypot(comp[:, 2], comp[:, 3])[np.argmax(comp[:, 0])])
        Hs = float(f['config'].attrs['Hs'])
        tq = t_end + LEADS
        A, om, kx, ky, ph = (comp[:, 0], comp[:, 1], comp[:, 2],
                             comp[:, 3], comp[:, 4])
        psi = (- om[None] * tq[:, None] + ph[None])
        gt = (A[None] * np.cos(psi)).sum(1)          # at origin (0,0)
    rr = np.hypot(P[:, 0], P[:, 1])
    edges = np.linspace(rr.min(), rr.max() + 1e-6, 13)
    b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
    take = []
    for j in range(12):
        idx = np.where(b == j)[0]
        k = min(len(idx), 167)
        if k:
            take.append(rng.choice(idx, k, replace=False))
    sel = np.concatenate(take)
    Q = np.column_stack([np.zeros(len(LEADS)), np.zeros(len(LEADS)), tq])
    with torch.no_grad():
        pr = mmse_gpu(torch.tensor(P[sel], device=DEV, dtype=torch.float32),
                      torch.tensor(zv[sel], device=DEV, dtype=torch.float32),
                      50.0, torch.tensor(Q, device=DEV, dtype=torch.float32),
                      0.05, kp_hint=kp, no_em=True, pexp=1.0,
                      smult=3.0).cpu().numpy()
    return pr, gt, Hs, kp


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    rng = np.random.default_rng(99)
    files = sorted(glob.glob('data/episode_*_dir180.h5'))
    data = []
    for p in files:
        pr, gt, Hs, kp = episode_track(p, rng)
        data.append(dict(file=os.path.basename(p), pr=pr, gt=gt,
                         Hs=Hs, kp=kp))
        print(f"{os.path.basename(p):48s} L0 corr {corr(pr, gt):+.2f}",
              flush=True)
    n = len(data)
    tr, va = data[:int(0.6 * n)], data[int(0.6 * n):]
    print(f"train {len(tr)} / val {len(va)} episodes")

    # accounting + L1: per-lead scale beta(lead) = <pred*gt>/<pred^2> on train
    Ptr = np.stack([d['pr'] / (d['Hs'] / 4) for d in tr])
    Gtr = np.stack([d['gt'] / (d['Hs'] / 4) for d in tr])
    beta = (Ptr * Gtr).sum(0) / np.maximum((Ptr ** 2).sum(0), 1e-9)
    print(f"beta(lead): {np.round(beta[::15], 2)}")

    # L2: tiny residual MLP
    def feats(d):
        prn = d['pr'] / (d['Hs'] / 4)
        env = np.abs(np.convolve(np.abs(prn), np.ones(9) / 9, 'same'))
        return np.stack([prn, LEADS / 8.0, np.full_like(prn, d['kp'] * 10),
                         np.full_like(prn, d['Hs'] / 4), env], 1)

    net = nn.Sequential(nn.Linear(5, 48), nn.SiLU(), nn.Linear(48, 48),
                        nn.SiLU(), nn.Linear(48, 1)).to(DEV)
    with torch.no_grad():
        net[-1].weight.mul_(0.01)
        net[-1].bias.zero_()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr = torch.tensor(np.concatenate([feats(d) for d in tr], 0),
                       dtype=torch.float32, device=DEV)
    Rtr = torch.tensor(np.concatenate(
        [(d['gt'] - d['pr']) / (d['Hs'] / 4) for d in tr], 0),
        dtype=torch.float32, device=DEV)
    for step in range(3000):
        ii = torch.randint(len(Xtr), (256,), device=DEV)
        loss = ((net(Xtr[ii]).squeeze(-1) - Rtr[ii]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    rows = []
    for d in va:
        s = d['Hs'] / 4
        l0 = d['pr']
        l1 = d['pr'] * beta * 1.0
        with torch.no_grad():
            dr = net(torch.tensor(feats(d), dtype=torch.float32,
                                  device=DEV)).squeeze(-1).cpu().numpy()
        l2 = d['pr'] + dr * s
        rows.append({'file': d['file'],
                     'L0': corr(l0, d['gt']), 'L1': corr(l1, d['gt']),
                     'L2': corr(l2, d['gt']),
                     'nr0': float(np.sqrt(np.mean((l0 - d['gt'])**2)) / s),
                     'nr1': float(np.sqrt(np.mean((l1 - d['gt'])**2)) / s),
                     'nr2': float(np.sqrt(np.mean((l2 - d['gt'])**2)) / s)})
    json.dump(rows, open(OUT, 'w'), indent=1)
    for k in ('L0', 'L1', 'L2'):
        print(f"{k} corr {np.mean([r[k] for r in rows]):+.3f}", end='  ')
    print()
    for k in ('nr0', 'nr1', 'nr2'):
        print(f"{k} {np.mean([r[k] for r in rows]):.3f}", end='  ')
    d1 = np.array([r['nr0'] - r['nr1'] for r in rows])
    d2 = np.array([r['nr1'] - r['nr2'] for r in rows])
    print(f"\nL1 gain(nrmse) {d1.mean():+.4f}±{d1.std(ddof=1)/np.sqrt(len(d1)):.4f}"
          f"   L2-over-L1 {d2.mean():+.4f}±{d2.std(ddof=1)/np.sqrt(len(d2)):.4f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
