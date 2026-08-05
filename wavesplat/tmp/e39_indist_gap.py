"""e39 STEP 2 — IN-DISTRIBUTION mechanism diagnostic (pre-training).

Provenance (e39_provenance) proved bwtask = seed 2026, fan_frac 0.55,
bimodal_frac 0.5, bw_inject 0.15 (z bit-exact) => the incumbent l1s2bw
TRAINED on 50% cross-Tp crossing seas, yet e37 shows +0.098 oracle-vs-net
headroom on crossing arena seas, +0.090 even on the in-distribution (CCW)
half. The data-gap hypothesis is DEAD. Remaining causes: optimization
(best-val@800 = <1 epoch), objective misalignment, or genuine
non-inferability from the 6ch maps.

THIS SCRIPT: on bwtask's own VAL samples (the first 666, exactly
d3_l1_s2head_train's split), through the SAME fixed solver (smult=3, free
arm; no heads, no GT peek), measure per-sample:
    net    corr / task-MSE of l1s2bw's gamma(X)
    oracle corr / task-MSE of the stored TRUE-energy label gamma (10^Y)
    v7fan  corr of the pretrain checkpoint (did the l1s2bw stage move it?)
split by the sample's (provenance-recovered) crossing flag.

READING:
  oracle-net gap >> 0 on crossing val, ~0 on unimodal val
      -> the net leaves crossing-specific loss on the table ON ITS OWN
         DISTRIBUTION => optimization/objective axis is real; retraining
         has a target. (And the e37 arena gap is NOT arena-specific.)
  gap ~ equal on both
      -> generic shrinkage gap, crossing is not special in-dist; the e37
         arena gap needs another explanation (kp normalization etc.).
  gap ~ 0 on both
      -> the net is already optimal-in-class in-dist; the arena gap is an
         arena/regime artifact => retraining CANNOT eat it (falsifies the
         retrain premise BEFORE burning GPU).

Usage: python3 tmp/e39_indist_gap.py [--n 240]
Output: results/direction3/e39_indist_gap.json
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from d3_prior_net_train import PriorNet, band_mask, NGRID, KREL
from d18_relax_estimator import omega
from d3_sa_achievability_gap import corr

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEPTH = 50.0
SMULT = 3.0
OUT = 'results/direction3/e39_indist_gap.json'
_MASK = band_mask()

BW_SEED, BW_FF, BW_BF, BW_BW = 2026, 0.55, 0.5, 0.15


def recover_flags(n=666):
    """Replay the bwtask stream to recover per-sample crossing flags
    (r2, tp2, diroff) for the first n samples. Mirrors gen_sample's rng."""
    from d3_sa_achievability_gap import dispersion_lam
    rng = np.random.default_rng(BW_SEED)
    N_OBS = 400
    flags = []
    for i in range(n):
        Tp = rng.uniform(5.0, 11.0)
        rng.uniform(1.0, 3.0)                     # Hs
        rng.uniform(1.5, 6.0)
        rng.uniform(15.0, 45.0)
        rng.uniform(0.0, 360.0)
        rng.integers(0, 2 ** 31 - 1)
        u_b = rng.uniform()
        bi = bool(u_b < BW_BF)
        r2 = tp2 = diroff = 0.0
        if bi:
            r2 = rng.uniform(0.15, 0.5)
            tp2 = float(np.clip(Tp * rng.uniform(0.6, 1.6), 4.0, 12.0))
            diroff = rng.uniform(40, 140)
        rng.uniform(0.0, 0.4)
        rng.uniform(0, 2 * np.pi)
        rng.uniform()                              # bw_lvl (bw_inject>0)
        rsel = rng.uniform()
        if rsel < BW_FF:
            rng.uniform(1.0, 12.0)
            rng.uniform(20.0, 100.0, N_OBS)
            rng.uniform(-np.pi / 3, np.pi / 3, N_OBS)
            mode = 'fan'
        elif rsel < BW_FF + (1 - BW_FF) / 3:
            rng.uniform(0, 2 * np.pi, N_OBS)
            rng.uniform(0, 1, N_OBS)
            mode = 'fixed'
        else:
            rng.uniform(0.1, 1.2)
            rng.uniform(0, 2 * np.pi, N_OBS)
            rng.uniform(0, 1, N_OBS)
            mode = 'moving'
        nfrac = rng.uniform(0.02, 0.10)
        Hs_chk = None
        rng.normal(0, 1.0, N_OBS)                  # scale irrelevant to stream
        if mode == 'fan':
            rng.uniform(0, 10.0, 128)
            rng.uniform(10.0, 110.0, 128)
            rng.uniform(-np.pi / 3, np.pi / 3, 128)
        else:
            rng.uniform(0, 1, 256)
            rng.uniform(0, 1, 256)
        flags.append(dict(bimodal=bi, r2=float(r2), tp2=float(tp2),
                          diroff=float(diroff), Tp=float(Tp), mode=mode,
                          nfrac=float(nfrac)))
    return flags


def solve_corr_mse(P, z, Q, gt, kp, sigma0, g_grid):
    """Fixed-solver (smult=3, free arm) prediction at Q; returns corr, nmse."""
    g = np.maximum(g_grid, 0.0)[_MASK]
    if g.sum() <= 0:
        return float('nan'), float('nan')
    g = g / g.sum()
    ax = np.linspace(-KREL, KREL, NGRID)
    KX, KY = np.meshgrid(ax, ax)
    kxb = (KX[_MASK] * kp).astype(np.float64)
    kyb = (KY[_MASK] * kp).astype(np.float64)
    om = omega(np.hypot(kxb, kyb) + 1e-12, DEPTH)
    t = lambda a: torch.tensor(np.asarray(a, np.float32), device=DEVICE)

    def design(Qp):
        psi = t(kxb)[None] * t(Qp)[:, 0:1] + t(kyb)[None] * t(Qp)[:, 1:2] \
            - t(om)[None] * t(Qp)[:, 2:3]
        return torch.cat([torch.cos(psi), torch.sin(psi)], dim=1)

    Phi = design(P)
    Phi_q = design(Q)
    gam = t(np.concatenate([g, g]) * float(np.var(z)))
    K = (Phi * gam[None]) @ Phi.T \
        + (SMULT * sigma0) ** 2 * torch.eye(len(z), device=DEVICE)
    w = Phi.T @ torch.linalg.solve(K, t(z))
    pred = ((Phi_q * gam[None]) @ w).cpu().numpy()
    c = corr(pred, gt)
    nmse = float(np.mean((pred - gt) ** 2) / max(np.var(gt), 1e-12))
    return float(c), nmse


def get_prior_net(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE)
    net = PriorNet().to(DEVICE)
    net.load_state_dict(ck['model'])
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=240)
    a = ap.parse_args()
    d = np.load('tmp/prior_train/bwtask_s000.npz')
    n = min(a.n, 666)
    flags = recover_flags(666)[:n]
    net = get_prior_net('results/direction3/prior_net_l1s2bw.pt')
    v7 = get_prior_net('results/direction3/prior_net_v7fan.pt')
    rows = []
    for i in range(n):
        P, z = d['P'][i].astype(np.float64), d['z'][i].astype(np.float64)
        Q, gt = d['Q'][i].astype(np.float64), d['gt_q'][i].astype(np.float64)
        kp, Hs, Tp, ucp, nfrac = [float(x) for x in d['meta'][i]]
        sigma0 = nfrac * Hs
        X = d['X'][i]
        with torch.no_grad():
            Y_net = net(torch.tensor(X[None], device=DEVICE))[0].cpu().numpy()
            Y_v7 = v7(torch.tensor(X[None], device=DEVICE))[0].cpu().numpy()
        g_net = 10.0 ** Y_net
        g_v7 = 10.0 ** Y_v7
        g_or = 10.0 ** d['Y'][i].astype(np.float64)   # true-energy label
        c_n, m_n = solve_corr_mse(P, z, Q, gt, kp, sigma0, g_net)
        c_o, m_o = solve_corr_mse(P, z, Q, gt, kp, sigma0, g_or)
        c_7, m_7 = solve_corr_mse(P, z, Q, gt, kp, sigma0, g_v7)
        rows.append(dict(i=i, bimodal=flags[i]['bimodal'],
                         r2=flags[i]['r2'], tp2=flags[i]['tp2'],
                         Tp=Tp, mode=flags[i]['mode'],
                         net=c_n, oracle=c_o, v7=c_7,
                         net_mse=m_n, oracle_mse=m_o))
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{n}", flush=True)

    def summar(sub, tag):
        if not sub:
            return None
        gap = np.array([r['oracle'] - r['net'] for r in sub])
        d7 = np.array([r['net'] - r['v7'] for r in sub])
        se = gap.std(ddof=1) / np.sqrt(len(gap))
        se7 = d7.std(ddof=1) / np.sqrt(len(d7))
        print(f"  {tag:22s} n={len(sub):3d}  net {np.mean([r['net'] for r in sub]):+.4f} "
              f"oracle {np.mean([r['oracle'] for r in sub]):+.4f}  "
              f"gap {gap.mean():+.4f}±{se:.4f} (win {(gap>0).sum()}/{len(gap)})  "
              f"| net-v7 {d7.mean():+.4f}±{se7:.4f}")
        return dict(tag=tag, n=len(sub),
                    net=float(np.mean([r['net'] for r in sub])),
                    oracle=float(np.mean([r['oracle'] for r in sub])),
                    gap=float(gap.mean()), gap_se=float(se),
                    gap_win=int((gap > 0).sum()),
                    net_minus_v7=float(d7.mean()), net_minus_v7_se=float(se7))

    print(f"\n=== e39 in-distribution gap (bwtask val, fixed smult=3, "
          f"free arm) ===")
    bi = [r for r in rows if r['bimodal']]
    un = [r for r in rows if not r['bimodal']]
    # cross-Tp strength split within bimodal: |tp2/Tp - 1|
    far = [r for r in bi if abs(r['tp2'] / r['Tp'] - 1) > 0.25]
    near = [r for r in bi if abs(r['tp2'] / r['Tp'] - 1) <= 0.25]
    s = dict(
        all=summar(rows, 'ALL'),
        unimodal=summar(un, 'unimodal'),
        crossing=summar(bi, 'crossing (all)'),
        crossing_far=summar(far, 'crossing far-Tp(>25%)'),
        crossing_near=summar(near, 'crossing near-Tp'),
    )
    json.dump(dict(rows=rows, summary=s,
                   provenance=dict(seed=BW_SEED, fan_frac=BW_FF,
                                   bimodal_frac=BW_BF, bw_inject=BW_BW)),
              open(OUT, 'w'), indent=1, default=float)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main()
