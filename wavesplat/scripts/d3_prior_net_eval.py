"""
d3_prior_net_eval.py — gate G1 for the amortized prior net: on the exact
10-sea continuous harness, does net-predicted gamma(k) beat the
hand-crafted prior (beamform-power^p + val-EM) through the SAME physics
MMSE? Compares:

  mmse_hand   shipped mmse_bank_estimate (power^p prior, val-EM)
  mmse_net    net gamma, val-selected noise mult only (no EM)
  mmse_netEM  net gamma as EM INIT, then val-gated SBL-EM

Gate G1 (pre-registered): mmse_net(EM) >= mmse_hand + 0.05 on moving
conditions. Off-distribution note: harness seas are Tp=7 Hs=2 gamma=3.3
spread=30 fixed — INSIDE the training distribution's support but a single
point of it; G1 here therefore tests amortization, not extrapolation.

Usage: python3 scripts/d3_prior_net_eval.py [--n_seas 10]
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d18_relax_estimator import beamform_top, omega
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_nomp import beamform_power, split_masks, mmse_bank_estimate
from d3_prior_net_train import PriorNet, band_mask, NGRID, KREL
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, corr,
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = 'results/direction3/prior_net_eval.json'

_NET_CACHE = {}


def get_net(ckpt='results/direction3/prior_net_v3.pt'):
    """Lazy singleton for harness use (d3_sa_uspeed --estimator mmse_net)."""
    if ckpt not in _NET_CACHE:
        ck = torch.load(ckpt, map_location=DEVICE)
        net = PriorNet().to(DEVICE)
        net.load_state_dict(ck['model'])
        net.eval()
        _NET_CACHE[ckpt] = net
    return _NET_CACHE[ckpt]


def net_gamma_mmse(net, P, z, depth, Pq, sigma0, seed=0, val_frac=0.25,
                   use_em=False, max_em=6):
    """MMSE with the net-predicted prior on the normalized 51x51 bank."""
    rng = np.random.default_rng(seed)
    n = len(z)
    vi = rng.choice(n, max(8, int(val_frac * n)), replace=False)
    vmask = np.zeros(n, bool)
    vmask[vi] = True
    Ptr, ztr = P[~vmask], z[~vmask]
    Pv, zv = P[vmask], z[vmask]
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          depth, klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          depth, klim=3.5 * kp, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    # net input maps (identical recipe to the data generator)
    m1, m2 = split_masks(P, mode='legs')
    _, _, p1 = beamform_power(P[m1], z[m1], depth, KREL * kp, NGRID)
    _, _, p2 = beamform_power(P[m2], z[m2], depth, KREL * kp, NGRID)
    p1 = (p1 / max(p1.max(), 1e-12)).reshape(NGRID, NGRID)
    p2 = (p2 / max(p2.max(), 1e-12)).reshape(NGRID, NGRID)
    pm = np.minimum(p1, p2)
    X = np.stack([p1, p2, pm,
                  np.log10(p1 + 1e-6) / 6 + 1, np.log10(p2 + 1e-6) / 6 + 1,
                  np.log10(pm + 1e-6) / 6 + 1]).astype(np.float32)
    with torch.no_grad():
        Ylog = net(torch.tensor(X[None], device=DEVICE))[0].cpu().numpy()
    # bank = annulus cells of the same normalized grid
    ax = np.linspace(-KREL * kp, KREL * kp, NGRID)
    KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY)
    sel = (kk > 0.35 * kp) & (kk < KREL * kp)
    kxb, kyb = KX[sel].astype(float), KY[sel].astype(float)
    B = len(kxb)
    om = omega(np.hypot(kxb, kyb) + 1e-12, depth)
    g = np.maximum(10.0 ** Ylog[sel], 1e-8)
    g = g / g.sum()
    gam = np.concatenate([g, g]) * np.var(z)

    def design(Q):
        psi = kxb[None] * Q[:, 0:1] + kyb[None] * Q[:, 1:2] \
            - om[None] * Q[:, 2:3]
        return np.concatenate([np.cos(psi), np.sin(psi)], axis=1)

    Phi_tr, Phi_v = design(Ptr), design(Pv)
    m = len(ztr)
    best = None
    for smult in (1.0, 3.0, 10.0):
        s2 = (smult * sigma0) ** 2
        K = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * np.eye(m)
        fv = (Phi_v * gam[None]) @ (Phi_tr.T @ np.linalg.solve(K, ztr))
        vr = float(np.sqrt(np.mean((fv - zv) ** 2)))
        if best is None or vr < best[0]:
            best = (vr, smult)
    vr0, smult = best
    s2 = (smult * sigma0) ** 2
    if use_em:
        best_em = (vr0, gam.copy())
        for _ in range(max_em):
            K = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * np.eye(m)
            Ki_Phi = np.linalg.solve(K, Phi_tr)
            mu = gam * (Phi_tr.T @ np.linalg.solve(K, ztr))
            sig = gam - gam ** 2 * np.einsum('ij,ij->j', Phi_tr, Ki_Phi)
            gb = 0.5 * (mu[:B] ** 2 + mu[B:] ** 2
                        + np.maximum(sig[:B], 0) + np.maximum(sig[B:], 0))
            gam = np.concatenate([gb, gb])
            fv = (Phi_v * gam[None]) @ (Phi_tr.T @ np.linalg.solve(
                (Phi_tr * gam[None]) @ Phi_tr.T + s2 * np.eye(m), ztr))
            vr = float(np.sqrt(np.mean((fv - zv) ** 2)))
            if vr < best_em[0] - 1e-5:
                best_em = (vr, gam.copy())
            else:
                break
        gam = best_em[1]
    Phi_all = design(P)
    K = (Phi_all * gam[None]) @ Phi_all.T + s2 * np.eye(n)
    w_pred = Phi_all.T @ np.linalg.solve(K, z)
    out = np.empty(len(Pq))
    for i in range(0, len(Pq), 4096):
        out[i:i + 4096] = (design(Pq[i:i + 4096]) * gam[None]) @ w_pred
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_seas', type=int, default=10)
    ap.add_argument('--ckpt', default='results/direction3/prior_net.pt')
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=DEVICE)
    net = PriorNet().to(DEVICE)
    net.load_state_dict(ck['model'])
    net.eval()
    print(f"ckpt step {ck['step']} val {ck['val']:.4f}")

    lam = dispersion_lam(TP, DEPTH)
    cp = lam / TP
    T = 10 * TP
    fp = 0.18 * lam
    r_fp = fp / 2
    side = 2.0 * lam
    gx = np.linspace(0, side, 96)
    gy = np.linspace(0, side, 96)
    Xg, Yg = np.meshgrid(gx, gy)
    mu = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
    Xs, Ys = Xg[mu], Yg[mu]
    take = np.random.default_rng(0).choice(len(Xs), 1500, replace=False)
    Xs, Ys = Xs[take], Ys[take]
    t_star = T
    Pq = np.column_stack([Xs, Ys, np.full(len(Xs), t_star)])

    conds = ('fixed', 'track043', 'track100')
    res = {c: {e: [] for e in ('hand', 'net', 'netEM')} for c in conds}
    for si in range(a.n_seas):
        c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                               seed=4000 + si)
        th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
              np.array([x.A for x in c]), np.array([x.phi for x in c]))
        om_t = omega_of(th[0], th[1], DEPTH)
        gt_u = render_modes(*th, DEPTH, Xs, Ys, t_star)
        sigma = NOISE_FRAC * HS
        for cname in conds:
            rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
            U = (0.43 if cname == 'track043'
                 else 1.0 if cname == 'track100' else 0) * cp
            P = sample_obs('fixed' if cname == 'fixed' else 'moving', U,
                           lam, cp, T, gx, gy, r_fp, rng)
            psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
                   - om_t[None] * P[:, 2:3] + th[3][None])
            z = (th[2][None] * np.cos(psi)).sum(1) \
                + rng.normal(0, sigma, N_OBS)
            fh = mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=sigma, seed=si)
            res[cname]['hand'].append(corr(fh, gt_u))
            fn = net_gamma_mmse(net, P, z, DEPTH, Pq, sigma, seed=si,
                                use_em=False)
            res[cname]['net'].append(corr(fn, gt_u))
            fe = net_gamma_mmse(net, P, z, DEPTH, Pq, sigma, seed=si,
                                use_em=True)
            res[cname]['netEM'].append(corr(fe, gt_u))
        print(f"sea {si}: " + " | ".join(
            f"{cn} hand {res[cn]['hand'][-1]:+.2f} "
            f"net {res[cn]['net'][-1]:+.2f} "
            f"netEM {res[cn]['netEM'][-1]:+.2f}" for cn in conds),
            flush=True)

    print("\n===== G1 gate: net gamma vs hand prior (mean corr) =====")
    summ = {}
    for cn in conds:
        summ[cn] = {e: float(np.mean(res[cn][e])) for e in res[cn]}
        d = np.array(res[cn]['netEM']) - np.array(res[cn]['hand'])
        summ[cn]['d_netEM_hand'] = float(d.mean())
        print(f"{cn:10s} hand {summ[cn]['hand']:+.3f}  "
              f"net {summ[cn]['net']:+.3f}  netEM {summ[cn]['netEM']:+.3f}"
              f"  d(netEM-hand) {d.mean():+.3f} "
              f"(win+ {int((d>0).sum())}/{len(d)})")
    json.dump(dict(summary=summ, rows=res, ckpt_step=int(ck['step'])),
              open(OUT, 'w'), indent=1, default=float)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
