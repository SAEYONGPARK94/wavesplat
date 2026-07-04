"""
d3_sa_moving_ident_probe.py — Does component IDENTIFICATION survive a moving-
track sampling operator? Zero-training probe of the static-trained detection
net vs RELAX on synthetic seas.

Context (2026-07-04 multi-record verdict): the moving-platform failure on real
broadband seas is an identifiability breakdown (forcing RELAX to more
components makes fields worse). The learned detector line (d3_beamform_net)
was trained with obs SCATTERED over the whole pool + random times — never a
moving track. Question 1 (this script): under a matched-budget moving track,
does the static-trained net's identification degrade more or less than RELAX?
Question 2 (if it degrades): does track-aware retraining recover it — run
d3_beamform_net-style training with mixed operators (--retrain).

Conditions (matched budget M=400, times linspace over Tobs = 2.2 Tp):
  scatter    obs at random pool points (net's training distribution — control)
  disk       fixed footprint 0.18 lam_p at domain center (real-protocol A)
  track043   ping-pong chord through center, U = 0.43 c_p
  track100   same, U = 1.0 c_p
Metrics per sea (nc in {15,30} JONSWAP + 2% noise, 20 seeds):
  chamfer    mean symmetric Chamfer distance between detected (kx,ky) and the
             top-10-amplitude true components (rad/m)
  corr/nrmse field skill on the full grid at t* = Tobs (nowcast)

Usage:
  python3 scripts/d3_sa_moving_ident_probe.py                # probe (Q1)
  python3 scripts/d3_sa_moving_ident_probe.py --retrain      # Q2: mixed-op net
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

import d3_beamform_net as BN
from d3_beamform_net import (DEVICE, M_OBS, DEPTH, K_TP, beamform, extract,
                             DetectCNN, target_heatmap)
from d40_learned_acquisition import pool_xy
from wavesim.wave_generator import jonswap_components
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from estimator_improved import iterative_relax_cv

G = 9.81
HS, TP = 2.0, 7.0
CKPT_STATIC = 'results/direction3/beamform_net.pt'
CKPT_MIXED = 'results/direction3/beamform_net_trackaware.pt'
OUT = 'results/direction3/sa_moving_ident_probe.json'


def sea(seed, nc):
    c = jonswap_components(Hs=HS, Tp=TP, n_components=nc, depth=DEPTH, seed=seed)
    return (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
            np.array([x.A for x in c]), np.array([x.omega for x in c]),
            np.array([x.phi for x in c]))


def field_of(comps, gx, gy, t):
    kx, ky, A, om, ph = comps
    X, Y = np.meshgrid(gx, gy)
    psi = (kx[None, None] * X[..., None] + ky[None, None] * Y[..., None]
           - om[None, None] * t + ph[None, None])
    return (A[None, None] * np.cos(psi)).sum(-1)


def obs_positions(mode, pool, lam_p, cp, Tobs, rng, U_ratio=None):
    """(M,3) space-time sample positions for the given operator."""
    tt = np.linspace(0, Tobs, M_OBS)
    x0, x1 = pool[:, 0].min(), pool[:, 0].max()
    y0, y1 = pool[:, 1].min(), pool[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if mode == 'scatter':
        idx = rng.choice(len(pool), M_OBS, replace=False)
        return np.column_stack([pool[idx, 0], pool[idx, 1],
                                rng.uniform(0, Tobs, M_OBS)])
    if mode == 'disk':
        r = 0.09 * lam_p                                  # diam 0.18 lam
        th = rng.uniform(0, 2 * np.pi, M_OBS)
        rr = r * np.sqrt(rng.uniform(0, 1, M_OBS))
        return np.column_stack([cx + rr * np.cos(th), cy + rr * np.sin(th), tt])
    # moving ping-pong chord along x through center, footprint jitter as disk
    U = U_ratio * cp
    L = (x1 - x0) * 0.9
    s = np.mod(U * tt, 2 * L)
    d = np.where(s <= L, s, 2 * L - s)
    r = 0.09 * lam_p
    th = rng.uniform(0, 2 * np.pi, M_OBS)
    rr = r * np.sqrt(rng.uniform(0, 1, M_OBS))
    return np.column_stack([cx - L / 2 + d + rr * np.cos(th),
                            cy + rr * np.sin(th), tt])


def chamfer(kd, kdy, kt, kty, w):
    """symmetric Chamfer between detected and top-10 true comps (by amplitude)."""
    top = np.argsort(w)[::-1][:10]
    T = np.column_stack([kt[top], kty[top]])
    if len(kd) == 0:
        return float(np.hypot(*T.T).mean())
    D = np.column_stack([kd, kdy])
    dm = np.hypot(D[:, None, 0] - T[None, :, 0], D[:, None, 1] - T[None, :, 1])
    return float(0.5 * (dm.min(1).mean() + dm.min(0).mean()))


@torch.no_grad()
def net_estimate(model, P, z, gx, gy, t_star):
    Bmap = beamform(torch.tensor(P[None], device=DEVICE, dtype=torch.float32),
                    torch.tensor(z[None], device=DEVICE, dtype=torch.float32))
    kx, ky = extract(model(Bmap)[0])
    if len(kx) == 0:
        return np.zeros((len(gy), len(gx))), kx, ky
    obs = np.column_stack([P, z]).astype(np.float32)
    A, phi = fit_amplitudes_phases(kx, ky, obs, DEPTH, lambda_reg=0.5)
    fld = render_wave_field(kx, ky, A, phi, DEPTH, gx, gy, t_star)
    return fld, kx, ky


def run_probe(model, tag, seeds=range(20)):
    pool = pool_xy().astype(np.float32)
    lam_p = G * TP ** 2 / (2 * np.pi)                    # deep-ish at 50 m
    cp = lam_p / TP
    Tobs = K_TP * TP
    gx = np.linspace(pool[:, 0].min(), pool[:, 0].max(), 96)
    gy = np.linspace(pool[:, 1].min(), pool[:, 1].max(), 96)
    modes = [('scatter', None), ('disk', None), ('track043', 0.43),
             ('track100', 1.0)]
    res = {m: {'relax': [], 'net': []} for m, _ in modes}
    for seed in seeds:
        nc = [5, 15, 30][seed % 3]
        comps = sea(1000 + seed, nc)
        gt = field_of(comps, gx, gy, Tobs)
        Hs_w = 4 * gt.std()
        for mode, ur in modes:
            rng = np.random.default_rng(7000 + seed)
            P = obs_positions(mode.replace('043', '').replace('100', ''),
                              pool, lam_p, cp, Tobs, rng, U_ratio=ur)
            kx, ky, A, om, ph = comps
            psi = (kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2]
                   - om[None] * P[:, 2:3] + ph[None])
            z = (A[None] * np.cos(psi)).sum(1) + rng.normal(0, 0.02, M_OBS)
            P32, z32 = P.astype(np.float32), z.astype(np.float32)
            # RELAX
            target = {'gx': gx, 'gy': gy, 't': Tobs, 'gt': gt, 'Hs': Hs_w}
            fr, N, _ = iterative_relax_cv(P32.copy(), z32.copy(), DEPTH, target,
                                          n_outer=2, seed=seed)
            # net
            fn, kdx, kdy = net_estimate(model, P32, z32, gx, gy, Tobs)
            for name, fld, kk in (('relax', fr, None), ('net', fn, (kdx, kdy))):
                c = float(np.corrcoef(fld.ravel(), gt.ravel())[0, 1])
                nr = float(np.sqrt(np.mean((fld - gt) ** 2)) / (Hs_w / 4))
                row = dict(seed=seed, nc=nc, corr=c, nrmse=nr)
                if kk is not None:
                    row['chamfer'] = chamfer(kk[0], kk[1], kx, ky, A)
                    row['n_det'] = int(len(kk[0]))
                else:
                    row['N'] = int(N)
                res[mode][name].append(row)
    return res


def summarize(res, label):
    print(f"\n=== {label} ===")
    print(f"{'mode':10s} {'RELAX corr':>11s} {'net corr':>9s} {'net chamfer':>12s} "
          f"{'net n_det':>10s} {'RELAX N':>8s}")
    out = {}
    for mode, d in res.items():
        rc = np.mean([r['corr'] for r in d['relax']])
        nc_ = np.mean([r['corr'] for r in d['net']])
        ch = np.mean([r['chamfer'] for r in d['net']])
        nd = np.mean([r['n_det'] for r in d['net']])
        NN = np.mean([r['N'] for r in d['relax']])
        print(f"{mode:10s} {rc:>11.3f} {nc_:>9.3f} {ch:>12.3f} {nd:>10.1f} {NN:>8.1f}")
        out[mode] = dict(relax_corr=float(rc), net_corr=float(nc_),
                         net_chamfer=float(ch), net_ndet=float(nd),
                         relax_N=float(NN))
    return out


def make_batch_mixed(Bsz, pool):
    """Track-aware training batch: 1/3 scatter (original), 1/3 disk, 1/3 track."""
    lam_ref = G * (8.0) ** 2 / (2 * np.pi)
    P = np.zeros((Bsz, M_OBS, 3), np.float32)
    Z = np.zeros((Bsz, M_OBS), np.float32)
    comps = []
    for b in range(Bsz):
        Hs = np.random.uniform(0.5, 3.5)
        Tp = np.random.uniform(5.0, 12.0)
        nc = int(np.random.choice([5, 15, 30]))
        c = jonswap_components(Hs=Hs, Tp=Tp, n_components=nc, depth=DEPTH,
                               seed=int(np.random.randint(1e9)))
        kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c])
        A = np.array([x.A for x in c]); om = np.array([x.omega for x in c])
        ph = np.array([x.phi for x in c])
        Tobs = K_TP * Tp
        lam_p = G * Tp ** 2 / (2 * np.pi)
        cp = lam_p / Tp
        rng = np.random.default_rng(np.random.randint(1e9))
        u = b % 3
        if u == 0:
            mode, ur = 'scatter', None
        elif u == 1:
            mode, ur = 'disk', None
        else:
            mode, ur = 'track', float(np.random.uniform(0.15, 1.2))
        Pb = obs_positions(mode, pool, lam_p, cp, Tobs, rng, U_ratio=ur)
        # random track direction: rotate positions about domain center
        if mode == 'track':
            cx = (pool[:, 0].min() + pool[:, 0].max()) / 2
            cy = (pool[:, 1].min() + pool[:, 1].max()) / 2
            th = np.random.uniform(0, np.pi)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            Pb[:, :2] = (Pb[:, :2] - [cx, cy]) @ R.T + [cx, cy]
        psi = (kx[None] * Pb[:, 0:1] + ky[None] * Pb[:, 1:2]
               - om[None] * Pb[:, 2:3] + ph[None])
        Z[b] = (A[None] * np.cos(psi)).sum(1) + np.random.normal(0, 0.02, M_OBS)
        P[b] = Pb
        comps.append((kx, ky, A ** 2))
    Pt = torch.tensor(P, device=DEVICE)
    Zt = torch.tensor(Z, device=DEVICE)
    Bmap = beamform(Pt, Zt)
    H = torch.stack([target_heatmap(torch.tensor(kx, device=DEVICE),
                                    torch.tensor(ky, device=DEVICE),
                                    torch.tensor(w, device=DEVICE))
                     for kx, ky, w in comps])
    return Bmap, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrain', action='store_true')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--seeds', type=int, default=20)
    args = ap.parse_args()

    out = {}
    model = DetectCNN().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_STATIC, map_location=DEVICE))
    model.eval()
    res = run_probe(model, 'static', seeds=range(args.seeds))
    out['static'] = summarize(res, 'static-trained net vs RELAX')

    if args.retrain:
        print(f"\nretraining track-aware net ({args.steps} steps)...")
        BN.make_batch = make_batch_mixed          # swap the batch generator
        model2 = BN.train(args.steps)
        torch.save(model2.state_dict(), CKPT_MIXED)
        print(f"saved {CKPT_MIXED}")
        model2.eval()
        res2 = run_probe(model2, 'trackaware', seeds=range(args.seeds))
        out['trackaware'] = summarize(res2, 'track-aware net vs RELAX')

    json.dump(out, open(OUT, 'w'), indent=1, default=float)
    print(f"\nsaved {OUT}")


if __name__ == '__main__':
    main()
