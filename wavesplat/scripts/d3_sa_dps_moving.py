"""
d3_sa_dps_moving.py — Does the DPS field/component prior (B2: frozen
set-diffusion + physics polish) survive a MOVING sampling operator?

Item (3) of the 2026-07-04 queue. The B2 posterior was only ever evaluated on
the episodes' native (scattered LiDAR-frame) observations. Here the SAME
frozen prior + polish runs on obs resampled from the episode GT frames under
the SA protocol operators:
  fixed      disk (0.18 lam_p) at the domain center
  track043   ping-pong diagonal at U = 0.43 c_p
  track100   U = 1.0 c_p
matched budget (150 obs = B2's M_CAP), same frame times, 2% noise (B2 SIGMA).
Comparators: RELAX on identical obs. Scores: full-grid nRMSE (B2 convention)
and corr on the common-unobserved region (SA convention).

Question: the moving-platform prize is support identification — a field-level
generative prior sidesteps explicit support estimation entirely. Does it cash
any of the moving-condition gap, or does it degrade like the greedy/dense/
learned-detect families?

Usage: python3 scripts/d3_sa_dps_moving.py [--episodes ...] [--nsamp 4]
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

from wavesplat.data_loader import EpisodeDataLoader
from d3_setdiff_prior import SetDenoiser
from d3_setdiff_posterior import (dps_sample, atoms_to_field, DEVICE, CKPT,
                                  SIGMA, W)
from d3_setdiff_posterior_b2 import field_set
from estimator_improved import iterative_relax_cv

G = 9.81
N_OBS = 150                      # = B2 M_CAP: keep the prior in-distribution
OUT = 'results/direction3/sa_dps_moving.json'


def build_obs(frames, ti, gx, gy, mode, U, r_fp, rng):
    """Obs resampled from GT frames along the operator; one batch per frame."""
    sel = frames[ti - W + 1:ti + 1]
    n_per = int(np.ceil(N_OBS / len(sel)))
    x0, x1, y0, y1 = gx.min(), gx.max(), gy.min(), gy.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    P0 = np.array([x0 + r_fp, y0 + r_fp])
    P1 = np.array([x1 - r_fp, y1 - r_fp])
    Ld = float(np.linalg.norm(P1 - P0))
    u = (P1 - P0) / Ld
    t0 = float(sel[0].t)
    P, z = [], []
    for f in sel:
        gt = f.gt_field.numpy()
        if mode == 'fixed':
            c = np.array([cx, cy])
        else:
            s = np.mod(U * (float(f.t) - t0), 2 * Ld)
            d = s if s <= Ld else 2 * Ld - s
            c = P0 + d * u
        th = rng.uniform(0, 2 * np.pi, n_per)
        rr = r_fp * np.sqrt(rng.uniform(0, 1, n_per))
        xs = np.clip(c[0] + rr * np.cos(th), x0, x1)
        ys = np.clip(c[1] + rr * np.sin(th), y0, y1)
        ix = np.round((xs - gx[0]) / (gx[1] - gx[0])).astype(int)
        iy = np.round((ys - gy[0]) / (gy[1] - gy[0])).astype(int)
        zz = gt[np.clip(iy, 0, gt.shape[0] - 1),
                np.clip(ix, 0, gt.shape[1] - 1)]
        P.append(np.column_stack([xs, ys, np.full(n_per, float(f.t))]))
        z.append(zz)
    P = np.vstack(P)[:N_OBS]
    z = np.concatenate(z)[:N_OBS] + rng.normal(0, SIGMA, N_OBS)
    return P.astype(np.float32), z.astype(np.float32)


def unobs_mask(gx, gy, r_fp, margin):
    Xg, Yg = np.meshgrid(gx, gy)
    cx, cy = (gx[0] + gx[-1]) / 2, (gy[0] + gy[-1]) / 2
    dA = np.hypot(Xg - cx, Yg - cy)
    P0 = np.array([gx[0] + r_fp, gy[0] + r_fp])
    P1 = np.array([gx[-1] - r_fp, gy[-1] - r_fp])
    seg = P1 - P0
    tpar = np.clip(((Xg - P0[0]) * seg[0] + (Yg - P0[1]) * seg[1])
                   / float(seg @ seg), 0, 1)
    dB = np.hypot(Xg - (P0[0] + tpar * seg[0]), Yg - (P0[1] + tpar * seg[1]))
    return (dA > r_fp + margin) & (dB > r_fp + margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', nargs='+', default=[
        'episode_BF4_15comp_seed0_uniform.h5',
        'episode_BF4_30comp_seed0_uniform.h5'])
    ap.add_argument('--guidance', type=float, default=0.1)
    ap.add_argument('--nsamp', type=int, default=4)
    ap.add_argument('--polish_iters', type=int, default=80)
    a = ap.parse_args()

    ck = torch.load(CKPT, map_location=DEVICE)
    stats = ck['stats']
    model = SetDenoiser(**ck.get('arch', {})).to(DEVICE)
    model.load_state_dict(ck['model'])
    model.eval()
    abar = ck['abar'].to(DEVICE)
    betas = ck['betas'].to(DEVICE)

    out = {}
    for ep in a.episodes:
        L = EpisodeDataLoader('data/' + ep)
        depth, Hs = L.depth, L.Hs_true
        gx = L.gt_grid_x.numpy()
        gy = L.gt_grid_y.numpy()
        lam_p = G * L.Tp_true ** 2 / (2 * np.pi)
        cp = lam_p / L.Tp_true
        r_fp = 0.09 * lam_p
        mu = unobs_mask(gx, gy, r_fp, 1.2 * r_fp)
        frames = L.get_tracking_frames(2.0)
        ti = 25
        tgt = frames[ti + 1]
        target = {'gx': gx, 'gy': gy, 't': float(tgt.t),
                  'gt': tgt.gt_field.numpy(), 'Hs': Hs}
        gt = target['gt']
        print(f"[{ep}] lam_p={lam_p:.0f} m patch "
              f"{gx.max()-gx.min():.0f}x{gy.max()-gy.min():.0f} m "
              f"({(gx.max()-gx.min())/lam_p:.1f} lam) unobs px {int(mu.sum())}")
        out[ep] = {}
        for mode, ur in [('fixed', None), ('track043', 0.43),
                         ('track100', 1.0)]:
            rng = np.random.default_rng(42)
            P, z = build_obs(frames, ti, gx, gy,
                             'fixed' if mode == 'fixed' else 'moving',
                             (ur or 0) * cp, r_fp, rng)
            # DPS + polish (B2)
            Xd = dps_sample(model, stats, abar, betas, P, z, depth,
                            guidance=a.guidance, n=a.nsamp, seed=0)
            f_pol, r_pol = field_set(Xd, stats, P, z, depth, target,
                                     polish=True, pit=a.polish_iters)
            fb = np.mean(np.stack(f_pol), axis=0)
            c_dps = float(np.corrcoef(fb[mu], gt[mu])[0, 1])
            # RELAX on identical obs
            fr, N, _ = iterative_relax_cv(P.copy(), z.copy(), depth, target,
                                          n_outer=2, seed=0)
            nr_rlx = float(np.sqrt(np.mean((fr - gt) ** 2)) / (Hs / 4))
            c_rlx = float(np.corrcoef(fr[mu], gt[mu])[0, 1])
            out[ep][mode] = dict(dps_nrmse=float(np.mean(r_pol)),
                                 dps_corr_unobs=c_dps,
                                 relax_nrmse=nr_rlx, relax_corr_unobs=c_rlx,
                                 relax_N=int(N))
            print(f"  {mode:9s} DPS+pol nrmse {np.mean(r_pol):.3f} "
                  f"unobs corr {c_dps:+.3f} | RELAX nrmse {nr_rlx:.3f} "
                  f"unobs corr {c_rlx:+.3f}")
    json.dump(out, open(OUT, 'w'), indent=1, default=float)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
