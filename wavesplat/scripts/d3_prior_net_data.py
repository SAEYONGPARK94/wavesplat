"""
d3_prior_net_data.py — training data generator for the amortized prior net
(DESIGN_amortized_prior_2026-07-05.md, module 1).

Each sample lives entirely in NORMALIZED k-space: the bank is the fixed
51x51 grid k/kp_est in [-2.5, 2.5]^2 (spacing 0.1 kp = mmse_bank_estimate's
dk_c), so the net is record-agnostic by construction.

  input  X: 3 x 51 x 51  [beamform power half1, half2, min]  (per-map max=1)
  label  Y: 51 x 51      log10(gamma* + eps), gamma* = sum A_m^2/2 of true
                          modes nearest-assigned to the cell (0 outside band)
  extras: P (400x3), z (400), kp_est, Hs, Tp, U/cp — for later task-loss.

Distribution: Tp~U[5,11], Hs~U[1,3], gamma_J~U[1.5,6], spread~U[15,45]deg,
mean dir~U[0,360), nc=200, noise frac~U[0.02,0.10], operator 1/3 fixed /
2/3 track with U/cp~U[0.1,1.2] (heading via sea rotation = random by dir).
HOS fraction: --hos_frac of samples get 3rd-order corrected z via
HOSSimulator sampling (slower; default 0 for the v1 linear batch).

Usage: python3 scripts/d3_prior_net_data.py --n 5000 --out tmp/prior_train/lin0
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d18_relax_estimator import beamform_top, omega
from d3_sa_nomp import beamform_power, split_masks
from d3_sa_achievability_gap import dispersion_lam, sample_obs, N_OBS

G = 9.81
DEPTH = 50.0
NGRID = 51                       # normalized bank grid (spacing 0.1 kp)
KREL = 2.5


def gen_sample(rng, fan_frac=0.0, bimodal_frac=0.0):
    Tp = rng.uniform(5.0, 11.0)
    Hs = rng.uniform(1.0, 3.0)
    gam_j = rng.uniform(1.5, 6.0)
    spread = rng.uniform(15.0, 45.0)
    mdir = rng.uniform(0.0, 360.0)
    seed = int(rng.integers(0, 2 ** 31 - 1))
    r2 = 0.0
    if rng.uniform() < bimodal_frac:      # crossing-sea augmentation
        r2 = rng.uniform(0.15, 0.5)
    c = jonswap_components(Hs=Hs * np.sqrt(1 - r2), Tp=Tp,
                           n_components=200, depth=DEPTH,
                           gamma=gam_j, direction_deg=mdir,
                           direction_spread_deg=spread, seed=seed)
    if r2 > 0:
        tp2 = float(np.clip(Tp * rng.uniform(0.6, 1.6), 4.0, 12.0))
        c = list(c) + list(jonswap_components(
            Hs=Hs * np.sqrt(r2), Tp=tp2, n_components=100, depth=DEPTH,
            gamma=gam_j, direction_deg=mdir + rng.uniform(40, 140),
            direction_spread_deg=spread, seed=seed + 7))
    kxm = np.array([x.kx for x in c])
    kym = np.array([x.ky for x in c])
    Am = np.array([x.A for x in c])
    phm = np.array([x.phi for x in c])
    k = np.hypot(kxm, kym)
    om = np.sqrt(G * k * np.tanh(k * DEPTH) + 1e-12)
    # ambient current (v4): true dispersion Doppler-shifted, model/labels
    # unchanged -> the net learns Doppler-TOLERANT priors (blind Uc
    # estimation is closed: 4 objectives picked spurious currents)
    uc_mag = rng.uniform(0.0, 0.4)
    uc_ang = rng.uniform(0, 2 * np.pi)
    om = om + kxm * uc_mag * np.cos(uc_ang) + kym * uc_mag * np.sin(uc_ang)

    lam = dispersion_lam(Tp, DEPTH)
    cp = lam / Tp
    T = 10 * Tp
    fp = 0.18 * lam
    r_fp = fp / 2
    side = 2.0 * lam
    gx = np.linspace(0, side, 96)
    gy = np.linspace(0, side, 96)
    rsel = rng.uniform()
    if rsel < fan_frac:                   # given-sensor forward fan
        mode = 'fan'
        U_abs = rng.uniform(1.0, 12.0)
        ucp = U_abs / cp
        tt = np.linspace(0, T, N_OBS)
        rr = rng.uniform(20.0, 100.0, N_OBS)
        aa = rng.uniform(-np.pi / 3, np.pi / 3, N_OBS)
        P = np.column_stack([U_abs * tt + rr * np.cos(aa),
                             rr * np.sin(aa), tt])
    elif rsel < fan_frac + (1 - fan_frac) / 3:
        mode, ucp = 'fixed', 0.0
        P = sample_obs(mode, 0.0, lam, cp, T, gx, gy, r_fp, rng)
    else:
        mode, ucp = 'moving', rng.uniform(0.1, 1.2)
        P = sample_obs(mode, ucp * cp, lam, cp, T, gx, gy, r_fp, rng)
    psi = (kxm[None] * P[:, 0:1] + kym[None] * P[:, 1:2]
           - om[None] * P[:, 2:3] + phm[None])
    nfrac = rng.uniform(0.02, 0.10)
    z = (Am[None] * np.cos(psi)).sum(1) + rng.normal(0, nfrac * Hs, N_OBS)

    # kp estimate exactly as at inference time (two-stage beamform zoom)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          DEPTH, klim=2.0, n=120)
    kp_e = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          DEPTH, klim=3.5 * kp_e, n=120)
    kp_e = max(np.hypot(bx, by), 1e-3)

    # 3-channel power maps on the normalized grid
    m1, m2 = split_masks(P, mode='legs')
    _, _, p1 = beamform_power(P[m1], z[m1], DEPTH, KREL * kp_e, NGRID)
    _, _, p2 = beamform_power(P[m2], z[m2], DEPTH, KREL * kp_e, NGRID)
    p1 = (p1 / max(p1.max(), 1e-12)).reshape(NGRID, NGRID)
    p2 = (p2 / max(p2.max(), 1e-12)).reshape(NGRID, NGRID)
    pm = np.minimum(p1, p2)
    # raw maps saturate near the peak; the order-48 prize lives in WEAK
    # modes -> log channels expose them (v2, after G1 v1 miss)
    X = np.stack([p1, p2, pm,
                  np.log10(p1 + 1e-6) / 6 + 1, np.log10(p2 + 1e-6) / 6 + 1,
                  np.log10(pm + 1e-6) / 6 + 1]).astype(np.float32)

    # label: true per-cell variance in normalized coords (nearest cell)
    Yv = np.zeros((NGRID, NGRID))
    ax = np.linspace(-KREL, KREL, NGRID)
    step = ax[1] - ax[0]
    ix = np.round((kxm / kp_e - ax[0]) / step).astype(int)
    iy = np.round((kym / kp_e - ax[0]) / step).astype(int)
    ok = (ix >= 0) & (ix < NGRID) & (iy >= 0) & (iy < NGRID)
    np.add.at(Yv, (iy[ok], ix[ok]), Am[ok] ** 2 / 2.0)
    Y = np.log10(Yv / max(Yv.max(), 1e-12) + 1e-6).astype(np.float32)

    # GT at random query pixels (task-loss target; excludes obs corridor
    # subtlety on purpose — random over the full patch is a superset of the
    # common-unobs scoring region)
    if mode == 'fan':                     # task queries = forecast region
        U_abs = ucp * cp
        n_enc = 128
        tq_e = T + rng.uniform(0, 10.0, n_enc)
        qx_e = U_abs * tq_e
        qy_e = np.zeros(n_enc)
        rrq = rng.uniform(10.0, 110.0, 128)
        aaq = rng.uniform(-np.pi / 3, np.pi / 3, 128)
        qx = np.concatenate([qx_e, U_abs * T + rrq * np.cos(aaq)])
        qy = np.concatenate([qy_e, rrq * np.sin(aaq)])
        tq = np.concatenate([tq_e, np.full(128, T)])
    else:
        qx = rng.uniform(0, side, 256)
        qy = rng.uniform(0, side, 256)
        tq = np.full(256, T)
    psi_q = (kxm[None] * qx[:, None] + kym[None] * qy[:, None]
             - om[None] * tq[:, None] + phm[None])
    gt_q = (Am[None] * np.cos(psi_q)).sum(1)
    Q = np.column_stack([qx, qy, tq]).astype(np.float32)

    return dict(X=X, Y=Y, P=P.astype(np.float32), z=z.astype(np.float32),
                Q=Q, gt_q=gt_q.astype(np.float32),
                kp_e=float(kp_e), Hs=float(Hs), Tp=float(Tp),
                ucp=float(ucp), nfrac=float(nfrac))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='tmp/prior_train/lin0')
    ap.add_argument('--shard', type=int, default=500)
    ap.add_argument('--fan_frac', type=float, default=0.0)
    ap.add_argument('--bimodal_frac', type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    rng = np.random.default_rng(a.seed)
    buf = []
    si = 0
    t0 = time.time()
    for i in range(a.n):
        buf.append(gen_sample(rng, a.fan_frac, a.bimodal_frac))
        if len(buf) == a.shard or i == a.n - 1:
            path = f"{a.out}_s{si:03d}.npz"
            np.savez_compressed(
                path,
                X=np.stack([b['X'] for b in buf]),
                Y=np.stack([b['Y'] for b in buf]),
                P=np.stack([b['P'] for b in buf]),
                z=np.stack([b['z'] for b in buf]),
                Q=np.stack([b['Q'] for b in buf]),
                gt_q=np.stack([b['gt_q'] for b in buf]),
                meta=np.array([[b['kp_e'], b['Hs'], b['Tp'], b['ucp'],
                                b['nfrac']] for b in buf], np.float32))
            print(f"shard {si} ({i + 1}/{a.n}) {time.time()-t0:.0f}s "
                  f"-> {path}", flush=True)
            buf, si = [], si + 1


if __name__ == '__main__':
    main()
