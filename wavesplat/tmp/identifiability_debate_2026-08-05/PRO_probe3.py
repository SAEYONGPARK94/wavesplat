"""PROBE 3 -- effective DoF of the observation operator, on the REAL stored
sample geometry of the S4 arm.  No wave run; the stored (x, y, t) of the
LiDAR returns is read and only linear algebra is done on it.

Why this is the load-bearing probe
----------------------------------
Two components at the SAME omega differ only through exp(i k.x): the aperture
is their only separating axis.  Two components at DIFFERENT omega differ
through exp(i(k.x - omega t)) as well: time is a second separating axis.
So moving angular variance from `across` to `within` is not a relabelling --
it removes a separating axis.  This probe measures how much.

Metric: participation ratio of the eigenvalues of the (amplitude-weighted)
Gram matrix of the design columns [cos(k.x - w t), sin(k.x - w t)],
PR = (sum lam)^2 / sum lam^2, in units of "effective real columns".
Normalised by 2N it is the fraction of the nominal parameter budget the
sensor can actually see.  It is a linear (amplitude-block / Fisher) proxy,
not the full nonlinear estimator.
"""
import glob
import json
import os
import sys

import h5py
import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
HERE = os.path.join(ROOT, 'tmp/identifiability_debate_2026-08-05')
sys.path.insert(0, ROOT)
from wavesim.wave_generator import jonswap_components  # noqa

HS, TP, DEPTH, SPREAD = 1.2, 7.0, 50.0, 30.0
SEEDS = [0, 1, 2, 3, 4]
DATA = os.path.join(ROOT, 'data_dirspread_2026-08-05')


def load_geom(seed, n_frames):
    """Return (x, y, t) of the valid LiDAR returns of the first n_frames."""
    fn = os.path.join(DATA, f'episode_BF4_30comp_seed{seed}_uniform_m3.h5')
    with h5py.File(fn, 'r') as f:
        P = f['frames/lidar_points'][:n_frames]
        M = f['frames/lidar_valid_mask'][:n_frames]
        ts = f['frames/timestamps'][:n_frames]
    xs, ys, tt = [], [], []
    for i in range(P.shape[0]):
        v = P[i][M[i]]
        xs.append(v[:, 0]); ys.append(v[:, 1])
        tt.append(np.full(v.shape[0], ts[i]))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(tt))


def eff_dof(A, om, th, k, x, y, t):
    """Participation ratio of the amplitude-weighted design Gram."""
    kx = k * np.cos(th)
    ky = k * np.sin(th)
    ph = np.outer(x, kx) + np.outer(y, ky) - np.outer(t, om)   # (P, N)
    Phi = np.concatenate([np.cos(ph) * A, np.sin(ph) * A], axis=1)  # (P, 2N)
    Phi = Phi / np.sqrt(len(x))
    lam = np.linalg.eigvalsh(Phi.T @ Phi)
    lam = np.clip(lam, 0, None)
    s = lam.sum()
    pr = float(s ** 2 / (lam ** 2).sum()) if s > 0 else 0.0
    nz = lam[lam > lam.max() * 1e-10]
    return dict(pr=pr, pr_frac=pr / (2 * len(A)),
                rank_1e6=int((lam > lam.max() * 1e-6).sum()),
                cond_log10=float(np.log10(lam.max() / nz.min())) if nz.size else np.nan)


def hier(N, M, delta, seed, spread_deg=SPREAD):
    base = jonswap_components(Hs=HS, Tp=TP, n_components=N, depth=DEPTH,
                              direction_spread_deg=spread_deg, seed=seed,
                              dirs_per_freq=M)
    A = np.array([c.A for c in base])
    om = np.array([c.omega for c in base])
    k = np.array([c.k for c in base])
    n_freq = N // M
    r = np.random.default_rng(10_000 + seed)
    b = r.normal(0.0, 1.0, n_freq)
    w = r.normal(0.0, 1.0, N)
    sig = np.deg2rad(spread_deg)
    sw = delta * sig
    sb = np.sqrt(max(sig ** 2 - sw ** 2, 0.0))
    bin_of = np.repeat(np.arange(n_freq), M)[:N]
    return A, om, sb * b[bin_of] + sw * w, k


def shipped(N, M, seed):
    c = jonswap_components(Hs=HS, Tp=TP, n_components=N, depth=DEPTH,
                           direction_spread_deg=SPREAD, seed=seed, dirs_per_freq=M)
    return (np.array([x.A for x in c]), np.array([x.omega for x in c]),
            np.array([np.arctan2(x.ky, x.kx) for x in c]),
            np.array([x.k for x in c]))


out = {}

# ---- aperture facts (MEASURED from the stored geometry) --------------------
g1 = load_geom(0, 1)
g20 = load_geom(0, 200)
ap = dict(frame1_n=int(len(g1[0])),
          frame1_x=[float(g1[0].min()), float(g1[0].max())],
          frame1_y=[float(g1[1].min()), float(g1[1].max())],
          win20s_n=int(len(g20[0])),
          win20s_x=[float(g20[0].min()), float(g20[0].max())],
          win20s_y=[float(g20[1].min()), float(g20[1].max())])
ap['frame1_D'] = float(np.hypot(ap['frame1_x'][1] - ap['frame1_x'][0],
                                ap['frame1_y'][1] - ap['frame1_y'][0]))
ap['win20s_D'] = float(np.hypot(ap['win20s_x'][1] - ap['win20s_x'][0],
                                ap['win20s_y'][1] - ap['win20s_y'][0]))
for lam, nm in ((21.761, 'lambda_min'), (60.444, 'lambda_peak')):
    ap[f'rayleigh_deg_{nm}_frame1'] = float(np.degrees(lam / ap['frame1_D']))
    ap[f'rayleigh_deg_{nm}_win20s'] = float(np.degrees(lam / ap['win20s_D']))
out['aperture'] = ap
print('=== aperture (MEASURED, stored geometry) ===')
for k_, v_ in ap.items():
    print(f'  {k_}: {v_}')

# ---- the three shipped arms ------------------------------------------------
print('\n=== PROBE 3a: effective DoF of the three S4 arms ===')
WINS = [(1, '1 frame (0.1 s)'), (30, '3 s'), (200, '20 s')]
p3a = []
for nf, lab in WINS:
    for name, N, M in (('A_M1_N30', 30, 1), ('B_M3_N30', 30, 3), ('C_M1_N10', 10, 1)):
        rs = []
        for s in SEEDS:
            x, y, t = load_geom(s, nf)
            A, om, th, k = shipped(N, M, s)
            rs.append(eff_dof(A, om, th, k, x, y, t))
        med = {kk: float(np.median([r[kk] for r in rs])) for kk in rs[0]}
        med.update(arm=name, N=N, M=M, n_frames=nf, window=lab, n_pts=int(len(x)))
        p3a.append(med)
        print(f"  {lab:16s} {name:9s} 2N={2*N:3d}  eff_dof={med['pr']:6.2f} "
              f"({100*med['pr_frac']:5.1f}% of budget)  rank={med["rank_1e6"]:5.1f}  "
              f"log10cond={med['cond_log10']:.1f}")
out['probe3a_arms'] = p3a

# ---- the hierarchical sweep -----------------------------------------------
print('\n=== PROBE 3b: eff DoF along the escape-b\' knob (N,n_freq,M all fixed) ===')
p3b = []
for nf, lab in WINS:
    print(f'  -- window {lab} --')
    for delta in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        rs = []
        for s in SEEDS:
            x, y, t = load_geom(s, nf)
            A, om, th, k = hier(30, 3, delta, s)
            rs.append(eff_dof(A, om, th, k, x, y, t))
        med = {kk: float(np.median([r[kk] for r in rs])) for kk in rs[0]}
        med.update(delta=delta, n_frames=nf, window=lab)
        p3b.append(med)
        print(f"    delta={delta:.2f}  eff_dof={med['pr']:6.2f} "
              f"({100*med['pr_frac']:5.1f}%)  rank={med["rank_1e6"]:5.1f}  "
              f"log10cond={med['cond_log10']:.1f}")
out['probe3b_hier'] = p3b

json.dump(out, open(os.path.join(HERE, 'PRO_probe3.json'), 'w'), indent=2)
print('\nwrote PRO_probe3.json')
