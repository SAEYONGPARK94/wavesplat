"""E-6: is BF2's residual weakness observation sampling, or the sea?

Prereg (frozen before numbers, 2026-07-13): docs/E2_MESH_AUDIT_prereg_2026-07-13.md

After the mesh was fixed, BF2 1-comp still scores worst on the wide fan
(0.36 vs 0.85-0.93 for BF3-BF5).  We attributed that to observation sampling --
an 8.2 m wave seen out to 200 m -- but never tested it.  It has been sitting in
the record as an assertion.  Two falsifiable predictions:

  P1 (range):  LiDAR return spacing grows with range, so points-per-wavelength
               falls with range.  If sampling is the binding constraint, BF2's
               rho must collapse with range exactly where its observed
               points-per-wavelength crosses ~2, while BF4 (lambda ~65 m) --
               same rays, same mesh -- stays flat.
               If instead BF2 is uniformly bad at all ranges, the sea (or the
               estimator), not sampling, is the limit.

  P2 (density): give the estimator MORE points from the same episodes.  If
               sampling-limited, BF2 improves with observation count; if the
               information is simply absent, more points of the same thing
               change nothing.

Both run on the FINE-mesh episodes, so the renderer is out of the picture.
"""
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_g0b_selection_aware import BAND, DEV, DT      # noqa: E402
from d3_gpu_mmse import build_bank, design, mmse_gpu  # noqa: E402

FINE = 'tmp/finemesh_clean'
OUT = 'results/direction3/e6_sampling_test.json'
SEEDS = [0, 1, 2, 3, 4]      # BF4 control needs all five, not one
RANGE_BINS = [(15, 50), (50, 85), (85, 120), (120, 160), (160, 200)]
# P2 sweep. 600 = the sparse regime the thesis assumes (50 returns/frame x
# 12 frames); 2000 = the deployment recipe; the rest exhaust the window.
N_OBS = [500, 600, 2000, 8000, 24000, 60000]


def load(path, rng, n_obs):
    with h5py.File(path, 'r') as f:
        comp = f['config/wave_components'][:]
        ts = f['frames/timestamps'][:]
        pts = f['frames/lidar_points']
        gx, gy = f['frames/gt_grid_x'][:], f['frames/gt_grid_y'][:]
        gz = f['frames/gt_wave_field']

        kp = float(np.hypot(comp[:, 2], comp[:, 3])[np.argmax(comp[:, 0])])
        lam = 2 * np.pi / kp
        obs_i = list(range(50, 249, 18))
        t_end = ts[obs_i[-1]]
        P, zv = [], []
        for i in obs_i:
            p = pts[i]
            P.append(np.column_stack([p[:, 0], p[:, 1], np.full(len(p), ts[i])]))
            zv.append(p[:, 2])
        P, zv = np.vstack(P), np.concatenate(zv)
        rr = np.hypot(P[:, 0], P[:, 1])

        # ceil, so n_obs=2000 gives the frozen recipe's 167/bin exactly.
        # int() gave 166 and quietly made this a different recipe.
        per_bin = max(-(-n_obs // 12), 1)
        edges = np.linspace(rr.min(), rr.max() + 1e-6, 13)
        b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
        take = []
        for j in range(12):
            idx = np.where(b == j)[0]
            k = min(len(idx), per_bin)
            if k:
                take.append(rng.choice(idx, k, replace=False))
        sel = np.concatenate(take)
        gt_end = gz[obs_i[-1]][:]
    return dict(Ps=P[sel], zs=zv[sel], kp=kp, lam=lam, t_end=t_end,
                gx=gx, gy=gy, gt_end=gt_end, P_all=P, r_all=rr)


def fit(ep):
    Ps, zs, kp = ep['Ps'], ep['zs'], ep['kp']
    t32 = torch.tensor(Ps, device=DEV, dtype=torch.float32)
    z32 = torch.tensor(zs, device=DEV, dtype=torch.float32)
    _, mdl = mmse_gpu(t32, z32, 50.0, t32[:64], 0.05, kp, seed=0,
                      return_model=True, no_em=True, pexp=1.0, smult=10.0)
    gam, s2 = mdl['gam'].to(DT), float(mdl['s2'])
    kxb, kyb, om = build_bank(kp, BAND, kp / 10.0, 50.0, DEV, torch.float32)
    kxb, kyb, om = kxb.to(DT), kyb.to(DT), om.to(DT)
    Phi = design(torch.tensor(Ps, device=DEV, dtype=DT), kxb, kyb, om)
    A0 = Phi.T @ Phi / s2 + torch.diag(1.0 / gam)
    a = torch.linalg.solve(A0, Phi.T @ torch.tensor(zs, device=DEV, dtype=DT) / s2)
    return (kxb, kyb, om), a


def rho_in(ep, bank, a, rmin, rmax):
    X, Y = np.meshgrid(ep['gx'], ep['gy'])
    R = np.hypot(X, Y)
    m = (R >= rmin) & (R <= rmax)
    if m.sum() < 20:
        return float('nan')
    Q = np.column_stack([X[m], Y[m], np.full(m.sum(), ep['t_end'])])
    C = design(torch.tensor(Q, device=DEV, dtype=DT), *bank)
    return float(np.corrcoef((C @ a).cpu().numpy(), ep['gt_end'][m])[0, 1])


def obs_pts_per_lambda(ep, rmin, rmax):
    """Points per wavelength actually delivered by the sensor in this annulus.

    Areal density n/A -> mean spacing 1/sqrt(density) -> lambda / spacing.
    """
    m = (ep['r_all'] >= rmin) & (ep['r_all'] < rmax)
    n = int(m.sum())
    if n == 0:
        return 0.0
    n_per_frame = n / 12.0                       # the window holds 12 frames
    area = np.pi * (rmax ** 2 - rmin ** 2) / 3.0  # 120 deg fan
    spacing = 1.0 / np.sqrt(max(n_per_frame / area, 1e-12))
    return float(ep['lam'] / spacing)


def main():
    res = {'prereg': 'docs/E2_MESH_AUDIT_prereg_2026-07-13.md',
           'note': 'all on FINE mesh -- renderer excluded', 'P1': {}, 'P2': {}}

    # ---- P1: rho and delivered sampling vs range, BF2 vs BF4 control --------
    for sea, name in (('BF2', 'episode_BF2_1comp_seed{}_fixedT.h5'),
                      ('BF4', 'episode_BF4_15comp_seed{}_fixedT.h5')):
        rows = []
        for s in SEEDS:
            p = os.path.join(FINE, name.format(s))
            if not os.path.exists(p):
                continue
            ep = load(p, np.random.default_rng(11 + s), 2000)
            bank, a = fit(ep)
            rows.append(dict(
                seed=s, lam=ep['lam'],
                rho=[rho_in(ep, bank, a, lo, hi) for lo, hi in RANGE_BINS],
                pts_lam=[obs_pts_per_lambda(ep, lo, hi) for lo, hi in RANGE_BINS]))
        if not rows:
            continue
        mono = int(sum(all(r['rho'][j] >= r['rho'][j + 1] - 1e-9
                           for j in range(len(RANGE_BINS) - 1)) for r in rows))
        res['P1'][sea] = dict(
            lam=float(np.mean([r['lam'] for r in rows])),
            bins=[list(b) for b in RANGE_BINS],
            rho=np.nanmean([r['rho'] for r in rows], axis=0).tolist(),
            pts_lam=np.nanmean([r['pts_lam'] for r in rows], axis=0).tolist(),
            n_seeds=len(rows),
            monotone_seeds=mono,          # G-E6-1 is a per-seed gate, not a mean
            seeds=rows)
        r = res['P1'][sea]
        print(f"[P1] {sea} (lam {r['lam']:.1f} m)")
        for i, (lo, hi) in enumerate(RANGE_BINS):
            print(f"   r {lo:3d}-{hi:3d} m: rho {r['rho'][i]:+.3f}   "
                  f"delivered {r['pts_lam'][i]:5.1f} pts/lambda")

    # ---- P2: does more observation help BF2? -------------------------------
    for sea, name in (('BF2', 'episode_BF2_1comp_seed{}_fixedT.h5'),
                      ('BF4', 'episode_BF4_15comp_seed{}_fixedT.h5')):
        out = {}
        for n in N_OBS:
            fans = []
            for s in SEEDS:
                p = os.path.join(FINE, name.format(s))
                if not os.path.exists(p):
                    continue
                ep = load(p, np.random.default_rng(11 + s), n)
                bank, a = fit(ep)
                fans.append(rho_in(ep, bank, a, 15.0, 200.0))
            out[str(n)] = dict(fan=float(np.mean(fans)),
                               se=float(np.std(fans, ddof=1) / np.sqrt(len(fans))),
                               n_obs_actual=int(len(ep['zs'])))
        res['P2'][sea] = out
        print(f"[P2] {sea} fan rho vs observation count")
        for n in N_OBS:
            o = out[str(n)]
            print(f"   n={n:5d} (actual {o['n_obs_actual']:5d}): "
                  f"fan {o['fan']:+.3f} +- {o['se']:.3f}")

    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
