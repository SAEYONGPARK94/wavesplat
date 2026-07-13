"""E-3b: re-score the benchmark-v2 BF2 row on an adequate raycast mesh.

Prereg: docs/E2_MESH_AUDIT_prereg_2026-07-13.md.

Table B4 reports BF2 1-comp as the benchmark's floor (fan 0.194, enc 0.168) and
its caption calls that a property of the sea.  It is not: the mesh renders that
8.2 m wave at 1.2 points per wavelength, so the LiDAR never saw a wave.  Same
H5 recipe as d3_h6_v2_e2e (the artifact that produced the published row), run
over the coarse and fine renderings of the same 5 seas.
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

SEEDS = [0, 1, 2, 3, 4]
NAME = 'episode_BF2_1comp_seed{}_fixedT.h5'
OUT = 'results/direction3/e3b_bf2_rescore.json'


def score(path, rng):
    with h5py.File(path, 'r') as f:
        comp = f['config/wave_components'][:]
        ts = f['frames/timestamps'][:]
        pts = f['frames/lidar_points']
        gx, gy = f['frames/gt_grid_x'][:], f['frames/gt_grid_y'][:]
        gz = f['frames/gt_wave_field']

        kp = float(np.hypot(comp[:, 2], comp[:, 3])[np.argmax(comp[:, 0])])
        obs_i = list(range(50, 249, 18))                 # 12 frames, ~20 s
        t_end = ts[obs_i[-1]]
        P, zv = [], []
        for i in obs_i:
            p = pts[i]
            P.append(np.column_stack([p[:, 0], p[:, 1], np.full(len(p), ts[i])]))
            zv.append(p[:, 2])
        P, zv = np.vstack(P), np.concatenate(zv)
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
        Ps, zs = P[sel], zv[sel]

        t32 = torch.tensor(Ps, device=DEV, dtype=torch.float32)
        z32 = torch.tensor(zs, device=DEV, dtype=torch.float32)
        _, mdl = mmse_gpu(t32, z32, 50.0, t32[:64], 0.05, kp, seed=0,
                          return_model=True, no_em=True, pexp=1.0, smult=10.0)
        gam, s2 = mdl['gam'].to(DT), float(mdl['s2'])
        kxb, kyb, om = build_bank(kp, BAND, kp / 10.0, 50.0, DEV, torch.float32)
        kxb, kyb, om = kxb.to(DT), kyb.to(DT), om.to(DT)
        Phi = design(torch.tensor(Ps, device=DEV, dtype=DT), kxb, kyb, om)
        A0 = Phi.T @ Phi / s2 + torch.diag(1.0 / gam)
        a = torch.linalg.solve(
            A0, Phi.T @ torch.tensor(zs, device=DEV, dtype=DT) / s2)

        X, Y = np.meshgrid(gx, gy)
        R = np.hypot(X, Y)
        iend = obs_i[-1]
        gt_end = gz[iend]

        def rho_region(rmin, rmax):
            m = (R >= rmin) & (R <= rmax)
            Q = np.column_stack([X[m], Y[m], np.full(m.sum(), t_end)])
            C = design(torch.tensor(Q, device=DEV, dtype=DT), kxb, kyb, om)
            return float(np.corrcoef((C @ a).cpu().numpy(), gt_end[m])[0, 1])

        ann = rho_region(15.0, 75.0)
        fan = rho_region(15.0, float(R.max()))

        iy, ix = np.unravel_index(np.argmin(R), R.shape)
        fut = [i for i in range(iend + 1, len(ts)) if ts[i] <= t_end + 5.0]
        Qe = np.column_stack([np.full(len(fut), X[iy, ix]),
                              np.full(len(fut), Y[iy, ix]), ts[fut]])
        Ce = design(torch.tensor(Qe, device=DEV, dtype=DT), kxb, kyb, om)
        fe = (Ce @ a).cpu().numpy()
        ge = np.array([gz[i][iy, ix] for i in fut])
        enc = float(np.corrcoef(fe, ge)[0, 1])

    return dict(ann=ann, fan=fan, enc=enc)


def main():
    res = {}
    for tag, root in (('coarse', 'data'), ('fine', 'tmp/finemesh')):
        recs = []
        for s in SEEDS:
            path = os.path.join(root, NAME.format(s))
            if not os.path.exists(path):
                raise SystemExit(f'missing {path}')
            r = score(path, np.random.default_rng(11 + s))
            r['seed'] = s
            recs.append(r)
            print(f'  [{tag}] seed{s}  ann {r["ann"]:+.3f}  fan {r["fan"]:+.3f}'
                  f'  enc {r["enc"]:+.3f}', flush=True)
        res[tag] = dict(
            episodes=recs,
            ann=float(np.mean([r['ann'] for r in recs])),
            fan=float(np.mean([r['fan'] for r in recs])),
            enc=float(np.mean([r['enc'] for r in recs])),
            fan_ge70=int(sum(r['fan'] >= 0.7 for r in recs)),
            enc_ge70=int(sum(r['enc'] >= 0.7 for r in recs)))

    res['published_B4_row'] = dict(fan=0.194, ann=0.223, enc=0.168,
                                   note='DRAFT_chAB_tables_2026-07-12.md L271')
    res['prereg'] = 'docs/E2_MESH_AUDIT_prereg_2026-07-13.md'
    json.dump(res, open(OUT, 'w'), indent=1)

    c, f = res['coarse'], res['fine']
    print(f'\n{"=" * 58}\n[E-3b] benchmark v2, BF2 1comp row (5 seeds)')
    print(f'  coarse mesh (published): ann {c["ann"]:.3f}  fan {c["fan"]:.3f}'
          f'  enc {c["enc"]:.3f}   fan>=0.7 {c["fan_ge70"]}/5')
    print(f'  fine mesh (corrected)  : ann {f["ann"]:.3f}  fan {f["fan"]:.3f}'
          f'  enc {f["enc"]:.3f}   fan>=0.7 {f["fan_ge70"]}/5')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
