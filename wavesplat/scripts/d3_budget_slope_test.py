"""
d3_budget_slope_test.py — final Q2 datum: within the real raydrop
pattern, sampling POLICY gains nothing at a fixed 600 budget (pairs
destructive, mix neutral). Does the BUDGET itself (600 -> 1200, i.e.
"pairs for free") open the slope? BF4 episode, 3 windows, random
subsample at both budgets, slope-map corr.

Usage: python3 scripts/d3_budget_slope_test.py
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesplat.data_loader import EpisodeDataLoader
from d3_sa_nomp import mmse_bank_estimate

G = 9.81
OUT = 'results/direction3/budget_slope_test.json'


def main():
    ep = 'episode_BF4_15comp_seed0_uniform.h5'
    L = EpisodeDataLoader('data/' + ep)
    Hs = L.Hs_true
    kp = (2 * np.pi / L.Tp_true) ** 2 / G
    frames = L.get_tracking_frames(0.2)
    dtf = float(frames[1].t - frames[0].t)
    nfw = int(round(12.0 / dtf))
    gx = L.gt_grid_x.numpy()[::2]
    gy = L.gt_grid_y.numpy()[::2]
    dxg = float(gx[1] - gx[0])
    Xg, Yg = np.meshgrid(gx, gy)
    rows = []
    for i0 in (nfw + 20, nfw + 80, nfw + 140):
        sel_fr = frames[i0 - nfw + 1:i0 + 1]
        P_all, z_all = [], []
        for f in sel_fr:
            P_all.append(np.column_stack([
                f.x_obs.numpy(), f.y_obs.numpy(),
                np.full(len(f.x_obs), float(f.t))]))
            z_all.append(f.z_obs.numpy())
        P_all = np.vstack(P_all)
        z_all = np.concatenate(z_all)
        tgt = frames[i0]
        gt = tgt.gt_field.numpy()[::2, ::2]
        gs_x = (gt[:, 2:] - gt[:, :-2]) / (2 * dxg)
        Pq = np.column_stack([Xg.ravel(), Yg.ravel(),
                              np.full(Xg.size, float(tgt.t))])
        row = dict(i0=int(i0))
        for cap in (600, 1200, 2400):
            rng = np.random.default_rng(0)
            ii = rng.choice(len(z_all), min(cap, len(z_all)),
                            replace=False)
            fq = mmse_bank_estimate(P_all[ii], z_all[ii], L.depth, Pq,
                                    sigma0=0.05 * Hs, kp_hint=kp,
                                    seed=i0).reshape(Xg.shape)
            fs_x = (fq[:, 2:] - fq[:, :-2]) / (2 * dxg)
            row[f'eta_{cap}'] = float(np.corrcoef(fq.ravel(),
                                                  gt.ravel())[0, 1])
            row[f'slope_{cap}'] = float(np.corrcoef(fs_x.ravel(),
                                                    gs_x.ravel())[0, 1])
        rows.append(row)
        print(f"[i0={i0}] slope 600 {row['slope_600']:+.2f} | 1200 "
              f"{row['slope_1200']:+.2f} | 2400 {row['slope_2400']:+.2f}",
              flush=True)
    for k in ('eta', 'slope'):
        line = f"{k:6s}:"
        for cap in (600, 1200, 2400):
            line += f"  {cap}: {np.mean([r[f'{k}_{cap}'] for r in rows]):+.3f}"
        print(line)
    json.dump(rows, open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
