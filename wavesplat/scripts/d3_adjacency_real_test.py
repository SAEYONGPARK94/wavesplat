"""
d3_adjacency_real_test.py — Q2 follow-up under the user's constraint
(2026-07-06): don't assume ideal scan-line clusters; use the ACTUAL
return pattern of the Falcon-spec simulation (raydrop physics keeps
returns clustered near sensor-facing crests: NN median ~2 m, ~19% of
returns have a <1 m neighbor, ~50% <2 m — measured across BF3/4/5).

The question becomes: with the SAME native returns and the SAME 600-obs
budget, does an adjacency-PRESERVING subsample policy (keep same-frame
mutual pairs < 2 m intact first, fill the rest randomly) beat the
current random policy on SLOPE skill? Random subsampling (600/5760 ~
10%) destroys pairs (both members survive ~1% of the time) — that loss
is free to recover in software.

Scoring: predict eta on the GT grid (stride 2) at the window end, finite
-difference both prediction and GT along x -> slope-map corr (+ eta-map
corr for reference).

Usage: python3 scripts/d3_adjacency_real_test.py
"""
import os
import sys
import json

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesplat.data_loader import EpisodeDataLoader
from d3_sa_nomp import mmse_bank_estimate

G = 9.81
OUT = 'results/direction3/adjacency_real_test.json'
EPISODES = ('episode_BF3_5comp_seed0_uniform.h5',
            'episode_BF4_15comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5')
W_MEM = 12.0
M_CAP = 600
PAIR_R = 2.0


def subsample(P, z, frames_id, policy, rng):
    n = len(z)
    if n <= M_CAP:
        return P, z
    if policy == 'random':
        ii = rng.choice(n, M_CAP, replace=False)
        return P[ii], z[ii]
    # pairs: same-frame mutual neighbors < PAIR_R, kept intact first
    keep = []
    for fid in np.unique(frames_id):
        m = np.flatnonzero(frames_id == fid)
        if len(m) < 2:
            continue
        t = cKDTree(P[m, :2])
        d, j = t.query(P[m, :2], k=2)
        for a_, (dd, jj) in enumerate(zip(d[:, 1], j[:, 1])):
            if dd < PAIR_R and j[jj, 1] == a_:      # mutual pair
                if a_ < jj:
                    keep.append((m[a_], m[jj]))
    rng.shuffle(keep)
    budget_pairs = M_CAP if policy == 'pairs' else int(M_CAP * 0.25)
    sel = []
    for a_, b_ in keep:
        if len(sel) + 2 > budget_pairs:
            break
        sel += [a_, b_]
    rest = np.setdiff1d(np.arange(n), np.array(sel, int))
    fill = rng.choice(rest, M_CAP - len(sel), replace=False)
    ii = np.concatenate([np.array(sel, int), fill])
    return P[ii], z[ii]


def main():
    rows = []
    for ep in EPISODES:
        L = EpisodeDataLoader('data/' + ep)
        Hs = L.Hs_true
        kp = (2 * np.pi / L.Tp_true) ** 2 / G
        frames = L.get_tracking_frames(0.2)
        dtf = float(frames[1].t - frames[0].t)
        nfw = int(round(W_MEM / dtf))
        gx = L.gt_grid_x.numpy()[::2]
        gy = L.gt_grid_y.numpy()[::2]
        dxg = float(gx[1] - gx[0])
        Xg, Yg = np.meshgrid(gx, gy)
        for i0 in (nfw + 20, nfw + 80, nfw + 140):
            sel_fr = frames[i0 - nfw + 1:i0 + 1]
            P_all, z_all, fid = [], [], []
            for j, f in enumerate(sel_fr):
                P_all.append(np.column_stack([
                    f.x_obs.numpy(), f.y_obs.numpy(),
                    np.full(len(f.x_obs), float(f.t))]))
                z_all.append(f.z_obs.numpy())
                fid.append(np.full(len(f.x_obs), j))
            P_all = np.vstack(P_all)
            z_all = np.concatenate(z_all)
            fid = np.concatenate(fid)
            tgt = frames[i0]
            gt = tgt.gt_field.numpy()[::2, ::2]
            gs_x = (gt[:, 2:] - gt[:, :-2]) / (2 * dxg)
            Pq = np.column_stack([Xg.ravel(), Yg.ravel(),
                                  np.full(Xg.size, float(tgt.t))])
            row = dict(ep=ep.split('_seed')[0], i0=int(i0))
            for pol in ('random', 'pairs', 'mix25'):
                rng = np.random.default_rng(0)
                P, z = subsample(P_all, z_all, fid, pol, rng)
                fq = mmse_bank_estimate(P, z, L.depth, Pq,
                                        sigma0=0.05 * Hs, kp_hint=kp,
                                        seed=i0).reshape(Xg.shape)
                fs_x = (fq[:, 2:] - fq[:, :-2]) / (2 * dxg)
                row[f'{pol}_eta'] = float(np.corrcoef(
                    fq.ravel(), gt.ravel())[0, 1])
                row[f'{pol}_slope'] = float(np.corrcoef(
                    fs_x.ravel(), gs_x.ravel())[0, 1])
            rows.append(row)
            print(f"[{row['ep']} i0={i0}] slope rnd "
                  f"{row['random_slope']:+.2f} pairs "
                  f"{row['pairs_slope']:+.2f} mix25 "
                  f"{row['mix25_slope']:+.2f}", flush=True)
    for k in ('eta', 'slope'):
        vals = {pol: np.mean([r[f'{pol}_{k}'] for r in rows])
                for pol in ('random', 'pairs', 'mix25')}
        print(f"{k:6s}: " + "  ".join(f"{pol} {v:+.3f}"
                                      for pol, v in vals.items()))
    json.dump(rows, open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
