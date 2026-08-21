"""Rx-4 null GAP FIX (2026-08-07, after machine restart).

DEFECT: nulls.py built CROWDED16 as {BF4,BF5}x{15,30}comp x seed{0,1} x {uniform,slope},
but PREREG sec.3 block C -- and the episode list actually RUN by run_rx4.py -- is
{BF4,BF5}x{15,30}comp x seed{0,1,2,3} x uniform ONLY.  Overlap is 8; the seed2/seed3
uniform episodes have NO null.  analyze_rx4.py would KeyError on them; rederive_rx4.py
would silently intersect them away and score crowded n=8 instead of 16.

This script does NOT touch nulls.json.  It:
  (1) recomputes the null for EVERY episode already in nulls.json with the same code
      path and requires BIT-EXACT reproduction (integrity: proves the convention and
      the loader still behave identically after the restart), and
  (2) computes the 8 missing episodes,
writing the union to nulls_full.json.

The null is a function of the ground-truth field alone (flat-sea z=0 predictor).  It
reads no result file, so extending the episode list after seeing results cannot bias it.

Convention below is copied VERBATIM from nulls.py (which copies B2 nulls12.py, which
copies rx2_init NEUTRAL_A3_nulls.py).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
DATA = os.path.join(REPO, 'data_indep2_2026-08-07')
from wavesplat.data_loader import EpisodeDataLoader  # noqa: E402

# the episode set the run actually used (PREREG sec.3)
CROWDED16 = [f'episode_BF{bf}_{nc}comp_seed{es}_uniform_ra_rf_ds.h5'
             for bf in (4, 5) for nc in (15, 30) for es in (0, 1, 2, 3)]
SPARSE8 = []

old = {}
targets = sorted(set(CROWDED16) | set(SPARSE8) | set(old))


def null_for(ep):
    L = EpisodeDataLoader(os.path.join(DATA, ep), device='cpu')
    frames = L.get_init_window(tp_factor=2.0)
    last = frames[-1]
    gt = last.gt_field.cpu().numpy()
    Hs = L.Hs_true
    tf = L.get_tracking_frames(tp_factor=2.0)
    zs = [float(np.sqrt(np.mean(f.gt_field.cpu().numpy() ** 2)) / max(Hs, 1e-6)) for f in tf]
    return dict(episode=ep, Hs_true=float(Hs), Tp_true=float(L.Tp_true),
                zero_null_init_nrmse=float(np.sqrt(np.mean(gt ** 2)) / max(Hs, 1e-6)),
                zero_null_tracking_nRMSE_mean=float(np.mean(zs)))


rows, worst, n_repro, n_new = [], 0.0, 0, 0
for ep in targets:
    r = null_for(ep)
    rows.append(r)
    if ep in old:
        n_repro += 1
        for k in ('zero_null_init_nrmse', 'zero_null_tracking_nRMSE_mean'):
            worst = max(worst, abs(r[k] - old[ep][k]))
        tag = 'repro'
    else:
        n_new += 1
        tag = '** NEW **'
    print(f"{ep[8:-3]:34s} null_track {r['zero_null_tracking_nRMSE_mean']:.6f}  {tag}",
          flush=True)

json.dump(rows, open(os.path.join(HERE, 'nulls_k6.json'), 'w'), indent=1)
print(f'\nreproduced {n_repro} existing episodes, worst |diff| = {worst:.3e}')
print(f'added {n_new} previously-missing episodes')
print(f'BIT-IDENTITY on pre-existing nulls: {"PASS" if worst == 0.0 else "FAIL"}')
missing = [e for e in CROWDED16 + SPARSE8 if e not in {r['episode'] for r in rows}]
print(f'still missing from block C+S: {missing}  -> {"PASS" if not missing else "FAIL"}')
