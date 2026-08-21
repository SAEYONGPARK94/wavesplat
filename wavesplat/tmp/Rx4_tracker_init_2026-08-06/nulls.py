"""Rx-4 flat-sea (z=0) null on the widened episode set.
Convention copied VERBATIM from tmp/B2_g1p1_ext_2026-08-06/nulls12.py (which itself copies
tmp/rx2_init_2026-08-05/NEUTRAL_A3_nulls.py), so the 12 P1 values must reproduce B2's file."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
from wavesplat.data_loader import EpisodeDataLoader

CROWDED16 = [f'episode_BF{bf}_{nc}comp_seed{es}_{bath}.h5'
             for bf in (4, 5) for nc in (15, 30)
             for es in (0, 1) for bath in ('uniform', 'slope')]
SPARSE8 = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
           'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
           'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
           'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5']
EPISODES = CROWDED16 + SPARSE8

rows = []
for ep in EPISODES:
    L = EpisodeDataLoader(os.path.join(DATA, ep), device='cpu')
    frames = L.get_init_window(tp_factor=2.0); last = frames[-1]
    gt = last.gt_field.cpu().numpy(); Hs = L.Hs_true
    tf = L.get_tracking_frames(tp_factor=2.0)
    zs = [float(np.sqrt(np.mean(f.gt_field.cpu().numpy() ** 2)) / max(Hs, 1e-6)) for f in tf]
    rows.append(dict(episode=ep, Hs_true=float(Hs), Tp_true=float(L.Tp_true),
                     zero_null_init_nrmse=float(np.sqrt(np.mean(gt ** 2)) / max(Hs, 1e-6)),
                     zero_null_tracking_nRMSE_mean=float(np.mean(zs))))
    print(f"{ep[8:-3]:34s} null_track {rows[-1]['zero_null_tracking_nRMSE_mean']:.4f}", flush=True)
json.dump(rows, open(os.path.join(HERE, 'nulls.json'), 'w'), indent=1)

print("\n-- reproduction control vs B2 nulls12.json (the 12 overlapping episodes) --")
old = {r['episode']: r for r in json.load(open(os.path.join(
    REPO, 'tmp/B2_g1p1_ext_2026-08-06/nulls12.json')))}
worst = 0.0
for r in rows:
    if r['episode'] not in old:
        continue
    d = abs(r['zero_null_tracking_nRMSE_mean']
            - old[r['episode']]['zero_null_tracking_nRMSE_mean'])
    worst = max(worst, d)
    print(f"  {r['episode'][8:-3]:34s} {r['zero_null_tracking_nRMSE_mean']:.6f} vs "
          f"{old[r['episode']]['zero_null_tracking_nRMSE_mean']:.6f}  |diff|={d:.2e}")
print(f"  worst |diff| = {worst:.3e}")
