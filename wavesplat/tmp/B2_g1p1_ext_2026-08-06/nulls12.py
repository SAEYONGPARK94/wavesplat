"""B2 — flat-sea (z=0) null on all 12 P1 episodes. Convention copied verbatim from
tmp/rx2_init_2026-08-05/NEUTRAL_A3_nulls.py so the 4 crowded values must reproduce."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(REPO, '..')))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
from wavesplat.data_loader import EpisodeDataLoader

CROWDED4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
EPISODES = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
            'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
            'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
            'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5'] + CROWDED4

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
json.dump(rows, open(os.path.join(HERE, 'nulls12.json'), 'w'), indent=1)

# reproduction control against rx2
old = {r['episode']: r for r in json.load(open(os.path.join(
    REPO, 'tmp/rx2_init_2026-08-05/NEUTRAL_A3_nulls.json')))}
print("\n-- reproduction control vs rx2 (crowded 4) --")
for e in CROWDED4:
    n = [r for r in rows if r['episode'] == e][0]['zero_null_tracking_nRMSE_mean']
    d = abs(n - old[e]['zero_null_tracking_nRMSE_mean'])
    print(f"  {e[8:-3]:34s} {n:.6f} vs {old[e]['zero_null_tracking_nRMSE_mean']:.6f}  |diff|={d:.2e}")
