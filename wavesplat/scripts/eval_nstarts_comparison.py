"""
eval_nstarts_comparison.py
n_starts=5 vs n_starts=10 comparison on 6 pilot episodes (seed0 × sea states).
Also tests whether catastrophic failure (BF4_15c_seed2) can be escaped with more starts.
"""
import sys, os, json, numpy as np, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from scripts.gp_disp_pilot import (
    eval_episode, build_K_disp_oracle, gp_predict,
    compute_metrics_simple, N_FRAMES, SIGMA_OBS
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_PATH  = os.path.join(os.path.dirname(__file__), '..', 'results',
                         'nstarts_comparison.json')

# 6 pilot episodes (seed0) + catastrophic failure case
PILOT_EPS = [
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]
EXTRA_EPS = [
    'episode_BF4_15comp_seed2_uniform.h5',  # catastrophic failure case
]


def run_comparison():
    # Load existing ws_multistart_5 results from full eval
    full_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                             'gp_disp_full_eval.json')
    with open(full_path) as f:
        full = json.load(f)
    ws5_per_ep = full['ws_multistart']['per_episode']
    oracle_per_ep = full['oracle']['per_episode']

    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
        print(f"Loaded existing {OUT_PATH}")
    else:
        results = {'ws_multistart_10': {}, 'ws_multistart_5_ref': {}}

    # Reference: ws_multistart_5 for pilot episodes
    for ep_name in PILOT_EPS + EXTRA_EPS:
        ep_key = ep_name.replace('episode_', '').replace('.h5', '')
        if ep_key in ws5_per_ep:
            results['ws_multistart_5_ref'][ep_key] = ws5_per_ep[ep_key]

    # Run ws_multistart_10 on pilot episodes (skip if already done)
    eps_todo = [ep for ep in PILOT_EPS + EXTRA_EPS
                if ep.replace('episode_', '').replace('.h5', '')
                   not in results.get('ws_multistart_10', {})]
    print(f"ws_multistart_10 episodes to run: {len(eps_todo)}")

    for ep_name in eps_todo:
        h5_path = os.path.join(DATA_DIR, ep_name)
        if not os.path.exists(h5_path):
            print(f"  SKIP (not found): {ep_name}")
            continue
        ep_key = ep_name.replace('episode_', '').replace('.h5', '')
        print(f"  {ep_key}...", end='', flush=True)
        t0 = time.time()
        try:
            m = eval_episode(h5_path, mode='ws_multistart_10', n_frames=N_FRAMES)
            results['ws_multistart_10'][ep_key] = m
            dt = time.time() - t0
            print(f" nRMSE={m['nRMSE']:.4f} VR={m['VR']:.3f} Pearson={m['Pearson']:.3f} ({dt:.0f}s)")
        except Exception as e:
            print(f" ERROR: {e}")
            import traceback; traceback.print_exc()

        with open(OUT_PATH, 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("n_starts Comparison (6 pilot episodes, seed0)")
    print("=" * 70)
    print(f"{'Episode':<30} {'oracle':>8} {'ms5':>8} {'ms10':>8} {'ms5→ms10':>10}")
    print("-" * 60)
    for ep_name in PILOT_EPS + EXTRA_EPS:
        ep_key = ep_name.replace('episode_', '').replace('.h5', '')
        or_v  = oracle_per_ep.get(ep_key, {}).get('nRMSE', float('nan'))
        ms5_v = results['ws_multistart_5_ref'].get(ep_key, {}).get('nRMSE', float('nan'))
        ms10_v = results['ws_multistart_10'].get(ep_key, {}).get('nRMSE', float('nan'))
        impr  = (ms5_v - ms10_v) / ms5_v * 100 if ms5_v > 0 else float('nan')
        print(f"{ep_key:<30} {or_v:>8.4f} {ms5_v:>8.4f} {ms10_v:>8.4f} {impr:>9.1f}%")

    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == '__main__':
    run_comparison()
