"""
ddkl_baseline_6ep.py — Same-protocol baselines on the DDKL 6 test episodes.

Computes GP K_disp ms5 (ws_ms5_reg_amp) and oracle on the EXACT same 6
seed0 test episodes DDKL is evaluated on, with the identical 30-frame
tracking protocol. This gives a fair apples-to-apples head-to-head, instead
of the 30-episode cross-protocol reference numbers.

Output: results/ddkl_baseline_6ep.json
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from scripts.gp_disp_pilot import eval_episode, TEST_EPISODES

DATA_DIR = 'data'


def main():
    print("Same-protocol baselines on DDKL 6 test episodes")
    print("=" * 60)
    modes = ['oracle', 'beamform', 'ws_ms5_reg_amp']
    results = {m: {} for m in modes}

    for mode in modes:
        print(f"\n--- mode: {mode} ---")
        t0 = time.time()
        for h5_name in TEST_EPISODES:
            h5_path = os.path.join(DATA_DIR, h5_name)
            if not os.path.exists(h5_path):
                print(f"  SKIP {h5_name}")
                continue
            ep_key = h5_name.replace('episode_', '').replace('.h5', '')
            avg = eval_episode(h5_path, mode=mode, n_frames=30,
                               lambda_reg=0.5)
            if avg is None:
                continue
            results[mode][ep_key] = avg
            print(f"  [{ep_key}] nRMSE={avg['nRMSE']:.4f}  "
                  f"Pearson={avg['Pearson']:.3f}  VR={avg['VR']:.3f}")
        dt = time.time() - t0
        vals = [results[mode][k]['nRMSE'] for k in results[mode]]
        pvals = [results[mode][k]['Pearson'] for k in results[mode]]
        print(f"  >>> {mode} mean nRMSE={np.mean(vals):.4f}  "
              f"Pearson={np.mean(pvals):.3f}  ({dt:.0f}s)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (same 6 episodes, 30-frame protocol)")
    print("=" * 60)
    summary = {}
    for mode in modes:
        vals = [results[mode][k]['nRMSE'] for k in results[mode]]
        pvals = [results[mode][k]['Pearson'] for k in results[mode]]
        vrs = [results[mode][k]['VR'] for k in results[mode]]
        summary[mode] = {
            'mean_nRMSE': float(np.mean(vals)),
            'mean_Pearson': float(np.mean(pvals)),
            'mean_VR': float(np.mean(vrs)),
        }
        print(f"  {mode:<18} nRMSE={np.mean(vals):.4f}  "
              f"Pearson={np.mean(pvals):.3f}  VR={np.mean(vrs):.3f}")

    out = {
        'description': 'Same-protocol baselines (oracle, beamform, ws_ms5_reg_amp) '
                       'on the 6 DDKL seed0 test episodes, lambda_reg=0.5, 30 frames',
        'test_episodes': TEST_EPISODES,
        'per_episode': results,
        'summary': summary,
    }
    with open('results/ddkl_baseline_6ep.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/ddkl_baseline_6ep.json")


if __name__ == '__main__':
    main()
