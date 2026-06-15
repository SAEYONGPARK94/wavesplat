"""
ws_ms10_reg_amp evaluation:
  n_starts=10 + regularized per-component amplitudes (default λ=2, tp=2)
  Goal: Fix catastrophic failures (BF4_30c seed3: 0.556 in ms5 baseline)

30 episodes (seed 0-4 × 6 sea states).
Result saved to results/gp_ms10_reg_amp_eval.json

Estimated time: ~600 min (10 seeds × 60s × 30 ep)
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.gp_disp_pilot import eval_episode, TEST_EPISODES_FULL, N_FRAMES

MODE = 'ws_ms10_reg_amp'
LAMBDA_REG = 2.0
INIT_TP_FACTOR = 2.0
SIGMA_OBS = 0.02
OUT_PATH = 'results/gp_ms10_reg_amp_eval.json'


def main():
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
        done = len(results.get('per_episode', {}))
        print(f"Loaded existing: {OUT_PATH} ({done} episodes done)")
    else:
        results = {'per_episode': {}, 'config': {
            'mode': MODE, 'lambda_reg': LAMBDA_REG,
            'init_tp_factor': INIT_TP_FACTOR, 'sigma_obs': SIGMA_OBS,
        }}

    per_ep = results.get('per_episode', {})

    print("=" * 70)
    print(f"ws_ms10_reg_amp (n_starts=10, λ={LAMBDA_REG}, tp={INIT_TP_FACTOR})")
    print(f"30 episodes — catastrophic failure recovery focus")
    print("=" * 70)

    for h5_name in TEST_EPISODES_FULL:
        h5_path = os.path.join('data', h5_name)
        ep_key = h5_name.replace('episode_', '').replace('.h5', '')

        if ep_key in per_ep:
            m = per_ep[ep_key]
            print(f"  {ep_key:<35}: skip  nRMSE={m['nRMSE']:.4f}")
            continue

        print(f"  {ep_key:<35}...", end='', flush=True)
        try:
            m = eval_episode(h5_path, mode=MODE, n_frames=N_FRAMES,
                             lambda_reg=LAMBDA_REG, sigma_obs=SIGMA_OBS,
                             init_tp_factor=INIT_TP_FACTOR)
            per_ep[ep_key] = m
            print(f" nRMSE={m['nRMSE']:.4f} VR={m['VR']:.3f} "
                  f"Pearson={m['Pearson']:.3f} Skill={m['Skill']:.3f}")
        except Exception as e:
            print(f" ERROR: {e}")
            import traceback; traceback.print_exc()

        if per_ep:
            avg = {k: float(np.nanmean([v[k] for v in per_ep.values()]))
                   for k in next(iter(per_ep.values()))}
            results['per_episode'] = per_ep
            results['average'] = avg
            with open(OUT_PATH, 'w') as f:
                json.dump(results, f, indent=2)

    if 'average' in results:
        a = results['average']
        print("\n" + "=" * 70)
        print(f"FINAL ({len(per_ep)} episodes): "
              f"nRMSE={a['nRMSE']:.4f} VR={a['VR']:.4f} "
              f"Pearson={a['Pearson']:.4f} Skill={a['Skill']:.4f}")
        print("=" * 70)
        print("\nComparison (30 ep):")
        print(f"  WaveSplat:        nRMSE=0.2633")
        print(f"  ws_ms5_reg_amp:   nRMSE=0.2232")
        print(f"  ws_ms10_reg_amp:  nRMSE={a['nRMSE']:.4f}  ← this run")
        print(f"  WavePINO S4:      nRMSE=0.2201")
        print(f"  Oracle K_disp:    nRMSE=0.1259")

    print(f"\nSaved: {OUT_PATH}")


if __name__ == '__main__':
    main()
