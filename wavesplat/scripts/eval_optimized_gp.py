"""
Optimized GP K_disp evaluation:
  mode:        ws_ms5_reg_amp  (n_starts=5 + regularized per-component amplitude)
  lambda_reg:  0.5             (best from lambda tuning)
  init_tp_factor: 4.0          (best Pearson from hyperparam tuning)
  sigma_obs:   0.02            (default)

30 episodes (seed 0-4 × 6 sea states).
Result saved to results/gp_optimized_eval.json
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.gp_disp_pilot import eval_episode, TEST_EPISODES_FULL, N_FRAMES

# Optimized hyperparams
LAMBDA_REG = 0.5
INIT_TP_FACTOR = 4.0
SIGMA_OBS = 0.02
MODE = 'ws_ms5_reg_amp'

OUT_PATH = 'results/gp_optimized_eval.json'


def main():
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
        print(f"Loaded existing: {OUT_PATH} ({len(results.get('per_episode', {}))} episodes done)")
    else:
        results = {'per_episode': {}, 'config': {
            'mode': MODE, 'lambda_reg': LAMBDA_REG,
            'init_tp_factor': INIT_TP_FACTOR, 'sigma_obs': SIGMA_OBS,
        }}

    per_ep = results.get('per_episode', {})

    print("=" * 70)
    print(f"Optimized GP (mode={MODE}, λ={LAMBDA_REG}, tp={INIT_TP_FACTOR}, σ={SIGMA_OBS})")
    print(f"30 episodes, n_frames={N_FRAMES}")
    print("=" * 70)

    for h5_name in TEST_EPISODES_FULL:
        h5_path = os.path.join('data', h5_name)
        ep_key = h5_name.replace('episode_', '').replace('.h5', '')

        if ep_key in per_ep:
            m = per_ep[ep_key]
            print(f"  {ep_key:<35}: already done "
                  f"nRMSE={m['nRMSE']:.4f} Pearson={m['Pearson']:.3f}")
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

        # Recompute average after each episode
        if per_ep:
            avg = {k: float(np.nanmean([v[k] for v in per_ep.values()]))
                   for k in next(iter(per_ep.values()))}
            results['per_episode'] = per_ep
            results['average'] = avg
            with open(OUT_PATH, 'w') as f:
                json.dump(results, f, indent=2)

    # Final summary
    if 'average' in results:
        a = results['average']
        print("\n" + "=" * 70)
        print(f"FINAL ({len(per_ep)} episodes): "
              f"nRMSE={a['nRMSE']:.4f} VR={a['VR']:.4f} "
              f"Pearson={a['Pearson']:.4f} Skill={a['Skill']:.4f}")
        print("=" * 70)

    # Comparison with baseline
    print("\nComparison (30 ep, same protocol):")
    print(f"  WaveSplat:       nRMSE=0.2633")
    print(f"  ws_ms5_reg_amp:  nRMSE=0.2232  (default λ=2, tp=2)")
    if 'average' in results:
        a = results['average']
        print(f"  ws_ms5_optimized:nRMSE={a['nRMSE']:.4f}  (λ={LAMBDA_REG}, tp={INIT_TP_FACTOR})")
    print(f"  WavePINO S4:     nRMSE=0.2201  (reference)")
    print(f"  Oracle K_disp:   nRMSE=0.1259  (ceiling)")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == '__main__':
    main()
