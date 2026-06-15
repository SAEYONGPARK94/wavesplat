"""
Hyperparameter tuning for GP K_disp (ws_reg_amp, n_starts=1):
  - lambda_reg  : {0.5, 1.0, 2.0, 5.0}    (default=2.0)
  - sigma_obs   : {0.01, 0.02, 0.05}       (default=0.02)
  - init_tp_factor: {2.0, 3.0, 4.0}        (default=2.0)

All experiments on 6 pilot episodes (seed0 × 6 sea states).
Result saved to results/hyperparam_tuning.json
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.gp_disp_pilot import eval_episode, N_FRAMES

PILOT_EPS = [
    'data/episode_BF3_5comp_seed0_uniform.h5',
    'data/episode_BF4_5comp_seed0_uniform.h5',
    'data/episode_BF4_15comp_seed0_uniform.h5',
    'data/episode_BF4_30comp_seed0_uniform.h5',
    'data/episode_BF5_15comp_seed0_uniform.h5',
    'data/episode_BF5_30comp_seed0_uniform.h5',
]

OUT_PATH = 'results/hyperparam_tuning.json'

# Grid: vary one parameter at a time from default
EXPERIMENTS = [
    # (name,  lambda_reg, sigma_obs, init_tp_factor)
    ('default',         2.0, 0.02, 2.0),   # baseline (already in lambda_tuning.json)
    ('lam_0.5',         0.5, 0.02, 2.0),
    ('lam_1.0',         1.0, 0.02, 2.0),
    ('lam_5.0',         5.0, 0.02, 2.0),
    ('sigma_0.01',      2.0, 0.01, 2.0),
    ('sigma_0.05',      2.0, 0.05, 2.0),
    ('tp_3.0',          2.0, 0.02, 3.0),
    ('tp_4.0',          2.0, 0.02, 4.0),
]


def run_experiment(name, lambda_reg, sigma_obs, init_tp_factor, results):
    if name in results and 'average' in results[name]:
        a = results[name]['average']
        print(f"  {name:<16}: already done "
              f"(nRMSE={a['nRMSE']:.4f} VR={a['VR']:.4f} Skill={a['Skill']:.4f})")
        return

    print(f"\n--- {name} (lam={lambda_reg}, sig={sigma_obs}, tp={init_tp_factor}) ---")
    per_ep = {}
    for h5_path in PILOT_EPS:
        ep_key = os.path.basename(h5_path).replace('episode_', '').replace('.h5', '')
        print(f"  {ep_key}...", end='', flush=True)
        try:
            m = eval_episode(h5_path, mode='ws_reg_amp', n_frames=N_FRAMES,
                             lambda_reg=lambda_reg, sigma_obs=sigma_obs,
                             init_tp_factor=init_tp_factor)
            per_ep[ep_key] = m
            print(f" nRMSE={m['nRMSE']:.4f} VR={m['VR']:.3f} "
                  f"Pearson={m['Pearson']:.3f} Skill={m['Skill']:.3f}")
        except Exception as e:
            print(f" ERROR: {e}")
            import traceback; traceback.print_exc()

    if per_ep:
        avg = {k: float(np.nanmean([v[k] for v in per_ep.values()]))
               for k in next(iter(per_ep.values()))}
        results[name] = {
            'per_episode': per_ep,
            'average': avg,
            'lambda_reg': lambda_reg,
            'sigma_obs': sigma_obs,
            'init_tp_factor': init_tp_factor,
        }
        with open(OUT_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  → nRMSE={avg['nRMSE']:.4f} VR={avg['VR']:.4f} "
              f"Pearson={avg['Pearson']:.4f} Skill={avg['Skill']:.4f}")


def main():
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
        print(f"Loaded existing: {OUT_PATH}")
    else:
        results = {}

    print("=" * 70)
    print("Hyperparameter Tuning for GP K_disp (ws_reg_amp, 6 pilot eps)")
    print("=" * 70)

    for name, lam, sig, tp in EXPERIMENTS:
        run_experiment(name, lam, sig, tp, results)

    # Summary
    print("\n" + "=" * 70)
    print("Summary (6 pilot episodes, ws_reg_amp n_starts=1)")
    print("=" * 70)
    print(f"{'Name':<16} {'lam':>5} {'sig':>6} {'tp':>5} {'nRMSE':>8} {'VR':>7} {'Pearson':>9} {'Skill':>8}")
    print("-" * 70)
    for name, lam, sig, tp in EXPERIMENTS:
        if name in results and 'average' in results[name]:
            a = results[name]['average']
            marker = ' ←best' if name == min(
                [(n, results[n]['average']['nRMSE']) for n, *_ in EXPERIMENTS
                 if n in results and 'average' in results[n]],
                key=lambda x: x[1])[0] else ''
            print(f"{name:<16} {lam:>5.1f} {sig:>6.3f} {tp:>5.1f} "
                  f"{a['nRMSE']:>8.4f} {a['VR']:>7.4f} "
                  f"{a['Pearson']:>9.4f} {a['Skill']:>8.4f}{marker}")

    print(f"\nSaved: {OUT_PATH}")


if __name__ == '__main__':
    main()
