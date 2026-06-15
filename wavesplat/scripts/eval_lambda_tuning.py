"""
Lambda tuning for ws_reg_amp: test lambda_reg in {0.5, 1.0, 2.0, 5.0}
on 6 pilot episodes (seed0 × 6 sea states).

Result saved to results/lambda_tuning.json
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

LAMBDAS = [0.5, 1.0, 2.0, 5.0]
OUT_PATH = 'results/lambda_tuning.json'


def main():
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
        print(f"Loaded existing: {OUT_PATH}")
    else:
        results = {}

    print("=" * 70)
    print("Lambda Tuning for ws_reg_amp (n_starts=1)")
    print("=" * 70)

    for lam in LAMBDAS:
        key = f'lambda_{str(lam).replace(".", "_")}'
        if key in results and 'average' in results[key]:
            print(f"lambda={lam}: already done (nRMSE={results[key]['average']['nRMSE']:.4f})")
            continue

        print(f"\n--- lambda={lam} ---")
        per_ep = {}
        for h5_path in PILOT_EPS:
            ep_key = os.path.basename(h5_path).replace('episode_', '').replace('.h5', '')
            print(f"  {ep_key}...", end='', flush=True)
            try:
                m = eval_episode(h5_path, mode='ws_reg_amp', n_frames=N_FRAMES,
                                 lambda_reg=lam)
                per_ep[ep_key] = m
                print(f" nRMSE={m['nRMSE']:.4f} VR={m['VR']:.3f} "
                      f"Pearson={m['Pearson']:.3f} Skill={m['Skill']:.3f} "
                      f"({m['elapsed_ms']:.0f}ms/frame)")
            except Exception as e:
                print(f" ERROR: {e}")
                import traceback; traceback.print_exc()

        if per_ep:
            avg = {k: float(np.nanmean([v[k] for v in per_ep.values()]))
                   for k in next(iter(per_ep.values()))}
            if key not in results:
                results[key] = {}
            results[key]['per_episode'] = per_ep
            results[key]['average'] = avg
            results[key]['lambda'] = lam
            with open(OUT_PATH, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  AVERAGE: nRMSE={avg['nRMSE']:.4f} VR={avg['VR']:.4f} "
                  f"Pearson={avg['Pearson']:.4f} Skill={avg['Skill']:.4f}")

    # Summary table
    print("\n" + "=" * 70)
    print("Lambda Tuning Summary (6 pilot episodes, ws_reg_amp n_starts=1)")
    print("=" * 70)
    print(f"{'lambda':>8} {'nRMSE':>8} {'VR':>8} {'Pearson':>9} {'Skill':>8}")
    print("-" * 45)
    for lam in LAMBDAS:
        key = f'lambda_{str(lam).replace(".", "_")}'
        if key in results and 'average' in results[key]:
            a = results[key]['average']
            print(f"{lam:>8.1f} {a['nRMSE']:>8.4f} {a['VR']:>8.4f} "
                  f"{a['Pearson']:>9.4f} {a['Skill']:>8.4f}")

    print(f"\nSaved: {OUT_PATH}")


if __name__ == '__main__':
    main()
