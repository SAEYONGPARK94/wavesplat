"""
run_extended_analysis.py
Extended experiments for paper:
1. Sparsity sweep: subsample observations to test degradation
2. BF-level trend analysis
3. Convergence speed analysis (per-frame nRMSE curves)
"""

import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader, FrameData
from wavesplat.run import run_estimation, set_seed
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.evaluation.metrics import compute_all_metrics, rmse, nrmse


# Representative episodes for sparsity sweep (one per BF)
SPARSITY_EPISODES = [
    'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
]
SPARSITY_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

# All seed0 uniform episodes for BF trend
BF_EPISODES = [
    ('BF2', '1comp'), ('BF3', '1comp'), ('BF3', '2comp'), ('BF3', '5comp'),
    ('BF4', '1comp'), ('BF4', '5comp'), ('BF4', '15comp'), ('BF4', '30comp'),
    ('BF5', '15comp'), ('BF5', '30comp'),
]

# Convergence: track per-frame nRMSE for selected episodes
CONVERGENCE_EPISODES = [
    'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]


def subsample_frame(frame, fraction):
    """Create a new FrameData with subsampled observations."""
    n = len(frame.x_obs)
    if fraction >= 1.0 or n == 0:
        return frame
    k = max(1, int(n * fraction))
    idx = torch.randperm(n)[:k]
    return FrameData(
        t=frame.t,
        x_obs=frame.x_obs[idx],
        y_obs=frame.y_obs[idx],
        z_obs=frame.z_obs[idx],
        confidence=frame.confidence[idx] if frame.confidence is not None else None,
        incidence_angles=frame.incidence_angles[idx] if frame.incidence_angles is not None else None,
        gt_field=frame.gt_field,
    )


class SubsampledLoader:
    """Wraps EpisodeDataLoader with observation subsampling."""

    def __init__(self, loader, fraction):
        self.loader = loader
        self.fraction = fraction
        # Copy attributes
        self.gt_grid_x = loader.gt_grid_x
        self.gt_grid_y = loader.gt_grid_y
        self.depth = loader.depth
        self.Hs_true = loader.Hs_true
        self.Tp_true = loader.Tp_true
        self.sea_state = loader.sea_state
        self.n_frames = loader.n_frames

    def get_init_window(self, tp_factor=2.0):
        frames = self.loader.get_init_window(tp_factor)
        return [subsample_frame(f, self.fraction) for f in frames]

    def get_tracking_frames(self, tp_factor=2.0):
        frames = self.loader.get_tracking_frames(tp_factor)
        return [subsample_frame(f, self.fraction) for f in frames]

    def __iter__(self):
        for frame in self.loader:
            yield subsample_frame(frame, self.fraction)


def run_ws_with_sparsity(h5_path, fraction):
    """Run WaveSplat-P1 with subsampled observations."""
    set_seed(42)
    loader = EpisodeDataLoader(h5_path)
    sub_loader = SubsampledLoader(loader, fraction)

    cfg = {
        'n_gaussians': [3, 8, 15],
        'depth': sub_loader.depth,
        'lr': 0.01,
        'lambda_l1': 0.001,
        'lambda_smooth': 0.0001,
        'tracking_steps': 10,
        'tracking_mode': 'w_phi',
        'tracking_lr': 0.005,
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'use_confidence': True,
        'adaptive_n': True,
        'prune_threshold': 0.10,
        'grow_residual_threshold': 0.05,
        'max_n': 30,
        'tracking_window': 3,
        'device': 'cpu',
        'seed': 42,
    }

    updater = SequentialUpdater(cfg)
    init_frames = sub_loader.get_init_window(tp_factor=2.0)
    updater.initialize(
        init_frames,
        x_range=(float(sub_loader.gt_grid_x[0]), float(sub_loader.gt_grid_x[-1])),
        y_range=(float(sub_loader.gt_grid_y[0]), float(sub_loader.gt_grid_y[-1])),
    )

    grid_x = sub_loader.gt_grid_x
    grid_y = sub_loader.gt_grid_y
    tracking_frames = sub_loader.get_tracking_frames(tp_factor=2.0)

    frame_results = []
    for frame in tracking_frames:
        updater.update(frame)
        pred = updater.predict_field(grid_x, grid_y, frame.t)
        pred_np = pred.cpu().numpy()
        gt_np = frame.gt_field.cpu().numpy()
        metrics = compute_all_metrics(
            pred_np, gt_np,
            grid_x.cpu().numpy(), grid_y.cpu().numpy(),
            sub_loader.Hs_true)
        frame_results.append(metrics)

    avg_pts = np.mean([len(f.x_obs) for f in tracking_frames])
    return {
        'avg_nRMSE': float(np.mean([r['nRMSE'] for r in frame_results])),
        'avg_pts': float(avg_pts),
        'final_n': updater.model.n_gaussians,
    }


def experiment1_sparsity_sweep():
    """Sparsity sweep: vary observation fraction."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    results = {}

    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Sparsity Sweep")
    print("=" * 80)

    for ep_name in SPARSITY_EPISODES:
        h5_path = os.path.join(data_dir, ep_name)
        if not os.path.exists(h5_path):
            print(f"  SKIP: {ep_name}")
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '').replace('_seed0_uniform', '')
        results[short] = {}
        print(f"\n[{short}]")

        for frac in SPARSITY_FRACTIONS:
            try:
                r = run_ws_with_sparsity(h5_path, frac)
                results[short][f"{int(frac*100)}%"] = r
                print(f"  {int(frac*100):>3}%: nRMSE={r['avg_nRMSE']:.4f} "
                      f"(avg_pts={r['avg_pts']:.0f}, N={r['final_n']})", flush=True)
            except Exception as e:
                print(f"  {int(frac*100):>3}%: FAILED ({e})")
                results[short][f"{int(frac*100)}%"] = {'error': str(e)}

    return results


def experiment2_bf_trend():
    """BF-level trend: extract per-BF average from existing P1 results."""
    results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'p1_comparison.json')
    if not os.path.exists(results_path):
        print("  p1_comparison.json not found")
        return None

    with open(results_path) as f:
        data = json.load(f)

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: BF-Level Trend Analysis")
    print("=" * 80)

    bf_groups = {}
    for ep, res in data.items():
        bf = ep.split('_')[0]  # BF2, BF3, BF4, BF5
        if bf not in bf_groups:
            bf_groups[bf] = {'ws': [], 'etkf': [], 'n_comp': []}

        ws_n = res.get('WS-P1', {}).get('avg_nRMSE', float('nan'))
        et_n = res.get('ETKF', {}).get('avg_nRMSE', float('nan'))

        # Extract n_comp
        parts = ep.split('_')
        n_comp = None
        for p in parts:
            if p.endswith('comp'):
                n_comp = int(p.replace('comp', ''))
                break

        if not np.isnan(ws_n):
            bf_groups[bf]['ws'].append(ws_n)
        if not np.isnan(et_n):
            bf_groups[bf]['etkf'].append(et_n)
        if n_comp is not None:
            bf_groups[bf]['n_comp'].append(n_comp)

    results = {}
    print(f"\n{'BF':<6} {'N_eps':>6} {'WS avg':>8} {'ETKF avg':>10} {'Δ%':>8} {'Avg N_comp':>10}")
    print("-" * 50)

    for bf in sorted(bf_groups.keys()):
        g = bf_groups[bf]
        ws_avg = np.mean(g['ws']) if g['ws'] else float('nan')
        et_avg = np.mean(g['etkf']) if g['etkf'] else float('nan')
        nc_avg = np.mean(g['n_comp']) if g['n_comp'] else float('nan')
        delta = (et_avg - ws_avg) / et_avg * 100 if not np.isnan(et_avg) else float('nan')

        results[bf] = {
            'n_episodes': len(g['ws']),
            'ws_avg_nRMSE': float(ws_avg),
            'etkf_avg_nRMSE': float(et_avg),
            'improvement_pct': float(delta),
            'avg_n_components': float(nc_avg),
        }

        print(f"{bf:<6} {len(g['ws']):>6} {ws_avg:>8.3f} {et_avg:>10.3f} "
              f"{delta:>+7.1f}% {nc_avg:>10.1f}")

    return results


def experiment3_convergence():
    """Convergence analysis: per-frame nRMSE curves."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    results = {}

    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Convergence Speed Analysis")
    print("=" * 80)

    for ep_name in CONVERGENCE_EPISODES:
        h5_path = os.path.join(data_dir, ep_name)
        if not os.path.exists(h5_path):
            print(f"  SKIP: {ep_name}")
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '').replace('_seed0_uniform', '')
        print(f"\n[{short}]", flush=True)

        config = {
            'init_method': 'v2_konly',
            'use_dispersion': True,
            'use_confidence': True,
            'adaptive_n': True,
            'tracking_window': 3,
            'seed': 42,
        }
        result = run_estimation(h5_path, config=config)

        # Extract per-frame nRMSE curve
        frame_nrmses = [r['nRMSE'] for r in result['frame_results']]
        frame_times = [r['t'] for r in result['frame_results']]

        # Find convergence metrics
        t_converge = result.get('T_converge')
        final_nrmse = result['avg_nRMSE']

        # Rolling average (window=5)
        if len(frame_nrmses) >= 5:
            rolling = np.convolve(frame_nrmses, np.ones(5)/5, mode='valid').tolist()
        else:
            rolling = frame_nrmses

        results[short] = {
            'frame_times': frame_times,
            'frame_nRMSE': frame_nrmses,
            'rolling_nRMSE_5': rolling,
            'T_converge': t_converge,
            'avg_nRMSE': float(final_nrmse),
            'n_frames': len(frame_nrmses),
            'nRMSE_at_10pct': float(frame_nrmses[max(0, len(frame_nrmses)//10)]) if frame_nrmses else None,
            'nRMSE_at_50pct': float(frame_nrmses[max(0, len(frame_nrmses)//2)]) if frame_nrmses else None,
        }

        print(f"  Final nRMSE={final_nrmse:.4f}, T_converge={t_converge}, "
              f"nRMSE@10%={results[short]['nRMSE_at_10pct']:.4f}, "
              f"nRMSE@50%={results[short]['nRMSE_at_50pct']:.4f}")

    return results


def main():
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # Experiment 1: Sparsity sweep
    all_results['sparsity_sweep'] = experiment1_sparsity_sweep()

    # Experiment 2: BF trend (from existing data)
    all_results['bf_trend'] = experiment2_bf_trend()

    # Experiment 3: Convergence
    all_results['convergence'] = experiment3_convergence()

    # Save
    out_path = os.path.join(output_dir, 'extended_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nAll results saved to {out_path}")


if __name__ == '__main__':
    main()
