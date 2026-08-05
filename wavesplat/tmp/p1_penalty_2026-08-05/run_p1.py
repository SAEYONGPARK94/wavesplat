"""P1 (Phase-2-only penalty removal) in the DEPLOYMENT configuration.

Pre-registration: tmp/p1_penalty_2026-08-05/PREREG.md (frozen before any new run).

Three arms x 12 episodes x 3 seeds = 108 runs.
  CTRL     : the shipped WS-P1 config (= tmp/rerun_2026-08-03 'WS-P1'), bit-identical.
  P1       : same, but lambda_l1 = lambda_smooth = 0 set on the updater AFTER
             initialize() returns -> Phase 1 is bit-identical, only the tracking
             objective changes.  Implemented as a monkey-patch in THIS file;
             nothing under wavesplat/ is modified.
  P1-both  : lambda_l1 = lambda_smooth = 0 passed in the config, i.e. both phases.
             Negative control -- the debate predicts this is WORSE.

Records the full panel per cell (nRMSE, VR, rho, SS, DirR2, Hs_error, final_n,
component detection rate, sum|w| vs the true sum|A|, init_nrmse).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/home/vexit/wavesplat')

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = '/home/vexit/wavesplat/wavesplat/data_finemesh_2026-07-23'
SEEDS = [42, 123, 456]

EPISODES = [
    'BF2_1comp_seed0_uniform', 'BF3_1comp_seed0_uniform',
    'BF3_2comp_seed0_uniform', 'BF3_2comp_seed0_slope',
    'BF3_5comp_seed0_uniform', 'BF3_5comp_seed0_slope',
    'BF4_1comp_seed0_uniform', 'BF4_5comp_seed0_uniform',
    'BF4_15comp_seed0_uniform', 'BF4_30comp_seed0_uniform',
    'BF5_15comp_seed0_uniform', 'BF5_30comp_seed0_uniform',
]

BASE = {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
        'adaptive_n': True, 'tracking_window': 3}

ARMS = {
    'CTRL':    (dict(BASE), False),
    'P1':      (dict(BASE), True),
    'P1-both': ({**BASE, 'lambda_l1': 0.0, 'lambda_smooth': 0.0}, False),
}


def _install_phase2_patch():
    """Zero the two penalties on the updater right after initialize() returns."""
    from wavesplat.optimizer import sequential_update as su
    orig = su.SequentialUpdater.initialize

    def patched(self, *a, **kw):
        out = orig(self, *a, **kw)
        self.lambda_l1 = 0.0
        self.lambda_smooth = 0.0
        return out
    su.SequentialUpdater.initialize = patched
    return orig


def _restore(orig):
    from wavesplat.optimizer import sequential_update as su
    su.SequentialUpdater.initialize = orig


def _detection_rate(spec, gt_components):
    """Hungarian match on (kx,ky), cost < 0.5 * max|k_gt| -- same rule as
    scripts/run_three_contributions.py:match_components, so numbers are comparable."""
    from scipy.optimize import linear_sum_assignment
    if spec is None or gt_components is None:
        return None, None, None
    gt = np.asarray(gt_components)
    gt_A, gt_kx, gt_ky = gt[:, 0], gt[:, 2], gt[:, 3]
    ex, ey = np.asarray(spec['kx']), np.asarray(spec['ky'])
    if len(ex) == 0 or len(gt_kx) == 0:
        return 0.0, 0.0, float(np.sum(np.abs(gt_A)))
    k_scale = max(float(np.max(np.sqrt(gt_kx ** 2 + gt_ky ** 2))), 0.01)
    cost = np.sqrt((ex[:, None] - gt_kx[None, :]) ** 2
                   + (ey[:, None] - gt_ky[None, :]) ** 2) / k_scale
    r, c = linear_sum_assignment(cost)
    keep = [(i, j) for i, j in zip(r, c) if cost[i, j] < 0.5]
    det = len(keep) / max(len(gt_A), 1)
    e_tot = float(np.sum(gt_A ** 2))
    e_det = float(np.sum(gt_A[[j for _, j in keep]] ** 2)) / max(e_tot, 1e-12)
    return float(det), float(e_det), float(np.sum(np.abs(gt_A)))


def run_cell(ep, seed, arm):
    from wavesplat.run import run_estimation
    import h5py
    cfg, patch = ARMS[arm]
    orig = _install_phase2_patch() if patch else None
    t0 = time.time()
    try:
        r = run_estimation(f'{DATA}/episode_{ep}.h5', {**cfg, 'seed': seed}, verbose=False)
    finally:
        if orig is not None:
            _restore(orig)

    fr = r['frame_results']

    def avg(k):
        v = [f[k] for f in fr if k in f and np.isfinite(f[k])]
        return float(np.mean(v)) if v else float('nan')

    with h5py.File(f'{DATA}/episode_{ep}.h5', 'r') as h:
        gtc = None
        for key in ('config/wave_components', 'wave_components'):
            if key in h:
                gtc = np.asarray(h[key])
                break
    spec = r.get('spectrum_params')
    det, e_det, sum_A = _detection_rate(spec, gtc)
    sum_w = float(np.sum(np.abs(spec['w']))) if spec is not None else float('nan')

    return {
        'arm': arm, 'episode': ep, 'seed': seed,
        'avg_nRMSE': float(r['avg_nRMSE']),
        'init_nrmse': float(r['init_nrmse']),
        'avg_RMSE_z': float(r['avg_RMSE_z']),
        'avg_Hs_error': float(r['avg_Hs_error']),
        'VR': avg('Variance_Ratio'),
        'rho': avg('Pearson_rho'),
        'SS': avg('Skill_Score'),
        'DirR2': avg('Directional_R2'),
        'RMSE_unobs': avg('RMSE_unobserved'),
        'obs_loss': float(r.get('avg_obs_loss', float('nan'))),
        'final_n': int(r.get('n_gaussians') or 0),
        'det_rate': det, 'energy_det_rate': e_det,
        'sum_w': sum_w, 'sum_A_true': sum_A,
        'sum_w_ratio': (sum_w / sum_A) if (sum_A and np.isfinite(sum_w)) else float('nan'),
        'n_frames': len(fr),
        'wall_s': time.time() - t0,
    }


if __name__ == '__main__':
    # worker mode: run_p1.py <arm> <episode> <seed> ; prints one JSON line
    if len(sys.argv) == 4:
        arm, ep, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
        try:
            rec = run_cell(ep, seed, arm)
        except Exception as e:            # a hole is recorded, not hidden
            rec = {'arm': arm, 'episode': ep, 'seed': seed,
                   'error': f'{type(e).__name__}: {e}'}
        print('@@' + json.dumps(rec, default=float), flush=True)
        sys.exit(0)
    # driver mode
    todo = [(a, e, s) for a in ARMS for e in EPISODES for s in SEEDS]
    print(f'{len(todo)} cells', flush=True)
    for a, e, s in todo:
        print(f'{a} {e} {s}')
