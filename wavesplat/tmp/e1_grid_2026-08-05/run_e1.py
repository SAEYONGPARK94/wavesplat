"""E1 — search-grid resolution 2x2 factorial (+ 07-23 replication arm E).

Pre-registered in tmp/e1_grid_2026-08-05/PREREG.md. Arm definitions are frozen by that file.

Protocol inherited verbatim from tmp/s1s2_2026-08-05/run_s1s2.py (panel, spectral_accuracy,
resumable jsonl, spawn pool) so cell A pairs bit-for-bit with arm S1-A0 there.

PRODUCTION CODE IS NOT MODIFIED. Every factor is applied by monkey-patching inside the
worker process:
  * factor GRID  -> wraps initializer._beamform_coarse AND coarse_to_fine._beamform_coarse
                    so the hardcoded (-4,4) at all THREE placement sites is overridden while
                    n_kx/n_ky keep their site-specific shipped values (81/61/61).
  * factor PEAK  -> initializer.PEAK_SELECT_MODE (already a module-level knob).
  * arm E        -> initializer.MIN_DK (already a module-level knob; the two 61-grid sites
                    re-import it inside the function, so they follow).

TIMING IS RECORDED BUT NOT CITABLE (this box's clock is under separate re-measurement).
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'results.jsonl')

SEEDS = [42, 123, 456]

# scripts/run_p1_comparison.py::EPISODES, verbatim.
EPISODES = [
    'episode_BF2_1comp_seed0_uniform.h5',
    'episode_BF3_1comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_slope.h5',
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]
PANEL = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
         'Directional_R2', 'Hs_error']

WIDE = (-4.0, 4.0)
NARROW = (-0.8, 0.8)

# cell -> (kx/ky range, PEAK_SELECT_MODE, MIN_DK)
CELLS = {
    'A': (WIDE,   'static_suppress',    0.15),   # control == shipped
    'B': (NARROW, 'static_suppress',    0.15),
    'C': (WIDE,   'recompute_residual', 0.15),
    'D': (NARROW, 'recompute_residual', 0.15),
    'E': (WIDE,   'static_suppress',    0.05),   # 2026-07-23 sweep treatment, replicated
}

# WS-P1, identical to tmp/s1s2_2026-08-05 arm S1-A0. sum(max_iters) == 900.
BASE_CFG = dict(init_method='v2_konly', use_dispersion=True, use_confidence=True,
                adaptive_n=True, tracking_window=3, prune_threshold=0.10, max_n=30,
                n_gaussians=[3, 8, 15], loss_thresholds=[0.1, 0.05, None],
                max_iters_per_stage=[200, 200, 500])


def apply_cell(cell):
    """Install the cell's factors by monkey-patch. Returns an undo callable."""
    import wavesplat.optimizer.initializer as I
    import wavesplat.optimizer.coarse_to_fine as C

    krange, mode, min_dk = CELLS[cell]
    orig_bf = I._beamform_coarse
    orig_mode, orig_mindk = I.PEAK_SELECT_MODE, I.MIN_DK

    def bf(x, y, z, t, depth, kx_range=WIDE, ky_range=WIDE, n_kx=81, n_ky=81):
        # range overridden; grid COUNT is whatever the call site asked for (81 / 61).
        return orig_bf(x, y, z, t, depth, krange, krange, n_kx, n_ky)

    I._beamform_coarse = bf
    C._beamform_coarse = bf          # coarse_to_fine imported the symbol at module load
    I.PEAK_SELECT_MODE = mode
    I.MIN_DK = min_dk

    def undo():
        I._beamform_coarse = orig_bf
        C._beamform_coarse = orig_bf
        I.PEAK_SELECT_MODE = orig_mode
        I.MIN_DK = orig_mindk
    return undo


# ---- spectral matching, verbatim from tmp/s1s2_2026-08-05/run_s1s2.py ----
def match_components(est_kx, est_ky, est_w, gt_kx, gt_ky, gt_A):
    N_est = len(est_kx); N_gt = len(gt_kx)
    if N_est == 0 or N_gt == 0:
        return [], list(range(N_gt)), list(range(N_est))
    k_scale = max(np.max(np.sqrt(gt_kx ** 2 + gt_ky ** 2)), 0.01)
    cost = np.zeros((N_est, N_gt))
    for i in range(N_est):
        for j in range(N_gt):
            dk = np.sqrt((est_kx[i] - gt_kx[j]) ** 2 + (est_ky[i] - gt_ky[j]) ** 2)
            cost[i, j] = dk / k_scale
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = [(r, c, cost[r, c]) for r, c in zip(row_ind, col_ind) if cost[r, c] < 0.5]
    matched_gt = {m[1] for m in matched}; matched_est = {m[0] for m in matched}
    unmatched_gt = [j for j in range(N_gt) if j not in matched_gt]
    unmatched_est = [i for i in range(N_est) if i not in matched_est]
    return matched, unmatched_gt, unmatched_est


def spectral_accuracy(est_params, gt_components):
    gt_A = gt_components[:, 0]; gt_omega = gt_components[:, 1]
    gt_kx = gt_components[:, 2]; gt_ky = gt_components[:, 3]
    est_kx = np.asarray(est_params['kx']); est_ky = np.asarray(est_params['ky'])
    est_w = np.abs(np.asarray(est_params['w'])); est_omega = np.asarray(est_params['omega'])
    matched, unmatched_gt, unmatched_est = match_components(
        est_kx, est_ky, est_w, gt_kx, gt_ky, gt_A)
    result = {'n_gt': int(len(gt_A)), 'n_est': int(len(est_kx)),
              'n_matched': int(len(matched)),
              'detection_rate': len(matched) / max(len(gt_A), 1)}
    if matched:
        k_mag_errors = []; omega_errors = []; amp_errors = []; direction_errors = []
        for est_idx, gt_idx, cost in matched:
            gt_kmag = np.sqrt(gt_kx[gt_idx] ** 2 + gt_ky[gt_idx] ** 2)
            est_kmag = np.sqrt(est_kx[est_idx] ** 2 + est_ky[est_idx] ** 2)
            k_mag_errors.append((est_kmag - gt_kmag) / max(gt_kmag, 1e-6))
            omega_errors.append((est_omega[est_idx] - gt_omega[gt_idx]) / max(gt_omega[gt_idx], 1e-6))
            amp_errors.append((est_w[est_idx] - gt_A[gt_idx]) / max(gt_A[gt_idx], 1e-6))
            gt_dir = np.degrees(np.arctan2(gt_ky[gt_idx], gt_kx[gt_idx]))
            est_dir = np.degrees(np.arctan2(est_ky[est_idx], est_kx[est_idx]))
            d_dir = abs(est_dir - gt_dir); d_dir = 360 - d_dir if d_dir > 180 else d_dir
            direction_errors.append(d_dir)
        result['k_mag_rel_error_mean'] = float(np.mean(np.abs(k_mag_errors)))
        result['omega_rel_error_mean'] = float(np.mean(np.abs(omega_errors)))
        result['amp_rel_error_mean'] = float(np.mean(np.abs(amp_errors)))
        result['direction_error_deg_mean'] = float(np.mean(direction_errors))
        matched_gt_indices = [m[1] for m in matched]
        matched_energy = float(np.sum(gt_A[matched_gt_indices] ** 2))
        total_energy = float(np.sum(gt_A ** 2))
        result['energy_detection_rate'] = matched_energy / max(total_energy, 1e-12)
    else:
        for k in ('k_mag_rel_error_mean', 'omega_rel_error_mean',
                  'amp_rel_error_mean', 'direction_error_deg_mean'):
            result[k] = float('nan')
        result['energy_detection_rate'] = 0.0
    return result


def build_jobs(cells=None, episodes=None, seeds=None):
    cells = cells or list(CELLS)
    episodes = episodes or EPISODES
    seeds = seeds or SEEDS
    return [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in cells for ep in episodes for s in seeds]


def run_one(job):
    sys.path.insert(0, REPO)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from wavesplat.run import run_estimation
    from wavesplat.data_loader import EpisodeDataLoader
    import wavesplat.optimizer.initializer as _init

    base = {k: job[k] for k in ('job_id', 'cell', 'episode', 'seed')}
    t0 = time.time()
    path = os.path.join(DATA, job['episode'])
    undo = None
    try:
        cfg = dict(BASE_CFG); cfg['seed'] = job['seed']
        gt_comp = np.asarray(EpisodeDataLoader(path).wave_components)
        _init.ORACLE_COMPONENTS = None
        undo = apply_cell(job['cell'])
        try:
            r = run_estimation(path, cfg)
        finally:
            undo(); undo = None

        frames = r['frame_results']
        avg = {k: float(np.nanmean([f[k] for f in frames if k in f])) for k in PANEL}
        ru = [f['RMSE_unobserved'] for f in frames
              if 'RMSE_unobserved' in f and np.isfinite(f.get('RMSE_unobserved', np.nan))]
        avg_ru = float(np.mean(ru)) if ru else float('nan')
        try:
            spec = spectral_accuracy(r['spectrum_params'], gt_comp)
        except Exception as se:
            spec = {'spec_error': repr(se)}
        return {**base, **avg, 'avg_RMSE_unobserved': avg_ru,
                'elapsed_ms': float(np.mean([f['elapsed_ms'] for f in frames]))
                if frames else float('nan'),
                'N': r.get('n_gaussians', -1),
                'init_nrmse': float(r.get('init_nrmse', float('nan'))),
                'spec': spec,
                'cell_cfg': {'range': CELLS[job['cell']][0],
                             'peak_mode': CELLS[job['cell']][1],
                             'min_dk': CELLS[job['cell']][2]},
                'wall_s': time.time() - t0}
    except Exception as e:
        import traceback
        if undo is not None:
            undo()
        return {**base, 'error': repr(e), 'tb': traceback.format_exc()[-1500:],
                'wall_s': time.time() - t0}


def _init_worker():
    sys.path.insert(0, REPO)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')


def main():
    import multiprocessing as mp
    cells = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    eps = None
    if len(sys.argv) > 2 and sys.argv[2] != '-':
        eps = [e for e in EPISODES if any(tag in e for tag in sys.argv[2].split(','))]
    n_workers = int(os.environ.get('WORKERS', '12'))
    jobs = build_jobs(cells, eps)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add(r['job_id'])
    todo = [j for j in jobs if j['job_id'] not in done]
    print(f'total {len(jobs)} | done {len(done)} | todo {len(todo)} | workers {n_workers}',
          flush=True)
    if not todo:
        print('nothing to do'); return
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(n_workers, initializer=_init_worker) as pool, open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            el = time.time() - t0
            tag = rec.get('nRMSE', rec.get('error', '?'))
            if isinstance(tag, float):
                tag = f'{tag:.4f}'
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<52s} {tag} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
