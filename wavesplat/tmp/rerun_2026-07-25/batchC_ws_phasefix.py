"""Batch C: TRACK_PHASE_PREDICT=False (redundant-phase-advance fix, tmp/TRACKER_AMPDECAY_2026-07-25.md)
100-episode x 5-seed confirmation arm. Protocol IDENTICAL to batchB_ws.py (the OFF/shipped arm)
so the two jsonls pair per (episode, seed). Records the same panel + RMSE_unobs + spectral.
Decision context: 12-ep pre-registered eval ADOPTED the fix (12/12, mean -0.1034); this run is
the power-gate before any default flip / paper propagation (discipline: finding != publishing).
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'batchC_ws_phasefix.jsonl')

SEEDS = [42, 1, 2, 3, 4]
P1 = {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
      'adaptive_n': True, 'tracking_window': 3}
PANEL = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score', 'Directional_R2', 'Hs_error']


# ---- spectral matching, verbatim from scripts/run_three_contributions.py ----
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
    matched, unmatched_gt, unmatched_est = match_components(est_kx, est_ky, est_w, gt_kx, gt_ky, gt_A)
    result = {'n_gt': int(len(gt_A)), 'n_est': int(len(est_kx)), 'n_matched': int(len(matched)),
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
        result['k_mag_rel_error_mean'] = float('nan')
        result['omega_rel_error_mean'] = float('nan')
        result['amp_rel_error_mean'] = float('nan')
        result['direction_error_deg_mean'] = float('nan')
        result['energy_detection_rate'] = 0.0
    return result


def episodes():
    return sorted(f for f in os.listdir(DATA) if f.endswith('.h5'))


SPECTRAL_EPS = {
    'episode_BF3_1comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5',
}


def build_jobs():
    """Order so partial cutoffs stay useful:
      1. spectral episodes, all 5 seeds  -> Task 3 (spectral) finishes first;
      2. seed 42 for every other episode -> a full 100-ep RMSE_unobs (>=1 seed each) next;
      3. seeds 1..4 for the rest, seed-major -> refines RMSE_unobs to median-of-5.
    """
    eps = episodes()
    jobs = []
    # group 1: spectral eps, all seeds
    for ep in eps:
        if ep in SPECTRAL_EPS:
            for s in SEEDS:
                jobs.append({'job_id': f'{ep}|{s}', 'episode': ep, 'seed': s})
    # group 2: seed 42 for the rest
    for ep in eps:
        if ep not in SPECTRAL_EPS:
            jobs.append({'job_id': f'{ep}|42', 'episode': ep, 'seed': 42})
    # group 3: remaining seeds, seed-major
    for s in [1, 2, 3, 4]:
        for ep in eps:
            if ep not in SPECTRAL_EPS:
                jobs.append({'job_id': f'{ep}|{s}', 'episode': ep, 'seed': s})
    return jobs


def run_one(job):
    sys.path.insert(0, REPO)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from wavesplat.run import run_estimation
    from wavesplat.data_loader import EpisodeDataLoader
    import wavesplat.optimizer.sequential_update as _su
    _su.TRACK_PHASE_PREDICT = False  # the fix arm; read at update() call time
    cfg = dict(P1); cfg['seed'] = job['seed']
    base = {k: job[k] for k in ('job_id', 'episode', 'seed')}
    t0 = time.time()
    try:
        path = os.path.join(DATA, job['episode'])
        r = run_estimation(path, cfg)
        frames = r['frame_results']
        avg = {k: float(np.nanmean([f[k] for f in frames if k in f]))
               for k in PANEL}
        ru = [f['RMSE_unobserved'] for f in frames
              if 'RMSE_unobserved' in f and np.isfinite(f.get('RMSE_unobserved', np.nan))]
        avg_ru = float(np.mean(ru)) if ru else float('nan')
        # spectral accuracy on the final spectral state
        try:
            loader = EpisodeDataLoader(path)
            gt_comp = np.asarray(loader.wave_components)
            spec = spectral_accuracy(r['spectrum_params'], gt_comp)
        except Exception as se:
            spec = {'spec_error': repr(se)}
        return {**base, **avg, 'avg_RMSE_unobserved': avg_ru,
                'elapsed_ms': float(np.mean([f['elapsed_ms'] for f in frames])) if frames else float('nan'),
                'N': r.get('n_gaussians', -1), 'spec': spec, 'wall_s': time.time() - t0}
    except Exception as e:
        return {**base, 'error': repr(e), 'wall_s': time.time() - t0}


def _init():
    sys.path.insert(0, REPO)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


def main():
    import multiprocessing as mp
    n_workers = int(os.environ.get('WORKERS', '4'))
    jobs = build_jobs()
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
    print(f'total {len(jobs)} | done {len(done)} | todo {len(todo)} | workers {n_workers}', flush=True)
    if not todo:
        print('nothing to do'); return
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(n_workers, initializer=_init) as pool, open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            el = time.time() - t0
            if i % 10 == 0 or i == len(todo):
                tag = rec.get('nRMSE', rec.get('error', '?'))
                print(f'[{i}/{len(todo)}] {rec["job_id"]:<48s} {tag} '
                      f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
