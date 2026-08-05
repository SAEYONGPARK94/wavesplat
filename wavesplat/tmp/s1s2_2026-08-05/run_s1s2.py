"""S1 (component-budget sweep) + S2 (oracle kx/ky init) driver.

Pre-registered in tmp/s1s2_2026-08-05/PREREG.md. Do not change arm definitions after the
first run: the prereg fixes them.

Protocol inherited verbatim from tmp/rerun_2026-07-25/batchC_ws_phasefix.py (panel,
spectral_accuracy, resumable jsonl, spawn pool) so results pair with the frozen 100-ep run.

Every arm holds sum(max_iters_per_stage) == 900, because
SequentialUpdater.initialize always executes exactly that many gradient steps -- so no arm
gets more optimisation than another, only a different component budget / placement.

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
EPISODES = [
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]
PANEL = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
         'Directional_R2', 'Hs_error']

# N_gt per episode (verified from config/wave_components shape).
N_GT = {
    'episode_BF4_15comp_seed0_uniform.h5': 15,
    'episode_BF4_30comp_seed0_uniform.h5': 30,
    'episode_BF5_15comp_seed0_uniform.h5': 15,
    'episode_BF5_30comp_seed0_uniform.h5': 30,
}

ARMS = ['S1-A0', 'S1-A1', 'S1-A2', 'S1-A3',
        'S2-C1', 'S2-T1', 'S2-T1f', 'S2-T2', 'S2-T4']
# POST-HOC EXPLORATORY (added 2026-08-05 AFTER the pre-registered arms had run, because
# S2-T1f's init_nrmse 0.063 vs tracking-mean 0.167 said the oracle's residual is created
# during TRACKING, not at init). NOT part of the prereg; cannot change the S1/S2 verdicts.
# tracking_lr = 0 makes the tracker a no-op (Adam with lr 0), so the model only propagates
# forward under the dispersion relation. tracking_steps=0 is not usable -- `loss` would be
# unbound in SequentialUpdater.update.
EXPLORATORY_ARMS = ['X-T1f-notrack', 'X-A0-notrack']
ORACLE_ARMS = {'S2-T1', 'S2-T1f', 'S2-T2', 'S2-T4', 'X-T1f-notrack'}


def arm_config(arm, episode):
    """Config overrides for one arm. PREREG sections 4 and 5.2."""
    n_gt = N_GT[episode]
    cfg = {'use_dispersion': True, 'use_confidence': True,
           'adaptive_n': True, 'tracking_window': 3,
           'prune_threshold': 0.10, 'max_n': 30}
    if arm == 'S1-A0':          # control == shipped stage ladder
        cfg.update(init_method='v2_konly', n_gaussians=[3, 8, 15],
                   loss_thresholds=[0.1, 0.05, None],
                   max_iters_per_stage=[200, 200, 500])
    elif arm == 'S1-A1':
        cfg.update(init_method='v2_konly', n_gaussians=[3, 8, 15, 22],
                   loss_thresholds=[0.1, 0.05, 0.05, None],
                   max_iters_per_stage=[200, 200, 200, 300])
    elif arm == 'S1-A2':
        cfg.update(init_method='v2_konly', n_gaussians=[3, 8, 15, 30],
                   loss_thresholds=[0.1, 0.05, 0.05, None],
                   max_iters_per_stage=[200, 200, 200, 300])
    elif arm == 'S1-A3':
        cfg.update(init_method='v2_konly', n_gaussians=[3, 8, 15, 30, 40],
                   loss_thresholds=[0.1, 0.05, 0.05, 0.05, None],
                   max_iters_per_stage=[200, 200, 200, 200, 100],
                   max_n=40)
    elif arm == 'S2-C1':        # matched-schedule control: same N, beamform placement
        cfg.update(init_method='v2_konly', n_gaussians=[n_gt],
                   loss_thresholds=[None], max_iters_per_stage=[900])
    elif arm == 'S2-T1':        # single-factor treatment: GT placement, k still trainable
        cfg.update(init_method='oracle', n_gaussians=[n_gt],
                   loss_thresholds=[None], max_iters_per_stage=[900])
    elif arm == 'S2-T1f':       # PRIMARY: GT placement, k frozen -> w/phi only
        cfg.update(init_method='oracle', n_gaussians=[n_gt],
                   loss_thresholds=[None], max_iters_per_stage=[900],
                   init_k_lr_scale=0.0)
    elif arm == 'S2-T2':        # GT placement, pruning disabled
        cfg.update(init_method='oracle', n_gaussians=[n_gt],
                   loss_thresholds=[None], max_iters_per_stage=[900],
                   prune_threshold=0.0)
    elif arm == 'S2-T4':        # GT placement, k frozen, adaptive-N off entirely
        cfg.update(init_method='oracle', n_gaussians=[n_gt],
                   loss_thresholds=[None], max_iters_per_stage=[900],
                   init_k_lr_scale=0.0, adaptive_n=False)
    elif arm == 'X-T1f-notrack':   # EXPLORATORY: S2-T1f with the tracker frozen
        cfg.update(init_method='oracle', n_gaussians=[n_gt],
                   loss_thresholds=[None], max_iters_per_stage=[900],
                   init_k_lr_scale=0.0, tracking_lr=0.0)
    elif arm == 'X-A0-notrack':    # EXPLORATORY: S1-A0 with the tracker frozen
        cfg.update(init_method='v2_konly', n_gaussians=[3, 8, 15],
                   loss_thresholds=[0.1, 0.05, None],
                   max_iters_per_stage=[200, 200, 500], tracking_lr=0.0)
    else:
        raise ValueError(arm)
    return cfg


# ---- spectral matching, verbatim from tmp/rerun_2026-07-25/batchC_ws_phasefix.py ----
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


def build_jobs(arms=None):
    arms = arms or ARMS
    jobs = []
    for arm in arms:
        for ep in EPISODES:
            for s in SEEDS:
                jobs.append({'job_id': f'{arm}|{ep}|{s}', 'arm': arm,
                             'episode': ep, 'seed': s})
    return jobs


def run_one(job):
    sys.path.insert(0, REPO)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from wavesplat.run import run_estimation
    from wavesplat.data_loader import EpisodeDataLoader
    import wavesplat.optimizer.initializer as _init

    base = {k: job[k] for k in ('job_id', 'arm', 'episode', 'seed')}
    t0 = time.time()
    path = os.path.join(DATA, job['episode'])
    try:
        cfg = arm_config(job['arm'], job['episode'])
        cfg['seed'] = job['seed']

        gt_comp = np.asarray(EpisodeDataLoader(path).wave_components)
        if job['arm'] in ORACLE_ARMS:
            _init.ORACLE_COMPONENTS = gt_comp
        else:
            _init.ORACLE_COMPONENTS = None

        try:
            r = run_estimation(path, cfg)
        finally:
            _init.ORACLE_COMPONENTS = None   # never leaks into a later job

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
                'spec': spec, 'cfg': {k: v for k, v in cfg.items()
                                      if k not in ('seed',)},
                'wall_s': time.time() - t0}
    except Exception as e:
        import traceback
        return {**base, 'error': repr(e), 'tb': traceback.format_exc()[-1500:],
                'wall_s': time.time() - t0}


def _init_worker():
    sys.path.insert(0, REPO)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')


def main():
    import multiprocessing as mp
    arms = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    n_workers = int(os.environ.get('WORKERS', '8'))
    jobs = build_jobs(arms)
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
