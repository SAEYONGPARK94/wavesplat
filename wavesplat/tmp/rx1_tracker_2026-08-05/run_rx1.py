"""Rx-1 driver: single-knob sweep on the oracle-placement baseline (PREREG.md sec.2).

Protocol inherited verbatim from tmp/s1s2_2026-08-05/run_s1s2.py (episodes, panel,
spectral_accuracy, oracle mechanism, spawn pool, resumable jsonl) so that X-BASE
reproduces the S2-T1f cell of debate J (medOfMed nRMSE 0.1602) up to the seed change.

X-BASE  = oracle k placement, k frozen at init, tracker ON   (== S2-T1f)
K1      = lambda_l1     -> 0      (config knob, both phases)
K2      = lambda_smooth -> 0      (config knob, both phases)
K3      = data term / sum(weights) instead of / n_obs   (env RX1_CONF_NORM=1)
K4      = Phase-2 objective = data term only            (env RX1_PHASE2_NOREG=1)
K5      = TRACK_PHASE_PREDICT flipped False -> True     (module flag, monkeypatched)
K12     = K1+K2 together (CONFIRMATION CELL ONLY, PREREG sec.2 cross-ban)

init_method is set EXPLICITLY in every cell (W9 contamination rule).
Seeds consumed from the reserved ledger [1515-1818]: 1515, 1516, 1517.
TIMING RECORDED BUT NOT CITABLE.
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

SEEDS = [1515, 1516, 1517]
EPISODES = [
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]
PANEL = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
         'Directional_R2', 'Hs_error']

N_GT = {
    'episode_BF4_15comp_seed0_uniform.h5': 15,
    'episode_BF4_30comp_seed0_uniform.h5': 30,
    'episode_BF5_15comp_seed0_uniform.h5': 15,
    'episode_BF5_30comp_seed0_uniform.h5': 30,
}

CELLS = ['X-BASE', 'K1', 'K2', 'K3', 'K4', 'K5', 'K12',
         # AMENDMENT 3: the displacement-budget lever, orthogonal to the 2x2
         'K6', 'K7', 'K12-S1',
         # AMENDMENT 3: second operating point = DEPLOYMENT placement (v2_konly ladder)
         'D-BASE', 'D-K12', 'D-K6',
         # tracker-frozen floors, so "% of harm removed" has a denominator at BOTH points
         'X-NOTRACK', 'D-NOTRACK']

# cells whose initialiser is the shipped beamform ladder, NOT oracle placement
DEPLOY_CELLS = {'D-BASE', 'D-K12', 'D-K6', 'D-NOTRACK'}

# env gates + module flags per cell; everything not listed stays at shipped default.
CELL_ENV = {
    'K3': {'RX1_CONF_NORM': '1'},
    'K4': {'RX1_PHASE2_NOREG': '1'},
}
CELL_FLAGS = {
    'K5': {'TRACK_PHASE_PREDICT': True},
}
RX1_ENV_KEYS = ('RX1_CONF_NORM', 'RX1_PHASE2_NOREG')


def cell_config(cell, episode):
    """One knob off the X-BASE config. PREREG sec.2 -- no bundling."""
    n_gt = N_GT[episode]
    cfg = {'use_dispersion': True, 'use_confidence': True,
           'adaptive_n': True, 'tracking_window': 3,
           'prune_threshold': 0.10, 'max_n': 30,
           # X-BASE == S2-T1f
           'init_method': 'oracle', 'n_gaussians': [n_gt],
           'loss_thresholds': [None], 'max_iters_per_stage': [900],
           'init_k_lr_scale': 0.0}
    if cell == 'X-BASE':
        pass
    elif cell == 'K1':
        cfg['lambda_l1'] = 0.0
    elif cell == 'K2':
        cfg['lambda_smooth'] = 0.0
    elif cell in ('K3', 'K4', 'K5'):
        pass                       # env / module flag only, config identical to X-BASE
    elif cell == 'K12':
        cfg['lambda_l1'] = 0.0
        cfg['lambda_smooth'] = 0.0
    # ---- AMENDMENT 3: displacement budget (penalties left at SHIPPED values) ----
    elif cell == 'K6':                      # budget 0.1 via steps
        cfg['tracking_steps'] = 1
    elif cell == 'K7':                      # budget 0.1 via lr  (budget-matched to K6)
        cfg['tracking_lr'] = 0.0005
    elif cell == 'X-NOTRACK':               # budget 0: Adam lr 0 -> tracker is a no-op
        cfg['tracking_lr'] = 0.0
    elif cell == 'K12-S1':                  # EXPLORATORY CROSSING, not part of the 2x2:
        cfg['lambda_l1'] = 0.0              # does zero-lambda buy anything ON TOP of a
        cfg['lambda_smooth'] = 0.0          # short trajectory? (attributability test)
        cfg['tracking_steps'] = 1
    # ---- AMENDMENT 3: deployment operating point (shipped beamform ladder) ----
    elif cell in DEPLOY_CELLS:
        cfg.update(init_method='v2_konly', n_gaussians=[3, 8, 15],
                   loss_thresholds=[0.1, 0.05, None],
                   max_iters_per_stage=[200, 200, 500])
        cfg.pop('init_k_lr_scale', None)    # shipped ladder trains k during init
        if cell == 'D-K12':
            cfg['lambda_l1'] = 0.0
            cfg['lambda_smooth'] = 0.0
        elif cell == 'D-K6':
            cfg['tracking_steps'] = 1
        elif cell == 'D-NOTRACK':
            cfg['tracking_lr'] = 0.0
    else:
        raise ValueError(cell)
    return cfg


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
    # amplitude budget: sum|w| recovered vs GT (shrinkage diagnostic, PREREG sec.3)
    result['sum_w_est'] = float(np.sum(est_w))
    result['sum_w_gt'] = float(np.sum(np.abs(gt_A)))
    return result


def build_jobs(cells=None, seeds=None, episodes=None):
    cells = cells or CELLS
    seeds = seeds or SEEDS
    episodes = episodes or EPISODES
    return [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in cells for ep in episodes for s in seeds]


def run_one(job):
    sys.path.insert(0, REPO)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    cell = job['cell']

    # per-job env gates: always cleared first, because pool workers are reused
    for k in RX1_ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in CELL_ENV.get(cell, {}).items():
        os.environ[k] = v

    from wavesplat.run import run_estimation
    from wavesplat.data_loader import EpisodeDataLoader
    import wavesplat.optimizer.initializer as _init
    import wavesplat.optimizer.sequential_update as _su

    # per-job module flags: reset to shipped default first
    _su.TRACK_PHASE_PREDICT = False
    for k, v in CELL_FLAGS.get(cell, {}).items():
        setattr(_su, k, v)

    base = {k: job[k] for k in ('job_id', 'cell', 'episode', 'seed')}
    t0 = time.time()
    path = os.path.join(DATA, job['episode'])
    try:
        cfg = cell_config(cell, job['episode'])
        cfg['seed'] = job['seed']
        assert 'init_method' in cfg, 'init_method must be explicit (W9 rule)'

        gt_comp = np.asarray(EpisodeDataLoader(path).wave_components)
        # oracle placement for the X-BASE family; the D-* cells are the deployment
        # operating point and must NOT see ground truth (AMENDMENT 3 item 4).
        _init.ORACLE_COMPONENTS = None if cell in DEPLOY_CELLS else gt_comp
        try:
            r = run_estimation(path, cfg)
        finally:
            _init.ORACLE_COMPONENTS = None
            _su.TRACK_PHASE_PREDICT = False
            for k in RX1_ENV_KEYS:
                os.environ.pop(k, None)

        frames = r['frame_results']
        avg = {k: float(np.nanmean([f[k] for f in frames if k in f])) for k in PANEL}

        # AMENDMENT 2: |VR-1| must be formed PER FRAME then averaged (averaging VR first
        # and transforming afterwards flips the sign on cells that push VR above 1).
        # Same for the exact split  m = (1-rho^2) + (sqrt(v)-rho)^2, whose two terms are
        # the DECORRELATION floor and the SHRINKAGE part.
        vr_f = np.array([f['Variance_Ratio'] for f in frames if 'Variance_Ratio' in f],
                        dtype=float)
        rho_f = np.array([f['Pearson_rho'] for f in frames if 'Pearson_rho' in f],
                         dtype=float)
        perframe = {}
        if vr_f.size and vr_f.size == rho_f.size:
            s = np.sqrt(np.clip(vr_f, 0, None))
            perframe = {
                'absVR1_framemean': float(np.nanmean(np.abs(vr_f - 1.0))),
                'decorr_framemean': float(np.nanmean(1.0 - rho_f ** 2)),
                'shrink_framemean': float(np.nanmean((s - rho_f) ** 2)),
                'm_framemean': float(np.nanmean(1.0 - 2 * rho_f * s + vr_f)),
                'vr_frames': [round(float(x), 6) for x in vr_f],
                'rho_frames': [round(float(x), 6) for x in rho_f],
            }
        ru = [f['RMSE_unobserved'] for f in frames
              if 'RMSE_unobserved' in f and np.isfinite(f.get('RMSE_unobserved', np.nan))]
        avg_ru = float(np.mean(ru)) if ru else float('nan')
        try:
            spec = spectral_accuracy(r['spectrum_params'], gt_comp)
        except Exception as se:
            spec = {'spec_error': repr(se)}
        return {**base, **avg, **perframe, 'avg_RMSE_unobserved': avg_ru,
                'elapsed_ms': float(np.mean([f['elapsed_ms'] for f in frames]))
                if frames else float('nan'),
                'N': r.get('n_gaussians', -1),
                'init_nrmse': float(r.get('init_nrmse', float('nan'))),
                'spec': spec,
                'cfg': {k: v for k, v in cfg.items() if k not in ('seed',)},
                'env': {k: os.environ.get(k, '') for k in RX1_ENV_KEYS},
                'flags': {'TRACK_PHASE_PREDICT': bool(CELL_FLAGS.get(cell, {})
                                                      .get('TRACK_PHASE_PREDICT', False))},
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
    cells = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    out = os.environ.get('RX1_OUT', OUT)
    seeds = [int(s) for s in os.environ['RX1_SEEDS'].split(',')] \
        if os.environ.get('RX1_SEEDS') else None
    episodes = os.environ['RX1_EPS'].split(',') if os.environ.get('RX1_EPS') else None
    n_workers = int(os.environ.get('WORKERS', '4'))
    jobs = build_jobs(cells, seeds, episodes)
    done = set()
    if os.path.exists(out):
        for line in open(out):
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
    with ctx.Pool(n_workers, initializer=_init_worker) as pool, open(out, 'a') as fh:
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
