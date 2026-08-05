"""Rx-2 -- initialisation-gap decomposition driver (PRO arm).

Pre-registered in tmp/rx2_init_2026-08-05/PREREG.md + PREREG_ADDENDUM_PRO.md.
Cell definitions are frozen by the addendum; do not edit them after the first run.

Protocol inherited VERBATIM from tmp/s1s2_2026-08-05/run_s1s2.py (panel, spectral_accuracy,
resumable jsonl, spawn pool, 1 BLAS thread per worker) so cell X-BASE pairs bit-for-bit with
arm S1-A0 there when run on seeds {42,123,456}.

PRODUCTION DEFAULTS ARE NOT CHANGED. Every factor is applied inside the worker process by
(a) environment gates read at CALL time (initializer._rx2_gate), or
(b) module-attribute monkey-patches that are undone in a finally block.

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

# Seed ledger [1518-1818]; this experiment consumes 1518/1519/1520 (median-of-3).
# The bit-identity batch alone uses the stored {42,123,456} so it can be compared to
# tmp/s1s2_2026-08-05/results.jsonl arm S1-A0 value-for-value.
SEEDS = [int(s) for s in os.environ.get('RX2_SEEDS', '1518,1519,1520').split(',')]

EPISODES = [
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]
PANEL = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
         'Directional_R2', 'Hs_error']
N_GT = {'episode_BF4_15comp_seed0_uniform.h5': 15,
        'episode_BF4_30comp_seed0_uniform.h5': 30,
        'episode_BF5_15comp_seed0_uniform.h5': 15,
        'episode_BF5_30comp_seed0_uniform.h5': 30}

WIDE = (-4.0, 4.0)
NARROW = (-0.8, 0.8)

# WS-P1 deployment operating point == tmp/s1s2_2026-08-05 arm S1-A0. sum(max_iters) == 900.
BASE_CFG = dict(init_method='v2_konly', use_dispersion=True, use_confidence=True,
                adaptive_n=True, tracking_window=3, prune_threshold=0.10, max_n=30,
                n_gaussians=[3, 8, 15], loss_thresholds=[0.1, 0.05, None],
                max_iters_per_stage=[200, 200, 500])

# cell -> dict(env=..., mod=..., krange=..., cfg=...)
#   env    : RX2_* gates, read at call time by initializer._rx2_gate
#   mod    : module attributes to override (undone in finally)
#   krange : coarse beamform (kx,ky) range at ALL THREE placement sites
#   cfg    : run_estimation config overrides on top of BASE_CFG
CELLS = {
    # --- controls -----------------------------------------------------------------
    'X-BASE':  dict(),                       # shipped, PRIMARY control
    # byte-duplicates of X-BASE: one per batch, so every batch carries a worker-leak probe
    'X-BASE2': dict(),
    'X-BASE3': dict(),
    'X-BASE4': dict(),
    # --- axis P: the residual-site greedy (12 of 15 atoms on the deployment ladder) --
    'P1': dict(env={'RX2_GROW_SUBTRACT': '1'}),
    'P2': dict(env={'RX2_GROW_RECOMPUTE': '1'}),
    'P3': dict(env={'RX2_GROW_RECOMPUTE': '1', 'RX2_GROW_TIGHT_SUPPRESS': '1'}),
    # --- axis W: deterministic (w, phi) seeding -------------------------------------
    'W1': dict(mod={'I.KONLY_RANDOM_WPHI': False}),
    'W2': dict(env={'RX2_GROW_FITPHI': '1'}),
    'W3': dict(mod={'I.KONLY_RANDOM_WPHI': False}, env={'RX2_GROW_FITPHI': '1'}),
    # --- axis S: schedule (same final N=15, same 900 total gradient steps) -----------
    'S1': dict(cfg=dict(n_gaussians=[3, 6, 9, 12, 15],
                        loss_thresholds=[0.1, 0.05, 0.05, 0.05, None],
                        max_iters_per_stage=[180, 180, 180, 180, 180])),
    'S2': dict(mod={'C.SCALE_INVARIANT_ADVANCE': True}),
    # --- anchor: best previously-measured placement intervention (E1 cell B) ---------
    'G1': dict(krange=NARROW),
    # --- STACKS (bundles, named as such; interaction reported explicitly) ------------
    'G1P1': dict(krange=NARROW, env={'RX2_GROW_SUBTRACT': '1'}),
    'G1P3': dict(krange=NARROW, env={'RX2_GROW_RECOMPUTE': '1',
                                     'RX2_GROW_TIGHT_SUPPRESS': '1'}),
    # --- upper-bound marker (NOT deployable) ----------------------------------------
    'ORACLE': dict(cfg='__oracle__'),
}
ORACLE_CELLS = {'ORACLE'}
RX2_ENV_KEYS = ('RX2_GROW_SUBTRACT', 'RX2_GROW_RECOMPUTE',
                'RX2_GROW_TIGHT_SUPPRESS', 'RX2_GROW_FITPHI')


def apply_cell(cell):
    """Install the cell's factors. Returns an undo callable. Worker-local."""
    import wavesplat.optimizer.initializer as I
    import wavesplat.optimizer.coarse_to_fine as C
    import wavesplat.optimizer.sequential_update as S
    spec = CELLS[cell]
    mods = {'I': I, 'C': C, 'S': S}

    # 1. environment gates -- always fully rewritten so a previous cell cannot leak
    for k in RX2_ENV_KEYS:
        os.environ[k] = '0'
    for k, v in (spec.get('env') or {}).items():
        os.environ[k] = v

    # 2. module attributes
    saved = {}
    for key, val in (spec.get('mod') or {}).items():
        mod_name, attr = key.split('.')
        m = mods[mod_name]
        saved[key] = getattr(m, attr)
        setattr(m, attr, val)

    # 3. beamform search range at all three placement sites
    orig_bf = I._beamform_coarse
    krange = spec.get('krange')
    if krange is not None:
        def bf(x, y, z, t, depth, kx_range=WIDE, ky_range=WIDE, n_kx=81, n_ky=81):
            # range overridden; grid COUNT stays whatever the call site asked for (81/61).
            return orig_bf(x, y, z, t, depth, krange, krange, n_kx, n_ky)
        I._beamform_coarse = bf
        C._beamform_coarse = bf   # coarse_to_fine imported the symbol at module load

    def undo():
        for k in RX2_ENV_KEYS:
            os.environ[k] = '0'
        for key, val in saved.items():
            mod_name, attr = key.split('.')
            setattr(mods[mod_name], attr, val)
        I._beamform_coarse = orig_bf
        C._beamform_coarse = orig_bf
    return undo


def cell_config(cell, episode):
    cfg = dict(BASE_CFG)
    spec = CELLS[cell].get('cfg')
    if spec == '__oracle__':
        cfg.update(init_method='oracle', n_gaussians=[N_GT[episode]],
                   loss_thresholds=[None], max_iters_per_stage=[900],
                   init_k_lr_scale=0.0)
    elif spec:
        cfg.update(spec)
    assert 'init_method' in cfg          # W9 discipline: never implicit
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
    return result


def build_jobs(cells=None):
    cells = cells or list(CELLS)
    return [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in cells for ep in EPISODES for s in SEEDS]


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
        cfg = cell_config(job['cell'], job['episode'])
        cfg['seed'] = job['seed']
        gt_comp = np.asarray(EpisodeDataLoader(path).wave_components)
        undo = apply_cell(job['cell'])
        _init.ORACLE_COMPONENTS = gt_comp if job['cell'] in ORACLE_CELLS else None
        try:
            r = run_estimation(path, cfg)
        finally:
            _init.ORACLE_COMPONENTS = None

        frames = r['frame_results']
        avg = {k: float(np.nanmean([f[k] for f in frames if k in f])) for k in PANEL}
        vr = np.array([f['Variance_Ratio'] for f in frames if 'Variance_Ratio' in f], float)
        rho = np.array([f['Pearson_rho'] for f in frames if 'Pearson_rho' in f], float)
        # PREREG: |VR-1| is formed PER FRAME and then averaged (never mean-then-transform).
        abs_vr = float(np.nanmean(np.abs(vr - 1.0))) if vr.size else float('nan')
        # frame-wise orthogonal decomposition m = (1-rho^2) + (sqrt(v)-rho)^2
        with np.errstate(invalid='ignore'):
            decorr = 1.0 - rho ** 2
            shrink = (np.sqrt(np.clip(vr, 0, None)) - rho) ** 2
        ru = [f['RMSE_unobserved'] for f in frames
              if 'RMSE_unobserved' in f and np.isfinite(f.get('RMSE_unobserved', np.nan))]
        try:
            spec = spectral_accuracy(r['spectrum_params'], gt_comp)
        except Exception as se:
            spec = {'spec_error': repr(se)}
        sp = r.get('spectrum_params') or {}
        return {**base, **avg,
                'absVR_framewise': abs_vr,
                'decorr_mean': float(np.nanmean(decorr)) if rho.size else float('nan'),
                'shrink_mean': float(np.nanmean(shrink)) if rho.size else float('nan'),
                'm_mean': float(np.nanmean(decorr + shrink)) if rho.size else float('nan'),
                'avg_RMSE_unobserved': float(np.mean(ru)) if ru else float('nan'),
                'elapsed_ms': float(np.mean([f['elapsed_ms'] for f in frames]))
                if frames else float('nan'),
                'N': r.get('n_gaussians', -1),
                'init_nrmse': float(r.get('init_nrmse', float('nan'))),
                'init_rmse': float(r.get('init_rmse', float('nan'))),
                'spec': spec,
                # discipline #2 from tmp/placement_field_debate_2026-08-05/VERDICT.md:
                # dump atom coordinates so the matcher radius can be re-chosen post hoc.
                'kx': [float(v) for v in np.asarray(sp.get('kx', []))],
                'ky': [float(v) for v in np.asarray(sp.get('ky', []))],
                'w': [float(v) for v in np.asarray(sp.get('w', []))],
                'vr_frames': [float(v) for v in vr],
                'rho_frames': [float(v) for v in rho],
                'cfg': {k: v for k, v in cfg.items() if k != 'seed'},
                'wall_s': time.time() - t0}
    except Exception as e:
        import traceback
        return {**base, 'error': repr(e), 'tb': traceback.format_exc()[-1500:],
                'wall_s': time.time() - t0}
    finally:
        if undo is not None:
            undo()


def _init_worker():
    sys.path.insert(0, REPO)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')


def main():
    import multiprocessing as mp
    global OUT
    cells = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    OUT = os.environ.get('RX2_OUT', OUT)
    n_workers = int(os.environ.get('WORKERS', '8'))
    jobs = build_jobs(cells)
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
    print(f'total {len(jobs)} | done {len(done)} | todo {len(todo)} | workers {n_workers} '
          f'| seeds {SEEDS}', flush=True)
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
                tag = f"{tag:.4f} init={rec.get('init_nrmse', float('nan')):.4f}"
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<46s} {tag} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
