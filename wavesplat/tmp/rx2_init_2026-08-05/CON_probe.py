"""CON diagnostic probes for Rx-2 (attack lines 2, 4, 5).

NOT the main Rx-2 sweep (PRO owns that). These are diagnostic cells only:

  attack 4 -- is the oracle `init_nrmse=0.0629` confounded with `init_k_lr_scale=0`?
              -> Q-ORC (oracle, k trainable) vs Q-ORCFRZ (oracle, k frozen)
              -> and the lever OUTSIDE the search space: Q-DEPFRZ = the *shipped* ladder
                 with k frozen at the beamform estimate. Nobody has ever run this.
  attack 2 -- can the OBSERVATIONS tell the two placements apart? Records the
              observation-space residual (in-sample and one-step-ahead held-out),
              which is the only signal a deployed initializer could ever select on.
              If deployable arms fit the data as well as the oracle, the gap is not a
              search failure -- it is unidentifiable from this sensor.
  attack 5 -- regime dependence: same contrast at n_comp = 1, 5 (BF4 + BF5).

Seeds 42/123/456 are REUSED ON PURPOSE: they are the seeds of tmp/s1s2_2026-08-05 and
tmp/rx1_tracker_2026-08-05, so replicate cells are bit-identity checks and every contrast
is paired. The reserve ledger [1518-1818] is NOT consumed.

`init_method` is explicit in every cell (W9 discipline).
Timing is recorded but NOT citable (box clock under separate re-measurement).
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'CON_probe.jsonl')
# HAZARD 2026-08-06 00:02 -- another agent was editing
# wavesplat/optimizer/{coarse_to_fine,initializer,sequential_update}.py WHILE this probe
# was starting, and the first launch crashed on a half-written `_beamform_residuals`.
# Every number below is therefore produced against a frozen copy of the package at git
# HEAD (53552f5), placed ahead of the live tree on sys.path.
SNAP = os.path.join(HERE, 'snap')

SEEDS = [42, 123, 456]

CROWDED = [
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]
# attack 5: same sea states, only n_comp varies
REGIME = [
    'episode_BF4_1comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF5_1comp_seed0_uniform.h5',
    'episode_BF5_5comp_seed0_uniform.h5',
]

PANEL = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
         'Directional_R2', 'Hs_error']

ORACLE_CELLS = {'Q-ORC', 'Q-ORCFRZ', 'R-ORCFRZ'}


def cell_config(cell, n_gt):
    cfg = {'use_dispersion': True, 'use_confidence': True,
           'adaptive_n': True, 'tracking_window': 3,
           'prune_threshold': 0.10, 'max_n': 30}
    if cell in ('Q-DEP', 'R-DEP'):
        # replicate of S1-A0 / the shipped ladder
        cfg.update(init_method='v2_konly', n_gaussians=[3, 8, 15],
                   loss_thresholds=[0.1, 0.05, None],
                   max_iters_per_stage=[200, 200, 500])
    elif cell in ('Q-DEPFRZ', 'R-DEPFRZ'):
        # NEW: shipped ladder, k frozen at the beamform estimate
        cfg.update(init_method='v2_konly', n_gaussians=[3, 8, 15],
                   loss_thresholds=[0.1, 0.05, None],
                   max_iters_per_stage=[200, 200, 500],
                   init_k_lr_scale=0.0)
    elif cell == 'Q-ORC':      # replicate of S2-T1
        cfg.update(init_method='oracle', n_gaussians=[n_gt],
                   loss_thresholds=[None], max_iters_per_stage=[900])
    elif cell in ('Q-ORCFRZ', 'R-ORCFRZ'):   # replicate of S2-T1f
        cfg.update(init_method='oracle', n_gaussians=[n_gt],
                   loss_thresholds=[None], max_iters_per_stage=[900],
                   init_k_lr_scale=0.0)
    else:
        raise ValueError(cell)
    return cfg


def build_jobs():
    jobs = []
    for cell in ('Q-DEP', 'Q-DEPFRZ', 'Q-ORC', 'Q-ORCFRZ'):
        for ep in CROWDED:
            for s in SEEDS:
                jobs.append({'job_id': f'{cell}|{ep}|{s}', 'cell': cell,
                             'episode': ep, 'seed': s})
    for cell in ('R-DEP', 'R-DEPFRZ', 'R-ORCFRZ'):
        for ep in REGIME:
            for s in SEEDS:
                jobs.append({'job_id': f'{cell}|{ep}|{s}', 'cell': cell,
                             'episode': ep, 'seed': s})
    return jobs


def _obs_resid(model, frame):
    """RMS observation-space residual of the current model on one frame [m]."""
    import torch
    if len(frame.x_obs) == 0:
        return float('nan')
    with torch.no_grad():
        pred = model(frame.x_obs, frame.y_obs, frame.t)
        r = (pred - frame.z_obs)
        return float(torch.sqrt((r ** 2).mean()))


def run_one(job):
    sys.path.insert(0, REPO)
    sys.path.insert(0, SNAP)          # frozen HEAD package wins over the live tree
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')
    import torch
    from wavesplat.run import DEFAULT_CONFIG
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.evaluation.metrics import compute_all_metrics, nrmse
    import wavesplat.optimizer.initializer as _init

    base = {k: job[k] for k in ('job_id', 'cell', 'episode', 'seed')}
    t0 = time.time()
    path = os.path.join(DATA, job['episode'])
    try:
        loader = EpisodeDataLoader(path)
        gt_comp = np.asarray(loader.wave_components)
        n_gt = int(gt_comp.shape[0])

        cfg = DEFAULT_CONFIG.copy()
        cfg.update(cell_config(job['cell'], n_gt))
        cfg['seed'] = job['seed']
        cfg['depth'] = loader.depth

        torch.manual_seed(job['seed'])
        np.random.seed(job['seed'])

        if job['cell'] in ORACLE_CELLS:
            _init.ORACLE_COMPONENTS = gt_comp
        else:
            _init.ORACLE_COMPONENTS = None

        try:
            init_frames = loader.get_init_window(tp_factor=2.0)
            updater = make_updater(cfg)
            init_result = updater.initialize(
                init_frames,
                x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
                y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
                verbose=False)

            grid_x, grid_y = loader.gt_grid_x, loader.gt_grid_y
            last_init = init_frames[-1]
            pred_init = updater.predict_field(grid_x, grid_y, last_init.t)
            init_nrmse = float(nrmse(pred_init.cpu().numpy(),
                                     last_init.gt_field.cpu().numpy(),
                                     loader.Hs_true))
            # observation-space fit of the INITIALISED model, in-sample (init window)
            init_obs_in = float(np.mean([_obs_resid(updater.model, f)
                                         for f in init_frames]))

            track = loader.get_tracking_frames(tp_factor=2.0)
            # held-out in time: the initialised model has never seen tracking frame 0
            init_obs_heldout = _obs_resid(updater.model, track[0]) if track else float('nan')

            frames_out, obs_in, obs_out = [], [], []
            for i, frame in enumerate(track):
                # ONE-STEP-AHEAD held-out residual, BEFORE this frame is used
                obs_out.append(_obs_resid(updater.model, frame))
                updater.update(frame)
                obs_in.append(_obs_resid(updater.model, frame))
                pred = updater.predict_field(grid_x, grid_y, frame.t)
                m = compute_all_metrics(
                    pred.cpu().numpy(), frame.gt_field.cpu().numpy(),
                    grid_x.cpu().numpy(), grid_y.cpu().numpy(), loader.Hs_true,
                    obs_x=frame.x_obs.cpu().numpy() if len(frame.x_obs) else None,
                    obs_y=frame.y_obs.cpu().numpy() if len(frame.y_obs) else None)
                frames_out.append(m)
        finally:
            _init.ORACLE_COMPONENTS = None

        avg = {k: float(np.nanmean([f[k] for f in frames_out if k in f]))
               for k in PANEL}
        ru = [f['RMSE_unobserved'] for f in frames_out
              if np.isfinite(f.get('RMSE_unobserved', np.nan))]
        sp = updater.model.get_spectrum_params()
        return {**base, **avg,
                'avg_RMSE_unobserved': float(np.mean(ru)) if ru else float('nan'),
                'N': int(updater.model.n_gaussians),
                'n_gt': n_gt,
                'Hs_true': float(loader.Hs_true),
                'init_nrmse': init_nrmse,
                'init_obs_rms_insample': init_obs_in,
                'init_obs_rms_heldout1': float(init_obs_heldout),
                'track_obs_rms_insample': float(np.nanmean(obs_in)),
                'track_obs_rms_heldout': float(np.nanmean(obs_out)),
                'init_iters': init_result.get('n_iters'),
                # dump atoms so any later re-analysis can redo matching (debate-J discipline 2)
                'kx': [float(v) for v in sp['kx']],
                'ky': [float(v) for v in sp['ky']],
                'w': [float(v) for v in sp['w']],
                'gt_kx': gt_comp[:, 2].tolist(), 'gt_ky': gt_comp[:, 3].tolist(),
                'gt_A': gt_comp[:, 0].tolist(),
                'cfg': {k: v for k, v in cfg.items() if k != 'seed'},
                'wall_s': time.time() - t0}
    except Exception as e:
        import traceback
        return {**base, 'error': repr(e), 'tb': traceback.format_exc()[-1500:],
                'wall_s': time.time() - t0}


def _init_worker():
    sys.path.insert(0, REPO)
    sys.path.insert(0, SNAP)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')


def main():
    import multiprocessing as mp
    n_workers = int(os.environ.get('WORKERS', '8'))
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
    print(f'total {len(jobs)} | done {len(done)} | todo {len(todo)} | '
          f'workers {n_workers}', flush=True)
    if not todo:
        print('nothing to do'); return
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(n_workers, initializer=_init_worker) as pool, open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            tag = rec.get('nRMSE', rec.get('error', '?'))
            if isinstance(tag, float):
                tag = f'{tag:.4f}'
            el = time.time() - t0
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<52s} {tag} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
