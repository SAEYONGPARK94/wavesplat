"""Batch A: finemesh baselines for the carried-over tables (Task 1 + Task 4 baseline arms).

Reproduces scripts/run_12ep_full_analysis.py / run_100ep_letkf_and_ws_metrics.py exactly
(stock ETKFGrid/LETKFGrid/GPRBaseline/GPRDispersion, RegularGridInterpolator, compute_all_metrics
with obs_x/obs_y so RMSE_unobserved is captured) but on data_finemesh_2026-07-23/, single filter
seed 42 (== original protocol). Stock ETKFGrid defaults obs_noise=0.05/max_obs=40 are identical to
the etkf_100ep_panel FastETKF config, so ETKF nRMSE here must reproduce the refreshed 0.332.

Per-method episode sets:
  ETKF, LETKF, GPR-1  -> all 100 episodes (feeds strong_baseline 100-ep + 12-ep subset + fivemetric
                        RMSE_unobs for ETKF/GPR-1; LETKF feeds the 100-ep head-to-head)
  GPR-disp(3f)        -> 12 representative episodes only (strong_baseline 12-ep row)

Resumable: appends JSONL keyed by (method|episode). Full panel + RMSE_unobserved recorded per episode.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'batchA_baselines.jsonl')

TABLE1_EPS = [
    'episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
    'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5',
]

PANEL = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
         'Directional_R2', 'Hs_error', 'RMSE_unobserved']


def all_eps():
    return sorted(f for f in os.listdir(DATA) if f.endswith('.h5'))


def build_jobs():
    eps100 = all_eps()
    jobs = []
    for m in ('ETKF', 'LETKF', 'GPR-1'):
        for ep in eps100:
            jobs.append({'method': m, 'episode': ep, 'job_id': f'{m}|{ep}'})
    for ep in TABLE1_EPS:
        jobs.append({'method': 'GPR-disp', 'episode': ep, 'job_id': f'GPR-disp|{ep}'})
    return jobs


def _grid_run(path, cls, kw):
    import numpy as np
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.evaluation.metrics import compute_all_metrics
    from scipy.interpolate import RegularGridInterpolator
    np.random.seed(42)
    loader = EpisodeDataLoader(path)
    gx = loader.gt_grid_x.cpu().numpy(); gy = loader.gt_grid_y.cpu().numpy()
    egx = np.linspace(gx[0], gx[-1], 40); egy = np.linspace(gy[0], gy[-1], 30)
    b = cls(egx, egy, depth=loader.depth, **kw)
    fm = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy(); y = frame.y_obs.cpu().numpy(); z = frame.z_obs.cpu().numpy()
        el = b.update(x, y, z)
        pred = b.predict_grid()
        interp = RegularGridInterpolator((egy, egx), pred, method='linear',
                                         bounds_error=False, fill_value=0.0)
        Xg, Yg = np.meshgrid(gx, gy)
        pf = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(len(gy), len(gx))
        m = compute_all_metrics(pf, frame.gt_field.cpu().numpy(), gx, gy, loader.Hs_true,
                                obs_x=x if len(x) else None, obs_y=y if len(y) else None)
        m['elapsed_ms'] = el; fm.append(m)
    return _agg(fm)


def _gpr_run(path, cls, kw):
    import numpy as np
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.evaluation.metrics import compute_all_metrics
    np.random.seed(42)
    loader = EpisodeDataLoader(path)
    gx = loader.gt_grid_x.cpu().numpy(); gy = loader.gt_grid_y.cpu().numpy()
    g = cls(**kw); fm = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy(); y = frame.y_obs.cpu().numpy(); z = frame.z_obs.cpu().numpy()
        el = g.update(x, y, z)
        pred = g.predict_grid(gx, gy)
        m = compute_all_metrics(pred, frame.gt_field.cpu().numpy(), gx, gy, loader.Hs_true,
                                obs_x=x if len(x) else None, obs_y=y if len(y) else None)
        m['elapsed_ms'] = el; fm.append(m)
    return _agg(fm)


def _agg(fm):
    import numpy as np
    out = {k: float(np.nanmean([m.get(k, np.nan) for m in fm])) for k in PANEL}
    out['ms'] = float(np.nanmean([m.get('elapsed_ms', np.nan) for m in fm]))
    return out


def run_one(job):
    sys.path.insert(0, REPO)
    from wavesplat.baselines.etkf_grid import ETKFGrid
    from wavesplat.baselines.letkf_grid import LETKFGrid
    from wavesplat.baselines.gpr_baseline import GPRBaseline
    from wavesplat.baselines.gpr_dispersion import GPRDispersion
    path = os.path.join(DATA, job['episode'])
    t0 = time.time()
    try:
        m = job['method']
        if m == 'ETKF':
            r = _grid_run(path, ETKFGrid, dict(n_ensemble=100, mult_inflation=1.0, add_inflation=0.005))
        elif m == 'LETKF':
            r = _grid_run(path, LETKFGrid, dict(n_ensemble=100, mult_inflation=1.0,
                                                add_inflation=0.005, loc_radius=50.0))
        elif m == 'GPR-1':
            r = _gpr_run(path, GPRBaseline, dict(kernel_type='matern', length_scale=20.0,
                                                 noise_sigma=0.02, window_frames=1))
        elif m == 'GPR-disp':
            r = _gpr_run(path, GPRDispersion, dict(length_scale=20.0, noise_sigma=0.02,
                                                   window_frames=3, depth=50.0))
        else:
            raise ValueError(m)
        rec = {'job_id': job['job_id'], 'method': m, 'episode': job['episode'],
               **r, 'wall_s': time.time() - t0}
    except Exception as e:
        rec = {'job_id': job['job_id'], 'method': job['method'], 'episode': job['episode'],
               'error': repr(e), 'wall_s': time.time() - t0}
    return rec


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
            tag = f"{rec.get('nRMSE', rec.get('error', '?'))}"
            if isinstance(rec.get('nRMSE'), float):
                tag = f"{rec['nRMSE']:.4f}"
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<50s} {tag} '
                  f'({rec["wall_s"]:.0f}s) ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
