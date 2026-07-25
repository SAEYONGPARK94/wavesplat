"""Batch A2: GPR-sliding (window=3, noise_sigma=0.05) on all 100 finemesh episodes.

This is the GPR arm of the paired-Wilcoxon table (scripts/run_multiseed_extended.py::run_gpr,
window=3, noise_sigma=0.05, single filter seed 42). ep100/gpr_100ep do NOT cover this config
(gpr_100ep is window=1, noise=0.02), so it is a genuine new run. nRMSE + panel recorded.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'batchA2_gpr3.jsonl')
PANEL = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score', 'Hs_error']


def run_one(ep):
    sys.path.insert(0, REPO)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import numpy as np
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.baselines.gpr_baseline import GPRBaseline
    from wavesplat.evaluation.metrics import compute_all_metrics
    path = os.path.join(DATA, ep)
    t0 = time.time()
    try:
        np.random.seed(42)
        loader = EpisodeDataLoader(path)
        gx = loader.gt_grid_x.cpu().numpy(); gy = loader.gt_grid_y.cpu().numpy()
        gpr = GPRBaseline(kernel_type='matern', length_scale=20.0, noise_sigma=0.05, window_frames=3)
        fm = []
        for frame in loader:
            x = frame.x_obs.cpu().numpy(); y = frame.y_obs.cpu().numpy(); z = frame.z_obs.cpu().numpy()
            ms = gpr.update(x, y, z)
            pred = gpr.predict_grid(gx, gy)
            m = compute_all_metrics(pred, frame.gt_field.cpu().numpy(), gx, gy, loader.Hs_true,
                                    obs_x=x if len(x) else None, obs_y=y if len(y) else None)
            m['elapsed_ms'] = ms; fm.append(m)
        agg = {k: float(np.nanmean([m.get(k, np.nan) for m in fm])) for k in PANEL}
        agg['ms'] = float(np.nanmean([m.get('elapsed_ms', np.nan) for m in fm]))
        return {'episode': ep, **agg, 'wall_s': time.time() - t0}
    except Exception as e:
        return {'episode': ep, 'error': repr(e), 'wall_s': time.time() - t0}


def _init():
    sys.path.insert(0, REPO)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


def main():
    import multiprocessing as mp
    n_workers = int(os.environ.get('WORKERS', '4'))
    eps = sorted(f for f in os.listdir(DATA) if f.endswith('.h5'))
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add(r['episode'])
    todo = [e for e in eps if e not in done]
    print(f'total {len(eps)} | done {len(done)} | todo {len(todo)} | workers {n_workers}', flush=True)
    if not todo:
        print('nothing to do'); return
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(n_workers, initializer=_init) as pool, open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            el = time.time() - t0
            if i % 10 == 0 or i == len(todo):
                print(f'[{i}/{len(todo)}] {rec["episode"]:<45s} '
                      f'{rec.get("nRMSE", rec.get("error", "?"))} ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
