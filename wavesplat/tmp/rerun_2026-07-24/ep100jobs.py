"""The headline: WaveSplat 6-metric evaluation over all 100 regenerated episodes, k=5 seeds.

This is what feeds the paper's abstract numbers (avg nRMSE 0.223 vs ETKF 0.319 over 100
episodes) and the five-metric table. Both inputs to those numbers have changed:
  * the dataset is now Nyquist-adequate (data_finemesh_2026-07-23/), and
  * per-episode numbers now carry k=5 seeds, per the binding reporting convention.
The aggregate is where the seed noise averages down by ~sqrt(100), so the headline is the
one number that should be *stable* -- which is exactly why it is worth measuring rather than
assuming.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'ep100_runs.jsonl')

SEEDS = [42, 1, 2, 3, 4]
P1 = {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
      'adaptive_n': True, 'tracking_window': 3}

METRIC_KEYS = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
               'Directional_R2', 'Hs_error']


def episodes():
    return sorted(f for f in os.listdir(DATA) if f.endswith('.h5'))


def build_jobs():
    return [{'job_id': f'{ep}|{s}', 'episode': ep, 'seed': s}
            for ep in episodes() for s in SEEDS]


def _run_one(job):
    sys.path.insert(0, REPO)
    from wavesplat.run import run_estimation
    cfg = dict(P1)
    cfg['seed'] = job['seed']
    base = {k: job[k] for k in ('job_id', 'episode', 'seed')}
    t0 = time.time()
    try:
        r = run_estimation(os.path.join(DATA, job['episode']), cfg)
        frames = r['frame_results']
        avg = {}
        for k in METRIC_KEYS:
            vals = [f[k] for f in frames
                    if k in f and np.isfinite(f.get(k, np.nan))]
            avg[k] = float(np.mean(vals)) if vals else float('nan')
        return {**base, **avg,
                'elapsed_ms': float(np.mean([f['elapsed_ms'] for f in frames]))
                if frames else float('nan'),
                'N': r.get('n_gaussians', -1),
                'init_nrmse': float(r.get('init_nrmse', float('nan'))),
                'wall_s': time.time() - t0}
    except Exception as e:
        return {**base, 'error': repr(e), 'wall_s': time.time() - t0}


def main():
    import multiprocessing as mp
    n_workers = int(os.environ.get('WS_WORKERS', '5'))
    jobs = build_jobs()
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'error' not in r:
                    done.add(r['job_id'])
    todo = [j for j in jobs if j['job_id'] not in done]
    print(f'episodes {len(episodes())} | total {len(jobs)} | done {len(done)} '
          f'| todo {len(todo)} | workers {n_workers}', flush=True)
    if not todo:
        return
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(n_workers) as pool, open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(_run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            el = time.time() - t0
            if i % 10 == 0 or i == len(todo):
                print(f'[{i}/{len(todo)}] {rec["job_id"]:<50s} '
                      f'{rec.get("nRMSE", rec.get("error", "?"))} '
                      f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
