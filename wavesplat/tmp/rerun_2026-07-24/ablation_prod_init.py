"""Follow-up to the ablation re-run: does the A1/A2 null survive the PRODUCTION initializer?

The ablation inherited `init_method='v1'` from the original study, but production is
'v2_konly', and the re-run showed the initializer change to be ~12x larger than either
ablated component (t = -10.2 vs |t| < 0.3). A null measured in a configuration whose
dominant factor is set differently from production does not transfer for free.

Same 6 BF3 episodes, same 5 seeds, only `init_method` changed to the production value.
Pre-registered read-out, fixed before the numbers: A2 is confirmed null under production
init if the paired mean |delta| is below the default arm's mean per-episode seed sd AND the
sign split is within 10/20 of even. Anything else re-opens it.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'ablation_prod_runs.jsonl')

SEEDS = [42, 1, 2, 3, 4]
EPISODES = [f'episode_BF3_{c}_seed0_{cond}.h5'
            for c in ('1comp', '2comp', '5comp') for cond in ('uniform', 'slope')]

# Production initializer, adaptive N and multi-frame tracking as shipped.
BASE = {'init_method': 'v2_konly', 'adaptive_n': True, 'tracking_window': 3}
ARMS = {
    'prod_Full':      {'use_dispersion': True,  'use_confidence': True},
    'prod_A1_noDisp': {'use_dispersion': False, 'use_confidence': True},
    'prod_A2_noConf': {'use_dispersion': True,  'use_confidence': False},
}


def build_jobs():
    return [{'job_id': f'{a}|{ep}|{s}', 'arm': a, 'episode': ep, 'seed': s}
            for a in ARMS for ep in EPISODES for s in SEEDS]


def _run_one(job):
    sys.path.insert(0, REPO)
    from wavesplat.run import run_estimation
    cfg = {**BASE, **ARMS[job['arm']], 'seed': job['seed']}
    base = {k: job[k] for k in ('job_id', 'arm', 'episode', 'seed')}
    t0 = time.time()
    try:
        r = run_estimation(os.path.join(DATA, job['episode']), cfg)
        return {**base, 'avg_nRMSE': float(r['avg_nRMSE']),
                'n_gaussians': r.get('n_gaussians'), 'wall_s': time.time() - t0}
    except Exception as e:
        return {**base, 'error': repr(e), 'wall_s': time.time() - t0}


def main():
    import multiprocessing as mp
    n_workers = int(os.environ.get('WS_WORKERS', '4'))
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
    print(f'total {len(jobs)} | done {len(done)} | todo {len(todo)}', flush=True)
    if not todo:
        return
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(n_workers) as pool, open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(_run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            el = time.time() - t0
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<58s} '
                  f'{rec.get("avg_nRMSE", rec.get("error", "?"))} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
