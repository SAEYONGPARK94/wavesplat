"""Evaluation: TRACK_PHASE_PREDICT=False (flag ON) on the 12 Table-1 episodes x 5 seeds.

Flag-OFF baseline is reused from tmp/rerun_2026-07-24/task13_runs.jsonl (finemesh P1), verified
bit-identical to ws_runs.jsonl P1. This driver only produces the flag-ON half. P1 config, CPU
(matches baseline device). Per-run we keep avg_nRMSE AND init_nrmse (init is identical to the
baseline per seed because the flag only touches tracking -- lets us verify the pairing and
measure the init->track gap shrink).

Adoption rule frozen in tmp/TRACKER_AMPDECAY_PREREG_2026-07-25.md.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'ampfix_runs.jsonl')

SEEDS = [42, 1, 2, 3, 4]
DEVICE = os.environ.get('WS_DEVICE', 'cpu')
P1 = {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
      'adaptive_n': True, 'tracking_window': 3, 'device': DEVICE}

PAIRS = [('BF2', '1comp'), ('BF3', '1comp'), ('BF3', '2comp'), ('BF3', '5comp'),
         ('BF4', '1comp'), ('BF4', '5comp'), ('BF4', '15comp'), ('BF4', '30comp'),
         ('BF5', '15comp'), ('BF5', '30comp')]
EPISODES = [f'episode_{bf}_{c}_seed0_uniform.h5' for bf, c in PAIRS] + [
    'episode_BF3_2comp_seed0_slope.h5', 'episode_BF3_5comp_seed0_slope.h5']


def build_jobs():
    return [{'job_id': f'{ep}|{s}', 'episode': ep, 'seed': s}
            for ep in EPISODES for s in SEEDS]


def _run_one(job):
    import torch
    torch.set_num_threads(int(os.environ.get('WS_THREADS', '2')))
    sys.path.insert(0, REPO)
    import wavesplat.optimizer.sequential_update as S
    S.TRACK_PHASE_PREDICT = False  # THE FIX (flag ON)
    from wavesplat.run import run_estimation
    cfg = dict(P1); cfg['seed'] = job['seed']
    t0 = time.time()
    try:
        r = run_estimation(os.path.join(DATA, job['episode']), cfg)
        rec = {**{k: job[k] for k in ('job_id', 'episode', 'seed')},
               'flag': 'TRACK_PHASE_PREDICT_False',
               'avg_nRMSE': float(r['avg_nRMSE']),
               'init_nrmse': float(r.get('init_nrmse', float('nan'))),
               'avg_Hs_error': float(r.get('avg_Hs_error', float('nan'))),
               'n_gaussians': r.get('n_gaussians'),
               'wall_s': time.time() - t0}
    except Exception as e:
        rec = {**{k: job[k] for k in ('job_id', 'episode', 'seed')},
               'error': repr(e), 'wall_s': time.time() - t0}
    return rec


def main():
    import multiprocessing as mp
    n_workers = int(os.environ.get('WS_WORKERS', '3'))
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
    print(f'total {len(jobs)} | done {len(done)} | todo {len(todo)} | workers {n_workers}',
          flush=True)
    if not todo:
        print('nothing to do'); return
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(n_workers) as pool, open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(_run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            el = time.time() - t0
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<46s} '
                  f'trk={rec.get("avg_nRMSE", float("nan")):.4f} '
                  f'init={rec.get("init_nrmse", float("nan")):.4f} '
                  f'({rec["wall_s"]:.0f}s) ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
