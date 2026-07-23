"""Task 13: why did better rendering help the standalone greedy+joint estimator by -73..-77%
while the WaveSplat tracker's aggregate did not move (12-episode pilot, t = -0.44)?

Hypothesis H: the gain is real but lands at INITIALIZATION, and the tracking stage discards
it. WaveSplat tracks in `w_phi` mode -- wavenumbers frozen, only amplitude and phase updated
-- with a known amplitude-decay failure mode (Adam momentum against an imperfect phase
prediction). If so, `init_nrmse` should improve on the fine mesh by roughly the amount the
standalone estimator improved, while `avg_nRMSE` (post-tracking) should not.

This is decisive between:
  H  (tracking washes out the gain)  -> init improves, tracked does not
  H' (the two estimators differ for some other reason) -> init does not improve either

Design: the same 12 episodes and 5 seeds as Table 1, run on BOTH meshes, recording the
initialization nRMSE and the tracked nRMSE from the same run so the comparison is paired at
the run level. The fine-mesh half is already in ws_runs.jsonl; only the old-mesh half is new.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(HERE, 'task13_runs.jsonl')

SEEDS = [42, 1, 2, 3, 4]
P1 = {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
      'adaptive_n': True, 'tracking_window': 3}

PAIRS = [('BF2', '1comp'), ('BF3', '1comp'), ('BF3', '2comp'), ('BF3', '5comp'),
         ('BF4', '1comp'), ('BF4', '5comp'), ('BF4', '15comp'), ('BF4', '30comp'),
         ('BF5', '15comp'), ('BF5', '30comp')]
EPISODES = [f'episode_{bf}_{c}_seed0_uniform.h5' for bf, c in PAIRS] + [
    'episode_BF3_2comp_seed0_slope.h5', 'episode_BF3_5comp_seed0_slope.h5']

MESHES = {'oldmesh': os.path.join(REPO, 'data'),
          'finemesh': os.path.join(REPO, 'data_finemesh_2026-07-23')}


def build_jobs():
    return [{'job_id': f'{m}|{ep}|{s}', 'mesh': m, 'episode': ep, 'seed': s}
            for m in MESHES for ep in EPISODES for s in SEEDS]


def _run_one(job):
    sys.path.insert(0, REPO)
    from wavesplat.run import run_estimation
    cfg = dict(P1)
    cfg['seed'] = job['seed']
    p = os.path.join(MESHES[job['mesh']], job['episode'])
    t0 = time.time()
    base = {k: job[k] for k in ('job_id', 'mesh', 'episode', 'seed')}
    if not os.path.exists(p):
        return {**base, 'error': 'missing', 'wall_s': 0.0}
    try:
        r = run_estimation(p, cfg)
        return {**base,
                'init_nrmse': float(r['init_nrmse']),
                'avg_nRMSE': float(r['avg_nRMSE']),
                'n_gaussians': r.get('n_gaussians'),
                'wall_s': time.time() - t0}
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
                  f'init={rec.get("init_nrmse", float("nan")):.4f} '
                  f'trk={rec.get("avg_nRMSE", float("nan")):.4f} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
