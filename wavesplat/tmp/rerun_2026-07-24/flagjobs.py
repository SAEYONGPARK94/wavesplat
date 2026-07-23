"""Tasks 9-11: sweep the module-level behaviour flags on the Nyquist-adequate dataset.

Each flag was added with its shipped value as the default so that nothing moved silently.
This driver is where they actually get measured, k=5 seeds per cell, against the same
default arm so every comparison is paired.

Arms
----
default              shipped behaviour (control; must reproduce ws_runs.jsonl)
recompute_residual   task 9. Instead of guessing a peak-separation radius, recompute the
                     beamform power map on the residual each greedy step so the radius stops
                     being load-bearing. Costs one coarse beamform per component.
min_dk_005           task 9 comparison point: the naive "just lower the radius" fix, already
                     known to be a regime-dependent trade. Included so the recompute arm is
                     judged against it and not only against the default.
deterministic        task 10. Remove both random-phase initialisation sites. Known to give
                     CV -> 0 on a sparse episode and to *improve* the crowded one, but to
                     give up the occasional lucky basin -- needs a wider episode set before
                     it can be adopted, which is what this provides.
scale_inv_grow       task 11-adjacent. Normalise the growth trigger by observed signal RMS.
scale_inv_l1         task 11-adjacent. Make the L1 penalty dimensionally consistent.

Read-out rule, fixed BEFORE the numbers (pre-registration):
  an arm counts as an improvement only if the paired per-episode median moves in the same
  direction on >= 4 of the 6 episodes AND the mean move exceeds the mean per-episode seed sd
  of the default arm. Anything else is reported as "not resolvable at this n".
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'flag_runs.jsonl')

SEEDS = [42, 1, 2, 3, 4]

P1 = {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
      'adaptive_n': True, 'tracking_window': 3}

EPISODES = [
    'episode_BF3_2comp_seed0_uniform.h5',    # sparse, highest variance
    'episode_BF3_5comp_seed0_uniform.h5',    # medium -- the regime min_dk=0.05 hurt
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',   # crowded -- the regime min_dk=0.05 helped
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
]

# arm -> {module: {FLAG: value}}
ARMS = {
    'default': {},
    'recompute_residual': {
        'wavesplat.optimizer.initializer': {'PEAK_SELECT_MODE': 'recompute_residual'}},
    'min_dk_005': {
        'wavesplat.optimizer.initializer': {'MIN_DK': 0.05}},
    'deterministic': {
        'wavesplat.optimizer.initializer': {'KONLY_RANDOM_WPHI': False},
        'wavesplat.optimizer.coarse_to_fine': {'GROW_RANDOM_PHI': False}},
    'scale_inv_grow': {
        'wavesplat.optimizer.sequential_update': {'SCALE_INVARIANT_GROW': True}},
    'scale_inv_l1': {
        'wavesplat.optimizer.sequential_update': {'SCALE_INVARIANT_L1': True}},
}


def build_jobs():
    return [{'job_id': f'{arm}|{ep}|{s}', 'arm': arm, 'episode': ep, 'seed': s}
            for arm in ARMS for ep in EPISODES for s in SEEDS]


def _run_one(job):
    import importlib
    sys.path.insert(0, REPO)
    from wavesplat.run import run_estimation

    applied = {}
    for mod_name, flags in ARMS[job['arm']].items():
        mod = importlib.import_module(mod_name)
        for k, v in flags.items():
            applied[(mod_name, k)] = getattr(mod, k)
            setattr(mod, k, v)

    cfg = dict(P1)
    cfg['seed'] = job['seed']
    t0 = time.time()
    try:
        r = run_estimation(os.path.join(DATA, job['episode']), cfg)
        rec = {**{k: job[k] for k in ('job_id', 'arm', 'episode', 'seed')},
               'avg_nRMSE': float(r['avg_nRMSE']),
               'avg_obs_loss': float(r.get('avg_obs_loss', float('nan'))),
               'n_gaussians': r.get('n_gaussians'),
               'wall_s': time.time() - t0}
    except Exception as e:
        rec = {**{k: job[k] for k in ('job_id', 'arm', 'episode', 'seed')},
               'error': repr(e), 'wall_s': time.time() - t0}
    finally:
        for (mod_name, k), v in applied.items():
            setattr(importlib.import_module(mod_name), k, v)
    return rec


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
    print(f'total {len(jobs)} | done {len(done)} | todo {len(todo)} | workers {n_workers}',
          flush=True)
    if not todo:
        return
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(n_workers) as pool, open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(_run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            el = time.time() - t0
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<62s} '
                  f'{rec.get("avg_nRMSE", rec.get("error", "?"))} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
