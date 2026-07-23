"""Task 3 — re-sweep the four production hyperparameters on the Nyquist-adequate dataset.

WHY: `lr`, `lambda_l1`, `prune_threshold` (tau_prune) and `grow_residual_threshold` (tau_grow)
were all tuned on BF3_5comp, on the old coarse render. Two problems with that:
  (a) the render was sub-Nyquist, so the tuning objective was partly measuring a rendering
      artifact;
  (b) BF3_5comp is a single regime, so the paper's "low sensitivity across BF2-5" argument was
      only ever verified inside it.
This is a coordinate sweep (one knob at a time around the shipped defaults), k=5 seeds per
cell, on four episodes chosen to span sparse / the original tuning point / crowded / worst.
Reporting is median +- sd per the binding convention; a knob only counts as mattering if the
move is larger than the seed spread it has to clear.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'sweep_runs.jsonl')

SEEDS = [42, 1, 2, 3, 4]

BASE = {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
        'adaptive_n': True, 'tracking_window': 3}

EPISODES = [
    ('sparse',   'episode_BF3_2comp_seed0_uniform.h5'),
    ('tuned_on', 'episode_BF3_5comp_seed0_uniform.h5'),
    ('crowded',  'episode_BF4_15comp_seed0_uniform.h5'),
    ('worst',    'episode_BF5_30comp_seed0_uniform.h5'),
]

# (knob, default, alternatives)
GRID = [
    ('lr',                       0.01,   [0.003, 0.03]),
    ('lambda_l1',                0.001,  [0.0001, 0.01]),
    ('prune_threshold',          0.10,   [0.05, 0.20]),
    ('grow_residual_threshold',  0.05,   [0.02, 0.10]),
]


def build_jobs():
    jobs = []
    seen = set()

    def add(knob, val, tag, ep_tag, ep):
        for s in SEEDS:
            jid = f'{ep}|{knob}={val}|{s}'
            if jid in seen:
                continue
            seen.add(jid)
            cfg = dict(BASE)
            if knob != 'default':
                cfg[knob] = val
            jobs.append({'job_id': jid, 'episode': ep, 'ep_tag': ep_tag,
                         'knob': knob, 'value': val, 'is_default': tag == 'default',
                         'config': cfg, 'seed': s})

    for ep_tag, ep in EPISODES:
        add('default', None, 'default', ep_tag, ep)
        for knob, dflt, alts in GRID:
            for v in alts:
                add(knob, v, 'alt', ep_tag, ep)
    return jobs


def _run_one(job):
    sys.path.insert(0, REPO)
    from wavesplat.run import run_estimation
    cfg = dict(job['config'])
    cfg['seed'] = job['seed']
    t0 = time.time()
    try:
        r = run_estimation(os.path.join(DATA, job['episode']), cfg)
        rec = {k: job[k] for k in ('job_id', 'episode', 'ep_tag', 'knob', 'value',
                                   'is_default', 'seed')}
        rec.update({'avg_nRMSE': float(r['avg_nRMSE']),
                    'avg_obs_loss': float(r.get('avg_obs_loss', float('nan'))),
                    'n_gaussians': r.get('n_gaussians'),
                    'wall_s': time.time() - t0})
    except Exception as e:
        rec = {k: job[k] for k in ('job_id', 'episode', 'ep_tag', 'knob', 'value',
                                   'is_default', 'seed')}
        rec.update({'error': repr(e), 'wall_s': time.time() - t0})
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
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'error' not in rec:
                    done.add(rec['job_id'])
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
