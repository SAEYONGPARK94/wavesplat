"""Job list + parallel runner for the 2026-07-24 full paper-table re-run.

WHY: every per-episode number in the paper is currently (a) computed on the pre-Nyquist
coarse mesh and (b) a single-seed point estimate from a bimodal distribution (CV up to 53%,
docs/REPORTING_CONVENTION_median_of_k_2026-07-23.md). This driver re-runs every WaveSplat
number the paper tables depend on, on `data_finemesh_2026-07-23/`, with k=5 seeds.

Design notes:
  * Jobs are de-duplicated across tables by (episode, config, seed) -- the raydrop and GPR
    tables share their WaveSplat arm with Table 1, so we pay for each run once.
  * Results are appended to a JSONL as they finish, so the run is resumable and a crash
    costs at most the in-flight jobs.
  * `spawn` start method: CUDA cannot be forked. Each worker imports torch once and then
    handles many jobs.
"""
import json
import os
import sys
import time
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'ws_runs.jsonl')

SEEDS = [42, 1, 2, 3, 4]

# ---------------------------------------------------------------- configurations
P0 = {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
      'adaptive_n': False, 'tracking_window': 1}
P1 = {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
      'adaptive_n': True, 'tracking_window': 3}

# Ablation configs are kept bit-identical to scripts/run_ablation_seeded.py (init_method
# 'v1'/'v2', not the production 'v2_konly') so the new numbers stay comparable to the old
# Table 2. The mismatch with production init is pre-existing and is flagged in the report.
ABL = {
    'Full':      {'use_dispersion': True,  'use_confidence': True,  'init_method': 'v1'},
    'A1_noDisp': {'use_dispersion': False, 'use_confidence': True,  'init_method': 'v1'},
    'A2_noConf': {'use_dispersion': True,  'use_confidence': False, 'init_method': 'v1'},
    'v2+disp':   {'use_dispersion': True,  'use_confidence': True,  'init_method': 'v2'},
}

# ---------------------------------------------------------------- episode sets
PAIRS = [('BF2', '1comp'), ('BF3', '1comp'), ('BF3', '2comp'), ('BF3', '5comp'),
         ('BF4', '1comp'), ('BF4', '5comp'), ('BF4', '15comp'), ('BF4', '30comp'),
         ('BF5', '15comp'), ('BF5', '30comp')]

TABLE1_EPS = [f'episode_{bf}_{c}_seed0_uniform.h5' for bf, c in PAIRS] + [
    'episode_BF3_2comp_seed0_slope.h5', 'episode_BF3_5comp_seed0_slope.h5']

RAYDROP_EPS = [f'episode_{bf}_{c}_seed0_{cond}.h5'
               for bf, c in PAIRS for cond in ('uniform', 'slope')]

ABL_EPS = [f'episode_BF3_{c}_seed0_{cond}.h5'
           for c in ('1comp', '2comp', '5comp') for cond in ('uniform', 'slope')]


def cfg_key(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def build_jobs():
    jobs = {}

    def add(ep, cfg_name, cfg, tables):
        for s in SEEDS:
            key = (ep, cfg_key(cfg), s)
            if key in jobs:
                jobs[key]['tables'] |= set(tables)
                continue
            jobs[key] = {'episode': ep, 'cfg_name': cfg_name, 'config': dict(cfg),
                         'seed': s, 'tables': set(tables)}

    for ep in TABLE1_EPS:
        add(ep, 'P0', P0, ['table1'])
        add(ep, 'P1', P1, ['table1', 'table3'])
    for ep in RAYDROP_EPS:
        add(ep, 'P1', P1, ['table4'])
    for ep in [f'episode_{bf}_{c}_seed0_uniform.h5' for bf, c in PAIRS]:
        add(ep, 'P1', P1, ['gpr'])
    for ep in ABL_EPS:
        for name, cfg in ABL.items():
            add(ep, name, cfg, ['table2'])

    out = []
    for (ep, ck, s), j in jobs.items():
        j['job_id'] = f'{ep}|{j["cfg_name"]}|{s}'
        j['tables'] = sorted(j['tables'])
        out.append(j)
    out.sort(key=lambda j: j['job_id'])
    return out


# ---------------------------------------------------------------- worker
def _init_worker():
    sys.path.insert(0, REPO)


def _run_one(job):
    sys.path.insert(0, REPO)
    from wavesplat.run import run_estimation
    path = os.path.join(DATA, job['episode'])
    cfg = dict(job['config'])
    cfg['seed'] = job['seed']
    t0 = time.time()
    try:
        r = run_estimation(path, cfg)
        rec = {'job_id': job['job_id'], 'episode': job['episode'],
               'cfg_name': job['cfg_name'], 'seed': job['seed'],
               'tables': job['tables'],
               'avg_nRMSE': float(r['avg_nRMSE']),
               'avg_RMSE_z': float(r['avg_RMSE_z']),
               'avg_Hs_error': float(r['avg_Hs_error']),
               'avg_inference_ms': float(r['avg_inference_ms']),
               'avg_obs_loss': float(r.get('avg_obs_loss', float('nan'))),
               'init_nrmse': float(r.get('init_nrmse', float('nan'))),
               'n_gaussians': r.get('n_gaussians'),
               'T_converge': r.get('T_converge'),
               'wall_s': time.time() - t0}
    except Exception as e:  # a failed job must not take the batch down
        rec = {'job_id': job['job_id'], 'episode': job['episode'],
               'cfg_name': job['cfg_name'], 'seed': job['seed'],
               'tables': job['tables'], 'error': repr(e),
               'wall_s': time.time() - t0}
    return rec


def main():
    import multiprocessing as mp

    n_workers = int(os.environ.get('WS_WORKERS', '3'))
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

    print(f'total jobs {len(jobs)} | already done {len(done)} | to run {len(todo)} '
          f'| workers {n_workers}', flush=True)
    if not todo:
        print('nothing to do')
        return

    ctx = mp.get_context('spawn')
    t0 = time.time()
    n = 0
    with ctx.Pool(n_workers, initializer=_init_worker) as pool, open(OUT, 'a') as fh:
        for rec in pool.imap_unordered(_run_one, todo, chunksize=1):
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            n += 1
            el = time.time() - t0
            eta = el / n * (len(todo) - n)
            val = rec.get('avg_nRMSE')
            tag = f'{val:.4f}' if val is not None else rec.get('error', '?')[:40]
            print(f'[{n}/{len(todo)}] {rec["job_id"]:<55s} {tag} '
                  f'({rec["wall_s"]:.0f}s) ETA {eta/60:.0f}min', flush=True)
    print(f'DONE in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
