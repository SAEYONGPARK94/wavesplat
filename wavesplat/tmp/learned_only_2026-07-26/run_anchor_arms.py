"""CANDIDATE 1: pure-learned-WITH-PEAK-ANCHOR tracker arm on the sparse 40 + a 10-episode
crowded no-harm spot check, both frozen anchor variants (0.375, 0.60). Same protocol as the
pure-learned arm (TRACK_PHASE_PREDICT=False, learned_init_iters=500, cached raw atoms, LANES,
median-of-5). Resumable JSONL. Proposals from existing caches (no GPU).

  nice -n 10 python3 tmp/learned_only_2026-07-26/run_anchor_arms.py --workers 4
"""
import os, sys, json, time, re, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool
import numpy as np

FINE = 'data_finemesh_2026-07-23/'
BATCHC = 'tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
VARIANTS = [0.375, 0.60]          # anchor_frac (kp-scaled)
SIGFRAC = 0.1
OUT = 'tmp/learned_only_2026-07-26/anchor_arms.jsonl'
CROWDED_SPOT = ([f'episode_BF4_15comp_seed{s}_uniform.h5' for s in range(5)] +
                [f'episode_BF4_30comp_seed{s}_uniform.h5' for s in range(5)])
CACHE = {}


def comp(ep): return int(re.search(r'_(\d+)comp', ep).group(1))


def _worker(args):
    ep, est_seed, diff_seed, afrac = args
    import wavesplat.optimizer.sequential_update as su
    su.TRACK_PHASE_PREDICT = False
    from wavesplat.run import run_estimation
    t0 = time.time()
    try:
        entry = CACHE.get((ep, diff_seed))
        if entry is None:
            raise RuntimeError(f'no cached atoms {ep} d{diff_seed}')
        cfg = {'adaptive_n': True, 'seed': est_seed, 'device': 'cpu',
               'init_method': 'learned', 'learned_diffusion_seed': diff_seed,
               'learned_init_iters': 500, 'learned_atoms': entry,
               'learned_peak_anchor': True, 'learned_anchor_frac': afrac,
               'learned_anchor_sigfrac': SIGFRAC}
        r = run_estimation(FINE + ep, config=cfg)
        return dict(episode=ep, comp=comp(ep), est_seed=est_seed, diff_seed=diff_seed,
                    anchor_frac=afrac, avg_nRMSE=float(r['avg_nRMSE']),
                    final_n=int(r['n_gaussians']), init_nrmse=float(r['init_nrmse']),
                    sec=time.time() - t0, err=None)
    except Exception as e:
        return dict(episode=ep, comp=comp(ep), est_seed=est_seed, diff_seed=diff_seed,
                    anchor_frac=afrac, avg_nRMSE=None, final_n=None, init_nrmse=None,
                    sec=time.time() - t0, err=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    global CACHE
    for src in ('tmp/wp_integration_2026-07-25/atoms_cache.jsonl',
                'tmp/unified_2026-07-26/atoms_cache_extra.jsonl'):
        for l in open(src):
            d = json.loads(l)
            if d.get('kx') is not None and d.get('err') is None:
                CACHE[(d['episode'], d['diff_seed'])] = dict(kx=d['kx'], ky=d['ky'], energy=d['energy'])
    eps_all = sorted(set(json.loads(l)['episode'] for l in open(BATCHC)))
    sparse = [e for e in eps_all if comp(e) in (1, 2)]
    assert len(sparse) == 40, len(sparse)
    targets = sparse + CROWDED_SPOT
    jobs = [(ep, es, ds, af) for ep in targets for (es, ds) in LANES for af in VARIANTS]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                d = json.loads(l)
                if d.get('err') is None:
                    done.add((d['episode'], d['est_seed'], d['diff_seed'], d['anchor_frac']))
            except Exception:
                pass
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in done]
    print(f"sparse={len(sparse)} crowded_spot={len(CROWDED_SPOT)} variants={VARIANTS}; "
          f"{len(jobs)} runs to do ({len(done)} done), {a.workers} workers", flush=True)
    t0 = time.time(); n = 0
    with open(OUT, 'a', buffering=1) as f, Pool(a.workers) as pool:
        for res in pool.imap_unordered(_worker, jobs):
            f.write(json.dumps(res) + '\n'); n += 1
            tag = ('ERR ' + res['err'][:40]) if res['err'] else f"avg {res['avg_nRMSE']:.3f} N{res['final_n']}"
            print(f"[{n:3d}/{len(jobs)}] {res['episode']:34s} e{res['est_seed']}d{res['diff_seed']} "
                  f"af{res['anchor_frac']} {tag} ({res['sec']:.0f}s)", flush=True)
    print(f"done: {n} runs, {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
