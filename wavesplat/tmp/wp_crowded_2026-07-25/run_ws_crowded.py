"""WS arm — phase-fixed WaveSplat on the crowded (+sparse control) finemesh episodes.
Per PREREG.md: median-of-5 estimator seeds (42,1,2,3,4); TRACK_PHASE_PREDICT=False set in each
worker (shipped default untouched); records init_nrmse and avg_nRMSE. 4 workers, threads=1.
Writes JSONL (one row per (episode, seed)) so a crash resumes without loss.
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool

FINE = 'data_finemesh_2026-07-23/'
SEEDS = (42, 1, 2, 3, 4)

CROWDED = [f'episode_BF{bf}_{nc}comp_seed{s}_{arm}.h5'
           for bf in (4, 5) for nc in (15, 30)
           for s in range(5) for arm in ('uniform', 'slope')]

SPARSE = [
    'episode_BF2_1comp_seed0_uniform.h5', 'episode_BF2_1comp_seed0_slope.h5',
    'episode_BF3_1comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_slope.h5',  'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed1_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_slope.h5',  'episode_BF4_5comp_seed1_uniform.h5',
]


def _worker(args):
    ep, seed = args
    import wavesplat.optimizer.sequential_update as su
    su.TRACK_PHASE_PREDICT = False          # phase-fix; default (True) untouched
    from wavesplat.run import run_estimation
    t0 = time.time()
    try:
        r = run_estimation(FINE + ep, config={'adaptive_n': True, 'seed': seed,
                                              'device': 'cpu'})
        return dict(episode=ep, seed=seed, init_nrmse=float(r['init_nrmse']),
                    avg_nRMSE=float(r['avg_nRMSE']), sec=time.time() - t0,
                    flag=su.TRACK_PHASE_PREDICT, err=None)
    except Exception as e:
        return dict(episode=ep, seed=seed, init_nrmse=None, avg_nRMSE=None,
                    sec=time.time() - t0, flag=su.TRACK_PHASE_PREDICT, err=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=['crowded', 'sparse', 'both'], default='both')
    ap.add_argument('--out', default='tmp/wp_crowded_2026-07-25/ws_runs.jsonl')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    eps = []
    if a.set in ('crowded', 'both'):
        eps += CROWDED
    if a.set in ('sparse', 'both'):
        eps += SPARSE
    jobs = [(ep, s) for ep in eps for s in SEEDS]

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                d = json.loads(line)
                if d.get('err') is None:
                    done.add((d['episode'], d['seed']))
            except Exception:
                pass
    jobs = [j for j in jobs if j not in done]
    print(f"WS arm: {len(eps)} episodes, {len(jobs)} (episode,seed) jobs to run "
          f"({len(done)} already done), {a.workers} workers", flush=True)

    t0 = time.time()
    n = 0
    with open(a.out, 'a', buffering=1) as f, Pool(a.workers) as pool:
        for res in pool.imap_unordered(_worker, jobs):
            f.write(json.dumps(res) + '\n')
            n += 1
            tag = 'ERR' if res['err'] else f"init {res['init_nrmse']:.3f} avg {res['avg_nRMSE']:.3f}"
            print(f"[{n:3d}/{len(jobs)}] {res['episode']:40s} s{res['seed']:<2d} "
                  f"{tag} ({res['sec']:.0f}s)", flush=True)
    print(f"WS arm done: {n} runs, {time.time()-t0:.0f}s wall", flush=True)


if __name__ == '__main__':
    main()
