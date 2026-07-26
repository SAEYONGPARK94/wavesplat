"""Analytic (WaveSplat) arm on the three discreteness cells.

Verbatim protocol from tmp/wp_crowded_2026-07-25/run_ws_crowded.py: median-of-5 estimator seeds
(42,1,2,3,4), TRACK_PHASE_PREDICT=False set per worker (shipped default untouched), adaptive_n,
device cpu; records init_nrmse and avg_nRMSE. Resumable JSONL.
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

DATA = 'tmp/discreteness_2026-07-26/data'
CELLS = ['a', 'b', 'c']
SEEDS = (42, 1, 2, 3, 4)
EPS = [f'episode_BF4_15comp_seed{s}_{r}.h5'
       for s in (10, 11, 12, 13, 14) for r in ('uniform', 'slope')]


def _worker(args):
    cell, ep, seed = args
    import wavesplat.optimizer.sequential_update as su
    su.TRACK_PHASE_PREDICT = False
    from wavesplat.run import run_estimation
    t0 = time.time()
    path = f'{DATA}/cell_{cell}/{ep}'
    try:
        r = run_estimation(path, config={'adaptive_n': True, 'seed': seed, 'device': 'cpu'})
        return dict(cell=cell, episode=ep, seed=seed,
                    init_nrmse=float(r['init_nrmse']), avg_nRMSE=float(r['avg_nRMSE']),
                    n_gaussians=int(r['n_gaussians']), sec=time.time() - t0,
                    flag=su.TRACK_PHASE_PREDICT, err=None)
    except Exception as e:
        return dict(cell=cell, episode=ep, seed=seed, init_nrmse=None, avg_nRMSE=None,
                    n_gaussians=None, sec=time.time() - t0,
                    flag=su.TRACK_PHASE_PREDICT, err=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='tmp/discreteness_2026-07-26/ws_runs.jsonl')
    ap.add_argument('--workers', type=int, default=5)
    a = ap.parse_args()
    jobs = [(c, ep, s) for c in CELLS for ep in EPS for s in SEEDS]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                d = json.loads(line)
                if d.get('err') is None:
                    done.add((d['cell'], d['episode'], d['seed']))
            except Exception:
                pass
    jobs = [j for j in jobs if j not in done]
    missing = [j for j in jobs if not os.path.exists(f'{DATA}/cell_{j[0]}/{j[1]}')]
    if missing:
        print(f"  skipping {len(missing)} jobs whose episode is not generated yet "
              f"(cells: {sorted(set(j[0] for j in missing))}) -- re-invoke later", flush=True)
        jobs = [j for j in jobs if os.path.exists(f'{DATA}/cell_{j[0]}/{j[1]}')]
    print(f"WS arm: {len(jobs)} jobs ({len(done)} done), {a.workers} workers", flush=True)
    t0 = time.time(); n = 0
    with open(a.out, 'a', buffering=1) as f, Pool(a.workers) as pool:
        for res in pool.imap_unordered(_worker, jobs):
            f.write(json.dumps(res) + '\n'); n += 1
            tag = 'ERR' if res['err'] else (f"init {res['init_nrmse']:.3f} "
                                           f"avg {res['avg_nRMSE']:.3f}")
            print(f"[{n:3d}/{len(jobs)}] {res['cell']} {res['episode']:38s} "
                  f"s{res['seed']:<2d} {tag} ({res['sec']:.0f}s)", flush=True)
    print(f"WS arm done: {n} runs, {time.time()-t0:.0f}s wall", flush=True)


if __name__ == '__main__':
    main()
