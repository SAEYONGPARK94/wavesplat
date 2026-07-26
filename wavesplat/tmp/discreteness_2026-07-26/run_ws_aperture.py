"""Analytic (WaveSplat) arm on the APERTURE cells (A-big, C-big).

Identical to run_ws.py except the init/tracking window is widened to tp_factor=8.0 (56.0 s), the
frozen aperture lever of PREREG_APERTURE_ADDENDUM.md §1. `wavesplat/run.py` hardcodes
`get_init_window(tp_factor=2.0)`, so the worker REBINDS the two loader methods in its own process
-- the shipped default is untouched, exactly as TRACK_PHASE_PREDICT=False is handled.
Asserts the window arithmetic (141 frames spanning 56.0 s) rather than assuming it.
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
CELLS = ['abig', 'cbig']
SEEDS = (42, 1, 2, 3, 4)
TPF = 8.0                       # frozen aperture lever
EPS = [f'episode_BF4_15comp_seed{s}_{r}.h5'
       for s in (20, 21, 22, 23, 24) for r in ('uniform', 'slope')]


def _patch_window():
    """Force tp_factor=TPF inside THIS process only."""
    from wavesplat.data_loader import EpisodeDataLoader
    if getattr(EpisodeDataLoader, '_aperture_patched', False):
        return
    _oi = EpisodeDataLoader.get_init_window
    _ot = EpisodeDataLoader.get_tracking_frames
    EpisodeDataLoader.get_init_window = lambda self, tp_factor=2.0: _oi(self, TPF)
    EpisodeDataLoader.get_tracking_frames = lambda self, tp_factor=2.0: _ot(self, TPF)
    EpisodeDataLoader._aperture_patched = True


def _worker(args):
    cell, ep, seed = args
    import wavesplat.optimizer.sequential_update as su
    su.TRACK_PHASE_PREDICT = False
    _patch_window()
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.run import run_estimation
    t0 = time.time()
    path = f'{DATA}/cell_{cell}/{ep}'
    try:
        L = EpisodeDataLoader(path)
        fr = L.get_init_window(2.0)          # patched -> TPF
        span = float(fr[-1].t) - float(fr[0].t)
        n_obs = int(sum(len(f.x_obs) for f in fr))
        assert len(fr) == 141 and abs(span - 56.0) < 1e-6, \
            f"aperture window assertion failed: {len(fr)} frames, {span:.3f}s"
        r = run_estimation(path, config={'adaptive_n': True, 'seed': seed, 'device': 'cpu'})
        return dict(cell=cell, episode=ep, seed=seed,
                    init_nrmse=float(r['init_nrmse']), avg_nRMSE=float(r['avg_nRMSE']),
                    n_gaussians=int(r['n_gaussians']), init_frames=len(fr),
                    init_span_s=span, init_n_obs=n_obs, tp_factor=TPF,
                    sec=time.time() - t0, flag=su.TRACK_PHASE_PREDICT, err=None)
    except Exception as e:
        return dict(cell=cell, episode=ep, seed=seed, init_nrmse=None, avg_nRMSE=None,
                    n_gaussians=None, init_frames=None, init_span_s=None, init_n_obs=None,
                    tp_factor=TPF, sec=time.time() - t0,
                    flag=su.TRACK_PHASE_PREDICT, err=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='tmp/discreteness_2026-07-26/ws_runs_aperture.jsonl')
    ap.add_argument('--workers', type=int, default=4)
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
    n_all = len(jobs)
    jobs = [j for j in jobs if os.path.exists(f'{DATA}/cell_{j[0]}/{j[1]}')]
    if len(jobs) != n_all:
        print(f"  skipping {n_all-len(jobs)} not-yet-generated episodes", flush=True)
    print(f"WS aperture arm (tp_factor={TPF}): {len(jobs)} jobs, {a.workers} workers", flush=True)
    t0 = time.time(); n = 0
    with open(a.out, 'a', buffering=1) as f, Pool(a.workers) as pool:
        for res in pool.imap_unordered(_worker, jobs):
            f.write(json.dumps(res) + '\n'); n += 1
            tag = ('ERR ' + str(res['err'])[:60]) if res['err'] else (
                f"init {res['init_nrmse']:.3f} avg {res['avg_nRMSE']:.3f}")
            print(f"[{n:3d}/{len(jobs)}] {res['cell']:5s} {res['episode']:38s} "
                  f"s{res['seed']:<2d} {tag} ({res['sec']:.0f}s)", flush=True)
    print(f"WS aperture arm done: {n} runs, {time.time()-t0:.0f}s wall", flush=True)


if __name__ == '__main__':
    main()
