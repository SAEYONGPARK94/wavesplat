"""STAGE-1 CPU tracker arms: learned-init vs v2_konly-init under the phase-fixed tracker.

Runs BOTH arms with identical metric extraction on the 40 crowded + 12 sparse finemesh
episodes, 5 paired lanes (est_seed, diff_seed)=(42,0),(1,1),(2,2),(3,3),(4,4). Per PREREG:
TRACK_PHASE_PREDICT=False per worker (shipped default untouched); learned atoms read from the
GPU precompute cache (no GPU in the worker). 4 workers, threads=1. Full 5-metric panel +
init decomposition. Resumable JSONL.

  nice -n 10 python3 tmp/wp_integration_2026-07-25/run_arms.py --arms both
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool
import numpy as np

FINE = 'data_finemesh_2026-07-23/'
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]   # (estimator_seed, diffusion_seed)
CACHE_PATH = 'tmp/wp_integration_2026-07-25/atoms_cache.jsonl'
# NB: 'nRMSE' excluded here -- avg_nRMSE is written explicitly from the run summary
# (identical value) to avoid a duplicate-keyword collision in the row dict().
PANEL_KEYS = ['Variance_Ratio', 'Pearson_rho', 'Skill_Score',
              'Directional_R2', 'Hs_error']

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

# populated in main(), inherited by workers via fork
CACHE = {}


def panel_from_frames(frame_results):
    out = {}
    for k in PANEL_KEYS:
        vals = [r[k] for r in frame_results
                if k in r and np.isfinite(r.get(k, np.nan))]
        out['avg_' + k] = float(np.mean(vals)) if vals else float('nan')
    return out


def _worker(args):
    ep, est_seed, diff_seed, arm = args
    import wavesplat.optimizer.sequential_update as su
    su.TRACK_PHASE_PREDICT = False           # phase-fix; module default (True) untouched
    from wavesplat.run import run_estimation
    t0 = time.time()
    cfg = {'adaptive_n': True, 'seed': est_seed, 'device': 'cpu'}
    raw_init = None
    try:
        if arm == 'v2_konly':
            cfg['init_method'] = 'v2_konly'
        elif arm in ('learned', 'learned_konly'):
            cfg['init_method'] = 'learned'
            cfg['learned_diffusion_seed'] = diff_seed
            cfg['learned_init_iters'] = 500
            if arm == 'learned_konly':
                cfg['learned_konly'] = True
            entry = CACHE.get((ep, diff_seed))
            if entry is None:
                raise RuntimeError(f'no cached atoms for {ep} d{diff_seed}')
            cfg['learned_atoms'] = entry
            # raw-init diagnostic (LS w/phi, NO joint fit) @ last init frame
            from wavesplat.data_loader import EpisodeDataLoader
            from wavesplat.optimizer.learned_init import raw_field_nrmse
            L = EpisodeDataLoader(FINE + ep)
            fr = L.get_init_window(2.0); last = fr[-1]
            raw_init = raw_field_nrmse(cfg, fr, L.depth, last.gt_field.numpy(),
                                       L.gt_grid_x.numpy(), L.gt_grid_y.numpy(),
                                       last.t, L.Hs_true, seed=est_seed)
        else:
            raise ValueError(arm)

        r = run_estimation(FINE + ep, config=cfg)
        panel = panel_from_frames(r['frame_results'])
        row = dict(episode=ep, est_seed=est_seed, diff_seed=diff_seed, arm=arm,
                   init_nrmse=float(r['init_nrmse']), avg_nRMSE=float(r['avg_nRMSE']),
                   final_n=int(r['n_gaussians']), raw_init_nrmse=raw_init,
                   sec=time.time() - t0, flag=su.TRACK_PHASE_PREDICT, err=None, **panel)
        return row
    except Exception as e:
        return dict(episode=ep, est_seed=est_seed, diff_seed=diff_seed, arm=arm,
                    init_nrmse=None, avg_nRMSE=None, final_n=None, raw_init_nrmse=raw_init,
                    sec=time.time() - t0, flag=None, err=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='both',
                    choices=['both', 'learned', 'v2_konly', 'learned_konly', 'all'])
    ap.add_argument('--set', default='both', choices=['crowded', 'sparse', 'both'])
    ap.add_argument('--out', default='tmp/wp_integration_2026-07-25/arm_runs.jsonl')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    # load atom cache
    global CACHE
    for line in open(CACHE_PATH):
        d = json.loads(line)
        if d.get('err') is None and d.get('kx') is not None:
            CACHE[(d['episode'], d['diff_seed'])] = dict(
                kx=d['kx'], ky=d['ky'], energy=d['energy'])
    print(f"loaded {len(CACHE)} cached atom sets", flush=True)

    eps = []
    if a.set in ('crowded', 'both'):
        eps += CROWDED
    if a.set in ('sparse', 'both'):
        eps += SPARSE
    if a.arms == 'both':
        arms = ['v2_konly', 'learned']
    elif a.arms == 'all':
        arms = ['v2_konly', 'learned', 'learned_konly']
    else:
        arms = [a.arms]

    jobs = [(ep, es, ds, arm) for ep in eps for (es, ds) in LANES for arm in arms]

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                d = json.loads(line)
                if d.get('err') is None:
                    done.add((d['episode'], d['est_seed'], d['diff_seed'], d['arm']))
            except Exception:
                pass
    jobs = [j for j in jobs if j not in done]
    print(f"{len(eps)} episodes x {len(arms)} arms x {len(LANES)} lanes; "
          f"{len(jobs)} runs to do ({len(done)} done), {a.workers} workers", flush=True)

    t0 = time.time(); n = 0
    with open(a.out, 'a', buffering=1) as f, Pool(a.workers) as pool:
        for res in pool.imap_unordered(_worker, jobs):
            f.write(json.dumps(res) + '\n')
            n += 1
            if res['err']:
                tag = 'ERR ' + res['err'][:50]
            else:
                tag = (f"init {res['init_nrmse']:.3f} avg {res['avg_nRMSE']:.3f} "
                       f"N{res['final_n']}")
                if res.get('raw_init_nrmse') is not None:
                    tag += f" raw {res['raw_init_nrmse']:.3f}"
            print(f"[{n:3d}/{len(jobs)}] {res['arm']:12s} {res['episode']:38s} "
                  f"e{res['est_seed']}d{res['diff_seed']} {tag} ({res['sec']:.0f}s)", flush=True)
    print(f"done: {n} runs, {time.time()-t0:.0f}s wall", flush=True)


if __name__ == '__main__':
    main()
