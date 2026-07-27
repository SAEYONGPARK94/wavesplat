"""MIXED-RETRAIN GATE G1/G2 step 2. Verbatim copy of
tmp/heldout40_2026-07-26/run_arms_heldout40.py; the ONLY edits are the --cache /
--out / --episodes arguments. Original untouched.

ORIGINAL DOCSTRING FOLLOWS
GPU-PHASE STEP 2 -- learned (and, as an integrity control, v2_konly) tracker arms
on the held-out 40.

VERBATIM copy of tmp/wp_integration_2026-07-25/run_arms.py with three edits: the episode
directory, the episode list (all 40 held-out episodes), and the cache/output paths.
Lanes, TRACK_PHASE_PREDICT=False, learned_init_iters=500 and the metric panel are unchanged.
The proposals come from the cache, so the workers themselves need no GPU.

Frozen protocol: tmp/heldout40_2026-07-26/PREREG.md Sec 1 and Sec 6.

  nice -n 10 python3 tmp/heldout40_2026-07-26/run_arms_heldout40.py --arms learned --workers 6

Integrity control (recommended, cheap): `--arms v2_konly` reproduces the CPU phase's
pure-v2 lanes through this second code path; they must agree with
tmp/heldout40_2026-07-26/v2_arm.jsonl to ~1e-10 (the same check the program ran on the
52-episode Stage-1 overlap).  A disagreement means the two protocols are NOT the same arm
and the held-out comparison is invalid.
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool
import numpy as np

FINE = 'data_confirm2_2026-07-27/'
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]   # (estimator_seed, diffusion_seed)
CACHE_PATH = 'tmp/confirm2_2026-07-27/atoms_confirm2_v1.jsonl'
PANEL_KEYS = ['Variance_Ratio', 'Pearson_rho', 'Skill_Score',
              'Directional_R2', 'Hs_error']

EPISODES = sorted(f for f in os.listdir(FINE) if f.endswith('.h5'))

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
    su.TRACK_PHASE_PREDICT = False           # phase-fix; module default untouched
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
            from wavesplat.data_loader import EpisodeDataLoader
            from wavesplat.optimizer.learned_init import raw_field_nrmse
            L = EpisodeDataLoader(FINE + ep)
            fr = L.get_init_window(2.0); last = fr[-1]
            raw_init = raw_field_nrmse(cfg, fr, L.depth, last.gt_field.numpy(),
                                       L.gt_grid_x.numpy(), L.gt_grid_y.numpy(),
                                       last.t, L.Hs_true, seed=est_seed)
        elif arm == 'auto':
            # live validation of the deployed policy; constants are the frozen defaults
            cfg['init_method'] = 'auto'
            cfg['learned_diffusion_seed'] = diff_seed
            cfg['learned_init_iters'] = 500
            entry = CACHE.get((ep, diff_seed))
            if entry is None:
                raise RuntimeError(f'no cached atoms for {ep} d{diff_seed}')
            cfg['learned_atoms'] = entry
        else:
            raise ValueError(arm)

        r = run_estimation(FINE + ep, config=cfg)
        panel = panel_from_frames(r['frame_results'])
        row = dict(episode=ep, est_seed=est_seed, diff_seed=diff_seed, arm=arm,
                   init_nrmse=float(r['init_nrmse']), avg_nRMSE=float(r['avg_nRMSE']),
                   final_n=int(r['n_gaussians']), raw_init_nrmse=raw_init,
                   # run.py exposes only these two router diagnostics in the summary;
                   # the guard's veto is already folded into `auto_selected`
                   # (sequential_update._auto_select_init), and bf_peak_ratio is
                   # recomputed independently by the frozen helper at score time.
                   auto_selected=r.get('auto_selected'),
                   auto_n_selected=r.get('auto_n_selected'),
                   sec=time.time() - t0, flag=su.TRACK_PHASE_PREDICT, err=None, **panel)
        return row
    except Exception as e:
        return dict(episode=ep, est_seed=est_seed, diff_seed=diff_seed, arm=arm,
                    init_nrmse=None, avg_nRMSE=None, final_n=None, raw_init_nrmse=raw_init,
                    sec=time.time() - t0, flag=None, err=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='learned',
                    choices=['learned', 'v2_konly', 'auto', 'learned_konly'])
    ap.add_argument('--out', default='tmp/confirm2_2026-07-27/arm_runs_confirm2_v1.jsonl')
    ap.add_argument('--cache', default=CACHE_PATH)
    ap.add_argument('--episodes', default='all')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()

    assert len(EPISODES) == 40, f'expected 40 held-out episodes, found {len(EPISODES)}'

    global CACHE
    if a.arms != 'v2_konly':
        for line in open(a.cache):
            d = json.loads(line)
            if d.get('err') is None and d.get('kx') is not None:
                CACHE[(d['episode'], d['diff_seed'])] = dict(
                    kx=d['kx'], ky=d['ky'], energy=d['energy'])
        print(f"loaded {len(CACHE)} cached atom sets", flush=True)

    eps = list(EPISODES)
    if a.episodes == 'crowded':
        eps = [e for e in eps if '15comp' in e or '30comp' in e]
    elif a.episodes == 'onecomp':
        eps = [e for e in eps if '_1comp_' in e]
    elif a.episodes == 'gates':                 # G1 crowded (16) + G2 one-comp (12)
        eps = [e for e in eps if '15comp' in e or '30comp' in e or '_1comp_' in e]
    print(f"episode subset '{a.episodes}': {len(eps)} episodes", flush=True)

    jobs = [(ep, es, ds, a.arms) for ep in eps for (es, ds) in LANES]

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
    print(f"{len(EPISODES)} episodes x 1 arm x {len(LANES)} lanes; "
          f"{len(jobs)} runs to do ({len(done)} done), {a.workers} workers", flush=True)

    t0 = time.time(); n = 0
    with open(a.out, 'a', buffering=1) as f, Pool(a.workers) as pool:
        for res in pool.imap_unordered(_worker, jobs):
            f.write(json.dumps(res) + '\n')
            n += 1
            tag = 'ERR ' + res['err'][:50] if res['err'] else f"N{res['final_n']}"
            print(f"[{n:3d}/{len(jobs)}] {res['arm']:12s} {res['episode']:44s} "
                  f"e{res['est_seed']}d{res['diff_seed']} {tag} ({res['sec']:.0f}s)", flush=True)
    print(f"done: {n} runs, {time.time()-t0:.0f}s wall", flush=True)


if __name__ == '__main__':
    main()
