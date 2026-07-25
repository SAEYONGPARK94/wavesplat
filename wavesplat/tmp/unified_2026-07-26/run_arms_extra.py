"""UNIFIED: learned tracker arm on the 48 finemesh episodes not covered by Stage-1.

Identical protocol to tmp/wp_integration_2026-07-25/run_arms.py, LEARNED arm only:
lanes (est_seed,diff_seed)=(42,0),(1,1),(2,2),(3,3),(4,4); TRACK_PHASE_PREDICT=False per
worker (shipped default untouched); learned atoms from the GPU precompute cache
(atoms_cache_extra.jsonl); full 5-metric panel + raw-init diagnostic. 4 workers, threads=1.
Resumable JSONL -> arm_runs_extra.jsonl.

  nice -n 10 python3 tmp/unified_2026-07-26/run_arms_extra.py
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool
import numpy as np

FINE = 'data_finemesh_2026-07-23/'
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
CACHE_PATH = 'tmp/unified_2026-07-26/atoms_cache_extra.jsonl'
BATCHC = 'tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'
STAGE1 = 'tmp/wp_integration_2026-07-25/arm_runs.jsonl'
PANEL_KEYS = ['Variance_Ratio', 'Pearson_rho', 'Skill_Score',
              'Directional_R2', 'Hs_error']
CACHE = {}


def missing_episodes():
    all100 = set(json.loads(l)['episode'] for l in open(BATCHC))
    covered = set(json.loads(l)['episode'] for l in open(STAGE1))
    return sorted(all100 - covered)


def panel_from_frames(frame_results):
    out = {}
    for k in PANEL_KEYS:
        vals = [r[k] for r in frame_results
                if k in r and np.isfinite(r.get(k, np.nan))]
        out['avg_' + k] = float(np.mean(vals)) if vals else float('nan')
    return out


def _worker(args):
    ep, est_seed, diff_seed = args
    import wavesplat.optimizer.sequential_update as su
    su.TRACK_PHASE_PREDICT = False
    from wavesplat.run import run_estimation
    t0 = time.time()
    cfg = {'adaptive_n': True, 'seed': est_seed, 'device': 'cpu',
           'init_method': 'learned', 'learned_diffusion_seed': diff_seed,
           'learned_init_iters': 500}
    raw_init = None
    try:
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
        r = run_estimation(FINE + ep, config=cfg)
        panel = panel_from_frames(r['frame_results'])
        return dict(episode=ep, est_seed=est_seed, diff_seed=diff_seed, arm='learned',
                    init_nrmse=float(r['init_nrmse']), avg_nRMSE=float(r['avg_nRMSE']),
                    final_n=int(r['n_gaussians']), raw_init_nrmse=raw_init,
                    sec=time.time() - t0, flag=su.TRACK_PHASE_PREDICT, err=None, **panel)
    except Exception as e:
        return dict(episode=ep, est_seed=est_seed, diff_seed=diff_seed, arm='learned',
                    init_nrmse=None, avg_nRMSE=None, final_n=None, raw_init_nrmse=raw_init,
                    sec=time.time() - t0, flag=None, err=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='tmp/unified_2026-07-26/arm_runs_extra.jsonl')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    global CACHE
    for line in open(CACHE_PATH):
        d = json.loads(line)
        if d.get('err') is None and d.get('kx') is not None:
            CACHE[(d['episode'], d['diff_seed'])] = dict(
                kx=d['kx'], ky=d['ky'], energy=d['energy'])
    print(f"loaded {len(CACHE)} cached atom sets", flush=True)

    eps = missing_episodes()
    assert len(eps) == 48, f'expected 48, got {len(eps)}'
    jobs = [(ep, es, ds) for ep in eps for (es, ds) in LANES]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                d = json.loads(line)
                if d.get('err') is None:
                    done.add((d['episode'], d['est_seed'], d['diff_seed']))
            except Exception:
                pass
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"{len(eps)} episodes x {len(LANES)} lanes; {len(jobs)} runs to do "
          f"({len(done)} done), {a.workers} workers", flush=True)

    t0 = time.time(); n = 0
    with open(a.out, 'a', buffering=1) as f, Pool(a.workers) as pool:
        for res in pool.imap_unordered(_worker, jobs):
            f.write(json.dumps(res) + '\n'); n += 1
            if res['err']:
                tag = 'ERR ' + res['err'][:50]
            else:
                tag = f"init {res['init_nrmse']:.3f} avg {res['avg_nRMSE']:.3f} N{res['final_n']}"
                if res.get('raw_init_nrmse') is not None:
                    tag += f" raw {res['raw_init_nrmse']:.3f}"
            print(f"[{n:3d}/{len(jobs)}] learned {res['episode']:38s} "
                  f"e{res['est_seed']}d{res['diff_seed']} {tag} ({res['sec']:.0f}s)", flush=True)
    print(f"done: {n} runs, {time.time()-t0:.0f}s wall", flush=True)


if __name__ == '__main__':
    main()
