"""MIXED-RETRAIN GATE G3. Verbatim copy of tmp/discreteness_2026-07-26/run_cond.py;
the ONLY edits are the --cells argument and the default --ckpt/--out. Original untouched.

ORIGINAL DOCSTRING FOLLOWS
Learned (cond-diffusion set-proposal) arm on the three discreteness cells.

`eval_episode` is imported VERBATIM from tmp/wp_crowded_2026-07-25/run_cond_crowded.py -- same
3 nowcast frames, same n=4 BMA + 80-iter polish, same /Hs normalizer, same checkpoint
(results/direction3/beamform_cond_diffusion.pt, read-only). Median-of-5 diffusion seeds
(0,1,2,3,4). Resumable JSONL.
"""
import os, sys, json, time, argparse
import warnings; warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp/wp_crowded_2026-07-25')

import torch
import d3_beamform_cond_diffusion as cd            # os.chdir's to repo root on import
from run_cond_crowded import eval_episode          # VERBATIM reuse

DATA = 'tmp/discreteness_2026-07-26/data'
CELLS = ['a', 'b', 'c']
SEEDS = (0, 1, 2, 3, 4)
EPS = [f'episode_BF4_15comp_seed{s}_{r}.h5'
       for s in (10, 11, 12, 13, 14) for r in ('uniform', 'slope')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='tmp/mixed_retrain_2026-07-26/cond_runs_mixed_v1.jsonl')
    ap.add_argument('--ckpt', default='results/direction3/beamform_cond_diffusion_mixed_v1.pt')
    ap.add_argument('--cells', nargs='+', default=['c'])
    a = ap.parse_args()

    stats = torch.load(cd.PRIOR_CKPT, map_location=cd.DEVICE)['stats']
    betas = cd.cosine_betas(cd.T).to(cd.DEVICE)
    abar = torch.cumprod(1 - betas, 0)
    model = cd.CondSetDenoiser().to(cd.DEVICE)
    model.load_state_dict(torch.load(a.ckpt, map_location=cd.DEVICE))
    model.eval()

    jobs = [(c, ep, s) for c in a.cells for ep in EPS for s in SEEDS]
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
        print(f"  skipping {n_all - len(jobs)} jobs whose episode is not generated yet "
              "-- re-invoke later", flush=True)
    print(f"cond arm on {cd.DEVICE}: {len(jobs)} jobs ({len(done)} done)", flush=True)

    t0 = time.time(); n = 0
    with open(a.out, 'a', buffering=1) as f:
        for (c, ep, s) in jobs:
            tt = time.time()
            try:
                v = eval_episode(model, stats, abar, betas,
                                 f'{DATA}/cell_{c}/{ep}', s)
                err = None
            except Exception as e:
                v = None; err = repr(e)
            row = dict(cell=c, episode=ep, seed=s, cond_polish_nrmse_Hs=v,
                       sec=time.time() - tt, err=err)
            f.write(json.dumps(row) + '\n'); n += 1
            print(f"[{n:3d}/{len(jobs)}] {c} {ep:38s} s{s} "
                  f"cond {'ERR' if err else f'{v:.3f}'} ({row['sec']:.1f}s)", flush=True)
    print(f"cond arm done: {n} runs, {time.time()-t0:.0f}s wall", flush=True)


if __name__ == '__main__':
    main()
