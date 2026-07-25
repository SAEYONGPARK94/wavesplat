"""GPU precompute of the FINE-TUNED-ckpt learned atoms for the integrated spot check (PREREG §5a,§7).

10 crowded episodes x 5 diffusion seeds, cond proposal + 80-iter polish on the init window
(get_init_window 2.0), pooled/selected exactly as Stage-1 (propose_atoms + pool_and_select).
Writes ft_atoms_cache.jsonl. Read-only ckpt; never touches data/.
"""
import os, sys, json, time
import warnings; warnings.filterwarnings('ignore')
import numpy as np
sys.path.insert(0, 'scripts')
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.learned_init import propose_atoms, pool_and_select

FINE = 'data_finemesh_2026-07-23/'
FT_CKPT = 'results/direction3/prior_net_e2eft_A.pt'   # overwritten by argv[1] if given
DIFF_SEEDS = (0, 1, 2, 3, 4)
OUT = 'tmp/wp_e2e_2026-07-25/ft_atoms_cache.jsonl'

SPOT10 = [
    'episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_15comp_seed0_slope.h5',
    'episode_BF4_15comp_seed1_uniform.h5', 'episode_BF4_15comp_seed1_slope.h5',
    'episode_BF4_30comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_slope.h5',
    'episode_BF4_30comp_seed1_uniform.h5', 'episode_BF4_30comp_seed1_slope.h5',
    'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_slope.h5',
]


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else FT_CKPT
    print(f"precompute ft atoms from {ckpt}", flush=True)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                d = json.loads(line)
                if d.get('err') is None:
                    done.add((d['episode'], d['diff_seed']))
            except Exception:
                pass
    jobs = [(ep, ds) for ep in SPOT10 for ds in DIFF_SEEDS if (ep, ds) not in done]
    print(f"{len(SPOT10)} eps x {len(DIFF_SEEDS)} seeds; {len(jobs)} to run ({len(done)} cached)",
          flush=True)
    frame_cache = {}
    t0 = time.time(); n = 0
    with open(OUT, 'a', buffering=1) as f:
        for (ep, ds) in jobs:
            tt = time.time()
            try:
                if ep not in frame_cache:
                    L = EpisodeDataLoader(FINE + ep)
                    frame_cache[ep] = (L.get_init_window(2.0), L.depth)
                frames, depth = frame_cache[ep]
                kx, ky, en = propose_atoms(frames, depth, ckpt=ckpt, nsamp=4, diffusion_seed=ds)
                sk, sy, se = pool_and_select(kx, ky, en, max_n=30)
                row = dict(episode=ep, diff_seed=ds,
                           kx=[float(v) for v in kx], ky=[float(v) for v in ky],
                           energy=[float(v) for v in en],
                           n_selected=int(len(sk)), sec=time.time() - tt, err=None)
                tag = f"n_sel={len(sk):2d}"
            except Exception as e:
                row = dict(episode=ep, diff_seed=ds, kx=None, ky=None, energy=None,
                           n_selected=None, sec=time.time() - tt, err=repr(e))
                tag = 'ERR'
            f.write(json.dumps(row) + '\n'); n += 1
            print(f"[{n:3d}/{len(jobs)}] {ep:40s} d{ds} {tag} ({row['sec']:.1f}s)", flush=True)
    print(f"done: {n} runs, {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
