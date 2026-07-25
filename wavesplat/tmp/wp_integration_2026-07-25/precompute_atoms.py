"""GPU precompute of the learned set-proposal atoms for STAGE-1 integration.

For each episode x diffusion_seed in {0,1,2,3,4}, run the cond-diffusion proposal + 80-iter
polish on the INIT WINDOW (get_init_window 2.0) and cache the raw pooled atoms (kx, ky, energy
across the 4 samples) so the CPU tracker workers can install them without touching the GPU.

Resumable via atoms_cache.jsonl. Read-only checkpoint; never touches data/.
  python3 tmp/wp_integration_2026-07-25/precompute_atoms.py
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np

# repo root == cwd expected; learned_init lazily imports d3_* (which chdir's to repo root)
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.learned_init import propose_atoms, pool_and_select

FINE = 'data_finemesh_2026-07-23/'
DIFF_SEEDS = (0, 1, 2, 3, 4)
OUT = 'tmp/wp_integration_2026-07-25/atoms_cache.jsonl'

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
EPISODES = CROWDED + SPARSE


def main():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                d = json.loads(line)
                if d.get('err') is None:
                    done.add((d['episode'], d['diff_seed']))
            except Exception:
                pass
    jobs = [(ep, ds) for ep in EPISODES for ds in DIFF_SEEDS if (ep, ds) not in done]
    print(f"precompute: {len(EPISODES)} episodes x {len(DIFF_SEEDS)} diff-seeds; "
          f"{len(jobs)} to run ({len(done)} cached)", flush=True)

    # cache loaders (init frames + depth) per episode to avoid re-reading h5
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
                kx, ky, en = propose_atoms(frames, depth, nsamp=4, diffusion_seed=ds)
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
            f.write(json.dumps(row) + '\n')
            n += 1
            print(f"[{n:3d}/{len(jobs)}] {ep:40s} d{ds} {tag} ({row['sec']:.1f}s)", flush=True)
    print(f"precompute done: {n} runs, {time.time()-t0:.0f}s wall", flush=True)


if __name__ == '__main__':
    main()
