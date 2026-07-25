"""UNIFIED: GPU precompute of the learned set-proposal atoms for the 48 finemesh
episodes NOT covered by Stage-1 (the 1/2/5-comp sparse+moderate regime). Identical
protocol to tmp/wp_integration_2026-07-25/precompute_atoms.py: propose_atoms(nsamp=4,
80-iter polish) on the init window (get_init_window 2.0), pool_and_select(max_n=30),
cache raw pooled atoms (kx,ky,energy over the 4 samples) + n_selected per diff_seed.

Resumable. Read-only checkpoint/data. Writes tmp/unified_2026-07-26/atoms_cache_extra.jsonl.
  python3 tmp/unified_2026-07-26/precompute_atoms_extra.py
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.learned_init import propose_atoms, pool_and_select

FINE = 'data_finemesh_2026-07-23/'
DIFF_SEEDS = (0, 1, 2, 3, 4)
OUT = 'tmp/unified_2026-07-26/atoms_cache_extra.jsonl'

# The 48 finemesh episodes present in batchC (100) but NOT in Stage-1 arm_runs (52).
BATCHC = 'tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'
STAGE1 = 'tmp/wp_integration_2026-07-25/arm_runs.jsonl'


def missing_episodes():
    all100 = set(json.loads(l)['episode'] for l in open(BATCHC))
    covered = set(json.loads(l)['episode'] for l in open(STAGE1))
    return sorted(all100 - covered)


def main():
    EPISODES = missing_episodes()
    assert len(EPISODES) == 48, f'expected 48 missing, got {len(EPISODES)}'
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
