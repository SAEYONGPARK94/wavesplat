"""MIXED-RETRAIN GATE G1/G2 step 1 -- set-proposal atoms for the held-out 40,
from the NEW checkpoint. Verbatim copy of
tmp/heldout40_2026-07-26/precompute_atoms_heldout40.py; the ONLY edits are the
--ckpt / --out arguments and the default output path. Original untouched.

ORIGINAL DOCSTRING FOLLOWS
GPU-PHASE STEP 1 -- learned set-proposal atoms for the held-out 40.

VERBATIM copy of tmp/wp_integration_2026-07-25/precompute_atoms.py with three edits:
the episode directory, the episode list (all 40 held-out episodes), and the output path.
The cache format is byte-for-byte the same schema as
tmp/wp_integration_2026-07-25/atoms_cache.jsonl:

  {"episode": str, "diff_seed": int, "kx": [float], "ky": [float],
   "energy": [float], "n_selected": int, "sec": float, "err": null|str}

Frozen protocol: tmp/heldout40_2026-07-26/PREREG.md Sec 6.
  python3 tmp/heldout40_2026-07-26/precompute_atoms_heldout40.py
"""
import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.learned_init import propose_atoms, pool_and_select

FINE = 'data_confirm40_2026-07-27/'
DIFF_SEEDS = (0, 1, 2, 3, 4)
OUT = 'tmp/confirm40_2026-07-27/atoms_confirm_v1.jsonl'
CKPT = 'results/direction3/beamform_cond_diffusion_mixed_v1.pt'

EPISODES = sorted(f for f in os.listdir(FINE) if f.endswith('.h5'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    globals()['OUT'] = a.out
    print(f'# checkpoint {a.ckpt}', flush=True)
    assert len(EPISODES) == 40, f'expected 40 held-out episodes, found {len(EPISODES)}'
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
                kx, ky, en = propose_atoms(frames, depth, ckpt=a.ckpt, nsamp=4, diffusion_seed=ds)
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
