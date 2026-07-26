"""MIXED-RETRAIN GATE G3 secondary. Verbatim copy of
tmp/discreteness_2026-07-26/run_secondary.py; the ONLY edits are --ckpt / --cells
and the default --out. Original untouched.

ORIGINAL DOCSTRING FOLLOWS
SECONDARY endpoints (PREREG §5): n_selected and bf_peak_ratio per episode.

n_selected  : learned_init.pool_and_select(*propose_atoms(init_window, depth, nsamp=4,
              diffusion_seed=d)) with the shipped knobs -- the exact cardinality the WavePrior
              UNIFIED router thresholds at 8. One value per (episode, diffusion seed 0..4).
bf_peak_ratio: initializer.beamform_peak_ratio(get_init_window(2.0), depth) -- the FROZEN GT-free
              guard signal, deterministic, one per episode.
Both use the shipped library functions unmodified.
"""
import os, sys, json, time, argparse
import warnings; warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

DATA = 'tmp/discreteness_2026-07-26/data'
CELLS = ['a', 'b', 'c']   # overridden by --cells
DSEEDS = (0, 1, 2, 3, 4)
EPS = [f'episode_BF4_15comp_seed{s}_{r}.h5'
       for s in (10, 11, 12, 13, 14) for r in ('uniform', 'slope')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='tmp/mixed_retrain_2026-07-26/secondary_mixed_v1.jsonl')
    ap.add_argument('--ckpt', default='results/direction3/beamform_cond_diffusion_mixed_v1.pt')
    ap.add_argument('--cells', nargs='+', default=['c'])
    a = ap.parse_args()
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.optimizer.initializer import beamform_peak_ratio
    from wavesplat.optimizer import learned_init as li

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                d = json.loads(line); done.add((d['cell'], d['episode']))
            except Exception:
                pass

    with open(a.out, 'a', buffering=1) as f:
        for c in a.cells:
            for ep in EPS:
                if (c, ep) in done:
                    continue
                t0 = time.time()
                path = f'{DATA}/cell_{c}/{ep}'
                try:
                    L = EpisodeDataLoader(path)
                    fr = L.get_init_window(2.0)
                    bfr = float(beamform_peak_ratio(fr, L.depth))
                    nsel, nraw = [], []
                    for d in DSEEDS:
                        rkx, rky, ren = li.propose_atoms(fr, L.depth, ckpt=a.ckpt, nsamp=4,
                                                         diffusion_seed=d)
                        kx, ky, en = li.pool_and_select(rkx, rky, ren,
                                                        energy_thresh=0.03,
                                                        dedup_dk=0.03, max_n=30)
                        nsel.append(int(len(kx))); nraw.append(int(len(rkx)))
                    row = dict(cell=c, episode=ep, bf_peak_ratio=bfr,
                               n_selected=nsel, n_selected_med=float(np.median(nsel)),
                               n_raw=nraw[0], sec=time.time() - t0, err=None)
                except Exception as e:
                    row = dict(cell=c, episode=ep, bf_peak_ratio=None, n_selected=None,
                               n_selected_med=None, n_raw=None,
                               sec=time.time() - t0, err=repr(e))
                f.write(json.dumps(row) + '\n')
                print(f"{c} {ep:38s} bfr={row['bf_peak_ratio']} "
                      f"nsel={row['n_selected']} ({row['sec']:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
