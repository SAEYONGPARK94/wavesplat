"""SECONDARY endpoints (PREREG §5): n_selected and bf_peak_ratio per episode.

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
CELLS = ['abig', 'cbig']
DSEEDS = (0, 1, 2, 3, 4)
EPS = [f'episode_BF4_15comp_seed{s}_{r}.h5'
       for s in (20, 21, 22, 23, 24) for r in ('uniform', 'slope')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='tmp/discreteness_2026-07-26/secondary_aperture.jsonl')
    a = ap.parse_args()
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.optimizer.initializer import beamform_peak_ratio
    from wavesplat.optimizer import learned_init as li
    # aperture lever: force tp_factor=8.0 in this process only (shipped default untouched)
    _oi = EpisodeDataLoader.get_init_window
    EpisodeDataLoader.get_init_window = lambda self, tp_factor=2.0: _oi(self, 8.0)

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                d = json.loads(line); done.add((d['cell'], d['episode']))
            except Exception:
                pass

    with open(a.out, 'a', buffering=1) as f:
        for c in CELLS:
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
                        rkx, rky, ren = li.propose_atoms(fr, L.depth, nsamp=4,
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
