"""GPU-PHASE STEP 4 -- bf_peak_ratio on the 40 held-out episodes (feeds bar S5).

Mirrors tmp/unified_guard_2026-07-26/score_task1.py:bf_ratios() exactly: the FROZEN helper
wavesplat.optimizer.initializer.beamform_peak_ratio on the 2.0 s init window each arm sees.
Writes tmp/heldout40_2026-07-26/bf_ratio.json.
"""
import os, json, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool

FINE = 'data_heldout40_2026-07-26/'
OUT = 'tmp/heldout40_2026-07-26/bf_ratio.json'
EPISODES = sorted(f for f in os.listdir(FINE) if f.endswith('.h5'))


def one(ep):
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.optimizer.initializer import beamform_peak_ratio
    L = EpisodeDataLoader(FINE + ep)
    return ep, float(beamform_peak_ratio(L.get_init_window(2.0), L.depth))


def main():
    assert len(EPISODES) == 40, len(EPISODES)
    with Pool(4) as p:
        res = dict(p.map(one, EPISODES))
    json.dump(res, open(OUT, 'w'), indent=0)
    print(f"wrote {OUT}: {len(res)} episodes")


if __name__ == '__main__':
    main()
