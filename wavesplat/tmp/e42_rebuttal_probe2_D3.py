"""e42 rebuttal probe 2 (D3): does the evidence RANK the snapped-true
hypothesis below its own picks (model-level wall / realization overfit), or
did the top-6 seeding EXCLUDE it (search gap), or is it weight dilution?
16 seas (si=100..115), ~20 s/sea.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e37_bimodal_ladder import build_sea_crosstp
from e42_detector_v2 import (JointEngine, fit_joint, DIRS, A_GRID, W_GRID,
                             true_rings)


def snap_dir(x):
    return float(DIRS[np.argmin(np.abs(np.angle(np.exp(1j * (DIRS - x)))))])


def snap(vals, x):
    v = np.asarray(vals)
    return float(v[np.argmin(np.abs(v - x))])


def main():
    t0 = time.time()
    out = []
    for si in range(100, 116):
        S = build_sea_crosstp(si)
        kp = S['kp']
        a1t, a2t = true_rings(S)
        d1s, d2s = snap_dir(S['d1']), snap_dir(S['d2'])
        a1s, a2s = snap(A_GRID, a1t), snap(A_GRID, a2t)
        ws = snap(W_GRID, S['r2'])

        F = fit_joint(S)                       # the scored machinery verbatim
        # rank of the snapped-true lobe1 in the 1-lobe evidence (seed check)
        eng = JointEngine(S)
        keys = [(round(float(a), 3), round(float(m), 5))
                for m in DIRS for a in A_GRID]
        eng.build_lobes(keys)
        cand1 = [(m, a) for m in DIRS for a in A_GRID]
        idx1 = np.array([eng._lobe[(round(float(a), 3), round(float(m), 5))]
                         for (m, a) in cand1])
        loge1 = eng.logev_pairs(idx1, -np.ones(len(idx1), int),
                                np.zeros(len(idx1)))
        # snapped lobe1 rank (both assignments: (d1s,a1s) and (d2s,a2s))
        def rank_of(mu, a):
            j = [k for k, (m, aa) in enumerate(cand1)
                 if abs(m - mu) < 1e-9 and abs(aa - a) < 1e-9][0]
            return int((loge1 > loge1[j]).sum())
        r1 = rank_of(d1s, a1s)
        r1b = rank_of(d2s, a2s)
        # logE of the snapped-true joint 5-tuple
        i1 = eng.idx(a1s, d1s)
        i2 = eng.idx(a2s, d2s)
        le_true = float(eng.logev_pairs([i1], [i2], [ws])[0])
        # also the swap-assignment version
        le_swap = float(eng.logev_pairs([i2], [i1], [1 - ws])[0])
        le_true = max(le_true, le_swap)
        gap = F['logE2'] - le_true
        seeded = min(r1, r1b) < 6
        out.append(dict(si=si, gap=gap, r1=r1, r1b=r1b, seeded=seeded,
                        dlogE=F['dlogE']))
        print(f"X{si} logE2max-logEtrue {gap:+8.2f} | lobe1 rank {r1:3d} "
              f"(swap {r1b:3d}) seeded {int(seeded)} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    g = np.array([r['gap'] for r in out])
    sd = np.array([r['seeded'] for r in out])
    print(f"\nlogE(max found) - logE(snapped truth): median {np.median(g):+.2f}"
          f"  q25 {np.quantile(g,.25):+.2f} q75 {np.quantile(g,.75):+.2f}")
    print(f"truth-lobe1 in top-6 seeds: {sd.sum()}/{len(sd)}")
    print(f"seas where truth OUTSCORES the search max (gap<0): {(g<0).sum()}")
    print(f"gap<=3 nats (truth competitive in posterior): {(g<=3).sum()}")


if __name__ == '__main__':
    main()
