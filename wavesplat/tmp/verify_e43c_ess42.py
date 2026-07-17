"""verify_e43c_ess42 — attack 6: does ESS(median 1403, FULL enumeration,
si=300 block) actually license "fit_bmaF vs fit_bma was not a trivial null"?

The e42 arm mixed KEEP_HYP=48 top hypotheses of the TOP-K=6 seeded 2-lobe set
(~11k hypotheses) on si=100..163. Measure, on THAT set and THOSE seas:
  - ESS of the 2-lobe posterior w2 (D2's claim: ESS ~ 1 => trivial null)
  - posterior mass captured by the kept top-48
  - ESS of mix_w (the weights fit_bmaF actually used)
Seas: si=100..105 (6 of the 64 actually scored in e42).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e37_bimodal_ladder import build_sea_crosstp
from e42_detector_v2 import fit_joint, JointEngine, DIRS, A_GRID, W_GRID, \
    MIN_SEP_DEG, TOPK, KEEP_HYP, _bma_weights
import e42_detector_v2 as v2

t0 = time.time()
print(f"e42 family: TOPK={TOPK} KEEP_HYP={KEEP_HYP}")
for si in range(100, 106):
    S = build_sea_crosstp(si)
    # replicate fit_joint's stage B set to get loge2 (fit_joint doesn't
    # return it) -- verbatim logic
    eng = JointEngine(S)
    keys = [(round(float(a), 3), round(float(m), 5))
            for m in DIRS for a in A_GRID]
    eng.build_lobes(keys)
    lut = {k: eng._lobe[k] for k in keys}
    cand1 = [(m, a) for m in DIRS for a in A_GRID]
    idx1 = np.array([lut[(round(float(a), 3), round(float(m), 5))]
                     for (m, a) in cand1])
    loge1 = eng.logev_pairs(idx1, -np.ones(len(idx1), int),
                            np.zeros(len(idx1)))
    order = np.argsort(-loge1)[:TOPK]
    II, JJ, WW = [], [], []
    for oi in order:
        m1, a1 = cand1[oi]
        i1 = idx1[oi]
        for m2 in DIRS:
            if abs(np.rad2deg(np.angle(np.exp(1j * (m2 - m1))))) < MIN_SEP_DEG:
                continue
            for a2 in A_GRID:
                i2 = lut[(round(float(a2), 3), round(float(m2), 5))]
                for w in W_GRID:
                    II.append(i1); JJ.append(i2); WW.append(w)
    loge2 = eng.logev_pairs(II, JJ, WW)
    w2 = _bma_weights(loge2)
    ess2 = 1.0 / (w2 ** 2).sum()
    keep = np.argsort(-w2)[:KEEP_HYP]
    mass48 = float(w2[keep].sum())
    # ESS of the mixture weights fit_bmaF actually used
    F = fit_joint(S)
    mw = np.asarray(F['mix_w'])
    ess_mix = 1.0 / (mw ** 2).sum()
    print(f"si {si}: |hyp2| {len(loge2)} | ESS(w2) {ess2:8.1f} | "
          f"top-48 mass {mass48:.3f} | ESS(mix_w) {ess_mix:6.1f} | p2 "
          f"{F['p2']:.3f} [{time.time()-t0:.0f}s]", flush=True)
