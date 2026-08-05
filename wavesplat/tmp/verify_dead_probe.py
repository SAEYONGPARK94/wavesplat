"""verify_dead_probe.py — fresh-context probe for e42 DEAD audit.
(1) determinism: recompute score_sea rows for a slice of scored seas and
    compare to the stored JSON rows.
(2) posterior degeneracy: ESS of the evidence weights (w1, w2, mix_w),
    mass captured by the top-KEEP_HYP selection, and the number of
    hypotheses surviving mixture_of_fields' 1e-3 prune.
Writes NOTHING to results/.
"""
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e37_bimodal_ladder import build_sea_crosstp, build_sea_bimodal
from e42_detector_v2 import (JointEngine, fit_joint, DIRS, A_GRID, W_GRID,
                             TOPK, MIN_SEP_DEG, KEEP_HYP, _bma_weights)
from e42_blind_recovery import score_sea

D = json.load(open('results/direction3/e42_blind_recovery.json'))
stored = {r['si']: r for r in D['rows']}
stored_uni = {r['j']: r for r in D['uni']}
tau = float(D['tau'])

def ess(w):
    w = np.asarray(w, float); w = w / w.sum()
    return 1.0 / np.sum(w ** 2)

def internals(S):
    """Duplicate fit_joint's stages to expose w1/w2; cross-check vs fit_joint."""
    eng = JointEngine(S)
    keys = [(round(float(a), 3), round(float(m), 5))
            for m in DIRS for a in A_GRID]
    eng.build_lobes(keys)
    lut = {k: eng._lobe[k] for k in keys}
    cand1 = [(m, a) for m in DIRS for a in A_GRID]
    idx1 = np.array([lut[(round(float(a), 3), round(float(m), 5))]
                     for (m, a) in cand1])
    loge1 = eng.logev_pairs(idx1, -np.ones(len(idx1), int), np.zeros(len(idx1)))
    w1 = _bma_weights(loge1)
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
    keep2 = np.argsort(-w2)[:KEEP_HYP]
    wk = w2[keep2] / w2[keep2].sum()
    l0 = max(loge1.max(), loge2.max())
    ev1 = float(np.exp(loge1 - l0).mean())
    ev2 = float(np.exp(loge2 - l0).mean())
    p2 = ev2 / (ev1 + ev2)
    k1 = np.argsort(-w1)[:KEEP_HYP]
    mix_w = np.concatenate([(1 - p2) * (w1[k1] / w1[k1].sum()), p2 * wk])
    mix_w = mix_w / mix_w.sum()
    return dict(loge1=loge1, w1=w1, loge2=loge2, w2=w2, p2=p2,
                mix_w=mix_w, mass1=float(w1[k1].sum()),
                mass2=float(w2[keep2].sum()),
                dlogE=float(loge2.max() - loge1.max()))

SIS = [100, 101, 102, 120, 128, 146]
print('=== (1) determinism + (2) posterior degeneracy, arena X slice ===')
keys_cmp = ['cur_grid', 'm2', 'ens1', 'oracle', 'fit1', 'fit_bma',
            'fit_bmaF', 'U_BMA', 'fit2_hard', 'dlogE', 'p2']
for si in SIS:
    S = build_sea_crosstp(si)
    r = score_sea(S, tau)
    s = stored[si]
    diffs = {k: abs((r[k] - s[k]) if np.isfinite(r[k]) and np.isfinite(s[k])
                    else (0.0 if (not np.isfinite(r[k])) == (not np.isfinite(s[k]))
                          else np.inf)) for k in keys_cmp}
    worst = max(diffs, key=diffs.get)
    I = internals(S)
    ok_dlog = abs(I['dlogE'] - s['dlogE']) < 1e-6
    nprune = int((I['mix_w'] > 1e-3 * I['mix_w'].max()).sum())
    print(f"si {si}: max|recomp-stored| = {diffs[worst]:.2e} ({worst}) | "
          f"ESS w1 {ess(I['w1']):6.2f}/504  mass_top48 {I['mass1']:.4f} | "
          f"ESS w2 {ess(I['w2']):6.2f}/{len(I['w2'])}  mass_top48 "
          f"{I['mass2']:.4f} | ESS mix {ess(I['mix_w']):5.2f}/96  "
          f"survive_1e-3 {nprune} | p2 {I['p2']:.4f} (stored {s['p2']:.4f}) "
          f"dlogE ok {ok_dlog}")

print('\n=== unimodal do-no-harm slice (j=0,1) ===')
for j in [0, 1]:
    S = build_sea_bimodal(3400 + j, unimodal=True)
    r = score_sea(S, tau, unimodal=True)
    s = stored_uni[j]
    ks = ['cur_grid', 'ens1', 'fit1', 'fit_bma', 'fit_bmaF', 'U_BMA', 'dlogE', 'p2']
    dmax = max(abs(r[k] - s[k]) for k in ks)
    print(f"j {j}: max|recomp-stored| = {dmax:.2e}  fired {r['fired']} "
          f"(stored {s['fired']})")
