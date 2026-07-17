"""verify_e43c_repro — fresh-context audit, independent re-derivation on GPU.

Attack 1 (blindness): re-implement the FULL arm's pipeline from the same
primitives, and run a GT-GARBLE CANARY: corrupt every ground-truth key in S
(d1, d2, Tp2, r2, m_kx, m_ky, m_A, m_ph) that the blind path must not read,
then recompute log-evidence + argmax + sc_full. If anything in the blind path
reads GT, the canary diverges (or crashes).

Attack 5 (construction): score the e42-style CONTINUOUS oracle
  (1-r2)*L1(d1_true) + r2*L2(Tp2_true ring, d2_true)   [x_marginals VERBATIM]
on the same seas, next to e43c's lattice-quantized node-oracle.

Also reproduces sc_full / sc_cur / sc_node_oracle / pct_true / ESS per sea and
compares to results/direction3/e43c_true_node_rank.json (read-only).

Seas: 300 (first), 322 (largest positive, catastrophic d1e=108 -> swap check),
341 (largest negative). Canary runs on 341 and 322.
"""
import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import DEPTH
from e37_bimodal_ladder import build_sea_crosstp, ang_err_deg, net_gamma_map
from e29_gate0_ladder import solve_with_grid
from e42_detector_v2 import JointEngine, DIRS, A_GRID, W_GRID, MIN_SEP_DEG
from e41_regime_prize_ladder import lobe_grid
from d3_sa_achievability_gap import dispersion_lam

J = {r['si']: r for r in
     json.load(open('results/direction3/e43c_true_node_rank.json'))['rows']}


def nearest(arr, v):
    return int(np.argmin(np.abs(np.asarray(arr) - v)))


def nearest_dir(v):
    return int(np.argmin(np.abs(np.angle(np.exp(1j * (DIRS - v))))))


def enumerate_and_score(S):
    """Blind path only: engine + full enumeration + argmax -> grid, scores."""
    eng = JointEngine(S)
    keys = [(round(float(x), 3), round(float(m), 5))
            for m in DIRS for x in A_GRID]
    eng.build_lobes(keys)
    lut = {k: eng._lobe[k] for k in keys}

    def idx_of(i_dir, i_a):
        return lut[(round(float(A_GRID[i_a]), 3),
                    round(float(DIRS[i_dir]), 5))]

    hyp, II, JJ, WW = [], [], [], []
    for i1 in range(len(DIRS)):
        for i2 in range(len(DIRS)):
            sep = abs(np.rad2deg(np.angle(np.exp(1j * (DIRS[i2] - DIRS[i1])))))
            if sep < MIN_SEP_DEG:
                continue
            for ia1 in range(len(A_GRID)):
                for ia2 in range(len(A_GRID)):
                    for iw in range(len(W_GRID)):
                        hyp.append((i1, ia1, i2, ia2, iw))
                        II.append(idx_of(i1, ia1))
                        JJ.append(idx_of(i2, ia2))
                        WW.append(W_GRID[iw])
    loge = eng.logev_pairs(II, JJ, WW)
    b = int(np.argmax(loge))
    j1, ja1, j2, ja2, jw = hyp[b]
    wb = float(W_GRID[jw])
    g_full = (1 - wb) * eng.grids[idx_of(j1, ja1)] \
        + wb * eng.grids[idx_of(j2, ja2)]
    sc_full = float(solve_with_grid(S, g_full))
    w = np.exp(loge - loge.max()); w /= w.sum()
    ess = float(1.0 / (w ** 2).sum())
    return dict(eng=eng, idx_of=idx_of, hyp=hyp, loge=loge, best=b,
                sc_full=sc_full, ess=ess, n_hyp=len(loge))


def garble(S, si):
    """Corrupt every GT key the blind path must not touch."""
    rng = np.random.default_rng(999000 + si)
    S2 = dict(S)
    S2['d1'] = float(rng.uniform(0, 2 * np.pi))
    S2['d2'] = float(rng.uniform(0, 2 * np.pi))
    S2['Tp2'] = float(rng.uniform(4, 12))
    S2['r2'] = float(rng.uniform(0, 1))
    for k in ('m_kx', 'm_ky', 'm_A', 'm_ph'):
        S2[k] = rng.permutation(np.asarray(S[k])) * rng.uniform(0.5, 2.0)
    S2['gt_u'] = S['gt_u']          # scoring target only, kept
    return S2


t0 = time.time()
for si, do_canary in [(300, False), (322, True), (341, True)]:
    S = build_sea_crosstp(si)
    r = J[si]
    E = enumerate_and_score(S)
    sc_cur = float(solve_with_grid(S, np.maximum(
        10.0 ** net_gamma_map(S), 1e-8)))
    # true node + node-oracle (GT allowed in THIS block only)
    a1t = (2 * np.pi / dispersion_lam(7.0, DEPTH)) / S['kp']
    a2t = (2 * np.pi / dispersion_lam(S['Tp2'], DEPTH)) / S['kp']
    Ti = dict(i_d1=nearest_dir(S['d1']), i_a1=nearest(A_GRID, a1t),
              i_d2=nearest_dir(S['d2']), i_a2=nearest(A_GRID, a2t),
              i_w=nearest(W_GRID, S['r2']))
    key_t = (Ti['i_d1'], Ti['i_a1'], Ti['i_d2'], Ti['i_a2'], Ti['i_w'])
    pos = {h: k for k, h in enumerate(E['hyp'])}
    kt = pos[key_t]
    pct_t = float((E['loge'] < E['loge'][kt]).sum() / E['n_hyp'])
    g_orc = (1 - S['r2']) * E['eng'].grids[E['idx_of'](Ti['i_d1'], Ti['i_a1'])] \
        + S['r2'] * E['eng'].grids[E['idx_of'](Ti['i_d2'], Ti['i_a2'])]
    sc_orc = float(solve_with_grid(S, g_orc))
    # e42-style CONTINUOUS oracle (x_marginals verbatim construction)
    g42 = (1 - S['r2']) * lobe_grid(S['kp'], [(7.0, 3.3, 1.0)],
                                    [(S['d1'], 30.0, 1.0)]) \
        + S['r2'] * lobe_grid(S['kp'], [(float(S['Tp2']), 3.3, 1.0)],
                              [(S['d2'], 30.0, 1.0)])
    sc_o42 = float(solve_with_grid(S, g42))
    # swap-corrected direction errors of the argmax
    j1, _, j2, _, _ = E['hyp'][E['best']]
    e_a = ang_err_deg(DIRS[j1], S['d1']) + ang_err_deg(DIRS[j2], S['d2'])
    e_b = ang_err_deg(DIRS[j1], S['d2']) + ang_err_deg(DIRS[j2], S['d1'])
    swap = e_b < e_a
    d1c = ang_err_deg(DIRS[j2 if swap else j1], S['d1'])
    d2c = ang_err_deg(DIRS[j1 if swap else j2], S['d2'])

    print(f"\n== si {si} ==  n_hyp {E['n_hyp']} (JSON {r['n_hyp']})")
    print(f"  sc_full {E['sc_full']:+.4f} (JSON {r['sc_full']:+.4f})  "
          f"d={E['sc_full']-r['sc_full']:+.1e}")
    print(f"  sc_cur  {sc_cur:+.4f} (JSON {r['sc_cur']:+.4f})  "
          f"d={sc_cur-r['sc_cur']:+.1e}")
    print(f"  node-orc {sc_orc:+.4f} (JSON {r['sc_node_oracle']:+.4f})  "
          f"d={sc_orc-r['sc_node_oracle']:+.1e}")
    print(f"  pct_true {pct_t:.4f} (JSON {r['pct_true']:.4f}) | "
          f"ESS {E['ess']:.1f} (JSON {r['ess']:.1f})")
    print(f"  e42-CONTINUOUS oracle {sc_o42:+.4f}  vs node-orc {sc_orc:+.4f} "
          f"(quantization cost {sc_orc-sc_o42:+.4f})")
    print(f"  argmax dir err raw d1 {ang_err_deg(DIRS[j1], S['d1']):.1f}/"
          f"d2 {ang_err_deg(DIRS[j2], S['d2']):.1f} | swap={swap} "
          f"corrected d1 {d1c:.1f}/d2 {d2c:.1f}")

    if do_canary:
        S2 = garble(S, si)
        E2 = enumerate_and_score(S2)
        same_argmax = E2['best'] == E['best']
        dmax = float(np.max(np.abs(E2['loge'] - E['loge'])))
        print(f"  CANARY(GT garbled): argmax same={same_argmax}  "
              f"max|dloge|={dmax:.2e}  sc_full {E2['sc_full']:+.4f} "
              f"(d={E2['sc_full']-E['sc_full']:+.1e})")
    print(f"  [{time.time()-t0:.0f}s]", flush=True)
