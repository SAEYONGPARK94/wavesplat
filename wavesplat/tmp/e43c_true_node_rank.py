"""e43c — TRUE-NODE EVIDENCE RANK: is the evidence surface FLAT or DECEIVED?

WHY. Two pre-registered experiments now agree the blind route does not work
(e42 fit_bmaF -0.0166 CI up +0.017; e43b v3_point -0.0137 CI up +0.016, both
DEAD). But WHY is not established -- my "evidence == uniform => identifiability
wall" inference was RETRACTED (the U_BMA control averaged an EVIDENCE-SELECTED
top-48, so it was never a uniform baseline). So: the route does not work; it is
NOT shown that it cannot.

This is the diagnostic D1 and D3 independently named as the tiebreaker, and the
one D1 pre-committed to as a self-death condition ("v3 failure + true-node rank
flat + CRB > threshold => I sign D1's death").

THE TEST. For each sea, take the lattice node nearest the TRUE theta and ask
where its log-evidence ranks among all hypotheses the fit considered.

  FLAT      true node ranks ~ uniformly at random (percentile ~ 0.5, and the
            evidence spread between best and true node is ~ noise)
            => the likelihood genuinely carries no information about theta
            => identifiability wall, established PROPERLY this time.

  DECEIVED  true node ranks POORLY while some wrong node ranks top
            => the likelihood carries information but points the wrong way
            => a bias/misspecification problem, NOT an information wall.
               (v2 already fixed the surface's worst deception: corr(d2_err,
               dlogE) went +0.419 -> +0.077.)

  INFORMED  true node ranks TOP but the pipeline still loses
            => information is there and found; the FAMILY/mapping destroys it.
               This would REOPEN the route and indict the 2-lobe parameterization.

CONTROL (essential): a PERMUTED-truth node -- the same lattice ring/weight but
with the direction pair re-drawn at random. Its rank distribution is the null.
If the true node's rank is indistinguishable from the permuted node's rank, the
surface is FLAT. This mirrors the PERM control that has repeatedly rescued this
project from over-reading sharpness.

ALSO reported: ESS (effective sample size) of the 2-lobe posterior -- D2's
rebuttal (never delivered; session limit) would have argued that if ESS ~ 1 then
"mixture-of-fields == grid-BMA" was a TRIVIAL null (both degenerate to one
hypothesis) rather than a finding. Settle it here.

Seas: fresh si=300..347 (n=48). si=0..31 / 100..163 / 200..263 are all burned.
Diagnostic only -- no verdict bar, no arm is scored against the incumbent.

Usage: python3 tmp/e43c_true_node_rank.py [--smoke] [--n 48]
Out:   results/direction3/e43c_true_node_rank.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import DEPTH
from e37_bimodal_ladder import build_sea_crosstp, ang_err_deg, net_gamma_map
from e29_gate0_ladder import solve_with_grid
from e42_detector_v2 import (JointEngine, DIRS, A_GRID, W_GRID, MIN_SEP_DEG)
from d3_sa_achievability_gap import dispersion_lam

OUT = 'results/direction3/e43c_true_node_rank.json'


def nearest(arr, v):
    return int(np.argmin(np.abs(np.asarray(arr) - v)))


def nearest_dir(v):
    d = np.angle(np.exp(1j * (DIRS - v)))
    return int(np.argmin(np.abs(d)))


def true_nodes(S):
    """Lattice indices closest to the TRUE theta."""
    a1t = (2 * np.pi / dispersion_lam(7.0, DEPTH)) / S['kp']
    a2t = (2 * np.pi / dispersion_lam(S['Tp2'], DEPTH)) / S['kp']
    return dict(i_d1=nearest_dir(S['d1']), i_a1=nearest(A_GRID, a1t),
                i_d2=nearest_dir(S['d2']), i_a2=nearest(A_GRID, a2t),
                i_w=nearest(W_GRID, S['r2']), a1t=float(a1t), a2t=float(a2t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n', type=int, default=48)
    a = ap.parse_args()
    if a.smoke:
        a.n = 4
    t0 = time.time()
    rows = []
    for si in range(300, 300 + a.n):
        S = build_sea_crosstp(si)
        eng = JointEngine(S)
        keys = [(round(float(x), 3), round(float(m), 5))
                for m in DIRS for x in A_GRID]
        eng.build_lobes(keys)
        lut = {k: eng._lobe[k] for k in keys}

        def idx_of(i_dir, i_a):
            return lut[(round(float(A_GRID[i_a]), 3),
                        round(float(DIRS[i_dir]), 5))]

        # full 2-lobe hypothesis enumeration (all dir pairs >= MIN_SEP, all
        # rings, all w) -- the honest denominator for a rank
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

        T = true_nodes(S)
        key_t = (T['i_d1'], T['i_a1'], T['i_d2'], T['i_a2'], T['i_w'])
        pos = {h: k for k, h in enumerate(hyp)}
        kt = pos.get(key_t)
        if kt is None:      # separation constraint excluded the true pair
            rows.append(dict(si=si, excluded=True))
            print(f'{si} true pair excluded by MIN_SEP', flush=True)
            continue
        # PERM control: true rings/weight, directions re-drawn at random
        rngp = np.random.default_rng(50000 + si)
        for _ in range(200):
            p1, p2 = rngp.integers(0, len(DIRS), 2)
            sep = abs(np.rad2deg(np.angle(np.exp(1j * (DIRS[p2] - DIRS[p1])))))
            if sep >= MIN_SEP_DEG and (p1, T['i_a1'], p2, T['i_a2'], T['i_w']) in pos:
                break
        kp_ = pos[(p1, T['i_a1'], p2, T['i_a2'], T['i_w'])]

        # ---- SCORE the full-enumeration argmax (the decisive follow-up).
        # v2's top-K=6 seeding achieved d1~24 deg / d2~23.7 deg. If the FULL
        # joint enumeration is far more accurate, the seeding -- not the
        # information -- was the bottleneck, and e42/e43b's DEAD verdicts were
        # measuring my own search, not the likelihood.
        best0 = int(np.argmax(loge))
        i1, ia1, i2, ia2, iw = hyp[best0]
        wbest = float(W_GRID[iw])
        g_full = (1 - wbest) * eng.grids[idx_of(i1, ia1)] \
            + wbest * eng.grids[idx_of(i2, ia2)]
        sc_full = solve_with_grid(S, g_full)
        Yn = net_gamma_map(S)
        sc_cur = solve_with_grid(S, np.maximum(10.0 ** Yn, 1e-8))
        g_orc = (1 - S['r2']) * eng.grids[idx_of(T['i_d1'], T['i_a1'])] \
            + S['r2'] * eng.grids[idx_of(T['i_d2'], T['i_a2'])]
        sc_node_orc = solve_with_grid(S, g_orc)

        n = len(loge)
        pct_t = float((loge < loge[kt]).sum() / n)
        pct_p = float((loge < loge[kp_]).sum() / n)
        best = int(np.argmax(loge))
        # ESS of the 2-lobe posterior
        w = np.exp(loge - loge.max()); w /= w.sum()
        ess = float(1.0 / (w ** 2).sum())
        rows.append(dict(
            si=si, excluded=False, n_hyp=n, pct_true=pct_t, pct_perm=pct_p,
            ess=ess, dlogE_best_minus_true=float(loge[best] - loge[kt]),
            dlogE_true_minus_perm=float(loge[kt] - loge[kp_]),
            best_d1_err=float(ang_err_deg(DIRS[hyp[best][0]], S['d1'])),
            best_d2_err=float(ang_err_deg(DIRS[hyp[best][2]], S['d2'])),
            true_w_post=float(w[kt]),
            sc_full=float(sc_full), sc_cur=float(sc_cur),
            sc_node_oracle=float(sc_node_orc)))
        r = rows[-1]
        print(f"{si} | true pct {pct_t:.4f} perm {pct_p:.3f} | "
              f"best-true {r['dlogE_best_minus_true']:6.1f} | "
              f"true-perm {r['dlogE_true_minus_perm']:+7.1f} nats | "
              f"ESS {ess:6.1f} | d1e {r['best_d1_err']:4.1f} "
              f"d2e {r['best_d2_err']:5.1f} | FULL {sc_full:+.3f} vs "
              f"cur {sc_cur:+.3f} = {sc_full-sc_cur:+.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    R = [r for r in rows if not r['excluded']]
    pt = np.array([r['pct_true'] for r in R])
    pp = np.array([r['pct_perm'] for r in R])
    ess = np.array([r['ess'] for r in R])
    tmp_ = np.array([r['dlogE_true_minus_perm'] for r in R])
    bmt = np.array([r['dlogE_best_minus_true'] for r in R])
    from scipy import stats
    t, p = stats.ttest_rel(pt, pp)
    print(f'\n=== TRUE-NODE RANK (n={len(R)}) ===')
    print(f'  true-node percentile : mean {pt.mean():.3f}  median {np.median(pt):.3f}')
    print(f'  PERM-node percentile : mean {pp.mean():.3f}  median {np.median(pp):.3f}')
    print(f'  paired diff          : {(pt-pp).mean():+.3f}  t {t:+.2f}  p {p:.4g}')
    print(f'  true - perm logE     : {tmp_.mean():+.1f} nats  '
          f'(>0 in {int((tmp_>0).sum())}/{len(R)})')
    print(f'  best - true logE     : {bmt.mean():.1f} nats (how far the argmax '
          f'outruns the truth)')
    print(f'  ESS of 2-lobe posterior: median {np.median(ess):.1f} '
          f'of {R[0]["n_hyp"]} hypotheses')
    print(f'  best-node d1 err {np.median([r["best_d1_err"] for r in R]):.1f} deg '
          f'| d2 err {np.median([r["best_d2_err"] for r in R]):.1f} deg')
    print(f'    (v2 top-K=6 seeding achieved d1 ~24 / d2 ~23.7 deg -- compare)')
    gf = np.array([r['sc_full'] - r['sc_cur'] for r in R])
    go = np.array([r['sc_node_oracle'] - r['sc_cur'] for r in R])
    seg = gf.std(ddof=1) / np.sqrt(len(gf))
    lo, hi = stats.t.interval(0.95, len(gf) - 1, loc=gf.mean(), scale=seg)
    print(f'\n=== ★ SCORED: full-enumeration argmax, point-committed ===')
    print(f'  FULL      - cur : {gf.mean():+.4f} +- {seg:.4f} '
          f'CI [{lo:+.4f},{hi:+.4f}]  win {int((gf>0).sum())}/{len(gf)}')
    print(f'  node-orc  - cur : {go.mean():+.4f}   (lattice-quantized oracle '
          f'= the ceiling this lattice can reach)')
    print(f'  e42/e43b blind arms were -0.017 / -0.014. e43 dose predicts '
          f'~+0.06 at d1~5/d2~10 deg.')
    # NOTE on the statistic. An earlier version gated on (pct_true - pct_perm),
    # which is WRONG: the percentile scale SATURATES. With the true node at
    # ~1.000 the difference cannot exceed (1 - pct_perm) no matter how large the
    # real separation is, so a decisive INFORMED surface was mislabelled FLAT.
    # The uncompressed scale is the log-evidence gap in NATS. Use it.
    tp_pos = float((tmp_ > 0).mean())
    if pt.mean() > 0.99 and tmp_.mean() > 5 and tp_pos >= 0.8:
        verd = ('INFORMED: the true node ranks at the very top of the '
                'enumeration and crushes the PERM null in nats -- the '
                'likelihood DOES locate theta. Any failure is the SEARCH or '
                'the FAMILY, not the information => route REOPENS')
    elif tmp_.mean() > 5 and tp_pos >= 0.8:
        verd = ('DECEIVED: true node beats PERM in nats but is not near the '
                'argmax -- information present, surface points wrong => bias, '
                'NOT an information wall')
    else:
        verd = ('FLAT: true node does not separate from a PERM node in nats -- '
                'the likelihood carries no usable information about theta => '
                'identifiability wall, properly established')
    print(f'\n  -> {verd}')
    json.dump(dict(rows=rows, verdict=verd,
                   pct_true_mean=float(pt.mean()), pct_perm_mean=float(pp.mean()),
                   paired_t=float(t), paired_p=float(p),
                   ess_median=float(np.median(ess))),
              open(OUT, 'w'), indent=1, default=float)
    print(f'saved -> {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
