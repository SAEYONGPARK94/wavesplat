"""e43e — FRESH-BLOCK replication of the e43c/e43d REOPEN + do-no-harm.

WHY. e43c scored a blind full-enumeration point-commit at FULL - cur = +0.0671
and e43d completed the control set on the same block:
    FULL - cur_grid +0.0671 CI [+0.0325,+0.1016] win 32/48
    FULL - cur_dep  +0.0812 CI [+0.0483,+0.1142] win 37/48
    FULL - m2       +0.0697 CI [+0.0285,+0.1108] win 35/48   <- the GATE
=> e37's "direction REOPENS" bar passes on si=300..347.

Two things are owed before that is a claim:
  1. si=300..347 is a SEEN block -- e43c ran there, and its first 4 seas were
     smoke-inspected before the n=48 run. Forking paths. This script replicates
     on a FRESH block, si=400..447, with the battery FROZEN in advance.
  2. do-no-harm was never measured for the FULL arm. FULL is UNGATED -- it always
     commits to two lobes. On a unimodal sea that could be actively harmful (cf.
     PERM: a pointy prior pointed the wrong way is harmful). If FULL does no harm
     on unimodal seas, no detector/threshold is needed AT ALL, which would be a
     considerably stronger result than the gated pipelines of e42/e43b.

PRE-REGISTERED (frozen before this run):
  SEAS    arena X, FRESH si=400..447 (n=48). Burned blocks: 0..31, 100..163,
          200..263, 300..347.
  UNI     unimodal do-no-harm, FRESH seeds 3700+j (n=24).
  ARMS    cur_grid, cur_dep, m2, ens1, oracle_x, FULL (blind, ungated).
  REOPEN  FULL - cur_grid AND FULL - cur_dep each >= +0.03 with 95% t-CI
          excluding 0 and win-rate >= 60%, AND FULL - m2 CI excludes 0.
  HARM    unimodal FULL - cur_grid >= -0.01.
  FAIL    if the block does not replicate, e43c's REOPEN was a seed artifact.

Cost ~38 s/sea (952,560-hypothesis enumeration) => ~45 min.
Out: results/direction3/e43e_fresh_battery.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import corr, incumbent_field
from e29_gate0_ladder import solve_with_grid
from e37_bimodal_ladder import build_sea_crosstp, build_sea_bimodal, mu_hat_of
from e42_detector_v2 import JointEngine, DIRS, A_GRID, W_GRID, MIN_SEP_DEG
from e37_bimodal_ladder import net_gamma_map
from e43d_gate import x_marginals, paired

OUT = 'results/direction3/e43e_fresh_battery.json'


def full_blind_grid(S):
    """Blind full-enumeration 2-lobe point-commit. Uses ONLY S['P'], S['z'],
    S['kp'], S['sigma0'] -- no ground truth anywhere in this function."""
    eng = JointEngine(S)
    keys = [(round(float(x), 3), round(float(m), 5))
            for m in DIRS for x in A_GRID]
    eng.build_lobes(keys)
    lut = {k: eng._lobe[k] for k in keys}

    def idx_of(i_dir, i_a):
        return lut[(round(float(A_GRID[i_a]), 3), round(float(DIRS[i_dir]), 5))]

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
    w = float(W_GRID[jw])
    g = (1 - w) * eng.grids[idx_of(j1, ja1)] + w * eng.grids[idx_of(j2, ja2)]
    return g, dict(d1=float(DIRS[j1]), a1=float(A_GRID[ja1]),
                   d2=float(DIRS[j2]), a2=float(A_GRID[ja2]), w=w,
                   n_hyp=len(loge))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--n_uni', type=int, default=24)
    a = ap.parse_args()
    if a.smoke:
        a.n, a.n_uni = 2, 2
    t0 = time.time()

    rows = []
    print(f'--- ARENA X, FRESH si=400..{400+a.n-1} ---', flush=True)
    for si in range(400, 400 + a.n):
        S = build_sea_crosstp(si)
        g, th = full_blind_grid(S)
        r = dict(si=si, FULL=solve_with_grid(S, g), **{f'th_{k}': v
                                                       for k, v in th.items()})
        r['cur_grid'] = solve_with_grid(
            S, np.maximum(10.0 ** net_gamma_map(S), 1e-8))
        r['cur_dep'] = float(corr(incumbent_field(
            S['P'], S['z'], S['Pq'], S['kp'], S['sigma0']), S['gt_u']))
        for k, gg in x_marginals(S).items():
            r[k] = solve_with_grid(S, gg)
        rows.append(r)
        print(f"X{si} FULL {r['FULL']:+.3f} cur {r['cur_grid']:+.3f} "
              f"dep {r['cur_dep']:+.3f} m2 {r['m2']:+.3f} "
              f"orc {r['oracle_x']:+.3f} | d {r['FULL']-r['cur_grid']:+.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    uni = []
    print('--- UNIMODAL do-no-harm (FRESH seeds 3700+j), FULL is UNGATED ---',
          flush=True)
    for j in range(a.n_uni):
        S = build_sea_bimodal(3700 + j, unimodal=True)
        g, th = full_blind_grid(S)
        r = dict(j=j, FULL=solve_with_grid(S, g))
        r['cur_grid'] = solve_with_grid(
            S, np.maximum(10.0 ** net_gamma_map(S), 1e-8))
        r['ens1'] = solve_with_grid(S, __import__('e41_regime_prize_ladder')
                                    .lobe_grid(S['kp'], [(7.0, 3.3, 1.0)],
                                               [(mu_hat_of(S), 30.0, 1.0)]))
        uni.append(r)
        print(f"U{j:2d} FULL {r['FULL']:+.3f} cur {r['cur_grid']:+.3f} "
              f"d {r['FULL']-r['cur_grid']:+.3f} [{time.time()-t0:.0f}s]",
              flush=True)

    print(f'\n=== ARENA X (n={len(rows)}) arm means ===')
    for k in ['cur_grid', 'cur_dep', 'm2', 'ens1', 'FULL', 'oracle_x']:
        print(f'  {k:9s}: {np.mean([r[k] for r in rows]):+.4f}')

    print('\n=== PRE-REGISTERED (fresh block) ===')
    res = {}
    for aa, bb in [('FULL', 'cur_grid'), ('FULL', 'cur_dep'), ('FULL', 'm2'),
                   ('FULL', 'ens1'), ('m2', 'cur_grid'),
                   ('oracle_x', 'cur_grid')]:
        r = paired(rows, aa, bb)
        res[f'{aa}-{bb}'] = r
        star = '*' if (r['lo'] > 0 or r['hi'] < 0) else ' '
        print(f"  {aa:8s} - {bb:8s}: {r['mean']:+.4f} +- {r['se']:.4f} "
              f"t {r['t']:+.2f} CI [{r['lo']:+.4f},{r['hi']:+.4f}]{star} "
              f"win {r['win']}/{r['n']}")

    hu = paired(uni, 'FULL', 'cur_grid')
    res['uni_FULL-cur_grid'] = hu
    print(f"\n=== UNIMODAL do-no-harm (n={hu['n']}, FULL UNGATED) ===")
    print(f"  FULL - cur_grid: {hu['mean']:+.4f} +- {hu['se']:.4f} "
          f"CI [{hu['lo']:+.4f},{hu['hi']:+.4f}]  win {hu['win']}/{hu['n']}  "
          f"(bar >= -0.01)")

    A, B, G = res['FULL-cur_grid'], res['FULL-cur_dep'], res['FULL-m2']
    bar = lambda x: x['mean'] >= 0.03 and x['lo'] > 0 and x['win'] / x['n'] >= 0.60
    reopen = bar(A) and bar(B) and G['lo'] > 0
    harm_ok = hu['mean'] >= -0.01
    print('\n=== VERDICT (fresh block) ===')
    print(f"  vs cur_grid : {bar(A)}   vs cur_dep : {bar(B)}   "
          f"GATE vs m2 : {G['lo'] > 0}")
    print(f"  do-no-harm  : {harm_ok}")
    print(f"  -> {'REOPEN REPLICATES on a fresh block' if reopen else 'FAILED to replicate -> e43c was a seed artifact'}"
          f"{' (but UNGATED FULL harms unimodal -> a gate is required)' if reopen and not harm_ok else ''}")
    json.dump(dict(rows=rows, uni=uni, contrasts=res, reopen=bool(reopen),
                   harm_ok=bool(harm_ok)), open(OUT, 'w'), indent=1,
              default=float)
    print(f'saved -> {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
