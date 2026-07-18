"""e52 — GUARD2 achievability: does the theta info-limit (E-51 CRB) actually CAP
field recovery at the net/blind level? Completes E-48 keystone 3 (model-class).

IDEA. A Cramer-Rao-efficient unbiased estimator has theta_hat ~ N(theta_true, CRB).
Build the ensemble field from theta_hat and score corr to GT -> E[corr] = the field
recovery an information-optimal estimator could reach (an UPPER bound, since a real
estimator's error >= CRB). Compare, on the SAME seas as E-51 (arena X, si=700..763):
  oracle_reg   field at TRUE theta                     (ceiling; the +0.069 prize)
  crb_limited  MC mean over theta_hat~N(theta,CRB)      (CRB-optimal achievable)
  cur          incumbent net                            (deployable baseline)
  ens1         1-lobe at mu_hat, true shape             (no-partition floor)
If crb_limited ~= cur << oracle_reg -> the net is near CRB-optimal and the oracle
prize sits ABOVE the information limit => blind~=net (E-49) because both are near the
info ceiling => the theta info-limit (esp. w) is the mechanism. Efficient: no 952k
blind re-run (E-49 already gave FULL~=cur in this arena).

PRE-REGISTERED (frozen):
  SEAS   arena X si=700..763 (n=64), SAME as E-51 (so CRB & fields are on one sea set).
  MC     M=32 draws theta_hat ~ N(theta_true, CRB); clip w in [0.01,0.99], Tp in [3,14].
  CONTRASTS (paired): oracle_reg-cur, crb_limited-cur, oracle_reg-crb_limited,
            crb_limited-ens1, oracle_reg-ens1.
  DECISION
    ACHIEVABILITY CONFIRMED (keystone3 model-class):
        prize exists    oracle_reg-cur >= +0.03, CI excl 0   AND
        net~=CRB-optimal |crb_limited-cur| : CI within +-0.03 (not decisively > net) AND
        gap>info-limit  oracle_reg-crb_limited >= +0.03, CI excl 0
      => net is near CRB-optimal; the oracle prize is above the info limit => the
         theta info-limit caps field recovery at ~net => explains blind~=net.
    NOT CONFIRMED:
        crb_limited-cur >= +0.03 CI excl 0 => a CRB-efficient estimator beats the net
        => blind~=net is partly estimator-suboptimality, not purely information.
  CAVEAT inherits E-51: Gaussian-approx (model-class, not random-phase) CRB.

Usage: python3 tmp/e52_crb_achievability.py [--smoke] [--n 64] [--M 32]
Out:   results/direction3/e52_crb_achievability.json
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

from e29_gennet_C import corr
from e29_gate0_ladder import solve_with_grid
from e37_bimodal_ladder import build_sea_crosstp, mu_hat_of, net_gamma_map
from e41_regime_prize_ladder import lobe_grid
from e42_detector_v2 import GAMMA, SPREAD
from e43d_gate import paired
from e51_crb import fim_crb

OUT = 'results/direction3/e52_crb_achievability.json'


def field_grid(kp, theta):
    d1, Tp1, d2, Tp2, w = theta
    return ((1 - w) * lobe_grid(kp, [(Tp1, GAMMA, 1.0)], [(d1, SPREAD, 1.0)])
            + w * lobe_grid(kp, [(Tp2, GAMMA, 1.0)], [(d2, SPREAD, 1.0)]))


def score_sea(S, M, rng):
    kp = S['kp']
    theta = [S['d1'], S.get('Tp1', 7.0), S['d2'], S['Tp2'], S['r2']]
    sc, cond, F = fim_crb(S, theta)
    CRB = np.linalg.inv(F)
    CRB = 0.5 * (CRB + CRB.T)                       # symmetrize
    try:
        L = np.linalg.cholesky(CRB + 1e-12 * np.eye(5))
    except np.linalg.LinAlgError:
        w_, v_ = np.linalg.eigh(CRB)
        L = v_ @ np.diag(np.sqrt(np.clip(w_, 0, None)))
    mu = mu_hat_of(S)
    row = dict(
        si=S.get('si'),
        oracle_reg=solve_with_grid(S, field_grid(kp, theta)),
        cur=solve_with_grid(S, np.maximum(10.0 ** net_gamma_map(S), 1e-8)),
        ens1=solve_with_grid(S, lobe_grid(kp, [(theta[1], GAMMA, 1.0)],
                                          [(mu, SPREAD, 1.0)])),
        r2=float(S['r2']), cond=float(cond))
    cs = []
    for _ in range(M):
        th = np.array(theta) + L @ rng.standard_normal(5)
        th[4] = float(np.clip(th[4], 0.01, 0.99))   # w
        th[1] = float(np.clip(th[1], 3.0, 14.0))    # Tp1
        th[3] = float(np.clip(th[3], 3.0, 14.0))    # Tp2
        cs.append(solve_with_grid(S, field_grid(kp, th.tolist())))
    row['crb_limited'] = float(np.mean(cs))
    row['crb_limited_sd'] = float(np.std(cs))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--M', type=int, default=32)
    a = ap.parse_args()
    n = 3 if a.smoke else a.n
    M = 6 if a.smoke else a.M
    rng = np.random.default_rng(0)
    t0 = time.time()

    rows = []
    print(f"--- ARENA X si=700..{700+n-1}, M={M} MC draws ---", flush=True)
    for si in range(700, 700 + n):
        try:
            S = build_sea_crosstp(si); S['si'] = si
            r = score_sea(S, M, rng)
        except Exception as e:
            print(f"  {si} EXCLUDED ({type(e).__name__}: {e})", flush=True)
            continue
        rows.append(r)
        if len(rows) % 10 == 1:
            print(f"  {si} oracle {r['oracle_reg']:+.3f} crb_lim "
                  f"{r['crb_limited']:+.3f} cur {r['cur']:+.3f} ens1 "
                  f"{r['ens1']:+.3f} | o-cur {r['oracle_reg']-r['cur']:+.3f} "
                  f"crb-cur {r['crb_limited']-r['cur']:+.3f} [{time.time()-t0:.0f}s]",
                  flush=True)

    print(f"\n=== ARENA X (n={len(rows)}) means ===")
    for k in ['oracle_reg', 'crb_limited', 'cur', 'ens1']:
        print(f"  {k:12s}: {np.mean([r[k] for r in rows]):+.4f}")
    res = {}
    print("  --- paired contrasts ---")
    for aa, bb in [('oracle_reg', 'cur'), ('crb_limited', 'cur'),
                   ('oracle_reg', 'crb_limited'), ('crb_limited', 'ens1'),
                   ('oracle_reg', 'ens1')]:
        r = paired(rows, aa, bb)
        res[f'{aa}-{bb}'] = r
        star = '*' if (r['lo'] > 0 or r['hi'] < 0) else ' '
        print(f"  {aa:11s} - {bb:11s}: {r['mean']:+.4f} +- {r['se']:.4f} "
              f"CI[{r['lo']:+.4f},{r['hi']:+.4f}]{star} win {r['win']}/{r['n']}")

    P = res['oracle_reg-cur']; Cc = res['crb_limited-cur']; G = res['oracle_reg-crb_limited']
    prize = (P['mean'] >= 0.03 and P['lo'] > 0)
    net_opt = (abs(Cc['mean']) <= 0.03 and Cc['lo'] > -0.03 and Cc['hi'] < 0.03)
    gap_info = (G['mean'] >= 0.03 and G['lo'] > 0)
    beats = (Cc['mean'] >= 0.03 and Cc['lo'] > 0)
    print(f"\n=== VERDICT (pre-registered) ===")
    print(f"  prize exists     oracle-cur >=+.03 CIexcl0 : {prize}  ({P['mean']:+.4f} CI[{P['lo']:+.4f},{P['hi']:+.4f}])")
    print(f"  net ~= CRB-opt   |crb_lim-cur| within +-.03: {net_opt}  ({Cc['mean']:+.4f} CI[{Cc['lo']:+.4f},{Cc['hi']:+.4f}])")
    print(f"  gap > info-limit oracle-crb_lim >=+.03 excl0: {gap_info}  ({G['mean']:+.4f} CI[{G['lo']:+.4f},{G['hi']:+.4f}])")
    print(f"  CRB-eff BEATS net crb_lim-cur >=+.03 excl0  : {beats}")
    if beats:
        verdict = ("NOT CONFIRMED: a CRB-efficient estimator beats the net "
                   "=> blind~=net is partly estimator-suboptimality, not pure info")
    elif prize and net_opt and gap_info:
        verdict = ("ACHIEVABILITY CONFIRMED (model-class): net ~= CRB-optimal and "
                   "the oracle prize is ABOVE the info limit => theta info-limit caps "
                   "field recovery at ~net => explains blind~=net (keystone3 model-class)")
    else:
        verdict = "PARTIAL / INCONCLUSIVE: report CIs, no forced label"
    print(f"  -> {verdict}")
    print("  CAVEAT: Gaussian-approx (model-class) CRB, not random-phase. E-49 gives FULL~=cur.")

    json.dump(dict(rows=rows, contrasts=res, verdict=verdict, M=M,
                   prize=bool(prize), net_opt=bool(net_opt), gap_info=bool(gap_info),
                   beats=bool(beats)), open(OUT, 'w'), indent=1, default=float)
    print(f"saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
