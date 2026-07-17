"""e43b — v3 PIPELINE: point-commit at the REAL theta_hat + incumbent fallback.

WHY. e42 scored fit_bmaF (48+48-hypothesis BMA) and got -0.0166 -> pre-registered
DEAD. D1's rebuttal showed that verdict is a FALLBACK ARTIFACT and I verified it:
  fired (43/64): fit_bmaF - cur = +0.0001 +- 0.0204
  NOT fired (21): fit_bmaF - cur = -0.0509 +- 0.0282     <- fit_bmaF -> fit1 as p2->0
  fit1 - cur = -0.0587  (fit1 is a big loser; using it as the fallback is my bug)
  post-hoc HYBRID (fired->fit_bmaF, else cur) = +0.0001 +- 0.0136, CI [-0.027,+0.027]
    => fails PRIMARY (+0.03) but is NOT dead (CI upper +0.027 > +0.02 bar).

D1 further claims the value is destroyed between theta_hat and gamma_out (48-grid
mixing + p2 model-average + fallback), and their pilot has point-commit at v2's
*empirical* error levels earning +0.0306 +- 0.0283 (n=23, CI includes 0).

That pilot PERTURBS TRUE THETA by errors resampled from the empirical pool with
RANDOM SIGN and INDEPENDENTLY PER PARAMETER. Two things make it untrustworthy:
  (a) independence destroys the error CORRELATION -- when the detector locks the
      wrong lobe, d1 AND d2 are wrong together (that is what `swap` records).
      Independent redraw therefore never reproduces a coherent wrong-lobe lock,
      which is exactly the failure that matters. => optimistic.
  (b) perturbing truth != using theta_hat. theta_hat is the argmax of the
      evidence, i.e. chosen to FIT THE DATA; its error is not sign-random. This
      could cut either way (a best-fitting wrong theta may reconstruct better
      than truth-plus-noise at the same angular error).
Arguing about noise models is pointless when we can just USE THE REAL theta_hat.
That is this script. No perturbation, no resampling: the actual v2 fit output.

PRE-REGISTERED (frozen before the scored run):
  SEAS    arena X, FRESH si=200..263 (n=64). si=100..163 is BURNED -- D1's
          rebuttal inspected its fired/not-fired subgroups, and I inspected them
          too while verifying. Reusing it would be forking paths.
  TAU     frozen from results/direction3/e42_detector_v2_calibration.json (1.96).
          NOT re-derived, NOT tuned.
  ARMS    cur_grid   incumbent (to beat)
          m2         honest all-marginal (GATE control; cur ~= m2)
          oracle     ceiling / ANCHOR (must reproduce ~ +0.0715 on these seas)
          v3_point   fired -> POINT-COMMIT at theta_hat, else cur_grid   [PRIMARY]
          v3_always  point-commit on every sea (isolates the fallback's value)
          v3_top4    fired -> top-4 narrow mixture, else cur_grid (D1's variant)
          fit_bmaF   e42's arm, for continuity
  PRIMARY v3_point - cur_grid >= +0.03 AND paired 95% t-CI excludes 0 AND
          win-rate >= 60%
  GATE    AND v3_point - m2 : 95% t-CI excludes 0
  DEAD    95% CI upper < +0.02  -> the blind route is dead even point-committed,
          and D1's "value destroyed downstream" claim is refuted.
  HARM    unimodal block (fresh seeds 3500+j, n=24): v3_point - cur >= -0.01.
          Structurally protected: unimodal seas mostly do not fire -> cur_grid.
  NOTE    detector is UNQUALIFIED at v2 calibration (TPR 58% < 60%) =>
          DETECTOR-LIMITED label stands.

Usage: python3 tmp/e43b_v3_pipeline.py [--smoke] [--n 64]
Out:   results/direction3/e43b_v3_pipeline.json
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

from e29_gennet_C import corr
from e29_gate0_ladder import solve_with_grid
from e37_bimodal_ladder import (build_sea_crosstp, build_sea_bimodal,
                                mu_hat_of, net_gamma_map)
from e41_regime_prize_ladder import lobe_grid, tp_of_k, TP2_RATIO_NODES, \
    D2_OFF_NODES, W_MARG
from e42_detector_v2 import fit_joint, SPREAD, GAMMA
from e42_blind_recovery import x_marginals, mixture_of_fields

OUT = 'results/direction3/e43b_v3_pipeline.json'
CAL = 'results/direction3/e42_detector_v2_calibration.json'


def one_lobe(kp, a, mu):
    """Identical construction to e42_detector_v2.ring_lobe_atoms."""
    return lobe_grid(kp, [(float(tp_of_k(a * kp)), GAMMA, 1.0)],
                     [(mu, SPREAD, 1.0)])


def point_grid(S, F):
    """POINT-COMMIT at the argmax theta_hat -- no mixing at all."""
    kp = S['kp']
    return (1 - F['w']) * one_lobe(kp, F['a1'], F['d1']) \
        + F['w'] * one_lobe(kp, F['a2'], F['d2'])


def topk_grid(F, k=4):
    """Narrow top-k mixture over the fit's own hypothesis set."""
    w = np.asarray(F['mix_w'], float)
    idx = np.argsort(-w)[:k]
    ww = w[idx] / w[idx].sum()
    return np.tensordot(ww, np.stack([F['mix_grids'][j] for j in idx]), axes=1)


def score_sea(S, tau, unimodal=False):
    row = {}
    Y = net_gamma_map(S)
    row['cur_grid'] = solve_with_grid(S, np.maximum(10.0 ** Y, 1e-8))
    if not unimodal:
        for k, g in x_marginals(S).items():
            row[k] = solve_with_grid(S, g)

    F = fit_joint(S)
    row['dlogE'] = F['dlogE']
    row['p2'] = F['p2']
    fired = bool(F['dlogE'] >= tau)
    row['fired'] = fired

    pg = point_grid(S, F)
    row['v3_always'] = solve_with_grid(S, pg)
    row['v3_point'] = row['v3_always'] if fired else row['cur_grid']
    row['v3_top4'] = solve_with_grid(S, topk_grid(F)) if fired \
        else row['cur_grid']
    f = mixture_of_fields(S, F['mix_grids'], F['mix_w'])
    row['fit_bmaF'] = float(corr(f, S['gt_u'])) if f is not None else np.nan
    return row


def paired(rows, a, b):
    d = np.array([r[a] - r[b] for r in rows
                  if np.isfinite(r[a]) and np.isfinite(r[b])])
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t, p = stats.ttest_1samp(d, 0)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se)
    return dict(mean=float(d.mean()), se=float(se), t=float(t), p=float(p),
                lo=float(lo), hi=float(hi), win=int((d > 0).sum()), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--n_uni', type=int, default=24)
    a = ap.parse_args()
    if a.smoke:
        a.n, a.n_uni = 4, 2
    cal = json.load(open(CAL))
    if cal.get('smoke'):
        print('REFUSING: calibration is a smoke run.')
        sys.exit(1)
    tau = float(cal['tau'])
    print(f"frozen TAU = {tau:.2f} (QUALIFIED={cal['qualified']}, "
          f"TPR {cal['tpr']:.0%}) -> DETECTOR-LIMITED", flush=True)
    t0 = time.time()

    rows = []
    print(f'--- ARENA X, FRESH si=200..{200+a.n-1} ---', flush=True)
    for si in range(200, 200 + a.n):
        S = build_sea_crosstp(si)
        r = score_sea(S, tau)
        r['si'] = si
        rows.append(r)
        print(f"X{si} cur {r['cur_grid']:+.3f} m2 {r['m2']:+.3f} | "
              f"pt {r['v3_point']:+.3f} alw {r['v3_always']:+.3f} "
              f"t4 {r['v3_top4']:+.3f} bmaF {r['fit_bmaF']:+.3f} | "
              f"orc {r['oracle']:+.3f} [f{int(r['fired'])}] "
              f"[{time.time()-t0:.0f}s]", flush=True)

    uni = []
    print('--- UNIMODAL do-no-harm (fresh seeds 3500+j) ---', flush=True)
    for j in range(a.n_uni):
        S = build_sea_bimodal(3500 + j, unimodal=True)
        r = score_sea(S, tau, unimodal=True)
        r['j'] = j
        uni.append(r)
    hu = paired(uni, 'v3_point', 'cur_grid')
    nf = sum(r['fired'] for r in uni)

    arms = ['cur_grid', 'm2', 'ens1', 'v3_point', 'v3_always', 'v3_top4',
            'fit_bmaF', 'oracle']
    print(f'\n=== ARENA X (n={len(rows)}) arm means ===')
    for k in arms:
        print(f'  {k:10s}: {np.nanmean([r[k] for r in rows]):+.4f}')

    print('\n=== PRE-REGISTERED CONTRASTS ===')
    res = {}
    for aa, bb in [('oracle', 'cur_grid'), ('m2', 'cur_grid'),
                   ('v3_point', 'cur_grid'), ('v3_point', 'm2'),
                   ('v3_always', 'cur_grid'), ('v3_top4', 'cur_grid'),
                   ('fit_bmaF', 'cur_grid'), ('v3_point', 'fit_bmaF'),
                   ('v3_point', 'v3_top4')]:
        r = paired(rows, aa, bb)
        res[f'{aa}-{bb}'] = r
        star = '*' if (r['lo'] > 0 or r['hi'] < 0) else ' '
        print(f"  {aa:9s} - {bb:9s}: {r['mean']:+.4f} +- {r['se']:.4f} "
              f"t {r['t']:+.2f} CI [{r['lo']:+.4f},{r['hi']:+.4f}]{star} "
              f"win {r['win']}/{r['n']}")

    # fired / not-fired split (the thing that broke e42)
    print('\n=== fired / not-fired split ===')
    for lab, sub in [('fired', True), ('not fired', False)]:
        rs = [r for r in rows if r['fired'] == sub]
        if len(rs) < 3:
            continue
        pr = paired(rs, 'v3_always', 'cur_grid')
        print(f"  {lab:10s} n={pr['n']:2d}: v3_always - cur = {pr['mean']:+.4f} "
              f"+- {pr['se']:.4f}  win {pr['win']}")
        res[f'split_{lab.replace(" ","_")}'] = pr
    res['uni_v3_point-cur_grid'] = hu
    print(f"\n=== UNIMODAL do-no-harm (n={hu['n']}) ===")
    print(f"  v3_point - cur_grid: {hu['mean']:+.4f} +- {hu['se']:.4f} "
          f"CI [{hu['lo']:+.4f},{hu['hi']:+.4f}]  (bar >= -0.01)  "
          f"false-fire {nf}/{len(uni)}")

    P, G = res['v3_point-cur_grid'], res['v3_point-m2']
    primary = (P['mean'] >= 0.03 and P['lo'] > 0 and P['win'] / P['n'] >= 0.60)
    gate = G['lo'] > 0
    dead = P['hi'] < 0.02
    harm_ok = hu['mean'] >= -0.01
    anchor = res['oracle-cur_grid']['mean']
    print('\n=== VERDICT (DETECTOR-LIMITED) ===')
    print(f"  ANCHOR oracle-cur = {anchor:+.4f} (expect ~ +0.0715)")
    print(f"  PRIMARY (v3_point-cur >= +0.03, CI excl 0, win>=60%): {primary}")
    print(f"  GATE    (v3_point-m2 CI excludes 0)                 : {gate}")
    print(f"  DEAD    (CI upper < +0.02)                          : {dead}")
    print(f"  do-no-harm (unimodal >= -0.01)                      : {harm_ok}")
    if primary and gate:
        v = 'RECOVERY: point-commit beats net AND marginal -> blind route REOPENS'
    elif dead:
        v = 'DEAD: blind route dead even point-committed; D1 downstream-loss claim REFUTED'
    else:
        v = 'INCONCLUSIVE: lands between the bars'
    print(f'  -> {v}')
    json.dump(dict(rows=rows, uni=uni, contrasts=res, tau=tau, verdict=v,
                   primary=bool(primary), gate=bool(gate), dead=bool(dead),
                   harm_ok=bool(harm_ok), anchor=float(anchor),
                   qualified=cal['qualified']),
              open(OUT, 'w'), indent=1, default=float)
    print(f'saved -> {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
