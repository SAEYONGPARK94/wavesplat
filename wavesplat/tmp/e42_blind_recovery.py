"""e42 — BLIND RECOVERY: does an exact-evidence joint fit recover any of the
+0.098 cross-Tp prize?

WHY THIS DECIDES MORE THAN ITSELF. All three competing E-42 network designs
(tmp/e42_design_D{1,2,3}.md) reduce to the same recipe: infer a joint posterior
over low-dim regime hypotheses theta=(d1,a1,d2,a2,w), then average. The
evidence fit in e42_detector_v2 IS that recipe, with an EXACT likelihood, a
WELL-SPECIFIED model in arena X, and zero learning -- i.e. the best case any
learned version could hope to reach. If exact evidence cannot recover theta
well enough to earn here, all three learned designs are refuted before a
single training step.

ARMS (all scored through the identical solver path, solve_with_grid):
  cur_grid   incumbent net (prior_net_l1s2bw) -- the thing to beat
  m2         honest all-marginal              -- MANDATORY control. cur ~= m2
                                                 (E-41: -0.005), so beating cur
                                                 WITHOUT beating m2 = recovering
                                                 no information. This is the
                                                 primary defense against
                                                 repeating E-37 (measuring our
                                                 own detector's harm/help and
                                                 calling it information).
  ens1       1-lobe at mu_hat                 -- beamform-only reference
  oracle     true (d1,d2,ring,w)              -- the ceiling; re-establishes
                                                 that the prize EXISTS on these
                                                 fresh seas
  fit1       global 1-lobe evidence BMA       -- threshold-free
  fit2_hard  2-lobe BMA gated at frozen TAU   -- v2 calibration TAU only. The
                                                 post-hoc TAU=3.5 of v1 is
                                                 contaminated and forbidden.
  fit_bma    grid-space BMA, threshold-free
  fit_bmaF   MIXTURE OF FIELDS                -- D2's point: mixing gamma GRIDS
                                                 averages covariances and stays
                                                 inside the incumbent's
                                                 single-gamma class; mixing
                                                 FIELDS is the actual posterior
                                                 mean E[f|z] = sum_k p(k|z)
                                                 E[f|z,k] and is outside it.
  U_BMA      uniform weights, same hypotheses -- control: is any gain from the
                                                 EVIDENCE weighting, or merely
                                                 from the 2-lobe family?

PRE-REGISTERED (frozen before the scored run):
  SEAS     arena X, FRESH si=100..163 (n=64). si=0..31 is the discovery set
           (the +0.098 and the E-41 ladder were measured there) -> forking
           paths; not used for the verdict. n=32 gives paired SE ~0.020 =>
           MDE ~+0.06, too coarse for a +0.03 bar; n=64 gives SE ~0.014.
  PRIMARY  fit_bmaF - cur_grid >= +0.03 AND paired 95% t-CI excludes 0 AND
           win-rate >= 60%   [the e37 pre-registered bar, kept]
  GATE     AND fit_bmaF - m2 : 95% t-CI excludes 0.  Without this, a "win" can
           be pure smear-matching and recovers nothing.
  DEAD     95% CI upper bound < +0.02  -> the evidence route is dead, and with
           it the shared core of D1/D2/D3.
  HARM     unimodal do-no-harm block (n=24): fit_bmaF - cur_grid >= -0.01.
  NOTE     Detector is UNQUALIFIED at v2 calibration (TPR 58% < 60% bar;
           median |d2 err| 23.7 deg PASSES the 25 deg bar) -> every 2-lobe
           result here is labeled DETECTOR-LIMITED.
  SCOPE    Arena X hardcodes Tp1=7, gamma=3.3, spread=30, and the detector
           assumes gamma=3.3/spread=30 => WELL-SPECIFIED here. A win in X alone
           is NOT a reopen; it is regime specialization until arena Y (open
           gamma/spread/Tp1 = misspecified) reproduces it. Y is a separate run.

Usage: python3 tmp/e42_blind_recovery.py [--smoke] [--n 64]
Out:   results/direction3/e42_blind_recovery.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import DEVICE, DEPTH, _MASK, corr
from e29_gate0_ladder import solve_with_grid, bank_atoms, SMULT
from e29_gennet_C import design_t
from e37_bimodal_ladder import (build_sea_crosstp, build_sea_bimodal,
                                mu_hat_of, ang_err_deg, net_gamma_map)
from e41_regime_prize_ladder import lobe_grid, TP2_RATIO_NODES, D2_OFF_NODES, W_MARG
from e42_detector_v2 import fit_joint, match_errors

OUT = 'results/direction3/e42_blind_recovery.json'
CAL = 'results/direction3/e42_detector_v2_calibration.json'


def field_with_grid(S, g_free_grid):
    """Identical to solve_with_grid but returns the FIELD, not its corr.
    Needed for the mixture-of-fields arm."""
    P, z, Pq = S['P'], S['z'], S['Pq']
    kp, s0 = S['kp'], S['sigma0']
    gb = np.zeros_like(g_free_grid)
    g = np.concatenate([g_free_grid[_MASK], gb[_MASK]])
    if g.sum() <= 0:
        return None
    g = g / g.sum()
    kxb, kyb, om = bank_atoms(kp)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    Phi = design_t(t(kxb), t(kyb), t(om), t(P))
    Phi_q = design_t(t(kxb), t(kyb), t(om), t(Pq))
    gam = t(np.concatenate([g, g]) * float(np.var(z)))
    K = (Phi * gam[None]) @ Phi.T + (SMULT * s0) ** 2 * torch.eye(
        len(z), device=DEVICE)
    w = Phi.T @ torch.linalg.solve(K, t(z))
    return ((Phi_q * gam[None]) @ w).cpu().numpy()


def mixture_of_fields(S, grids, weights, tol=1e-3):
    """E[f|z] = sum_k p(k|z) E[f|z,k]. Prunes hypotheses below tol mass."""
    w = np.asarray(weights, float)
    keep = np.where(w > tol * w.max())[0]
    w = w[keep] / w[keep].sum()
    acc = None
    for wk, j in zip(w, keep):
        f = field_with_grid(S, grids[j])
        if f is None:
            continue
        acc = wk * f if acc is None else acc + wk * f
    return acc


def x_marginals(S):
    """Arena-X marginal grids, VERBATIM from e41 score_x (same convention)."""
    kp = S['kp']
    mu_hat = mu_hat_of(S)
    L1 = lambda mu: lobe_grid(kp, [(7.0, 3.3, 1.0)], [(mu, 30.0, 1.0)])

    def L2(center=None, mu2=None, ring_true=False):
        rad = [(S['Tp2'], 3.3, 1.0)] if ring_true else \
            [(7.0 * r, 3.3, 1.0) for r in TP2_RATIO_NODES]
        ang = [(mu2, 30.0, 1.0)] if mu2 is not None else \
            [(center + s * np.deg2rad(o), 30.0, 1.0)
             for s in (+1, -1) for o in D2_OFF_NODES]
        return lobe_grid(kp, rad, ang)

    wm = W_MARG
    r2 = S['r2']
    return dict(
        ens1=L1(mu_hat),
        m2=(1 - wm) * L1(mu_hat) + wm * L2(mu_hat),
        oracle=(1 - r2) * L1(S['d1']) + r2 * L2(mu2=S['d2'], ring_true=True),
    )


def score_sea(S, tau, unimodal=False):
    row = {}
    Y = net_gamma_map(S)
    row['cur_grid'] = solve_with_grid(S, np.maximum(10.0 ** Y, 1e-8))
    if not unimodal:
        M = x_marginals(S)
        for k, g in M.items():
            row[k] = solve_with_grid(S, g)
    else:
        row['ens1'] = solve_with_grid(S, lobe_grid(
            S['kp'], [(7.0, 3.3, 1.0)], [(mu_hat_of(S), 30.0, 1.0)]))

    F = fit_joint(S)
    row['dlogE'] = F['dlogE']
    row['p2'] = F['p2']
    row['fired'] = bool(F['dlogE'] >= tau)
    row['fit1'] = solve_with_grid(S, F['fit1_grid'])
    row['fit_bma'] = solve_with_grid(S, F['bma_grid'])
    row['fit2_hard'] = solve_with_grid(
        S, F['fit2_grid'] if row['fired'] else F['fit1_grid'])

    # mixture of FIELDS (outside the single-gamma class)
    f = mixture_of_fields(S, F['mix_grids'], F['mix_w'])
    row['fit_bmaF'] = float(corr(f, S['gt_u'])) if f is not None else float('nan')

    # uniform-weight control over the SAME hypothesis set
    row['U_BMA'] = solve_with_grid(
        S, np.mean(np.stack(F['mix_grids']), axis=0))

    if not unimodal:
        E = match_errors(F, S)
        row.update({f'e_{k}': v for k, v in E.items()})
        row['mu_err'] = ang_err_deg(F['mu_hat'], S['d1'])
        row['r2'] = S['r2']
        row['Tp2'] = S['Tp2']
    return row


def paired(rows, a, b):
    d = np.array([r[a] - r[b] for r in rows if np.isfinite(r[a])
                  and np.isfinite(r[b])])
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
    tau = float(cal['tau'])
    if cal.get('smoke'):
        print('REFUSING: calibration JSON is a smoke run. Run the full '
              'tmp/e42_detector_v2.py first.')
        sys.exit(1)
    print(f"frozen TAU = {tau:.2f}  (detector QUALIFIED={cal['qualified']}"
          f", TPR {cal['tpr']:.0%})", flush=True)
    t0 = time.time()

    rows = []
    print(f'--- ARENA X, FRESH si=100..{100+a.n-1} ---', flush=True)
    for si in range(100, 100 + a.n):
        S = build_sea_crosstp(si)
        r = score_sea(S, tau)
        r['si'] = si
        rows.append(r)
        print(f"X{si} muE {r['mu_err']:5.1f} | cur {r['cur_grid']:+.3f} "
              f"m2 {r['m2']:+.3f} ens1 {r['ens1']:+.3f} | f1 {r['fit1']:+.3f} "
              f"bma {r['fit_bma']:+.3f} bmaF {r['fit_bmaF']:+.3f} "
              f"U {r['U_BMA']:+.3f} | orc {r['oracle']:+.3f} "
              f"[fired {int(r['fired'])}] [{time.time()-t0:.0f}s]", flush=True)

    uni = []
    print(f'--- UNIMODAL do-no-harm block (seeds 3400+j) ---', flush=True)
    for j in range(a.n_uni):
        S = build_sea_bimodal(3400 + j, unimodal=True)
        r = score_sea(S, tau, unimodal=True)
        r['j'] = j
        uni.append(r)
        print(f"U{j:2d} cur {r['cur_grid']:+.3f} bmaF {r['fit_bmaF']:+.3f} "
              f"[fired {int(r['fired'])}] [{time.time()-t0:.0f}s]", flush=True)

    arms = ['cur_grid', 'm2', 'ens1', 'fit1', 'fit2_hard', 'fit_bma',
            'fit_bmaF', 'U_BMA', 'oracle']
    print(f'\n=== ARENA X (n={len(rows)}) arm means ===')
    for k in arms:
        v = np.array([r[k] for r in rows])
        print(f'  {k:10s}: {np.nanmean(v):+.4f}')

    print(f'\n=== PRE-REGISTERED CONTRASTS ===')
    res = {}
    for aa, bb in [('oracle', 'cur_grid'), ('m2', 'cur_grid'),
                   ('fit_bmaF', 'cur_grid'), ('fit_bmaF', 'm2'),
                   ('fit_bma', 'cur_grid'), ('fit_bma', 'm2'),
                   ('fit1', 'cur_grid'), ('fit2_hard', 'cur_grid'),
                   ('fit_bmaF', 'U_BMA'), ('fit_bmaF', 'fit_bma'),
                   ('fit1', 'ens1'), ('U_BMA', 'cur_grid')]:
        r = paired(rows, aa, bb)
        res[f'{aa}-{bb}'] = r
        star = '*' if (r['lo'] > 0 or r['hi'] < 0) else ' '
        print(f"  {aa:10s} - {bb:9s}: {r['mean']:+.4f} +- {r['se']:.4f} "
              f"t {r['t']:+.2f} CI [{r['lo']:+.4f},{r['hi']:+.4f}]{star} "
              f"win {r['win']}/{r['n']}")

    hu = paired(uni, 'fit_bmaF', 'cur_grid')
    res['uni_fit_bmaF-cur_grid'] = hu
    print(f"\n=== UNIMODAL do-no-harm (n={hu['n']}) ===")
    print(f"  fit_bmaF - cur_grid: {hu['mean']:+.4f} +- {hu['se']:.4f} "
          f"CI [{hu['lo']:+.4f},{hu['hi']:+.4f}]  (bar >= -0.01)")
    nfire = sum(r['fired'] for r in uni)
    print(f"  unimodal false-fire rate: {nfire}/{len(uni)} = {nfire/len(uni):.0%}")

    P = res['fit_bmaF-cur_grid']
    G = res['fit_bmaF-m2']
    win_ok = P['win'] / P['n'] >= 0.60
    primary = (P['mean'] >= 0.03 and P['lo'] > 0 and win_ok)
    gate = G['lo'] > 0
    dead = P['hi'] < 0.02
    print(f"\n=== VERDICT (detector UNQUALIFIED -> DETECTOR-LIMITED) ===")
    print(f"  PRIMARY (fit_bmaF-cur >= +0.03, CI excl 0, win>=60%): {primary}")
    print(f"  GATE    (fit_bmaF-m2 CI excludes 0)                 : {gate}")
    print(f"  DEAD    (CI upper < +0.02)                          : {dead}")
    if primary and gate:
        v = 'RECOVERY: blind evidence beats the net AND beats the marginal'
    elif dead:
        v = 'DEAD: evidence route refuted -> shared core of D1/D2/D3 refuted'
    else:
        v = 'INCONCLUSIVE / partial'
    print(f"  -> {v}")
    print(f"  SCOPE: arena X is WELL-SPECIFIED for this detector. A win here "
          f"is specialization until arena Y reproduces it.")

    json.dump(dict(rows=rows, uni=uni, contrasts=res, tau=tau,
                   qualified=cal['qualified'], verdict=v,
                   primary=bool(primary), gate=bool(gate), dead=bool(dead)),
              open(OUT, 'w'), indent=1, default=float)
    print(f"saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
