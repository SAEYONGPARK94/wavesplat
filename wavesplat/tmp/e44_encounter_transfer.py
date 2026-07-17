"""e44 — ★ENCOUNTER TRANSFER: does the regime prize exist on the DEPLOYMENT task?

WHY THIS THREATENS THE WHOLE ARC. Every number in E-41..E-43 scores
corr(field, truth) on a domain-wide query grid at t=T. But this project's LOCKED
direction is "short-horizon encounter forecasting for control with a given
sensor" -- the field the controller actually needs is the elevation ALONG THE
PLATFORM TRACK over the next 0-10 s. Nobody ever checked that the +0.078 cross-Tp
prize exists on THAT functional.

Designer N2's argument for why this, and only this, is worth running: under the
arena's generative model, any blind z->field estimator is dominated by the Bayes
predictor, whose FORM IS the "infer theta then average" family. So "outside the
family" can only be (i) a worse approximation of the same function, (ii) the
realization channel -- CLOSED (RL deconvolution at infinite SNR + GT-peek ties
the incumbent, -0.0003), or (iii) external information. Even the non-Gaussian
structure (deterministic amplitudes, fixed frequency lattice) dies outside the
family: the frequency lattice is unobservable at this window (dOmega*T = 1.57 rad
<< 2*pi), and the constant-modulus 4th-order structure can only cash out by
sharpening p(theta|z), which is INSIDE the family.
=> No new architecture is warranted. But whether the SCORE FUNCTIONAL matches the
   deployment task is a property of the TASK, not of z, and no evidence
   computation can answer it. This can.

N2 probe (n=16, si=900..915, burned): field anchor orc-cur +0.056 +- 0.041
(reproduces the prize) vs encounter 0-10 s +0.021 +- 0.093 (UNRESOLVED).
Per-sea sd 0.372 measured => n=960 gives MDE ~ +-0.024 for ~5 min. Free.

PRE-REGISTERED (frozen before this run; N2's spec):
  SEAS   arena X, FRESH si=1000..1959 (n=960). Burned: 0..31, 100..163, 200..263,
         300..347, 400..447, 900..915.
  UNI    unimodal do-no-harm, fresh seeds 4000+j (n=240).
  QUERY  platform track extrapolated past T=70 s, t in (T, T+10], 40 points, no
         jitter; gt rendered from the true modes (same convention as gt_u).
  ARMS   cur   incumbent net
         m2    honest all-marginal  [MANDATORY: cur ~= m2, so beating cur without
               beating m2 recovers nothing]
         orc   supplied TRUE theta (the ceiling / the thing E-43 sells)
         buoy  supplied theta at buoy grade   (d+-5deg,  Tp+-5%,  r2+-0.05)
         ww3   supplied theta at forecast grade (d+-15deg, Tp+-10%, r2+-0.10)
  ANCHOR field-corr orc - cur must reproduce ~ +0.06..+0.08, else the script is
         wrong and nothing else is read.
  PASS   TRANSFER: enc10 orc - cur >= +0.03 AND 95% t-CI excludes 0 AND win >= 55%
         AND Wilcoxon p < 0.01 AND (orc - m2) clears the same bar.
  KILL   enc10 (orc - cur) 95% CI upper < +0.02
         => the +0.078 prize is an artifact of the field metric and does NOT
            exist on the deployment task
         => the whole regime arc is STOP-and-write, and E-43's external-info spec
            (buoy 89% / WW3 58%) is DEMOTED to a statement about a mapping rather
            than about control.
  NOTE   Neither outcome rescues any new architecture. This pre-registration has
         no survival interest for its author (N2) or for me.

CAVEAT (N2's, kept): corr over 40 smooth track points has effective dof ~5-10 and
heavy tails (measured d = -1.16). corr may not represent control utility (phase /
timing). A null here is evidence about corr-on-track, not proof of irrelevance.

Usage: python3 tmp/e44_encounter_transfer.py [--smoke] [--n 960]
Out:   results/direction3/e44_encounter_transfer.json
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
from e37_bimodal_ladder import build_sea_crosstp, build_sea_bimodal, net_gamma_map
from e43d_gate import x_marginals, paired
from e43_external_regime_dose import supplied_grid, perturb, BUNDLES
from e44_probe_N2_encounter_transfer import (track_future, render_at, N_ENC,
                                             T_END, HORIZON)

OUT = 'results/direction3/e44_encounter_transfer.json'


def field_of(S, g, Pq):
    """corr of the MMSE field under prior g, evaluated at query Pq."""
    import torch
    from e29_gennet_C import DEVICE, _MASK, design_t
    from e29_gate0_ladder import bank_atoms, SMULT
    P, z = S['P'], S['z']
    kp, s0 = S['kp'], S['sigma0']
    gg = np.concatenate([g[_MASK], np.zeros_like(g)[_MASK]])
    if gg.sum() <= 0:
        return None
    gg = gg / gg.sum()
    kxb, kyb, om = bank_atoms(kp)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    Phi = design_t(t(kxb), t(kyb), t(om), t(P))
    Phi_q = design_t(t(kxb), t(kyb), t(om), t(Pq))
    gam = t(np.concatenate([gg, gg]) * float(np.var(z)))
    K = (Phi * gam[None]) @ Phi.T + (SMULT * s0) ** 2 * torch.eye(
        len(z), device=DEVICE)
    w = Phi.T @ torch.linalg.solve(K, t(z))
    return ((Phi_q * gam[None]) @ w).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n', type=int, default=960)
    ap.add_argument('--n_uni', type=int, default=240)
    a = ap.parse_args()
    if a.smoke:
        a.n, a.n_uni = 8, 4
    t0 = time.time()
    rng_master = np.random.default_rng(4404)

    rows = []
    print(f'--- ARENA X, FRESH si=1000..{1000+a.n-1} (n={a.n}) ---', flush=True)
    for k, si in enumerate(range(1000, 1000 + a.n)):
        S = build_sea_crosstp(si)
        te = T_END + np.linspace(HORIZON / N_ENC, HORIZON, N_ENC)
        Pq_e = track_future(S, te)
        gt_e = render_at(S, Pq_e)
        M = x_marginals(S)
        rng = np.random.default_rng(94000 + si)
        grids = {'cur': np.maximum(10.0 ** net_gamma_map(S), 1e-8),
                 'm2': M['m2'],
                 'orc': supplied_grid(S, S['d1'], S['d2'], S['Tp2'], S['r2'])}
        for name, (sd, sT, sw) in BUNDLES.items():
            if name == 'coarse':
                continue
            d1, d2, Tp2, r2 = perturb(S, rng, sd=sd, sT=sT, sw=sw)
            grids[name] = supplied_grid(S, d1, d2, Tp2, r2)
        row = dict(si=si, U=float(S['U']))
        for nm, g in grids.items():
            row[f'fld_{nm}'] = solve_with_grid(S, g)
            fe = field_of(S, g, Pq_e)
            if fe is None:
                row[f'enc10_{nm}'] = row[f'enc05_{nm}'] = \
                    row[f'enc510_{nm}'] = np.nan
                continue
            h = N_ENC // 2
            row[f'enc10_{nm}'] = float(corr(fe, gt_e))
            row[f'enc05_{nm}'] = float(corr(fe[:h], gt_e[:h]))
            row[f'enc510_{nm}'] = float(corr(fe[h:], gt_e[h:]))
        rows.append(row)
        if k % 100 == 0 or a.smoke:
            print(f"  si{si} fld d {row['fld_orc']-row['fld_cur']:+.3f} | "
                  f"enc10 d {row['enc10_orc']-row['enc10_cur']:+.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    uni = []
    print(f'--- UNIMODAL do-no-harm (fresh 4000+j, n={a.n_uni}) ---', flush=True)
    for j in range(a.n_uni):
        S = build_sea_bimodal(4000 + j, unimodal=True)
        te = T_END + np.linspace(HORIZON / N_ENC, HORIZON, N_ENC)
        Pq_e = track_future(S, te)
        gt_e = render_at(S, Pq_e)
        g_cur = np.maximum(10.0 ** net_gamma_map(S), 1e-8)
        r = dict(j=j)
        for nm, g in [('cur', g_cur)]:
            fe = field_of(S, g, Pq_e)
            r[f'enc10_{nm}'] = float(corr(fe, gt_e))
        uni.append(r)

    print(f'\n=== ANCHOR: field-corr (must reproduce the +0.06..+0.08 prize) ===')
    an = paired(rows, 'fld_orc', 'fld_cur')
    print(f"  fld orc - cur : {an['mean']:+.4f} +- {an['se']:.4f} "
          f"CI [{an['lo']:+.4f},{an['hi']:+.4f}] win {an['win']}/{an['n']}")
    anchor_ok = 0.04 <= an['mean'] <= 0.11
    print(f'  anchor OK: {anchor_ok}')

    print(f'\n=== ★ENCOUNTER TASK (the deployment functional) ===')
    res = {'anchor_fld_orc-cur': an}
    for tag in ['enc10', 'enc05', 'enc510']:
        print(f'  -- {tag} --')
        for aa, bb in [('orc', 'cur'), ('orc', 'm2'), ('m2', 'cur'),
                       ('buoy', 'cur'), ('ww3', 'cur')]:
            r = paired(rows, f'{tag}_{aa}', f'{tag}_{bb}')
            res[f'{tag}_{aa}-{bb}'] = r
            star = '*' if (r['lo'] > 0 or r['hi'] < 0) else ' '
            print(f"    {aa:5s} - {bb:4s}: {r['mean']:+.4f} +- {r['se']:.4f} "
                  f"CI [{r['lo']:+.4f},{r['hi']:+.4f}]{star} "
                  f"win {r['win']}/{r['n']}")

    d = np.array([r['enc10_orc'] - r['enc10_cur'] for r in rows])
    d = d[np.isfinite(d)]
    try:
        w_p = float(stats.wilcoxon(d)[1])
    except Exception:
        w_p = float('nan')
    P = res['enc10_orc-cur']
    G = res['enc10_orc-m2']
    transfer = (P['mean'] >= 0.03 and P['lo'] > 0 and P['win'] / P['n'] >= 0.55
                and w_p < 0.01 and G['mean'] >= 0.03 and G['lo'] > 0)
    kill = P['hi'] < 0.02
    ratio = P['mean'] / an['mean'] if an['mean'] else float('nan')
    print(f'\n=== VERDICT ===')
    print(f"  Wilcoxon p (enc10 orc-cur) = {w_p:.3g}")
    print(f"  transfer ratio enc10/field = {ratio:.2f}")
    print(f"  TRANSFER (>=+0.03, CI excl 0, win>=55%, wilcox<0.01, "
          f"and orc-m2 same): {transfer}")
    print(f"  KILL (enc10 CI upper < +0.02): {kill}")
    if transfer:
        v = ('TRANSFERS: the regime prize exists on the deployment task -> '
             'E-43 external-info spec inherits control relevance')
    elif kill:
        v = ('KILLED: the prize is an artifact of the FIELD metric and does not '
             'exist on the deployment task -> whole regime arc is STOP-and-write; '
             'E-43 spec demoted to a statement about a mapping')
    else:
        v = 'INCONCLUSIVE between the bars'
    print(f'  -> {v}')
    print('  CAVEAT: corr over 40 smooth track points has eff. dof ~5-10; a null '
          'is evidence about corr-on-track, not proof of control-irrelevance.')
    json.dump(dict(rows=rows, uni=uni, contrasts=res, wilcoxon_p=w_p,
                   anchor_ok=bool(anchor_ok), transfer=bool(transfer),
                   kill=bool(kill), ratio=float(ratio), verdict=v),
              open(OUT, 'w'), indent=1, default=float)
    print(f'saved -> {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
