"""e49 — OPEN-ARENA (arena Y) BLIND: does the blind~=0 negative generalize off
the well-specified arena X?  See tmp/e49_open_blind_prereg.md for the frozen
pre-registration (arena, arms, decision rules, scope caveats).

WHY. Every blind run (e42/e43b/e43c/e43e/e46) used build_sea_crosstp = arena X,
which HARDCODES Tp1=7, gamma=3.3, spread=30. The e42 SCOPE note says arena Y
(open gamma/spread/Tp1 = misspecified) "is a separate run" -- never done. E-48's
persona debate flagged this as the one cheap, killer-class experiment: it decides
whether blind~=0 is a property of the observation class or an artifact of the
narrow arena. Arena Y here = the incumbent net's TRAINING distribution
(d3_prior_net_data: Tp~U(5,11), gamma~U(1.5,6), spread~U(15,45)), so the net is
IN-distribution (max-favourable baseline) while the fixed-shape blind enumeration
is misspecified (max-unfavourable). If the prize still exists and blind still
cannot recover it on the net's own home turf, the negative generalizes.

SCOPE CAVEAT (frozen): FULL is a FIXED-shape enumeration (gamma3.3/spread30), so a
non-recovery here is robustness of THIS estimator, not the full info-theoretic
closure (shape-marginalized blind / 5-dim CRB still owed).

Usage: python3 tmp/e49_open_arena_blind.py [--smoke] [--n 96] [--n_uni 24]
Out:   results/direction3/e49_open_blind.json
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

from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, N_OBS)
from e29_gennet_C import corr, incumbent_field, DEPTH, two_stage_kp
from e29_gate0_ladder import solve_with_grid
from e37_bimodal_ladder import mu_hat_of, net_gamma_map
from e41_regime_prize_ladder import lobe_grid
from e43e_fresh_battery import full_blind_grid
from e43d_gate import paired

OUT = 'results/direction3/e49_open_blind.json'


def build_sea_open(si, unimodal=False):
    """build_sea_crosstp with the SHAPE axes OPENED to the net's training
    distribution (Tp1, gamma, spread all drawn per-sea; both lobes share them).
    FRESH block: rng 29000+31*si, jonswap seeds 26000+si / 27000+si."""
    rng = np.random.default_rng(29000 + 31 * si)
    Tp1 = float(rng.uniform(5.0, 11.0))
    gam_j = float(rng.uniform(1.5, 6.0))
    spread = float(rng.uniform(15.0, 45.0))
    Hs = rng.uniform(1.5, 3.0)
    r2 = 0.0 if unimodal else rng.uniform(0.2, 0.5)
    d1 = rng.uniform(0, 360)
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    d2 = d1 + sign * rng.uniform(60, 150)
    Tp2 = float(np.clip(Tp1 * rng.uniform(0.6, 1.6), 4.0, 12.0))
    c = list(jonswap_components(Hs=Hs * np.sqrt(1 - r2), Tp=Tp1,
                                n_components=100, depth=DEPTH, gamma=gam_j,
                                direction_deg=d1, direction_spread_deg=spread,
                                seed=26000 + si))
    if r2 > 0:
        c += list(jonswap_components(Hs=Hs * np.sqrt(r2), Tp=Tp2,
                                     n_components=100, depth=DEPTH, gamma=gam_j,
                                     direction_deg=d2,
                                     direction_spread_deg=spread,
                                     seed=27000 + si))
    kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c])
    Am = np.array([x.A for x in c]); ph = np.array([x.phi for x in c])
    om = omega_of(kx, ky, DEPTH)
    lam = dispersion_lam(Tp1, DEPTH)
    cp = lam / Tp1
    T = 10 * Tp1
    fp = 0.18 * lam
    r_fp = fp / 2
    side = 2.0 * lam
    gx = np.linspace(0, side, 96)
    gy = np.linspace(0, side, 96)
    Xg, Yg = np.meshgrid(gx, gy)
    mu = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
    Xs, Ys = Xg[mu], Yg[mu]
    take = np.random.default_rng(0).choice(len(Xs), min(1200, len(Xs)),
                                           replace=False)
    Xs, Ys = Xs[take], Ys[take]
    gt_u = render_modes(kx, ky, Am, ph, DEPTH, Xs, Ys, T)
    Pq = np.column_stack([Xs, Ys, np.full(len(Xs), T)])
    U = rng.uniform(0.0, 1.5) * cp
    nfrac = rng.uniform(0.02, 0.09)
    P = sample_obs('fixed' if U < 1e-6 else 'moving', U, lam, cp, T, gx, gy,
                   r_fp, rng)
    psi = (kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2]
           - om[None] * P[:, 2:3] + ph[None])
    z = (Am[None] * np.cos(psi)).sum(1) + rng.normal(0, nfrac * Hs, N_OBS)
    kp = two_stage_kp(P, z)
    return dict(P=P, z=z, Pq=Pq, gt_u=gt_u, kp=kp, sigma0=float(nfrac * Hs),
                Hs=float(Hs), m_kx=kx, m_ky=ky, m_A=Am, m_ph=ph,
                d1=float(np.deg2rad(d1)), d2=float(np.deg2rad(d2)),
                r2=float(r2), Tp1=Tp1, Tp2=Tp2, gamma=gam_j, spread=spread,
                U=float(U))


def score_sea(S):
    """All arms use the sea's TRUE shape where they are meant to be oracles;
    FULL is the fixed-shape blind enumeration (misspecified in arena Y)."""
    kp, r2, sp, gm = S['kp'], S['r2'], S['spread'], S['gamma']
    mu_hat = mu_hat_of(S)
    ens1_g = lobe_grid(kp, [(S['Tp1'], gm, 1.0)], [(mu_hat, sp, 1.0)])
    orc_g = (1 - r2) * lobe_grid(kp, [(S['Tp1'], gm, 1.0)], [(S['d1'], sp, 1.0)])
    if r2 > 0:
        orc_g = orc_g + r2 * lobe_grid(kp, [(S['Tp2'], gm, 1.0)],
                                       [(S['d2'], sp, 1.0)])
    g_full, th = full_blind_grid(S)
    row = dict(
        cur_grid=solve_with_grid(S, np.maximum(10.0 ** net_gamma_map(S), 1e-8)),
        cur_dep=float(corr(incumbent_field(S['P'], S['z'], S['Pq'], kp,
                                           S['sigma0']), S['gt_u'])),
        ens1=solve_with_grid(S, ens1_g),
        oracle_reg=solve_with_grid(S, orc_g),
        FULL=solve_with_grid(S, g_full),
        mu_err=float(np.rad2deg(np.abs(np.angle(
            np.exp(1j * (mu_hat - S['d1'])))))),
        Tp1=S['Tp1'], Tp2=S['Tp2'], gamma=gm, spread=sp, r2=r2)
    return row


def report(rows, label, contrasts):
    print(f'\n=== {label} (n={len(rows)}) arm means ===')
    for k in ['cur_grid', 'cur_dep', 'ens1', 'oracle_reg', 'FULL']:
        print(f'  {k:11s}: {np.nanmean([r[k] for r in rows]):+.4f}')
    res = {}
    print('  --- paired contrasts ---')
    for aa, bb in contrasts:
        r = paired(rows, aa, bb)
        res[f'{aa}-{bb}'] = r
        star = '*' if (r['lo'] > 0 or r['hi'] < 0) else ' '
        print(f"  {aa:10s} - {bb:10s}: {r['mean']:+.4f} +- {r['se']:.4f} "
              f"t {r['t']:+.2f} CI [{r['lo']:+.4f},{r['hi']:+.4f}]{star} "
              f"win {r['win']}/{r['n']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n', type=int, default=96)
    ap.add_argument('--n_uni', type=int, default=24)
    a = ap.parse_args()
    if a.smoke:
        a.n, a.n_uni = 2, 2
    t0 = time.time()

    rows = []
    print(f'--- ARENA Y (open shape), FRESH si=600..{600+a.n-1} ---', flush=True)
    for si in range(600, 600 + a.n):
        try:
            S = build_sea_open(si)
            r = score_sea(S)
        except Exception as e:                       # MNAR: two_stage_kp etc.
            print(f"  Y{si} EXCLUDED ({type(e).__name__}: {e})", flush=True)
            continue
        r['si'] = si
        rows.append(r)
        if si % 10 == 0 or a.smoke:
            print(f"Y{si} muE {r['mu_err']:5.1f} | cur {r['cur_grid']:+.3f} "
                  f"ens1 {r['ens1']:+.3f} orc {r['oracle_reg']:+.3f} "
                  f"FULL {r['FULL']:+.3f} | F-cur {r['FULL']-r['cur_grid']:+.3f} "
                  f"orc-cur {r['oracle_reg']-r['cur_grid']:+.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    uni = []
    print(f'\n--- UNIMODAL do-no-harm (open shape, seeds 3900+j) ---', flush=True)
    for j in range(a.n_uni):
        try:
            S = build_sea_open(3900 + j, unimodal=True)
            r = score_sea(S)
        except Exception as e:
            print(f"  U{j} EXCLUDED ({type(e).__name__})", flush=True)
            continue
        r['j'] = j
        uni.append(r)

    C_MAIN = [('oracle_reg', 'ens1'), ('oracle_reg', 'cur_grid'),
              ('cur_grid', 'ens1'),
              ('FULL', 'cur_grid'), ('FULL', 'ens1'), ('FULL', 'cur_dep'),
              ('cur_dep', 'cur_grid')]
    res = report(rows, 'ARENA Y-open', C_MAIN)

    hu = paired(uni, 'FULL', 'cur_grid')
    print(f"\n=== UNIMODAL do-no-harm (n={hu['n']}) ===")
    print(f"  FULL - cur_grid: {hu['mean']:+.4f} +- {hu['se']:.4f} "
          f"CI [{hu['lo']:+.4f},{hu['hi']:+.4f}]  (bar >= -0.01)")

    # Verdict labels refined pre-run per independent audit, grounded in E-29
    # (const-net -0.0424: the in-dist net partially captures regime, so
    # oracle_reg-cur can be small even when partition info is real). PRIZE
    # primary = information prize oracle_reg-ens1. Bars/arena/arms/seeds unchanged.
    PI = res['oracle_reg-ens1']       # information prize (thesis primary)
    PD = res['oracle_reg-cur_grid']   # deployment-residual prize over the net
    NA = res['cur_grid-ens1']         # does the in-dist net already capture it?
    Bc = res['FULL-cur_grid']
    Be = res['FULL-ens1']
    prize_info = (PI['mean'] >= 0.03 and PI['lo'] > 0)
    prize_deploy = (PD['mean'] >= 0.03 and PD['lo'] > 0)
    net_already = (NA['mean'] > 0 and NA['lo'] > 0)
    blind_null = (Bc['hi'] < 0.03 and Be['hi'] < 0.03)
    blind_breaks = (
        (Bc['mean'] >= 0.03 and Bc['lo'] > 0 and Bc['win'] / Bc['n'] >= 0.60)
        or (Be['mean'] >= 0.03 and Be['lo'] > 0 and Be['win'] / Be['n'] >= 0.60))
    harm_ok = (hu['mean'] >= -0.01)
    print(f"\n=== VERDICT (pre-registered) ===")
    print(f"  PRIZE_INFO   orc-ens1 >=+0.03 CI excl0 : {prize_info}  "
          f"({PI['mean']:+.4f} CI[{PI['lo']:+.4f},{PI['hi']:+.4f}] n{PI['n']})")
    print(f"  PRIZE_DEPLOY orc-cur  >=+0.03 CI excl0 : {prize_deploy}  "
          f"({PD['mean']:+.4f} CI[{PD['lo']:+.4f},{PD['hi']:+.4f}] n{PD['n']})")
    print(f"  NET_ALREADY  cur-ens1 >0  CI excl0     : {net_already}  "
          f"({NA['mean']:+.4f} CI[{NA['lo']:+.4f},{NA['hi']:+.4f}])")
    print(f"  BLIND_NULL   F-cur & F-ens1 CI_hi<+0.03: {blind_null}  "
          f"(F-cur {Bc['mean']:+.4f} hi{Bc['hi']:+.4f}; F-ens1 {Be['mean']:+.4f} hi{Be['hi']:+.4f})")
    print(f"  BLIND_BREAKS F beats a baseline>=+.03 w60: {blind_breaks}")
    print(f"  do-no-harm   uni F-cur >= -0.01 (weak)  : {harm_ok}  "
          f"({hu['mean']:+.4f} CI[{hu['lo']:+.4f},{hu['hi']:+.4f}] n{hu['n']})")
    if blind_breaks:
        verdict = ('BREAKS: this fixed-shape blind enumeration recovers value in '
                   'the open arena -> the negative was arena-X specific; rethink')
    elif prize_info and blind_null:
        note = (' (in-dist net already captures it -> no residual deployable prize '
                'over net, but partition info is real)'
                if net_already and not prize_deploy else '')
        verdict = ('CONFIRMED: partition info has value in the open arena but this '
                   'fixed-shape blind enumeration cannot recover it -> negative '
                   'generalizes off arena X' + note)
    elif not prize_info:
        verdict = ('INCONCLUSIVE-NOPRIZE: true-shape partition knowledge does not '
                   'beat the no-partition prior (orc-ens1 CI incl 0) -> channel '
                   'value itself is regime-specific')
    else:
        verdict = 'INCONCLUSIVE / partial (report CIs; do not force a label)'
    print(f"  -> {verdict}")
    print("  R2 NOTE: FULL is fixed gamma3.3/spread30 while ens1/oracle_reg use "
          "TRUE shape -> FULL is shape-handicapped, biasing toward the negative "
          "(protects against a false BREAK).")
    print("  SCOPE: robustness of THIS blind estimator, not full info-closure "
          "(shape-marginalized blind / 5-dim CRB still owed).")

    json.dump(dict(rows=rows, uni=uni, contrasts=res, uni_FULL_cur=hu,
                   verdict=verdict, prize_info=bool(prize_info),
                   prize_deploy=bool(prize_deploy), net_already=bool(net_already),
                   blind_null=bool(blind_null), blind_breaks=bool(blind_breaks),
                   harm_ok=bool(harm_ok)),
              open(OUT, 'w'), indent=1, default=float)
    print(f"saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
