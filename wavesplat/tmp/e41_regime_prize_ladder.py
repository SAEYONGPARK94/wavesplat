"""e41 — REGIME-AXIS PRIZE DECOMPOSITION (oracle ladder / ANOVA over ensemble
parameters).

CONTEXT: e37 established ens2_oracle - cur_grid = +0.098 +- 0.020 on cross-Tp
crossing seas (n=32). This script measures the TOTAL size of the regime axis
and which ensemble parameters carry it, on two arenas:

  ARENA X  cross-Tp (e37 build_sea_crosstp si=0..31 VERBATIM, comparability
           with the established +0.098). Varied ensemble params: d1, d2,
           ring2 (Tp2), w (r2).  Tp1=7, gamma=3.3, spread=30 are arena
           constants (their "knowledge" is degenerate here -> measured on Y).
  ARENA Y  open-bimodal (NEW, rng 11000+31*si): Tp1~U(5.5,9.5),
           gam_i~U(1.5,6) per lobe, spread_i~U(15,45) per lobe, plus the X
           params. Measures the FULL parameter set incl. ring1(kp), gamma,
           spread.

METHOD (pre-registered before running; smoke n=4 only for code debugging):
  For each parameter p: MAIN effect = score(all-marginal + p true) - score(all
  marginal); TOTAL effect = score(all true) - score(all true except p
  marginal). Marginals = honest prior-predictive smears over the arena's own
  generating distribution (quadrature mixtures), EXCEPT d1 whose blind default
  is mu_hat (beamform read-off, established free) and ring1 whose blind
  default is "peak at kp_hat" (two-stage kp, established free).
  Solver: solve_with_grid (e29_gate0_ladder), SMULT=3 FIXED, free arm only,
  no GT-peeked hyperparams. Score both cur_grid and cur_dep (standing rule).
  n=32 per arena (standing rule: n>=32).

Usage: python3 tmp/e41_regime_prize_ladder.py [--n 32] [--smoke] [--arena XY]
Output: results/direction3/e41_regime_prize_ladder.json
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

from e29_gennet_C import (deposit_grid, corr, DEPTH, _MASK, _KXN, _KYN,
                          incumbent_field, two_stage_kp, NGRID, KREL)
from e29_gate0_ladder import universal_radial, ensemble_grid, solve_with_grid
from e37_bimodal_ladder import (build_sea_crosstp, build_sea_bimodal,
                                net_gamma_map, mu_hat_of, ang_err_deg,
                                true_radial_grid)
from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, N_OBS)

OUT = 'results/direction3/e41_regime_prize_ladder.json'
G = 9.81

# ---- frozen marginal quadratures (the arena's own generating distribution)
D2_OFF_NODES = 60.0 + 90.0 * (np.arange(6) + 0.5) / 6.0     # U(60,150), 6/side
TP2_RATIO_NODES = 0.75 + 0.65 * (np.arange(7) + 0.5) / 7.0  # U(0.75,1.4), 7
W_MARG = 0.35                                               # E[U(0.2,0.5)]
GAM_NODES = 1.5 + 4.5 * (np.arange(3) + 0.5) / 3.0          # U(1.5,6), 3
SP_NODES = 15.0 + 30.0 * (np.arange(3) + 0.5) / 3.0         # U(15,45), 3
TP1_Y = (5.5, 9.5)
NANG = 720                                                  # 0.5deg angle grid


# ==========================================================================
#  fast vectorized deposit (verified against deposit_grid in selftest)
# ==========================================================================
def deposit_fast(kxs, kys, en, kp):
    sc = (NGRID - 1) / (2 * KREL)
    u = (np.asarray(kxs) / kp + KREL) * sc
    v = (np.asarray(kys) / kp + KREL) * sc
    ok = (u >= 0) & (u <= NGRID - 1) & (v >= 0) & (v <= NGRID - 1)
    u, v, e = u[ok], v[ok], np.asarray(en)[ok]
    i0 = np.floor(v).astype(int)
    j0 = np.floor(u).astype(int)
    fv, fu = v - i0, u - j0
    grid = np.zeros((NGRID, NGRID))
    for di, wv in ((0, 1 - fv), (1, fv)):
        for dj, wu in ((0, 1 - fu), (1, fu)):
            m = (i0 + di <= NGRID - 1) & (j0 + dj <= NGRID - 1)
            np.add.at(grid, (i0[m] + di, j0[m] + dj), e[m] * wv[m] * wu[m])
    return grid


# ==========================================================================
#  analytic lobe builders (radial mixture x angular mixture)
# ==========================================================================
_RAD_CACHE = {}


def radial_profile(Tp, gam):
    """(kmag, normalized energy) of JONSWAP(Tp, gam), ensemble radial
    (seed 1234, spread ~0), cached. Identical recipe to e37 true_radial."""
    key = (round(float(Tp), 4), round(float(gam), 3))
    if key not in _RAD_CACHE:
        c = jonswap_components(Hs=2.0, Tp=float(Tp), n_components=200,
                               depth=DEPTH, gamma=float(gam),
                               direction_deg=0.0, direction_spread_deg=1e-6,
                               seed=1234)
        kmag = np.array([np.hypot(x.kx, x.ky) for x in c])
        e = np.array([x.A for x in c]) ** 2 / 2
        _RAD_CACHE[key] = (kmag, e / e.sum())
    return _RAD_CACHE[key]


def radial_mix(specs):
    """specs: list of (Tp, gam, weight). Returns concatenated (kmag, en)."""
    ks, es = [], []
    wtot = sum(w for _, _, w in specs)
    for Tp, gam, w in specs:
        kmag, en = radial_profile(Tp, gam)
        ks.append(kmag)
        es.append(en * (w / wtot))
    return np.concatenate(ks), np.concatenate(es)


def angle_mix(specs):
    """specs: list of (mu_rad, spread_deg, weight). Wrapped-normal mixture on
    a global NANG grid. Returns (theta_nodes, weights) with weights sum 1."""
    th = -np.pi + (np.arange(NANG) + 0.5) * (2 * np.pi / NANG)
    w = np.zeros(NANG)
    for mu, sp, wt in specs:
        s = np.deg2rad(sp)
        d = np.angle(np.exp(1j * (th - mu)))
        g = np.exp(-0.5 * (d / s) ** 2)
        w += wt * g / g.sum()
    keep = w > w.max() * 1e-6
    return th[keep], w[keep] / w.sum()


def lobe_grid(kp, rad_specs, ang_specs):
    kmag, en = radial_mix(rad_specs)
    th, wth = angle_mix(ang_specs)
    KX = np.outer(kmag, np.cos(th)).ravel()
    KY = np.outer(kmag, np.sin(th)).ravel()
    E = np.outer(en, wth).ravel()
    return deposit_fast(KX, KY, E, kp)


def tp_of_k(k):
    """Tp whose spectral peak sits at wavenumber k (finite depth)."""
    om = np.sqrt(G * k * np.tanh(k * DEPTH))
    return 2 * np.pi / om


# ==========================================================================
#  ARENA Y builder — open bimodal cross-Tp
# ==========================================================================
def build_sea_openbi(si):
    rng = np.random.default_rng(11000 + 31 * si)
    Tp1 = rng.uniform(*TP1_Y)
    Hs = rng.uniform(1.5, 3.0)
    r2 = rng.uniform(0.2, 0.5)
    d1 = rng.uniform(0, 360)
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    d2 = d1 + sign * rng.uniform(60, 150)
    Tp2 = float(Tp1 * rng.uniform(0.75, 1.4))
    gam1, gam2 = rng.uniform(1.5, 6.0), rng.uniform(1.5, 6.0)
    sp1, sp2 = rng.uniform(15.0, 45.0), rng.uniform(15.0, 45.0)
    c = list(jonswap_components(Hs=Hs * np.sqrt(1 - r2), Tp=Tp1,
                                n_components=100, depth=DEPTH, gamma=gam1,
                                direction_deg=d1, direction_spread_deg=sp1,
                                seed=6000 + si))
    c += list(jonswap_components(Hs=Hs * np.sqrt(r2), Tp=Tp2,
                                 n_components=100, depth=DEPTH, gamma=gam2,
                                 direction_deg=d2, direction_spread_deg=sp2,
                                 seed=7000 + si))
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
                r2=float(r2), Tp1=float(Tp1), Tp2=Tp2,
                gam1=float(gam1), gam2=float(gam2),
                sp1=float(sp1), sp2=float(sp2), U=float(U))


# ==========================================================================
#  ARENA X ladder (params: d1, d2, ring2, w; Tp1/gam/sp arena constants)
# ==========================================================================
def score_x(S):
    kp = S['kp']
    mu_hat = mu_hat_of(S)
    d1, d2, r2, Tp2 = S['d1'], S['d2'], S['r2'], S['Tp2']

    L1 = lambda mu: lobe_grid(kp, [(7.0, 3.3, 1.0)], [(mu, 30.0, 1.0)])

    def L2(center=None, mu2=None, ring_true=False):
        """lobe 2: angle true (mu2) or smeared around center; ring true or
        smeared over the Tp2 ratio prior."""
        rad = [(Tp2, 3.3, 1.0)] if ring_true else \
            [(7.0 * r, 3.3, 1.0) for r in TP2_RATIO_NODES]
        if mu2 is not None:
            ang = [(mu2, 30.0, 1.0)]
        else:
            ang = [(center + s * np.deg2rad(o), 30.0, 1.0)
                   for s in (+1, -1) for o in D2_OFF_NODES]
        return lobe_grid(kp, rad, ang)

    row = dict(Hs=S['Hs'], r2=r2, Tp2=Tp2, dsep=ang_err_deg(d1, d2),
               mu_hat=mu_hat, mu_err=ang_err_deg(mu_hat, d1))
    Y = net_gamma_map(S)
    row['cur_grid'] = solve_with_grid(S, np.maximum(10.0 ** Y, 1e-8))
    f_dep = incumbent_field(S['P'], S['z'], S['Pq'], kp, S['sigma0'])
    row['cur_dep'] = corr(f_dep, S['gt_u'])
    row['ens1'] = solve_with_grid(S, L1(mu_hat))
    row['uni_d1'] = solve_with_grid(S, L1(d1))
    wm = W_MARG
    row['m2'] = solve_with_grid(S, (1 - wm) * L1(mu_hat) + wm * L2(mu_hat))
    # main effects (one true, rest marginal)
    row['M_d1'] = solve_with_grid(S, (1 - wm) * L1(d1) + wm * L2(d1))
    row['M_d2'] = solve_with_grid(S, (1 - wm) * L1(mu_hat) + wm * L2(mu2=d2))
    row['M_ring'] = solve_with_grid(
        S, (1 - wm) * L1(mu_hat) + wm * L2(mu_hat, ring_true=True))
    row['M_w'] = solve_with_grid(S, (1 - r2) * L1(mu_hat) + r2 * L2(mu_hat))
    # total effects (all true except one)
    o_l1t = L1(d1)
    o_l2t = L2(mu2=d2, ring_true=True)
    row['T_d1'] = solve_with_grid(
        S, (1 - r2) * L1(mu_hat) + r2 * L2(mu2=d2, ring_true=True))
    row['T_d2'] = solve_with_grid(S, (1 - r2) * o_l1t + r2 * L2(d1, ring_true=True))
    row['T_ring'] = solve_with_grid(S, (1 - r2) * o_l1t + r2 * L2(mu2=d2))
    row['T_w'] = solve_with_grid(S, (1 - wm) * o_l1t + wm * o_l2t)
    row['oracle'] = solve_with_grid(S, (1 - r2) * o_l1t + r2 * o_l2t)
    # e37-convention oracle (anchor cross-check; should match 'oracle' closely)
    go = (1 - r2) * true_radial_grid(S, 7.0, d1) \
        + r2 * true_radial_grid(S, Tp2, d2)
    row['oracle_e37'] = solve_with_grid(S, go)
    gf = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, kp)
    row['real'] = solve_with_grid(S, gf)
    return row


# ==========================================================================
#  ARENA Y ladder (params: d1, d2, ring1, ring2, w, gam, sp)
# ==========================================================================
def score_y(S):
    kp = S['kp']
    mu_hat = mu_hat_of(S)
    d1, d2, r2 = S['d1'], S['d2'], S['r2']
    Tp1, Tp2 = S['Tp1'], S['Tp2']
    Tp1_hat = float(tp_of_k(kp))          # blind ring1 default: peak at kp_hat

    def lobe1(dir_true, ring_true, gam_true, sp_true):
        """Lobe 1: each param true or marginal (prior-predictive smear).
        Blind defaults (free info): direction=mu_hat, ring peak at kp_hat."""
        gams = [S['gam1']] if gam_true else GAM_NODES
        sps = [S['sp1']] if sp_true else SP_NODES
        Tps = [Tp1] if ring_true else [Tp1_hat]
        angs = [(d1 if dir_true else mu_hat, sp, 1.0) for sp in sps]
        rad = [(Tp, g, 1.0) for Tp in Tps for g in gams]
        return lobe_grid(kp, rad, angs)

    # NOTE on lobe-2 marginal center: the d2 prior is defined relative to d1.
    # In arms where d1 is MARGINAL the honest center is mu_hat; where d1 is
    # TRUE it is d1. Passed via `cen2` below.
    def two(d1_t, d2_t, r1_t, r2_t, w_t, g_t, s_t):
        w = r2 if w_t else W_MARG
        g1 = lobe1(d1_t, r1_t, g_t, s_t)
        # rebuild lobe2 with correct center
        cen = d1 if d1_t else mu_hat
        if d2_t:
            angs = [(d2, sp, 1.0) for sp in ([S['sp2']] if s_t else SP_NODES)]
        else:
            angs = [(cen + s * np.deg2rad(o), sp, 1.0)
                    for s in (+1, -1) for o in D2_OFF_NODES
                    for sp in ([S['sp2']] if s_t else SP_NODES)]
        base_Tp1 = Tp1 if r1_t else Tp1_hat
        Tps = [Tp2] if r2_t else [base_Tp1 * r for r in TP2_RATIO_NODES]
        gams = [S['gam2']] if g_t else GAM_NODES
        rad = [(Tp, g, 1.0) for Tp in Tps for g in gams]
        g2 = lobe_grid(kp, rad, angs)
        return (1 - w) * g1 + w * g2

    row = dict(Hs=S['Hs'], r2=r2, Tp1=Tp1, Tp2=Tp2,
               dsep=ang_err_deg(d1, d2), mu_err=ang_err_deg(mu_hat, d1))
    row['ring1_true_rel'] = float(
        (2 * np.pi / dispersion_lam(Tp1, DEPTH)) / kp)
    Y = net_gamma_map(S)
    row['cur_grid'] = solve_with_grid(S, np.maximum(10.0 ** Y, 1e-8))
    f_dep = incumbent_field(S['P'], S['z'], S['Pq'], kp, S['sigma0'])
    row['cur_dep'] = corr(f_dep, S['gt_u'])
    row['ens1'] = solve_with_grid(S, lobe1(False, False, False, False))
    F = dict(d1=False, d2=False, r1=False, r2f=False, w=False, g=False,
             s=False)

    def arm(**kw):
        a = dict(F)
        a.update(kw)
        return solve_with_grid(S, two(a['d1'], a['d2'], a['r1'], a['r2f'],
                                      a['w'], a['g'], a['s']))
    row['m2'] = arm()
    row['M_d1'] = arm(d1=True)
    row['M_d2'] = arm(d2=True)
    row['M_ring1'] = arm(r1=True)
    row['M_ring2'] = arm(r2f=True)
    row['M_w'] = arm(w=True)
    row['M_gam'] = arm(g=True)
    row['M_sp'] = arm(s=True)
    T = dict(d1=True, d2=True, r1=True, r2f=True, w=True, g=True, s=True)

    def armT(**kw):
        a = dict(T)
        a.update(kw)
        return solve_with_grid(S, two(a['d1'], a['d2'], a['r1'], a['r2f'],
                                      a['w'], a['g'], a['s']))
    row['T_d1'] = armT(d1=False)
    row['T_d2'] = armT(d2=False)
    row['T_ring1'] = armT(r1=False)
    row['T_ring2'] = armT(r2f=False)
    row['T_w'] = armT(w=False)
    row['T_gam'] = armT(g=False)
    row['T_sp'] = armT(s=False)
    row['oracle'] = armT()
    gf = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, kp)
    row['real'] = solve_with_grid(S, gf)
    return row


# ==========================================================================
def paired(rows, a, b):
    d = np.array([r[a] - r[b] for r in rows], float)
    m = np.isfinite(d)
    d = d[m]
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    return dict(mean=float(d.mean()), se=float(se),
                t=float(d.mean() / se) if se > 0 else float('nan'),
                lo=float(d.mean() - 1.96 * se),
                hi=float(d.mean() + 1.96 * se),
                win=int((d > 0).sum()), n=n)


def summarize(rows, params, tag):
    print(f"\n=== ARENA {tag} (n={len(rows)}) — arm means ===")
    arms = [k for k in rows[0] if k not in
            ('Hs', 'r2', 'Tp1', 'Tp2', 'dsep', 'mu_hat', 'mu_err', 'kp_ratio',
             'ring1_true_rel')]
    for k in arms:
        v = np.array([r[k] for r in rows], float)
        print(f"  {k:11s}: {np.nanmean(v):+.4f}")
    out = dict(arm_means={k: float(np.nanmean([r[k] for r in rows]))
                          for k in arms})
    print(f"\n--- axis size (paired vs cur) ---")
    for aa, bb in [('oracle', 'cur_grid'), ('oracle', 'cur_dep'),
                   ('m2', 'cur_grid'), ('ens1', 'cur_grid'),
                   ('oracle', 'm2'), ('oracle', 'ens1'),
                   ('real', 'oracle')]:
        c = paired(rows, aa, bb)
        out[f'{aa}-{bb}'] = c
        print(f"  {aa:9s} - {bb:9s}: {c['mean']:+.4f} +- {c['se']:.4f} "
              f"(t {c['t']:+.2f}, CI [{c['lo']:+.4f},{c['hi']:+.4f}], "
              f"win {c['win']}/{c['n']})")
    print(f"\n--- per-parameter prize (MAIN = M_p - m2 | TOTAL = oracle - T_p)"
          f" ---")
    mains, totals = {}, {}
    for p in params:
        cm = paired(rows, f'M_{p}', 'm2')
        ct = paired(rows, 'oracle', f'T_{p}')
        mains[p], totals[p] = cm, ct
        print(f"  {p:6s}: MAIN {cm['mean']:+.4f} +- {cm['se']:.4f} "
              f"(CI [{cm['lo']:+.4f},{cm['hi']:+.4f}]) | "
              f"TOTAL {ct['mean']:+.4f} +- {ct['se']:.4f} "
              f"(CI [{ct['lo']:+.4f},{ct['hi']:+.4f}])")
    sm = sum(c['mean'] for c in mains.values())
    st = sum(c['mean'] for c in totals.values())
    gap = paired(rows, 'oracle', 'm2')['mean']
    print(f"  sum(MAIN) {sm:+.4f} | sum(TOTAL) {st:+.4f} | "
          f"oracle-m2 {gap:+.4f}  (초과분 = 상호작용)")
    out['mains'], out['totals'] = mains, totals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--arena', default='XY')
    a = ap.parse_args()
    if a.smoke:
        a.n = 4
    t0 = time.time()
    res = {}
    if 'X' in a.arena:
        rows = []
        for si in range(a.n):
            S = build_sea_crosstp(si)
            r = score_x(S)
            rows.append(r)
            print(f"X{si:2d} r2 {r['r2']:.2f} Tp2 {r['Tp2']:.1f} "
                  f"| cur {r['cur_grid']:+.3f} ens1 {r['ens1']:+.3f} "
                  f"m2 {r['m2']:+.3f} orc {r['oracle']:+.3f} "
                  f"(e37 {r['oracle_e37']:+.3f}) real {r['real']:+.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        res['X_rows'] = rows
        res['X'] = summarize(rows, ['d1', 'd2', 'ring', 'w'], 'X (cross-Tp)')
    if 'Y' in a.arena:
        rows = []
        for si in range(a.n):
            S = build_sea_openbi(si)
            r = score_y(S)
            rows.append(r)
            print(f"Y{si:2d} Tp1 {r['Tp1']:.1f} Tp2 {r['Tp2']:.1f} "
                  f"r2 {r['r2']:.2f} | cur {r['cur_grid']:+.3f} "
                  f"ens1 {r['ens1']:+.3f} m2 {r['m2']:+.3f} "
                  f"orc {r['oracle']:+.3f} real {r['real']:+.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        res['Y_rows'] = rows
        res['Y'] = summarize(rows, ['d1', 'd2', 'ring1', 'ring2', 'w',
                                    'gam', 'sp'], 'Y (open bimodal)')
    json.dump(res, open(OUT, 'w'), indent=1, default=float)
    print(f"\nsaved -> {OUT}  ({time.time()-t0:.0f}s)")


def selftest():
    rng = np.random.default_rng(0)
    kx = rng.normal(0, 0.1, 500)
    ky = rng.normal(0, 0.1, 500)
    e = rng.uniform(0, 1, 500)
    g1 = deposit_grid(kx, ky, e, 0.08)
    g2 = deposit_fast(kx, ky, e, 0.08)
    err = np.abs(g1 - g2).max()
    print(f"deposit_fast vs deposit_grid: max abs err {err:.2e}")
    assert err < 1e-12
    # ensemble_grid vs lobe_grid single lobe consistency (quadrature diff)
    kmag, en = universal_radial()
    ga = ensemble_grid(kmag, en, 0.7, 0.08)
    gb = lobe_grid(0.08, [(7.0, 3.3, 1.0)], [(0.7, 30.0, 1.0)])
    ca = ga / ga.sum(); cb = gb / gb.sum()
    print(f"ensemble_grid vs lobe_grid: corr "
          f"{np.corrcoef(ca.ravel(), cb.ravel())[0,1]:.6f}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
        selftest()
    else:
        main()
