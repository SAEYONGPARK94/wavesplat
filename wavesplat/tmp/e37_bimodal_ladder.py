"""e37 — BIMODAL LADDER: the one untested surviving path.

CONTEXT (verified from raw JSON before this script was written):
  e30: incumbent == constant prior + direction read-off on the unimodal arena
       (B-A = -0.0129 +- 0.0242, 6/16; direction read-off B-C = +0.2352).
  e36 (script lost, JSON verified): bimodal arena, own +0.3078 / mean +0.3138
       / ens(1-lobe) +0.3016 / real +0.4818 -> oracle prize +0.174, net takes 0%.
  GAP: nobody built a CORRECT 2-lobe hand prior from the map's two visible
       lobes. Two-lobe structure is an ENSEMBLE-level feature (unlike
       realization speckle) so it should be representable AND inferable.

PRE-REGISTERED DESIGN (frozen before the full run; smoke n=4 for code
debugging only, no score-driven tuning of detector or arena):
  ARENA  n=32 bimodal seas, si=0..31, rng(9100+31*si):
     Tp1=Tp2=7.0 (shared radial profile -> direction structure is the ONLY
     difference between 1-lobe and 2-lobe hand priors), Hs_tot~U(1.5,3.0),
     r2~U(0.2,0.5) secondary energy share, d1~U(0,360),
     d2=d1+sign*U(60,150) deg, spread 30 deg each, 100+100 components
     (seeds 4000+si / 5000+si), U~U(0,1.5)*cp, nfrac~U(0.02,0.09),
     NO bound waves. GT at 1200 held-out pixels, corr metric.
     + 8 unimodal control seas (r2=0) for detector false-positive rate.
  SOLVER identical for every gamma arm: solve_with_grid (e29_gate0_ladder),
     SMULT=3.0 FIXED, blur 0, free arm only, no GT-peeked hyperparams.
  ARMS
     cur_grid    net's own gamma through the ladder solver (e30 'own' recipe)
     cur_dep     deployed incumbent path (net_gamma_mmse + SmultHead + bw arm)
     mean        net's across-sea MEAN aligned log-gamma map (recomputed on
                 the 16 unimodal e30 seas), rotated to this sea's mu_hat
     ens1        hand 1-lobe: universal radial x N(mu_hat, 30 deg)
     ens2_blind  hand 2-lobe from the DATA-ONLY map detector (below)
     ens2_oracle hand 2-lobe at TRUE directions d1,d2, TRUE weights (1-r2,r2)
     real        oracle realization deposit (true A^2/2), same fixed solver
  DETECTOR (frozen): pm=min(p1,p2) map channel; radial shell k/kp in
     [0.6,1.6]; 72-bin angular energy profile; circular Gaussian smooth
     sigma=10 deg; theta1=argmax; suppress +-45 deg; theta2=argmax of rest;
     ACCEPT second lobe iff prof(theta2) >= 0.25*prof(theta1) (else fall
     back to ens1); lobe angle refined by circular mean in +-15 deg window;
     weights w_i proportional to prof(theta_i), normalized.
  DECISION RULE (pre-registered):
     WIN (direction REOPENS): mean(ens2_blind - cur) >= +0.03 with paired
        95% CI excluding 0 AND win-rate >= 60%, against BOTH cur_grid and
        cur_dep.
     If ens2_oracle beats both curs by >= +0.03 (CI excl 0) but ens2_blind
        does not -> lobe structure valuable but NOT inferable -> door closes.
     If ens2_oracle does not beat curs -> 2-lobe ensemble structure itself
        is worth ~nothing (prize is realization-level) -> door closes.
     Anything smaller than +0.03 or CI crossing 0 = FAILURE (n=16 negatives
        in this session had MDE ~ +-0.07; n=32 halves that; small-n "wins"
        have repeatedly evaporated -> do not overclaim).

Usage: python3 tmp/e37_bimodal_ladder.py [--n 32] [--smoke]
Output: results/direction3/e37_bimodal_ladder.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (build_sea_c, deposit_grid, bank_atoms, design_t,
                          corr, DEVICE, DEPTH, _MASK, _KXN, _KYN,
                          load_incumbent, incumbent_field, maps6,
                          two_stage_kp, NGRID, KREL)
from e29_gate0_ladder import universal_radial, ensemble_grid, solve_with_grid
from wavesim.wave_generator import jonswap_components
from d18_relax_estimator import beamform_top
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_achievability_gap import dispersion_lam, sample_obs, \
    masks_common_unobs, N_OBS

OUT = 'results/direction3/e37_bimodal_ladder.json'
SPREAD_DEG = 30.0

# frozen detector params (pre-registered)
DET_SHELL = (0.6, 1.6)          # radial window in k/kp
DET_NBIN = 72                   # 5-degree angular bins
DET_SMOOTH_DEG = 10.0
DET_EXCL_DEG = 45.0
DET_ACCEPT = 0.25
DET_REFINE_DEG = 15.0


# ==========================================================================
#  bimodal arena
# ==========================================================================
def build_sea_bimodal(si, unimodal=False):
    rng = np.random.default_rng(9100 + 31 * si)
    Tp = 7.0
    Hs = rng.uniform(1.5, 3.0)
    r2 = 0.0 if unimodal else rng.uniform(0.2, 0.5)
    d1 = rng.uniform(0, 360)
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    d2 = d1 + sign * rng.uniform(60, 150)
    c = list(jonswap_components(Hs=Hs * np.sqrt(1 - r2), Tp=Tp,
                                n_components=100, depth=DEPTH, gamma=3.3,
                                direction_deg=d1,
                                direction_spread_deg=SPREAD_DEG,
                                seed=4000 + si))
    if r2 > 0:
        c += list(jonswap_components(Hs=Hs * np.sqrt(r2), Tp=Tp,
                                     n_components=100, depth=DEPTH, gamma=3.3,
                                     direction_deg=d2,
                                     direction_spread_deg=SPREAD_DEG,
                                     seed=5000 + si))
    kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c])
    Am = np.array([x.A for x in c]); ph = np.array([x.phi for x in c])
    om = omega_of(kx, ky, DEPTH)
    lam = dispersion_lam(Tp, DEPTH)
    cp = lam / Tp
    T = 10 * Tp
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
                r2=float(r2), U=float(U))


# ==========================================================================
#  incumbent gamma map + constant("mean") prior (e30 recipe, recomputed)
# ==========================================================================
def net_gamma_map(S):
    net, _ = load_incumbent()
    X = maps6(S['P'], S['z'], S['kp'])
    with torch.no_grad():
        return net(torch.tensor(X[None], device=DEVICE))[0].cpu().numpy()


def rotate_map(M, ang):
    ca, sa = np.cos(ang), np.sin(ang)
    sx = ca * _KXN + sa * _KYN
    sy = -sa * _KXN + ca * _KYN
    sc = (NGRID - 1) / (2 * KREL)
    u = (sx + KREL) * sc
    v = (sy + KREL) * sc
    return map_coordinates(M, [v, u], order=1, mode='constant', cval=0.0)


def mu_hat_of(S):
    bx, by = beamform_top(S['P'].astype(np.float32),
                          S['z'].astype(np.float32),
                          DEPTH, klim=3.5 * S['kp'], n=120)
    return float(np.arctan2(by, bx))


def build_mean_aligned(n=16):
    """The net's across-sea mean aligned log-gamma map, on the SAME 16
    unimodal e30 seas (build_sea_c 'spread')."""
    maps, mus = [], []
    for si in range(n):
        S = build_sea_c('spread', si)
        maps.append(net_gamma_map(S))
        mus.append(mu_hat_of(S))
    aligned = np.stack([rotate_map(maps[i], -mus[i]) for i in range(n)])
    return aligned.mean(0)


# ==========================================================================
#  frozen 2-lobe detector (data only)
# ==========================================================================
def detect_lobes(S):
    """Returns (thetas, weights, accepted2). Data-only: pm map channel."""
    X = maps6(S['P'], S['z'], S['kp'])
    pm = X[2]                                    # min(p1,p2), 51x51
    kk = np.hypot(_KXN, _KYN)
    shell = (kk >= DET_SHELL[0]) & (kk <= DET_SHELL[1])
    th = np.arctan2(_KYN, _KXN)
    e = np.where(shell, pm, 0.0)
    idx = np.clip(((th + np.pi) / (2 * np.pi) * DET_NBIN).astype(int),
                  0, DET_NBIN - 1)
    prof = np.zeros(DET_NBIN)
    np.add.at(prof, idx[shell], e[shell])
    binw = 360.0 / DET_NBIN
    sig = DET_SMOOTH_DEG / binw
    xs = np.arange(-int(4 * sig), int(4 * sig) + 1)
    ker = np.exp(-0.5 * (xs / sig) ** 2)
    ker /= ker.sum()
    prof = np.convolve(np.tile(prof, 3), ker, 'same')[DET_NBIN:2 * DET_NBIN]

    centers = -np.pi + (np.arange(DET_NBIN) + 0.5) * np.deg2rad(binw)

    def refine(p):
        d = np.angle(np.exp(1j * (centers - centers[p])))
        w = np.where(np.abs(d) <= np.deg2rad(DET_REFINE_DEG), prof, 0.0)
        return float(np.angle((w * np.exp(1j * centers)).sum()))

    p1 = int(np.argmax(prof))
    excl = np.abs(np.angle(np.exp(1j * (centers - centers[p1])))) \
        <= np.deg2rad(DET_EXCL_DEG)
    rest = np.where(excl, -np.inf, prof)
    p2 = int(np.argmax(rest))
    accepted = bool(prof[p2] >= DET_ACCEPT * prof[p1])
    t1 = refine(p1)
    if not accepted:
        return [t1], [1.0], False
    t2 = refine(p2)
    w = np.array([prof[p1], prof[p2]], float)
    w /= w.sum()
    return [t1, t2], list(w), True


def two_lobe_grid(S, thetas, weights):
    kmag_u, en_u = universal_radial()
    g = np.zeros((NGRID, NGRID))
    for t, w in zip(thetas, weights):
        g += w * ensemble_grid(kmag_u, en_u, t, S['kp'])
    return g


# ==========================================================================
def ang_err_deg(a, b):
    return float(np.rad2deg(np.abs(np.angle(np.exp(1j * (a - b))))))


def ang_profile(grid2d):
    """Angular energy profile of a (linear-scale) 51x51 map, det shell."""
    kk = np.hypot(_KXN, _KYN)
    shell = (kk >= DET_SHELL[0]) & (kk <= DET_SHELL[1])
    th = np.arctan2(_KYN, _KXN)
    idx = np.clip(((th + np.pi) / (2 * np.pi) * DET_NBIN).astype(int),
                  0, DET_NBIN - 1)
    prof = np.zeros(DET_NBIN)
    np.add.at(prof, idx[shell], np.where(shell, grid2d, 0.0)[shell])
    binw = 360.0 / DET_NBIN
    sig = DET_SMOOTH_DEG / binw
    xs = np.arange(-int(4 * sig), int(4 * sig) + 1)
    ker = np.exp(-0.5 * (xs / sig) ** 2)
    ker /= ker.sum()
    return np.convolve(np.tile(prof, 3), ker, 'same')[DET_NBIN:2 * DET_NBIN]


def lobe2_capture(grid2d, d1, d2):
    """DIAGNOSTIC only: prof(d2)/prof(d1) of a gamma/energy map."""
    prof = ang_profile(grid2d)
    binw = np.deg2rad(360.0 / DET_NBIN)

    def at(ang):
        i = int(np.clip((ang + np.pi) / binw, 0, DET_NBIN - 1))
        return prof[i]
    return float(at(d2) / max(at(d1), 1e-30))


def score_sea(S, mean_aligned):
    kmag_u, en_u = universal_radial()
    mu_hat = mu_hat_of(S)
    row = dict(Hs=S['Hs'], r2=S['r2'],
               dsep=ang_err_deg(S['d1'], S['d2']), mu_hat=mu_hat)
    # net arms
    Y = net_gamma_map(S)
    row['cur_grid'] = solve_with_grid(S, np.maximum(10.0 ** Y, 1e-8))
    f_dep = incumbent_field(S['P'], S['z'], S['Pq'], S['kp'], S['sigma0'])
    row['cur_dep'] = corr(f_dep, S['gt_u'])
    # constant prior
    g_mean = np.maximum(10.0 ** rotate_map(mean_aligned, mu_hat), 1e-8)
    row['mean'] = solve_with_grid(S, g_mean)
    # hand 1-lobe
    row['ens1'] = solve_with_grid(S, ensemble_grid(kmag_u, en_u, mu_hat,
                                                   S['kp']))
    # hand 2-lobe blind
    thetas, weights, acc = detect_lobes(S)
    row['det2'] = acc
    row['ens2_blind'] = solve_with_grid(S, two_lobe_grid(S, thetas, weights))
    # detector diagnostics: match detected lobes to true directions
    if acc:
        e11 = ang_err_deg(thetas[0], S['d1'])
        e12 = ang_err_deg(thetas[0], S['d2'])
        if e11 <= e12:
            row['derr1'], row['derr2'] = e11, ang_err_deg(thetas[1], S['d2'])
        else:
            row['derr1'], row['derr2'] = ang_err_deg(thetas[1], S['d1']), e12
        row['w2'] = float(min(weights))
    else:
        row['derr1'] = min(ang_err_deg(thetas[0], S['d1']),
                           ang_err_deg(thetas[0], S['d2']))
        row['derr2'] = float('nan')
        row['w2'] = 0.0
    # hand 2-lobe oracle (true dirs, true weights)
    if S['r2'] > 0:
        row['ens2_oracle'] = solve_with_grid(
            S, two_lobe_grid(S, [S['d1'], S['d2']],
                             [1 - S['r2'], S['r2']]))
    else:
        row['ens2_oracle'] = row['ens1']
    # oracle realization (fixed solver, NO GT-peeked blur/smult)
    gf = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, S['kp'])
    row['real'] = solve_with_grid(S, gf)
    # DIAGNOSTIC (score-neutral): does the net's gamma carry lobe 2?
    if S['r2'] > 0:
        row['cap_net'] = lobe2_capture(np.maximum(10.0 ** Y, 1e-8),
                                       S['d1'], S['d2'])
        row['cap_true'] = lobe2_capture(gf, S['d1'], S['d2'])
        X6 = maps6(S['P'], S['z'], S['kp'])
        row['cap_map'] = lobe2_capture(X6[2], S['d1'], S['d2'])
    return row


def paired(rows, a, b):
    d = np.array([r[a] - r[b] for r in rows], float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    return dict(mean=float(d.mean()), se=float(se),
                t=float(d.mean() / se) if se > 0 else float('nan'),
                lo=float(d.mean() - 1.96 * se),
                hi=float(d.mean() + 1.96 * se),
                win=int((d > 0).sum()), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--n_uni', type=int, default=8)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        a.n, a.n_uni = 4, 2
    t0 = time.time()
    print('--- building constant("mean") prior from 16 unimodal e30 seas ---',
          flush=True)
    mean_aligned = build_mean_aligned(16)
    print(f'    done ({time.time()-t0:.0f}s)', flush=True)

    rows, rows_uni = [], []
    for si in range(a.n):
        S = build_sea_bimodal(si)
        r = score_sea(S, mean_aligned)
        rows.append(r)
        print(f"bi {si:2d} r2 {r['r2']:.2f} sep {r['dsep']:5.1f} "
              f"| curG {r['cur_grid']:+.3f} curD {r['cur_dep']:+.3f} "
              f"mean {r['mean']:+.3f} ens1 {r['ens1']:+.3f} "
              f"| e2b {r['ens2_blind']:+.3f} (det {int(r['det2'])} "
              f"d1e {r['derr1']:4.0f} d2e {r['derr2']:4.0f} w2 {r['w2']:.2f}) "
              f"e2o {r['ens2_oracle']:+.3f} real {r['real']:+.3f}",
              flush=True)
    for si in range(a.n_uni):
        S = build_sea_bimodal(1000 + si, unimodal=True)
        r = score_sea(S, mean_aligned)
        rows_uni.append(r)
        print(f"uni{si:2d}          | curG {r['cur_grid']:+.3f} "
              f"curD {r['cur_dep']:+.3f} ens1 {r['ens1']:+.3f} "
              f"| e2b {r['ens2_blind']:+.3f} (det {int(r['det2'])}) "
              f"real {r['real']:+.3f}", flush=True)

    arms = ['cur_grid', 'cur_dep', 'mean', 'ens1', 'ens2_blind',
            'ens2_oracle', 'real']
    print(f"\n=== e37 BIMODAL LADDER (n={len(rows)}, smult=3 fixed, "
          f"free arm, no GT peek) ===")
    for k in arms:
        v = np.array([r[k] for r in rows])
        print(f"  {k:12s}: {v.mean():+.4f}")
    det_rate = float(np.mean([r['det2'] for r in rows]))
    fp_rate = float(np.mean([r['det2'] for r in rows_uni])) if rows_uni \
        else float('nan')
    print(f"  detector: 2-lobe accept rate bimodal {det_rate:.0%} | "
          f"unimodal FALSE-POSITIVE {fp_rate:.0%}")
    d1e = np.array([r['derr1'] for r in rows])
    d2e = np.array([r['derr2'] for r in rows])
    print(f"  lobe dir err: primary {np.nanmean(d1e):.1f} deg, "
          f"secondary {np.nanmean(d2e):.1f} deg")
    for k in ('cap_net', 'cap_true', 'cap_map'):
        v = np.array([r[k] for r in rows if k in r])
        print(f"  lobe2 capture {k:8s}: median {np.median(v):.3f} "
              f"(q25 {np.quantile(v, .25):.3f} q75 {np.quantile(v, .75):.3f})")

    print("\n--- pre-registered contrasts (win needs >= +0.03, CI excl 0, "
          "win-rate >= 60%, vs BOTH curs) ---")
    cons = {}
    for aa, bb in [('ens2_blind', 'cur_grid'), ('ens2_blind', 'cur_dep'),
                   ('ens2_oracle', 'cur_grid'), ('ens2_oracle', 'cur_dep'),
                   ('ens2_oracle', 'ens1'), ('ens2_blind', 'ens1'),
                   ('ens1', 'cur_grid'), ('mean', 'cur_grid'),
                   ('mean', 'cur_dep'),
                   ('real', 'ens2_oracle'), ('real', 'cur_dep')]:
        c = paired(rows, aa, bb)
        cons[f'{aa}-{bb}'] = c
        print(f"  {aa:11s} - {bb:11s}: {c['mean']:+.4f} +- {c['se']:.4f} "
              f"(t {c['t']:+.2f}, CI [{c['lo']:+.4f},{c['hi']:+.4f}], "
              f"win {c['win']}/{c['n']})")

    def winner(key):
        c1, c2 = cons[f'{key}-cur_grid'], cons[f'{key}-cur_dep']
        return all(c['mean'] >= 0.03 and c['lo'] > 0
                   and c['win'] / c['n'] >= 0.6 for c in (c1, c2))

    w_blind, w_oracle = winner('ens2_blind'), winner('ens2_oracle')
    if w_blind:
        verdict = 'REOPEN: ens2_blind beats the net (pre-registered WIN)'
    elif w_oracle:
        verdict = ('CLOSED (inference gap): true 2-lobe prior wins but the '
                   'map-detected one does not')
    else:
        verdict = ('CLOSED (representation not binding): even the TRUE '
                   '2-lobe ensemble prior does not beat the net by +0.03')
    print(f"\nVERDICT: {verdict}")
    json.dump(dict(rows=rows, rows_uni=rows_uni, contrasts=cons,
                   det_rate=det_rate, fp_rate=fp_rate,
                   verdict=verdict, w_blind=bool(w_blind),
                   w_oracle=bool(w_oracle)),
              open(OUT, 'w'), indent=1, default=float)
    print(f"saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()


# ==========================================================================
#  EXPLORATORY (post-registration, labeled): cross-Tp bimodal variant.
#  Tp2 = 7 * U(0.75, 1.4)  -> lobe 2 at a DIFFERENT radial ring (k2/kp1 in
#  ~[0.51, 1.78], inside the bank). This closes the strongest remaining
#  loophole: e36's lost arena / the training augmentation were cross-Tp.
#  Blind detector extension: radial centroid of the pm map in a +-22.5 deg
#  wedge around each detected lobe; universal radial SHAPE re-centred there.
# ==========================================================================
def build_sea_crosstp(si):
    rng = np.random.default_rng(9700 + 31 * si)
    Tp1 = 7.0
    Hs = rng.uniform(1.5, 3.0)
    r2 = rng.uniform(0.2, 0.5)
    d1 = rng.uniform(0, 360)
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    d2 = d1 + sign * rng.uniform(60, 150)
    Tp2 = float(Tp1 * rng.uniform(0.75, 1.4))
    c = list(jonswap_components(Hs=Hs * np.sqrt(1 - r2), Tp=Tp1,
                                n_components=100, depth=DEPTH, gamma=3.3,
                                direction_deg=d1,
                                direction_spread_deg=SPREAD_DEG,
                                seed=4000 + si))
    c += list(jonswap_components(Hs=Hs * np.sqrt(r2), Tp=Tp2,
                                 n_components=100, depth=DEPTH, gamma=3.3,
                                 direction_deg=d2,
                                 direction_spread_deg=SPREAD_DEG,
                                 seed=5000 + si))
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
                r2=float(r2), Tp2=Tp2, U=float(U))


def radial_centroid(S, theta):
    """pm-map radial centroid (k/kp units) in a +-22.5 deg wedge at theta."""
    X = maps6(S['P'], S['z'], S['kp'])
    pm = X[2]
    kk = np.hypot(_KXN, _KYN)
    th = np.arctan2(_KYN, _KXN)
    wedge = (np.abs(np.angle(np.exp(1j * (th - theta))))
             <= np.deg2rad(22.5)) & (kk >= 0.4) & (kk <= 2.2)
    w = pm[wedge]
    if w.sum() <= 0:
        return 1.0
    return float((kk[wedge] * w).sum() / w.sum())


def scaled_radial_grid(S, theta, rcent):
    """Universal radial SHAPE re-centred at rcent (k/kp), one lobe."""
    kmag_u, en_u = universal_radial()
    kp_u = kmag_u[np.argmax(en_u)]
    kn = kmag_u / kp_u                        # normalized shape, peak at 1
    kmag = kn * rcent * S['kp']               # absolute k, peak at rcent*kp
    return ensemble_grid(kmag, en_u, theta, S['kp'])


def true_radial_grid(S, Tp, theta):
    c = jonswap_components(Hs=2.0, Tp=Tp, n_components=200, depth=DEPTH,
                           gamma=3.3, direction_deg=0.0,
                           direction_spread_deg=1e-6, seed=1234)
    kmag = np.array([np.hypot(x.kx, x.ky) for x in c])
    e = np.array([x.A for x in c]) ** 2 / 2
    return ensemble_grid(kmag, e / e.sum(), theta, S['kp'])


def score_sea_crosstp(S):
    row = dict(Hs=S['Hs'], r2=S['r2'], Tp2=S['Tp2'],
               dsep=ang_err_deg(S['d1'], S['d2']))
    Y = net_gamma_map(S)
    row['cur_grid'] = solve_with_grid(S, np.maximum(10.0 ** Y, 1e-8))
    f_dep = incumbent_field(S['P'], S['z'], S['Pq'], S['kp'], S['sigma0'])
    row['cur_dep'] = corr(f_dep, S['gt_u'])
    kmag_u, en_u = universal_radial()
    mu_hat = mu_hat_of(S)
    row['ens1'] = solve_with_grid(S, ensemble_grid(kmag_u, en_u, mu_hat,
                                                   S['kp']))
    thetas, weights, acc = detect_lobes(S)
    g = np.zeros((NGRID, NGRID))
    for t, w in zip(thetas, weights):
        g += w * scaled_radial_grid(S, t, radial_centroid(S, t))
    row['ens2_blind'] = solve_with_grid(S, g)
    row['det2'] = acc
    go = (1 - S['r2']) * true_radial_grid(S, 7.0, S['d1']) \
        + S['r2'] * true_radial_grid(S, S['Tp2'], S['d2'])
    row['ens2_oracle'] = solve_with_grid(S, go)
    gf = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, S['kp'])
    row['real'] = solve_with_grid(S, gf)
    row['cap_net'] = lobe2_capture(np.maximum(10.0 ** Y, 1e-8),
                                   S['d1'], S['d2'])
    row['cap_true'] = lobe2_capture(gf, S['d1'], S['d2'])
    return row


def main_crosstp(n=16):
    rows = []
    for si in range(n):
        S = build_sea_crosstp(si)
        r = score_sea_crosstp(S)
        rows.append(r)
        print(f"xt {si:2d} r2 {r['r2']:.2f} Tp2 {r['Tp2']:.1f} "
              f"sep {r['dsep']:5.1f} | curG {r['cur_grid']:+.3f} "
              f"curD {r['cur_dep']:+.3f} ens1 {r['ens1']:+.3f} "
              f"| e2b {r['ens2_blind']:+.3f} e2o {r['ens2_oracle']:+.3f} "
              f"real {r['real']:+.3f} capN {r['cap_net']:.2f} "
              f"capT {r['cap_true']:.2f}", flush=True)
    print(f"\n=== EXPLORATORY cross-Tp bimodal (n={n}) ===")
    for k in ['cur_grid', 'cur_dep', 'ens1', 'ens2_blind', 'ens2_oracle',
              'real']:
        v = np.array([r[k] for r in rows], float)
        print(f"  {k:12s}: {np.nanmean(v):+.4f} (n={np.isfinite(v).sum()})")
    for aa, bb in [('ens2_blind', 'cur_grid'), ('ens2_blind', 'cur_dep'),
                   ('ens2_oracle', 'cur_grid'), ('ens2_oracle', 'cur_dep'),
                   ('ens2_oracle', 'ens1'), ('real', 'ens2_oracle')]:
        x = np.array([r[aa] for r in rows], float)
        y = np.array([r[bb] for r in rows], float)
        m = np.isfinite(x) & np.isfinite(y)
        dd = x[m] - y[m]
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        print(f"  {aa:11s} - {bb:11s}: {dd.mean():+.4f} +- {se:.4f} "
              f"(win {(dd > 0).sum()}/{len(dd)})")
    json.dump(dict(rows=rows, exploratory=True),
              open('results/direction3/e37_crosstp_exploratory.json', 'w'),
              indent=1, default=float)
    print('saved -> results/direction3/e37_crosstp_exploratory.json')
