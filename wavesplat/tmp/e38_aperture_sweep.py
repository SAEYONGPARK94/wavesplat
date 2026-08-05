"""e38 — DECISIVE APERTURE SWEEP (adjudication of Agent1-aperture vs Agent2-speckle)

PRE-REGISTERED DESIGN (written before running):

Paired seas (same si => same Hs, mean direction, U draw, noise fraction).
Sweep the observation along two orthogonal axes from the base arena
(T=70 s, n=400):

  Th    T= 35 s  n= 200   (half window, rate const)
  base  T= 70 s  n= 400   (replicates the e29/e32 arena)
  T2n2  T=140 s  n= 800   (2x window, rate const)
  T4n4  T=280 s  n=1600   (4x window, rate const)
  nT0   T= 70 s  n=1600   (points ONLY: 4x points, same aperture)
  Tn0   T=280 s  n= 400   (aperture ONLY: 4x window, same 400 points)

Arms per (sea, condition) — smult=3 FIXED, exponents FIXED, no GT peek:
  ens   constant hand prior  ens_grid_norm(mu_hat)            [rung2]
  real  oracle realized gamma (free arm)                      [rung4]
  bf1   pure data prior  gamma = bf_map^1.0
  eb05  modulated data prior  gamma = ens * bf_map^0.5
  cur   the learned incumbent (l1s2bw; OOD beyond base — caveat)
  partial  spearman(bf, gamma_true | gamma_ens) on band cells [no estimator]

Quantities:
  headroom  H = real - ens          (oracle realization prize over the constant)
  extract   X_eb = eb05 - ens ;  X_bf = bf1 - ens

PRE-REGISTERED PREDICTIONS:
  Agent 1 (observation/aperture wall, opens with better observation):
     H shrinks along Th->base->T2n2->T4n4 (at T4n4: H < 0.5 * H(base));
     X turns >= 0 once resolution suffices; "wall is 400 points" strictly
     predicts nT0 (points only) already shrinks H / lifts X.
  Agent 2 (speckle/SNR wall, aperture cannot cash):
     H persists (within ~20% of base) at ALL conditions incl. T4n4;
     X stays <= 0 everywhere; partial stays ~0.08-0.12 flat.
  Mixed outcome (H persists but X rises to >0 with aperture) = reconciliation:
     aperture buys extraction of a prize that itself does not close.

DECISION RULE: primary = paired X_eb(T4n4) CI and paired dH = H(T4n4)-H(base);
secondary = monotone trend across the 4 rate-const conditions; nT0 vs Tn0
separates points from aperture. n=16 paired first; key contrasts at n=32.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (REGIMES, deposit_grid, two_stage_kp, incumbent_field,
                          corr, DEPTH, _MASK, NGRID, KREL)
from e29_gate0_ladder import solve_with_grid
from e30_conditionality_probe import radial_of, ens_grid_norm
from e34_speckle import partial_spearman
from d3_sa_nomp import beamform_power
from d18_relax_estimator import beamform_top
from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_achievability_gap import (dispersion_lam, masks_common_unobs,
                                     hashabs)

OUT = 'results/direction3/e38_aperture_sweep.json'
CONDS = [('Th', 35.0, 200), ('base', 70.0, 400), ('T2n2', 140.0, 800),
         ('T4n4', 280.0, 1600), ('nT0', 70.0, 1600), ('Tn0', 280.0, 400)]


def sample_obs_n(mode, U, T, gx, gy, r_fp, rng, n_obs):
    """verbatim copy of d3_sa_achievability_gap.sample_obs with n parametrized."""
    tt = np.linspace(0, T, n_obs)
    cx, cy = (gx[0] + gx[-1]) / 2, (gy[0] + gy[-1]) / 2
    th = rng.uniform(0, 2 * np.pi, n_obs)
    rr = r_fp * np.sqrt(rng.uniform(0, 1, n_obs))
    jx, jy = rr * np.cos(th), rr * np.sin(th)
    if mode == 'fixed':
        return np.column_stack([cx + jx, cy + jy, tt])
    P0 = np.array([gx[0] + r_fp, gy[0] + r_fp])
    P1 = np.array([gx[-1] - r_fp, gy[-1] - r_fp])
    L = float(np.linalg.norm(P1 - P0))
    u = (P1 - P0) / L
    s = np.mod(U * tt, 2 * L)
    d = np.where(s <= L, s, 2 * L - s)
    xy = P0[None] + d[:, None] * u[None]
    return np.column_stack([xy[:, 0] + jx, xy[:, 1] + jy, tt])


def build_sea_ap(si, T_s, n_obs):
    """build_sea_c with (T, n_obs) parametrized; rng stream for
    (Hs, direction, U, nfrac) identical across conditions => paired seas."""
    R = REGIMES['spread']
    rng = np.random.default_rng(7000 + 31 * si + hashabs('spread'))
    Tp = 7.0
    Hs = rng.uniform(*R['hs'])
    c = jonswap_components(Hs=Hs, Tp=Tp, n_components=200, depth=DEPTH,
                           gamma=3.3, direction_deg=rng.uniform(0, 360),
                           direction_spread_deg=30.0, seed=4000 + si)
    kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c])
    Am = np.array([x.A for x in c]); ph = np.array([x.phi for x in c])
    om = omega_of(kx, ky, DEPTH)
    lam = dispersion_lam(Tp, DEPTH)
    cp = lam / Tp
    T = T_s
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
    U = rng.uniform(*R['U']) * cp
    nfrac = rng.uniform(*R['nf'])
    P = sample_obs_n('fixed' if U < 1e-6 else 'moving', U, T, gx, gy, r_fp,
                     rng, n_obs)
    psi = (kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2]
           - om[None] * P[:, 2:3] + ph[None])
    z = (Am[None] * np.cos(psi)).sum(1) + rng.normal(0, nfrac * Hs, n_obs)
    bw_lvl = rng.uniform(*R['bw'])
    if bw_lvl > 1e-4:
        K = min(12, len(Am)); top = np.argsort(Am)[-K:]
        bkx, bky = 2 * kx[top], 2 * ky[top]
        bom, bph = 2 * om[top], 2 * ph[top]
        bA = 0.5 * np.hypot(kx[top], ky[top]) * Am[top] ** 2
        psib = (bkx[None] * P[:, 0:1] + bky[None] * P[:, 1:2]
                - bom[None] * P[:, 2:3] + bph[None])
        zb = (bA[None] * np.cos(psib)).sum(1)
        bw_sc = (np.sqrt(bw_lvl) * (Hs / 4)) / max(np.std(zb), 1e-9)
        z = z + bw_sc * zb
        psib_q = (bkx[None] * Xs[:, None] + bky[None] * Ys[:, None]
                  - bom[None] * T + bph[None])
        gt_u = gt_u + bw_sc * (bA[None] * np.cos(psib_q)).sum(1)
    kp = two_stage_kp(P, z)
    return dict(P=P, z=z, Pq=Pq, gt_u=gt_u, kp=kp, sigma0=float(nfrac * Hs),
                Hs=float(Hs), m_kx=kx, m_ky=ky, m_A=Am, m_ph=ph,
                U=float(U), lam=float(lam))


def main(n_seas=16):
    kn, en = radial_of(7.0, 3.3)
    out = {}
    for tag, T_s, n_obs in CONDS:
        rows = []
        print(f'\n===== {tag}: T={T_s:g}s n={n_obs} =====', flush=True)
        for si in range(n_seas):
            S = build_sea_ap(si, T_s, n_obs)
            kp = S['kp']
            bx, by = beamform_top(S['P'].astype(np.float32),
                                  S['z'].astype(np.float32),
                                  DEPTH, klim=3.5 * kp, n=120)
            mu_hat = np.arctan2(by, bx)
            g_ens = ens_grid_norm(kn, en, mu_hat, kp, 30.0)
            g_real = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, kp)
            _, _, pf = beamform_power(S['P'], S['z'], DEPTH, KREL * kp, NGRID)
            pf = (pf / max(pf.max(), 1e-12)).reshape(NGRID, NGRID)
            bfp = np.maximum(pf, 1e-8)

            r = dict(si=si, U=S['U'], path_lam=S['U'] * T_s / S['lam'])
            r['ens'] = solve_with_grid(S, g_ens)
            r['real'] = solve_with_grid(S, g_real)
            r['bf1'] = solve_with_grid(S, bfp)
            r['eb05'] = solve_with_grid(S, g_ens * bfp ** 0.5)
            try:
                r['cur'] = float(corr(incumbent_field(
                    S['P'], S['z'], S['Pq'], kp, S['sigma0']), S['gt_u']))
            except Exception as e:
                r['cur'] = float('nan')
            r['partial'] = partial_spearman(pf[_MASK], g_real[_MASK],
                                            g_ens[_MASK])
            rows.append(r)
            print(f"  sea{si:2d} ens {r['ens']:+.3f} real {r['real']:+.3f} "
                  f"bf1 {r['bf1']:+.3f} eb05 {r['eb05']:+.3f} "
                  f"cur {r['cur']:+.3f} part {r['partial']:+.3f} "
                  f"path {r['path_lam']:4.1f}L", flush=True)
        out[tag] = rows
        a = lambda k: np.array([x[k] for x in rows], float)
        H = a('real') - a('ens')
        Xeb = a('eb05') - a('ens'); Xbf = a('bf1') - a('ens')
        se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
        print(f"  --> {tag}: ens {a('ens').mean():+.4f} real {a('real').mean():+.4f} "
              f"cur {np.nanmean(a('cur')):+.4f}")
        print(f"      H(real-ens) {H.mean():+.4f}±{se(H):.4f} | "
              f"X_eb {Xeb.mean():+.4f}±{se(Xeb):.4f} (pos {(Xeb>0).sum()}/{len(Xeb)}) | "
              f"X_bf {Xbf.mean():+.4f}±{se(Xbf):.4f} (pos {(Xbf>0).sum()}/{len(Xbf)}) | "
              f"partial {a('partial').mean():+.4f}", flush=True)
        json.dump(out, open(OUT, 'w'), indent=1)

    # ---- paired contrasts
    print("\n\n================= PAIRED CONTRASTS =================")
    base = out['base']
    for tag, _, _ in CONDS:
        if tag == 'base':
            continue
        rows = out[tag]
        a = lambda rs, k: np.array([x[k] for x in rs], float)
        dH = (a(rows, 'real') - a(rows, 'ens')) - (a(base, 'real') - a(base, 'ens'))
        dX = (a(rows, 'eb05') - a(rows, 'ens')) - (a(base, 'eb05') - a(base, 'ens'))
        se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {tag:5s} vs base: dH {dH.mean():+.4f}±{se(dH):.4f} "
              f"(t {dH.mean()/se(dH):+.2f}) | dX_eb {dX.mean():+.4f}±{se(dX):.4f} "
              f"(t {dX.mean()/se(dX):+.2f})")
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
