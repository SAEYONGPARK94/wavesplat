"""e46 — ★HOS LEG: does the buoy-grade external-supply headline survive
nonlinearity, or is it a linear-superposition artifact?

WHY. Standing user directive: "검증 데이터셋엔 항상 HOS 비선형 포함, 헤드라인은
HOS서 재현" (validation must include HOS nonlinear seas; headlines must
reproduce under HOS). A two-agent audit found the whole E-29..E-45 arc violated
it: every number below was measured in a LINEAR JONSWAP superposition arena.
  surviving headline (post-audit, linear arena):
    prize   oracle-cur (field)      +0.0729 +- 0.0034  (6 blocks)
    transfer enc10 orc-cur          +0.0431 +- 0.0077  (n=957, ratio 0.60)
    buoy    buoy-grade supply       +0.0349 CI [+0.0084,+0.0615]
    ww3     forecast-grade supply   DEAD (frozen-axis -0.0092; phantom -0.0158)
    e45 phantom refs (linear): fld ph_buoy-cur -0.0074 CI[-0.0194,+0.0045],
                               fld ph_ww3-cur  -0.0208 CI[-0.0338,-0.0078]
This script pays the debt: the honest HOS (M=3) analog of build_sea_crosstp,
same geometry / track / observation operator / sigma0 / SMULT=3 / solver, same
arms, same functionals.

================= PRE-REGISTERED (frozen before any scoring) =================
Smoke block si=4990..4993 (crossTp) / si=6990..6991 (unimodal) is quarantined:
used ONLY for crash / NaN / timing checks; its numbers are not read for tuning.

ARENA X_H (HOS cross-Tp). Same parameter/track/noise draws as
  e37_bimodal_ladder.build_sea_crosstp — identical rng stream 9700+31*si in the
  identical call order (Hs~U(1.5,3), r2~U(0.2,0.5), d1~U(0,360),
  d2=d1+-U(60,150) deg, Tp2=7*U(0.75,1.4), spread 30 deg both partitions,
  U~U(0,1.5)*cp, nfrac~U(0.02,0.09), N_OBS=400 over T=70 s, aperture
  side=2*lam1, GT=1200 held-out pixels at t=T, same masks and rng(0)
  subsample) — BUT the surface is a Higher-Order Spectral evolution:
    M=3, dealiasing ON (HOSSimulator factor ceil((M+1)/2)=2; the documented
      "M=1 dealiasing-off NaN blow-up" path is NOT used),
    periodic domain 8*lam1 x 8*lam1 (611 m), 64x64 spectral modes
      (dk=kp1/8, kmax=4*kp1, 7.2 deg angular spacing at the peak ring),
    depth 50 m, RK4 dt=0.15 s (omega_max*dt=0.27, same margin as the F1/E-16
      HOS precedent DT=0.10 at omega_max 2.78), spin-up 3*Tp1=21 s before
      arena t=0.
  IC: two JONSWAP partitions via HOSSimulator.initialize_jonswap called once
  per partition (Hs*sqrt(1-r2) @ Tp1,d1 seed 4000+si; Hs*sqrt(r2) @ Tp2,d2
  seed 5000+si), eta_hat and phi_s_hat summed (linear IC; bound harmonics
  develop during spin-up). Steepness = the arena's own: kp1*Hs/4 in
  [0.031, 0.062] at lam1=76.4 m — mild enough for M=3 stability, strong
  enough for measurable skewness (2nd-order theory ~3*kp*sigma ~ 0.09-0.19).
LINEAR TWIN X_L. The same IC modes evolved linearly (exact linear limit of
  the same discretization: eta_hat(t) = eta_hat(0)*exp(-i*omega*t)), same
  observation noise vector -> per-sea paired physics control.
  ★HARNESS GATE: twin fld orc-cur mean must land in [0.04, 0.11] (the e44
  anchor band). If the twin gate FAILS the harness is wrong and NOTHING ELSE
  IS READ. (Putting the harness gate on the twin — not on the HOS leg — keeps
  "prize dies under HOS" observable as a physics outcome instead of being
  indistinguishable from a bug.)
SEAS. Fresh blocks: crossTp si=5000..5191 (n=192), unimodal si=6000..6063
  (n_uni=64, the build_sea_bimodal 9100+31*si stream with r2=0, d2 drawn as
  in e45). Burned and avoided: si 0..31, 100..163, 200..263, 300..347,
  400..447, 900..915, 1000..1959; unimodal 3000+/3200+/3300+/3400+/3500+/
  4000+. FALLBACK (timing only, decided before reading any contrast): if the
  smoke projects > 3 h for the full block on this machine, n=96 / n_uni=48.
ARMS (identical solver solve_with_grid, SMULT=3, free arm, no GT peek; same
  theta draws re-used in BOTH worlds so H and L are draw-paired):
  cur   incumbent net (10**net_gamma_map)
  m2    honest all-marginal [MANDATORY: cur~=m2 in this arc; beating cur
        without beating m2 recovers nothing]
  orc   supplied TRUE theta (e43 supplied_grid)
  buoy  theta at buoy grade (d+-5deg, Tp2+-5%, r2+-0.05), R=5 draws/sea,
        per-sea mean BEFORE pairing (E-14 audit rule)
  unimodal block (e45 arms, R=3): uni_orc, uni_buoy, ph_buoy, ph_ww3
        (phantom second lobe that does not exist).
FUNCTIONALS. field corr at Pq (primary/anchor) AND encounter corr on the
  track_future extrapolation (enc10 = 0-10 s, enc05 = 0-5 s, 40 pts).
NONLINEARITY SANITY (must pass, else the HOS leg is a linear run in disguise
  and NO verdict is issued): mean elevation skewness of the HOS field at t=T
  >= +0.05 AND exceeds the twin's mean skewness by >= +0.04. Also reported:
  excess kurtosis, Hs drift |Hs(T)-Hs_target|/Hs_target, spectral tail
  pile-up (energy fraction at k > 0.85*kmax, t=T vs t=0), corr(z_H, z_L),
  NaN/instability exclusions (> 5% excluded -> harness problem, no verdict).
MDE (t-test, alpha=.05, power 80%, n=192). Field buoy-cur: assumed per-sea
  paired sd 0.10 (e43 linear history: se 0.0133 @ n=64) -> MDE +-0.020;
  robust to sd inflation 1.5x (sd 0.15 -> MDE 0.030) vs the +0.035 target.
  Fallback n=96 -> MDE 0.029 (still below target, marginal). Encounter
  orc-cur: sd 0.238 (e44) -> MDE 0.048 vs expected +0.043 => encounter is
  SECONDARY (~74% power); buoy-enc expected ~+0.02 => PILOT, CI reported, no
  claim either way. Unimodal n=64, R=3: fld phantom CI half-width ~+-0.020
  (sd 0.08 assumed) — adequate for the -0.01 do-no-harm bar as a mean bar,
  disclosed as such.
BARS (HOS block, field functional = primary):
  ANCHOR-H  fld orc-cur mean >= +0.03 with 95% t-CI excluding 0 = the prize
            survives nonlinearity. If the twin gate passes but ANCHOR-H has
            CI upper < +0.02 -> the PRIZE ITSELF is a linear artifact ->
            headline scoped to linear seas (physics verdict, not harness).
  PASS      fld buoy-cur >= +0.03 AND t-CI excludes 0 AND fld buoy-m2 t-CI
            excludes 0 AND phantom do-no-harm: ph_buoy-cur mean >= -0.01 on
            BOTH fld and enc10 (unimodal HOS block).
  KILL      fld buoy-cur 95% t-CI upper < +0.02 -> the buoy headline does
            NOT survive nonlinearity -> deliverable scoped to linear seas.
  Neither -> INCONCLUSIVE, reported with achieved MDE. A null is a
            legitimate outcome and will be reported as such.
STATS. t-based CIs everywhere (scipy stats.t.interval; NEVER 1.96 — the
  e41:356 bug). Deterministic integer seeds only (never Python hash() — the
  E-28-B disqualification). Per-sea means over R draws BEFORE pairing.
  Wilcoxon reported for enc10 orc-cur. Exploratory readouts (labeled): the
  draw-paired dd = (arm-cur)_H - (arm-cur)_L per sea.
==============================================================================

Usage: python3 tmp/e46_hos_repro.py [--smoke] [--n 192] [--n_uni 64]
Out:   results/direction3/e46_hos_repro.json
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

from e29_gennet_C import corr, DEPTH, two_stage_kp                # noqa: E402
from e29_gate0_ladder import solve_with_grid                      # noqa: E402
from e37_bimodal_ladder import net_gamma_map, SPREAD_DEG          # noqa: E402
from e43_external_regime_dose import supplied_grid, perturb, BUNDLES  # noqa: E402
from e43d_gate import x_marginals, paired                         # noqa: E402
from e44_encounter_transfer import field_of                       # noqa: E402
from e44_probe_N2_encounter_transfer import (track_future, N_ENC,  # noqa: E402
                                             T_END, HORIZON)
from e45_attack_dnh_enc import one_lobe, two_lobe                 # noqa: E402
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,  # noqa: E402
                                     masks_common_unobs, N_OBS)
from wavesim.hos_simulator import HOSSimulator, HOSConfig         # noqa: E402

OUT = 'results/direction3/e46_hos_repro.json'

G = 9.81
TP1 = 7.0
LAM1 = dispersion_lam(TP1, DEPTH)          # ~76.4 m at depth 50
KP1 = 2 * np.pi / LAM1
L_DOM = 8.0 * LAM1                          # periodic HOS domain
NSPEC = 64                                  # spectral modes per axis
M_HOS = 3                                   # NEVER 1 (aliases -> NaN, e16 note)
DT = 0.15                                   # RK4 step; omega_max*dt ~ 0.27
SPINUP = 3.0 * TP1                          # 21 s before arena t=0
RB = 5                                      # buoy draws per sea (e43 conv.)
R_UNI = 3                                   # phantom/uni draws per sea (e45)


# ==========================================================================
#  HOS + twin sea builder
# ==========================================================================
def advance_to(sim, t_phys):
    while sim.t < t_phys - 1e-9:
        sim.step(min(DT, t_phys - sim.t))


def eta_hos_at(sim, xs, ys):
    """Direct spectral evaluation of the CURRENT HOS surface at points."""
    xs = np.atleast_1d(np.asarray(xs, float))
    ys = np.atleast_1d(np.asarray(ys, float))
    a = sim.eta_hat.ravel()
    m = np.abs(a) > 1e-15
    a = a[m]
    kx = sim.KX.ravel()[m]
    ky = sim.KY.ravel()[m]
    out = np.zeros(len(xs))
    B = 2048
    for i in range(0, len(a), B):
        ph = np.outer(kx[i:i + B], xs) + np.outer(ky[i:i + B], ys)
        out += np.real(a[i:i + B][:, None] * np.exp(1j * ph)).sum(0)
    return out


def eta_lin_at(tw, xs, ys, t_phys):
    """Linear twin at points sharing ONE time t_phys (scalar)."""
    xs = np.atleast_1d(np.asarray(xs, float))
    ys = np.atleast_1d(np.asarray(ys, float))
    a = tw['a'] * np.exp(-1j * tw['om'] * t_phys)
    out = np.zeros(len(xs))
    B = 2048
    for i in range(0, len(a), B):
        ph = np.outer(tw['kx'][i:i + B], xs) + np.outer(tw['ky'][i:i + B], ys)
        out += np.real(a[i:i + B][:, None] * np.exp(1j * ph)).sum(0)
    return out


def eta_lin_obs(tw, P, spinup):
    """Linear twin at obs points, each with its OWN time (vectorized)."""
    out = np.zeros(len(P))
    t = P[:, 2] + spinup
    B = 2048
    for i in range(0, len(tw['kx']), B):
        ph = (np.outer(tw['kx'][i:i + B], P[:, 0])
              + np.outer(tw['ky'][i:i + B], P[:, 1])
              - np.outer(tw['om'][i:i + B], t))
        out += np.real(tw['a'][i:i + B][:, None] * np.exp(1j * ph)).sum(0)
    return out


def tail_frac(eta_hat, K, kmax):
    e = np.abs(eta_hat) ** 2
    return float(e[K > 0.85 * kmax].sum() / max(e.sum(), 1e-30))


def moments(f):
    f = f.ravel()
    z = (f - f.mean()) / max(f.std(), 1e-30)
    return float((z ** 3).mean()), float((z ** 4).mean() - 3.0)


def build_sea_hos(si, unimodal=False):
    """HOS analog of build_sea_crosstp / build_sea_bimodal(unimodal=True).

    Identical parameter rng streams and call order; the surface (obs z, GT
    field, encounter GT) comes from an M=3 HOS evolution. Returns the HOS
    world SH, the draw-paired linear twin SL, encounter query/GT for both,
    and nonlinearity diagnostics. None if the HOS run went non-finite.
    """
    if unimodal:
        rng = np.random.default_rng(9100 + 31 * si)   # build_sea_bimodal stream
        Hs = rng.uniform(1.5, 3.0)
        r2 = 0.0
        d1 = rng.uniform(0, 360)
        sign = 1.0 if rng.uniform() < 0.5 else -1.0
        d2 = d1 + sign * rng.uniform(60, 150)
        Tp2 = TP1
    else:
        rng = np.random.default_rng(9700 + 31 * si)   # build_sea_crosstp stream
        Hs = rng.uniform(1.5, 3.0)
        r2 = rng.uniform(0.2, 0.5)
        d1 = rng.uniform(0, 360)
        sign = 1.0 if rng.uniform() < 0.5 else -1.0
        d2 = d1 + sign * rng.uniform(60, 150)
        Tp2 = float(TP1 * rng.uniform(0.75, 1.4))

    # ---- HOS IC: two JONSWAP partitions on the periodic grid --------------
    sim = HOSSimulator(HOSConfig(Lx=L_DOM, Ly=L_DOM, Nx=NSPEC, Ny=NSPEC,
                                 depth=DEPTH, M=M_HOS))
    sim.initialize_jonswap(Hs=Hs * np.sqrt(1 - r2) if r2 > 0 else Hs, Tp=TP1,
                           gamma=3.3, direction_deg=d1,
                           direction_spread_deg=SPREAD_DEG, seed=4000 + si)
    eta1, phi1 = sim.eta_hat.copy(), sim.phi_s_hat.copy()
    if r2 > 0:
        sim.initialize_jonswap(Hs=Hs * np.sqrt(r2), Tp=Tp2, gamma=3.3,
                               direction_deg=d2,
                               direction_spread_deg=SPREAD_DEG, seed=5000 + si)
        sim.eta_hat = sim.eta_hat + eta1
        sim.phi_s_hat = sim.phi_s_hat + phi1
    sim.t = 0.0

    # linear twin = the same IC modes (masked for point sums, full for ifft)
    m = np.abs(sim.eta_hat) > 1e-15
    tw = dict(kx=sim.KX[m].copy(), ky=sim.KY[m].copy(),
              a=sim.eta_hat[m].copy(), om=sim.omega[m].copy())
    eh0 = sim.eta_hat.copy()
    om_grid = sim.omega.copy()
    tail0 = tail_frac(sim.eta_hat, sim.K, sim.K.max())

    # ---- geometry / track / noise: verbatim arena conventions -------------
    lam = LAM1
    cp = lam / TP1
    T = 10 * TP1                            # 70 s
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
    Pq = np.column_stack([Xs, Ys, np.full(len(Xs), T)])
    U = rng.uniform(0.0, 1.5) * cp
    nfrac = rng.uniform(0.02, 0.09)
    P = sample_obs('fixed' if U < 1e-6 else 'moving', U, lam, cp, T, gx, gy,
                   r_fp, rng)
    noise = rng.normal(0, nfrac * Hs, N_OBS)

    # ---- evolve HOS through the obs times, then GT, then encounter --------
    advance_to(sim, SPINUP)
    zH = np.empty(N_OBS)
    for i in range(N_OBS):
        advance_to(sim, SPINUP + P[i, 2])
        zH[i] = eta_hos_at(sim, P[i, 0], P[i, 1])[0]
    advance_to(sim, SPINUP + T)
    if not np.isfinite(sim.eta_hat).all() or not np.isfinite(zH).all():
        return None
    gtH = eta_hos_at(sim, Xs, Ys)
    fH = sim.get_eta()
    skH, kuH = moments(fH)
    hs_T = 4.0 * float(fH.std())
    tailT = tail_frac(sim.eta_hat, sim.K, sim.K.max())

    te = T_END + np.linspace(HORIZON / N_ENC, HORIZON, N_ENC)
    S_geom = dict(U=float(U))
    Pq_e = track_future(S_geom, te)
    gtH_e = np.empty(N_ENC)
    for i in range(N_ENC):
        advance_to(sim, SPINUP + te[i])
        gtH_e[i] = eta_hos_at(sim, Pq_e[i, 0], Pq_e[i, 1])[0]
    if not np.isfinite(gtH_e).all():
        return None

    # ---- linear twin renders (same modes, same times, same noise) ---------
    zL = eta_lin_obs(tw, P, SPINUP)
    gtL = eta_lin_at(tw, Xs, Ys, SPINUP + T)
    gtL_e = np.array([eta_lin_at(tw, Pq_e[i, 0], Pq_e[i, 1],
                                 SPINUP + te[i])[0] for i in range(N_ENC)])
    # twin moments on the SAME full-domain grid as fH (support-matched
    # contrast; exact linear evolution of the identical IC)
    fL = np.real(np.fft.ifft2(eh0 * np.exp(-1j * om_grid * (SPINUP + T)))) \
        * (NSPEC * NSPEC)
    skL, kuL = moments(fL)

    base = dict(P=P, Pq=Pq, sigma0=float(nfrac * Hs), Hs=float(Hs),
                d1=float(np.deg2rad(d1)), d2=float(np.deg2rad(d2)),
                r2=float(r2), Tp2=float(Tp2), U=float(U))
    zHn, zLn = zH + noise, zL + noise
    SH = dict(base, z=zHn, gt_u=gtH, kp=two_stage_kp(P, zHn))
    SL = dict(base, z=zLn, gt_u=gtL, kp=two_stage_kp(P, zLn))
    diag = dict(skew_H=skH, skew_L=skL, exkurt_H=kuH, exkurt_L=kuL,
                hs_drift=float(abs(hs_T - Hs) / Hs),
                tail0=tail0, tailT=tailT,
                zcorr=float(corr(zH, zL)),
                steep=float(KP1 * Hs / 4.0))
    return dict(SH=SH, SL=SL, Pq_e=Pq_e, gtH_e=gtH_e, gtL_e=gtL_e, diag=diag)


# BUG-GUARD note: eta_lin_at with per-point times — the vector call above
# passes a SCALAR t only; obs times differ per point, so compute zL properly:
def eta_lin_obs(tw, P, spinup):
    """Linear twin at obs points, each with its own time (vectorized)."""
    out = np.zeros(len(P))
    B = 2048
    t = P[:, 2] + spinup
    for i in range(0, len(tw['kx']), B):
        ph = (np.outer(tw['kx'][i:i + B], P[:, 0])
              + np.outer(tw['ky'][i:i + B], P[:, 1])
              - np.outer(tw['om'][i:i + B], t))
        out += np.real(tw['a'][i:i + B][:, None] * np.exp(1j * ph)).sum(0)
    return out


# ==========================================================================
#  arm scoring (one world)
# ==========================================================================
def enc_scores(S, g, Pq_e, gt_e):
    fe = field_of(S, g, Pq_e)
    if fe is None:
        return np.nan, np.nan
    h = N_ENC // 2
    return float(corr(fe, gt_e)), float(corr(fe[:h], gt_e[:h]))


def score_crosstp_world(S, Pq_e, gt_e, buoy_thetas):
    row = {}
    grids = {'cur': np.maximum(10.0 ** net_gamma_map(S), 1e-8),
             'm2': x_marginals(S)['m2'],
             'orc': supplied_grid(S, S['d1'], S['d2'], S['Tp2'], S['r2'])}
    for nm, g in grids.items():
        row[f'fld_{nm}'] = solve_with_grid(S, g)
        row[f'enc10_{nm}'], row[f'enc05_{nm}'] = enc_scores(S, g, Pq_e, gt_e)
    fl, e10, e05 = [], [], []
    for (d1, d2, Tp2, r2) in buoy_thetas:
        g = supplied_grid(S, d1, d2, Tp2, r2)
        fl.append(solve_with_grid(S, g))
        a, b = enc_scores(S, g, Pq_e, gt_e)
        e10.append(a)
        e05.append(b)
    row['fld_buoy'] = float(np.nanmean(fl))
    row['enc10_buoy'] = float(np.nanmean(e10))
    row['enc05_buoy'] = float(np.nanmean(e05))
    return row


def score_uni_world(S, Pq_e, gt_e, draws):
    """e45 arms. draws = dict with lists of pre-drawn thetas (shared H/L)."""
    row = {}
    for nm, g in [('cur', np.maximum(10.0 ** net_gamma_map(S), 1e-8)),
                  ('uni_orc', one_lobe(S, S['d1']))]:
        row[f'fld_{nm}'] = solve_with_grid(S, g)
        row[f'enc10_{nm}'], _ = enc_scores(S, g, Pq_e, gt_e)
    for arm in ['uni_buoy', 'ph_buoy', 'ph_ww3']:
        fl, e10 = [], []
        for th in draws[arm]:
            g = one_lobe(S, th[0]) if arm == 'uni_buoy' else two_lobe(S, *th)
            fl.append(solve_with_grid(S, g))
            a, _ = enc_scores(S, g, Pq_e, gt_e)
            e10.append(a)
        row[f'fld_{arm}'] = float(np.nanmean(fl))
        row[f'enc10_{arm}'] = float(np.nanmean(e10))
    return row


def uni_draws(S, si):
    """Pre-draw e45-style theta perturbations once (shared by H and L)."""
    rng = np.random.default_rng(97000 + si)
    d = {'uni_buoy': [], 'ph_buoy': [], 'ph_ww3': []}
    for _ in range(R_UNI):
        d['uni_buoy'].append((S['d1'] + np.deg2rad(rng.normal(0, 5.0)),))
        Tp2 = 7.0 * rng.uniform(0.75, 1.4)
        r2 = rng.uniform(0.2, 0.5)
        for grade, sdg in [('ph_buoy', 5.0), ('ph_ww3', 15.0)]:
            d1p = S['d1'] + np.deg2rad(rng.normal(0, sdg))
            d2p = S['d2'] + np.deg2rad(rng.normal(0, sdg))
            Tp2p = float(np.clip(Tp2 * (1 + rng.normal(
                0, (5.0 if grade == 'ph_buoy' else 10.0) / 100)), 3.0, 16.0))
            r2p = float(np.clip(r2 + rng.normal(
                0, 0.05 if grade == 'ph_buoy' else 0.10), 0.05, 0.6))
            d[grade].append((d1p, d2p, Tp2p, r2p))
    return d


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n', type=int, default=192)
    ap.add_argument('--n_uni', type=int, default=64)
    ap.add_argument('--si0', type=int, default=5000)
    ap.add_argument('--si0_uni', type=int, default=6000)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    if a.smoke:
        a.n, a.n_uni, a.si0, a.si0_uni = 2, 2, 4990, 6990
        a.out = 'tmp/e46_smoke.json'
    t0 = time.time()

    rows, excl = [], []
    print(f'--- HOS cross-Tp block si={a.si0}..{a.si0+a.n-1} (n={a.n}, '
          f'M={M_HOS}, dt={DT}, grid {NSPEC}^2 on {L_DOM:.0f} m) ---',
          flush=True)
    for k, si in enumerate(range(a.si0, a.si0 + a.n)):
        W = build_sea_hos(si)
        if W is None:
            excl.append(si)
            print(f'  si{si} EXCLUDED (non-finite HOS state)', flush=True)
            continue
        SH, SL = W['SH'], W['SL']
        rngb = np.random.default_rng(96000 + 7 * si + 11)
        sd, sT, sw = BUNDLES['buoy']
        buoy_thetas = [perturb(SH, rngb, sd=sd, sT=sT, sw=sw)
                       for _ in range(RB)]
        row = dict(si=si, U=SH['U'], r2=SH['r2'], Tp2=SH['Tp2'], **W['diag'])
        for pre, S, gt_e in [('H', SH, W['gtH_e']), ('L', SL, W['gtL_e'])]:
            r = score_crosstp_world(S, W['Pq_e'], gt_e, buoy_thetas)
            row.update({f'{pre}_{kk}': v for kk, v in r.items()})
        rows.append(row)
        if k % 8 == 0 or a.smoke:
            print(f"  si{si} skH {row['skew_H']:+.3f} skL {row['skew_L']:+.3f}"
                  f" | H fld orc-cur {row['H_fld_orc']-row['H_fld_cur']:+.3f}"
                  f" buoy-cur {row['H_fld_buoy']-row['H_fld_cur']:+.3f}"
                  f" | L orc-cur {row['L_fld_orc']-row['L_fld_cur']:+.3f}"
                  f" [{time.time()-t0:.0f}s]", flush=True)
            json.dump(dict(rows=rows, partial=True), open(a.out, 'w'),
                      indent=1, default=float)

    rows_uni, excl_uni = [], []
    print(f'--- HOS unimodal phantom block si={a.si0_uni}.. (n={a.n_uni}) ---',
          flush=True)
    for k, si in enumerate(range(a.si0_uni, a.si0_uni + a.n_uni)):
        W = build_sea_hos(si, unimodal=True)
        if W is None:
            excl_uni.append(si)
            print(f'  si{si} EXCLUDED (non-finite HOS state)', flush=True)
            continue
        SH, SL = W['SH'], W['SL']
        draws = uni_draws(SH, si)
        row = dict(si=si, **W['diag'])
        for pre, S, gt_e in [('H', SH, W['gtH_e']), ('L', SL, W['gtL_e'])]:
            r = score_uni_world(S, W['Pq_e'], gt_e, draws)
            row.update({f'{pre}_{kk}': v for kk, v in r.items()})
        rows_uni.append(row)
        if k % 8 == 0 or a.smoke:
            print(f"  si{si} skH {row['skew_H']:+.3f}"
                  f" | H ph_buoy-cur fld "
                  f"{row['H_fld_ph_buoy']-row['H_fld_cur']:+.3f}"
                  f" [{time.time()-t0:.0f}s]", flush=True)

    # ====================== SANITY / GATES ================================
    print(f'\n=== NONLINEARITY SANITY (crossTp n={len(rows)}, '
          f'excluded {len(excl)}) ===')
    g = lambda kk, R=rows: np.array([r[kk] for r in R], float)
    skH, skL = g('skew_H'), g('skew_L')
    print(f"  skew  HOS {skH.mean():+.4f} (sd {skH.std(ddof=1):.4f}) | "
          f"twin {skL.mean():+.4f} | d {np.mean(skH-skL):+.4f}")
    print(f"  exkurt HOS {g('exkurt_H').mean():+.4f} | "
          f"twin {g('exkurt_L').mean():+.4f}")
    print(f"  Hs drift {g('hs_drift').mean():.4f} | tail0 "
          f"{g('tail0').mean():.2e} -> tailT {g('tailT').mean():.2e} | "
          f"corr(zH,zL) {g('zcorr').mean():.4f} | steep "
          f"{g('steep').min():.3f}..{g('steep').max():.3f}")
    nl_ok = bool(skH.mean() >= 0.05 and (skH - skL).mean() >= 0.04)
    excl_ok = bool(len(excl) <= 0.05 * a.n and len(excl_uni) <= 0.05 * a.n_uni)
    print(f"  NONLINEARITY PRESENT: {nl_ok} | exclusions OK: {excl_ok}")

    res = {}
    print(f'\n=== HARNESS GATE (linear twin, n={len(rows)}) ===')
    an_L = paired(rows, 'L_fld_orc', 'L_fld_cur')
    res['twin_fld_orc-cur'] = an_L
    print(f"  twin fld orc-cur: {an_L['mean']:+.4f} +- {an_L['se']:.4f} "
          f"CI [{an_L['lo']:+.4f},{an_L['hi']:+.4f}] win {an_L['win']}/{an_L['n']}")
    harness_ok = bool(0.04 <= an_L['mean'] <= 0.11)
    print(f"  HARNESS GATE (twin anchor in [0.04,0.11]): {harness_ok}")

    print(f'\n=== ★HOS BLOCK — field functional (primary) ===')
    for aa, bb in [('orc', 'cur'), ('orc', 'm2'), ('m2', 'cur'),
                   ('buoy', 'cur'), ('buoy', 'm2')]:
        for pre in ['H', 'L']:
            r = paired(rows, f'{pre}_fld_{aa}', f'{pre}_fld_{bb}')
            res[f'{pre}_fld_{aa}-{bb}'] = r
            star = '*' if (r['lo'] > 0 or r['hi'] < 0) else ' '
            print(f"  [{pre}] fld {aa:4s} - {bb:4s}: {r['mean']:+.4f} "
                  f"+- {r['se']:.4f} CI [{r['lo']:+.4f},{r['hi']:+.4f}]{star} "
                  f"win {r['win']}/{r['n']}")

    print(f'\n=== HOS BLOCK — encounter functional (secondary/pilot) ===')
    for tag in ['enc10', 'enc05']:
        for aa, bb in [('orc', 'cur'), ('orc', 'm2'), ('buoy', 'cur'),
                       ('buoy', 'm2'), ('m2', 'cur')]:
            for pre in ['H', 'L']:
                r = paired(rows, f'{pre}_{tag}_{aa}', f'{pre}_{tag}_{bb}')
                res[f'{pre}_{tag}_{aa}-{bb}'] = r
                star = '*' if (r['lo'] > 0 or r['hi'] < 0) else ' '
                print(f"  [{pre}] {tag} {aa:4s} - {bb:4s}: {r['mean']:+.4f} "
                      f"+- {r['se']:.4f} CI [{r['lo']:+.4f},{r['hi']:+.4f}]"
                      f"{star} win {r['win']}/{r['n']}")
    d_enc = np.array([r['H_enc10_orc'] - r['H_enc10_cur'] for r in rows])
    d_enc = d_enc[np.isfinite(d_enc)]
    try:
        wilcox_p = float(stats.wilcoxon(d_enc)[1])
    except Exception:
        wilcox_p = float('nan')
    print(f"  Wilcoxon p (H enc10 orc-cur) = {wilcox_p:.3g}")

    print(f'\n=== EXPLORATORY (labeled): draw-paired physics effect '
          f'dd = (arm-cur)_H - (arm-cur)_L ===')
    for arm in ['orc', 'buoy', 'm2']:
        dd = np.array([(r[f'H_fld_{arm}'] - r['H_fld_cur'])
                       - (r[f'L_fld_{arm}'] - r['L_fld_cur']) for r in rows])
        dd = dd[np.isfinite(dd)]
        n = len(dd)
        se = dd.std(ddof=1) / np.sqrt(n)
        lo, hi = stats.t.interval(0.95, n - 1, loc=dd.mean(), scale=se)
        res[f'dd_fld_{arm}'] = dict(mean=float(dd.mean()), se=float(se),
                                    lo=float(lo), hi=float(hi), n=n)
        print(f"  dd fld {arm:4s}: {dd.mean():+.4f} +- {se:.4f} "
              f"CI [{lo:+.4f},{hi:+.4f}]")

    print(f'\n=== UNIMODAL PHANTOM do-no-harm (HOS, n={len(rows_uni)}) ===')
    for tag in ['fld', 'enc10']:
        for arm in ['uni_orc', 'uni_buoy', 'ph_buoy', 'ph_ww3']:
            for pre in ['H', 'L']:
                r = paired(rows_uni, f'{pre}_{tag}_{arm}', f'{pre}_{tag}_cur')
                res[f'{pre}_{tag}_{arm}-cur'] = r
                star = '*' if (r['lo'] > 0 or r['hi'] < 0) else ' '
                print(f"  [{pre}] {tag:5s} {arm:8s} - cur: {r['mean']:+.4f} "
                      f"+- {r['se']:.4f} CI [{r['lo']:+.4f},{r['hi']:+.4f}]"
                      f"{star} win {r['win']}/{r['n']}")

    # ====================== VERDICT =======================================
    print(f'\n=== VERDICT (pre-registered bars) ===')
    A = res.get('H_fld_orc-cur')
    Bc = res.get('H_fld_buoy-cur')
    Bm = res.get('H_fld_buoy-m2')
    PHf = res.get('H_fld_ph_buoy-cur')
    PHe = res.get('H_enc10_ph_buoy-cur')
    anchor_h = bool(A and A['mean'] >= 0.03 and A['lo'] > 0)
    prize_killed = bool(A and A['hi'] < 0.02)
    dnh = bool(PHf and PHe and PHf['mean'] >= -0.01 and PHe['mean'] >= -0.01)
    pass_ = bool(Bc and Bm and Bc['mean'] >= 0.03 and Bc['lo'] > 0
                 and Bm['lo'] > 0 and dnh)
    kill = bool(Bc and Bc['hi'] < 0.02)
    print(f"  nonlinearity present        : {nl_ok}")
    print(f"  exclusions <= 5%            : {excl_ok}")
    print(f"  HARNESS GATE (twin)         : {harness_ok}")
    print(f"  ANCHOR-H prize survives     : {anchor_h}"
          f"  (prize killed: {prize_killed})")
    print(f"  do-no-harm (ph_buoy >= -.01): {dnh}")
    print(f"  PASS (buoy survives HOS)    : {pass_}")
    print(f"  KILL (buoy CI hi < +0.02)   : {kill}")
    if not (nl_ok and excl_ok):
        v = 'NO VERDICT: HOS leg invalid (nonlinearity absent or exclusions)'
    elif not harness_ok:
        v = 'NO VERDICT: harness gate failed on the linear twin'
    elif pass_:
        v = ('PASS: the buoy-grade external-supply headline SURVIVES HOS '
             'nonlinearity')
    elif kill and prize_killed:
        v = ('KILLED: the regime prize itself is a linear-superposition '
             'artifact; headline scoped to linear seas')
    elif kill:
        v = ('KILLED: prize survives but buoy-grade supply does not clear '
             'the bar under HOS; headline scoped to linear seas')
    else:
        v = 'INCONCLUSIVE between the bars (report with achieved MDE)'
    print(f'  -> {v}')

    json.dump(dict(rows=rows, rows_uni=rows_uni, contrasts=res,
                   excluded=excl, excluded_uni=excl_uni,
                   wilcoxon_p_enc10=wilcox_p,
                   nl_ok=nl_ok, excl_ok=excl_ok, harness_ok=harness_ok,
                   anchor_h=anchor_h, prize_killed=prize_killed, dnh=dnh,
                   pass_=pass_, kill=kill, verdict=v,
                   cfg=dict(NSPEC=NSPEC, L_DOM=L_DOM, M=M_HOS, DT=DT,
                            SPINUP=SPINUP, RB=RB, R_UNI=R_UNI,
                            n=a.n, n_uni=a.n_uni, si0=a.si0,
                            si0_uni=a.si0_uni)),
              open(a.out, 'w'), indent=1, default=float)
    print(f'saved -> {a.out}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
