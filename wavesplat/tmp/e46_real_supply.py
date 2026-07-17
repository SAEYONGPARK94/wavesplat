"""e46 — ★REAL-DATA LEG of the surviving deliverable (buoy-grade external
regime supply, scored on the deployment functional).

WHY. The one headline that survived the e45 two-agent audit is 100% synthetic:
in the cross-Tp JONSWAP arena the sea-regime prize transfers to the encounter
functional (enc10 orc-cur +0.0431+-0.0077, n=957) and BUOY-grade external
theta recovers +0.0349 CI[+0.0084,+0.0615], while forecast(WW3)-grade supply
is DEAD (frozen-axis error -0.0092 CI contains 0; phantom-lobe do-no-harm
-0.0158 < -0.01 bar). The arena's 2-lobe family is well-specified BY
CONSTRUCTION (lobe_grid builds exactly the ensemble build_sea_crosstp samples
from). On real seas the family is MISSPECIFIED. If buoy-grade supply still
pays on real stereo data, the claim escapes "synthetic JONSWAP only"; if not,
the deliverable is scoped to simulation.

DATA. Guimaraes 2020 stereo surfaces (data/real_bergamasco), 12 records
(LJ18n excluded: documented negative-control 40 m patch under lam~300 m swell
— cannot host any lam-scaled track). Truth = the dense stereo field.

================================ PRE-REGISTRATION =============================
(frozen before any real-data scoring; the synthetic self-test may only gate
 the harness, never tune constants against real scores)

PRECONDITION DELIVERABLE (added per coordinator 2026-07-17, still BEFORE any
real-data scoring. Context: the coordinator's own snapshot-FFT bimodality
screen was DISCARDED after a unimodal control fired 6/6 false positives —
a direction-folded snapshot |FFT2|^2 cannot separate systems in principle,
and its exclusion mask handed back the first boundary point, the exact
e37-detector/PERM failure mode "a detector that never abstains measures
nothing".)
  REGIME CENSUS — does the arc's target regime (cross-Tp bimodal) exist in
  these records AT ALL? Screen = the same dispersion-gated (theta, ln k)
  machinery as the supply estimator: TIME axis included (3D FFT, omega>0
  half-space resolves the 180 deg ambiguity), PER-SITE depth in the
  dispersion shell, record-level average spectrum over all windows.
  STRUCTURAL ABSTENTION: peak 2 must be a GENUINE 2D local maximum
  (theta-wrapped 8-neighborhood) of the smoothed hist — a monotone JONSWAP
  shoulder or a mask-boundary point can never fire.
  SCREEN VALIDATION, reported BEFORE real data is touched:
    unimodal control   n=6  build_sea_bimodal(5100+j, unimodal=True)
                       through the identical pipeline: false-positive rate
                       must be <= 1/6, else the screen is not evidence and
                       the census is NOT reported as a finding.
    cross-Tp control   n=6  build_sea_crosstp(si=5000..5005, fresh block):
                       detection >= 4/6, mean direction error <= 20 deg,
                       mean |dTp|/Tp <= 12%.
  CROSS-CHECK: spectral Tp(dominant lobe) vs time-domain peak_period per
  record; ratio outside [0.77, 1.30] flagged (the depth-bug signature the
  coordinator hit on LJ: spectral 5.1 s vs time 12.0 s at wrong depth).
  If NO record is genuinely cross-Tp bimodal, the supply experiment cannot
  test the surviving claim on its target regime — that is a standalone
  finding (claim stays synthetic-scoped for a reason unrelated to the
  estimator), and the run is reported as census + do-no-harm only.

RECORDS / WINDOWS
  12 records, non-overlapping 80 s windows (70 s obs + 10 s horizon), start
  spacing 85 s, max 10 windows/record => expected n ~ 110. Windows within a
  record share a sea state; records within a day share a sea event. PRIMARY
  clustering = sea-event (10 clusters: BS11-1001, BS11-1004{a,c,d}, BS13-0922,
  BS13-0925, BS13-0930, AA14, AA15-0305, AA15-0515, YS17, LJ18). Naive
  per-window stats are reported alongside; the VERDICT reads the clustered CI
  (the E-14 audit found exactly this error class in this project).

OBSERVATION (E-9 sparse protocol / arena mirror)
  Ping-pong inset diagonal (better-valid of SW-NE / NW-SE, chosen by valid
  fraction along the segment — score-blind), speed U ~ U(0.3,1.5)*cp_ref
  capped at 12 m/s (design speed), footprint jitter disc r_fp = fp_ref/2,
  N_OBS = 400 draws over 70 s, z by NaN-aware bilinear at the nearest frame
  (frame time used as the sample time), NaN dropped; window DROPPED if
  n_valid < 250. Declared noise sigma0 = 0.05 * Hs_hat (Hs_hat = 4*std(z));
  N(0, sigma0) ADDED to z; solver noise floor = (3*sigma0)^2 (SMULT=3, arena
  convention, no per-window selection). Stereo error rides on top — absorbed
  by the floor, stated honestly.

SUPPLIED THETA (the "co-located buoy")
  Per window: dispersion-gated 3D FFT of the DENSE 70 s cube -> directional
  wavenumber energy on (theta, ln k); record-level LEAVE-ONE-WINDOW-OUT
  average (supply is regime-level, shares NO realization with the scored
  window); frozen partition: 72x48 hist, smooth (10 deg, 0.15 in ln k),
  peak 2 outside (dTheta<40 AND dlnk<0.30) exclusion, accept if
  E2 >= 0.20*E1; soft-assignment centroids -> (d_i, Tp_i(depth), r2),
  r2 clipped [0.05,0.6]. This dense-field estimate is an UPPER BOUND on what
  a real buoy reports (full spatial info vs point f-theta) — stated.
  Family constants gamma=3.3, spread=30 stay FROZEN: that IS the
  misspecification under test.

ARMS (all through the identical fixed solver; depth-correct dispersion)
  cur   incumbent net (10**net_gamma_map), zero-shot on real maps (precedent:
        G3 / 8-record 3-way)
  m2    honest all-marginal [MANDATORY GATE: cur~=m2 throughout the arc, so
        beating cur without beating m2 recovers nothing]:
        0.65*L1(mu_hat, Tp(kp_hat)) + 0.35*L2smear(mu_hat +- D2_OFF_NODES,
        Tp(kp_hat)*TP2_RATIO_NODES) — x_marginals verbatim with the arena
        constant Tp1=7 replaced by the free two-stage kp read-off
  ens1  smooth 1-lobe ensemble baseline = L1 alone
  orc   supplied LOWO dense-theta, exact (clean external supply ceiling)
  buoy  orc degraded at buoy grade: d_i+-5deg, Tp_i+-5% (BOTH lobes — Tp1 is
        NOT frozen here, stricter than e43), r2+-0.05; R=5 draws averaged
        per window BEFORE pairing (E-14: one draw per cell measures draw
        variance, not the effect)

SCORE
  enc10/enc05: corr along the extrapolated track (no jitter), t in (70,80],
  40 points, gt = dense field (drop window if <25 valid). fld: corr at up to
  1200 valid pixels at the last obs frame outside the track corridor
  (dist > r_fp + 0.6*fp_ref), for comparability with the arena metric.

HARNESS GATE (must pass BEFORE the real run is read at all)
  = the SCREEN VALIDATION above (6 cross-Tp + 6 unimodal synthetic seas,
  FRESH si; burned blocks 0..31, 100..163, 200..263, 300..347, 400..447,
  900..915, 1000..1959, 4000..4239) rendered to dense cubes and pushed
  through the IDENTICAL pipeline (loader, track, partition, arms), plus
  fld orc-cur mean > 0 on the cross-Tp seas and a plane-wave micro-test of
  the FFT direction disambiguation.
  If the self-test PASSES and the real-data anchor FAILS, that is a
  SCIENTIFIC negative (family dead on real seas), not a harness bug.

ANCHOR (real data): fld orc-cur clustered mean > 0 and plausible (< +0.30).
  If the anchor fails, nothing downstream is read as confirmatory.

BARS (primary contrast = enc10 buoy-cur, sea-event-clustered t-CI)
  PASS  mean >= +0.03 AND clustered CI excludes 0 AND (buoy-m2) clustered CI
        excludes 0.
  KILL  clustered CI upper < +0.02  => the spec does NOT survive real seas
        => deliverable scoped to synthetic JONSWAP permanently.
  DNH   on windows whose LOWO supply is classified UNIMODAL: buoy-cur mean
        >= -0.01 (real-world phantom-lobe do-no-harm analog).

POWER (stated up front — this arc's disease is claims below their own MDE)
  e44 measured per-sea sd(enc10 orc-cur) = 0.372. If real per-window sd is
  similar, n~110 gives naive MDE ~ 2.0*0.372/sqrt(110) ~ 0.071 > the +0.03
  bar, and the 10-cluster CI is wider still. THEREFORE this run is
  pre-declared a PILOT unless the measured clustered MDE < 0.03: it can
  (i) detect effects ~ +-0.07-0.10, (ii) kill on a strongly negative mean,
  (iii) measure the real-data effect sd for a powered design. A verdict
  between the bars at MDE > bar is "underpowered pilot", not "null".

STATS  t-based CIs ONLY (the 1.96 bug recurred at e41:356). Deterministic
  seeds only (never Python hash()). No modification of existing scripts.

AMENDMENTS (2026-07-17, after --smoke, BEFORE the real scoring run; the
smoke exposed 2 BS13a windows' scores, but none of these decisions depend
on which arm scored higher there — they are driven by geometry/validity
diagnostics and by SYNTHETIC-control accuracy only):
  A1  RADIAL COORDINATE FROM omega. Smoke census showed BS13a spectral
      Tp 2.0 s vs time-domain 3.3 s (the pre-registered cross-check FLAG
      fired): the BS13 patch is 19x19 m ~ 1 lambda_p, so the spatial-FFT
      k-axis cannot resolve the peak (and kmin censored it). Fix = the
      buoy principle: for gated cells the hist radial coordinate is
      u = ln k_disp(omega_bin) (dispersion-inverted, per-site depth);
      direction still from the (kx,ky) cell. 70 s windows give
      d_omega/omega_p ~ 5% (BS) vs k-bin >= 100% there. Known bias:
      surface-current Doppler shifts omega by k*Uc (~3% at BS) — accepted.
      kmin relaxed to 1.0x the box fundamental (direction validity only).
  A2  kp_hat GUARDED BY RECORD METADATA. two_stage_kp on 70 s sparse BS
      tracks jumped 0.14 -> 1.17 rad/m between windows of the same sea,
      poisoning every arm's bank equally. Stage-1 klim = min(2.0,
      3.5*kp_ref), final kp_hat clipped to [0.4, 2.5]*kp_ref, with
      kp_ref = 2*pi/(fp_ref/0.18) from the record characterization used
      project-wide (static per-record metadata, identical for all arms —
      not a GT peek; E-11 precedent: sparse/metadata hints are free).
  A3  TRACK INSIDE THE VALID BOX. YS17 windows all dropped: its valid
      region is an offset sub-box (marginal-valid x[-255,-95] y[-85,74]).
      The diagonal is now inset within the marginal>=0.4 valid bounding
      box (fallback: full grid).
  A4  SELF-TEST additionally reports (diagnostic, not gated) partition
      accuracy at a 1.0-lambda box — the BS13/LJ aperture regime.
  A5  OFF-SHELL NOISE-FLOOR SUBTRACTION per omega bin (median power of
      cells with |om-om_d| > 2.5*gate). Cause: gated cell count grows ~k^2,
      so integrated stereo noise tilts the histogram to high omega (BS13a
      census Tp 1.8 s vs time-domain 3.3 s, flagged by the pre-registered
      cross-check).
  A6  PARTITION v2 — SCREEN SCOPE NARROWED TO THE ARENA REGIME. The
      unimodal control kept firing: a broad 30-deg-spread single lobe
      SPLITS into two realization local maxima ~41 deg apart (mass 0.44 vs
      0.56 — no mass test can separate them), and the ln-k escape route
      lets within-lobe omega-tail lumps in. The arc's target regime is
      DIRECTION-SEPARATED cross-Tp (d-sep in [60,150] deg), so the
      exclusion becomes dTheta < 55 deg at ALL k, plus soft-mass ratio
      >= 0.12. STATED BLIND SPOTS of the screen: (i) same-direction
      swell+windsea (out of screen scope — it is not the arena regime),
      (ii) swell-side secondaries with lambda_2 >~ box/1.5 (aperture).
  A7  SELF-TEST FIDELITY = the real supply path: each synthetic sea is
      rendered as 4 quasi-independent 70-80 s segments; census = 4-average
      hist, supply/accuracy = 3-average LOWO hist (the real leg averages
      5-13 windows per record). Single-window partition accuracy is NOT
      the deployed path and is no longer gated.
  A8  HARNESS ANCHOR = TRUE-theta ORACLE, n=32 (the first anchor gate was
      wrong, and diagnosing it produced a finding). The n=6 estimated-theta
      anchor FAILED (-0.084). A/B bisecting showed: (i) arena-NATIVE
      scoring of the same fresh seas reproduces the prize (orcT-cur +0.097,
      6/6 protocol); (ii) my pipeline with TRUE theta at the arena's own U
      reproduces +0.056; (iii) interpolation/noise/query are innocent —
      the per-sea prize is TRACK-DRAW sensitive (sd ~0.12 across U draws),
      so ANY n=6 anchor is underpowered by construction. Gate becomes:
      orcT-cur over n=32 fresh seas (si 5000..5031), mean must be in
      (0, +0.20).
      ★FINDING (reported, not gated): the LOWO dense-field partition at a
      2-lambda box supplies theta at ~(12 deg, 10% Tp) grade — between the
      arc's 'buoy' (5 deg/5%) and DEAD 'ww3' (15 deg/10%) bundles — and
      its supplied arm scored -0.084 vs cur on the same seas. "A dense
      stereo patch gives buoy-grade regime supply" is FALSE at 2-lambda
      aperture; the real leg's orc arm is therefore detector-limited, not
      a clean ceiling. An aperture-grade curve (boxes 1.0/2.0/3.8 lambda)
      is reported to interpret per-record supply grade (YS17 ~ 3.8).
  A9  PER-OMEGA DENSITY NORMALIZATION (buoy/Welch convention) + None
      guards. The u-histogram summed omega bins per u-bin, and du =
      2 domega/omega (deep) makes that count grow ~omega -> broad spectra
      tilt to high frequency (BS13a census Tp 1.8 s vs Welch 4.8 s — the
      pre-registered cross-check flag caught it mid-run). H is now
      normalized by the omega-bin count per u-bin. Also: a supplied ring
      outside the bank band (possible for short-Tp seas after clipping)
      yields a zero grid -> the arm is NaN for that window (was a crash).

Usage:  python3 tmp/e46_real_supply.py [--smoke] [--selftest-only]
                                       [--records YS17 AA15a ...]
Out:    results/direction3/e46_real_supply.json
==============================================================================
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

from e29_gennet_C import (load_incumbent, DEVICE, _MASK, _KXN, _KYN,
                          NGRID, KREL)
from e41_regime_prize_ladder import deposit_fast, angle_mix, \
    TP2_RATIO_NODES, D2_OFF_NODES, W_MARG
from d3_sa_nomp import beamform_power, split_masks
from d18_relax_estimator import beamform_top
from d3_sa_multirecord import MRecord
from d3_falcon_real_operator import bilinear
from realdata_analysis import peak_period
from wavesim.wave_generator import jonswap_components

OUT = 'results/direction3/e46_real_supply.json'
G = 9.81
T_END = 70.0          # obs window [s]
HORIZON = 10.0        # encounter horizon [s]
N_ENC = 40
N_OBS = 400           # arena convention (E-9 sparse: never the dense mesh)
SMULT = 3.0           # fixed solver noise multiplier (e29_gate0_ladder)
NFRAC = 0.05          # declared noise fraction of Hs (arena mid)
N_FLD = 1200          # field-query pixels (arena: 1200 held-out px)
U_CAP = 12.0          # m/s, design-speed cap
WIN_SPACING = 85.0    # s between window starts (80 used + 5 guard)
MAX_WIN = 10          # per record
MIN_OBS = 250         # window dropped below this
MIN_ENC = 25          # of N_ENC valid gt points

# frozen partition constants (pre-registered; v2 per amendment A6)
NTH_H, NU_H = 72, 48
U_LO, U_HI = np.log(0.02), np.log(6.3)          # ln k bounds [rad/m]
SM_TH_DEG, SM_U = 10.0, 0.15
EXCL_TH_DEG = 55.0                              # peak-2 exclusion, ALL k (A6)
ACCEPT2 = 0.20
MIN_MASS2 = 0.12                                # soft-mass abstention (A6)
SOFT_TH_DEG, SOFT_U = 25.0, 0.26                # soft-assignment scales
TP_CLIP = (1.5, 25.0)

BUOY_SD, BUOY_ST, BUOY_SW = 5.0, 5.0, 0.05      # e43 BUNDLES['buoy']
R_BUOY = 5

# 12 records; fp_ref/Tp_ref from tmp/record_characterization via
# d3_sa_multirecord.RECORDS (BS11c/d, BS13c: same-site siblings, values
# interpolated — they only set track geometry presets, never the score).
RECORDS46 = [
    ('BS11a', 'data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc',
     30.0, 2.5, 3.0, 'BS11-1004'),
    ('BS11b', 'data/real_bergamasco/BS_2011/Surfaces_20111001_161800_short.nc',
     30.0, 1.7, 2.5, 'BS11-1001'),
    ('BS11c', 'data/real_bergamasco/BS_2011/Surfaces_20111004_130700_short.nc',
     30.0, 2.5, 3.0, 'BS11-1004'),
    ('BS11d', 'data/real_bergamasco/BS_2011/Surfaces_20111004_153000_short.nc',
     30.0, 2.5, 3.0, 'BS11-1004'),
    ('BS13a', 'data/real_bergamasco/BS_2013/Surfaces_20130925_121501_short.nc',
     30.0, 3.4, 3.5, 'BS13-0925'),
    ('BS13b', 'data/real_bergamasco/BS_2013/Surfaces_20130930_102001_short.nc',
     30.0, 3.1, 3.3, 'BS13-0930'),
    ('BS13c', 'data/real_bergamasco/BS_2013/Surfaces_20130922_130001_short.nc',
     30.0, 3.2, 3.4, 'BS13-0922'),
    ('AA14', 'data/real_bergamasco/AA_2014/Surfaces_20140327_091000.nc',
     17.0, 6.3, 4.7, 'AA14'),
    ('AA15a', 'data/real_bergamasco/AA_2015/Surfaces_20150305_103500.nc',
     17.0, 9.3, 5.8, 'AA15-0305'),
    ('AA15b', 'data/real_bergamasco/AA_2015/Surfaces_20150515_090000.nc',
     17.0, 7.2, 5.1, 'AA15-0515'),
    ('YS17', 'data/real_bergamasco/YS_2017/Surfaces_20170513_050000.nc',
     50.0, 10.5, 6.1, 'YS17'),
    ('LJ18g', 'data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid2_short.nc',
     45.0, 54.0, 13.0, 'LJ18'),
]


# ==========================================================================
#  depth-parameterized numerics (copies of the arena path, depth freed;
#  existing scripts are NOT modified)
# ==========================================================================
def omega_d(k, depth):
    return np.sqrt(G * k * np.tanh(k * depth) + 1e-12)


def k_disp(om, depth):
    """Invert om^2 = g k tanh(k d) (Newton-free fixed point, as
    dispersion_lam)."""
    k = om ** 2 / G
    for _ in range(60):
        k = om ** 2 / (G * np.tanh(np.maximum(k, 1e-8) * depth))
    return k


def tp_of_k_d(k, depth):
    return float(2 * np.pi / omega_d(k, depth))


def bank_atoms_d(kp, depth):
    """e29_gennet_C.bank_atoms with depth freed (free + bound-wave arms)."""
    kxb = _KXN[_MASK] * kp
    kyb = _KYN[_MASK] * kp
    om = omega_d(np.hypot(kxb, kyb) + 1e-12, depth)
    return (np.concatenate([kxb, 2.0 * kxb]),
            np.concatenate([kyb, 2.0 * kyb]),
            np.concatenate([om, 2.0 * om]))


def design_np_t(kxb, kyb, om, Q):
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    kxb, kyb, om, Q = t(kxb), t(kyb), t(om), t(Q)
    psi = kxb[None] * Q[:, 0:1] + kyb[None] * Q[:, 1:2] - om[None] * Q[:, 2:3]
    return torch.cat([torch.cos(psi), torch.sin(psi)], dim=1)


def solve_fields(P, z, kp, sigma0, depth, g_free, queries):
    """One MMSE factorization per arm, evaluated at several query sets.
    Mirrors e29_gate0_ladder.solve_with_grid / e44.field_of exactly
    (free arm only, zero bound-wave gamma, SMULT fixed)."""
    g = np.concatenate([g_free[_MASK], np.zeros(int(_MASK.sum()))])
    if g.sum() <= 0:
        return [None] * len(queries)
    g = g / g.sum()
    kxb, kyb, om = bank_atoms_d(kp, depth)
    Phi = design_np_t(kxb, kyb, om, P)
    gam = torch.tensor(np.concatenate([g, g]) * float(np.var(z)),
                       dtype=torch.float32, device=DEVICE)
    K = (Phi * gam[None]) @ Phi.T + (SMULT * sigma0) ** 2 * torch.eye(
        len(z), device=DEVICE)
    w = Phi.T @ torch.linalg.solve(
        K, torch.tensor(z, dtype=torch.float32, device=DEVICE))
    out = []
    for Q in queries:
        Phi_q = design_np_t(kxb, kyb, om, Q)
        out.append(((Phi_q * gam[None]) @ w).cpu().numpy())
    return out


def corr(a, b):
    if a is None or b is None:
        return float('nan')
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.size != b.size or a.size < 3:
        return float('nan')
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def two_stage_kp_d(P, z, depth, kp_ref):
    """A2: stage-1 klim bounded by record metadata; final kp clipped to
    [0.4, 2.5]*kp_ref (identical for all arms — no per-arm information)."""
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          depth, klim=min(2.0, 3.5 * kp_ref), n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          depth, klim=3.5 * kp, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    return float(np.clip(kp, 0.4 * kp_ref, 2.5 * kp_ref))


def mu_hat_of_d(P, z, kp, depth):
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          depth, klim=3.5 * kp, n=120)
    return float(np.arctan2(by, bx))


def maps6_d(P, z, kp, depth):
    """e29_gennet_C.maps6 with depth freed (the incumbent's input)."""
    m1, m2 = split_masks(P, mode='legs')
    _, _, p1 = beamform_power(P[m1], z[m1], depth, KREL * kp, NGRID)
    _, _, p2 = beamform_power(P[m2], z[m2], depth, KREL * kp, NGRID)
    p1 = (p1 / max(p1.max(), 1e-12)).reshape(NGRID, NGRID)
    p2 = (p2 / max(p2.max(), 1e-12)).reshape(NGRID, NGRID)
    pm = np.minimum(p1, p2)
    return np.stack([p1, p2, pm,
                     np.log10(p1 + 1e-6) / 6 + 1, np.log10(p2 + 1e-6) / 6 + 1,
                     np.log10(pm + 1e-6) / 6 + 1]).astype(np.float32)


def cur_gamma(P, z, kp, depth):
    net, _ = load_incumbent()
    X = maps6_d(P, z, kp, depth)
    with torch.no_grad():
        Y = net(torch.tensor(X[None], device=DEVICE))[0].cpu().numpy()
    return np.maximum(10.0 ** Y, 1e-8)


_RADC = {}


def radial_profile_d(Tp, gam, depth):
    """e41.radial_profile with depth freed (JONSWAP ensemble radial)."""
    key = (round(float(Tp), 4), round(float(gam), 3), round(float(depth), 1))
    if key not in _RADC:
        c = jonswap_components(Hs=2.0, Tp=float(Tp), n_components=200,
                               depth=float(depth), gamma=float(gam),
                               direction_deg=0.0, direction_spread_deg=1e-6,
                               seed=1234)
        kmag = np.array([np.hypot(x.kx, x.ky) for x in c])
        e = np.array([x.A for x in c]) ** 2 / 2
        _RADC[key] = (kmag, e / e.sum())
    return _RADC[key]


def lobe_grid_d(kp, depth, Tp, mu_rad, spread=30.0, gam=3.3):
    kmag, en = radial_profile_d(Tp, gam, depth)
    th, wth = angle_mix([(mu_rad, spread, 1.0)])
    KX = np.outer(kmag, np.cos(th)).ravel()
    KY = np.outer(kmag, np.sin(th)).ravel()
    E = np.outer(en, wth).ravel()
    return deposit_fast(KX, KY, E, kp)


def supplied_grid_d(kp, depth, lobes):
    """lobes: list of (Tp, d_rad, weight). gamma=3.3/spread=30 frozen."""
    g = np.zeros((NGRID, NGRID))
    wtot = sum(w for _, _, w in lobes)
    for Tp, d, w in lobes:
        g += (w / wtot) * lobe_grid_d(kp, depth, Tp, d)
    return g


def m2_grid_d(kp, depth, mu_hat):
    """x_marginals 'm2' verbatim, arena Tp1=7 -> free Tp(kp_hat, depth)."""
    Tp1h = float(np.clip(tp_of_k_d(kp, depth), *TP_CLIP))
    L1 = lobe_grid_d(kp, depth, Tp1h, mu_hat)
    kmag2, en2 = [], []
    for r in TP2_RATIO_NODES:
        km, en = radial_profile_d(np.clip(Tp1h * r, *TP_CLIP), 3.3, depth)
        kmag2.append(km)
        en2.append(en / len(TP2_RATIO_NODES))
    kmag2, en2 = np.concatenate(kmag2), np.concatenate(en2)
    ang = [(mu_hat + s * np.deg2rad(o), 30.0, 1.0)
           for s in (+1, -1) for o in D2_OFF_NODES]
    th, wth = angle_mix(ang)
    KX = np.outer(kmag2, np.cos(th)).ravel()
    KY = np.outer(kmag2, np.sin(th)).ravel()
    E = np.outer(en2, wth).ravel()
    L2 = deposit_fast(KX, KY, E, kp)
    return (1 - W_MARG) * L1 + W_MARG * L2, L1


def perturb_lobes(lobes, rng):
    """Buoy-grade degradation on ALL supplied axes (stricter than e43:
    Tp1/d1 are perturbed too — there is no frozen arena axis here)."""
    out = []
    for Tp, d, w in lobes:
        out.append((float(np.clip(Tp * (1 + rng.normal(0, BUOY_ST / 100)),
                                  *TP_CLIP)),
                    d + np.deg2rad(rng.normal(0, BUOY_SD)), w))
    if len(out) == 2:
        r2 = float(np.clip(out[1][2] + rng.normal(0, BUOY_SW), 0.05, 0.6))
        out = [(out[0][0], out[0][1], 1 - r2), (out[1][0], out[1][1], r2)]
    return out


# ==========================================================================
#  dense-field regime spectrum (the "buoy" measurement) + frozen partition
# ==========================================================================
def win_hist(Z, gx, gy, dt, depth):
    """Dispersion-gated directional energy on the frozen (theta, ln k) hist.
    Z: (nt, ny, nx) dense cube of the 70 s OBS part, NaN allowed.
    Returns (72,48) energy hist.  TIME axis included: a traveling component
    cos(kx x + ky y - om t) lands at (kx, ky, -om) under the e^{-i...} FFT
    convention; by Hermitian symmetry of the real field the SAME component
    appears at time-frequency +om with wavevector (-kx, -ky).  We therefore
    rfft the time axis (om >= 0) and read the direction as (-kx, -ky) —
    this resolves the 180 deg ambiguity that killed the snapshot screen.
    Streaming per om-bin (memory: one padded 2D plane at a time)."""
    from scipy import fft as sfft
    nt, ny, nx = Z.shape
    dxs = float(np.median(np.diff(gx)))
    dys = float(np.median(np.diff(gy)))
    v = np.isfinite(Z).all(0)
    Zc = np.where(np.isfinite(Z), Z, 0.0).astype(np.float32)
    Zc = Zc - Zc.mean()
    Zc *= np.hanning(nt).astype(np.float32)[:, None, None]
    Zc *= (np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
           * v).astype(np.float32)[None]
    Zt = sfft.rfft(Zc, axis=0)                    # (nt//2+1, ny, nx) c64
    del Zc
    w_pos = 2 * np.pi * np.fft.rfftfreq(nt, d=dt)
    # padded 2x spatial grids (sub-bin interpolation), direction = (-kx,-ky)
    kx = 2 * np.pi * np.fft.fftfreq(2 * nx, d=dxs)
    ky = 2 * np.pi * np.fft.fftfreq(2 * ny, d=dys)
    KX, KY = kx[None, :], ky[:, None]
    K = np.hypot(KX + 0 * KY, KY + 0 * KX)
    omd = omega_d(K, depth)
    dw = 2 * np.pi / (nt * dt)
    gate_w = np.maximum(0.30 * omd, 3 * dw)
    th_all = np.arctan2(-(KY + 0 * KX), -(KX + 0 * KY))
    kmin = 0.999 * max(2 * np.pi / (nx * dxs), 2 * np.pi / (ny * dys))
    mkeep = (K > max(kmin, np.exp(U_LO))) & (K < np.exp(U_HI))
    i_th_all = np.clip(((th_all + np.pi) / (2 * np.pi) * NTH_H).astype(int),
                       0, NTH_H - 1)
    H = np.zeros((NTH_H, NU_H))
    cnt_u = np.zeros(NU_H)
    for it in range(1, len(w_pos)):
        om_t = w_pos[it]
        if om_t < 1.5 * dw:
            continue
        m = mkeep & (np.abs(om_t - omd) <= gate_w)
        if not m.any():
            continue
        # A1: radial coordinate from omega via the dispersion shell (the
        # buoy principle) — robust to ~1-lambda apertures where the
        # spatial k-axis cannot resolve the peak.
        u_t = np.log(max(k_disp(float(om_t), depth), 1e-12))
        if not (U_LO <= u_t <= U_HI):
            continue
        iu = int(np.clip((u_t - U_LO) / (U_HI - U_LO) * NU_H, 0, NU_H - 1))
        Pxy = np.abs(sfft.fft2(Zt[it], s=(2 * ny, 2 * nx))) ** 2
        # A5: off-shell noise floor (stereo noise integrates ~k^2 cells)
        moff = mkeep & (np.abs(om_t - omd) > 2.5 * gate_w)
        floor = float(np.median(Pxy[moff])) if moff.any() else 0.0
        e = np.maximum(Pxy[m] - floor, 0.0)
        np.add.at(H, (i_th_all[m], np.full(int(m.sum()), iu)), e)
        cnt_u[iu] += 1.0
    # A9: per-omega DENSITY (buoy/Welch convention). Without this, a u-bin
    # accumulates ~omega-proportionally many omega bins (du = 2 domega/omega
    # in deep water), tilting broad spectra to high frequency (BS13a census
    # Tp 1.8 s vs Welch 4.8 s -- the pre-registered cross-check flag).
    H = H / np.maximum(cnt_u, 1.0)[None, :]
    return H


def _smooth_hist(H):
    from scipy.ndimage import gaussian_filter1d
    binw_th = 360.0 / NTH_H
    binw_u = (U_HI - U_LO) / NU_H
    Hs_ = gaussian_filter1d(H, SM_TH_DEG / binw_th, axis=0, mode='wrap')
    return gaussian_filter1d(Hs_, SM_U / binw_u, axis=1, mode='nearest')


def _local_max(S, i, j):
    """Genuine 2D local max, theta axis wrapped. A monotone JONSWAP
    shoulder or a mask-boundary point can never satisfy this — the
    structural abstention that the e37 detector / the coordinator's
    snapshot screen lacked."""
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            jj = j + dj
            if jj < 0 or jj >= NU_H:
                continue
            if S[(i + di) % NTH_H, jj] > S[i, j]:
                return False
    return True


def partition_hist(H, depth):
    """Frozen 2-lobe partition. Returns (lobes[(Tp,d,w)...], accepted2)."""
    S = _smooth_hist(H)
    th_c = -np.pi + (np.arange(NTH_H) + 0.5) * (2 * np.pi / NTH_H)
    u_c = U_LO + (np.arange(NU_H) + 0.5) * ((U_HI - U_LO) / NU_H)
    TH = th_c[:, None] + 0 * u_c[None, :]
    UU = 0 * th_c[:, None] + u_c[None, :]
    i1 = np.unravel_index(np.argmax(S), S.shape)
    t1, u1 = th_c[i1[0]], u_c[i1[1]]
    dth1 = np.abs(np.angle(np.exp(1j * (TH - t1))))
    # A6: exclusion = dTheta < 55 deg at ALL k. The screen looks ONLY for
    # direction-separated cross-Tp (the arena regime, d-sep in [60,150]).
    excl = dth1 < np.deg2rad(EXCL_TH_DEG)
    S2 = np.where(excl, -np.inf, S)
    # peak 2: highest NON-excluded bin that is a genuine 2D local max and
    # clears the relative-height bar; otherwise ABSTAIN (unimodal).
    i2 = None
    for f in np.argsort(S2.ravel())[::-1][:400]:
        ii, jj = np.unravel_index(int(f), S.shape)
        if not np.isfinite(S2[ii, jj]) or S[ii, jj] < ACCEPT2 * S[i1]:
            break
        if _local_max(S, ii, jj):
            i2 = (ii, jj)
            break

    def centroid(t0, u0, wsoft):
        m = wsoft * H
        tot = m.sum()
        if tot <= 0:
            return t0, u0, 0.0
        d = float(np.angle((m * np.exp(1j * TH)).sum()))
        u = float((m * UU).sum() / tot)
        return d, u, float(tot)

    def soft(t0, u0):
        dth = np.angle(np.exp(1j * (TH - t0))) / np.deg2rad(SOFT_TH_DEG)
        du = (UU - u0) / SOFT_U
        return np.exp(-0.5 * (dth ** 2 + du ** 2))

    accepted = i2 is not None
    if accepted:
        t2, u2 = th_c[i2[0]], u_c[i2[1]]
        w1r, w2r = soft(t1, u1), soft(t2, u2)
        tot = w1r + w2r + 1e-30
        w1n, w2n = w1r / tot, w2r / tot
        d1, uu1, m1 = centroid(t1, u1, w1n)
        d2, uu2, m2_ = centroid(t2, u2, w2n)
        # A6: soft-mass abstention
        if m2_ / max(m1 + m2_, 1e-30) < MIN_MASS2:
            accepted = False
    if not accepted:
        w1 = soft(t1, u1)
        d, u, _ = centroid(t1, u1, w1)
        Tp = float(np.clip(tp_of_k_d(np.exp(u), depth), *TP_CLIP))
        return [(Tp, d, 1.0)], False
    r2 = float(np.clip(m2_ / max(m1 + m2_, 1e-30), 0.05, 0.6))
    Tp1 = float(np.clip(tp_of_k_d(np.exp(uu1), depth), *TP_CLIP))
    Tp2 = float(np.clip(tp_of_k_d(np.exp(uu2), depth), *TP_CLIP))
    return [(Tp1, d1, 1 - r2), (Tp2, d2, r2)], True


# ==========================================================================
#  window construction + scoring (shared by real records and the self-test)
# ==========================================================================
def track_state(gx, gy, r_fp, U, valid_frac_fn):
    """Inset diagonal, better-valid of the two (score-blind)."""
    x0, x1 = gx[0] + r_fp, gx[-1] - r_fp
    y0, y1 = gy[0] + r_fp, gy[-1] - r_fp
    cands = [(np.array([x0, y0]), np.array([x1, y1])),
             (np.array([x0, y1]), np.array([x1, y0]))]
    fr = [valid_frac_fn(P0, P1) for P0, P1 in cands]
    P0, P1 = cands[int(np.argmax(fr))]
    return P0, P1


def track_pos(t_rel, P0, P1, U):
    L = float(np.linalg.norm(P1 - P0))
    u = (P1 - P0) / L
    s = np.mod(U * np.asarray(t_rel), 2 * L)
    d = np.where(s <= L, s, 2 * L - s)
    return P0[None] + d[:, None] * u[None]


def corridor_mask(Xg, Yg, P0, P1, dmin):
    seg = P1 - P0
    L2 = float(seg @ seg)
    tpar = np.clip(((Xg - P0[0]) * seg[0] + (Yg - P0[1]) * seg[1]) / L2, 0, 1)
    dB = np.hypot(Xg - (P0[0] + tpar * seg[0]), Yg - (P0[1] + tpar * seg[1]))
    return dB > dmin


def build_window(rec, i0, depth, fp_ref, cp_ref, rng):
    """Everything score-blind: track, sparse z, queries, dense hist."""
    dt = rec.dt
    nf80 = int(round((T_END + HORIZON) / dt)) + 2
    if i0 + nf80 > rec.T_all:
        return None
    Zwin = rec.window(i0, nf80)
    gx, gy = rec.gx, rec.gy
    r_fp = fp_ref / 2
    U = float(min(rng.uniform(0.3, 1.5) * cp_ref, U_CAP))

    v2 = np.isfinite(Zwin[0])

    def vf(P0, P1):
        ts = np.linspace(0, 1, 60)
        xs = P0[0] + ts * (P1[0] - P0[0])
        ys = P0[1] + ts * (P1[1] - P0[1])
        ix = np.clip(np.searchsorted(gx, xs), 0, len(gx) - 1)
        iy = np.clip(np.searchsorted(gy, ys), 0, len(gy) - 1)
        return float(v2[iy, ix].mean())

    # A3: diagonal inside the marginal-valid bounding box (YS17-type
    # offset valid regions), fallback = full grid
    cm, rm = v2.mean(0), v2.mean(1)
    xs_v = np.where(cm >= 0.4)[0]
    ys_v = np.where(rm >= 0.4)[0]
    if (len(xs_v) >= 8 and len(ys_v) >= 8
            and gx[xs_v[-1]] - gx[xs_v[0]] > 2 * r_fp + 10
            and gy[ys_v[-1]] - gy[ys_v[0]] > 2 * r_fp + 10):
        bgx = np.array([gx[xs_v[0]], gx[xs_v[-1]]])
        bgy = np.array([gy[ys_v[0]], gy[ys_v[-1]]])
    else:
        bgx = np.array([gx[0], gx[-1]])
        bgy = np.array([gy[0], gy[-1]])
    P0, P1 = track_state(bgx, bgy, r_fp, U, vf)

    # ---- sparse obs (E-9 protocol): 400 draws, jitter, nearest frame
    tt = np.linspace(0, T_END, N_OBS)
    it = np.clip(np.round(tt / dt).astype(int), 0, nf80 - 1)
    t_frame = it * dt
    xy = track_pos(t_frame, P0, P1, U)
    thj = rng.uniform(0, 2 * np.pi, N_OBS)
    rrj = r_fp * np.sqrt(rng.uniform(0, 1, N_OBS))
    xs = xy[:, 0] + rrj * np.cos(thj)
    ys = xy[:, 1] + rrj * np.sin(thj)
    z = np.array([bilinear(Zwin[i], gx, gy, x, y)
                  for i, x, y in zip(it, xs, ys)])
    ok = np.isfinite(z)
    if ok.sum() < MIN_OBS:
        del Zwin
        return None
    P = np.column_stack([xs[ok], ys[ok], t_frame[ok]])
    z = z[ok]
    Hs_hat = 4.0 * float(np.std(z))
    sigma0 = NFRAC * Hs_hat
    z = z + rng.normal(0, sigma0, len(z))

    # ---- encounter query: extrapolated track, no jitter
    te = T_END + np.linspace(HORIZON / N_ENC, HORIZON, N_ENC)
    ite = np.clip(np.round(te / dt).astype(int), 0, nf80 - 1)
    te_f = ite * dt
    xye = track_pos(te_f, P0, P1, U)
    gt_e = np.array([bilinear(Zwin[i], gx, gy, x, y)
                     for i, (x, y) in zip(ite, xye)])
    oke = np.isfinite(gt_e)
    if oke.sum() < MIN_ENC:
        del Zwin
        return None
    Pq_e = np.column_stack([xye[oke, 0], xye[oke, 1], te_f[oke]])
    gt_e = gt_e[oke]
    h = int(oke[:N_ENC // 2].sum())              # first-5s count

    # ---- field query at last obs frame, outside corridor
    itF = int(round(T_END / dt))
    ZF = Zwin[itF]
    Xg, Yg = np.meshgrid(gx, gy)
    mq = np.isfinite(ZF) & corridor_mask(Xg, Yg, P0, P1,
                                         r_fp + 0.6 * fp_ref)
    idx = np.flatnonzero(mq)
    if len(idx) >= 200:
        pick = np.random.default_rng(rng.integers(1 << 31)).choice(
            idx, min(N_FLD, len(idx)), replace=False)
        iyq, ixq = np.unravel_index(pick, ZF.shape)
        Pq_f = np.column_stack([gx[ixq], gy[iyq],
                                np.full(len(pick), itF * dt)])
        gt_f = ZF[iyq, ixq]
    else:
        Pq_f, gt_f = None, None

    # ---- dense regime hist (obs 70 s only; decimated to <=0.35 s)
    tdec = max(1, int(round(0.35 / dt)))
    nf70 = int(round(T_END / dt))
    Zdec = Zwin[:nf70:tdec]
    H = win_hist(Zdec, gx, gy, dt * tdec, depth)
    tp_t, _ = peak_period(Zdec, np.arange(Zdec.shape[0]) * dt * tdec)
    del Zwin
    return dict(P=P, z=z, sigma0=sigma0, Hs=Hs_hat, U=U,
                Pq_e=Pq_e, gt_e=gt_e, h5=h, Pq_f=Pq_f, gt_f=gt_f,
                hist=H, n_obs=int(ok.sum()), n_enc=int(oke.sum()),
                tp_time=float(tp_t))


def score_window(w, depth, supply_lobes, rng_buoy, kp_ref):
    kp = two_stage_kp_d(w['P'], w['z'], depth, kp_ref)
    mu = mu_hat_of_d(w['P'], w['z'], kp, depth)
    grids = {}
    grids['cur'] = cur_gamma(w['P'], w['z'], kp, depth)
    grids['m2'], grids['ens1'] = m2_grid_d(kp, depth, mu)
    grids['orc'] = supplied_grid_d(kp, depth, supply_lobes)
    row = dict(kp=float(kp), mu_deg=float(np.rad2deg(mu)))
    queries = [w['Pq_e']] + ([w['Pq_f']] if w['Pq_f'] is not None else [])
    for nm in ('cur', 'm2', 'ens1', 'orc'):
        fs = solve_fields(w['P'], w['z'], kp, w['sigma0'], depth,
                          grids[nm], queries)
        fe = fs[0]
        row[f'enc10_{nm}'] = corr(fe, w['gt_e'])
        row[f'enc05_{nm}'] = corr(fe[:w['h5']], w['gt_e'][:w['h5']]) \
            if (fe is not None and w['h5'] >= 10) else float('nan')
        row[f'fld_{nm}'] = corr(fs[1], w['gt_f']) if len(fs) > 1 \
            else float('nan')
    # buoy: R=5 degraded draws, scores averaged BEFORE pairing.
    # A supply ring outside the bank band gives a zero grid -> None fields
    # -> NaN (honest missing arm), handled by corr's guards.
    acc = {k: [] for k in ('enc10', 'enc05', 'fld')}
    for r in range(R_BUOY):
        lb = perturb_lobes(supply_lobes, rng_buoy)
        fs = solve_fields(w['P'], w['z'], kp, w['sigma0'], depth,
                          supplied_grid_d(kp, depth, lb), queries)
        fe = fs[0]
        acc['enc10'].append(corr(fe, w['gt_e']))
        acc['enc05'].append(corr(fe[:w['h5']], w['gt_e'][:w['h5']])
                            if (fe is not None and w['h5'] >= 10)
                            else float('nan'))
        acc['fld'].append(corr(fs[1], w['gt_f']) if len(fs) > 1
                          else float('nan'))
    for k, v in acc.items():
        row[f'{k}_buoy'] = float(np.nanmean(v)) if np.isfinite(v).any() \
            else float('nan')
    return row


# ==========================================================================
#  statistics: t-CIs, naive AND sea-event clustered
# ==========================================================================
def t_ci(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return dict(mean=float(np.mean(d)) if n else float('nan'),
                    se=float('nan'), lo=float('nan'), hi=float('nan'),
                    n=n, win=int((d > 0).sum()) if n else 0)
    se = d.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se)
    return dict(mean=float(d.mean()), se=float(se), lo=float(lo),
                hi=float(hi), n=n, win=int((d > 0).sum()))


def paired_cl(rows, a, b):
    d, cl = [], []
    for r in rows:
        if np.isfinite(r.get(a, np.nan)) and np.isfinite(r.get(b, np.nan)):
            d.append(r[a] - r[b])
            cl.append(r['cluster'])
    d, cl = np.array(d), np.array(cl)
    naive = t_ci(d)
    cmeans = [d[cl == c].mean() for c in sorted(set(cl))]
    clust = t_ci(cmeans)
    clust['clusters'] = sorted(set(cl))
    return dict(naive=naive, clustered=clust)


# ==========================================================================
#  synthetic self-test (harness gate)
# ==========================================================================
class FakeRec:
    def __init__(self, cube, gx, gy, dt):
        self.cube, self.gx, self.gy, self.dt = cube, gx, gy, dt
        self.T_all = cube.shape[0]

    def window(self, i0, n):
        return self.cube[i0:i0 + n].copy()


def render_cube(S, gx, gy, ts, depth):
    from d3_sa_fim_identifiability import omega_of
    kx = torch.tensor(S['m_kx'], dtype=torch.float32, device=DEVICE)
    ky = torch.tensor(S['m_ky'], dtype=torch.float32, device=DEVICE)
    A = torch.tensor(S['m_A'], dtype=torch.float32, device=DEVICE)
    ph = torch.tensor(S['m_ph'], dtype=torch.float32, device=DEVICE)
    om = torch.tensor(omega_of(S['m_kx'], S['m_ky'], depth),
                      dtype=torch.float32, device=DEVICE)
    Xg, Yg = np.meshgrid(gx, gy)
    X = torch.tensor(Xg.ravel(), dtype=torch.float32, device=DEVICE)
    Y = torch.tensor(Yg.ravel(), dtype=torch.float32, device=DEVICE)
    out = np.empty((len(ts), len(gy), len(gx)), np.float32)
    for i, t in enumerate(ts):
        psi = kx[None] * X[:, None] + ky[None] * Y[:, None] \
            - om[None] * float(t) + ph[None]
        out[i] = ((A[None] * torch.cos(psi)).sum(1)
                  ).cpu().numpy().reshape(len(gy), len(gx))
    return out


def ang_err_deg(a, b):
    return float(np.rad2deg(np.abs(np.angle(np.exp(1j * (a - b))))))


def selftest(n=6, n_uni=6, enforce=True, verbose=True):
    from e37_bimodal_ladder import build_sea_crosstp, build_sea_bimodal
    from d3_sa_achievability_gap import dispersion_lam
    depth = 50.0
    # micro-test: single plane wave, known direction
    lam0 = 60.0
    k0 = 2 * np.pi / lam0
    om0 = omega_d(k0, depth)
    gxm = np.arange(0, 120.0, 2.0)
    gym = np.arange(0, 120.0, 2.0)
    tsm = np.arange(0, 70.0, 0.35)
    Xg, Yg = np.meshgrid(gxm, gym)
    d_true = np.deg2rad(30.0)
    cube = np.stack([np.cos(k0 * np.cos(d_true) * Xg
                            + k0 * np.sin(d_true) * Yg - om0 * t)
                     for t in tsm]).astype(np.float32)
    Hm = win_hist(cube, gxm, gym, 0.35, depth)
    lb, _ = partition_hist(Hm, depth)
    e_dir = ang_err_deg(lb[0][1], d_true)
    e_tp = abs(lb[0][0] - 2 * np.pi / om0) / (2 * np.pi / om0)
    micro_ok = (e_dir < 15.0) and (e_tp < 0.10)
    if verbose:
        print(f'  [micro] plane wave: dir err {e_dir:.1f} deg, '
              f'Tp err {e_tp:.1%} -> {"OK" if micro_ok else "FAIL"}',
              flush=True)

    rows = []
    for si in range(5000, 5000 + n):
        S = build_sea_crosstp(si)
        Tp1 = 7.0
        lam = dispersion_lam(Tp1, depth)
        side = 2.0 * lam
        gx = np.linspace(0, side, 96)
        gy = np.linspace(0, side, 96)
        dtc = 0.1
        ts = np.arange(0, T_END + HORIZON + 2 * dtc, dtc)
        cube = render_cube(S, gx, gy, ts, depth)
        rec = FakeRec(cube, gx, gy, dtc)
        fp = 0.18 * lam
        cp = lam / Tp1
        rng = np.random.default_rng(46600 + si)
        w = build_window(rec, 0, depth, fp, cp, rng)
        if w is None:
            rows.append(dict(si=si, fail='window'))
            continue
        # A7: supply = LOWO mirror — 3 later quasi-independent segments
        # (hist-only, dt 0.35), exactly as the real leg averages the other
        # windows of the record
        hists = []
        for j in range(1, 4):
            ts_j = np.arange(0, T_END, 0.35) + j * (T_END + HORIZON + 2.0)
            cube_j = render_cube(S, gx, gy, ts_j, depth)
            hists.append(win_hist(cube_j, gx, gy, 0.35, depth))
        H_lowo = np.mean(hists, 0)
        lobes, acc = partition_hist(H_lowo, depth)
        # match detected lobes to truth
        d1t, d2t = S['d1'], S['d2']
        errs = dict(si=si, det2=bool(acc), r2_true=S['r2'],
                    Tp2_true=S['Tp2'])
        if acc:
            e_a = ang_err_deg(lobes[0][1], d1t) + ang_err_deg(lobes[1][1], d2t)
            e_b = ang_err_deg(lobes[0][1], d2t) + ang_err_deg(lobes[1][1], d1t)
            if e_a <= e_b:
                m = [(lobes[0], Tp1, d1t), (lobes[1], S['Tp2'], d2t)]
            else:
                m = [(lobes[0], S['Tp2'], d2t), (lobes[1], Tp1, d1t)]
            errs['derr'] = float(np.mean([ang_err_deg(l[1], dt_)
                                          for l, _, dt_ in m]))
            errs['tperr'] = float(np.mean([abs(l[0] - tp_) / tp_
                                           for l, tp_, _ in m]))
            errs['r2_hat'] = float(lobes[1][2])
        else:
            errs['derr'] = min(ang_err_deg(lobes[0][1], d1t),
                               ang_err_deg(lobes[0][1], d2t))
            errs['tperr'] = abs(lobes[0][0] - Tp1) / Tp1
        rb = np.random.default_rng(46700 + si)
        r = score_window(w, depth, lobes, rb, 2 * np.pi / lam)
        errs.update(r)
        # A8: TRUE-theta oracle through the same pipeline (harness anchor)
        true_lobes = [(7.0, S['d1'], 1 - S['r2']),
                      (S['Tp2'], S['d2'], S['r2'])]
        kp_w = two_stage_kp_d(w['P'], w['z'], depth, 2 * np.pi / lam)
        gT = supplied_grid_d(kp_w, depth, true_lobes)
        qs = [w['Pq_e']] + ([w['Pq_f']] if w['Pq_f'] is not None else [])
        fsT = solve_fields(w['P'], w['z'], kp_w, w['sigma0'], depth, gT, qs)
        errs['enc10_orcT'] = corr(fsT[0], w['gt_e'])
        errs['fld_orcT'] = corr(fsT[1], w['gt_f']) if len(fsT) > 1 \
            else float('nan')
        rows.append(errs)
        if verbose:
            print(f"  [self si{si}] det2 {int(acc)} derr "
                  f"{errs['derr']:5.1f} tperr {errs['tperr']:.2%} | "
                  f"fld orcT-cur {errs['fld_orcT'] - r['fld_cur']:+.3f} "
                  f"orc-cur {r['fld_orc'] - r['fld_cur']:+.3f} "
                  f"enc10 orc-cur {r['enc10_orc'] - r['enc10_cur']:+.3f}",
                  flush=True)

    # A8: anchor extension seas (orcT + cur only, field task)
    anc = [rr['fld_orcT'] - rr['fld_cur'] for rr in rows
           if np.isfinite(rr.get('fld_orcT', np.nan))
           and np.isfinite(rr.get('fld_cur', np.nan))]
    n_anchor = max(0, (32 if enforce else 4) - n)
    for si in range(5000 + n, 5000 + n + n_anchor):
        S = build_sea_crosstp(si)
        lam = dispersion_lam(7.0, depth)
        gx = np.linspace(0, 2 * lam, 96)
        gy = np.linspace(0, 2 * lam, 96)
        dtc = 0.1
        ts = np.arange(0, T_END + HORIZON + 2 * dtc, dtc)
        cube = render_cube(S, gx, gy, ts, depth)
        rec = FakeRec(cube, gx, gy, dtc)
        rng = np.random.default_rng(46600 + si)
        w = build_window(rec, 0, depth, 0.18 * lam, lam / 7.0, rng)
        if w is None or w['Pq_f'] is None:
            continue
        kp_w = two_stage_kp_d(w['P'], w['z'], depth, 2 * np.pi / lam)
        g_cur = cur_gamma(w['P'], w['z'], kp_w, depth)
        gT = supplied_grid_d(kp_w, depth,
                             [(7.0, S['d1'], 1 - S['r2']),
                              (S['Tp2'], S['d2'], S['r2'])])
        c1 = corr(solve_fields(w['P'], w['z'], kp_w, w['sigma0'], depth,
                               g_cur, [w['Pq_f']])[0], w['gt_f'])
        c2 = corr(solve_fields(w['P'], w['z'], kp_w, w['sigma0'], depth,
                               gT, [w['Pq_f']])[0], w['gt_f'])
        if np.isfinite(c1) and np.isfinite(c2):
            anc.append(c2 - c1)
    anc = np.array(anc, float)
    anchor32 = float(anc.mean()) if len(anc) else float('nan')
    anc_se = float(anc.std(ddof=1) / np.sqrt(len(anc))) if len(anc) > 2 \
        else float('nan')
    if verbose:
        print(f'  [A8 anchor] fld orcT-cur n={len(anc)}: '
              f'{anchor32:+.4f} +- {anc_se:.4f}', flush=True)
    # ---- A4/A8b diagnostic: aperture-grade curve (boxes 1.0 / 3.8 lam;
    # 2.0 comes from the screen block). Interprets per-record supply grade
    # on real data (BS13/LJ ~ 1.0, AA ~ 1.5-2, YS ~ 3.8).
    aperture = {}
    for box in (1.0, 3.8):
        des, tes, dets = [], [], []
        for si in range(5000, 5004 if enforce else 5002):
            S = build_sea_crosstp(si)
            lam = dispersion_lam(7.0, depth)
            npx = 64 if box <= 2.0 else 96
            gxb = np.linspace(0, box * lam, npx)
            gyb = np.linspace(0, box * lam, npx)
            hs = []
            for seg in range(4):
                ts_b = np.arange(0, T_END, 0.35) \
                    + seg * (T_END + HORIZON + 2.0)
                cb = render_cube(S, gxb, gyb, ts_b, depth)
                hs.append(win_hist(cb, gxb, gyb, 0.35, depth))
            lb_b, acc_b = partition_hist(np.mean(hs, 0), depth)
            dets.append(bool(acc_b))
            if acc_b:
                e_a = ang_err_deg(lb_b[0][1], S['d1']) \
                    + ang_err_deg(lb_b[1][1], S['d2'])
                e_b = ang_err_deg(lb_b[0][1], S['d2']) \
                    + ang_err_deg(lb_b[1][1], S['d1'])
                if e_a <= e_b:
                    mm = [(lb_b[0], 7.0, S['d1']), (lb_b[1], S['Tp2'],
                                                    S['d2'])]
                else:
                    mm = [(lb_b[0], S['Tp2'], S['d2']), (lb_b[1], 7.0,
                                                         S['d1'])]
                des.append(np.mean([ang_err_deg(l[1], dd) for l, _, dd
                                    in mm]))
                tes.append(np.mean([abs(l[0] - tp_) / tp_ for l, tp_, _
                                    in mm]))
        aperture[box] = dict(det=float(np.mean(dets)) if dets else 0.0,
                             derr=float(np.mean(des)) if des else None,
                             tperr=float(np.mean(tes)) if tes else None)
        if verbose:
            a = aperture[box]
            print(f"  [aperture {box:.1f} lam] det2 {a['det']:.0%} "
                  f"derr {a['derr'] if a['derr'] is None else round(a['derr'], 1)} "
                  f"tperr {a['tperr'] if a['tperr'] is None else round(100 * a['tperr'], 1)}%"
                  f" [diagnostic only]", flush=True)

    # ---- UNIMODAL CONTROL (the screen's false-positive rate; the
    # coordinator's snapshot screen died 6/6 here — this gate is why the
    # census can be called evidence at all)
    fp = []
    for j in range(n_uni):
        S = build_sea_bimodal(5100 + j, unimodal=True)
        lam = dispersion_lam(7.0, depth)
        side = 2.0 * lam
        gx = np.linspace(0, side, 96)
        gy = np.linspace(0, side, 96)
        # A7: census-path mirror — average of 4 segment hists
        hs = []
        for seg in range(4):
            ts_j = np.arange(0, T_END, 0.35) + seg * (T_END + HORIZON + 2.0)
            cube_j = render_cube(S, gx, gy, ts_j, depth)
            hs.append(win_hist(cube_j, gx, gy, 0.35, depth))
        lb_u, acc = partition_hist(np.mean(hs, 0), depth)
        fp.append(bool(acc))
        if verbose:
            xtra = ''
            if acc:
                xtra = '  [' + ' + '.join(
                    f'Tp{t:.1f}@{np.rad2deg(d):.0f}*{w_:.2f}'
                    for t, d, w_ in lb_u) + ']'
            print(f'  [uni {j}] 2-lobe fired: {int(acc)}{xtra}', flush=True)
    fpr = float(np.mean(fp)) if fp else 1.0

    ok_rows = [r for r in rows if 'fld_orc' in r]
    det = np.mean([r['det2'] for r in ok_rows]) if ok_rows else 0.0
    derr = np.mean([r['derr'] for r in ok_rows]) if ok_rows else 999
    tperr = np.mean([r['tperr'] for r in ok_rows]) if ok_rows else 9
    danc = np.mean([r['fld_orc'] - r['fld_cur'] for r in ok_rows]) \
        if ok_rows else -9
    grade_pen = np.nanmean([r['fld_orc'] - r['fld_orcT'] for r in ok_rows]) \
        if ok_rows else float('nan')
    passed = (micro_ok and len(ok_rows) >= max(3, n - 1)
              and det >= 4.0 / 6.0 - 1e-9 and derr <= 20.0
              and tperr <= 0.12 and fpr <= 1.0 / 6.0 + 1e-9
              and 0.0 < anchor32 < 0.20)
    if not enforce:
        passed = True          # smoke: report only, never gate
    if verbose:
        print(f'  [selftest] TPR(det2) {det:.0%} derr {derr:.1f}deg '
              f'tperr {tperr:.1%} | unimodal FPR {fpr:.0%} '
              f'({int(np.sum(fp))}/{len(fp)}) | ★A8 anchor(orcT-cur, '
              f'n={len(anc)}) {anchor32:+.4f} | est-theta supply orc-cur '
              f'{danc:+.4f} (grade penalty orc-orcT {grade_pen:+.4f}) '
              f'-> {"PASS" if passed else "FAIL"}'
              f'{" [smoke: gate not enforced]" if not enforce else ""}',
              flush=True)
    return passed, dict(micro_dir_err=e_dir, micro_tp_err=e_tp,
                        det=float(det), derr=float(derr),
                        tperr=float(tperr), fpr=float(fpr),
                        anchor32=float(anchor32), anchor32_se=anc_se,
                        anchor_est=float(danc),
                        grade_penalty=float(grade_pen),
                        aperture=aperture,
                        rows=rows, passed=bool(passed))


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--selftest-only', action='store_true')
    ap.add_argument('--census-only', action='store_true')
    ap.add_argument('--records', nargs='+', default=None)
    ap.add_argument('--max_win', type=int, default=MAX_WIN)
    a = ap.parse_args()
    t0 = time.time()

    print('=== HARNESS GATE + SCREEN VALIDATION (synthetic, BEFORE any real '
          'data): cross-Tp TPR + unimodal FPR ===', flush=True)
    if a.smoke:
        ok, st = selftest(2, 2, enforce=False)
    else:
        ok, st = selftest(6, 6, enforce=True)
    if not ok:
        print('HARNESS GATE FAILED — real run aborted (fix the pipeline, '
              'not the bars).')
        json.dump(dict(selftest=st, aborted=True), open(OUT, 'w'),
                  indent=1, default=float)
        return
    if a.selftest_only:
        json.dump(dict(selftest=st), open(OUT, 'w'), indent=1, default=float)
        return

    recs = RECORDS46
    if a.records:
        recs = [r for r in RECORDS46 if r[0] in a.records]
    if a.smoke:
        recs = [r for r in RECORDS46 if r[0] in ('BS13a', 'YS17')]
        a.max_win = 2

    rows = []
    rec_diag = {}
    for ri, (name, path, depth, fp_ref, tp_ref, cluster) in enumerate(recs):
        lam_ref = fp_ref / 0.18
        cp_ref = lam_ref / tp_ref
        rec = MRecord(path, stride=1)
        if rec.n0 * rec.n1 > 60000:
            del rec
            rec = MRecord(path, stride=2)
        dur = rec.t[-1] - rec.t[0]
        nwin = min(a.max_win, int(dur // WIN_SPACING))
        i0s = [int(round((5.0 + j * WIN_SPACING) / rec.dt))
               for j in range(nwin)]
        print(f'--- {name} (depth {depth} m, lam_ref {lam_ref:.0f} m, '
              f'Tp_ref {tp_ref} s, {nwin} windows, cluster {cluster}) ---',
              flush=True)
        wins = []
        for wi, i0 in enumerate(i0s):
            rng = np.random.default_rng(46000 + 1009 * ri + 17 * wi)
            w = build_window(rec, i0, depth, fp_ref, cp_ref, rng)
            if w is None:
                print(f'  win{wi}: DROPPED (obs/enc validity)', flush=True)
                continue
            w['wi'] = wi
            wins.append(w)
        if not wins:
            print(f'  {name}: no usable windows', flush=True)
            del rec
            continue
        # ---- REGIME CENSUS (precondition deliverable): record-average
        Hrec = np.mean([w['hist'] for w in wins], 0)
        lob_c, acc_c = partition_hist(Hrec, depth)
        tp_med = float(np.nanmedian([w['tp_time'] for w in wins]))
        tp_dom = float(lob_c[0][0])
        ratio = tp_dom / tp_med if tp_med > 0 else float('nan')
        tp_flag = not (0.77 <= ratio <= 1.30)
        cens = dict(bimodal=bool(acc_c),
                    lobes=[(float(t), float(np.rad2deg(d)), float(wt))
                           for t, d, wt in lob_c],
                    tp_spec=tp_dom, tp_time_med=tp_med,
                    tp_ratio=float(ratio), tp_flag=bool(tp_flag),
                    n_win=len(wins))
        stxt = ' + '.join(f'Tp{t:.1f}@{d:.0f}deg*{w_:.2f}'
                          for t, d, w_ in cens['lobes'])
        print(f"  ★CENSUS {name}: {'CROSS-Tp BIMODAL' if acc_c else 'unimodal'}"
              f"  [{stxt}]  Tp_spec {tp_dom:.1f}s vs Tp_time {tp_med:.1f}s "
              f"(ratio {ratio:.2f}{' ⚠FLAG' if tp_flag else ''})", flush=True)
        rec_diag[name] = dict(census=cens)
        if a.census_only:
            del rec
            continue
        # LOWO supply per window
        sup_diag = []
        for w in wins:
            others = [x['hist'] for x in wins if x['wi'] != w['wi']]
            Hsup = np.mean(others, 0) if others else w['hist']
            lobes, acc = partition_hist(Hsup, depth)
            w['supply'] = lobes
            w['acc2'] = acc
            w['lowo'] = bool(others)
        for w in wins:
            rng_b = np.random.default_rng(46000 + 1009 * ri
                                          + 17 * w['wi'] + 7)
            r = score_window(w, depth, w['supply'], rng_b,
                             2 * np.pi / lam_ref)
            r.update(record=name, wi=w['wi'], cluster=cluster,
                     depth=depth, U=w['U'], Hs=w['Hs'],
                     n_obs=w['n_obs'], n_enc=w['n_enc'],
                     acc2=bool(w['acc2']), lowo=w['lowo'],
                     supply=[(float(Tp), float(np.rad2deg(d)), float(wt))
                             for Tp, d, wt in w['supply']])
            rows.append(r)
            s = w['supply']
            stxt = ' + '.join(f'Tp{Tp:.1f}@{np.rad2deg(d):.0f}deg*{wt:.2f}'
                              for Tp, d, wt in s)
            print(f"  win{w['wi']:2d} U{w['U']:4.1f} Hs{w['Hs']:5.2f} "
                  f"n{w['n_obs']} [{stxt}] | "
                  f"enc10 cur {r['enc10_cur']:+.3f} m2 {r['enc10_m2']:+.3f} "
                  f"orc {r['enc10_orc']:+.3f} buoy {r['enc10_buoy']:+.3f} | "
                  f"fld cur {r['fld_cur']:+.3f} orc {r['fld_orc']:+.3f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        rec_diag[name]['bimodal_rate_lowo'] = float(
            np.mean([w['acc2'] for w in wins]))
        del rec

    # ================= PRECONDITION: census table =================
    print(f'\n=== ★REGIME CENSUS (screen: TPR {st["det"]:.0%}, unimodal FPR '
          f'{st["fpr"]:.0%} — validated BEFORE real data) ===')
    n_bi = 0
    for k, v in rec_diag.items():
        c = v['census']
        s = ' + '.join(f"Tp{t:.1f}@{d:.0f}*{w:.2f}" for t, d, w in c['lobes'])
        n_bi += int(c['bimodal'])
        print(f"  {k:6s}: {'CROSS-Tp' if c['bimodal'] else 'unimodal'}"
              f"  [{s}]  Tp_time {c['tp_time_med']:.1f}s "
              f"ratio {c['tp_ratio']:.2f}{' ⚠' if c['tp_flag'] else ''}")
    regime_present = n_bi > 0
    print(f"  -> target regime present in {n_bi}/{len(rec_diag)} records")
    if a.census_only or not rows:
        json.dump(dict(selftest=st, rec_diag=rec_diag,
                       regime_present=bool(regime_present),
                       census_only=True),
                  open(OUT, 'w'), indent=1, default=float)
        print(f'saved -> {OUT}  ({time.time() - t0:.0f}s)')
        return

    # ================= summary =================
    print(f'\n=== e46 REAL LEG SUMMARY (n={len(rows)} windows, '
          f'{len(set(r["cluster"] for r in rows))} sea-event clusters) ===')
    res = {}
    for tag in ('fld', 'enc10', 'enc05'):
        print(f'  -- {tag} --')
        for aa, bb in [('orc', 'cur'), ('buoy', 'cur'), ('orc', 'm2'),
                       ('buoy', 'm2'), ('m2', 'cur'), ('ens1', 'cur'),
                       ('orc', 'ens1'), ('buoy', 'ens1')]:
            r = paired_cl(rows, f'{tag}_{aa}', f'{tag}_{bb}')
            res[f'{tag}_{aa}-{bb}'] = r
            nv, cv = r['naive'], r['clustered']
            star = '*' if (cv['lo'] > 0 or cv['hi'] < 0) else ' '
            print(f"    {aa:5s}-{bb:5s}: naive {nv['mean']:+.4f}"
                  f"+-{nv['se']:.4f} CI[{nv['lo']:+.4f},{nv['hi']:+.4f}] "
                  f"win {nv['win']}/{nv['n']} | clustered {cv['mean']:+.4f} "
                  f"CI[{cv['lo']:+.4f},{cv['hi']:+.4f}]{star} "
                  f"(n_cl {cv['n']})")

    # arm means
    print('  -- arm means (enc10 / fld) --')
    for nm in ('cur', 'm2', 'ens1', 'orc', 'buoy'):
        e = np.nanmean([r[f'enc10_{nm}'] for r in rows])
        f = np.nanmean([r[f'fld_{nm}'] for r in rows])
        print(f'    {nm:5s}: enc10 {e:+.4f}  fld {f:+.4f}')

    # LOWO bimodality (supply-level, per window)
    print('  -- per-record LOWO supply bimodal rate --')
    for k, v in rec_diag.items():
        print(f"    {k:6s}: census "
              f"{'CROSS-Tp' if v['census']['bimodal'] else 'unimodal'}, "
              f"LOWO windows fired {v.get('bimodal_rate_lowo', 0):.0%}")

    # verdict
    A = res['fld_orc-cur']['clustered']
    Pc = res['enc10_buoy-cur']['clustered']
    Gc = res['enc10_buoy-m2']['clustered']
    anchor_ok = 0.0 < A['mean'] < 0.30
    mde = float('nan')
    if np.isfinite(Pc['se']):
        mde = stats.t.ppf(0.975, Pc['n'] - 1) * Pc['se']
    uni = [r for r in rows if not r['acc2']]
    dnh = float(np.nanmean([r['enc10_buoy'] - r['enc10_cur'] for r in uni])) \
        if uni else float('nan')
    passed = (Pc['mean'] >= 0.03 and Pc['lo'] > 0 and Gc['lo'] > 0)
    killed = Pc['hi'] < 0.02
    pilot = not (mde < 0.03)
    print('\n=== VERDICT (pre-registered) ===')
    print(f"  ANCHOR fld orc-cur clustered: {A['mean']:+.4f} "
          f"CI[{A['lo']:+.4f},{A['hi']:+.4f}] -> "
          f"{'OK' if anchor_ok else 'FAIL'}")
    print(f"  PRIMARY enc10 buoy-cur clustered: {Pc['mean']:+.4f} "
          f"CI[{Pc['lo']:+.4f},{Pc['hi']:+.4f}]  (gate buoy-m2 "
          f"{Gc['mean']:+.4f} CI[{Gc['lo']:+.4f},{Gc['hi']:+.4f}])")
    print(f"  MDE (clustered) = {mde:.4f}  -> "
          f"{'POWERED' if not pilot else 'PILOT (MDE > +0.03 bar)'}")
    print(f"  do-no-harm on unimodal windows: {dnh:+.4f} "
          f"(n={len(uni)}, bar >= -0.01)")
    print(f"  PASS: {passed}   KILL: {killed}")
    if not regime_present:
        v = ('TARGET REGIME ABSENT: no record is cross-Tp bimodal under the '
             'validated screen -> the real leg CANNOT test the surviving '
             'claim on its target regime; claim stays synthetic-scoped for '
             'a reason unrelated to the estimator. Numbers above are '
             'census + do-no-harm only.')
    elif not anchor_ok:
        v = ('ANCHOR FAIL: with a passing harness gate, even CLEAN '
             'dense-derived theta does not pay on real seas -> the 2-lobe '
             'family is misspecified on real data; deliverable scoped to '
             'synthetic JONSWAP (scientific negative, not a harness bug)')
    elif passed and not pilot:
        v = 'BUOY-GRADE SPEC SURVIVES REAL SEAS'
    elif killed:
        v = ('KILLED: buoy-grade supply does not pay on real seas -> '
             'deliverable scoped to synthetic JONSWAP permanently')
    else:
        v = ('INCONCLUSIVE/PILOT: between the bars at the achievable n; '
             'report effect size + sd for a powered design')
    print(f'  -> {v}')
    json.dump(dict(selftest=st, rows=rows, contrasts=res,
                   rec_diag=rec_diag, regime_present=bool(regime_present),
                   anchor_ok=bool(anchor_ok),
                   passed=bool(passed), killed=bool(killed),
                   pilot=bool(pilot), mde_clustered=mde, dnh_unimodal=dnh,
                   verdict=v),
              open(OUT, 'w'), indent=1, default=float)
    print(f'saved -> {OUT}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
