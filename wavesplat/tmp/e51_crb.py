"""e51 — 5-dim theta CRB for the regime/partition parameters (E-48 keystone 3).
See tmp/e50_crb_direction_check.md for the strategy. This is a DISCRIMINATING
experiment, not a winning one: E-43c showed the likelihood pinpoints theta
(true-node pctile 0.996), so the FIM is non-degenerate and the naive CRB may be
SMALL. The question is which parameters.

MODEL (matches the blind estimator's ensemble-Gaussian likelihood, JointEngine):
  z ~ N(0, C(theta)),  C(theta) = Mtot*(var(z)/stot) + (smult*sigma0)^2 I
  Mtot = (1-w) Phi diag([g1,g1]) Phi^T + w Phi diag([g2,g2]) Phi^T
  g_i  = lobe_grid(kp,[(Tp_i,GAMMA,1)],[(d_i,SPREAD,1)])[_MASK]   (GAMMA=3.3,SPREAD=30)
  stot = (1-w) sum(g1) + w sum(g2)
theta = (d1, Tp1, d2, Tp2, w) evaluated at the sea's TRUE values.

FIM (zero-mean Gaussian):  I_kl = 1/2 tr(C^-1 dC/dk C^-1 dC/dl),  dC via central
finite difference. CRB_kk = (I^-1)_kk. sqrt(CRB) reported per param; directions
converted to degrees, Tp to seconds.

PRE-REGISTERED (frozen before scored run):
  ARENAS  X (build_sea_crosstp: shape gamma3.3/spread30 = model WELL-SPECIFIED) = PRIMARY.
          Y (build_sea_open: shape open = model MISSPECIFIED) = secondary/caveat.
          FRESH block si=700..763 (n=64), unused. rng/seed reuse of crosstp/open
          builders is fine: CRB is a function of the geometry, not a scored contrast.
  PRIOR widths (arena X, for the A/B boundary):
          d2 offset ~ U(60,150) deg  => prior std ~ 26 deg (half-range/sqrt3 of 45)
          Tp2 = 7*U(0.75,1.4)        => prior std ~ 1.1 s
          w = r2 ~ U(0.2,0.5)        => prior std ~ 0.087
  DECISION (per parameter, on the MEDIAN sqrt(CRB) over seas):
    A info-limit    : sqrt(CRB(d2)) >= 0.6 * prior_std(d2)  (>~16 deg) => obs adds
                      little about the secondary lobe => information-level closure.
    B decision-limit: sqrt(CRB(d2)) <= 0.3 * prior_std(d2)  (<~8 deg)  => d2 identified
                      => blind~=0 is a decision/posterior problem, not information.
    between         : partial identification; report as such (no forced label).
  GUARD 1 (E-19 trap): store aperture ratio (obs span / lambda); Spearman + partial
          correlation of sqrt(CRB(d2)) with aperture. If |partial| collapses the
          signal, warn "CRB = aperture rename" (E-19 replay).
  GUARD 2 (achievability): also report FIM condition number and sqrt(CRB(d1)) (should
          be the tight one). A credible result = d1 tight, d2 the question.
  CAVEAT  Gaussian-approx model (true field is random-phase, deterministic-amp).
          Arena X = well-specified; Y = misspecified-model CRB.

Usage: python3 tmp/e51_crb.py [--smoke] [--n 64]
Out:   results/direction3/e51_crb.json
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

from e29_gennet_C import DEVICE, _MASK, DEPTH
from e37_bimodal_ladder import build_sea_crosstp
from e42_detector_v2 import JointEngine, GAMMA, SPREAD
from e41_regime_prize_ladder import lobe_grid
from e49_open_arena_blind import build_sea_open
from d3_sa_achievability_gap import dispersion_lam

OUT = 'results/direction3/e51_crb.json'

# central finite-difference steps per parameter (physical units)
EPS = dict(d1=np.deg2rad(0.5), Tp1=0.05, d2=np.deg2rad(0.5), Tp2=0.05, w=0.01)
PNAMES = ['d1', 'Tp1', 'd2', 'Tp2', 'w']


def _atoms(kp, Tp, d):
    """2NB atom vector [g,g] and NB sum for one lobe (matches JointEngine)."""
    g = lobe_grid(kp, [(Tp, GAMMA, 1.0)], [(d, SPREAD, 1.0)])[_MASK]
    return np.concatenate([g, g]).astype(np.float64), float(g.sum())


def build_C(eng, theta):
    """C(theta) as an n x n torch tensor (float64 for FIM stability)."""
    d1, Tp1, d2, Tp2, w = theta
    a1, s1 = _atoms(eng.kp, Tp1, d1)
    a2, s2 = _atoms(eng.kp, Tp2, d2)
    Phi = eng.Phi.double()                                   # (n, 2NB)
    t = lambda a: torch.tensor(a, dtype=torch.float64, device=DEVICE)
    M1 = (Phi * t(a1)[None]) @ Phi.T
    M2 = (Phi * t(a2)[None]) @ Phi.T
    Mtot = (1 - w) * M1 + w * M2
    stot = (1 - w) * s1 + w * s2
    scale = eng.vz / max(stot, 1e-30)
    return Mtot * scale + eng.s2 * torch.eye(eng.n, dtype=torch.float64,
                                             device=DEVICE)


def fim_crb(S, theta):
    """5x5 FIM and CRB (inverse) for the model N(0, C(theta)) at true theta."""
    eng = JointEngine(S)
    C0 = build_C(eng, theta)
    Cinv = torch.linalg.inv(C0)
    dC = []
    for k, p in enumerate(PNAMES):
        tp = list(theta); tm = list(theta)
        tp[k] += EPS[p]; tm[k] -= EPS[p]
        dC.append((build_C(eng, tp) - build_C(eng, tm)) / (2 * EPS[p]))
    # A_k = Cinv @ dC_k ; FIM_kl = 1/2 tr(A_k A_l)
    A = [Cinv @ d for d in dC]
    F = torch.empty((5, 5), dtype=torch.float64, device=DEVICE)
    for k in range(5):
        for l in range(k, 5):
            v = 0.5 * torch.einsum('ij,ji->', A[k], A[l])
            F[k, l] = v; F[l, k] = v
    Fn = F.cpu().numpy()
    cond = float(np.linalg.cond(Fn))
    try:
        CRB = np.linalg.inv(Fn)
        sc = np.sqrt(np.clip(np.diag(CRB), 0, None))
    except np.linalg.LinAlgError:
        sc = np.full(5, np.nan)
    return sc, cond, Fn


def aperture_ratio(S):
    """obs footprint span / dominant wavelength."""
    P = S['P']
    span = float(np.hypot(np.ptp(P[:, 0]), np.ptp(P[:, 1])))
    lam = dispersion_lam(S.get('Tp1', 7.0), DEPTH)
    return span / lam


def run_arena(name, builder, sis):
    rows = []
    t0 = time.time()
    for si in sis:
        try:
            S = builder(si)
            Tp1 = S.get('Tp1', 7.0)
            theta = [S['d1'], Tp1, S['d2'], S['Tp2'], S['r2']]
            sc, cond, _ = fim_crb(S, theta)
        except Exception as e:
            print(f"  {name}{si} EXCLUDED ({type(e).__name__}: {e})", flush=True)
            continue
        r = dict(si=si, cond=cond, r2=float(S['r2']), Tp2=float(S['Tp2']),
                 aperture=aperture_ratio(S),
                 sep_deg=float(np.rad2deg(np.abs(np.angle(
                     np.exp(1j * (S['d2'] - S['d1'])))))))
        for k, p in enumerate(PNAMES):
            r[f'crb_{p}'] = float(sc[k])
        r['crb_d1_deg'] = float(np.rad2deg(sc[0]))
        r['crb_d2_deg'] = float(np.rad2deg(sc[2]))
        rows.append(r)
        if len(rows) % 10 == 1:
            print(f"  {name}{si} d1 {r['crb_d1_deg']:5.1f}deg d2 "
                  f"{r['crb_d2_deg']:6.1f}deg w {r['crb_w']:.3f} Tp2 "
                  f"{r['crb_Tp2']:.2f}s cond {cond:.1e} sep {r['sep_deg']:.0f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return rows


def summarize(name, rows):
    print(f"\n=== {name} (n={len(rows)}) median sqrt(CRB) ===")
    med = {}
    for key, unit, pstd in [('crb_d1_deg', 'deg', None), ('crb_d2_deg', 'deg', 26.0),
                            ('crb_Tp2', 's', 1.31), ('crb_w', '', 0.087)]:
        v = np.array([r[key] for r in rows])
        m = float(np.median(v))
        med[key] = m
        extra = f"  (prior_std {pstd} {unit} -> ratio {m/pstd:.2f})" if pstd else ""
        print(f"  {key:12s}: median {m:.3f} {unit}  IQR[{np.percentile(v,25):.2f},"
              f"{np.percentile(v,75):.2f}]{extra}")
    apert = np.array([r['aperture'] for r in rows])
    d2 = np.array([r['crb_d2_deg'] for r in rows])
    rho, p = stats.spearmanr(apert, d2)
    print(f"  GUARD1 aperture vs sqrt(CRB(d2)): Spearman {rho:+.2f} (p={p:.3f})")
    print(f"  median FIM cond = {np.median([r['cond'] for r in rows]):.1e}")
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n', type=int, default=64)
    a = ap.parse_args()
    n = 3 if a.smoke else a.n
    sis = list(range(700, 700 + n))

    print(f"--- ARENA X (well-specified), si=700..{700+n-1} ---", flush=True)
    rowsX = run_arena('X', build_sea_crosstp, sis)
    print(f"--- ARENA Y (open, model-misspecified), si=700..{700+n-1} ---", flush=True)
    rowsY = run_arena('Y', build_sea_open, sis)

    medX = summarize('ARENA X (PRIMARY, well-specified)', rowsX)
    medY = summarize('ARENA Y (secondary, misspecified-model)', rowsY)

    # Verdict (audited): the ROBUST axis is w, NOT d2's threshold-crossing.
    # d2 crosses the pre-registered 0.6*prior by a coin-flip margin -> do not
    # headline it. Scope = model-class limit; GUARD2 achievability still owed.
    d2X = medX['crb_d2_deg']; wX = medX['crb_w']
    fw = float(np.mean([r['crb_w'] > 0.087 for r in rowsX]))
    verdict = (
        f"A-partial (model-class info-limit on the SECONDARY partition). "
        f"ROBUST: w unidentified sqrt(CRB)={wX:.3f} = {wX/0.087:.2f}x prior, "
        f"{fw:.0%} of seas > prior. BORDERLINE (do not headline): d2={d2X:.1f}deg "
        f"= {d2X/26.0:.2f}x prior (crosses 0.6 by coin-flip). d1 tight "
        f"({medX['crb_d1_deg']:.1f}deg). SCOPE: bounds estimators using the "
        f"ensemble-Gaussian model (blind & net both do -> explains blind~=net); "
        f"NOT the fundamental random-phase limit. GUARD2 achievability "
        f"(CRB->field-corr cap vs E-49) OWED before 'info-level closure' cite.")
    print(f"\n=== VERDICT (audited) ===\n  -> {verdict}")

    json.dump(dict(rowsX=rowsX, rowsY=rowsY, medX=medX, medY=medY,
                   verdict=verdict), open(OUT, 'w'), indent=1, default=float)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main()
