"""e41 — EVIDENCE (type-II ML) 2-lobe detector + CALIBRATION (frozen before
scoring).

WHY: e37's blind detector (angular profile of the pm map + wedge radial
centroid, accept 0.25) had a 100% unimodal false-positive rate and a mean
lobe-2 direction error of 66.5 deg -> its -0.059 loss is detector quality,
not information. THIS detector uses the datum the map throws away: the 70 s
TIME axis. The evidence of z under N(0, Phi diag(gamma) Phi^T + s^2 I) with a
parametric 2-lobe gamma ties omega to |k| through the dispersion relation in
Phi itself, so the RING (the +0.102 dominating term) is estimated jointly
with direction from the full spatio-temporal likelihood, not from a
single-snapshot angular profile. The family is 5-dimensional
(d1, a1, d2, a2, w) — NOT the 1904-atom ill-posed problem.

PRE-REGISTERED (frozen before any scored run; smoke on si>=900 only):
  ENGINE  noise = (3*sigma0)^2 (identical to the scoring solver smult=3),
          gamma scale = var(z) (identical convention), free bank only
          (arena has no bound waves), spread fixed 30 deg (arena constant =
          regime knowledge, same status as Tp1=7 for the universal radial).
  SEARCH  stage 1 (1-lobe): d1 in mu_hat+{-40..40 step 10} x a1 in
          {0.80,0.90,1.00,1.10,1.20,1.35}; refine d1 +-6 step 3 x
          a1*{0.93,1.00,1.07}.
          stage 2 (2-lobe): lobe1 frozen at stage-1 argmax; d2 in
          d1_hat+-{45..165 step 10} x a2 in {0.55..1.85 step 0.1} x w in
          {0.12,0.20,0.28,0.36,0.44,0.52}; refine (d2 +-8 step 4) x
          (a2 +-0.06 step 0.03) x (w +-0.06 step 0.06); then refit lobe1
          (d1 +-6 step 3 x a1*{0.93,1,1.07}) with lobe2 fixed.
  ACCEPT  DeltaLogE = max logE(2-lobe) - max logE(1-lobe) >= TAU, where TAU =
          95th percentile of DeltaLogE on 24 CALIBRATION unimodal seas
          (build_sea_bimodal(3000+j, unimodal=True), j=0..23) — disjoint
          seeds from every scored sea.
  QUALIFY (detector itself) TPR >= 60% at TAU on 24 calibration positives
          (build_sea_crosstp(500+j), j=0..23) AND median |d2 err| <= 25 deg
          among accepted. If unqualified, scoring still runs but is labeled
          detector-limited.
  OUTPUT ARMS (defined here, scored in e41_blind_recovery.py)
          fit1      = evidence-weighted BMA over the 1-lobe family
          fit2_hard = if DeltaLogE >= TAU: BMA over the 2-lobe family,
                      else fit1                              [PRIMARY]
          fit2_point= if accepted: argmax point grid (sharpness diagnostic)
          fit_bma   = full Bayesian model average over 1-lobe U 2-lobe
                      (equal model prior, quadrature marginals) [SECONDARY]

Usage: python3 tmp/e41_evidence_detector.py [--smoke]   (calibration main)
Output: results/direction3/e41_detector_calibration.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (corr, DEVICE, DEPTH, _MASK, KXN_NP, KYN_NP,
                          design_t, NGRID, KREL)
from e29_gate0_ladder import solve_with_grid
from e37_bimodal_ladder import (build_sea_crosstp, build_sea_bimodal,
                                mu_hat_of, ang_err_deg)
from e41_regime_prize_ladder import lobe_grid, tp_of_k
from d18_relax_estimator import omega
from d3_sa_achievability_gap import dispersion_lam

OUT = 'results/direction3/e41_detector_calibration.json'
NB = len(KXN_NP)

# frozen search grids
A1_GRID = np.array([0.80, 0.90, 1.00, 1.10, 1.20, 1.35])
D1_GRID = np.deg2rad(np.arange(-40, 41, 10.0))
D1_REF = np.deg2rad(np.arange(-6, 7, 3.0))
A1_REF = np.array([0.93, 1.00, 1.07])
D2_OFF = np.deg2rad(np.arange(45, 166, 10.0))
A2_GRID = np.arange(0.55, 1.851, 0.10)
W_GRID = np.array([0.12, 0.20, 0.28, 0.36, 0.44, 0.52])
D2_REF = np.deg2rad(np.arange(-8, 9, 4.0))
A2_REF = np.arange(-0.06, 0.061, 0.03)
W_REF = np.array([-0.06, 0.0, 0.06])
SPREAD = 30.0
CHUNK = 128


class EvidenceEngine:
    """log evidence of z under N(0, Phi diag(gam) Phi^T + (3 s0)^2 I),
    free bank of this sea, gamma-scale var(z) — the scoring solver's exact
    covariance model."""

    def __init__(self, S, smult=3.0):
        kp = S['kp']
        kxb = KXN_NP * kp
        kyb = KYN_NP * kp
        om = omega(np.hypot(kxb, kyb) + 1e-12, DEPTH)
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32,
                                   device=DEVICE)
        self.Phi = design_t(t(kxb), t(kyb), t(om), t(S['P']))
        self.z = t(S['z'])
        self.s2 = float((smult * S['sigma0']) ** 2)
        self.vz = float(np.var(S['z']))
        self.n = len(S['z'])
        self.eye = torch.eye(self.n, device=DEVICE)

    def logev(self, Gb):
        """Gb: (B, NB) numpy atom energies (unnormalized). Returns (B,)."""
        out = np.empty(len(Gb))
        for i in range(0, len(Gb), CHUNK):
            g = torch.tensor(np.asarray(Gb[i:i + CHUNK]),
                             dtype=torch.float32, device=DEVICE)
            g = g / g.sum(1, keepdim=True).clamp_min(1e-30) * self.vz
            gam = torch.cat([g, g], 1)
            K = torch.einsum('nk,bk,mk->bnm', self.Phi, gam, self.Phi)
            K = K + self.s2 * self.eye[None]
            L = torch.linalg.cholesky(K)
            logdet = 2 * torch.log(
                torch.diagonal(L, dim1=1, dim2=2)).sum(1)
            rhs = self.z[None, :, None].expand(len(K), -1, -1).contiguous()
            al = torch.cholesky_solve(rhs, L)
            quad = (self.z[None, :, None] * al).sum((1, 2))
            out[i:i + CHUNK] = (-0.5 * (quad + logdet + self.n
                                        * np.log(2 * np.pi))).cpu().numpy()
        return out


def ring_lobe_atoms(kp, a, mu, cache):
    """Masked atom energies of one lobe with ring a (=peak k / kp) at
    direction mu, spread 30 deg. Deposit-grid convention (solver-consistent:
    unnormalized after masking)."""
    key = (round(float(a), 4), round(float(mu), 5))
    if key not in cache:
        Tp_eff = float(tp_of_k(a * kp))
        g = lobe_grid(kp, [(Tp_eff, 3.3, 1.0)], [(mu, SPREAD, 1.0)])
        cache[key] = (g[_MASK].astype(np.float64), g)
    return cache[key]


def _bma(grids, loges):
    w = np.exp(loges - loges.max())
    w /= w.sum()
    return np.tensordot(w, np.stack(grids), axes=1)


def fit_two_lobe(S, verbose=False):
    """Blind evidence fit. Returns dict with params, DeltaLogE, and the four
    output grids (51x51, solver-ready)."""
    eng = EvidenceEngine(S)
    kp = S['kp']
    mu_hat = mu_hat_of(S)
    cache = {}

    # ---- stage 1: 1-lobe
    c1 = [(mu_hat + dd, aa) for dd in D1_GRID for aa in A1_GRID]
    G1 = [ring_lobe_atoms(kp, a, m, cache)[0] for (m, a) in c1]
    e1 = eng.logev(G1)
    m0, a0 = c1[int(np.argmax(e1))]
    c1r = [(m0 + dd, a0 * fa) for dd in D1_REF for fa in A1_REF]
    G1r = [ring_lobe_atoms(kp, a, m, cache)[0] for (m, a) in c1r]
    e1r = eng.logev(G1r)
    cand1 = c1 + c1r
    loge1 = np.concatenate([e1, e1r])
    i1 = int(np.argmax(loge1))
    d1h, a1h = cand1[i1]
    logE1 = float(loge1[i1])
    grids1 = [ring_lobe_atoms(kp, a, m, cache)[1] for (m, a) in cand1]
    fit1_grid = _bma(grids1, loge1)

    # ---- stage 2: 2-lobe (lobe1 frozen at stage-1 argmax)
    g1a, g1grid = ring_lobe_atoms(kp, a1h, d1h, cache)
    pairs = [(d1h + sg * off, a2) for sg in (+1, -1) for off in D2_OFF
             for a2 in A2_GRID]
    L2at = [ring_lobe_atoms(kp, a2, m2, cache)[0] for (m2, a2) in pairs]
    cand2, G2 = [], []
    for j, (m2, a2) in enumerate(pairs):
        for w in W_GRID:
            cand2.append((d1h, a1h, m2, a2, w))
            G2.append((1 - w) * g1a + w * L2at[j])
    e2 = eng.logev(G2)
    jb = int(np.argmax(e2))
    _, _, m2b, a2b, wb = cand2[jb]
    # refine (d2, a2, w)
    candr, Gr = [], []
    for dd in D2_REF:
        for da in A2_REF:
            a2n = a2b + da
            if not (0.45 <= a2n <= 2.0):
                continue
            l2 = ring_lobe_atoms(kp, a2n, m2b + dd, cache)[0]
            for dw in W_REF:
                wn = float(np.clip(wb + dw, 0.06, 0.6))
                candr.append((d1h, a1h, m2b + dd, a2n, wn))
                Gr.append((1 - wn) * g1a + wn * l2)
    er = eng.logev(Gr)
    cand2 += candr
    e2 = np.concatenate([e2, er])
    jb = int(np.argmax(e2))
    _, _, m2b, a2b, wb = cand2[jb]
    # refit lobe1 with lobe2 fixed
    l2b = ring_lobe_atoms(kp, a2b, m2b, cache)[0]
    candf, Gf = [], []
    for dd in D1_REF:
        for fa in A1_REF:
            candf.append((d1h + dd, a1h * fa, m2b, a2b, wb))
            l1 = ring_lobe_atoms(kp, a1h * fa, d1h + dd, cache)[0]
            Gf.append((1 - wb) * l1 + wb * l2b)
    ef = eng.logev(Gf)
    cand2 += candf
    loge2 = np.concatenate([e2, ef])
    i2 = int(np.argmax(loge2))
    d1f, a1f, d2f, a2f, wf = cand2[i2]
    logE2 = float(loge2[i2])

    # grids for BMA over the 2-lobe family (mix in grid space)
    def grid_of(c):
        _d1, _a1, _d2, _a2, _w = c
        return (1 - _w) * ring_lobe_atoms(kp, _a1, _d1, cache)[1] \
            + _w * ring_lobe_atoms(kp, _a2, _d2, cache)[1]
    # memory: only need weights above threshold
    w2 = np.exp(loge2 - loge2.max())
    w2 /= w2.sum()
    keep = np.where(w2 > 1e-4)[0]
    fit2_grid = np.tensordot(w2[keep] / w2[keep].sum(),
                             np.stack([grid_of(cand2[j]) for j in keep]),
                             axes=1)
    point_grid = grid_of(cand2[i2])

    # full BMA: model evidences = quadrature means over each family
    l0 = max(loge1.max(), loge2.max())
    ev1 = float(np.exp(loge1 - l0).mean())
    ev2 = float(np.exp(loge2 - l0).mean())
    p2 = ev2 / (ev1 + ev2)
    bma_grid = (1 - p2) * fit1_grid + p2 * fit2_grid

    return dict(mu_hat=float(mu_hat), d1=float(d1f), a1=float(a1f),
                d2=float(d2f), a2=float(a2f), w=float(wf),
                logE1=logE1, logE2=logE2, dlogE=float(logE2 - logE1),
                p2=float(p2), fit1_grid=fit1_grid, fit2_grid=fit2_grid,
                point_grid=point_grid, bma_grid=bma_grid)


def true_rings(S):
    """(a1_true, a2_true) = true peak wavenumbers / kp_hat."""
    Tp1 = S.get('Tp1', 7.0)
    k1 = 2 * np.pi / dispersion_lam(Tp1, DEPTH)
    k2 = 2 * np.pi / dispersion_lam(S['Tp2'], DEPTH)
    return float(k1 / S['kp']), float(k2 / S['kp'])


def match_errors(F, S):
    """Assign fitted (lobe1, lobe2) to true (d1, d2) by direction, report
    direction + ring errors of the SECONDARY true lobe (the prize carrier)."""
    a1t, a2t = true_rings(S)
    # candidate assignments: fitted lobe1->true1 & lobe2->true2, or swapped
    e_direct = ang_err_deg(F['d1'], S['d1']) + ang_err_deg(F['d2'], S['d2'])
    e_swap = ang_err_deg(F['d1'], S['d2']) + ang_err_deg(F['d2'], S['d1'])
    if e_direct <= e_swap:
        return dict(d1_err=ang_err_deg(F['d1'], S['d1']),
                    d2_err=ang_err_deg(F['d2'], S['d2']),
                    a1_err=float(F['a1'] - a1t),
                    a2_err=float(F['a2'] - a2t),
                    a2_rel=float(F['a2'] / a2t - 1),
                    w_err=float(F['w'] - S['r2']), swap=False)
    return dict(d1_err=ang_err_deg(F['d2'], S['d1']),
                d2_err=ang_err_deg(F['d1'], S['d2']),
                a1_err=float(F['a2'] - a1t),
                a2_err=float(F['a1'] - a2t),
                a2_rel=float(F['a1'] / a2t - 1),
                w_err=float((1 - F['w']) - S['r2']), swap=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n_neg', type=int, default=24)
    ap.add_argument('--n_pos', type=int, default=24)
    a = ap.parse_args()
    if a.smoke:
        a.n_neg, a.n_pos = 3, 3
    t0 = time.time()
    neg, pos = [], []
    print('--- calibration NEGATIVES (unimodal, seeds 3000+j) ---', flush=True)
    for j in range(a.n_neg):
        S = build_sea_bimodal(3000 + j, unimodal=True)
        F = fit_two_lobe(S)
        neg.append(dict(j=j, dlogE=F['dlogE'], p2=F['p2'],
                        d1=F['d1'], a1=F['a1'], w=F['w']))
        print(f"neg{j:2d} dlogE {F['dlogE']:8.2f} p2 {F['p2']:.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    print('--- calibration POSITIVES (cross-Tp, seeds crosstp 500+j) ---',
          flush=True)
    for j in range(a.n_pos):
        S = build_sea_crosstp(500 + j)
        F = fit_two_lobe(S)
        E = match_errors(F, S)
        pos.append(dict(j=j, dlogE=F['dlogE'], p2=F['p2'], r2=S['r2'],
                        Tp2=S['Tp2'], dsep=ang_err_deg(S['d1'], S['d2']),
                        **E))
        print(f"pos{j:2d} dlogE {F['dlogE']:8.2f} p2 {F['p2']:.3f} "
              f"d2e {E['d2_err']:5.1f} a2rel {E['a2_rel']:+.3f} "
              f"we {E['w_err']:+.3f} swap {int(E['swap'])} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    dneg = np.array([r['dlogE'] for r in neg])
    dpos = np.array([r['dlogE'] for r in pos])
    tau = float(np.percentile(dneg, 95))
    tpr = float((dpos >= tau).mean())
    fpr_cal = float((dneg >= tau).mean())
    acc = [r for r in pos if r['dlogE'] >= tau]
    med_d2 = float(np.median([r['d2_err'] for r in acc])) if acc else float('nan')
    med_a2 = float(np.median([abs(r['a2_rel']) for r in acc])) if acc else float('nan')
    med_w = float(np.median([abs(r['w_err']) for r in acc])) if acc else float('nan')
    qualified = bool(tpr >= 0.60 and med_d2 <= 25.0)
    print(f"\n=== CALIBRATION (frozen) ===")
    print(f"  TAU = {tau:.2f}  (95th pct of {len(dneg)} unimodal dlogE; "
          f"cal FPR {fpr_cal:.0%})")
    print(f"  TPR at TAU = {tpr:.0%}  ({int((dpos>=tau).sum())}/{len(dpos)})")
    print(f"  accepted-positive median errors: d2 {med_d2:.1f} deg | "
          f"a2 rel {med_a2:.3f} | w {med_w:.3f}")
    print(f"  neg dlogE: {np.sort(dneg).round(1).tolist()}")
    print(f"  pos dlogE: {np.sort(dpos).round(1).tolist()}")
    print(f"  DETECTOR QUALIFIED (TPR>=60% & med d2err<=25deg): {qualified}")
    json.dump(dict(neg=neg, pos=pos, tau=tau, tpr=tpr, fpr_cal=fpr_cal,
                   med_d2_err=med_d2, med_a2_rel=med_a2, med_w_err=med_w,
                   qualified=qualified, smoke=a.smoke),
              open(OUT, 'w'), indent=1, default=float)
    print(f"saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
