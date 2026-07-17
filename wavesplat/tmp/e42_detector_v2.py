"""e42 — v2 JOINT evidence detector + FROZEN calibration.

WHY v2. Every fix below was forced by a defect found in e41's raw calibration
data (results/direction3/e41_detector_calibration.json), not by taste:

  F1 GLOBAL d1.  e41 searched d1 only in mu_hat +- 40 deg (D1_GRID). But 10/24
     calibration positives had mu_err > 45 deg (up to 154 deg = a wrong-lobe
     beamform lock). fit1 could not reach them BY CONSTRUCTION. e41's apparent
     "+26.8 deg d1 improvement" was delivered 9/10 times by the 2-lobe stage
     via lobe SWAP, and it COST -11.1 deg on the 14 seas where mu_hat was
     already good. -> d1 is now searched over the full circle.

  F2 TOP-K SEEDING (K=6), not greedy freeze. e41 froze lobe-1 at the stage-1
     argmax before ever looking at lobe 2. When stage 1 locks the wrong lobe,
     the whole fit is conditioned on it.

  F3 w FLOOR 0.15.  23/24 e41 negatives sat EXACTLY at the w floor (0.06);
     24/24 at w <= 0.12; the two negatives that set TAU=14.64 had w =
     0.060/0.120. That is a tiny nuisance lobe absorbing realization speckle,
     not a second wave system. With w >= 0.15 the 2-lobe hypothesis MEANS
     "a real second system", and the nuisance channel is closed structurally
     rather than filtered post hoc.

  F4 a2 GRID -> 0.50.  True a2 = k(Tp2)/kp spans [0.51, 1.78] (Tp2 = 7 *
     U(0.75,1.4)); e41's A2_GRID started at 0.55 and clipped the long-swell end.

THE FAMILY CHANGED => e41's TAU is VOID. TAU is re-derived here on FRESH
negatives, and the calibration seeds are disjoint from every scored sea.

PRE-REGISTERED (frozen before any scored run):
  ENGINE   noise (3*sigma0)^2, gamma scale var(z), free bank only, spread 30
           deg and gamma 3.3 assumed (= arena X constants; declared regime
           knowledge -- this makes the model WELL-SPECIFIED in arena X, the
           friendliest case that exists, and MISSPECIFIED in arena Y).
  CAL NEG  build_sea_bimodal(3200+j, unimodal=True), j=0..23   (e41 used 3000+j)
  CAL POS  build_sea_crosstp(600+j),                 j=0..23   (e41 used 500+j)
  TAU      95th percentile of DeltaLogE on CAL NEG.
  QUALIFY  TPR >= 60% at TAU on CAL POS AND median |d2 err| <= 25 deg among
           accepted. If unqualified, scoring still runs, labeled
           detector-limited.

Usage: python3 tmp/e42_detector_v2.py [--smoke]
Out:   results/direction3/e42_detector_v2_calibration.json
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

from e29_gennet_C import DEVICE, DEPTH, _MASK, KXN_NP, KYN_NP, design_t
from e37_bimodal_ladder import (build_sea_crosstp, build_sea_bimodal,
                                mu_hat_of, ang_err_deg)
from e41_regime_prize_ladder import lobe_grid, tp_of_k
from d18_relax_estimator import omega
from d3_sa_achievability_gap import dispersion_lam

OUT = 'results/direction3/e42_detector_v2_calibration.json'

# ---- frozen v2 search family -------------------------------------------
NDIR = 36                                    # global, 10 deg step  (F1)
DIRS = np.arange(NDIR) * (2 * np.pi / NDIR)
A_GRID = np.round(np.arange(0.50, 1.851, 0.10), 3)   # 14 nodes        (F4)
W_GRID = np.array([0.15, 0.24, 0.33, 0.42, 0.51])    # floor 0.15      (F3)
TOPK = 6                                             #                 (F2)
MIN_SEP_DEG = 45.0
SPREAD = 30.0
GAMMA = 3.3
CHUNK = 96
LOBE_CHUNK = 48
KEEP_HYP = 48        # hypotheses retained for the mixture-of-fields arm


class JointEngine:
    """Exact log-evidence of z under N(0, Phi diag(gam) Phi^T + (3 s0)^2 I).

    Precomputes M_i = Phi diag([g_i,g_i]) Phi^T for every distinct lobe once,
    so a 2-lobe hypothesis is a LINEAR combine + Cholesky rather than a fresh
    einsum. gamma convention is identical to solve_with_grid: the MIXED atom
    vector is normalized to sum 1 and scaled by var(z).
    """

    def __init__(self, S, smult=3.0):
        kp = S['kp']
        kxb, kyb = KXN_NP * kp, KYN_NP * kp
        om = omega(np.hypot(kxb, kyb) + 1e-12, DEPTH)
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32,
                                   device=DEVICE)
        self.Phi = design_t(t(kxb), t(kyb), t(om), t(S['P']))     # (n, 2NB)
        self.z = t(S['z'])
        self.s2 = float((smult * S['sigma0']) ** 2)
        self.vz = float(np.var(S['z']))
        self.n = len(S['z'])
        self.eye = torch.eye(self.n, device=DEVICE)
        self.kp = kp
        self._lobe = {}      # key -> index into self.M
        self.M = None        # (L, n, n)
        self.sums = None     # (L,) atom-energy sums (pre-clipping mass)
        self.grids = []      # (L,) 51x51 numpy

    def build_lobes(self, keys):
        """keys: list of (a, mu). Precompute M and grids for each."""
        todo = [k for k in keys if k not in self._lobe]
        for k in todo:
            self._lobe[k] = len(self.grids)
            a, mu = k
            Tp_eff = float(tp_of_k(a * self.kp))
            self.grids.append(lobe_grid(self.kp, [(Tp_eff, GAMMA, 1.0)],
                                        [(mu, SPREAD, 1.0)]))
        if not todo:
            return
        atoms = np.stack([self.grids[self._lobe[k]][_MASK] for k in todo])
        atoms_t = torch.tensor(atoms, dtype=torch.float32, device=DEVICE)
        Ms = []
        for i in range(0, len(atoms_t), LOBE_CHUNK):
            g = atoms_t[i:i + LOBE_CHUNK]
            gam = torch.cat([g, g], 1)                       # (b, 2NB)
            Ms.append(torch.einsum('nk,bk,mk->bnm', self.Phi, gam, self.Phi))
        Mnew = torch.cat(Ms)
        s_new = atoms_t.sum(1)
        self.M = Mnew if self.M is None else torch.cat([self.M, Mnew])
        self.sums = s_new if self.sums is None else torch.cat([self.sums, s_new])

    def idx(self, a, mu):
        return self._lobe[(round(float(a), 3), round(float(mu), 5))]

    def logev_pairs(self, ii, jj, ww):
        """ii, jj: lobe indices (jj may be -1 for the 1-lobe case). ww: weights.
        Hypothesis atom vector = (1-w) g_ii + w g_jj  (w=0 when jj<0)."""
        ii = np.asarray(ii); jj = np.asarray(jj); ww = np.asarray(ww, float)
        out = np.empty(len(ii))
        for c in range(0, len(ii), CHUNK):
            i_c = torch.as_tensor(ii[c:c + CHUNK], device=DEVICE)
            w_c = torch.as_tensor(ww[c:c + CHUNK], dtype=torch.float32,
                                  device=DEVICE)
            j_raw = jj[c:c + CHUNK]
            two = j_raw >= 0
            Mi = self.M[i_c]
            si = self.sums[i_c]
            if two.any():
                j_c = torch.as_tensor(np.where(two, j_raw, 0), device=DEVICE)
                tw = torch.as_tensor(two, device=DEVICE)
                wv = torch.where(tw, w_c, torch.zeros_like(w_c))
                Mtot = (1 - wv)[:, None, None] * Mi \
                    + wv[:, None, None] * self.M[j_c]
                stot = (1 - wv) * si + wv * self.sums[j_c]
            else:
                Mtot, stot = Mi, si
            scale = (self.vz / stot.clamp_min(1e-30))[:, None, None]
            K = Mtot * scale + self.s2 * self.eye[None]
            L = torch.linalg.cholesky(K)
            logdet = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
            rhs = self.z[None, :, None].expand(len(K), -1, -1).contiguous()
            al = torch.cholesky_solve(rhs, L)
            quad = (self.z[None, :, None] * al).sum((1, 2))
            out[c:c + CHUNK] = (-0.5 * (quad + logdet + self.n
                                        * np.log(2 * np.pi))).cpu().numpy()
        return out

    def grid_of(self, i, j=-1, w=0.0):
        if j < 0:
            return self.grids[i]
        return (1 - w) * self.grids[i] + w * self.grids[j]


def _bma_weights(loge):
    p = np.exp(loge - loge.max())
    return p / p.sum()


def fit_joint(S):
    """v2 blind joint fit. No ground truth touched."""
    eng = JointEngine(S)
    mu_hat = mu_hat_of(S)

    # ---- all distinct lobes on the global (dir x a) lattice
    keys = [(round(float(a), 3), round(float(m), 5))
            for m in DIRS for a in A_GRID]
    eng.build_lobes(keys)
    lut = {k: eng._lobe[k] for k in keys}
    cand1 = [(m, a) for m in DIRS for a in A_GRID]
    idx1 = np.array([lut[(round(float(a), 3), round(float(m), 5))]
                     for (m, a) in cand1])

    # ---- stage A: 1-lobe, GLOBAL d1                                    (F1)
    loge1 = eng.logev_pairs(idx1, -np.ones(len(idx1), int),
                            np.zeros(len(idx1)))
    w1 = _bma_weights(loge1)
    fit1_grid = np.tensordot(w1, np.stack([eng.grids[i] for i in idx1]),
                             axes=1)
    b1 = int(np.argmax(loge1))
    d1_1lobe, a1_1lobe = cand1[b1]
    logE1 = float(loge1[b1])

    # ---- stage B: joint over TOP-K lobe-1 seeds                        (F2)
    order = np.argsort(-loge1)[:TOPK]
    hyp, II, JJ, WW = [], [], [], []
    for oi in order:
        m1, a1 = cand1[oi]
        i1 = idx1[oi]
        for m2 in DIRS:
            if abs(np.rad2deg(np.angle(np.exp(1j * (m2 - m1))))) < MIN_SEP_DEG:
                continue
            for a2 in A_GRID:
                i2 = lut[(round(float(a2), 3), round(float(m2), 5))]
                for w in W_GRID:                                        # (F3)
                    hyp.append((m1, a1, m2, a2, float(w)))
                    II.append(i1); JJ.append(i2); WW.append(w)
    loge2 = eng.logev_pairs(II, JJ, WW)
    b2 = int(np.argmax(loge2))
    d1f, a1f, d2f, a2f, wf = hyp[b2]
    logE2 = float(loge2[b2])

    # ---- 2-lobe BMA (threshold-free)
    w2 = _bma_weights(loge2)
    keep = np.argsort(-w2)[:KEEP_HYP]
    wk = w2[keep] / w2[keep].sum()
    grids_k = np.stack([eng.grid_of(II[j], JJ[j], WW[j]) for j in keep])
    fit2_grid = np.tensordot(wk, grids_k, axes=1)

    # ---- full BMA over {1-lobe} U {2-lobe}, equal model prior
    l0 = max(loge1.max(), loge2.max())
    ev1 = float(np.exp(loge1 - l0).mean())
    ev2 = float(np.exp(loge2 - l0).mean())
    p2 = ev2 / (ev1 + ev2)
    bma_grid = (1 - p2) * fit1_grid + p2 * fit2_grid

    # ---- hypothesis set for the MIXTURE-OF-FIELDS arm (D2's escape from the
    #      single-gamma class): mixing GRIDS averages covariances and stays in
    #      the incumbent's class; mixing FIELDS is the actual posterior mean.
    k1 = np.argsort(-w1)[:KEEP_HYP]
    mix_grids = [eng.grids[idx1[j]] for j in k1] + [g for g in grids_k]
    mix_w = np.concatenate([(1 - p2) * (w1[k1] / w1[k1].sum()),
                            p2 * wk])
    mix_w = mix_w / mix_w.sum()

    return dict(mu_hat=float(mu_hat), d1=float(d1f), a1=float(a1f),
                d2=float(d2f), a2=float(a2f), w=float(wf),
                d1_1lobe=float(d1_1lobe), a1_1lobe=float(a1_1lobe),
                logE1=logE1, logE2=logE2, dlogE=float(logE2 - logE1),
                p2=float(p2), fit1_grid=fit1_grid, fit2_grid=fit2_grid,
                bma_grid=bma_grid, mix_grids=mix_grids, mix_w=mix_w)


def true_rings(S):
    Tp1 = S.get('Tp1', 7.0)
    k1 = 2 * np.pi / dispersion_lam(Tp1, DEPTH)
    k2 = 2 * np.pi / dispersion_lam(S['Tp2'], DEPTH)
    return float(k1 / S['kp']), float(k2 / S['kp'])


def match_errors(F, S):
    a1t, a2t = true_rings(S)
    e_dir = ang_err_deg(F['d1'], S['d1']) + ang_err_deg(F['d2'], S['d2'])
    e_swp = ang_err_deg(F['d1'], S['d2']) + ang_err_deg(F['d2'], S['d1'])
    if e_dir <= e_swp:
        return dict(d1_err=ang_err_deg(F['d1'], S['d1']),
                    d2_err=ang_err_deg(F['d2'], S['d2']),
                    a2_rel=float(F['a2'] / a2t - 1),
                    w_err=float(F['w'] - S['r2']), swap=False)
    return dict(d1_err=ang_err_deg(F['d2'], S['d1']),
                d2_err=ang_err_deg(F['d1'], S['d2']),
                a2_rel=float(F['a1'] / a2t - 1),
                w_err=float((1 - F['w']) - S['r2']), swap=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n_neg', type=int, default=24)
    ap.add_argument('--n_pos', type=int, default=24)
    a = ap.parse_args()
    if a.smoke:
        a.n_neg, a.n_pos = 2, 3
    t0 = time.time()
    neg, pos = [], []
    print('--- CAL NEGATIVES (unimodal, seeds 3200+j) ---', flush=True)
    for j in range(a.n_neg):
        S = build_sea_bimodal(3200 + j, unimodal=True)
        F = fit_joint(S)
        neg.append(dict(j=j, dlogE=F['dlogE'], p2=F['p2'], w=F['w']))
        print(f"neg{j:2d} dlogE {F['dlogE']:8.2f} p2 {F['p2']:.3f} "
              f"w {F['w']:.2f} [{time.time()-t0:.0f}s]", flush=True)
    print('--- CAL POSITIVES (cross-Tp, seeds 600+j) ---', flush=True)
    for j in range(a.n_pos):
        S = build_sea_crosstp(600 + j)
        F = fit_joint(S)
        E = match_errors(F, S)
        mu_e = ang_err_deg(F['mu_hat'], S['d1'])
        d1_1l = ang_err_deg(F['d1_1lobe'], S['d1'])
        pos.append(dict(j=j, dlogE=F['dlogE'], p2=F['p2'], r2=S['r2'],
                        Tp2=S['Tp2'], mu_err=mu_e, d1_1lobe_err=d1_1l, **E))
        print(f"pos{j:2d} dlogE {F['dlogE']:8.2f} muE {mu_e:5.1f} "
              f"d1_1L {d1_1l:5.1f} d1e {E['d1_err']:5.1f} "
              f"d2e {E['d2_err']:5.1f} a2r {E['a2_rel']:+.3f} "
              f"sw {int(E['swap'])} [{time.time()-t0:.0f}s]", flush=True)

    dneg = np.array([r['dlogE'] for r in neg])
    dpos = np.array([r['dlogE'] for r in pos])
    tau = float(np.percentile(dneg, 95))
    tpr = float((dpos >= tau).mean())
    acc = [r for r in pos if r['dlogE'] >= tau]
    med_d2 = float(np.median([r['d2_err'] for r in acc])) if acc else float('nan')
    med_a2 = float(np.median([abs(r['a2_rel']) for r in acc])) if acc else float('nan')
    qualified = bool(tpr >= 0.60 and med_d2 <= 25.0)
    mu_all = np.array([r['mu_err'] for r in pos])
    d1_1l = np.array([r['d1_1lobe_err'] for r in pos])
    print('\n=== v2 CALIBRATION (frozen family) ===')
    print(f"  TAU = {tau:.2f}  (95th pct of {len(dneg)} unimodal dlogE)")
    print(f"  TPR at TAU = {tpr:.0%} ({int((dpos>=tau).sum())}/{len(dpos)})")
    print(f"  accepted median: |d2 err| {med_d2:.1f} deg | |a2 rel| {med_a2:.3f}")
    print(f"  neg dlogE: {np.sort(dneg).round(1).tolist()}")
    print(f"  neg w    : {np.sort([r['w'] for r in neg]).round(2).tolist()}")
    print(f"  pos dlogE: {np.sort(dpos).round(1).tolist()}")
    print(f"  QUALIFIED (TPR>=60% & med d2err<=25deg): {qualified}")
    print(f"\n  --- F1 check: does GLOBAL 1-lobe search fix the mu_hat lock? ---")
    print(f"  mu_hat   d1 err: median {np.median(mu_all):5.1f} mean {mu_all.mean():5.1f}")
    print(f"  1-lobe   d1 err: median {np.median(d1_1l):5.1f} mean {d1_1l.mean():5.1f}")
    cat = mu_all > 45
    if cat.any():
        print(f"  on the {cat.sum()} catastrophic (mu_err>45) seas: "
              f"mu {np.median(mu_all[cat]):.0f} -> 1-lobe {np.median(d1_1l[cat]):.0f} deg")
    if (~cat).any():
        print(f"  on the {(~cat).sum()} healthy seas:              "
              f"mu {np.median(mu_all[~cat]):.0f} -> 1-lobe {np.median(d1_1l[~cat]):.0f} deg")
    json.dump(dict(neg=neg, pos=pos, tau=tau, tpr=tpr, med_d2_err=med_d2,
                   med_a2_rel=med_a2, qualified=qualified, smoke=a.smoke,
                   family=dict(ndir=NDIR, a_grid=A_GRID.tolist(),
                               w_grid=W_GRID.tolist(), topk=TOPK,
                               min_sep_deg=MIN_SEP_DEG)),
              open(OUT, 'w'), indent=1, default=float)
    print(f"saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
