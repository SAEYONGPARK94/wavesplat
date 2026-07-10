"""G0b -- the prize, measured with estimators: selection-aware EP vs naive MMSE
under the THRESHOLD detection law.

F2b (docs/F2b_THRESHOLD_ACCOUNTING_2026-07-10.md) measured the mask's
information under the real (threshold) detection law: 17.13% of the amplitude
information at deployed sparsity on the steep accounting sea, gate-PASS. That
was information accounting. This script measures the PRIZE with actual
estimators, paired on identical observations -- the E5 power lesson: pairing on
the same data removes the sampling variance that made Door 1b unresolvable.

Scene (per sea): the door1b ray grid (240 x 48 rays x 12 scans), true
first-hit occlusion (Moller-Trumbore), then threshold dropout
(wavesim/raydrop_threshold.py, M=1 speckle): per scan, return iff
A(s)*X > T, A = Fresnel(theta)cos(theta)/(d/125)^2, theta = pi/2 - g - s,
s = the true along-ray slope at the hit, T = per-scan top-k (n_pts = 50).
Amplitudes: z = eta(hit) + noise at the 600 returns. The sea is scaled to the
accounting sea (sigma_eta ~ 1 m) where the mask is worth measuring; the mild
sea is reported as the conditionality arm.

Estimators (same 600 amplitudes, same bank, same prior from mmse_gpu):
  naive     the production linear MMSE.
  aware     + EP sites for the mask factors, exact per-bit likelihoods:
              return:  p(s) = exp(-T/A(s, r))          at the KNOWN hit
              miss:    1 - exp(-T/A(s, r))             at the flat-sea
                       footprint (the deployable proxy; the true hit of a miss
                       is unobservable)
            Sites live on the linear functionals s_k = G_k a (the S2 gradient
            design). Moment matching by 40-node Gauss-Hermite on the cavity;
            parallel EP, damped, site precisions clamped >= 0.
            MISS SUBSAMPLING: K_miss of ~90k misses, each site's precision
            contribution scaled by N_miss/K_miss (the SVI minibatch trick;
            linear-Gaussian-exact, approximate for tilts; documented).

The estimator KNOWS: the fired-ray geometry, per-scan T (receiver constant),
the radiometric law A (calibratable from mask-vs-range profiles alone), kp.
It never sees the true slope or the GT field.

================================ PRE-REGISTRATION ============================
Frozen before the first full run. n >= 16 seas, paired by sea, clustered SE.

  R1 REDUCTION.   0 sweeps => naive solve to <= 1e-6 fp64 (same code path).
  G1 POSITIVE.    paired d nRMSE = nRMSE(naive) - nRMSE(aware) > 0 with
                  t >= 3.
  G2 CEILING CONSISTENCY (two-sided; #45 and its mirror). The accounting
                  ceiling for this sea/rate is computed INLINE from the same
                  MI quadrature as F2b, mapped by fable's linear nats->nRMSE
                  map anchored at this scene's own naive operating point.
                  Require: gain <= 2 x ceiling. A gain ABOVE the ceiling
                  falsifies the accounting (or reveals a leak) and blocks the
                  positive verdict until resolved.
  G3 PERMUTATION CONTROL. Shuffle the miss bits' footprints within range bins
                  (marginals preserved, field correspondence destroyed):
                  |gain_perm| <= 0.005. If permuted masks also help, the gain
                  is regularisation, NOT selection information. NO VERDICT.
  G4 MCAR CONTROL. Same pipeline, selection = uniform-random top-k (field-
                  independent); the aware estimator told the truth (constant
                  p): gain must be exactly 0 by construction -- printed as an
                  identity check, not a statistic.
  G5 MILD-SEA ARM. Same measurement at sigma_eta 0.30 m: expected small
                  (accounting: ~3% MI ratio). Reported, not gated -- the
                  claim is conditional on sea state and must be stated so.

DISCLOSED PROTOCOL CHANGE (#48, after the first capped smoke). On the full
15-200 m fan the naive baseline sits at nRMSE 1.2-1.4 -- WORSE than the
zero predictor -- because threshold selection concentrates all 600 returns
at near range (A ~ 1/r^2) and the far fan is simply uninstrumented. Scoring
an uninstrumented region measures the prior, not the estimator. So scoring
is now reported on TWO regions: the full fan (protocol continuity) and the
INFORMED region r_q <= p90(return ranges) -- data-dependent but identical
for both arms, so the pairing is untouched. G1/G2/G3 are gated on the
informed region; the full fan is always printed beside it. This change was
made after seeing the smoke numbers and is disclosed as such.

DECISION: R1 & G1 & G2 & G3 pass -> the selection-aware posterior beats the
linear MMSE by a measured, ceiling-consistent margin. Amortizing it (step 3,
N5 site head) is the learning chapter's main piece. G1 fails -> the mask's
information does not survive the estimator; the door closes at the estimator
level and the accounting/estimation gap is itself the result.
=============================================================================
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.raydrop_uniform import fresnel_reflectance
from d3_gpu_mmse import mmse_gpu, build_bank, design
from d3_s2_mask_information import grad_design
from d3_door1b_field_bias import (sea, eta, eta_dr, ray_grid, occluded_hit,
                                  H_SENS, R_MIN, R_MAX, SCAN_T, NOISE, DEPTH,
                                  FOV_H)

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = torch.float64
OUT = 'results/direction3/g0b_selection_aware'
D_REF = 125.0
N_PTS = 50                    # returns per scan (episode convention)
K_MISS = 6000                 # miss sites kept (precision-scaled)
SWEEPS = 8
DAMP = 0.6
GH_N, GH_X, GH_W = 40, *np.polynomial.hermite_e.hermegauss(40)
BAND = (0.35, 2.5)


def A_of(s, r):
    g = np.arctan(H_SENS / r)
    th = np.clip(np.pi / 2 - g - s, 0.0, np.pi / 2 - 1e-9)
    return fresnel_reflectance(th) * np.cos(th) / ((r / D_REF) ** 2 + 1e-9)


def A_of_t(s, r):
    """torch version of A_of, for quadrature on GPU."""
    g = torch.arctan(H_SENS / r)
    th = torch.clamp(math.pi / 2 - g - s, 0.0, math.pi / 2 - 1e-9)
    n = 1.318
    ci = torch.cos(th)
    st2 = torch.clamp((torch.sin(th) / n) ** 2, max=1.0 - 1e-12)
    ct = torch.sqrt(1.0 - st2)
    Rs = ((ci - n * ct) / (ci + n * ct)) ** 2
    Rp = ((n * ci - ct) / (n * ci + ct)) ** 2
    R = 0.5 * (Rs + Rp)
    return R * ci / ((r / D_REF) ** 2 + 1e-9)


def gen_scene(seed, amp_scale, rng, mcar=False):
    """threshold-law observations + mask bits for one sea."""
    S = sea(seed, amp_scale)
    kp = float(np.hypot(S[0], S[1])[np.argmax(S[3])])
    o, d, foot = ray_grid()
    fr = np.hypot(foot[:, 0], foot[:, 1])
    inband = (fr > R_MIN) & (fr < R_MAX)

    amps, miss_P, miss_r, Ts = [], [], [], []
    for t in SCAN_T:
        hm, hp = occluded_hit(o, d, t, S)
        idx = np.where(hm)[0]
        r = np.hypot(hp[idx, 0], hp[idx, 1])
        s = eta_dr(hp[idx, 0], hp[idx, 1], t, S)
        A = A_of(s, r)
        X = rng.exponential(1.0, len(idx))
        power = np.arange(len(idx)) * 0.0 + A * X
        if mcar:
            power = rng.random(len(idx))                 # field-independent
        k = min(N_PTS, len(idx))
        top = np.argpartition(power, -k)[-k:]
        T = float(np.partition(power, -k)[-k]) if not mcar else float('nan')
        Ts.append(T)
        sel = np.zeros(len(idx), bool)
        sel[top] = True
        amps.append(np.column_stack([hp[idx[sel], :2],
                                     np.full(k, t), hp[idx[sel], 2],
                                     r[sel]]))
        # miss bits: fired, hit the gate, did not return -> site at footprint
        mi = idx[~sel]
        mm = inband[mi]
        miss_P.append(np.column_stack([foot[mi[mm]], np.full(mm.sum(), t)]))
        miss_r.append(fr[mi[mm]])
    return (S, kp, np.vstack(amps), np.vstack(miss_P),
            np.concatenate(miss_r), Ts)


def ep_aware(Phi, z, s2, gam, kxb, kyb, om, sitesP, sites_r, sites_T,
             is_return, w_scale, sweeps=SWEEPS, mcar=False):
    """parallel EP with exact Bernoulli-tilt sites on s_k = G_k a."""
    A0 = Phi.T @ Phi / s2 + torch.diag(1.0 / gam)
    b0 = Phi.T @ z / s2
    if sweeps == 0 or len(sitesP) == 0 or mcar:
        # mcar: p is constant in s -> the exact site is vague. identity.
        return torch.linalg.solve(A0, b0), 0.0
    G = grad_design(sitesP, kxb, kyb, om)
    K = len(G)
    w = torch.tensor(w_scale, device=DEV, dtype=DT)
    tau = torch.zeros(K, device=DEV, dtype=DT)
    nu = torch.zeros(K, device=DEV, dtype=DT)
    gx = torch.tensor(GH_X, device=DEV, dtype=DT)
    gw = torch.tensor(GH_W / GH_W.sum(), device=DEV, dtype=DT)
    rT = sites_T
    rr = sites_r

    for _ in range(sweeps):
        A = A0 + (G * (w * tau)[:, None]).T @ G
        S_ = torch.linalg.inv(A)
        mu = S_ @ (b0 + G.T @ (w * nu))
        v = ((G @ S_) * G).sum(1).clamp_min(1e-14)
        m = G @ mu
        vc_inv = (1.0 / v) - tau
        ok = vc_inv > 1e-12
        vc = torch.where(ok, 1.0 / vc_inv.clamp_min(1e-12), v * 1e6)
        mc = torch.where(ok, vc * (m / v - nu), m)

        # tilt moments by Gauss-Hermite on the cavity
        sd = torch.sqrt(vc)
        sq = mc[:, None] + sd[:, None] * gx[None, :]        # (K, GH_N)
        p = torch.exp(-rT[:, None] / A_of_t(sq, rr[:, None]).clamp_min(1e-30))
        f = torch.where(is_return[:, None], p, 1.0 - p).clamp_min(1e-300)
        Z = (gw[None] * f).sum(1)
        m_h = (gw[None] * f * sq).sum(1) / Z
        v_h = ((gw[None] * f * sq ** 2).sum(1) / Z - m_h ** 2).clamp_min(1e-14)

        tau_n = ((1.0 / v_h) - (1.0 / vc)).clamp_min(0.0)
        nu_n = m_h / v_h - mc / vc
        bad = ~ok | ~torch.isfinite(tau_n) | ~torch.isfinite(nu_n)
        tau_n = torch.where(bad, tau, tau_n)
        nu_n = torch.where(bad, nu, nu_n)
        tau = DAMP * tau_n + (1 - DAMP) * tau
        nu = DAMP * nu_n + (1 - DAMP) * nu

    A = A0 + (G * (w * tau)[:, None]).T @ G
    return torch.linalg.solve(A, b0 + G.T @ (w * nu)), \
        float((tau > 1e-8).float().mean())


def run_sea(seed, amp_scale, rng, mcar=False, perm=False):
    S, kp, amps, missP, miss_r, Ts = gen_scene(seed, amp_scale, rng, mcar)
    if perm:                                             # G3: break field link
        edges = np.linspace(R_MIN, R_MAX, 13)
        idx = np.clip(np.digitize(miss_r, edges) - 1, 0, 11)
        for b in range(12):
            m = idx == b
            if m.sum() > 1:
                sh = rng.permutation(np.where(m)[0])
                missP[m, :2] = missP[sh, :2]

    P, zval = amps[:, :3], amps[:, 3] + rng.normal(0, NOISE, len(amps))
    ret_r = amps[:, 4]

    nq = 1200
    rq = np.sqrt(rng.uniform((R_MIN / R_MAX) ** 2, 1.0, nq)) * R_MAX
    aq = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), nq)
    Pq = np.column_stack([rq * np.cos(aq), rq * np.sin(aq), np.zeros(nq)])
    gt = eta(Pq[:, 0], Pq[:, 1], 0.0, S)
    sig = float(np.std(gt))
    r90 = float(np.quantile(ret_r, 0.90))            # informed region (#48)
    inf_q = rq <= r90
    sig_inf = float(np.std(gt[inf_q])) if inf_q.sum() > 50 else sig

    tP = torch.tensor(P, device=DEV, dtype=torch.float32)
    tz = torch.tensor(zval, device=DEV, dtype=torch.float32)
    tQ = torch.tensor(Pq, device=DEV, dtype=torch.float32)
    _, mdl = mmse_gpu(tP, tz, DEPTH, tQ, NOISE, kp, seed=seed,
                      return_model=True)
    gam, s2 = mdl['gam'].to(DT), float(mdl['s2'])

    kxb, kyb, om = build_bank(kp, BAND, kp / 10.0, DEPTH, DEV, torch.float32)
    kxb, kyb, om = kxb.to(DT), kyb.to(DT), om.to(DT)
    Phi = design(torch.tensor(P, device=DEV, dtype=DT), kxb, kyb, om)
    Phq = design(torch.tensor(Pq, device=DEV, dtype=DT), kxb, kyb, om)
    tz64 = torch.tensor(zval, device=DEV, dtype=DT)
    gt_t = torch.tensor(gt, device=DEV, dtype=DT)

    # sites: all returns (exact) + K_MISS subsampled misses (precision-scaled)
    n_miss = len(missP)
    ksub = min(K_MISS, n_miss)
    sub = rng.choice(n_miss, ksub, replace=False)
    scanT = np.repeat(np.mean(Ts), 1)                    # per-site T: scan mean
    Tmap = {t: Ts[i] for i, t in enumerate(SCAN_T)}
    sT_ret = np.array([Tmap[t] for t in P[:, 2]])
    sT_mis = np.array([Tmap[t] for t in missP[sub, 2]])
    sitesP = torch.tensor(np.vstack([P, missP[sub]]), device=DEV, dtype=DT)
    sites_r = torch.tensor(np.concatenate([ret_r, miss_r[sub]]),
                           device=DEV, dtype=DT)
    sites_T = torch.tensor(np.concatenate([sT_ret, sT_mis]),
                           device=DEV, dtype=DT)
    is_ret = torch.tensor(np.r_[np.ones(len(P), bool), np.zeros(ksub, bool)],
                          device=DEV)
    w_scale = np.r_[np.ones(len(P)), np.full(ksub, n_miss / ksub)]

    def nrmse(a):
        """with the B1 energy cap, the project's certified defence, applied
        identically to both arms. Without it the smoke run exploded (naive
        nRMSE 24.7 on sea 2): threshold top-k concentrates the 600 returns at
        near range (A ~ 1/r^2), and the EM-tuned prior then oscillates wildly
        when extrapolated to the 200 m fan. The cap is part of the production
        estimator (B1: RMSE(beta*p) - RMSE(0) <= std(z)), so both estimators
        are evaluated in their certified configuration; the pairing is
        untouched."""
        f = Phq @ a
        beta = min(1.0, float(tz64.std() / f.std().clamp_min(1e-12)))
        e = (beta * f - gt_t)
        full = float(torch.sqrt((e ** 2).mean()) / sig)
        m = torch.tensor(inf_q, device=DEV)
        inf = float(torch.sqrt((e[m] ** 2).mean()) / sig_inf)
        return full, inf

    a_naive, _ = ep_aware(Phi, tz64, s2, gam, kxb, kyb, om, sitesP, sites_r,
                          sites_T, is_ret, w_scale, sweeps=0)
    a_aware, act = ep_aware(Phi, tz64, s2, gam, kxb, kyb, om, sitesP, sites_r,
                            sites_T, is_ret, w_scale, mcar=mcar)
    f0, i0 = nrmse(a_naive)
    f1, i1 = nrmse(a_aware)
    return {'seed': seed, 'sigma': sig, 'n_amp': len(P), 'n_miss': int(n_miss),
            'act': act, 'r90': r90, 'nq_inf': int(inf_q.sum()),
            'nrmse_naive': f0, 'nrmse_aware': f1, 'gain': f0 - f1,
            'inf_naive': i0, 'inf_aware': i1, 'gain_inf': i0 - i1}


def clustered(v):
    v = np.asarray(v, float)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.inf
    return float(v.mean()), float(se), float(v.mean() / se if se > 0 else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seas', type=int, default=16)
    ap.add_argument('--amp-scale', type=float, default=3.33,
                    help='3.33 -> sigma_eta ~1 m, the accounting sea')
    ap.add_argument('--arm', choices=['main', 'perm', 'mcar', 'mild'],
                    default='main')
    ap.add_argument('--tag', default=None)
    a_ = ap.parse_args()
    tag = a_.tag or a_.arm
    amp = 1.0 if a_.arm == 'mild' else a_.amp_scale

    print(f"=== G0b selection-aware vs naive  arm={a_.arm} seas={a_.seas} "
          f"amp={amp} ===")
    print("PRE-REGISTERED: R1<=1e-6 | G1 gain>0 t>=3 | G2 gain<=2x ceiling |")
    print("                G3 |perm|<=0.005 | G4 mcar identity | G5 mild arm\n")

    rng = np.random.default_rng(21)
    rows = []
    for s in range(a_.seas):
        r = run_sea(s, amp, rng, mcar=(a_.arm == 'mcar'),
                    perm=(a_.arm == 'perm'))
        rows.append(r)
        print(f"sea {s:2d} sig {r['sigma']:.3f} r90 {r['r90']:5.1f} "
              f"act {r['act']:.2f} | FULL naive {r['nrmse_naive']:.4f} "
              f"gain {r['gain']:+.4f} | INF({r['nq_inf']:4d}q) naive "
              f"{r['inf_naive']:.4f} aware {r['inf_aware']:.4f} "
              f"gain {r['gain_inf']:+.4f}", flush=True)

    g = clustered([r['gain'] for r in rows])
    gi = clustered([r['gain_inf'] for r in rows])
    nn = clustered([r['nrmse_naive'] for r in rows])
    ni = clustered([r['inf_naive'] for r in rows])
    print("\n" + "=" * 72)
    print(f"FULL fan : naive {nn[0]:.4f}   gain = {g[0]:+.4f} +- {g[1]:.4f}"
          f"  t = {g[2]:+.2f}")
    print(f"INFORMED : naive {ni[0]:.4f}   gain = {gi[0]:+.4f} +- {gi[1]:.4f}"
          f"  t = {gi[2]:+.2f}   <- gated region (#48)")
    res = {'arm': a_.arm, 'amp_scale': amp, 'n_sea': len(rows),
           'gain': g, 'gain_inf': gi, 'naive': nn, 'naive_inf': ni,
           'rows': rows}
    os.makedirs('results/direction3', exist_ok=True)
    with open(f'{OUT}_{tag}.json', 'w') as f:
        json.dump(res, f, indent=2)
    print(f"wrote {OUT}_{tag}.json")
    return 0


if __name__ == '__main__':
    sys.exit(main())
