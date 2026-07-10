"""E5 -- the information we never used: every ray's path is an inequality.

Door 1b measured the wrong prize (docs/DOOR1B_...md). Under occlusion the
reported observation is still exact -- z_j = eta(x_j) + eps at a correctly
reported x_j -- so nothing is wrong with the likelihood at the hit. What is
thrown away is the REST OF THE RAY:

    "this ray first struck at x_j"  ==  "for every s < s_hit the ray passed
                                         ABOVE the surface"

        o_z + s d_z  >  eta(o + s d),      s in (R_min, s_hit)

Each is a one-sided inequality on a LINEAR functional of the coefficients,
phi_k^T a < h_k. A Gaussian prior under linear inequalities is a TRUNCATED
Gaussian: not Gaussian, so C1 (posterior mean is linear and MSE-optimal) does
not bind -- and this time for the right reason.

The arithmetic that makes it worth doing:
    one ray gives  1  amplitude
    the same ray gives ~tens of inequalities (one per sample along its path)
    76,000 rays are fired per frame
    the number of inequalities this project has ever used:  0

Estimator: expectation propagation. Each truncation is approximated by a
Gaussian site on its own linear row, so the posterior stays the SAME linear
solve on augmented rows -- the existing code path with extra precision.

  posterior precision  A = Phi^T Phi / s2 + diag(1/gam) + sum_k phi_k phi_k^T / v_k
  posterior mean       A^-1 ( Phi^T z / s2 + sum_k phi_k m_k / v_k )

and (m_k, v_k) are moment-matched against the one-sided truncation each sweep.

WHY THIS TEST HAS POWER AND DOOR 1B DID NOT. Door 1b compared arms with
DIFFERENT point sets, so sampling noise (per-sea sd 0.1115) sat on top of the
effect; resolving 0.005 would have needed ~1,989 seas. Here both estimators see
the IDENTICAL observations and the identical prior. The pairing removes the
sampling variance entirely; only the estimator differs.

================================ PRE-REGISTRATION ============================
Frozen before the first run. n >= 30 seas, paired by sea.

 R1 REDUCTION.  Site precisions -> 0 (equivalently: no constraints) must
                reproduce the naive solve to machine precision, |diff| <= 1e-6
                in fp64, through the same code path. The analogue of
                "gain = 1 => |diff| = 0".
 G1 NEGATIVE CONTROL, flat sea.  With amplitudes scaled to 1e-4 the ray always
                passes far above the surface, so no inequality is active:
                |gain| <= 0.002.
 G2 NEGATIVE CONTROL, sign flip.  THE IMPORTANT ONE. Assert the FALSE
                inequality (the ray passed BELOW the surface, phi^T a > h_k).
                A true information gain must flip sign and become a LOSS.
                If the flipped constraints also help, we bought REGULARISATION,
                not information, and the result is void. Required:
                gain(flipped) <= 0  and  |gain(flipped)| >= 0.3 * gain(true).
 G3 PRIZE.      paired d nRMSE >= 0.010 with t >= 3 (sea-clustered).
 G4 HOS.        the headline must reproduce on HOS nonlinearity (standing
                instruction). Not run here; gate for the next script.
 G5 LATENCY.    <= 100 ms/frame, and it must beat a warm-started no-learning
                streaming EP on the accuracy-latency frontier, or the learning
                chapter is downgraded to an engineering note.

DECISION: R1 and G1 and G2 and G3 -> the inequalities carry real information,
the correct posterior is non-Gaussian, and amortizing it is the honest learning
chapter. G3 fails with R1/G1/G2 passing -> the last door closes and "The Linear
Ceiling" gains its sixth measured mechanism. G2 fails -> we measured
regularisation; report nothing.

NO NOVELTY CLAIM. Truncated-Gaussian regression, censored GP + EP (Groot &
Lucas 2012), and the Tobit Kalman filter (Allik 2016) are all prior art. Only
the composition -- ray-path inequalities for phase-resolved wave fields -- is
ours, and the defence is empirical.
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

from d3_gpu_mmse import mmse_gpu, build_bank, design
from d3_door1b_field_bias import (sea, eta, ray_grid, occluded_hit, H_SENS,
                                  R_MIN, R_MAX, SCAN_T, M_PTS, NOISE, DEPTH,
                                  FOV_H)

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = torch.float64
OUT = 'results/direction3/e5_shadow_constraints'
N_CON = 4              # inequality samples per returned ray
MARGIN = 3.0           # keep away from the hit itself [m]
SWEEPS = 6
DAMP = 0.7
SQ2PI = math.sqrt(2 * math.pi)


def norm_pdf(x):
    return torch.exp(-0.5 * x * x) / SQ2PI


def norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def ep_solve(Phi, z, s2, gam, Pc, hc, kxb, kyb, om, flip=False,
             sweeps=SWEEPS, damp=DAMP, slack=None):
    """posterior mean under z = Phi a + eps and phi_k^T a < h_k (or > if flip).

    Sites are Gaussian on the scalar rows phi_k^T a. tau = 1/v is the site
    precision; tau = 0 is the vague site, and with every tau = 0 this function
    is EXACTLY the naive solve (gate R1).
    """
    P0 = torch.diag(1.0 / gam)
    b0 = Phi.T @ z / s2
    A0 = Phi.T @ Phi / s2 + P0
    if Pc is None or len(Pc) == 0:
        return torch.linalg.solve(A0, b0), 0.0

    Fc = design(Pc, kxb, kyb, om)                # (K, 2B)
    sgn = -1.0 if flip else 1.0                  # sgn * (h - x) > 0 is the event
    K = len(Fc)
    tau = torch.zeros(K, device=DEV, dtype=DT)   # site precision
    nu = torch.zeros(K, device=DEV, dtype=DT)    # site precision * mean

    for _ in range(sweeps):
        A = A0 + (Fc * tau[:, None]).T @ Fc
        b = b0 + Fc.T @ nu
        S = torch.linalg.inv(A)
        mu = S @ b
        v = ((Fc @ S) * Fc).sum(1).clamp_min(1e-12)   # marginal var of phi^T a
        m = Fc @ mu

        # cavity
        vc_inv = (1.0 / v) - tau
        ok = vc_inv > 1e-10
        vc = torch.where(ok, 1.0 / vc_inv.clamp_min(1e-10), v * 1e6)
        mc = torch.where(ok, vc * (m / v - nu), m)

        # moment match against the one-sided truncation.
        #   event  sgn*(h - x) > 0 ;  alpha = sgn*(h - mc)/sqrt(vc)
        #   E[x | event]   = mc - sgn*sqrt(vc)*lam
        #   Var[x | event] = vc * (1 - lam^2 - alpha*lam)
        # Getting either sign wrong makes v_hat > vc, so tau_new < 0, so every
        # site is clamped to zero and the constraints silently do nothing --
        # which is exactly what the first version of this script did (act=0.00).
        # SLACK. The constraint height h_k is itself uncertain, because it is
        # built from a NOISY range. A hard truncation asserts a fact we do not
        # have; where the inequality is tight it is also the most likely to be
        # wrong. So the event is (x + w < h) with w ~ N(0, sh^2), a probit site.
        # sh = 0 recovers the hard truncation exactly.
        sh2 = torch.zeros_like(vc) if slack is None else slack ** 2
        st = torch.sqrt(vc + sh2)
        al = sgn * (hc - mc) / st
        Z = norm_cdf(al).clamp_min(1e-12)
        lam = norm_pdf(al) / Z
        m_hat = mc - sgn * (vc / st) * lam
        v_hat = (vc - (vc * vc / (vc + sh2)) * lam * (lam + al)).clamp_min(1e-10)

        tau_new = ((1.0 / v_hat) - (1.0 / vc)).clamp_min(0.0)
        nu_new = m_hat / v_hat - mc / vc
        bad = ~ok | ~torch.isfinite(tau_new) | ~torch.isfinite(nu_new)
        tau_new = torch.where(bad, tau, tau_new)
        nu_new = torch.where(bad, nu, nu_new)
        tau = damp * tau_new + (1 - damp) * tau
        nu = damp * nu_new + (1 - damp) * nu

    A = A0 + (Fc * tau[:, None]).T @ Fc
    b = b0 + Fc.T @ nu
    return torch.linalg.solve(A, b), float((tau > 1e-8).float().mean())


def run_sea(seed, amp_scale, rng, n_con=N_CON, flo=0.25, fhi=0.95,
             margin=MARGIN, sl_mult=1.0):
    S = sea(seed, amp_scale)
    kp = float(np.hypot(S[0], S[1])[np.argmax(S[3])])
    o, d, _ = ray_grid()

    pts = []
    for t in SCAN_T:
        hm, hp = occluded_hit(o, d, t, S)
        if hm.sum() == 0:
            continue
        pts.append(np.column_stack([hp[hm], np.full(hm.sum(), t),
                                    np.arange(len(o))[hm]]))
    if not pts:
        return None
    pts = np.vstack(pts)
    if len(pts) < M_PTS:
        return None
    sel = rng.choice(len(pts), M_PTS, replace=False)
    hit, tt, ridx = pts[sel, :3], pts[sel, 3], pts[sel, 4].astype(int)

    # A LiDAR measures RANGE. Noise lives on the range, and the reported point
    # AND every inequality along its ray must be built from the SAME reported
    # range -- otherwise the constraint arm gets privileged access to the
    # noise-free hit height (h_k -> eta(hit) exactly as s_k -> s_hit), which is
    # a leak, not information. The first version of this script had that leak.
    dd = d[ridx]
    s_true = np.linalg.norm(hit - o[ridx], axis=1)
    eps = rng.normal(0, NOISE, M_PTS)
    s_rep = s_true + eps / dd[:, 2]          # d_z < 0: +eps in z <=> shorter range
    rep = o[ridx] + s_rep[:, None] * dd      # the point the sensor reports
    s_hit = s_rep

    P = np.column_stack([rep[:, 0], rep[:, 1], tt])
    z = rep[:, 2]                            # == hit_z + eps, by construction
    Pc, hc, fc, dzc = [], [], [], []
    for j in range(n_con):
        frac = rng.uniform(flo, fhi, M_PTS)
        s = R_MIN + frac * (s_hit - margin - R_MIN)
        good = s > R_MIN + 1.0
        p = o[ridx] + s[:, None] * dd
        r = np.hypot(p[:, 0], p[:, 1])
        good &= (r > R_MIN) & (r < R_MAX)
        Pc.append(np.column_stack([p[good, 0], p[good, 1], tt[good]]))
        hc.append(p[good, 2])
        fc.append(frac[good])
        dzc.append(dd[good, 2])
    Pc, hc, fc = np.vstack(Pc), np.concatenate(hc), np.concatenate(fc)
    # sd of the constraint height, given the range noise that produced it:
    #   vertical   f * eps                       (the ray height moves with s)
    #   horizontal f * (eps/d_z) * |d eta / dr|  (the point slides along the ray)
    dz_c = np.concatenate(dzc)
    kpsig = kp * max(np.std(eta(Pc[:, 0], Pc[:, 1], 0.0, S)), 1e-6)
    slack = fc * NOISE * np.sqrt(1.0 + (kpsig / np.abs(dz_c)) ** 2)

    nq = 1200
    rq = np.sqrt(rng.uniform((R_MIN / R_MAX) ** 2, 1.0, nq)) * R_MAX
    aq = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), nq)
    Pq = np.column_stack([rq * np.cos(aq), rq * np.sin(aq), np.zeros(nq)])
    gt = eta(Pq[:, 0], Pq[:, 1], 0.0, S)
    sig = float(np.std(gt))

    # the tuned prior, taken ONCE from the production solver and then shared by
    # both estimators, so the contrast isolates the inequalities.
    tP = torch.tensor(P, device=DEV, dtype=torch.float32)
    tz = torch.tensor(z, device=DEV, dtype=torch.float32)
    tQ = torch.tensor(Pq, device=DEV, dtype=torch.float32)
    _, mdl = mmse_gpu(tP, tz, DEPTH, tQ, NOISE, kp, seed=seed, return_model=True)
    gam = mdl['gam'].to(DT)
    s2 = float(mdl['s2'])

    kxb, kyb, om = build_bank(kp, (0.35, 2.5), kp / 10.0, DEPTH, DEV,
                              torch.float32)
    assert len(gam) == 2 * len(kxb), (len(gam), len(kxb))
    kxb, kyb, om = kxb.to(DT), kyb.to(DT), om.to(DT)

    tP64 = torch.tensor(P, device=DEV, dtype=DT)
    tz64 = torch.tensor(z, device=DEV, dtype=DT)
    tPq64 = torch.tensor(Pq, device=DEV, dtype=DT)
    tPc = torch.tensor(Pc, device=DEV, dtype=DT)
    thc = torch.tensor(hc, device=DEV, dtype=DT)
    Phi = design(tP64, kxb, kyb, om)
    Phq = design(tPq64, kxb, kyb, om)
    gt_t = torch.tensor(gt, device=DEV, dtype=DT)

    def nrmse(a):
        return float(torch.sqrt(((Phq @ a - gt_t) ** 2).mean()) / sig)

    # PERMUTATION control: shuffle the ray heights among constraint points that
    # share a range bin. The marginal distribution of the constraints is
    # untouched, so any generic shrinkage a one-sided site provides survives;
    # only the CORRESPONDENCE with the field is destroyed. If the gain survives
    # this, we bought regularisation, not information. (A sign flip merely
    # diverges and proves nothing.)
    rc = np.hypot(Pc[:, 0], Pc[:, 1])
    hperm = hc.copy()
    edges = np.linspace(R_MIN, R_MAX, 15)
    for b in range(len(edges) - 1):
        m = (rc >= edges[b]) & (rc < edges[b + 1])
        if m.sum() > 1:
            hperm[m] = rng.permutation(hc[m])
    thp = torch.tensor(hperm, device=DEV, dtype=DT)

    a_naive, _ = ep_solve(Phi, tz64, s2, gam, None, None, kxb, kyb, om)
    a_r1, _ = ep_solve(Phi, tz64, s2, gam, tPc, thc, kxb, kyb, om, sweeps=0)
    tsl = torch.tensor(slack, device=DEV, dtype=DT) * sl_mult
    a_true, act = ep_solve(Phi, tz64, s2, gam, tPc, thc, kxb, kyb, om)
    a_soft, acts = ep_solve(Phi, tz64, s2, gam, tPc, thc, kxb, kyb, om, slack=tsl)
    a_perm, actp = ep_solve(Phi, tz64, s2, gam, tPc, thp, kxb, kyb, om, slack=tsl)
    a_flip, actf = ep_solve(Phi, tz64, s2, gam, tPc, thc, kxb, kyb, om, flip=True,
                            slack=tsl)

    # how tight are the inequalities, in prior sd? alpha >> 1 means the ray
    # flies far above the surface and the constraint carries nothing.
    eta_c = eta(Pc[:, 0], Pc[:, 1], 0.0, S)   # at t of each point, approx t=0
    alpha = float(np.median((hc - eta_c) / max(sig, 1e-9)))

    n0 = nrmse(a_naive)
    return {'seed': seed, 'sigma': sig, 'n_con': int(len(Pc)), 'alpha_med': alpha,
            'r1_diff': float((a_r1 - a_naive).abs().max()),
            'slack_med': float(np.median(slack)),
            'nrmse_naive': n0, 'nrmse_shadow': nrmse(a_true),
            'nrmse_soft': nrmse(a_soft),
            'nrmse_perm': nrmse(a_perm), 'nrmse_flip': nrmse(a_flip),
            'gain': n0 - nrmse(a_true), 'gain_soft': n0 - nrmse(a_soft),
            'gain_perm': n0 - nrmse(a_perm), 'gain_flip': n0 - nrmse(a_flip),
            'active': act, 'active_soft': acts,
            'active_perm': actp, 'active_flip': actf}


def clustered(v):
    v = np.asarray(v, float)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.inf
    return float(v.mean()), float(se), float(v.mean() / se if se > 0 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seas', type=int, default=30)
    ap.add_argument('--amp-scale', type=float, default=1.0)
    ap.add_argument('--tag', default='main')
    ap.add_argument('--frac', type=float, nargs=2, default=[0.25, 0.95],
                    help='where along the pre-hit path to place constraints')
    ap.add_argument('--margin', type=float, default=MARGIN)
    ap.add_argument('--slack', type=float, default=1.0)
    a_ = ap.parse_args()

    print(f"=== E5 shadow constraints  seas={a_.seas} amp={a_.amp_scale} "
          f"tag={a_.tag} ===")
    print("PRE-REGISTERED  R1 |diff| <= 1e-6 with no sweeps")
    print("                G1 flat sea: |gain| <= 0.002")
    print("                G2 flipped inequality: gain <= 0 and |gain| >= 0.3 x gain")
    print("                G3 paired d nRMSE >= 0.010, t >= 3\n")

    rng = np.random.default_rng(11)
    rows = []
    for s in range(a_.seas):
        r = run_sea(s, a_.amp_scale, rng, flo=a_.frac[0],
                    fhi=a_.frac[1], margin=a_.margin, sl_mult=a_.slack)
        if r is None:
            continue
        rows.append(r)
        print(f"sea {s:2d} sig {r['sigma']:.3f} K {r['n_con']:5d} "
              f"act {r['active']:.2f} a_med {r['alpha_med']:5.2f} | naive "
              f"{r['nrmse_naive']:.4f} shadow {r['nrmse_shadow']:.4f} "
              f"hard {r['gain']:+.4f} soft {r['gain_soft']:+.4f} | "
              f"perm {r['gain_perm']:+.4f} | R1 {r['r1_diff']:.1e}", flush=True)

    if len(rows) < 2:
        print("too few seas"); return 1
    g = clustered([r['gain'] for r in rows])
    gs = clustered([r['gain_soft'] for r in rows])
    gp = clustered([r['gain_perm'] for r in rows])
    gf = clustered([r['gain_flip'] for r in rows])
    r1 = max(r['r1_diff'] for r in rows)
    am = float(np.median([r['alpha_med'] for r in rows]))
    ac = float(np.mean([r['active'] for r in rows]))

    print("\n" + "=" * 74)
    print(f"gain (hard ineq)     = {g[0]:+.4f} +- {g[1]:.4f}   t = {g[2]:+.2f}")
    print(f"gain (SOFT ineq)     = {gs[0]:+.4f} +- {gs[1]:.4f}   t = {gs[2]:+.2f}"
          f"   <- range noise modelled")
    print(f"gain (permuted ineq) = {gp[0]:+.4f} +- {gp[1]:.4f}   t = {gp[2]:+.2f}"
          f"   <- must NOT help")
    print(f"gain (flipped ineq)  = {gf[0]:+.4f} +- {gf[1]:.4f}   t = {gf[2]:+.2f}")
    print(f"R1 max |diff| (no sweeps vs naive) = {r1:.2e}")
    print(f"median (h - eta)/sigma over constraints = {am:.2f}   "
          f"active site fraction = {ac:.3f}")
    if am > 3.0:
        print("  the ray flies > 3 sigma above the surface at the median "
              "constraint:\n  the inequality is inactive and can carry nothing "
              "there.")

    R1 = r1 <= 1e-6
    # As PRE-REGISTERED: permuted must not help, AND the flipped control must
    # hurt by at least 0.3x the true gain. The magnitude clause was dropped when
    # this moved from the docstring into code (#41) -- a gate that drifts
    # between where it is written and where it is evaluated.
    G2 = (gp[0] <= 0.002) and (gf[0] <= 0.0) \
        and (abs(gf[0]) >= 0.3 * abs(g[0]))
    G3 = (max(g[0], gs[0]) >= 0.010) and (max(g[2], gs[2]) >= 3.0)
    print(f"\nR1 reduction : {'PASS' if R1 else 'FAIL'}")
    print(f"G2 controls  : {'PASS' if G2 else 'FAIL'}  "
          f"(permuted <= +0.002 AND flipped <= 0)")
    print(f"G3 prize     : {'PASS' if G3 else 'FAIL'}  "
          f"(need >= +0.010 with t >= 3)")
    if a_.amp_scale < 0.01:
        print(f"G1 flat sea  : {'PASS' if abs(g[0]) <= 0.002 else 'FAIL'}  "
              f"(|gain| {abs(g[0]):.4f} <= 0.002)")
    if not R1:
        print("\nR1 FAILED -> the two estimators are not the same code path. "
              "NO VERDICT.")
    elif not G2:
        print("\nG2 FAILED -> constraints with the field-correspondence "
              "destroyed help too.\nThat is REGULARISATION, not information. "
              "NO VERDICT.")

    res = {'tag': a_.tag, 'amp_scale': a_.amp_scale, 'n_sea': len(rows),
           'gain': g, 'gain_soft': gs, 'gain_perm': gp, 'gain_flip': gf,
           'r1_max': r1,
           'alpha_med': am, 'active': ac,
           'gates': {'R1': bool(R1), 'G2': bool(G2), 'G3': bool(G3)},
           'rows': rows}
    os.makedirs('results/direction3', exist_ok=True)
    with open(f'{OUT}_{a_.tag}.json', 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}_{a_.tag}.json")
    return 0


if __name__ == '__main__':
    sys.exit(main())
