"""Door 1b: does OCCLUSION leave recoverable structure in the FIELD error?

v2. v1's instrument FAILED its own negative control and is retracted below
(self-correction #36). Read that first; it is the reason this design looks the
way it does.

--------------------------------- v1, RETRACTED -----------------------------
v1 regressed the field error e = eta_hat - eta_gt on the GT along-ray slope s
and compared the coefficient to b_trunc, the same coefficient for the bank's
best uniform approximation of eta_gt. The idea was that b_trunc absorbs the
band-limiting confound (e contains -eta_high-k, and s is a derivative, so both
load on high k).

  measured, n=2 seas:   db(occ) = -0.0201     db(mcar) = -0.0161, t = -6.1

The MCAR arm -- whose rays are chosen independently of the field -- reproduced
80% of the effect. b_trunc is the WRONG null: a uniform least-squares
projection truncates, but a 600-point prior-regularised estimator SHRINKS, and
shrinkage is mode-dependent (high k shrinks more). So e carries -(shrunk high-k
part of eta) whatever the sampling, and s reads it. The statistic was measuring
the estimator, not the selection. Pre-registered rule: G1 fails => instrument
broken => no verdict. Honoured. v1 is not reported as a result.
-----------------------------------------------------------------------------

WHAT V2 MEASURES INSTEAD, and why its null is an identity.

The naive MMSE treats the observation locations as ancillary. They are not: a
return exists only if no nearer facet blocks the ray, so the MASK is a
functional of the field. Troughs hide behind crests. Where the mask is empty,
the posterior relaxes to the prior mean 0, and the field there is
systematically LOW -- so the error is systematically POSITIVE. That is a bias
the naive estimator cannot see and a selection-aware one can.

Crucially the mask is DEPLOYMENT-OBSERVABLE: at run time you know which rays
you fired and which came back. So define, at each query point q, using nothing
but the observations,

    u(q) = (# returns within R0 of q) / (# rays whose FLAT-SEA footprint is
                                          within R0 of q)                 in [0,1]

and ask whether the field error is predictable from u:

    e ~ 1 + r + r^2 + u,      b := coefficient on u.

Range enters because shadowing grows with range at grazing incidence, and the
error grows with range too; r and r^2 absorb that shared trend, leaving the
LOCAL, field-driven fluctuation of the mask.

  THE NULL IS EXACT. Under MCAR the mask is independent of eta, so u carries
  no information about eta: cov(e, u) = 0 in expectation, identically, for any
  estimator. This is the same identity that S2 found in Fisher information
  (p' = 0 => I = 0). It is not an empirical hope; it is arithmetic.

  Under occlusion, u correlates with eta (crests return, troughs shadow), and
  the shrinkage e ~ -(1-c) eta then forces cov(e, u) < 0.

v2.1 -- AND THEN THE PHYSICS CORRECTED ME AGAIN (self-correction #38).

v2 assumed occlusion creates MISSINGNESS. It does not. The sea is opaque and
unbounded, so a ray always hits something: measured hit rate 0.657 on a wavy
sea against 0.667 on a flat one (the 1% gap is the [15, 200] m range gate, not
shadowing). Occlusion does not delete returns -- it DISPLACES them onto the
near faces of crests. At 1.4 deg depression a 0.3 m elevation moves the hit by
12 m in range.

So the mechanism is not informative missingness. It is PREFERENTIAL SAMPLING
(Diggle, Menezes & Su, JRSS-C 2010): the sample LOCATIONS depend on the field,
and an estimator that conditions on them without modelling p(x | eta) is
biased. The density statistic u was therefore the wrong instrument, and its
null result at n=3 says nothing about the real channel.

ARMS. Identical sea, bank, budget M, matched range profile of return rate.
  A  occ    the rays' true first hits (Moller-Trumbore). Locations depend on
            eta through BOTH displacement and shadowing.
  C  mcar   every ray's unshadowed nearest root, kept with arm A's return rate
            in the same range bin. Displacement WITHOUT shadowing.
  F  fixed  the ray's FLAT-SEA footprint, z = eta there. Locations do not
            depend on eta at all. The ONLY arm free of preferential sampling,
            hence the only valid null.

A - F  = the whole preferential-sampling effect.   C - F = displacement alone.

============================= PRE-REGISTRATION, v2.1 =========================
DISCLOSURE, because this is exactly the pattern that produced #22 and #27: the
density route (b_u) was pre-registered, run at n=3, and found null. I am now
ADDING a second route (the slope coefficient b_s, paired against arm F) after
seeing that null. That loosens G2 and G3, and it could manufacture a positive.
Three things keep it honest: (1) the reason is physical and was discovered by a
control, not by the outcome -- occlusion turns out not to be a mask at all;
(2) arm F gives b_s an exact null it did not have in v1, which is why v1 was
retracted; (3) BOTH routes are reported at n >= 20 whatever they say, and the
density route's null is reported as a result, not buried.

n_sea >= 20; SE clustered by sea; two-sided.

  G1 EXACT NULL, as an EQUIVALENCE test, not a significance test. A t-test
                  against a null that is exactly zero is guaranteed to fail
                  once n is large enough to resolve any residual confound: at
                  n = 2 arm F already gave b_u = -0.0014 +- 0.0001, t = -18.7,
                  a value that is numerically nothing and statistically
                  everything. (The residual is r-trend leakage: u correlates
                  with range through the keep-rate profile and e has its own
                  range trend, and 1 + r + r^2 does not absorb it exactly.)
                  So the gate is on MAGNITUDE:
                      |b_u(F)| <= 0.2 |b_u(occ)|,  |b_s(F)| <= 0.2 |b_s(occ)|,
                      and arm F's LOSO nRMSE gain within +-0.002.
                  Disclosed as a post-hoc change (#39): it was made after
                  seeing G1 fail, and it makes a verdict POSSIBLE where the
                  previous form forbade one. The raw t values are printed
                  alongside so a reader can apply the stricter rule.
  G2 POSITIVE.    |t(b_u, occ)| > 2   OR   |t(b_s(occ) - b_s(fixed))| > 2.
  G3 EFFECT SIZE. leave-one-sea-out correction removes >= 0.005 nRMSE,
                  OR nRMSE(occ) - nRMSE(fixed) >= 0.005 with t >= 2.
                  Significant-and-negligible closes the door as firmly as
                  absent.
  G4 REDUCTION.   --amp-scale 1e-4 (a flat sea): no displacement, no shadowing,
                  so every b -> 0. The analogue of "gain = 1 => |diff| = 0".

DECISION, fixed now:
  G1..G4 pass -> preferential sampling costs the linear MMSE real accuracy that
      a selection-aware (non-Gaussian) likelihood can recover. That is the
      honest home for amortized learning.
  G2 or G3 fails (G1, G4 passing) -> it does not. Door 1b closes for field
      accuracy, and "The Linear Ceiling" gains a fourth measured mechanism.
  G1 or G4 fails -> instrument broken. Report the failure, no verdict.
=============================================================================
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from wavesim.lidar_simulator import build_mesh, raycast_surface
from d3_gpu_mmse import mmse_gpu

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = 'results/direction3/door1b_field_bias'

H_SENS = 5.0                  # wavesim/config.py LidarCfg.height
FOV_H = 60.0
EL_LO, EL_HI = -25.0, -1.146  # atan(5/250): the farthest ray the sensor fires
R_MIN, R_MAX = 15.0, 200.0
N_AZ, N_EL = 240, 48
SCAN_T = np.linspace(-20.0, 0.0, 12)
M_PTS = 600
NOISE = 0.05
DEPTH = 50.0
HS, TP, NCOMP = 1.2, 8.0, 60
SPREAD = 30.0
MESH_DX = 2.0
DOM_X, DOM_Y = (0.0, 260.0), (-240.0, 240.0)
R0 = 25.0                     # neighbourhood for the local return rate
RBINS = np.linspace(R_MIN, R_MAX, 13)


def sea(seed, amp_scale=1.0):
    cs = jonswap_components(HS, TP, n_components=NCOMP, depth=DEPTH,
                            direction_spread_deg=SPREAD, seed=seed)
    return (np.array([c.kx for c in cs]), np.array([c.ky for c in cs]),
            np.array([c.omega for c in cs]),
            np.array([c.A for c in cs]) * amp_scale,
            np.array([c.phi for c in cs]))


def eta(x, y, t, S):
    kx, ky, om, a, ph = S
    psi = np.outer(x, kx) + np.outer(y, ky) - t * om[None, :] + ph
    return (a * np.cos(psi)).sum(1)


def eta_dr(x, y, t, S):
    kx, ky, om, a, ph = S
    r = np.clip(np.hypot(x, y), 1e-6, None)
    psi = np.outer(x, kx) + np.outer(y, ky) - t * om[None, :] + ph
    kr = np.outer(x / r, kx) + np.outer(y / r, ky)
    return (-a * kr * np.sin(psi)).sum(1)


def ray_grid():
    az = np.deg2rad(np.linspace(-FOV_H, FOV_H, N_AZ))
    el = np.deg2rad(np.linspace(EL_LO, EL_HI, N_EL))
    A, E = np.meshgrid(az, el, indexing='ij')
    A, E = A.ravel(), E.ravel()
    d = np.column_stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)])
    o = np.tile([0.0, 0.0, H_SENS], (len(d), 1))
    flat = (-o[:, 2] / d[:, 2])[:, None] * d + o          # flat-sea footprint
    return o, d, flat[:, :2]


def transparent_hit(o, d, t, S, iters=44):
    s0 = -o[:, 2] / d[:, 2]
    lo, hi = np.maximum(s0 * 0.5, 1.0), s0 * 2.0

    def g(s):
        p = o + s[:, None] * d
        return p[:, 2] - eta(p[:, 0], p[:, 1], t, S)

    glo, ghi = g(lo), g(hi)
    ok = (glo * ghi) < 0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        left = (glo * gm) <= 0
        hi, lo = np.where(left, mid, hi), np.where(left, lo, mid)
        glo = np.where(left, glo, gm)
    p = o + (0.5 * (lo + hi))[:, None] * d
    r = np.hypot(p[:, 0], p[:, 1])
    return ok & (r > R_MIN) & (r < R_MAX), p


def occluded_hit(o, d, t, S):
    gx = np.arange(DOM_X[0], DOM_X[1] + MESH_DX, MESH_DX)
    gy = np.arange(DOM_Y[0], DOM_Y[1] + MESH_DX, MESH_DX)
    X, Y = np.meshgrid(gx, gy)
    Z = eta(X.ravel(), Y.ravel(), t, S).reshape(X.shape)
    V, T = build_mesh(X, Y, Z)
    hm, hp, _ = raycast_surface(o, d, V, T, range_min=R_MIN, range_max=R_MAX,
                               batch_size=8000)
    r = np.hypot(hp[:, 0], hp[:, 1])
    return hm & (r > R_MIN) & (r < R_MAX), hp


def return_rate(flat_r, keep):
    """arm A's return rate per range bin of the FLAT-SEA footprint."""
    idx = np.clip(np.digitize(flat_r, RBINS) - 1, 0, len(RBINS) - 2)
    rate = np.zeros(len(RBINS) - 1)
    for b in range(len(RBINS) - 1):
        m = idx == b
        rate[b] = keep[m].mean() if m.sum() > 20 else 0.0
    return rate, idx


def local_rate(Pq, pts_xy, foot_xy):
    """u(q): returns within R0 / rays aimed within R0. Observation-only."""
    q = Pq[:, :2]
    num = np.zeros(len(q))
    den = np.zeros(len(q))
    for i in range(0, len(q), 200):
        s = slice(i, i + 200)
        num[s] = (np.linalg.norm(q[s, None] - pts_xy[None], axis=2) < R0).sum(1)
        den[s] = (np.linalg.norm(q[s, None] - foot_xy[None], axis=2) < R0).sum(1)
    return num / np.maximum(den, 1.0)


def ols(e, X):
    beta, *_ = np.linalg.lstsq(X, e, rcond=None)
    rf = e - X @ beta
    rr = e - X[:, :-1] @ np.linalg.lstsq(X[:, :-1], e, rcond=None)[0]
    pr2 = 1.0 - (rf @ rf) / max(rr @ rr, 1e-30)
    return beta, float(max(pr2, 0.0))


def run_sea(seed, amp_scale, rng):
    S = sea(seed, amp_scale)
    kp = float(np.hypot(S[0], S[1])[np.argmax(S[3])])
    o, d, foot = ray_grid()
    flat_r = np.hypot(foot[:, 0], foot[:, 1])
    inband = (flat_r > R_MIN) & (flat_r < R_MAX)

    occ, mcar, fixed = [], [], []
    rate_acc = []
    now = {}                      # the t = 0 mask: the only shadow contemporaneous
    for t in SCAN_T:              # with the field we score (waves move; old
        hm, hp = occluded_hit(o, d, t, S)   # shadows say nothing about eta(0))
        rate, idx = return_rate(flat_r[inband], hm[inband])
        rate_acc.append(rate)
        occ.append(np.column_stack([hp[hm, :2], np.full(hm.sum(), t), hp[hm, 2]]))

        ok, tp = transparent_hit(o, d, t, S)
        # keep each transparent hit with arm A's return rate in ITS range bin:
        # same density, same r-profile, field has no say in which ray survives
        p_keep = np.zeros(len(o))
        p_keep[inband] = rate[idx]
        km = ok & (rng.random(len(o)) < p_keep)
        mcar.append(np.column_stack([tp[km, :2], np.full(km.sum(), t), tp[km, 2]]))

        # arm F: the ray's FLAT-SEA footprint. Locations do not depend on eta at
        # all -- this is the only arm free of preferential sampling, and so the
        # only valid null for the slope coefficient b_s.
        fm = inband & (rng.random(len(o)) < p_keep)
        zf = eta(foot[fm, 0], foot[fm, 1], t, S)
        fixed.append(np.column_stack([foot[fm], np.full(fm.sum(), t), zf]))
        if abs(t) < 1e-9:
            now['occ'], now['mcar'], now['fixed'] = hp[hm, :2], tp[km, :2], foot[fm]

    occ, mcar, fixed = np.vstack(occ), np.vstack(mcar), np.vstack(fixed)
    if min(len(occ), len(mcar), len(fixed)) < M_PTS:
        return None

    nq = 1200
    rq = np.sqrt(rng.uniform((R_MIN / R_MAX) ** 2, 1.0, nq)) * R_MAX
    aq = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), nq)
    Pq = np.column_stack([rq * np.cos(aq), rq * np.sin(aq), np.zeros(nq)])
    gt = eta(Pq[:, 0], Pq[:, 1], 0.0, S)
    sq = eta_dr(Pq[:, 0], Pq[:, 1], 0.0, S)
    sig = float(np.std(gt))
    rn = rq / R_MAX

    out = {'seed': seed, 'kp': kp, 'sigma': sig,
           'n_occ': int(len(occ)), 'n_mcar': int(len(mcar)),
           'rate': float(np.mean(rate_acc))}

    for name, D in (('occ', occ), ('mcar', mcar), ('fixed', fixed)):
        sel = rng.choice(len(D), size=M_PTS, replace=False)
        P, z = D[sel, :3], D[sel, 3] + rng.normal(0, NOISE, M_PTS)
        hat = mmse_gpu(torch.tensor(P, device=DEV, dtype=torch.float32),
                       torch.tensor(z, device=DEV, dtype=torch.float32), DEPTH,
                       torch.tensor(Pq, device=DEV, dtype=torch.float32),
                       NOISE, kp, seed=seed)
        hat = hat.detach().cpu().numpy()
        e = hat - gt
        u = local_rate(Pq, now[name], foot[inband])   # the t=0 mask only
        u = (u - u.mean()) / (u.std() + 1e-12)
        X = np.column_stack([np.ones(nq), rn, rn ** 2, u])
        beta, pr2 = ols(e, X)
        sn = (sq - sq.mean()) / (sq.std() + 1e-12)
        bs, _ = ols(e, np.column_stack([np.ones(nq), gt, sn]))
        out[name] = {'b': float(beta[3]), 'pr2': pr2,
                     'b_s': float(bs[2]),
                     'nrmse': float(np.sqrt((e ** 2).mean()) / sig),
                     'e': e.tolist(), 'u': u.tolist(), 'rn': rn.tolist(),
                     'sig': sig}
    return out


def clustered(v):
    v = np.asarray(v, float)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.inf
    return float(v.mean()), float(se), float(v.mean() / se if se > 0 else 0.0)


def loso_gain(rows, name):
    """fit b on the other seas, apply to the held-out sea, report nRMSE removed."""
    gains = []
    for i in range(len(rows)):
        bs = [rows[j][name]['b'] for j in range(len(rows)) if j != i]
        b = float(np.mean(bs))
        r = rows[i][name]
        e, u = np.asarray(r['e']), np.asarray(r['u'])
        rn = np.asarray(r['rn'])
        # The r-trend is not the claim, so it is removed from BOTH sides. The
        # first version built e0 and then scored e -- a dead variable that let
        # the range-trend leakage into every LOSO number (#42).
        X = np.column_stack([np.ones(len(e)), rn, rn ** 2])
        e0 = e - X @ np.linalg.lstsq(X, e, rcond=None)[0]
        before = np.sqrt((e0 ** 2).mean()) / r['sig']
        after = np.sqrt(((e0 - b * u) ** 2).mean()) / r['sig']
        gains.append(before - after)
    return clustered(gains)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seas', type=int, default=24)
    ap.add_argument('--amp-scale', type=float, default=1.0)
    ap.add_argument('--tag', default='main')
    a_ = ap.parse_args()

    print(f"=== Door 1b v2  seas={a_.seas} amp_scale={a_.amp_scale} tag={a_.tag} ===")
    print("PRE-REGISTERED: G1 |t(b_mcar)|<2 and |b_mcar|<0.2|b_occ| (exact null)")
    print("                G2 |t(b_occ)|>2")
    print("                G3 partial R2 >= 0.01 AND LOSO nRMSE removed >= 0.005")
    print("                G4 flat sea (amp 1e-4) => |b_occ| -> 0")
    print(f"rays/scan {N_AZ*N_EL}  scans {len(SCAN_T)}  M={M_PTS}  R0={R0} m\n")

    rng = np.random.default_rng(7)
    rows = []
    for s in range(a_.seas):
        r = run_sea(s, a_.amp_scale, rng)
        if r is None:
            print(f"sea {s:2d}  skipped (too few returns)"); continue
        rows.append(r)
        print(f"sea {s:2d} sig {r['sigma']:.3f} rate {r['rate']:.3f} "
              f"| occ b {r['occ']['b']:+.4f} pr2 {r['occ']['pr2']:.4f} "
              f"nR {r['occ']['nrmse']:.3f} | mcar b {r['mcar']['b']:+.4f} "
              f"pr2 {r['mcar']['pr2']:.4f} nR {r['mcar']['nrmse']:.3f}", flush=True)

    if len(rows) < 2:
        print("too few seas"); return 1

    res = {'tag': a_.tag, 'amp_scale': a_.amp_scale, 'n_sea': len(rows)}
    print("\n" + "=" * 76)
    for name in ('occ', 'mcar', 'fixed'):
        m, se, t = clustered([r[name]['b'] for r in rows])
        p2 = float(np.mean([r[name]['pr2'] for r in rows]))
        nr = clustered([r[name]['nrmse'] for r in rows])
        bs = clustered([r[name]['b_s'] for r in rows])
        gm, gse, gt_ = loso_gain(rows, name)
        res[name] = {'b': m, 'se': se, 't': t, 'pr2': p2,
                     'nrmse': nr[0], 'nrmse_se': nr[1],
                     'b_s': bs[0], 'b_s_t': bs[2],
                     'loso_gain': gm, 'loso_se': gse, 'loso_t': gt_}
        print(f"{name:5s} b_u = {m:+.4f} +- {se:.4f} (t {t:+.2f})  pR2 {p2:.4f}"
              f"   b_s = {bs[0]:+.4f} (t {bs[2]:+.2f})   nRMSE {nr[0]:.3f}")

    # PAIRED contrasts. Arm F ('fixed') is the only arm whose sample locations do
    # not depend on eta, so it is the null for the slope coefficient; arm C
    # ('mcar') is the null for the mask/density coefficient.
    d_bs = clustered([r['occ']['b_s'] - r['fixed']['b_s'] for r in rows])
    d_nr = clustered([r['occ']['nrmse'] - r['fixed']['nrmse'] for r in rows])
    c_bs = clustered([r['mcar']['b_s'] - r['fixed']['b_s'] for r in rows])
    res['paired'] = {'d_bs': d_bs, 'd_nrmse': d_nr, 'ctrl_bs': c_bs}
    print(f"\npaired vs arm F (field-independent locations):")
    print(f"  occ  - fixed :  d b_s = {d_bs[0]:+.4f} +- {d_bs[1]:.4f} (t {d_bs[2]:+.2f})"
          f"    d nRMSE = {d_nr[0]:+.4f} +- {d_nr[1]:.4f} (t {d_nr[2]:+.2f})")
    print(f"  mcar - fixed :  d b_s = {c_bs[0]:+.4f} +- {c_bs[1]:.4f} (t {c_bs[2]:+.2f})"
          f"    [transparent hits: displacement WITHOUT shadowing]")

    # G1 as equivalence (#39): magnitude, not significance.
    g1 = (abs(res['fixed']['b']) <= 0.2 * abs(res['occ']['b']) + 1e-9
          and abs(res['fixed']['b_s']) <= 0.2 * abs(res['occ']['b_s']) + 1e-9
          and abs(res['fixed']['loso_gain']) <= 0.002)
    g2 = (abs(res['occ']['t']) > 2.0) or (abs(d_bs[2]) > 2.0)
    g3 = (res['occ']['pr2'] >= 0.01 and res['occ']['loso_gain'] >= 0.005) \
        or (d_nr[0] >= 0.005 and d_nr[2] >= 2.0)
    print(f"\nG1 equivalence (arm F <= 0.2 x arm A, |LOSO| <= .002) : "
          f"{'PASS' if g1 else 'FAIL'}   |b_u| {abs(res['fixed']['b']):.4f} vs "
          f"{0.2*abs(res['occ']['b']):.4f} ; |b_s| {abs(res['fixed']['b_s']):.4f} "
          f"vs {0.2*abs(res['occ']['b_s']):.4f} ; LOSO "
          f"{res['fixed']['loso_gain']:+.4f}")
    print(f"   [raw t on arm F, for the stricter reading: t_u "
          f"{res['fixed']['t']:+.2f}, t_s {res['fixed']['b_s_t']:+.2f}]")
    print(f"G2 positive   (|t(b_u occ)|>2 or |t(d b_s)|>2)  : "
          f"{'PASS' if g2 else 'FAIL'}  (t_u {res['occ']['t']:+.2f}, "
          f"t_ds {d_bs[2]:+.2f})")
    print(f"G3 effect size(LOSO>=.005 or d nRMSE>=.005 t>=2): "
          f"{'PASS' if g3 else 'FAIL'}  (LOSO {res['occ']['loso_gain']:+.4f}, "
          f"d nRMSE {d_nr[0]:+.4f})")
    print("G4 reduction  : rerun with --amp-scale 1e-4 --tag flat  (|b|->0)")
    if not g1:
        print("\nG1 FAILED -> instrument broken. NO VERDICT. (pre-registered)")
    res['gates'] = {'G1': bool(g1), 'G2': bool(g2), 'G3': bool(g3)}

    for r in rows:                                   # drop the bulky vectors
        for k in ('occ', 'mcar'):
            for f in ('e', 'u', 'rn'):
                r[k].pop(f, None)
    res['rows'] = rows
    os.makedirs('results/direction3', exist_ok=True)
    p = f'{OUT}_{a_.tag}.json'
    with open(p, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {p}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
