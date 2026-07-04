"""
d3_sa_achievability_gap.py — How much of the FIM bound do real estimators
achieve under FIXED vs MOVING sampling? Synthetic seas with KNOWN modes
(no extraction error, no model mismatch) on the real-run protocol geometry.

The 2026-07-04 real-data FIM analysis found the information bound predicts
positive motion gains while RELAX shows harm — implying an ALGORITHMIC gap.
This script measures that gap directly and decomposes it:

  bound        Bayesian FIM/CRB predicted corr (true modes, true sigma)
  oracle_ls    true (kx,ky) given, A/phi by ridge LS  -> identification solved,
               only linear fitting left. bound - oracle_ls = fitting gap.
  oracle_ref   oracle_ls + joint nonconvex refine (physics polish)
  relax        iterative_relax_cv (greedy identification + CV + refine)
  ridge        fixed-dictionary ridge (no identification)
  net          track-aware detection net (beamform_net_trackaware.pt) + LS
               (skipped if checkpoint missing)

If oracle_ls ~ bound while relax << oracle_ls under moving sampling, the
moving-platform loss is IDENTIFICATION (learnable target confirmed at its
theoretical prize); if even oracle_ls falls short, the loss is conditioning.

Protocol mirrors the real runs: patch 2 lam x 2 lam, T = 10 Tp, 400 obs at
linspace times, fixed disk fp = 0.18 lam vs ping-pong diagonal track at
U = {0.43, 1.0} c_p, scoring on the common-unobserved region.

Usage: python3 scripts/d3_sa_achievability_gap.py [--n_seas 12] [--smoke]
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from estimator_improved import iterative_relax_cv, refine_params
from d3_sa_fim_identifiability import (design_matrix, prior_precision,
                                       omega_of, render_modes)

G = 9.81
DEPTH = 50.0
TP, HS = 7.0, 2.0
LAM = G * TP ** 2 / (2 * np.pi) * np.tanh(0.7)   # ~kd=0.7-ish; recompute below
NOISE_FRAC = 0.05
N_OBS = 400
OUT = 'results/direction3/sa_achievability_gap.json'


def dispersion_lam(Tp, depth):
    om = 2 * np.pi / Tp
    k = om ** 2 / G
    for _ in range(60):
        k = om ** 2 / (G * np.tanh(k * depth))
    return 2 * np.pi / k


def true_field(theta, depth, X, Y, t):
    kx, ky, A, phi = theta
    om = omega_of(kx, ky, depth)
    psi = (kx[None] * X.ravel()[:, None] + ky[None] * Y.ravel()[:, None]
           - om[None] * t + phi[None])
    return ((A[None] * np.cos(psi)).sum(1)).reshape(X.shape)


def sample_obs(mode, U, lam, cp, T, gx, gy, r_fp, rng):
    tt = np.linspace(0, T, N_OBS)
    cx, cy = (gx[0] + gx[-1]) / 2, (gy[0] + gy[-1]) / 2
    th = rng.uniform(0, 2 * np.pi, N_OBS)
    rr = r_fp * np.sqrt(rng.uniform(0, 1, N_OBS))
    jx, jy = rr * np.cos(th), rr * np.sin(th)
    if mode == 'fixed':
        return np.column_stack([cx + jx, cy + jy, tt])
    # ping-pong diagonal (SW->NE), inset by r_fp
    P0 = np.array([gx[0] + r_fp, gy[0] + r_fp])
    P1 = np.array([gx[-1] - r_fp, gy[-1] - r_fp])
    L = float(np.linalg.norm(P1 - P0))
    u = (P1 - P0) / L
    s = np.mod(U * tt, 2 * L)
    d = np.where(s <= L, s, 2 * L - s)
    xy = P0[None] + d[:, None] * u[None]
    return np.column_stack([xy[:, 0] + jx, xy[:, 1] + jy, tt])


def masks_common_unobs(gx, gy, r_fp, margin):
    Xg, Yg = np.meshgrid(gx, gy)
    cx, cy = (gx[0] + gx[-1]) / 2, (gy[0] + gy[-1]) / 2
    dA = np.hypot(Xg - cx, Yg - cy)
    P0 = np.array([gx[0] + r_fp, gy[0] + r_fp])
    P1 = np.array([gx[-1] - r_fp, gy[-1] - r_fp])
    seg = P1 - P0
    L2 = float(seg @ seg)
    tpar = np.clip(((Xg - P0[0]) * seg[0] + (Yg - P0[1]) * seg[1]) / L2, 0, 1)
    dB = np.hypot(Xg - (P0[0] + tpar * seg[0]), Yg - (P0[1] + tpar * seg[1]))
    return (dA > r_fp + margin) & (dB > r_fp + margin)


def fim_bound(theta, P, sigma, depth, Xs, Ys, t_star, dk_bin,
              theta_resid=None):
    """CRB-implied corr for estimating the theta modes from obs at P.
    theta_resid: unmodeled modes treated as a COLORED noise field with
    per-mode variance A^2/2 (random phase) -> GLS FIM with
    C = Phi diag(A^2/2) Phi^T (+ same at the query points for the target
    variance). This is the structured-residual treatment: white-sigma_eff
    over-penalizes because wave-like residual is largely separable."""
    D = design_matrix(theta, P, depth)
    lam_inv = prior_precision(theta, dk_bin)
    n = len(P)
    if theta_resid is None:
        Ci_D = D / sigma ** 2
        J = D.T @ Ci_D + np.diag(lam_inv)
    else:
        rkx, rky, rA, _ = theta_resid
        om = omega_of(rkx, rky, depth)
        psi = (rkx[None] * P[:, 0:1] + rky[None] * P[:, 1:2]
               - om[None] * P[:, 2:3])
        # random-phase mode == cos and sin quadratures, each var A^2/4
        Phi = np.concatenate([np.cos(psi), np.sin(psi)], axis=1)
        w = np.concatenate([rA ** 2 / 4.0, rA ** 2 / 4.0])
        C = (Phi * w[None]) @ Phi.T + sigma ** 2 * np.eye(n)
        Ci_D = np.linalg.solve(C, D)
        J = D.T @ Ci_D + np.diag(lam_inv)
    Ji = np.linalg.inv(J)
    Pq = np.column_stack([Xs, Ys, np.full(len(Xs), t_star)])
    Gd = design_matrix(theta, Pq, depth)
    b = float(np.mean(np.einsum('ij,jk,ik->i', Gd, Ji, Gd)))
    fld = render_modes(*theta, depth, Xs, Ys, t_star)
    return float(np.sqrt(max(0.0, 1.0 - b / max(np.var(fld), 1e-12))))


def corr(a, b):
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_seas', type=int, default=12)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--spectrum', choices=['discrete', 'continuous'],
                    default='discrete',
                    help='continuous: nc=200, model class = top-32 true modes,'
                         ' rest = structured residual (bound calibration)')
    args = ap.parse_args()
    n_seas = 2 if args.smoke else args.n_seas

    lam = dispersion_lam(TP, DEPTH)
    cp = lam / TP
    T = 10 * TP
    fp = 0.18 * lam
    r_fp = fp / 2
    side = 2.0 * lam
    gx = np.linspace(0, side, 96)
    gy = np.linspace(0, side, 96)
    Xg, Yg = np.meshgrid(gx, gy)
    mu = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
    Xs, Ys = Xg[mu], Yg[mu]
    if len(Xs) > 1500:
        take = np.random.default_rng(0).choice(len(Xs), 1500, replace=False)
        Xs, Ys = Xs[take], Ys[take]
    dk_bin = 2 * np.pi / side
    t_star = T
    print(f"lam={lam:.1f} m cp={cp:.2f} m/s patch {side:.0f} m (2 lam), "
          f"T={T:.0f} s, fp={fp:.1f} m, unobs px {int(mu.sum())}")

    # optional track-aware net
    net = None
    try:
        import torch
        from d3_beamform_net import DetectCNN, DEVICE
        from d3_sa_moving_ident_probe import net_estimate, CKPT_MIXED
        net = DetectCNN().to(DEVICE)
        net.load_state_dict(torch.load(CKPT_MIXED, map_location=DEVICE))
        net.eval()
    except Exception as e:
        print(f"net skipped: {e}")

    conds = [('fixed', None), ('track043', 0.43), ('track100', 1.0)]
    res = {c: {} for c, _ in conds}
    for si in range(n_seas):
        nc = (200 if args.spectrum == 'continuous'
              else (15 if si % 2 == 0 else 30))
        c = jonswap_components(Hs=HS, Tp=TP, n_components=nc, depth=DEPTH,
                               seed=4000 + si)
        theta_full = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
                      np.array([x.A for x in c]), np.array([x.phi for x in c]))
        if args.spectrum == 'continuous':
            order = np.argsort(theta_full[2])[::-1]
            top, rest = order[:32], order[32:]
            theta = tuple(v[top] for v in theta_full)
            theta_resid = tuple(v[rest] for v in theta_full)
        else:
            theta, theta_resid = theta_full, None
        om = omega_of(theta_full[0], theta_full[1], DEPTH)
        gt = true_field(theta_full, DEPTH, Xg, Yg, t_star)
        gt_u = render_modes(*theta_full, DEPTH, Xs, Ys, t_star)
        sigma = NOISE_FRAC * HS
        Hs_w = 4 * gt.std()
        for cname, ur in conds:
            rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
            U = (ur or 0) * cp
            P = sample_obs('fixed' if cname == 'fixed' else 'moving', U, lam,
                           cp, T, gx, gy, r_fp, rng)
            psi = (theta_full[0][None] * P[:, 0:1]
                   + theta_full[1][None] * P[:, 1:2]
                   - om[None] * P[:, 2:3] + theta_full[3][None])
            z = (theta_full[2][None] * np.cos(psi)).sum(1) \
                + rng.normal(0, sigma, N_OBS)
            P32 = P.astype(np.float32)
            z32 = z.astype(np.float32)
            row = res[cname].setdefault(si, {})
            row['bound'] = fim_bound(theta, P, sigma, DEPTH, Xs, Ys, t_star,
                                     dk_bin)
            if theta_resid is not None:
                var_resid = float((theta_resid[2] ** 2).sum() / 2.0)
                sig_w = float(np.sqrt(sigma ** 2 + var_resid))
                row['bound_white'] = fim_bound(theta, P, sig_w, DEPTH, Xs, Ys,
                                               t_star, dk_bin)
                row['bound_col'] = fim_bound(theta, P, sigma, DEPTH, Xs, Ys,
                                             t_star, dk_bin,
                                             theta_resid=theta_resid)
            # oracle_ls: true k given, ridge LS for A/phi
            obs = np.column_stack([P32, z32]).astype(np.float32)
            A_ls, phi_ls = fit_amplitudes_phases(
                theta[0].astype(np.float32), theta[1].astype(np.float32),
                obs, DEPTH, lambda_reg=0.1)
            f_ols = render_modes(theta[0], theta[1], np.asarray(A_ls, float),
                                 np.asarray(phi_ls, float), DEPTH, Xs, Ys,
                                 t_star)
            row['oracle_ls'] = corr(f_ols, gt_u)
            # stronger ridge: rules out under-regularization as the cause of
            # an oracle collapse on dense continuous spectra
            A_l2, phi_l2 = fit_amplitudes_phases(
                theta[0].astype(np.float32), theta[1].astype(np.float32),
                obs, DEPTH, lambda_reg=1.0)
            row['oracle_ls2'] = corr(
                render_modes(theta[0], theta[1], np.asarray(A_l2, float),
                             np.asarray(phi_l2, float), DEPTH, Xs, Ys, t_star),
                gt_u)
            # oracle_ref: + joint nonconvex refine from the true k init
            target = {'gx': gx, 'gy': gy, 't': t_star,
                      'gt': gt, 'Hs': Hs_w}
            try:
                _, _, f_ref = refine_params(theta[0].copy(), theta[1].copy(),
                                            P32.copy(), z32.copy(), DEPTH,
                                            target, iters=400)
                f_ref_u = f_ref[mu]
                # subsample to the same pixels
                row['oracle_ref'] = corr(
                    render_from_grid(f_ref, gx, gy, Xs, Ys), gt_u)
            except Exception as e:
                row['oracle_ref'] = np.nan
            # dense-bin ridge: ALL patch-FFT-lattice bins within the sea band,
            # both propagation signs, ridge lambda=1.0 — no selection, no
            # greedy, no learning. If this reaches oracle_ls2, the moving-
            # platform fix is "replace greedy with dense linear GLS"; if not,
            # bin SELECTION is the real value of detection.
            kp_here = 2 * np.pi / lam
            nb = int(np.ceil(2.5 * kp_here / dk_bin))
            kxb, kyb = np.meshgrid(np.arange(-nb, nb + 1) * dk_bin,
                                   np.arange(0, nb + 1) * dk_bin)
            kk = np.hypot(kxb, kyb)
            selb = (kk > 0.35 * kp_here) & (kk < 2.5 * kp_here)
            kxd = np.concatenate([kxb[selb], -kxb[selb]]).astype(np.float32)
            kyd = np.concatenate([kyb[selb], -kyb[selb]]).astype(np.float32)
            A_d, ph_d = fit_amplitudes_phases(kxd, kyd, obs, DEPTH,
                                              lambda_reg=1.0)
            row['dense'] = corr(
                render_modes(kxd.astype(float), kyd.astype(float),
                             np.asarray(A_d, float), np.asarray(ph_d, float),
                             DEPTH, Xs, Ys, t_star), gt_u)
            row['dense_nbins'] = int(len(kxd))
            # relax
            fld, N, _ = iterative_relax_cv(P32.copy(), z32.copy(), DEPTH,
                                           target, n_outer=2, seed=si)
            row['relax'] = corr(render_from_grid(fld, gx, gy, Xs, Ys), gt_u)
            row['relax_N'] = N
            # ridge (fixed dictionary)
            from d3_sa_gonogo import ridge_field
            kp = 2 * np.pi / lam
            frd = ridge_field(P32, z32, DEPTH, kp, gx, gy, t_star)
            row['ridge'] = corr(render_from_grid(frd, gx, gy, Xs, Ys), gt_u)
            # net
            if net is not None:
                from d3_sa_moving_ident_probe import net_estimate
                fn, _, _ = net_estimate(net, P32, z32, gx, gy, t_star)
                row['net'] = corr(render_from_grid(np.asarray(fn), gx, gy,
                                                   Xs, Ys), gt_u)
        print(f"sea {si} (nc={nc}): " + " | ".join(
            f"{cn} bound {res[cn][si]['bound']:.2f} "
            f"oLS {res[cn][si]['oracle_ls']:.2f} "
            f"rlx {res[cn][si]['relax']:.2f}" for cn, _ in conds))

    # summary
    print("\n===== ACHIEVABILITY (mean corr on common_unobs) =====")
    ests = ['bound']
    if args.spectrum == 'continuous':
        ests += ['bound_white', 'bound_col']
    ests += ['oracle_ls', 'oracle_ls2', 'oracle_ref', 'dense', 'relax',
             'ridge'] + (['net'] if net is not None else [])
    print(f"{'cond':10s}" + "".join(f"{e:>11s}" for e in ests))
    summ = {}
    for cname, _ in conds:
        line = f"{cname:10s}"
        summ[cname] = {}
        for e in ests:
            v = float(np.nanmean([res[cname][si].get(e, np.nan)
                                  for si in res[cname]]))
            summ[cname][e] = v
            line += f"{v:>11.3f}"
        print(line)
    print("\nmotion gain (track - fixed) per estimator:")
    for e in ests:
        for cname in ('track043', 'track100'):
            gsum = summ[cname][e] - summ['fixed'][e]
            print(f"  {e:10s} {cname}: {gsum:+.3f}")
    path = (OUT if args.spectrum == 'discrete'
            else OUT.replace('.json', '_cont.json'))
    json.dump(dict(summary=summ, rows={c: res[c] for c in res},
                   lam=lam, T=T, fp=fp, n_seas=n_seas,
                   spectrum=args.spectrum),
              open(path, 'w'), indent=1, default=float)
    print(f"saved {path}")


def render_from_grid(fld, gx, gy, Xs, Ys):
    """Sample a gridded field at (Xs, Ys) by nearest index."""
    ix = np.clip(np.round((Xs - gx[0]) / (gx[1] - gx[0])).astype(int), 0,
                 len(gx) - 1)
    iy = np.clip(np.round((Ys - gy[0]) / (gy[1] - gy[0])).astype(int), 0,
                 len(gy) - 1)
    return np.asarray(fld)[iy, ix]


def hashabs(s):
    return sum(ord(ch) for ch in s) % 97


if __name__ == '__main__':
    main()
