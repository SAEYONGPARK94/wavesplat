"""
d3_sa_fim_identifiability.py — Bayesian FIM/CRB identifiability of wave
components under FIXED vs MOVING sampling operators, evaluated on the 56 real
windows of the multi-record study, and validated against the observed gains.

Why (2026-07-04): the multi-record collapse falsified the universal D_eff/lam
gain law; the drill-down located the moving-platform bottleneck at component
IDENTIFIABILITY under the sampling operator. This script computes that
identifiability from first principles, per real window:

  1. TRUE MODES: top-M linear modes of the full stereo cube (spatial FFT,
     per-bin complex demodulation e^{+i w t} vs e^{-i w t} resolves the
     propagation sign; amplitude self-test gate included — the E1 factor-2
     lesson).
  2. FIM: wave model eta = sum_j A_j cos(kx_j x + ky_j y - w_j t + phi_j),
     w = dispersion(|k|, depth), theta = (A, phi, kx, ky) x M. Bayesian
     J_post = D^T D / sigma^2 + Lambda^{-1} with a spectrum-scaled prior
     (without data: A ~ its own scale, phi flat-ish, k within a bin width).
     Sampling positions reproduce the run protocol exactly (same build_obs,
     same tracks, same seeds structure).
  3. FIELD BOUND: per common-unobs pixel x*, b(x*) = g^T J_post^{-1} g with
     g = grad_theta eta(x*, t*): the best any estimator (learned or not) can
     do in this model class. Predicted corr_bound = sqrt(1 - mean(b)/var).
  4. VALIDATION: predicted A-baseline strength vs observed A_abs (does theory
     explain fixed-sensor starvation?), predicted motion gain vs observed
     RELAX/RIDGE gains per (window, ratio).

Usage:
  python3 scripts/d3_sa_fim_identifiability.py --selftest   # gates only
  python3 scripts/d3_sa_fim_identifiability.py              # full (all windows)
"""
import os
import sys
import json
import time
import zlib
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_sa_multirecord import MRecord, RECORDS, valid_aware_track
from d3_sa_gonogo import build_obs, region_masks, diagonal_track
from d3_sa_uspeed import resolve_prop_sign, centered_segment

G = 9.81
M_MODES = 32
SIGMA_FRAC = 0.05          # obs noise sigma = SIGMA_FRAC * Hs (sensitivity-checked)
N_PIX_BOUND = 1200         # pixels sampled from common_unobs for the field bound
JSON_RUN = 'results/direction3/sa_multirecord.json'
OUT = 'results/direction3/sa_fim_identifiability.json'


# ------------------------------------------------------------------ dispersion
def omega_of(kx, ky, depth):
    k = np.hypot(kx, ky) + 1e-12
    return np.sqrt(G * k * np.tanh(k * depth))


def domega_dk(kx, ky, depth):
    """(d w/d kx, d w/d ky) = c_g * k_hat."""
    k = np.hypot(kx, ky) + 1e-12
    kd = k * depth
    om = omega_of(kx, ky, depth)
    cg = G * (np.tanh(kd) + kd / np.cosh(kd) ** 2) / (2 * om)
    return cg * kx / k, cg * ky / k


# ------------------------------------------------------------------ modes
def extract_modes(Zwin, dt, dx, dy, depth, M=M_MODES, n_frames_fft=96,
                  gx=None, gy=None):
    """Top-M travelling modes from the space-time cube.
    Per spatial FFT bin k, demodulate the frame spectra with e^{+i w t} and
    e^{-i w t}: for a real field the bin holds a(k) e^{-iwt} + b*(-k) e^{+iwt};
    |<F_t e^{+iwt}>| estimates |a| (the +k traveller), |<F_t e^{-iwt}>| the
    -k traveller aliased into this bin. Winner sets the propagation sign.
    Returns kx, ky, A, phi (one-sided, A of eta = A cos(k.x - w t + phi))."""
    nf, n0, n1 = Zwin.shape
    sel = np.linspace(0, nf - 1, min(n_frames_fft, nf)).astype(int)
    t_sel = sel * dt
    Zf = np.nan_to_num(Zwin[sel])
    valid_frac = max(float(np.isfinite(Zwin[0]).mean()), 1e-3)
    F = np.fft.fft2(Zf, axes=(1, 2))                       # (K, n0, n1)
    kxv = 2 * np.pi * np.fft.fftfreq(n1, dx)
    kyv = 2 * np.pi * np.fft.fftfreq(n0, dy)
    KX, KY = np.meshgrid(kxv, kyv)
    OM = omega_of(KX, KY, depth)
    ph = np.exp(1j * OM[None] * t_sel[:, None, None])
    a_pos = (F * ph).mean(0)                               # +k travellers
    a_neg = (F * np.conj(ph)).mean(0)                      # -k travellers (conj pair)
    # half-plane fold: consider each bin once (kx > 0 or (kx==0 & ky>0))
    half = (KX > 1e-9) | ((np.abs(KX) <= 1e-9) & (KY > 1e-9))
    amp_pos = np.abs(a_pos) * half
    amp_neg = np.abs(a_neg) * half
    use_neg = amp_neg > amp_pos
    amp = np.where(use_neg, amp_neg, amp_pos)
    flat = amp.ravel()
    top = np.argsort(flat)[::-1][:M]
    N = n0 * n1
    kx_o, ky_o = [], []
    for idx in top:
        i, j = np.unravel_index(idx, amp.shape)
        neg = bool(use_neg[i, j])
        # -k traveller in bin k appears as conj(b) at +k: the physical mode is
        # at wavevector -k (sign from the demodulation winner)
        kx_o.append(float(-KX[i, j] if neg else KX[i, j]))
        ky_o.append(float(-KY[i, j] if neg else KY[i, j]))
    kx_o, ky_o = np.array(kx_o), np.array(ky_o)
    # amplitude/phase by LS on VALID pixels only — the zero-filled FFT biases
    # A and phi badly on NaN-heavy records (AA valid 0.43); bin selection and
    # propagation sign above are robust to the fill, the linear refit is exact.
    # ABSOLUTE grid coordinates are essential: phases referenced to a corner
    # origin shift by k*origin per mode when rendered at absolute (gx, gy),
    # which anti-correlates the whole modal field (bug found 2026-07-04).
    if gx is None:
        gx = np.arange(n1) * dx
    if gy is None:
        gy = np.arange(n0) * dy
    A_o, phi_o = refit_modes_ls(Zwin, sel, t_sel, kx_o, ky_o, depth, gx, gy)
    return kx_o, ky_o, A_o, phi_o


def refit_modes_ls(Zwin, frame_sel, t_sel, kx, ky, depth, gx, gy,
                   max_px=2500, seed=0):
    """LS amplitudes/phases of fixed-(kx,ky) modes on valid pixels only.
    eta = sum_j c_j cos(psi0_j) + s_j sin(psi0_j), psi0 = kx x + ky y - w t."""
    om = omega_of(kx, ky, depth)
    rng = np.random.default_rng(seed)
    rows_P, rows_z = [], []
    X, Y = np.meshgrid(gx, gy)
    for fi, tv in zip(frame_sel[::3], t_sel[::3]):
        fr = Zwin[fi]
        ok = np.isfinite(fr)
        idx = np.argwhere(ok)
        take = rng.choice(len(idx), min(max_px // len(frame_sel[::3]) + 1,
                                        len(idx)), replace=False)
        ij = idx[take]
        rows_P.append(np.column_stack([X[ij[:, 0], ij[:, 1]],
                                       Y[ij[:, 0], ij[:, 1]],
                                       np.full(len(ij), tv)]))
        rows_z.append(fr[ij[:, 0], ij[:, 1]])
    P = np.vstack(rows_P)
    z = np.concatenate(rows_z).astype(float)
    psi0 = kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2] - om[None] * P[:, 2:3]
    Dm = np.concatenate([np.cos(psi0), np.sin(psi0)], axis=1)
    M = len(kx)
    reg = 1e-6 * len(z) * np.eye(2 * M)
    beta = np.linalg.solve(Dm.T @ Dm + reg, Dm.T @ z)
    c, s = beta[:M], beta[M:]
    return np.hypot(c, s), np.arctan2(-s, c)


def render_modes(kx, ky, A, phi, depth, X, Y, t):
    om = omega_of(kx, ky, depth)
    psi = (kx[None] * X[:, None] + ky[None] * Y[:, None]
           - om[None] * t + phi[None])
    return (A[None] * np.cos(psi)).sum(1)


# ------------------------------------------------------------------ FIM
def design_matrix(theta, P, depth):
    """D (n x 4M): d eta / d (A, phi, kx, ky) at obs points P=(x,y,t)."""
    kx, ky, A, phi = theta
    om = omega_of(kx, ky, depth)
    dwx, dwy = domega_dk(kx, ky, depth)
    x, y, t = P[:, 0:1], P[:, 1:2], P[:, 2:3]
    psi = kx[None] * x + ky[None] * y - om[None] * t + phi[None]
    c, s = np.cos(psi), np.sin(psi)
    dA = c
    dphi = -A[None] * s
    dkx = -A[None] * s * (x - dwx[None] * t)
    dky = -A[None] * s * (y - dwy[None] * t)
    return np.concatenate([dA, dphi, dkx, dky], axis=1)    # (n, 4M)


def prior_precision(theta, dk_bin):
    """Lambda^{-1} diag: A within its own scale, phi ~ pi/sqrt(3) (flat),
    kx/ky within one FFT bin width."""
    kx, ky, A, phi = theta
    M = len(kx)
    var = np.concatenate([
        np.maximum(A, 1e-6) ** 2,
        np.full(M, (np.pi / np.sqrt(3)) ** 2),
        np.full(M, (0.5 * dk_bin) ** 2),
        np.full(M, (0.5 * dk_bin) ** 2)])
    return 1.0 / var


def field_bound(theta, Jinv, depth, Xs, Ys, t_star):
    """mean_x g^T Jinv g over sampled pixels (posterior field variance bound)."""
    P = np.column_stack([Xs, Ys, np.full(len(Xs), t_star)])
    Gd = design_matrix(theta, P, depth)                     # (npix, 4M)
    return float(np.mean(np.einsum('ij,jk,ik->i', Gd, Jinv, Gd)))


def predicted_corr(theta, P_obs, depth, sigma, Xs, Ys, t_star, dk_bin):
    D = design_matrix(theta, P_obs, depth)
    lam_inv = prior_precision(theta, dk_bin)
    J = D.T @ D / sigma ** 2 + np.diag(lam_inv)
    Jinv = np.linalg.inv(J)
    b = field_bound(theta, Jinv, depth, Xs, Ys, t_star)
    fld = render_modes(*_reorder(theta), depth, Xs, Ys, t_star)
    var = float(np.var(fld))
    r2 = max(0.0, 1.0 - b / max(var, 1e-12))
    # identifiability: data-dominated directions of Lambda^{1/2} J_data Lambda^{1/2}
    Jd = D.T @ D / sigma ** 2
    S = np.sqrt(1.0 / lam_inv)
    eig = np.linalg.eigvalsh(S[:, None] * Jd * S[None, :])
    n_ident = int((eig > 1.0).sum())
    return float(np.sqrt(r2)), n_ident


def _reorder(theta):
    kx, ky, A, phi = theta
    return kx, ky, A, phi


# ------------------------------------------------------------------ self-tests
def selftest():
    rng = np.random.default_rng(0)
    depth, dx, dy, dt = 30.0, 0.5, 0.5, 0.1
    n0 = n1 = 96
    nf = 240
    # non-zero origin on purpose: regression gate for the corner-origin phase
    # bug (phases must be referenced to ABSOLUTE coordinates)
    gx = np.arange(n1) * dx + 37.3
    gy = np.arange(n0) * dy - 12.9
    X, Y = np.meshgrid(gx, gy)
    # G1: single mode, both propagation signs
    ok = True
    dkx = 2 * np.pi / (n1 * dx)                  # on-bin modes (no leakage)
    dky = 2 * np.pi / (n0 * dy)
    for sgn in (+1, -1):
        kx0, ky0, A0, ph0 = sgn * 3 * dkx, sgn * 1 * dky, 0.42, 0.9
        om0 = omega_of(kx0, ky0, depth)
        Z = np.stack([A0 * np.cos(kx0 * X + ky0 * Y - om0 * (f * dt) + ph0)
                      for f in range(nf)])
        kx, ky, A, phi = extract_modes(Z, dt, dx, dy, depth, M=1, gx=gx, gy=gy)
        errA = abs(A[0] - A0) / A0
        errk = np.hypot(kx[0] - kx0, ky[0] - ky0)
        errp = abs((phi[0] - ph0 + np.pi) % (2 * np.pi) - np.pi)
        print(f"G1 sign {sgn:+d}: A err {errA:.3f} k err {errk:.4f} "
              f"phi err {errp:.3f}")
        ok &= errA < 0.05 and errk < 0.03 and errp < 0.1
    # G2: CRB(A) for one mode, dense fixed obs ~ 2 sigma^2 / n
    theta = (np.array([0.3]), np.array([0.1]), np.array([0.5]), np.array([0.2]))
    n = 400
    P = np.column_stack([rng.uniform(0, 5, n), rng.uniform(0, 5, n),
                         np.linspace(0, 30, n)])
    D = design_matrix(theta, P, depth)
    sigma = 0.02
    J = D.T @ D / sigma ** 2
    crbA = np.linalg.inv(J)[0, 0]
    print(f"G2: CRB(A) {crbA:.2e} vs 2 sigma^2/n {2 * sigma**2 / n:.2e} "
          f"(same order expected)")
    ok &= crbA < 10 * 2 * sigma ** 2 / n
    print("selftest:", "PASS" if ok else "FAIL")
    return ok


# ------------------------------------------------------------------ main
def run_all(args):
    store = json.load(open(JSON_RUN))
    out = []
    for key, cfg in RECORDS.items():
        if key not in store:
            continue
        run = store[key]
        rec = MRecord(cfg['path'], stride=cfg.get('stride', 1))
        nf = int(round(cfg['T'] / rec.dt))
        r_fp = cfg['fp'] / 2.0
        track_fn = valid_aware_track if rec.valid.mean() < 0.98 else None
        rec_seed = zlib.crc32(key.encode()) % 100000
        for wi, w in enumerate(run['windows']):
            tic = time.time()
            i0 = w['i0']
            Zwin = rec.window(i0, nf)
            theta_kx, theta_ky, theta_A, theta_phi = extract_modes(
                Zwin, rec.dt, rec.dx, rec.dy, cfg['depth'],
                gx=rec.gx, gy=rec.gy)
            theta = (theta_kx, theta_ky, theta_A, theta_phi)
            Hs = w['Hs_win']
            sigma = SIGMA_FRAC * Hs
            dk_bin = 2 * np.pi / max(rec.gx.max() - rec.gx.min(),
                                     rec.gy.max() - rec.gy.min())
            t_off = float(rec.t[i0])
            t_star = float(rec.t[i0 + nf - 1]) - t_off
            # geometry identical to the run
            pdir = w['peak_dir']
            if track_fn is not None:
                P0d, P1d, _ = track_fn(rec, r_fp, pdir)
            else:
                P0d, P1d, _ = diagonal_track(rec.gx, rec.gy, r_fp, pdir)
            prop = w['prop_dir']
            pvec = np.array([np.cos(np.radians(prop)), np.sin(np.radians(prop))])
            if float((P1d - P0d) @ pvec) < 0:
                P0d, P1d = P1d, P0d
            masks = region_masks(rec, r_fp, 0.6 * cfg['fp'], P0d, P1d)
            Xg, Yg = np.meshgrid(rec.gx, rec.gy)
            mu = masks['common_unobs']
            pix = np.argwhere(mu)
            take = np.random.default_rng(0).choice(
                len(pix), min(N_PIX_BOUND, len(pix)), replace=False)
            Xs = Xg[mu][take]
            Ys = Yg[mu][take]
            L_diag = float(np.linalg.norm(P1d - P0d))
            # extraction-quality gate: modal field vs the actual last frame
            gt = Zwin[-1]
            okp = np.argwhere(np.isfinite(gt))
            tk = np.random.default_rng(1).choice(len(okp), min(2000, len(okp)),
                                                 replace=False)
            ij = okp[tk]
            fld_m = render_modes(theta_kx, theta_ky, theta_A, theta_phi,
                                 cfg['depth'], Xg[ij[:, 0], ij[:, 1]],
                                 Yg[ij[:, 0], ij[:, 1]], t_star)
            extract_q = float(np.corrcoef(fld_m, gt[ij[:, 0], ij[:, 1]])[0, 1])
            row = dict(record=key, wi=wi, i0=int(i0), Hs=Hs,
                       lam_p=w['lam_p'], extract_q=extract_q, conds={})
            # A (fixed) + moving conditions at the run's ratios
            rng = np.random.default_rng(rec_seed + 10 * wi)
            P_A, _, _, _ = build_obs(Zwin, rec, i0, run['n_obs'], r_fp,
                                     'fixed', rng)
            P_A[:, 2] -= t_off
            cA, nA = predicted_corr(theta, P_A, cfg['depth'], sigma,
                                    Xs, Ys, t_star, dk_bin)
            row['conds']['A'] = dict(corr_pred=cA, n_ident=nA)
            for r in run['ratios']:
                U = r * w['cp']
                L_seg = min(U * cfg['T'], L_diag)
                seg = centered_segment(P0d, P1d, L_seg)
                rngB = np.random.default_rng(rec_seed + 10 * wi + 1)
                P_B, _, _, _ = build_obs(Zwin, rec, i0, run['n_obs'], r_fp,
                                         'moving', rngB, U=U, track=seg)
                P_B[:, 2] -= t_off
                cB, nB = predicted_corr(theta, P_B, cfg['depth'], sigma,
                                        Xs, Ys, t_star, dk_bin)
                row['conds'][f'B_r{r:g}'] = dict(corr_pred=cB, n_ident=nB,
                                                 gain_pred=cB - cA)
            del Zwin
            print(f"  {key} w{wi} [{time.time()-tic:.0f}s] q={extract_q:.2f} "
                  f"A pred {cA:.2f} (N_id {nA}) | " +
                  " ".join(f"r{r:g}:{row['conds'][f'B_r{r:g}']['gain_pred']:+.2f}"
                           f"/{row['conds'][f'B_r{r:g}']['n_ident']}"
                           for r in run['ratios']))
            out.append(row)
        rec.f.close()
    json.dump(out, open(OUT, 'w'), indent=1, default=float)
    print(f"saved {OUT}")
    return out


def validate(out, store, q_min=0.5):
    """Theory vs experiment. Windows with extraction quality < q_min are
    excluded (their 'theory' is a bound for the wrong modes)."""
    n_all = len(out)
    out = [r for r in out if r.get('extract_q', 1.0) >= q_min]
    print(f"\nextraction gate: {len(out)}/{n_all} windows pass q>={q_min}")
    pred_A, obs_A_rx, obs_A_rd = [], [], []
    gp, go_rx, go_rd, recs = [], [], [], []
    for row in out:
        run = store[row['record']]
        w = run['windows'][row['wi']]
        A = w['conditions']['A']
        pred_A.append(row['conds']['A']['corr_pred'])
        obs_A_rx.append(A['mean']['common_unobs']['corr_bl'])
        obs_A_rd.append(A['mean_ridge']['common_unobs']['corr_bl'])
        for name, c in row['conds'].items():
            if name == 'A' or name not in w['conditions']:
                continue
            e = w['conditions'][name]
            gp.append(c['gain_pred'])
            go_rx.append(e['mean']['common_unobs']['corr_bl']
                         - A['mean']['common_unobs']['corr_bl'])
            go_rd.append(e['mean_ridge']['common_unobs']['corr_bl']
                         - A['mean_ridge']['common_unobs']['corr_bl'])
            recs.append(row['record'])

    def cc(a, b):
        a, b = np.asarray(a), np.asarray(b)
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1])

    print("\n===== THEORY vs EXPERIMENT =====")
    print(f"A-baseline: corr(pred, obs) RELAX {cc(pred_A, obs_A_rx):+.2f} "
          f"RIDGE {cc(pred_A, obs_A_rd):+.2f}  (n={len(pred_A)} windows)")
    print(f"motion gain: corr(pred, obs) RELAX {cc(gp, go_rx):+.2f} "
          f"RIDGE {cc(gp, go_rd):+.2f}  (n={len(gp)} points)")
    print("\nper-record mean (pred gain | obs RELAX | obs RIDGE):")
    res_rec = {}
    for k in sorted(set(recs)):
        ii = [i for i, r in enumerate(recs) if r == k]
        pg = float(np.mean([gp[i] for i in ii]))
        ox = float(np.mean([go_rx[i] for i in ii]))
        od = float(np.mean([go_rd[i] for i in ii]))
        res_rec[k] = dict(pred=pg, obs_rx=ox, obs_rd=od)
        print(f"  {k:7s} pred {pg:+.3f} | RELAX {ox:+.3f} | RIDGE {od:+.3f}")
    sgn = [np.sign(res_rec[k]['pred']) == np.sign(res_rec[k]['obs_rx'])
           for k in res_rec]
    print(f"record-level sign agreement (RELAX): {int(np.sum(sgn))}/{len(sgn)}")
    return dict(corr_A_rx=cc(pred_A, obs_A_rx), corr_A_rd=cc(pred_A, obs_A_rd),
                corr_gain_rx=cc(gp, go_rx), corr_gain_rd=cc(gp, go_rd),
                per_record=res_rec,
                sign_agree_rx=int(np.sum(sgn)), n_records=len(sgn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if not selftest():
        sys.exit(1)
    if args.selftest:
        return
    out = run_all(args)
    store = json.load(open(JSON_RUN))
    summary = validate(out, store)
    payload = dict(sigma_frac=SIGMA_FRAC, M=M_MODES, rows=out, summary=summary)
    json.dump(payload, open(OUT, 'w'), indent=1, default=float)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
