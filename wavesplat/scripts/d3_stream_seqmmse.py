"""
d3_stream_seqmmse.py — GIVEN-SENSOR validation, step 3 (2026-07-06):
STREAMING sequential MMSE (Kalman/RLS form) = control-compatible
(<= 100 ms/frame) version of mmse_bank_estimate.

Model: weight-space view of the bank MMSE. State w = [a_b, b_b] (cos/sin
coefficients of the B bank components), prior N(0, diag(gam)). The
dispersion time-dependence is already inside the design row
psi = kx x + ky y - om t, so w is STATIC for a stationary linear sea and
the batch MMSE is exactly recursive least squares with prior:

    P_0 = diag(gam);  per frame:  S = Phi P Phi^T + s2 I
    K = P Phi^T S^-1; w += K (z - Phi w); P -= K Phi P            (RLS)

Finite memory (sea nonstationarity / slow current drift) enters as an
AR(1)-toward-prior dynamics with the group-coherence time as the
physical interpretation:  w <- a w,  P <- a^2 P + (1 - a^2) diag(gam),
a = exp(-dt / tau). a = 1 reproduces batch MMSE on all past obs EXACTLY
(the built-in equivalence test asserts this to 1e-6). Encounter forecast
at any future (x, y, t) is phi(q)^T w with variance phi^T P phi — per
query O(nB), microseconds.

Two-timescale architecture (design; the slow loop is stubbed here):
  fast loop  = this filter, every frame (10 FPS), O(n^2 m) per frame;
  slow loop  = gamma/bank refresh (beamform power -> val-select -> EM,
               or the amortized net prior) every ~10 s on a trailing
               buffer, then rebuild P, w from the buffer (one batch
               solve, can run async).
Literature honesty: RLS/Kalman wave-field assimilation exists (CN-RLS,
DSWP); the contribution here is the measured latency + equivalence on
the GIVEN Falcon operator and the net-prior slot, not the filter.

Usage: python3 scripts/d3_stream_seqmmse.py [--episode ...] [--tau 30]
       [--prune 600]
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

from wavesplat.data_loader import EpisodeDataLoader
from d3_sa_nomp import omega

G = 9.81
OUT = 'results/direction3/stream_seqmmse.json'


def make_bank(kp, depth, dk_frac=0.1):
    """Same bank geometry as mmse_bank_estimate (annulus 0.35-2.5 kp)."""
    dk = kp * dk_frac
    ax = np.arange(-2.5 * kp, 2.5 * kp + dk / 2, dk)
    KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY)
    sel = (kk > 0.35 * kp) & (kk < 2.5 * kp)
    kxb, kyb = KX[sel].astype(float), KY[sel].astype(float)
    om = omega(np.hypot(kxb, kyb) + 1e-12, depth)
    return kxb, kyb, om


def design(kxb, kyb, om, Q):
    psi = kxb[None] * Q[:, 0:1] + kyb[None] * Q[:, 1:2] - om[None] * Q[:, 2:3]
    return np.concatenate([np.cos(psi), np.sin(psi)], axis=1)


def gamma_beamform(kxb, kyb, om, P, z, sigma0, depth, seed=0):
    """Slow-loop prior: beamform power shape + val-selected (pexp, smult)
    — the non-EM core of mmse_bank_estimate, factored for refresh use."""
    rng = np.random.default_rng(seed)
    n = len(z)
    vi = rng.choice(n, max(8, n // 4), replace=False)
    vm = np.zeros(n, bool)
    vm[vi] = True
    B = len(kxb)
    pw = np.empty(B)
    for i in range(0, B, 512):
        psi = (kxb[i:i + 512, None] * P[None, :, 0]
               + kyb[i:i + 512, None] * P[None, :, 1]
               - om[i:i + 512, None] * P[None, :, 2])
        pw[i:i + 512] = (z[None] * np.cos(psi)).sum(1) ** 2 \
            + (z[None] * np.sin(psi)).sum(1) ** 2
    pw = pw / max(pw.max(), 1e-12)
    Phi_tr = design(kxb, kyb, om, P[~vm])
    Phi_v = design(kxb, kyb, om, P[vm])
    ztr, zv = z[~vm], z[vm]
    best = None
    for pexp in (0.5, 1.0, 2.0):
        g = pw ** pexp
        g = g / g.sum()
        gam = np.concatenate([g, g]) * np.var(z)
        for smult in (1.0, 3.0, 10.0):
            s2 = (smult * sigma0) ** 2
            K = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * np.eye(len(ztr))
            fv = (Phi_v * gam[None]) @ (Phi_tr.T @ np.linalg.solve(K, ztr))
            vr = float(np.sqrt(np.mean((fv - zv) ** 2)))
            if best is None or vr < best[0]:
                best = (vr, gam, s2)
    return best[1], best[2]


class StreamMMSE:
    def __init__(self, kxb, kyb, om, gam, s2, tau=None, dt=0.1, prune=None):
        if prune is not None and prune < len(kxb):
            top = np.argsort(gam[:len(kxb)])[-prune:]
            kxb, kyb, om = kxb[top], kyb[top], om[top]
            gam = np.concatenate([gam[:len(gam) // 2][top],
                                  gam[len(gam) // 2:][top]])
        self.kxb, self.kyb, self.om = kxb, kyb, om
        self.gam, self.s2 = gam, s2
        self.n = len(gam)
        self.w = np.zeros(self.n)
        self.P = np.diag(gam).copy()
        self.a = 1.0 if tau is None else float(np.exp(-dt / tau))

    def step(self, Pf, zf):
        """One frame: AR(1) fade toward prior, then RLS update.
        Returns the normalized innovation squared (NIS, ~1 if
        calibrated) — a causal per-frame health/gate signal."""
        if self.a < 1.0:
            a2 = self.a ** 2
            self.w *= self.a
            self.P *= a2
            self.P[np.diag_indices(self.n)] += (1 - a2) * self.gam
        Phi = design(self.kxb, self.kyb, self.om, Pf)
        PPt = self.P @ Phi.T
        S = Phi @ PPt + self.s2 * np.eye(len(zf))
        Si = np.linalg.inv(S)
        nu = zf - Phi @ self.w
        K = PPt @ Si
        self.w += K @ nu
        self.P -= K @ PPt.T
        self.P = 0.5 * (self.P + self.P.T)   # keep symmetric (300+ steps)
        return float(nu @ Si @ nu / len(zf))

    def rebuild(self, P_obs, z_obs, gam_full, kx_full, ky_full, om_full,
                prune=None):
        """Slow-loop gamma refresh: re-prune the bank by the NEW gamma
        and rebuild (w, P) from a batch solve on the trailing buffer —
        state transfer without filter drift. O(n_obs^3), run ~every 10 s
        (async in a real deployment)."""
        B = len(kx_full)
        if prune is not None and prune < B:
            top = np.argsort(gam_full[:B])[-prune:]
            self.kxb, self.kyb, self.om = (kx_full[top], ky_full[top],
                                           om_full[top])
            self.gam = np.concatenate([gam_full[:B][top],
                                       gam_full[B:][top]])
        else:
            self.kxb, self.kyb, self.om = kx_full, ky_full, om_full
            self.gam = gam_full
        self.n = len(self.gam)
        Phi = design(self.kxb, self.kyb, self.om, P_obs)
        K = (Phi * self.gam[None]) @ Phi.T + self.s2 * np.eye(len(z_obs))
        Ki_z = np.linalg.solve(K, z_obs)
        self.w = self.gam * (Phi.T @ Ki_z)
        Ki_Phi = np.linalg.solve(K, Phi * self.gam[None])
        self.P = np.diag(self.gam) - (Phi * self.gam[None]).T @ Ki_Phi
        self.P = 0.5 * (self.P + self.P.T)

    def predict(self, Q, with_var=False):
        Phi = design(self.kxb, self.kyb, self.om, Q)
        if not with_var:
            return Phi @ self.w, None
        return Phi @ self.w, np.einsum('ij,ij->i', Phi @ self.P, Phi)


def batch_mmse(kxb, kyb, om, gam, s2, P, z, Q):
    Phi = design(kxb, kyb, om, P)
    K = (Phi * gam[None]) @ Phi.T + s2 * np.eye(len(z))
    return (design(kxb, kyb, om, Q) * gam[None]) @ (
        Phi.T @ np.linalg.solve(K, z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episode', default='episode_BF4_15comp_seed0_uniform.h5')
    ap.add_argument('--tau', type=float, default=30.0,
                    help='AR(1) memory time constant [s]; batch-equiv test '
                         'always runs with tau=None')
    ap.add_argument('--prune', type=int, default=400,
                    help='keep top-K bank components by gamma (latency)')
    ap.add_argument('--m_scan_cap', type=int, default=40,
                    help='points kept per frame (Falcon frames are dense)')
    a = ap.parse_args()
    L = EpisodeDataLoader('data/' + a.episode)
    depth, Hs = L.depth, L.Hs_true
    kp = 2 * np.pi / (G * L.Tp_true ** 2 / (2 * np.pi))
    frames = L.get_tracking_frames(0.2)
    dtf = float(frames[1].t - frames[0].t)
    rng = np.random.default_rng(0)

    def frame_obs(f):
        x, y, z_ = f.x_obs.numpy(), f.y_obs.numpy(), f.z_obs.numpy()
        if len(z_) > a.m_scan_cap:
            ii = rng.choice(len(z_), a.m_scan_cap, replace=False)
            x, y, z_ = x[ii], y[ii], z_[ii]
        return np.column_stack([x, y, np.full(len(z_), float(f.t))]), z_

    kxb, kyb, om = make_bank(kp, depth)
    print(f"bank B={len(kxb)} (state n={2 * len(kxb)}), "
          f"episode dt={dtf:.2f}s", flush=True)

    # ---- slow-loop prior from the first 12 s of frames
    n_init = int(round(12.0 / dtf))
    P0, z0 = zip(*[frame_obs(f) for f in frames[:n_init]])
    Pb, zb = np.vstack(P0), np.concatenate(z0)
    ii = np.random.default_rng(1).choice(len(zb), min(600, len(zb)), False)
    gam, s2 = gamma_beamform(kxb, kyb, om, Pb[ii], zb[ii], 0.05 * Hs, depth)
    print(f"slow loop: gamma set from {len(ii)} obs "
          f"(s2={s2:.4f})", flush=True)

    # ---- equivalence test: stream(tau=None) == batch, 20 frames
    st = StreamMMSE(kxb, kyb, om, gam, s2, tau=None)
    Pe, ze = [], []
    for f in frames[:20]:
        Pf, zf = frame_obs(f)
        st.step(Pf, zf)
        Pe.append(Pf)
        ze.append(zf)
    gx = L.gt_grid_x.numpy()[::4]
    gy = L.gt_grid_y.numpy()[::4]
    Xg, Yg = np.meshgrid(gx, gy)
    tq = float(frames[19].t)
    Q = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, tq)])
    f_st, _ = st.predict(Q)
    f_ba = batch_mmse(kxb, kyb, om, gam, s2, np.vstack(Pe),
                      np.concatenate(ze), Q)
    err = float(np.max(np.abs(f_st - f_ba)) / (np.std(f_ba) + 1e-12))
    gt19 = frames[19].gt_field.numpy()[::4, ::4].ravel()
    print(f"EQUIVALENCE stream==batch: max|diff|/std = {err:.2e} "
          f"({'PASS' if err < 1e-6 else 'FAIL'}); "
          f"batch corr vs GT (2s obs) = "
          f"{np.corrcoef(f_ba, gt19)[0, 1]:+.3f}", flush=True)
    assert err < 1e-6

    # ---- control: batch MMSE, PRUNED bank, 12 s of streaming-rate obs
    stp = StreamMMSE(kxb, kyb, om, gam, s2, tau=None, prune=a.prune)
    i_mid = n_init + 60
    Pc, zc = zip(*[frame_obs(f) for f in frames[i_mid - n_init:i_mid]])
    Pc, zc = np.vstack(Pc), np.concatenate(zc)
    if len(zc) > 1500:
        ic = np.random.default_rng(2).choice(len(zc), 1500, replace=False)
        Pc, zc = Pc[ic], zc[ic]
    tq = float(frames[i_mid].t)
    Qc = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, tq)])
    f_c = batch_mmse(stp.kxb, stp.kyb, stp.om, stp.gam, stp.s2, Pc, zc, Qc)
    gt_c = frames[i_mid].gt_field.numpy()[::4, ::4].ravel()
    print(f"CONTROL batch, pruned bank K={a.prune}, {len(zc)} obs/12s: "
          f"corr vs GT = {np.corrcoef(f_c, gt_c)[0, 1]:+.3f}", flush=True)

    # ---- full-episode streaming with finite memory + latency
    st = StreamMMSE(kxb, kyb, om, gam, s2, tau=a.tau, dt=dtf, prune=a.prune)
    gxf = L.gt_grid_x.numpy()[::2]
    gyf = L.gt_grid_y.numpy()[::2]
    Xf, Yf = np.meshgrid(gxf, gyf)
    lat, scores = [], []
    for i, f in enumerate(frames):
        Pf, zf = frame_obs(f)
        t1 = time.perf_counter()
        st.step(Pf, zf)
        lat.append(time.perf_counter() - t1)
        if i >= n_init and (i - n_init) % 30 == 0:
            for dt_f, tag in ((0.0, 'now'), (2.0, 'f2'), (4.0, 'f4')):
                j = i + int(round(dt_f / dtf))
                if j >= len(frames):
                    continue
                Qf = np.column_stack([Xf.ravel(), Yf.ravel(),
                                      np.full(Xf.size, float(frames[j].t))])
                pred, _ = st.predict(Qf)
                gt = frames[j].gt_field.numpy()[::2, ::2].ravel()
                scores.append(dict(
                    i=i, tag=tag,
                    corr=float(np.corrcoef(pred, gt)[0, 1]),
                    nrmse=float(np.sqrt(np.mean((pred - gt) ** 2))
                                / (Hs / 4))))
    lat = np.array(lat) * 1e3
    res = dict(episode=a.episode, tau=a.tau, prune=a.prune,
               B=len(st.kxb), n=st.n,
               lat_ms_median=float(np.median(lat)),
               lat_ms_p95=float(np.percentile(lat, 95)),
               equivalence_err=err, scores=scores)
    for tag in ('now', 'f2', 'f4'):
        cs = [s['corr'] for s in scores if s['tag'] == tag]
        res[f'corr_{tag}'] = float(np.mean(cs))
        print(f"stream corr {tag}: {np.mean(cs):+.3f} over {len(cs)} evals",
              flush=True)
    print(f"latency/frame: median {res['lat_ms_median']:.1f} ms, "
          f"p95 {res['lat_ms_p95']:.1f} ms (state n={st.n}, "
          f"m/frame<={a.m_scan_cap}) -> "
          f"{'PASS' if res['lat_ms_p95'] < 100 else 'FAIL'} 100 ms budget")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
