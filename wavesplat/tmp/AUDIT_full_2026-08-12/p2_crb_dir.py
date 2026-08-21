"""P2 run-0 gate: does the sea direction change the INFORMATION available to the
deployed (stationary) sensor?  Field-nRMSE CRB on the real observation geometry.

Intervention = rotate the true (kx,ky) of a stored episode by Delta.  |k| and omega
are invariant, so this is exactly what `direction_deg` does, with the sensor bit-identical.
"""
import glob
import sys

import h5py
import numpy as np

g = 9.81


def omega_of(k, d):
    return np.sqrt(g * k * np.tanh(k * d))


def dwdk(k, d):
    kd = k * d
    return 0.5 * np.sqrt(g * np.tanh(kd) / k) * (1.0 + 2.0 * kd / np.sinh(2.0 * kd))


def field_crb(xo, yo, to, A, kx, ky, phi, d, gx, gy, tg, sigma):
    """CRB on the field RMSE over the eval grid, params (A_i, phi_i, kx_i, ky_i)."""
    n = len(A)
    k = np.hypot(kx, ky)
    om = omega_of(k, d)
    cg = dwdk(k, d)
    psi = kx[None, :] * xo[:, None] + ky[None, :] * yo[:, None] - om[None, :] * to[:, None] + phi[None, :]
    s, c = np.sin(psi), np.cos(psi)
    # d psi / d kx  =  x - cg*(kx/k)*t   (dispersion hard constraint)
    dpx = xo[:, None] - cg[None, :] * (kx / k)[None, :] * to[:, None]
    dpy = yo[:, None] - cg[None, :] * (ky / k)[None, :] * to[:, None]
    J = np.concatenate([c, -A[None, :] * s, -A[None, :] * s * dpx, -A[None, :] * s * dpy], axis=1)
    F = J.T @ J / sigma ** 2
    # grid Jacobian at a single eval time
    X, Y = np.meshgrid(gx, gy)
    xg, yg = X.ravel(), Y.ravel()
    psg = kx[None, :] * xg[:, None] + ky[None, :] * yg[:, None] - om[None, :] * tg + phi[None, :]
    sg, cg_ = np.sin(psg), np.cos(psg)
    dgx = xg[:, None] - cg[None, :] * (kx / k)[None, :] * tg
    dgy = yg[:, None] - cg[None, :] * (ky / k)[None, :] * tg
    G = np.concatenate([cg_, -A[None, :] * sg, -A[None, :] * sg * dgx, -A[None, :] * sg * dgy], axis=1)
    # ridge for numerical conditioning (same for every arm -> fair comparison)
    lam = 1e-12 * np.trace(F) / F.shape[0]
    C = np.linalg.solve(F + lam * np.eye(F.shape[0]), np.eye(F.shape[0]))
    mse = np.einsum('ij,jk,ik->i', G, C, G).mean()
    return float(np.sqrt(max(mse, 0.0)))


def main():
    pats = [('data_s9_sparse_2026-08-11/*.h5', 'SPARSE'),
            ('data_s9_crowded_2026-08-11/*.h5', 'CROWDED')]
    deltas = [0, 45, 90, 135, 180]
    SIGMA = 0.02  # LiDAR noise floor used throughout the repo
    print(f"{'regime':9s} {'episode':44s} " + " ".join(f"{'d=%d' % dd:>9s}" for dd in deltas) + "   max/min")
    agg = {lab: {dd: [] for dd in deltas} for _, lab in pats}
    for pat, lab in pats:
        for f in sorted(glob.glob(pat))[:6]:
            with h5py.File(f, 'r') as h:
                wc = h['config/wave_components'][:]
                A, om0, kx0, ky0, phi, Q = wc.T
                d = float(h['config'].attrs['depth'])
                P = h['frames/lidar_points'][:]
                M = h['frames/lidar_valid_mask'][:].astype(bool)
                ts = h['frames/timestamps'][:]
                Tp = float(h['config'].attrs['Tp'])
                gx = h['frames/gt_grid_x'][:][::4]
                gy = h['frames/gt_grid_y'][:][::4]
            # deployed initialisation window = tp_factor 2.0 x Tp
            Tw = 2.0 * Tp
            fr = np.where(ts <= Tw)[0]
            xo, yo, to = [], [], []
            for i in fr:
                m = M[i]
                xo.append(P[i][m][:, 0]); yo.append(P[i][m][:, 1])
                to.append(np.full(m.sum(), ts[i]))
            xo = np.concatenate(xo); yo = np.concatenate(yo); to = np.concatenate(to)
            vals = []
            for dd in deltas:
                th = np.radians(dd)
                kx = kx0 * np.cos(th) - ky0 * np.sin(th)
                ky = kx0 * np.sin(th) + ky0 * np.cos(th)
                v = field_crb(xo, yo, to, A, kx, ky, phi, d, gx, gy, Tw, SIGMA)
                vals.append(v)
                agg[lab][dd].append(v)
            r = max(vals) / min(vals)
            print(f"{lab:9s} {f.split('/')[-1][:44]:44s} " +
                  " ".join(f"{v:9.5f}" for v in vals) + f"   {r:6.3f}x")
    print()
    for lab in agg:
        med = [np.median(agg[lab][dd]) for dd in deltas]
        print(f"{lab:9s} median over episodes: " + " ".join(f"{'d%d' % dd}={v:.5f}" for dd, v in zip(deltas, med)) +
              f"   |  180/0 = {med[-1] / med[0]:.4f}x   max/min = {max(med) / min(med):.4f}x")
        # paired sign test 180 vs 0
        a = np.array(agg[lab][0]); b = np.array(agg[lab][180])
        print(f"          paired: 180 better than 0 in {int((b < a).sum())}/{len(a)} episodes; "
              f"median rel change {np.median((b - a) / a) * 100:+.3f}%")


main()
