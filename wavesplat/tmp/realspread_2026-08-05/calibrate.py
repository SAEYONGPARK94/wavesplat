"""Instrument response of the 3-D-FFT sigma_1 estimator: KNOWN spread in, measured out.

A finite patch of size L smears a plane wave over dk ~ 2pi/L in wavenumber, i.e.
over ~lambda/L in angle.  So a *finite-aperture* directional estimate is biased
HIGH relative to a buoy, which has no aperture.  This script measures that bias
as a function of the only dimensionless number that controls it, L/lambda.

Truth model: Longuet-Higgins cos^{2s}(theta/2), for which r1 = s/(s+1) exactly,
hence sigma_1_true = sqrt(2/(s+1)) rad -- an analytic handle on the input.

Synthesis trick: all components share one omega, so
    eta(t,x,y) = cos(wt) Re{Psi} + sin(wt) Im{Psi},  Psi = sum_j A_j e^{i(k_j.x+phi_j)}
which is rank-2 in time.  Cost is one outer product, not N_dir field evaluations.
"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spread_est import analyse_block, moments_from_E, band_summary, spread_from_moments, G

OUT = os.path.dirname(os.path.abspath(__file__))
N = 128            # grid points per axis
NT = 1024
DT = 0.1
LAM = 40.0         # m, carrier wavelength (f0 = 0.198 Hz, inside 0.04-0.5 band)
NDIR = 360
SEED = 20260805


def sigma_to_s(sig_deg):
    if sig_deg <= 0:
        return np.inf
    s = 2.0 / np.radians(sig_deg) ** 2 - 1.0
    return max(s, 1e-3)


def make_field(nx, ny, dx, dy, nt, dt, lam, sig_deg, theta0, rng, snr_db=None,
               n_freq=1, bw=0.0):
    k0 = 2 * np.pi / lam
    w0 = np.sqrt(G * k0)
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    s = sigma_to_s(sig_deg)
    th = np.linspace(-np.pi, np.pi, NDIR, endpoint=False)
    if np.isinf(s):
        D = np.zeros(NDIR); D[np.argmin(abs(th - theta0))] = 1.0
    else:
        d = np.cos(0.5 * (th - theta0)) ** 2
        D = np.power(np.clip(d, 0, None), s)
        D /= D.sum()
    A = np.sqrt(D)
    fr = [1.0] if n_freq == 1 else np.linspace(1 - bw, 1 + bw, n_freq)
    eta = np.zeros((nt, nx, ny), np.float32)
    tt = np.arange(nt) * dt
    for m in fr:
        k = k0 * m ** 2                # keep deep-water tie: w = sqrt(gk)
        w = np.sqrt(G * k)
        ph = rng.uniform(0, 2 * np.pi, NDIR)
        kx = k * np.cos(th); ky = k * np.sin(th)
        Psi = np.zeros((nx, ny), complex)
        for j in range(NDIR):
            if A[j] <= 0:
                continue
            Psi += A[j] * np.exp(1j * (kx[j] * x[:, None] + ky[j] * y[None, :] + ph[j]))
        Psi /= np.sqrt(len(fr))
        eta += (np.cos(w * tt)[:, None, None] * Psi.real[None].astype(np.float32) +
                np.sin(w * tt)[:, None, None] * Psi.imag[None].astype(np.float32)).astype(np.float32)
    if snr_db is not None:
        p = float(np.var(eta))
        eta += rng.normal(0, np.sqrt(p * 10 ** (-snr_db / 10)), eta.shape).astype(np.float32)
    # analytic truth from the SAME discrete direction set (quadrature-exact)
    E = D
    a1 = float((E * np.cos(th)).sum() / E.sum())
    b1 = float((E * np.sin(th)).sum() / E.sum())
    return eta, float(spread_from_moments(a1, b1))


def run(nx, ny, L_over_lam, sig_deg, theta0=0.6, snr_db=None, n_freq=1, bw=0.0,
        seed=SEED):
    rng = np.random.default_rng(seed)
    L = LAM * L_over_lam
    dx = L / (nx - 1)
    eta, sig_true = make_field(nx, ny, dx, dx, NT, DT, LAM, sig_deg, theta0, rng,
                               snr_db=snr_db, n_freq=n_freq, bw=bw)
    f, kx, ky, Esp = analyse_block(eta, DT, dx, dx)
    out = {}
    for tag, shell in (('shell', True), ('allk', False)):
        a1, b1, S, ridge, onsh = moments_from_E(f, kx, ky, Esp, use_shell=shell,
                                                depth=None, Lmin=L)
        out[tag] = band_summary(f, a1, b1, S)
    return dict(L_over_lam=L_over_lam, dx=dx, samples_per_lam=LAM / dx,
                sigma_true_deg=sig_true, sigma_nominal_deg=sig_deg,
                snr_db=snr_db, n_freq=n_freq,
                sigma_meas_band=out['shell']['spread_band_deg'],
                sigma_meas_total=out['shell']['spread_total_deg'],
                sigma_meas_allk_total=out['allk']['spread_total_deg'])


def main():
    res = []
    ratios = [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    sigs = [0, 10, 20, 30, 40, 50]
    for r in ratios:
        for s in sigs:
            d = run(N, N, r, s)
            res.append(d)
            print(f"L/lam={r:5.1f} sig_true={d['sigma_true_deg']:6.2f} -> "
                  f"band={d['sigma_meas_band']:6.2f} total={d['sigma_meas_total']:6.2f} "
                  f"allk={d['sigma_meas_allk_total']:6.2f} (n/lam={d['samples_per_lam']:.1f})",
                  flush=True)
    # --- secondary: seed sensitivity, noise, finite bandwidth ---------------
    extra = []
    for sd in (1, 2, 3):
        extra.append(dict(kind='seed', **run(N, N, 4, 20, seed=SEED + sd)))
    for snr in (20, 10, 5):
        extra.append(dict(kind='noise', **run(N, N, 4, 20, snr_db=snr)))
    for nf, bw in ((3, 0.06), (5, 0.10)):
        extra.append(dict(kind='bandwidth', **run(N, N, 4, 20, n_freq=nf, bw=bw)))
    for e in extra:
        print(f"[{e['kind']}] L/lam={e['L_over_lam']} snr={e['snr_db']} nf={e['n_freq']} "
              f"true={e['sigma_true_deg']:.2f} -> total={e['sigma_meas_total']:.2f} "
              f"allk={e['sigma_meas_allk_total']:.2f}", flush=True)
    json.dump({'grid': N, 'nt': NT, 'dt': DT, 'lambda_m': LAM, 'ndir': NDIR,
               'main': res, 'extra': extra},
              open(os.path.join(OUT, 'calibration.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
