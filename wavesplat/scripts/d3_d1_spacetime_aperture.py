"""D1: is the wall spatial aperture, or space-time aperture?

All day the story was "waves longer than the 185 m aperture are not in the
model, and cannot be". But a plane wave's phase at a sample is k.x - omega.t,
and we sample over 20 s as well as over 185 m. A 566 m swell in 45 m of water
travels at c = 21 m/s: during the observation window it moves 420 m through
the fan. The solver already uses (x, y, t).

So the aperture that matters should be the SPACE-TIME one. Identifying (|k|,
direction) needs the sampled phase to vary by order 2 pi across the data:

    k.L + omega.T  >~  2 pi        i.e.        L/lambda + T/T_wave  >~  1

If that is the right criterion, then the bank's floor is not set by the fan's
length at all -- it is set by how long we look. Widening the band worked today
(G5, A1b, C1) because the time dimension was already supplying the aperture and
we had simply cut those atoms out of the model.

TEST. A single long-crested wave of wavelength lambda, direction unknown to the
estimator. The bank is an annulus [0.35, 2.5] k centred on it, dk = k/10, so
the estimator must find the direction and the exact wavenumber among ~1900
atoms -- it is not handed the answer. Sweep lambda and the observation window T
at a FIXED budget of 600 returns, so a longer window buys time-aperture and
pays in nothing but revisit rate.

  if the wall is spatial      recovery depends on L/lambda alone; the T columns
                              are flat
  if the wall is space-time   recovery follows L/lambda + T/T_wave, and a long
                              window rescues a wave the fan cannot span

The scored domain is the fixed canonical rectangle, never a function of T.
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

from d3_gpu_mmse import mmse_gpu
import d3_static_field as SF

OUT = 'results/direction3/d1_spacetime_aperture.json'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = 9.81
DEPTH = 45.0
BUDGET = 600
N_SCAN = 40
LAMBDAS = [100.0, 200.0, 300.0, 400.0, 600.0, 800.0]
WINDOWS = [2.5, 5.0, 10.0, 20.0, 40.0]
SEEDS = (0, 1, 2)
AMP = 1.0
NOISE = 0.02
L_SPACE = SF.R_FAN[1] - SF.R_FAN[0]        # 185 m


def wave(lam):
    k = 2 * np.pi / lam
    om = np.sqrt(G * k * np.tanh(k * DEPTH))
    return k, om, 2 * np.pi / om, om / k          # k, omega, period, c_phase


def observe(rng, T):
    """Fan samples over a window of length T. Budget fixed: only the revisit
    rate changes, never the number of returns."""
    per = BUDGET // N_SCAN
    ts = np.repeat(np.linspace(-T, 0.0, N_SCAN)[:, None], per, 1).ravel()
    rr = rng.uniform(*SF.R_FAN, BUDGET)
    aa = rng.uniform(-SF.HALF_FOV, SF.HALF_FOV, BUDGET)
    return np.column_stack([rr * np.cos(aa), rr * np.sin(aa), ts])


def query():
    gx = np.arange(SF.DOM_X[0], SF.DOM_X[1], 8.0)
    gy = np.arange(SF.DOM_Y[0], SF.DOM_Y[1], 8.0)
    X, Y = np.meshgrid(gx, gy)
    return np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])


def run(lam, T, seed):
    k, om, per, c = wave(lam)
    rng = np.random.default_rng(1000 + seed)
    th = rng.uniform(-np.pi, np.pi)
    kx, ky = k * np.cos(th), k * np.sin(th)
    phi = rng.uniform(-np.pi, np.pi)

    P = observe(rng, T)
    Pq = query()
    eta = lambda Q: AMP * np.cos(kx * Q[:, 0] + ky * Q[:, 1] - om * Q[:, 2]
                                 + phi)
    z = eta(P) + rng.normal(0, NOISE, len(P))
    gt = eta(Pq)

    t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    pr = mmse_gpu(t(P), t(z), DEPTH, t(Pq), sigma0=0.05 * 2 * AMP,
                  kp_hint=k, band=(0.35, 2.5))
    pr = (pr - pr.mean()).cpu().numpy()
    g = gt - gt.mean()
    return float(np.sqrt(np.mean((pr - g) ** 2)) / g.std())


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    Pq = query()
    print(f"spatial aperture L = {L_SPACE:.0f} m, budget {BUDGET} returns, "
          f"scored on {len(Pq)} fixed points\n")

    hdr = ''.join(f"{f'T={T:g}s':>10}" for T in WINDOWS)
    print(f"{'lambda':>8}{'T_wave':>9}{'c':>7}{'L/lam':>8}{hdr}")
    res = []
    for lam in LAMBDAS:
        k, om, per, c = wave(lam)
        row = f"{lam:>8.0f}{per:>9.1f}{c:>7.1f}{L_SPACE/lam:>8.2f}"
        for T in WINDOWS:
            v = np.mean([run(lam, T, s) for s in SEEDS])
            row += f"{v:>10.3f}"
            res.append(dict(lam=lam, T=T, T_wave=per, c=c,
                            spatial=L_SPACE / lam, temporal=T / per,
                            total=L_SPACE / lam + T / per, nrmse=v))
        print(row, flush=True)

    print("\n1.000 = as good as predicting zero.  Bank is centred on the true "
          "k,\nso failure here is identifiability, not representation.\n")

    a = np.array([[d['spatial'], d['temporal'], d['total'], d['nrmse']]
                  for d in res])
    ok = a[:, 3] < 0.5
    print(f"corr(nRMSE, L/lambda)              = "
          f"{np.corrcoef(a[:,0], a[:,3])[0,1]:+.3f}")
    print(f"corr(nRMSE, T/T_wave)              = "
          f"{np.corrcoef(a[:,1], a[:,3])[0,1]:+.3f}")
    print(f"corr(nRMSE, L/lambda + T/T_wave)   = "
          f"{np.corrcoef(a[:,2], a[:,3])[0,1]:+.3f}   <- the space-time claim")
    print(f"\nrecovered (nRMSE < 0.5): {ok.sum()}/{len(a)} cells")
    if ok.any():
        print(f"  their L/lambda + T/T_wave: min {a[ok,2].min():.2f}  "
              f"median {np.median(a[ok,2]):.2f}")
    if (~ok).any():
        print(f"  failed cells:              max {a[~ok,2].max():.2f}")
    sp_only = a[(a[:, 0] < 0.5) & ok]
    print(f"\ncells with L/lambda < 0.5 (fan cannot span the wave) that are "
          f"still recovered: {len(sp_only)}")
    if len(sp_only):
        print("  -> the spatial aperture is NOT the wall. Time supplies it.")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
