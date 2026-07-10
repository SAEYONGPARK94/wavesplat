"""G3: search for the sea that breaks WaveSplat, by gradient ascent through it.

Yesterday's "storm = model-class wall" was retracted: five seeds of clean
synthetic storm never reproduced the real record's degradation. That left an
open question with no method attached -- WHICH sea, if any, does the physics
MMSE actually fail on? Every attempt so far guessed a structure (storm,
bimodal, out-of-band) and tested it.

P0 made the solver differentiable and 30-60x faster. So stop guessing. Fix the
sea's total variance and let gradient ascent choose the spectrum that maximises
the estimator's own nRMSE. Whatever it finds is the model class's weak axis,
found by optimisation rather than by intuition.

    eta(x, t) = sum_i a_i cos(k_i . x - omega_i t + phi_i)

with (k_i, omega_i) pinned to the dispersion shell and (a, phi) free. The
amplitudes are renormalised to a fixed variance every step, so the attacker
cannot win by making the sea bigger -- nRMSE is scale-free anyway, but the
observation noise floor is not.

THE CONTROL THAT MATTERS. We already know out-of-band energy is a big lever
(storm -31.6%, bimodal -33.0%). An attacker that simply parks energy outside
the bank's [0.35, 2.5] kp annulus has rediscovered that, not found something
new. So every adversarial sea is scored on:

    in-band energy fraction   vs the same statistic for random seas

  * if the attack's damage tracks out-of-band energy   -> rediscovery
  * if it finds high-nRMSE seas that are still IN band -> a genuinely new
    failure axis, and worth characterising (direction? phase alignment?)

Baseline: 256 random seas at the same variance give the null distribution of
nRMSE. The adversarial sea must clear its 99th percentile to count at all.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu
import d3_static_field as SF

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = 'results/direction3/g3_adversarial_sea.json'
G = 9.81
DEPTH = 45.0
KP = 2 * np.pi / 80.0            # lam_p 80 m
NCOMP = 96
SIGMA = 1.0                      # sea std, m  (Hs = 4 m)
NOISE = 0.02
BAND = (0.35, 2.5)


def sea_shell(rng, n=NCOMP):
    """Candidate components on the dispersion shell, spanning 0.15-5 kp so the
    optimiser is FREE to leave the bank's band if that is the easiest attack."""
    k = KP * np.exp(rng.uniform(np.log(0.15), np.log(5.0), n))
    th = rng.uniform(-np.pi, np.pi, n)
    kx, ky = k * np.cos(th), k * np.sin(th)
    om = np.sqrt(G * k * np.tanh(k * DEPTH))
    return (torch.tensor(kx, device=DEV, dtype=torch.float32),
            torch.tensor(ky, device=DEV, dtype=torch.float32),
            torch.tensor(om, device=DEV, dtype=torch.float32),
            torch.tensor(k, device=DEV, dtype=torch.float32))


def eta(P, kx, ky, om, a, ph):
    psi = (P[:, 0:1] * kx[None] + P[:, 1:2] * ky[None]
           - P[:, 2:3] * om[None] + ph[None])
    return (a[None] * torch.cos(psi)).sum(1)


def norm_amp(raw):
    a = torch.nn.functional.softplus(raw) + 1e-6
    return a * (SIGMA * np.sqrt(2.0) / torch.linalg.norm(a))   # var = sum a^2/2


def geometry(seed=0):
    rng = np.random.default_rng(seed)
    cx, cy = 0.0, 0.0
    hv = np.array([1.0, 0.0])
    P = []
    for t_s in np.arange(-SF.W_OBS, 1e-9, SF.SCAN_DT):
        rr = rng.uniform(*SF.R_FAN, 40)
        aa = rng.uniform(-SF.HALF_FOV, SF.HALF_FOV, 40)
        P.append(np.column_stack([cx + rr * np.cos(aa), cy + rr * np.sin(aa),
                                  np.full(40, t_s)]))
    P = np.vstack(P)
    P = P[rng.choice(len(P), SF.M_CAP, replace=False)]
    gx = np.arange(SF.DOM_X[0], SF.DOM_X[1], 6.0)
    gy = np.arange(SF.DOM_Y[0], SF.DOM_Y[1], 6.8)
    X, Y = np.meshgrid(gx, gy)
    ins, _ = SF.fan_mask(X, Y, cx, cy, hv)
    Pq = np.column_stack([X.ravel()[ins.ravel()], Y.ravel()[ins.ravel()],
                          np.zeros(ins.sum())])
    return (torch.tensor(P, device=DEV, dtype=torch.float32),
            torch.tensor(Pq, device=DEV, dtype=torch.float32))


def nrmse(P, Pq, kx, ky, om, a, ph, gen):
    z = eta(P, kx, ky, om, a, ph)
    z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
    gt = eta(Pq, kx, ky, om, a, ph)
    pr = mmse_gpu(P, z, DEPTH, Pq, sigma0=0.05 * 4 * SIGMA, kp_hint=KP,
                  band=BAND)
    # Normalise by the sea's TRUE std, not by the realised std over the query
    # points. The first version divided by gt.std(), and gradient ascent
    # promptly drove that denominator to zero: it pushed the energy to
    # lam >> 200 m, where the field inside the fan is nearly a plane, and
    # "won" nRMSE = 72 without the estimator ever failing. That attacks the
    # metric, not the model class.
    return torch.sqrt(((pr - gt) ** 2).mean()) / SIGMA


def in_band(a, k):
    e = a ** 2
    m = (k > BAND[0] * KP) & (k < BAND[1] * KP)
    return float((e[m].sum() / e.sum()).detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--restarts', type=int, default=4)
    ap.add_argument('--null', type=int, default=256)
    a_ = ap.parse_args()

    P, Pq = geometry()
    print(f"fan observations {len(P)}   query points {len(Pq)}", flush=True)
    gen = torch.Generator(device=DEV).manual_seed(0)

    # ---- null distribution: random seas at the same variance ---------------
    rng = np.random.default_rng(1)
    null, nb = [], []
    t0 = time.time()
    for i in range(a_.null):
        kx, ky, om, k = sea_shell(rng)
        raw = torch.tensor(rng.normal(0, 1, NCOMP), device=DEV,
                           dtype=torch.float32)
        ph = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                          dtype=torch.float32)
        with torch.no_grad():
            amp = norm_amp(raw)
            null.append(float(nrmse(P, Pq, kx, ky, om, amp, ph, gen)))
            nb.append(in_band(amp, k))
        if (i + 1) % 64 == 0:
            print(f"  null {i+1}/{a_.null}  ({time.time()-t0:.0f} s)", flush=True)
    null, nb = np.array(null), np.array(nb)
    q99 = float(np.quantile(null, 0.99))
    print(f"\nnull nRMSE  mean {null.mean():.3f}  sd {null.std():.3f}  "
          f"p99 {q99:.3f}   in-band {nb.mean():.3f}+-{nb.std():.3f}", flush=True)
    print(f"null corr(nRMSE, in-band energy) = "
          f"{np.corrcoef(null, nb)[0,1]:+.3f}", flush=True)

    # ---- gradient ascent ---------------------------------------------------
    best = []
    for r in range(a_.restarts):
        rng = np.random.default_rng(100 + r)
        kx, ky, om, k = sea_shell(rng)
        raw = torch.tensor(rng.normal(0, 1, NCOMP), device=DEV,
                           dtype=torch.float32, requires_grad=True)
        ph = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                          dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([raw, ph], lr=0.05)
        hist = []
        for s in range(a_.steps):
            amp = norm_amp(raw)
            loss = -nrmse(P, Pq, kx, ky, om, amp, ph, gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (s + 1) % 50 == 0:
                hist.append(float(-loss))
                print(f"  restart {r} step {s+1:4d}  nRMSE {-float(loss):.3f}  "
                      f"in-band {in_band(norm_amp(raw), k):.3f}", flush=True)
        with torch.no_grad():
            amp = norm_amp(raw)
            v = float(nrmse(P, Pq, kx, ky, om, amp, ph, gen))
            ib = in_band(amp, k)
            gtsd = float(eta(Pq, kx, ky, om, amp, ph).std())
            # Does the posterior know it is wrong? z-std = err / predictive sd;
            # a calibrated posterior gives ~1. Hand-built out-of-band seas
            # reach nRMSE 1.6 at most, so whatever the optimiser found is not
            # band placement alone -- record the spectrum so it can be read.
            z = eta(P, kx, ky, om, amp, ph)
            z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
            gt = eta(Pq, kx, ky, om, amp, ph)
            pr, var = mmse_gpu(P, z, DEPTH, Pq, sigma0=0.05 * 4 * SIGMA,
                               kp_hint=KP, band=BAND, return_var=True)
            sd = torch.sqrt(var.clamp_min(1e-12))
            zstd = float(((pr - gt) / sd).std())
            psd = float(sd.mean())
            e = (amp ** 2).detach()
            kk = k.detach()
            lo = float((e[kk < BAND[0] * KP].sum() / e.sum()))
            hi = float((e[kk > BAND[1] * KP].sum() / e.sum()))
        best.append({'restart': r, 'nrmse': v, 'in_band': ib,
                     'gt_std_in_fan': gtsd, 'z_std': zstd, 'pred_sd': psd,
                     'energy_below_band': lo, 'energy_above_band': hi,
                     'hist': hist})
        print(f"    z-std {zstd:.2f}  pred sd {psd:.3f} m   "
              f"energy below band {lo:.3f}  above band {hi:.3f}", flush=True)
        print(f"restart {r}:  nRMSE {v:.3f}  (null p99 {q99:.3f})  "
              f"in-band {ib:.3f}  (null {nb.mean():.3f})  "
              f"gt std in fan {gtsd:.3f} m  (sea std {SIGMA:.2f})", flush=True)

    top = max(best, key=lambda d: d['nrmse'])
    beats = top['nrmse'] > q99
    band_expl = top['in_band'] < nb.mean() - 2 * nb.std()

    print(f"\n=== G3 ===")
    print(f"  best adversarial nRMSE {top['nrmse']:.3f}   null p99 {q99:.3f}   "
          f"{'CLEARS' if beats else 'does not clear'} the null")
    print(f"  its in-band energy {top['in_band']:.3f}   null "
          f"{nb.mean():.3f}+-{nb.std():.3f}")
    if beats and band_expl:
        print("  -> the attack parked energy OUT of band: rediscovery of the "
              "band lever, not a new axis.")
    elif beats:
        print("  -> a high-nRMSE sea that is still IN band: a failure axis the "
              "band story does not explain. Characterise it.")
    else:
        print("  -> gradient ascent found nothing beyond the random null. The "
              "model class has no sharp weak axis at this variance.")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'null_mean': float(null.mean()), 'null_sd': float(null.std()),
                   'null_p99': q99, 'null_inband_mean': float(nb.mean()),
                   'null_inband_sd': float(nb.std()),
                   'null_corr_nrmse_inband': float(np.corrcoef(null, nb)[0, 1]),
                   'restarts': best, 'beats_null': bool(beats),
                   'explained_by_band': bool(band_expl)}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
