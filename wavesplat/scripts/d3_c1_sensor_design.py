"""C1: stop asking what a given sensor can do. Ask what sensor the task requires.

The thesis frame is "given the Falcon LiDAR, forecast the encounter". Every
result so far takes the sensor as fixed: FOV +-60 deg, range 15-200 m, and the
estimator does what it can with 600 returns. The open question in the direction
note is the inverse one -- requirement-inverted observation specification --
and until now there was no way to answer it, because the estimator was a
non-differentiable numpy routine.

It is differentiable now, and verified against it (P0: |diff| = 0.0000). So the
sensor's own parameters can be optimised by gradient descent through the wave
estimator itself.

The trick is to reparameterise the sampling so the geometry has a gradient:

    r = R0 + (R1 - R0) . u        u, v ~ U(0,1), DRAWN ONCE AND FROZEN
    a = phi . (2v - 1)
    x = cx + r cos a,   y = cy + r sin a

Now (R0, R1, phi) are continuous knobs, the observations z = eta(x, y, t) are
differentiable in them, and the solver carries the gradient to the field error.

THE TRAP, and it is the same one that beat me twice today (errors #16, #18):
if the scored domain follows the fan, the optimiser wins by SHRINKING WHAT IT
IS GRADED ON. A 5 deg fan scoring a 5 deg wedge is trivially accurate and
useless. So the scored domain is the canonical rectangle, FIXED, independent of
every sensor parameter. The sensor must estimate the same patch of ocean no
matter how it looks at it.

Held fixed, because they are the real engineering constraints:
    observation budget      600 returns   (points per second x window)
    observation window      20 s
    hardware limits         R0 >= 5 m, R1 <= 250 m, 5 deg <= phi <= 90 deg

Optimised: R0, R1, phi.
Reported against the Falcon's actual spec (phi = 60 deg, R = 15-200 m).
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
from d3_g3_adversarial_sea import (eta, norm_amp, sea_shell, DEV, DEPTH, KP,
                                   NCOMP, SIGMA, NOISE)
import d3_static_field as SF

OUT = 'results/direction3/c1_sensor_design.json'
S0 = 0.05 * 4 * SIGMA
BAND = (0.15, 2.5)               # the band the adaptive selector converges to
BUDGET = 600
W_OBS = 20.0
N_SCAN = 40
R_MIN, R_MAX = 5.0, 250.0
PHI_MIN, PHI_MAX = np.deg2rad(5.0), np.deg2rad(90.0)
FALCON = (15.0, 200.0, np.deg2rad(60.0))


def scored_domain():
    """The canonical rectangle. Fixed. The sensor does not get to choose it."""
    gx = np.arange(SF.DOM_X[0], SF.DOM_X[1], 8.0)
    gy = np.arange(SF.DOM_Y[0], SF.DOM_Y[1], 8.0)
    X, Y = np.meshgrid(gx, gy)
    return torch.tensor(np.column_stack([X.ravel(), Y.ravel(),
                                         np.zeros(X.size)]),
                        device=DEV, dtype=torch.float32)


def frozen_draws(seed=0):
    rng = np.random.default_rng(seed)
    per = BUDGET // N_SCAN
    u = torch.tensor(rng.uniform(0, 1, (N_SCAN, per)), device=DEV,
                     dtype=torch.float32)
    v = torch.tensor(rng.uniform(0, 1, (N_SCAN, per)), device=DEV,
                     dtype=torch.float32)
    t = torch.tensor(np.repeat(np.linspace(-W_OBS, 0.0, N_SCAN)[:, None],
                               per, 1), device=DEV, dtype=torch.float32)
    return u, v, t


def fan(u, v, t, R0, R1, phi):
    r = R0 + (R1 - R0) * u
    a = phi * (2 * v - 1)
    return torch.stack([(r * torch.cos(a)).reshape(-1),
                        (r * torch.sin(a)).reshape(-1),
                        t.reshape(-1)], 1)


def unpack(theta):
    """Keep the spec inside the hardware box, differentiably."""
    R0 = R_MIN + (R_MAX - R_MIN - 10.0) * torch.sigmoid(theta[0])
    R1 = R0 + 10.0 + (R_MAX - R0 - 10.0) * torch.sigmoid(theta[1])
    phi = PHI_MIN + (PHI_MAX - PHI_MIN) * torch.sigmoid(theta[2])
    return R0, R1, phi


def loss_on(seas, Pq, u, v, t, R0, R1, phi, gen):
    P = fan(u, v, t, R0, R1, phi)
    tot = 0.0
    for kx, ky, om, a, ph in seas:
        z = eta(P, kx, ky, om, a, ph)
        z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
        gt = eta(Pq, kx, ky, om, a, ph)
        gt = gt - gt.mean()
        pr = mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP, band=BAND)
        pr = pr - pr.mean()
        tot = tot + torch.sqrt(((pr - gt) ** 2).mean()) / gt.std()
    return tot / len(seas)


def draw_seas(rng, n):
    out = []
    for _ in range(n):
        kx, ky, om, k = sea_shell(rng)
        a = norm_amp(torch.tensor(rng.normal(0, 1, NCOMP), device=DEV,
                                  dtype=torch.float32))
        ph = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                          dtype=torch.float32)
        out.append((kx, ky, om, a, ph))
    return out


def evaluate(spec, Pq, u, v, t, gen, n=48, seed=99):
    R0, R1, phi = [torch.tensor(float(x), device=DEV) for x in spec]
    rng = np.random.default_rng(seed)
    vals = []
    with torch.no_grad():
        for s in draw_seas(rng, n):
            vals.append(float(loss_on([s], Pq, u, v, t, R0, R1, phi, gen)))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=250)
    ap.add_argument('--seas', type=int, default=4)
    a_ = ap.parse_args()

    Pq = scored_domain()
    u, v, t = frozen_draws()
    gen = torch.Generator(device=DEV).manual_seed(0)
    print(f"scored domain: {len(Pq)} points on the fixed rectangle "
          f"x[{SF.DOM_X[0]:.0f},{SF.DOM_X[1]:.0f}] y[{SF.DOM_Y[0]:.0f},"
          f"{SF.DOM_Y[1]:.0f}]")
    print(f"budget {BUDGET} returns over {W_OBS:.0f} s, {N_SCAN} scans\n")

    base, base_sd = evaluate(FALCON, Pq, u, v, t, gen)
    print(f"Falcon spec   R {FALCON[0]:.0f}-{FALCON[1]:.0f} m, "
          f"phi {np.rad2deg(FALCON[2]):.0f} deg   ->  nRMSE/zero "
          f"{base:.4f} ± {base_sd:.4f}\n")

    theta = torch.zeros(3, device=DEV, requires_grad=True)
    with torch.no_grad():                       # start at the Falcon spec
        theta[0] = np.log(1e-9 + (15.0 - R_MIN) / (R_MAX - R_MIN - 10.0 -
                                                   (15.0 - R_MIN)))
        theta[1] = 0.0
        theta[2] = 0.0
    opt = torch.optim.Adam([theta], lr=0.05)
    rng = np.random.default_rng(1)
    hist = []
    for s in range(a_.steps):
        R0, R1, phi = unpack(theta)
        L = loss_on(draw_seas(rng, a_.seas), Pq, u, v, t, R0, R1, phi, gen)
        opt.zero_grad(set_to_none=True)
        L.backward()
        opt.step()
        if (s + 1) % 25 == 0:
            with torch.no_grad():
                R0, R1, phi = unpack(theta)
            hist.append(dict(step=s + 1, loss=float(L), R0=float(R0),
                             R1=float(R1), phi_deg=float(np.rad2deg(
                                 phi.detach().cpu()))))
            print(f"  step {s+1:4d}  loss {float(L):.4f}   R {float(R0):.1f}-"
                  f"{float(R1):.1f} m   phi {np.rad2deg(float(phi)):.1f} deg",
                  flush=True)

    with torch.no_grad():
        R0, R1, phi = unpack(theta)
    spec = (float(R0), float(R1), float(phi))
    opt_v, opt_sd = evaluate(spec, Pq, u, v, t, gen)

    print(f"\ndesigned spec R {spec[0]:.1f}-{spec[1]:.1f} m, "
          f"phi {np.rad2deg(spec[2]):.1f} deg  ->  nRMSE/zero "
          f"{opt_v:.4f} ± {opt_sd:.4f}")
    print(f"Falcon                                        -> {base:.4f}")
    print(f"improvement {100*(base-opt_v)/base:+.1f}%   "
          f"(held-out seas, same budget of {BUDGET} returns)")

    # what does each knob buy on its own?
    print("\none-at-a-time, from the Falcon spec:")
    for i, name, vals in ((0, 'R0 (m)', [5, 10, 15, 30, 60]),
                          (1, 'R1 (m)', [100, 150, 200, 250]),
                          (2, 'phi (deg)', [15, 30, 45, 60, 90])):
        line = []
        for x in vals:
            s = list(FALCON)
            s[i] = np.deg2rad(x) if i == 2 else float(x)
            if s[1] <= s[0] + 10:
                continue
            m, _ = evaluate(tuple(s), Pq, u, v, t, gen, n=24)
            line.append(f"{x}: {m:.3f}")
        print(f"  {name:<11} " + "   ".join(line))

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'falcon': dict(R0=FALCON[0], R1=FALCON[1],
                                  phi_deg=float(np.rad2deg(FALCON[2])),
                                  score=base, sd=base_sd),
                   'designed': dict(R0=spec[0], R1=spec[1],
                                    phi_deg=float(np.rad2deg(spec[2])),
                                    score=opt_v, sd=opt_sd),
                   'budget': BUDGET, 'history': hist}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
