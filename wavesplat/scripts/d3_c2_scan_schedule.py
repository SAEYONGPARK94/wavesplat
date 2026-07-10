"""C2: the axis C1 left open -- spend the budget on time, or on space?

C1 optimised the fan's range and field of view through the differentiable
solver and found the Falcon's own hardware corner. It could not touch the one
knob that costs nothing to turn: given 600 returns in a 20 s window, how many
scans should they be split into?

  few scans, dense    each scan maps the fan well; the window is sampled coarsely
  many scans, sparse  each scan is a thin sketch; the wave is followed in time

The trade is between spatial aperture and temporal aperture, and E3 just showed
that the temporal one carries information the spatial one does not -- a static
bias cannot survive a time-forward test because waves propagate and it does not.
So the sweep is not merely engineering. It asks how much of what we know comes
from watching the sea move.

Budget, window, fan and scored domain are all held fixed. Only the schedule
moves. Twelve seas per cell, scored in-fan against the truth, normalised so the
zero-predictor is 1.000.

Also reported: the same sweep under the injected range bias, because if the
temporal aperture is what discriminates, more scans should help there MORE.
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
from d3_l1_learned_prior import S0
from d3_o1_defect_ablation import synth_sea, corr_noise
from d3_g3_adversarial_sea import eta, DEV, DEPTH, KP, SIGMA, NOISE
import d3_static_field as SF

OUT = 'results/direction3/c2_scan_schedule.json'
BUDGET = 600
W_OBS = 20.0
SCANS = (2, 5, 10, 20, 40, 80, 200)
BAND = (0.15, 2.5)
LEADS = (0.0, 5.0)   # score the field at t0 = 0 (nowcast) and at +5 s (forecast)


def geometry(n_scan, rng):
    per = BUDGET // n_scan
    if per < 1:
        return None
    ts = np.repeat(np.linspace(-W_OBS, 0.0, n_scan)[:, None], per, 1).ravel()
    rr = rng.uniform(*SF.R_FAN, len(ts))
    aa = rng.uniform(-SF.HALF_FOV, SF.HALF_FOV, len(ts))
    P = np.column_stack([rr * np.cos(aa), rr * np.sin(aa), ts])
    return torch.tensor(P, device=DEV, dtype=torch.float32)


def query(lead):
    """The scored instant.

    DESIGN LEAK, caught after the first run (#29). Scan times are
    linspace(-W, 0), so the LAST scan always lands exactly on the nowcast
    instant t = 0. With two scans, 300 of the 600 returns sit at the scored
    time; with forty, only fifteen do. The sweep then measures "how many points
    are at the scored instant", not how the budget should be spread in time,
    and it duly reported that two scans beat forty by 0.072.

    A forecast lead removes the coincidence: no schedule has an observation at
    the scored instant, and predicting it requires the dispersion relation,
    which is what a temporal aperture buys.
    """
    gx = np.arange(SF.DOM_X[0], SF.DOM_X[1], 8.0)
    gy = np.arange(SF.DOM_Y[0], SF.DOM_Y[1], 8.0)
    X, Y = np.meshgrid(gx, gy)
    ins, _ = SF.fan_mask(X, Y, 0.0, 0.0, np.array([1.0, 0.0]))
    return torch.tensor(np.column_stack([X.ravel()[ins.ravel()],
                                         Y.ravel()[ins.ravel()],
                                         np.full(ins.sum(), lead)]),
                        device=DEV, dtype=torch.float32)


def run_cell(n_scan, cfg, n_sea, gen, Pq):
    vals = []
    for i in range(n_sea):
        rng = np.random.default_rng(9000 + i)
        s = synth_sea(rng)
        P = geometry(n_scan, rng)
        if P is None:
            return None
        kx, ky, om, a, ph = s
        with torch.no_grad():
            z = eta(P, kx, ky, om, a, ph)
            if cfg == 'bias':
                r = torch.hypot(P[:, 0], P[:, 1])
                z = z + 0.5 * SIGMA * (r / 200.0) ** 2
            z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
            gt = eta(Pq, kx, ky, om, a, ph)
            gt = gt - gt.mean()
            pr = mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP, band=BAND)
            pr = pr - pr.mean()
            vals.append(float(torch.sqrt(((pr - gt) ** 2).mean()) / gt.std()))
    v = np.array(vals)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seas', type=int, default=12)
    a_ = ap.parse_args()
    gen = torch.Generator(device=DEV).manual_seed(0)
    print(f"budget {BUDGET} returns, window {W_OBS:.0f} s, fan "
          f"{SF.R_FAN[0]:.0f}-{SF.R_FAN[1]:.0f} m\n")

    res = {}
    for lead in LEADS:
        Pq = query(lead)
        tag = 'nowcast t=0' if lead == 0 else f'forecast t=+{lead:.0f}s'
        print(f"--- {tag} ---")
        print(f"{'scans':>7}{'pts/scan':>10}{'clean':>18}{'bias':>18}")
        r = {}
        for ns in SCANS:
            a = run_cell(ns, 'none', a_.seas, gen, Pq)
            b = run_cell(ns, 'bias', a_.seas, gen, Pq)
            if a is None:
                continue
            print(f"{ns:>7}{BUDGET//ns:>10}"
                  f"{a[0]:>12.3f}±{a[1]:.3f}{b[0]:>12.3f}±{b[1]:.3f}",
                  flush=True)
            r[ns] = dict(clean=a[0], clean_se=a[1], bias=b[0], bias_se=b[1])
        ks = sorted(r)
        bc = min(ks, key=lambda k: r[k]['clean'])
        bb = min(ks, key=lambda k: r[k]['bias'])
        print(f"  best clean {bc} scans -> {r[bc]['clean']:.3f}   "
              f"best bias {bb} scans -> {r[bb]['bias']:.3f}   "
              f"Falcon-like 40: {r[40]['clean']:.3f} / {r[40]['bias']:.3f}\n",
              flush=True)
        res[tag] = r

    print("1.000 = as good as predicting zero.")
    print("the nowcast column is CONFOUNDED (#29): the last scan lands on the")
    print("scored instant, so few scans = many points at t=0. Read the forecast.")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({t: {str(k): {kk: float(vv) for kk, vv in v.items()}
                       for k, v in r.items()} for t, r in res.items()}, f,
                  indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
