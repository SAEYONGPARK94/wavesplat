"""O1: put the ocean's defects into the simulator, one at a time.

N1 ended on a wall that is not a wall. The critic could not learn to correct
the residual rule's bias because, on our synthetic seas, THE RESIDUAL RULE HAS
NO BIAS: its regret is 0.0067, three and a half percent of the label spread.
The same rule leaks +0.016 on real records. You cannot learn to fix a defect
your training data does not contain.

So the question moves to the simulator. Which of the real world's defects, all
of which our synthesis omits, is the one that breaks the residual criterion?

Five candidates, each measured ALONE and then together, so the answer names a
mechanism rather than a mood:

  dead     30-40% of the fan removed in CLUSTERS, not at random. The stereo
           mask kills contiguous patches; our synthetic fan is uniformly
           covered, and uniform coverage is exactly what makes a held-out point
           an easy interpolation.

  cnoise   spatially correlated noise (10 m scale, 5 cm) instead of the 2 cm
           independent Gaussian we have always used. Correlated error is what a
           flexible prior can absorb into its coefficients while a stiff one
           cannot -- and the residual cannot tell absorption from fit.

  nonstat  the sea changes inside the 20 s window. Our components have fixed
           amplitudes; a real wind sea does not.

  bias     a smooth range-dependent sensor bias, growing as (r/200)^2. LJ18g's
           reconstruction bows by 7.6 m at range (ledger C1).

  kperr    kp_hint is estimated per window on real data and handed to us exactly
           in synthesis. A 15% error moves the whole bank.

PRE-REGISTERED, before running:
  the combined configuration must raise the residual rule's regret from 0.0067
  to >= 0.015, approaching the +0.016 it leaks on real records. If it does not,
  the list above is wrong and the simulator, not the critic, is what we do not
  understand yet.

Reported as regret against the sea's own best of twenty candidates, with the
oracle spread alongside -- because a defect that makes every candidate equally
bad raises no regret and teaches nothing.
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

from d3_gpu_mmse import mmse_gpu, build_bank
from d3_l1_learned_prior import RadialGain, sea, CANDS, S0
from d3_l3_trust_and_coverage import ood_sea
from d3_g3_adversarial_sea import geometry, eta, DEV, DEPTH, KP, SIGMA, NOISE
from d3_a2_real_revalidation import HELD
import d3_static_field as SF

OUT = 'results/direction3/o1_defect_ablation.json'
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CONFIGS = ['none', 'dead', 'cnoise', 'nonstat', 'bias', 'kperr', 'all']
DEAD_FRAC = 0.35
CNOISE_STD, CNOISE_LEN = 0.05, 10.0
KP_ERR = 0.15


def synth_sea(rng):
    r = rng.random()
    if r < 0.4:
        return sea(rng, 'swell')
    if r < 0.8:
        return sea(rng, 'inband')
    return ood_sea(rng, 0.2, 2.4)


def corr_noise(P, rng, std, ell):
    """Correlated field noise: a few random plane waves at the given scale."""
    J = 24
    q = 2 * np.pi / ell
    th = rng.uniform(-np.pi, np.pi, J)
    ph = rng.uniform(-np.pi, np.pi, J)
    qs = q * np.stack([np.cos(th), np.sin(th)], 1)
    x = P[:, :2].cpu().numpy()
    n = np.cos(x @ qs.T + ph[None]).sum(1) / np.sqrt(J / 2)
    return torch.tensor(std * n, device=DEV, dtype=torch.float32)


def observe(P, Pq, s, rng, gen, cfg):
    kx, ky, om, a, ph = s
    keep = torch.ones(len(P), dtype=torch.bool, device=DEV)
    if cfg in ('dead', 'all'):
        # contiguous holes, not random thinning
        xy = P[:, :2].cpu().numpy()
        lo, hi = xy.min(0), xy.max(0)
        drop = np.zeros(len(xy), bool)
        while drop.mean() < DEAD_FRAC:
            c = rng.uniform(lo, hi)
            r = rng.uniform(15.0, 35.0)
            drop |= (np.hypot(*(xy - c).T) < r)
        keep = torch.tensor(~drop, device=DEV)

    if cfg in ('nonstat', 'all'):
        # amplitudes drift inside the window; the truth at t0 uses the same law
        psi = torch.tensor(rng.uniform(-np.pi, np.pi, len(a)), device=DEV,
                           dtype=torch.float32)

        def env(t):
            return 1.0 + 0.5 * torch.sin(2 * np.pi * t / 40.0 + psi[None])

        def field(Q):
            pz = (Q[:, 0:1] * kx[None] + Q[:, 1:2] * ky[None]
                  - Q[:, 2:3] * om[None] + ph[None])
            return ((a[None] * env(Q[:, 2:3])) * torch.cos(pz)).sum(1)
    else:
        def field(Q):
            return eta(Q, kx, ky, om, a, ph)

    z = field(P)
    if cfg in ('cnoise', 'all'):
        z = z + corr_noise(P, rng, CNOISE_STD, CNOISE_LEN)
    else:
        z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
    if cfg in ('bias', 'all'):
        r = torch.hypot(P[:, 0], P[:, 1])
        z = z + 0.5 * SIGMA * (r / 200.0) ** 2

    gt = field(Pq)
    gt = gt - gt.mean()
    return P[keep], z[keep], gt


def gains_for(net, kp, depth):
    out = {}
    for band in CANDS:
        kxb, kyb, _ = build_bank(kp, band, kp / 10.0, depth, DEV,
                                 torch.float32)
        kmag = torch.hypot(kxb, kyb)
        with torch.no_grad():
            g = torch.exp(net.f(torch.log(kmag / kp).unsqueeze(1))
                          .squeeze(1).clamp(-6.0, 6.0))
        for al in ALPHAS:
            out[(band, al)] = None if al == 0.0 else g ** al
    return out


def detrend_range(P, z):
    """Regress {1, r, r^2} out of the observations.

    The ablation named the mechanism: a smooth range-dependent sensor bias is
    partly representable by the bank's longest atoms. A wide band -- or a
    learned prior that favours low k -- absorbs the bias, which LOWERS the
    held-out residual while planting a spurious long wave in the field. The
    residual rewards exactly what the field punishes.

    If that reading is right, removing the representable part of the bias before
    the fit should restore the residual criterion, with no learning at all.
    Three coefficients, fitted on the observations, never on the truth.
    """
    r = torch.hypot(P[:, 0], P[:, 1])
    A = torch.stack([torch.ones_like(r), r / 200.0, (r / 200.0) ** 2], 1)
    c = torch.linalg.lstsq(A, z.unsqueeze(1)).solution.squeeze(1)
    return z - A @ c


def one_sea(net, P0, Pq, rng, gen, cfg, detrend=False):
    s = synth_sea(rng)
    P, z, gt = observe(P0, Pq, s, rng, gen, cfg)
    if len(z) < 150:
        return None
    if detrend:
        z = detrend_range(P, z)
    kp = KP * (1 + rng.normal(0, KP_ERR)) if cfg in ('kperr', 'all') else KP
    kp = float(np.clip(kp, 0.5 * KP, 1.8 * KP))
    gains = gains_for(net, kp, DEPTH)

    n = len(z)
    idx = torch.randperm(n, device=DEV, generator=gen)
    tr, he = idx[:int((1 - HELD) * n)], idx[int((1 - HELD) * n):]
    zero = float(gt.std())
    res, rat = [], []
    with torch.no_grad():
        for (band, al), g in gains.items():
            pv = mmse_gpu(P[tr], z[tr], DEPTH, P[he], sigma0=S0, kp_hint=kp,
                          band=band, prior_gain=g)
            res.append(float(torch.sqrt(((pv - z[he]) ** 2).mean())))
            pr = mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=kp, band=band,
                          prior_gain=g)
            pr = pr - pr.mean()
            rat.append(float(torch.sqrt(((pr - gt) ** 2).mean()) / zero))
    res, rat = np.array(res), np.array(rat)
    best = rat.min()
    return dict(regret=rat[int(np.argmin(res))] - best, best=best,
                spread=rat.max() - best, n_obs=len(z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seas', type=int, default=40)
    ap.add_argument('--cfgs', nargs='*', default=CONFIGS)
    ap.add_argument('--detrend', action='store_true',
                    help='regress {1, r, r^2} out of z first')
    a_ = ap.parse_args()

    net = RadialGain().to(DEV)
    net.load_state_dict(torch.load('results/direction3/l1_prior_gain.pt'))
    net.eval()
    P0, Pq = geometry()
    gen = torch.Generator(device=DEV).manual_seed(0)

    print(f"detrend {{1, r, r^2}} from observations: {a_.detrend}\n")
    print(f"{'defect':<10}{'regret':>18}{'oracle best':>13}"
          f"{'spread':>10}{'obs':>7}")
    res = {}
    for cfg in a_.cfgs:
        rows = []
        for i in range(a_.seas):
            rng = np.random.default_rng(5000 + i)
            r = one_sea(net, P0, Pq, rng, gen, cfg, a_.detrend)
            if r:
                rows.append((r['regret'], r['best'], r['spread'], r['n_obs']))
        v = np.array(rows)
        se = v[:, 0].std(ddof=1) / np.sqrt(len(v))
        print(f"{cfg:<10}{v[:,0].mean():>11.4f} ± {se:.4f}"
              f"{v[:,1].mean():>13.3f}{v[:,2].mean():>10.3f}"
              f"{v[:,3].mean():>7.0f}", flush=True)
        res[cfg] = dict(regret=v[:, 0].mean(), se=se, best=v[:, 1].mean(),
                        spread=v[:, 2].mean(), n=len(v),
                        n_obs=v[:, 3].mean())

    print(f"\nregret = field nRMSE of the residual rule's pick, minus the sea's "
          f"own best\nof twenty candidates. Clean synthesis gave 0.0067; the "
          f"real records leak +0.016.\n")
    base = res.get('none', {}).get('regret', np.nan)
    for cfg, d in res.items():
        if cfg == 'none':
            continue
        print(f"  {cfg:<10} regret {d['regret']:.4f}  "
              f"({d['regret']-base:+.4f} vs clean)   "
              f"oracle spread {d['spread']:.3f}")

    if 'all' in res:
        r = res['all']['regret']
        ok = r >= 0.015
        print(f"\n=== O1 gate (pre-registered) ===")
        print(f"  combined regret >= 0.015 : {'PASS' if ok else 'FAIL'}  "
              f"({r:.4f})")
        if not ok:
            print("  the defect list is wrong. The simulator, not the critic, "
                  "is what we\n  do not understand -- and no learned selector "
                  "can be trained until it is.")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in res.items()}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
