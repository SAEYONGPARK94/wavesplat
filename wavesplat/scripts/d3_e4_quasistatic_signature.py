"""E4: read the absorbed bias off the model, not off the residual.

Idea (3) on the list was "a network that looks at the residual's STRUCTURE
rather than its scalar RMS". E3 makes that idea look naive, and the reason is
worth keeping.

A model that absorbs the range bias FITS THE OBSERVATIONS WELL. Its held-out
residual is small and structureless. There is nothing there for a network to
read. What E3 found is that the absorbed bias is exposed by TIME: the fit gives
its spurious long wave the wrong omega, and the next scan catches it out.

So the signature does not live in the residual. It lives in the MODEL. A static
bias is best matched by atoms whose phase barely turns inside the 20 s window --

    omega * W_OBS  <<  2 pi

-- so the absorbing fit dumps energy into quasi-static atoms. That fraction is
computable from the fitted coefficients, needs no ground truth, costs one
matrix-vector product, and is not a network at all.

    q = sum of coefficient energy at omega*W < pi
        --------------------------------------
             total coefficient energy

Three questions, in order:
  1. does q separate the candidates that wreck the field under bias?
     (corr(q, field regret) across the twenty candidates, per sea)
  2. does the criterion `residual + lam * q` beat the residual alone?
  3. and does it beat the time-forward residual, which needs no tuning at all?

If (3) is no -- and E3's mechanism says it should be no, because q is a proxy
for what time-forward measures directly -- then the answer to idea (3) is that
the structure worth reading was temporal, and we already read it.
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
from d3_l1_learned_prior import RadialGain, S0
from d3_o1_defect_ablation import synth_sea, observe, gains_for
from d3_g3_adversarial_sea import geometry, DEV, DEPTH, KP
from d3_a2_real_revalidation import HELD

OUT = 'results/direction3/e4_quasistatic.json'
W_OBS = 20.0
PEXP, SMULT = 1.0, 1.0
LAMS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)


def solve(P, z, Pq, band, gain, model=False):
    return mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP, band=band,
                    prior_gain=gain, pexp=PEXP, smult=SMULT, no_em=True,
                    return_model=model)


def quasi_static(P, z, band, gain):
    """Fraction of fitted coefficient energy in atoms that barely turn."""
    with torch.no_grad():
        _, mo = solve(P, z, P[:1], band, gain, model=True)
        K, gam, Phi = mo['K'], mo['gam'], mo['Phi']
        w = gam * (Phi.T @ torch.linalg.solve(K, z))     # posterior mean coeffs
        kxb, kyb, om = build_bank(KP, band, KP / 10.0, DEPTH, DEV,
                                  torch.float32)
        e = (w ** 2).reshape(2, -1).sum(0)               # cos & sin per atom
        slow = (om * W_OBS) < np.pi
        return float(e[slow].sum() / e.sum())


def sea_row(P, z, Pq, gt, gains, gen):
    n = len(z)
    idx = torch.randperm(n, device=DEV, generator=gen)
    tr, he = idx[:int((1 - HELD) * n)], idx[int((1 - HELD) * n):]
    t = P[:, 2]
    cut = torch.quantile(t, 1 - HELD)
    ttr = torch.nonzero(t <= cut).squeeze(1)
    the = torch.nonzero(t > cut).squeeze(1)
    if len(the) < 20:
        return None
    res, rtime, qs, rat = [], [], [], []
    with torch.no_grad():
        for (band, al), g in gains.items():
            pv = solve(P[tr], z[tr], P[he], band, g)
            res.append(float(torch.sqrt(((pv - z[he]) ** 2).mean())))
            pt = solve(P[ttr], z[ttr], P[the], band, g)
            rtime.append(float(torch.sqrt(((pt - z[the]) ** 2).mean())))
            qs.append(quasi_static(P, z, band, g))
            pr = solve(P, z, Pq, band, g)
            pr = pr - pr.mean()
            rat.append(float(torch.sqrt(((pr - gt) ** 2).mean()) / gt.std()))
    return np.array(res), np.array(rtime), np.array(qs), np.array(rat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seas', type=int, default=25)
    a_ = ap.parse_args()

    net = RadialGain().to(DEV)
    net.load_state_dict(torch.load('results/direction3/l1_prior_gain.pt'))
    net.eval()
    P0, Pq = geometry()
    gen = torch.Generator(device=DEV).manual_seed(0)
    out = {}

    for cfg in ('none', 'bias'):
        corr, reg_res, reg_time, reg_lam = [], [], [], {l: [] for l in LAMS}
        qs_all = []
        for i in range(a_.seas):
            rng = np.random.default_rng(5000 + i)
            s = synth_sea(rng)
            P, z, gt = observe(P0, Pq, s, rng, gen, cfg)
            if len(z) < 150:
                continue
            gains = gains_for(net, KP, DEPTH)
            r = sea_row(P, z, Pq, gt, gains, gen)
            if r is None:
                continue
            res, rtime, qs, rat = r
            best = rat.min()
            corr.append(np.corrcoef(qs, rat)[0, 1])
            reg_res.append(rat[int(np.argmin(res))] - best)
            reg_time.append(rat[int(np.argmin(rtime))] - best)
            qs_all.append(qs.mean())
            rn = res / (res.mean() + 1e-9)
            for l in LAMS:
                reg_lam[l].append(rat[int(np.argmin(rn + l * qs))] - best)
        c = np.array(corr)
        print(f"\n=== {cfg} seas ({len(c)}) ===")
        print(f"corr(quasi-static energy, field regret) across candidates: "
              f"{np.nanmean(c):+.3f} ± {np.nanstd(c)/np.sqrt(len(c)):.3f}"
              f"   (mean q = {np.mean(qs_all):.3f})")
        print(f"{'lambda':>8}{'regret':>12}")
        for l in LAMS:
            v = np.array(reg_lam[l])
            print(f"{l:>8.2f}{v.mean():>12.4f}")
        print(f"{'residual':>8}{np.mean(reg_res):>12.4f}")
        print(f"{'time-fwd':>8}{np.mean(reg_time):>12.4f}   <- E3's winner")
        out[cfg] = dict(corr=float(np.nanmean(c)),
                        q_mean=float(np.mean(qs_all)),
                        residual=float(np.mean(reg_res)),
                        time=float(np.mean(reg_time)),
                        lam={str(l): float(np.mean(reg_lam[l])) for l in LAMS})

    b = out['bias']
    best_lam = min(b['lam'], key=lambda k: b['lam'][k])
    print(f"\n=== E4 ===")
    print(f"  bias seas: residual {b['residual']:.4f}   "
          f"best lambda {best_lam} -> {b['lam'][best_lam]:.4f}   "
          f"time-forward {b['time']:.4f}")
    if b['lam'][best_lam] > b['time']:
        print("  the model-side signature does not beat simply holding out the "
              "future.\n  The structure worth reading was temporal, and E3 "
              "already read it.")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
