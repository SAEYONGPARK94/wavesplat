"""L3: when may we trust the learned prior, and how wrong can we be?

L1 left the learned gamma(k) unusable. In distribution it beats the hand rule
and even the discrete-band oracle (swell 0.216 -> 0.187). Out of distribution
it is 45% worse than the hand rule (high-k seas: 0.143 -> 0.208), because it
suppressed atoms where the training seas happened to carry no energy. That is
what a prior is; the problem is deploying one without noticing when the sea has
changed.

We already own a shift detector: the held-out observation residual. It chose
the band all day. Ask it to choose the PRIOR too.

  L3a  TRUST RULE.  Fit each candidate on 75% of the fan, score it on the
       remaining 25%, take the lower residual. No ground truth anywhere.

       THE TRAP, and it is the whole experiment: the learned prior runs on the
       wide band and has more effective freedom, so it can fit the observations
       better while fitting the FIELD worse. A residual rule would then pick it
       exactly when it should not. If that happens, the rule is worse than
       useless and L1 stays undeployable.

  L3b  COVERAGE.  Split-conformal on the predictive sd. Calibrate one scalar q
       on synthetic seas so that |pr - gt| <= q . sd holds for 90% of pixels,
       then measure the realised coverage in distribution and out of it.
       Conformal guarantees coverage only under exchangeability, which OOD
       breaks by construction. The honest question is how badly.

Together with B1's deterministic bound this is what a controller needs:
  B1   it can never be worse than doing nothing by more than std(z~)
  L3a  when to use the learned prior at all
  L3b  a band around the estimate, and its true coverage
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
from d3_l1_learned_prior import (RadialGain, bank_kmag, sea, obs, ratio,
                                 hand_select, WIDE, CANDS, dm, S0, HELD)
from d3_g3_adversarial_sea import geometry, DEV, DEPTH, KP, SIGMA

OUT = 'results/direction3/l3_trust_and_coverage.json'
CKPT = 'results/direction3/l1_prior_gain.pt'
G = 9.81
ALPHA = 0.10                       # target miscoverage


def ood_sea(rng, lo, hi, n=64):
    k = KP * np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    th = rng.uniform(-np.pi, np.pi, n)
    a = np.abs(rng.normal(0, 1, n))
    a *= np.sqrt(SIGMA ** 2 / ((a ** 2).sum() / 2))
    t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    return (t(k * np.cos(th)), t(k * np.sin(th)),
            t(np.sqrt(G * k * np.tanh(k * DEPTH))), t(a),
            t(rng.uniform(-np.pi, np.pi, n)))


POPS = {
    'swell (in-dist)': lambda r: sea(r, 'swell'),
    'in-band (in-dist)': lambda r: sea(r, 'inband'),
    'highk (OOD)': lambda r: ood_sea(r, 1.5, 2.4),
    'flat (OOD)': lambda r: ood_sea(r, 0.2, 2.4),
}


def split(n, gen):
    idx = torch.randperm(n, device=DEV, generator=gen)
    return idx[:int((1 - HELD) * n)], idx[int((1 - HELD) * n):]


def held_resid(P, z, tr, he, band, gain=None):
    pv = mmse_gpu(P[tr], z[tr], DEPTH, P[he], sigma0=S0, kp_hint=KP, band=band,
                  prior_gain=gain)
    return float(torch.sqrt(((pv - z[he]) ** 2).mean()))


def trust_rule(P, z, gen, gain):
    """Pick learned-prior-on-wide-band vs hand-selected-band, by held-out
    observation residual. Ground truth is never touched."""
    tr, he = split(len(z), gen)
    hb = hand_select(P, z, gen)
    r_hand = held_resid(P, z, tr, he, hb)
    r_learn = held_resid(P, z, tr, he, WIDE, gain)
    return ('learned', WIDE, gain) if r_learn < r_hand else ('hand', hb, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--cal', type=int, default=40)
    a_ = ap.parse_args()

    P, Pq = geometry()
    gen = torch.Generator(device=DEV).manual_seed(0)
    kmag, _ = bank_kmag()
    net = RadialGain().to(DEV)
    net.load_state_dict(torch.load(CKPT))
    net.eval()
    with torch.no_grad():
        gain = net(kmag)

    # ---------------- L3a: the trust rule -------------------------------
    print("L3a  trust rule: pick the prior by held-out OBSERVATION residual\n")
    print(f"{'population':<20}{'hand':>8}{'learned':>9}{'TRUST':>8}"
          f"{'oracle*':>9}{'picks learned':>15}")
    res = {}
    for name, mk in POPS.items():
        h, l, t, o, pk = [], [], [], [], 0
        for i in range(a_.n):
            rng = np.random.default_rng(120000 + i)
            s = mk(rng)
            z, gt = obs(P, Pq, s, gen)
            with torch.no_grad():
                rh = float(ratio(P, z, Pq, gt, hand_select(P, z, gen)))
                rl = float(ratio(P, z, Pq, gt, WIDE, gain))
                which, band, g = trust_rule(P, z, gen, gain)
                rt = float(ratio(P, z, Pq, gt, band, g))
            h.append(rh); l.append(rl); t.append(rt); o.append(min(rh, rl))
            pk += (which == 'learned')
        res[name] = dict(hand=np.mean(h), learned=np.mean(l), trust=np.mean(t),
                         oracle=np.mean(o), frac_learned=pk / a_.n)
        print(f"{name:<20}{np.mean(h):>8.3f}{np.mean(l):>9.3f}"
              f"{np.mean(t):>8.3f}{np.mean(o):>9.3f}{pk/a_.n*100:>14.0f}%")
    print("\n*oracle = per-sea best of {hand, learned}, chosen with the truth.\n")

    ok = all(res[k]['trust'] <= res[k]['hand'] + 0.01 for k in POPS)
    gain_ind = res['swell (in-dist)']['hand'] - res['swell (in-dist)']['trust']
    harm_ood = res['highk (OOD)']['trust'] - res['highk (OOD)']['hand']
    print(f"in-distribution gain kept:  {gain_ind:+.4f}   "
          f"(learned alone: {res['swell (in-dist)']['hand']-res['swell (in-dist)']['learned']:+.4f})")
    print(f"out-of-distribution harm:   {harm_ood:+.4f}   "
          f"(learned alone: {res['highk (OOD)']['learned']-res['highk (OOD)']['hand']:+.4f})")
    print(f"trust rule never worse than hand by >0.01: {ok}\n")

    # ---------------- L3b: conformal coverage ----------------------------
    print("L3b  split-conformal on the predictive sd, alpha = 0.10\n")
    scores = []
    for i in range(a_.cal):
        rng = np.random.default_rng(200000 + i)
        s = sea(rng, 'swell' if i % 2 else 'inband')      # calibration = in-dist
        z, gt = obs(P, Pq, s, gen)
        with torch.no_grad():
            pr, var = mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP,
                               band=WIDE, prior_gain=gain, return_var=True)
            pr = dm(pr)
            sd = torch.sqrt(var.clamp_min(1e-12))
            scores.append((torch.abs(pr - gt) / sd).cpu().numpy())
    sc = np.concatenate(scores)
    q = float(np.quantile(sc, 1 - ALPHA))
    print(f"calibration on {a_.cal} in-distribution seas: q = {q:.3f}   "
          f"(a calibrated Gaussian sd would give 1.645)")

    print(f"\n{'population':<20}{'coverage':>10}{'target':>9}{'z-std':>9}")
    cov = {}
    for name, mk in POPS.items():
        c, zs = [], []
        for i in range(a_.n):
            rng = np.random.default_rng(300000 + i)
            s = mk(rng)
            z, gt = obs(P, Pq, s, gen)
            with torch.no_grad():
                pr, var = mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP,
                                   band=WIDE, prior_gain=gain, return_var=True)
                pr = dm(pr)
                sd = torch.sqrt(var.clamp_min(1e-12))
                c.append(float((torch.abs(pr - gt) <= q * sd).float().mean()))
                zs.append(float(((pr - gt) / sd).std()))
        cov[name] = dict(coverage=np.mean(c), z_std=np.mean(zs))
        print(f"{name:<20}{np.mean(c):>10.3f}{1-ALPHA:>9.2f}{np.mean(zs):>9.2f}")

    print("\nconformal guarantees 0.90 only under exchangeability with the "
          "calibration set.\nOOD breaks it by construction; the question was "
          "how badly.")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'trust': {k: {kk: float(vv) for kk, vv in v.items()}
                             for k, v in res.items()},
                   'q': q, 'alpha': ALPHA,
                   'coverage': {k: {kk: float(vv) for kk, vv in v.items()}
                                for k, v in cov.items()}}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
