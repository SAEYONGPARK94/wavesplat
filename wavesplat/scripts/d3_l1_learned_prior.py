"""L1: learn the one thing theory allows -- gamma(k) -- and close a measured gap.

Today established where learning may and may not go.

  may not   the posterior mean. On a Gaussian sea with Gaussian noise it is
            linear and MSE-optimal; no learner beats the physics MMSE.
  may not   anything read off 13 real records. The labelled-sea count is the
            binding axis (G1), and 13 is below it.
  may       gamma(k), the prior over the bank. Physics does not fix it. It is
            a modelling choice, and today a hand-made one: pick the best of
            four candidate bands by held-out observation residual.

That hand rule leaves a MEASURED gap. In A1b's worst physical cell it scores
0.582 where the oracle band scores 0.490. An oracle exists, so the ceiling is
visible -- the rarest thing in this project.

D1 then sharpened the target. The wall was never the aperture; it was the band
floor we chose. Lowering the floor adds atoms, and on a multi-component sea
those atoms compete with the wind sea for the same observations. So the thing
to learn is not "the band" but the DEGREES-OF-FREEDOM TRADE: which low-k atoms
earn their keep.

So: fix the band wide (0.15, 2.5) and learn a radial gain g(k/kp) that
multiplies the beamform prior. Setting g = 0 below 0.35 kp reproduces the
narrow band exactly; g = 1 everywhere reproduces the wide one. The learned
prior therefore STRICTLY GENERALISES the hand selector, and the comparison is
honest.

Trained by task loss through the differentiable solver (verified: prior_gain =
1 reproduces the analytic estimator to |diff| = 0.0). Seas are synthetic and
unlimited, so leave-sea-out is a real test.

PRE-REGISTERED GATE, on 48 held-out seas never seen in training:
  1. beat the hand selector by dnRMSE >= 0.02
  2. close >= 2/3 of the hand selector's gap to the oracle band
  3. negative control: on in-band-only seas, where the hand selector is already
     oracle, the learned prior must not LOSE more than 0.01
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu, build_bank
from d3_g3_adversarial_sea import geometry, eta, norm_amp, DEV, DEPTH, KP, SIGMA, NOISE

OUT = 'results/direction3/l1_learned_prior.json'
CKPT = 'results/direction3/l1_prior_gain.pt'
G = 9.81
WIDE = (0.15, 2.5)
CANDS = [(0.35, 2.5), (0.25, 2.5), (0.15, 2.5), (0.15, 4.0)]
S0 = 0.05 * 4 * SIGMA
HELD = 0.25


def dm(v):
    return v - v.mean()


class RadialGain(nn.Module):
    """g(k/kp): one number per bank atom, a function of its wavenumber only.

    Radial by construction -- direction must stay with the beamform prior,
    which is where the observations speak. A direction-dependent learned gain
    would be learning the posterior, which the Gaussian theorem forbids.
    """

    def __init__(self, h=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h),
                               nn.SiLU(), nn.Linear(h, 1))
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)          # starts at gain = 1 exactly

    def forward(self, kmag):
        x = torch.log(kmag / KP).unsqueeze(1)
        return torch.exp(self.f(x).squeeze(1).clamp(-6.0, 6.0))


def bank_kmag():
    kxb, kyb, _ = build_bank(KP, WIDE, KP / 10.0, DEPTH, DEV, torch.float32)
    return torch.hypot(kxb, kyb), len(kxb)


def sea(rng, kind):
    """in-band: nothing below the floor. swell: 60% of the variance below it."""
    if kind == 'inband':
        n = 64
        k = KP * np.exp(rng.uniform(np.log(0.5), np.log(2.0), n))
        a = np.abs(rng.normal(0, 1, n))
        a *= np.sqrt(SIGMA ** 2 / ((a ** 2).sum() / 2))
    else:
        n1, n2 = 24, 48
        k = np.concatenate([
            KP * np.exp(rng.uniform(np.log(0.15), np.log(0.33), n1)),
            KP * np.exp(rng.uniform(np.log(0.5), np.log(2.0), n2))])
        a = np.abs(rng.normal(0, 1, n1 + n2))
        a[:n1] *= np.sqrt(0.6 * SIGMA ** 2 / ((a[:n1] ** 2).sum() / 2))
        a[n1:] *= np.sqrt(0.4 * SIGMA ** 2 / ((a[n1:] ** 2).sum() / 2))
        n = n1 + n2
    th = rng.uniform(-np.pi, np.pi, n)
    t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    return (t(k * np.cos(th)), t(k * np.sin(th)),
            t(np.sqrt(G * k * np.tanh(k * DEPTH))), t(a),
            t(rng.uniform(-np.pi, np.pi, n)))


def obs(P, Pq, s, gen):
    kx, ky, om, a, ph = s
    z = eta(P, kx, ky, om, a, ph)
    z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
    gt = dm(eta(Pq, kx, ky, om, a, ph))
    return z, gt


def ratio(P, z, Pq, gt, band, gain=None):
    pr = dm(mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP, band=band,
                     prior_gain=gain))
    return torch.sqrt(((pr - gt) ** 2).mean()) / gt.std()


def hand_select(P, z, gen):
    with torch.no_grad():
        n = len(z)
        idx = torch.randperm(n, device=DEV, generator=gen)
        tr, he = idx[:int((1 - HELD) * n)], idx[int((1 - HELD) * n):]
        best, bb = None, CANDS[0]
        for b in CANDS:
            pv = mmse_gpu(P[tr], z[tr], DEPTH, P[he], sigma0=S0, kp_hint=KP,
                          band=b)
            r = float(torch.sqrt(((pv - z[he]) ** 2).mean()))
            if best is None or r < best:
                best, bb = r, b
    return bb


def evaluate(P, Pq, net, kmag, gen, kinds, n, seed0=5000):
    out = {k: {m: [] for m in ('fix35', 'fix15', 'hand', 'oracle', 'learned')}
           for k in kinds}
    with torch.no_grad():
        gain = net(kmag)
    for kind in kinds:
        for i in range(n):
            rng = np.random.default_rng(seed0 + i)
            s = sea(rng, kind)
            z, gt = obs(P, Pq, s, gen)
            with torch.no_grad():
                out[kind]['fix35'].append(float(ratio(P, z, Pq, gt, CANDS[0])))
                out[kind]['fix15'].append(float(ratio(P, z, Pq, gt, WIDE)))
                out[kind]['oracle'].append(min(float(ratio(P, z, Pq, gt, b))
                                               for b in CANDS))
                out[kind]['hand'].append(
                    float(ratio(P, z, Pq, gt, hand_select(P, z, gen))))
                out[kind]['learned'].append(
                    float(ratio(P, z, Pq, gt, WIDE, gain)))
    return {k: {m: float(np.mean(v)) for m, v in d.items()}
            for k, d in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--seas', type=int, default=2)
    ap.add_argument('--eval', type=int, default=48)
    a_ = ap.parse_args()

    P, Pq = geometry()
    gen = torch.Generator(device=DEV).manual_seed(0)
    kmag, B = bank_kmag()
    print(f"bank {B} atoms over {WIDE} kp, learning a radial gain on it\n")

    net = RadialGain().to(DEV)
    base = evaluate(P, Pq, net, kmag, gen, ('swell', 'inband'), 16)
    print("at init (gain = 1, so 'learned' must equal 'fix15'):")
    for k, d in base.items():
        print(f"  {k:<7} fix35 {d['fix35']:.3f}  fix15 {d['fix15']:.3f}  "
              f"hand {d['hand']:.3f}  oracle {d['oracle']:.3f}  "
              f"learned {d['learned']:.3f}")

    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a_.steps)
    rng = np.random.default_rng(1)
    print("\ntraining (task loss through the solver):", flush=True)
    for st in range(1, a_.steps + 1):
        gain = net(kmag)
        loss = 0.0
        for _ in range(a_.seas):
            kind = 'swell' if rng.random() < 0.6 else 'inband'
            s = sea(rng, kind)
            z, gt = obs(P, Pq, s, gen)
            loss = loss + ratio(P, z, Pq, gt, WIDE, gain)
        loss = loss / a_.seas
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if st % 50 == 0:
            with torch.no_grad():
                g = net(kmag)
                lo = float(g[kmag < 0.35 * KP].mean())
                hi = float(g[kmag >= 0.35 * KP].mean())
            print(f"  step {st:4d}  loss {float(loss):.4f}   "
                  f"gain below floor {lo:.3f}  above {hi:.3f}", flush=True)

    print(f"\nheld-out seas ({a_.eval} per population, never trained on):")
    res = evaluate(P, Pq, net, kmag, gen, ('swell', 'inband'), a_.eval,
                   seed0=90000)
    for k, d in res.items():
        print(f"  {k:<7} fix35 {d['fix35']:.3f}  fix15 {d['fix15']:.3f}  "
              f"hand {d['hand']:.3f}  oracle {d['oracle']:.3f}  "
              f"LEARNED {d['learned']:.3f}")

    sw = res['swell']
    gap = sw['hand'] - sw['oracle']
    closed = (sw['hand'] - sw['learned']) / gap if abs(gap) > 1e-9 else 0.0
    beat = sw['hand'] - sw['learned']
    lose = res['inband']['learned'] - res['inband']['hand']
    print(f"\n=== L1 gate (pre-registered) ===")
    print(f"  1) beat hand by >= 0.020 on swell:  {beat:+.4f}  "
          f"{'PASS' if beat >= 0.02 else 'FAIL'}")
    print(f"  2) close >= 2/3 of oracle gap:      {closed*100:.0f}%  "
          f"(gap {gap:.4f})  {'PASS' if closed >= 2/3 else 'FAIL'}")
    print(f"  3) lose <= 0.010 on in-band:        {lose:+.4f}  "
          f"{'PASS' if lose <= 0.01 else 'FAIL'}")

    with torch.no_grad():
        g = net(kmag).cpu().numpy()
        km = (kmag / KP).cpu().numpy()
    prof = [(float(a), float(np.mean(g[(km >= a) & (km < b)])))
            for a, b in ((0.15, 0.25), (0.25, 0.35), (0.35, 0.6), (0.6, 1.0),
                         (1.0, 1.6), (1.6, 2.5))]
    print("\nlearned radial gain, by k/kp band:")
    for a, v in prof:
        print(f"  k/kp >= {a:<5} gain {v:.3f}")

    torch.save(net.state_dict(), CKPT)
    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'init': base, 'heldout': res, 'gate':
                   {'beat_hand': beat, 'closed_frac': closed, 'inband_loss': lose},
                   'gain_profile': prof}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
