"""L2: a prior trained against an adversary, and made to keep its calibration.

L1 learned a gamma(k) that wins in distribution and loses out of it. L3 then
found the deeper cost: on the adversarial sea the learned prior keeps its
accuracy (nRMSE/zero = 0.282) while its posterior variance collapses -- z-std
reached 68,710. It is confidently wrong. Accuracy-based selection cannot see
that, so the trust rule picked it anyway.

Both failures have the same root: nothing in L1's objective ever asked the
prior to be right about seas it was not shown, or honest about its own
uncertainty. So put both in the loss.

  minimax     an adversarial sea is maintained and refreshed by a few gradient
              steps against the CURRENT prior each iteration, then the prior is
              updated on a mixture of random and adversarial seas. The prior is
              never allowed to settle on a distribution.

  calibration |z-std - 1| is added to the loss. z-std is the std of
              (pr - gt)/sd; a calibrated posterior gives 1. It needs the truth
              at training time, which we have, and never at run time.

  floor       the gain is clamped to [0.082, 12.2] instead of [0.0025, 403].
              A prior that can drive an atom's variance to zero can drive the
              posterior variance to zero, and L3 showed it does.

PRE-REGISTERED GATES, all on seas never trained on:
  1. adversarial (found against THIS prior): nRMSE <= hand rule + 0.01
  2. out-of-distribution high-k seas:        nRMSE <= hand rule + 0.01
  3. calibration health, every population:   z-std in [0.5, 2.0]
  4. in-distribution gain retained:          hand - learned >= 0.02 on swell

Gate 3 is the one L1 would fail (z-std 68,710). Gate 1 is the one it was never
asked.
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

from d3_gpu_mmse import mmse_gpu
from d3_l1_learned_prior import (bank_kmag, sea, obs, ratio, hand_select,
                                 WIDE, CANDS, dm, S0)
from d3_l3_trust_and_coverage import ood_sea
from d3_g3_adversarial_sea import (geometry, eta, norm_amp, sea_shell, DEV,
                                   DEPTH, KP, NCOMP, SIGMA, NOISE)

OUT = 'results/direction3/l2_adversarial_prior.json'
CKPT = 'results/direction3/l2_prior_gain.pt'
G = 9.81
LAM_CAL = 0.3                 # weight on |z-std - 1|
CLAMP = 2.5                   # gain in [e^-2.5, e^2.5] = [0.082, 12.2]
NQ = 192                      # query subsample for the variance term


class SafeGain(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h),
                               nn.SiLU(), nn.Linear(h, 1))
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)

    def forward(self, kmag):
        x = torch.log(kmag / KP).unsqueeze(1)
        return torch.exp(self.f(x).squeeze(1).clamp(-CLAMP, CLAMP))


def loss_on(P, Pq, s, gen, gain, qi):
    z, gt = obs(P, Pq, s, gen)
    pr, var = mmse_gpu(P, z, DEPTH, Pq[qi], sigma0=S0, kp_hint=KP, band=WIDE,
                       prior_gain=gain, return_var=True)
    g = gt[qi] - gt[qi].mean()
    pr = pr - pr.mean()
    acc = torch.sqrt(((pr - g) ** 2).mean()) / g.std()
    sd = torch.sqrt(var.clamp_min(1e-10))
    zstd = ((pr - g) / sd).std()
    return acc + LAM_CAL * torch.abs(zstd - 1.0), acc, zstd


def refresh_adversary(P, Pq, gen, gain, state, steps=4):
    """A persistent adversarial sea, pushed a few steps against the CURRENT
    prior every outer iteration. Full inner maximisation is unaffordable; a
    moving adversary is the standard cheap surrogate."""
    kx, ky, om, raw, ph = state
    opt = torch.optim.Adam([raw, ph], lr=0.05)
    for _ in range(steps):
        z = eta(P, kx, ky, om, norm_amp(raw), ph)
        z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
        gt = dm(eta(Pq, kx, ky, om, norm_amp(raw), ph))
        pr = dm(mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP, band=WIDE,
                         prior_gain=gain.detach()))
        # Normalise by the sea's FIXED spectral std. Dividing by gt.std() lets
        # the adversary win by shrinking the denominator instead of breaking
        # the estimator -- errors #16 and #18, and the first version of this
        # file made it a third time. The tell was that `hand` and the learned
        # prior scored almost identically (4.945 vs 5.065) on the sea it found:
        # a metric attack hurts every estimator equally.
        loss = -torch.sqrt(((pr - gt) ** 2).mean()) / SIGMA
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return (kx, ky, om, norm_amp(raw).detach(), ph.detach())


POPS = {
    'swell (in-dist)': lambda r: sea(r, 'swell'),
    'in-band (in-dist)': lambda r: sea(r, 'inband'),
    'highk (OOD)': lambda r: ood_sea(r, 1.5, 2.4),
}


def evaluate(P, Pq, gen, gain, n=32):
    out = {}
    for name, mk in POPS.items():
        h, l, zs = [], [], []
        for i in range(n):
            rng = np.random.default_rng(400000 + i)
            s = mk(rng)
            z, gt = obs(P, Pq, s, gen)
            with torch.no_grad():
                h.append(float(ratio(P, z, Pq, gt, hand_select(P, z, gen))))
                pr, var = mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP,
                                   band=WIDE, prior_gain=gain, return_var=True)
                pr = dm(pr)
                sd = torch.sqrt(var.clamp_min(1e-10))
                l.append(float(torch.sqrt(((pr - gt) ** 2).mean()) / gt.std()))
                zs.append(float(((pr - gt) / sd).std()))
        out[name] = dict(hand=np.mean(h), learned=np.mean(l), z_std=np.mean(zs))
    return out


def fresh_adversary(P, Pq, gen, gain, seed, steps=300):
    rng = np.random.default_rng(seed)
    kx, ky, om, _ = sea_shell(rng)
    raw = torch.tensor(rng.normal(0, 1, NCOMP), device=DEV, dtype=torch.float32,
                       requires_grad=True)
    ph = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                      dtype=torch.float32, requires_grad=True)
    st = refresh_adversary(P, Pq, gen, gain, (kx, ky, om, raw, ph), steps)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--eval', type=int, default=32)
    a_ = ap.parse_args()

    P, Pq = geometry()
    gen = torch.Generator(device=DEV).manual_seed(0)
    kmag, _ = bank_kmag()
    qi = torch.randperm(len(Pq), device=DEV, generator=gen)[:NQ]

    net = SafeGain().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a_.steps)
    rng = np.random.default_rng(2)

    # the persistent adversary
    ar = np.random.default_rng(77)
    kx, ky, om, _ = sea_shell(ar)
    raw = torch.tensor(ar.normal(0, 1, NCOMP), device=DEV, dtype=torch.float32,
                       requires_grad=True)
    ph = torch.tensor(ar.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                      dtype=torch.float32, requires_grad=True)
    adv_state = (kx, ky, om, raw, ph)

    print("training: minimax + calibration term |z-std - 1|\n", flush=True)
    for st in range(1, a_.steps + 1):
        gain = net(kmag)
        adv = refresh_adversary(P, Pq, gen, gain, adv_state, steps=2)
        adv_state = (adv[0], adv[1], adv[2], raw, ph)

        kind = 'swell' if rng.random() < 0.5 else 'inband'
        seas = [sea(rng, kind), adv]
        L, accs, zss = 0.0, [], []
        for s in seas:
            li, acc, zs = loss_on(P, Pq, s, gen, gain, qi)
            L = L + li
            accs.append(float(acc))
            zss.append(float(zs))
        L = L / len(seas)
        opt.zero_grad(set_to_none=True)
        L.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if st % 50 == 0:
            print(f"  step {st:4d}  loss {float(L):.4f}   "
                  f"acc rand {accs[0]:.3f} adv {accs[1]:.3f}   "
                  f"z-std rand {zss[0]:.2f} adv {zss[1]:.2f}", flush=True)

    with torch.no_grad():
        gain = net(kmag)
    res = evaluate(P, Pq, gen, gain, a_.eval)
    print(f"\nheld-out seas ({a_.eval} each):")
    print(f"{'population':<20}{'hand':>8}{'L2':>8}{'z-std':>9}")
    for k, d in res.items():
        print(f"{k:<20}{d['hand']:>8.3f}{d['learned']:>8.3f}{d['z_std']:>9.2f}")

    # a fresh adversary, found against the FINAL prior
    st = fresh_adversary(P, Pq, gen, gain, 909, steps=400)
    z, gt = obs(P, Pq, st, gen)
    with torch.no_grad():
        rh = float(ratio(P, z, Pq, gt, hand_select(P, z, gen)))
        pr, var = mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP, band=WIDE,
                           prior_gain=gain, return_var=True)
        pr = dm(pr)
        sd = torch.sqrt(var.clamp_min(1e-10))
        rl = float(torch.sqrt(((pr - gt) ** 2).mean()) / gt.std())
        zl = float(((pr - gt) / sd).std())
    print(f"\nadversary found against the FINAL prior:")
    print(f"  hand {rh:.3f}    L2 {rl:.3f}    z-std {zl:.2f}")

    g1 = rl <= rh + 0.01
    g2 = res['highk (OOD)']['learned'] <= res['highk (OOD)']['hand'] + 0.01
    zs = [d['z_std'] for d in res.values()] + [zl]
    g3 = all(0.5 <= v <= 2.0 for v in zs)
    g4 = (res['swell (in-dist)']['hand'] - res['swell (in-dist)']['learned']) >= 0.02
    print(f"\n=== L2 gates (pre-registered) ===")
    print(f"  1) adversarial <= hand + 0.01 : {'PASS' if g1 else 'FAIL'}"
          f"  ({rl:.3f} vs {rh:.3f})")
    print(f"  2) OOD high-k <= hand + 0.01  : {'PASS' if g2 else 'FAIL'}"
          f"  ({res['highk (OOD)']['learned']:.3f} vs "
          f"{res['highk (OOD)']['hand']:.3f})")
    print(f"  3) z-std in [0.5, 2] everywhere: {'PASS' if g3 else 'FAIL'}"
          f"  ({min(zs):.2f} .. {max(zs):.2f})   [L1 gave 68710]")
    print(f"  4) in-dist gain >= 0.02        : {'PASS' if g4 else 'FAIL'}"
          f"  ({res['swell (in-dist)']['hand']-res['swell (in-dist)']['learned']:+.3f})")

    torch.save(net.state_dict(), CKPT)
    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'heldout': {k: {kk: float(vv) for kk, vv in v.items()}
                               for k, v in res.items()},
                   'adversarial': dict(hand=rh, l2=rl, z_std=zl),
                   'gates': dict(adv=bool(g1), ood=bool(g2), cal=bool(g3),
                                 gain=bool(g4))}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
