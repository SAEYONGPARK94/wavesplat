"""E3: the criterion, done properly. Fixed-prior evidence, BMA, time-forward CV.

E2 measured evidence, SURE and GCV on a gamma the solver had already fitted to
the data -- held-out choice of (pexp, smult), then SBL-EM. An evidence computed
on a tuned prior has no Occam factor left to spend, and SURE's sigma^2 was
itself a selected quantity. All three failed, and the failure said nothing about
the criteria.

The solver now takes `pexp, smult, no_em` and freezes the prior. So:

  1  EVIDENCE, fixed prior.  -2 log N(z | 0, K) = z' K^-1 z + log|K|.
     Now log|K| is a real model evidence for the candidate. A model that gives
     large prior variance to the longest atoms -- and thereby absorbs a
     range-dependent sensor bias, which O1 showed lowers the held-out residual
     -- pays for the volume of data it could have explained and did not.

  2  BMA. Do not select at all. Weight the twenty fields by their evidence and
     average. Selection error is the thing we have been fighting since E1; model
     averaging is the classical way to refuse the fight.

  4  TIME-FORWARD residual. Every held-out split we have tried -- random points,
     4-fold, spatial wedges -- hides points BETWEEN their neighbours in time.
     The last scan is different: predicting it requires the model to have got
     the dispersion right, not just to interpolate. Hold out the final 25% of
     the window.

SURE and GCV come along with the true sigma_0 this time, since the prior is
frozen and sigma^2 is no longer a fitted quantity.

PRE-REGISTERED, on synthetic seas with and without the injected sensor bias,
twenty candidates, regret against the sea's own best:
  1. on BIAS seas, fixed-prior evidence regret < 0.03 (the held-out residual
     gives 0.1297, E2's post-hoc evidence 0.1068)
  2. on CLEAN seas, no criterion worse than the residual by more than 0.01
  3. BMA's field error must beat the residual rule's selected field
  4. NEGATIVE CONTROL: with the prior frozen, SURE must now behave -- if it is
     still ten times worse than the residual on clean seas, the freeze did not
     take and the whole comparison is void.
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
from d3_l1_learned_prior import RadialGain, S0
from d3_o1_defect_ablation import synth_sea, observe, gains_for
from d3_g3_adversarial_sea import geometry, DEV, DEPTH, KP
from d3_a2_real_revalidation import HELD

OUT = 'results/direction3/e3_fixed_prior_criteria.json'
PEXP, SMULT = 1.0, 1.0
CRIT = ('residual', 'resid_time', 'evidence', 'sure', 'gcv')


def solve(P, z, Pq, depth, kp, band, gain, sig0, model=False):
    return mmse_gpu(P, z, depth, Pq, sigma0=sig0, kp_hint=kp, band=band,
                    prior_gain=gain, pexp=PEXP, smult=SMULT, no_em=True,
                    return_model=model)


def one_candidate(P, z, Pq, gt, depth, kp, band, gain, sig0, tr, he, ttr, the):
    n = len(z)
    with torch.no_grad():
        pv = solve(P[tr], z[tr], P[he], depth, kp, band, gain, sig0)
        r_rand = float(torch.sqrt(((pv - z[he]) ** 2).mean()))

        pt = solve(P[ttr], z[ttr], P[the], depth, kp, band, gain, sig0)
        r_time = float(torch.sqrt(((pt - z[the]) ** 2).mean()))

        _, mo = solve(P, z, P[:1], depth, kp, band, gain, sig0, model=True)
        K, s2 = mo['K'], float(mo['s2'])
        Ki = torch.linalg.inv(K)
        _, logdet = torch.linalg.slogdet(K)
        ev = float(z @ (Ki @ z) + logdet)

        H = (K - s2 * torch.eye(n, device=DEV, dtype=K.dtype)) @ Ki
        dof = float(torch.diagonal(H).sum())
        rss = float(((z - H @ z) ** 2).sum())
        sure = rss / n - s2 + 2 * s2 * dof / n
        gcv = (rss / n) / max((1 - dof / n) ** 2, 1e-6)

        pr = solve(P, z, Pq, depth, kp, band, gain, sig0)
        pr = pr - pr.mean()
        rat = float(torch.sqrt(((pr - gt) ** 2).mean()) / gt.std())
    return dict(residual=r_rand, resid_time=r_time, evidence=ev, sure=sure,
                gcv=gcv), rat, pr, dof


def score_sea(P, z, Pq, gt, kp, depth, gains, sig0, gen):
    n = len(z)
    idx = torch.randperm(n, device=DEV, generator=gen)
    tr, he = idx[:int((1 - HELD) * n)], idx[int((1 - HELD) * n):]
    t = P[:, 2]
    cut = torch.quantile(t, 1 - HELD)
    ttr, the = torch.nonzero(t <= cut).squeeze(1), torch.nonzero(t > cut).squeeze(1)
    if len(the) < 20:
        return None

    C, rats, fields, dofs = [], [], [], []
    for (band, al), g in gains.items():
        c, rat, pr, dof = one_candidate(P, z, Pq, gt, depth, kp, band, g, sig0,
                                        tr, he, ttr, the)
        C.append([c[k] for k in CRIT])
        rats.append(rat)
        fields.append(pr)
        dofs.append(dof)
    C, r = np.array(C), np.array(rats)
    best = r.min()
    out = {k: r[int(np.argmin(C[:, i]))] - best for i, k in enumerate(CRIT)}

    # BMA over the evidence: w ~ exp(-ev/2), ev is already -2 log N
    ev = C[:, CRIT.index('evidence')]
    w = np.exp(-0.5 * (ev - ev.min()))
    w = w / w.sum()
    with torch.no_grad():
        pr_bma = sum(float(wi) * f for wi, f in zip(w, fields))
        r_bma = float(torch.sqrt(((pr_bma - gt) ** 2).mean()) / gt.std())
    out['bma'] = r_bma - best
    out['_eff_models'] = float(1.0 / (w ** 2).sum())
    out['_dof'] = float(np.mean(dofs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seas', type=int, default=30)
    a_ = ap.parse_args()

    net = RadialGain().to(DEV)
    net.load_state_dict(torch.load('results/direction3/l1_prior_gain.pt'))
    net.eval()
    P0, Pq = geometry()
    gen = torch.Generator(device=DEV).manual_seed(0)
    keys = list(CRIT) + ['bma']
    res = {}

    print(f"prior frozen at pexp={PEXP}, smult={SMULT}, no EM\n")
    print(f"{'seas':<12}" + "".join(f"{c:>16}" for c in keys)
          + f"{'eff models':>12}{'dof':>8}")
    for cfg in ('none', 'bias'):
        rows = {k: [] for k in keys}
        eff, dof = [], []
        for i in range(a_.seas):
            rng = np.random.default_rng(5000 + i)
            s = synth_sea(rng)
            P, z, gt = observe(P0, Pq, s, rng, gen, cfg)
            if len(z) < 150:
                continue
            gains = gains_for(net, KP, DEPTH)
            o = score_sea(P, z, Pq, gt, KP, DEPTH, gains, S0, gen)
            if o is None:
                continue
            for k in keys:
                rows[k].append(o[k])
            eff.append(o['_eff_models'])
            dof.append(o['_dof'])
        m = {k: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))))
             for k, v in rows.items()}
        res[cfg] = m
        print(f"{'synth ' + cfg:<12}"
              + "".join(f"{m[c][0]:>10.4f}±{m[c][1]:.4f}" for c in keys)
              + f"{np.mean(eff):>12.1f}{np.mean(dof):>8.1f}", flush=True)

    print("\nregret = field nRMSE of the choice minus the sea's best of twenty.")
    print("bma does not choose; its regret is the averaged field's.\n")

    b, c = res['bias'], res['none']
    g1 = b['evidence'][0] < 0.03
    g2 = all(c[k][0] <= c['residual'][0] + 0.01 for k in
             ('evidence', 'sure', 'gcv', 'resid_time'))
    g3 = b['bma'][0] < b['residual'][0] and c['bma'][0] < c['residual'][0]
    g4 = c['sure'][0] <= 10 * c['residual'][0]
    print("=== E3 gates (pre-registered) ===")
    print(f"  1) evidence regret < 0.03 on bias : {'PASS' if g1 else 'FAIL'}"
          f"  ({b['evidence'][0]:.4f} vs residual {b['residual'][0]:.4f})")
    print(f"  2) nothing much worse on clean    : {'PASS' if g2 else 'FAIL'}")
    print(f"  3) BMA beats the residual choice  : {'PASS' if g3 else 'FAIL'}"
          f"  (bias {b['bma'][0]:.4f} vs {b['residual'][0]:.4f}, "
          f"clean {c['bma'][0]:.4f} vs {c['residual'][0]:.4f})")
    print(f"  4) freeze took (SURE sane)        : {'PASS' if g4 else 'FAIL'}"
          f"  (clean SURE {c['sure'][0]:.4f})")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({k: {kk: list(vv) for kk, vv in v.items()}
                   for k, v in res.items()}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
