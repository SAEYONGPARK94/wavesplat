"""A: does anything we learned today survive contact with the ocean?

Today's learning results all live on synthetic seas:

  L1   a radial prior gain g(k/kp) that beats the hand rule and the discrete
       oracle in distribution, and is 45% worse out of it
  L3a  a trust rule that picks between the learned prior and the hand rule by
       held-out observation residual, keeping 97% of the gain and none of the harm

NONE of it has touched a real record. The gain is a function of k/kp, so it
should transfer by construction: build the bank around the record's own kp and
evaluate the same network. Whether "by construction" survives a JONSWAP peak,
real noise and 30-40% dead pixels is the experiment.

Three outcomes, and each changes what the thesis is:

  transfer works        the learning chapter stands
  transfer is neutral   the learning chapter is a synthetic curiosity
  transfer is harmful   the trust rule must catch it -- and if it does, THAT is
                        the result: the gate matters more than the prior

Five seeds per record, demeaned, scored in-fan, normalised by the truth's own
std so the zero-predictor scores 1.000. z-std is reported because L3 showed the
learned prior destroys calibration on seas it does not know.
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
from d3_l1_learned_prior import RadialGain
from d3_a2_real_revalidation import draw, CANDS, SEEDS, HELD
import d3_static_field as SF

OUT = 'results/direction3/a_real_learning_transfer.json'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RECS = ['YS17', 'AA14', 'AA15a', 'LJ18g']
WIDE = (0.15, 2.5)


def gain_for(net, kp, depth):
    """The SAME network, re-anchored on this record's own peak wavenumber.
    g is a function of k/kp, so the bank moves with the sea and the network
    does not have to know anything about the record."""
    kxb, kyb, _ = build_bank(kp, WIDE, kp / 10.0, depth, DEV, torch.float32)
    kmag = torch.hypot(kxb, kyb)
    with torch.no_grad():
        x = torch.log(kmag / kp).unsqueeze(1)
        return torch.exp(net.f(x).squeeze(1).clamp(-6.0, 6.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recs', nargs='*', default=RECS)
    a_ = ap.parse_args()

    net = RadialGain().to(DEV)
    net.load_state_dict(torch.load('results/direction3/l1_prior_gain.pt'))
    net.eval()
    t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    gen = torch.Generator(device=DEV).manual_seed(0)
    res = {}

    print(f"{'record':<8}{'lam_p':>7}{'hand':>16}{'LEARNED':>16}"
          f"{'TRUST':>16}{'oracle':>9}   z-std(learn)  trusts learned")
    for rec in a_.recs:
        sc = SF.scene_real(rec)
        Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
        eta = sc['make_eta']()
        g_all = eta(Xq.ravel(), Yq.ravel(), sc['t0'])
        cx, cy = sc['center']
        inside, _ = SF.fan_mask(Xq, Yq, cx, cy, sc['hvec'])
        m = inside.ravel() & np.isfinite(g_all)
        Pq = t(np.column_stack([Xq.ravel()[m], Yq.ravel()[m],
                                np.full(m.sum(), sc['t0'])]))
        gtq = t(g_all[m])
        gtq = gtq - gtq.mean()
        zero = float(gtq.std())
        sig0 = 0.05 * sc['Hs']
        gain = gain_for(net, sc['kp'], sc['depth'])

        rows, picks = [], 0
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            P, Z = draw(sc, rng)
            Pt, zt = t(P), t(Z)

            def sol(band, g=None, var=False):
                out = mmse_gpu(Pt, zt, sc['depth'], Pq, sigma0=sig0,
                               kp_hint=sc['kp'], band=band, prior_gain=g,
                               return_var=var)
                if var:
                    pr, v = out
                    return pr - pr.mean(), v
                return out - out.mean()

            def rat(pr):
                return float(torch.sqrt(((pr - gtq) ** 2).mean()) / zero)

            with torch.no_grad():
                n = len(zt)
                idx = torch.randperm(n, device=DEV, generator=gen)
                tr, he = idx[:int((1 - HELD) * n)], idx[int((1 - HELD) * n):]

                # hand rule: best of four bands by held-out observation residual
                best, hb = None, CANDS[0]
                for b in CANDS:
                    pv = mmse_gpu(Pt[tr], zt[tr], sc['depth'], Pt[he],
                                  sigma0=sig0, kp_hint=sc['kp'], band=b)
                    r = float(torch.sqrt(((pv - zt[he]) ** 2).mean()))
                    if best is None or r < best:
                        best, hb = r, b
                r_hand_resid = best

                # learned prior on the wide band, same held-out residual
                pv = mmse_gpu(Pt[tr], zt[tr], sc['depth'], Pt[he], sigma0=sig0,
                              kp_hint=sc['kp'], band=WIDE, prior_gain=gain)
                r_learn_resid = float(torch.sqrt(((pv - zt[he]) ** 2).mean()))

                use_learned = r_learn_resid < r_hand_resid
                picks += use_learned

                pr_h = sol(hb)
                pr_l, var_l = sol(WIDE, gain, var=True)
                sd_l = torch.sqrt(var_l.clamp_min(1e-10))
                zsl = float(((pr_l - gtq) / sd_l).std())
                pr_t = pr_l if use_learned else pr_h
                orc = min(rat(sol(b)) for b in CANDS)
            rows.append((rat(pr_h), rat(pr_l), rat(pr_t), orc, zsl))

        v = np.array(rows)
        print(f"{rec:<8}{2*np.pi/sc['kp']:>7.0f}"
              f"{v[:,0].mean():>11.3f}±{v[:,0].std():.3f}"
              f"{v[:,1].mean():>11.3f}±{v[:,1].std():.3f}"
              f"{v[:,2].mean():>11.3f}±{v[:,2].std():.3f}"
              f"{v[:,3].mean():>9.3f}   {v[:,4].mean():>10.2f}"
              f"   {picks}/{len(SEEDS)}", flush=True)
        res[rec] = dict(lam_p=2 * np.pi / sc['kp'], hand=v[:, 0].mean(),
                        learned=v[:, 1].mean(), trust=v[:, 2].mean(),
                        oracle=v[:, 3].mean(), z_std=v[:, 4].mean(),
                        trust_picks_learned=picks / len(SEEDS),
                        hand_sd=v[:, 0].std(), trust_sd=v[:, 2].std())

    print("\n1.000 = as good as predicting zero.  +- is the SEED sd.\n")
    dl = np.mean([res[r]['hand'] - res[r]['learned'] for r in res])
    dt = np.mean([res[r]['hand'] - res[r]['trust'] for r in res])
    zs = [res[r]['z_std'] for r in res]
    print(f"learned prior vs hand, averaged over records: {dl:+.4f}")
    print(f"trust rule  vs hand, averaged over records:   {dt:+.4f}")
    print(f"z-std of the learned prior on real seas: "
          f"{min(zs):.2f} .. {max(zs):.2f}   (calibrated = 1.0)")
    worst = max(res, key=lambda r: res[r]['learned'] - res[r]['hand'])
    print(f"worst record for the learned prior: {worst}  "
          f"({res[worst]['learned']:.3f} vs hand {res[worst]['hand']:.3f})")
    print(f"and the trust rule there:            {res[worst]['trust']:.3f}  "
          f"(picked learned on {res[worst]['trust_picks_learned']*100:.0f}% "
          f"of seeds)")

    if dl > 0.01:
        print("\nTRANSFER WORKS: the learned prior helps on real seas.")
    elif dl > -0.01:
        print("\nTRANSFER NEUTRAL: the learning chapter is synthetic-only.")
    else:
        print("\nTRANSFER HARMFUL: the prior hurts. Then the only question is "
              "whether\nthe trust rule caught it -- see the TRUST column.")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in res.items()}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
