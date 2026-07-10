"""E1: learning proposes, the observations dispose.

Today killed one use of learning and quietly demonstrated another.

  killed     a learned gamma(k) deployed as THE prior. On real records it is
             0.0345 worse than the hand rule, it destroys calibration (z-std up
             to 3.27), and adversarial training does not fix it.

  survived   on LJ18g -- the storm, the long swell, the record where the band
             actually matters -- that same learned prior BEATS the hand rule:
             0.679 vs 0.719. The gate picked it on 3 of 5 seeds.

So the learned prior is not wrong. It is right somewhere and wrong elsewhere,
and today's gate already knows which is which, using held-out observations and
never the truth.

The gate was given two choices. Give it a family.

    gain_alpha(k) = g(k/kp) ** alpha ,    alpha in {0, .25, .5, .75, 1}

alpha = 0 is exactly the analytic prior (gain = 1 everywhere, verified
|diff| = 0.0). alpha = 1 is L1's learned prior. In between is a continuum of
partially-trusted priors. Crossed with the four candidate bands, the gate ranks
twenty physics estimators by their residual on 25% held-out observations and
returns one. Every candidate is a certified MMSE solve; learning never emits a
field, only a proposal.

WHAT THIS BUYS, if it works: the storm gain without the calm-sea harm, chosen
per record, per window, with no ground truth and no retraining.

WHAT WOULD KILL IT, and it is a real risk: twenty candidates ranked on ~150
held-out points is a selection problem. The gate may overfit the residual and
pick a prior that explains those 150 points while wrecking the field. Two
choices could not overfit; twenty might. That is the experiment.

PRE-REGISTERED, on the same 30 real record-window cells as C:
  1. beat the hand rule (alpha = 0, four bands) by >= +0.010 paired
  2. lose <= 0.005 on the records where the hand rule already reaches oracle
     (YS17, AA15b -- calm, energy inside the band)
  3. the selection must not be overfitting: gap to the ORACLE OVER THE SAME 20
     candidates must not exceed the hand rule's gap to its own 4-candidate
     oracle by more than 0.010
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
from d3_a2_real_revalidation import draw, CANDS, HELD
import d3_static_field as SF

OUT = 'results/direction3/e1_prior_family_gate.json'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RECS = ['AA14', 'AA15a', 'AA15b', 'LJ18g', 'YS17']
WINDOWS = (0, 1, 2, 3, 4, 5)
SEEDS = (11, 12, 13)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
FOLDS = 4          # set by --folds
SPLIT = 'random'   # set by --split
CALM = ('YS17', 'AA15b')


def gains_for(net, kp, depth):
    """One gain vector per (band, alpha). alpha = 0 reproduces the analytic
    prior exactly, so the family strictly contains the hand rule."""
    out = {}
    for band in CANDS:
        kxb, kyb, _ = build_bank(kp, band, kp / 10.0, depth, DEV,
                                 torch.float32)
        kmag = torch.hypot(kxb, kyb)
        with torch.no_grad():
            x = torch.log(kmag / kp).unsqueeze(1)
            g = torch.exp(net.f(x).squeeze(1).clamp(-6.0, 6.0))
        for a in ALPHAS:
            out[(band, a)] = None if a == 0.0 else g ** a
    return out


def cell(sc, gains, gen, t):
    Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
    eta = sc['make_eta']()
    g_all = eta(Xq.ravel(), Yq.ravel(), sc['t0'])
    cx, cy = sc['center']
    inside, _ = SF.fan_mask(Xq, Yq, cx, cy, sc['hvec'])
    m = inside.ravel() & np.isfinite(g_all)
    if m.sum() < 150:
        return None
    Pq = t(np.column_stack([Xq.ravel()[m], Yq.ravel()[m],
                            np.full(m.sum(), sc['t0'])]))
    gtq = t(g_all[m])
    gtq = gtq - gtq.mean()
    zero = float(gtq.std())
    if not np.isfinite(zero) or zero < 1e-6:
        return None
    sig0 = 0.05 * sc['Hs']

    rows = []
    for sd in SEEDS:
        rng = np.random.default_rng(1000 + sd)
        P, Z = draw(sc, rng)
        if len(Z) < 200:
            continue
        Pt, zt = t(P), t(Z)
        n = len(zt)
        if SPLIT == 'random':
            idx = torch.randperm(n, device=DEV, generator=gen)
            folds = [idx[i::FOLDS] for i in range(FOLDS)]
        else:
            # BLOCK split. Random held-out points sit between their neighbours,
            # so the residual only ever measures INTERPOLATION -- while the
            # field is scored between and beyond the samples. A prior with more
            # freedom can interpolate better and extrapolate worse, which is
            # exactly the bias we measured: averaging over folds made the
            # overfit worse (+0.0157 -> +0.0181), so it was never noise.
            # Holding out a contiguous angular wedge forces the criterion to
            # measure what the task measures.
            ang = torch.atan2(Pt[:, 1] - cy, Pt[:, 0] - cx)
            off = 2 * np.pi * (sd % FOLDS) / FOLDS
            a01 = torch.remainder(ang + off, 2 * np.pi) / (2 * np.pi)
            folds = [torch.nonzero((a01 >= i / FOLDS) &
                                   (a01 < (i + 1) / FOLDS)).squeeze(1)
                     for i in range(FOLDS)]
            folds = [f for f in folds if len(f) >= 20]
            if len(folds) < 2:
                continue

        def field(band, g):
            pr = mmse_gpu(Pt, zt, sc['depth'], Pq, sigma0=sig0,
                          kp_hint=sc['kp'], band=band, prior_gain=g)
            return pr - pr.mean()

        def resid(band, g):
            """K-fold held-out observation residual.

            The single 25% split gives ~150 scoring points. With four
            candidates that is enough; with twenty it is not -- E1 measured the
            gate losing +0.0157 more of its own oracle's headroom than the hand
            rule loses of its. Before blaming the criterion, give it less noise:
            average the residual over K disjoint folds. If the overfit closes,
            the gate was starved, not wrong."""
            tot = 0.0
            for f in range(len(folds)):
                he = folds[f]
                tr = torch.cat([folds[j] for j in range(len(folds)) if j != f])
                pv = mmse_gpu(Pt[tr], zt[tr], sc['depth'], Pt[he], sigma0=sig0,
                              kp_hint=sc['kp'], band=band, prior_gain=g)
                tot += float(torch.sqrt(((pv - zt[he]) ** 2).mean()))
            return tot / len(folds)

        def rat(pr):
            return float(torch.sqrt(((pr - gtq) ** 2).mean()) / zero)

        with torch.no_grad():
            # hand rule: alpha = 0 only
            r_h, b_h = None, CANDS[0]
            for b in CANDS:
                r = resid(b, None)
                if r_h is None or r < r_h:
                    r_h, b_h = r, b
            hand = rat(field(b_h, None))
            hand_oracle = min(rat(field(b, None)) for b in CANDS)

            # family: 4 bands x 5 alphas
            r_f, key = None, (CANDS[0], 0.0)
            for (b, a), g in gains.items():
                r = resid(b, g)
                if r_f is None or r < r_f:
                    r_f, key = r, (b, a)
            fam = rat(field(key[0], gains[key]))
            fam_oracle = min(rat(field(b, g)) for (b, a), g in gains.items())
        rows.append((hand, hand_oracle, fam, fam_oracle, key[1]))
    if not rows:
        return None
    v = np.array(rows)
    return v.mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recs', nargs='*', default=RECS)
    ap.add_argument('--split', choices=('random', 'block'),
                    default='random')
    ap.add_argument('--folds', type=int, default=4,
                    help='K-fold residual. 4 keeps the original '
                         '75/25 train/score ratio; K=1 is not a split.')
    a_ = ap.parse_args()
    global FOLDS
    if a_.folds < 2:
        raise SystemExit('--folds must be >= 2; K=1 leaves no training points')
    FOLDS = a_.folds
    global SPLIT
    SPLIT = a_.split

    net = RadialGain().to(DEV)
    net.load_state_dict(torch.load('results/direction3/l1_prior_gain.pt'))
    net.eval()
    t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    gen = torch.Generator(device=DEV).manual_seed(0)

    print(f"{'record':<8}{'win':>5}{'hand':>9}{'hand orc':>10}"
          f"{'FAMILY':>9}{'fam orc':>10}{'alpha':>8}")
    cells, per = [], {}
    for rec in a_.recs:
        rows = []
        for w in WINDOWS:
            try:
                sc = SF.scene_real(rec, win=w)
                # kp is estimated per window, so the bank -- and therefore the
                # gain vector's length -- changes with it. Building the gains
                # once from window 0 raised RuntimeError on every other window.
                gains = gains_for(net, sc['kp'], sc['depth'])
                c = cell(sc, gains, gen, t)
            except Exception as e:
                print(f"{rec:<8}{w:>5}   skip: {type(e).__name__}", flush=True)
                continue
            if c is None:
                continue
            print(f"{rec:<8}{w:>5}{c[0]:>9.3f}{c[1]:>10.3f}"
                  f"{c[2]:>9.3f}{c[3]:>10.3f}{c[4]:>8.2f}", flush=True)
            rows.append(c)
            cells.append(c)
        if rows:
            r = np.array(rows)
            per[rec] = dict(hand=r[:, 0].mean(), hand_oracle=r[:, 1].mean(),
                            family=r[:, 2].mean(), fam_oracle=r[:, 3].mean(),
                            alpha=r[:, 4].mean(), n=len(rows))

    print(f"\n{'record':<8}{'hand':>9}{'FAMILY':>9}{'gain':>9}"
          f"{'mean alpha':>12}{'sel. overfit':>14}")
    for rec, d in per.items():
        g = d['hand'] - d['family']
        of = (d['family'] - d['fam_oracle']) - (d['hand'] - d['hand_oracle'])
        print(f"{rec:<8}{d['hand']:>9.3f}{d['family']:>9.3f}{g:>+9.3f}"
              f"{d['alpha']:>12.2f}{of:>+14.3f}")

    a = np.array(cells)
    d = a[:, 0] - a[:, 2]                         # hand - family, paired
    se = d.std(ddof=1) / np.sqrt(len(d))
    calm = np.array([per[r]['hand'] - per[r]['family'] for r in CALM
                     if r in per])
    of = (a[:, 2] - a[:, 3]) - (a[:, 0] - a[:, 1])
    print(f"\npaired over {len(d)} cells:")
    print(f"  family beats hand by {d.mean():+.4f} ± {se:.4f} (se)   "
          f"t = {d.mean()/se:+.2f}   wins {int((d>0).sum())}/{len(d)}")
    print(f"  mean alpha chosen: {a[:,4].mean():.2f}  "
          f"(0 = analytic prior, 1 = fully learned)")
    print(f"  selection overfit (family gap − hand gap to their own oracles): "
          f"{of.mean():+.4f}   worst {of.max():+.4f}")

    g1 = d.mean() >= 0.010
    g2 = (calm.min() >= -0.005) if len(calm) else False
    g3 = of.mean() <= 0.010
    print(f"\n=== E1 gates (pre-registered) ===")
    print(f"  1) beat hand by >= +0.010 : {'PASS' if g1 else 'FAIL'}"
          f"  ({d.mean():+.4f})")
    print(f"  2) calm records lose <= 0.005 : {'PASS' if g2 else 'FAIL'}"
          f"  (worst {calm.min():+.4f})" if len(calm) else "  2) n/a")
    print(f"  3) no selection overfit (<= +0.010) : {'PASS' if g3 else 'FAIL'}"
          f"  ({of.mean():+.4f})")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'per_record': {k: {kk: float(vv) for kk, vv in v.items()}
                                  for k, v in per.items()},
                   'paired_gain': float(d.mean()), 'se': float(se),
                   'mean_alpha': float(a[:, 4].mean()),
                   'overfit_mean': float(of.mean()),
                   'gates': dict(beat=bool(g1), calm=bool(g2),
                                 overfit=bool(g3))}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
