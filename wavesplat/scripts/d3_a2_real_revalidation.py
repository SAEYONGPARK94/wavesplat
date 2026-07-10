"""A2: everything built today, re-run on real seas with five seeds.

Today produced a certificate (B1), an adaptive band selector (G5, HOS-confirmed
in A1), an energy cap that costs nothing, and a trust rule for the learned
prior (L3a). NONE of it has been checked on the ocean. Worse, there is an open
contradiction:

    synthetic   widening the bank's band is free or helpful, always
    real LJ18g  widening HURT, in-fan 0.550 -> 0.609

D1 predicts the resolution. Lowering the floor adds atoms; on a multi-component
sea those atoms compete with the wind sea for the same 600 returns. Synthetic
seas are clean and their spectra are log-uniform, so the competition is mild.
Real records have a JONSWAP-like peak, real noise and 30-40% dead pixels, so
the degrees-of-freedom bill comes due. If that is right, the adaptive selector
should REFUSE to widen on the records where widening hurts -- which is exactly
what it is for.

And a second debt: today showed that the LJ18g "oracle pathology" was mostly
single-seed noise (+0.235 -> +0.105 over five seeds, seed sd +-0.066). Every
real number in this project is n = 1. Here every cell is five seeds.

Per record, per seed, all demeaned, scored in-fan, normalised by the truth's
own std so the zero-predictor scores 1.000:

    fixed .35     the deployed default
    fixed .15     the wide band D1 says is representationally necessary
    oracle        best of the four candidates, chosen on the TRUTH (unusable)
    ADAPTIVE      chosen by residual on 25% held-out observations (G5)
    + cap         the B1 safeguard on top of ADAPTIVE

Reported with seed standard deviations, because that is the lesson of the day.
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
import d3_static_field as SF

OUT = 'results/direction3/a2_real_revalidation.json'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RECS = ['YS17', 'AA14', 'AA15a', 'BS13a', 'LJ18g']
CANDS = [(0.35, 2.5), (0.25, 2.5), (0.15, 2.5), (0.15, 4.0)]
SEEDS = (11, 12, 13, 14, 15)
HELD = 0.25


def draw(sc, rng):
    """Fan observations of the record's own surface, over the scan window."""
    cx, cy = sc['center']
    hx, hy = sc['hvec']
    eta = sc['make_eta']()
    t0 = sc['t0']
    dx = (sc['gx'].min(), sc['gx'].max())
    dy = (sc['gy'].min(), sc['gy'].max())
    P, Z = [], []
    for t_s in np.arange(t0 - SF.W_OBS, t0 + 1e-9, SF.SCAN_DT):
        rr = rng.uniform(*SF.R_FAN, 4 * SF.M_PER_SCAN)
        aa = rng.uniform(-SF.HALF_FOV, SF.HALF_FOV, 4 * SF.M_PER_SCAN)
        ca, sa = np.cos(aa), np.sin(aa)
        xs = cx + rr * (hx * ca - hy * sa)
        ys = cy + rr * (hx * sa + hy * ca)
        keep = ((xs >= dx[0]) & (xs <= dx[1]) & (ys >= dy[0]) & (ys <= dy[1]))
        xs, ys = xs[keep], ys[keep]
        if len(xs) == 0:
            continue
        zz = eta(xs, ys, t_s)
        ok = np.isfinite(zz)
        if not ok.any():
            continue
        P.append(np.column_stack([xs[ok], ys[ok], np.full(ok.sum(), t_s)]))
        Z.append(zz[ok] + rng.normal(0, SF.NOISE, ok.sum()))
    P, Z = np.vstack(P), np.concatenate(Z)
    if len(Z) > SF.M_CAP:
        ii = rng.choice(len(Z), SF.M_CAP, replace=False)
        P, Z = P[ii], Z[ii]
    return P, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recs', nargs='*', default=RECS)
    a_ = ap.parse_args()

    t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    gen = torch.Generator(device=DEV).manual_seed(0)
    res = {}

    print(f"{'record':<8}{'Hs':>6}{'lam_p':>7}"
          f"{'fixed .35':>16}{'fixed .15':>16}{'oracle':>10}"
          f"{'ADAPTIVE':>16}{'+cap':>9}   widened")
    for rec in a_.recs:
        try:
            sc = SF.scene_real(rec)
        except Exception as e:
            print(f"{rec:<8} skip: {e}")
            continue
        Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
        eta = sc['make_eta']()
        gt_all = eta(Xq.ravel(), Yq.ravel(), sc['t0'])
        cx, cy = sc['center']
        inside, _ = SF.fan_mask(Xq, Yq, cx, cy, sc['hvec'])
        m = inside.ravel() & np.isfinite(gt_all)
        Pq = np.column_stack([Xq.ravel()[m], Yq.ravel()[m],
                              np.full(m.sum(), sc['t0'])])
        gtq = t(gt_all[m])
        gtq = gtq - gtq.mean()
        zero = float(gtq.std())
        sig0 = 0.05 * sc['Hs']

        rows, wide = [], 0
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            P, Z = draw(sc, rng)
            Pt, zt, Qt = t(P), t(Z), t(Pq)

            def sol(band, gain=None):
                pr = mmse_gpu(Pt, zt, sc['depth'], Qt, sigma0=sig0,
                              kp_hint=sc['kp'], band=band, prior_gain=gain)
                return pr - pr.mean()

            def rat(band):
                with torch.no_grad():
                    pr = sol(band)
                    return float(torch.sqrt(((pr - gtq) ** 2).mean()) / zero)

            # adaptive: residual on 25% held-out OBSERVATIONS, no truth
            with torch.no_grad():
                n = len(zt)
                idx = torch.randperm(n, device=DEV, generator=gen)
                tr, he = idx[:int((1 - HELD) * n)], idx[int((1 - HELD) * n):]
                best, bb = None, CANDS[0]
                for b in CANDS:
                    pv = mmse_gpu(Pt[tr], zt[tr], sc['depth'], Pt[he],
                                  sigma0=sig0, kp_hint=sc['kp'], band=b)
                    r = float(torch.sqrt(((pv - zt[he]) ** 2).mean()))
                    if best is None or r < best:
                        best, bb = r, b
                pr_a = sol(bb)
                r_a = float(torch.sqrt(((pr_a - gtq) ** 2).mean()) / zero)
                beta = min(1.0, float((zt - zt.mean()).std())
                           / float(pr_a.std() + 1e-9))
                r_c = float(torch.sqrt(((beta * pr_a - gtq) ** 2).mean()) / zero)
            wide += (bb[0] <= 0.15)
            rows.append((rat(CANDS[0]), rat((0.15, 2.5)),
                         min(rat(b) for b in CANDS), r_a, r_c, beta))

        v = np.array(rows)
        print(f"{rec:<8}{sc['Hs']:>6.1f}{2*np.pi/sc['kp']:>7.0f}"
              f"{v[:,0].mean():>11.3f}±{v[:,0].std():.3f}"
              f"{v[:,1].mean():>11.3f}±{v[:,1].std():.3f}"
              f"{v[:,2].mean():>10.3f}"
              f"{v[:,3].mean():>11.3f}±{v[:,3].std():.3f}"
              f"{v[:,4].mean():>9.3f}   {wide}/{len(SEEDS)}", flush=True)
        res[rec] = dict(Hs=sc['Hs'], lam_p=2 * np.pi / sc['kp'],
                        fix35=v[:, 0].mean(), fix35_sd=v[:, 0].std(),
                        fix15=v[:, 1].mean(), fix15_sd=v[:, 1].std(),
                        oracle=v[:, 2].mean(),
                        adaptive=v[:, 3].mean(), adaptive_sd=v[:, 3].std(),
                        capped=v[:, 4].mean(), beta=v[:, 5].mean(),
                        widened=wide / len(SEEDS))

    print("\n1.000 = as good as predicting zero.  +- is the SEED sd.\n")
    for rec, d in res.items():
        cost = d['fix15'] - d['fix35']
        gain_a = d['fix35'] - d['adaptive']
        print(f"{rec:<8} widening costs {cost:+.3f}   adaptive vs fixed .35 "
              f"{gain_a:+.3f}   vs oracle {d['adaptive']-d['oracle']:+.3f}   "
              f"cap beta {d['beta']:.3f}")

    bad = [r for r, d in res.items() if d['fix15'] > d['fix35'] + 0.01]
    print(f"\nrecords where widening HURTS: {bad if bad else 'none'}")
    for r in bad:
        d = res[r]
        print(f"  {r}: adaptive widened on {d['widened']*100:.0f}% of seeds, "
              f"and scored {d['adaptive']:.3f} vs fixed .35 {d['fix35']:.3f}")
    worst = max(res.values(), key=lambda d: d['adaptive'] - d['oracle'])
    print(f"\nlargest adaptive-vs-oracle gap: "
          f"{worst['adaptive']-worst['oracle']:+.3f}")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in res.items()}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
