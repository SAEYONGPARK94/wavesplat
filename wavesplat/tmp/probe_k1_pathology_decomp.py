"""K1: decompose the LJ18g oracle pathology (+0.235).

Yesterday: giving the estimator DENSE observations everywhere made the in-fan
field WORSE than giving it sparse noisy fan observations (0.550 -> 0.785). No
band, aperture, bimodality or storm-physics explanation survived.

The comparison had three differences at once, not one:

  fan_noisy   n =  600   noise 0.02 m   fan 15-200 m
  all_dense   n = 2400   noise 0        uniform over the whole stereo patch

so it confounds (a) a 4x observation count at FIXED sigma0 -- the exact
regularization coupling that already produced one retraction -- with (b) noise
and (c) where the observations come from.

And (c) now has teeth: LJ_2018's reconstructed GT degrades with range. After
removing the static per-pixel bias, the temporal-mean surface bows to 7.6 m and
excess kurtosis reaches +8 beyond 130 m, while YS/AA/BS stay under 0.09 m at
any range (tmp/probe_range_quality_all.py). `all_dense` draws its observations
from that far field. `fan_noisy` also reaches 200 m, but only inside +-60 deg.

Six configs, all n = 600 unless stated, isolate one difference at a time:

  A fan_noisy   reproduce the baseline           (600, noise, fan 15-200)
  B all_dense   reproduce the pathology          (2400, clean, everywhere)
  C all_dense + sigma0 * sqrt(2400/600)          <- the I1 fix, alone
  D all         equal count                      (600, clean, everywhere)
  E all_near    equal count, clean GT only        (600, clean, R <= 65)
  F fan         noise removed                    (600, clean, fan 15-200)

Scored twice: on the usual in-fan mask, and on the NEAR-FAN ring (R <= 65 m)
where LJ_2018's GT is trustworthy.

YS17 is the control: its GT is flat at every range, so no config should show
the pathology there. If it does, the cause is not the reference.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

from d3_sa_nomp import mmse_bank_estimate
import d3_static_field as SF

N_BASE = 600
N_DENSE = 2400
R_CLEAN = 65.0          # LJ18g: |bias| 0.31-0.39 m, kurt ~1 inside this
SEEDS = (11, 12, 13)


def sample(sc, kind, n_target, noise, rng, r_max=None):
    """Draw n_target observations of the scene's own eta over W_OBS seconds."""
    cx, cy = sc['center']
    hx, hy = sc['hvec']
    eta = sc['make_eta']()
    t0 = sc['t0']
    dom_x = (sc['gx'].min(), sc['gx'].max())
    dom_y = (sc['gy'].min(), sc['gy'].max())
    scan_ts = np.arange(t0 - SF.W_OBS, t0 + 1e-9, SF.SCAN_DT)
    per = int(np.ceil(4 * n_target / len(scan_ts)))     # oversample, then trim

    P, Z = [], []
    for t_s in scan_ts:                                 # ascending: HOS-safe
        if kind == 'all':
            xs = rng.uniform(*dom_x, per)
            ys = rng.uniform(*dom_y, per)
        else:
            hi = SF.R_FAN[1] if r_max is None else min(SF.R_FAN[1], r_max)
            rr = rng.uniform(SF.R_FAN[0], hi, per)
            aa = rng.uniform(-SF.HALF_FOV, SF.HALF_FOV, per)
            ca, sa = np.cos(aa), np.sin(aa)
            xs = cx + rr * (hx * ca - hy * sa)
            ys = cy + rr * (hx * sa + hy * ca)
        keep = ((xs >= dom_x[0]) & (xs <= dom_x[1]) &
                (ys >= dom_y[0]) & (ys <= dom_y[1]))
        if kind == 'all' and r_max is not None:
            keep &= np.hypot(xs - cx, ys - cy) <= r_max
        xs, ys = xs[keep], ys[keep]
        if len(xs) == 0:
            continue
        zz = eta(xs, ys, t_s)
        ok = np.isfinite(zz)
        if not ok.any():
            continue
        P.append(np.column_stack([xs[ok], ys[ok], np.full(ok.sum(), t_s)]))
        Z.append(zz[ok] + (rng.normal(0, noise, ok.sum()) if noise else 0.0))
    P, Z = np.vstack(P), np.concatenate(Z)
    if len(Z) > n_target:
        ii = rng.choice(len(Z), n_target, replace=False)
        P, Z = P[ii], Z[ii]
    return P, Z


CONFIGS = [
    # tag          kind   n         noise      sig_mult  r_max
    ('A fan_noisy', 'fan', N_BASE,  SF.NOISE,  1.0,      None),
    ('B all_dense', 'all', N_DENSE, 0.0,       1.0,      None),
    ('C B+sig fix',  'all', N_DENSE, 0.0,      2.0,      None),
    ('D all n=600',  'all', N_BASE,  0.0,      1.0,      None),
    ('E all R<=65',  'all', N_BASE,  0.0,      1.0,      R_CLEAN),
    ('F fan clean',  'fan', N_BASE,  0.0,      1.0,      None),
]


def main():
    for rec in ('YS17', 'LJ18g'):
        sc = SF.scene_real(rec)
        Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
        eta = sc['make_eta']()
        gt = eta(Xq.ravel(), Yq.ravel(), sc['t0']).reshape(Xq.shape)
        Pq = np.column_stack([Xq.ravel(), Yq.ravel(),
                              np.full(Xq.size, sc['t0'])])
        cx, cy = sc['center']
        inside, R = SF.fan_mask(Xq, Yq, cx, cy, sc['hvec'])
        near = inside & (R <= R_CLEAN)
        sg = float(np.nanstd(gt))
        sig0 = 0.05 * sc['Hs']

        print(f"\n=== {rec}   Hs {sc['Hs']:.1f} m   "
              f"near-fan px {near.sum()}  in-fan px {inside.sum()} ===")
        print(f"{'config':<14}{'n':>6}{'in-fan':>18}{'near-fan (clean GT)':>24}")
        base = {}
        for tag, kind, n, noise, sm, rmax in CONFIGS:
            vin, vnr = [], []
            for sd in SEEDS:
                rng = np.random.default_rng(sd)
                P, Z = sample(sc, kind, n, noise, rng, rmax)
                fq = mmse_bank_estimate(P, Z, sc['depth'], Pq,
                                        sigma0=sig0 * sm, kp_hint=sc['kp'],
                                        seed=sc['seed'], band=(0.35, 2.5))
                pr = fq.reshape(Xq.shape)
                for m, acc in ((inside, vin), (near, vnr)):
                    mm = m & np.isfinite(pr) & np.isfinite(gt)
                    acc.append(float(np.sqrt(np.mean((pr[mm] - gt[mm]) ** 2)) / sg))
            vin, vnr = np.array(vin), np.array(vnr)
            base.setdefault('A', (vin.mean(), vnr.mean()))
            print(f"{tag:<14}{n:>6}"
                  f"{vin.mean():>11.3f}+-{vin.std():.3f}"
                  f"{vnr.mean():>17.3f}+-{vnr.std():.3f}")
            if tag.startswith('A'):
                base['A'] = (vin.mean(), vnr.mean())
            else:
                print(f"{'':14}{'':6}{vin.mean()-base['A'][0]:>+11.3f}  "
                      f"{'':4}{vnr.mean()-base['A'][1]:>+13.3f}   (vs A)")

    print("\nreading: if the pathology (B worse than A) survives the sigma fix "
          "(C)\nand the equal-count control (D), but dies when observations are "
          "confined to\nthe clean near field (E), then it is the corrupted "
          "far-field reference.\nYS17 must show no pathology in any config.")


if __name__ == '__main__':
    sys.exit(main())
