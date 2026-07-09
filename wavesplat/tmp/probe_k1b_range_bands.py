"""K1b: the LJ18g pathology, with the controls fixed.

K1a taught three things and one of them was a design error of mine.

  * The pathology is far smaller than reported once seeds are averaged.
    Single seed (the published number): 0.550 -> 0.785, +0.235.
    Three seeds, matched sampler:      0.658 +- 0.052 -> 0.724 +- 0.030, +0.066.

  * `all_dense` was never an oracle. It spends its budget uniformly over the
    whole stereo rectangle, most of which lies OUTSIDE the scored fan, while
    `fan_noisy` spends all of it inside. Fewer useful observations, not more.
    Confirmed: at equal count (600) the uniform allocation is WORSE (0.812)
    than the dense one (0.724). "More data hurts" was reading a budget
    reallocation as an information gain.

  * MY ERROR (#14): config E confined observations to R <= 65 m to get clean
    ground truth, and thereby collapsed the aperture. YS17 lost +0.328 from
    that alone. It cannot separate reference quality from aperture. Dropped.

  * LJ18g's near field is 100% removed by the robust dead-pixel mask, so the
    clean ring I wanted to score on does not exist for that record.

So test the range hypothesis the other way round: hold the estimate fixed and
score it in RANGE BANDS inside the fan. Geometry alone makes error grow with
range, and YS17 supplies that baseline growth (its reconstruction is flat to
0.04 m at 240 m). If LJ18g's error grows faster than YS17's, in the same bands
where its reconstruction bias runs 0.4 m -> 7.6 m, the reference is implicated.
If both grow alike, the reference is exonerated and the pathology is geometry.
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
from probe_k1_pathology_decomp import sample, N_BASE, N_DENSE

SEEDS = (11, 12, 13, 14, 15)
BANDS = [(15, 65), (65, 98), (98, 130), (130, 200)]

CONFIGS = [
    ('A fan_noisy', 'fan', N_BASE,  SF.NOISE, 1.0),
    ('B all_dense', 'all', N_DENSE, 0.0,      1.0),
    ('C B+sig fix', 'all', N_DENSE, 0.0,      2.0),
    ('D all n=600', 'all', N_BASE,  0.0,      1.0),
    ('F fan clean', 'fan', N_BASE,  0.0,      1.0),
]


def main():
    out = {}
    for rec in ('YS17', 'LJ18g'):
        sc = SF.scene_real(rec)
        Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
        eta = sc['make_eta']()
        gt = eta(Xq.ravel(), Yq.ravel(), sc['t0']).reshape(Xq.shape)
        Pq = np.column_stack([Xq.ravel(), Yq.ravel(),
                              np.full(Xq.size, sc['t0'])])
        cx, cy = sc['center']
        inside, R = SF.fan_mask(Xq, Yq, cx, cy, sc['hvec'])
        sg = float(np.nanstd(gt))
        sig0 = 0.05 * sc['Hs']

        print(f"\n=== {rec}   Hs {sc['Hs']:.1f} m   in-fan {inside.sum()} px "
              f"  GT finite in-fan {np.isfinite(gt[inside]).mean()*100:.0f}% ===")
        hdr = ''.join(f"{f'{a}-{b}m':>13}" for a, b in BANDS)
        print(f"{'config':<14}{'in-fan':>14}{hdr}")

        for tag, kind, n, noise, sm in CONFIGS:
            acc = {k: [] for k in ['all'] + [f'{a}-{b}' for a, b in BANDS]}
            for sd in SEEDS:
                rng = np.random.default_rng(sd)
                P, Z = sample(sc, kind, n, noise, rng, None)
                pr = mmse_bank_estimate(P, Z, sc['depth'], Pq, sigma0=sig0 * sm,
                                        kp_hint=sc['kp'], seed=sc['seed'],
                                        band=(0.35, 2.5)).reshape(Xq.shape)
                fin = np.isfinite(pr) & np.isfinite(gt)

                def nr(m):
                    mm = m & fin
                    if mm.sum() < 50:
                        return np.nan
                    return float(np.sqrt(np.mean((pr[mm] - gt[mm]) ** 2)) / sg)

                acc['all'].append(nr(inside))
                for a, b in BANDS:
                    acc[f'{a}-{b}'].append(nr(inside & (R >= a) & (R < b)))

            row = f"{tag:<14}"
            m = np.nanmean(acc['all'])
            row += f"{m:>8.3f}+-{np.nanstd(acc['all']):.3f}"
            for a, b in BANDS:
                v = np.array(acc[f'{a}-{b}'])
                row += ("          nan" if np.all(np.isnan(v))
                        else f"{np.nanmean(v):>8.3f}+-{np.nanstd(v):.2f}")
            print(row)
            if tag.startswith('A'):
                out[rec] = {f'{a}-{b}': np.nanmean(acc[f'{a}-{b}'])
                            for a, b in BANDS}

    print("\n--- range growth of the FAN-OBSERVED estimate (config A) ---")
    print(f"{'band':<12}{'YS17':>10}{'LJ18g':>10}   "
          f"(YS17 reconstruction is flat at every range)")
    for a, b in BANDS:
        k = f'{a}-{b}'
        y, l = out['YS17'].get(k, np.nan), out['LJ18g'].get(k, np.nan)
        print(f"{k+' m':<12}{y:>10.3f}{l:>10.3f}")
    ys = np.array([out['YS17'][f'{a}-{b}'] for a, b in BANDS])
    lj = np.array([out['LJ18g'][f'{a}-{b}'] for a, b in BANDS])
    ok = np.isfinite(ys) & np.isfinite(lj)
    if ok.sum() >= 3:
        r = np.array([0.5 * (a + b) for a, b in BANDS])[ok]
        sy = np.polyfit(r, ys[ok], 1)[0] * 100
        sl = np.polyfit(r, lj[ok], 1)[0] * 100
        print(f"\nslope per 100 m:   YS17 {sy:+.3f}   LJ18g {sl:+.3f}")
        print("LJ18g steeper => its far-field reference is implicated.")
        print("similar        => the pathology is geometry, and the reference "
              "is exonerated.")


if __name__ == '__main__':
    sys.exit(main())
