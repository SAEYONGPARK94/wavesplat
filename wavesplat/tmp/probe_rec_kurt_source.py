"""Is record-level excess kurtosis wave physics, or stereo spikes?

LORO probing exposed a spread of 4.4 in record-mean excess kurtosis. Second-
order Stokes on a real sea buys about +0.2; a steep sea a few tenths. A spread
of 4.4 is not wave physics.

Hypothesis: the tail is stereo reconstruction outliers — isolated bad-match
pixels that survive the validity mask.

Test: recompute each record's kurtosis after winsorising every patch at +-4
sigma. Clipping removes <0.01% of pixels if the field is smooth, and removes
the spikes if it is not. If kurtosis collapses, the record-level statistic is
measuring the sensor, not the sea.

Consequence if confirmed: excess kurtosis cannot be the CL1 regressor on real
records until CL7 (per-pixel trust) separates wave from artifact. On synthetic
seas, where kurtosis is known by construction, CL1 is unaffected.
"""
import os
import sys

import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

from d3_ssl_pretrain import Patches, HELD_OUT

N_BATCH, BS = 64, 64
CLIP = 4.0


def kurt(x, axis=1):
    x = x - x.mean(axis, keepdims=True)
    v = x.var(axis) + 1e-12
    return (x ** 4).mean(axis) / v ** 2 - 3.0


def main():
    ds = Patches(HELD_OUT, exclude=True)
    rng = np.random.default_rng(3)
    raw, cli, frac, src = [], [], [], []
    for _ in range(N_BATCH):
        xb, ridx = ds.batch(rng, BS, want_rec=True)
        p = xb.numpy().reshape(len(xb), -1).astype(np.float64)
        raw.append(kurt(p))
        s = p.std(1, keepdims=True)
        out = np.abs(p) > CLIP * s
        cli.append(kurt(np.clip(p, -CLIP * s, CLIP * s)))
        frac.append(out.mean(1))
        src.append(ridx)
    raw, cli = np.concatenate(raw), np.concatenate(cli)
    frac, src = np.concatenate(frac), np.concatenate(src)

    print(f"{'record':<26} {'raw kurt':>9} {'clipped':>9} {'drop':>7} "
          f"{'>4sig px':>9}")
    rows = []
    for i in np.unique(src):
        m = src == i
        rows.append((ds.names[i], raw[m].mean(), cli[m].mean(),
                     raw[m].mean() - cli[m].mean(), frac[m].mean() * 100))
    for n, r, c, d, f in sorted(rows, key=lambda z: -z[1]):
        print(f"{n[:26]:<26} {r:+9.3f} {c:+9.3f} {d:7.3f} {f:8.3f}%")

    R = np.array([r[1] for r in rows])
    C = np.array([r[2] for r in rows])
    print(f"\nspread across records:  raw {R.std():.3f}  ->  clipped {C.std():.3f}")
    print(f"max record kurt:        raw {R.max():+.3f}  ->  clipped {C.max():+.3f}")
    if C.std() < 0.4 * R.std():
        print("\n[!] record-level kurtosis is dominated by isolated spikes.")
        print("    It measures the stereo reconstruction, not the sea state.")
        print("    => not usable as the CL1 regressor on real data before CL7.")


if __name__ == '__main__':
    sys.exit(main())
