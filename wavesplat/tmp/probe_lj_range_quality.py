"""Does stereo ground truth degrade with range?

RETRACTION FIRST. The previous probe compared Grid1 and Grid2 kurtosis and
called the difference a reconstruction artifact. That was wrong: Grid1 is a
40x40 m window at 0.1 m spacing, Grid2 is 300x250 m at 0.5 m. Under a
lam ~ 300 m swell those footprints sample genuinely different statistics. The
comparison was not apples-to-apples and the claim is withdrawn.

What survives, and is not explained by footprint: Grid2's TEMPORAL MEAN surface
has a spatial std of 6.2 m. Averaged over the whole record the sea surface must
be flat to well under a metre. A 6 m bowl in the mean is the stereo
reconstruction, not the ocean.

Test, needing no ground truth: WASS triangulation error grows with range, so if
the mean-surface distortion is reconstruction error, then
   mean-surface |bias|, elevation std, excess kurtosis and >4sigma tail
should all grow with distance from the camera, while the true sea is
statistically homogeneous over 300 m.

Two readings:
  A. per-range-bin statistics inside Grid2.
  B. Grid2 cropped to Grid1's 40 m near-field footprint, vs Grid1 itself
     — now an apples-to-apples pair.

Consequence if range-dependence is strong: our real-data fan spans 15-200 m,
so GT quality is not constant across the evaluation domain. "beyond-range has
no skill" and the unexplained LJ18g oracle pathology (+0.235) would then be
partly a property of the reference, not of the estimator.
"""
import os
import sys

import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

import h5py
from realdata_analysis import _unpack

LJ = 'data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_%s_short.nc'
NT = 96


def load(g):
    f = h5py.File(LJ % g, 'r')
    X = _unpack(f['_nc4_non_coord_X'], np.s_[:])
    Y = _unpack(f['Y'], np.s_[:])
    T = f['Z'].shape[0]
    idx = np.linspace(0, T - 1, NT).astype(int)
    Z = np.stack([_unpack(f['Z'], np.s_[int(i)]) for i in idx])
    m = _unpack(f['mask_Z'], np.s_[0]) if 'mask_Z' in f else None
    f.close()
    valid = np.isfinite(Z).all(0)
    if m is not None:
        valid &= np.isfinite(m)
    return X, Y, Z, valid


def moments(v):
    v = v[np.isfinite(v)]
    if v.size < 500:
        return np.nan, np.nan, np.nan
    s = v.std()
    c = v - v.mean()
    return s, (c ** 4).mean() / s ** 4 - 3.0, float((np.abs(c) > 4 * s).mean() * 100)


def main():
    X, Y, Z, valid = load('Grid2')
    print(f"Grid2  X [{X.min():.0f}, {X.max():.0f}] m   "
          f"Y [{Y.min():.0f}, {Y.max():.0f}] m   valid {valid.mean()*100:.1f}%")

    bias = np.where(valid, Z.mean(0), np.nan)          # temporal mean surface
    Zc = Z - bias                                      # de-biased, as SSL does
    R = np.hypot(X - X[valid].mean() * 0, Y)           # camera near Y = 0
    R = np.hypot(X, Y) if abs(Y.min()) > 1 else np.abs(Y)

    print(f"\n{'range (m)':<14}{'n px':>8}{'|bias| m':>10}{'eta std':>9}"
          f"{'kurt':>9}{'>4sig':>9}")
    edges = np.linspace(np.nanmin(R[valid]), np.nanmax(R[valid]), 7)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = valid & (R >= a) & (R < b)
        if m.sum() < 500:
            continue
        s, k, t = moments(Zc[:, m].ravel())
        ab = float(np.nanmean(np.abs(bias[m] - np.nanmedian(bias[valid]))))
        rows.append((0.5 * (a + b), m.sum(), ab, s, k, t))
        print(f"{a:6.0f}-{b:<7.0f}{m.sum():8d}{ab:10.2f}{s:9.3f}{k:+9.3f}{t:9.3f}")

    r = np.array([x[0] for x in rows])
    print("\ncorrelation with range:")
    for j, nm in ((2, '|bias|'), (3, 'eta std'), (4, 'kurt'), (5, '>4sig tail')):
        v = np.array([x[j] for x in rows])
        print(f"  {nm:<12} corr {np.corrcoef(r, v)[0,1]:+.3f}   "
              f"near {v[0]:.3f} -> far {v[-1]:.3f}")

    # B. apples-to-apples: Grid2 cropped to a 40 m near-field square
    X1, Y1, Z1, v1 = load('Grid1')
    print(f"\nGrid1  X [{X1.min():.0f}, {X1.max():.0f}]  Y [{Y1.min():.0f}, "
          f"{Y1.max():.0f}]  ({X1.max()-X1.min():.0f} m window)")
    xc, yc = 0.5 * (X1.min() + X1.max()), 0.5 * (Y1.min() + Y1.max())
    crop = valid & (np.abs(X - xc) <= 20) & (np.abs(Y - yc) <= 20)
    if crop.sum() > 500:
        s2, k2, t2 = moments(Zc[:, crop].ravel())
        b1 = np.where(v1, Z1.mean(0), np.nan)
        s1, k1, t1 = moments((Z1 - b1)[:, v1].ravel())
        print(f"  Grid1 (0.1 m px):        std {s1:.3f}  kurt {k1:+.3f}  "
              f">4sig {t1:.3f}%")
        print(f"  Grid2 cropped to 40 m:   std {s2:.3f}  kurt {k2:+.3f}  "
              f">4sig {t2:.3f}%")
        print(f"  same footprint, |dkurt| = {abs(k1-k2):.3f}")
    else:
        print("  Grid1 footprint does not overlap Grid2 coords; crop skipped.")


if __name__ == '__main__':
    sys.exit(main())
