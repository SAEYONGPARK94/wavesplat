"""Is range-dependent GT degradation an LJ quirk, or a property of stereo?

probe_lj_range_quality found, in LJ_2018 Grid2 and after removing the static
per-pixel bias:  corr(|mean-surface|, range) = +0.951, growing to 7.6 m, and
excess kurtosis climbing 0.96 -> 8.28 across 15-200 m.

n = 1. Before that becomes a claim about our evaluation domain, it has to
appear in the other sites. Same probe, all 13 records.

A real sea is statistically homogeneous over a few hundred metres. Any
systematic growth of |bias|, std or kurtosis with distance from the camera is
the reconstruction.
"""
import glob
import os
import sys

import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

import h5py
from realdata_analysis import _unpack

NT = 64
NBIN = 6


def moments(v):
    v = v[np.isfinite(v)]
    if v.size < 500:
        return np.nan, np.nan
    s = v.std()
    c = v - v.mean()
    return s, (c ** 4).mean() / s ** 4 - 3.0


def run(path):
    f = h5py.File(path, 'r')
    X = _unpack(f['_nc4_non_coord_X'], np.s_[:])
    Y = _unpack(f['Y'], np.s_[:])
    T = f['Z'].shape[0]
    idx = np.linspace(0, T - 1, NT).astype(int)
    Z = np.stack([_unpack(f['Z'], np.s_[int(i)]) for i in idx])
    f.close()

    valid = np.isfinite(Z).all(0)
    if valid.sum() < 5000:
        return None
    bias = np.where(valid, Z.mean(0), np.nan)
    Zc = Z - bias
    R = np.hypot(X, Y) if abs(Y.min()) > 1 else np.abs(Y)

    lo, hi = np.nanmin(R[valid]), np.nanmax(R[valid])
    edges = np.linspace(lo, hi, NBIN + 1)
    r, ab, sd, ku = [], [], [], []
    med = np.nanmedian(bias[valid])
    for a, b in zip(edges[:-1], edges[1:]):
        m = valid & (R >= a) & (R < b)
        if m.sum() < 500:
            continue
        s, k = moments(Zc[:, m].ravel())
        if not np.isfinite(k):
            continue
        r.append(0.5 * (a + b))
        ab.append(float(np.nanmean(np.abs(bias[m] - med))))
        sd.append(s)
        ku.append(k)
    if len(r) < 4:
        return None
    r = np.array(r)
    c = lambda v: float(np.corrcoef(r, np.array(v))[0, 1])
    return dict(span=hi - lo, extent=float(X.max() - X.min()),
                c_bias=c(ab), c_std=c(sd), c_kurt=c(ku),
                bias_near=ab[0], bias_far=ab[-1],
                kurt_near=ku[0], kurt_far=ku[-1], hs=4 * np.mean(sd))


def main():
    files = sorted(glob.glob('data/real_bergamasco/*/*.nc'))
    print(f"{'record':<34}{'extent':>8}{'4std':>7}"
          f"{'c(bias)':>9}{'c(kurt)':>9}{'bias n->f':>14}{'kurt n->f':>16}")
    rows = []
    for p in files:
        try:
            r = run(p)
        except Exception as e:
            print(f"{os.path.basename(p)[:34]:<34}  skip: {e}")
            continue
        if r is None:
            continue
        rows.append((os.path.basename(p), r))
        print(f"{os.path.basename(p)[9:43]:<34}{r['extent']:7.0f}m"
              f"{r['hs']:7.1f}{r['c_bias']:+9.3f}{r['c_kurt']:+9.3f}"
              f"{r['bias_near']:6.2f}->{r['bias_far']:5.2f} m"
              f"{r['kurt_near']:+8.2f}->{r['kurt_far']:+6.2f}")

    cb = np.array([r['c_bias'] for _, r in rows])
    ck = np.array([r['c_kurt'] for _, r in rows])
    print(f"\n{len(rows)} records")
    print(f"  corr(|bias|,range) > +0.5 :  {(cb > 0.5).sum()}/{len(rows)}"
          f"   median {np.median(cb):+.3f}")
    print(f"  corr(kurt,  range) > +0.5 :  {(ck > 0.5).sum()}/{len(rows)}"
          f"   median {np.median(ck):+.3f}")
    ext = np.array([r['extent'] for _, r in rows])
    far = np.array([r['bias_far'] for _, r in rows])
    if len(rows) > 3:
        print(f"  corr(far |bias|, grid extent) = "
              f"{np.corrcoef(ext, far)[0,1]:+.3f}  "
              f"<- degradation tracks how far out the grid reaches")


if __name__ == '__main__':
    sys.exit(main())
