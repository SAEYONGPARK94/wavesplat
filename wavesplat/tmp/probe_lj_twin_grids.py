"""The same sea, at the same instant, reconstructed onto two grids.

LJ_2018 holds Surfaces_20180103_093938_Grid1 and _Grid2: one scene, one
timestamp, two stereo reconstruction grids. Whatever the sea's excess kurtosis
was at 09:39:38, it was ONE number.

If the two grids disagree, the disagreement is manufactured by the
reconstruction. That is a controlled experiment the ocean handed us for free,
and it needs no ground truth.

This bears directly on the unexplained LJ18g oracle pathology (+0.235 nRMSE),
where aperture, bimodality, white GT noise, storm physics and band placement
were all excluded.
"""
import os
import sys

import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

from d3_sa_multirecord import MRecord, _unpack

FILES = ['data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid1_short.nc',
         'data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid2_short.nc']
NT = 96


def stats(f):
    rec = MRecord(f, stride=1, robust=False)
    idx = np.linspace(0, rec.T_all - 1, NT).astype(int)
    fr = np.stack([_unpack(rec.zd, np.s_[int(i)]) for i in idx])
    fr[:, ~rec.valid] = np.nan
    bias = np.nanmean(fr, axis=0)
    z = fr - bias                                  # same calibration as SSL
    v = z[np.isfinite(z)]
    s = v.std()
    k = ((v - v.mean()) ** 4).mean() / s ** 4 - 3.0
    sk = ((v - v.mean()) ** 3).mean() / s ** 3
    tail = float((np.abs(v - v.mean()) > 4 * s).mean() * 100)
    return dict(name=os.path.basename(f), shape=rec.valid.shape,
                T=rec.T_all, valid=float(rec.valid.mean() * 100),
                std=float(s), skew=float(sk), kurt=float(k), tail=tail,
                bias_std=float(np.nanstd(bias)))


def main():
    rows = [stats(f) for f in FILES]
    for r in rows:
        print(f"{r['name']}")
        print(f"   grid {r['shape']}  frames {r['T']}  valid {r['valid']:.1f}%")
        print(f"   surface std   {r['std']:.3f} m     static bias std "
              f"{r['bias_std']:.3f} m")
        print(f"   skew {r['skew']:+.3f}   excess kurt {r['kurt']:+.3f}   "
              f">4sig {r['tail']:.3f}%\n")

    dk = abs(rows[0]['kurt'] - rows[1]['kurt'])
    print(f"same scene, same instant:  |dkurt| = {dk:.3f}")
    print("second-order Stokes on a real sea moves excess kurtosis by ~0.2.")
    if dk > 1.0:
        print(f"\n[!] The two reconstructions of ONE sea disagree by {dk:.2f} in "
              f"excess kurtosis.\n    Record-level kurtosis here is a property of "
              f"the reconstruction, not the sea.\n    Non-Gaussianity measured on "
              f"LJ_2018 cannot be attributed to wave physics.")


if __name__ == '__main__':
    sys.exit(main())
