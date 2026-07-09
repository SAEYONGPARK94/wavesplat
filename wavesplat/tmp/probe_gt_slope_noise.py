"""Same white-noise discriminator, applied to the GT surface SLOPE.

var(slope, h) = var(slope_true) + sigma_s^2 / (2 h^2)

sigma_s is the effective elevation noise seen ACROSS SPACE. It need not equal
the temporal sigma: stereo reconstruction error is correlated between adjacent
pixels but not between frames, so spatial differencing can be far quieter than
temporal differencing. That is exactly what this measures.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

import d3_required_horizon as RH

HX = [1.0, 2.0, 4.0, 8.0, 12.0]


def main():
    for rec in ['YS17', 'LJ18g']:
        print(f"=== {rec} ===")
        var_s = []
        for hx in HX:
            RH.H_X, RH.H_T = hx, 0.1
            vs = []
            for win in range(3):
                sc = RH.build_real(rec, win)
                _, ref = RH.gt_series(sc, [RH.LEADS, RH.REF_LEADS])
                vs.append(np.nanvar(ref[:, 1]))
            var_s.append(np.mean(vs))
            print(f"  h_x={hx:4.1f} m  std(slope)={np.sqrt(var_s[-1]):.5f}")

        x = 1.0 / (2.0 * np.array(HX) ** 2)
        y = np.array(var_s)
        A = np.vstack([x, np.ones_like(x)]).T
        sig2, v_true = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = y - A @ np.array([sig2, v_true])
        r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
        sg = np.sqrt(max(sig2, 0))
        st = np.sqrt(max(v_true, 0))
        print(f"  sigma_spatial = {sg:.4f} m,  std(slope_true) = {st:.5f}")
        print(f"  R^2 = {r2:.4f}")
        for hx in HX:
            n = sg / (np.sqrt(2) * hx)
            print(f"    h_x={hx:4.1f}: noise {n:.5f} vs signal {st:.5f} "
                  f"-> SNR {st/max(n,1e-9):.2f}")
        print()
    RH.H_X, RH.H_T = 2.0, 0.1


if __name__ == '__main__':
    sys.exit(main())
