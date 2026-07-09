"""Is the GT vertical velocity noise-dominated? Separate signal from noise by
how std(w) scales with the differencing interval h.

Central difference of a signal + white measurement noise (std sigma):

    var(w_hat, h) = var(w_true) + (sigma^2 / (2 h^2))          [white noise]

so a plot of var vs 1/h^2 is a straight line whose slope gives sigma and whose
intercept gives the true velocity variance. A smooth signal gives slope ~ 0.
Genuine unresolved short waves would instead saturate, not grow like 1/h^2 —
that is the discriminator, not an assumption.

Runs on GT only, over the long reference lead grid.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

import d3_required_horizon as RH

HS = [0.1, 0.2, 0.3, 0.4, 0.5]


def main():
    for rec in ['YS17', 'LJ18g']:
        print(f"=== {rec} ===")
        var_w, var_a = [], []
        for ht in HS:
            RH.H_X, RH.H_T = 2.0, ht
            vs, as_ = [], []
            for win in range(3):
                sc = RH.build_real(rec, win)
                _, ref = RH.gt_series(sc, [RH.LEADS, RH.REF_LEADS])
                vs.append(np.nanvar(ref[:, 2]))
                as_.append(np.nanvar(ref[:, 3]))
            var_w.append(np.mean(vs))
            var_a.append(np.mean(as_))
            print(f"  h_t={ht:.1f}s  var(w)={var_w[-1]:.4f}  "
                  f"std(w)={np.sqrt(var_w[-1]):.3f}   "
                  f"std(alpha)={np.sqrt(var_a[-1]):.3f}")

        x = 1.0 / (2.0 * np.array(HS) ** 2)          # coefficient of sigma^2
        y = np.array(var_w)
        A = np.vstack([x, np.ones_like(x)]).T
        sig2, v_true = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = y - A @ np.array([sig2, v_true])
        r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"  fit var(w) = sigma^2/(2h^2) + var_true :")
        print(f"    sigma  (frame elevation noise) = {np.sqrt(max(sig2,0)):.4f} m")
        print(f"    std(w_true)                    = {np.sqrt(max(v_true,0)):.4f} m/s")
        print(f"    R^2 = {r2:.4f}   (high R^2 => white-noise law holds)")
        for ht in HS:
            n = np.sqrt(max(sig2, 0)) / (np.sqrt(2) * ht)
            s = np.sqrt(max(v_true, 0))
            print(f"    h_t={ht:.1f}: noise std {n:.3f} vs signal {s:.3f} "
                  f"-> SNR {s/max(n,1e-9):.2f}")
        print()
    RH.H_X, RH.H_T = 2.0, 0.1


if __name__ == '__main__':
    sys.exit(main())
