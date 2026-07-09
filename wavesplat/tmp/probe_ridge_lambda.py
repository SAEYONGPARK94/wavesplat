"""The ridge picked lambda=1e-4, the smallest value in its grid.

A regularization path that ends at its own boundary has not been minimized —
the true optimum may lie below. That matters here: the linear map is the
BASELINE the CNN must beat, and an under-optimized baseline manufactures a
nonlinear "gain" out of nothing.

Sweep lambda far below the previous grid on the same 40k cache and find the
interior minimum.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

CACHE = 'tmp/cbf_cache_40k.npz'
N_VAL = 1000


def main():
    d = np.load(CACHE)
    X, F = d['X'], d['F']
    Xf = X.reshape(len(X), -1)
    Ff = F.reshape(len(F), -1)
    Xtr, Ftr = Xf[N_VAL:], Ff[N_VAL:]
    Xva = Xf[:N_VAL].astype(np.float64)
    Fva = Ff[:N_VAL].astype(np.float64)
    print(f"train {len(Xtr)}  val {N_VAL}  features {Xtr.shape[1]}")

    p = Xtr.shape[1]
    A = np.zeros((p, p)); B = np.zeros((p, Ftr.shape[1]))
    for i in range(0, len(Xtr), 4000):
        xb = Xtr[i:i + 4000].astype(np.float64)
        A += xb.T @ xb
        B += xb.T @ Ftr[i:i + 4000].astype(np.float64)
    tr = np.trace(A) / p
    print(f"gram done, trace/p = {tr:.4e}")

    best = (np.inf, None)
    for lam in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        try:
            W = np.linalg.solve(A + lam * tr * np.eye(p), B)
        except np.linalg.LinAlgError:
            print(f"  lam={lam:.0e}  singular")
            continue
        vl = float(np.mean((Xva @ W - Fva) ** 2))
        amp = float((Xva @ W).std() / Fva.std())
        mark = ''
        if vl < best[0]:
            best = (vl, lam); mark = '  <-- best so far'
        print(f"  lam={lam:.0e}  val {vl:.4f}  amp {amp:.3f}{mark}")
    print(f"\ninterior optimum: lam={best[1]:.0e}, val {best[0]:.4f}")
    print("compare: cnn best val 0.5561, resid best val 0.5544")


if __name__ == '__main__':
    sys.exit(main())
