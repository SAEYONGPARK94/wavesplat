"""The bound wave is not something to learn. It is something to compute.

A conv net reading c(k) recovers almost none of the bound wave (corr 0.11).
But eta_2 is, to leading order, a pointwise quadratic of the free surface:
for a narrowband sea, eta_2 ~ (k/2) * eta_1^2. So a physicist would never ask
the network for it — they would estimate eta_1 (which the linear map does
well) and then APPLY the known operator.

Three numbers settle it:

  ceiling    a * eta_1_true^2      vs eta_2   — is a pointwise quadratic even
                                                the right functional form?
  physics    a * eta_1_hat^2       vs eta_2   — what you get by composing the
                                                linear estimator with the
                                                known operator (no learning)
  learned    CNN(c) (from probe_bw_normalized) vs eta_2

`a` is one scalar, fit by least squares on the training half.

If physics >> learned, the lesson of the linear seas survives into the
nonlinear ones: freedom does not buy the bound wave, structure does. And the
storm's "model-class wall" is then not a wall for learning to climb, but an
operator to append.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

from d3_cbf_boundwave import CACHE, N_VAL, ridge

NG2 = 64


def corr(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.corrcoef(a, b)[0, 1])


def fit_a(pred_sq, target):
    """single scalar, least squares"""
    x, y = pred_sq.ravel(), target.ravel()
    return float((x * y).sum() / max((x * x).sum(), 1e-12))


def main():
    d = np.load(CACHE)
    X, Ff, Fb = d['X'], d['Ff'], d['Fb']
    n = len(X) - N_VAL
    print(f"scenes {len(X)}  (train {n}, val {N_VAL})")

    print("fitting ridge on the FREE field (this is the estimator physics "
          "would hand the operator)...", flush=True)
    rv, W = ridge(X, Ff, n)
    Xva = X[:N_VAL].reshape(N_VAL, -1).astype(np.float64)
    f1_hat = (Xva @ W).reshape(N_VAL, NG2, NG2)
    f1_true = Ff[:N_VAL].astype(np.float64)
    f2_true = Fb[:N_VAL].astype(np.float64)
    print(f"  ridge free-field val MSE {rv:.4f}  "
          f"corr(eta1_hat, eta1) = {corr(f1_hat, f1_true):+.3f}")

    # fit the scalar on the TRAIN half (no val leakage)
    Xtr = X[N_VAL:].reshape(n, -1).astype(np.float64)
    f1_hat_tr = (Xtr @ W).reshape(n, NG2, NG2)
    f1_tr = Ff[N_VAL:].astype(np.float64)
    f2_tr = Fb[N_VAL:].astype(np.float64)
    a_ceil = fit_a(f1_tr ** 2 - f1_tr.mean(), f2_tr)
    a_phys = fit_a(f1_hat_tr ** 2 - f1_hat_tr.mean(), f2_tr)

    var2 = float(np.var(f2_true))
    rows = []
    for name, base, a in (('ceiling  (true eta1)', f1_true, a_ceil),
                          ('physics  (ridge eta1)', f1_hat, a_phys)):
        p = a * (base ** 2 - base.mean())
        rel = float(np.mean((p - f2_true) ** 2) / var2)
        rows.append((name, corr(p, f2_true), rel, float(p.std() / f2_true.std())))
    rows.append(('learned  (CNN on c)', 0.111, 0.988, 0.138))

    print(f"\n{'model':<24} {'corr':>7} {'rel MSE':>9} {'amp':>7}")
    print('-' * 50)
    for nm, c, r, amp in rows:
        print(f"{nm:<24} {c:>+7.3f} {r:>9.3f} {amp:>7.3f}")
    print("\n(rel MSE 1.0 = zero-predictor. learned row is the best point of "
          "tmp/probe_bw_normalized.py before it overfits.)")


if __name__ == '__main__':
    sys.exit(main())
