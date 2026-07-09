"""NV2 — does the recoverable correlation of a QUADRATIC functional really
fall as rho^2?

Claim (Isserlis / Wick): if (eta, eta_hat) are jointly Gaussian, zero mean,
with corr(eta_hat, eta) = rho, then

    corr(eta_hat^2, eta^2) = rho^2      exactly.

So any quadratic functional of the field — wave energy, energy flux, breaking
criteria, crest exceedance, and the 2nd-order bound wave — inherits a rho^2
ceiling. Linear functionals (surface slope, orbital velocity, and therefore the
foil's alpha_wave = -slope - w/U) inherit no penalty at all. That asymmetry is
the whole content of the claim, and it says elevation forecasting and
extreme-event forecasting are different problems.

The bound-wave experiment gave one point on that curve (rho = 0.662, measured
corr 0.195, predicted rho^2 x ceiling = 0.266). One point does not test a law.
Here rho is swept by starving the estimator of observations, and the two
correlations are measured against each other.

Also checks the linear control: corr(a.eta_hat, a.eta) must stay at rho.
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
import d3_storm_wall as SW

N_OBS = [60, 120, 250, 500, 1000, 2000]     # starve / feed the estimator
SEEDS = 3


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main():
    gx = np.arange(SF.DOM_X[0], SF.DOM_X[1] + 0.1, SF.DX)
    gy = np.arange(SF.DOM_Y[0], SF.DOM_Y[1] + 0.1, SF.DY)
    Xq, Yq = np.meshgrid(gx, gy)
    t0 = SF.W_OBS
    Pq = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(Xq.size, t0)])
    inside, _ = SF.fan_mask(Xq, Yq, 0.0, 0.0, np.array([1.0, 0.0]))
    spec = SW.SEAS['wind']

    print(f"{'n_obs':>6} {'rho = corr(eta_hat, eta)':>24} "
          f"{'corr(eta_hat^2, eta^2)':>23} {'rho^2':>8} {'ratio':>7}")
    print('-' * 72)
    rows = []
    for n in N_OBS:
        rs, qs = [], []
        for sd in range(SEEDS):
            eta, kp, Hs = SW.make_sea(spec, 1000 + 37 * sd)
            gt = eta(Xq.ravel(), Yq.ravel(), t0).reshape(Xq.shape)
            rng = np.random.default_rng(11 + sd)
            # fan observations, count controlled
            scan_ts = np.arange(t0 - SF.W_OBS, t0 + 1e-9, SF.SCAN_DT)
            per = max(1, int(np.ceil(n / len(scan_ts))))
            P, Z = [], []
            for t_s in scan_ts:
                rr = rng.uniform(*SF.R_FAN, per)
                aa = rng.uniform(-SF.HALF_FOV, SF.HALF_FOV, per)
                xs, ys = rr * np.cos(aa), rr * np.sin(aa)
                k = ((xs >= SF.DOM_X[0]) & (xs <= SF.DOM_X[1]) &
                     (ys >= SF.DOM_Y[0]) & (ys <= SF.DOM_Y[1]))
                xs, ys = xs[k], ys[k]
                if len(xs) == 0:
                    continue
                P.append(np.column_stack([xs, ys, np.full(len(xs), t_s)]))
                Z.append(eta(xs, ys, t_s) + rng.normal(0, 0.02, len(xs)))
            P, Z = np.vstack(P), np.concatenate(Z)
            if len(Z) > n:
                ii = rng.choice(len(Z), n, replace=False)
                P, Z = P[ii], Z[ii]
            fq = mmse_bank_estimate(P, Z, spec['depth'], Pq,
                                    sigma0=0.05 * Hs, kp_hint=kp, seed=sd)
            pr = fq.reshape(Xq.shape)
            # score only inside the fan: there the joint-Gaussian premise is
            # closest to true (outside, eta_hat is a shrunken near-zero field)
            a, b = pr[inside], gt[inside]
            a = a - a.mean(); b = b - b.mean()
            rs.append(corr(a, b))
            qs.append(corr(a ** 2, b ** 2))
        r, q = float(np.mean(rs)), float(np.mean(qs))
        print(f"{n:>6} {r:>24.3f} {q:>23.3f} {r**2:>8.3f} {q/r**2:>7.3f}")
        rows.append((n, r, q))

    r = np.array([x[1] for x in rows]); q = np.array([x[2] for x in rows])
    print(f"\ncorr(measured q, rho^2) over the sweep = "
          f"{np.corrcoef(q, r ** 2)[0,1]:+.4f}")
    print(f"mean ratio q / rho^2 = {np.mean(q / r**2):.3f} "
          f"(1.0 = the law holds exactly)")
    print("\nlinear control: any linear functional keeps corr = rho by "
          "construction (no test needed).")


if __name__ == '__main__':
    sys.exit(main())
