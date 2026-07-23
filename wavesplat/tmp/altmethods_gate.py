"""
altmethods_gate.py — empirical gate for NON-beamforming wave-characterization
methods, vs the beamforming (matched-filter) baselines from blasso_feasibility.py.

Families tested here (both avoid matched-filter peak-picking):
  - BANK-RIDGE  : dispersion-constrained wavenumber bank + regularized (Tikhonov/MMSE)
                  linear amplitude estimate. == the "covariance/likelihood / bank-MMSE" family.
  - BANK-LASSO  : same bank + L1 sparse recovery (compressed sensing). A 3rd mechanism
                  (neither peak-search nor ridge).
Comparison references (beamforming family, already computed in the BLASSO run):
  5comp_s0: greedy+joint 0.0748, oracle-k 0.0698
  5comp_s1: greedy+joint 0.0787, oracle-k 0.0693
  15comp_s0: greedy+joint 0.1088, oracle-k 0.1474
"""
import sys, numpy as np, time
sys.path.insert(0, 'tmp')
from blasso_feasibility import load_obs, omega_of_k, fit_amp, model_eval, field_nrmse

RNG = np.random.default_rng(0)

# beamforming (greedy+joint) & oracle-k references from the BLASSO feasibility run
BEAM_REF = {
    'episode_BF3_5comp_seed0_uniform.h5':  dict(beam=0.0748, oracle=0.0698),
    'episode_BF3_5comp_seed1_uniform.h5':  dict(beam=0.0787, oracle=0.0693),
    'episode_BF4_15comp_seed0_uniform.h5': dict(beam=0.1088, oracle=0.1474),
}


def build_bank(step, kmax=1.2):
    g = np.arange(-kmax, kmax + 1e-9, step)
    KX, KY = np.meshgrid(g, g, indexing='xy')
    return KX.ravel(), KY.ravel()


def design(kxb, kyb, x, y, t, depth):
    om = omega_of_k(kxb, kyb, depth)
    th = np.outer(x, kxb) + np.outer(y, kyb) - np.outer(t, om)      # (M, B)
    return np.hstack([np.cos(th), -np.sin(th)])                     # (M, 2B)  -> model = Gm @ [a; b]


def ridge_solve(Gm, z, lam_rel=1e-2):
    A = Gm.T @ Gm
    d = lam_rel * np.trace(A) / A.shape[0]
    return np.linalg.solve(A + d * np.eye(A.shape[0]), Gm.T @ z)


def mmse_solve(Gm, z, kxb, kyb, k0=0.30, reg_rel=0.05):
    """Bank-MMSE: prior coeff variance follows a generic swell-favouring spectral
    shape (shrinks high-|k| and DC), so it is a FAIR family-① test, not flat ridge."""
    km = np.hypot(kxb, kyb)
    pvar = 1.0 / (1.0 + (km / k0) ** 4)            # coeff prior variance vs |k|
    pvar = np.concatenate([pvar, pvar])             # a, b blocks
    A = Gm.T @ Gm
    sig2 = reg_rel * np.trace(A) / A.shape[0]
    return np.linalg.solve(A + sig2 * np.diag(1.0 / pvar), Gm.T @ z)


def spec_norm_sq(Gm, iters=25):
    v = RNG.standard_normal(Gm.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(iters):
        w = Gm.T @ (Gm @ v)
        v = w / (np.linalg.norm(w) + 1e-30)
    return float(v @ (Gm.T @ (Gm @ v)))


def fista(Gm, z, lam, iters=400):
    L = spec_norm_sq(Gm) * 1.02
    x = np.zeros(Gm.shape[1]); y = x.copy(); tt = 1.0
    for _ in range(iters):
        grad = Gm.T @ (Gm @ y - z)
        xn = np.sign(y - grad / L) * np.maximum(np.abs(y - grad / L) - lam / L, 0.0)
        tn = (1 + np.sqrt(1 + 4 * tt * tt)) / 2
        y = xn + ((tt - 1) / tn) * (xn - x)
        x = xn; tt = tn
    return x


def bank_field_nrmse(kxb, kyb, coef, obs, frame):
    gx, gy, gtf, Hs, depth = obs['gx'], obs['gy'], obs['gtf'], obs['Hs'], obs['depth']
    tt = obs['ts'][frame]
    X, Y = np.meshgrid(gx, gy, indexing='xy')
    Gm = design(kxb, kyb, X.ravel(), Y.ravel(), np.full(X.size, tt), depth)
    pred = (Gm @ coef).reshape(X.shape)
    return np.sqrt(np.mean((pred - gtf[frame]) ** 2)) / Hs


def weak_comp_recovered(kxb, kyb, coef, wc, thresh_frac=0.3):
    """Is the weakest true comp within |Δk|<.06 of a bank atom carrying real power?"""
    B = len(kxb); w = np.hypot(coef[:B], coef[B:])
    wi = wc[:, 0].argmin()
    kg = np.array([wc[wi, 2], wc[wi, 3]])
    d = np.hypot(kxb - kg[0], kyb - kg[1])
    near = d < 0.06
    if not near.any():
        return False, 0.0
    p_near = w[near].max()
    return (p_near > thresh_frac * wc[wi, 0]), p_near


def run(fn, step=0.06):
    obs = load_obs('data/' + fn)
    M = len(obs['x'])
    sub = RNG.choice(M, size=min(2600, M), replace=False)
    xs, ys, ts, zs = obs['x'][sub], obs['y'][sub], obs['t'][sub], obs['z'][sub]
    kxb, kyb = build_bank(step)
    B = len(kxb)
    Gm = design(kxb, kyb, xs, ys, ts, obs['depth'])
    t0 = time.time()
    c_ridge = ridge_solve(Gm, zs, lam_rel=1e-2)
    c_mmse = mmse_solve(Gm, zs, kxb, kyb)
    lam = 0.06 * np.abs(Gm.T @ zs).max()
    c_lasso = fista(Gm, zs, lam)
    fr = obs['nfr'] // 2
    # average field nRMSE over a few frames (not just one) for robustness
    frames = [obs['nfr'] // 4, obs['nfr'] // 2, 3 * obs['nfr'] // 4]
    nr_ridge = np.mean([bank_field_nrmse(kxb, kyb, c_ridge, obs, f) for f in frames])
    nr_mmse = np.mean([bank_field_nrmse(kxb, kyb, c_mmse, obs, f) for f in frames])
    nr_lasso = np.mean([bank_field_nrmse(kxb, kyb, c_lasso, obs, f) for f in frames])
    wl = np.hypot(c_lasso[:B], c_lasso[B:])
    n_active = int((wl > 0.02 * wl.max()).sum())
    ref = BEAM_REF[fn]
    print(f"\n{fn}  (bank={B} atoms, {len(sub)} obs, {time.time()-t0:.1f}s)")
    print(f"    field nRMSE (avg 3 frames):  BEAM(g+j)={ref['beam']:.4f}   "
          f"BANK-MMSE={nr_mmse:.4f}   BANK-RIDGE={nr_ridge:.4f}   BANK-LASSO={nr_lasso:.4f}   oracle-k={ref['oracle']:.4f}")
    rr = weak_comp_recovered(kxb, kyb, c_mmse, obs['wc'])
    rl = weak_comp_recovered(kxb, kyb, c_lasso, obs['wc'])
    wi = obs['wc'][:, 0].argmin()
    print(f"    weak comp |k|={np.hypot(obs['wc'][wi,2],obs['wc'][wi,3]):.3f} A={obs['wc'][wi,0]:.3f}: "
          f"MMSE {'HIT' if rr[0] else 'miss'}(p={rr[1]:.3f})  LASSO {'HIT' if rl[0] else 'miss'}(p={rl[1]:.3f})  "
          f"[LASSO n_active={n_active}]")


if __name__ == "__main__":
    for fn in BEAM_REF:
        try:
            run(fn)
        except Exception as e:
            import traceback; traceback.print_exc()
