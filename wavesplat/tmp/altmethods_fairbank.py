"""
altmethods_fairbank.py — FAIR arm of the non-beamforming family gate.

Why: the first gate (`altmethods_gate.py`) used a naive FLAT bank (kx,ky in +-1.2,
step .06 -> 1681 atoms = 3362 unknowns vs 2600 obs, UNDERDETERMINED) with a generic
1/(1+(k/k0)^4) prior. The project's OWN bank estimator (`scripts/d3_stream_seqmmse.py`
:make_bank / gamma_beamform, == mmse_bank_estimate, the estimator the companion
encounter-forecast paper actually deploys) uses a *kp-centred annulus* 0.35-2.5 kp with
dk = 0.1 kp and a *beamform-power-shaped* prior. Testing the family with the strawman
bank would be a rigged gate.

This script re-runs the family with that geometry (bank size ~ a few hundred atoms,
overdetermined) and reports against the same BEAM (greedy+joint) and zero-predictor
references on the SAME frames.

Pre-registered read (stated before the numbers):
  - If BANK-MMSE(fair) still sits at/above the zero-predictor level -> the family is
    genuinely unable to reconstruct the FIELD off the observed strip, and the gate is a
    clean negative for "non-beamforming families as a novelty lever".
  - If BANK-MMSE(fair) lands near BEAM -> the earlier 4x gap was a strawman artifact and
    the family is competitive; the gate then says nothing about novelty either way, only
    that the flat-bank probe was invalid.
  - Either way, note that field-nRMSE-at-grid is NOT the metric the companion paper used
    (it used encounter correlation at the ship track), so a loss here does not contradict
    the companion's "bank beat component identification" result.
"""
import sys, numpy as np, time
sys.path.insert(0, 'tmp')
sys.path.insert(0, 'scripts')
from blasso_feasibility import (load_obs, greedy, joint_refine, field_nrmse,
                                build_grid, corr_at)
from altmethods_gate import design, ridge_solve, mmse_solve, fista, bank_field_nrmse

RNG = np.random.default_rng(0)
EPS = [('episode_BF3_5comp_seed0_uniform.h5', 5),
       ('episode_BF3_5comp_seed1_uniform.h5', 5),
       ('episode_BF4_15comp_seed0_uniform.h5', 15)]


def kp_annulus_bank(kp, dk_frac=0.10):
    """Same geometry as scripts/d3_stream_seqmmse.py:make_bank (annulus .35-2.5 kp)."""
    dk = kp * dk_frac
    ax = np.arange(-2.5 * kp, 2.5 * kp + dk / 2, dk)
    KX, KY = np.meshgrid(ax, ax, indexing='xy')
    kk = np.hypot(KX, KY)
    sel = (kk > 0.35 * kp) & (kk < 2.5 * kp)
    return KX[sel].astype(float), KY[sel].astype(float)


def beamform_prior(kxb, kyb, x, y, t, z, depth, pexp=1.0, smult=2.0):
    """Beamform-power-shaped coefficient prior variance (non-EM core of
    gamma_beamform): gamma ~ (|corr|^2)^pexp, smoothed by a floor = smult*mean."""
    c = corr_at(kxb, kyb, x, y, t, z, depth)
    p = np.abs(c) ** 2
    p = p / (p.max() + 1e-30)
    g = p ** pexp
    g = g + smult * g.mean()
    return g / g.mean()


def mmse_solve_gamma(Gm, z, gam, reg_rel=0.05):
    pvar = np.concatenate([gam, gam])
    A = Gm.T @ Gm
    sig2 = reg_rel * np.trace(A) / A.shape[0]
    return np.linalg.solve(A + sig2 * np.diag(1.0 / pvar), Gm.T @ z)


def run(fn, n_true, nsub=2600):
    obs = load_obs('data/' + fn)
    frames = [obs['nfr'] // 4, obs['nfr'] // 2, 3 * obs['nfr'] // 4]
    M = len(obs['x'])
    sub = RNG.choice(M, size=min(nsub, M), replace=False)
    xs, ys, ts, zs = obs['x'][sub], obs['y'][sub], obs['t'][sub], obs['z'][sub]

    # kp from the true components (energy-weighted mean |k|) -- generous to the family
    wc = obs['wc']
    km = np.hypot(wc[:, 2], wc[:, 3])
    kp = float(np.sum(wc[:, 0] ** 2 * km) / np.sum(wc[:, 0] ** 2))

    kxb, kyb = kp_annulus_bank(kp)
    B = len(kxb)
    Gm = design(kxb, kyb, xs, ys, ts, obs['depth'])
    gam = beamform_prior(kxb, kyb, xs, ys, ts, zs, obs['depth'])

    c_bf = mmse_solve_gamma(Gm, zs, gam)
    c_ridge = ridge_solve(Gm, zs, lam_rel=1e-2)
    lam = 0.06 * np.abs(Gm.T @ zs).max()
    c_lasso = fista(Gm, zs, lam)

    zvar = float(np.mean(zs ** 2))
    fit = {k: float(np.mean((Gm @ c - zs) ** 2) / zvar)
           for k, c in [('bfmmse', c_bf), ('ridge', c_ridge), ('lasso', c_lasso)]}

    gridk = build_grid()
    lam_b = 0.03 * np.abs(corr_at(*gridk, xs, ys, ts, zs, obs['depth']).max())
    gj = joint_refine(greedy(obs, n_true + 1, gridk, sub), obs, sub, lam_b)
    beam = np.mean([field_nrmse(gj, obs, f) for f in frames])
    z0 = np.mean([float(np.sqrt(np.mean(obs['gtf'][f] ** 2)) / obs['Hs']) for f in frames])

    nr = {k: np.mean([bank_field_nrmse(kxb, kyb, c, obs, f) for f in frames])
          for k, c in [('bfmmse', c_bf), ('ridge', c_ridge), ('lasso', c_lasso)]}

    print(f"\n{fn}   kp={kp:.3f}  bank={B} atoms (2B={2*B} unknowns vs {len(sub)} obs)"
          f"  -> {'UNDERDET' if len(sub) < 2*B else 'overdetermined'}")
    print(f"   field nRMSE (avg 3 frames):  ZERO-PRED={z0:.4f}   BEAM(g+j)={beam:.4f}   "
          f"BANK-MMSE(beamform prior)={nr['bfmmse']:.4f}   BANK-RIDGE={nr['ridge']:.4f}   "
          f"BANK-LASSO={nr['lasso']:.4f}")
    print(f"   in-sample obs residual/var:  BF-MMSE={fit['bfmmse']:.4f}  RIDGE={fit['ridge']:.4f}"
          f"  LASSO={fit['lasso']:.4f}")


if __name__ == "__main__":
    for fn, nt in EPS:
        t0 = time.time()
        try:
            run(fn, nt)
        except Exception:
            import traceback; traceback.print_exc()
        print(f"   ({time.time()-t0:.1f}s)")
