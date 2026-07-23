"""
altmethods_diag.py — sanity/comparability diagnostics for the altmethods gate.

The first run reported BANK-* field nRMSE ~0.30 vs BEAM 0.075-0.11. 0.25-0.30 is
suspiciously close to the ZERO-PREDICTOR level (a documented past metric artifact in
this repo), so before reporting "the bank family loses" we check:
  (1) zero-predictor nRMSE on the SAME frames  -> is 0.30 just "predicting nothing"?
  (2) BEAM (greedy+joint) recomputed on the SAME 3 frames -> apples-to-apples
  (3) in-sample observation residual for each bank solution -> does the bank FIT the
      data (and merely fail to extrapolate), or does it not even fit?
  (4) bank conditioning: n_obs vs n_unknowns (2*B).
"""
import sys, numpy as np, time
sys.path.insert(0, 'tmp')
from blasso_feasibility import (load_obs, omega_of_k, greedy, joint_refine, field_nrmse,
                                build_grid, corr_at)
from altmethods_gate import build_bank, design, ridge_solve, mmse_solve, fista, bank_field_nrmse

RNG = np.random.default_rng(0)
EPS = [('episode_BF3_5comp_seed0_uniform.h5', 5),
       ('episode_BF3_5comp_seed1_uniform.h5', 5),
       ('episode_BF4_15comp_seed0_uniform.h5', 15)]


def zero_pred_nrmse(obs, frame):
    return float(np.sqrt(np.mean(obs['gtf'][frame] ** 2)) / obs['Hs'])


def run(fn, n_true, step=0.06, nsub=2600):
    obs = load_obs('data/' + fn)
    frames = [obs['nfr'] // 4, obs['nfr'] // 2, 3 * obs['nfr'] // 4]
    M = len(obs['x'])
    sub = RNG.choice(M, size=min(nsub, M), replace=False)
    xs, ys, ts, zs = obs['x'][sub], obs['y'][sub], obs['t'][sub], obs['z'][sub]

    kxb, kyb = build_bank(step)
    B = len(kxb)
    Gm = design(kxb, kyb, xs, ys, ts, obs['depth'])
    c_ridge = ridge_solve(Gm, zs, lam_rel=1e-2)
    c_mmse = mmse_solve(Gm, zs, kxb, kyb)
    lam = 0.06 * np.abs(Gm.T @ zs).max()
    c_lasso = fista(Gm, zs, lam)

    zvar = float(np.mean(zs ** 2))
    res = {k: float(np.mean((Gm @ c - zs) ** 2) / zvar)
           for k, c in [('mmse', c_mmse), ('ridge', c_ridge), ('lasso', c_lasso)]}

    # BEAM on the SAME frames (same recipe as blasso_feasibility.run_episode)
    gridk = build_grid()
    lam = 0.03 * np.abs(corr_at(*gridk, xs, ys, ts, zs, obs['depth']).max())
    gj = joint_refine(greedy(obs, n_true + 1, gridk, sub), obs, sub, lam)
    beam_f = [field_nrmse(gj, obs, f) for f in frames]

    z0 = [zero_pred_nrmse(obs, f) for f in frames]
    print(f"\n{fn}")
    print(f"  n_obs={len(sub)}  n_unknowns=2B={2*B}  -> {'UNDERDETERMINED' if len(sub) < 2*B else 'ok'}")
    print(f"  ZERO-PREDICTOR nRMSE (same 3 frames): {np.mean(z0):.4f}   {[f'{v:.3f}' for v in z0]}")
    print(f"  BEAM greedy+joint  (same 3 frames)  : {np.mean(beam_f):.4f}   {[f'{v:.3f}' for v in beam_f]}")
    print(f"  BANK field nRMSE  MMSE={np.mean([bank_field_nrmse(kxb,kyb,c_mmse,obs,f) for f in frames]):.4f}"
          f"  RIDGE={np.mean([bank_field_nrmse(kxb,kyb,c_ridge,obs,f) for f in frames]):.4f}"
          f"  LASSO={np.mean([bank_field_nrmse(kxb,kyb,c_lasso,obs,f) for f in frames]):.4f}")
    print(f"  in-sample obs residual / var:  MMSE={res['mmse']:.4f}  RIDGE={res['ridge']:.4f}  LASSO={res['lasso']:.4f}")


if __name__ == "__main__":
    for fn, nt in EPS:
        t0 = time.time()
        run(fn, nt)
        print(f"  ({time.time()-t0:.1f}s)")
