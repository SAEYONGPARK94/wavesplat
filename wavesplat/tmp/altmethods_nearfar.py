"""
altmethods_nearfar.py — disambiguate WHY the bank family loses on wide-grid field nRMSE.

Hypothesis: the wide-grid metric structurally rewards *global parametric extrapolation*.
The component (beamforming) model is a sum of plane waves valid everywhere; the bank-MMSE
is a shrinkage estimator that (correctly, by design) shrinks toward zero where it has no
information — i.e. off the observed strip. If so, the two should be much closer NEAR the
observations and diverge FAR from them, and "BANK loses 3x" would be a statement about the
metric's extrapolation demand, not about the family's estimation quality.

Test: split grid points by distance to the nearest observation (in the frame's local
space), and report field nRMSE within each distance band for BEAM vs BANK vs zero-pred.
Pre-registered read:
  - BANK ~ BEAM in the near band, BANK -> zero-pred in the far band  => confirms the
    extrapolation explanation; the earlier verdict must be scoped to "off-strip field
    reconstruction", not to the family.
  - BANK loses even in the near band  => the family really is worse at the estimation task
    itself, and the verdict stands unqualified.
"""
import sys, numpy as np, time
sys.path.insert(0, 'tmp')
sys.path.insert(0, 'scripts')
from blasso_feasibility import (load_obs, greedy, joint_refine, build_grid, corr_at,
                                model_eval)
from d3_sa_nomp import mmse_bank_estimate

RNG = np.random.default_rng(0)
BANDS = [(0, 20), (20, 50), (50, 100), (100, 1e9)]   # metres to nearest observation


def run(fn, n_true, nsub=2600):
    obs = load_obs('data/' + fn)
    f = obs['nfr'] // 2
    tf = obs['ts'][f]
    M = len(obs['x'])
    sub = RNG.choice(M, size=min(nsub, M), replace=False)
    xs, ys, ts, zs = obs['x'][sub], obs['y'][sub], obs['t'][sub], obs['z'][sub]

    gx, gy = obs['gx'], obs['gy']
    X, Y = np.meshgrid(gx, gy, indexing='xy')
    xq, yq = X.ravel(), Y.ravel()
    truth = obs['gtf'][f].ravel()

    # distance from each grid point to the nearest observation IN THE SAME FRAME window
    # (obs within +-1 frame interval of tf; falls back to all obs if too few)
    dt = np.abs(ts - tf)
    win = dt < max(np.median(np.diff(np.unique(ts))) * 1.5, 1e-9)
    if win.sum() < 50:
        win = dt < np.quantile(dt, 0.02)
    ox, oy = xs[win], ys[win]
    d = np.sqrt(np.min((xq[:, None] - ox[None]) ** 2 + (yq[:, None] - oy[None]) ** 2, axis=1))

    # BEAM
    gridk = build_grid()
    lam_b = 0.03 * np.abs(corr_at(*gridk, xs, ys, ts, zs, obs['depth']).max())
    gj = joint_refine(greedy(obs, n_true + 1, gridk, sub), obs, sub, lam_b)
    pred_beam = model_eval(gj, xq, yq, np.full(xq.size, tf), obs['depth'])

    # BANK (project's own tuned estimator)
    P = np.column_stack([xs, ys, ts])
    Pq = np.column_stack([xq, yq, np.full(xq.size, tf)])
    pred_bank = mmse_bank_estimate(P, zs, obs['depth'], Pq, 0.05 * obs['Hs'])

    print(f"\n{fn}  frame={f}  n_obs_in_frame={win.sum()}  grid={xq.size} pts")
    print(f"  {'band [m]':>14} {'n':>7} {'ZERO':>8} {'BEAM':>8} {'BANK':>8}")
    for lo, hi in BANDS:
        m = (d >= lo) & (d < hi)
        if m.sum() == 0:
            continue
        z0 = np.sqrt(np.mean(truth[m] ** 2)) / obs['Hs']
        nb = np.sqrt(np.mean((pred_beam[m] - truth[m]) ** 2)) / obs['Hs']
        nk = np.sqrt(np.mean((pred_bank[m] - truth[m]) ** 2)) / obs['Hs']
        lbl = f"{lo}-{'inf' if hi > 1e8 else int(hi)}"
        print(f"  {lbl:>14} {m.sum():>7} {z0:>8.4f} {nb:>8.4f} {nk:>8.4f}")
    # energy retained by each predictor (shrinkage diagnostic)
    print(f"  pred RMS / truth RMS:  BEAM={np.sqrt(np.mean(pred_beam**2))/np.sqrt(np.mean(truth**2)):.3f}"
          f"   BANK={np.sqrt(np.mean(pred_bank**2))/np.sqrt(np.mean(truth**2)):.3f}")


if __name__ == "__main__":
    for fn, nt in [('episode_BF3_5comp_seed0_uniform.h5', 5),
                   ('episode_BF4_15comp_seed0_uniform.h5', 15)]:
        t0 = time.time()
        try:
            run(fn, nt)
        except Exception:
            import traceback; traceback.print_exc()
        print(f"  ({time.time()-t0:.1f}s)")
