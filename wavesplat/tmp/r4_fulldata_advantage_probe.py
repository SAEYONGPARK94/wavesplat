"""
r4_fulldata_advantage_probe.py — red-team attack #1b: is BEAM's per-atom refine/fit_amp
actually restricted to the SAME 2600-point subsample as BANK, as ALTMETHODS_GATE_RESULT
claims ("All arms use the same 2600-point observation subsample")?

blasso_feasibility.greedy() does:
    x, y, t, z = obs['x'], obs['y'], obs['t'], obs['z']      # FULL episode (all frames)
    xs, ys, ts = x[sub_idx], y[sub_idx], t[sub_idx]          # subsample, x/y/t only
    ...
    cm = corr_at(KXf, KYf, xs, ys, ts, rs, depth)            # peak SEARCH: subsample only
    kx, ky = refine_single(KXf[j], KYf[j], x, y, t, r, depth)  # refine: FULL data
    a, b   = fit_amp(kx, ky, x, y, t, r, depth)                # amplitude: FULL data

So the coarse peak search is subsample-restricted but the off-grid refine + amplitude fit
(the part that actually sets the final atom parameters) uses ALL ~15k points spanning all
301 frames of the episode -- ~5.75x more data than BANK ever sees, and temporal span far
beyond BANK's fixed subsample. This script builds a "fair" greedy that restricts refine
and fit_amp to the SAME sub_idx throughout, and compares near/far-band field nRMSE against
the as-published (full-data) greedy, on the same frame/bands as ALTMETHODS_GATE Round 4.
"""
import sys, time, numpy as np
sys.path.insert(0, 'tmp')
from blasso_feasibility import (load_obs, build_grid, corr_at, refine_single, fit_amp,
                                 model_eval)

RNG = np.random.default_rng(0)


def greedy_fair(obs, n_atoms, gridk, sub_idx):
    """Same as blasso_feasibility.greedy but refine_single/fit_amp restricted to sub_idx
    (true 'same inputs as BANK' condition)."""
    x, y, t, z, depth = obs['x'], obs['y'], obs['t'], obs['z'], obs['depth']
    xs, ys, ts, zs = x[sub_idx], y[sub_idx], t[sub_idx], z[sub_idx]
    KXf, KYf = gridk
    r = zs.copy()
    atoms = []
    for _ in range(n_atoms):
        cm = corr_at(KXf, KYf, xs, ys, ts, r, depth)
        j = cm.argmax()
        kx, ky = refine_single(KXf[j], KYf[j], xs, ys, ts, r, depth)
        a, b = fit_amp(kx, ky, xs, ys, ts, r, depth)
        atoms.append((kx, ky, a, b))
        r = r - model_eval([(kx, ky, a, b)], xs, ys, ts, depth)
    return atoms


def joint_refine_sub(atoms, obs, sub_idx, lam):
    from blasso_feasibility import joint_refine
    return joint_refine(atoms, obs, sub_idx, lam)   # already subsample-restricted


def nrmse(pred, truth, m, Hs):
    return float(np.sqrt(np.mean((pred[m] - truth[m]) ** 2)) / Hs)


def run(fn, n_true):
    print("\n" + "=" * 78)
    print(fn)
    obs = load_obs('data/' + fn)
    M = len(obs['x'])
    sub = RNG.choice(M, size=min(2600, M), replace=False)
    xs, ys, ts, zs = obs['x'][sub], obs['y'][sub], obs['t'][sub], obs['z'][sub]
    print(f"  full episode points M={M}   subsample n={len(sub)}   "
          f"ratio={M/len(sub):.2f}x   frames spanned by subsample vs full "
          f"={obs['nfr']} (both use all {obs['nfr']} frames' worth of TIME span; "
          f"full has {M/obs['nfr']:.0f} pts/frame avg, sub has {len(sub)/obs['nfr']:.1f})")

    gridk = build_grid()
    lam_b = 0.03 * np.abs(corr_at(*gridk, xs, ys, ts, zs, obs['depth']).max())

    from blasso_feasibility import greedy as greedy_fulldata, joint_refine

    f = obs['nfr'] // 2
    tf = obs['ts'][f]
    truth = obs['gtf'][f].ravel()
    gx, gy = obs['gx'], obs['gy']
    X, Y = np.meshgrid(gx, gy, indexing='xy')
    xq, yq = X.ravel(), Y.ravel()

    # distance to nearest SAME-window observation (reuse aswritten logic for banding)
    dt = np.abs(ts - tf)
    thresh = max(np.median(np.diff(np.unique(ts))) * 1.5, 1e-9)
    win = dt < thresh
    if win.sum() < 50:
        win = dt < np.quantile(dt, 0.02)
    ox, oy = xs[win], ys[win]
    d = np.sqrt(np.min((xq[:, None] - ox[None]) ** 2 + (yq[:, None] - oy[None]) ** 2, axis=1))
    near = d < 20
    far = d >= 100

    t0 = time.time()
    g_full = greedy_fulldata(obs, n_true + 1, gridk, sub)
    gj_full = joint_refine(g_full, obs, sub, lam_b)
    pred_full = model_eval(gj_full, xq, yq, np.full(xq.size, tf), obs['depth'])
    t_full = time.time() - t0

    t0 = time.time()
    g_fair = greedy_fair(obs, n_true + 1, gridk, sub)
    gj_fair = joint_refine(g_fair, obs, sub, lam_b)
    pred_fair = model_eval(gj_fair, xq, yq, np.full(xq.size, tf), obs['depth'])
    t_fair = time.time() - t0

    print(f"  BEAM as-published (refine/fit_amp on FULL {M} pts):   "
          f"near0-20={nrmse(pred_full,truth,near,obs['Hs']):.4f}  "
          f"far>100={nrmse(pred_full,truth,far,obs['Hs']):.4f}  "
          f"full-grid={nrmse(pred_full,truth,np.ones_like(near,bool),obs['Hs']):.4f}  [{t_full:.1f}s]")
    print(f"  BEAM 'fair' (refine/fit_amp on SAME {len(sub)}-pt subsample as BANK): "
          f"near0-20={nrmse(pred_fair,truth,near,obs['Hs']):.4f}  "
          f"far>100={nrmse(pred_fair,truth,far,obs['Hs']):.4f}  "
          f"full-grid={nrmse(pred_fair,truth,np.ones_like(near,bool),obs['Hs']):.4f}  [{t_fair:.1f}s]")

    # compare recovered (kx,ky) between the two to see how much the full-data
    # refine step actually moves the atom locations
    ka_full = np.array([(a[0], a[1]) for a in g_full])
    ka_fair = np.array([(a[0], a[1]) for a in g_fair])
    print(f"  atom |k| shift (full-refine vs fair-refine), per atom (pre-joint):")
    for i in range(len(ka_full)):
        print(f"    atom{i}: full k=({ka_full[i,0]:+.3f},{ka_full[i,1]:+.3f})  "
              f"fair k=({ka_fair[i,0]:+.3f},{ka_fair[i,1]:+.3f})  "
              f"|dk|={np.hypot(*(ka_full[i]-ka_fair[i])):.4f}")


if __name__ == "__main__":
    run('episode_BF3_5comp_seed0_uniform.h5', 5)
    run('episode_BF4_15comp_seed0_uniform.h5', 15)
