"""
r4_nearfar_power_probe.py — red-team probe for ALTMETHODS_GATE Round 4 near-band claim.

Re-derives, independently of altmethods_nearfar.py:
  (A) whether the `win` fallback branch silently fires (attack #2), and what the
      "in-frame" window actually spans in frames/metres of ship travel;
  (B) the near-band (0-20m) BEAM vs BANK nRMSE across MANY frames (not just frame=150)
      to get an actual SE on the 0.0716 vs 0.0743 "parity" claim (attack #3);
  (C) sensitivity of BEAM to the oracle atom count n_true+1 (attack #1) by sweeping
      atom count and re-scoring the near band.

Run from wavesplat/wavesplat/.
"""
import sys, time, numpy as np
sys.path.insert(0, 'tmp')
sys.path.insert(0, 'scripts')
from blasso_feasibility import load_obs, greedy, joint_refine, build_grid, corr_at, model_eval
from d3_sa_nomp import mmse_bank_estimate

RNG = np.random.default_rng(0)


def win_mask(ts, tf, mode):
    dt = np.abs(ts - tf)
    thresh = max(np.median(np.diff(np.unique(ts))) * 1.5, 1e-9)
    w = dt < thresh
    if mode == 'aswritten' and w.sum() < 50:
        w = dt < np.quantile(dt, 0.02)
    elif mode == 'exact0':
        w = dt == 0.0
    return w, dt


def nrmse(pred, truth, m, Hs):
    return float(np.sqrt(np.mean((pred[m] - truth[m]) ** 2)) / Hs)


def probe(fn, n_true, frames_step=30, nsub=2600, n_atom_sweep=None):
    print("\n" + "=" * 78)
    print(fn, "n_true=", n_true)
    obs = load_obs('data/' + fn)
    M = len(obs['x'])
    sub = RNG.choice(M, size=min(nsub, M), replace=False)
    xs, ys, ts, zs = obs['x'][sub], obs['y'][sub], obs['t'][sub], obs['z'][sub]
    gx, gy = obs['gx'], obs['gy']
    X, Y = np.meshgrid(gx, gy, indexing='xy')
    xq, yq = X.ravel(), Y.ravel()

    # --- window diagnostic at frame 150 (matches the published run) ---
    tf150 = obs['ts'][150]
    for mode in ('aswritten', 'exact0'):
        w, dt = win_mask(ts, tf150, mode)
        span = np.sort(np.unique(np.round(dt[w] / 0.1).astype(int))) if w.sum() else []
        print(f"  [window @frame150] mode={mode:9s} n_in_win={w.sum():4d}  "
          f"frame-offsets included={list(span)}")

    # --- BEAM fit (oracle n_true+1), independent of frame ---
    gridk = build_grid()
    lam_b = 0.03 * np.abs(corr_at(*gridk, xs, ys, ts, zs, obs['depth']).max())
    t0 = time.time()
    gj_oracle = joint_refine(greedy(obs, n_true + 1, gridk, sub), obs, sub, lam_b)
    print(f"  BEAM fit (n_atoms={n_true+1}, oracle) done in {time.time()-t0:.1f}s")

    # --- atom-count sweep (attack #1): does BEAM need the oracle count? ---
    if n_atom_sweep:
        f_mid = obs['nfr'] // 2
        tfm = obs['ts'][f_mid]
        truth_mid = obs['gtf'][f_mid].ravel()
        w_mid, _ = win_mask(ts, tfm, 'aswritten')
        ox, oy = xs[w_mid], ys[w_mid]
        d_mid = np.sqrt(np.min((xq[:, None] - ox[None]) ** 2 + (yq[:, None] - oy[None]) ** 2, axis=1))
        near_mid = d_mid < 20
        print(f"  --- atom-count sweep (frame {f_mid}, near-band n={near_mid.sum()}) ---")
        for na in n_atom_sweep:
            t1 = time.time()
            gj_k = joint_refine(greedy(obs, na, gridk, sub), obs, sub, lam_b)
            pred_k = model_eval(gj_k, xq, yq, np.full(xq.size, tfm), obs['depth'])
            nr_near = nrmse(pred_k, truth_mid, near_mid, obs['Hs'])
            nr_all = nrmse(pred_k, truth_mid, np.ones_like(near_mid, bool), obs['Hs'])
            print(f"    n_atoms={na:3d} (n_true+{na-n_true:+d}): near0-20={nr_near:.4f}  "
                  f"full-grid={nr_all:.4f}   [{time.time()-t1:.1f}s]")

    # --- multi-frame near-band BEAM vs BANK (attack #3: power) ---
    frames = list(range(30, obs['nfr'] - 29, frames_step))
    print(f"  --- multi-frame near-band (0-20m) sweep, frames={frames} ---")
    Pq_all = np.column_stack([np.tile(xq, len(frames)), np.tile(yq, len(frames)),
                               np.repeat(obs['ts'][frames], xq.size)])
    t0 = time.time()
    pred_bank_all = mmse_bank_estimate(np.column_stack([xs, ys, ts]), zs, obs['depth'],
                                        Pq_all, 0.05 * obs['Hs'])
    print(f"  BANK batched query ({len(frames)} frames) done in {time.time()-t0:.1f}s")

    rows = []
    for i, f in enumerate(frames):
        tf = obs['ts'][f]
        truth = obs['gtf'][f].ravel()
        pred_beam = model_eval(gj_oracle, xq, yq, np.full(xq.size, tf), obs['depth'])
        pred_bank = pred_bank_all[i * xq.size:(i + 1) * xq.size]

        w_asw, _ = win_mask(ts, tf, 'aswritten')
        ox, oy = xs[w_asw], ys[w_asw]
        d = np.sqrt(np.min((xq[:, None] - ox[None]) ** 2 + (yq[:, None] - oy[None]) ** 2, axis=1))
        near = d < 20
        far = d >= 100

        nb_beam = nrmse(pred_beam, truth, near, obs['Hs'])
        nb_bank = nrmse(pred_bank, truth, near, obs['Hs'])
        fb_beam = nrmse(pred_beam, truth, far, obs['Hs']) if far.sum() else np.nan
        fb_bank = nrmse(pred_bank, truth, far, obs['Hs']) if far.sum() else np.nan
        rows.append((f, near.sum(), nb_beam, nb_bank, far.sum(), fb_beam, fb_bank))
        print(f"    frame={f:3d} n_near={near.sum():4d} BEAM={nb_beam:.4f} BANK={nb_bank:.4f} "
              f"Delta(BANK-BEAM)={nb_bank-nb_beam:+.4f}   |  n_far={far.sum():4d} "
              f"BEAM_far={fb_beam:.4f} BANK_far={fb_bank:.4f}")

    rows = np.array([(r[2], r[3]) for r in rows])  # BEAM, BANK near-band per frame
    d = rows[:, 1] - rows[:, 0]
    print(f"  SUMMARY near-band (0-20m), n_frames={len(d)}:")
    print(f"    BEAM  mean={rows[:,0].mean():.4f} sd={rows[:,0].std(ddof=1):.4f}")
    print(f"    BANK  mean={rows[:,1].mean():.4f} sd={rows[:,1].std(ddof=1):.4f}")
    print(f"    Delta(BANK-BEAM) mean={d.mean():+.4f}  sd={d.std(ddof=1):.4f}  "
          f"SE(mean)={d.std(ddof=1)/np.sqrt(len(d)):.4f}  "
          f"t={d.mean()/(d.std(ddof=1)/np.sqrt(len(d))+1e-12):+.2f}  n={len(d)}")
    return rows


if __name__ == "__main__":
    probe('episode_BF3_5comp_seed0_uniform.h5', 5,
          n_atom_sweep=[3, 5, 6, 8, 12, 20])
    probe('episode_BF4_15comp_seed0_uniform.h5', 15,
          n_atom_sweep=[8, 15, 16, 20, 25])
