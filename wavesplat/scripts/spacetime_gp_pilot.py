"""
spacetime_gp_pilot.py — Oracle de-risk pilot for the Space-Time Dispersion-Fused GP.

Tests the core hypothesis before building the full practical pipeline:
  Does fusing multiple frames through the dispersion relation (the -omega*dt phase
  term) add usable information that a single ~50-obs frame cannot?

Kernel (space-time dispersion-constrained spectral mixture):
  K((x,t),(x',t')) = sum_n (A_n^2/2) cos(k_n . (x-x') - omega_n (t-t'))
  with omega_n = sqrt(g |k_n| tanh(|k_n| d))   [enforced by construction]

We use ORACLE components {A_n, k_n, omega_n} (from wave_components) to isolate the
question "does temporal fusion help?" from "can we estimate components?".

Two metrics, on an easy (5-comp) AND a complex (30-comp) episode:
  M1 SUPER-RES : reconstruct frame t*'s field using (a) only t*'s ~50 obs [per-frame]
                 vs (b) a W-frame window ending at t* [fused].  Lower nRMSE = fusion helps.
  M2 FORECAST  : condition on W frames ending at t0, predict t0+h.
                 Compare vs persistence (hold the t0 reconstruction).  Lower = real forecast.

Output: results/spacetime_gp_pilot.json
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader

G = 9.81
SIGMA_OBS = 0.02

EPISODES = [
    ('BF4_5c',  'episode_BF4_5comp_seed0_uniform.h5'),   # oracle-easy
    ('BF4_30c', 'episode_BF4_30comp_seed0_uniform.h5'),  # complex, single-frame starved
    ('BF5_30c', 'episode_BF5_30comp_seed0_uniform.h5'),  # complex, high sea state
]


def oracle_components(loader):
    """Return A[n], omega[n], kx[n], ky[n] from wave_components (col layout
    [A, omega, kx, ky, phi, dir]; omega verified == dispersion(k))."""
    wc = np.array(loader.wave_components)
    A = wc[:, 0]
    omega = wc[:, 1]
    kx = wc[:, 2]
    ky = wc[:, 3]
    return A, omega, kx, ky


def build_K_spacetime(P1, P2, A, omega, kx, ky):
    """Space-time dispersion kernel.
    P1: (N1, 3) = (x, y, t); P2: (M, 3).
    K = sum_n (A_n^2/2) cos(kx_n*dx + ky_n*dy - omega_n*dt)
    """
    dx = P1[:, 0:1] - P2[None, :, 0]   # (N1, M)
    dy = P1[:, 1:2] - P2[None, :, 1]
    dt = P1[:, 2:3] - P2[None, :, 2]
    K = np.zeros((P1.shape[0], P2.shape[0]))
    for n in range(len(A)):
        phase = kx[n] * dx + ky[n] * dy - omega[n] * dt
        K += 0.5 * A[n] ** 2 * np.cos(phase)
    return K


def gp_mean(K_oo, K_po, z_obs, sigma=SIGMA_OBS):
    N = len(z_obs)
    Anoise = K_oo + (sigma ** 2 + 1e-6) * np.eye(N)
    try:
        alpha = np.linalg.solve(Anoise, z_obs)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(Anoise, z_obs, rcond=None)[0]
    return K_po @ alpha


def metrics(pred, gt, Hs):
    err = pred - gt
    nRMSE = float(np.sqrt(np.mean(err ** 2)) / max(Hs, 1e-6))
    sp, sg = float(np.std(pred)), float(np.std(gt))
    rho = float(np.corrcoef(pred.ravel(), gt.ravel())[0, 1]) if sp > 1e-8 and sg > 1e-8 else 0.0
    return nRMSE, rho


def collect_frame_obs(frame):
    x = frame.x_obs.cpu().numpy()
    y = frame.y_obs.cpu().numpy()
    z = frame.z_obs.cpu().numpy()
    t = float(frame.t)
    P = np.column_stack([x, y, np.full_like(x, t)])
    return P, z


def run_episode(tag, h5_name, W=10, forecast_horizons=(5, 10, 20, 30)):
    loader = EpisodeDataLoader(os.path.join('data', h5_name))
    A, omega, kx, ky = oracle_components(loader)
    Hs = loader.Hs_true
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    Xg, Yg = np.meshgrid(gx, gy)
    grid_xy = np.column_stack([Xg.ravel(), Yg.ravel()])  # (Ngrid, 2)

    frames = loader.get_tracking_frames(tp_factor=2.0)
    n = len(frames)

    # ---- M1 SUPER-RES: reconstruct target frame field, per-frame vs fused ----
    perframe_nrmse, perframe_rho = [], []
    fused_nrmse, fused_rho = [], []
    # evaluate on target frames spaced through the episode (need W history each)
    targets = list(range(W, min(n, W + 60)))  # up to 60 target frames
    for ti in targets:
        target = frames[ti]
        t_star = float(target.t)
        gt = target.gt_field.cpu().numpy()
        grid_pts = np.column_stack([grid_xy, np.full(len(grid_xy), t_star)])

        # (a) per-frame: condition on target frame obs only
        P1, z1 = collect_frame_obs(target)
        K_oo = build_K_spacetime(P1, P1, A, omega, kx, ky)
        K_po = build_K_spacetime(grid_pts, P1, A, omega, kx, ky)
        pred_pf = gp_mean(K_oo, K_po, z1).reshape(gt.shape)
        r, rho = metrics(pred_pf, gt, Hs)
        perframe_nrmse.append(r); perframe_rho.append(rho)

        # (b) fused: condition on W frames ending at target (incl. target)
        Ps, zs = [], []
        for j in range(ti - W + 1, ti + 1):
            Pj, zj = collect_frame_obs(frames[j])
            Ps.append(Pj); zs.append(zj)
        Pw = np.vstack(Ps); zw = np.concatenate(zs)
        K_oo = build_K_spacetime(Pw, Pw, A, omega, kx, ky)
        K_po = build_K_spacetime(grid_pts, Pw, A, omega, kx, ky)
        pred_fs = gp_mean(K_oo, K_po, zw).reshape(gt.shape)
        r, rho = metrics(pred_fs, gt, Hs)
        fused_nrmse.append(r); fused_rho.append(rho)

    m1 = {
        'perframe_nRMSE': float(np.mean(perframe_nrmse)),
        'perframe_Pearson': float(np.mean(perframe_rho)),
        'fused_nRMSE': float(np.mean(fused_nrmse)),
        'fused_Pearson': float(np.mean(fused_rho)),
        'n_targets': len(targets),
        'W': W,
    }
    m1['fusion_gain_pct'] = float(
        (m1['perframe_nRMSE'] - m1['fused_nRMSE']) / m1['perframe_nRMSE'] * 100)

    # ---- M2 FORECAST: condition on W frames ending at t0, predict t0+h ----
    forecast = {}
    # use several t0 anchors
    anchors = list(range(W, min(n - max(forecast_horizons), W + 50), 5))
    for h in forecast_horizons:
        fc_nrmse, pers_nrmse = [], []
        for t0 in anchors:
            # history window [t0-W+1 .. t0]
            Ps, zs = [], []
            for j in range(t0 - W + 1, t0 + 1):
                Pj, zj = collect_frame_obs(frames[j])
                Ps.append(Pj); zs.append(zj)
            Pw = np.vstack(Ps); zw = np.concatenate(zs)
            K_oo = build_K_spacetime(Pw, Pw, A, omega, kx, ky)

            # forecast target frame t0+h
            tgt = frames[t0 + h]
            t_fut = float(tgt.t)
            gt_fut = tgt.gt_field.cpu().numpy()

            grid_fut = np.column_stack([grid_xy, np.full(len(grid_xy), t_fut)])
            K_po_fut = build_K_spacetime(grid_fut, Pw, A, omega, kx, ky)
            pred_fc = gp_mean(K_oo, K_po_fut, zw).reshape(gt_fut.shape)
            r_fc, _ = metrics(pred_fc, gt_fut, Hs)
            fc_nrmse.append(r_fc)

            # persistence: reconstruct field at t0 (last anchor frame time), hold it
            t0_time = float(frames[t0].t)
            grid_t0 = np.column_stack([grid_xy, np.full(len(grid_xy), t0_time)])
            K_po_t0 = build_K_spacetime(grid_t0, Pw, A, omega, kx, ky)
            pred_t0 = gp_mean(K_oo, K_po_t0, zw).reshape(gt_fut.shape)
            r_pers, _ = metrics(pred_t0, gt_fut, Hs)  # compare t0 recon vs FUTURE gt
            pers_nrmse.append(r_pers)

        forecast[f'h{h}'] = {
            'dt_sec': h * 0.1,
            'forecast_nRMSE': float(np.mean(fc_nrmse)),
            'persistence_nRMSE': float(np.mean(pers_nrmse)),
            'beats_persistence': bool(np.mean(fc_nrmse) < np.mean(pers_nrmse)),
        }

    return {'M1_superres': m1, 'M2_forecast': forecast, 'Hs': Hs, 'n_frames': n}


def main():
    print("Space-Time Dispersion-Fused GP — ORACLE de-risk pilot")
    print("=" * 64)
    results = {}
    for tag, h5 in EPISODES:
        if not os.path.exists(os.path.join('data', h5)):
            print(f"SKIP {tag}")
            continue
        print(f"\n### {tag} ({h5}) ###")
        res = run_episode(tag, h5)
        results[tag] = res

        m1 = res['M1_superres']
        print(f"  [M1 super-res, W={m1['W']}, {m1['n_targets']} targets]")
        print(f"    per-frame oracle : nRMSE={m1['perframe_nRMSE']:.4f}  "
              f"Pearson={m1['perframe_Pearson']:.3f}")
        print(f"    fused (space-time): nRMSE={m1['fused_nRMSE']:.4f}  "
              f"Pearson={m1['fused_Pearson']:.3f}")
        print(f"    >>> fusion gain: {m1['fusion_gain_pct']:+.1f}% nRMSE")

        print(f"  [M2 forecast vs persistence]")
        for hk, fc in res['M2_forecast'].items():
            flag = "WIN" if fc['beats_persistence'] else "lose"
            print(f"    +{fc['dt_sec']:.1f}s: forecast={fc['forecast_nRMSE']:.4f}  "
                  f"persist={fc['persistence_nRMSE']:.4f}  [{flag}]")

    with open('results/spacetime_gp_pilot.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/spacetime_gp_pilot.json")


if __name__ == '__main__':
    main()
