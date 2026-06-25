"""
spacetime_gp_forecast.py — Wave forecasting with space-time dispersion GP.

Evaluates nRMSE as a function of forecast horizon Δt ∈ [0, 6] seconds.
Baseline: zero-mean persistence (predict η(x, t*) as η(x, t*+Δt)).

Ship safety context: need prediction skill at Δt ≥ 2-3s (motion response lag).

Method: condition on W=10 frames ending at t*, predict grid at t* + Δt.
Output: results/spacetime_gp_forecast.json
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from scripts.gp_disp_pilot import estimate_components_wavesplat_opt

G = 9.81
SIGMA_OBS = 0.02
W = 10          # conditioning window frames
COMP_MODE = 'ws_reg_amp'
N_TARGETS = 5   # fast-look (was 20); 9 horizons × grid solves make this the heaviest
                # eval, so keep N low — skill-vs-horizon curve averages over episodes

# Forecast horizons [seconds]
HORIZONS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]

# 15 episodes (3 sea states × 5 seeds) — BF3/BF4/BF5, 5-comp for clean signal
FORECAST_EPISODES = [
    (f'BF3_5c_s{s}', f'episode_BF3_5comp_seed{s}_uniform.h5') for s in range(5)
] + [
    (f'BF4_5c_s{s}', f'episode_BF4_5comp_seed{s}_uniform.h5') for s in range(5)
] + [
    (f'BF4_15c_s{s}', f'episode_BF4_15comp_seed{s}_uniform.h5') for s in range(5)
]


def comps_to_arrays(comps, depth):
    A = comps[:, 0]; kx = comps[:, 2]; ky = comps[:, 3]
    k = np.sqrt(kx**2 + ky**2 + 1e-12)
    omega = np.sqrt(G * k * np.tanh(k * depth))
    return A, omega, kx, ky


def build_K_spacetime(P1, P2, A, omega, kx, ky):
    dx = P1[:, 0:1] - P2[None, :, 0]
    dy = P1[:, 1:2] - P2[None, :, 1]
    dt = P1[:, 2:3] - P2[None, :, 2]
    K = np.zeros((P1.shape[0], P2.shape[0]))
    for n in range(len(A)):
        K += 0.5 * A[n]**2 * np.cos(kx[n]*dx + ky[n]*dy - omega[n]*dt)
    return K


def gp_mean(K_oo, K_po, z, sigma=SIGMA_OBS):
    N = len(z)
    A = K_oo + (sigma**2 + 1e-6) * np.eye(N)
    try:
        alpha = np.linalg.solve(A, z)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(A, z, rcond=None)[0]
    return K_po @ alpha


def frame_obs(f):
    x = f.x_obs.cpu().numpy(); y = f.y_obs.cpu().numpy(); z = f.z_obs.cpu().numpy()
    return np.column_stack([x, y, np.full_like(x, float(f.t))]), z


def eval_forecast_episode(tag, h5_name):
    loader = EpisodeDataLoader(os.path.join('data', h5_name))
    Hs = loader.Hs_true; depth = loader.depth
    gx = loader.gt_grid_x.cpu().numpy(); gy = loader.gt_grid_y.cpu().numpy()
    Xg, Yg = np.meshgrid(gx, gy)
    grid_xy = np.column_stack([Xg.ravel(), Yg.ravel()])

    comps = estimate_components_wavesplat_opt(
        loader, apply_reg_amp=True, n_starts=1, lambda_reg=0.5)
    Ae, oe, kxe, kye = comps_to_arrays(comps, depth)

    frames = loader.get_tracking_frames(tp_factor=2.0)
    n = len(frames)
    dt_frame = float(frames[1].t) - float(frames[0].t)

    # Find targets where we also have frames at all horizons available
    max_horizon = max(HORIZONS)
    max_horizon_frames = int(np.ceil(max_horizon / dt_frame)) + 1
    targets = list(range(W, n - max_horizon_frames))[:N_TARGETS]

    # horizon → list of nRMSE per frame
    horizon_nrmse = {h: [] for h in HORIZONS}
    horizon_persist = {h: [] for h in HORIZONS}

    for ti in targets:
        t_star = float(frames[ti].t)

        # Build conditioning window (past W frames ending at t*)
        Ps, zs = [], []
        for j in range(ti - W + 1, ti + 1):
            Pj, zj = frame_obs(frames[j]); Ps.append(Pj); zs.append(zj)
        Pw = np.vstack(Ps); zw = np.concatenate(zs)

        # Current field for persistence baseline
        current_gt = frames[ti].gt_field.cpu().numpy()

        # Pre-compute K_oo once (conditioning data doesn't change with horizon)
        K_oo = build_K_spacetime(Pw, Pw, Ae, oe, kxe, kye)

        for dh in HORIZONS:
            t_pred = t_star + dh
            # Find the closest future frame
            future_ti = ti + int(round(dh / dt_frame))
            if future_ti >= n:
                continue
            future_frame = frames[future_ti]
            gt = future_frame.gt_field.cpu().numpy()
            grid_pts = np.column_stack([grid_xy, np.full(len(grid_xy), t_pred)])

            # GP forecast
            K_po = build_K_spacetime(grid_pts, Pw, Ae, oe, kxe, kye)
            pred = gp_mean(K_oo, K_po, zw).reshape(gt.shape)

            nRMSE_gp = float(np.sqrt(np.mean((pred - gt)**2)) / max(Hs, 1e-6))
            nRMSE_persist = float(np.sqrt(np.mean((current_gt - gt)**2)) / max(Hs, 1e-6))

            horizon_nrmse[dh].append(nRMSE_gp)
            horizon_persist[dh].append(nRMSE_persist)

    return {
        'tag': tag, 'Hs': Hs, 'n_targets': len(targets),
        'horizons': {
            str(h): {
                'gp_nRMSE': float(np.mean(horizon_nrmse[h])) if horizon_nrmse[h] else None,
                'persist_nRMSE': float(np.mean(horizon_persist[h])) if horizon_persist[h] else None,
                'skill': float(1 - np.mean(horizon_nrmse[h]) / max(np.mean(horizon_persist[h]), 1e-6))
                         if horizon_nrmse[h] and horizon_persist[h] else None,
            }
            for h in HORIZONS
        }
    }


def main():
    print("Space-Time GP Forecast Evaluation")
    print(f"Horizons: {HORIZONS} s")
    print("=" * 60)

    all_results = {}
    skipped = []

    for tag, h5 in FORECAST_EPISODES:
        fpath = os.path.join('data', h5)
        if not os.path.exists(fpath) or os.path.getsize(fpath) < 1000:
            skipped.append(tag); continue

        print(f"[{tag}]", end='', flush=True)
        t0 = time.time()
        try:
            r = eval_forecast_episode(tag, h5)
            all_results[tag] = r
            elapsed = time.time() - t0
            h0 = r['horizons']['0.0']
            h3 = r['horizons']['3.0']
            print(f" Δt=0s: GP={h0['gp_nRMSE']:.4f}  "
                  f"Δt=3s: GP={h3['gp_nRMSE']:.4f} persist={h3['persist_nRMSE']:.4f} "
                  f"skill={h3['skill']:+.3f}  ({elapsed:.0f}s)")
        except Exception as e:
            print(f" ERROR: {e}")
            skipped.append(tag)

    if not all_results:
        print("No episodes evaluated."); return

    # ── Aggregate by horizon ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FORECAST SKILL BY HORIZON (mean over episodes)")
    print(f"{'Horizon':>8}  {'GP nRMSE':>10}  {'Persist nRMSE':>14}  {'Skill':>8}  {'GP wins?':>8}")

    horizon_summary = {}
    for h in HORIZONS:
        gp_vals = [all_results[t]['horizons'][str(h)]['gp_nRMSE']
                   for t in all_results if all_results[t]['horizons'][str(h)]['gp_nRMSE'] is not None]
        pe_vals = [all_results[t]['horizons'][str(h)]['persist_nRMSE']
                   for t in all_results if all_results[t]['horizons'][str(h)]['persist_nRMSE'] is not None]
        if not gp_vals:
            continue
        gp_avg = float(np.mean(gp_vals))
        pe_avg = float(np.mean(pe_vals)) if pe_vals else 1.0
        skill = float(1 - gp_avg / max(pe_avg, 1e-6))
        wins = gp_avg < pe_avg
        print(f"{h:>7.1f}s  {gp_avg:>10.4f}  {pe_avg:>14.4f}  {skill:>+8.3f}  {'YES' if wins else 'NO':>8}")
        horizon_summary[h] = {'gp_nRMSE': gp_avg, 'persist_nRMSE': pe_avg, 'skill': skill}

    # Find skill horizon (when GP stops beating persistence)
    skill_horizons = [(h, horizon_summary[h]['skill']) for h in HORIZONS if h in horizon_summary]
    positive_horizons = [h for h, sk in skill_horizons if sk > 0]
    max_skill_horizon = max(positive_horizons) if positive_horizons else 0.0
    print(f"\n>>> GP beats persistence up to Δt = {max_skill_horizon:.1f}s")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        'description': 'Space-time GP wave forecasting vs persistence',
        'config': {'comp_mode': COMP_MODE, 'W': W, 'horizons': HORIZONS,
                   'sigma_obs': SIGMA_OBS, 'n_targets_per_ep': N_TARGETS},
        'n_evaluated': len(all_results),
        'skipped': skipped,
        'max_skill_horizon_sec': max_skill_horizon,
        'per_episode': all_results,
        'horizon_summary': {str(h): v for h, v in horizon_summary.items()},
    }
    out_path = 'results/spacetime_gp_forecast.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
