"""
spacetime_gp_practical.py — PRACTICAL space-time dispersion-fused GP.

Tests how much of the oracle fusion gain survives with ESTIMATED components.
Reuses the exact component estimator that GP K_disp uses (ws_reg_amp / beamform);
the space-time kernel needs only {A_n, k_n} (omega_n via dispersion) — NO phase
estimation, NO new training. The only change vs the static GP K_disp baseline is
the -omega_n*dt temporal term + joint conditioning on a W-frame window.

Baselines on the SAME 6 test episodes (30-frame protocol):
  - per-frame GP K_disp (static kernel, single frame)  [== existing method]
  - fused space-time GP (W-frame window, dispersion kernel)
  - (oracle fused, for reference ceiling)

Output: results/spacetime_gp_practical.json
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from scripts.gp_disp_pilot import estimate_components_wavesplat_opt, estimate_components_beamform

G = 9.81
SIGMA_OBS = 0.02

TEST_EPISODES = [
    ('BF3_5c',  'episode_BF3_5comp_seed0_slope.h5'),
    ('BF4_5c',  'episode_BF4_5comp_seed0_uniform.h5'),
    ('BF4_15c', 'episode_BF4_15comp_seed0_uniform.h5'),
    ('BF4_30c', 'episode_BF4_30comp_seed0_uniform.h5'),
    ('BF5_15c', 'episode_BF5_15comp_seed0_uniform.h5'),
    ('BF5_30c', 'episode_BF5_30comp_seed0_uniform.h5'),
]


def comps_to_arrays(comps, depth):
    """comps: (n,6) = [A,_,kx,ky,_,_]. Returns A, omega(dispersion), kx, ky."""
    A = comps[:, 0]
    kx = comps[:, 2]
    ky = comps[:, 3]
    k = np.sqrt(kx**2 + ky**2 + 1e-12)
    omega = np.sqrt(G * k * np.tanh(k * depth))
    return A, omega, kx, ky


def oracle_arrays(loader):
    wc = np.array(loader.wave_components)
    return wc[:, 0], wc[:, 1], wc[:, 2], wc[:, 3]


def build_K_static(P1, P2, A, kx, ky):
    dx = P1[:, 0:1] - P2[None, :, 0]
    dy = P1[:, 1:2] - P2[None, :, 1]
    K = np.zeros((P1.shape[0], P2.shape[0]))
    for n in range(len(A)):
        K += 0.5 * A[n]**2 * np.cos(kx[n]*dx + ky[n]*dy)
    return K


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
    Anoise = K_oo + (sigma**2 + 1e-6) * np.eye(N)
    try:
        alpha = np.linalg.solve(Anoise, z)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(Anoise, z, rcond=None)[0]
    return K_po @ alpha


def metrics(pred, gt, Hs):
    err = pred - gt
    nRMSE = float(np.sqrt(np.mean(err**2)) / max(Hs, 1e-6))
    sp, sg = float(np.std(pred)), float(np.std(gt))
    rho = float(np.corrcoef(pred.ravel(), gt.ravel())[0, 1]) if sp > 1e-8 and sg > 1e-8 else 0.0
    return nRMSE, rho


def frame_obs(f):
    x = f.x_obs.cpu().numpy(); y = f.y_obs.cpu().numpy(); z = f.z_obs.cpu().numpy()
    return np.column_stack([x, y, np.full_like(x, float(f.t))]), z


def eval_episode(tag, h5_name, comp_mode='ws_reg_amp', W=10, n_targets=30):
    loader = EpisodeDataLoader(os.path.join('data', h5_name))
    Hs = loader.Hs_true; depth = loader.depth
    gx = loader.gt_grid_x.cpu().numpy(); gy = loader.gt_grid_y.cpu().numpy()
    Xg, Yg = np.meshgrid(gx, gy)
    grid_xy = np.column_stack([Xg.ravel(), Yg.ravel()])

    # Estimate components (the same way GP K_disp does)
    t0 = time.time()
    if comp_mode == 'ws_reg_amp':
        comps = estimate_components_wavesplat_opt(
            loader, apply_reg_amp=True, n_starts=1, lambda_reg=0.5)
    elif comp_mode == 'ws_ms5_reg_amp':
        comps = estimate_components_wavesplat_opt(
            loader, apply_reg_amp=True, n_starts=5, lambda_reg=0.5)
    elif comp_mode == 'beamform':
        comps = estimate_components_beamform(loader, n_gaussians=15)
    else:
        raise ValueError(comp_mode)
    est_time = time.time() - t0
    Ae, oe, kxe, kye = comps_to_arrays(comps, depth)
    Ao, oo, kxo, kyo = oracle_arrays(loader)

    frames = loader.get_tracking_frames(tp_factor=2.0)
    n = len(frames)
    targets = list(range(W, min(n, W + n_targets)))

    pf, fused, fused_or = [], [], []   # per-frame est / fused est / fused oracle
    pf_rho, fused_rho = [], []
    for ti in targets:
        target = frames[ti]
        t_star = float(target.t)
        gt = target.gt_field.cpu().numpy()
        grid_pts = np.column_stack([grid_xy, np.full(len(grid_xy), t_star)])

        # per-frame static GP (estimated comps) -- the existing GP K_disp method
        P1, z1 = frame_obs(target)
        K_oo = build_K_static(P1[:, :2], P1[:, :2], Ae, kxe, kye)
        K_po = build_K_static(grid_xy, P1[:, :2], Ae, kxe, kye)
        pred = gp_mean(K_oo, K_po, z1).reshape(gt.shape)
        r, rho = metrics(pred, gt, Hs); pf.append(r); pf_rho.append(rho)

        # fused space-time GP (estimated comps)
        Ps, zs = [], []
        for j in range(ti - W + 1, ti + 1):
            Pj, zj = frame_obs(frames[j]); Ps.append(Pj); zs.append(zj)
        Pw = np.vstack(Ps); zw = np.concatenate(zs)
        K_oo = build_K_spacetime(Pw, Pw, Ae, oe, kxe, kye)
        K_po = build_K_spacetime(grid_pts, Pw, Ae, oe, kxe, kye)
        pred = gp_mean(K_oo, K_po, zw).reshape(gt.shape)
        r, rho = metrics(pred, gt, Hs); fused.append(r); fused_rho.append(rho)

        # fused oracle (reference ceiling)
        K_oo = build_K_spacetime(Pw, Pw, Ao, oo, kxo, kyo)
        K_po = build_K_spacetime(grid_pts, Pw, Ao, oo, kxo, kyo)
        pred = gp_mean(K_oo, K_po, zw).reshape(gt.shape)
        r, _ = metrics(pred, gt, Hs); fused_or.append(r)

    res = {
        'comp_mode': comp_mode, 'W': W, 'n_targets': len(targets),
        'est_time_sec': est_time,
        'perframe_nRMSE': float(np.mean(pf)),
        'perframe_Pearson': float(np.mean(pf_rho)),
        'fused_nRMSE': float(np.mean(fused)),
        'fused_Pearson': float(np.mean(fused_rho)),
        'fused_oracle_nRMSE': float(np.mean(fused_or)),
    }
    res['fusion_gain_pct'] = float(
        (res['perframe_nRMSE'] - res['fused_nRMSE']) / res['perframe_nRMSE'] * 100)
    return res


def main():
    comp_mode = sys.argv[1] if len(sys.argv) > 1 else 'ws_reg_amp'
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(f"PRACTICAL space-time GP — comp_mode={comp_mode}, W={W}")
    print("=" * 64)
    results = {}
    for tag, h5 in TEST_EPISODES:
        if not os.path.exists(os.path.join('data', h5)):
            print(f"SKIP {tag}"); continue
        print(f"\n### {tag} ###")
        r = eval_episode(tag, h5, comp_mode=comp_mode, W=W)
        results[tag] = r
        print(f"  per-frame (est, GP K_disp): nRMSE={r['perframe_nRMSE']:.4f}  P={r['perframe_Pearson']:.3f}")
        print(f"  fused     (est, space-time): nRMSE={r['fused_nRMSE']:.4f}  P={r['fused_Pearson']:.3f}  "
              f"[{r['fusion_gain_pct']:+.1f}%]")
        print(f"  fused     (oracle ceiling):  nRMSE={r['fused_oracle_nRMSE']:.4f}")

    # summary
    def avg(key): return float(np.mean([results[t][key] for t in results]))
    print("\n" + "=" * 64)
    print("SUMMARY (same 6 test eps)")
    print(f"  per-frame GP K_disp (est) : nRMSE={avg('perframe_nRMSE'):.4f}  P={avg('perframe_Pearson'):.3f}")
    print(f"  fused space-time   (est)  : nRMSE={avg('fused_nRMSE'):.4f}  P={avg('fused_Pearson'):.3f}")
    print(f"  fused space-time (oracle) : nRMSE={avg('fused_oracle_nRMSE'):.4f}")
    print(f"  >>> mean fusion gain: {avg('fusion_gain_pct'):+.1f}%")

    out = {
        'description': f'Practical space-time dispersion-fused GP ({comp_mode}, W={W}) '
                       'vs per-frame GP K_disp, same 6 test episodes',
        'config': {'comp_mode': comp_mode, 'W': W, 'sigma_obs': SIGMA_OBS},
        'per_episode': results,
        'summary': {
            'perframe_nRMSE': avg('perframe_nRMSE'),
            'fused_nRMSE': avg('fused_nRMSE'),
            'fused_oracle_nRMSE': avg('fused_oracle_nRMSE'),
            'mean_fusion_gain_pct': avg('fusion_gain_pct'),
        },
    }
    with open(f'results/spacetime_gp_practical_{comp_mode}_W{W}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: results/spacetime_gp_practical_{comp_mode}_W{W}.json")


if __name__ == '__main__':
    main()
