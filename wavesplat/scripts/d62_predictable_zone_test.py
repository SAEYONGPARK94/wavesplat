"""
d62 — MAKE-OR-BREAK novelty test: FIM (CRB) predictable zone vs geometric (group-velocity) DPZ.

The canonical "predictable zone" in deterministic sea-wave prediction is GEOMETRIC: the region
the measured energy can propagate into at the group velocity (Naaijen; Lee-Nam-Liu-Yang 2024).
Our Direction-I claim is that under SPARSE, MOVING ship-LiDAR the binding constraint is instead
Fisher-information IDENTIFIABILITY, and the true predictable boundary is set by the CRB field-σ,
not the c_g cone.

This script tests the claim on NONLINEAR HOS data (standing directive: validation must be HOS):
  1. observe a LiDAR window [0, t_now] on an HOS sea, run the deployable RELAX estimator,
  2. FORECAST the field over a downstream grid at horizons t* = t_now + Δt,
  3. measure the ACTUAL pointwise error e(x,y,t*)/Hs vs the HOS ground truth,
  4. compute two competing predictors of where the forecast is good:
       - FIM:  σ_z(x,y,t*)/Hs  = CRB field-std from the estimated components' Fisher information,
       - GEO:  d_geo(x,y,t*)   = distance to the energy advected from the footprint by the
                                 estimated components' group-velocity vectors (the geometric DPZ),
  5. ask WHICH predictor tracks the actual error better (Spearman, AUC), and whether the two
     zones even DIFFER (IoU at matched area).

Verdict logic:
  - FIM clearly beats GEO at predicting where forecast works AND the zones differ  -> core CONFIRMED.
  - they coincide / GEO wins                                                        -> novelty COLLAPSES
                                                                                       (must know now).

Run: python3 scripts/d62_predictable_zone_test.py
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import torch
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

from wavesim.hos_simulator import HOSSimulator, HOSConfig
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from d18_relax_estimator import relax_step, MCAP
from d17_recursive_estimator import refine_params
from d40_learned_acquisition import pool_xy

# FIM machinery (analytic Jacobian, float64) — reuse Direction-I primitives
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1_crb import jacobian, fisher, DT, DEVICE

DEPTH = 50.0
G = 9.81
DT_FRAME = 0.1
K_TP = 2.2
SIGMA = 0.02
HORIZONS = [0.0, 2.0, 4.0, 6.0]
# HOS seas (tag, Hs, Tp) — span mild→severe; complex partially-identified regimes
SEAS = [('BF3_5c', 0.8, 5.5), ('BF4_15c', 1.2, 7.0), ('BF5_30c', 2.0, 9.0)]
SEEDS = [0, 1, 2, 3]
TAU = 0.25          # zero-predictor skill threshold (e/Hs < TAU == "actually predictable")
TAU2 = 0.15


def cg_np(k, depth=DEPTH):
    kd = k * depth
    th = np.tanh(kd)
    w = np.sqrt(G * k * th + 1e-30)
    return G * (th + kd * (1 - th ** 2)) / (2 * w + 1e-30)


def build_window(Hs, Tp, seed, gx, gy, M=3):
    """Run HOS, collect LiDAR obs over [0,t_now] (n_back frames) and GT grid frames at
    t_now and the forecast horizons."""
    cfg = HOSConfig(Lx=600.0, Ly=400.0, Nx=64, Ny=64, depth=DEPTH, M=M)
    sim = HOSSimulator(cfg)
    sim.initialize_jonswap(Hs=Hs, Tp=Tp, seed=seed, direction_deg=0.0, direction_spread_deg=30.0)
    Xg, Yg = np.meshgrid(gx, gy)
    pool = pool_xy()
    n_back = int(round(K_TP * Tp / DT_FRAME))
    h_steps = sorted({int(round(h / DT_FRAME)) for h in HORIZONS})
    t_now_step = n_back
    last = t_now_step + max(h_steps)

    obs_P, obs_z = [], []
    gt_at = {}              # step -> (t, GT grid)
    t = 0.0
    for i in range(last + 1):
        if i > 0:
            sim.step(DT_FRAME); t += DT_FRAME
        if i <= t_now_step:                                   # accumulate observation window
            idx = np.random.default_rng(1000 + i).choice(len(pool), 60, replace=False)
            ox, oy = pool[idx, 0], pool[idx, 1]
            Zo, *_ = sim.evaluate_at(ox.reshape(1, -1), oy.reshape(1, -1))
            zo = Zo.ravel() + np.random.default_rng(i).normal(0, SIGMA, len(ox))
            obs_P.append(np.column_stack([ox, oy, np.full_like(ox, t)]))
            obs_z.append(zo)
        rel = i - t_now_step
        if i == t_now_step or (rel in h_steps and rel >= 0):
            Zg, *_ = sim.evaluate_at(Xg, Yg)
            gt_at[i - t_now_step] = (t, Zg.astype(np.float64))
    P = np.vstack(obs_P).astype(np.float64)
    z = np.concatenate(obs_z).astype(np.float64)
    t_now = t_now_step * DT_FRAME
    return dict(P=P, z=z, t_now=t_now, gt_at=gt_at, Hs=Hs, n_back=n_back)


def fim_sigma_map(theta_np, Px, Py, Pt, gx, gy, t_star):
    """CRB field-std σ_z(x,y,t*) over the grid from the estimated components' FIM.
    theta_np: (N,4)=[A,kx,ky,phi]. Returns (Ny,Nx) of σ_z (metres)."""
    theta = torch.tensor(theta_np, dtype=DT, device=DEVICE)
    x = torch.tensor(Px, dtype=DT, device=DEVICE)
    y = torch.tensor(Py, dtype=DT, device=DEVICE)
    t = torch.tensor(Pt, dtype=DT, device=DEVICE)
    F = fisher(theta, x, y, t, DEPTH, SIGMA)
    n = F.shape[0]
    Finv = torch.linalg.inv(F + 1e-8 * torch.trace(F) / n * torch.eye(n, dtype=DT, device=F.device))
    Xg, Yg = np.meshgrid(gx, gy)
    xf = torch.tensor(Xg.ravel(), dtype=DT, device=DEVICE)
    yf = torch.tensor(Yg.ravel(), dtype=DT, device=DEVICE)
    tf = torch.full_like(xf, float(t_star))
    Jg = jacobian(theta, xf, yf, tf, DEPTH)                  # (P,4N)
    var = torch.einsum('pa,ab,pb->p', Jg, Finv, Jg)
    sig = torch.sqrt(torch.clamp(var, min=0.0)).cpu().numpy()
    return sig.reshape(Xg.shape)


def geo_distance_map(kx, ky, Px, Py, Pt, gx, gy, t_star, sub=400):
    """Geometric DPZ predictor: min distance from each grid point to the observation cloud
    advected forward to t* by EACH estimated component's group-velocity vector (the most
    generous, physically-faithful geometric predictable zone)."""
    k = np.sqrt(kx ** 2 + ky ** 2) + 1e-12
    cg = cg_np(k)
    vx, vy = cg * kx / k, cg * ky / k                        # (N,) component group-velocity vectors
    M = len(Px)
    if M > sub:
        sel = np.random.default_rng(0).choice(M, sub, replace=False)
        Px, Py, Pt = Px[sel], Py[sel], Pt[sel]
    dt = t_star - Pt                                         # lead per obs point (>=0)
    # advected cloud: (obs m) x (component i)
    ax = (Px[:, None] + vx[None, :] * dt[:, None]).ravel()
    ay = (Py[:, None] + vy[None, :] * dt[:, None]).ravel()
    cloud = np.column_stack([ax, ay])
    Xg, Yg = np.meshgrid(gx, gy)
    grid = np.column_stack([Xg.ravel(), Yg.ravel()])
    d = cdist(grid, cloud).min(axis=1)
    return d.reshape(Xg.shape)


def estimate(P, z, t_now):
    """Deployable RELAX (3 passes) on the window; fit amplitudes/phases; return components."""
    gx_d = np.linspace(P[:, 0].min(), P[:, 0].max(), 40)
    gy_d = np.linspace(P[:, 1].min(), P[:, 1].max(), 40)
    target = {'gx': gx_d, 'gy': gy_d, 't': t_now, 'gt': np.zeros((40, 40)), 'Hs': 1.0}
    kx, ky = np.zeros(0), np.zeros(0)
    for _ in range(3):
        kx, ky = relax_step(kx, ky, P, z, DEPTH)
        kx, ky, _ = refine_params(kx, ky, P, z, DEPTH, target)
    obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z])
    if len(obs) > MCAP:
        obs = obs[np.random.default_rng(1).choice(len(obs), MCAP, replace=False)]
    A, phi = fit_amplitudes_phases(kx, ky, obs, DEPTH, lambda_reg=0.5)
    return kx, ky, A, phi


def iou_at_matched_area(score_lower_better_a, score_lower_better_b, good_mask):
    """Threshold both predictors so their predicted-good area == actual good area, then IoU of
    the two predicted-good masks (does FIM define the same zone as GEO?)."""
    frac = good_mask.mean()
    if frac <= 0 or frac >= 1:
        return None
    ta = np.quantile(score_lower_better_a, frac)
    tb = np.quantile(score_lower_better_b, frac)
    ma = score_lower_better_a <= ta
    mb = score_lower_better_b <= tb
    inter = (ma & mb).sum(); union = (ma | mb).sum()
    return float(inter / max(union, 1))


def main():
    out = 'results/direction3'
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    gx = np.linspace(0.0, 320.0, 90)        # downstream (+x propagation) → straddles the boundary
    gy = np.linspace(-110.0, 110.0, 64)

    pooled = dict(err=[], fim=[], geo=[])
    per_horizon = {h: dict(err=[], fim=[], geo=[]) for h in HORIZONS if h > 0}
    rows = []
    panel_cache = None

    for tag, Hs, Tp in SEAS:
        for seed in SEEDS:
            w = build_window(Hs, Tp, seed, gx, gy)
            kx, ky, A, phi = estimate(w['P'], w['z'], w['t_now'])
            theta = np.column_stack([A, kx, ky, phi])
            Hs_n = float(w['gt_at'][0][1].std() * 4)
            print(f"[{tag} s{seed}] N_est={len(kx)}  Hs={Hs_n:.2f}  t_now={w['t_now']:.1f}s "
                  f"({time.time()-t0:.0f}s)", flush=True)
            for h in HORIZONS:
                step = int(round(h / DT_FRAME))
                if step not in w['gt_at']:
                    continue
                t_star, GT = w['gt_at'][step]
                fcast = render_wave_field(kx, ky, A, phi, DEPTH, gx, gy, t_star)
                err = np.abs(fcast - GT) / Hs_n
                sig = fim_sigma_map(theta, w['P'][:, 0], w['P'][:, 1], w['P'][:, 2], gx, gy, t_star) / Hs_n
                dgeo = geo_distance_map(kx, ky, w['P'][:, 0], w['P'][:, 1], w['P'][:, 2], gx, gy, t_star)
                good = err < TAU
                row = dict(tag=tag, seed=seed, horizon=h,
                           frac_actual_good=float(good.mean()),
                           rmse=float(np.sqrt(np.mean((fcast - GT) ** 2)) / Hs_n))
                if 0 < good.mean() < 1:
                    row['spearman_fim'] = float(spearmanr(sig.ravel(), err.ravel()).correlation)
                    row['spearman_geo'] = float(spearmanr(dgeo.ravel(), err.ravel()).correlation)
                    row['auc_fim'] = float(roc_auc_score((~good).ravel(), sig.ravel()))
                    row['auc_geo'] = float(roc_auc_score((~good).ravel(), dgeo.ravel()))
                    row['iou_fim_geo'] = iou_at_matched_area(sig.ravel(), dgeo.ravel(), good.ravel())
                rows.append(row)
                if h > 0:
                    pooled['err'].append(err.ravel()); pooled['fim'].append(sig.ravel()); pooled['geo'].append(dgeo.ravel())
                    per_horizon[h]['err'].append(err.ravel()); per_horizon[h]['fim'].append(sig.ravel()); per_horizon[h]['geo'].append(dgeo.ravel())
                if panel_cache is None and tag == 'BF5_30c' and seed == 0 and h == 4.0:
                    panel_cache = dict(gx=gx, gy=gy, err=err, sig=sig, dgeo=dgeo, GT=GT,
                                       fcast=fcast, tag=tag, h=h, tau=TAU)

    # pooled metrics (forecast horizons only)
    e = np.concatenate(pooled['err']); f = np.concatenate(pooled['fim']); g = np.concatenate(pooled['geo'])
    good = e < TAU
    summary = dict(
        n_cells=int(e.size),
        spearman_fim=float(spearmanr(f, e).correlation),
        spearman_geo=float(spearmanr(g, e).correlation),
        auc_fim=float(roc_auc_score((~good), f)),
        auc_geo=float(roc_auc_score((~good), g)),
        auc_fim_tau15=float(roc_auc_score((e >= TAU2), f)),
        auc_geo_tau15=float(roc_auc_score((e >= TAU2), g)),
        iou_fim_geo=iou_at_matched_area(f, g, good),
        frac_good=float(good.mean()),
    )
    ph = {}
    for h in per_horizon:
        if not per_horizon[h]['err']:
            continue
        eh = np.concatenate(per_horizon[h]['err']); fh = np.concatenate(per_horizon[h]['fim']); gh = np.concatenate(per_horizon[h]['geo'])
        gd = eh < TAU
        ph[str(h)] = dict(auc_fim=float(roc_auc_score((~gd), fh)) if 0 < gd.mean() < 1 else None,
                          auc_geo=float(roc_auc_score((~gd), gh)) if 0 < gd.mean() < 1 else None,
                          spearman_fim=float(spearmanr(fh, eh).correlation),
                          spearman_geo=float(spearmanr(gh, eh).correlation),
                          frac_good=float(gd.mean()))

    result = dict(config=dict(seas=SEAS, seeds=SEEDS, horizons=HORIZONS, tau=TAU, sigma=SIGMA,
                              grid=[len(gx), len(gy)]),
                  pooled=summary, per_horizon=ph, rows=rows)
    json.dump(result, open(f'{out}/predictable_zone_test.json', 'w'), indent=2)

    print("\n================ POOLED (forecast horizons, HOS) ================")
    print(f"  cells={summary['n_cells']}  actually-predictable frac={summary['frac_good']:.2f}")
    print(f"  Spearman(predictor, |err|):   FIM {summary['spearman_fim']:+.3f}   GEO {summary['spearman_geo']:+.3f}   (higher=better predictor)")
    print(f"  AUC(predict where forecast fails, τ=0.25):  FIM {summary['auc_fim']:.3f}   GEO {summary['auc_geo']:.3f}")
    print(f"  AUC(τ=0.15):  FIM {summary['auc_fim_tau15']:.3f}   GEO {summary['auc_geo_tau15']:.3f}")
    print(f"  IoU(FIM-zone, GEO-zone) @matched area = {summary['iou_fim_geo']:.3f}  (1.0 ⇒ identical zones)")
    print("  per-horizon AUC (FIM / GEO):")
    for h in sorted(ph, key=float):
        r = ph[h]
        print(f"    Δt={float(h):.0f}s  FIM {r['auc_fim']:.3f} / GEO {r['auc_geo']:.3f}   (good frac {r['frac_good']:.2f})")

    winner = 'FIM' if summary['auc_fim'] > summary['auc_geo'] else 'GEO'
    margin = abs(summary['auc_fim'] - summary['auc_geo'])
    print(f"\n  => better predictor of actual forecast skill: {winner}  (AUC margin {margin:.3f})")
    print(f"  => zones {'DIFFER' if (summary['iou_fim_geo'] or 1) < 0.7 else 'COINCIDE'} (IoU {summary['iou_fim_geo']:.2f})")

    if panel_cache is not None:
        make_panel(panel_cache, out)
    print(f"\nsaved {out}/predictable_zone_test.json   ({time.time()-t0:.0f}s)")


def make_panel(pc, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gx, gy = pc['gx'], pc['gy']
    extent = [gx.min(), gx.max(), gy.min(), gy.max()]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), sharex=True, sharey=True)
    panels = [('actual |forecast−GT|/Hs', pc['err'], 0, 0.4, 'magma_r'),
              ('FIM  σ_z/Hs  (CRB)', pc['sig'], 0, 0.4, 'viridis_r'),
              ('GEO  dist-to-advected-energy [m]', pc['dgeo'], 0, pc['dgeo'].max(), 'cividis_r')]
    for a, (title, M, vmin, vmax, cmap) in zip(ax, panels):
        im = a.imshow(np.clip(M, vmin, vmax), origin='lower', extent=extent, aspect='auto',
                      cmap=cmap, vmin=vmin, vmax=vmax)
        a.contour(gx, gy, pc['err'], levels=[pc['tau']], colors='red', linewidths=1.6)
        a.set_title(title, fontsize=10); a.set_xlabel('x [m]')
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    ax[0].set_ylabel('y [m]')
    fig.suptitle(f"Predictable zone — {pc['tag']}, Δt={pc['h']:.0f}s (HOS forecast). "
                 f"red = actual e/Hs={pc['tau']} boundary. Does FIM σ or GEO distance match it?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{out}/fig_predictable_zone_test.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out}/fig_predictable_zone_test.png")


if __name__ == '__main__':
    main()
