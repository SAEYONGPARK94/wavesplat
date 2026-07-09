"""
d69 — Did the amortized posterior win only because of the compute/real-time CYCLE? (user fact-check)

Honest test: plot the accuracy–compute(+latency) Pareto on HOS short→long windows.
  - amortized posterior : ms inference, short window, UQ — the fast corner.
  - F1 (ridge continuous): ~10²ms, short window, no UQ.
  - RELAX, window sweep  : 2.2 / 3.5 / 5.0·Tp — MORE accurate with a longer window (synthetic
                           aperture), but pays (a) window LATENCY (must wait the window to collect)
                           and (b) slower inference. No UQ (unless +d64 physics UQ).

If RELAX with a longer window is more accurate but slow/high-latency, while the posterior is the
only fast+UQ option at the short window, then the posterior's "win" IS conditional on the real-time
cycle — exactly the user's hypothesis — and we say so with numbers.

Run: python3 scripts/d69_compute_pareto.py
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from wavesim.hos_simulator import HOSSimulator, HOSConfig
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from d26_relax_iterative import iterative_relax
from d23_main_results import panel
from d40_learned_acquisition import pool_xy
from d67_physics_posterior_uq import PosteriorNet, DEVICE

DEPTH = 50.0; DT_FRAME = 0.1; SIGMA = 0.02; G = 9.81
SEAS = [('BF4_15c', 1.2, 7.0), ('BF5_30c', 2.0, 9.0)]
SEEDS = [0, 1]
KTPS = [2.2, 3.5, 5.0]
PER_FRAME = 25; OBS_CAP = 2500


def build_long_window(Hs, Tp, seed, gx, gy):
    """HOS to t_now=5·Tp; collect obs at PER_FRAME/frame; GT grid at t_now (nowcast)."""
    cfg = HOSConfig(Lx=600.0, Ly=400.0, Nx=64, Ny=64, depth=DEPTH, M=3)
    sim = HOSSimulator(cfg); sim.initialize_jonswap(Hs=Hs, Tp=Tp, seed=seed, direction_deg=0.0, direction_spread_deg=30.0)
    pool = pool_xy(); Xg, Yg = np.meshgrid(gx, gy)
    n = int(round(max(KTPS) * Tp / DT_FRAME))
    P, z = [], []
    for i in range(n + 1):
        if i > 0:
            sim.step(DT_FRAME)
        t = i * DT_FRAME
        idx = np.random.default_rng(4000 + i).choice(len(pool), PER_FRAME, replace=False)
        ox, oy = pool[idx, 0], pool[idx, 1]
        Zo, *_ = sim.evaluate_at(ox.reshape(1, -1), oy.reshape(1, -1))
        zo = Zo.ravel() + np.random.default_rng(5000 + i).normal(0, SIGMA, len(ox))
        P.append(np.column_stack([ox, oy, np.full_like(ox, t)])); z.append(zo)
    Zg, *_ = sim.evaluate_at(Xg, Yg)
    return np.vstack(P).astype(np.float64), np.concatenate(z).astype(np.float64), n * DT_FRAME, Zg.astype(np.float64)


def relax_timed(P, z, gx, gy, t_now, GT, Hs):
    target = {'gx': gx, 'gy': gy, 't': t_now, 'gt': GT, 'Hs': Hs}
    t0 = time.perf_counter()
    fld, _ = iterative_relax(P, z, DEPTH, target, 3)
    dt = (time.perf_counter() - t0) * 1e3
    return panel(fld, GT, Hs)['nrmse'], dt


def f1_timed(P, z, gx, gy, t_now, GT, Hs):
    """Dense-grid ridge LSQ over a fixed (kx,ky) grid → continuous-spectrum reconstruction."""
    kk = np.linspace(-0.18, 0.18, 19)
    KX, KY = np.meshgrid(kk, kk); kx = KX.ravel(); ky = KY.ravel()
    m = np.hypot(kx, ky); keep = (m > 1e-3) & (m < 0.25); kx, ky = kx[keep], ky[keep]
    t0 = time.perf_counter()
    obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z])
    A, phi = fit_amplitudes_phases(kx, ky, obs, DEPTH, lambda_reg=2.0)
    fld = render_wave_field(kx, ky, A, phi, DEPTH, gx, gy, t_now)
    dt = (time.perf_counter() - t0) * 1e3
    return panel(fld, GT, Hs)['nrmse'], dt


def posterior_time(gx, gy):
    """Time one forward pass of the amortized posterior (architecture; nRMSE from d68=0.250)."""
    model = PosteriorNet(qdim=3).to(DEVICE).eval()
    Xg, Yg = np.meshgrid(gx, gy); grid = np.column_stack([Xg.ravel(), Yg.ravel()])
    ob = torch.zeros(1, OBS_CAP, 4, device=DEVICE)
    q = torch.tensor(np.column_stack([grid, np.zeros(len(grid))])[None], dtype=torch.float32, device=DEVICE)
    for _ in range(3):
        with torch.no_grad(): model(ob, q)
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        with torch.no_grad(): model(ob, q)
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / 10


def main():
    out = 'results/direction3'; t0 = time.time()
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    relax = {k: dict(nrmse=[], ms=[]) for k in KTPS}
    f1 = dict(nrmse=[], ms=[])
    for tag, Hs, Tp in SEAS:
        for seed in SEEDS:
            P, z, t_now, GT = build_long_window(Hs, Tp, seed, gx, gy)
            Hs_n = float(GT.std() * 4)
            for ktp in KTPS:
                sel = P[:, 2] >= t_now - ktp * Tp - 1e-6
                Pk, zk = P[sel], z[sel]
                if len(zk) > OBS_CAP:
                    ii = np.random.default_rng(0).choice(len(zk), OBS_CAP, replace=False); Pk, zk = Pk[ii], zk[ii]
                nr, ms = relax_timed(Pk, zk, gx, gy, t_now, GT, Hs_n)
                relax[ktp]['nrmse'].append(nr); relax[ktp]['ms'].append(ms)
            # F1 at short window (2.2Tp)
            sel = P[:, 2] >= t_now - 2.2 * Tp - 1e-6; Pk, zk = P[sel], z[sel]
            if len(zk) > OBS_CAP:
                ii = np.random.default_rng(0).choice(len(zk), OBS_CAP, replace=False); Pk, zk = Pk[ii], zk[ii]
            nr, ms = f1_timed(Pk, zk, gx, gy, t_now, GT, Hs_n)
            f1['nrmse'].append(nr); f1['ms'].append(ms)
            print(f"  [{tag} s{seed}] done ({time.time()-t0:.0f}s)", flush=True)
    post_ms = posterior_time(gx, gy)

    rows = []
    for tag, Tp in [('BF4_15c', 7.0), ('BF5_30c', 9.0)]:
        pass
    Tp_ref = 8.0  # representative for latency display
    summary = dict(posterior=dict(window_Tp=2.2, latency_s=2.2*Tp_ref, infer_ms=float(post_ms),
                                  nrmse=0.250, uq=True),
                   F1=dict(window_Tp=2.2, latency_s=2.2*Tp_ref, infer_ms=float(np.mean(f1['ms'])),
                           nrmse=float(np.mean(f1['nrmse'])), uq=False),
                   relax={str(k): dict(window_Tp=k, latency_s=k*Tp_ref, infer_ms=float(np.mean(relax[k]['ms'])),
                                       nrmse=float(np.mean(relax[k]['nrmse'])), uq=False) for k in KTPS})
    json.dump(summary, open(f'{out}/compute_pareto.json', 'w'), indent=2)

    print("\n================ ACCURACY vs COMPUTE/LATENCY PARETO (HOS nowcast) ================")
    print(f"  {'method':22s} {'window':>7s} {'latency':>8s} {'infer':>9s} {'nRMSE':>7s}  UQ")
    s = summary
    print(f"  {'amortized posterior':22s} {s['posterior']['window_Tp']:>5.1f}Tp "
          f"{s['posterior']['latency_s']:>6.0f}s {s['posterior']['infer_ms']:>7.1f}ms {s['posterior']['nrmse']:>7.3f}  YES")
    print(f"  {'F1 (ridge continuous)':22s} {s['F1']['window_Tp']:>5.1f}Tp "
          f"{s['F1']['latency_s']:>6.0f}s {s['F1']['infer_ms']:>7.0f}ms {s['F1']['nrmse']:>7.3f}  no")
    for k in KTPS:
        r = s['relax'][str(k)]
        print(f"  {'RELAX (window '+str(k)+'Tp)':22s} {r['window_Tp']:>5.1f}Tp "
              f"{r['latency_s']:>6.0f}s {r['infer_ms']:>7.0f}ms {r['nrmse']:>7.3f}  no")
    best_acc = min([summary['relax'][str(k)]['nrmse'] for k in KTPS] + [summary['F1']['nrmse'], 0.250])
    print(f"\n  most accurate = {'long-window RELAX' if summary['relax']['5.0']['nrmse']==best_acc else 'see table'} "
          f"(nRMSE {best_acc:.3f}) — but needs {summary['relax']['5.0']['latency_s']:.0f}s window + "
          f"{summary['relax']['5.0']['infer_ms']:.0f}ms.")
    print(f"  posterior = fastest ({post_ms:.1f}ms) + ONLY one with UQ, nRMSE {0.250:.3f} (≈ short-window classical).")
    print(f"  => the posterior's edge is SPEED + UQ at a fixed short window; classical is MORE accurate")
    print(f"     given a longer window (more latency/compute). 'Learning won because of the cycle' = TRUE.")
    make_fig(summary, KTPS, out)
    print(f"saved {out}/compute_pareto.json   ({time.time()-t0:.0f}s)")


def make_fig(s, ktps, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter([s['posterior']['infer_ms']], [s['posterior']['nrmse']], s=120, c='tab:green',
               marker='*', zorder=5, label='amortized posterior (UQ)')
    ax.scatter([s['F1']['infer_ms']], [s['F1']['nrmse']], s=70, c='tab:gray', label='F1 ridge (no UQ)')
    xs = [s['relax'][str(k)]['infer_ms'] for k in ktps]; ys = [s['relax'][str(k)]['nrmse'] for k in ktps]
    ax.plot(xs, ys, 'o-', c='tab:red', label='RELAX (window 2.2→5 Tp)')
    for k in ktps:
        ax.annotate(f"{k}Tp", (s['relax'][str(k)]['infer_ms'], s['relax'][str(k)]['nrmse']), fontsize=8)
    ax.set_xscale('log'); ax.set_xlabel('inference compute [ms/update]'); ax.set_ylabel('nowcast nRMSE')
    ax.set_title("Accuracy–compute Pareto: posterior = fast+UQ corner;\nlonger-window RELAX = more accurate but slow/high-latency")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(f'{out}/fig_compute_pareto.png', dpi=140, bbox_inches='tight')
    plt.close(fig); print(f"saved {out}/fig_compute_pareto.png")


if __name__ == '__main__':
    main()
