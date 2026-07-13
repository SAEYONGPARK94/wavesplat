"""E-7: see it. Truth, what the sensor returned, and what we estimated.

Two figures for the mesh-artifact arc:

fig_mesh_wavefield : per sea, three panels -- the true surface at the scoring
  instant with the LiDAR returns drawn on it, the estimate from those returns,
  and the return error against analytic truth.  Rows contrast the same BF2 sea
  rendered on the coarse mesh (returns are facet noise; the estimate is
  garbage) and on the fine mesh (returns are the wave; the estimate comes
  back), with a BF4 row as the control that was never badly broken.

fig_mesh_performance : the four measurements of the arc on one page --
  what the shipped mesh could not represent (E-2), the paired shift it put
  into the central gradient (E-3), the BF2 rescore with its n=5 spread
  (E-3b), and BF2's remaining range-dependent weakness (E-6).
"""
import json
import os
import sys

import h5py
import numpy as np
import torch

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import matplotlib                                     # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.gridspec import GridSpec              # noqa: E402

from d3_g0b_selection_aware import BAND, DEV, DT      # noqa: E402
from d3_gpu_mmse import build_bank, design, mmse_gpu  # noqa: E402
from wavesim.gt_generator import evaluate_field       # noqa: E402
from wavesim.wave_generator import WaveComponent      # noqa: E402

C_HAND = '#2a78d6'
C_NET = '#1baf7a'
C_BAD = '#d1495b'
C_INK = '#0b0b0b'
C_INK2 = '#52514e'
C_MUT = '#898781'
C_GRID = '#e1e0d9'
C_AXIS = '#c3c2b7'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.edgecolor': C_AXIS, 'axes.labelcolor': C_INK2,
    'xtick.color': C_MUT, 'ytick.color': C_MUT,
    'axes.grid': False,
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
    'savefig.facecolor': '#fcfcfb',
})
OUT = 'results/direction3'


def estimate(path, rng):
    """Run the deployment recipe on one episode; return truth, obs, estimate."""
    with h5py.File(path, 'r') as f:
        comp = f['config/wave_components'][:]
        ts = f['frames/timestamps'][:]
        pts = f['frames/lidar_points']
        gx, gy = f['frames/gt_grid_x'][:], f['frames/gt_grid_y'][:]
        gz = f['frames/gt_wave_field']

        kp = float(np.hypot(comp[:, 2], comp[:, 3])[np.argmax(comp[:, 0])])
        obs_i = list(range(50, 249, 18))
        t_end = float(ts[obs_i[-1]])
        P, zv = [], []
        for i in obs_i:
            p = pts[i]
            P.append(np.column_stack([p[:, 0], p[:, 1], np.full(len(p), ts[i])]))
            zv.append(p[:, 2])
        P, zv = np.vstack(P), np.concatenate(zv)
        rr = np.hypot(P[:, 0], P[:, 1])
        edges = np.linspace(rr.min(), rr.max() + 1e-6, 13)
        b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
        take = []
        for j in range(12):
            idx = np.where(b == j)[0]
            k = min(len(idx), 167)
            if k:
                take.append(rng.choice(idx, k, replace=False))
        sel = np.concatenate(take)
        Ps, zs = P[sel], zv[sel]
        gt_end = gz[obs_i[-1]][:]
        last = pts[obs_i[-1]]                 # returns of the scoring frame

    t32 = torch.tensor(Ps, device=DEV, dtype=torch.float32)
    z32 = torch.tensor(zs, device=DEV, dtype=torch.float32)
    _, mdl = mmse_gpu(t32, z32, 50.0, t32[:64], 0.05, kp, seed=0,
                      return_model=True, no_em=True, pexp=1.0, smult=10.0)
    gam, s2 = mdl['gam'].to(DT), float(mdl['s2'])
    kxb, kyb, om = build_bank(kp, BAND, kp / 10.0, 50.0, DEV, torch.float32)
    kxb, kyb, om = kxb.to(DT), kyb.to(DT), om.to(DT)
    Phi = design(torch.tensor(Ps, device=DEV, dtype=DT), kxb, kyb, om)
    A0 = Phi.T @ Phi / s2 + torch.diag(1.0 / gam)
    a = torch.linalg.solve(A0, Phi.T @ torch.tensor(zs, device=DEV, dtype=DT) / s2)

    X, Y = np.meshgrid(gx, gy)
    Q = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, t_end)])
    C = design(torch.tensor(Q, device=DEV, dtype=DT), kxb, kyb, om)
    est = (C @ a).cpu().numpy().reshape(X.shape)

    R = np.hypot(X, Y)
    m = (R >= 15) & (R <= R.max())
    rho = float(np.corrcoef(est[m], gt_end[m])[0, 1])
    mn = (R >= 15) & (R <= 75)
    rho_near = float(np.corrcoef(est[mn], gt_end[mn])[0, 1])

    # what the sensor actually reported vs the analytic surface, same points
    comps = [WaveComponent(A=r[0], omega=r[1], kx=r[2], ky=r[3], phi=r[4],
                           Q=r[5]) for r in comp]
    z_true, _, _, _ = evaluate_field(comps, last[:, 0], last[:, 1], t_end)
    op_err = float(np.std(last[:, 2] - z_true))
    lam = float(2 * np.pi / kp)

    return dict(X=X, Y=Y, truth=gt_end, est=est, rho=rho,
                rho_near=rho_near, lam=lam,
                obs_xy=last[:, :2], obs_z=last[:, 2], obs_err=last[:, 2] - z_true,
                op_err=op_err, t_end=t_end)


def fig_wavefield():
    rng = np.random.default_rng(11)
    rows = [
        ('BF2, mesh 6.2 m', 'data/episode_BF2_1comp_seed0_fixedT.h5', C_BAD),
        ('BF2, mesh 1.34 m', 'tmp/finemesh_clean/episode_BF2_1comp_seed0_fixedT.h5', C_NET),
        ('BF4, mesh 1.34 m', 'tmp/finemesh_clean/episode_BF4_15comp_seed0_fixedT.h5', C_HAND),
    ]
    fig = plt.figure(figsize=(12.4, 9.4), dpi=190)
    gs = GridSpec(3, 3, figure=fig, hspace=0.30, wspace=0.22,
                  left=0.06, right=0.97, top=0.88, bottom=0.06)

    for r, (label, path, col) in enumerate(rows):
        e = estimate(path, np.random.default_rng(11))
        v = float(np.abs(e['truth']).max())

        ax = fig.add_subplot(gs[r, 0])
        ax.pcolormesh(e['X'], e['Y'], e['truth'], cmap='RdBu_r', vmin=-v, vmax=v,
                      shading='auto', rasterized=True)
        ax.scatter(e['obs_xy'][:, 0], e['obs_xy'][:, 1], s=0.35, c='k',
                   alpha=0.35, linewidths=0, rasterized=True)
        ax.set_title(f'true surface + LiDAR returns\nλ = {e["lam"]:.1f} m',
                     fontsize=9, color=C_INK2)
        ax.set_ylabel(label, fontsize=11, color=col, fontweight='bold')

        ax2 = fig.add_subplot(gs[r, 1])
        ax2.pcolormesh(e['X'], e['Y'], e['est'], cmap='RdBu_r', vmin=-v, vmax=v,
                       shading='auto', rasterized=True)
        ax2.set_title('estimate from those returns\n'
                      f'near-field ρ = {e["rho_near"]:+.3f}   |   '
                      f'wide-fan ρ = {e["rho"]:+.3f}',
                      fontsize=9, color=col, fontweight='bold')

        ax3 = fig.add_subplot(gs[r, 2])
        sc = ax3.scatter(e['obs_xy'][:, 0], e['obs_xy'][:, 1], s=1.6,
                         c=e['obs_err'], cmap='PuOr', vmin=-0.15, vmax=0.15,
                         linewidths=0, rasterized=True)
        ax3.set_title('return error vs analytic truth\n'
                      f'std = {e["op_err"]:.4f} m  (noise model 0.02)',
                      fontsize=9, color=C_INK2)
        plt.colorbar(sc, ax=ax3, fraction=0.04, pad=0.02).set_label('m', size=8)

        for a in (ax, ax2, ax3):
            a.set_xlim(0, 250)
            a.set_ylim(-100, 100)
            a.set_aspect('equal')
            a.tick_params(labelsize=7)
            if r == 2:
                a.set_xlabel('x [m]', fontsize=8)

    fig.suptitle('What the coarse mesh destroyed, and what came back\n'
                 'same sea, same estimator — only the renderer changed '
                 '(rows 1 vs 2)',
                 fontsize=13, color=C_INK, y=0.965, fontweight='bold')
    p = f'{OUT}/fig_mesh_wavefield_2026-07-13.png'
    fig.savefig(p, bbox_inches='tight')
    print('wrote', p)


def fig_performance():
    audit = json.load(open(f'{OUT}/e2_mesh_audit.json'))
    e3 = json.load(open(f'{OUT}/e3_clean_rejudge.json'))
    e3b = json.load(open(f'{OUT}/e3b_bf2_rescore.json'))
    e6 = json.load(open(f'{OUT}/e6_sampling_test.json'))

    fig = plt.figure(figsize=(13.0, 8.6), dpi=190)
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.26,
                  left=0.07, right=0.97, top=0.87, bottom=0.08)

    # (a) what the shipped mesh could not represent
    ax = fig.add_subplot(gs[0, 0])
    gsel = [g for g in audit['groups'] if g['sea'] in ('BF2', 'BF3', 'BF4', 'BF5')]
    names = [f"{g['sea']}\n{g['n_components']}c" for g in gsel]
    vals = [100 * g['var_under_resolved'] for g in gsel]
    cols = [C_BAD if g['verdict'] == 'INADEQUATE' else '#e8a33d' for g in gsel]
    ax.bar(range(len(gsel)), vals, color=cols, width=0.72)
    ax.axhline(5, color=C_INK2, lw=1, ls='--')
    ax.text(-0.4, 8.5, 'adequate ≤ 5%', fontsize=7, color=C_INK2, ha='left')
    ax.set_ylim(0, 112)
    ax.set_xticks(range(len(gsel)))
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel('wave variance the mesh\ncannot represent  [%]', fontsize=9)
    ax.set_title('(a) the shipped mesh (6.2 m) vs the seas it rendered\n'
                 'every benchmark sea state is at best degraded',
                 fontsize=10, color=C_INK, fontweight='bold', loc='left')
    ax.grid(axis='y', color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    # (b) dose-response: the worse the render, the more it moved the verdict
    ax = fig.add_subplot(gs[0, 1])
    seas = ['BF3', 'BF4', 'BF5']
    shift = [e3['fine']['by_sea'][s]['delta'] - e3['coarse']['by_sea'][s]['delta']
             for s in seas]
    ptsl = [next(g['pts_per_lambda_peak'] for g in audit['groups']
                 if g['sea'] == s and g['n_components'] in (5, 15)) for s in seas]
    ax.scatter(ptsl, shift, s=90, color=C_NET, zorder=3)
    for s, x, y in zip(seas, ptsl, shift):
        ax.annotate(s, (x, y), textcoords='offset points', xytext=(8, 4),
                    fontsize=9, color=C_INK2)
    ax.axhline(0, color=C_AXIS, lw=1)
    ax.set_xlabel('mesh points per peak wavelength (coarse render)', fontsize=9)
    ax.set_ylabel('Δ moved by fixing the mesh', fontsize=9)
    ax.set_title('(b) the shift scales with the damage (3 points, corroborating)\n'
                 'well-rendered BF5 barely moved; starved BF3 moved most',
                 fontsize=10, color=C_INK, fontweight='bold', loc='left')
    ax.grid(color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    # (c) BF2 rescore, with the n=5 spread that forbids over-claiming
    ax = fig.add_subplot(gs[1, 0])
    mets = ['ann', 'fan', 'enc']
    cs = [e3b['coarse'][m] for m in mets]
    fs = [e3b['fine'][m] for m in mets]
    x = np.arange(3)
    ax.bar(x - 0.19, cs, 0.36, color=C_MUT, label='mesh 6.2 m')
    ax.bar(x + 0.19, fs, 0.36, color=C_NET, label='mesh 1.34 m')
    for i, m in enumerate(mets):
        d = np.array([f[m] for f in e3b['fine']['episodes']]) - \
            np.array([c[m] for c in e3b['coarse']['episodes']])
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        ax.scatter(np.full(5, i + 0.19),
                   [f[m] for f in e3b['fine']['episodes']],
                   s=14, color=C_INK, zorder=3, alpha=0.75)
        ax.text(i, max(cs[i], fs[i]) + 0.09, f't={t:.1f}', ha='center',
                fontsize=8, color=(C_INK if abs(t) > 2.5 else C_BAD),
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['near field\n(ann)', 'wide fan', 'encounter'],
                       fontsize=8.5)
    ax.set_ylabel('ρ', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.set_title('(c) BF2 rescored — only the near field is a real recovery\n'
                 'dots = 5 seeds; fan and encounter are not significant',
                 fontsize=10, color=C_INK, fontweight='bold', loc='left')
    ax.grid(axis='y', color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    # (d) the sampling story does not survive its own seeds
    ax = fig.add_subplot(gs[1, 1])
    bins = e6['P1']['BF2']['bins']
    ctr = [0.5 * (b[0] + b[1]) for b in bins]
    for r in e6['P1']['BF4']['seeds']:
        ax.plot(ctr, r['rho'], '-', color=C_HAND, lw=1, alpha=0.5)
    for r in e6['P1']['BF2']['seeds']:
        flat = min(r['rho']) > 0.6
        ax.plot(ctr, r['rho'], '-o', ms=3, lw=1.4,
                color=(C_NET if flat else C_BAD), alpha=0.85)
        if flat:
            ax.annotate(f"seed{r['seed']}", (ctr[-1], r['rho'][-1]),
                        textcoords='offset points', xytext=(4, -2),
                        fontsize=7, color=C_NET)
    ax.plot(ctr, e6['P1']['BF2']['rho'], 'o-', color=C_INK, lw=2.4,
            label='BF2 mean (λ 8 m)', zorder=5)
    ax.plot(ctr, e6['P1']['BF4']['rho'], '-', color=C_HAND, lw=2.4,
            label='BF4 control, 5 seeds (λ 65 m)', zorder=4)
    ax.set_ylim(-0.3, 1.05)
    ax.axhline(0, color=C_AXIS, lw=0.8)
    ax.set_xlabel('range from sensor [m]', fontsize=9)
    ax.set_ylabel('ρ  (fine mesh, 2000 obs)', fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc='lower left')
    ax.set_title('(d) the sampling explanation FAILS its own seeds\n'
                 'only 1/5 BF2 seeds decays; two (green) stay flat at the same '
                 'λ — so it is identification, not sampling',
                 fontsize=10, color=C_INK, fontweight='bold', loc='left')
    ax.grid(color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('The mesh artifact: what it broke, how much it moved, '
                 'and what it did not explain',
                 fontsize=14, color=C_INK, y=0.955, fontweight='bold')
    p = f'{OUT}/fig_mesh_performance_2026-07-13.png'
    fig.savefig(p, bbox_inches='tight')
    print('wrote', p)


if __name__ == '__main__':
    fig_wavefield()
    fig_performance()
