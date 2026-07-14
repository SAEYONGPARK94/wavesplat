"""Two centerpiece figures from committed JSONs (no recompute).

fig_gradient : the misspecification gradient — d(learned-hand) whole-field
  corr across three validation stages (clean linear synthetic / HOS M=3
  nonlinear / real stereo). The thesis sentence as one plot.
fig_d20_control : real-replay closed-loop residual motion by record x arm
  x actuator delay (D-20).
"""
import json
import os
import sys

import numpy as np
from scipy import stats

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from d3_cluster_se import cluster_se
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_HAND = '#2a78d6'
C_NET = '#1baf7a'
C_INK = '#0b0b0b'
C_INK2 = '#52514e'
C_MUT = '#898781'
C_GRID = '#e1e0d9'
C_AXIS = '#c3c2b7'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.edgecolor': C_AXIS, 'axes.labelcolor': C_INK2,
    'xtick.color': C_MUT, 'ytick.color': C_MUT,
    'axes.grid': True, 'grid.color': C_GRID, 'grid.linewidth': 0.6,
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
    'savefig.facecolor': '#fcfcfb',
})
OUT = 'results/direction3'


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linewidth=0.6)
    ax.grid(axis='x', visible=False)


def stage_deltas():
    """per-unit d(learned-hand), whole-field corr, three stages."""
    # Clean linear synthetic: the same 35 seas re-rendered at 1.34 m node
    # spacing instead of 6.2 m.  The stored episodes left 14-77% of their
    # variance below the mesh's resolving power, so the published -0.041 was
    # measured through simulator clutter (E-2/E-3, 2026-07-13).  On the fine
    # mesh the stage is -0.032: the anchor holds -- learned still loses where
    # nothing is misspecified -- but a third of the old gap was the renderer.
    # Not "adequate" for all 35: BF3_5comp (6 eps) still sits at 4.7 points per
    # wavelength on its shortest significant component.  Dropping those 6 gives
    # -0.030 (t -6.1), so the verdict does not rest on them.
    # E-9 (PI directive, 2026-07-13): all three stages are quoted at ONE
    # observation budget -- the sparse one the thesis actually assumes (600 obs
    # per window = 50 returns/frame).  The real stage was already there
    # (m_cap = 600); the other two are now re-quoted to match.  Comparing stages
    # at different budgets would compare budgets, not seas.
    e3 = json.load(open(f'{OUT}/e3_clean_rejudge.json'))
    clean = [e['delta'] for e in e3['episodes']['fine_sparse']]
    hos = [r['learned'] - r['hand'] for r in
           json.load(open(f'{OUT}/a1_field_hos_sparse600.json'))]
    real = [r['field_learned'] - r['field_hand'] for r in
            json.load(open(f'{OUT}/field_wide_eval.json'))]
    return [np.array(clean), np.array(hos), np.array(real)]


def stage_clusters():
    """Cluster label per unit, per stage. E-12e (2026-07-14): the figure used an
    iid SE over units that are nested inside a handful of seas/records.

    The project's convention (d3_cluster_se.py) is "the sea law is the cluster".
    Applied honestly:
      clean : sea x n_comp                    -> G = 4
      HOS   : sea state (Hs)                  -> G = 2  (!!)  -- DEGENERATE:
              cluster-robust inference needs the between-cluster variance and
              two clusters cannot supply it (t(1) critical value = 12.7). The
              HOS leg of the gradient rests on TWO sea states, 4 seeds each.
              The figure therefore uses the SEED cluster (G=4) and the caption
              must say the across-sea generalization is NOT established here.
              Assumption-light backstop: the sign test is 8/8, p = 0.008.
      real  : record                          -> G = 5
    """
    e3 = json.load(open(f'{OUT}/e3_clean_rejudge.json'))
    clean = [f"{e['sea']}_{e['n_comp']}" for e in e3['episodes']['fine_sparse']]
    hos_j = json.load(open(f'{OUT}/a1_field_hos_sparse600.json'))
    # NOT the seed: seed 0 appears at BOTH Hs 1.2 and Hs 3.0, so it does not
    # name a sea. The sea state is the cluster, exactly as the clean stage
    # clusters on the sea law. That gives G=2 and the between-sea variance is
    # NOT ESTIMABLE -- so no CI is drawn for this stage (see fig_gradient).
    hos = [round(r['Hs'], 2) for r in hos_j]         # G=2
    hos_sea = hos
    real = [r['record'] for r in
            json.load(open(f'{OUT}/field_wide_eval.json'))]
    return [clean, hos, real], hos_sea


def fig_gradient():
    data = stage_deltas()
    STAGE_CLUSTERS, hos_sea = stage_clusters()
    labels = ['clean linear\nsynthetic\n(35 episodes,\nfine mesh 1.34 m)',
              'HOS M=3\nnonlinear\n(8 cells)',
              'real stereo\nseas\n(60 windows)']
    sub = ['no misspecification', 'bound waves,\nnonlinear evolution',
           'broadband, directional\nstructure, recon noise']
    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=200)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        x = i + rng.uniform(-0.13, 0.13, len(d))
        ax.scatter(x, d, s=26, color=C_NET if d.mean() > 0 else C_HAND,
                   alpha=0.55, edgecolors='none', zorder=2)
        m = d.mean()
        # E-12e (2026-07-14): the bars were iid SE x 1.96 on clustered data.
        # Cluster-robust SE + the t(G-1) critical value. Clusters: clean = sea
        # law, HOS = sea state (only TWO -> inference degenerate, see below),
        # real = record.
        se, n_clu = cluster_se(d, STAGE_CLUSTERS[i])
        # Do not draw a CI you cannot compute. With G=2 clusters the
        # between-cluster variance has one degree of freedom and the interval is
        # meaningless (t(1) critical value 12.7). The HOS stage has TWO sea
        # states; its bar is omitted and the stage is marked as such.
        if n_clu >= 3:
            tcrit = float(stats.t.ppf(0.975, n_clu - 1))
            ax.errorbar(i, m, yerr=tcrit * se, fmt='o', ms=9, lw=2.2,
                        capsize=6, color=C_INK, zorder=4)
        else:
            tcrit = np.nan
            ax.plot([i], [m], 'o', ms=9, color=C_INK, zorder=4)
            ax.annotate(f'no CI:\nonly {n_clu} seas',
                        (i, m), xytext=(0, 26), textcoords='offset points',
                        ha='center', fontsize=8, color='#b3261e',
                        fontweight='bold')
        ax.text(i + 0.2, m, f'{m:+.3f}', fontsize=11, color=C_INK,
                va='center', fontweight='bold')
        ax.text(i, ax.get_ylim()[0], '', fontsize=8)
        print(f'stage {i}: n={len(d)} G={n_clu} mean {m:+.4f} '
              f'CR-se {se:.4f} t {m/se:+.2f} crit {tcrit:.3f} '
              f'CI [{m-tcrit*se:+.4f}, {m+tcrit*se:+.4f}]')
    # the honest HOS number: cluster on the SEA STATE, as the clean stage does
    d_h = data[1]
    se_s, G_s = cluster_se(d_h, hos_sea)
    from scipy.stats import binomtest
    sg = binomtest((d_h > 0).sum(), len(d_h), 0.5).pvalue
    print(f'\n  HOS leg, clustered on the SEA STATE (project convention): '
          f'G={G_s} -> cluster-robust inference is DEGENERATE '
          f'(t(1) crit 12.7; CI would be [-0.080, +0.261], includes 0).')
    print(f'  The HOS stage has TWO seas (Hs 1.2, 3.0) x 4 seeds. '
          f'What survives assumption-light: sign test {(d_h > 0).sum()}/'
          f'{len(d_h)}, p={sg:.4f}. The across-sea claim is NOT established; '
          f'more HOS sea states are required.\n')
    ax.axhline(0, color=C_AXIS, lw=1.2, zorder=1)
    ax.text(-0.48, 0.006, 'learning\nwins', fontsize=8.5, color=C_NET,
            va='bottom', ha='left')
    ax.text(-0.48, -0.012, 'physics prior\nwins', fontsize=8.5,
            color=C_HAND, va='top', ha='left')
    ax.set_xlim(-0.55, 2.55)
    ax.set_xticks(range(3), labels)
    for i, t in enumerate(sub):
        ax.annotate(t, (i, 1), xycoords=('data', 'axes fraction'),
                    xytext=(0, -14), textcoords='offset points',
                    ha='center', fontsize=8, color=C_MUT)
    ax.set_ylabel('whole-field corr:  learned $-$ physics prior')
    ax.set_title('Learning wins exactly where physics is misspecified\n'
                 'same estimator pair, same observation budget '
                 '(600 obs/window = 50 returns/frame); '
                 'dots = units, bars = 95% CI',
                 fontsize=10.5, color=C_INK, loc='left')
    style(ax)
    fig.tight_layout()
    p = f'{OUT}/fig_gradient_2026-07-12.png'
    fig.savefig(p)
    plt.close(fig)
    print('wrote', p)


def fig_d20():
    d = json.load(open(f'{OUT}/d20_replay_eval.json'))
    rows = d['rows']
    recs = ['YS17', 'AA15b', 'LJ18g']
    arms = [('pct_persist', 'persistence', C_MUT),
            ('pct_hand', 'physics-prior FF', C_HAND),
            ('pct_learned', 'learned FF', C_NET)]
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.6), dpi=200,
                            sharey=True)
    for ax, tau in zip(axs, (0.5, 1.0)):
        x = np.arange(len(recs))
        for j, (key, lab, col) in enumerate(arms):
            v = [np.mean([r[key] for r in rows
                          if r['record'] == nm and r['tau'] == tau])
                 for nm in recs]
            ax.bar(x + (j - 1) * 0.26, v, 0.24, color=col,
                   label=lab if tau == 0.5 else None)
            for xi, vi in zip(x, v):
                ax.text(xi + (j - 1) * 0.26, vi + 0.02, f'{vi:.2f}',
                        ha='center', fontsize=7.5, color=C_INK2)
        ax.axhline(1.0, color=C_AXIS, lw=1.2, ls='--')
        ax.text(2.45, 1.02, 'uncompensated', fontsize=8, color=C_MUT,
                ha='right')
        ax.set_xticks(x, recs)
        ax.set_title(f'actuator delay $\\tau$ = {tau} s', fontsize=10,
                     loc='left', color=C_INK)
        style(ax)
    axs[0].set_ylabel('residual heave RMS\n(fraction of uncompensated)')
    axs[0].legend(loc='upper left', frameon=False, fontsize=9)
    g2 = d['gate_D20_2']
    fig.suptitle('Real-sea replay closed loop: the learned estimator\'s '
                 'gain transfers to control\n(learned FF vs physics-prior '
                 f'FF: paired $\\Delta$ {g2["pooled_paired_mean"]:+.3f}, '
                 't $-$4.0, 3/3 records; streaming re-estimation, '
                 'RAO$\\approx$1 heave proxy)',
                 fontsize=11, color=C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = f'{OUT}/fig_d20_control_2026-07-12.png'
    fig.savefig(p)
    plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    fig_gradient()
    fig_d20()
