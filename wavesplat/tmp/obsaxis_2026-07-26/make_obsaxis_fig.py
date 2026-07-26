"""W9 figure — observable-coordinate phase diagram (crux A + F).

Left  : the GT-free coordinate the runtime can read (bf_peak_ratio), synthetic + real.
Right : the simulator knob the runtime cannot read (n_components), synthetic only.
Same y-axis on both panels; no dual axes anywhere.

Frozen rule + verdict: tmp/obsaxis_2026-07-26/PRESTATE.md (written before computing).
Numbers: tmp/obsaxis_2026-07-26/obsaxis_stats.json (compute_obsaxis.py).
Palette: tmp/figs_opus_diag_2026-07-26/common.py.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'figs_opus_diag_2026-07-26'))
import common as C  # noqa: E402

plt = C.apply_style()
from matplotlib.lines import Line2D  # noqa: E402

D = json.load(open(os.path.join(HERE, 'obsaxis_stats.json')))
S, R, ST = D['synthetic'], D['real'], D['stats']
OUTPNG = os.path.join(C.ROOT, 'results/gifs/diag_observable_axis.png')

MARK = {'sparse': 'o', 'moderate': 's', 'crowded': '^'}   # secondary encoding, never colour alone
YLO, YHI = -0.36, 0.335
REAL_INK = '#3b2f6e'   # arena marker: deliberately outside the three regime hues
L = 0.058              # left text margin (figure fraction)


def clamp(v):
    return min(max(v, YLO + 0.012), YHI - 0.012)


def panel(ax, xs, ys, regs, xlabel, jitter=False, rng=None):
    for g in C.REGIME_ORDER:
        i = [k for k, r in enumerate(regs) if r == g]
        x = np.array([xs[k] for k in i], float)
        if jitter:
            x = x * np.exp(rng.uniform(-0.075, 0.075, size=x.size))
        ax.scatter(x, [clamp(ys[k]) for k in i], s=44, marker=MARK[g],
                   c=C.REGIME_COLOR[g], edgecolors=C.SURFACE, linewidths=0.9,
                   alpha=0.9, zorder=3)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylim(YLO, YHI)
    ax.axhline(0, color=C.SECOND, lw=1.0, zorder=2)


fig = plt.figure(figsize=(13.4, 8.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1], left=L + 0.008, right=0.985,
                      top=0.845, bottom=0.375, wspace=0.13)
axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# ---- panel A : the observable coordinate -------------------------------------------
panel(axA, [r['bfr'] for r in S], [r['delta'] for r in S], [r['regime'] for r in S],
      'bf_peak_ratio   (GT-free: 2nd / 1st beamform peak, initializer.py:292)')
axA.axvline(C.GUARD_RATIO, color=C.MUTED, ls=(0, (5, 3)), lw=1.3, zorder=2)
axA.text(C.GUARD_RATIO * 0.92, YHI - 0.015, 'guard veto  < 0.05', rotation=90,
         ha='right', va='top', fontsize=8.5, color=C.SECOND)

off = []
for r in R:
    axA.scatter([r['bfr']], [clamp(r['delta'])], s=58, marker='D',
                facecolors=(REAL_INK if r['collapse'] else 'none'), edgecolors=REAL_INK,
                linewidths=1.4, zorder=5)
    if r['delta'] < YLO:
        off.append(r)
        axA.annotate('', xy=(r['bfr'], YLO + 0.004), xytext=(r['bfr'], YLO + 0.034),
                     arrowprops=dict(arrowstyle='-|>', color=REAL_INK, lw=1.3), zorder=5)
axA.text(0.015, 0.028,
         'real windows off scale below:  ' +
         ',  '.join(f"{r['record']}_w{r['win']}  Δ = {r['delta']:+.2f}"
                    for r in sorted(off, key=lambda z: z['delta'])),
         transform=axA.transAxes, ha='left', va='bottom', fontsize=8.5, color=REAL_INK)

# ---- panel B : the simulator knob ---------------------------------------------------
panel(axB, [float(r['comp']) for r in S], [r['delta'] for r in S], [r['regime'] for r in S],
      'n_components   (simulator knob — not readable at runtime)',
      jitter=True, rng=np.random.default_rng(0))
axB.set_xticks([1, 2, 5, 15, 30])
axB.set_xticklabels(['1', '2', '5', '15', '30'])
axB.tick_params(labelleft=False)
axB.text(0.5, 0.035, 'the 30 real windows cannot be drawn on this axis —\n'
         'that is crux F, not an omission', transform=axB.transAxes, ha='center',
         va='bottom', fontsize=9, color=C.SECOND, style='italic')

# ---- rho boxes ----------------------------------------------------------------------
box = dict(boxstyle='round,pad=0.42', facecolor=C.SURFACE, edgecolor=C.GRID, linewidth=1.0)
axA.text(0.017, 0.975, f"Spearman |ρ| = {ST['primary']['bf']['abs_rho']:.3f}\n"
         f"(ρ = {ST['primary']['bf']['rho']:+.3f},  p = {ST['primary']['bf']['p']:.1e},  n = 100)",
         transform=axA.transAxes, va='top', ha='left', fontsize=10.5, color=C.INK, bbox=box)
axB.text(0.017, 0.975, f"Spearman |ρ| = {ST['primary']['ncomp']['abs_rho']:.3f}\n"
         f"(ρ = {ST['primary']['ncomp']['rho']:+.3f},  p = {ST['primary']['ncomp']['p']:.1e},  n = 100)",
         transform=axB.transAxes, va='top', ha='left', fontsize=10.5, color=C.INK, bbox=box)

axA.set_ylabel('Δ nRMSE   =   learned init   −   analytic init      (per episode)')
axA.text(0.0093, -0.010, 'learned better ↓', fontsize=9, color=C.SECOND, va='top')
axA.text(0.0093, 0.010, 'analytic better ↑', fontsize=9, color=C.SECOND, va='bottom')

# ---- legend -------------------------------------------------------------------------
handles = [Line2D([], [], ls='', marker=MARK[g], mfc=C.REGIME_COLOR[g], mec=C.SURFACE,
                  ms=8, label=C.REGIME_LABEL[g]) for g in C.REGIME_ORDER]
handles += [
    Line2D([], [], ls='', marker='D', mfc='none', mec=REAL_INK, mew=1.4, ms=8,
           label='real stereo window, normal (n=27)'),
    Line2D([], [], ls='', marker='D', mfc=REAL_INK, mec=REAL_INK, ms=8,
           label='real stereo window, analytic collapse (n=3)')]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.52, 0.325), ncol=3,
           fontsize=9, handletextpad=0.4, columnspacing=2.2, labelcolor=C.SECOND)

# ---- title / protocol ---------------------------------------------------------------
dec = ST['decision']
fig.text(L, 0.975, 'The regime boundary is much sharper on the simulator knob '
         'than on the observable', fontsize=15.5, color=C.INK, va='top')
fig.text(L, 0.938,
         'Δ = median-of-5-lane nRMSE(learned set-proposal init) − median-of-5-lane '
         'nRMSE(analytic v2_konly);  100 finemesh episodes, lanes (42,0)(1,1)(2,2)(3,3)(4,4).\n'
         'Real overlay: learned_now − hand_now on the 30 Bergamasco stereo windows '
         '(nowcast map nRMSE, /gt_std).  Zero new estimator compute — re-derived from the shipped JSONs.\n'
         'Pre-stated rule, frozen before computing (tmp/obsaxis_2026-07-26/PRESTATE.md): '
         'if |ρ_bf_peak_ratio| ≥ |ρ_n_components| − 0.05 the boundary is not a relabeling\n'
         'of the answer key;  if n_components dominates, P3 is right and Ch.5 must say so in those words.',
         fontsize=9.2, color=C.SECOND, va='top', linespacing=1.55)

# ---- verdict banner -----------------------------------------------------------------
vcol = C.ORANGE if dec['verdict'] == 'FOR_P3' else C.AQUA
fig.patches.append(plt.Rectangle((L, 0.170), 0.985 - L, 0.072, transform=fig.transFigure,
                                 facecolor=vcol, alpha=0.10, edgecolor=vcol, linewidth=1.0,
                                 zorder=0))
fig.text(L + 0.012, 0.223, 'VERDICT (pre-stated):', fontsize=11, color=C.INK,
         va='center', fontweight='bold')
fig.text(L + 0.163, 0.223,
         f"|ρ_bf| = {dec['abs_rho_bf']:.3f}   <   |ρ_ncomp| − 0.05 = "
         f"{dec['abs_rho_ncomp'] - dec['tol']:.3f}    ⇒    n_components dominates.",
         fontsize=11, color=C.INK, va='center')
fig.text(L + 0.012, 0.191,
         'The phase diagram is drawn on a coordinate the runtime cannot read — '
         'P3 is right, and Ch.5 says so in those words.',
         fontsize=11, color=C.INK, va='center')

# ---- honesty note -------------------------------------------------------------------
fig.text(L, 0.147,
         'Honesty (pre-committed before the numbers): the 30 real windows cluster at '
         'bf_peak_ratio 0.51–0.95 (IQR), a band where only 6 of 100 synthetic episodes live and\n'
         'none of them is sparse — the synthetic grid barely populates it.  Inside that band '
         '4 of 6 synthetic episodes favour the learned init while 20 of 29 real windows favour\n'
         'the analytic one.  Real bf_peak_ratio spans 0.11–1.00 and 0/30 fall below the 0.05 '
         'guard, so the guard never fires on real data.  The axis IS measurable outside the\n'
         'simulator (crux F answered); it was only *varied* inside it.  Real Δ uses the /gt_std '
         'nowcast-map normaliser and synthetic Δ uses /Hs, so the real points are placed for\n'
         'coordinate and sign — not for magnitude comparison against the synthetic cloud.',
         fontsize=8.4, color=C.MUTED, va='top', linespacing=1.6)

fig.savefig(OUTPNG)
print('wrote', OUTPNG)
print(f"  |rho| bf={dec['abs_rho_bf']:.4f}  ncomp={dec['abs_rho_ncomp']:.4f}  "
      f"verdict={dec['verdict']}")
