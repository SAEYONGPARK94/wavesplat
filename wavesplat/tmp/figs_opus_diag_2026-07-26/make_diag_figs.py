"""diag_* synthetic diagnostic figures for the unified estimator (2026-07-26).

  diag_scatter_v2_vs_auto.png : per-episode pure-v2 vs auto+guard (100 pts)
  diag_routing.png            : router decision surface (n_selected) + guard signal (bf ratio)
  diag_oracle_gap.png         : where the 0.0875 -> 0.0818 oracle gap lives
  diag_seed_variance.png      : 5-seed spread before vs after the tracker phase fix

Run from the wavesplat/ package root:
    python3 tmp/figs_opus_diag_2026-07-26/make_diag_figs.py
Reconstruction is verified against tmp/unified_2026-07-26/switch_results.json and
tmp/unified_guard_2026-07-26/task1_results.json (see verify() at the bottom).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (AQUA, BLUE, GRID, GUARD_RATIO, INK, MUTED, ORANGE, OUT,
                    REGIME_COLOR, REGIME_LABEL, REGIME_ORDER, SECOND, SURFACE,
                    SWITCH_JSON, TASK1_JSON, THRESH_ALT, THRESH_DEPLOYED,
                    apply_style, assemble, mean_over)

plt = apply_style()

PROTO = ('protocol: per-episode value = median over 5 lanes (est_seed, diff_seed); '
         '100 finemesh episodes; tracker phase fix ON')
BOX = dict(facecolor=SURFACE, edgecolor='none', alpha=0.88, pad=1.8)
ROWS, FLIPS = assemble(THRESH_DEPLOYED, GUARD_RATIO)
EPS = sorted(ROWS)
SUB = {r: [e for e in EPS if ROWS[e]['regime'] == r] for r in REGIME_ORDER}


def sup(fig, title, subtitle):
    fig.text(0.012, 0.975, title, ha='left', va='top', fontsize=14.5,
             color=INK, fontweight='bold')
    fig.text(0.012, 0.928, subtitle, ha='left', va='top', fontsize=9.2, color=SECOND)


def foot(fig, text):
    fig.text(0.012, 0.012, text, ha='left', va='bottom', fontsize=8.0, color=MUTED)


def regime_legend(ax, loc='lower right', title=None, **kw):
    h = [plt.Line2D([], [], marker='o', ls='none', ms=7.5, mfc=REGIME_COLOR[r],
                    mec=SURFACE, mew=1.2, label=REGIME_LABEL[r]) for r in REGIME_ORDER]
    lg = ax.legend(handles=h, loc=loc, fontsize=8.6, title=title, frameon=True,
                   facecolor=SURFACE, edgecolor='none', framealpha=0.88,
                   title_fontsize=8.6, labelcolor=SECOND, handletextpad=0.5, **kw)
    lg.set_zorder(10)


# ---------------------------------------------------------------- figure 1
def fig_scatter():
    fig = plt.figure(figsize=(12.2, 7.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1.0], left=0.062, right=0.985,
                          top=0.795, bottom=0.145, wspace=0.24)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    lo, hi = 0.0028, 0.45
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls='--', zorder=1)
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=SECOND, alpha=0.05, zorder=0)

    counts = {}
    for r in REGIME_ORDER:
        x = np.array([ROWS[e]['pure_v2'] for e in SUB[r]])
        y = np.array([ROWS[e]['guard'] for e in SUB[r]])
        ax.plot(x, y, 'o', ms=8, mfc=REGIME_COLOR[r], mec=SURFACE, mew=1.3,
                ls='none', zorder=3, alpha=0.92)
        w = int((y < x - 1e-12).sum())
        ls_ = int((y > x + 1e-12).sum())
        counts[r] = (w, len(x) - w - ls_, ls_)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('pure analytic init (v2_konly) nRMSE  [log]', labelpad=8)
    ax.set_ylabel("unified  init_method='auto' + beamform guard  nRMSE  [log]")
    ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks]); ax.set_yticklabels([str(t) for t in ticks])
    ax.grid(True, which='major', color=GRID, lw=0.8)
    ax.grid(False, which='minor')

    tw = sum(c[0] for c in counts.values())
    tt = sum(c[1] for c in counts.values())
    tl = sum(c[2] for c in counts.values())
    ax.text(0.97, 0.10, 'unified better (below diagonal)', fontsize=9.4, color=SECOND,
            ha='right', va='bottom', fontweight='bold', transform=ax.transAxes)
    ax.text(0.03, 0.965, 'unified worse (above diagonal)', fontsize=9.4, color=SECOND,
            ha='left', va='top', fontweight='bold', transform=ax.transAxes)

    # direct labels: the one surviving large misroute + the largest crowded win
    worst = max(EPS, key=lambda e: ROWS[e]['guard'] - ROWS[e]['pure_v2'])
    ax.annotate(f"{ROWS[worst]['short']} — the one big surviving misroute\n"
                f"guard did not fire (bf ratio {ROWS[worst]['bfr']:.2f} > {GUARD_RATIO})",
                xy=(ROWS[worst]['pure_v2'], ROWS[worst]['guard']),
                xycoords='data', textcoords=ax.transAxes, xytext=(0.03, 0.885),
                fontsize=8.6, color=INK, ha='left', va='top', bbox=BOX, zorder=6,
                arrowprops=dict(arrowstyle='->', color=SECOND, lw=0.9))
    best = max(EPS, key=lambda e: ROWS[e]['pure_v2'] - ROWS[e]['guard'])
    ax.annotate(f"largest single gain {ROWS[best]['short']}\n"
                f"{ROWS[best]['pure_v2']:.2f} → {ROWS[best]['guard']:.2f}",
                xy=(ROWS[best]['pure_v2'], ROWS[best]['guard']),
                xycoords='data', textcoords=ax.transAxes, xytext=(0.955, 0.42),
                fontsize=8.6, color=INK, ha='right', va='center', bbox=BOX, zorder=6,
                arrowprops=dict(arrowstyle='->', color=SECOND, lw=0.9))
    regime_legend(ax, loc='upper center', bbox_to_anchor=(0.5, -0.115), ncol=3)

    # right: per-regime outcome counts
    ypos = np.arange(3)[::-1]
    for i, r in enumerate(REGIME_ORDER):
        w, t, l = counts[r]
        left = 0.0
        for val, col, alpha, hatch in ((w, REGIME_COLOR[r], 1.0, None),
                                       (t, MUTED, 0.35, None),
                                       (l, SECOND, 0.85, '///')):
            if val:
                axb.barh(ypos[i], val, left=left, height=0.5, color=col, alpha=alpha,
                         edgecolor=SURFACE, lw=2, hatch=hatch)
                if val >= 3:
                    axb.text(left + val / 2, ypos[i], str(val), ha='center', va='center',
                             fontsize=9, color='white' if alpha > 0.6 else SECOND,
                             fontweight='bold', zorder=5)
                else:
                    axb.text(left + val + 0.6, ypos[i], str(val), ha='left', va='center',
                             fontsize=9, color=SECOND, fontweight='bold')
            left += val
    axb.set_yticks(ypos)
    axb.set_yticklabels([REGIME_LABEL[r].split(' (')[0] for r in REGIME_ORDER], fontsize=9.5)
    axb.set_xlim(0, 45); axb.set_ylim(-0.55, 2.55)
    axb.set_xlabel('episodes')
    hb = [plt.matplotlib.patches.Patch(facecolor=BLUE, edgecolor=SURFACE,
                                       label='unified better'),
          plt.matplotlib.patches.Patch(facecolor=MUTED, alpha=0.35, edgecolor=SURFACE,
                                       label='identical (routed analytic)'),
          plt.matplotlib.patches.Patch(facecolor=SECOND, alpha=0.85, edgecolor=SURFACE,
                                       hatch='///', label='unified worse')]
    lgb = axb.legend(handles=hb, loc='center right', bbox_to_anchor=(1.0, 0.335),
                     fontsize=8.4, labelcolor=SECOND, frameon=True, facecolor=SURFACE,
                     edgecolor='none', framealpha=0.9)
    lgb.set_zorder(10)
    axb.set_title('b) per-episode outcome by regime\n(colour fill takes the regime hue)',
                  fontsize=10.5, color=INK, loc='left', pad=10)
    ax.set_title('a) per-episode nRMSE, unified vs pure analytic', fontsize=10.5,
                 color=INK, loc='left', pad=10)
    axb.grid(axis='y', visible=False)

    m_v2 = mean_over(ROWS, 'pure_v2'); m_g = mean_over(ROWS, 'guard')
    sup(fig, 'Unified estimator vs pure analytic init — per-episode outcome (100 episodes)',
        f"one config: init_method='auto', auto_guard=True, guard ratio {GUARD_RATIO}, "
        f"router n_selected ≥ {THRESH_DEPLOYED}\n"
        f'mean nRMSE {m_v2:.4f} → {m_g:.4f} · {tw} episodes better, {tt} identical '
        f'(routed analytic), {tl} worse\n{PROTO}')
    foot(fig, 'sources: tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl · '
              'tmp/wp_integration_2026-07-25+unified_2026-07-26 arm_runs · '
              'guard_bf_ratio.json — reconstruction reproduces task1_results.json exactly')
    p = os.path.join(OUT, 'diag_scatter_v2_vs_auto.png')
    fig.savefig(p); plt.close(fig)
    return p, counts


# ---------------------------------------------------------------- figure 2
def fig_routing():
    fig = plt.figure(figsize=(13.0, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], left=0.055, right=0.985,
                          top=0.845, bottom=0.115, wspace=0.2)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    comps = [1, 2, 5, 15, 30]
    xpos = {c: i for i, c in enumerate(comps)}
    rng = np.random.default_rng(7)

    # ---- left: router signal
    for e in EPS:
        R = ROWS[e]
        x = xpos[R['comp']] + rng.uniform(-0.22, 0.22)
        R['_x'] = x
        ns = R['ns_lanes']
        if ns and max(ns) > min(ns):
            ax.plot([x, x], [min(ns), max(ns)], color=REGIME_COLOR[R['regime']],
                    lw=1.0, alpha=0.35, zorder=2)
    for r in REGIME_ORDER:
        keep = [e for e in SUB[r] if not ROWS[e]['any_veto']]
        ax.plot([ROWS[e]['_x'] for e in keep], [ROWS[e]['ns_med'] for e in keep], 'o',
                ms=8, mfc=REGIME_COLOR[r], mec=SURFACE, mew=1.2, ls='none', zorder=3,
                alpha=0.9)
    vet = [e for e in EPS if ROWS[e]['any_veto']]
    ax.plot([ROWS[e]['_x'] for e in vet], [ROWS[e]['ns_med'] for e in vet], 'o', ms=11,
            mfc='none', mec=INK, mew=1.9, ls='none', zorder=4)
    ax.plot([ROWS[e]['_x'] for e in vet], [ROWS[e]['ns_med'] for e in vet], 'x', ms=6,
            color=INK, mew=1.6, ls='none', zorder=5)

    for thr, ls, lab in ((THRESH_DEPLOYED, '-', f'threshold {THRESH_DEPLOYED} — shipped code default'),
                         (THRESH_ALT, ':', f'threshold {THRESH_ALT} — handoff ladder table')):
        ax.axhline(thr, color=SECOND, lw=1.3, ls=ls, zorder=1)
        ax.text(2.85, thr + 0.30, lab, fontsize=8.4, color=SECOND, ha='right', va='bottom',
                bbox=BOX, zorder=6)
    ax.text(2.85, 20.5, 'above the line → route to learned proposal\n'
            'below the line → route to analytic v2_konly', fontsize=8.8, color=SECOND,
            ha='right', va='center', bbox=BOX, zorder=6, fontweight='bold')
    ax.set_xticks(range(5)); ax.set_xticklabels([str(c) for c in comps])
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(0, 33)
    ax.set_xlabel('true number of wave components (ground truth, not available at run time)')
    ax.set_ylabel('n_selected  (learned proposal, median over 5 diffusion seeds)')
    ax.set_title('a) router signal — GT-free crowding proxy', fontsize=10.5, loc='left',
                 color=INK, pad=8)
    ax.annotate('BF2 1-comp over-proposal:\nthe net emits ~30 near-degenerate atoms\n'
                'on a single clean swell, so the router\nreads "crowded" when the sea is not',
                xy=(0.24, 30), xytext=(1.05, 29.0), fontsize=8.6, color=INK, ha='left',
                va='top', bbox=BOX, zorder=6,
                arrowprops=dict(arrowstyle='->', color=SECOND, lw=1.0))
    h = [plt.Line2D([], [], marker='o', ls='none', ms=7.5, mfc=REGIME_COLOR[r], mec=SURFACE,
                    mew=1.2, label=REGIME_LABEL[r].split(' (')[0]) for r in REGIME_ORDER]
    h.append(plt.Line2D([], [], marker='o', ls='none', ms=9, mfc='none', mec=INK, mew=1.8,
                        label=f'guard veto ({len(vet)} episodes)'))
    lg = ax.legend(handles=h, loc='center left', bbox_to_anchor=(0.275, 0.435), fontsize=8.6,
                   labelcolor=SECOND, frameon=True, facecolor=SURFACE, edgecolor='none',
                   framealpha=0.9)
    lg.set_zorder(10)

    # ---- right: guard signal
    routed = [e for e in EPS if ROWS[e]['ns_med'] >= THRESH_DEPLOYED]
    order = {'sparse': 0, 'moderate': 1, 'crowded': 2}
    for e in EPS:
        R = ROWS[e]
        R['_x2'] = order[R['regime']] + rng.uniform(-0.26, 0.26)
    for r in REGIME_ORDER:
        no = [e for e in SUB[r] if e not in routed]
        ye = [e for e in SUB[r] if e in routed]
        ax2.plot([ROWS[e]['_x2'] for e in no], [ROWS[e]['bfr'] for e in no], 'o', ms=6,
                 mfc=SURFACE, mec=REGIME_COLOR[r], mew=1.3, ls='none', alpha=0.55, zorder=2)
        ax2.plot([ROWS[e]['_x2'] for e in ye], [ROWS[e]['bfr'] for e in ye], 'o', ms=8,
                 mfc=REGIME_COLOR[r], mec=SURFACE, mew=1.2, ls='none', zorder=3)
    ax2.plot([ROWS[e]['_x2'] for e in vet], [ROWS[e]['bfr'] for e in vet], 'o', ms=11,
             mfc='none', mec=INK, mew=1.9, ls='none', zorder=4)
    ax2.plot([ROWS[e]['_x2'] for e in vet], [ROWS[e]['bfr'] for e in vet], 'x', ms=6,
             color=INK, mew=1.6, ls='none', zorder=5)

    ax2.axhline(GUARD_RATIO, color=INK, lw=1.4, zorder=1)
    cmin = min(ROWS[e]['bfr'] for e in SUB['crowded'])
    ax2.axhline(cmin, color=ORANGE, lw=1.1, ls='--', zorder=1)
    ax2.text(2.45, GUARD_RATIO * 0.94, f'guard threshold {GUARD_RATIO}', fontsize=8.6,
             color=INK, ha='right', va='top', bbox=BOX, zorder=6)
    ax2.text(2.45, cmin * 1.05, f'lowest crowded bf ratio {cmin:.3f} — the guard '
             'threshold sits below it,\nso no crowded episode can be vetoed',
             fontsize=8.6, color=SECOND, ha='right', va='bottom', bbox=BOX, zorder=6)
    ax2.set_yscale('log')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([REGIME_LABEL[r].split(' (')[0] for r in REGIME_ORDER])
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylabel('beamform dominant-peak ratio (observations only, GT-free)')
    ax2.set_title('b) guard signal — why the veto is safe', fontsize=10.5, loc='left',
                  color=INK, pad=8)
    ax2.grid(axis='x', visible=False)
    h2 = [plt.Line2D([], [], marker='o', ls='none', ms=7.5, mfc=SECOND, mec=SURFACE,
                     label=f'router says learned (n_selected ≥ {THRESH_DEPLOYED})'),
          plt.Line2D([], [], marker='o', ls='none', ms=6, mfc=SURFACE, mec=SECOND, mew=1.3,
                     label='router says analytic'),
          plt.Line2D([], [], marker='o', ls='none', ms=9, mfc='none', mec=INK, mew=1.8,
                     label='guard veto → analytic')]
    ax2.legend(handles=h2, loc='lower right', fontsize=8.5, labelcolor=SECOND)

    t1 = json.load(open(TASK1_JSON))
    sup(fig, 'Router decision surface and the beamform guard that corrects it',
        f"guard fires on {len(vet)} episodes (all BF2 1-comp) — sparse mean "
        f"{t1['current_auto_sparse']:.4f} → {t1['by_guard_ratio']['0.05']['sparse']:.4f}, "
        f"crowded flips {FLIPS}\n{PROTO}; n_selected is per diffusion seed, so the router "
        f"is evaluated per lane")
    foot(fig, 'guard rule: route learned AND beamform dominant-peak ratio < 0.05 → veto to '
              'v2_konly (never promotes analytic→learned) · '
              'wavesplat/optimizer/sequential_update.py:536-568')
    p = os.path.join(OUT, 'diag_routing.png')
    fig.savefig(p); plt.close(fig)
    return p, len(vet)


# ---------------------------------------------------------------- figure 3
def fig_oracle_gap():
    gaps = sorted(((ROWS[e]['guard'] - ROWS[e]['oracle'], e) for e in EPS), reverse=True)
    tot = sum(g for g, _ in gaps)
    TOPN = 16
    fig = plt.figure(figsize=(13.4, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], left=0.148, right=0.98,
                          top=0.795, bottom=0.11, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    top = gaps[:TOPN]
    y = np.arange(TOPN)[::-1]
    for i, (g, e) in enumerate(top):
        R = ROWS[e]
        ax.barh(y[i], g, height=0.62, color=REGIME_COLOR[R['regime']],
                edgecolor=SURFACE, lw=1.5)
        ax.text(g + tot * 0.006, y[i], f'{g:.3f}', va='center', ha='left', fontsize=8.2,
                color=SECOND)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{ROWS[e]['short']}  (ns={ROWS[e]['ns_med']:.0f})" for _, e in top],
                       fontsize=8.4)
    ax.set_xlim(0, top[0][0] * 1.13)
    ax.set_xlabel('per-episode routing cost:  auto+guard nRMSE − oracle-switch nRMSE')
    ax.set_title(f'a) top {TOPN} routing costs (of {sum(1 for g, _ in gaps if g > 1e-9)} '
                 f'episodes with any cost)', fontsize=10.5, loc='left', color=INK, pad=8)
    ax.grid(axis='y', visible=False)
    regime_legend(ax, loc='lower right')
    ax.text(0.985, 0.245, 'no crowded episode reaches the top 16 —\n'
            'crowded routing is already at the oracle', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8.5, color=MUTED)
    g0, e0 = gaps[0]
    ax.annotate(f'single episode = {100 * g0 / tot:.0f}% of the whole oracle gap\n'
                f'(analytic {ROWS[e0]["pure_v2"]:.3f} vs learned {ROWS[e0]["pure_learned"]:.3f}; '
                f'bf ratio {ROWS[e0]["bfr"]:.2f} → guard does not fire)',
                xy=(g0 * 0.55, y[0]), xytext=(g0 * 0.30, y[0] - 4.2), fontsize=8.6,
                color=INK, ha='left',
                arrowprops=dict(arrowstyle='->', color=SECOND, lw=1.0))

    # cumulative
    cum = np.cumsum([g for g, _ in gaps]) / tot
    ax2.plot(np.arange(1, 101), cum, color=INK, lw=2.0, zorder=3)
    for e_i, dx, dy in ((1, 4, -0.085), (8, 5, -0.085), (35, 5, 0.055)):
        ax2.plot([e_i], [cum[e_i - 1]], 'o', ms=8, mfc=SURFACE, mec=INK, mew=1.8, zorder=4)
        ax2.annotate(f'{e_i} ep → {100 * cum[e_i - 1]:.0f}%', xy=(e_i, cum[e_i - 1]),
                     xytext=(e_i + dx, cum[e_i - 1] + dy), fontsize=8.8, color=SECOND)
    ax2.axhline(1.0, color=GRID, lw=1.0)
    ax2.set_xlim(0, 101); ax2.set_ylim(0, 1.20)
    ax2.set_xlabel('episodes, sorted by routing cost (descending)')
    ax2.set_ylabel('cumulative share of the total oracle gap')
    ax2.set_title('b) the gap is not diffuse — it is a handful of episodes', fontsize=10.5,
                  loc='left', color=INK, pad=8)
    neg = sum(1 for g, _ in gaps if g < -1e-9)
    ax2.text(0.97, 0.10, f'{sum(1 for g, _ in gaps if abs(g) <= 1e-9)} episodes cost exactly 0.'
             f'\nThe curve overshoots 100% and returns because {neg} episodes\n'
             f'score BETTER than the per-episode oracle — per-lane routing\n'
             f'can mix the two arms inside one episode.',
             transform=ax2.transAxes, ha='right', va='bottom', fontsize=8.5, color=MUTED)

    shares = {r: sum(ROWS[e]['guard'] - ROWS[e]['oracle'] for e in SUB[r]) for r in REGIME_ORDER}
    sup(fig, 'Where the remaining oracle gap lives (0.0875 deployed → 0.0818 oracle)',
        f"mean gap {tot / 100:.4f} nRMSE over 100 episodes · share of the total gap by regime: "
        + ', '.join(f'{r} {100 * shares[r] / tot:+.0f}%' for r in REGIME_ORDER)
        + f'\n{PROTO}\noracle = per-episode min(pure analytic, pure learned) — an upper bound '
          'that requires ground truth and is NOT achievable at run time')
    foot(fig, 'sources: tmp/unified_2026-07-26/switch_results.json (misroutes) + per-episode '
              're-assembly · oracle is NOT achievable at run time')
    p = os.path.join(OUT, 'diag_oracle_gap.png')
    fig.savefig(p); plt.close(fig)
    return p, tot


# ---------------------------------------------------------------- figure 4
def fig_seed_variance():
    sd_b = {e: float(np.std(ROWS[e]['bug_lanes'], ddof=1)) for e in EPS}
    sd_c = {e: float(np.std(ROWS[e]['v2_lanes'], ddof=1)) for e in EPS}
    rg_b = {e: float(np.ptp(ROWS[e]['bug_lanes'])) for e in EPS}
    rg_c = {e: float(np.ptp(ROWS[e]['v2_lanes'])) for e in EPS}

    fig = plt.figure(figsize=(15.0, 6.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.85], left=0.05, right=0.985,
                          top=0.815, bottom=0.145, wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    lo, hi = 4e-4, 0.25
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls='--', zorder=1)
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=SECOND, alpha=0.05, zorder=0)
    for r in REGIME_ORDER:
        ax.plot([sd_b[e] for e in SUB[r]], [sd_c[e] for e in SUB[r]], 'o', ms=7.5,
                mfc=REGIME_COLOR[r], mec=SURFACE, mew=1.2, ls='none', zorder=3, alpha=0.92)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.set_xlabel('5-seed s.d. of nRMSE — tracker bug ON (shipped default)')
    ax.set_ylabel('5-seed s.d. — tracker phase fix ON')
    ax.set_title('a) per-episode seed spread, paired', fontsize=10.5, loc='left',
                 color=INK, pad=8)
    ax.text(0.19, 5.5e-4, 'spread shrank\n(below diagonal)', fontsize=9, color=SECOND,
            fontweight='bold', ha='right', va='bottom')
    ax.text(0.60, 0.965, 'spread grew (above diagonal)', fontsize=9, color=SECOND,
            fontweight='bold', ha='center', va='top', transform=ax.transAxes)
    regime_legend(ax, loc='upper left', bbox_to_anchor=(0.015, 0.86))

    # ECDF of 5-seed range
    for name, d, col, ls in (('tracker bug ON', rg_b, MUTED, '--'),
                             ('tracker phase fix ON', rg_c, INK, '-')):
        v = np.sort(np.array([d[e] for e in EPS]))
        ax2.step(np.concatenate([[1e-5], v]),
                 np.concatenate([[0], np.arange(1, 101) / 100]), where='post',
                 color=col, lw=2.0, ls=ls, label=name)
    ax2.axvline(0.1, color=GRID, lw=1.0)
    nb, nc = sum(1 for e in EPS if rg_b[e] > 0.1), sum(1 for e in EPS if rg_c[e] > 0.1)
    ax2.text(0.108, 0.32, f'range > 0.10 nRMSE\n{nb} → {nc} episodes', fontsize=8.8,
             color=SECOND, ha='left', va='center', bbox=BOX, zorder=6)
    ax2.set_xscale('log'); ax2.set_xlim(1e-4, 0.6); ax2.set_ylim(0, 1.02)
    ax2.set_xlabel('per-episode 5-seed range (max − min nRMSE)  [log]')
    ax2.set_ylabel('cumulative fraction of the 100 episodes')
    ax2.set_title('b) the bimodal tail collapses', fontsize=10.5, loc='left', color=INK, pad=8)
    ax2.legend(loc='upper left', fontsize=8.8, labelcolor=SECOND)

    # per-regime bars — the honest caveat
    w = 0.34
    for i, r in enumerate(REGIME_ORDER):
        b = float(np.mean([sd_b[e] for e in SUB[r]]))
        c = float(np.mean([sd_c[e] for e in SUB[r]]))
        red = sum(1 for e in SUB[r] if sd_c[e] < sd_b[e])
        ax3.bar(i - w / 2 - 0.01, b, width=w, color=MUTED, alpha=0.55, edgecolor=SURFACE, lw=2)
        ax3.bar(i + w / 2 + 0.01, c, width=w, color=REGIME_COLOR[r], edgecolor=SURFACE, lw=2)
        ax3.text(i - w / 2 - 0.01, b + 0.0016, f'{b:.4f}', ha='center', fontsize=8.2, color=SECOND)
        ax3.text(i + w / 2 + 0.01, c + 0.0016, f'{c:.4f}', ha='center', fontsize=8.2, color=SECOND)
        ax3.text(i, -0.0075, f'{red}/{len(SUB[r])} shrank', ha='center', fontsize=8.2,
                 color=MUTED)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels([REGIME_LABEL[r].split(' (')[0] for r in REGIME_ORDER])
    ax3.set_ylim(0, 0.068)
    ax3.set_ylabel('mean 5-seed s.d. of nRMSE')
    ax3.set_title('c) it is a sparse-regime effect', fontsize=10.5, loc='left', color=INK, pad=8)
    ax3.grid(axis='x', visible=False)
    ax3.text(0.98, 0.96, 'grey = bug ON,  colour = fix ON', transform=ax3.transAxes,
             ha='right', va='top', fontsize=8.4, color=MUTED)
    ax3.text(0.98, 0.87, 'crowded spread does NOT shrink', transform=ax3.transAxes,
             ha='right', va='top', fontsize=8.4, color=ORANGE)

    mb = float(np.mean(list(sd_b.values()))); mc = float(np.mean(list(sd_c.values())))
    lb = float(np.mean([ROWS[e]['bug_med'] for e in EPS]))
    lc = float(np.mean([ROWS[e]['v2_med'] for e in EPS]))
    sup(fig, 'Run-to-run seed spread before and after the tracker double phase-advance fix',
        f'mean 5-seed s.d. {mb:.4f} → {mc:.4f} (shrank on '
        f'{sum(1 for e in EPS if sd_c[e] < sd_b[e])}/100 episodes) — descriptive only, '
        f'no pre-registered test was run on spread\n'
        f'the pre-registered result is the LEVEL change: mean nRMSE {lb:.4f} → {lc:.4f}, '
        f'100/100 episodes improved, Wilcoxon p=3.9e-18 — that p value applies to the level, '
        f'not to the spread\n' + PROTO)
    foot(fig, 'sources: tmp/rerun_2026-07-25/batchB_ws.jsonl (bug ON) vs '
              'batchC_ws_phasefix.jsonl (fix ON) · same 100 episodes, same 5 estimator seeds')
    p = os.path.join(OUT, 'diag_seed_variance.png')
    fig.savefig(p); plt.close(fig)
    return p, (mb, mc, lb, lc)


# ---------------------------------------------------------------- verify
def verify():
    sw = json.load(open(SWITCH_JSON))
    t1 = json.load(open(TASK1_JSON))
    m8 = sw['by_threshold']['8']['means']
    checks = [
        ('pure_v2 100ep', mean_over(ROWS, 'pure_v2'), m8['pure_v2']),
        ('pure_learned 100ep', mean_over(ROWS, 'pure_learned'), m8['pure_learned']),
        ('oracle 100ep', mean_over(ROWS, 'oracle'), m8['oracle']),
        ('auto (no guard) 100ep', mean_over(ROWS, 'auto'), t1['current_auto_100']),
        ('auto+guard 100ep', mean_over(ROWS, 'guard'), t1['by_guard_ratio']['0.05']['mean_100']),
        ('auto+guard sparse', mean_over(ROWS, 'guard', SUB['sparse']),
         t1['by_guard_ratio']['0.05']['sparse']),
        ('auto+guard crowded', mean_over(ROWS, 'guard', SUB['crowded']),
         t1['by_guard_ratio']['0.05']['crowded']),
        ('auto+guard moderate', mean_over(ROWS, 'guard', SUB['moderate']),
         t1['by_guard_ratio']['0.05']['moderate']),
        ('crowded flips', float(FLIPS), float(t1['primary']['flips'])),
    ]
    print('\nVERIFICATION (reconstruction vs shipped result JSONs)')
    ok = True
    for name, got, ref in checks:
        d = abs(got - ref)
        ok &= d < 1e-3
        print(f'  {name:26s} got={got:.6f} ref={ref:.6f}  |d|={d:.2e}  '
              f'{"OK" if d < 1e-3 else "MISMATCH"}')
    # threshold sensitivity note
    r6, _ = assemble(THRESH_ALT, GUARD_RATIO)
    print(f'  threshold-{THRESH_ALT} variant  auto+guard 100ep = '
          f'{mean_over(r6, "guard"):.6f} (deployed threshold {THRESH_DEPLOYED}: '
          f'{mean_over(ROWS, "guard"):.6f})')
    print(f'  ALL WITHIN 1e-3: {ok}')
    return ok


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    ok = verify()
    p1, counts = fig_scatter()
    p2, nvet = fig_routing()
    p3, tot = fig_oracle_gap()
    p4, stats = fig_seed_variance()
    print('\nwrote:')
    for p in (p1, p2, p3, p4):
        print(f'  {p}  {os.path.getsize(p) / 1024:.0f} KB')
    print('\nscatter counts (win/tie/loss):', counts)
    print('guard vetoes:', nvet, ' total oracle gap:', round(tot, 4))
