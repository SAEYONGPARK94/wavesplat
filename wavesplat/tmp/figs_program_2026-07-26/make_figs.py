"""Program-summary figures for the 2026-07-25/26 WavePrior arc (Discord + handoff).

All numbers are read from the session artifacts (no re-runs):
  fig1 program_journey.png        : 100ep estimator ladder (bug -> phasefix -> unified+guard)
  fig2 program_phasefix.png       : tracker fix by complexity (batchB vs batchC)
  fig3 program_regime_cross.png   : per-regime pure-v2 / pure-learned / auto+guard
  fig4 program_real_fallback.png  : real-data amp-guard fallback (30 windows)

Run from wavesplat/ package root:  python3 tmp/figs_program_2026-07-26/make_figs.py
"""
import json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'results/gifs'
BATCHB = 'tmp/rerun_2026-07-25/batchB_ws.jsonl'          # tracker bug ON (shipped)
BATCHC = 'tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'  # tracker fixed
S1_ARMS = 'tmp/wp_integration_2026-07-25/arm_runs.jsonl'
X_ARMS = 'tmp/unified_2026-07-26/arm_runs_extra.jsonl'
TASK1 = 'tmp/unified_guard_2026-07-26/task1_results.json'
SWITCH = 'tmp/unified_2026-07-26/switch_results.json'
FALLBACK = 'tmp/unified_2026-07-26/fallback_results.json'
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

# palette (dataviz reference, light mode)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, SEC, MUT, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'text.color': INK, 'axes.edgecolor': MUT, 'axes.labelcolor': SEC,
    'xtick.color': SEC, 'ytick.color': SEC, 'axes.grid': False,
    'font.family': 'sans-serif', 'font.size': 11,
})


def comp_of(ep):
    return int(re.search(r'_(\d+)comp', ep).group(1))


def regime(ep):
    c = comp_of(ep)
    return 'sparse' if c <= 2 else ('moderate' if c == 5 else 'crowded')


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float('nan')


def load_pure_arms():
    """Per-episode median-of-5-lane pure arms, exactly as analyze_switch.py."""
    v2 = {}
    eps = set()
    for l in open(BATCHC):
        d = json.loads(l)
        v2[(d['episode'], d['seed'])] = d['nRMSE']
        eps.add(d['episode'])
    learned = {}
    for src in (S1_ARMS, X_ARMS):
        for l in open(src):
            d = json.loads(l)
            if d.get('arm') == 'learned' and d.get('err') is None \
                    and d.get('avg_nRMSE') is not None:
                learned[(d['episode'], d['est_seed'], d['diff_seed'])] = d['avg_nRMSE']
    pv2, pln = {}, {}
    for ep in sorted(eps):
        pv2[ep] = med([v2.get((ep, E)) for (E, D) in LANES])
        pln[ep] = med([learned.get((ep, E, D)) for (E, D) in LANES])
    return pv2, pln


def bar_labels(ax, bars, fmt='{:.3f}', dy=0.002):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(b.get_height()), ha='center', va='bottom',
                fontsize=9.5, color=INK)


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


# ---------------------------------------------------------------- fig 1 journey
def fig_journey():
    sw = json.load(open(SWITCH))['by_threshold']['6']['means']
    t1 = json.load(open(TASK1))
    b_off = np.mean([np.median(v) for v in _batch_by_ep(BATCHB).values()])
    steps = [
        ('Shipped default\n(tracker double\nphase-advance bug)', b_off, MUT),
        ('Tracker fix\n(pure analytic v2)', sw['pure_v2'], BLUE),
        ('Pure learned init\n(cond-diffusion)', sw['pure_learned'], ORANGE),
        ("init_method='auto'\n(GT-free router)", sw['switch_perlane'], AQUA),
        ('auto + guard\n(DEPLOYED)', t1['by_guard_ratio']['0.05']['mean_100'], AQUA),
    ]
    oracle = sw['oracle']
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    xs = np.arange(len(steps))
    bars = ax.bar(xs, [s[1] for s in steps], width=0.62,
                  color=[s[2] for s in steps], zorder=3)
    bars[-1].set_edgecolor(INK); bars[-1].set_linewidth(1.4)
    bar_labels(ax, bars, '{:.4f}')
    ax.axhline(oracle, color=MUT, ls=(0, (4, 3)), lw=1.2, zorder=2)
    ax.text(-0.42, oracle - 0.004, f'oracle switch  {oracle:.4f}',
            color=SEC, fontsize=9.5, ha='left', va='top')
    ax.set_xticks(xs, [s[0] for s in steps], fontsize=9.5)
    ax.set_ylabel('mean nRMSE  (100 episodes, lower = better)')
    ax.set_ylim(0, 0.215)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    style(ax)
    ax.set_title('One estimator, two days: 0.199 → 0.0875\n'
                 '100 ep × 5 seeds, median-of-5 per episode, data_finemesh',
                 fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(f'{OUT}/program_journey.png', dpi=160)
    plt.close(fig)
    print('fig1 saved', b_off)


def _batch_by_ep(path):
    by = {}
    for l in open(path):
        d = json.loads(l)
        by.setdefault(d['episode'], []).append(d['nRMSE'])
    return by


# ---------------------------------------------------------------- fig 2 phasefix
def fig_phasefix():
    off, on = _batch_by_ep(BATCHB), _batch_by_ep(BATCHC)
    groups = [('1comp', 1), ('2comp', 2), ('5comp', 5), ('15comp', 15), ('30comp', 30)]
    xo, xn = [], []
    for _, c in groups:
        eo = [med(v) for ep, v in off.items() if comp_of(ep) == c]
        en = [med(v) for ep, v in on.items() if comp_of(ep) == c]
        xo.append(np.mean(eo)); xn.append(np.mean(en))
    xo.append(np.mean([med(v) for v in off.values()]))
    xn.append(np.mean([med(v) for v in on.values()]))
    labels = [g[0] for g in groups] + ['ALL (100 ep)']
    xs = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    b1 = ax.bar(xs - w / 2, xo, w, color=MUT, zorder=3, label='before fix (shipped)')
    b2 = ax.bar(xs + w / 2, xn, w, color=BLUE, zorder=3, label='after fix (TRACK_PHASE_PREDICT=False)')
    bar_labels(ax, b1); bar_labels(ax, b2)
    ax.set_xticks(xs, labels)
    ax.set_ylabel('mean nRMSE (lower = better)')
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.legend(frameon=False, loc='upper left', fontsize=9.5)
    style(ax)
    ax.set_title('Tracker double phase-advance fix — improves 100/100 episodes '
                 '(Wilcoxon p=3.9e-18)\nsparse seas become near-exact; the crowding '
                 'wall survives', fontsize=11.5, pad=10)
    fig.tight_layout()
    fig.savefig(f'{OUT}/program_phasefix.png', dpi=160)
    plt.close(fig)
    print('fig2 saved')


# ---------------------------------------------------------------- fig 3 regimes
def fig_regimes():
    pv2, pln = load_pure_arms()
    t1 = json.load(open(TASK1))
    g = t1['by_guard_ratio']['0.05']
    regs = ['sparse', 'moderate', 'crowded']
    nice = {'sparse': 'sparse\n1–2 comp (40 ep)',
            'moderate': 'moderate\n5 comp (20 ep)',
            'crowded': 'crowded\n15–30 comp (40 ep)'}
    v2m = {r: np.mean([v for ep, v in pv2.items() if regime(ep) == r]) for r in regs}
    lnm = {r: np.mean([v for ep, v in pln.items() if regime(ep) == r]) for r in regs}
    auto = {'sparse': g['sparse'], 'moderate': g['moderate'], 'crowded': g['crowded']}
    xs = np.arange(3); w = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    b1 = ax.bar(xs - w - 0.015, [v2m[r] for r in regs], w, color=BLUE, zorder=3,
                label='pure analytic (v2_konly)')
    b2 = ax.bar(xs, [lnm[r] for r in regs], w, color=ORANGE, zorder=3,
                label='pure learned (cond-diffusion init)')
    b3 = ax.bar(xs + w + 0.015, [auto[r] for r in regs], w, color=AQUA, zorder=3,
                label="auto + guard (single config, GT-free)")
    for b in (b1, b2, b3):
        bar_labels(ax, b, dy=0.0025)
    ax.set_xticks(xs, [nice[r] for r in regs])
    ax.set_ylabel('mean nRMSE (lower = better)')
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9.5)
    style(ax)
    ax.set_title('Clean regime cross — the GT-free router keeps (nearly)\n'
                 'the best of both regimes in a single config', fontsize=12, pad=10)
    ax.text(1.02, 0.225, 'crowded: learned wins\n36/40, p=5e-11', fontsize=9, color=SEC)
    ax.text(-0.38, 0.095, 'sparse: v2 wins; learned-only\nparity sealed FAIL (proposal OOD)',
            fontsize=9, color=SEC)
    fig.tight_layout()
    fig.savefig(f'{OUT}/program_regime_cross.png', dpi=160)
    plt.close(fig)
    print('fig3 saved', v2m, lnm, auto)


# ---------------------------------------------------------------- fig 4 real
def fig_real():
    fb = json.load(open(FALLBACK))
    agg, det = fb['aggregate'], fb['coll_detail']
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3),
                                  gridspec_kw={'width_ratios': [1, 1.35]})
    # left: 30-window aggregate
    vals = [agg['hand_mean'], agg['learned_all_mean'], agg['fallback_mean']]
    labs = ['analytic\n(hand)', 'learned\nalways', 'hand + amp-guard\nfallback (DEPLOYED)']
    bars = ax.bar(np.arange(3), vals, 0.58, color=[BLUE, ORANGE, AQUA], zorder=3)
    bars[2].set_edgecolor(INK); bars[2].set_linewidth(1.4)
    bar_labels(ax, bars, '{:.3f}', dy=0.008)
    ax.set_xticks(np.arange(3), labs, fontsize=9.5)
    ax.set_ylabel('mean nRMSE, 30 real stereo windows')
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    style(ax)
    ax.set_title('Real data (Bergamasco stereo):\naggregate', fontsize=11)
    # right: the 3 analytic collapse windows
    xs = np.arange(3); w = 0.36
    hand = [d['hand'] for d in det]; dep = [d['fallback'] for d in det]
    names = [d['window'] + ('\n(guard fired)' if d['fires'] else '\n(mild — not fired)')
             for d in det]
    b1 = ax2.bar(xs - w / 2, hand, w, color=BLUE, zorder=3, label='analytic (hand)')
    b2 = ax2.bar(xs + w / 2, dep, w, color=AQUA, zorder=3, label='deployed (guard fallback)')
    bar_labels(ax2, b1, '{:.2f}', dy=0.03); bar_labels(ax2, b2, '{:.2f}', dy=0.03)
    ax2.axhline(1.0, color=MUT, ls=(0, (4, 3)), lw=1.1)
    ax2.text(1.41, 1.04, 'nRMSE = 1', color=SEC, fontsize=8.5, ha='center')
    ax2.set_xticks(xs, names, fontsize=9)
    ax2.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax2.legend(frameon=False, fontsize=9.5)
    style(ax2)
    ax2.set_title('Analytic collapse windows: guard catches the\n'
                  'two severe ones; false alarms 0/27', fontsize=11)
    fig.suptitle('Real-data role of learning = collapse insurance, not accuracy '
                 '(GT-free amplitude-ratio guard, τ=1.5)', fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f'{OUT}/program_real_fallback.png', dpi=160)
    plt.close(fig)
    print('fig4 saved')


if __name__ == '__main__':
    fig_journey()
    fig_phasefix()
    fig_regimes()
    fig_real()
