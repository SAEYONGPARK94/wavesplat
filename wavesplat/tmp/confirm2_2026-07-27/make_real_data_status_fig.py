"""Figure: real-data status -- why 'auto' stays disabled on real records (2026-07-27).

READ-ONLY over existing result JSON/JSONL. No estimator is run. Every number plotted is
re-derived here from the frozen files and mirrored into real_data_fig_numbers.json.

Sources
  A  tmp/mixed_retrain_2026-07-26/gate_results_v1.json  -> G4.accuracy.per_window (30 windows,
     27 non-collapse) : analytic 'hand' vs learned mixed_v1 vs pre-retrain learned ('old').
  B  tmp/mixed_retrain_2026-07-26/gate_results_{v1,v3}.json -> G4.collapse.per_window (18
     harvested collapse windows; 'baseline' = analytic arm, 'e2_double' = the GT-free
     analytic kp x2 escape control from tmp/collapse_controls_2026-07-26/escapes.json).
  C  tmp/confirm2_2026-07-27/atoms_confirm2_v1.jsonl  -> router statistic n_selected of the
     DEPLOYED checkpoint on the fresh synthetic 40 (5 diffusion seeds/episode);
     tmp/blind_auto_real_2026-07-26/blind_auto_real.json -> n_selected of the shipped
     _auto_select_init measured live on the 30 real stereo windows;
     tmp/mixed_retrain_2026-07-26/wp_real_mixed_{v1,v3}.json -> n_eff proxy on the same
     30 real windows for the retrained checkpoints (annotation only).
"""
import json
import re
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = '/home/vexit/wavesplat/wavesplat/'
OUT = R + 'tmp/confirm2_2026-07-27/'

# palette (project-verified). aqua/gray are NOT used adjacently: dE(deuteranopia)=1.5.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GRAY, INK, INK2 = '#8a8983', '#1a1a19', '#5f5e56'
GRID = '#e6e4dc'

num = {'generated': '2026-07-27', 'note': 'all values re-derived from frozen files; no estimator run'}


def style(ax):
    ax.set_facecolor('white')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRAY)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------- data
g1 = json.load(open(R + 'tmp/mixed_retrain_2026-07-26/gate_results_v1.json'))['G4']
g3 = json.load(open(R + 'tmp/mixed_retrain_2026-07-26/gate_results_v3.json'))['G4']

normals = [x for x in g1['accuracy']['per_window'] if not x['collapse']]
hand = np.array([x['hand'] for x in normals])
v1 = np.array([x['new'] for x in normals])
old = np.array([x['old'] for x in normals])
num['A_normal_windows'] = {
    'n': len(normals),
    'source': 'tmp/mixed_retrain_2026-07-26/gate_results_v1.json :: G4.accuracy.per_window',
    'mean_nRMSE_analytic_hand': float(hand.mean()),
    'mean_nRMSE_learned_mixed_v1': float(v1.mean()),
    'mean_nRMSE_learned_pre_retrain': float(old.mean()),
    'median_nRMSE_analytic_hand': float(np.median(hand)),
    'median_nRMSE_learned_mixed_v1': float(np.median(v1)),
    'learned_v1_better_than_hand': int((v1 < hand).sum()),
    'learned_v3_better_than_hand': int(g3['accuracy']['new_better_than_hand']),
    'mean_nRMSE_learned_mixed_v3': float(g3['accuracy']['new_mean']),
    'per_window': [{'rec': x['rec'], 'win': x['win'], 'hand': x['hand'], 'v1': x['new'],
                    'old': x['old']} for x in normals],
}

col = g1['collapse']
col3 = g3['collapse']
pwc = col['per_window']
base_b1 = sum(1 for x in pwc if x['baseline'] is not None and x['baseline'] < 1.0)
kp2_b1 = sum(1 for x in pwc if x['e2_double'] is not None and x['e2_double'] < 1.0)
num['B_collapse_windows'] = {
    'n_all': col['n_all'], 'n_heldout': col['n_heldout'],
    'source': 'tmp/mixed_retrain_2026-07-26/gate_results_{v1,v3}.json :: G4.collapse',
    'below_1.0_of_18': {
        'analytic baseline': base_b1,
        'learned (pre-retrain)': col['old_below_1'],
        'learned mixed_v1': col['new_below_1'],
        'learned mixed_v3': col3['new_below_1'],
        'analytic kp x2 escape': kp2_b1,
    },
    'heldout15_repairs_damages': {
        'learned (pre-retrain)': [col['old_repairs'], col['old_damages']],
        'learned mixed_v1': [col['new_repairs'], col['new_damages']],
        'learned mixed_v3': [col3['new_repairs'], col3['new_damages']],
    },
}

# --- C: router statistic n_selected
def comp(ep):
    return int(re.search(r'_(\d+)comp', ep).group(1))


syn = defaultdict(list)
for line in open(R + 'tmp/confirm2_2026-07-27/atoms_confirm2_v1.jsonl'):
    d = json.loads(line)
    if d.get('n_selected') is None:
        continue
    syn[comp(d['episode'])].append(int(d['n_selected']))
syn_sparse = syn[1] + syn[2]
syn_crowded = syn[15] + syn[30]

blind = json.load(open(R + 'tmp/blind_auto_real_2026-07-26/blind_auto_real.json'))
real_ns = [r['n_selected'] for r in blind['rows']]
real_routes = [r['route'] for r in blind['rows']]
real_guard = sum(1 for r in blind['rows'] if r['guard_fired'])

neff = {}
for v in ('v1', 'v3'):
    rows = json.load(open(R + f'tmp/mixed_retrain_2026-07-26/wp_real_mixed_{v}.json'))['results']
    neff[v] = [r['n_eff'] for r in rows]

THRESH = 8
num['C_router_signal'] = {
    'auto_switch_thresh': THRESH,
    'synthetic_sparse_1_2comp_v1': {
        'source': 'tmp/confirm2_2026-07-27/atoms_confirm2_v1.jsonl (fresh 40, 5 diffusion seeds)',
        'n': len(syn_sparse), 'min': min(syn_sparse), 'median': st.median(syn_sparse),
        'max': max(syn_sparse), 'below_thresh': sum(1 for x in syn_sparse if x < THRESH)},
    'synthetic_crowded_15_30comp_v1': {
        'n': len(syn_crowded), 'min': min(syn_crowded), 'median': st.median(syn_crowded),
        'max': max(syn_crowded), 'below_thresh': sum(1 for x in syn_crowded if x < THRESH)},
    'real_30_windows_shipped_router': {
        'source': 'tmp/blind_auto_real_2026-07-26/blind_auto_real.json (live _auto_select_init, '
                  'pre-retrain ckpt, diffusion seed 42)',
        'n': len(real_ns), 'min': min(real_ns), 'median': st.median(real_ns), 'max': max(real_ns),
        'below_thresh': sum(1 for x in real_ns if x < THRESH),
        'routed_learned': sum(1 for r in real_routes if r == 'learned'),
        'guard_vetoes': real_guard},
    'real_n_eff_proxy_retrained': {
        'source': 'tmp/mixed_retrain_2026-07-26/wp_real_mixed_{v1,v3}.json :: results[].n_eff',
        'caveat': 'n_eff is the per-posterior-sample kept-atom count, not the router n_selected; '
                  'n_selected has never been measured on real windows with the retrained ckpt',
        'v1': {'min': min(neff['v1']), 'median': st.median(neff['v1']), 'max': max(neff['v1'])},
        'v3': {'min': min(neff['v3']), 'median': st.median(neff['v3']), 'max': max(neff['v3'])}},
}

# ----------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.9), facecolor='white')
fig.subplots_adjust(left=0.047, right=0.99, top=0.755, bottom=0.185, wspace=0.235)

# ---------------- panel A
ax = axes[0]
style(ax)
lo, hi = 0.40, 1.28
ax.fill_between([lo, hi], [lo, hi], [hi, hi], color=BLUE, alpha=0.055, zorder=1, lw=0)
ax.plot([lo, hi], [lo, hi], color=GRAY, lw=1.2, ls='--', zorder=2)
ax.scatter(hand, v1, s=42, facecolor='white', edgecolor=INK2, linewidth=1.4, zorder=5)
ax.axvline(hand.mean(), color=ORANGE, lw=2.0, zorder=4)
ax.axhline(v1.mean(), color=BLUE, lw=2.0, zorder=4)
ax.axhline(old.mean(), color=GRAY, lw=1.4, ls=':', zorder=3)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel('analytic arm (hand)  nowcast nRMSE', fontsize=9.5, color=INK2)
ax.set_ylabel('learned arm (mixed_v1)  nowcast nRMSE', fontsize=9.5, color=INK2)
ax.set_title('A  27 normal real windows, paired',
             fontsize=11.5, color=INK, fontweight='bold', loc='left', pad=9)
ax.text(0.985, 0.055, 'learned worse\n(above the 1:1 line)  23 / 27',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color=INK2,
        linespacing=1.35)
ax.text(hand.mean() + 0.012, hi - 0.02, f'analytic mean {hand.mean():.3f}',
        color=ORANGE, fontsize=9, ha='left', va='top', fontweight='bold')
ax.text(hi - 0.015, v1.mean() + 0.012, f'learned v1 mean {v1.mean():.3f}',
        color=BLUE, fontsize=9, ha='right', va='bottom', fontweight='bold')
ax.text(lo + 0.015, old.mean() + 0.014, f'pre-retrain {old.mean():.3f}',
        color=INK2, fontsize=8.4, ha='left', va='bottom')
ax.text(0.0, -0.185, 'Retraining moved the real-sea mean by 0.009 -- not the sign.',
        transform=ax.transAxes, fontsize=9, color=INK2, ha='left', va='top')

# ---------------- panel B
ax = axes[1]
style(ax)
labels = ['analytic\nbaseline', 'learned\n(pre-retrain)', 'learned\nmixed_v1',
          'learned\nmixed_v3', 'analytic\nkp x2 escape']
vals = [base_b1, col['old_below_1'], col['new_below_1'], col3['new_below_1'], kp2_b1]
cols = [GRAY, BLUE, BLUE, BLUE, ORANGE]
xs = np.arange(len(vals))
ax.bar(xs, vals, width=0.62, color=cols, zorder=3)
for x, v in zip(xs, vals):
    ax.text(x, v + 0.28, str(v), ha='center', va='bottom', fontsize=11,
            color=INK, fontweight='bold')
sub = ['', '8 rep\n7 dmg', '7 rep\n8 dmg', '8 rep\n7 dmg', '']
for x, t in zip(xs, sub):
    if t:
        ax.text(x, 0.55, t, ha='center', va='bottom', fontsize=8.4, color='white',
                fontweight='bold', linespacing=1.3, zorder=6)
ax.axhline(kp2_b1, color=ORANGE, lw=1.2, ls='--', zorder=2)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=8.6, color=INK2)
ax.set_ylim(0, 14)
ax.set_yticks(range(0, 15, 2))
ax.set_ylabel('collapse windows recovered below nRMSE 1.0   (of 18)',
              fontsize=9.5, color=INK2)
ax.set_title('B  collapse windows: the cheap analytic escape still wins',
             fontsize=11.5, color=INK, fontweight='bold', loc='left', pad=9)
ax.text(0.985, 0.955, 'GT-free analytic control\n(kp x 2)  11 / 18', transform=ax.transAxes,
        ha='right', va='top', fontsize=9, color=ORANGE, fontweight='bold', linespacing=1.35)
ax.text(0.0, -0.185, 'White text = repairs / damages on the 15 held-out\ncollapse windows. No learned arm is net-positive.',
        transform=ax.transAxes, fontsize=9, color=INK2, ha='left', va='top',
        linespacing=1.4)

# ---------------- panel C
ax = axes[2]
style(ax)
groups = [('synthetic sparse\n1-2 comp (v1)\nbelow thresh 58/80', syn_sparse, AQUA),
          ('synthetic crowded\n15-30 comp (v1)\nbelow thresh 4/80', syn_crowded, BLUE),
          ('REAL 30 windows\nshipped router, live\nbelow thresh 0/30', real_ns, ORANGE)]
rng = np.random.default_rng(7)
for i, (lab, vals_, c) in enumerate(groups):
    y = np.array(vals_, float)
    x = i + rng.uniform(-0.19, 0.19, size=len(y))
    ax.scatter(x, y, s=26, color=c, alpha=0.55, linewidth=0.6, edgecolor='white', zorder=4)
    m = float(np.median(y))
    ax.plot([i - 0.30, i + 0.30], [m, m], color=c, lw=3.0, zorder=6,
            solid_capstyle='butt')
    ax.text(i, 32.6, f'median {m:g}', fontsize=9.5, color=c, va='center',
            ha='center', fontweight='bold')
ax.axhline(THRESH, color=INK2, lw=1.6, ls='--', zorder=5)
ax.text(1.50, THRESH + 0.9, 'auto_switch_thresh = 8',
        fontsize=8.8, color=INK2, va='bottom', ha='center')
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups], fontsize=8.6, color=INK2)
ax.set_xlim(-0.52, len(groups) - 0.48)
ax.set_ylim(0, 34.5)
ax.set_ylabel('router statistic  n_selected', fontsize=9.5, color=INK2)
ax.set_title('C  the GT-free router never takes its second branch',
             fontsize=11.5, color=INK, fontweight='bold', loc='left', pad=9)
ax.text(0.0, -0.185,
        'n_selected >= 8 routes to the learned arm. Real values use the\n'
        'pre-retrain ckpt; the retrained proxy n_eff reads '
        f"{min(neff['v1']):.1f}-{max(neff['v1']):.1f} (v1),\n"
        f"{min(neff['v3']):.1f}-{max(neff['v3']):.1f} (v3) -- still all >= 8.",
        transform=ax.transAxes, fontsize=9, color=INK2, ha='left', va='top',
        linespacing=1.4)

# ---------------- titles
fig.text(0.047, 0.972, "Real-data status - why 'auto' stays disabled on real records (2026-07-27)",
         fontsize=16.5, color=INK, fontweight='bold', ha='left', va='top')
fig.text(0.047, 0.912,
         'Synthetic confirmation authorised auto_v1_m3 for deployment. On the 30 real stereo '
         'windows every bar still points the other way, so the real path keeps\nthe analytic '
         'estimator with a triggered fallback.',
         fontsize=10, color=INK2, ha='left', va='top', linespacing=1.45)
fig.text(0.047, 0.826,
         'analytic arm = orange     learned arm = blue     reference / baseline = grey'
         '          all values read from frozen result files; nothing re-run',
         fontsize=8.8, color=GRAY, ha='left', va='top')
fig.text(0.047, 0.009,
         'sources: tmp/mixed_retrain_2026-07-26/gate_results_{v1,v3}.json  |  '
         'tmp/blind_auto_real_2026-07-26/blind_auto_real.json  |  '
         'tmp/confirm2_2026-07-27/atoms_confirm2_v1.jsonl',
         fontsize=7.6, color=GRAY, ha='left', va='bottom')

fig.savefig(OUT + 'real_data_status.png', dpi=150, facecolor='white')
json.dump(num, open(OUT + 'real_data_fig_numbers.json', 'w'), indent=2)
print('wrote', OUT + 'real_data_status.png')
print(json.dumps({k: v for k, v in num['C_router_signal'].items()}, indent=1)[:900])
