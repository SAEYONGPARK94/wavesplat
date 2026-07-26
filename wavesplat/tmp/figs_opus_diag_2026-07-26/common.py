"""Shared data assembly + style for the diag_* synthetic diagnostic figures (2026-07-26).

Reconstruction is byte-faithful to the scoring scripts that produced the shipped numbers:
  tmp/unified_2026-07-26/analyze_switch.py   (router, thresholds 6/8/10)
  tmp/unified_guard_2026-07-26/score_task1.py (guard, THRESH=8, ratio 0.05)
Protocol: per-episode value = median over the 5 lanes (est_seed, diff_seed).
"""
import json
import os
import re

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

BATCHB = os.path.join(ROOT, 'tmp/rerun_2026-07-25/batchB_ws.jsonl')
BATCHC = os.path.join(ROOT, 'tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl')
S1_ARMS = os.path.join(ROOT, 'tmp/wp_integration_2026-07-25/arm_runs.jsonl')
X_ARMS = os.path.join(ROOT, 'tmp/unified_2026-07-26/arm_runs_extra.jsonl')
S1_CACHE = os.path.join(ROOT, 'tmp/wp_integration_2026-07-25/atoms_cache.jsonl')
X_CACHE = os.path.join(ROOT, 'tmp/unified_2026-07-26/atoms_cache_extra.jsonl')
BF_JSON = os.path.join(ROOT, 'tmp/unified_guard_2026-07-26/guard_bf_ratio.json')
SWITCH_JSON = os.path.join(ROOT, 'tmp/unified_2026-07-26/switch_results.json')
TASK1_JSON = os.path.join(ROOT, 'tmp/unified_guard_2026-07-26/task1_results.json')
OUT = os.path.join(ROOT, 'results/gifs')

LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
THRESH_DEPLOYED = 8      # wavesplat/optimizer/sequential_update.py:508 default
THRESH_ALT = 6           # threshold reported in the handoff ladder table
GUARD_RATIO = 0.05

# ---- style -----------------------------------------------------------------
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
SECOND = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BLUE = '#2a78d6'    # analytic  / sparse
ORANGE = '#eb6834'  # learned   / crowded
AQUA = '#1baf7a'    # unified   / moderate

REGIME_COLOR = {'sparse': BLUE, 'moderate': AQUA, 'crowded': ORANGE}
REGIME_ORDER = ['sparse', 'moderate', 'crowded']
REGIME_LABEL = {'sparse': 'sparse (1-2 comp, n=40)',
                'moderate': 'moderate (5 comp, n=20)',
                'crowded': 'crowded (15-30 comp, n=40)'}


def apply_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': SECOND, 'ytick.color': SECOND,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 12, 'axes.labelsize': 10.5,
        'legend.frameon': False, 'figure.dpi': 160,
    })
    return plt


# ---- data ------------------------------------------------------------------
def comp(ep):
    return int(re.search(r'_(\d+)comp', ep).group(1))


def regime(c):
    return 'sparse' if c in (1, 2) else ('moderate' if c == 5 else 'crowded')


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float('nan')


def short(ep):
    """episode_BF2_1comp_seed0_slope.h5 -> BF2/1c/s0/slope"""
    m = re.match(r'episode_(BF\d+)_(\d+)comp_seed(\d+)_(\w+)\.h5', ep)
    if not m:
        return ep
    return f"{m.group(1)}/{m.group(2)}c/s{m.group(3)}/{m.group(4)[:3]}"


def load_raw():
    v2, eps = {}, set()
    for l in open(BATCHC):
        d = json.loads(l)
        v2[(d['episode'], d['seed'])] = d['nRMSE']
        eps.add(d['episode'])
    bug = {}
    for l in open(BATCHB):
        d = json.loads(l)
        bug[(d['episode'], d['seed'])] = d['nRMSE']
    learned = {}
    for src in (S1_ARMS, X_ARMS):
        for l in open(src):
            d = json.loads(l)
            if d.get('arm') == 'learned' and d.get('err') is None and d.get('avg_nRMSE') is not None:
                learned[(d['episode'], d['est_seed'], d['diff_seed'])] = d['avg_nRMSE']
    nsel = {}
    for src in (S1_CACHE, X_CACHE):
        for l in open(src):
            d = json.loads(l)
            if d.get('n_selected') is not None:
                nsel[(d['episode'], d['diff_seed'])] = d['n_selected']
    bfr = json.load(open(BF_JSON))
    return sorted(eps), v2, bug, learned, nsel, bfr


def assemble(thresh=THRESH_DEPLOYED, guard_ratio=GUARD_RATIO):
    """Per-episode table. Routing is evaluated per lane (faithful to the live path)."""
    eps, v2, bug, learned, nsel, bfr = load_raw()
    assert len(eps) == 100, len(eps)
    rows = {}
    crowded_flips = 0
    for ep in eps:
        v2l = [v2.get((ep, E)) for (E, D) in LANES]
        bgl = [bug.get((ep, E)) for (E, D) in LANES]
        lnl = [learned.get((ep, E, D)) for (E, D) in LANES]
        nsl = [nsel.get((ep, D)) for (E, D) in LANES]
        auto_lane, guard_lane, picks, gpicks = [], [], [], []
        for (E, D), vv, lv, ns in zip(LANES, v2l, lnl, nsl):
            base = 'learned' if (ns is not None and ns >= thresh) else 'v2'
            g = base
            if base == 'learned' and guard_ratio is not None and bfr.get(ep, 1.0) < guard_ratio:
                g = 'v2'
            auto_lane.append(lv if base == 'learned' else vv)
            guard_lane.append(lv if g == 'learned' else vv)
            picks.append(base)
            gpicks.append(g)
            if comp(ep) in (15, 30) and g != base:
                crowded_flips += 1
        c = comp(ep)
        pv, pl = med(v2l), med(lnl)
        rows[ep] = dict(
            episode=ep, short=short(ep), comp=c, regime=regime(c),
            pure_v2=pv, pure_learned=pl,
            auto=med(auto_lane), guard=med(guard_lane), oracle=min(pv, pl),
            ns_med=med([n for n in nsl if n is not None]),
            ns_lanes=[n for n in nsl if n is not None],
            bfr=bfr.get(ep),
            n_learned_lanes=sum(p == 'learned' for p in picks),
            n_guard_lanes=sum(p == 'learned' for p in gpicks),
            veto=bool(any(p == 'learned' for p in picks) and all(p == 'v2' for p in gpicks)),
            any_veto=bool(any(a == 'learned' and b == 'v2' for a, b in zip(picks, gpicks))),
            v2_lanes=[x for x in v2l if x is not None],
            bug_lanes=[x for x in bgl if x is not None],
            bug_med=med(bgl), v2_med=pv,
        )
    return rows, crowded_flips


def mean_over(rows, key, subset=None):
    S = subset if subset is not None else list(rows)
    return float(np.nanmean([rows[e][key] for e in S]))
