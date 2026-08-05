"""Rx-1 analysis, PREREG sec.1 aggregation convention (fixed before any run).

- per (cell, episode): MEDIAN over the 3 seeds of each metric
- primary statistic: MEDIAN of the PAIRED per-episode deltas (cell - X-BASE)
- sign convention: nRMSE / init_nrmse / Hs_error / amp_rel_error / |VR-1|  = smaller better
                   rho / skill                                            = larger  better
  VR is ALWAYS reported as |VR-1| for direction (PREREG sec.1)
- two-sided sign test, k/n and p reported together; n=4 floor p=0.125 -> never "significant"

Axis decomposition (PREREG sec.3). For zero-mean fields the variance-normalised MSE is
    m = MSE/var(gt) = 1 - 2*rho*sqrt(v) + v,     v = Variance_Ratio
and the shipped metrics satisfy  Skill_Score = 1 - m  and  nRMSE = sqrt(m)/4  (Hs = 4*sigma).
We verify both identities on the raw panel, then split each knob's delta-m into
    v-axis   = m(rho_base, v_knob) - m(rho_base, v_base)      (shrinkage)
    rho-axis = m(rho_knob, v_base) - m(rho_base, v_base)      (decorrelation)
    residual = total - v-axis - rho-axis                       (interaction)
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

SMALLER_BETTER = ['nRMSE', 'init_nrmse', 'absVR_minus_1', 'Hs_error', 'amp_rel_error']
LARGER_BETTER = ['Pearson_rho', 'Skill_Score']
ALL_METRICS = ['nRMSE', 'Pearson_rho', 'absVR_minus_1', 'Hs_error',
               'amp_rel_error', 'Skill_Score', 'Variance_Ratio', 'init_nrmse',
               'sum_w_ratio', 'avg_RMSE_unobserved', 'Directional_R2']
CELL_ORDER = ['X-BASE', 'K1', 'K2', 'K3', 'K4', 'K5', 'K12']


def binom_two_sided(k, n):
    """Exact two-sided sign test p (point-probability method, symmetric binomial)."""
    if n == 0:
        return float('nan')
    from math import comb
    kk = min(k, n - k)
    p = sum(comb(n, i) for i in range(0, kk + 1)) / 2 ** n * 2
    return min(1.0, p)


def load(paths):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' in r or 'cell' not in r:
                continue
            rows.append(r)
    # de-dup by job_id, keep last
    seen = {}
    for r in rows:
        seen[r['job_id']] = r
    return list(seen.values())


def enrich(r):
    vr = r.get('Variance_Ratio', float('nan'))
    r['absVR_minus_1'] = abs(vr - 1.0)
    spec = r.get('spec', {}) or {}
    r['amp_rel_error'] = spec.get('amp_rel_error_mean', float('nan'))
    sw, sg = spec.get('sum_w_est'), spec.get('sum_w_gt')
    r['sum_w_ratio'] = (sw / sg) if (sw is not None and sg) else float('nan')
    r['detection_rate'] = spec.get('detection_rate', float('nan'))
    return r


def main():
    paths = sys.argv[1:] or [os.path.join(HERE, 'results.jsonl')]
    rows = [enrich(r) for r in load(paths)]
    cells = defaultdict(lambda: defaultdict(list))     # cell -> ep -> [rows]
    for r in rows:
        cells[r['cell']][r['episode']].append(r)

    eps = sorted({r['episode'] for r in rows})
    print(f'rows={len(rows)}  cells={sorted(cells)}  episodes={len(eps)}')
    for c in sorted(cells):
        counts = {e: len(v) for e, v in cells[c].items()}
        print(f'  {c:<8s} seeds/ep {counts}')
    print()

    # ---- median over seeds ----
    med = defaultdict(dict)      # cell -> ep -> {metric: median}
    for c, byep in cells.items():
        for e, rs in byep.items():
            med[c][e] = {m: float(np.nanmedian([r.get(m, float('nan')) for r in rs]))
                         for m in ALL_METRICS}

    # ---- identity checks on raw rows ----
    print('=== identity audit (raw per-run, frame-averaged panel) ===')
    d1, d2 = [], []
    for r in rows:
        ss, rho, v = r['Skill_Score'], r['Pearson_rho'], r['Variance_Ratio']
        d1.append(abs(r['nRMSE'] - math.sqrt(max(1 - ss, 0)) / 4))
        d2.append(abs((1 - ss) - (1 - 2 * rho * math.sqrt(max(v, 0)) + v)))
    print(f'  |nRMSE - sqrt(1-SS)/4|            max {max(d1):.2e}  median {np.median(d1):.2e}')
    print(f'  |(1-SS) - (1-2*rho*sqrt(v)+v)|    max {max(d2):.2e}  median {np.median(d2):.2e}')
    print()

    # ---- cell levels ----
    print('=== cell levels (median over seeds, then mean/median over 4 episodes) ===')
    hdr = ['cell'] + ['nRMSE', 'rho', 'VR', '|VR-1|', 'Hs_err', 'amp_err', 'skill', 'init', 'sumW/GT']
    print('  ' + ''.join(f'{h:>10s}' for h in hdr))
    for c in [x for x in CELL_ORDER if x in med] + [x for x in sorted(med) if x not in CELL_ORDER]:
        vals = []
        for m in ['nRMSE', 'Pearson_rho', 'Variance_Ratio', 'absVR_minus_1',
                  'Hs_error', 'amp_rel_error', 'Skill_Score', 'init_nrmse', 'sum_w_ratio']:
            vals.append(float(np.median([med[c][e][m] for e in eps if e in med[c]])))
        print(f'  {c:>10s}' + ''.join(f'{v:10.4f}' for v in vals))
    print()

    # ---- paired deltas vs X-BASE ----
    base = med.get('X-BASE')
    if base is None:
        print('no X-BASE'); return
    print('=== PRIMARY + SECONDARY: paired per-episode deltas vs X-BASE ===')
    print('  (delta = cell - X-BASE; "better" counts use the pre-registered direction)')
    out = {}
    for c in [x for x in CELL_ORDER if x in med and x != 'X-BASE']:
        out[c] = {}
        print(f'\n  --- {c} ---')
        for m in ['nRMSE', 'Pearson_rho', 'absVR_minus_1', 'Hs_error',
                  'amp_rel_error', 'Skill_Score', 'init_nrmse', 'Variance_Ratio',
                  'sum_w_ratio']:
            deltas, per = [], []
            for e in eps:
                if e not in med[c] or e not in base:
                    continue
                d = med[c][e][m] - base[e][m]
                if np.isfinite(d):
                    deltas.append(d); per.append((e, d))
            if not deltas:
                continue
            n = len(deltas)
            if m in SMALLER_BETTER:
                k = sum(1 for d in deltas if d < 0)
            elif m in LARGER_BETTER:
                k = sum(1 for d in deltas if d > 0)
            else:
                k = sum(1 for d in deltas if d != 0)     # direction-free
            p = binom_two_sided(k, n)
            tag = '' if m in SMALLER_BETTER + LARGER_BETTER else '  (no direction)'
            print(f'    {m:<16s} medD {np.median(deltas):+9.4f}   better {k}/{n}  '
                  f'p={p:.4f}{tag}   per-ep ' +
                  ' '.join(f'{d:+.4f}' for _, d in per))
            out[c][m] = {'median_delta': float(np.median(deltas)), 'k': k, 'n': n,
                         'p': p, 'per_ep': {e: float(d) for e, d in per}}

    # ---- exact orthogonal split: m = (1-rho^2) + (sqrt(v)-rho)^2 ----
    print('\n=== EXACT SPLIT  m = (1-rho^2) + (sqrt(v)-rho)^2 ===')
    print('  (1-rho^2)      = DECORRELATION floor, irreducible by any scalar rescale')
    print('  (sqrt(v)-rho)^2= SHRINKAGE/miscalibration, removable by one scalar')
    print(f"  {'cell':>7s}{'m':>9s}{'decorr':>9s}{'shrink':>9s}{'shrink%':>9s}"
          f"{'d_m':>9s}{'d_decorr':>10s}{'d_shrink':>10s}")

    def split(rho, v):
        s = math.sqrt(max(v, 0))
        return 1 - rho ** 2, (s - rho) ** 2

    base_sp = {e: split(base[e]['Pearson_rho'], base[e]['Variance_Ratio']) for e in eps
               if e in base}
    for c in [x for x in CELL_ORDER if x in med]:
        ms, ds, ss_, dm, dd, dss = [], [], [], [], [], []
        for e in eps:
            if e not in med[c] or e not in base_sp:
                continue
            d, s = split(med[c][e]['Pearson_rho'], med[c][e]['Variance_Ratio'])
            bd, bs = base_sp[e]
            ms.append(d + s); ds.append(d); ss_.append(s)
            dm.append((d + s) - (bd + bs)); dd.append(d - bd); dss.append(s - bs)
        if not ms:
            continue
        M, D, S = np.median(ms), np.median(ds), np.median(ss_)
        print(f'  {c:>7s}{M:9.4f}{D:9.4f}{S:9.4f}{100*S/M:8.1f}%'
              f'{np.median(dm):9.4f}{np.median(dd):10.4f}{np.median(dss):10.4f}')
        if c != 'X-BASE':
            out.setdefault(c, {})['exact_split'] = {
                'd_m': float(np.median(dm)), 'd_decorr': float(np.median(dd)),
                'd_shrink': float(np.median(dss)),
                'k_decorr_better': int(sum(1 for x in dd if x < 0)),
                'k_shrink_better': int(sum(1 for x in dss if x < 0)),
                'n': len(dm)}

    # ---- axis decomposition ----
    print('\n=== AXIS DECOMPOSITION: m = 1 - 2*rho*sqrt(v) + v  (per episode, then median) ===')
    print('  v-axis = shrinkage channel, rho-axis = decorrelation channel')
    print('  NOTE m here is reconstructed from the frame-averaged (rho, v); the panel also')
    print('  carries m = 1 - Skill_Score directly. Closure error is printed above.')

    def mfun(rho, v):
        return 1 - 2 * rho * math.sqrt(max(v, 0)) + v

    for c in [x for x in CELL_ORDER if x in med and x != 'X-BASE']:
        tot, va, ra, inter = [], [], [], []
        for e in eps:
            if e not in med[c]:
                continue
            rb, vb = base[e]['Pearson_rho'], base[e]['Variance_Ratio']
            rk, vk = med[c][e]['Pearson_rho'], med[c][e]['Variance_Ratio']
            m0 = mfun(rb, vb)
            t = mfun(rk, vk) - m0
            v_only = mfun(rb, vk) - m0
            r_only = mfun(rk, vb) - m0
            tot.append(t); va.append(v_only); ra.append(r_only)
            inter.append(t - v_only - r_only)
        if not tot:
            continue
        T, V, R, I = (float(np.median(x)) for x in (tot, va, ra, inter))
        share_v = 100 * V / T if abs(T) > 1e-12 else float('nan')
        share_r = 100 * R / T if abs(T) > 1e-12 else float('nan')
        print(f'  {c:>6s}  d_m {T:+8.4f} = v-axis {V:+8.4f} ({share_v:6.1f}%) '
              f'+ rho-axis {R:+8.4f} ({share_r:6.1f}%) + inter {I:+8.4f}')
        out.setdefault(c, {})['axis'] = {'d_m': T, 'v_axis': V, 'rho_axis': R,
                                         'interaction': I,
                                         'v_share_pct': share_v, 'rho_share_pct': share_r}

    # ---- MDE: seed noise ----
    print('\n=== MDE / seed noise (PREREG sec.5) ===')
    spreads = []
    for c, byep in cells.items():
        for e, rs in byep.items():
            if len(rs) >= 2:
                spreads.append(np.std([r['nRMSE'] for r in rs], ddof=1))
    if spreads:
        print(f'  per-(cell,episode) seed std of nRMSE: median {np.median(spreads):.4f}  '
              f'max {np.max(spreads):.4f}  n_cells {len(spreads)}')
        se_med3 = np.median(spreads) / math.sqrt(3)
        print(f'  s.e. of a median-of-3 cell value ~ {se_med3:.4f}; '
              f'paired delta noise ~ {se_med3 * math.sqrt(2):.4f}')
    print('  n=4 episodes -> two-sided sign test floor p=0.125 at 4/4. '
          'No result in this experiment can be called significant.')

    # ---- K1+K2 additivity check (PREREG sec.2 cross-ban) ----
    if all(x in out for x in ('K1', 'K2', 'K12')):
        a = out['K1']['nRMSE']['median_delta'] + out['K2']['nRMSE']['median_delta']
        b = out['K12']['nRMSE']['median_delta']
        print(f'\n=== K12 vs K1+K2 additivity (confirmation cell only) ===')
        print(f'  K1+K2 sum {a:+.4f}   K12 measured {b:+.4f}   '
              f'non-additivity {b - a:+.4f}')

    with open(os.path.join(HERE, 'rx1_summary.json'), 'w') as fh:
        json.dump({'levels': {c: {e: med[c][e] for e in med[c]} for c in med},
                   'contrasts': out}, fh, indent=1)
    print('\nwrote rx1_summary.json')


if __name__ == '__main__':
    main()
