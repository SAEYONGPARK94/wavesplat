"""S4 arm analysis -- follows PREREG_ARM.md sections 2 and 5, nothing else.

PRIMARY  : nRMSE, one-sided sign test on the 5 paired differences B - A,
           direction pre-specified as "B worse" (PREREG_ARM section 4).
           n=5 => two-sided floor 0.0625, so only the one-sided 5/5 result
           (p=0.03125) can clear alpha=0.05.  Everything else is descriptive.
AGGREGATE: median of paired differences (pairing by episode seed).
Each episode's WS metric = median-of-3 over estimator seeds [42,123,456].
"""
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

HERE = '/home/vexit/wavesplat/wavesplat/tmp/s4_within_spread_2026-08-05'
PANEL = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
         'Directional_R2', 'Hs_error', 'avg_RMSE_unobserved', 'elapsed_ms', 'N']
SPEC = ['detection_rate', 'energy_detection_rate', 'n_est', 'n_matched',
        'k_mag_rel_error_mean', 'direction_error_deg_mean', 'amp_rel_error_mean']
# "worse" direction, pre-registered.  +1 = higher is worse, -1 = lower is worse,
# 0 = no direction (diagnostic only, never tested).
WORSE = {'nRMSE': +1, 'Pearson_rho': -1, 'Skill_Score': -1, 'Directional_R2': -1,
         'Hs_error': +1, 'avg_RMSE_unobserved': +1, 'detection_rate': -1,
         'energy_detection_rate': -1, 'k_mag_rel_error_mean': +1,
         'direction_error_deg_mean': +1, 'amp_rel_error_mean': +1,
         'Variance_Ratio': 0, 'elapsed_ms': 0, 'N': 0, 'n_est': 0, 'n_matched': 0,
         'VR_dev': +1}

recs = [json.loads(l) for l in open(os.path.join(HERE, 'eval.jsonl'))]
recs = [r for r in recs if 'error' not in r]

# per (method, arm, ep_seed) -> metric dict (WS: median-of-3)
cell = defaultdict(lambda: defaultdict(list))
realized = {}
for r in recs:
    key = (r['method'], r['arm'], r['ep_seed'])
    for k in PANEL:
        if k in r and r[k] is not None:
            cell[key][k].append(float(r[k]))
    if 'RMSE_unobserved' in r:
        cell[key]['avg_RMSE_unobserved'].append(float(r['RMSE_unobserved']))
    for k in SPEC:
        if k in r.get('spec', {}):
            cell[key][k].append(float(r['spec'][k]))
    realized[(r['arm'], r['ep_seed'])] = r['realized']

M = {k: {m: float(np.median(v)) for m, v in d.items() if len(v)} for k, d in cell.items()}
for v in M.values():
    if 'Variance_Ratio' in v:
        v['VR_dev'] = abs(1.0 - v['Variance_Ratio'])

SEEDS = sorted({k[2] for k in M})
out = {'n_pairs': len(SEEDS), 'seeds': SEEDS, 'manipulation_check': {}, 'levels': {},
       'paired': {}}

# ---- manipulation check on the SAVED episodes -------------------------------
print('=== (3) MANIPULATION CHECK -- realized sigma_1 of the saved episodes ===')
print(f"{'arm':10s} {'bins':>5s} {'within':>8s} {'across':>8s} {'total':>8s} "
      f"{'Hs':>7s} {'M_attr':>7s} {'spread_attr':>12s}")
for arm in ('A_M1_N30', 'B_M3_N30', 'C_M1_N10'):
    rs = [realized[(arm, s)] for s in SEEDS if (arm, s) in realized]
    if not rs:
        continue
    med = {k: float(np.median([r[k] for r in rs])) for k in
           ('n_freq_bins', 'sigma1_within_deg', 'sigma1_across_deg',
            'sigma1_total_deg', 'Hs_from_components')}
    out['manipulation_check'][arm] = dict(
        median=med, per_seed={str(s): realized[(arm, s)] for s in SEEDS if (arm, s) in realized},
        mesh_verdicts=sorted({r['mesh_verdict'] for r in rs}))
    print(f"{arm:10s} {med['n_freq_bins']:5.0f} {med['sigma1_within_deg']:8.2f} "
          f"{med['sigma1_across_deg']:8.2f} {med['sigma1_total_deg']:8.2f} "
          f"{med['Hs_from_components']:7.4f} {rs[0]['dirs_per_freq']:7d} "
          f"{rs[0]['direction_spread_deg']:12.1f}   verdicts="
          f"{sorted({r['mesh_verdict'] for r in rs})}")
print('  CDIP measured (all6):  within 29.28  across 21.78  total 35.35')

# ---- levels ------------------------------------------------------------------
ALL = PANEL[:-1] + ['VR_dev', 'N'] + SPEC
print('\n=== (4a) LEVELS -- median over the 5 episode seeds ===')
for meth in ('WS', 'LETKF'):
    arms = [a for a in ('A_M1_N30', 'B_M3_N30', 'C_M1_N10')
            if any((meth, a, s) in M for s in SEEDS)]
    if not arms:
        continue
    print(f'\n-- {meth} --')
    hdr = f"{'metric':24s}" + ''.join(f'{a:>13s}' for a in arms)
    print(hdr)
    for k in ALL:
        row = {}
        for a in arms:
            v = [M[(meth, a, s)][k] for s in SEEDS
                 if (meth, a, s) in M and k in M[(meth, a, s)]]
            if v:
                row[a] = float(np.median(v))
        if not row:
            continue
        out['levels'].setdefault(meth, {})[k] = row
        print(f'{k:24s}' + ''.join(f'{row.get(a, float("nan")):13.4f}' for a in arms))

# ---- paired tests ------------------------------------------------------------
def paired(meth, treat, ctrl):
    res = {}
    for k in ALL:
        d, pa, pc = [], [], []
        for s in SEEDS:
            kt, kc = (meth, treat, s), (meth, ctrl, s)
            if kt in M and kc in M and k in M[kt] and k in M[kc]:
                d.append(M[kt][k] - M[kc][k]); pa.append(M[kt][k]); pc.append(M[kc][k])
        if len(d) < 2:
            continue
        d = np.array(d)
        w = WORSE.get(k, 0)
        n_worse = int(np.sum(d * w > 0)) if w else 0
        n_nz = int(np.sum(d != 0))
        r = dict(n=len(d), median_diff=float(np.median(d)),
                 diffs=[float(x) for x in d],
                 mean_ctrl=float(np.mean(pc)), mean_treat=float(np.mean(pa)),
                 sd_diff=float(np.std(d, ddof=1)) if len(d) > 1 else float('nan'),
                 worse_direction=w)
        if w and n_nz:
            r['n_in_worse_direction'] = f'{n_worse}/{n_nz}'
            r['sign_p_onesided_worse'] = float(
                binomtest(n_worse, n_nz, 0.5, alternative='greater').pvalue)
            r['sign_p_twosided'] = float(binomtest(n_worse, n_nz, 0.5).pvalue)
            try:
                r['wilcoxon_p_twosided'] = float(wilcoxon(d).pvalue)
            except Exception:
                r['wilcoxon_p_twosided'] = float('nan')
            # achieved MDE at 80% power for a 5/5 one-sided sign test
            r['achieved_MDE_80pct'] = float(1.712 * r['sd_diff'])
        res[k] = r
    return res


for meth in ('WS', 'LETKF'):
    for treat, ctrl, tag in (('B_M3_N30', 'A_M1_N30', 'B_minus_A'),
                             ('C_M1_N10', 'A_M1_N30', 'C_minus_A')):
        if not any((meth, treat, s) in M for s in SEEDS):
            continue
        r = paired(meth, treat, ctrl)
        out['paired'][f'{meth}|{tag}'] = r
        print(f'\n=== (4b) PAIRED  {meth}  {tag}  (n={r["nRMSE"]["n"]}) ===')
        print(f"{'metric':24s}{'medianΔ':>11s}{'sdΔ':>10s}{'worse':>8s}"
              f"{'p_1side':>10s}{'p_2side':>10s}{'MDE80':>10s}")
        for k in ALL:
            if k not in r:
                continue
            x = r[k]
            print(f"{k:24s}{x['median_diff']:11.4f}{x['sd_diff']:10.4f}"
                  f"{x.get('n_in_worse_direction', '  --'):>8s}"
                  f"{x.get('sign_p_onesided_worse', float('nan')):10.4f}"
                  f"{x.get('sign_p_twosided', float('nan')):10.4f}"
                  f"{x.get('achieved_MDE_80pct', float('nan')):10.4f}")

json.dump(out, open(os.path.join(HERE, 'analysis.json'), 'w'), indent=2)
print(f"\nwrote {os.path.join(HERE, 'analysis.json')}")
