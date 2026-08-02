"""전파 2단계 — 구본(2026-04-07) vs 재실행(2026-08-03) 대조.

⛔ 베이스라인은 재실행하지 않았다(구본 승계). WS 팔만 비교한다.
보고는 median-of-3 + 산포. 논문 주장이 뒤집히는지 확인하는 것이 목적.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
OLD = os.path.join(ROOT, 'results')


def med(seed_results, key='avg_nRMSE'):
    v = [s[key] for s in seed_results if 'error' not in s and s.get(key) is not None]
    return (float(np.median(v)), float(np.std(v)), len(v)) if v else (None, None, 0)


rep = {}

# ---------------- ablation -------------------------------------------------
new = json.load(open(os.path.join(HERE, 'ablation_seeded_rerun.json')))
old = json.load(open(os.path.join(OLD, 'ablation_seeded.json')))


def abl_map(rows):
    return {(r['episode'], r['config']): med(r['seed_results'])[0] for r in rows}


O, N = abl_map(old), abl_map(new)
eps = sorted({k[0] for k in N})
cfgs = ['Full', 'A1_noDisp', 'A2_noConf', 'v2+disp']
abl = {'per_config_median': {}, 'delta_vs_full': {}}
for c in cfgs:
    o = [O[(e, c)] for e in eps if O.get((e, c)) is not None]
    n = [N[(e, c)] for e in eps]
    abl['per_config_median'][c] = {'old': float(np.median(o)), 'new': float(np.median(n))}
for c in cfgs[1:]:
    dn = [N[(e, c)] - N[(e, 'Full')] for e in eps]
    do = [O[(e, c)] - O[(e, 'Full')] for e in eps
          if O.get((e, c)) is not None and O.get((e, 'Full')) is not None]
    abl['delta_vs_full'][c] = {
        'old_median': float(np.median(do)), 'old_worse_than_full': f'{sum(x > 0 for x in do)}/{len(do)}',
        'new_median': float(np.median(dn)), 'new_worse_than_full': f'{sum(x > 0 for x in dn)}/{len(dn)}'}
rep['ablation'] = abl

# ---------------- p1 / adaptive N -----------------------------------------
p = os.path.join(HERE, 'p1_comparison_rerun.json')
if os.path.exists(p):
    new = json.load(open(p))
    old = json.load(open(os.path.join(OLD, 'p1_comparison.json')))
    rows = []
    for ep, v in new.items():
        r = {'episode': ep}
        for arm in ('WS-P0', 'WS-P1'):
            if arm in v and isinstance(v[arm], dict) and 'seed_results' in v[arm]:
                m, sd, k = med(v[arm]['seed_results'])
                r[f'{arm}_new'] = m
                r[f'{arm}_new_std'] = sd
                r[f'{arm}_new_n'] = k
                fn = [s.get('final_n') for s in v[arm]['seed_results'] if 'error' not in s]
                r[f'{arm}_final_n_new'] = float(np.median(fn)) if fn else None
            ov = old.get(ep, {}).get(arm)
            if isinstance(ov, dict):
                r[f'{arm}_old'] = ov.get('avg_nRMSE')
                r[f'{arm}_final_n_old'] = ov.get('final_n')
        for k in ('ETKF', 'ETKF-medium', 'GPR'):
            if isinstance(old.get(ep, {}).get(k), dict):
                r[f'{k}_carried'] = old[ep][k].get('avg_nRMSE')
        rows.append(r)
    def col(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(v)) if v else None
    rep['p1'] = {
        'n_episodes': len(rows),
        'mean_WS-P0_old': col('WS-P0_old'), 'mean_WS-P0_new': col('WS-P0_new'),
        'mean_WS-P1_old': col('WS-P1_old'), 'mean_WS-P1_new': col('WS-P1_new'),
        'adaptive_n_gain_old': (col('WS-P0_old') - col('WS-P1_old'))
        if col('WS-P0_old') and col('WS-P1_old') else None,
        'adaptive_n_gain_new': (col('WS-P0_new') - col('WS-P1_new'))
        if col('WS-P0_new') and col('WS-P1_new') else None,
        'P1_better_than_P0_new': f"{sum(1 for r in rows if r.get('WS-P1_new') is not None and r.get('WS-P0_new') is not None and r['WS-P1_new'] < r['WS-P0_new'])}/{len(rows)}",
        'P1_better_than_P0_old': f"{sum(1 for r in rows if r.get('WS-P1_old') is not None and r.get('WS-P0_old') is not None and r['WS-P1_old'] < r['WS-P0_old'])}/{len(rows)}",
        'rows': rows,
    }

# ---------------- raydrop --------------------------------------------------
p = os.path.join(HERE, 'raydrop_comparison_rerun.json')
if os.path.exists(p):
    new = json.load(open(p))
    old = json.load(open(os.path.join(OLD, 'raydrop_comparison.json')))
    rows = []
    for key, v in new.items():
        r = {'pair': key}
        for model in ('uniform', 'slope'):
            if model in v and 'seed_results' in v[model].get('WS-P1', {}):
                r[f'WS_{model}_new'] = med(v[model]['WS-P1']['seed_results'])[0]
            ov = old.get(key, {}).get(model, {})
            if isinstance(ov.get('WS-P1'), dict):
                r[f'WS_{model}_old'] = ov['WS-P1'].get('avg_nRMSE')
            if isinstance(ov.get('ETKF'), dict):
                r[f'ETKF_{model}_carried'] = ov['ETKF'].get('avg_nRMSE')
        for tag in ('new', 'old'):
            a, b = r.get(f'WS_slope_{tag}'), r.get(f'WS_uniform_{tag}')
            if a is not None and b is not None:
                r[f'slope_minus_uniform_{tag}'] = a - b
        rows.append(r)
    def dd(tag):
        v = [r[f'slope_minus_uniform_{tag}'] for r in rows if r.get(f'slope_minus_uniform_{tag}') is not None]
        return {'median': float(np.median(v)), 'slope_worse': f'{sum(x > 0 for x in v)}/{len(v)}'} if v else None
    rep['raydrop'] = {'n_pairs': len(rows), 'slope_minus_uniform_old': dd('old'),
                      'slope_minus_uniform_new': dd('new'), 'rows': rows}

with open(os.path.join(HERE, 'stage2_comparison.json'), 'w') as f:
    json.dump(rep, f, indent=2)
print(json.dumps({k: (v if k != 'p1' and k != 'raydrop' else
                      {kk: vv for kk, vv in v.items() if kk != 'rows'})
                  for k, v in rep.items()}, indent=2))
