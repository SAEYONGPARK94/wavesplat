"""N5 — 100ep 위상수정 원자료를 CDIP 관측 확산 기준으로 층화 (대칭감사).

⛔ **새 런 0건.** 기존 결과에 외부 기준으로 가중치를 붙이는 재가중(re-weighting)이지
새 성능 측정이 아니다. 네 방법 전부 **동일 층화·동일 프로토콜**로 처리한다.
⛔ 층화 임계는 CDIP 실측의 min / p05 로 **동결**한다(재선택 = p-hacking, 토론② C3).
"""
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
WS = os.path.join(ROOT, 'tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl')
BL = os.path.join(ROOT, 'tmp/rerun_2026-07-25/batchA_baselines.jsonl')

sup = json.load(open(os.path.join(HERE, 'cdip_support_raw.json')))
spread = {s['episode']: s['spread_total_deg'] for s in sup['synthetic']}
ncomp = {s['episode']: s['n_components'] for s in sup['synthetic']}
real = np.concatenate([np.asarray(r['spread_total_deg'], float) for r in sup['real']])
RMIN, RP05 = float(real.min()), float(np.percentile(real, 5))


def med(path):
    d = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for line in open(path):
        r = json.loads(line)
        ep = r['episode'].replace('episode_', '').replace('.h5', '')
        d[r.get('method', 'WS')][ep]['nRMSE'].append(r['nRMSE'])
        d[r.get('method', 'WS')][ep]['SS'].append(r['Skill_Score'])
    return {m: {e: {k: float(np.median(v)) for k, v in kv.items()} for e, kv in eps.items()}
            for m, eps in d.items()}


M = med(WS)
M.update(med(BL))
METHODS = ['WS', 'ETKF', 'LETKF', 'GPR-1']
BANDS = [('all', lambda s: True),
         ('observed_ge_min', lambda s: s >= RMIN),
         ('core_ge_p05', lambda s: s >= RP05),
         ('never_observed_lt_min', lambda s: s < RMIN)]

out = {'thresholds_deg': {'real_min': RMIN, 'real_p05': RP05},
       'n_real_records': int(real.size), 'note': 're-weighting of existing runs; no new runs',
       'bands': {}}
for name, f in BANDS:
    eps = [e for e in M['WS'] if e in spread and f(spread[e])]
    row = {'n_episodes': len(eps),
           'composition_by_n_components': {int(n): sum(1 for e in eps if ncomp[e] == n)
                                           for n in sorted({ncomp[e] for e in eps})}}
    for m in METHODS:
        v = [M[m][e]['nRMSE'] for e in eps if e in M.get(m, {})]
        s = [M[m][e]['SS'] for e in eps if e in M.get(m, {})]
        row[m] = {'mean_nRMSE': float(np.mean(v)), 'mean_SS': float(np.mean(s)), 'n': len(v)}
    for base in ('ETKF', 'LETKF'):
        d = np.array([M['WS'][e]['nRMSE'] - M[base][e]['nRMSE'] for e in eps if e in M[base]])
        row[f'WS_vs_{base}'] = {
            'rel_improvement_pct': float(100 * (row[base]['mean_nRMSE'] - row['WS']['mean_nRMSE'])
                                         / row[base]['mean_nRMSE']),
            'WS_better': f'{int((d < 0).sum())}/{len(d)}',
            'sign_p': float(stats.binomtest(int((d < 0).sum()), len(d)).pvalue),
            'wilcoxon_p': float(stats.wilcoxon(d).pvalue) if len(d) > 5 else None}
    out['bands'][name] = row

with open(os.path.join(HERE, 'stratified_by_cdip.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
