"""지지집합 비교 — 서술 통계만. ⛔성능 주장 없음."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'cdip_support_raw.json')))

real = d['real']
syn = d['synthetic']

R = {k: np.concatenate([np.asarray(r[k], float) for r in real])
     for k in ('Hs', 'Tp', 'spread_band_deg', 'spread_total_deg')}
S = {k: np.array([s[k] for s in syn], float)
     for k in ('Hs', 'Tp', 'spread_band_deg', 'spread_total_deg', 'n_components')}

out = {'n_real_records': int(len(R['Hs'])), 'n_real_stations': len(real),
       'n_synthetic_episodes': len(syn), 'scope': d['scope'], 'estimator': d['estimator']}

def q(a):
    a = a[np.isfinite(a)]
    return {'n': int(a.size), 'min': float(a.min()), 'p05': float(np.percentile(a, 5)),
            'p25': float(np.percentile(a, 25)), 'median': float(np.median(a)),
            'p75': float(np.percentile(a, 75)), 'p95': float(np.percentile(a, 95)),
            'max': float(a.max()), 'mean': float(a.mean())}

out['marginals'] = {k: {'real': q(R[k]), 'synthetic': q(S[k])}
                    for k in ('Hs', 'Tp', 'spread_band_deg', 'spread_total_deg')}

# --- 피복률: 실측 레코드가 합성 주변부 상자 안에 있는가 (marginal box) ---
def box_cover(keys):
    m = np.ones(len(R['Hs']), bool)
    for k in keys:
        lo, hi = np.nanmin(S[k]), np.nanmax(S[k])
        m &= (R[k] >= lo) & (R[k] <= hi)
    return float(m.mean()), {k: [float(np.nanmin(S[k])), float(np.nanmax(S[k]))] for k in keys}

for name, keys in [('Hs', ['Hs']), ('Tp', ['Tp']), ('Hs+Tp', ['Hs', 'Tp']),
                   ('spread_total', ['spread_total_deg']),
                   ('Hs+Tp+spread_total', ['Hs', 'Tp', 'spread_total_deg'])]:
    frac, box = box_cover(keys)
    out.setdefault('real_inside_synthetic_box', {})[name] = {'fraction': frac, 'synthetic_box': box}

# --- 역방향: 합성 에피소드가 실측 5-95% 구간 안에 있는가 ---
rev = {}
for k in ('Hs', 'Tp', 'spread_total_deg'):
    lo, hi = np.percentile(R[k], 5), np.percentile(R[k], 95)
    rev[k] = {'real_p05_p95': [float(lo), float(hi)],
              'synthetic_fraction_inside': float(((S[k] >= lo) & (S[k] <= hi)).mean())}
out['synthetic_inside_real_p05p95'] = rev

# --- 성분 수별 합성 전대역 확산 (기술만; n_components 재정의 아님) ---
byn = {}
for n in sorted(set(S['n_components'].astype(int))):
    m = S['n_components'] == n
    byn[int(n)] = {'n_episodes': int(m.sum()),
                   'spread_total_deg_median': float(np.median(S['spread_total_deg'][m])),
                   'spread_band_deg_median': float(np.median(S['spread_band_deg'][m]))}
out['synthetic_by_n_components'] = byn

# --- 실측 확산 분포에서 합성 중앙값의 분위 위치 ---
out['synthetic_median_percentile_in_real'] = {
    k: float((R[k] < np.median(S[k])).mean() * 100)
    for k in ('Hs', 'Tp', 'spread_total_deg')}

with open(os.path.join(HERE, 'support_summary.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2)[:4000])
