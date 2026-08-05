"""사전등록(PREREG.md) 그대로의 분석. 검정 선택·층화 임계 재선택 없음.

⛔#16 준수: 모든 p 값에 **검정명**을 붙인다.
"""
import json
import os
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, 'relax_extctrl.json')))
EPS = D['episodes']

# 08-03 에 CDIP 로 동결된 층화 (재선택 금지): n_comp>=5 <=> 관측된 확산 >=12.18 deg
ON = [e for e in EPS if any(f'_{c}comp' in e for c in (5, 15, 30))]
OFF = [e for e in EPS if e not in ON]


def sign_test_two_sided(wins, n):
    """부호검정 양측. wins = 한쪽이 이긴 횟수, n = 동점 제외 짝수."""
    if n == 0:
        return 1.0
    k = max(wins, n - wins)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilcoxon(d):
    """Wilcoxon signed-rank, 정규근사(타이 보정 없음). n<10 이면 근사임을 명기."""
    d = np.asarray([x for x in d if x != 0], float)
    n = len(d)
    if n == 0:
        return None, 0
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    w = min(r[d > 0].sum(), r[d < 0].sum())
    mu = n * (n + 1) / 4
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sd == 0:
        return None, n
    z = (w - mu) / sd
    from math import erfc
    return float(erfc(abs(z) / np.sqrt(2))), n


def med(ep, arm, key):
    v = EPS[ep][arm][key]
    return None if v is None else v['median']


def compare(arm_a, arm_b, key, eps, label):
    """arm_a 를 arm_b 에 대해 짝지어 검정. 음수 delta = arm_a 가 더 좋다(nRMSE)."""
    rows, deltas, wins = [], [], 0
    for ep in eps:
        a, b = med(ep, arm_a, key), med(ep, arm_b, key)
        if a is None or b is None:
            continue
        d = a - b
        deltas.append(d)
        wins += (d < 0)
        rows.append((ep, b, a, d))
    n = len(deltas)
    p_sign = sign_test_two_sided(wins, n)
    p_wil, n_wil = wilcoxon(deltas)
    print(f'\n── {label}  [{arm_a} vs {arm_b}] · {key} · n={n}')
    print(f"   {'episode':30s} {arm_b:>9s} {arm_a:>9s} {'delta':>9s}")
    for ep, b, a, d in rows:
        print(f'   {ep:30s} {b:9.4f} {a:9.4f} {d:+9.4f}')
    print(f'   중앙 delta = {np.median(deltas):+.4f}   평균 = {np.mean(deltas):+.4f}')
    print(f'   {arm_a} 승 = {wins}/{n}   부호검정(양측) p = {p_sign:.4f}'
          f'   Wilcoxon signed-rank(정규근사) p = '
          f'{"n/a" if p_wil is None else f"{p_wil:.4f}"} (n={n_wil})')
    return {'arm_a': arm_a, 'arm_b': arm_b, 'key': key, 'label': label, 'n': n,
            'wins_a': wins, 'median_delta': float(np.median(deltas)),
            'mean_delta': float(np.mean(deltas)),
            'p_sign_two_sided': p_sign, 'p_wilcoxon_normal_approx': p_wil,
            'rows': [{'episode': e, arm_b: b, arm_a: a, 'delta': d} for e, b, a, d in rows]}


print('=' * 78)
print('구성요소 수준 외부 대조 — RELAX 초기화기 vs 우리 beamforming')
print(f"데이터 {D['_data']} · 시드 {D['_seeds']} (median-of-3) · 팔 {D['_arms']}")
print('=' * 78)

out = {'_prereg': D['_prereg'], 'results': {}}

# ---------- PRIMARY ----------
print('\n' + '#' * 78)
print('# PRIMARY (사전등록) — EXT-M(relax_konly) vs OURS(v2_konly), e2e avg_nRMSE')
print('#' * 78)
out['results']['PRIMARY'] = compare('EXT-M', 'OURS', 'avg_nRMSE', list(EPS), 'PRIMARY 전체 12ep')

# 효과 대 시드잡음 (사전등록 판정 조건 2)
stds = []
for ep in EPS:
    for arm in ('OURS', 'EXT-M'):
        s = EPS[ep][arm]['avg_nRMSE']
        if s:
            stds.append(s['std'])
noise = float(np.median(stds))
eff = abs(out['results']['PRIMARY']['median_delta'])
print(f'\n   시드 std 중앙값 = {noise:.4f}   |중앙 효과| = {eff:.4f}   '
      f'비 = {eff / noise if noise else float("inf"):.2f}배')
out['results']['seed_noise_median_std'] = noise
out['results']['effect_over_noise'] = eff / noise if noise else None

p = out['results']['PRIMARY']['p_sign_two_sided']
verdict = ('PASS' if (p < 0.05 and eff > noise) else 'INCONCLUSIVE (검정력 부족)')
print(f"\n   ★ 사전등록 판정: {verdict}"
      f"   [p<0.05? {p < 0.05} · 효과>잡음? {eff > noise}]")
out['results']['VERDICT'] = verdict

# ---------- SECONDARY ----------
print('\n' + '#' * 78)
print('# SECONDARY-a — 구성요소 본체: init_nrmse (초기화기가 직접 내놓는 것)')
print('#' * 78)
out['results']['SEC_a_init'] = compare('EXT-M', 'OURS', 'init_nrmse', list(EPS), 'init nRMSE 12ep')

print('\n' + '#' * 78)
print('# SECONDARY-b — EXT-F(relax, full k/w/phi) 서술용')
print('#' * 78)
out['results']['SEC_b_full'] = compare('EXT-F', 'OURS', 'avg_nRMSE', list(EPS), 'EXT-F 전체 12ep')

print('\n' + '#' * 78)
print('# SECONDARY-c — 동결 층화 (임계 재선택 없음)')
print('#' * 78)
print(f'  on-support (>=5comp, 관측 확산 >=12.18deg): n={len(ON)}  {ON}')
print(f'  off-support (<5comp, 실측 0건):            n={len(OFF)}  {OFF}')
print('  ⚠ off-support n=5 는 양측 부호검정 바닥 0.0625 ⟹ 사전등록대로 **서술 전용**')
out['results']['SEC_c_on'] = compare('EXT-M', 'OURS', 'avg_nRMSE', ON, 'on-support')
out['results']['SEC_c_off'] = compare('EXT-M', 'OURS', 'avg_nRMSE', OFF, 'off-support (서술 전용)')

# ---------- 보고 의무 ----------
print('\n' + '#' * 78)
print('# 보고 의무 — 지연 / final_n / 시드산포')
print('#' * 78)
print(f"  {'episode':30s} " + ' '.join(f'{a:>18s}' for a in ('OURS', 'EXT-M', 'EXT-F')))
for ep in EPS:
    cells = []
    for arm in ('OURS', 'EXT-M', 'EXT-F'):
        ms = EPS[ep][arm]['avg_inference_ms']
        fn = EPS[ep][arm]['final_n']
        cells.append(f"{ms['median']:6.1f}ms N={fn['median']:4.1f}" if ms else '        n/a')
    print(f'  {ep:30s} ' + ' '.join(f'{c:>18s}' for c in cells))

for arm in ('OURS', 'EXT-M', 'EXT-F'):
    v = [EPS[e][arm]['avg_inference_ms']['median'] for e in EPS if EPS[e][arm]['avg_inference_ms']]
    s = [EPS[e][arm]['avg_nRMSE']['std'] for e in EPS if EPS[e][arm]['avg_nRMSE']]
    print(f'  {arm:6s} per-frame 중앙 {np.median(v):6.2f} ms   시드 std 중앙 {np.median(s):.4f}')

# 발산 점검 (스모크에서 EXT-F 1건 nRMSE>1 관측됨)
print('\n  발산/이상 (any-seed nRMSE > 0.5):')
bad = 0
for ep in EPS:
    for arm in ('OURS', 'EXT-M', 'EXT-F'):
        for s in EPS[ep][arm]['seed_results']:
            if 'error' in s:
                print(f'    ERROR {ep} {arm} seed={s["seed"]}: {s["error"]}'); bad += 1
            elif s.get('avg_nRMSE', 0) > 0.5:
                print(f'    {ep:28s} {arm:6s} seed={s["seed"]:4d} nRMSE={s["avg_nRMSE"]:.3f}'); bad += 1
if not bad:
    print('    없음')

json.dump(out, open(os.path.join(HERE, 'analysis.json'), 'w'), indent=2, default=float)
print(f"\n-> {os.path.join(HERE, 'analysis.json')}")
