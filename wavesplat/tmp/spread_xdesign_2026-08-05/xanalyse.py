"""확산 교차설계 분석 — 사전등록(PREREG.md) 그대로. 검정 재선택 없음.

PRIMARY = 상호작용 대비  I(n,s) = ΔWS(n,s) − ΔLETKF(n,s),  Δ = nRMSE(60°) − nRMSE(30°)
n = 10 짝 (2 성분수 × 5 생성시드). 양측 부호검정 (10/10 → 0.0020 · 9/10 → 0.0215).
"""
import json
import os
from math import comb, erfc

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, 'xdesign.json')))
C = D['cells']
NCOMPS = [5, 30]
SEEDS = [0, 1, 2, 3, 4]


def sign_test(wins, n):
    if n == 0:
        return 1.0
    k = max(wins, n - wins)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def wilcoxon(d):
    d = np.asarray([x for x in d if x != 0], float)
    n = len(d)
    if n == 0:
        return None, 0
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    w = min(r[d > 0].sum(), r[d < 0].sum())
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sd == 0:
        return None, n
    z = (w - n * (n + 1) / 4) / sd
    return float(erfc(abs(z) / np.sqrt(2))), n


def val(arm, nc, sd, method, key):
    c = C.get(f'{arm}|{nc}comp|seed{sd}')
    if not c:
        return None
    v = c[method].get(key)
    if v is None:
        return None
    return v['median'] if isinstance(v, dict) else v


print('=' * 76)
print('확산 교차설계 — 2(n_comp) × 2(명목 확산) × 5(생성시드)')
print('=' * 76)

# ---------- 조작 점검 ----------
man_p = os.path.join(os.path.dirname(HERE), '..', 'data_spread60_2026-08-05', '_manifest.json')
man_p = os.path.abspath(man_p)
if os.path.exists(man_p):
    man = json.load(open(man_p))
    ctrl = man.get('_ctrl_realised_spread_deg', {})
    print('\n★ 조작 점검 — 실현 확산 [deg] (이게 실패하면 이하 전부 무효)')
    for nc in NCOMPS:
        c = [ctrl.get(f'{nc}comp_seed{s}') for s in SEEDS]
        c = [x for x in c if x is not None]
        t = [man.get(f'episode_BF4_{nc}comp_seed{s}_uniform.h5', {}).get('realised_spread_deg')
             for s in SEEDS]
        t = [x for x in t if x is not None]
        if c and t:
            ov = 'OVERLAP' if max(c) > min(t) else '분리'
            print(f'  {nc:2d}comp  30°팔 중앙 {np.median(c):6.2f} [{min(c):.2f},{max(c):.2f}]'
                  f'  →  60°팔 중앙 {np.median(t):6.2f} [{min(t):.2f},{max(t):.2f}]   ({ov})')
    print(f"  대조팔 승계 검증: {man.get('_ctrl_inherit_verified')}")

# ---------- PRIMARY ----------
print('\n' + '#' * 76)
print('# PRIMARY — 상호작용 I = ΔWS − ΔLETKF   (Δ = nRMSE(60°) − nRMSE(30°))')
print('#' * 76)
print(f"  {'cell':14s} {'ΔWS':>9s} {'ΔLETKF':>9s} {'I':>9s}")
I, dws, dlk, rows = [], [], [], []
for nc in NCOMPS:
    for sd in SEEDS:
        a = val('ctrl30', nc, sd, 'WS', 'nRMSE'); b = val('treat60', nc, sd, 'WS', 'nRMSE')
        p = val('ctrl30', nc, sd, 'LETKF', 'nRMSE'); q = val('treat60', nc, sd, 'LETKF', 'nRMSE')
        if None in (a, b, p, q):
            print(f'  {nc}comp/seed{sd}: 결측 — 제외'); continue
        dw, dl = b - a, q - p
        I.append(dw - dl); dws.append(dw); dlk.append(dl)
        rows.append({'n_comp': nc, 'gen_seed': sd, 'dWS': dw, 'dLETKF': dl, 'I': dw - dl})
        print(f'  {nc:2d}comp/seed{sd}  {dw:+9.4f} {dl:+9.4f} {dw - dl:+9.4f}')

n = len(I)
wins = sum(1 for x in I if x > 0)
p_sign = sign_test(wins, n)
p_wil, _ = wilcoxon(I)
print(f'\n  중앙 ΔWS   = {np.median(dws):+.4f}   (확산↑ 시 WS 열화)')
print(f'  중앙 ΔLETKF= {np.median(dlk):+.4f}   (확산↑ 시 LETKF 열화)')
print(f'  중앙 I     = {np.median(I):+.4f}')
print(f'  I>0 = {wins}/{n}   부호검정(양측) p = {p_sign:.4f}   '
      f'Wilcoxon signed-rank(정규근사) p = {"n/a" if p_wil is None else f"{p_wil:.4f}"}')
verdict = ('H1 지지 (이산 표현이 확산에 특별히 취약)' if (p_sign < 0.05 and np.median(I) > 0)
           else 'H1 미지지 — ' + ('방향은 H1 이나 검정력 부족' if np.median(I) > 0
                                  else '★H1 반대 방향: (B) 후보가 반증된다'))
print(f'\n  ★ 사전등록 판정: {verdict}')

# ---------- 층별 (서술 전용, n=5 바닥 0.0625) ----------
print('\n' + '#' * 76)
print('# 층별 (서술 전용 — n=5 양측 부호검정 바닥 0.0625)')
print('#' * 76)
strata = {}
for nc in NCOMPS:
    sub = [r for r in rows if r['n_comp'] == nc]
    if not sub:
        continue
    ii = [r['I'] for r in sub]
    w = sum(1 for x in ii if x > 0)
    print(f'  {nc:2d}comp: 중앙 ΔWS {np.median([r["dWS"] for r in sub]):+.4f}  '
          f'중앙 ΔLETKF {np.median([r["dLETKF"] for r in sub]):+.4f}  '
          f'중앙 I {np.median(ii):+.4f}  I>0 {w}/{len(ii)}  p={sign_test(w, len(ii)):.4f}')
    strata[nc] = {'median_I': float(np.median(ii)), 'wins': w, 'n': len(ii),
                  'p_sign': sign_test(w, len(ii))}

# ---------- 수준값과 SECONDARY ----------
print('\n' + '#' * 76)
print('# 수준값 (median-of-3) 및 SECONDARY 패널')
print('#' * 76)
for key in ('nRMSE', 'Skill_Score', 'Directional_R2', 'Variance_Ratio', 'Pearson_rho'):
    print(f'\n  [{key}]')
    print(f"    {'n_comp':>7s} {'WS 30°':>9s} {'WS 60°':>9s} {'LK 30°':>9s} {'LK 60°':>9s}")
    for nc in NCOMPS:
        cells = []
        for m in ('WS', 'LETKF'):
            for arm in ('ctrl30', 'treat60'):
                v = [val(arm, nc, s, m, key) for s in SEEDS]
                v = [x for x in v if x is not None]
                cells.append(np.median(v) if v else np.nan)
        print(f'    {nc:7d} {cells[0]:9.4f} {cells[1]:9.4f} {cells[2]:9.4f} {cells[3]:9.4f}')

# ---------- 시드잡음 대비 ----------
stds = []
for k, c in C.items():
    s = c['WS'].get('nRMSE')
    if isinstance(s, dict):
        stds.append(s['std'])
noise = float(np.median(stds)) if stds else float('nan')
print(f'\n  WS 시드 std 중앙 = {noise:.4f}   |중앙 ΔWS| = {abs(np.median(dws)):.4f} '
      f'({abs(np.median(dws)) / noise:.2f}배)   |중앙 I| = {abs(np.median(I)):.4f} '
      f'({abs(np.median(I)) / noise:.2f}배)')

json.dump({'rows': rows, 'n': n, 'wins_I_pos': wins, 'p_sign_two_sided': p_sign,
           'p_wilcoxon_normal_approx': p_wil, 'median_I': float(np.median(I)),
           'median_dWS': float(np.median(dws)), 'median_dLETKF': float(np.median(dlk)),
           'verdict': verdict, 'strata': strata, 'ws_seed_noise_median_std': noise},
          open(os.path.join(HERE, 'xanalysis.json'), 'w'), indent=2, default=float)
print(f"\n-> {os.path.join(HERE, 'xanalysis.json')}")
