"""R2-P1 판정 집계 — `r2_support.jsonl` 에서 **결정적 대비만** 뽑는다.

부분 자료에서도 돌아간다(레짐별로 짝이 맞는 에피소드만 쓴다).
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, 'r2_support.jsonl')
SHIP_KEYS = ['10_20', '20_30', '30_50', '50_75', '75_100', '100_150', '150_200', '200_inf']
OBS_KEYS = ['0_5', '5_10', '10_20', '20_30', '30_50', '50_75', '75_100', '100_150', '150_inf']
OBSC = ['o0_5', 'o5_15', 'o15_40', 'o40_inf']
RHO = ['rho0', 'rho1', 'rho2_3', 'rho4_7', 'rho8+']
S_KEYS = ['s<20', 's20_40', 's40_70', 's70_110', 's110_160', 's160+']
Q_KEYS = ['q0_20', 'q20_50', 'q50_100', 'q100+']

rows = [json.loads(l) for l in open(P)]


def rmse(b):
    return np.sqrt(b[0] / b[1]) if b and b[1] else np.nan


def block(rg):
    sub = [r for r in rows if r['regime'] == rg]
    base = {r['episode']: r for r in sub if r['cell'] == 'SHIP'}
    T = {r['episode']: r for r in sub if r['cell'] == 'NT'}
    return sorted(set(base) & set(T)), base, T


def line(ks, base, T, bk, label, ind=2):
    a = np.array([rmse(T[k]['bins'].get(bk)) for k in ks])
    b = np.array([rmse(base[k]['bins'].get(bk)) for k in ks])
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        print(' ' * ind + f'{label:<26} (n<3)')
        return None
    rel = 100 * (a[m].mean() - b[m].mean()) / b[m].mean()
    med = 100 * np.median((a[m] - b[m]) / b[m])
    w = int(np.sum(a[m] > b[m]))
    p = stats.binomtest(w, int(m.sum()), 0.5).pvalue
    area = np.mean([base[k]['bins'].get(bk, [0, 0, 0])[1] / base[k]['n_frames'] / 4800
                    for k in ks])
    print(' ' * ind + f'{label:<26} 면적{100*area:>5.2f}%  S={b[m].mean():.4f} '
          f'NT={a[m].mean():.4f}  Δ={a[m].mean()-b[m].mean():+.5f}  '
          f'rel={rel:+8.1f}%  med={med:+8.1f}%  손상 {w:>2}/{m.sum()}  p={p:.3g}')
    return dict(rel=rel, med=med, w=w, n=int(m.sum()), S=b[m].mean(), T=a[m].mean(),
                d=a[m].mean() - b[m].mean(), area=area)


def main():
    print(f'# n_rows={len(rows)}')
    for rg in ('crowded', 'sparse'):
        ks, base, T = block(rg)
        if len(ks) < 3:
            print(f'\n[{rg}] 짝 {len(ks)} — 생략')
            continue
        th = [base[k]['meta']['theta'] for k in ks]
        print(f'\n{"="*104}\n[{rg}]  NT − SHIP   짝 n={len(ks)}  '
              f'θ 범위 [{min(th):+.1f}, {max(th):+.1f}]°  '
              f'프레임당 반환점 {np.mean([base[k]["meta"]["n_obs"] for k in ks]):.1f}')

        print('\n R1 배-거리 고리 (BS-3 재현)')
        for k in SHIP_KEYS:
            line(ks, base, T, 'S:' + k, k)
        print('\n R2 최근접-관측거리 셸 (= 저장소가 쓴 support 자)')
        for k in OBS_KEYS:
            line(ks, base, T, 'O:' + k, k)
        print('\n ★R5 국소 관측밀도 ρ₁₀ (= 진짜 관측률)')
        for k in RHO:
            line(ks, base, T, 'R:' + k, k)

        print('\n ★★★R3 2D 교차표 — 배-거리 고정, 관측거리만 변화')
        for kr in SHIP_KEYS[:5]:
            print(f'   [배-거리 {kr}]')
            for ko in OBSC:
                line(ks, base, T, f'X:{kr}|{ko}', ko, ind=5)
        print('\n ★★★R4 FOV 콘 자연실험 — 같은 배-거리, 콘 안(관측) vs 콘 밖(영구 미관측)')
        for kr in SHIP_KEYS[:6]:
            print(f'   [배-거리 {kr}]')
            for side in ('in', 'out'):
                line(ks, base, T, f'C:{kr}|{side}', f'cone-{side}', ind=5)

        print('\n ★★★D4/D5 같은 배-거리 고리를 두 자로 반씩 가른다')
        print(f'   {"고리":<10}{"dobs 하위 rel":>15}{"dobs 상위 rel":>15}{"Δ(지지)":>11}'
              f'{"s 하위 rel":>13}{"s 상위 rel":>13}{"Δ(경과)":>11}')
        for kr in SHIP_KEYS:
            def q(bk):
                a = np.array([rmse(T[k]['bins'].get(bk)) for k in ks])
                b = np.array([rmse(base[k]['bins'].get(bk)) for k in ks])
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 3:
                    return None
                return 100 * (a[m].mean() - b[m].mean()) / b[m].mean()
            dl, dh = q(f'RD:{kr}|d_lo'), q(f'RD:{kr}|d_hi')
            sl, sh = q(f'RS:{kr}|s_lo'), q(f'RS:{kr}|s_hi')
            if None in (dl, dh, sl, sh):
                continue
            print(f'   {kr:<10}{dl:>14.1f}%{dh:>14.1f}%{dh-dl:>+10.1f}pp'
                  f'{sl:>12.1f}%{sh:>12.1f}%{sh-sl:>+10.1f}pp')

        print('\n ★D1 파향축 s 밴드 (경과시간 자)')
        for k in S_KEYS:
            line(ks, base, T, 'D:' + k, k)
        print('\n ★D2 파향 직교축 |q| 밴드')
        for k in Q_KEYS:
            line(ks, base, T, 'Q:' + k, k)

        print('\n ★D3 s × |q| — s 고정하고 횡거리만 (경과시간 고정, 거리 변화)')
        for a_ in S_KEYS[:4]:
            print(f'   [{a_}]')
            for b_ in Q_KEYS:
                line(ks, base, T, f'SQ:{a_}|{b_}', b_, ind=5)

        # ── 면적가중 ΔMSE 예산: 이득 쪽 / 손해 쪽이 각각 어디에 있는가 ──
        def budget(keys, pref, title):
            print(f'\n ★면적가중 ΔMSE 예산 — {title}   (양수 = 추적기가 이득)')
            tot = 0.0
            for k in keys:
                bk = pref + k
                a = np.array([rmse(T[e]['bins'].get(bk)) for e in ks])
                b = np.array([rmse(base[e]['bins'].get(bk)) for e in ks])
                fr = np.array([base[e]['bins'].get(bk, [0, 0, 0])[1]
                               / base[e]['n_frames'] / 4800 for e in ks])
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 3:
                    continue
                v = float(np.mean((a[m] ** 2 - b[m] ** 2) * fr[m]))
                tot += v
                print(f'   {k:<12} 면적{100*fr.mean():>6.2f}%   ΔMSE·a = {v:+.4e}')
            print(f'   {"합":<12}                ΔMSE·a = {tot:+.4e}'
                  f'   ({"NT 우세" if tot < 0 else "SHIP 우세"})')
        budget(RHO, 'R:', '국소 관측밀도 ρ₁₀')
        budget([f'{k}|{s}' for k in SHIP_KEYS for s in ('in', 'out')], 'C:',
               '배-거리 × FOV 콘')

        # 팔별 절대 프로파일 (경사를 누가 만드는가)
        print('\n ★팔별 절대 프로파일 (배-거리)')
        for nm, D in (('SHIP', base), ('NT', T)):
            v = [np.nanmean([rmse(D[k]['bins'].get('S:' + kk)) for k in ks])
                 for kk in SHIP_KEYS]
            print(f'   {nm:<5} ' + ' '.join(f'{x:.4f}' for x in v)
                  + f'   ramp far/near = {v[-1]/v[0]:.2f}x')


if __name__ == '__main__':
    main()
