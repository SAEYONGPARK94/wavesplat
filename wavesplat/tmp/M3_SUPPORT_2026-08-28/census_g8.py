"""★선별 술어 ③ 의 **전수 census** (AMENDMENT_1 부록 B, 2026-08-29).

선언 시드 풀 전체 × 6칸을 `gates_m3.hs4std_ratio` **로만** 판정한다 — 레이캐스트 0회 ·
GT 파일 0개 · 추정 0회. 목적은 *"규칙을 넣었을 때 몇 개를 건너뛰는가"* 를 **기록에 남기는 것**
이고, ⛔**문턱을 고치는 근거가 아니다**(그것은 기각된 처방 (나) 다 — `AMENDMENT_1.md` §A3·P3).

    python3 tmp/M3_SUPPORT_2026-08-28/census_g8.py > tmp/M3_SUPPORT_2026-08-28/census_g8_2026-08-29.txt
"""
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import gates_m3 as G                                           # noqa: E402


def main():
    t0 = time.time()
    print('# M3 선별 술어 ③ 전수 census — |4·std/Hs − 1| ≤ HS4STD_MAX_DEV')
    print(f'# 상수 HS4STD_MAX_DEV = {G.HS4STD_MAX_DEV} (⛔불변 — G-8 과 같은 심볼)')
    print(f'# 풀 = SEED_POOL[{G.SEED_POOL[0]}..{G.SEED_POOL[-1]}] ({len(G.SEED_POOL)}시드) '
          f'× {len(G.CELL_ORDER)}셀 × {len(G.NCOMPS)}ncomp')
    print('# ⛔추정 0회 · 레이캐스트 0회 · 결과 파일 0회 열람\n')
    n_cand = n_bad = 0
    rows = []
    for cell in G.CELL_ORDER:
        for nc in G.NCOMPS:
            r = [G.hs4std_ratio(cell, nc, s) for s in G.SEED_POOL]
            dev = [abs(x - 1) for x in r]
            bad = [s for s, d in zip(G.SEED_POOL, dev) if d > G.HS4STD_MAX_DEV]
            n_cand += len(r)
            n_bad += len(bad)
            rows.append((cell, nc, st.stdev(r), max(dev), bad,
                         G.FROZEN_SEEDS[f'{cell}|{nc}']))
    w = f'{"셀":<8}{"ncomp":>6}{"sd(비)":>10}{"max|dev|":>10}{"0.25/sd":>9}  거부시드'
    print(w)
    print('-' * 78)
    for cell, nc, sd, mx, bad, _ in rows:
        print(f'{cell:<8}{nc:>6}{sd:>10.4f}{mx:>10.4f}{G.HS4STD_MAX_DEV / sd:>8.2f}σ  {bad}')
    print('-' * 78)
    print(f'후보 {n_cand}개 · **거부 {n_bad}개 ({100 * n_bad / n_cand:.2f}%)** '
          f'· {time.time() - t0:.2f} s')
    print('\n# 선별된 8시드 (거부분을 건너뛴 뒤 풀 순서상 앞의 8)')
    for cell, nc, _, _, _, sel in rows:
        print(f'  {cell:<8}{nc:>3}comp  {list(sel)}')
    print(f'\n# FROZEN_REJECTED = {G.FROZEN_REJECTED}')
    print(f'# SEEDS_CONSUMED  = {G.SEEDS_CONSUMED[0]}–{G.SEEDS_CONSUMED[-1]} '
          f'({len(G.SEEDS_CONSUMED)}개, 거부분 포함)')
    print('\n# ⚠읽기 주의 (AMENDMENT_1 §A2d·§B3):')
    print('#   · SPR5 는 실현 파고 **산포**가 30° 셀의 2.6–3.5배인 셀이다(위치 편향은 거의 없다).')
    print('#   · 그래서 같은 상수가 셀마다 다른 엄격도다 — SPR5|30 은 2.65σ, DIR180|15 는 14.3σ.')
    print('#   · SPR5|15 는 거부 0건이지만 풀 최대가 문턱을 **스친다**(0.2397).')
    print('#   · ⛔이 census 는 문턱을 옮기는 근거가 아니라 **표본을 고르는 근거**다(§P3).')
    verify()


def verify():
    """★처방도 검증하라 — **생성 전 예측 vs 저장 파일 실측**을 48건 전수 대조한다.

    선별 술어 ③ 이 `G-8` 과 같은 것을 재는지는 산문이 아니라 이 대조가 답한다.
    (`diag_g8_2026-08-29.py [B]` 는 **진단 시점 표본**(1695–1702)만 보므로, 부록 B 로 새로
    생성된 `SPR5|30|1703` 은 여기서만 검산된다.)
    """
    import h5py
    import numpy as np
    print('\n# ★생성 전 예측 vs 저장 GT 실측 (48/48 전수) — "처방도 검증하라"')
    worst = (0.0, None)
    n = 0
    for cell in G.CELL_ORDER:
        for nc in G.NCOMPS:
            for sd in G.FROZEN_SEEDS[f'{cell}|{nc}']:
                path = os.path.join(G.cell_dir(cell), G.fname(cell, nc, sd))
                if not os.path.exists(path):
                    print(f'#   ⚠미생성 {cell}|{nc}|{sd}')
                    continue
                with h5py.File(path, 'r') as h:
                    meas = 4 * float(np.std(h['frames/gt_wave_field'][:]))
                pred = G.hs4std_ratio(cell, nc, sd) * G.bf4_seastate()[0]
                d = abs(pred - meas)
                n += 1
                if d > worst[0]:
                    worst = (d, (cell, nc, sd))
                if sd == 1703:
                    print(f'#   ★신규 {cell}|{nc}|{sd}: 예측 {pred:.6f} · 실측 {meas:.6f} '
                          f'· |Δ| {d:.3e} · |dev| {abs(meas / G.bf4_seastate()[0] - 1):.4f}')
    print(f'#   전수 {n}/{G.N_EP} · max|예측 − 실측| = {worst[0]:.3e} @ {worst[1]}')
    assert n == G.N_EP and worst[0] < 1e-12, (n, worst)
    print('#   ⟹ ★선별 술어 ③ 과 `G-8` 은 **같은 양을 잰다**(부동소수 수준으로 동일).')


if __name__ == '__main__':
    main()
