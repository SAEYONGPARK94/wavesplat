#!/usr/bin/env python3
"""`T19` **동결 전 프로브 P-A** — 채점 격자가 오라클을 지탱하는가.

⛔**판정이 아니다.** 사전등록 미동결, 게이트·분석기·러너 0개, 추정기 런 **0회**.
적대적 심사(2026-09-06)가 낸 BLOCKER 세 건 — **B-1(NaN)** · **B-2(N_MODE=25 도달 불가)** ·
**M-6(Nyquist 를 틀린 격자에서 잰다)** — 을 **한 번에** 확정하기 위한 것이다.

재는 것 (72셀 = 12 레코드 × 창 6)
--------------------------------
채점 표적 `target = scene_station(...)['maps_gt'][0]` (4 m 질의격자) 에 대해:
  · `shape` · 유한 픽셀 수 · **NaN 수**
  · `np.fft.fft2(target)` 이 **NaN 을 내는가** (⟹ `oracle()` 의 파워 정렬이 무너진다)
  · **사용 가능한 켤레류 수** = `oracle()` 의 선택 루프가 실제로 모을 수 있는 최대 모드 수
  · `len(ks)` — 등록값 `N_MODE=25` 를 요청했을 때 **실제로 잡히는 수**
  · `k_max = π / Δq` (질의격자의 Nyquist) 와 `Δq`

⛔이 스크립트는 `T11` 의 `oracle()` 을 **호출하지 않는다** — 그 선택 루프만 **같은 논리로**
다시 돌려 개수를 센다(적합·lstsq 0회). 논리 동일성은 `run_t11.py:126-141` 대조로 유지한다.

사용
----
    cd /home/vexit/wavesplat/wavesplat
    systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 \
        python3 tmp/T19_REALORACLE_2026-09-05/probe_A_grid.py
"""
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp/B2_REAL_2026-08-23'),
          os.path.join(ROOT, 'tmp/T17_REALSCALE_2026-09-05')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

N_MODE_REG = 25          # T19 PREREG §5.2 의 등록값


def classes_and_len(target, nmode):
    """`run_t11.oracle()` 의 선택 루프와 **같은 논리**로 (사용가능 켤레류 수, len(ks)) 를 센다."""
    F = np.fft.fft2(target)
    P = np.abs(F) ** 2
    nan_in_P = bool(np.isnan(P).any())
    N0, N1 = P.shape
    order = np.argsort(P.ravel())[::-1]
    seen, n_ks = set(), 0
    for o in order:
        i, j = np.unravel_index(o, P.shape)
        if i == 0 and j == 0:
            continue
        key = min((int(i), int(j)), (int((N0 - i) % N0), int((N1 - j) % N1)))
        if key in seen:
            continue
        seen.add(key)
        n_ks += 1
        if n_ks == nmode:
            break
    # 전체 사용가능 켤레류 수 (상한 없이)
    seen_all = set()
    for i in range(N0):
        for j in range(N1):
            if i == 0 and j == 0:
                continue
            seen_all.add(min((i, j), ((N0 - i) % N0, (N1 - j) % N1)))
    return nan_in_P, len(seen_all), n_ks


def main():
    warnings.filterwarnings('ignore')
    import gates_t17 as G
    from scene_station import scene_station
    os.chdir(ROOT)

    rows = []
    print(f"{'cell':14s}{'shape':>9s}{'npx':>6s}{'nan':>6s}{'finite':>7s}"
          f"{'fftNaN':>7s}{'classes':>8s}{'len(ks)':>8s}{'dq[m]':>7s}{'kmax':>7s}")
    for rec in G.RECORDS:
        for win in G.WINS:
            sc = scene_station(rec, win=win, W=G.W_OBS, m_scan=G.M_SCAN,
                               m_cap=G.M_CAP, noise=G.NOISE, cap_mode=G.CAP_MODE)
            tgt = np.asarray(sc['maps_gt'][0], float)
            gxq, gyq = np.asarray(sc['gxq'], float), np.asarray(sc['gyq'], float)
            dq = float(min(gxq[1] - gxq[0], gyq[1] - gyq[0]))
            nan = int(np.isnan(tgt).sum())
            fin = int(np.isfinite(tgt).sum())
            fftnan, ncls, nks = classes_and_len(tgt, N_MODE_REG)
            kmax = float(np.pi / dq)
            r = dict(record=rec, win=int(win), shape=list(tgt.shape),
                     npx=int(tgt.size), n_nan=nan, n_finite=fin,
                     fft_nan=fftnan, n_classes=int(ncls), n_ks_at_25=int(nks),
                     dq=dq, k_max=kmax)
            rows.append(r)
            print(f"{rec+'|w'+str(win):14s}{str(tuple(tgt.shape)):>9s}{tgt.size:6d}"
                  f"{nan:6d}{fin:7d}{str(fftnan):>7s}{ncls:8d}{nks:8d}"
                  f"{dq:7.2f}{kmax:7.3f}", flush=True)

    out = os.path.join(HERE, 'probe_A_grid.jsonl')
    with open(out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    n_nan_cells = sum(1 for r in rows if r['n_nan'] > 0)
    n_fftnan = sum(1 for r in rows if r['fft_nan'])
    n_short = sum(1 for r in rows if r['n_ks_at_25'] < N_MODE_REG)
    print(f'\n{"="*78}')
    print(f'★B-1  NaN 을 가진 셀        : {n_nan_cells}/72')
    print(f'★B-1  fft2 가 NaN 인 셀     : {n_fftnan}/72   '
          f'⟹ 이 셀들에서 oracle() 의 파워 정렬은 인덱스 순서가 된다')
    print(f'★B-2  N_MODE=25 미달 셀     : {n_short}/72   '
          f'(실사용 모드 수 최소 {min(r["n_ks_at_25"] for r in rows)})')
    print(f'★M-6  질의격자 Nyquist k_max: '
          f'{min(r["k_max"] for r in rows):.3f} – {max(r["k_max"] for r in rows):.3f} rad/m')
    print(f'\n{len(rows)} 행 → {out}')
    print('⛔동결 전 프로브다 — 어떤 주장도 지탱하지 않는다.')


if __name__ == '__main__':
    main()
