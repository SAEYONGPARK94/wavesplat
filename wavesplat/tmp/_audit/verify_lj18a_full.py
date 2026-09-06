#!/usr/bin/env python3
"""★`LJ18a` **전장(full-length)** 다운로드 무결성 확인 — ⛔가운데 프레임까지 읽는다.

⛔**판정 도구가 아니다.** 어떤 배치의 verdict 도 바꾸지 않는다. 추정기 런 0회.

## 왜 이 스크립트가 있는가

2026-09-05 에 같은 파일을 **append(`>>`) 기반으로 이어받아** 22.4 GB(정본의 **129%**)를
만들었다. 그런데 파일이 **열렸고 `Z[0]` 도 읽혔다** — 손상은 **가운데 프레임**에서
`wrong B-tree signature` 로 드러났다.
⟹ ★***맨 앞 프레임만 읽는 확인은 이 부류의 손상을 통과시킨다.***
그래서 이 스크립트는 **앞·가운데·뒤**를 전부 읽고, 가운데를 **여러 지점** 읽는다.

사용
----
    cd /home/vexit/wavesplat/wavesplat
    python3 tmp/_audit/verify_lj18a_full.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PATH = os.path.join(ROOT, 'data/real_bergamasco_full/Surfaces_20180103_093938_Grid1.nc')
WANT = 17371101809          # ifremer 카탈로그의 Content-Length
SHORT = os.path.join(ROOT, 'data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid1_short.nc')


def main():
    os.chdir(ROOT)
    if not os.path.exists(PATH):
        print(f'⛔파일 없음: {PATH}'); return 2
    n = os.path.getsize(PATH)
    print(f'크기 {n:,} / 기대 {WANT:,}  ({100*n/WANT:.3f}%)')
    if n != WANT:
        print('⛔크기 불일치 — 아래 열람은 의미가 없다. 재개하거나 다시 받아라.')
        return 1
    print('✅크기 일치\n')

    import h5py
    with h5py.File(PATH, 'r') as f:
        keys = list(f.keys())
        print(f'최상위 키 {keys}')
        # 표면 변수 찾기 (short 파일과 같은 규약을 기대한다)
        name = None
        for cand in ('Z', 'z', 'surface', 'eta'):
            if cand in f:
                name = cand; break
        if name is None:
            big = sorted(((v.size, k) for k, v in f.items()
                          if hasattr(v, 'shape') and len(getattr(v, 'shape', ())) == 3),
                         reverse=True)
            if not big:
                print('⛔3차원 데이터셋을 못 찾았다 — 규약 확인 필요'); return 1
            name = big[0][1]
        d = f[name]
        T = d.shape[0]
        print(f'표면 변수 `{name}` shape={d.shape} dtype={d.dtype}  프레임 {T}\n')

        # ★앞·가운데(여러 점)·뒤 — 가운데를 반드시 포함한다
        probes = sorted(set([0, 1, T // 4, T // 2 - 1, T // 2, T // 2 + 1,
                             3 * T // 4, T - 2, T - 1]))
        print(f"{'frame':>8s}{'finite%':>10s}{'min':>10s}{'median':>10s}{'max':>10s}{'std':>10s}")
        bad = 0
        for i in probes:
            try:
                a = np.asarray(d[int(i)], dtype=np.float64)
            except Exception as e:            # noqa: BLE001 — 손상은 어떤 예외로든 온다
                print(f'{i:8d}  ⛔읽기 실패: {type(e).__name__}: {e}')
                bad += 1
                continue
            fin = np.isfinite(a)
            if not fin.any():
                print(f'{i:8d}{0.0:10.2f}   ⛔유한 픽셀 0')
                bad += 1
                continue
            v = a[fin]
            print(f'{i:8d}{100*fin.mean():10.2f}{v.min():10.3f}{np.median(v):10.3f}'
                  f'{v.max():10.3f}{v.std():10.3f}')
        print()
        if bad:
            print(f'⛔{bad}/{len(probes)} 프레임에서 실패 — **손상이다.** 지우고 다시 받아라.')
            return 1
        print(f'✅{len(probes)}/{len(probes)} 프레임 정상 (가운데 3점 포함)')

        # 짧은 판본과 겹치는 프레임이 있으면 대조 (있을 때만)
        if os.path.exists(SHORT):
            with h5py.File(SHORT, 'r') as g:
                if name in g and g[name].shape[1:] == d.shape[1:]:
                    a0 = np.asarray(d[0], np.float64); b0 = np.asarray(g[name][0], np.float64)
                    m = np.isfinite(a0) & np.isfinite(b0)
                    if m.any():
                        print(f'\n짧은 판본 대조(프레임 0, 공통 유한 {int(m.sum())} px): '
                              f'max|Δ| = {np.abs(a0[m]-b0[m]).max():.3e}')
                        print('⚠같은 시각이 아닐 수 있다 — 값이 다르면 그것만으로 손상이 아니다.')
                else:
                    print(f'\n짧은 판본은 격자/변수 규약이 달라 대조 생략'
                          f' (short shape={g[name].shape if name in g else "n/a"})')
    print('\n⛔이 출력은 어떤 배치의 판정도 바꾸지 않는다 — 파일 무결성만 본다.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
