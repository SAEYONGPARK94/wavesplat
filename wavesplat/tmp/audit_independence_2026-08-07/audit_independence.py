"""① 과거 "N/N 전승" 헤드라인의 실제 독립 단위 감사 (2026-08-07).

동기: Rx-4 에서 블록 C 16 에피소드가 위상 벡터 8개·진폭 벡터 4개뿐이고,
G1P1/Rx-2/B3/rx1 이 쓴 CROWDED4 는 **위상이 넷 다 비트 동일**임이 드러났다.
추적기가 움직이는 파라미터가 정확히 위상·진폭이므로, 추적기 실험 전체가
위상 1종 위에서 측정됐다는 뜻이다.

이 스크립트는 **결과 파일을 전혀 읽지 않는다.** 오직 에피소드 HDF5 의 생성기
파라미터(`config/wave_components`)와 진실면(`frames/gt_wave_field`)만 본다.
따라서 어떤 판정도 여기서 편향될 수 없다.

`config/wave_components` 열 규약 (관측으로 확인):
  col0 = 진폭 A   col1 = 각주파수 ω   col2 = kx   col3 = ky   col4 = 위상 φ   col5 = (설정 스칼라)
"""
import hashlib
import json
import os
import sys

import h5py
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')

CROWDED4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
SPARSE8 = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
           'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
           'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
           'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5']
P1_12 = SPARSE8 + CROWDED4
CROWDED16 = [f'episode_BF{bf}_{nc}comp_seed{s}_uniform.h5'
             for bf in (4, 5) for nc in (15, 30) for s in range(4)]

SETS = [
    ('CROWDED4  (G1P1 / Rx-2 / B3 / rx1 의 "4/4")', CROWDED4),
    ('P1_12     (B2 확장 / p1 비교의 "12")', P1_12),
    ('SPARSE8   (Rx-4 블록 S 의 "8")', SPARSE8),
    ('CROWDED16 (Rx-4 블록 C 의 "16")', CROWDED16),
]

COLS = {'진폭 A': 0, 'omega': 1, '위상 phi': 4}

# ⚠2026-08-07 정정: 처음에 kx(col2)·ky(col3) 를 **바이트**로 세어 "10종" 이라고 적었다.
# 그건 draw 가 아니다 -- 생성기는 방향각 theta 와 크기 |k| 를 따로 뽑아
# kx = |k|cos(theta) 로 저장하고(`wave_generator.py`), BF 사다리는 |k| 만 바꾼다.
# 그래서 **같은 theta draw 가 서로 다른 kx 바이트**를 낳는다. 난수 단위는 theta 이므로
# theta 를 허용오차로 묶어서 센다. P1_12 에서 10 -> **5** 로 줄고, 그 5개 분할은
# 위상 5개 분할과 **정확히 동일**(완전 교락)이다.
TOL = 1e-4


def load(ep):
    with h5py.File(os.path.join(DATA, ep), 'r') as f:
        wc = np.array(f['config/wave_components'])
        fld = np.array(f['frames/gt_wave_field'])
    return wc, fld


def key_of(vec):
    """비트 수준 신원. 길이가 다르면 자동으로 다른 키가 된다."""
    return hashlib.md5(np.ascontiguousarray(vec, dtype='<f8').tobytes()).hexdigest()


def floor_p(c):
    """C 개 독립 클러스터에서 양측 부호검정이 낼 수 있는 최소 p (전승 시)."""
    return min(1.0, 2.0 / 2 ** c) if c > 0 else 1.0


def audit(name, eps):
    print('=' * 88)
    print(f'### {name}   —  에피소드 {len(eps)}개')
    data = {}
    for ep in eps:
        try:
            data[ep] = load(ep)
        except (OSError, KeyError) as e:
            print(f'  !! 읽기 실패 {ep}: {e}')
    if not data:
        return None

    n = len(data)
    out = {'set': name, 'n_episodes': n}

    def n_draws(vecs):
        """허용오차 기준 서로 다른 draw 의 수 (바이트가 아니라 난수 단위)."""
        seen = []
        for v in vecs:
            if not any(len(v) == len(s) and np.allclose(v, s, atol=TOL, rtol=0)
                       for s in seen):
                seen.append(v)
        return len(seen)

    print(f"  {'축':12s} {'draw 종수':>10s}   {'전승 시 바닥 p':>14s}")
    for label, c in COLS.items():
        k = n_draws([wc[:, c] for wc, _ in data.values()])
        out[label] = k
        print(f'  {label:12s} {k:10d}   {floor_p(k):14.4f}'
              + ('   <-- 전부 동일!' if k == 1 else ''))
    # 방향은 theta = atan2(ky, kx) 로 센다 (위 정정 주석 참조)
    k_th = n_draws([np.arctan2(wc[:, 3], wc[:, 2]) for wc, _ in data.values()])
    out['방향 theta'] = k_th
    print(f'  {"방향 theta":12s} {k_th:10d}   {floor_p(k_th):14.4f}')

    fkeys = {}
    for ep, (_, fld) in data.items():
        fkeys.setdefault(key_of(fld), []).append(ep)
    out['distinct_fields'] = len(fkeys)
    print(f'  {"진실면 전체":12s} {len(fkeys):12d} {len(fkeys):10d}   {floor_p(len(fkeys)):14.4f}')
    dupes = {k: v for k, v in fkeys.items() if len(v) > 1}
    if dupes:
        print('  ★ 비트 동일한 진실면 묶음 (관측단위가 아니라 같은 파동장을 두 번 본 것):')
        for v in dupes.values():
            print('      ' + '  ==  '.join(e[8:-3] for e in v))

    lim = min([out[l] for l in COLS] + [out['방향 theta'], out['distinct_fields']])
    print(f'  ⟹ 이 집합의 **가장 좁은 독립축 = {lim}** ⟹ 클러스터 부호검정 바닥 p = **{floor_p(lim):.4f}**'
          + ('  ⛔ 0.05 도달 불가' if floor_p(lim) > 0.05 else ''))
    out['narrowest_axis'] = lim
    out['floor_p'] = floor_p(lim)
    return out


def main():
    print('① 과거 "N/N 전승" 헤드라인 독립성 감사')
    print(f'데이터: {DATA}')
    print('결과 파일 미참조 — 생성기 파라미터와 진실면만 읽는다.\n')
    rows = [r for r in (audit(nm, eps) for nm, eps in SETS) if r]
    print('=' * 88)
    print('### 요약 — "N/N" 이라고 쓴 N 과 실제 독립단위\n')
    print(f"  {'집합':46s} {'쓴 N':>5s} {'진폭':>5s} {'위상':>5s} {'방향':>5s} {'진실면':>6s} {'바닥 p':>8s}")
    for r in rows:
        print(f"  {r['set']:46s} {r['n_episodes']:5d} {r['진폭 A']:5d} {r['위상 phi']:5d} "
              f"{r['방향 theta']:5d} {r['distinct_fields']:6d} {r['floor_p']:8.4f}")
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'independence.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    sys.exit(main())
