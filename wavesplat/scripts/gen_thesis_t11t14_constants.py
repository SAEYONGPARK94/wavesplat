#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`thesis/t11t14_constants.tex` 생성 — `T11`~`T14` 의 수치를 **손으로 적지 마라**.

선례 = `scripts/gen_deploy_constants.py`(학위논문 배포 상수) ·
       `scripts/gen_paper_constants.py`(논문 결과 상수, 동결 verdict 입력).

⛔**왜 학위논문 전용 파일을 따로 만드는가.** `thesis/thesis_main.tex` 는
`../paper/paper_constants` 를 `\\input` 하지만, 그 파일은 **논문 원고의 소유**이고
지금 다른 작업 단위가 그 디렉터리를 고치고 있다. 학위논문에만 들어가는 네 배치의
상수를 그쪽에 얹으면 두 문서의 개정 주기가 묶인다 ⟹ **별도 파일**로 낸다.

★이 스크립트는 동결 `verdict_t1{1,2,3,4}.json` 에서 값을 읽고, 아래 `PINS` 와
어긋나면 `AssertionError` 로 죽는다. ⟹ 배치가 재분석되면 학위논문이 **조용히**
낡지 않는다. ★재생성이 곧 검사다(`scripts/check_all.py` 가 매번 돌린다).

⛔**인쇄하지 않는 것**(2026-09-04/05 인용금지 묶음, `reference_citation_bans.md`):
  · `T14` 의 **라벨과 부분 통계 일체** --- 그 배치는 `G-EQUIVAR` 로 **차단**되었다.
    게이트 계수$\\cdot$차단자 이름$\\cdot$MDE$\\cdot$취약성 진단만 낸다.
  · `T11`$\\cdot$`T13` 의 **절대 nRMSE 중앙값** --- `RESCALE`/`FIX` 는 배포 구성이 아니다.
  · 저 코퍼스의 **층별(Hs$\\cdot$각도) 계수** --- 라벨을 만들지 않는 기술통계다.

사용:
    python3 scripts/gen_thesis_t11t14_constants.py           # 검증 + 생성
    python3 scripts/gen_thesis_t11t14_constants.py --show    # 값만 출력
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
TMP = os.path.join(ROOT, 'tmp')
OUT = os.path.join(ROOT, 'thesis', 't11t14_constants.tex')

T11 = 'T11_EXTACC_2026-09-03/verdict_t11.json'
T12 = 'T12_ISOLATE_2026-09-04/verdict_t12.json'
T13 = 'T13_DEPTHFIX_2026-09-04/verdict_t13.json'
T14 = 'T14_AUTOSCALE_2026-09-04/verdict_t14.json'

_cache = {}


def V(rel):
    if rel not in _cache:
        _cache[rel] = json.load(open(os.path.join(TMP, rel), encoding='utf-8'))
    return _cache[rel]


def get(rel, path):
    cur = V(rel)
    for k in path.split('.'):
        cur = cur[int(k)] if isinstance(cur, list) else cur[k]
    return cur


def ratio(rel, wpath, npath):
    return f'{get(rel, wpath)}/{get(rel, npath)}'


def sci(p, digits=1):
    """`5.96e-08` -> `6.0\\times10^{-8}` (수학 모드 안에서 쓴다)."""
    s = f'{float(p):.{digits}e}'
    m, e = s.split('e')
    return f'{m}\\times10^{{{int(e)}}}'


def gate_detail(rel, needle):
    for g in get(rel, 'gates'):
        if needle in g['name']:
            return g['name'], str(g.get('detail'))
    raise AssertionError(f'⛔게이트를 찾지 못했다: {needle} ({rel})')


# ── ★결과를 본 뒤에 바뀌면 죽는 핀 ───────────────────────────────────────
def check_pins():
    bad = []

    def eq(label, got, want):
        if str(got) != str(want):
            bad.append(f'{label}: 동결값 {got!r} != 핀 {want!r}')

    # T11 --- BAND-MISMATCH
    eq('T11 라벨', get(T11, 'label'), 'BAND-MISMATCH')
    eq('T11 게이트', get(T11, 'gates_pass'), '17/17')
    eq('T11 행수', get(T11, 'n_rows'), 96)
    eq('T11 PRIMARY', ratio(T11, 'primary.w', 'primary.n'), '24/24')
    eq('T11 SEC-A', ratio(T11, 'sec_a_ship.w', 'sec_a_ship.n'), '8/24')
    eq('T11 SEC-B', ratio(T11, 'sec_b_oracle.w', 'sec_b_oracle.n'), '24/24')
    eq('T11 차단', get(T11, 'blocked_by'), [])
    # ★`s=2` 를 손으로 적지 않는다 --- 게이트 이름이 정본이다.
    name, _ = gate_detail(T11, 'G-RESCALE')
    if 's=2.0' not in name:
        bad.append(f'T11 재척도 배율이 게이트 이름에 없다: {name}')

    # T12 --- MULTIPLE-AXES-CARRY
    eq('T12 라벨', get(T12, 'label'), 'MULTIPLE-AXES-CARRY')
    eq('T12 게이트', get(T12, 'gates_pass'), '25/25')
    eq('T12 행수', get(T12, 'n_rows'), 424)
    eq('T12 PRIMARY', ratio(T12, 'primary.w', 'primary.n'), '24/24')
    eq('T12 SEC 문턱', get(T12, 'sec_family_thresh'), 19)
    eq('T12 SEC 최대', get(T12, 'sec_max'), 19)
    eq('T12 C-MINDK 개입', get(T12, 'effect_vs_ship.C-MINDK'), '6/24 다름')
    eq('T12 미실행 축', get(T12, 'not_exercised'), [])
    eq('T12 no-op 팔', get(T12, 'no_op_arms'), [])

    # T13 --- FIX-UNDETECTED (WITHIN-PERTURBATION-FLOOR)
    eq('T13 라벨', get(T13, 'label'), 'FIX-UNDETECTED (WITHIN-PERTURBATION-FLOOR)')
    eq('T13 게이트', get(T13, 'gates_pass'), '24/24')
    eq('T13 행수', get(T13, 'n_rows'), 256)
    eq('T13 PRIMARY', ratio(T13, 'primary.w', 'primary.n'), '26/52')
    eq('T13 바닥 안', get(T13, 'ctrl.within_floor'), True)
    eq('T13 CTRL 이동', get(T13, 'ctrl.moved'), False)

    # T14 --- ⛔라벨 없음(차단)
    if get(T14, 'label') is not None:
        bad.append(f"T14 에 라벨이 생겼다: {get(T14, 'label')!r} — 문안을 다시 쓰라")
    eq('T14 label_msg', get(T14, 'label_msg'), 'BLOCKED')
    eq('T14 게이트', get(T14, 'gates_pass'), '31/32')
    eq('T14 차단자', get(T14, 'blocked_by'), ['G-EQUIVAR'])
    eq('T14 행수', get(T14, 'n_rows'), 212)

    # ★네 배치를 잇는 앵커 --- 하나라도 비트 재현을 잃으면 사슬 주장이 죽는다.
    for rel, needle, want in ((T12, 'G-ANCHOR', 'BIT'),
                              (T13, 'G-ANCHOR-SHIP', '불일치 0'),
                              (T13, 'G-ANCHOR-RESCALE', '불일치 0'),
                              (T14, 'G-ANCHOR-T11', '불일치 0')):
        _, det = gate_detail(rel, needle)
        if want not in det:
            bad.append(f'{rel} {needle}: 「{want}」 가 사라졌다 — {det}')

    if bad:
        raise AssertionError('⛔동결 판정과 어긋난다:\n  ' + '\n  '.join(bad))


def fresh_ship_zero():
    """`T13` 의 **신규 절반**에서 `SHIP<ZERO` --- 공시 문자열이 정본이다.

    ⚠구조화 필드가 아니라 `disclosures` 안에 있다(개정문 §1 이 그 자리를 정했다)
    ⟹ **정규식으로 뽑고**, 못 찾으면 죽는다. ⛔손으로 옮기지 않는다.
    """
    for d in get(T13, 'disclosures'):
        m = re.search(r'SHIP<ZERO\s+(\d+)/(\d+)', d)
        if m:
            return f'{m.group(1)}/{m.group(2)}'
    raise AssertionError('⛔T13 공시에서 신규 절반의 SHIP<ZERO 를 찾지 못했다')


def band_fragile():
    """`G-BAND-FRAGILE` 의 여유 --- 중앙과 **최소**를 함께 낸다(공시 의무)."""
    r = [b['kp_over_dk1'] for b in get(T14, 'band_fragile')]
    r.sort()
    n = len(r)
    med = r[n // 2] if n % 2 else 0.5 * (r[n // 2 - 1] + r[n // 2])
    return f'{med:.2f}', f'{min(r):.2f}'


def rows():
    out = []

    def add(k, v):
        out.append((k, v))

    # ── T11 -------------------------------------------------------------
    add('TElevenGates', get(T11, 'gates_pass'))
    add('TElevenRows', str(get(T11, 'n_rows')))
    add('TElevenRuns', str(get(T11, 'new_runs')))
    add('TElevenPrimary', ratio(T11, 'primary.w', 'primary.n'))
    add('TElevenPrimaryP', sci(get(T11, 'primary.p_one_sided')))
    add('TElevenShip', ratio(T11, 'sec_a_ship.w', 'sec_a_ship.n'))
    add('TElevenOracle', ratio(T11, 'sec_b_oracle.w', 'sec_b_oracle.n'))
    add('TElevenThresh', str(get(T11, 'mde.crit_w')))
    add('TElevenMDE', f"{get(T11, 'mde.q80'):.2f}")
    add('TElevenUnrejLo', str(get(T11, 'mde.unrejectable_w.0')))
    add('TElevenUnrejHi', str(get(T11, 'mde.unrejectable_w.1')))
    # 아카이브 크기는 게이트가 **직접 잰** 바이트다(러너 리터럴 금지).
    _, det = gate_detail(T11, 'G-SIZE')
    add('TElevenArchiveGB', f'{int(det) / 1e9:.1f}')

    # ── T12 -------------------------------------------------------------
    add('TTwelveGates', get(T12, 'gates_pass'))
    add('TTwelveRows', str(get(T12, 'n_rows')))
    add('TTwelvePrimary', ratio(T12, 'primary.w', 'primary.n'))
    add('TTwelvePrimaryP', sci(get(T12, 'primary.p_one_sided')))
    add('TTwelvePrimaryThresh', str(get(T12, 'primary.thresh')))
    add('TTwelveHeld', ratio(T12, 'secr_exploratory.A-MAPSCALE.w',
                             'secr_exploratory.A-MAPSCALE.n'))
    add('TTwelveHeldShip', ratio(T12, 'secr_exploratory.SHIP.w',
                                 'secr_exploratory.SHIP.n'))
    add('TTwelveAnchorShip', ratio(T12, 'anchors.SHIP.w', 'anchors.SHIP.n'))
    add('TTwelveSecThresh', str(get(T12, 'sec_family_thresh')))
    for macro, arm in (('TTwelveMapcell', 'A-MAPCELL'), ('TTwelveBand', 'A-BAND'),
                       ('TTwelveDedup', 'B-DEDUP'), ('TTwelveMindk', 'C-MINDK'),
                       ('TTwelveGrowgrid', 'C-GROWGRID'), ('TTwelveVplr', 'D-VPLR'),
                       ('TTwelveMapdepth', 'E-MAPDEPTH'), ('TTwelveSteep', 'F-STEEP')):
        add(macro, ratio(T12, f'sec_family.{arm}.w', f'sec_family.{arm}.n'))
    add('TTwelveMapcellP', f"{get(T12, 'sec_family.A-MAPCELL.p_one_sided'):.4f}")
    add('TTwelveMindkDiff', get(T12, 'effect_vs_ship.C-MINDK').split()[0])
    add('TTwelveRho', f"{get(T12, 'rho_recovery.A-MAPSCALE.median'):.3f}")
    add('TTwelveRhoQOne', f"{get(T12, 'rho_recovery.A-MAPSCALE.q1'):.3f}")
    add('TTwelveRhoQThree', f"{get(T12, 'rho_recovery.A-MAPSCALE.q3'):.3f}")
    add('TTwelveSecMDE', f"{get(T12, 'mde.sec_q80'):.4f}")

    # ── T13 -------------------------------------------------------------
    add('TThirteenGates', get(T13, 'gates_pass'))
    add('TThirteenRows', str(get(T13, 'n_rows')))
    add('TThirteenPrimary', ratio(T13, 'primary.w', 'primary.n'))
    add('TThirteenThreshWin', str(get(T13, 'primary.thresh_win')))
    add('TThirteenThreshHurt', str(get(T13, 'primary.thresh_hurt')))
    add('TThirteenFixZero', ratio(T13, 'sec_a.w', 'sec_a.n'))
    add('TThirteenShipZero', ratio(T13, 'ship_vs_zero.w', 'ship_vs_zero.n'))
    add('TThirteenReread', '{}/{}'.format(*get(T13, 'strata_descriptive.T11-24')))
    add('TThirteenFresh', '{}/{}'.format(*get(T13, 'strata_descriptive.FRESH-28')))
    add('TThirteenFreshShipZero', fresh_ship_zero())
    add('TThirteenFloor', f"{get(T13, 'ctrl.floor_median_abs_rel'):.4f}")
    add('TThirteenEffect', f"{get(T13, 'ctrl.effect_median_rel'):.6f}")
    add('TThirteenMDE', f"{get(T13, 'mde.q80'):.4f}")
    add('TThirteenUnrejLo', str(get(T13, 'mde.unrejectable_w.0')))
    add('TThirteenUnrejHi', str(get(T13, 'mde.unrejectable_w.1')))

    # ── T14 (⛔라벨·부분 통계 없음) ---------------------------------------
    add('TFourteenGates', get(T14, 'gates_pass'))
    add('TFourteenRows', str(get(T14, 'n_rows')))
    add('TFourteenBlocker', '\\texttt{' + get(T14, 'blocked_by.0') + '}')
    add('TFourteenMDE', f"{get(T14, 'mde.primary_q80'):.4f}")
    med, lo = band_fragile()
    add('TFourteenBandMed', med)
    add('TFourteenBandMin', lo)
    add('TFourteenShipZero', ratio(T14, 'ship_vs_zero.w', 'ship_vs_zero.n'))

    # ── 네 배치 공통(✅인용 가능한 단서형) --------------------------------
    _, det = gate_detail(T14, 'G-INVERT')
    add('TInvert', re.search(r'위반\s*(\S+)', det).group(1))
    _, det = gate_detail(T14, 'G-S-BLIND')
    add('TSBlind', re.search(r'위반\s*(\S+)', det).group(1))
    return out


def render():
    check_pins()
    body = '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in rows())
    return (
        '% AUTO-GENERATED by scripts/gen_thesis_t11t14_constants.py -- DO NOT EDIT BY HAND.\n'
        '% 입력 = 동결 verdict_t11/t12/t13/t14.json. 학위논문은 이 매크로만 쓴다.\n'
        '% ⛔T14 는 차단된 배치다 --- 라벨도 부분 통계도 여기에 없다(있으면 그것이 결함이다).\n'
        + body + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', action='store_true')
    # ★[2026-09-05] `gen_paper_constants.py` 와 같은 규약 --- 손편집.드리프트 탐지.
    #   ⛔`check_all.py` 는 재생성으로 검사하지만(git 이 차이를 보여준다) CI 나 훅에서
    #   **쓰기 없이** 대조해야 할 때가 있다.
    ap.add_argument('--check', action='store_true',
                    help='디스크의 파일이 동결 verdict 에서 생성한 것과 같은가(쓰지 않는다)')
    a = ap.parse_args()
    txt = render()
    if a.check:
        cur = open(OUT, encoding='utf-8').read() if os.path.exists(OUT) else None
        if cur != txt:
            print(f'⛔{os.path.relpath(OUT, ROOT)} 가 생성값과 다르다 --- '
                  '손편집이거나 동결 판정이 움직였다. 생성기를 다시 돌려라.')
            return 1
        print(f'[OK  ] {os.path.relpath(OUT, ROOT)} == 생성값 ({len(rows())} 매크로)')
        return 0
    if a.show:
        print(txt)
        return 0
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(txt)
    print(txt)
    print(f'기록: {OUT}  ({len(rows())} 매크로)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
