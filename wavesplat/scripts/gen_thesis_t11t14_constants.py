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
# ★[2026-09-05] 파일 이름은 `t11t14` 로 둔다 --- 이름을 바꾸면 thesis 의 \input 과
#   scripts/check_all.py 의 항목까지 함께 움직여야 하고, 그 churn 은 이 커밋의 몫이
#   아니다. ⚠이름 부채로 남긴다: 이 파일은 실제로 T11--T16 을 담는다.
T15 = 'T15_HOMEREG_2026-09-05/verdict_t15.json'
T16 = 'T16_BANDFIX_2026-09-05/verdict_t16.json'

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

    # T15 --- HOME-REGRESSION (⚠노출된 코퍼스: 사전의 학습 봉투)
    eq('T15 라벨', get(T15, 'label'), 'HOME-REGRESSION')
    eq('T15 게이트', get(T15, 'gates_pass'), '22/22')
    eq('T15 행수', get(T15, 'n_rows'), 832)
    eq('T15 차단', get(T15, 'blocked_by'), [])
    eq('T15 혼잡', ratio(T15, 'primary.crowded.w', 'primary.crowded.n'), '23/64')
    eq('T15 희소', ratio(T15, 'primary.sparse.w', 'primary.sparse.n'), '31/64')
    # ★해침 문턱을 **정확히** 만났다는 것이 이 배치의 문면이다 --- 문턱이 움직이면 죽는다.
    eq('T15 해침 문턱', get(T15, 'primary.crowded.thresh_hurt'), 23)
    # ★★[2026-09-05] **구조 핀** --- 학위논문 산문이 계수가 아니라 **관계**를 주장하는 자리.
    #   ⛔계수보다 먼저 조용히 낡는 것이 이쪽이다(논문 생성기 `_t15_t16()` 과 같은 규약).
    if int(get(T15, 'primary.crowded.w')) != int(get(T15, 'primary.crowded.thresh_hurt')):
        bad.append('⛔"해침 문턱을 정확히 만난다" 가 거짓이 됐다 --- 그 자구를 고쳐라')
    eq('T15 혼잡 등가(바닥 밖)', get(T15, 'equivalence.crowded.within_floor'), False)
    eq('T15 희소 등가(바닥 안)', get(T15, 'equivalence.sparse.within_floor'), True)
    # ⚠**바닥 팔이 희소에서 도움 문턱을 정확히 만난다** --- "빌려온 바닥" 문장의 하중이다.
    if int(V(T15)['curve']['sparse']['FLOOR-VPLR']['w']) != int(get(T15, 'primary.sparse.thresh_help')):
        bad.append('⛔바닥 팔이 희소에서 도움 문턱을 더는 정확히 만나지 않는다 --- 문안을 고쳐라')
    eq('T15 스모크 노출 수', len(get(T15, 'smoke_exposed_blocks')), 2)
    eq('T15 양성대조 혼잡 복제', get(T15, 'control.crowded.replicated'), True)
    eq('T15 양성대조 희소 복제', get(T15, 'control.sparse.replicated'), True)

    # T16 --- 두 라벨의 결합. ⛔`BANDFIX-` 접두어는 폐기됐다(verdict §9.2-6b).
    eq('T16 라벨(S)', get(T16, 'label_S'), 'AUTOSCALE-BEATS-ZERO-BOTH-RULES')
    eq('T16 라벨(D)', get(T16, 'label_D'), 'D-TREATMENT-BELOW-SCALE')
    eq('T16 게이트', get(T16, 'gates_pass'), '46/46')
    eq('T16 행수', get(T16, 'n_rows'), 366)
    eq('T16 차단', get(T16, 'blocked_by'), [])
    # ★`T14` 를 막았던 것과 **같은 종류로 다시 규정된** 등변성 게이트를 통과한다
    #   --- 그것이 이 배치의 존재 이유다. ⛔*"T14 의 그 게이트를 통과한다"* 라 적지 마라
    #   (저쪽 허용오차는 `c=2.0` 한 방향에만 걸렸고 규약도 다르다 --- verdict §9.3-15·16).
    #   ⚠이 주석 자신이 2026-09-05 적대적 심사에서 **정정을 되감는 자리**로 지목됐다.
    name, det = gate_detail(T16, 'G-EQUIVAR ')
    if '위반 0' not in det:
        bad.append(f'T16 G-EQUIVAR 가 더는 무위반이 아니다: {det}')
    # ★음성 읽기 차단: 옛 규칙도 제로필드를 이긴다(SEC-E). 이 값이 사라지면 문면이 거짓이 된다.
    eq('T16 SEC-E', ratio(T16, 'SEC.SEC-E AUTO-BAND<ZERO (★사다리 안 · 무보정).w',
                          'SEC.SEC-E AUTO-BAND<ZERO (★사다리 안 · 무보정).n'), '58/60')
    # ★★[2026-09-05] 구조 핀 --- 산문이 주장하는 **관계** 셋.
    eq('T16 배포 팔 유의열세', get(T16, 'sig_worse.SHIP'), True)
    eq('T16 후보 유의열세 없음', get(T16, 'sig_worse.AUTO-BANDR'), False)
    eq('T16 D 축 자보다 작음', get(T16, 'D_drive.T1_treatment_below_scale'), True)
    eq('T16 앵커 등급', get(T16, 'anchor_grade'), 'T1')
    # ⛔[2026-09-05 적대적 심사 m8] 산문이 *"기준은 $\\ln 1.05 = \\TSixteenDriveRef$"* 라
    #   적는데 **`1.05` 자체는 아무 데도 핀돼 있지 않았다** --- 설계 기준이 1.10 으로 바뀌면
    #   매크로만 움직여 **거짓 항등식**이 인쇄된다. 자를 산수로 붙든다.
    import math as _math
    if abs(float(get(T16, 'D_drive.ln105')) - _math.log(1.05)) > 1e-12:
        bad.append('⛔`D_drive.ln105` 가 더는 ln(1.05) 가 아니다 --- 산문의 항등식을 고쳐라')

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


def pct(x, signed=True):
    """상대차를 백분율 문자열로. ⛔손으로 적지 마라 --- 부호가 논거의 절반이다."""
    return (f'{x*100:+.1f}' if signed else f'{x*100:.1f}') + '\\%'


def sweep_ladder():
    """★스윕 **부분층 위의 `f=2`** --- ⛔동결 `verdict_t15.json` 에 **없는 수**다(재집계).

    ⛔왜 필요한가(2026-09-05 적대적 심사 BLOCKER, 논문 쪽과 같은 사유):
      원고 초판이 스윕 세 팔(√2 · 2√2 · 4)만 보고 *"배율이 커질수록 벌점이 **단조**
      증가한다(용량--반응)"* 라 적었다. 그런데 **PRIMARY 처치 `f=2` 도 그 32블록에서
      돌았고**, 그 값을 사다리에 넣으면 +14.1% → **+8.3%** → +22.9% → +31.0% 로
      **단조가 깨진다**. ⟹ 세 팔만 고른 것이 자의적이었고 그 자의가 없는 주장을 만들었다.

    ★자기검사 셋(어긋나면 죽는다):
      · 부분층에서 **세 스윕 팔을 재계산해 동결 verdict 과 대조**한다(계수·중앙 상대차 전건)
        ⟹ 부분층 식별이 옳다는 실행 증명이다.
      · **사다리가 단조가 되면 죽는다** ⟹ 학위논문의 「단조가 아니다」 문장이 조용히 낡지 않는다.
      · **절대 중앙값의 순위가 상대차의 순위와 같아져도 죽는다**(그 병기 문장의 근거다).
    ⛔이 수는 **재집계**이므로 원고가 그 지위를 밝혀야 한다(선례 `W-15`).
    """
    import statistics as _st
    d = os.path.join(TMP, 'T15_HOMEREG_2026-09-05')
    rws = []
    for f in ('results_t15_FULL.jsonl', 'results_t15_SUB.jsonl'):
        with open(os.path.join(d, f), encoding='utf-8') as fh:
            rws += [json.loads(x) for x in fh]
    by, lay = {}, {}
    for r in rws:
        by.setdefault(r['block'], {})[r['arm']] = r['nrmse']
        lay[r['block']] = r['layer']
    sub = [b for b in by if lay[b] == 'crowded' and 'F4' in by[b]]
    curve = V(T15)['curve']['crowded']
    assert len(sub) == int(curve['F4']['n']), '⛔스윕 부분층 크기가 갈린다'

    def stat(arm):
        w = sum(1 for b in sub if by[b][arm] < by[b]['SHIP'])
        rel = _st.median((by[b][arm] - by[b]['SHIP']) / by[b]['SHIP'] for b in sub)
        med = _st.median(by[b][arm] for b in sub)
        return w, rel, med

    for arm in ('F1.414', 'F2.828', 'F4'):
        w, rel, _ = stat(arm)
        assert w == int(curve[arm]['w']), f'⛔{arm} 재집계 계수가 동결값과 다르다'
        assert abs(rel - curve[arm]['median_rel']) < 1e-12, f'⛔{arm} 재집계 중앙값이 다르다'
    order = ('F1.414', 'F2', 'F2.828', 'F4')
    ladder = [stat(a)[1] for a in order]
    assert not all(a < b for a, b in zip(ladder, ladder[1:])), \
        '⛔스윕 사다리가 단조가 됐다 --- 학위논문의 「단조가 아니다」 문장을 다시 써라'
    assert all(x > 0 for x in ladder), '⛔어떤 배율이 배포 팔을 이겼다 --- 문안을 다시 써라'
    absm = [stat(a)[2] for a in order]
    rank = lambda v: [i for i, _ in sorted(enumerate(v), key=lambda t: t[1])]
    assert rank(ladder) != rank(absm), \
        '⛔절대 중앙값 순위가 상대차 순위와 같아졌다 --- 그 병기 문장을 다시 써라'
    w2, rel2, _ = stat('F2')
    return {'TFifteenDoseN': str(len(sub)),
            'TFifteenSweepLo': f"{curve['F1.414']['w']}/{curve['F1.414']['n']}",
            'TFifteenSweepLoRel': pct(curve['F1.414']['median_rel']),
            'TFifteenSubFTwo': f'{w2}/{len(sub)}',
            'TFifteenSubFTwoRel': pct(rel2),
            'TFifteenSweepMid': f"{curve['F2.828']['w']}/{curve['F2.828']['n']}",
            'TFifteenSweepMidRel': pct(curve['F2.828']['median_rel']),
            'TFifteenDoseRel': pct(curve['F4']['median_rel'])}


def peek_counts():
    """⚠본실행 도중 읽은 **부분 집계** --- `NOTE_INTERIM_PEEK.md` 에서 읽는다.

    ⛔손으로 옮겨 적지 않는다. 저장소 규율은 *"봤지만 안 말한다 는 공시가 아니다"* 이고
    (`AMENDMENT_1_PREDATA.md §1.1` 축자), 그래서 **보인 세 계수를 전부** 낸다 ---
    논문 초판은 PRIMARY 하나만 인쇄하고 둘을 빠뜨렸다(적대적 심사 MAJOR).
    ★분모도 검사한다: 30 은 **혼잡 64** 중 30 이지 나뉘지 않은 128 중 30 이 아니다.
    """
    note = open(os.path.join(TMP, 'T15_HOMEREG_2026-09-05',
                             'NOTE_INTERIM_PEEK.md'), encoding='utf-8').read()
    mp = re.search(r'\*\*(\d+)/(\d+) 블록\*\*', note)
    mw = re.search(r'F2 < SHIP\s+(\d+)/(\d+)', note)
    mc = re.search(r'SHIP < ZERO\s+(\d+)/(\d+)', note)
    mf = re.search(r'FLOOR-VPLR < SHIP\s+(\d+)/(\d+)', note)
    assert mp and mw and mc and mf, '⛔중간 열람 공시의 형태가 바뀌었다'
    assert mw.group(2) == mc.group(2) == mf.group(2) == mp.group(1), \
        '⛔중간 열람 세 계수의 분모가 갈린다'
    assert '혼잡' in note, '⛔중간 열람이 혼잡 층이라는 진술이 사라졌다'
    assert int(mp.group(2)) == 128 and int(mp.group(1)) < 64, \
        '⛔중간 열람 시점의 분모 관계가 바뀌었다(혼잡 64 안이어야 한다)'
    return {'TFifteenPeekAt': mp.group(1), 'TFifteenPeekWin': mw.group(1),
            'TFifteenPeekCtrl': mc.group(1), 'TFifteenPeekFloor': mf.group(1)}


def t16_gate_atoms():
    """★`T16` 의 **게이트 유래** 원자 --- 등변성 자·격자 자·속성 노출.

    ⛔이 셋은 verdict 의 구조화 필드가 아니라 **게이트 이름과 세부 문자열**에 있다
      ⟹ 정규식으로 뽑고 못 찾으면 죽는다(손복사 금지 규약).
    ⚠`G-EQUIVAR` 는 접두어가 같은 형제가 둘 있다(`-RAW`·`-OLD`) ⟹ 뒤에 공백을 요구한다.
    """
    v16 = V(T16)
    name, det = gate_detail(T16, 'G-EQUIVAR ')
    m = re.search(r'≤(\d)e-0?(\d+)', name)
    assert m and m.group(1) == '1', f'⛔G-EQUIVAR 허용오차 표기가 바뀌었다: {name}'
    mv = re.match(r'위반\s+(\d+)', det)
    assert mv and mv.group(1) == '0', f'⛔G-EQUIVAR 에 위반이 생겼다: {det}'
    # ⛔개정문 2 의 **완화**. 학위논문이 그것을 공시하므로 두 수를 여기서 핀한다.
    gname, gdet = gate_detail(T16, 'G-GRID')
    mg = re.search(r'/max ≤ ([0-9.e-]+)', gname)
    assert mg, f'⛔G-GRID 자구가 바뀌었다: {gname}'
    tol = float(mg.group(1))
    assert abs(tol - 1e-4) < 1e-12, f'⛔G-GRID 허용오차가 움직였다: {tol}'
    mm = re.search(r"x/y 상대차 \{[^}]*'median': ([0-9.e+-]+)", gdet)
    assert mm, f'⛔G-GRID 세부 형태가 바뀌었다: {gdet[:120]}'
    meas = float(mm.group(1))
    assert meas < tol, '⛔측정 비대칭이 완화된 허용오차를 넘었다'
    # ⚠**속성 노출** --- 판정 60블록은 *성과* 미노출이지만 36개는 앞 배치가 **속성**을 열었다.
    ma = re.search(r'속성 노출 (\d+)/(\d+)', v16['grade'])
    assert ma and int(ma.group(2)) == 60, f"⛔T16 grade 문면이 바뀌었다: {v16['grade']}"
    assert 'blind' in v16['grade'], '⛔T16 의 blind 등급 자구가 사라졌다'
    return {'TSixteenEquivarTol': f'10^{{-{int(m.group(2))}}}',
            'TSixteenEquivarViol': mv.group(1),
            'TSixteenGridTol': '10^{-4}',
            'TSixteenGridMeas': sci(meas, 2),
            'TSixteenAttrExposed': ma.group(1),
            'TSixteenN': ma.group(2)}


def t16_scale_shape():
    """★*"상수인 것은 규칙이 **읽는 양**이지 규칙이 **고르는 배율**이 아니다"* 를 붙든다.

    ⛔[2026-09-05] 학위논문 초판이 이 둘을 뒤집어 적었다(*"규칙이 고르는 배율은 거의 상수"*).
      동결 행이 반증한다 --- `s_rule` 고유값 `AUTO-BANDR` **41/60** · `AUTO-BAND` **35/60**
      (CV 16.5%·23.3%). 상수인 것은 전 파일이 공유하는 **격자·개구**이고, 그래서
      1단 반경 `klim₁` 이 팔마다 **단일값**(0.6033 · 2.0)이다.
    ⚠verdict §9.3-2 는 *"`s` 와 `klim₁` 이 **거의/완전히** 상수"* 로 둘을 묶어 적었는데
      축퇴 논증(`klim₁` 3.3배 → `k̂_p` 1%)은 `klim₁` 쪽이다 ⟹ 원고는 정확한 쪽을 적는다.
    ⛔매크로를 내지 않는다 --- 이 함수는 **문장을 지키는 assert** 다(런 0회, 동결 행만 읽는다).
    """
    d = os.path.join(TMP, 'T16_BANDFIX_2026-09-05', 'results_t16.jsonl')
    P = [json.loads(x) for x in open(d, encoding='utf-8')]
    P = [r for r in P if r.get('tier') == 'TIER-P']
    assert len({r['L_obs_x'] for r in P}) == 1 and len({r['L_obs_y'] for r in P}) == 1, \
        '⛔관측 개구가 더는 코퍼스 상수가 아니다 --- "규칙이 읽는 양이 상수" 문장을 다시 써라'
    for arm in ('AUTO-BANDR', 'AUTO-BAND'):
        k = {r['klim1'] for r in P if r['arm'] == arm}
        sv = {r['s_rule'] for r in P if r['arm'] == arm}
        assert len(k) == 1, f'⛔{arm} 의 1단 반경이 더는 단일값이 아니다: {len(k)} 개'
        assert len(sv) > 1, \
            f'⛔{arm} 의 배율이 상수가 됐다 --- 원고의 "그 배율은 블록마다 갈린다" 를 다시 써라'
    return {}


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
    # ★★[2026-09-05 수치 정합 감사] ⚠**아래 MDE 매크로 셋은 검정력이 0.80 에 미달한다.**
    #   ⛔동결 verdict 를 고치지 않는다(증거다) ⟹ 여기에 실측 검정력을 적어 둔다.
    #     · `\TTwelveSecMDE`    0.8332 → **0.79995** (규약값 0.8333)
    #     · `\TFourteenMDE`     0.7630 → **0.79986** (규약값 0.7631)
    #     · `\TFifteenMDE`      0.3185 → **0.79978** (해침 다리, 규약값 0.3184)
    #     · `\TSixteenLandingMDE` 0.7074 → **0.79991** · `\TSixteenSkillMDE` 0.6915 → **0.79984**
    #   ⚠부족분은 전부 <=2.2e-04 이고 **어떤 라벨도 움직이지 않는다**(MDE 는 라벨을 만들지
    #   않는다 — 문턱 `w>=k` 가 만든다. `T11`~`T16` 전 게이트에서 수치 비교 0건을 확인했다).
    #   ⟹ 문안을 고칠 필요는 없으나, *"검정력 0.80"* 이라 **단정**하지 말고 "약 0.80" 이 옳다.
    #   ★규약과 재계산 = `tmp/_audit/sign_test_mde.py` · 래칫 = `scripts/check_all.py MDE_KNOWN`.
    #   ✅`\TElevenMDE`(0.76)·`\TThirteenMDE`(0.6795) 는 **보수적**(0.80094·0.80156)이라 안전하다.
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

    # ── T15 (자국 코퍼스에서 같은 노브를 옮기면) ---------------------------
    add('TFifteenGates', get(T15, 'gates_pass'))
    # ★[2026-09-05 심사 m5] 동결 라벨을 **인쇄한다** --- 감사 앵커다(논문은 이미 인쇄한다).
    add('TFifteenLabel', get(T15, 'label'))
    add('TFifteenRows', str(get(T15, 'n_rows')))
    add('TFifteenCrowded', ratio(T15, 'primary.crowded.w', 'primary.crowded.n'))
    add('TFifteenCrowdedP', f"{get(T15, 'primary.crowded.p_hurt_one_sided'):.4f}")
    add('TFifteenCrowdedThresh', str(get(T15, 'primary.crowded.thresh_hurt')))
    add('TFifteenSparse', ratio(T15, 'primary.sparse.w', 'primary.sparse.n'))
    add('TFifteenSparseP', f"{get(T15, 'primary.sparse.p_hurt_one_sided'):.2f}")
    add('TFifteenCtrl', ratio(T15, 'control.crowded.w', 'control.crowded.n'))
    # ⚠*"두 층에서 복제된다"* 는 **두 계수**의 주장이다 --- 하나만 인쇄하면 다른
    #   층이 조용히 무너져도 문장이 남는다(이 저장소의 「N-1곳」 결함).
    add('TFifteenCtrlSparse', ratio(T15, 'control.sparse.w', 'control.sparse.n'))
    add('TFifteenDose', ratio(T15, 'curve.crowded.F4.w', 'curve.crowded.F4.n'))
    add('TFifteenMDE', f"{get(T15, 'mde.q80'):.4f}")
    add('TFifteenUnrejLo', str(get(T15, 'mde.unrejectable_w.0')))
    add('TFifteenUnrejHi', str(get(T15, 'mde.unrejectable_w.1')))
    # ★[2026-09-05] 논문 §5.8 과 **같은 공시 수준**으로 올린다 --- 학위논문에만
    #   빠져 있던 다섯(중간 열람 · 게이트 완화 · 속성 노출 · 스윕 부분층 · 개입 바닥).
    add('TFifteenN', str(get(T15, 'primary.crowded.n')))
    add('TFifteenCorpusN',
        str(get(T15, 'primary.crowded.n') + get(T15, 'primary.sparse.n')))
    add('TFifteenHelpThresh', str(get(T15, 'primary.crowded.thresh_help')))
    add('TFifteenSmokeExposed', str(len(get(T15, 'smoke_exposed_blocks'))))
    # ★결과 **전에** 정의된 처치정합 잡음 바닥 --- 크기 인용의 병기 의무를 진다.
    #   ⚠부호 없는 크기(중앙 |상대차|)다. ⛔처치의 벌점(부호 있음)과 섞지 마라.
    add('TFifteenFloor',
        pct(get(T15, 'equivalence.crowded.floor_median_abs_rel'), signed=False))
    add('TFifteenRel', pct(get(T15, 'equivalence.crowded.effect_median_rel')))
    # ⚠**바닥 팔은 이 코퍼스에서 널이라는 보장이 없다**(verdict disclosures §10).
    #   혼잡에서는 어느 문턱에도 못 닿지만 **희소에서는 도움 문턱을 정확히 만난다**.
    fc, fs = V(T15)['curve']['crowded']['FLOOR-VPLR'], V(T15)['curve']['sparse']['FLOOR-VPLR']
    add('TFifteenFloorCrowded', f"{fc['w']}/{fc['n']}")
    add('TFifteenFloorSparse', f"{fs['w']}/{fs['n']}")

    # ── T16 (참값 없이 그 배율을 고르는 규칙) ------------------------------
    #   ⛔비교자는 **제로필드**다 --- 외부 기준선 행 0개(verdict §9.2-1).
    add('TSixteenGates', get(T16, 'gates_pass'))
    add('TSixteenRows', str(get(T16, 'n_rows')))
    S = 'primary_family.S (기술 축, 단측, Bonferroni α/2)'
    D = 'primary_family.D (착지 축, 양측, Bonferroni α/2)'
    add('TSixteenSkill', ratio(T16, S + '.w', S + '.n'))
    add('TSixteenSkillP', sci(get(T16, S + '.p')))
    add('TSixteenSkillThresh', str(get(T16, S + '.thresh')))
    E = 'SEC.SEC-E AUTO-BAND<ZERO (★사다리 안 · 무보정)'
    add('TSixteenOldRule', ratio(T16, E + '.w', E + '.n'))
    add('TSixteenPosCtrl', ratio(T16, 'positive_control.FIX2<ZERO.w',
                                 'positive_control.FIX2<ZERO.n'))
    add('TSixteenShipZero', ratio(T16, 'ship_vs_zero.w', 'ship_vs_zero.n'))
    add('TSixteenLanding', '{}/{}'.format(get(T16, D + '.w'), get(T16, D + '.n_eff')))
    add('TSixteenLandingP', f"{get(T16, D + '.p_two_sided'):.2f}")
    add('TSixteenLandingThresh', str(get(T16, 'thresholds.D')))
    add('TSixteenLandingMDE', f"{get(T16, 'mde.D_q80'):.4f}")
    add('TSixteenSkillMDE', f"{get(T16, 'mde.S_q80'):.4f}")
    add('TSixteenDrive', f"{get(T16, 'D_drive.median_abs_dln_s'):.4f}")
    add('TSixteenDriveRef', f"{get(T16, 'D_drive.ln105'):.4f}")
    # ⛔`D` 축의 구동량은 **부호가 섞인다** --- 그 분할 없이 인용 금지(verdict §9.2-6).
    add('TSixteenDrivePos', str(get(T16, 'D_drive.n_pos')))
    add('TSixteenDriveNeg', str(get(T16, 'D_drive.n_neg')))
    add('TSixteenSigWorseThresh', str(get(T16, 'thresholds.SIG_WORSE')))
    add('TSixteenOldRuleThresh', str(get(T16, E + '.thresh')))
    # ⚠옛 규칙의 등변 편차는 **허용오차 없는 기록 전용**이고 방향마다 다르다
    #   ⟹ 한쪽만 인쇄하면 체리피킹이다. 두 방향을 **함께** 낸다.
    add('TSixteenEquivarOldDown', f"{get(T16, 'equivar_old.c05.median'):.3f}")
    add('TSixteenEquivarOldUp', f"{get(T16, 'equivar_old.c20.median'):.3f}")
    # ★**설계가 결과 전에 스스로 적은 사전확률**(verdict §0-4) --- ⛔이 사실을 결과
    #   뒤에 재서술하지 마라. 인쇄하지 않으면 독자가 59/60 을 놀라운 것으로 읽는다.
    add('TSixteenPriorD', f"{V(T16)['prediction']['label_probs']['D_axis_given_F']['D-TREATMENT-BELOW-SCALE']:.3f}")

    # ── 네 배치 공통(✅인용 가능한 단서형) --------------------------------
    _, det = gate_detail(T14, 'G-INVERT')
    add('TInvert', re.search(r'위반\s*(\S+)', det).group(1))
    _, det = gate_detail(T14, 'G-S-BLIND')
    add('TSBlind', re.search(r'위반\s*(\S+)', det).group(1))

    # ★[2026-09-05] 동결 행 재집계 · 공시 노트 · 게이트 문자열에서 나오는 원자들.
    for k, v in list(sweep_ladder().items()) + list(peek_counts().items()) \
            + list(t16_gate_atoms().items()) + list(t16_scale_shape().items()):
        add(k, v)
    return out


def render():
    check_pins()
    body = '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in rows())
    return (
        '% AUTO-GENERATED by scripts/gen_thesis_t11t14_constants.py -- DO NOT EDIT BY HAND.\n'
        '% 입력 = 동결 verdict_t11..t16.json. 학위논문은 이 매크로만 쓴다.\n'
        '% ⚠파일 이름은 역사적 이유로 t11t14 이나 실제 범위는 T11--T16 이다.\n'
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
