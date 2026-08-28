"""M2 게이트 — **판정 규칙·문턱·팔·표본·해석 규칙의 유일한 정의**.

    python3 tmp/M2_LOCTUNE_2026-08-28/gates_m2.py --selftest   # 자가진단(런 0회·파일 0개)
    python3 tmp/M2_LOCTUNE_2026-08-28/gates_m2.py              # 런 0회 게이트

★**손복사 상수 0건**을 목표로 한다:
  · 문턱(NOMINAL/CONFIRM/WEAK/NEG_MIN/FAMILY_ALPHA) → `gates_b26` 에서 import
  · 현행 기준선 loc(25.0)·앙상블(600)·배경(jonswap) → `gates_b11.SPEC` 에서 import
  · loc 사다리 {6.25,12.5,25,50,100} → `run_s17.ARMS` 에서 **읽는다**
  · S17 적격성 상수(zero-loss ≤ 0.10 · skill ≤ −0.05)·동점규칙 → `analyze_s17.py`
    **소스에서 파싱**한다(⛔손으로 옮기지 않는다 — "손복사 기준 상수 사냥꾼" 처방)
  · 표본(에피소드 목록)·코너 정의 → `gates_b26.select()` / `gates_b26.CORNERS`
  · λ_p → `wavesim.wave_generator.wavenumber_from_omega` (저장소 자신의 분산관계 해)
"""
import glob
import hashlib
import json
import os
import re
import statistics as st
import sys
from math import comb, pi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B26D = os.path.join(ROOT, 'tmp', 'B26_SEASTATE_2026-08-27')
B12D = os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23')
B11D = os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23')
S16D = os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')
S17D = os.path.join(ROOT, 'tmp', 'S17_WHITE25_2026-08-18')
for _p in (HERE, B26D, B12D, B11D, S16D, S17D, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_b26 as B26                                        # noqa: E402
from gates_b11 import (EST_SEEDS, FILTER_SEEDS, SPEC,          # noqa: F401,E402
                       sign_test_one_sided, med_by_episode, paired)

# ─────────────────────────────────────────────────────────────────────────────
# 0. 방향 고정 — ★**결과 전에 박는다** (정지규칙 긴장의 정당화 조건)
# ─────────────────────────────────────────────────────────────────────────────
DIRECTION = (
    'BASELINE-STRENGTHENING-ONLY :: 이 배치는 기준선을 **강하게만** 만든다. '
    '① 개입은 `loc_radius` 하나이고 그 후보 격자는 **S17 이 동결한 사다리**에서 '
    '**자료(λ_p)로 도출**된다 — 사후 자유도 0. '
    '② 선택은 `BEST = argmin_a W_a`(현행 `SPEC25` 를 **후보 풀에 포함**)이므로 '
    '`W_BEST ≤ W_SPEC25` 가 **항등적으로** 성립한다 = 부호 마진은 커질 수 없다. '
    '③ 인용 크기는 `MAGNITUDE_FLOOR`(= 두 med(rel) 중 **덜 유리한** 쪽)로 막는다 '
    '= 크기 마진도 커질 수 없다. '
    '⟹ ★**어느 분기로 가도 M2 는 마진을 줄이거나·지우거나·그대로 두며, 절대 키우지 못한다.**')

# ─────────────────────────────────────────────────────────────────────────────
# 1. 후보 loc 격자 — ★**손으로 고르지 않는다. 자료(λ_p)에서 도출한다.**
# ─────────────────────────────────────────────────────────────────────────────
LOC_INCUMBENT = float(SPEC['loc'])          # 25.0  ← gates_b11.SPEC (B-26 기준선의 정의)
BG_INCUMBENT = SPEC['background']           # 'jonswap'
NENS_INCUMBENT = int(SPEC['n_ensemble'])    # 600


def ladder():
    """★S17 이 동결한 loc 사다리를 **그 배치의 팔 정의에서 읽는다**."""
    import run_s17
    return tuple(sorted({float(a['loc']) for a in run_s17.ARMS.values()}))


def lam_p(Tp, depth):
    """λ_p = 2π/k_p, k_p 는 **저장소 자신의** 분산관계 해에서. ⛔손공식 금지."""
    from wavesim.wave_generator import wavenumber_from_omega
    return 2.0 * pi / wavenumber_from_omega(2.0 * pi / float(Tp), depth=float(depth))


def sweep_corpus_lambda():
    """`loc=25` 를 고른 스윕이 돌았던 코퍼스(S16 혼잡 64)의 λ_p 중앙값.

    ⛔코퍼스를 손으로 적지 않는다 — `run_s16.CROWDED_DIR` 를 그대로 걸어
    파일명의 BF 를 `wavesim.pipeline.SEA_STATE` 로 옮긴다."""
    import run_s16
    from wavesim.pipeline import SEA_STATE
    depth = B26.gen_params()['depth']
    lams, comp = [], {}
    for d in run_s16.CROWDED_DIR:
        for p in sorted(glob.glob(os.path.join(ROOT, d, '*.h5'))):
            m = re.match(r'episode_(BF\d)_(\d+)comp_', os.path.basename(p))
            if not m or int(m.group(2)) < 15:
                continue
            bf = m.group(1)
            comp[bf] = comp.get(bf, 0) + 1
            lams.append(lam_p(SEA_STATE[bf]['Tp'], depth))
    return dict(n=len(lams), composition=comp, median=st.median(lams),
                by_bf={bf: lam_p(SEA_STATE[bf]['Tp'], depth) for bf in sorted(comp)})


def loc_star(corner):
    """★도출식 (한 줄):

        loc*(corner) = LOC_INCUMBENT × λ_p(corner) / λ_p(스윕 코퍼스)

    = *"`loc=25` 를 고른 코퍼스와 **같은 loc/λ_p** 를 이 코너에 준다"*.
    """
    depth = B26.gen_params()['depth']
    return (LOC_INCUMBENT * lam_p(B26.CORNERS[corner]['Tp'], depth)
            / sweep_corpus_lambda()['median'])


def candidates(corner):
    """★동결 선택 규칙 (사다리는 S17, 규칙은 이 세 줄):

    C(corner) = { L ∈ 사다리 : loc* ≤ L ≤ LOC_INCUMBENT }
              ∪ { 사다리 바닥 }  if loc* < 사다리 바닥
    """
    L = ladder()
    ls = loc_star(corner)
    c = {x for x in L if ls <= x <= LOC_INCUMBENT}
    if ls < min(L):
        c.add(min(L))
    return tuple(sorted(c))


# ★동결 기대값 — 결과 0행에서 위 함수를 돌려 얻었다. `G-L2` 가 매 실행 재도출·대조한다.
FROZEN_CANDIDATES = {'SS-REAL': (6.25, 12.5, 25.0),
                     'SS-SHORT': (6.25, 12.5, 25.0),
                     'SS-BF4': (25.0,)}
FROZEN_LADDER = (6.25, 12.5, 25.0, 50.0, 100.0)

# ─────────────────────────────────────────────────────────────────────────────
# 2. 팔 — ⛔빈 dict 금지. `run_s16.ARMS` 와 **같은 스키마**로 등록한다.
#    현행 `SPEC25` 는 이미 `run_s16.ARMS` 에 있고 **건드리지 않는다**.
# ─────────────────────────────────────────────────────────────────────────────
INCUMBENT_ARM = 'SPEC25'
NEW_ARMS = {
    'SPEC12': dict(kind='spec', background=BG_INCUMBENT, loc=12.5),
    'SPEC6': dict(kind='spec', background=BG_INCUMBENT, loc=6.25),
}
ARM_LOC = {'SPEC25': 25.0, 'SPEC12': 12.5, 'SPEC6': 6.25}
ARM_ORDER = ('SPEC25', 'SPEC12', 'SPEC6')
SHIP_ARM = 'SHIP-NOW'

PRIMARY_CORNER = 'SS-REAL'          # 임무의 질문이 걸린 칸
SECONDARY_CORNER = 'SS-SHORT'       # 같은 λ_p (Tp 동일) ⟹ 두 번째 인용 크기(−42.17%)를 함께 해결
CONTROL_CORNER = 'SS-BF4'           # ★도출이 {25.0} 만 주는 칸 = **정합된 끝** (용량-반응 대조)
CORNERS_M2 = (PRIMARY_CORNER, SECONDARY_CORNER, CONTROL_CORNER)
# ★동결 — `G-1` 이 이것과 대조한다(코너 집합이 조용히 바뀌는 것을 막는다).
#   ⛔`SS-LOWH` 는 **결과 전에** 제외했다: λ_p 가 `SS-BF4` 와 같으므로 대조 칸의 **복제**이고,
#   도출은 그 칸에도 `{25.0}` 만 준다(= 재조율 대상 아님). 사유는 PREREG §2.2.
FROZEN_CORNERS = ('SS-REAL', 'SS-SHORT', 'SS-BF4')

# ─────────────────────────────────────────────────────────────────────────────
# 3. 문턱 — ★**B-26 에서 import**. ⛔이 파일에 문턱 리터럴이 없다.
# ─────────────────────────────────────────────────────────────────────────────
NOMINAL, CONFIRM, WEAK = B26.NOMINAL, B26.CONFIRM, B26.WEAK
NEG_MIN = B26.NEG_MIN
FAMILY_ALPHA = B26.FAMILY_ALPHA          # 0.05/4 — ★B-26 의 4코너 가족을 **그대로 승계**
ALPHA = B26.ALPHA
N_PER_CORNER = B26.N_PER_CORNER          # 24
# D-2(기술적): 조율 팔이 현행 팔보다 나은가 — 단측 부호검정, 명목 α, 코너 간 Holm
D2_THRESH = NOMINAL                      # ≥17/24  (p=0.031957 ≤ 0.05)


def threshold_table(n=None):
    n = n or N_PER_CORNER
    return {'n': n, 'source': 'gates_b26 (import, 손복사 0)',
            'p_nominal': sign_test_one_sided(NOMINAL, n),
            'p_confirm': sign_test_one_sided(CONFIRM, n),
            'p_weak_le': sum(comb(n, k) for k in range(0, WEAK + 1)) / 2 ** n,
            'family_alpha': FAMILY_ALPHA,
            'confirm_within_family': sign_test_one_sided(CONFIRM, n) <= FAMILY_ALPHA,
            'weak_is_sibling': WEAK == n - NOMINAL,
            'd2_thresh': D2_THRESH,
            'p_d2': sign_test_one_sided(D2_THRESH, n)}


# ─────────────────────────────────────────────────────────────────────────────
# 4. 적격성·선택 — ★S17 의 **동결 프로토콜**. 상수도 규칙도 소스에서 읽는다.
# ─────────────────────────────────────────────────────────────────────────────
S17_ANALYZER = os.path.join(S17D, 'analyze_s17.py')
S17_TIEBREAK_SIG = '(W[a], -rel[a], COORD[a][2], a)'


def s17_rules():
    """★`analyze_s17.py` **소스에서** R-1a′(b)·R-1b′ 의 두 상수와 동점규칙을 읽는다.

    ⛔이 두 수를 이 파일에 리터럴로 적지 않는다 (`AMENDMENT_1 A-4/A-5` 가 원문,
    그 원문을 **집행한 코드**가 정본이다)."""
    src = open(S17_ANALYZER, encoding='utf-8').read()
    m1 = re.search(r'loss\s*<=\s*([0-9.]+)', src)
    m2 = re.search(r'skill\s*<=\s*(-[0-9.]+)', src)
    if not (m1 and m2):
        raise RuntimeError('analyze_s17.py 에서 적격성 상수를 파싱하지 못했다')
    return dict(zero_loss_max=float(m1.group(1)), skill_max=float(m2.group(1)),
                tiebreak_in_source=(S17_TIEBREAK_SIG in src))


def arm_stats(ship, arm_med, zero, raw_rows):
    """한 팔의 통계 — 적격성 판정에 필요한 전부."""
    ks = sorted(set(arm_med) & set(zero))
    az = paired(arm_med, zero)
    skill = st.median([(arm_med[k] - zero[k]) / zero[k] for k in ks]) if ks else None
    loss = sum(1 for r in raw_rows
               if r.get('zero_nRMSE') is not None
               and r['avg_nRMSE'] >= r['zero_nRMSE'])
    nraw = sum(1 for r in raw_rows if r.get('zero_nRMSE') is not None)
    pr = paired(ship, arm_med)
    return dict(n=pr['n'], W=pr['better'], rel=pr['rel_median_pct'],
                p=pr['p_one_sided'], mean=(st.mean(arm_med.values()) if False
                                           else (st.mean(list(arm_med.values()))
                                                 if arm_med else None)),
                median=(st.median(list(arm_med.values())) if arm_med else None),
                a_lt_zero=az['better'], a_lt_zero_n=az['n'],
                skill=skill, zero_loss=(loss / nraw if nraw else None),
                zero_loss_raw=[loss, nraw])


def eligible(name, s):
    """`P_elig` 판정 — ★두 규약을 **둘 다** 낸다.

    · `b26`  : B-26 이 동결한 음성 대조 (부호 ≥ NEG_MIN/24). **PRIMARY 는 이것을 쓴다**
               (B-26 의 라벨 기계를 그대로 승계하기 위함).
    · `s17`  : S17 의 적격성 (R-1a′(b) zero-loss ≤ 10% ∧ R-1b′ skill ≤ −5%).
               **PRIMARY 를 바꾸지 않는다** — §해석 규칙의 **공시 의무**를 발동시킨다.
    """
    r = s17_rules()
    b26_ok = (s['a_lt_zero_n'] > 0 and s['a_lt_zero'] >= NEG_MIN)
    s17_ok = (b26_ok and s['zero_loss'] is not None and s['skill'] is not None
              and s['zero_loss'] <= r['zero_loss_max'] and s['skill'] <= r['skill_max'])
    return dict(arm=name, b26=bool(b26_ok), s17=bool(s17_ok),
                zero_loss_max=r['zero_loss_max'], skill_max=r['skill_max'])


def select_best(stats, elig, key='b26'):
    """★S17 §1.1 동결: `BEST = argmin_a W_a`,
    동점 1차 med(rel) **큰** 쪽 · 2차 `loc` **작은** 쪽 · 3차 이름 사전순."""
    pool = [a for a in ARM_ORDER if a in stats and elig[a][key]]
    if not pool:
        return None
    return sorted(pool, key=lambda a: (stats[a]['W'], -stats[a]['rel'],
                                       ARM_LOC[a], a))[0]


def magnitude_floor(stats, best):
    """★크기 바닥 — **결과 전 동결**.

    인용 크기 = `max(med(rel)_BEST, med(rel)_SPEC25)` = 둘 중 **덜 유리한** 쪽.
    ⟹ M2 는 인용 크기를 **줄일 수만 있다**. (§0 DIRECTION ③)"""
    a = stats[best]['rel']
    b = stats[INCUMBENT_ARM]['rel']
    if a is None or b is None:
        return None, None
    cited = max(a, b)
    return cited, (INCUMBENT_ARM if cited == b and b != a else best)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ★결과 전 해석 규칙 — 어느 결과가 나와도 무엇을 쓸지 **지금** 정한다
# ─────────────────────────────────────────────────────────────────────────────
VERDICTS = ('TUNING-SHRINKS-MARGIN', 'TUNING-NULL', 'TUNING-HURTS',
            'NO-ELIGIBLE-BASELINE', 'LABEL-FALLS')

READING = {
    'TUNING-SHRINKS-MARGIN': (
        '★**조율된 기준선이 유의하게 낫다.** ⛔***`SS-REAL` 의 `−42.42%` 를 크기로 '
        '인용하는 것을 철회한다*** — 그 수는 미조율 `loc=25` 기준선에 대한 것이다. '
        '남는 것은 **부호**(24/24)와 ★**조율된 기준선 대비 새 크기**이고, 인용값은 '
        '`MAGNITUDE_FLOOR` 가 고른 쪽이다. `SS-SHORT` 의 `−42.17%` 도 같은 규칙으로 '
        '갱신한다. **F2 공시가 "미정량 인플레" 에서 "정량됨"으로 바뀐다.**'),
    'TUNING-NULL': (
        '★**F2 가 닫힌다** — `loc/λ_p` 4.02배 불일치가 짧은 주기 코너의 마진을 '
        '설명하지 못한다 ⟹ `−42.42%` 를 **크기로 인용할 수 있다**. '
        '⛔단 *"loc 이 무관하다"* 로 쓰지 마라 — **MDE 를 반드시 병기**한다 '
        '(`mde` 필드: 라벨이 떨어지려면 기준선이 몇 % 좋아져야 하는가). '
        '⚠그리고 이 널은 **loc 축 한정**이다 — 필터 격자 40×30 은 이 배치가 '
        '건드리지 않았고(도달 불가, §도달성) 배경·팽창·앙상블도 마찬가지다.'),
    'TUNING-HURTS': (
        '★**F2 가 강한 형태로 닫힌다** — 사다리를 아래로 내리면 기준선이 **나빠진다** '
        '⟹ `loc=25` 는 이 코너들에서 **이미 최적점 이하**다. `−42.42%` 인용 가능. '
        '★`S17` 의 백색 배경 결과(`loc 25→12.5` 0/64, `12.5→6.25` 0/64)가 '
        '**스펙트럼 배경·짧은 주기 코너로 확장**된다. MDE 병기 의무는 동일.'),
    'NO-ELIGIBLE-BASELINE': (
        '⛔**적격 기준선이 없다** — 사다리의 어느 칸도 `P_elig` 에 들어가지 못한다. '
        '⟹ `SS-REAL` 에서 **우열을 말하지 않는다**(`B-9b`·`B-11` 규약). '
        '크기 인용 철회, `SHIP < ZERO` 를 보고한다. ★이 라벨도 결과다: '
        '실측 해상상태에서는 **조율된 최강 기준선조차** 자격 문턱을 못 넘는다.'),
    'LABEL-FALLS': (
        '★★**조율만으로 `B-26` 의 코너 라벨이 떨어진다.** ⛔***`SS-REAL` = '
        '`HOLDS (CONFIRMATORY)` 를 철회하고 새 라벨로 갱신한다*** — 크기뿐 아니라 '
        '**판정 자체가 미조율 기준선의 산물**이었다는 뜻이다. '
        '논문 §「작동 경계」의 해상상태 절을 다시 쓴다.'),
}

# ★공시 의무 (라벨과 **독립**으로 발동한다)
DISCLOSURE = {
    's17_fails': (
        '⚠**공시 의무**: 선택된 비교자가 `S17` 의 기술점수 크기 게이트 `R-1b′` '
        '(제로필드 대비 중앙 ≥5% 개선)를 **통과하지 못한다**. 마진을 인용할 때 '
        '*"비교자는 부호검정은 통과하나 동결 조율 프로토콜의 5% 기술 바닥은 넘지 못한다"* '
        '를 **함께** 적는다. (런 0회 실측: `SPEC25` 는 `SS-REAL` −3.96% · '
        '`SS-SHORT` −4.04% 로 **둘 다 미달**, `SS-BF4` 는 −16.17% 로 통과.)'),
    'grid_floor': (
        '⚠**공시 의무**: `BEST` 의 loc 이 사다리 바닥 6.25 m 다. 이 격자(40×30, '
        'dx 6.1538 m)에서 GC 지지폭 `2c` = **2.03 dx** 이므로 **국소화 최적점은 '
        '이 격자에서 정의 가능한 하한**이고 **loc 은 격자 해상도와 교락**된다 '
        '(`S17 AMENDMENT_1 A-6`·`RESULT §6.5` 승계).'),
    'control_moved': (
        '⚠**공시 의무**: **정합 대조 칸 `SS-BF4` 에서도** 조율이 기준선을 개선했다. '
        '⟹ 이것은 **코너 특이 교락(F2)이 아니라 전역 미조율**이다 — `B-1`·`B-11`·'
        '`B-12`·`B-16`·`B-19`·`B-22` 의 `SPEC25` 대비 크기가 **전부** 같은 인플레를 '
        '안고 있다는 뜻이므로, 그 사실을 **저장소 차원의 공시**로 올린다.'),
    'derivation_off': (
        '⚠**공시 의무**: 도출은 `SS-BF4` 에 `{25.0}` 만 준다(loc* = 18.99 m ⟹ '
        '사다리에서 25 가 최근접). 그 칸에서 6.25·12.5 를 돌린 것은 **도출 밖 대조**다.'),
}


def verdict_label(corner_res):
    """★라벨을 **코드가** 고른다. 우선순위는 결과 전 동결."""
    if corner_res['best'] is None:
        return 'NO-ELIGIBLE-BASELINE'
    if corner_res['b26_label_new'] != corner_res['b26_label_frozen']:
        return 'LABEL-FALLS'
    if corner_res['d2_best_better']:
        return 'TUNING-SHRINKS-MARGIN'
    if corner_res['d2_any_worse'] and not corner_res['d2_any_better']:
        return 'TUNING-HURTS'
    return 'TUNING-NULL'


# ─────────────────────────────────────────────────────────────────────────────
# 6. 표본 — B-26 의 **그 파일들**. 생성 0회.
# ─────────────────────────────────────────────────────────────────────────────
def episodes():
    eps, _ = B26.select()
    out = [e for e in eps if e['corner'] in CORNERS_M2]
    for e in out:
        e['bf'] = e['corner']
    return out


B26_RESULTS = os.path.join(B26D, 'results_b26_*.jsonl')


def frozen_rows():
    """★`B-26` 의 동결 행 — `SHIP-NOW` 와 `SPEC25` 를 **다시 돌리지 않고 그대로 쓴다**."""
    rows = []
    for p in sorted(glob.glob(B26_RESULTS)):
        if 'smoke' in os.path.basename(p):
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r and r.get('corner') in CORNERS_M2:
                rows.append(r)
    return rows


# ★양성대조 대상 — 결과 전 동결. 코너별 ncomp 15/30 각 1개(사전 순).
def pc_episodes():
    out = []
    for c in CORNERS_M2:
        for nc in B26.NCOMPS:
            cand = sorted([e for e in episodes()
                           if e['corner'] == c and e['ncomp'] == nc],
                          key=lambda e: e['episode'])
            out.append(cand[0])
    return out


PC_PATH = os.path.join(HERE, 'pc_m2.json')


# ─────────────────────────────────────────────────────────────────────────────
# 7. 게이트
# ─────────────────────────────────────────────────────────────────────────────
# ★0행에서 **참이 되어도 되는** 게이트만 여기 적는다. 자가진단 [2] 가 이 밖의 공허한 통과를
#   FAIL 로 센다. 넷이 무해한 이유: **행이 있는지는 `G-12` 가 따로 요구**하고
#   `G-PC1`/`G-PC2` 는 `pc_n > 0` 을 스스로 요구하므로 0행에서 이미 FAIL 이다.
ROWLESS_OK = ('G-11', 'G-Z', 'G-16', 'G-18')


def _gid(name):
    """게이트 이름에서 식별자만 뽑는다(장식 ★ 제거) — 자가진단이 이름 장식에 걸리지 않게."""
    return name.lstrip('★ ').split()[0]
MESH_GRID = (40, 30)              # run_s16.run_letkf 의 격자 (도달 불가 — G-L4 가 확인)


def _grid_dxdy():
    """실제 에피소드의 `gt_grid` 에서 필터 격자 dx·dy 를 **재도출**한다."""
    import h5py
    import numpy as np
    p = episodes()[0]['path']
    with h5py.File(p, 'r') as f:
        gx = f['frames/gt_grid_x'][:]
        gy = f['frames/gt_grid_y'][:]
    egx = np.linspace(gx[0], gx[-1], MESH_GRID[0])
    egy = np.linspace(gy[0], gy[-1], MESH_GRID[1])
    return float(egx[1] - egx[0]), float(egy[1] - egy[0])


def gates_runfree():
    out = []
    L = ladder()
    out.append(('G-L1 loc 사다리 == S17 팔 정의', L == FROZEN_LADDER, str(L)))
    sw = sweep_corpus_lambda()
    out.append(('G-L1b 스윕 코퍼스 = S16 혼잡 64 (BF4 32 + BF5 32)',
                sw['n'] == 64 and sw['composition'] == {'BF4': 32, 'BF5': 32},
                f'n={sw["n"]} {sw["composition"]} λ_median={sw["median"]:.4f}'))
    ok, msg = True, []
    for c in CORNERS_M2:
        got = candidates(c)
        exp = FROZEN_CANDIDATES.get(c)
        ok &= (got == exp)
        msg.append(f'{c}: loc*={loc_star(c):.4f} → {got}')
    out.append(('★G-L2 후보 격자 재도출 == 동결', ok, ' · '.join(msg)))
    dx, dy = _grid_dxdy()
    sup = {a: (2 * ARM_LOC[a] / dx) for a in ARM_ORDER}
    out.append(('★G-L3 GC 지지폭 == S17 A-6 표 (2c/dx = 8.12·4.06·2.03)',
                all(abs(sup[a] - v) < 0.01 for a, v in
                    (('SPEC25', 8.125), ('SPEC12', 4.0625), ('SPEC6', 2.0312))),
                f'dx={dx:.4f} dy={dy:.4f} ' +
                ' '.join(f'{a}:{sup[a]:.2f}dx' for a in ARM_ORDER)))
    try:
        r = s17_rules()
        out.append(('★G-L5 S17 적격성 상수·동점규칙을 소스에서 파싱',
                    r['zero_loss_max'] == 0.10 and r['skill_max'] == -0.05
                    and r['tiebreak_in_source'],
                    f'zero_loss≤{r["zero_loss_max"]} skill≤{r["skill_max"]} '
                    f'tiebreak_in_source={r["tiebreak_in_source"]}'))
    except Exception as e:                                     # noqa: BLE001
        out.append(('★G-L5 S17 적격성 상수·동점규칙을 소스에서 파싱', False, repr(e)[:90]))
    out.append(('★G-L6 선택규칙 양성대조: S17 동결 판정 재현 (BEST=SPEC25)',
                _pc_select_s17(), 'verdict_s17.json 의 W/rel/elig → SPEC25'))
    t = threshold_table()
    out.append(('G-19 문턱 산술 (B-26 에서 import)',
                t['confirm_within_family'] and t['weak_is_sibling'],
                f'NOMINAL {NOMINAL} p={t["p_nominal"]:.5f} · CONFIRM {CONFIRM} '
                f'p={t["p_confirm"]:.5f} ≤ {FAMILY_ALPHA} · D-2 {D2_THRESH} '
                f'p={t["p_d2"]:.5f}'))
    d = B26.deploy_constants()
    out.append(('G-20 배포 상수 == 패키지 단일출처 == 사전등록값',
                all(d[f'DEFAULT_CONFIG.{k}'] == v for k, v in B26.DEPLOY_EXPECTED.items())
                and d['DEFAULT_CONFIG.varpro_steps'] == d['SU.VARPRO_STEPS_DEFAULT']
                and d['DEFAULT_CONFIG.learned_init_iters'] == d['SU.LEARNED_INIT_ITERS_DEFAULT']
                and d['DEFAULT_CONFIG.auto_switch_thresh'] == d['SU.AUTO_SWITCH_THRESH_DEFAULT']
                and d['DEFAULT_CONFIG.lambda_l1'] == d['SU.LAMBDA_L1_DEFAULT'],
                str({k: d[f'DEFAULT_CONFIG.{k}'] for k in B26.DEPLOY_EXPECTED})))
    eps = episodes()
    out.append(('G-1 표본 == B-26 세 코너 72 에피소드 (코너 집합도 동결과 대조)',
                tuple(CORNERS_M2) == FROZEN_CORNERS
                and len(eps) == len(FROZEN_CORNERS) * N_PER_CORNER
                and all(e['corner'] in FROZEN_CORNERS for e in eps),
                f'{len(eps)}/{len(FROZEN_CORNERS) * N_PER_CORNER} {tuple(CORNERS_M2)}'))
    out.append(('★G-L4 도달성: loc 은 ARMS 로 도달 · 격자 40×30 은 **도달 불가**',
                _reachability()['loc_reachable']
                and not _reachability()['grid_reachable'],
                json.dumps(_reachability(), ensure_ascii=False)))
    return out


def _reachability():
    """★도달 가능성을 **소스에서** 확인한다 (산문이 아니라 파싱)."""
    src = open(os.path.join(S16D, 'run_s16.py'), encoding='utf-8').read()
    return dict(
        loc_reachable=bool(re.search(r"loc_radius=spec\['loc'\]", src)),
        loc_sites=len(re.findall(r"loc_radius=spec\['loc'\]", src)),
        grid_reachable=bool(re.search(r'np\.linspace\([^)]*,\s*(?:nx|NX|spec\[)', src)),
        grid_literal=bool(re.search(r'egx = np\.linspace\(gx\[0\], gx\[-1\], 40\)', src)),
        readback=bool(re.search(r'loc_radius=float\(f\.loc_radius\)', src)))


def _pc_select_s17():
    """★선택 규칙의 **교차 배치 양성대조** — S17 의 동결 W/rel/elig 를 먹여
    `BEST == 'SPEC25'` 가 나오는지. (동점 W=59 인 `SPEC25`/`SPEC100` 이 동점규칙을 실제로 태운다.)"""
    v = json.load(open(os.path.join(S17D, 'verdict_s17.json')))
    W, rel, elig = v['W'], v['rel'], set(v['elig'])
    loc = {'WHITER50': 50.0, 'WHITER25': 25.0, 'WHITER12': 12.5, 'WHITER6': 6.25,
           'WHITERHS50': 50.0, 'SPEC25': 25.0, 'SPEC50': 50.0, 'SPEC100': 100.0}
    pool = [a for a in sorted(W) if a in elig]
    best = sorted(pool, key=lambda a: (W[a], -rel[a] * 100, loc[a], a))[0]
    return best == v['best'] == 'SPEC25'


def gates_data(exists=None):
    exists = exists or os.path.exists
    out = []
    eps = episodes()
    miss = [e['episode'] for e in eps if not exists(e['path'])]
    out.append(('G-6 72 에피소드 파일 존재 (생성 0회)', not miss,
                f'결측 {len(miss)}' + (f' 예: {miss[0]}' if miss else '')))
    fr = frozen_rows()
    sh = med_by_episode(fr, SHIP_ARM)
    sp = med_by_episode(fr, INCUMBENT_ARM)
    out.append(('★G-F1 B-26 동결 행 수입: SHIP 72 · SPEC25 72 에피소드',
                len(sh) == 72 and len(sp) == 72, f'{len(sh)} / {len(sp)}'))
    return out


def gates_run(rows, frozen=None):
    """추정 후. `rows` = M2 의 새 행 (+ 양성대조 행)."""
    fr = frozen_rows() if frozen is None else frozen
    ok = [r for r in rows if 'error' not in r]
    out = [('G-11 에러 0행', len(rows) == len(ok), f'{len(rows) - len(ok)}')]
    new = [r for r in ok if r.get('arm') in NEW_ARMS and not r.get('pc')]
    for a in ('SPEC12', 'SPEC6'):
        n = len([r for r in new if r['arm'] == a])
        out.append((f'G-12 {a} 216행', n == 72 * len(FILTER_SEEDS), f'{n}/216'))
    # ★개입 검증 — 항등식이 아니라 **필터 객체 되읽기** 두 곳 + 파생량
    bad = []
    for r in new:
        want = ARM_LOC[r['arm']]
        ec = r.get('effective_config') or {}
        if (r.get('loc_radius') != want or float(ec.get('loc_radius', -1)) != want
                or r.get('background') != BG_INCUMBENT
                or r.get('n_ensemble') != NENS_INCUMBENT
                or r.get('filter_class') != 'LETKFGridSpec'):
            bad.append(r.get('episode'))
    out.append(('★G-16 개입 되읽기: `f.loc_radius` **와** `readback(f)` 둘 다 == 팔의 loc '
                '(+ 배경 jonswap · 앙상블 600 · LETKFGridSpec)', not bad, f'위반 {len(bad)}'))
    # ★개입이 **no-op 이 아니다** — 세 팔이 비트 동일이면 잰 것이 없다
    #   (`B-2f` 교훈: `MAXN-10` 이 결합 팔과 12/12 비트 동일이었다).
    fspec = {(r['episode'], r['filter_seed']): r['avg_nRMSE'] for r in fr
             if r.get('arm') == INCUMBENT_ARM and r.get('filter_seed') is not None}
    noop = {a: 0 for a in NEW_ARMS}
    ncmp = {a: 0 for a in NEW_ARMS}
    for r in new:
        k = (r.get('episode'), r.get('filter_seed'))
        if k in fspec:
            ncmp[r['arm']] += 1
            noop[r['arm']] += (r['avg_nRMSE'] == fspec[k])
    out.append(('★G-16b 개입이 no-op 이 아니다 (팔이 SPEC25 와 비트 동일한 런의 비율 < 50%)',
                all(ncmp[a] > 0 and noop[a] / ncmp[a] < 0.5 for a in NEW_ARMS),
                ' '.join(f'{a}:{noop[a]}/{ncmp[a]}' for a in NEW_ARMS)))
    # ★단일요인 — loc 말고는 아무것도 안 움직였다 (readback 을 SPEC25 동결 행과 비교)
    keys = ('n_ensemble', 'mult_inflation', 'add_inflation', 'obs_noise', 'max_obs',
            'depth', 'dt', 'nx', 'ny', 'n_grid', 'n_state', 'dx', 'dy', 'g')
    ref = {}
    for r in fr:
        if r.get('arm') == INCUMBENT_ARM and r.get('cap_effective_config'):
            ref = r['cap_effective_config']
            break
    if not ref:
        for r in fr:
            if r.get('arm') == INCUMBENT_ARM and r.get('effective_config'):
                ref = r['effective_config']
                break
    diff = set()
    if ref:
        for r in new:
            ec = r.get('effective_config') or {}
            for k in keys:
                if k in ref and k in ec and ec[k] != ref[k]:
                    diff.add(k)
    out.append(('★G-13 단일요인: `loc_radius` 외 필터 설정 차이 0',
                bool(ref) and not diff, f'ref={"있음" if ref else "없음"} 차이={sorted(diff)}'))
    # ★공짜 양성대조 — zero_nRMSE 는 GT 만의 함수 ⟹ 팔이 달라도 **비트 동일**이어야 한다
    fz = med_by_episode(fr, INCUMBENT_ARM, field='zero_nRMSE')
    mism = []
    for a in ('SPEC12', 'SPEC6'):
        za = med_by_episode(new, a, field='zero_nRMSE')
        mism += [e for e in za if e in fz and za[e] != fz[e]]
    out.append(('★G-Z zero_nRMSE 가 팔 간 **비트 동일** (GT 만의 함수)',
                not mism, f'불일치 {len(mism)}'))
    # ★양성대조 — SPEC25·SHIP 재실행이 B-26 동결 행과 비트 동일
    pc = [r for r in ok if r.get('pc')]
    pcm = _pc_compare(pc, fr)
    out.append(('★G-PC1 SPEC25 재실행 == B-26 동결 행 (비트 동일)',
                pcm['spec_n'] > 0 and pcm['spec_bad'] == 0,
                f'{pcm["spec_n"]}행 불일치 {pcm["spec_bad"]} max|Δ|={pcm["spec_max"]}'))
    out.append(('★G-PC2 SHIP-NOW 재실행 == B-26 동결 행 (비트 동일)',
                pcm['ship_n'] > 0 and pcm['ship_bad'] == 0,
                f'{pcm["ship_n"]}행 불일치 {pcm["ship_bad"]} max|Δ|={pcm["ship_max"]}'))
    out.append(('★G-18 계측 투명: `cap_avg_nRMSE` == `avg_nRMSE` 비트 동일',
                all(r.get('cap_avg_nRMSE') == r['avg_nRMSE'] for r in new
                    if 'cap_avg_nRMSE' in r),
                f'{sum(1 for r in new if "cap_avg_nRMSE" in r)}행'))
    return out


def _pc_compare(pc, frozen):
    idx = {}
    for r in frozen:
        k = (r.get('arm'), r.get('episode'),
             r.get('est_seed') if r.get('arm') == SHIP_ARM else r.get('filter_seed'))
        idx[k] = r.get('avg_nRMSE')
    res = dict(spec_n=0, spec_bad=0, spec_max=0.0, ship_n=0, ship_bad=0, ship_max=0.0)
    for r in pc:
        arm = r.get('arm')
        k = (arm, r.get('episode'),
             r.get('est_seed') if arm == SHIP_ARM else r.get('filter_seed'))
        if k not in idx:
            continue
        tag = 'ship' if arm == SHIP_ARM else 'spec'
        d = abs(r['avg_nRMSE'] - idx[k])
        res[tag + '_n'] += 1
        res[tag + '_max'] = max(res[tag + '_max'], d)
        if r['avg_nRMSE'] != idx[k]:
            res[tag + '_bad'] += 1
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 8. 집계 + 판정
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(new_rows, frozen=None):
    fr = frozen_rows() if frozen is None else frozen
    new = [r for r in new_rows if 'error' not in r and not r.get('pc')]
    out = {}
    for c in CORNERS_M2:
        f_c = [r for r in fr if r.get('corner') == c]
        n_c = [r for r in new if r.get('corner') == c]
        ship = med_by_episode(f_c, SHIP_ARM)
        zero = med_by_episode(f_c, INCUMBENT_ARM, field='zero_nRMSE')
        med = {INCUMBENT_ARM: med_by_episode(f_c, INCUMBENT_ARM)}
        raw = {INCUMBENT_ARM: [r for r in f_c if r.get('arm') == INCUMBENT_ARM]}
        for a in NEW_ARMS:
            med[a] = med_by_episode(n_c, a)
            raw[a] = [r for r in n_c if r.get('arm') == a]
        stats = {a: arm_stats(ship, med[a], zero, raw[a])
                 for a in ARM_ORDER if med[a]}
        elig = {a: eligible(a, stats[a]) for a in stats}
        best_b26 = select_best(stats, elig, 'b26')
        best_s17 = select_best(stats, elig, 's17')
        # D-2 (기술적): 조율 팔 vs 현행 팔
        d2 = {}
        for a in NEW_ARMS:
            if a in med and med[a]:
                d2[a] = paired(med[a], med[INCUMBENT_ARM])
        # B-26 라벨 기계를 그대로 (동결 라벨 vs 새 비교자 라벨)
        neg_inc = paired(med[INCUMBENT_ARM], zero)
        lab_frozen = B26.corner_label(paired(ship, med[INCUMBENT_ARM]), neg_inc)[0]
        if best_b26:
            neg_b = paired(med[best_b26], zero)
            lab_new, msg_new = B26.corner_label(paired(ship, med[best_b26]), neg_b)
        else:
            lab_new, msg_new = 'BASELINE-NO-SKILL', '적격 기준선 없음'
        cited, cited_from = (magnitude_floor(stats, best_b26)
                             if best_b26 else (None, None))
        out[c] = dict(
            corner=c, Hs=B26.CORNERS[c]['Hs'], Tp=B26.CORNERS[c]['Tp'],
            lam_p=lam_p(B26.CORNERS[c]['Tp'], B26.gen_params()['depth']),
            loc_star=loc_star(c), candidates=list(candidates(c)),
            stats=stats, elig=elig, best=best_b26, best_s17=best_s17,
            d2=d2,
            d2_best_better=bool(best_b26 in d2 and d2[best_b26]['better'] >= D2_THRESH),
            d2_any_better=any(v['better'] >= D2_THRESH for v in d2.values()),
            d2_any_worse=all(v['better'] <= N_PER_CORNER - D2_THRESH
                             for v in d2.values()) if d2 else False,
            b26_label_frozen=lab_frozen, b26_label_new=lab_new,
            b26_label_new_msg=msg_new,
            cited_rel_pct=cited, cited_from=cited_from,
            mde=mde(ship, med[INCUMBENT_ARM]),
            ship_vs_zero=paired(ship, zero))
    return out


def mde(ship, spec):
    """★MDE — 기준선을 균일 인수 f 로 강화했을 때 라벨이 떨어지는 지점 (S17 §0.3 규약)."""
    ks = sorted(set(ship) & set(spec))
    if not ks:
        return None
    rat = sorted(ship[k] / spec[k] for k in ks)
    n = len(ks)

    def need(w):        # W < w 가 되려면 f ≤ rat[w-1]
        return rat[w - 1] if 1 <= w <= n else None
    return {'n': n,
            'f_below_confirm': need(CONFIRM), 'f_below_nominal': need(NOMINAL),
            'f_to_weak': need(WEAK + 1),
            'pct_below_confirm': (1 - need(CONFIRM)) * 100 if need(CONFIRM) else None,
            'pct_below_nominal': (1 - need(NOMINAL)) * 100 if need(NOMINAL) else None,
            'pct_to_weak': (1 - need(WEAK + 1)) * 100 if need(WEAK + 1) else None}


def holm(pvals):
    """코너 간 Holm (D-2 는 기술적이므로 명목 α)."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, out, prev = len(order), {}, 0.0
    for i, k in enumerate(order):
        thr = ALPHA / (m - i)
        prev = max(prev, pvals[k])
        out[k] = bool(prev <= thr)
    return out


def disclosures(agg):
    """★공시 의무는 **인용 크기를 내는 두 칸**(PRIMARY·SECONDARY)을 함께 본다."""
    d = []
    cited = [agg.get(c, {}) for c in (PRIMARY_CORNER, SECONDARY_CORNER)]
    if any(x.get('best') and not x['elig'][x['best']]['s17'] for x in cited):
        d.append('s17_fails')
    if any(x.get('best') and ARM_LOC.get(x['best']) == min(FROZEN_LADDER)
           for x in cited):
        d.append('grid_floor')
    ctl = agg.get(CONTROL_CORNER, {})
    if ctl.get('d2_any_better'):
        d.append('control_moved')
    d.append('derivation_off')
    return d


# ─────────────────────────────────────────────────────────────────────────────
# 9. ★자가진단 — *게이트도 시험한다* (런 0회 · 파일 0개)
# ─────────────────────────────────────────────────────────────────────────────
def _fixture(scale=None, worse=False):
    """정상 케이스 픽스처. ⚠값은 **임의 상수** — 실측을 미리 보지 않는다."""
    # ⚠임의 상수. 방향이 섞여 있어 **결과를 미리 보는 것이 아니다**
    #   (팔이 서로 달라야 `G-16b` 가 정상 케이스에서 통과한다).
    scale = scale or {'SPEC25': 1.000, 'SPEC12': 1.004, 'SPEC6': 0.996}
    rows, frozen = [], []
    ec = dict(loc_radius=25.0, n_ensemble=600, mult_inflation=1.0, add_inflation=0.005,
              obs_noise=0.05, max_obs=40, depth=50.0, dt=0.1, nx=40, ny=30,
              n_grid=1200, n_state=2400, dx=6.1538, dy=6.8966, g=9.81)
    for e in episodes():
        h = (hash(e['episode']) % 100) / 1000.0
        z = 0.25 + h * 0.02
        base = 0.24 + h * 0.02
        shipv = base * 0.58
        for s in EST_SEEDS:
            frozen.append(dict(arm=SHIP_ARM, episode=e['episode'], corner=e['corner'],
                               est_seed=s, avg_nRMSE=shipv, zero_nRMSE=z))
        for s in FILTER_SEEDS:
            frozen.append(dict(arm=INCUMBENT_ARM, episode=e['episode'],
                               corner=e['corner'], filter_seed=s,
                               avg_nRMSE=base * scale['SPEC25'], zero_nRMSE=z,
                               loc_radius=25.0, n_ensemble=600, background='jonswap',
                               filter_class='LETKFGridSpec', effective_config=dict(ec)))
        for a in ('SPEC12', 'SPEC6'):
            for s in FILTER_SEEDS:
                v = base * scale[a] * (1.15 if worse else 1.0)
                rows.append(dict(arm=a, episode=e['episode'], corner=e['corner'],
                                 filter_seed=s, avg_nRMSE=v, zero_nRMSE=z,
                                 cap_avg_nRMSE=v, loc_radius=ARM_LOC[a],
                                 n_ensemble=600, background='jonswap',
                                 filter_class='LETKFGridSpec',
                                 effective_config=dict(ec, loc_radius=ARM_LOC[a])))
    for e in pc_episodes():
        for s in FILTER_SEEDS:
            m = [r for r in frozen if r['arm'] == INCUMBENT_ARM
                 and r['episode'] == e['episode'] and r['filter_seed'] == s][0]
            rows.append(dict(m, pc=True))
        for s in EST_SEEDS:
            m = [r for r in frozen if r['arm'] == SHIP_ARM
                 and r['episode'] == e['episode'] and r['est_seed'] == s][0]
            rows.append(dict(m, pc=True))
    return rows, frozen


def _perturbations():
    """게이트마다 **그것을 깨는 개입**. 발화하지 않으면 그 게이트는 무용하다."""
    out = []

    def mk(name, gate, fn):
        out.append((name, gate, fn))

    def p_err(rows, fr):
        rows = list(rows) + [dict(arm='SPEC12', episode='X', error='boom')]
        return rows, fr
    mk('행 하나에 error', 'G-11', p_err)

    def p_missing(rows, fr):
        return [r for r in rows if not (r['arm'] == 'SPEC12' and not r.get('pc'))][:-1], fr
    mk('SPEC12 행 삭제', 'G-12', p_missing)

    def p_loc(rows, fr):
        rows = [dict(r) for r in rows]
        for r in rows:
            if r['arm'] == 'SPEC12' and not r.get('pc'):
                r['loc_radius'] = 25.0
        return rows, fr
    mk('SPEC12 의 loc 되읽기가 25', 'G-16', p_loc)

    def p_ec(rows, fr):
        rows = [dict(r, effective_config=dict(r.get('effective_config') or {}))
                for r in rows]
        for r in rows:
            if r['arm'] == 'SPEC6' and not r.get('pc'):
                r['effective_config']['loc_radius'] = 25.0
        return rows, fr
    mk('SPEC6 의 readback loc 이 25', 'G-16', p_ec)

    def p_2factor(rows, fr):
        rows = [dict(r, effective_config=dict(r.get('effective_config') or {}))
                for r in rows]
        for r in rows:
            if r['arm'] == 'SPEC12' and not r.get('pc'):
                r['effective_config']['n_ensemble'] = 300
        return rows, fr
    mk('★두 번째 요인(앙상블)이 함께 움직였다', 'G-13', p_2factor)

    def p_zero(rows, fr):
        rows = [dict(r) for r in rows]
        for r in rows:
            if r['arm'] == 'SPEC6' and not r.get('pc'):
                r['zero_nRMSE'] = (r['zero_nRMSE'] or 0.25) * 1.001
        return rows, fr
    mk('★zero_nRMSE 가 팔에 따라 달라졌다', 'G-Z', p_zero)

    def p_pc(rows, fr):
        rows = [dict(r) for r in rows]
        for r in rows:
            if r.get('pc') and r['arm'] == INCUMBENT_ARM:
                r['avg_nRMSE'] = r['avg_nRMSE'] * (1 + 1e-12)
        return rows, fr
    mk('★SPEC25 재실행이 1e-12 만큼 다르다', 'G-PC1', p_pc)

    def p_pc2(rows, fr):
        rows = [dict(r) for r in rows]
        for r in rows:
            if r.get('pc') and r['arm'] == SHIP_ARM:
                r['avg_nRMSE'] = r['avg_nRMSE'] * (1 + 1e-12)
        return rows, fr
    mk('★SHIP 재실행이 1e-12 만큼 다르다', 'G-PC2', p_pc2)

    def p_noop(rows, fr):
        rows = [dict(r) for r in rows]
        fspec = {(r['episode'], r['filter_seed']): r['avg_nRMSE'] for r in fr
                 if r.get('arm') == INCUMBENT_ARM and r.get('filter_seed') is not None}
        for r in rows:
            k = (r.get('episode'), r.get('filter_seed'))
            if r['arm'] in NEW_ARMS and not r.get('pc') and k in fspec:
                r['avg_nRMSE'] = fspec[k]
        return rows, fr
    mk('★개입이 완전한 no-op (세 팔 비트 동일)', 'G-16b', p_noop)

    def p_shim(rows, fr):
        rows = [dict(r) for r in rows]
        for r in rows:
            if r['arm'] == 'SPEC12' and not r.get('pc') and 'cap_avg_nRMSE' in r:
                r['cap_avg_nRMSE'] = r['avg_nRMSE'] * 1.0000001
        return rows, fr
    mk('★계측 shim 이 값을 바꿨다', 'G-18', p_shim)
    return out


def _runfree_perturbations():
    """전역을 갈아 끼워 런 0회 게이트가 발화하는지."""
    out = []

    def p1():
        g = globals()
        old = g['FROZEN_LADDER']
        g['FROZEN_LADDER'] = (1.0, 2.0)
        return 'G-L1', lambda: g.__setitem__('FROZEN_LADDER', old)

    def p2():
        g = globals()
        old = dict(g['FROZEN_CANDIDATES'])
        g['FROZEN_CANDIDATES'] = {k: (99.0,) for k in old}
        return 'G-L2', lambda: g.__setitem__('FROZEN_CANDIDATES', old)

    def p3():
        g = globals()
        old = g['LOC_INCUMBENT']
        g['LOC_INCUMBENT'] = 50.0
        return 'G-L2', lambda: g.__setitem__('LOC_INCUMBENT', old)

    def p4():
        g = globals()
        old = dict(g['ARM_LOC'])
        g['ARM_LOC'] = dict(old, SPEC6=25.0)
        return 'G-L3', lambda: g.__setitem__('ARM_LOC', old)

    def p5():
        g = globals()
        old = g['S17_ANALYZER']
        g['S17_ANALYZER'] = os.path.join(S17D, 'PREREG.md')
        return 'G-L5', lambda: g.__setitem__('S17_ANALYZER', old)

    def p6():
        g = globals()
        old = g['NOMINAL'], g['CONFIRM'], g['WEAK']
        g['CONFIRM'], g['WEAK'] = 12, 3
        return 'G-19', lambda: (g.__setitem__('NOMINAL', old[0]),
                                g.__setitem__('CONFIRM', old[1]),
                                g.__setitem__('WEAK', old[2]))

    def p7():
        g = globals()
        old = g['CORNERS_M2']
        g['CORNERS_M2'] = CORNERS_M2 + ('SS-LOWH',)
        return 'G-1', lambda: g.__setitem__('CORNERS_M2', old)

    def p8():
        g = globals()
        old = g['MESH_GRID']
        g['MESH_GRID'] = (20, 15)
        return 'G-L3', lambda: g.__setitem__('MESH_GRID', old)
    return [p1, p2, p3, p4, p5, p6, p7, p8]


def _label_reachability():
    """★라벨 5종이 **전부 도달 가능**한가 (불리한 라벨만 조용히 막히는 결함 부류)."""
    got = {}
    base = dict(best='SPEC12', b26_label_frozen='HOLDS (CONFIRMATORY)',
                b26_label_new='HOLDS (CONFIRMATORY)', d2_best_better=False,
                d2_any_better=False, d2_any_worse=False)
    got[verdict_label(dict(base, best=None))] = 1
    got[verdict_label(dict(base, b26_label_new='UNDECIDED'))] = 1
    got[verdict_label(dict(base, d2_best_better=True, d2_any_better=True))] = 1
    got[verdict_label(dict(base, d2_any_worse=True))] = 1
    got[verdict_label(base)] = 1
    return set(got) == set(VERDICTS) and all(v in READING for v in VERDICTS)


def selftest(verbose=True):
    log, ok = [], True

    def say(s):
        log.append(s)
        if verbose:
            print(s, flush=True)

    rows, fr = _fixture()
    g1 = list(gates_runfree()) + list(gates_data()) + list(gates_run(rows, fr))
    n_ok = sum(1 for g in g1 if g[1])
    say(f'[1] 픽스처 전 게이트: {n_ok}/{len(g1)}')
    for g in g1:
        if not g[1]:
            say(f'    ⛔FAIL {g[0]}: {g[2]}')
    ok &= (n_ok == len(g1))

    empty = list(gates_run([], []))
    hollow = [g[0] for g in empty if g[1] and _gid(g[0]) not in ROWLESS_OK]
    say(f'[2] 공허한 통과 (0행): {len(hollow)}건 (허용 {ROWLESS_OK})')
    for h in hollow:
        say(f'    ⛔{h}')
    ok &= (not hollow)

    fired = 0
    for name, gate, fn in _perturbations():
        r2, f2 = fn(rows, fr)
        gg = list(gates_run(r2, f2))
        hit = any((not g[1]) and _gid(g[0]) == gate for g in gg)
        fired += hit
        if not hit:
            say(f'    ⛔변형이 발화하지 않았다: {name} → {gate}')
    say(f'[3] 변형 {fired}/{len(_perturbations())} 발화')
    ok &= (fired == len(_perturbations()))

    rf = _runfree_perturbations()
    fired2 = 0
    for p in rf:
        gate, undo = p()
        try:
            gg = list(gates_runfree())
            hit = any((not g[1]) and _gid(g[0]) == gate for g in gg)
            fired2 += hit
            if not hit:
                say(f'    ⛔런0회 변형이 발화하지 않았다 → {gate}')
        finally:
            undo()
    say(f'[4] 런 0회 변형 {fired2}/{len(rf)} 발화')
    ok &= (fired2 == len(rf))

    lr = _label_reachability()
    say(f'[5] 라벨 도달성 {len(VERDICTS)}종 + READING 키: {"PASS" if lr else "⛔FAIL"}')
    ok &= lr

    t = threshold_table()
    ta = t['confirm_within_family'] and t['weak_is_sibling']
    say(f'[6] 문턱 산술: {"PASS" if ta else "⛔FAIL"} '
        f'(CONFIRM {CONFIRM} p={t["p_confirm"]:.6f} ≤ {FAMILY_ALPHA})')
    ok &= ta

    # [7] 방향 불변식 — ★MAGNITUDE_FLOOR 가 실제로 크기를 못 키우는가
    s = {'SPEC25': dict(W=24, rel=-42.0), 'SPEC12': dict(W=23, rel=-55.0)}
    cited, _ = magnitude_floor(s, 'SPEC12')
    dir_ok = (cited == -42.0)
    say(f'[7] 방향 불변식 (MAGNITUDE_FLOOR): {"PASS" if dir_ok else "⛔FAIL"} '
        f'(BEST −55% 여도 인용은 {cited}%)')
    ok &= dir_ok

    say(f'\n{"✅자가진단 PASS" if ok else "⛔자가진단 FAIL"}')
    return ok, log


def main():
    if '--selftest' in sys.argv:
        sys.exit(0 if selftest()[0] else 1)
    gs = list(gates_runfree()) + list(gates_data())
    for n_, q_, d_ in gs:
        print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
    print(f'\n런 0회 게이트 {sum(1 for g in gs if g[1])}/{len(gs)}')
    print(f'\n{DIRECTION}')
    sys.exit(0 if all(g[1] for g in gs) else 1)


if __name__ == '__main__':
    main()
