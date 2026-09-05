#!/usr/bin/env python3
"""★동결 verdict 의 **MDE 상수**를 규약대로 재계산해 대조한다 (런 0회 · 산술만).

    python3 tmp/_audit/check_mde_constants.py            # 사람이 읽는 표
    python3 tmp/_audit/check_mde_constants.py --keys      # 래칫용 이름 집합(`KEY<TAB>…`)
    python3 tmp/_audit/check_mde_constants.py --selftest  # ★항진 검사

⛔**계산의 유일출처는 `tmp/_audit/sign_test_mde.py`** 다 — 이 파일은 산수를 하지 않는다.
규약(*"MDE 를 참값보다 낮게 보고하지 않는다"*)의 서술도 그 파일에 있다.

──────────────────────────────────────────────────────────────────────────────
★판정 기준은 격자 일치가 아니라 **실제 검정력**이다
──────────────────────────────────────────────────────────────────────────────
  `⛔미달`   : 인쇄값의 **정확 검정력 < 목표** ⟹ 「검정력 .80」이라 광고하는데 아니다.
               ***이것만이 결함이다*** — 설계가 실제보다 예민하다고 말하기 때문이다.
  `✅규약`   : 인쇄값 == `mde_q()` (4자리 격자 위 달성 가능 최솟값).
  `⚠보수적` : 검정력은 목표 이상인데 규약값과 다르다(자릿수가 다르거나 더 무른 값).
               ⛔**결함이 아니다** — 방향이 안전하다. 세기만 하고 실패시키지 않는다.

⚠**동결 배치는 고치지 않는다.** 그 수는 증거다 ⟹ `⛔미달` 은 `scripts/check_all.py` 의
`MDE_KNOWN` **이름집합 래칫**에 등록된다(선례 = `SEED_BAND_KNOWN`·`SHA256_KNOWN_FAIL`).
새 미달이 들어오거나 기지 미달이 사라지면 `check_all` 이 실패한다.

──────────────────────────────────────────────────────────────────────────────
★설계 인자(n · 문턱 · 다리 · 목표검정력)를 어디서 얻는가
──────────────────────────────────────────────────────────────────────────────
  (A) **자기기술** — verdict 자신이 `n` 과 문턱(`thresh`/`crit_w`/`k_plus`/`threshold`)을
      싣고 있으면 **그것을 읽는다**. ★손으로 옮기는 값이 0 이 되므로 이쪽이 항상 낫다.
      ⟹ 새 배치는 `mde` 블록에 `n`·`thresh`·`power` 를 넣어라. 그러면 등록이 필요 없다.
  (B) **등록(`ENTRIES`)** — 자기기술이 아닌 동결 배치. `crit` 은 반드시 `pin` 으로
      **배치 자신의 파일에서 되읽어** 대조한다(`src:` = 배치 소스의 상수, `json:` = 같은
      verdict 의 다른 필드). 되읽기가 불가능한 것만 `pin=None` + `src` 산문 출처를 적는다.
  (C) **미등록** — 위 둘 다 아닌데 MDE 처럼 보이는 수를 실은 배치. 배치 **이름**으로
      래칫한다. ⛔침묵시키지 않는다(★이 저장소 규율: *"동결한 적이 없다" 가 더 나쁘다*).
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from sign_test_mde import mde_q, mde_exact, power_at   # noqa: E402

# ⚠부동소수 이분법의 마지막 자리는 판정하지 않는다 — 우리가 겨누는 부족분은 >=1e-5 다.
POWER_TOL = 1e-9

# ── (A) 자기기술 인식 ───────────────────────────────────────────────────────
CRIT_FIELDS = ('thresh', 'crit_w', 'k_plus', 'threshold')
# ⚠**맨 `p\d\d` 는 받지 않는다** — 자가진단 중 `S13b` 의 `cv.crowded.p90`(CV 백분위)이
#   MDE 로 오탐됐다. 이 저장소에서 MDE 의 기호는 **`q`** 다.
Q_PAT = re.compile(r'^(?:mde_[qp]|q)(\d{2})$')      # q80 · mde_q80 · mde_p80

# ── (C) MDE 처럼 보이는 수 (미등록 탐지용, 넓게 잡는다) ─────────────────────
LOOKS_MDE = re.compile(r'(^|\.)mde|(^|\.)q\d{2}$', re.I)
# 설계 인자·부수 통계는 「인쇄된 MDE」가 아니다.
NOT_A_VALUE = {'n', 'power', 'alpha', 'side', 'grade', 'one_sided', 'unrejectable_w',
               'unrejectable_S', 'unrejectable_D',
               # ⚠`_rec` = `T17` 의 **레코드 접기** 밴드. 비-`rec` 쌍둥이가 이미 면제인데
               #   이 둘만 빠져 있었다 — 「N곳 전부의 N」 부류다(승수 정수이지 MDE 아님).
               'unrejectable_S_rec', 'unrejectable_D_rec',
               'p_one_tail', 'k_plus', 'thresh',
               'crit_w', 'threshold', 'note', 'mde_note', 'holm_first_alpha',
               'min_wins', 'p_at_min', 'p_at_min_minus_1', 'predicted_K', 'predicted'}


def E(batch, vfile, path, n, crit, power, side='upper', pin=None, src=''):
    return dict(batch=batch, vfile=vfile, path=path, n=n, crit=crit,
                power=power, side=side, pin=pin, src=src)


# ── (B) 등록 ────────────────────────────────────────────────────────────────
# ⚠`crit` 은 **되읽기로 확인된다**(pin). 되읽기가 없는 둘은 산문 출처를 적었다.
ENTRIES = [
    # ── T12: n=24 PRIMARY(17) · SEC Holm 첫 단계(19) · SECR n=28(19) · 층 n=12(10)
    E('T12_ISOLATE_2026-09-04', 'verdict_t12.json', 'mde.primary_q50', 24, 17, 0.50,
      pin='src:THRESH_PRIMARY'),
    E('T12_ISOLATE_2026-09-04', 'verdict_t12.json', 'mde.primary_q80', 24, 17, 0.80,
      pin='src:THRESH_PRIMARY'),
    E('T12_ISOLATE_2026-09-04', 'verdict_t12.json', 'mde.primary_q90', 24, 17, 0.90,
      pin='src:THRESH_PRIMARY'),
    E('T12_ISOLATE_2026-09-04', 'verdict_t12.json', 'mde.sec_q80', 24, 19, 0.80,
      pin='src:THRESH_SEC'),
    E('T12_ISOLATE_2026-09-04', 'verdict_t12.json', 'mde.secr_q80', 28, 19, 0.80,
      pin='src:THRESH_SECR'),
    E('T12_ISOLATE_2026-09-04', 'verdict_t12.json', 'mde.stratum_q80', 12, 10, 0.80,
      src='PREREG.md:281 「층(Hs 2층, n=12) 최소 정수 10/12」 — ⚠상수 핀 없음'),
    # ── T13: n=52(33) · 층 되읽기 T11-24(17) · FRESH-28(19)
    E('T13_DEPTHFIX_2026-09-04', 'verdict_t13.json', 'mde.q50', 52, 33, 0.50,
      pin='src:THRESH_WIN'),
    E('T13_DEPTHFIX_2026-09-04', 'verdict_t13.json', 'mde.q80', 52, 33, 0.80,
      pin='src:THRESH_WIN'),
    E('T13_DEPTHFIX_2026-09-04', 'verdict_t13.json', 'mde.q90', 52, 33, 0.90,
      pin='src:THRESH_WIN'),
    E('T13_DEPTHFIX_2026-09-04', 'verdict_t13.json', 'mde.q80_t11', 24, 17, 0.80,
      pin='src:THRESH_T11'),
    E('T13_DEPTHFIX_2026-09-04', 'verdict_t13.json', 'mde.q80_fresh', 28, 19, 0.80,
      pin='src:THRESH_FRESH'),
    # ── T14: n=28 PRIMARY Bonferroni α/2(20) · SEC 무보정(19)
    E('T14_AUTOSCALE_2026-09-04', 'verdict_t14.json', 'mde.primary_q50', 28, 20, 0.50,
      pin='src:THRESH_PRIMARY'),
    E('T14_AUTOSCALE_2026-09-04', 'verdict_t14.json', 'mde.primary_q80', 28, 20, 0.80,
      pin='src:THRESH_PRIMARY'),
    E('T14_AUTOSCALE_2026-09-04', 'verdict_t14.json', 'mde.primary_q90', 28, 20, 0.90,
      pin='src:THRESH_PRIMARY'),
    E('T14_AUTOSCALE_2026-09-04', 'verdict_t14.json', 'mde.sec_q80', 28, 19, 0.80,
      src='PREREG.md:192 「SEC(무보정, w>=19)」 — ⚠상수 핀 없음(THRESH_CTRL 은 다른 검정)'),
    # ── T15: ★**해침 다리**(기각 W<=23) ⟹ side='lower' · 셀 n=16 Holm α/8 (W<=2)
    E('T15_HOMEREG_2026-09-05', 'verdict_t15.json', 'mde.q50', 64, 23, 0.50,
      side='lower', pin='src:THRESH_HURT'),
    E('T15_HOMEREG_2026-09-05', 'verdict_t15.json', 'mde.q80', 64, 23, 0.80,
      side='lower', pin='src:THRESH_HURT'),
    E('T15_HOMEREG_2026-09-05', 'verdict_t15.json', 'mde.q90', 64, 23, 0.90,
      side='lower', pin='src:THRESH_HURT'),
    E('T15_HOMEREG_2026-09-05', 'verdict_t15.json', 'mde.cell_q80', 16, 2, 0.80,
      side='lower', src='PREREG.md:202 「셀(n=16, Holm α/8): 해침 임계 W<=2」'),
    # ── T16: 문턱은 **계산**되지만 verdict 의 `mde` 블록이 n·문턱을 싣지 않는다.
    #   ⟹ `REGISTERED` 딕셔너리(사전등록 핀)에서 되읽는다. n_eff_table 은 (A) 가 본다.
] + [
    E('T16_BANDFIX_2026-09-05', 'verdict_t16.json', f'mde.{tag}_q{int(pw * 100)}',
      n, crit, pw, pin=f'src:{pinname}')
    for tag, n, crit, pinname in (('S', 60, 39, 'THRESH_S'), ('D', 60, 40, 'THRESH_D'),
                                  ('F', 60, 37, 'THRESH_F'),
                                  ('SECE', 60, 37, 'THRESH_SECE'),
                                  ('BLIND', 24, 17, 'THRESH_BLIND'))
    for pw in (0.50, 0.80, 0.90)
] + [
    # ── 이전 아크 (같은 산수 · 같은 규약이 적용된다) ────────────────────────
    E('T4_GEOMETRY_2026-08-30', 'verdict_t4.json', 'mde.q80', 48, 32, 0.80,
      pin='json:thresholds.CONFIRM'),
    E('M10_GEOMETRY_2026-08-30', 'verdict_m10.json', 'thresholds.q80', 48, 32, 0.80,
      pin='json:thresholds.CONFIRM'),
    E('T7_SPECTRAL_2026-09-02', 'verdict_t7.json', 'thresholds.q80', 48, 32, 0.80,
      pin='json:thresholds.CONFIRM'),
    # ⛔★이 한 줄이 이 감사의 최대 소득이다 — `analyze_t7b.py:38` 이 `q80` 만 **리터럴**로
    #   적었고(같은 줄의 나머지 넷은 전부 `G.` 에서 계산된다) 그 리터럴이 **n=48 배치의 값**이다.
    E('T7b_SPECTRAL_2026-09-02', 'verdict_t7b.json', 'thresholds.q80', 32, 23, 0.80,
      pin='json:thresholds.CONFIRM'),
    # ⚠`T8`·`T9` 는 귀무가 `p0=0.75` 다(부호검정이 아니다). ★그래도 `mde_q()` 는 그대로
    #   쓴다 — 문턱을 **받아서** 풀기 때문에 귀무는 문턱 유도에만 들어간다(파일에서 되읽는다).
    E('T8_ENSDEFAULT_2026-09-02', 'verdict_t8.json', 'thresholds.mde_p80', 64, 56, 0.80,
      pin='json:thresholds.K_MIN'),
    E('T9_ENSHARM_2026-09-02', 'verdict_t9.json', 'thresholds.mde_p80', 64, 56, 0.80,
      pin='json:thresholds.K_MIN'),
    E('T9_ENSHARM_2026-09-02', 'verdict_t9_BLOCKED_2PID.json', 'thresholds.mde_p80',
      64, 56, 0.80, pin='json:thresholds.K_MIN'),
    # ⚠같은 양이 한 파일에 **두 번** 인쇄된 곳 — 둘 다 본다(한쪽만 고치는 부류 방지).
    E('T4_GEOMETRY_2026-08-30', 'verdict_t4.json', 'thresholds.q80', 48, 32, 0.80,
      pin='json:thresholds.CONFIRM'),
    E('T16_BANDFIX_2026-09-05', 'verdict_t16.json',
      'primary_family.D (착지 축, 양측, Bonferroni α/2).q80', 60, 40, 0.80,
      pin='src:THRESH_D'),
    E('T16_BANDFIX_2026-09-05', 'verdict_t16.json',
      'SEC.SEC-BLIND (미접촉 24 · ⛔angle 과 완전 교락).q80', 24, 17, 0.80,
      pin='src:THRESH_BLIND'),
] + [
    # ── T17 (⛔`BLOCKED` = 라벨 없음. ★그래도 MDE 는 검사한다 — 차단된 배치의 설계
    #   상수도 다음 사전등록이 인용하는 수이고, 인용되면 같은 규약을 져야 한다.)
    #   ★**이 저장소에서 18/18 전건 «규약» 이 나온 첫 배치다**(부족분 0.00e+00).
    #   n 출처 = `gates_t17.py`: 셀 72 · 레코드 `N_REC=12` · 일자 `N_DATE=10`(`:136`).
    #   ⚠일자만 n 이 다르다 — 12 로 넘겨 잡으면 세 값이 «보수적» 로 **오분류**된다.
    E('T17_REALSCALE_2026-09-05', 'verdict_t17.json', f'mde.{tag}_q{int(pw * 100)}',
      n, crit, pw, pin=f'json:thresholds.{tag}')
    for tag, n, crit in (('S', 72, 45), ('D', 72, 46), ('F', 72, 44),
                         ('REC_S', 12, 10), ('REC_D', 12, 11), ('DATE', 10, 9))
    for pw in (0.50, 0.80, 0.90)
] + [
    # ── T18 (⛔`BLOCKED` = 라벨 없음. 차단은 `G-ATOMSRC` 이고 MDE 와 무관하다.)
    #   n 출처 = `gates_t18.py`: 축 I·SEC 는 `N_EP=48`, `SEC-C` 는 `SECC_N_EP=16`.
    #   ⚠`T18` 은 `round(mde_q(...), 4)` 로 **가까운 쪽** 반올림한다 ⟹ 넷이 규약보다
    #   1e-4 수준으로 낮다(⛔라벨은 정수 문턱이 만들므로 **판정 불변**). 동결이라 고치지
    #   않고 `MDE_KNOWN` 에 등록한다.
    E('T18_FACTORIAL_2026-09-05', 'verdict_t18.json', f'mde.{tag}_q{int(pw * 100)}',
      n, crit, pw, pin=f'json:thresholds.{pin}')
    for tag, n, crit, pin in (('I', 48, 32, 'I'), ('SEC', 48, 31, 'SEC'),
                              ('SECC', 16, 13, 'SECC'))
    for pw in (0.50, 0.80, 0.90)
    if not (tag == 'SECC' and pw != 0.80)
] + [
    # ⚠같은 양을 **두 번째로** 인쇄하는 곳(`T4`·`T16`·`T17` 과 같은 부류) — 둘 다 본다.
    E('T18_FACTORIAL_2026-09-05', 'verdict_t18.json', 'primary.q80', 48, 32, 0.80,
      pin='json:thresholds.I'),
    E('T18_FACTORIAL_2026-09-05', 'verdict_t18.json',
      'SEC.SEC-A (사전-OFF 열의 양자화 효과, 무보정 단측).q80', 48, 31, 0.80,
      pin='json:thresholds.SEC'),
    E('T18_FACTORIAL_2026-09-05', 'verdict_t18.json',
      'SEC.SEC-B (양자화 열의 사전 효과, 무보정 단측).q80', 48, 31, 0.80,
      pin='json:thresholds.SEC'),
    E('T18_FACTORIAL_2026-09-05', 'verdict_t18.json',
      'SEC.SEC-C (신선 시드 45·46·47 · 앞 16, 무보정).q80', 16, 13, 0.80,
      pin='json:thresholds.SECC'),
] + [
    # ⚠`n_eff_table` 은 **`n` 이 dict 키**라 (A) 자기기술 탐지가 못 본다(`T16` 은 필드였다).
    #   ⟹ 명시 등록. `k` 는 같은 블록에서 되읽는다.
    E('T18_FACTORIAL_2026-09-05', 'verdict_t18.json', f'n_eff_table.{n}.q80',
      n, k, 0.80, pin=f'json:n_eff_table.{n}.k')
    #   ⛔★내가 여기 `k` 를 **손으로 적었다가 두 개 틀렸다**(36→24·24→17). `pin` 되읽기가
    #   즉시 잡았다 — 이 검사기가 존재하는 이유가 정확히 그 부류다. 아래는 verdict 실측값이다.
    for n, k in ((48, 32), (44, 29), (40, 27), (36, 25), (32, 23), (28, 20),
                 (24, 18), (20, 15), (16, 13), (12, 10), (8, 8), (6, 6))
] + [
    # ⚠가족 비용표 — Holm 보정 후 꼬리 α 라 `k` 가 다르다(블록에서 되읽는다).
    E('T18_FACTORIAL_2026-09-05', 'verdict_t18.json',
      f'prediction.family_cost.{fam}.q80', n, k, 0.80,
      pin=f'json:prediction.family_cost.{fam}.k')
    for fam, n, k in (('2x2_family1', 48, 32), ('2x2_family3', 48, 33),
                      ('2x2x2_family7', 48, 34))
] + [
    # ⚠`T17` 도 같은 양을 **두 번** 인쇄한다(`mde` 블록 + `primary_family` 블록).
    #   ★둘 다 본다 — 한쪽만 고치면 통과하는 구멍을 만들지 않는다(`T4`·`T16` 선례).
    E('T17_REALSCALE_2026-09-05', 'verdict_t17.json',
      'primary_family.S (기술 축, 단측, Bonferroni α/2).q80', 72, 45, 0.80,
      pin='json:thresholds.S'),
    E('T17_REALSCALE_2026-09-05', 'verdict_t17.json',
      'primary_family.S (기술 축, 단측, Bonferroni α/2).병기_레코드접기.q80',
      12, 10, 0.80, pin='json:thresholds.REC_S'),
    E('T17_REALSCALE_2026-09-05', 'verdict_t17.json',
      'primary_family.D (착지 축, 양측, Bonferroni α/2).q80', 72, 46, 0.80,
      pin='json:thresholds.D'),
    E('T17_REALSCALE_2026-09-05', 'verdict_t17.json',
      'primary_family.D (착지 축, 양측, Bonferroni α/2).병기_레코드접기.q80',
      12, 11, 0.80, pin='json:thresholds.REC_D'),
]

# ── (C) 미등록 — ⛔침묵이 아니라 **공시**다 ─────────────────────────────────
#   이름집합 래칫 자체는 `scripts/check_all.py` 의 `MDE_KNOWN` 이 소유한다(래칫은 한 곳).
#   여기는 *"왜 재도출할 수 없는가"* 만 적는다 — 보고서에 그 사유가 같이 인쇄된다.
UNREGISTERED_WHY = {
    'M2_LOCTUNE_2026-08-28':
        '`mde.{f_below_*, pct_to_weak}` = 관측 효과크기의 **분위**이지 승률 MDE 가 아니다',
    'R7_FLOOR_2026-09-03':
        '`mde` = 0.714285… = **5/7 레코드 비율**. 이항 검정력이 아니다',
}


# ── 값 읽기 ─────────────────────────────────────────────────────────────────
def _dig(obj, path):
    for seg in path.split('.'):
        if isinstance(obj, dict) and seg in obj:
            obj = obj[seg]
        else:
            return None
    return obj


def _pin_from_src(batch, name):
    """배치 자신의 `.py` 에서 `NAME = <int>` 또는 `'NAME': <int>` 를 되읽는다."""
    got = set()
    for f in sorted(glob.glob(os.path.join(ROOT, 'tmp', batch, '*.py'))):
        try:
            txt = open(f, encoding='utf-8').read()
        except OSError:
            continue
        for m in re.finditer(rf'^\s*{re.escape(name)}\s*=\s*(\d+)\s*(?:#.*)?$',
                             txt, re.M):
            got.add(int(m.group(1)))
        for m in re.finditer(rf'''['"]{re.escape(name)}['"]\s*:\s*(\d+)''', txt):
            got.add(int(m.group(1)))
    return got


def _resolve_pin(e, vd):
    """`(값 또는 None, 설명)`. ⛔해결 실패는 조용한 기본값이 아니라 **결함**이다."""
    pin = e['pin']
    if not pin:
        return None, e['src'] or '⚠핀 없음'
    kind, _, name = pin.partition(':')
    if kind == 'json':
        v = _dig(vd, name)
        return (int(v) if isinstance(v, (int, float)) else None), f'verdict.{name}'
    got = _pin_from_src(e['batch'], name)
    if len(got) == 1:
        return got.pop(), f'{name} (배치 소스)'
    if not got:
        return None, f'⛔{name} 되읽기 실패'
    return None, f'⛔{name} 이 소스에서 {sorted(got)} 로 갈린다'


# ── 대조 ────────────────────────────────────────────────────────────────────
def _judge(n, crit, side, power, printed):
    """`(라벨, 규약값, 실제 검정력, 부족분)`."""
    safe = mde_q(n, crit, power, side=side)
    got = power_at(printed, n, crit, side)
    if got < power - POWER_TOL:
        return '⛔미달', safe, got, power - got
    if safe is not None and abs(printed - safe) < 1e-12:
        return '✅규약', safe, got, 0.0
    return '⚠보수적', safe, got, 0.0


def _auto_rows(vd, vfile, batch, skip):
    """(A) 자기기술 블록 — `n`·문턱·`qNN` 을 **파일에서** 읽는다."""
    rows = []

    def walk(o, path, parent_key):
        if not isinstance(o, dict):
            if isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, f'{path}[{i}]', parent_key)
            return
        crit = next((o[c] for c in CRIT_FIELDS if isinstance(o.get(c), int)), None)
        n = o.get('n')
        if n is None and re.fullmatch(r'\d+', parent_key or ''):
            n = int(parent_key)          # `n_eff_table` 처럼 부모 키가 n 인 표
        if isinstance(n, int) and isinstance(crit, int):
            side = o.get('side', 'upper')
            for k, v in o.items():
                m = Q_PAT.match(k)
                if not m or not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                p = f'{path}.{k}'.lstrip('.')
                if p in skip:
                    continue
                rows.append(dict(batch=batch, vfile=vfile, path=p, n=n, crit=crit,
                                 power=int(m.group(1)) / 100.0, side=side,
                                 printed=float(v), how='자기기술',
                                 critsrc='verdict(같은 블록)'))
        for k, v in o.items():
            walk(v, f'{path}.{k}'.lstrip('.'), k)

    walk(vd, '', None)
    return rows


def collect():
    """`(rows, problems)` — `problems` 는 래칫 키 목록."""
    rows, problems = [], []
    by_file = {}
    for e in ENTRIES:
        by_file.setdefault((e['batch'], e['vfile']), []).append(e)

    covered_batches = set()
    for (batch, vfile), es in sorted(by_file.items()):
        p = os.path.join(ROOT, 'tmp', batch, vfile)
        if not os.path.exists(p):
            problems.append(f'MISSING:{batch}/{vfile}')
            continue
        vd = json.load(open(p, encoding='utf-8'))
        covered_batches.add(batch)
        for e in es:
            printed = _dig(vd, e['path'])
            if not isinstance(printed, (int, float)):
                problems.append(f"MISSING:{batch}:{e['path']}")
                continue
            pinv, why = _resolve_pin(e, vd)
            if pinv is not None and pinv != e['crit']:
                problems.append(f"PIN:{batch}:{e['path']}")
                why = f'⛔핀 불일치 등록 {e["crit"]} vs {pinv} ({why})'
            elif e['pin'] and pinv is None:
                problems.append(f"PIN:{batch}:{e['path']}")
            rows.append(dict(batch=batch, vfile=vfile, path=e['path'], n=e['n'],
                             crit=e['crit'], power=e['power'], side=e['side'],
                             printed=float(printed), how='등록', critsrc=why))

    # (A) + (C) — 모든 동결 verdict 를 훑는다.
    reg_paths = {(e['batch'], e['vfile'], e['path']) for e in ENTRIES}
    for p in sorted(glob.glob(os.path.join(ROOT, 'tmp', '*', 'verdict*.json'))):
        batch = os.path.basename(os.path.dirname(p))
        vfile = os.path.basename(p)
        try:
            vd = json.load(open(p, encoding='utf-8'))
        except (OSError, ValueError):
            continue
        skip = {q for (b, f, q) in reg_paths if b == batch and f == vfile}
        auto = _auto_rows(vd, vfile, batch, skip)
        rows.extend(auto)
        if auto:
            covered_batches.add(batch)
        # (C) 미등록 — MDE 처럼 보이는 **수**가 남아 있는가
        seen = skip | {r['path'] for r in auto}
        left = []

        def walk(o, path):
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f'{path}.{k}'.lstrip('.'))
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, f'{path}[{i}]')
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                # ⚠`a.b[0]` 의 잎은 `b` 다 — 첨자를 떼지 않으면 설계 인자 목록이 새는다.
                leaf = re.sub(r'\[\d+\]$', '', path.split('.')[-1])
                if leaf in NOT_A_VALUE or path in seen:
                    return
                if LOOKS_MDE.search(path):
                    left.append(path)

        walk(vd, '')
        # ⛔`covered_batches` 로 면제하지 않는다 — 자기기술 블록 **하나**가 같은 배치의
        #   손타이핑 MDE 를 가려 주면 안 된다(이 저장소의 「한 곳만 고친다」 부류).
        if left:
            problems.append(f'UNREGISTERED:{batch}')

    for r in rows:
        lab, safe, got, short = _judge(r['n'], r['crit'], r['side'], r['power'],
                                       r['printed'])
        r.update(label=lab, safe=safe, got=got, short=short,
                 exact=mde_exact(r['n'], r['crit'], r['power'], r['side']))
        if lab == '⛔미달':
            problems.append(f"{r['batch']}:{r['path']}")
    return rows, sorted(set(problems))


ARROW = {'upper': '↑', 'lower': '↓'}


def report(rows, problems):
    print('── MDE 상수 정합 (부호검정 · 런 0회) ' + '─' * 36)
    print(f'   규약 = {mde_q.__module__}.mde_q — '
          '4자리 격자 위에서 **실제로** 목표 검정력을 달성하는 값 (⛔낮게 보고하지 않는다)')
    cur = None
    for r in sorted(rows, key=lambda r: (r['batch'], r['path'])):
        if r['batch'] != cur:
            cur = r['batch']
            print(f'\n[{cur}]')
        safe = '—' if r['safe'] is None else f"{r['safe']:.4f}"
        extra = f"  부족 {r['short']:.2e}" if r['label'] == '⛔미달' else ''
        print(f"  {r['label']} {r['path']:24s} n={r['n']:>3} k={r['crit']:>3} "
              f"{ARROW[r['side']]}{r['power']:.2f}  인쇄 {r['printed']:<8.6g} "
              f"규약 {safe}  연속 {r['exact']:.6f}  검정력 {r['got']:.8f}{extra}"
              f"   [{r['how']}]")
    n_bad = sum(1 for r in rows if r['label'] == '⛔미달')
    n_cons = sum(1 for r in rows if r['label'] == '⚠보수적')
    print('\n' + '─' * 72)
    print(f'   대조 {len(rows)}건 — ✅규약 {len(rows) - n_bad - n_cons} · '
          f'⚠보수적 {n_cons}(안전) · ⛔미달 {n_bad}')
    other = [p for p in problems if p.split(':')[0] in
             ('UNREGISTERED', 'MISSING', 'PIN')]
    for p in other:
        why = UNREGISTERED_WHY.get(p.split(':', 1)[1], '')
        print(f'   ⛔{p}' + (f'  — {why}' if why else ''))
    if n_bad:
        print('   ⛔**동결 배치를 고치지 마라** — 그 수는 증거다. '
              '`scripts/check_all.py` 의 `MDE_KNOWN` 이 이름으로 래칫한다.')
    return 0 if not problems else 1


# ── ★항진 검사 ──────────────────────────────────────────────────────────────
def _selftest():
    """*"이 검사기가 무엇이든 통과시키지는 않는가"* — 규칙이 규칙인지 본다."""
    bad = []

    def chk(c, m):
        if not c:
            bad.append(m)

    # 1. 실제 결함을 잡는가 (`T14 primary_q80` = 0.7630, 검정력 0.79986).
    lab, safe, got, short = _judge(28, 20, 'upper', 0.80, 0.7630)
    chk(lab == '⛔미달' and safe == 0.7631, f'T14 재현 실패: {lab} {safe}')
    chk(abs(short - (0.80 - got)) < 1e-15, '부족분 계산이 어긋난다')
    # 2. 규약값 자신은 통과하는가.
    chk(_judge(28, 20, 'upper', 0.80, 0.7631)[0] == '✅규약', '규약값이 통과하지 않는다')
    # 3. 보수적인 값을 결함으로 오판하지 않는가 (`T11 q80` = 0.760).
    chk(_judge(24, 17, 'upper', 0.80, 0.760)[0] == '⚠보수적', 'T11 q80 오판')
    # 4. ★**해침 다리에서 방향이 뒤집히는가** — 위쪽으로 어긋난 값이 미달이어야 한다.
    chk(_judge(64, 23, 'lower', 0.90, 0.2952)[0] == '⛔미달', 'T15 q90 하측 오판')
    chk(_judge(64, 23, 'lower', 0.50, 0.3617)[0] == '⚠보수적', 'T15 q50 하측 오판')
    # 5. ⛔**항진 방지** — 아무 수나 통과하지 않는다.
    chk(_judge(24, 17, 'upper', 0.80, 0.60)[0] == '⛔미달', '터무니없는 값이 통과했다')
    chk(_judge(32, 23, 'upper', 0.80, 0.7109)[0] == '⛔미달', 'T7b 오판')
    # 6. 되읽기(pin)가 **실제로** 배치 소스를 읽는가.
    chk(_pin_from_src('T13_DEPTHFIX_2026-09-04', 'THRESH_WIN') == {33},
        'THRESH_WIN 되읽기 실패')
    chk(_pin_from_src('T13_DEPTHFIX_2026-09-04', 'THRESH_NOSUCH') == set(),
        '없는 상수가 되읽혔다')
    # 7. 자기기술 인식이 **실제로** 행을 만드는가(T11 = n·crit_w 를 싣는다).
    p = os.path.join(ROOT, 'tmp', 'T11_EXTACC_2026-09-03', 'verdict_t11.json')
    rows = _auto_rows(json.load(open(p, encoding='utf-8')), 'v', 'T11', set())
    chk(len(rows) == 3 and all(r['n'] == 24 and r['crit'] == 17 for r in rows),
        f'자기기술 인식 실패: {rows}')
    if bad:
        print('⛔항진 검사 실패')
        for m in bad:
            print('   ' + m)
        return 1
    print('✅check_mde_constants 항진 검사 통과 '
          '(결함검출·규약·보수적·하측방향·항진·핀되읽기·자기기술)')
    return 0


def main():
    ap = argparse.ArgumentParser()
    # ⚠줄 단위다 — `T16` 의 verdict 키에는 **공백과 한글 산문**이 들어 있어
    #   공백 구분 프로토콜이 키를 조각냈다(실측 후 고쳤다).
    ap.add_argument('--keys', action='store_true',
                    help='래칫용 이름집합 — `KEY<TAB>이름` 한 줄에 하나')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    rows, problems = collect()
    if a.keys:
        for k in problems:
            print('KEY\t' + k)
        return 0
    return report(rows, problems)


if __name__ == '__main__':
    sys.exit(main())
