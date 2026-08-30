"""T-2(STAT) 단일출처 — ★**제안 표현에 「재적합」을 이식한다: 실측 예보 지평의 정상성 축**

⛔**이 파일이 판정의 유일한 권위다.** `PREREG.md` 는 이 파일의 산문 번역이고
두 곳이 어긋나면 **코드가 이긴다**. 문턱·팔·표본·라벨·해석 규칙·게이트 전부 여기 있다.
⛔**결과 0행에서 동결한다.**

## ★이름 충돌 공시 (결과 전)
저장소에 `T2` 라는 이름이 **이미 둘** 있다 — `tmp/_audit/recheck_t2_2026-08-27.py`(논문 수치
재집계기, 축 A/B)와 `M7_SUPERVISION_2026-08-28/verdict_m7_e1.json` 의 `T-2_data` 키(`DATA` 축
읽기 등급). ⟹ **이 배치의 정식 이름은 `T2-STAT` 이고 라벨 접두어도 그렇게 쓴다.**
⛔ *"T-2"* 단독으로 인용하지 마라.

## 0. 질문 (한 문장)

***"배포 실측 경로가 예보 지평에서 갖는 유일하게 측정된 기술 담지자(= 주기적 재적합)를
제안 표현에 그대로 이식하면, 제안 표현도 실측에서 기술을 얻는가?"***

## 1. ★왜 이 질문이 지금까지 물어진 적이 없는가 — 런 0회로 확인된 것

`premise_t2.py` 가 동결 결과 파일에서만 재도출한다(⛔손복사 상수 0건). 요지:

| 팔 (동결 자료) | 참 제로필드 대비, `fc_mean`(lead 2–14) |
|---|---|
| `SHIP` 동결 (`B-3` `FROZEN`) | **16/36** · 비 1.033 ⟹ ⛔**무기술** |
| `PROP` 동결 (`B-7b` `PROP-B7B`) | **16/36** · 비 1.099 ⟹ ⛔**무기술** |
| `SHIP` 재적합 2 s (`B-3` `REFIT2`) | ★**32/36** · 비 0.793 ⟹ ✅**기술** |

⟹ ★★***예보 지평에서는 배포 경로도 동결하면 제안 표현과 똑같이 무기술이다(16/36 == 16/36).
실측 예보 기술의 유일한 측정된 담지자는 표현이 아니라 재적합이다.*** 그리고 그 담지자는
제안 표현에 **한 번도 이식된 적이 없다** — 저장소 자신이 그렇게 적어 두었다
(`M8_LADDER_2026-08-28/DIFF_INVENTORY.md` §3, D8 행 축자:
*"PROP 쪽 재적합(re-init 롤링)은 미측정"*).

⚠**범위**: 위 등가는 **예보 지평 한정**이다. **나우캐스트(lead 0)에서는 두 팔이 갈린다**
(`SHIP` 33/36·비 0.73 vs `PROP` 17/36·비 1.04) — 그 갈림은 이 배치가 **설명하지 않는다**.

## 2. ★비대칭 (결과 전 명시 — 이 배치를 정당화하는 것이 이것이다)

* **양성이면**: 소거표 **최초의 양성 행**. 척추가 *배제* 에서 ***위치지음*** 으로 바뀐다 —
  *"경계는 표현이 아니라 **적합의 시간축 갱신**에 있다"*. 기여 ②c(실시간 순차 추적)가
  `bank_mmse` 만의 성질이 아니라 **제안 표현에도 옮겨 붙는다**.
* **음성이면**: 소거표 한 줄, 판정 불변. ★**단 그 한 줄은 앞의 아홉 줄과 다르다** —
  *"배포 경로의 유일한 실측 예보 기술 담지자를 제안 표현에 그대로 심었는데 아무 일도
  없었다"* 이고, ⟹ ***"우리 팔에 공정한 기회를 준 적이 없다" 는 마지막 변호가 닫힌다.***

⛔**어느 분기에서도 이 배치는 배포 변경을 제안하지 않는다**(§7 금지 목록).
"""
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B2 = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
B3 = os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23')
B7 = os.path.join(ROOT, 'tmp', 'B7_PROP_TRACK_2026-08-23')
for _p in (HERE, ROOT, B2, B3, B7):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ★표본은 `gates_b3` 가 단일출처다(⛔레코드 목록 손복사 금지).
from gates_b3 import RECORDS, WINDOWS, SEED, NOISE, N_PAIRS  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# 1. 팔 — ⛔빈 dict 금지(`B-7` 교훈). 모든 키를 **명시**한다.
# ─────────────────────────────────────────────────────────────────────────────
# `EST` = 추정기 config 위에 얹는 것.  `HARN` = **하네스만** 읽는 키(패키지에 안 넘긴다;
# ⛔"설정을 넣었다 != 설정이 먹었다" 부류를 구조적으로 막는다).
_BANK = dict(real_domain_gate=True, init_method='auto', real_estimator='bank_mmse',
             real_bank_ensemble_k=3, bank_refit_async=False)
_PROP = dict(real_domain_gate=False, init_method='learned', bank_refit_async=False)

ARMS = {
    'SHIP-FROZEN': dict(EST=dict(_BANK, bank_refit_every_s=None), HARN=dict(prop_refit_s=None)),
    'SHIP-REFIT2': dict(EST=dict(_BANK, bank_refit_every_s=2.0), HARN=dict(prop_refit_s=None)),
    'PROP-FROZEN': dict(EST=dict(_PROP, bank_refit_every_s=None), HARN=dict(prop_refit_s=None)),
    'PROP-REFIT2': dict(EST=dict(_PROP, bank_refit_every_s=None), HARN=dict(prop_refit_s=2.0)),
}
ARM_ORDER = ['SHIP-FROZEN', 'SHIP-REFIT2', 'PROP-FROZEN', 'PROP-REFIT2']
BANK_ARMS = ['SHIP-FROZEN', 'SHIP-REFIT2']
PROP_ARMS = ['PROP-FROZEN', 'PROP-REFIT2']
FROZEN_ARMS = ['SHIP-FROZEN', 'PROP-FROZEN']
REFIT_ARMS = ['SHIP-REFIT2', 'PROP-REFIT2']
PAIRS_REFIT = [('SHIP-REFIT2', 'SHIP-FROZEN'), ('PROP-REFIT2', 'PROP-FROZEN')]

# ★`PROP-REFIT2` 의 개입 정의 (하네스가 소유, 러너가 이 상수만 읽는다):
#   `t_now - t_last_init >= PROP_REFIT_S` 이면 **롤링 창으로 `initialize()` 를 다시 부른다**.
#   창 규약은 bank 의 `_bank_update`/`_bank_fit_window` 와 **글자 그대로 같다**:
PROP_REFIT_S = 2.0          # == bank `bank_refit_every_s`
MEMORY_S = 40.0             # == `bank_memory_s` 기본값
OBS_CAP = 600               # == `bank_max_obs` 기본값, 같은 `default_rng(0)` 추출
# ⟹ 두 팔이 **같은 시각에 같은 관측 집합**으로 재적합한다. `G-MIRROR` 가 해시로 강제한다.

# ★재적합 횟수는 프로토콜에서 **도출**한다(⛔"7" 을 손으로 적지 않는다).
LEADS_OBS = [float(i) for i in range(1, 15)]     # 관측이 들어오는 lead (lead 0 은 없음)


def expected_refits(step=PROP_REFIT_S, leads=None):
    """`t - last >= step` 규칙이 관측 프레임 위에서 발화하는 횟수."""
    leads = LEADS_OBS if leads is None else leads
    last, n = 0.0, 0
    for t in leads:
        if t - last >= step:
            n += 1
            last = t
    return n


N_REFIT_EXPECTED = expected_refits()             # = 7 (B-3 동결 행이 이 값을 갖는다)

# ─────────────────────────────────────────────────────────────────────────────
# 2. 채점 — ⛔`ZERO_REF=1.0` 금지. 기준은 **레코드·창·lead 별 참 제로필드**.
# ─────────────────────────────────────────────────────────────────────────────
FIELD = 'fc_mean'
LEAD_FC = (2.0, 14.0)        # PRIMARY/SECONDARY-A 채점 구간 (⛔결과 전 확정)
LEAD_NULLCTRL = (0.0, 2.0)   # 음성대조: 이 구간은 두 팔이 **비트 동일**이어야 한다
LEAD_NOW = (0.0, 1.0)        # 서술 전용(⛔판정 아님) — 재적합이 구조적으로 무관한 구간


def zero_perlead(record, win):
    from zero_ref import zero_ref_perlead
    return zero_ref_perlead(record, win=win, noise=NOISE)


def _win_mean(d, lo, hi):
    v = [x for k, x in d.items() if lo <= float(k) <= hi and x is not None]
    return sum(v) / len(v) if v else None


def fc(row, lo=None, hi=None):
    lo, hi = (LEAD_FC if lo is None else (lo, hi))[0], (LEAD_FC[1] if hi is None else hi)
    return _win_mean(row.get('per_lead') or {}, lo, hi)


def fc_zero(record, win, lo=None, hi=None):
    lo, hi = (LEAD_FC if lo is None else (lo, hi))[0], (LEAD_FC[1] if hi is None else hi)
    return _win_mean(zero_perlead(record, win), lo, hi)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 문턱 — `math.comb` 이 직접 낸다. ⛔손복사 0건.
# ─────────────────────────────────────────────────────────────────────────────
ALPHA = 0.05
N_FAMILY = 2                                   # PRIMARY · SECONDARY-A (확증 주장 둘)
FAMILY_ALPHA = ALPHA / N_FAMILY                # 0.025


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def thresholds(n=N_PAIRS):
    nominal = next(w for w in range(n + 1) if sign_test_one_sided(w, n) <= ALPHA)
    confirm = next(w for w in range(n + 1) if sign_test_one_sided(w, n) <= FAMILY_ALPHA)
    return dict(n=n, NOMINAL=nominal, CONFIRM=confirm, NOT_EARN=n - nominal,
                p_nominal=sign_test_one_sided(nominal, n),
                p_confirm=sign_test_one_sided(confirm, n),
                alpha=ALPHA, family_alpha=FAMILY_ALPHA)


def mde(n=N_PAIRS, power=0.80, thresh=None):
    """★문턱에서 `power` 를 내는 최소 참 승률. ⛔판정문에 **반드시 병기**한다."""
    t = thresholds(n)['CONFIRM'] if thresh is None else thresh
    q = 0.5
    while sum(comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(t, n + 1)) < power:
        q += 0.001
    return dict(n=n, thresh=t, power=power, q80=round(q, 3))


# ─────────────────────────────────────────────────────────────────────────────
# 4. 집계 · 판정
# ─────────────────────────────────────────────────────────────────────────────
def key(r):
    return (r['record'], r['win'])


def _map(rows, arm, field=FIELD):
    return {key(r): r[field] for r in rows
            if r.get('arm') == arm and 'error' not in r and r.get(field) is not None}


def paired(rows, arm, ref, field=FIELD):
    import statistics as st
    a, b = _map(rows, arm, field), _map(rows, ref, field)
    ks = sorted(set(a) & set(b))
    n = len(ks)
    w = sum(1 for k in ks if a[k] < b[k])
    return dict(arm=arm, ref=ref, n=n, better=w,
                ties=sum(1 for k in ks if a[k] == b[k]),
                rel_median_pct=(st.median([(a[k] - b[k]) / b[k] * 100 for k in ks])
                                if ks else None),
                median_arm=st.median([a[k] for k in ks]) if ks else None,
                median_ref=st.median([b[k] for k in ks]) if ks else None,
                p_one_sided=sign_test_one_sided(w, n) if n else None)


def vs_zero(rows, arm, lo=None, hi=None):
    """★참 제로필드 대비. ⛔상수 1.0 을 쓰지 않는다 — `zero_ref` 가 낸다."""
    import statistics as st
    ks, rel, w = [], [], 0
    for r in rows:
        if r.get('arm') != arm or 'error' in r:
            continue
        v = fc(r, lo, hi)
        z = fc_zero(r['record'], r['win'], lo, hi)
        if v is None or z in (None, 0):
            continue
        ks.append(key(r))
        rel.append(v / z)
        w += (v < z)
    n = len(ks)
    return dict(arm=arm, n=n, better=w, ratio_median=st.median(rel) if rel else None,
                p_one_sided=sign_test_one_sided(w, n) if n else None)


def interaction(rows):
    """SECONDARY-B — ⚠**EXPLORATORY**. 가족에 넣지 않는다(⛔확증 금지).

    셀마다 로그 이득 `g = log(fc_REFIT / fc_FROZEN)` 을 팔 종류별로 내고,
    `g_prop < g_ship` (= 제안 쪽이 재적합에서 **더** 번다) 인 셀 수를 센다.
    """
    import math
    import statistics as st
    m = {a: _map(rows, a) for a in ARM_ORDER}
    ks = sorted(set.intersection(*[set(m[a]) for a in ARM_ORDER])) if all(m.values()) else []
    gp = {k: math.log(m['PROP-REFIT2'][k] / m['PROP-FROZEN'][k]) for k in ks}
    gs = {k: math.log(m['SHIP-REFIT2'][k] / m['SHIP-FROZEN'][k]) for k in ks}
    w = sum(1 for k in ks if gp[k] < gs[k])
    return dict(n=len(ks), prop_more=w, p_one_sided=sign_test_one_sided(w, len(ks)) if ks else None,
                g_prop_median=st.median(gp.values()) if ks else None,
                g_ship_median=st.median(gs.values()) if ks else None,
                grade='EXPLORATORY', mde=mde(len(ks)) if ks else None)


def label_primary(v):
    """PRIMARY = `PROP-REFIT2` 가 **참 제로필드**를 이기는 창 수 (`fc_mean`, lead 2–14)."""
    t = thresholds(v['n'])
    w, n = v['better'], v['n']
    if w >= t['CONFIRM']:
        return ('T2-STAT :: PROP-EARNS-REAL-SKILL-VIA-REFIT (CONFIRMATORY)',
                f'{w}/{n} (p={v["p_one_sided"]:.5f} <= 가족단위 {FAMILY_ALPHA}) ⟹ ★소거표 '
                f'최초의 **양성 행**. 척추가 *배제* → **위치지음**: 경계는 표현이 아니라 '
                f'적합의 **시간축 갱신**에 있다. ⛔단 §7 금지 전건 유효(배포 권고·예산 주장 불가).')
    if w >= t['NOMINAL']:
        return ('T2-STAT :: PROP-EARNS-REAL-SKILL-VIA-REFIT (EXPLORATORY)',
                f'{w}/{n} (p={v["p_one_sided"]:.5f}) — 명목 통과이나 가족단위 미달. '
                f'⚠**확증 아님** · MDE 병기 의무.')
    if w <= t['NOT_EARN']:
        return ('T2-STAT :: PROP-NO-SKILL-EVEN-WITH-REFIT',
                f'{w}/{n} ⟹ ★배포 경로의 **유일하게 측정된 실측 예보 기술 담지자**를 제안 '
                f'표현에 그대로 이식했는데 기술이 생기지 않았다. ⟹ *"공정한 기회를 준 적이 '
                f'없다"* 는 변호가 닫힌다. **소거표 한 줄, 다른 판정 전부 불변.**')
    return ('T2-STAT :: PROP-SKILL-UNDECIDED',
            f'부호로 갈리지 않는다({w}/{n}, p={v["p_one_sided"]:.4f}). ⛔승격 금지 · '
            f'MDE({mde(v["n"])["q80"]}) 병기.')


def label_secondary_a(p):
    """SECONDARY-A = `PROP-REFIT2` < `PROP-FROZEN` (`fc_mean`). `B-3` PRIMARY 의 정확한 거울."""
    t = thresholds(p['n'])
    w, n = p['better'], p['n']
    if w >= t['CONFIRM']:
        return ('T2-STAT :: PROP-REFIT-HELPS (CONFIRMATORY)',
                f'{w}/{n} (p={p["p_one_sided"]:.5f}) — 재적합은 **표현에 무관한 시간축의 '
                f'성질**이다. ⛔이것만으로 *"제안 표현이 실측에서 작동한다"* 를 쓸 수 없다 '
                f'(PRIMARY 를 보라).')
    if w >= t['NOMINAL']:
        return ('T2-STAT :: PROP-REFIT-HELPS (EXPLORATORY)', f'{w}/{n} 명목만.')
    if w <= t['NOT_EARN']:
        return ('T2-STAT :: PROP-REFIT-NULL-OR-HURTS',
                f'{w}/{n} ⟹ 재적합은 제안 표현에서 `B-3` 의 효과를 **재현하지 않는다** — '
                f'`REFIT-HELPS` 는 `bank_mmse` 국소 성질이다.')
    return ('T2-STAT :: PROP-REFIT-UNDECIDED', f'{w}/{n}.')


# ★결과 전 고정된 **읽기 규칙**. ⛔"널이면 무엇을 쓰나" 에 답이 없으면 설계가 아니다.
READING = {
    'PRIMARY+': '★소거표 최초 양성 행. 척추 = 위치지음. 기여 ②c 가 제안 표현으로 확장. '
                '⛔배포 권고 아님(§7-3) · ⛔예산 주장 아님(§7-4).',
    'PRIMARY- & SEC_A+': '★정확한 문장 = *"재적합은 두 표현이 공유하는 시간축의 성질이지만, '
                         '이 코퍼스에서 제안 표현이 기술에 도달하기에는 충분하지 않다."* '
                         '남는 결손은 **재적합이 구조적으로 무관한 lead 0–1** 에 있다 ⟹ '
                         '다음 개입의 표적은 예보 지평이 아니라 **나우캐스트**다.',
    'PRIMARY- & SEC_A-': '★가장 강한 소거 문장(위 label_primary 참조). 시간축 축 전체가 닫힌다.',
    'UNDECIDED': '⛔승격 금지. MDE 를 병기하고 소거표에 *"검출 불가"* 로 적는다 '
                 '(⛔*"차이 없다"* 금지 — `M8b` 개명 사건의 교훈).',
}

DIRECTION = ('LOCATE-OR-ELIMINATE. 이 배치는 "실측 예보 기술의 담지자가 표현인가 시간축 '
             '갱신인가" 만 묻는다. ⛔"경계의 원인은 X 다" 로 넘겨 읽지 않는다 — 양성이어도 '
             '주장은 *"기술 담지자가 시간축에 있다"* 까지이고 나우캐스트 결손은 미해결로 남는다.')

# ─────────────────────────────────────────────────────────────────────────────
# 5. 양성/음성 대조 — 값은 `premise_t2.json` 이 **동결 행에서 재도출**한다(⛔손복사 0건)
# ─────────────────────────────────────────────────────────────────────────────
PREMISE = os.path.join(HERE, 'premise_t2.json')
ANCHORS = {                      # (동결 파일, cell 값, 이 배치의 팔)
    'SHIP-FROZEN': (os.path.join(B3, 'results_b3.jsonl'), 'FROZEN'),
    'SHIP-REFIT2': (os.path.join(B3, 'results_b3.jsonl'), 'REFIT2'),
    'PROP-FROZEN': (os.path.join(B7, 'results_b7b.jsonl'), 'PROP-B7B'),
}
# ⛔`PROP-REFIT2` 에는 앵커가 **없다** — 그것이 이 배치의 신규 칸이다.

# ★배포 기본값 트립와이어 (선례 = `M8_LADDER/gates_m8.DEPLOY_EXTRA_EXPECTED`).
#   ⛔손복사 상수가 아니라 **트립와이어**다: 출하 기본값이 바뀌면 여기서 멈춘다.
DEPLOY_PIN = {
    'bank_refit_every_s': None, 'bank_refit_async': False,
    'real_estimator': 'bank_mmse', 'real_bank_ensemble_k': 3,
    'init_method': 'auto', 'varpro_steps': 400, 'learned_init_iters': 0,
    'auto_switch_thresh': 0, 'lambda_l1': 1e-4, 'tracking_mode': 'w_phi',
    'adaptive_n': True, 'use_confidence': True, 'k_snap_dk': None, 'device': 'cpu',
}


def assert_shipping_defaults():
    from wavesplat.run import DEFAULT_CONFIG
    bad = {k: (v, DEFAULT_CONFIG.get(k)) for k, v in DEPLOY_PIN.items()
           if DEFAULT_CONFIG.get(k) != v}
    if bad:
        raise AssertionError(f'⛔출하 기본값이 이 사전등록 이후 움직였다: {bad}')
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 6. 게이트 (⛔0행 가드 · ⛔필터 항진 금지 · ★개입 계수는 **가지** 단위)
# ─────────────────────────────────────────────────────────────────────────────
def gates(rows, premise=None):
    import json
    out = []
    ok = [r for r in rows if 'error' not in r]
    exp = len(ARMS) * N_PAIRS
    A = out.append

    A(('G-0  런 수', len(ok) == exp, f'{len(ok)}/{exp}'))
    A(('G-0b 에러 0', len(rows) == len(ok), f'{len(rows) - len(ok)}'))
    A(('G-0c 0행 가드', len(ok) > 0 and all(len(_map(ok, a)) == N_PAIRS for a in ARM_ORDER),
       str({a: len(_map(ok, a)) for a in ARM_ORDER})))
    A(('G-0d 셀 전건', len({key(r) for r in ok}) == N_PAIRS, ''))

    # ── 경로 되읽기 (⛔팔 정의를 되읽으면 항등식이다 — 추정기 **객체**에서 읽는다)
    bank = [r for r in ok if r['arm'] in BANK_ARMS]
    prop = [r for r in ok if r['arm'] in PROP_ARMS]
    A(('G-1a bank 팔이 bank 경로', bool(bank) and all(
        (r.get('n_atoms') or 0) == 0 and r.get('real_gate') is True
        and r.get('init_method_readback') == 'bank_mmse' for r in bank),
       str(sorted({(r.get('n_atoms'), r.get('real_gate'),
                    r.get('init_method_readback')) for r in bank}))[:90]))
    A(('G-1b prop 팔이 learned 경로', bool(prop) and all(
        (r.get('n_atoms') or 0) > 0 and r.get('real_gate') is False
        and r.get('init_method_readback') == 'learned' for r in prop),
       str(sorted({(r.get('real_gate'), r.get('init_method_readback')) for r in prop}))[:90]))

    # ── ★개입 계수기 = **가지** 단위 (B-2f 교훈). 기대값은 프로토콜에서 도출된다.
    for arm, fld, want in (('SHIP-FROZEN', 'refit_calls', 0),
                           ('SHIP-REFIT2', 'refit_calls', N_REFIT_EXPECTED),
                           ('PROP-FROZEN', 'reinit_calls', 0),
                           ('PROP-REFIT2', 'reinit_calls', N_REFIT_EXPECTED)):
        got = sorted({r.get(fld) for r in ok if r['arm'] == arm})
        A((f'G-2 {arm} {fld}=={want}', got == [want], str(got)[:40]))
    A(('G-2x 교차 무개입', all(r.get('reinit_calls') in (0, None) for r in ok if r['arm'] in BANK_ARMS)
       and all(r.get('refit_calls') in (0, None) for r in ok if r['arm'] in PROP_ARMS), ''))

    # ── ★음성대조: lead 0·1·2 는 두 팔이 **비트 동일**(첫 재적합은 t=2 관측 투입 뒤다).
    for ref, frz in PAIRS_REFIT:
        a = {key(r): [(r.get('per_lead') or {}).get(f'{l}.0') for l in (0, 1, 2)]
             for r in ok if r['arm'] == ref}
        b = {key(r): [(r.get('per_lead') or {}).get(f'{l}.0') for l in (0, 1, 2)]
             for r in ok if r['arm'] == frz}
        ks = sorted(set(a) & set(b))
        A((f'G-3 음성대조 lead0-2 비트동일 [{frz}|{ref}]',
           bool(ks) and all(a[k] == b[k] for k in ks),
           f'{sum(1 for k in ks if a[k] == b[k])}/{len(ks)}'))

    # ── ★G-MIRROR: 두 표현이 **같은 시각에 같은 관측 집합**으로 재적합했는가
    a = {(r['record'], r['win']): r.get('refit_window_hashes')
         for r in ok if r['arm'] == 'SHIP-REFIT2'}
    b = {(r['record'], r['win']): r.get('refit_window_hashes')
         for r in ok if r['arm'] == 'PROP-REFIT2'}
    ks = sorted(set(a) & set(b))
    A(('G-MIRROR 재적합 창 해시 일치', bool(ks) and all(
        a[k] is not None and a[k] == b[k] and len(a[k]) == N_REFIT_EXPECTED for k in ks),
       f'{sum(1 for k in ks if a[k] == b[k])}/{len(ks)}'))
    A(('G-CAP 재적합 창 <= OBS_CAP', all(
        all(m <= OBS_CAP for m in (r.get('refit_window_n') or []))
        for r in ok if r['arm'] in REFIT_ARMS), ''))

    # ── ★G-ANCHOR: 동결 배치와의 **비트** 재현 (⛔M-8 선례: 실패하면 라벨 없이 닫는다)
    for arm, fld in (('SHIP-FROZEN', 'anchor_bitmatch'), ('SHIP-REFIT2', 'anchor_bitmatch'),
                     ('PROP-FROZEN', 'anchor_bitmatch')):
        v = [r.get(fld) for r in ok if r['arm'] == arm]
        A((f'G-ANCHOR {arm} 비트 {N_PAIRS}/{N_PAIRS}',
           len(v) == N_PAIRS and all(x is True for x in v), f'{sum(1 for x in v if x is True)}/{len(v)}'))

    # ── 구성 되읽기 (⛔입력 dict 복사 금지 — S13b 교훈: **객체**에서 되읽는다)
    A(('G-5 refit 되읽기', all(
        r.get('refit_readback') == ARMS[r['arm']]['EST']['bank_refit_every_s'] for r in ok), ''))
    A(('G-5b prop_refit 되읽기(하네스)', all(
        r.get('prop_refit_readback') == ARMS[r['arm']]['HARN']['prop_refit_s'] for r in ok), ''))
    A(('G-5c 출하 기본값 핀', all(r.get('deploy_pin_ok') is True for r in ok), ''))

    # ── 장치·스레드 스탬프 (⚠배치마다 다르다 — 행에 남기고 **단일 스탬프**를 요구한다)
    st = sorted({(r.get('device'), r.get('cuda_available'), str(r.get('omp_threads')))
                 for r in ok})
    A(('G-8 장치/스레드 단일 스탬프', len(st) == 1, str(st)[:90]))

    # ── 채점 기준 (⛔`ZERO_REF=1.0` 금지)
    A(('G-7 참 제로필드 전건 존재', all(
        fc_zero(r['record'], r['win']) not in (None, 0) for r in ok), ''))
    A(('G-7b ZERO_REF 상수 부재', not any(
        'ZERO_REF' in open(os.path.join(HERE, f), encoding='utf-8', errors='replace').read()
        for f in os.listdir(HERE) if f.endswith('.py')), ''))

    # ── ★양성대조: 동결 자료가 낸 세 수를 이 배치가 재현하는가
    if premise is None and os.path.exists(PREMISE):
        premise = json.load(open(PREMISE))
    if premise:
        for arm, kk in (('SHIP-FROZEN', 'ship_frozen_vs_zero'),
                        ('SHIP-REFIT2', 'ship_refit_vs_zero'),
                        ('PROP-FROZEN', 'prop_frozen_vs_zero')):
            got, wnt = vs_zero(ok, arm)['better'], premise[kk]['better']
            A((f'G-PC {arm} vs 제로 == 동결 {wnt}', got == wnt, f'{got}'))
        got = paired(ok, 'SHIP-REFIT2', 'SHIP-FROZEN')['better']
        A((f'G-PC 짝 재적합 이득 == 동결 {premise["ship_refit_paired"]["better"]}',
           got == premise['ship_refit_paired']['better'], f'{got}'))

    # ── 필터 항진 금지: 위 필터가 실제로 무언가를 배제할 수 있는가
    A(('G-9 필터 비항진', len({r['arm'] for r in ok}) == len(ARMS)
       and len(BANK_ARMS) + len(PROP_ARMS) == len(ARMS), ''))
    return out


def analyze(rows, premise=None):
    ok = [r for r in rows if 'error' not in r]
    pri = vs_zero(ok, 'PROP-REFIT2')
    sec = paired(ok, 'PROP-REFIT2', 'PROP-FROZEN')
    g = gates(ok, premise)
    lab, why = label_primary(pri)
    lab2, why2 = label_secondary_a(sec)
    blocked = [n for n, p, _ in g if not p]
    return dict(
        label=None if blocked else lab, why=why,
        label_secondary=None if blocked else lab2, why_secondary=why2,
        blocked_by=blocked,                 # ⛔게이트 실패 시 **라벨 없이 닫는다**(M-8 선례)
        direction=DIRECTION, reading=READING,
        primary=pri, secondary_a=sec, secondary_b=interaction(ok),
        thresholds=thresholds(), mde=mde(),
        descriptive_nowcast={a: vs_zero(ok, a, *LEAD_NOW) for a in ARM_ORDER},
        gates=[{'name': n, 'pass': bool(p), 'detail': d} for n, p, d in g],
        n_rows=len(rows))


def selftest():
    t = thresholds()
    assert t['n'] == 36 and t['CONFIRM'] == 25 and t['NOMINAL'] == 24 and t['NOT_EARN'] == 12, t
    assert abs(t['p_confirm'] - 0.014408) < 1e-6 and abs(t['p_nominal'] - 0.032623) < 1e-6, t
    assert N_REFIT_EXPECTED == 7, N_REFIT_EXPECTED
    assert mde()['q80'] == 0.742, mde()
    assert len(RECORDS) == 12 and WINDOWS == [1, 3, 5] and N_PAIRS == 36
    assert set(ARM_ORDER) == set(ARMS) and len(ARMS) == 4
    for a in ARM_ORDER:                      # ⛔빈 dict 팔 금지
        assert ARMS[a]['EST'] and 'bank_refit_every_s' in ARMS[a]['EST']
        assert 'prop_refit_s' in ARMS[a]['HARN']
    assert ARMS['PROP-REFIT2']['HARN']['prop_refit_s'] == PROP_REFIT_S
    assert set(READING) == {'PRIMARY+', 'PRIMARY- & SEC_A+', 'PRIMARY- & SEC_A-', 'UNDECIDED'}
    print(f'gates_t2 selftest: PASS  n={t["n"]} CONFIRM={t["CONFIRM"]} '
          f'NOMINAL={t["NOMINAL"]} NOT_EARN={t["NOT_EARN"]} '
          f'family_alpha={FAMILY_ALPHA} refits={N_REFIT_EXPECTED} q80={mde()["q80"]}')
    return True


if __name__ == '__main__':
    selftest()
