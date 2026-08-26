"""B-18 단일출처 — **밀도 × 잡음 2요인**. 러너와 분석기는 이 파일만 읽는다.

⛔이 파일은 **결과가 하나라도 나오기 전에** 커밋된다(git 앵커). 문턱·라벨·팔·판정 규칙은
전부 아래에 있고 **사후에 바꾸지 않는다**.

## 왜 이 배치인가 (STATUS_2026-08-24 §7–8 의 1순위)
2026-08-24 마감 시점에 실측 음성의 원인이 **소거법으로만** 좁혀졌다:
  · 잡음의 *형태*   ⛔아니다 — `B-14b` `OBS-MODEL-DOES-NOT-MEDIATE` (네 모형 최대 8/12)
    (⚠`CORR`/`CLUST` 팔은 σ 축에서 **점 일치 미검증**이다: `obs_models.py:69-72` 의 `CORR` 는
     `rng.normal(0,1,Z.shape)` 로 **격자 전체**를 소비하는데 `noise<=0` 가지는 `len(pick)` 만
     소비한다 ⟹ 상태 전진이 다르고, `results_b14b_*.jsonl` 에 `p_hash` 키가 없어 대조된 적도 없다.
     ✅**B-18 은 `IID` 만 쓰므로 무손상**이다 — 그 경로는 σ 3수준이 전부 `len(pick)` 을 소비한다.
     그 사실을 검사하는 것이 `G-2` 다.)
  · 표현의 *용량*   ⛔아니다 — `B-14` 축 B (조밀·무잡음 10/12, 밀도에 단조 감소)
  · 센서 모형(합성) ⛔결론을 바꾸지 않는다 — `B-15` 17/20
  · ★**관측의 희소성** — **남는다** (480 → 12,000 관측에서 4/12 → 10/12)
그리고 `STATUS.md` §7 이 축자로 적었다:
    *"소거법이다. '희소성이 원인' 을 직접 개입으로 확인한 것이 아니고
      **밀도×잡음 2요인은 미측정**이다. ⟹ 다음 배치의 1순위."*
★**이 배치가 그 소거를 직접 확인으로 바꾼다.**

## 왜 2요인이라야 하는가 (한 요인씩으로는 못 답한다)
  · `B-14` 축 B 는 **σ=0 에서만** 밀도를 올렸다 ⟹ *"배포 잡음에서도 밀도가 구하는가"* 는 미측정.
  · `B-2b`/`B-14b` 는 **m=12 에서만** 잡음을 낮췄다 ⟹ *"밀도가 높으면 잡음이 무해해지는가"* 는 미측정.
  · 두 축이 **상승적**(둘 다 필요)인지 **상쇄적**(하나면 충분)인지는 **완전요인**만이 답한다.
    ★`W10` 이 같은 부류(벌점 2요인)를 2×2 로 갈랐다 — 그 판정 문언(단독 몫 / 합계 몫 / 널)을 본떴다.

## 요인 수준 — ⛔새로 고르지 않았다. **동결 문서에서 상속**한다.
  · 밀도 `DENSITIES`               = `gates_b14.py:52`  [12, 48, 150, 300]  (프레임당 점수)
  · 총 관측 예산 규칙              = `gates_b14.py:53`  m_cap = 40 × m_scan
                                     (실측 nfw = 400–600 프레임이라 **캡이 항상 구속**한다
                                      ⟹ 실제 총 관측 = 480 / 1,920 / 6,000 / 12,000, 레코드 무관)
  · 잡음 `NOISE_LEVELS`            = `gates_b14.py:45`  [0.02, 0.005, 0.0]  (= `run_b2b.py:16` 과 동일)
  · 레코드·창·시드·추정기 팔       = `gates_b14.py:27–35`
  · 배포 예산 600                  = `scene_station.py:68` 의 `m_cap=600` 기본값(= `run_b14.py:39` 축 A)

## 팔 (⛔빈 dict 금지 · 배포 상수 3개를 명시)
`ARM_PROP` = `gates_b14.PROP_CFG` + 2026-08-22 배포 상수 4개를 **명시**.
⚠명시값이 현 `DEFAULT_CONFIG` 와 같은지는 게이트 `G-7b` 가 **코드에서 되읽어** 검사한다
(손복사 상수를 남기지 않기 위해 값이 아니라 **일치**를 검사한다).

## ⛔2026-08-26 수리 — 초판의 `G-6` 은 **항등식이라 실패할 수 없었다** (결과 0행 시점)
`sequential_update.py:399 self.config = config` 이므로 `upd.config[k] == cfg[k]` 는 방금 쓴 것을
되읽는 것이다. 그리고 그 항등식은 **읽히지도 않는 죽은 노브**(`real_domain_gate` ·
`auto_switch_thresh`, 둘 다 `init_method=='auto'` 경로에서만 읽힌다)를 통과시켰다.
⟹ `READBACK_KEYS` 를 폐기하고 `DERIVED_ATTRS` / `COUNTER_EXPECT` / `ROUTER_ATTRS` 셋으로
쪼갰다(각각 `G-6` / `G-6b` / `G-6c`). 상세는 그 상수들 위의 주석.

## ★라벨 공간 (결과 전 공시)
`C_QUIET`·`D_BOTH` 두 모서리는 `B-14` 축 B 가 **이미 잰 셀**이다 ⟹ `LABELS_PRACTICALLY_UNREACHABLE`
와 `prior_exposure_axisB()` 를 볼 것. PREREG §12 가 그 수를 **자료에서 도출해** 적는다.
"""
import glob
import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
B14 = os.path.abspath(os.path.join(HERE, '..', 'B14_OBSMODEL_2026-08-24'))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
for p in (B14, B2):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── 상속 (⛔재기재 금지 — 동결 파일에서 import 한다) ─────────────────────────
from gates_b14 import (RECORDS, WIN, SEED, N,                 # noqa: E402
                       NOISE_LEVELS, DENSITIES, DENSITY_CAP_PER_FRAME,
                       PROP_CFG)

# ── 팔 ──────────────────────────────────────────────────────────────────────
# 2026-08-22 배포 상수. ⚠값 자체는 `G-7b` 가 `wavesplat.run.DEFAULT_CONFIG` 와 대조한다.
SHIPPED_2026_08_22 = {'varpro_steps': 400, 'learned_init_iters': 0,
                      'auto_switch_thresh': 0, 'lambda_l1': 1e-4}
ARM_PROP = dict(PROP_CFG, **SHIPPED_2026_08_22)      # ⛔`{}` 아님

# ── ★개입 되읽기 계약 — ⛔`upd.config` 를 읽는 것은 **항등식이다** ──────────────
# `sequential_update.py:399` 가 `self.config = config` 로 **입력 dict 을 복사 없이 붙든다.**
# ⟹ `upd.config[k] == cfg[k]` 는 방금 쓴 것을 되읽는 것이라 **실패할 수 없고**, 더 나쁘게는
#    **한 번도 읽히지 않는 죽은 노브를 통과시킨다**(W8 `G-4` 공허한 통과와 같은 부류).
#    초판의 `READBACK_KEYS` 6키 게이트가 정확히 그것이었다 ⟹ 아래 셋으로 대체한다.
# 증거가 되는 것은 **config dict 과 독립으로 존재하는 것**뿐이다:
#
#  (1) **파생 속성** — `__init__` 이 config 에서 꺼내 자기 속성으로 스냅샷한 것.
#      `:402 self.lambda_l1` · `:413 self.init_method`.
#      ⚠`init_method` 는 `initialize()` 안에서 **바뀔 수 있다**(`:757`, auto 경로) ⟹
#        런 **뒤에** 읽으면 *"실제로 돈 팔"* 이 된다. 우리 팔은 명시 `'learned'` 라 불변.
#      ⛔`varpro_steps`·`learned_init_iters` 는 속성이 **아니다**(`:913`·`:889` 가 함수 안에서
#        `self.config.get` 을 직접 읽는다) ⟹ 여기 넣으면 다시 항등식이 된다. (2) 로 잡는다.
DERIVED_ATTRS = ('lambda_l1', 'init_method')
#
#  (2) **개입 계수기** — `initialize()` 반환 dict. **함수 본문이 실제로 쓴 값**이다.
#      `res['n_iters']`            = `:889` 가 읽은 `learned_init_iters` (결합 적합 반복수)
#      `res['varpro_steps']`       = `:913` 이 읽은 `varpro_steps`
#      `res['varpro']['vp_steps']` = VARPRO 가 **실제로 돈** 스텝 수(`varpro.py:138 executed`)
#      ⚠`res['varpro']` 는 건너뛴 경우에도 **truthy dict** 이다(`varpro.py:201` 이 먼저
#        `{'vp_entry':1,…}` 를 만들고 `vp_skipped` 를 적어 반환) ⟹ 존재로 발화를 판정하지 않는다.
COUNTER_EXPECT = {'n_iters': 'learned_init_iters',   # 행 `counters` 키 -> ARM_PROP 키
                  'varpro_steps': 'varpro_steps',
                  'vp_steps_exec': 'varpro_steps'}
#
#  (3) **라우터 진단 속성** — `:417`·`:418`·`:427` 이 **무조건** None 으로 초기화하고
#      `auto` 경로에서만 채운다. 전 행 None = `if self.init_method == 'auto'`(`:689`) 블록에
#      **한 번도 들어가지 않았다**는 증거다.
ROUTER_ATTRS = ('_auto_selected', '_auto_n_selected', '_auto_real_gate')
#
# ★그리고 (3) 이 **도달 불가 키 둘**을 확정한다 — 팔이 `init_method` 를 명시하기 때문이다:
INERT_KEYS = {
    'real_domain_gate': "`:706` 이 `if self.init_method == 'auto':`(`:689`) 블록 **안**에서만 "
                        "읽는다 ⟹ 명시 `init_method='learned'` 에서는 읽히지 않는다",
    'auto_switch_thresh': "`:1008 _auto_select_init` 안에서만 읽고 그 함수는 auto 경로(`:757`)"
                          "에서만 불린다 ⟹ 같은 이유로 읽히지 않는다",
}
# ⛔**지우지 않고 팔에 남긴 이유**(결과 전에 적는다): `real_domain_gate` 는 동결 `PROP_CFG` 의
#   축자 상속이고 *"`B-2`/`B-14` 와 글자 그대로 같은 팔"* 이라는 것이 `G-1` 비트 동일의 전제다.
#   `auto_switch_thresh` 는 2026-08-22 배포 상수 3종의 하나라 `G-7b`(현 `DEFAULT_CONFIG` 대조)에
#   필요하다. ⟹ ★**문서용으로 남기되 "개입이 먹었다" 의 증거로는 세지 않는다.**
#   미발화 자체는 `G-6c` 가 증명한다.

GRID = 'GRID'                # 완전요인 팔: m_cap = DENSITY_CAP_PER_FRAME × m_scan
PC = 'PC-DEPLOY'             # 양성대조 팔: 배포 예산(m_scan=12, m_cap=600)
ARM_ORDER = [GRID, PC]
PC_M_SCAN, PC_CAP = 12, 600
FIELD = 'nowcast'

M_LO, M_HI = min(DENSITIES), max(DENSITIES)          # 12, 300
NZ_HI, NZ_LO = max(NOISE_LEVELS), min(NOISE_LEVELS)  # 0.02(배포), 0.0(무잡음)

# ★결과 전 확정: 네 모서리. 이름은 판정 문언에 그대로 쓰인다.
CORNERS = {
    'A_DEPLOY': (M_LO, NZ_HI),   # 배포 근사   — 희소 + 배포 잡음
    'B_DENSE':  (M_HI, NZ_HI),   # 밀도만      — 조밀 + 배포 잡음
    'C_QUIET':  (M_LO, NZ_LO),   # 잡음만      — 희소 + 무잡음
    'D_BOTH':   (M_HI, NZ_LO),   # 둘 다       — 조밀 + 무잡음
}
CORNER_ORDER = ['A_DEPLOY', 'B_DENSE', 'C_QUIET', 'D_BOTH']

# ── 문턱 (결과 전 계산, n=12 단측 부호검정) ─────────────────────────────────
#   10/12 -> 0.01929 · 11/12 -> 0.00317 · 12/12 -> 0.00024
#   PRIMARY 가족 = 모서리 4개 ⟹ α = 0.05/4 = 0.0125 ⟹ **11/12 이라야 확증**
#   양측(상호작용)도 10/12 -> 0.0386 < 0.05 이므로 같은 NOMINAL 을 쓴다.
NOMINAL = 10
CONFIRM = 11
FAMILY_ALPHA = 0.05 / len(CORNERS)

# ── 양성대조 참조 (동결 파일) ───────────────────────────────────────────────
REF_B14B_GLOB = os.path.join(B14, 'results_b14b_*.jsonl')   # 수정 후 `IID` 열 (36행)
REF_B2B = os.path.join(B2, 'results_b2b.jsonl')             # 원조 B-2b (noise=0.02)
REF_B14 = os.path.join(B14, 'results_b14.jsonl')            # 원조 B-14 (축 B = σ=0 밀도 사다리)
# ⚠BS 7 레코드는 `scene_station.py` 의 수심 정정(17.0 → 30.0, 2026-08-23)으로
#   `B-2b` 와의 비트 일치가 **원리적으로 불가능**하다(`RESULT_B14.md` §1 ①).
#   ⟹ B-2b 사슬은 비-BS 5 레코드에서만 요구한다.
NON_BS = [r for r in RECORDS if not r.startswith('BS')]
BIT_TOL = 1e-9


# ── 검정 ────────────────────────────────────────────────────────────────────
def sign_test_one_sided(w, n=N):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def sign_test_two_sided(w, n=N):
    return min(1.0, 2 * min(sign_test_one_sided(w, n),
                            sign_test_one_sided(n - w, n)))


def cells():
    """(arm, record, m_scan, noise, m_cap) — 144 완전요인 + 36 양성대조 = **180**."""
    for rec in RECORDS:
        for m in DENSITIES:
            for nz in NOISE_LEVELS:
                yield (GRID, rec, m, nz, DENSITY_CAP_PER_FRAME * m)
    for rec in RECORDS:
        for nz in NOISE_LEVELS:
            yield (PC, rec, PC_M_SCAN, nz, PC_CAP)


N_GRID = len(RECORDS) * len(DENSITIES) * len(NOISE_LEVELS)      # 144
N_PC = len(RECORDS) * len(NOISE_LEVELS)                          # 36


# ── 판정 라벨 (결과 전 확정) ────────────────────────────────────────────────
def label_shape(w):
    """`w` = {모서리 이름: 참 제로필드를 이긴 레코드 수}. **verdict.label 은 이것 하나다.**

    ★해석: *"회복"* = 그 조건에서 **참 제로필드(무기술 기준)를 이기는 레코드 수 ≥ NOMINAL**.
    ⛔`ZERO_REF=1.0` 을 쓰지 않는다 — 기준은 `zero_ref.py` 가 자료에서 낸다.

    ⚠**두 가지는 사실상 발화하지 않는다** — `LABELS_PRACTICALLY_UNREACHABLE` 참조.
    `C_QUIET`·`D_BOTH` 는 `B-14` 축 B 의 σ=0 열과 같은 셀이고 그 값이 이미 공개돼 있다.
    ⛔그래도 **가지를 지우지 않는다**(지우면 사후에 라벨 공간을 좁힌 것이 된다).
    """
    a, b, c, d = (w['A_DEPLOY'], w['B_DENSE'], w['C_QUIET'], w['D_BOTH'])
    if a >= NOMINAL:
        return ('PREMISE-FAILS',
                f'배포 근사 모서리가 이미 {a}/{N} 로 회복돼 있다 ⟹ *"실측 음성"* 전제가 '
                f'이 구성에서 성립하지 않는다. 2요인 질문을 읽지 않는다')
    dens, noi, both = b >= NOMINAL, c >= NOMINAL, d >= NOMINAL
    if dens and not noi:
        base, dec = ('SPARSITY-IS-THE-BINDING-CONSTRAINT', b)
        txt = (f'배포 잡음(σ={NZ_HI})에서 밀도만 올려도 {b}/{N} 회복, 잡음만 낮추면 {c}/{N} '
               f'⟹ ★구속 조건은 **관측의 희소성**이다')
    elif noi and not dens:
        base, dec = ('NOISE-IS-THE-BINDING-CONSTRAINT', c)
        txt = (f'배포 밀도(m={M_LO})에서 잡음만 낮춰도 {c}/{N} 회복, 밀도만 올리면 {b}/{N} '
               f'⟹ 구속 조건은 **관측 잡음**이다 (⚠`B-14b` 최대 8/12 와 충돌 — §해석규칙)')
    elif dens and noi:
        base, dec = ('EITHER-SUFFICIENT', min(b, c))
        txt = (f'밀도만 {b}/{N} · 잡음만 {c}/{N} — **각각 단독으로 충분**하다 '
               f'⟹ ⛔*"구속 조건"* 이라는 **단수 표현을 쓸 수 없다**')
    elif both:
        base, dec = ('BOTH-REQUIRED', d)
        txt = (f'단독으로는 회복하지 못하고(밀도 {b}/{N} · 잡음 {c}/{N}) '
               f'둘 다일 때만 {d}/{N} ⟹ ★**상호작용**: 두 요인이 함께라야 회복한다')
    else:
        return ('NEITHER-SUFFICIENT',
                f'어느 조합도 회복하지 못한다 (밀도 {b}/{N} · 잡음 {c}/{N} · 둘 다 {d}/{N}) '
                f'⟹ ★실측 음성의 원인은 **측정된 네 후보 밖**에 있다')
    return (base if dec >= CONFIRM else base + '-NOMINAL',
            txt + (f' [가족단위 통과, p={sign_test_one_sided(dec):.4g}]' if dec >= CONFIRM
                   else f' [**명목만** {dec}/{N}, p={sign_test_one_sided(dec):.4g} — 탐색적]'))


def label_dose(w, name):
    """SECONDARY 짝 검정(단측). 기준값이 소거되므로 `ZERO_REF` 사고에 면역이다."""
    if w >= CONFIRM:
        return (f'{name}-HELPS', f'{w}/{N}, p={sign_test_one_sided(w):.4g}')
    if w >= NOMINAL:
        return (f'{name}-HELPS-NOMINAL', f'{w}/{N}, p={sign_test_one_sided(w):.4g} — 명목만')
    return (f'{name}-NULL', f'{w}/{N}, p={sign_test_one_sided(w):.4g}')


def label_interaction(w_syn):
    """상호작용(양측). `w_syn` = 가법 대비 `I = (A−B) − (C−D)` 가 **음수**인 레코드 수.

    `I < 0` ⟹ 밀도 이득이 **무잡음에서 더 크다** = **상승(synergy)**  ⟹ "둘 다 필요" 쪽.
    `I > 0` ⟹ 밀도 이득이 **배포 잡음에서 더 크다** = **포화(saturating)** ⟹ "하나면 충분" 쪽.
    ⚠가법 척도에서 두 방향(밀도의 잡음 의존 / 잡음의 밀도 의존)은 **같은 대비**다.
    """
    p = sign_test_two_sided(w_syn)
    if w_syn >= NOMINAL:
        return ('INTERACTION-SYNERGY', f'{w_syn}/{N} 에서 I<0, 양측 p={p:.4g}')
    if (N - w_syn) >= NOMINAL:
        return ('INTERACTION-SATURATING', f'{N - w_syn}/{N} 에서 I>0, 양측 p={p:.4g}')
    return ('ADDITIVE-NO-INTERACTION', f'I<0 {w_syn}/{N}, 양측 p={p:.4g}')


# ── 게이트 (전부 여기서 낸다; 분석기는 이 함수만 부른다) ─────────────────────
def _load_ref_b14b():
    ref = {}
    for p in sorted(glob.glob(REF_B14B_GLOB)):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' in r or r.get('obs_model') != 'IID':
                continue
            ref[(r['record'], r['noise'])] = r['nowcast']
    return ref


def _load_ref_b2b():
    ref = {}
    if not os.path.exists(REF_B2B):
        return ref
    for line in open(REF_B2B):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' not in r and abs(r['noise'] - NZ_HI) < 1e-12:
            ref[r['record']] = r['nowcast']
    return ref


def load_ref_b14_axisB():
    """동결 `results_b14.jsonl` 의 **축 B**(IID · σ=0 · 밀도 사다리) 를 (record, m_scan) 색인으로.

    ★이 배치의 `GRID` σ=0 열은 **같은 호출**이다(`run_b14.py:39` `cap = m_scan*40`,
    `model='IID'`, `noise=0.0`) ⟹ **사전 노출**이고, §12 가 그 수를 여기서 도출해 적는다.
    """
    ref = {}
    if not os.path.exists(REF_B14):
        return ref
    for line in open(REF_B14):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' in r or r.get('axis') != 'B':
            continue
        ref[(r['record'], r['m_scan'])] = r
    return ref


def prior_exposure_axisB():
    """★사전 노출을 **자료에서 도출한다**(⛔손복사 금지). {m_scan: (beats, n, n_obs)}."""
    ref, out = load_ref_b14_axisB(), {}
    for m in DENSITIES:
        rows = [r for (rec, mm), r in ref.items() if mm == m]
        out[m] = (sum(1 for r in rows if r['beats_zero']), len(rows),
                  (rows[0]['n_obs_used'] if rows else None))
    return out


# ── ★라벨 공간은 런 전에 **사실상 반쯤 결정돼 있다** (결과 전 공시) ─────────────
# `C_QUIET`(m=12, σ=0) 와 `D_BOTH`(m=300, σ=0) 은 위 축 B 가 **이미 잰 셀**이다.
# 그 값을 `label_shape` 에 대입하면 `NOISE-IS-THE-BINDING-CONSTRAINT` 와 `EITHER-SUFFICIENT`
# 는 `C_QUIET >= NOMINAL` 을 요구하는데 축 B 는 그 셀에서 `NOMINAL` 에 **한참 못 미쳤다**.
# ⟹ 아래 둘은 **원리적 도달 불가가 아니라 사실상 발화하지 않는 가지**다. ⛔지우지 않는다
#    (지우면 사후에 라벨 공간을 좁힌 것이 된다). PREREG §7 의 충돌 처리 규칙도 그대로 둔다.
LABELS_PRACTICALLY_UNREACHABLE = ('NOISE-IS-THE-BINDING-CONSTRAINT', 'EITHER-SUFFICIENT')
# ⚠**그러나 "확정" 은 아니다** — 2026-08-24 난수 수리(`obs_models.py:68-69`, `np.zeros` →
#   `rng.normal(0,0.0,n)`)로 σ=0 의 **점 집합이 바뀌었다** ⟹ 축 B 의 수가 이 배치에서
#   비트 재현되지 **않는다**(`G-1c` 가 그 사실 자체를 검사한다). 사전 노출은 실질적이되
#   값의 재현은 보장되지 않는다.


def _true_zero():
    from zero_ref import zero_ref_nowcast
    return {rec: zero_ref_nowcast(rec, win=WIN, noise=NZ_HI) for rec in RECORDS}


def _eq(got, want):
    """JSON 왕복 뒤의 동치. float 은 비트 허용오차로, 그밖은 정확 일치로 본다."""
    if isinstance(want, bool) or want is None:
        return got is want
    if isinstance(want, float):
        return isinstance(got, (int, float)) and not isinstance(got, bool) \
            and abs(float(got) - want) <= BIT_TOL * max(1.0, abs(want))
    return got == want


def gates(rows):
    """러너 산출 전체(오류 행 포함)를 받아 게이트를 판정한다."""
    ok = [r for r in rows if 'error' not in r]
    # ⛔**공허한 통과 금지** — 0행에서 참이 되는 게이트를 만들지 않는다(W8 `G-4` 부류).
    out = [('G-5 오류 0행', bool(rows) and len(ok) == len(rows),
            f'{len(rows) - len(ok)} 행 오류 / 전체 {len(rows)}')]

    n_grid = sum(1 for r in ok if r['arm'] == GRID)
    n_pc = sum(1 for r in ok if r['arm'] == PC)
    out.append(('G-0 표본 수', n_grid == N_GRID and n_pc == N_PC,
                f'GRID {n_grid}/{N_GRID} · PC {n_pc}/{N_PC}'))

    # ── G-1 ★양성대조 (ABORT 게이트) — PC 팔 = B-14b `IID` 열과 **비트 동일** ──
    ref = _load_ref_b14b()
    mine = {(r['record'], r['noise']): r[FIELD] for r in ok if r['arm'] == PC}
    common = sorted(set(ref) & set(mine))
    dmax = max((abs(mine[k] - ref[k]) for k in common), default=float('nan'))
    n_bit = sum(1 for k in common if abs(mine[k] - ref[k]) <= BIT_TOL)
    out.append((f'G-1 ★양성대조: PC 팔 = B-14b IID (ABORT)',
                bool(common) and n_bit == len(common) and len(common) == N_PC,
                f'{n_bit}/{len(common)} 비트 동일 (최대 Δ={dmax:.3e})'))

    # ── G-1b 사슬: PC σ=0.02 의 **비-BS 5 레코드** = 원조 B-2b ──────────────
    ref2 = _load_ref_b2b()
    c2 = [k for k in NON_BS if k in ref2 and (k, NZ_HI) in mine]
    n2 = sum(1 for k in c2 if abs(mine[(k, NZ_HI)] - ref2[k]) <= BIT_TOL)
    out.append(('G-1b 사슬: PC σ=0.02 = B-2b (비-BS 5개)',
                bool(c2) and n2 == len(c2) and len(c2) == len(NON_BS),
                f'{n2}/{len(c2)} 비트 동일 (BS 7 은 수심 정정으로 원리적 불가)'))

    # ── G-1c ★σ=0 열 이식 충실도 — B-14 축 B 대조 (48셀) ────────────────────
    #   ⛔**비트 동일을 요구하지 않는다.** 2026-08-24 난수 수리(`obs_models.py:68-69`)가
    #     σ=0 에서 `np.zeros(n)` → `rng.normal(0,0.0,n)` 로 바뀌어 **이후 프레임의
    #     `rng.choice` 가 달라진다** ⟹ 점 집합이 원리적으로 다르다(결과 전 실측 확인).
    #   ⟹ 검사하는 것은 **구조 보존 + 수리 발화** 둘이다:
    #      (i) 축 B 와 같은 (레코드 × 밀도) 48셀이 다 있고 `n_obs_used` 가 정확히 같다
    #      (ii) `z_hash` 가 **전건 다르다** = 낡은 장면 빌더로 조용히 되돌아가지 않았다
    #   회복 수의 대조 자체는 **검정이 아니라 서술**이다(분석기가 verdict 에 나란히 싣는다).
    refB = load_ref_b14_axisB()
    mine0 = {(r['record'], r['m_scan']): r for r in ok
             if r['arm'] == GRID and r['noise'] == NZ_LO}
    c3 = sorted(set(refB) & set(mine0))
    same_obs = sum(1 for k in c3 if mine0[k]['n_obs_used'] == refB[k]['n_obs_used'])
    diff_z = sum(1 for k in c3 if mine0[k]['z_hash'] != refB[k]['z_hash'])
    n_exp0 = len(RECORDS) * len(DENSITIES)
    out.append(('G-1c σ=0 열 = B-14 축 B 구조 일치 + rng 수리 발화(⛔비트 동일 아님)',
                len(c3) == n_exp0 and same_obs == len(c3) and diff_z == len(c3),
                f'{len(c3)}/{n_exp0} 셀 대조 · n_obs 일치 {same_obs} · z_hash 상이 {diff_z}'))

    # ── G-2 ★난수 소비 동수: 같은 (팔, 레코드, 밀도)에서 σ 3수준의 P 가 동일 ──
    #   (2026-08-24 `B-14` 축 A 를 죽인 결함 부류. 이것이 "짝 개입" 의 증거다.)
    grp = {}
    for r in ok:
        grp.setdefault((r['arm'], r['record'], r['m_scan']), set()).add(r['p_hash'])
    bad = [k for k, v in grp.items() if len(v) != 1]
    out.append(('G-2 ★난수 소비 동수(σ 축에서 P 비트 동일)', bool(grp) and not bad,
                f'{len(grp) - len(bad)}/{len(grp)} 군 동일'))

    # ── G-3 밀도 개입 발화: 밀도가 다르면 P 가 다르다 + 예산 되읽기 ────────
    per = {}
    for r in ok:
        if r['arm'] == GRID:
            per.setdefault(r['record'], set()).add(r['p_hash'])
    dens_fired = all(len(v) == len(DENSITIES) for v in per.values()) and bool(per)
    cap_ok = all(r['n_obs_used'] == r['m_cap'] for r in ok)
    exp_cap = all(r['m_cap'] == DENSITY_CAP_PER_FRAME * r['m_scan']
                  for r in ok if r['arm'] == GRID) and \
        all(r['m_cap'] == PC_CAP and r['m_scan'] == PC_M_SCAN for r in ok if r['arm'] == PC)
    out.append(('G-3 밀도 개입 발화 + 예산 되읽기',
                dens_fired and cap_ok and exp_cap,
                f'밀도별 P 상이 {sum(1 for v in per.values() if len(v) == len(DENSITIES))}'
                f'/{len(per)} · n_obs_used==m_cap {cap_ok} · 캡 규칙 {exp_cap}'))

    # ── G-3b 잡음 개입 발화: 같은 (레코드,밀도)에서 σ 3수준의 z 가 서로 다르다 ─
    zg = {}
    for r in ok:
        zg.setdefault((r['arm'], r['record'], r['m_scan']), set()).add(r['z_hash'])
    z_ok = all(len(v) == len(NOISE_LEVELS) for v in zg.values()) and bool(zg)
    out.append(('G-3b 잡음 개입 발화(σ 축에서 z 상이)', z_ok,
                f'{sum(1 for v in zg.values() if len(v) == len(NOISE_LEVELS))}/{len(zg)} 군'))

    # ── G-4 채점 기준 단일출처: 행의 zero_ref == zero_ref.py 산출 (⛔상수 1.0 금지) ─
    try:
        tz = _true_zero()
        dz = max(abs(r['zero_ref'] - tz[r['record']]) for r in ok) if ok else float('nan')
        distinct = len({round(v, 6) for v in tz.values()})
        z4 = dz <= BIT_TOL and distinct > 1 and 1.0 not in {round(v, 6) for v in tz.values()}
        d4 = f'최대 Δ={dz:.3e} · 기준 {distinct} 종 (범위 {min(tz.values()):.4f}–{max(tz.values()):.4f})'
    except Exception as e:                                    # noqa: BLE001
        z4, d4 = False, f'{type(e).__name__}: {e}'
    out.append(('G-4 채점 기준 = zero_ref.py 단일출처 (⛔1.0 금지)', z4, d4))

    # ── G-6 ★파생 속성 일치 — ⛔`upd.config` 되읽기(=항등식)가 아니다 ───────
    exp_d = {k: ARM_PROP[k] for k in DERIVED_ATTRS}
    d_have = bool(ok) and all(all(k in (r.get('derived') or {}) for k in DERIVED_ATTRS)
                              for r in ok)
    d_bad = [r for r in ok
             if not all(_eq((r.get('derived') or {}).get(k), v) for k, v in exp_d.items())]
    out.append(('G-6 파생 속성 일치(⛔config 되읽기 아님 — sequential_update.py:402/:413)',
                d_have and not d_bad,
                f'{len(ok) - len(d_bad)}/{len(ok)} 행 · '
                + ' · '.join(f'upd.{k}={v}' for k, v in exp_d.items())))

    # ── G-6b ★개입 계수기 — 함수 본문이 실제로 쓴 값 (도달 가능성의 증거) ───
    exp_c = {rk: ARM_PROP[ak] for rk, ak in COUNTER_EXPECT.items()}
    c_have = bool(ok) and all(all(k in (r.get('counters') or {}) for k in COUNTER_EXPECT)
                              for r in ok)
    c_bad = [r for r in ok
             if not all(_eq((r.get('counters') or {}).get(k), v) for k, v in exp_c.items())]
    skipped = [r for r in ok if (r.get('counters') or {}).get('vp_skipped') is not None]
    out.append(('G-6b ★개입 계수기(실행된 VARPRO 스텝 · 결합적합 반복수)',
                c_have and not c_bad and not skipped,
                f'{len(ok) - len(c_bad)}/{len(ok)} 행 일치 · VARPRO 건너뜀 {len(skipped)} · '
                + ' · '.join(f'{k}=={v}' for k, v in exp_c.items())))

    # ── G-6c ★라우터 미발화 ⟹ 도달 불가 키 확인 ────────────────────────────
    #   `_auto_*` 는 `__init__`(:417/:418/:427)이 무조건 None 으로 두고 `auto` 경로에서만
    #   채운다. 전 행 None = `:689` 블록 미진입 = `real_domain_gate`(:706) ·
    #   `auto_switch_thresh`(:1008) 가 **한 번도 읽히지 않았다**.
    r_have = bool(ok) and all(all(k in (r.get('router') or {}) for k in ROUTER_ATTRS)
                              for r in ok)
    r_bad = [r for r in ok
             if any((r.get('router') or {}).get(k) is not None for k in ROUTER_ATTRS)]
    out.append(('G-6c ★라우터 미발화 확인 ⟹ 팔의 도달 불가 키 2종은 문서용',
                r_have and not r_bad,
                f'{len(ok) - len(r_bad)}/{len(ok)} 행에서 {ROUTER_ATTRS} 전부 None · '
                + ' · '.join(f'{k}: {v}' for k, v in INERT_KEYS.items())))

    # ── G-7 팔 정의 비어있지 않음 / G-7b 명시 배포 상수 == 현 DEFAULT_CONFIG ─
    out.append(('G-7 팔 정의 비어있지 않음(⛔`{}` 금지)', bool(ARM_PROP), str(ARM_PROP)))
    try:
        from wavesplat.run import DEFAULT_CONFIG
        cur = {k: DEFAULT_CONFIG.get(k) for k in SHIPPED_2026_08_22}
        out.append(('G-7b 명시 배포 상수 == DEFAULT_CONFIG', cur == SHIPPED_2026_08_22,
                    f'{cur} vs {SHIPPED_2026_08_22}'))
    except Exception as e:                                    # noqa: BLE001
        out.append(('G-7b 명시 배포 상수 == DEFAULT_CONFIG', False, f'{type(e).__name__}: {e}'))

    # ── G-8 채점 창 동일성 (레코드 안에서 15셀 전부 같은 창) ────────────────
    win = {}
    for r in ok:
        win.setdefault(r['record'], set()).add(
            (r['win'], r['i0'], round(r['t0_abs'], 9), r['npx'], r['n_gt_finite']))
    w_ok = all(len(v) == 1 for v in win.values()) and bool(win)
    out.append(('G-8 채점 창 동일(레코드별 1종)', w_ok,
                f'{sum(1 for v in win.values() if len(v) == 1)}/{len(win)} 레코드'))
    return out
