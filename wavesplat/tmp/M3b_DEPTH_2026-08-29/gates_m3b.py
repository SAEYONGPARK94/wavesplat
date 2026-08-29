"""M3b 단일출처 — ★상수·문턱·팔·라벨·해석 규칙은 **전부 여기 있다.**
`PREREG.md` 의 숫자는 이 파일의 산문 사본이고, 불일치하면 **코드가 이긴다.**

## 무엇을 재는가
`M3`(2026-08-28)의 `DEP12` 96런은 사전등록한 처치를 **집행하지 않았다**
(`AMENDMENT_2` §8.3 = `UNMEASURABLE (TREATMENT-NOT-DELIVERED)`).
`wavesim/data_writer.load_episode` 가 h5 의 `depth` 를 **읽지 않아**
`wavesplat/data_loader.py:46` 의 하드코딩 폴백 `50.0` 이 **두 팔 모두**에 갔다.
⟹ M3b 는 **같은 동결 코퍼스**(생성 0회 · 신규 시드 0개)에 수심을 **실제로 집행**하고,
그 개입이 무엇을 바꾸는지 잰다.

## ⛔M3b 가 답하지 못하는 것 (결과 전에 못박는다)
학습 **사전**(체크포인트 + 그 beamform 조건자)은 **어느 팔에서도 d=50 이다** —
`scripts/d3_beamform_net.py:30 DEPTH = 50.0` 은 모듈 상수라 인자를 받지 않고,
체크포인트의 학습 자료 자체가 `d3_beamform_cond_diffusion.py:96,108` 에서 `depth=DEPTH`
로 생성됐다. ⟹ ★***이 배치는 "사전이 수심 지지집합 밖에서 버는가" 에 답할 수 없다.***
그래서 ship-계열 팔 이름은 **전부 `CKPT50` 로 시작한다** — 이름이 한계를 들고 다닌다.
⛔팔 이름에 `SHIP` 문자열을 쓰지 않는다(§ARMS 주석).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
M3 = os.path.join(ROOT, 'tmp', 'M3_SUPPORT_2026-08-28')
B12 = os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23')
B11 = os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23')
B26 = os.path.join(ROOT, 'tmp', 'B26_SEASTATE_2026-08-27')
S16 = os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')
for _p in (HERE, M3, B12, B11, B26, S16, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── ★손복사 0: 집계·검정·파생측정·되읽기 목록을 동결 배치에서 그대로 import ─────
from gates_b11 import (EST_SEEDS, FILTER_SEEDS, SPEC,           # noqa: F401,E402
                       sign_test_one_sided, med_by_episode, paired)
from gates_m3 import (_read_ep, derived_coords, deploy_constants,  # noqa: F401,E402
                      DEPLOY_EXPECTED, ROUTE_EXPECTED, G,
                      DISP_TRUE_MAX, DISP_50_MIN,
                      SHIP_CAPTURE as M3_SHIP_CAPTURE,
                      SPEC_CAPTURE as M3_SPEC_CAPTURE)

# ═════════════════════════════════════════════════════════════════════════════
# 1. 표본 — ★동결 M3 코퍼스 재사용. 생성 0회 · 신규 시드 0개 · 디스크 증가 0.
# ═════════════════════════════════════════════════════════════════════════════
CELL = 'DEP12'
DATA = os.path.join(M3, 'data_m3', CELL)
TRUE_DEPTH = 12.0                    # h5 `config.attrs['depth']` 참값 (G-2 가 되읽는다)
FALLBACK_DEPTH = 50.0                # `wavesplat/data_loader.py:46` 하드코딩 폴백
NCOMPS = (15, 30)
SEEDS = tuple(range(1695, 1703))     # M3 가 이미 소모 (1695–1702). ★M3b 소모 = 0
N_EP = len(NCOMPS) * len(SEEDS)      # 16
SEA_STATE = 'BF4'
SEEDS_CONSUMED = ()                  # ★신규 소모 없음 — 원장을 건드리지 않는다

# ★PC-A(비트불변) 대상 = 참 수심이 50.0 인 **다른** 코퍼스의 에피소드.
#   저장소 전수 스캔(1,496개 episode_*.h5)에서 depth≠50 은 위 16개뿐이므로,
#   이 두 에피소드에서 비트 동일이면 나머지 1,480개에 대해서도 같은 값이 흐른다.
PCA_EPISODES = ('SS-BF4_15comp_seed1641_uniform_ra_rf_ds',
                'SS-BF4_30comp_seed1641_uniform_ra_rf_ds')
PCA_SEED = 42
B26_DATA = os.path.join(B26, 'data_b26')
# ★PC-B(M3 재현) 대상 = 동결 M3 결과의 DEP12 행 전건 (96행)
M3_RESULTS_GLOB = os.path.join(M3, 'results_m3_*.jsonl')


def fname(ncomp, seed):
    return f'episode_{SEA_STATE}_{ncomp}comp_seed{seed}_uniform_ra_rf_ds.h5'


def episodes():
    """16 에피소드. ⛔파일명으로 처치를 주장하지 않는다 — `gates_data` 가 attrs 를 되읽는다."""
    out = []
    for ncomp in NCOMPS:
        for seed in SEEDS:
            out.append(dict(path=os.path.join(DATA, fname(ncomp, seed)),
                            episode=fname(ncomp, seed)[len('episode_'):-3],
                            cell=CELL, bf=SEA_STATE, ncomp=ncomp, seed=seed))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 2. ★팔 — ⛔빈 dict `{}` 0개. 세 축(RT · 조건자 · 체크포인트)을 전부 명시한다.
# ═════════════════════════════════════════════════════════════════════════════
# RT = runtime depth: 추정계·필터가 **실제로 푸는** 분산관계의 수심.
#   구현은 **단 하나의 shim** = `wavesplat.data_loader.load_episode` 다.
#   ★양 팔이 모두 `EpisodeDataLoader` 를 통과하므로 **비대칭 전달이 구조적으로 불가능**하다
#     (= `AMENDMENT_2` §4 가 경고한 "한쪽만 참 수심" 을 설계로 배제).
#   ⛔`wavesplat/run.py:216` 을 고쳐서 호출자 인자로 넣지 **않는다** — 그러면 ship 팔에만
#     있는 노브가 되어 정확히 그 비대칭을 만든다(§PREREG "run.py 정책").
RT50 = 'RT50'          # shim 미설치 = **오늘의 배포 거동**(폴백 50.0)
RT12 = 'RT12'          # shim 설치   = h5 참값 12.0

ARMS = {
    # ── ship 계열: 이름이 전부 `CKPT50` 로 시작한다 = 체크포인트는 어느 팔에서도 d=50 ──
    'CKPT50-RT50': dict(
        family='ship', rt=RT50, cond_depth=50.0, ckpt_depth=50.0,
        runner='run_b12.run_ship', config='dict(DEFAULT_CONFIG) + seed',
        seeds=EST_SEEDS, seed_field='est_seed',
        note='★= M3 `SHIP-NOW` 의 DEP12 팔 = **오늘 배포된 시스템이 얕은 물에서 하는 것**. '
             'PC-B 가 동결 M3 행과 비트 대조한다.'),
    'CKPT50-RT12': dict(
        family='ship', rt=RT12, cond_depth=50.0, ckpt_depth=50.0,
        runner='run_b12.run_ship', config='dict(DEFAULT_CONFIG) + seed',
        seeds=EST_SEEDS, seed_field='est_seed',
        note='★처치. ⛔**오늘의 배포가 아니다** — `load_episode` 수리를 **선행 조건**으로 '
             '가정한 조건부 배포다. ⛔사전은 여전히 d=50 ⟹ 지지집합 질문에 답하지 못한다.'),
    'CKPT50-RT12-C12': dict(
        family='ship', rt=RT12, cond_depth=12.0, ckpt_depth=50.0,
        runner='run_b12.run_ship (+ d3_beamform_net.OM_G 재바인딩)',
        config='dict(DEFAULT_CONFIG) + seed', seeds=EST_SEEDS, seed_field='est_seed',
        note='⚠**EXPLORATORY 전용 · 승격 금지.** 조건자 ω-격자를 d=12 로 재계산한다. '
             '★이것은 조건자를 **분리하지 않는다** — 오지정 하나(조건자 vs 참 물리)를 '
             '다른 오지정(조건자 입력 vs 체크포인트 학습 분포)과 **맞바꾼다**.'),
    # ── LETKF 계열: 수심이 **순수 런타임 인자**라 완전 수리가 가능하다 ──
    'SPEC25-RT50': dict(
        family='spec', rt=RT50, cond_depth=None, ckpt_depth=None,
        runner='run_b12.run_spec -> run_s16.run_letkf("SPEC25")',
        config='jonswap · loc_radius 25 · n_ensemble 600', seeds=FILTER_SEEDS,
        seed_field='filter_seed',
        note='★= M3 `SPEC25` 의 DEP12 팔. PC-B 가 동결 M3 행과 비트 대조한다.'),
    'SPEC25-RT12': dict(
        family='spec', rt=RT12, cond_depth=None, ckpt_depth=None,
        runner='run_b12.run_spec -> run_s16.run_letkf("SPEC25")',
        config='jonswap · loc_radius 25 · n_ensemble 600', seeds=FILTER_SEEDS,
        seed_field='filter_seed',
        note='★처치 · **완전 수리**(배경 스펙트럼 `letkf_spec.py:166` 과 예보 연산자 '
             '`letkf_grid.py:80` 이 같은 참 수심을 받는다).'),
}
ZERO_FROM = 'SPEC25-RT50'    # ZERO = 기준선 행의 `zero_nRMSE` (수심 무관 — PC-D 가 증명)

SHIP_ARMS = tuple(a for a, d in ARMS.items() if d['family'] == 'ship')
SPEC_ARMS = tuple(a for a, d in ARMS.items() if d['family'] == 'spec')
ARM_ORDER = ('CKPT50-RT50', 'CKPT50-RT12', 'CKPT50-RT12-C12',
             'SPEC25-RT50', 'SPEC25-RT12')

# ★되읽기 목록 — M3 것을 **상속**하고 두 결함을 메운다.
#   ① ship 쪽 실효 수심: `SequentialUpdater.__init__` 객체 되읽기(러너 shim). ★패키지 무수정.
#      (`run_estimation` 요약에 `depth` 키가 없어 `SHIP_CAPTURE` 로는 잡을 수 없었다 —
#       그래서 M3 의 `G-17` 이 **기준선 쪽에만** 걸렸다: `AMENDMENT_2` §4.)
#   ② 셸 — "계산되는데 저장 안 됨" **여섯 번째**. 양 팔 모두 붙인다.
SHIP_CAPTURE = M3_SHIP_CAPTURE + ('avg_shell_rmse', 'shell_frac', 'avg_RMSE_near')
SPEC_CAPTURE = M3_SPEC_CAPTURE + ('shell_rmse', 'zero_shell', 'shell_frac')
OBJ_CAPTURE = ('lambda_l1', 'init_method', 'tracking_mode', 'real_estimator', 'depth')

# ⚠셸 키 규약이 두 팔에서 **다르다** (런 0회 소스 확인):
#   spec  `run_s16.py:55-56  SHELL_KEYS = ['obs:%d_%s' ...]`      → `obs:0_5` (접두어 있음)
#   ship  `wavesplat/run.py:426  _key = f'{int(_a)}_{...}'`        → `0_5`     (접두어 없음)
#   경계는 동일(`EDGES` 두 곳이 같은 리스트). ⟹ 비교 전 `SHELL_MAP` 으로 맞춘다.
SHELL_BARE = ('0_5', '5_10', '10_20', '20_30', '30_50', '50_75', '75_100',
              '100_150', '150_inf')
SHELL_MAP = {b: 'obs:' + b for b in SHELL_BARE}     # ship 키 → spec 키
# ⛔`shell_frac` 은 **두 팔에서 정의가 다르다**: ship = `frame_results[0]` 첫 프레임 하나
#   (`run.py:601`), spec = 채점창 평균(`run_s16.py:270 _sm(fr)`). ⟹ 팔 간 비교 금지,
#   팔별로만 보고한다. ⛔`run_s16.median_arm:172-176` 은 접두어를 벗겨 조회하므로
#   셸에 대해 **항상 nan** 을 낸다(잠복 결함) — 이 배치는 그 함수를 쓰지 않는다.

# ═════════════════════════════════════════════════════════════════════════════
# 3. ★주장 가족과 문턱 — ⛔M3 와 **동일**. 사후 완화가 산술적으로 불가능하다.
# ═════════════════════════════════════════════════════════════════════════════
# n=16 단측 부호검정. N_FAMILY=3 이든 2 든 문턱은 13/16 으로 **같다**
# (p(13)=0.010635 ≤ 0.05/3=0.016667 ≤ 0.025; p(12)=0.038406 은 둘 다 초과).
# ⟹ 주장 수를 어떻게 세도 문턱이 안 움직인다 = 내가 행사할 재량이 0.
N_PER_CELL = N_EP                    # 16
N_FAMILY = 3
FAMILY_ALPHA = 0.05 / N_FAMILY
NOMINAL, CONFIRM, WEAK = 12, 13, 4
NEG_MIN = 12                         # 비교자가 ZERO 를 이겨야 하는 최소
SZ_MIN = 12                          # 처치 팔이 ZERO 를 이겨야 하는 최소
TIE_POLICY = ('짝이 비트 동일하면 **승리로 세지 않는다** — `gates_b11.paired` 의 '
              '`a[k] < b[k]` 엄격부등호를 그대로 쓴다(보수적).')

# ★주장 3개. `(a, b)` = "a 가 b 보다 낮은 에피소드 수" 를 센다.
CLAIMS = {
    'C1-BASELINE-DEPTH': dict(
        a='SPEC25-RT12', b='SPEC25-RT50', in_family=True,
        q='기준선의 DEP12 붕괴 중 얼마가 **수심 오지정**이 만든 것인가?',
        clean='★단일요인. LETKF 에서 수심은 순수 런타임 인자이므로 이 대조는 '
              '**완전히 깨끗하다**(배경 스펙트럼·예보 연산자가 같은 값을 받는다).'),
    'C2-CKPT50-DEPTH': dict(
        a='CKPT50-RT12', b='CKPT50-RT50', in_family=True,
        q='사전을 d=50 에 **고정한 채** 런타임 수심만 참값으로 주면 배포 계열이 회복하는가?',
        clean='★단일요인. 조건자·체크포인트가 **두 팔에서 동일하게 d=50** 이므로 '
              '움직이는 것은 런타임 수심 하나다. ⟹ 사전이 50 에 갇혀 있다는 사실은 '
              '이 대조를 **오염시키지 않는다**(그것이 상수이기 때문).'),
    'C3-SHALLOW-COMPARE': dict(
        a='CKPT50-RT12', b='SPEC25-RT12', in_family=True,
        q='수심을 양 팔에 집행한 뒤에도 배포 계열이 기준선을 이기는가?',
        clean='⛔**비대칭 대조**: `SPEC25-RT12` 는 **완전 수리**, `CKPT50-RT12` 는 '
              '**부분 수리**(사전 d=50 잔존) ⟹ 이 비교는 **배포 계열에 불리한 쪽으로 편향**돼 '
              '있다. 그래서 읽기가 비대칭이다(§READING_C3).'),
}
CLAIM_ORDER = ('C1-BASELINE-DEPTH', 'C2-CKPT50-DEPTH', 'C3-SHALLOW-COMPARE')

# ★가족 밖 — α 를 쓰지 않는다. 보고만 한다.
SECONDARY = {
    'S1-SHELL': dict(
        kind='registered-secondary', direction_declared=True,
        text='수심은 **예보 연산자**에 들어간다(`letkf_grid.py:80` → `:98` 상태 전이; '
             'ship 은 `spectral_gaussian.py:97-101` 의 ω(k;d)). ⟹ 수심 수리의 이득은 '
             '**관측에서 먼 셸에서 커야 한다**(가까운 셸은 관측이 직접 구속). '
             '결과 전 선언: C1/C2 가 유지되면 상대 이득이 셸 거리에 대해 '
             '**바깥 절반 > 안쪽 절반** 이어야 한다. Spearman ρ(셸 index, 이득)와 '
             '안/바깥 중앙값을 보고한다. ⛔승격 금지 · ⛔`shell_frac` 팔 간 비교 금지.'),
}
EXPLORATORY = {
    'E1-COND': dict(
        a='CKPT50-RT12-C12', b='CKPT50-RT12',
        text='⚠**EXPLORATORY. 어느 방향이든 승격 금지.** 이 대조는 조건자를 분리하지 '
             '**못한다**(오지정을 맞바꾼다). ★유일하게 유익한 결과는 **널**이다: '
             '두 팔이 갈리지 않고 상대차 중앙이 작으면 *"조건자 상수는 이 수심에서 '
             '하중을 받지 않는다"* 로 잔차를 {체크포인트 사전, 얕은-물 레짐} 으로 좁힌다. '
             '비-널이면 ⛔귀속 없이 수만 보고한다.'),
}


def threshold_table(n=N_PER_CELL):
    """문턱을 **산술로 재도출**한다(손복사 0). 게이트가 매 실행 호출한다."""
    return dict(n=n,
                p_nominal=sign_test_one_sided(NOMINAL, n),
                p_confirm=sign_test_one_sided(CONFIRM, n),
                family_alpha=FAMILY_ALPHA, n_family=N_FAMILY,
                confirm_within_family=sign_test_one_sided(CONFIRM, n) <= FAMILY_ALPHA,
                nominal_outside_family=sign_test_one_sided(NOMINAL, n) > FAMILY_ALPHA,
                weak_is_complement=(WEAK == n - NOMINAL),
                power_at_0_80=_power(NOMINAL, n, 0.80),
                power_confirm_at_0_80=_power(CONFIRM, n, 0.80))


def _power(w, n, p):
    from math import comb
    return sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(w, n + 1))


# ═════════════════════════════════════════════════════════════════════════════
# 4. ★결과 전 해석 규칙 — ⛔"널이면 무엇을 쓰나" 에 전부 답이 있다.
# ═════════════════════════════════════════════════════════════════════════════
SKILL_GATE = (
    '★먼저 기술 게이트를 읽는다. `SPEC25-RT12 < ZERO` < {NEG} 이면 **비교자 무기술** ⟹ '
    '`C3` 는 **읽을 수 없다**(우열 금지). 양 팔 모두 ZERO 를 못 이기면 '
    '`STILL-NO-SKILL` ⟹ `C1`·`C2` 의 개선은 ***무기술 영역 안에서의 개선***이고 '
    '⛔**기술로 읽지 않는다**.').format(NEG=NEG_MIN)

CLAIM_LABELS = ('HOLDS (CONFIRMATORY)', 'HOLDS (NOMINAL)', 'UNDECIDED', 'REVERSES')

READING = {
    'C1-BASELINE-DEPTH': {
        'HOLDS (CONFIRMATORY)':
            '★기준선의 얕은-물 붕괴는 **수심 오지정이 상당 부분 만들었다**. ⟹ '
            '`AMENDMENT_2` 의 `TREATMENT-NOT-DELIVERED` 가 **추론에서 측정이 된다**. '
            '⛔단 이것은 *장치* 에 대한 진술이고, 여전히 *레짐* 에 대한 진술이 아니다.',
        'HOLDS (NOMINAL)': '⚠명목만 ⟹ `EXPLORATORY`. 위 문장을 가족단위 미달 병기와 함께.',
        'UNDECIDED':
            '★***수심을 실제로 집행해도 기준선은 n=16 으로 갈리지 않았다*** ⟹ '
            'M3 `DEP12` 의 기준선 무기술은 **수심 오지정 하나로 설명되지 않는다**. '
            '이것은 `AMENDMENT_2` §5.2–5.3 의 사후 용량-반응을 **지지하지 않는 쪽**이고, '
            '그 분석을 승격하지 않은 결정이 옳았음을 확인한다. MDE 공시 의무. '
            '⛔결과 보고 n 증축 금지 — 새 사전등록으로만.',
        'REVERSES':
            '★참 수심을 주면 기준선이 **더 나빠진다**. 먼저 `PC-D`·`G-2`·`G-3`(코퍼스가 '
            'd=12 분산관계를 만족하는지)을 재확인한다. 통과하면 보고문은 '
            '*"12 m 배경/국소화 조합이 이 필터에 불리하다"* 이고, ⚠**`F2` 부류 공시 의무**: '
            'λ_p 76.5→63.6 m 로 `loc_radius=25` 대비 loc/λ_p 0.327→0.393 이 함께 움직인다 '
            '(⛔`loc_radius` 재조율은 **M2 정지규칙**에 걸린다 — 새 사전등록으로만).'},
    'C2-CKPT50-DEPTH': {
        'HOLDS (CONFIRMATORY)':
            '★사전을 d=50 에 둔 채 **런타임 물리만 고쳐도** 배포 계열이 회복한다 ⟹ '
            '`load_episode` 수리는 얕은 물에서 **실효가 있다**. ⛔단 *"사전이 지지집합 밖에서 '
            '번다"* 로 읽지 마라 — 사전은 이 배치에서 **한 번도 지지집합을 떠나지 않았다**.',
        'HOLDS (NOMINAL)': '⚠명목만 ⟹ `EXPLORATORY`.',
        'UNDECIDED':
            '★***런타임 수심만으로는 배포 계열이 갈리지 않는다.*** 잔차의 후보는 '
            '{사전 지지집합, 조건자 상수, 얕은-물 레짐} 이고 **이 배치는 셋을 가르지 못한다**. '
            '`E1` 이 조건자에 대해서만 느슨한 경계를 준다. MDE 공시 의무.',
        'REVERSES':
            '★★가장 정보량이 큰 결과: ***정합성이 정확성보다 중요하다.*** d=50 사전이 낸 '
            '원자를 d=12 물리로 전개하면 **둘 다 50 일 때보다 나쁘다**. ⟹ 처방이 구체화된다 '
            '— *"`load_episode` 만 고치면 안 되고 **수심을 사전까지 인자화**해야 한다"* '
            '(= `B-23` 의 구체형). ⟹ ★**배포 경고**: 수리를 단독 배포할 때 얕은 물에 대한 '
            '주의 문구가 **필수**가 된다(§FIX_POLICY 의 caveat 분기).'},
    'C3-SHALLOW-COMPARE': {
        'HOLDS (CONFIRMATORY)':
            '★**보수적으로 읽을 수 있다** — 배포 계열은 부분 수리만 받고도 완전 수리된 '
            '기준선을 이겼다. 허용 문장: *"얕은 물(12 m) 코퍼스에서, 수심을 양 팔에 '
            '집행한 뒤에도 배포 계열이 `SPEC25` 를 이긴다."* '
            '⛔**금지**: *"사전이 수심 지지집합 밖에서 번다"* · *"경계는 수심 축에 없다"* · '
            '⛔크기를 `B-1`/`B-12`/`B-16`/`B-26`/`M1` 과 직접 비교 금지.',
        'HOLDS (NOMINAL)': '⚠명목만 ⟹ `EXPLORATORY`. 위 허용 문장 + 미달 병기.',
        'UNDECIDED':
            '⛔**해석 불가**(결과 전 선언). 비대칭 수리 때문에 *"사전이 진다"* 로 읽을 수 '
            '없다 — 배포 계열만 부분 수리를 받았기 때문이다. 수만 보고한다.',
        'REVERSES':
            '⛔**해석 불가**(결과 전 선언). 위와 같은 사유. ⛔*"기준선이 낫다"* 금지 · '
            '⛔*"사전이 지지집합 밖에서 진다"* 금지. 수만 보고하고, 가르는 후속은 '
            '**사전의 수심 인자화**(`B-23`)뿐임을 적는다.'},
}

# ★C1×C2 분해 — 라벨 두 개만 읽는다(추가 검정 0개; 08-23 층간대비 교훈 계승).
DECOMPOSITION = {
    ('hold', 'hold'):
        '★`DEPTH-MISSPEC-CARRIED-BOTH` — 두 팔 모두에서 수심 오지정이 하중을 졌다. '
        'M3 `DEP12` VOID 의 사유가 측정으로 확인된다. 남은 지지집합 질문은 **미측정**이고 '
        '`B-23`(수심 인자화)이 유일한 경로다.',
    ('hold', 'undecided'):
        '★`DEPTH-MISSPEC-BASELINE-ONLY` — 기준선만 회복 ⟹ 배포 계열의 붕괴는 수심 '
        '오지정으로 설명되지 않는다. 후보 {사전 지지집합, 조건자, 레짐} 중 어느 것도 '
        '**지목되지 않는다**(배제만: 런타임 물리는 아니다).',
    ('hold', 'reverses'):
        '★★`CONSISTENCY-OVER-CORRECTNESS` — 기준선은 참 수심으로 회복하는데 배포 계열은 '
        '악화한다 ⟹ 사전과 물리의 **불일치**가 원인이다. `B-23` 의 형태가 '
        '*"재학습"* 이 아니라 **"수심 조건화"** 로 좁혀진다.',
    ('undecided', 'hold'):
        '★`DEPTH-MISSPEC-CKPT50-ONLY` — 배포 계열만 회복 ⟹ 기준선의 얕은-물 붕괴는 '
        '수심 밖 원인. ⚠**loc/λ_p 공시 의무**(위 `C1 REVERSES` 항 참조).',
    ('undecided', 'undecided'):
        '★`DEPTH-NOT-THE-CARRIER` — ***수심을 실제로 집행해도 어느 팔도 갈리지 않는다.*** '
        'M3 `DEP12` 의 무기술은 수심 오지정 하나로 설명되지 않는다. ⛔단 M3 의 VOID 판정은 '
        '**불변**이다 — VOID 는 *결과* 가 아니라 *집행 실패* 에 대한 진술이라 이 배치의 '
        '어떤 결과로도 되돌아가지 않는다(§VOID_IS_INDEPENDENT).',
    ('reverses', 'reverses'):
        '⚠`DELIVERY-HURTS-BOTH` — 참 수심이 양 팔을 다 해친다. 코퍼스/채점 경보. '
        '`PC-D`·`G-2`·`G-3` 재확인 후에도 남으면 그대로 보고하고 ⛔귀속하지 않는다.',
}

VOID_IS_INDEPENDENT = (
    '★★M3 `DEP12` 의 `UNMEASURABLE (TREATMENT-NOT-DELIVERED)` 는 **이 배치의 결과와 '
    '논리적으로 독립**이다. 그 판정은 *"96런이 d=12 를 받지 않았다"* 는 소스·되읽기 사실에 '
    '근거하고, M3b 가 무엇을 재든 그 사실은 바뀌지 않는다. ⛔따라서 M3b 의 어떤 결과도 '
    'M3 의 `DEP12` 수치(5/16 · +4.63% · 음성대조 6/16)를 인용 가능하게 만들지 않는다.')

# ★`load_episode` 패키지 수리 정책 — **결과와 무관하게 결과 전에 확정**한다.
FIX_POLICY = (
    '★`wavesim/data_writer.load_episode` 에 `data["depth"] = f["config"].attrs["depth"]` 를 '
    '**추가한다**. 근거 둘 다 결과 독립이다: (a) 파일에 있는 값을 읽지 않는 것은 결함이다 '
    '(b) `PC-A` 가 비트불변을 **증명**한다(저장소 1,496개 episode h5 전건이 `depth` attr 을 '
    '가지고, 그 중 1,480개가 정확히 `50.0` = 폴백과 같은 값). '
    '⟹ 오늘 존재하는 어떤 코퍼스에서도 거동이 바뀌지 않는다. '
    '★결과가 정하는 것은 **수리 여부가 아니라 동반 문구**뿐이다: `C2` 가 `REVERSES` 면 '
    '*"얕은 물은 사전이 인자화될 때까지 미지원"* 이라는 배포 주의를 함께 넣는다.')

RUNPY_POLICY = (
    '★`wavesplat/run.py:216 cfg["depth"] = loader.depth` 는 **고치지 않는다**(결과 전 확정). '
    '① M3b 가 필요로 하지 않는다 — RT 인자는 상류 `load_episode` 한 곳에서 전달되고, 양 팔이 '
    '모두 `EpisodeDataLoader` 를 통과하므로 **비대칭이 구조적으로 불가능**하다. '
    '`run.py:216` 을 호출자 우선으로 바꾸면 ship 팔에만 있는 노브가 되어 정확히 '
    '`AMENDMENT_2` §4 가 경고한 비대칭을 **새로 만든다**. '
    '② 배포 거동 변경이며 측정 필요가 0 이다(저장소 규칙: 배포 변경은 자기 사전등록 + '
    '배포-후 게이트 출력을 요구한다). '
    '③ `load_episode` 수리 후에는 `loader.depth` 가 **참값**이므로 이 덮어쓰기는 결함이 '
    '아니라 **우선순위 정책**이 된다: *"에피소드의 참값이 호출자 의도를 이긴다"*. '
    '⚠⛔*"죽은 키"* 는 정확하지 않다 — **경로 의존 키**다: 실측 경로(`SequentialUpdater` 를 '
    '직접 구성해 `load_episode` 를 타지 않는 `tmp/B2_REAL_2026-08-23/scene_station.py` 등)에서 '
    '`cfg["depth"]` 는 **살아 있다**. ⟹ 처방은 (i) `run.py:216` 에 정책 주석 (ii) 회귀 테스트 '
    '`test_depth_precedence`(합성 h5 에서 호출자 `config={"depth": X}` 가 무시되고 에피소드 '
    '참값이 이긴다 — 현 거동을 못박는다) (iii) 매 런 `PC-C` 되읽기.')

# ★스모크 규약 — M3 의 `G-17` 이 288런 전에 발화 불가였던 결함의 일반 처방.
SMOKE_RULE = (
    '★***스모크는 게이트가 발화할 수 있는 (셀 × 팔) 교차곱을 덮어야 하고, 게이트를 '
    '스모크에서 끄지 말고 기대 개수만 매개변수화해야 한다.*** '
    'M3 의 실패는 두 겹이었다: (a) `run_m3.episodes(smoke=True)` 가 `PRIMARY_CELL` **1셀**만 '
    '골라 `DEP12` 를 건드리지 않았고 (b) `run_m3.main()` 이 `if not smoke:` 로 '
    '`gates_data()` 를 **통째로 건너뛰었다** ⟹ `G-17` 은 288런이 끝나기 전에는 '
    '**구조적으로 발화 불가**였다. M3b 는 `SMOKE_PLAN` = 전 셀 × **전 팔** × 시드 1 이고, '
    '`gates_run(rows, expect=...)` 로 행수만 매개변수화한다. `G-SMOKE-COVER` 가 이를 '
    '기계로 강제한다(⛔산문 아님).')
SMOKE_PLAN = dict(cells=(CELL,), arms=ARM_ORDER, n_episodes=1, n_seeds=1)


def claim_label(pr):
    """주장 하나의 라벨. `pr` = `paired(a, b)`."""
    w, n = pr['better'], pr['n']
    if n == 0:
        return ('UNDECIDED', '0 짝')
    if w >= CONFIRM:
        return ('HOLDS (CONFIRMATORY)', f'{w}/{n} p={pr["p_one_sided"]:.6f}')
    if w >= NOMINAL:
        return ('HOLDS (NOMINAL)', f'{w}/{n} p={pr["p_one_sided"]:.6f} — 가족단위 미달')
    if w <= WEAK:
        return ('REVERSES', f'{w}/{n} — 부호가 반대로 간다')
    return ('UNDECIDED', f'{w}/{n} — 부호로 갈리지 않는다')


def decompose(labels):
    def st_(c):
        L = labels[c]
        return ('hold' if L.startswith('HOLDS')
                else 'reverses' if L == 'REVERSES' else 'undecided')
    key = (st_('C1-BASELINE-DEPTH'), st_('C2-CKPT50-DEPTH'))
    return key, DECOMPOSITION.get(
        key, '조합이 표에 없다 ⟹ 단일 축 귀속 금지, 두 라벨을 그대로 보고한다.')


def aggregate(rows):
    ok = [r for r in rows if 'error' not in r]
    med = {a: med_by_episode(ok, a) for a in ARM_ORDER}
    zero = med_by_episode(ok, ZERO_FROM, field='zero_nRMSE')
    zero_rt12 = med_by_episode(ok, 'SPEC25-RT12', field='zero_nRMSE')
    out = dict(med=med, zero=zero, zero_rt12=zero_rt12, claims={})
    for c in CLAIM_ORDER:
        d = CLAIMS[c]
        out['claims'][c] = paired(med[d['a']], med[d['b']])
    out['neg_rt12'] = paired(med['SPEC25-RT12'], zero)
    out['neg_rt50'] = paired(med['SPEC25-RT50'], zero)
    out['sz_rt12'] = paired(med['CKPT50-RT12'], zero)
    out['sz_rt50'] = paired(med['CKPT50-RT50'], zero)
    out['e1'] = paired(med['CKPT50-RT12-C12'], med['CKPT50-RT12'])
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 5. 게이트
# ═════════════════════════════════════════════════════════════════════════════
def _src(rel):
    with open(os.path.join(ROOT, rel), encoding='utf-8') as fh:
        return fh.read()


def defect_facts():
    """★결함을 **소스에서 재도출**한다. `RT50` 팔이 '오늘의 배포' 라는 전제가 매 실행 검증된다."""
    import re
    dw = _src('wavesim/data_writer.py')
    body = dw[dw.index('def load_episode('):]
    body = body[:body.index('\n\ndef ') if '\n\ndef ' in body else len(body)]
    dl = _src('wavesplat/data_loader.py')
    rp = _src('wavesplat/run.py')
    bn = _src('scripts/d3_beamform_net.py')
    m = re.search(r"self\.depth\s*=\s*float\(self\.data\.get\('depth',\s*([0-9.]+)\)\)", dl)
    mb = re.search(r'^DEPTH\s*=\s*([0-9.]+)', bn, re.M) or re.search(r'DEPTH\s*=\s*([0-9.]+)', bn)
    return dict(
        load_episode_reads_depth=("attrs['depth']" in body or 'attrs["depth"]' in body),
        loader_fallback=float(m.group(1)) if m else None,
        runpy_overwrites=("cfg['depth'] = loader.depth" in rp),
        cond_depth=float(mb.group(1)) if mb else None)


def gates_runfree():
    out = []
    f = defect_facts()
    out.append(('★G-A 결함 재도출: `load_episode` 가 depth 를 **아직 안 읽는다** '
                '(= `RT50` 팔이 오늘의 배포다)',
                f['load_episode_reads_depth'] is False,
                f"reads_depth={f['load_episode_reads_depth']}"))
    out.append(('★G-B 폴백 상수 = 50.0 (`data_loader.py:46`)',
                f['loader_fallback'] == FALLBACK_DEPTH, str(f['loader_fallback'])))
    out.append(('G-C `run.py:216` 무조건 덮어쓰기 존재 (정책 = 무변, RUNPY_POLICY)',
                f['runpy_overwrites'] is True, str(f['runpy_overwrites'])))
    out.append(('★G-D 조건자 상수 = 50.0 (`d3_beamform_net.py:30`) — 팔 이름 `CKPT50` 의 근거',
                f['cond_depth'] == 50.0, str(f['cond_depth'])))
    t = threshold_table()
    out.append(('★G-E 문턱 산술 (M3 와 동일 · 사후 완화 불가)',
                bool(t['confirm_within_family'] and t['nominal_outside_family']
                     and t['weak_is_complement']),
                f"NOMINAL {NOMINAL} p={t['p_nominal']:.5f} · CONFIRM {CONFIRM} "
                f"p={t['p_confirm']:.5f} ≤ {FAMILY_ALPHA:.5f} · 검정력@0.80 "
                f"{t['power_at_0_80']:.3f}"))
    dc = deploy_constants()
    ok = all(dc[f'DEFAULT_CONFIG.{k}'] == v for k, v in DEPLOY_EXPECTED.items())
    out.append(('★G-F 배포 상수 = 단일출처 = 사전등록값', ok, str(dc)))
    out.append(('★G-G 신규 시드 소모 0 · 신규 생성 0', SEEDS_CONSUMED == (), 'SEEDS_CONSUMED=()'))
    out.append(('★G-H 팔 정의에 빈 dict 0개 · ship 팔 이름은 전부 `CKPT50` 접두',
                all(bool(v) for v in ARMS.values())
                and all(a.startswith('CKPT50') for a in SHIP_ARMS)
                and not any('SHIP' in a for a in ARMS), str(ARM_ORDER)))
    out.append(('★G-SMOKE-COVER 스모크 계획이 전 팔·전 셀을 덮는다 (M3 `G-17` 결함의 처방)',
                tuple(SMOKE_PLAN['arms']) == ARM_ORDER and SMOKE_PLAN['cells'] == (CELL,),
                str(SMOKE_PLAN)))
    return out


def gates_data(read=None, exists=None):
    read = read or _read_ep
    exists = exists or os.path.exists
    out = []
    eps = episodes()
    have = [e for e in eps if exists(e['path'])]
    out.append(('G-1 동결 코퍼스 16 에피소드 존재 (M3 재사용)',
                len(have) == N_EP, f'{len(have)}/{N_EP}'))
    if len(have) != N_EP:
        return out
    A = [read(e['path']) for e in have]
    out.append(('★G-2 attrs depth 되읽기 = 12.0 전건 (⛔파일명 아님)',
                all(abs(a['depth'] - TRUE_DEPTH) < 1e-9 for a in A),
                str(sorted({round(a['depth'], 6) for a in A}))))
    D = [derived_coords(a) for a in A]
    out.append(('★G-3 분산관계 파생측정: 참 수심 잔차 ≤ 1e-6 전건 (코퍼스가 진짜 d=12)',
                all(d['disp_resid_true'] <= DISP_TRUE_MAX for d in D),
                f"max {max(d['disp_resid_true'] for d in D):.2e}"))
    out.append(('★G-4 d=50 가정 잔차 ≥ 0.10 전건 (오지정이 실재 = 처치에 지렛대가 있다)',
                all(d['disp_resid_50'] >= DISP_50_MIN for d in D),
                f"min {min(d['disp_resid_50'] for d in D):.3f}"))
    return out


def _by(rows, arm):
    return [r for r in rows if 'error' not in r and r.get('arm') == arm]


def gates_run(rows, expect=None):
    """★`expect` 는 **행수만** 매개변수화한다 — 나머지 게이트는 스모크에서도 동일하게 돈다."""
    expect = expect or {a: N_EP * len(ARMS[a]['seeds']) for a in ARM_ORDER}
    out = [('G-5 에러 0', all('error' not in r for r in rows),
            f"{sum(1 for r in rows if 'error' in r)}")]
    # ⚠게이트 id 는 **팔마다 유일**해야 한다 — 같은 이름 다섯 개면 자가진단의 dict 조회가
    #   마지막 것만 남겨 **누락을 놓친다**(초판이 실제로 그랬고 T20 이 잡았다).
    for i, a in enumerate(ARM_ORDER):
        out.append((f'G-6{"abcde"[i]} {a} 행수', len(_by(rows, a)) == expect[a],
                    f'{len(_by(rows, a))}/{expect[a]}'))
    sets = {a: {r['episode'] for r in _by(rows, a)} for a in ARM_ORDER}
    out.append(('★G-7 다섯 팔의 에피소드 집합 완전 일치',
                len({frozenset(s) for s in sets.values()}) == 1,
                str({a: len(s) for a, s in sets.items()})))
    # ★★PC-C — M3 `G-17` 의 대칭 수리: **두 팔 모두** 실효 수심을 되읽는다.
    bad = []
    for a in ARM_ORDER:
        want = TRUE_DEPTH if ARMS[a]['rt'] == RT12 else FALLBACK_DEPTH
        for r in _by(rows, a):
            got = r.get('obj_depth') if ARMS[a]['family'] == 'ship' else r.get('cap_depth')
            if got is None or abs(float(got) - want) > 1e-9:
                bad.append((a, r['episode'], got, want))
    out.append(('★★G-8 (PC-C) 실효 수심 되읽기 == 팔의 선언 RT — **양 팔 전건** '
                '(M3 `G-17` 이 기준선에만 걸렸던 결함의 대칭 수리)',
                not bad, f'위반 {len(bad)}' + (f' 예: {bad[:2]}' if bad else '')))
    # ★PC-D — ZERO 는 수심 무관이어야 한다 (shim 이 채점에 새지 않았다는 증명)
    z50 = med_by_episode(rows, 'SPEC25-RT50', field='zero_nRMSE')
    z12 = med_by_episode(rows, 'SPEC25-RT12', field='zero_nRMSE')
    ks = sorted(set(z50) & set(z12))
    out.append(('★G-9 (PC-D) `zero_nRMSE` 가 RT50/RT12 에서 **비트 동일** '
                '(RT shim 이 채점 대상에 새지 않았다)',
                bool(ks) and all(z50[k] == z12[k] for k in ks),
                f'{sum(1 for k in ks if z50[k] == z12[k])}/{len(ks)}'))
    # 라우팅 — 배포 경로가 실제로 학습 팔로 갔는가 (ITT)
    rt = [r for a in SHIP_ARMS for r in _by(rows, a)]
    out.append(('G-10 라우터가 전건 learned (ITT)',
                bool(rt) and all(r.get('cap_auto_selected') == ROUTE_EXPECTED for r in rt),
                str(sorted({r.get('cap_auto_selected') for r in rt}))))
    out.append(('G-11 varpro 전건 발화 · steps = 배포값',
                bool(rt) and all(r.get('cap_varpro_fired') is True
                                 and r.get('cap_varpro_steps') == DEPLOY_EXPECTED['varpro_steps']
                                 for r in rt),
                str(sorted({(r.get('cap_varpro_fired'), r.get('cap_varpro_steps'))
                            for r in rt}))))
    # ★셸 — "계산되는데 저장 안 됨" 여섯 번째의 수리가 실제로 행에 도달했는가
    sp = [r for a in SPEC_ARMS for r in _by(rows, a)]
    out.append(('★G-12 셸이 행에 실렸다 — spec `shell_rmse` 9키 · ship `avg_shell_rmse` 9키',
                bool(sp) and bool(rt)
                and all(set((r.get('cap_shell_rmse') or {})) == set(SHELL_MAP.values())
                        for r in sp)
                and all(set((r.get('cap_avg_shell_rmse') or {})) == set(SHELL_BARE)
                        for r in rt),
                f'spec {len(sp)} · ship {len(rt)}'))
    # 기준선 되읽기 (M3 G-17 의 (a)(b) 부분 계승)
    out.append(('G-13 `SPEC25` 되읽기: loc 25 · ens 600 · jonswap 전건',
                bool(sp) and all(r.get('cap_loc_radius') == SPEC['loc']
                                 and r.get('cap_n_ensemble') == SPEC['n_ensemble']
                                 and r.get('cap_background') == SPEC['background']
                                 for r in sp),
                str(sorted({(r.get('cap_loc_radius'), r.get('cap_n_ensemble'),
                             r.get('cap_background')) for r in sp}))))
    # ★조건자 팔이 실제로 재바인딩됐는가 (⛔의도표 금지 — 러너가 OM_G 를 되읽어 행에 싣는다)
    c12 = _by(rows, 'CKPT50-RT12-C12')
    other = [r for a in ('CKPT50-RT50', 'CKPT50-RT12') for r in _by(rows, a)]
    out.append(('★G-14 조건자 되읽기: C12 팔만 `cap_cond_depth`==12.0, 나머지 ship 팔은 50.0',
                bool(c12) and bool(other)
                and all(r.get('cap_cond_depth') == 12.0 for r in c12)
                and all(r.get('cap_cond_depth') == 50.0 for r in other),
                f"C12 {sorted({r.get('cap_cond_depth') for r in c12})} · "
                f"기타 {sorted({r.get('cap_cond_depth') for r in other})}"))
    return out


def gates_pca(pca):
    """PC-A: `load_episode` 수리가 **비트불변**인가 (참 수심 50 코퍼스). FAIL ⟹ BATCH-VOID."""
    rows = (pca or {}).get('rows') or []
    n_exp = 2 * len(PCA_EPISODES) * 2      # 팔 2 × 에피소드 2 × {shim off, on}
    pairs = {}
    for r in rows:
        pairs.setdefault((r['arm'], r['episode']), {})[r['shim']] = r['avg_nRMSE']
    eq = [k for k, v in pairs.items() if len(v) == 2 and list(v.values())[0] == list(v.values())[1]]
    return [('★★G-PCA (PC-A) `load_episode` 수리가 비트불변 — shim off/on 이 '
             '`avg_nRMSE` 비트 동일 (⛔FAIL ⟹ BATCH-VOID)',
             len(rows) == n_exp and len(eq) == len(pairs) and bool(pairs),
             f'{len(eq)}/{len(pairs)} 짝 비트동일 · {len(rows)}/{n_exp} 행'),
            ('G-PCA2 (PC-A) shim 이 참 수심 50 을 실제로 주입했다 (되읽기)',
             bool(rows) and all(r.get('depth_readback') == 50.0 for r in rows),
             str(sorted({r.get('depth_readback') for r in rows})))]


def gates_pcb(pcb):
    """PC-B: `RT50` 팔이 동결 M3 DEP12 행을 재현하는가.
    ⛔**FAIL 은 BATCH-VOID 가 아니다** — 결과 전 선언(§READING_PCB)."""
    rows = (pcb or {}).get('rows') or []
    fz = (pcb or {}).get('frozen') or {}
    n = sum(1 for r in rows if fz.get(r['key']) == r['avg_nRMSE'])
    return [('★G-PCB (PC-B) `RT50` 96행이 동결 M3 DEP12 행과 비트 동일 '
             '(⚠FAIL 은 VOID 아님 — READING_PCB)',
             bool(rows) and n == len(rows) and len(rows) == len(fz),
             f'{n}/{len(rows)} 비트동일 · 동결 {len(fz)}행')]


READING_PCB = (
    '★`PC-B` 의 판정은 **결과 전에** 정한다. **PASS** ⟹ 이 나무가 M3 를 재현한다 — 그리고 '
    '★`AMENDMENT_1` 이 `sequential_update.py` 드리프트에 대해 *"영향 = 구조적 도달 불가"* 를 '
    '**정적 논증으로만** 세우고 스스로 *"실행 대조가 아니다"* 라 적어 둔 구멍이 **닫힌다**. '
    '**FAIL** ⟹ (a) ⛔그 정적 논증이 **반증**된다 — 즉시 공시 (b) M3 의 DEP12 수치는 '
    '재현 불가로 표시된다 (c) ★그러나 **M3b 의 내부 짝 비교는 무손상**이다: RT50·RT12 가 '
    '**같은 나무·같은 세션**에서 돌았기 때문이다 ⟹ `C1`·`C2`·`C3` 는 그대로 읽는다. '
    '⛔FAIL 을 본 뒤 문턱·팔·읽기를 바꾸지 않는다.')


# ═════════════════════════════════════════════════════════════════════════════
# 6. 자가진단 — ★게이트가 **실제로 발화하는지** 변형으로 확인한다.
# ═════════════════════════════════════════════════════════════════════════════
def _rows(win_a=16, arm_a='SPEC25-RT12', arm_b='SPEC25-RT50'):
    """합성 행. 기본은 전 게이트 PASS."""
    out = []
    for i, e in enumerate(episodes()):
        for a in ARM_ORDER:
            d = ARMS[a]
            want = TRUE_DEPTH if d['rt'] == RT12 else FALLBACK_DEPTH
            base = 0.30 if a in (arm_b,) else 0.20
            if a == arm_a and i >= win_a:
                base = 0.40
            for s in d['seeds']:
                r = dict(arm=a, episode=e['episode'], cell=CELL, avg_nRMSE=base,
                         zero_nRMSE=0.25, cap_zero_nRMSE=0.25,
                         cap_auto_selected=ROUTE_EXPECTED, cap_varpro_fired=True,
                         cap_varpro_steps=DEPLOY_EXPECTED['varpro_steps'],
                         cap_loc_radius=SPEC['loc'], cap_n_ensemble=SPEC['n_ensemble'],
                         cap_background=SPEC['background'])
                r[d['seed_field']] = s
                if d['family'] == 'ship':
                    r['obj_depth'] = want
                    r['cap_cond_depth'] = d['cond_depth']
                    r['cap_avg_shell_rmse'] = {k: 0.1 for k in SHELL_BARE}
                    r['cap_shell_frac'] = {k: 0.1 for k in SHELL_BARE}
                else:
                    r['cap_depth'] = want
                    r['cap_shell_rmse'] = {v: 0.1 for v in SHELL_MAP.values()}
                    r['cap_zero_shell'] = {v: 0.1 for v in SHELL_MAP.values()}
                    r['cap_shell_frac'] = {v: 0.1 for v in SHELL_MAP.values()}
                out.append(r)
    return out


def gate_id(name):
    return name.split()[0].lstrip('★⛔⚠')


def selftest(verbose=True):
    """25항 변형 하네스. ⛔러너는 이것이 통과하지 않으면 추정을 돌리지 않는다."""
    fails = []

    def chk(tag, cond):
        if not cond:
            fails.append(tag)
        if verbose:
            print(f'  [{"ok" if cond else "XX"}] {tag}')

    # (1) 문턱 산술
    t = threshold_table()
    chk('T1 CONFIRM p ≤ 가족 α', t['p_confirm'] <= FAMILY_ALPHA)
    chk('T2 NOMINAL p > 가족 α', t['p_nominal'] > FAMILY_ALPHA)
    chk('T3 WEAK = n − NOMINAL', WEAK == N_PER_CELL - NOMINAL)
    chk('T4 문턱이 N_FAMILY 2/3 어디서도 같다',
        sign_test_one_sided(CONFIRM, N_PER_CELL) <= 0.025
        and sign_test_one_sided(NOMINAL, N_PER_CELL) > 0.025)
    chk('T5 검정력@0.80 ≈ 0.80', 0.75 <= t['power_at_0_80'] <= 0.85)
    # (2) 라벨
    chk('T6 13/16 ⟹ CONFIRMATORY',
        claim_label(dict(better=13, n=16, p_one_sided=0.0106))[0] == 'HOLDS (CONFIRMATORY)')
    chk('T7 12/16 ⟹ NOMINAL',
        claim_label(dict(better=12, n=16, p_one_sided=0.0384))[0] == 'HOLDS (NOMINAL)')
    chk('T8 4/16 ⟹ REVERSES',
        claim_label(dict(better=4, n=16, p_one_sided=1.0))[0] == 'REVERSES')
    chk('T9 8/16 ⟹ UNDECIDED',
        claim_label(dict(better=8, n=16, p_one_sided=0.6))[0] == 'UNDECIDED')
    chk('T10 READING 이 네 라벨 전부를 세 주장에 대해 갖는다',
        all(set(READING[c]) == set(CLAIM_LABELS) for c in CLAIM_ORDER))
    chk('T11 DECOMPOSITION 키가 전부 유효 상태쌍',
        all(set(k) <= {'hold', 'undecided', 'reverses'} for k in DECOMPOSITION))
    # (3) 팔 정의
    chk('T12 빈 dict 팔 0개', all(bool(v) for v in ARMS.values()))
    chk('T13 ship 팔 전부 CKPT50 접두 · 이름에 SHIP 0회',
        all(a.startswith('CKPT50') for a in SHIP_ARMS) and not any('SHIP' in a for a in ARMS))
    chk('T14 RT 라벨이 두 값뿐', {d['rt'] for d in ARMS.values()} == {RT50, RT12})
    chk('T15 ckpt_depth 는 ship 전 팔에서 50 (= 환원 불가 한계)',
        {ARMS[a]['ckpt_depth'] for a in SHIP_ARMS} == {50.0})
    # (4) 게이트가 발화하는가 — 변형 주입
    good = _rows()
    gr = dict((gate_id(n_), q_) for n_, q_, _ in gates_run(good))
    chk('T16 정상 행에서 gates_run 전건 PASS', all(gr.values()))
    bad = _rows()
    for r in bad:
        if r['arm'] == 'SPEC25-RT12':
            r['cap_depth'] = FALLBACK_DEPTH             # ← M3 가 당한 바로 그 결함
    chk('T17 ★수심 미전달을 G-8 이 잡는다 (M3 결함 재현)',
        not dict((gate_id(n_), q_) for n_, q_, _ in gates_run(bad))['G-8'])
    bad = _rows()
    for r in bad:
        if r['arm'] == 'CKPT50-RT12':
            r['obj_depth'] = FALLBACK_DEPTH             # ← ship 쪽 비대칭 (M3 가 못 잡던 것)
    chk('T18 ★★ship 쪽 미전달도 G-8 이 잡는다 (M3 `G-17` 의 사각지대)',
        not dict((gate_id(n_), q_) for n_, q_, _ in gates_run(bad))['G-8'])
    bad = _rows()
    for r in bad:
        if r['arm'] == 'SPEC25-RT12':
            r['zero_nRMSE'] = 0.2500001                 # ← shim 이 채점에 샜다
    chk('T19 ZERO 누출을 G-9 가 잡는다',
        not dict((gate_id(n_), q_) for n_, q_, _ in gates_run(bad))['G-9'])
    bad = [r for r in _rows() if r['arm'] != 'CKPT50-RT12-C12']
    gb = dict((gate_id(n_), q_) for n_, q_, _ in gates_run(bad))
    chk('T20 팔 누락을 G-6c/G-7 이 잡는다', (not gb['G-6c']) and (not gb['G-7']))
    chk('T20b 게이트 id 가 전부 유일 (같은 이름이면 누락을 놓친다)',
        len({gate_id(n_) for n_, _, _ in gates_run(good)}) == len(gates_run(good)))
    bad = _rows()
    for r in bad:
        if r['arm'] == 'CKPT50-RT12-C12':
            r['cap_cond_depth'] = 50.0                  # ← 재바인딩이 안 먹었다
    chk('T21 조건자 미개입을 G-14 가 잡는다',
        not dict((gate_id(n_), q_) for n_, q_, _ in gates_run(bad))['G-14'])
    bad = _rows()
    for r in bad:
        r.pop('cap_shell_rmse', None)
        r.pop('cap_avg_shell_rmse', None)
    chk('T22 셸 유실을 G-12 가 잡는다',
        not dict((gate_id(n_), q_) for n_, q_, _ in gates_run(bad))['G-12'])
    # (5) 집계
    agg = aggregate(_rows(win_a=16))
    chk('T23 처치가 전승이면 C1 = 16/16', agg['claims']['C1-BASELINE-DEPTH']['better'] == 16)
    agg = aggregate(_rows(win_a=0))
    chk('T24 처치가 전패면 C1 = 0/16', agg['claims']['C1-BASELINE-DEPTH']['better'] == 0)
    chk('T25 동점은 승리로 세지 않는다 (TIE_POLICY)',
        paired({'e': 0.2}, {'e': 0.2})['better'] == 0)
    # (6) PC 게이트
    chk('T26 PC-A 가 불일치를 잡는다',
        not gates_pca(dict(rows=[
            dict(arm='X', episode='e', shim='off', avg_nRMSE=0.1, depth_readback=50.0),
            dict(arm='X', episode='e', shim='on', avg_nRMSE=0.2, depth_readback=50.0)]))[0][1])
    chk('T27 PC-A 가 0행에서 PASS 하지 않는다', not gates_pca({})[0][1])
    chk('T28 PC-B 가 0행에서 PASS 하지 않는다', not gates_pcb({})[0][1])
    # (7) 런프리 게이트가 저장소 현 상태에서 참인가
    for n_, q_, d_ in gates_runfree():
        chk(f'T29 runfree {gate_id(n_)}', q_)
    if verbose:
        print(f'\n자가진단 {"PASS" if not fails else "FAIL"} — 실패 {len(fails)}: {fails}')
    return (not fails), fails


if __name__ == '__main__':
    ok, _ = selftest()
    print()
    for n_, q_, d_ in gates_runfree():
        print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
    print()
    for n_, q_, d_ in gates_data():
        print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
    sys.exit(0 if ok else 1)
