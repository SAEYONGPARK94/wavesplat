"""B-19 단일출처 — **표현 절제를 희소 층으로 확장** (2026-08-25).

⚠**2026-08-26 수리판** (런 전, 결과 0행). 고친 것은 §"수리 기록" 에 적었다.
동결 해시(`FROZEN_md5.txt`)는 이 수리 뒤 **다시 계산**했다.

## 왜
`B-16` 이 저장소 최초로 **표현만 절제**했고(혼잡 46/48 · −38.61% · p=4.18e-12, 용량-반응 47/48)
그 `RESULT` 가 스스로 *"희소 층은 미측정"* 이라 적었다. ⟹ **그 칸을 채운다.**

★**이 배치가 중요한 이유는 두 서사가 여기서 만나기 때문이다**:
  A) *"희소의 큰 상대 이득은 **표현 아티팩트**"*(`HOSGATE`: 비선형화 시 우리 오차만 3.6배)
  B) *"연속 파수가 값을 한다"*(`B-16`, 혼잡)
그리고 `B-11b` 가 A 의 ②번 명제(*기준선 대비 우위도 인위적*)를 **반증**했다(15/16, −74.7%).
⚠단 그 층에서는 **기준선 자신이 무기술을 간신히 넘을 뿐**이다(`SPEC25<ZERO` 12/16 p=0.038).
⟹ 결과 전 예상·해석 규칙·긴장은 `PREREG.md` §2 에 적혀 있다(여기서 반복하지 않는다).

## 개입 — `B-16` 과 **동일**
`sequential_update._snap_k_to_grid`: 초기화 끝(VARPRO 직후)에 파수를 간격 `dk` 격자에
반올림하고 `(w, φ)` 를 LS 로 다시 맞춘다. 기본 `k_snap_dk=None` 은 완전한 no-op.
⛔`B-16` 의 `SHIP-NOW` 행을 재사용하지 않는다(층이 다르고, 재사용은 독립 표본 오류).

## 격자 간격 — ⚠**`DK1`/`DK2` 는 `gates_b16` 의 *반올림된 리터럴* 이다** (정직하게 적는다)
`HOSGATE` 코퍼스 64 에피소드가 **모두 같은 채점 격자**를 쓴다:
`gt_grid_x` 10–250 m(L=240) · `gt_grid_y` −100–100 m(L=200) ⟹ Δk_x=0.02617994, Δk_y=0.03141593.

⛔**초판이 *"`gates_b16` 에서 import 한다(손복사 0)"* 라 적은 것은 거짓이었다** — `gates_b16.py:48-49`
가 `DK1 = 0.02618  # = 2π/240` 이라는 **손으로 반올림한 리터럴**이고, 남의 손복사를 import 하는
것은 손복사 0 이 아니다(2026-08-26 지적).

★**그럼에도 도출식(`2*pi/240`)으로 바꾸지 않는다. 사유 둘**:
 1. `DK1` 은 문서용 상수가 아니라 **개입값 그 자체**다. `B-16` 은 이미 `0.02618` 로 **돌았다**.
    여기서 값을 바꾸면 PREREG §3 이 선언한 *"개입은 `B-16` 과 **동일**"* 이 깨지고,
    두 층의 절제가 서로 다른 격자가 된다.
 2. `gates_b16.py` 는 **이미 돌아간 배치의 동결 파일**이다. 고치면 그 배치의 팔 정의가
    실제로 돌아간 것과 달라진다(= 재현 불가).
⟹ **대신 정직하게 계측한다**: `G-10` 이 16 에피소드 각각에서 `2π/Lx` 를 **자료에서 재도출**해
`DK1` 과 대조하고, **반올림 편차의 실측값을 게이트 상세에 적는다**(런 전 실측: 절대 6.12e-08 ·
상대 2.34e-06 ⟹ 허용오차 1e-5 의 1/163, 그리고 이 배치가 재는 상대 섭동 Δk₁/|k| ≈ 0.14 의
약 1.7e-05 배 = **수치적으로 무해**). ⚠이것은 *"손복사가 없다"* 가 아니라
*"손복사가 있고, 그 크기를 쟀으며, 재도출로 대조한다"* 는 뜻이다.
`2π/Ly > DK1` (= 등방 선택이 y 축에서 **우리 팔에 유리**) 도 같은 게이트가 검사한다.

## 표본 — n = 16 (결과 전 고정)
`HOSGATE` 희소 HOS: `ncomp < 15` 그리고 `seed ∈ gates_b11b.SPARSE_SEEDS` (1585–1588)
= BF4/BF5 × {2,5}comp × 4 시드 = **16**. 추정기 시드 (42,43,44) median-of-3 ⟹ **144 런**.
⚠희소 HOS 는 이 16개가 전부다(`gen_b12.py:13` 이 *"희소는 만들지 않는다"*).
⛔**결과를 보고 n 을 늘리지 않는다**(`B-8b` 정지규칙).

★**비선형성 실현률은 이 표본에서 혼잡과 다르다** — `PREREG.md` §5.1 에 3-분할을 적었고
`G-13` 이 HDF5 에서 재도출해 `MESH_EXPECTED` 와 대조한다. ⛔n=16 에 재층화는 검정력이
없으므로 **나누지 않고 적기만** 한다(층 분해는 탐색적 예약).

## 판정 — 문턱은 `thresholds(n)` 이 `math.comb` 로 **직접** 낸다
⛔`B-16` 의 48-표본 문턱을 옮겨 쓰지 않는다. 형제 규약:
`NOMINAL` = p≤0.05 최소 · `CONFIRM` = p≤가족단위 α 최소 · `NOT_EARN` = `n − NOMINAL`.
n=16 → (12, 13, 4). 가족단위 α = 0.05/2 (주장은 PRIMARY·DOSE 둘, 음성대조는 게이트).

## 음성 대조 둘 — 그리고 **왜 하나만 무효화하는가**
`N-1` `SHIP-NOW < ZERO` ≥ NOMINAL — 미달이면 **PRIMARY 무효**(뺄 기술이 없으면 절제가 무의미).
`N-2` `KGRID1 < ZERO` ≥ NOMINAL — 미달이어도 **무효가 아니라 라벨 전환**이다.
★사유(결과 전): `KGRID1` 은 독립 방법이 아니라 **우리 팔의 의도된 열화**다. 그것이 무기술로
떨어지는 것은 *개입이 강했다*는 뜻이므로, 무효화 조건으로 걸면 **효과가 클수록 판정이
무효가 되는 단조 역전 게이트**가 된다 — `B-12` 가 결과 전에 고친 *"우연의 절반에서 발화하는
결론 라벨"* 과 같은 부류의 설계 결함이다. ⟹ 라벨을 갈라 두 문장을 구분한다.

⛔`ZERO` 에 상수 1.0 을 쓰지 않는다(2026-08-23 `ZERO_REF` 결함 부류). 참 제로필드를
**채점 창에서 직접 계산**한다. ★`B-16` 은 `zero_nRMSE`·`final_n` 을 행에 요청해 놓고
`run_estimation` 이 그 키를 반환하지 않아 **432행 전부 None** 이었고 게이트가 검사하지 않아
조용히 통과했다 ⟹ G-7·G-9 가 그 부류를 막는다.

## 수리 기록 (2026-08-26, 런 전 · 결과 0행)
1. ★★**BLOCKER: 되읽기가 항등식이었다.** 초판 러너는 `lambda_l1=cfg.get('lambda_l1')` 처럼
   **방금 쓴 dict 을 다시 읽어** 행에 실었다. `sequential_update.py:399 self.config = config`
   가 입력 dict 을 복사 없이 붙들므로 `행[k] == cfg[k]` 는 `cfg[k] == cfg[k]` 이고 **실패할 수
   없다**. ⟹ 되읽기를 **추정기 객체의 파생 속성**(`:402 lambda_l1` · `:408 tracking_mode` ·
   `:413/:757 init_method` · `:441 seed` …)에서 읽도록 바꿨다(`effective_from_updater`).
   게이트 이름도 *"배포 상수 되읽기"* → **"객체 파생 속성 일치"** 로 고쳤다.
   ⚠**파생 증거가 없는 키는 게이트에서 뺐다** — `NO_DERIVED_EVIDENCE` 참조.
2. **게이트 단일출처.** 초판은 G-*/N-* 전부가 **분석기 안**에 있었다(`gates_b19` 에 `def gates`
   가 0개) ⟹ 러너가 런 전에 게이트를 자가진단할 방법이 원리적으로 없었다.
   이제 `gates(rows)`·`aggregate(rows)`·`cmp_pair`·`cmp_zero` 가 전부 여기 있고
   분석기는 **호출만** 한다. `selftest()` 가 **가짜 행으로 전 게이트의 발화**를 확인한다.
3. **공허한 통과 제거.** 결과 0행에서 `G-0`·`G-2`·`G-8` 이 `all([])` 로 PASS 했고
   `G-10` 은 **에피소드 목록이 비면** PASS 했다(코퍼스 부재를 통과로 읽는다).
   ⟹ 전 게이트에 존재 조건을 붙이고, `selftest()` 가 `rows=[]` 에서 `ROWLESS_OK` 밖의
   게이트가 하나도 통과하지 않음을 **기계로** 확인한다.
4. **`DK1`/`DK2` 서술 정정** — 위 §"격자 간격" 참조(값은 바꾸지 않는다).
5. **`mesh_verdict` 3-분할 공개** — `MESH_EXPECTED` + `G-13`, PREREG §5.1.
"""
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B16 = os.path.abspath(os.path.join(HERE, '..', 'B16_KGRID_2026-08-24'))
B11 = os.path.abspath(os.path.join(HERE, '..', 'B11_HOS_BASELINE_2026-08-23'))
for _p in (HERE, ROOT, B16, B11):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 격자 간격: `gates_b16` 에서 import. ⚠**그 파일의 값이 손으로 반올림한 리터럴**이다 ──
#    (모듈 독스트링 §"격자 간격" 이 사유와 실측 편차를 적었다. G-10 이 자료에서 재도출해 대조.)
from gates_b16 import DK1, DK2                              # noqa: E402
# ── 표본 정의: B-11b 에서 import (희소 시드·코퍼스 경로), 시드는 B-11 에서 ──────────
from gates_b11b import HOS, SPARSE_SEEDS                    # noqa: E402
from gates_b11 import EST_SEEDS                             # noqa: E402

SPARSE_MAX_NCOMP = 15          # `ncomp < 15` = 희소 (run_b11b.py:30 의 조건절 그대로)
N_EP = 2 * 2 * len(SPARSE_SEEDS)        # BF4/BF5 × {2,5}comp × 4 시드 = 16

# ── 팔 ────────────────────────────────────────────────────────────────────────
# ⛔빈 dict 로 팔을 정의하지 않는다(2026-08-23 R2 F-7: `{}` 는 런타임 기본값 스냅샷).
ARMS = {'SHIP-NOW': {'k_snap_dk': None},
        'KGRID1':   {'k_snap_dk': DK1},
        'KGRID2':   {'k_snap_dk': DK2}}
ARM_ORDER = ['SHIP-NOW', 'KGRID1', 'KGRID2']

# ─────────────────────────────────────────────────────────────────────────────
# 되읽기 규약 (★2026-08-26 BLOCKER 수리) — **무엇이 증거이고 무엇이 아닌가**
# ─────────────────────────────────────────────────────────────────────────────
# ⛔**증거가 아닌 것**: 러너가 만든 설정 dict 을 다시 읽은 값. `run_estimation` 은
#   `cfg = DEFAULT_CONFIG.copy(); cfg.update(config)` 로 **자기 사본**을 만들고
#   `SequentialUpdater.__init__` 은 `self.config = config` 로 그 사본을 붙든다
#   (`sequential_update.py:399`) ⟹ 러너 쪽 dict 을 되읽는 것은 항등식이다.
#
# ✅**A. 객체 파생 속성** — `__init__` 이 config 와 **독립적으로** 스냅샷한 것만 쓴다.
#    줄번호는 `wavesplat/optimizer/sequential_update.py` 기준(2026-08-26 실측).
DERIVED_KEYS = ('lambda_l1',      # :402  config.get('lambda_l1', LAMBDA_L1_DEFAULT)
                'tracking_mode',  # :408
                'tracking_lr',    # :409
                'init_method',    # :413 스냅샷 → :757 이 **경로로 덮어쓴다**
                'real_estimator',  # :432
                'max_steepness',  # :440
                'seed',           # :441
                'depth')          # :400 (에피소드가 덮어쓴다 — 보고용)
#    ⚠`init_method` 는 `initialize()` 가 라우팅 결과로 덮어쓴다(`:757`). 따라서 런 뒤의
#    되읽기는 **설정값 'auto' 가 아니라 고른 경로**다 — 그것이 더 강한 증거다
#    ('auto' 가 객체에 닿지 않았다면 라우팅 자체가 없고 값은 `INIT_METHOD_CLASS_FALLBACK`
#     = 'v2_konly' 로 남는다).
DERIVED_EXPECTED = {'lambda_l1': 1e-4, 'tracking_mode': 'w_phi',
                    'tracking_lr': 0.005, 'real_estimator': 'bank_mmse'}
ROUTE_EXPECTED = 'learned'      # 파생 `init_method`(=경로) 와 `auto_selected` 둘 다 이 값
UPDATER_CLASS_EXPECTED = 'MedianEnsembleUpdater'   # init_method='auto' + ensemble_k=3
#
# ✅**B. 결과 dict 의 계수기** — 코드가 **실제로 쓴 값**을 되돌려준다(설정 사본이 아니다).
COUNTER_EXPECTED = {'varpro_steps': 400,   # sequential_update.py:913→:934, run.py:530
                    'varpro_fired': True}
#
# ✅**C. 행동 지문** — `learned_init_iters=0` 이면 결합 경사하강이 한 스텝도 안 돌아
#    `loss_history` 가 비고 `init_loss_n == 0` 이 된다(run.py:554 및 그 주석).
#
# ⚠**D. 파생 증거가 없는 키 — 게이트가 검사하지 않는다(문서용).**
NO_DERIVED_EVIDENCE = {
    'auto_switch_thresh':
        '`_auto_select_init`(:1008) 이 `self.config` 에서 직접 읽고 속성으로 스냅샷하지 '
        '않는다 ⟹ 객체 되읽기가 불가능하다. 이 배치가 가진 증거는 **간접**이다: '
        '`init_method=\'auto\'` 이므로 그 함수는 반드시 실행되고(G-6 의 `auto_selected` 가 '
        '그것을 증명한다), 임계 0 이면 라우터가 항상 learned 를 고른다. '
        '⚠**0 과 배포 전 8 을 구별하려면** 어떤 행에서 `auto_n_selected < 8` 이면서 경로가 '
        'learned 여야 한다 — G-6 상세가 그 판별 여부를 적는다. 판별되지 않으면 '
        '⛔"임계 0 이 먹었다" 고 쓰지 마라(두 값이 이 표본에서 구별 불가라고 써라). '
        '★2026-08-26 1셀 배선시험(BF4_2comp_seed1585, 시드 42)에서 `auto_n_selected = 4 < 8` '
        '이면서 경로가 learned 였다 ⟹ 이 층에서는 판별이 **기대**되지만, 기대를 게이트로 '
        '삼지 않고 G-6 이 전 행에서 다시 재어 상세에 적는다.',
    'learned_init_iters':
        '`_initialize_learned`(:889) 가 `self.config` 에서 직접 읽는다 ⟹ 속성 없음. '
        '대신 **행동 지문** `init_loss_n == 0`(G-5b) 이 객체측 증거다.',
}
#
# ✅**E. 배포 상수 트립와이어** — 아래 표는 *단일출처의 사본*이 아니라 **핀**이다.
#    `deploy_constants()` 가 패키지 단일출처에서 읽어 오고 `G-5d` 가 이 핀과 대조한다.
#    ⟹ 출하 기본값이 나중에 바뀌면 이 배치는 조용히 재라우팅되는 대신 **G-5d 에서 멈춘다.**
DEPLOY_EXPECTED = {'varpro_steps': 400, 'learned_init_iters': 0,
                   'auto_switch_thresh': 0, 'lambda_l1': 1e-4,
                   'init_method': 'auto'}
#
# ✅**F. 환경 트립와이어** — `lambda_l1` 은 두 곳에서 **일시적으로** 옛 값으로 바뀐다
#    (`:660` `S8_L1_TRACK_ONLY` · `:1385` `S8_L1_PHASE1_ONLY`, 둘 다 `finally` 로 복원).
#    복원되므로 런 뒤 되읽기는 옳게 보이지만 **런은 틀린다** ⟹ 환경변수를 행에 싣고 검사한다.
ENV_PROBE_KEYS = ('S8_L1_TRACK_ONLY', 'S8_L1_PHASE1_ONLY', 'S5_WINDOW')


def effective_from_updater(upd, n_captured=None) -> dict:
    """★추정기 **객체**에서 파생 속성을 읽는다 (설정 dict 을 다시 읽지 않는다).

    `MedianEnsembleUpdater` 는 미지 속성을 member 0 으로 전달한다(`:1895 __getattr__`)
    ⟹ 두 클래스 어느 쪽이 잡혀도 같은 키를 읽는다.
    """
    if upd is None:
        return None
    def _jsonable(v):
        # numpy 스칼라만 파이썬 값으로 내린다. ⛔`float()` 로 일괄 변환하지 않는다 —
        # 초판이 그렇게 해서 `seed` 42 가 42.0 이 됐다(비교는 통과하지만 행이 거짓을 적는다).
        if hasattr(v, 'item') and not isinstance(v, (str, bytes, bool)):
            try:
                return v.item()
            except Exception:                                # noqa: BLE001
                return v
        return v
    out = {k: _jsonable(getattr(upd, k, '<missing>')) for k in DERIVED_KEYS}
    out['updater_class'] = type(upd).__name__
    out['primary_class'] = type(getattr(upd, '_primary', upd)).__name__
    out['n_captured'] = n_captured
    out['env_probe'] = {k: os.environ.get(k) for k in ENV_PROBE_KEYS}
    return out


def deploy_constants() -> dict:
    """패키지 단일출처에서 배포 상수를 읽어 온다 (지연 import: 분석기는 torch 불필요)."""
    from wavesplat.optimizer.sequential_update import (
        VARPRO_STEPS_DEFAULT, LEARNED_INIT_ITERS_DEFAULT,
        AUTO_SWITCH_THRESH_DEFAULT, LAMBDA_L1_DEFAULT, INIT_METHOD_DEFAULT)
    return {'varpro_steps': VARPRO_STEPS_DEFAULT,
            'learned_init_iters': LEARNED_INIT_ITERS_DEFAULT,
            'auto_switch_thresh': AUTO_SWITCH_THRESH_DEFAULT,
            'lambda_l1': LAMBDA_L1_DEFAULT,
            'init_method': INIT_METHOD_DEFAULT}


# ── 판정 문턱 (코드가 계산한다) ───────────────────────────────────────────────
ALPHA = 0.05
FAMILY_ALPHA = 0.05 / 2         # 주장 둘: PRIMARY · DOSE (음성대조는 게이트)
MIN_N_USED = 14                 # 표본 손실이 이보다 크면 SAMPLE-LOSS-UNDECIDED
TP_FACTOR = 2.0                 # 채점 창: t > TP_FACTOR * Tp (run.py 의 값)
AUTO_SWITCH_THRESH_PRE_UNIFY = 8   # 배포 전 값 — G-6 의 판별 기준 (구별 가능성 보고용)
DK_REDERIVE_TOL = 1e-5          # G-10 허용오차 (실측 반올림 편차 6.12e-08 의 163배)


def sign_test_one_sided(w, n=N_EP):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def thresholds(n=N_EP):
    """(NOMINAL, CONFIRM, NOT_EARN). 형제 규약: NOT_EARN = n − NOMINAL (`B-12`)."""
    nominal = next((w for w in range(n + 1) if sign_test_one_sided(w, n) <= ALPHA), None)
    confirm = next((w for w in range(n + 1)
                    if sign_test_one_sided(w, n) <= FAMILY_ALPHA), None)
    return nominal, confirm, (n - nominal if nominal is not None else None)


# ── ZERO (참 제로필드 기준) ───────────────────────────────────────────────────
def zero_nrmse(h5_path, tp_factor=TP_FACTOR):
    """채점 창(`t > tp_factor·Tp`)에서 제로 예측의 nRMSE 를 **직접** 계산한다.

    `run.py` 의 `avg_nRMSE` 는 추적 프레임의 `nRMSE` 평균이고(`run.py:493`),
    프레임 창은 `EpisodeDataLoader.get_tracking_frames(tp_factor=2.0)` = `t > 2·Tp` 다.
    여기서는 **같은 창·같은 정규화(Hs_true)·같은 함수**(`metrics.nrmse`)로 pred=0 을 채점한다.
    ⛔상수 1.0 을 쓰지 않는다 — 2026-08-23 실측 `ZERO_REF` 결함 부류.
    """
    import numpy as np
    from wavesim.data_writer import load_episode
    from wavesplat.evaluation.metrics import nrmse
    d = load_episode(h5_path)
    ts = np.asarray(d['timestamps'], float)
    hs = float(d['Hs_true'])
    tp = float(d['Tp_true'])
    idx = [i for i in range(len(ts)) if ts[i] > tp_factor * tp]
    vals = []
    for i in idx:
        g = np.asarray(d['gt_wave_field'][i], dtype=np.float32)   # loader 와 같은 dtype
        vals.append(nrmse(np.zeros_like(g), g, hs))
    return dict(zero_nRMSE=float(np.mean(vals)) if vals else None,
                zero_n_frames=len(idx),
                zero_t_first=float(ts[idx[0]]) if idx else None,
                zero_t_last=float(ts[idx[-1]]) if idx else None,
                Hs_true=hs, Tp_true=tp)


def grid_dk(h5_path):
    """채점 격자에서 Δk = 2π/L 을 **재도출**한다 (G-10)."""
    import math
    import h5py
    with h5py.File(h5_path, 'r') as h:
        gx = h['frames/gt_grid_x'][:]
        gy = h['frames/gt_grid_y'][:]
    return (2 * math.pi / (float(gx[-1]) - float(gx[0])),
            2 * math.pi / (float(gy[-1]) - float(gy[0])))


# ── 표본의 비선형성 실현률 (m-9, 2026-08-26 신설) ─────────────────────────────
# ★`HOSGATE` 생성기는 에피소드마다 메시 해상도 판정을 HDF5 속성으로 남긴다
#   (`config.attrs['mesh_verdict']`). **희소 표본의 분포는 혼잡과 다르다**:
#     혼잡 48 : ADEQUATE 31 · DEGRADED 17
#     희소 16 : ADEQUATE 11 · DEGRADED  2 · INADEQUATE 3   ← 혼잡에 없는 등급이 3개
#   ⟹ 이 표본을 `"비선형 HOS M=3 참값"` 으로 **단일 서술하지 않는다**(PREREG §5.1).
#   아래 표는 손복사가 아니라 **핀**이다 — `G-13` 이 HDF5 에서 재도출해 대조한다.
MESH_EXPECTED = {'ADEQUATE': 11, 'DEGRADED': 2, 'INADEQUATE': 3}
MESH_ATTR = 'mesh_verdict'


def mesh_census(eps=None):
    """희소 16 에피소드의 `mesh_verdict` 를 **HDF5 에서 세어** 돌려준다."""
    import h5py
    out = {}
    for e in (eps if eps is not None else episodes()):
        with h5py.File(e['path'], 'r') as h:
            v = h['config'].attrs.get(MESH_ATTR)
        if isinstance(v, bytes):
            v = v.decode()
        out[e['episode']] = str(v)
    counts = {}
    for v in out.values():
        counts[v] = counts.get(v, 0) + 1
    return counts, out


# ── 양성대조: `B-11b`(08-23) 의 동결 행 ───────────────────────────────────────
# ★이 배치가 `SHIP-NOW` 를 **재사용하지 않고 다시 돌리기 때문에** 가능한 검사다.
#   같은 에피소드·같은 시드·같은 출하 구성이므로 **비트 동일**이어야 한다.
#   틀어지면 08-23 이후의 코드 변경(k_snap 노브·B-13 채점창 키 등)이 no-op 이 아니었다는 뜻이다.
#   ★그리고 이것이 러너의 **관측 훅**(`make_updater` 감싸기, run_b19.py 참조)이 거동을
#   바꾸지 않았다는 증명이기도 하다 — B-11b 는 훅 없이 돌았다.
B11B_RESULTS = os.path.join(B11, 'results_b11b.jsonl')
POSCTRL_RTOL = 0.0             # 0 = 비트 동일 요구
ZERO_RTOL = 1e-5               # ZERO 교차검증(부동소수 누적 차이만 허용)


def _b11b_rows():
    import json
    out = []
    with open(B11B_RESULTS) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                out.append(r)
    return out


def b11b_ship():
    """`B-11b` 의 동결 `SHIP-NOW` 행: {(episode, est_seed): avg_nRMSE}."""
    return {(r['episode'], r['est_seed']): float(r['avg_nRMSE'])
            for r in _b11b_rows() if r.get('arm') == 'SHIP-NOW'}


def b11b_zero():
    """`B-11b` 의 `SPEC25` 행이 함께 기록한 제로필드 기준: {episode: zero_nRMSE}.

    우리는 `zero_nrmse()` 로 **독립 계산**하므로 이것은 교차검증용이다
    (두 경로가 서로 다른 코드로 같은 양을 낸다).
    """
    return {r['episode']: float(r['zero_nRMSE']) for r in _b11b_rows()
            if r.get('arm') == 'SPEC25' and r.get('zero_nRMSE') is not None}


def episodes():
    """희소 HOS 에피소드. 조건절은 `run_b11b.crowded_episodes` 의 것을 그대로 쓴다."""
    import glob
    import re
    out = []
    for p in sorted(glob.glob(os.path.join(HOS, '*.h5'))):
        m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(.+)\.h5', os.path.basename(p))
        if m and int(m.group(2)) < SPARSE_MAX_NCOMP and int(m.group(3)) in SPARSE_SEEDS:
            out.append(dict(path=p, episode=os.path.basename(p)[len('episode_'):-3],
                            bf=m.group(1), ncomp=int(m.group(2)), gen_seed=int(m.group(3))))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 집계 · 비교 (★단일출처: 게이트와 분석기가 **같은 함수**를 부른다)
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(rows):
    """median-of-3 집계 + 유효 표본 선정. 게이트와 분석기가 공유한다.

    ⚠**중복 구현 금지** — 이 저장소가 반복해서 물린 결함 부류가
    *"같은 기능이 두 파일에 구현돼 한쪽만 계측한 것"* 이다(λ 3중 기재 · W4 개입 기대값 4중).
    """
    import statistics as st
    ok = [r for r in rows if 'error' not in r]
    cell = {}
    for r in ok:
        cell.setdefault((r['arm'], r['episode']), []).append(r)
    val = {k: st.median([x['avg_nRMSE'] for x in v]) for k, v in cell.items()}
    eps_all = sorted({e for (_, e) in val})
    good = []
    for e in eps_all:
        rs = [r for r in ok if r['episode'] == e]
        route_ok = all(r.get('auto_selected') == ROUTE_EXPECTED for r in rs)
        snap_ok = all(bool(r.get('k_snap_fired')) == (r['arm'] != 'SHIP-NOW') for r in rs)
        full = all((a, e) in val for a in ARM_ORDER)
        if route_ok and snap_ok and full:
            good.append(e)
    zero = {r['episode']: r['zero_nRMSE'] for r in ok
            if r['episode'] in good and r.get('zero_nRMSE') is not None}
    return dict(ok=ok, cell=cell, val=val, eps_all=eps_all, good=good,
                zero=zero, n_used=len(good))


def cmp_pair(val, eps, a, b):
    """짝 비교 `a < b`. 동점은 승리로 세지 않는다(strict `<`, `B-16` 상속)."""
    import statistics as st
    common = [e for e in eps if (a, e) in val and (b, e) in val]
    w = sum(1 for e in common if val[(a, e)] < val[(b, e)])
    return dict(n=len(common), better=w,
                n_tie=sum(1 for e in common if val[(a, e)] == val[(b, e)]),
                p_one_sided=sign_test_one_sided(w, len(common)) if common else None,
                rel_median_pct=(st.median([(val[(a, e)] - val[(b, e)]) / val[(b, e)] * 100
                                           for e in common]) if common else None),
                median_a=st.median([val[(a, e)] for e in common]) if common else None,
                median_b=st.median([val[(b, e)] for e in common]) if common else None)


def cmp_zero(val, zero, eps, a):
    """음성 대조 `a < ZERO` (참 제로필드 기준)."""
    import statistics as st
    common = [e for e in eps if (a, e) in val and e in zero]
    w = sum(1 for e in common if val[(a, e)] < zero[e])
    return dict(n=len(common), better=w,
                p_one_sided=sign_test_one_sided(w, len(common)) if common else None,
                rel_median_pct=(st.median([(val[(a, e)] - zero[e]) / zero[e] * 100
                                           for e in common]) if common else None))


# ─────────────────────────────────────────────────────────────────────────────
# 게이트 (★단일출처 — 분석기는 `gates(rows)` 를 **호출만** 한다)
# ─────────────────────────────────────────────────────────────────────────────
# ⛔**공허한 통과 금지** — 결과 0행에서 참이 되는 게이트를 만들지 않는다(W8 `G-4` 부류,
#    `gates_b18.py:223` 이 같은 날 축자로 적었다). 자료만 읽는 게이트도 **자료가 비면
#    통과하면 안 된다**(초판 G-10 이 그랬다).
#    아래 셋만이 행 없이도 판정할 수 있는 게이트다(자료·패키지만 읽는다).
ROWLESS_OK = ('G-5d', 'G-10', 'G-13')


def _srt(vals):
    """`None` 이 섞인 값 집합도 정렬한다 (게이트 상세 출력용).

    ⚠초판은 `sorted({...})` 를 그냥 썼다 — **결함 행이 있을 때만** `int < None` 으로
    터진다(= 게이트가 FAIL 을 보고하는 대신 분석기 전체가 죽는다). 자가진단이 잡았다.
    """
    return sorted(vals, key=lambda x: (x is None, repr(x)))


def gates(rows, agg=None):
    """러너 산출 전체(오류 행 포함)를 받아 `[(이름, 통과여부, 상세)]` 를 낸다."""
    import statistics as st
    if agg is None:
        agg = aggregate(rows)
    ok, cell, val = agg['ok'], agg['cell'], agg['val']
    eps_all, good, zero = agg['eps_all'], agg['good'], agg['zero']
    out = []

    def g(name, cond, detail):
        out.append((name, bool(cond), str(detail)))

    g('G-0 오류 0행', bool(rows) and len(ok) == len(rows),
      f'{len(rows) - len(ok)} 행 오류 / {len(rows)} 행')

    g('G-1 표본 = 3팔 × 16 에피소드',
      len(val) == len(ARM_ORDER) * N_EP and len(eps_all) == N_EP,
      f'{len(val)}/{len(ARM_ORDER) * N_EP} 셀 · 에피소드 {len(eps_all)}/{N_EP}')

    bad_seed = {k: sorted(x['est_seed'] for x in v) for k, v in cell.items()
                if sorted(x['est_seed'] for x in v) != sorted(EST_SEEDS)}
    g('G-2 셀마다 추정기 시드 3개 전건', bool(cell) and not bad_seed,
      f'{len(cell) - len(bad_seed)}/{len(cell)} 셀 정상'
      + (f' · 이탈 예: {list(bad_seed.items())[:2]}' if bad_seed else ''))

    fired = {a: sum(1 for r in ok if r['arm'] == a and r.get('k_snap_fired'))
             for a in ARM_ORDER}
    tot = {a: sum(1 for r in ok if r['arm'] == a) for a in ARM_ORDER}
    g('G-3 개입 발화 (스냅 팔 전건 · SHIP 0건)',
      bool(ok) and tot.get('SHIP-NOW') and fired.get('SHIP-NOW') == 0
      and all(fired.get(a) == tot.get(a) and tot.get(a) for a in ('KGRID1', 'KGRID2')),
      ' · '.join(f'{a} {fired.get(a)}/{tot.get(a)}' for a in ARM_ORDER))

    dks = {a: sorted({r.get('k_snap_dk') for r in ok if r['arm'] == a},
                     key=lambda x: (x is None, x)) for a in ARM_ORDER}
    g('G-4 dk 되읽기 (계수기 `init_result.k_snap.dk`)',
      bool(ok) and dks.get('SHIP-NOW') == [None] and dks.get('KGRID1') == [DK1]
      and dks.get('KGRID2') == [DK2], str(dks))

    # ── G-5 ★BLOCKER 수리: 설정 dict 이 아니라 **추정기 객체**에서 읽은 값과 대조 ──
    eff = [r.get('effective') for r in ok]
    have = bool(ok) and all(isinstance(e, dict) and all(k in e for k in DERIVED_KEYS)
                            for e in eff)
    bad_der = sorted({k for r in ok for k, v in DERIVED_EXPECTED.items()
                      if (r.get('effective') or {}).get(k) != v})
    seed_ok = bool(ok) and all((r.get('effective') or {}).get('seed') == r['est_seed']
                               for r in ok)
    cls_ok = bool(ok) and all((r.get('effective') or {}).get('updater_class')
                              == UPDATER_CLASS_EXPECTED for r in ok)
    cap_ok = bool(ok) and all((r.get('effective') or {}).get('n_captured') == 1 for r in ok)
    env_ok = bool(ok) and all(all(v is None for v in
                                  ((r.get('effective') or {}).get('env_probe') or {}).values())
                              for r in ok)
    g('G-5 ★객체 파생 속성 일치 (⛔cfg 항등식 아님) + 시드·클래스·환경',
      have and not bad_der and seed_ok and cls_ok and cap_ok and env_ok,
      f'기대 {DERIVED_EXPECTED} · 이탈 키 {bad_der} · 키존재 {have} · seed 일치 {seed_ok} '
      f'· 클래스 {cls_ok} · 포착수 1 {cap_ok} · S8/S5 환경변수 미설정 {env_ok}')

    # ⛔**2026-08-26 V4 반증**: 초판은 `(r.get('init_loss_n') or 0) == 0` 이었다.
    #   `None or 0 == 0` 이므로 **키가 사라져도 통과**한다 = `B-16` 의 `final_n` 부류
    #   (계산되는데 저장되지 않은 값을 게이트가 조용히 통과시킨 사고)의 재현이다.
    #   ⟹ `is not None` 을 명시로 요구한다. 실측 1셀에서 이 키는 `0` 으로 채워진다.
    g('G-5b 행동 지문 init_loss_n == 0 (⛔None 은 통과 아님)',
      bool(ok) and all(r.get('init_loss_n') is not None and r['init_loss_n'] == 0
                       for r in ok),
      f"None {sum(1 for r in ok if r.get('init_loss_n') is None)}/{len(ok)} · "
      f'값 {_srt({r.get("init_loss_n") for r in ok})}')

    bad_cnt = sorted({k for r in ok for k, v in COUNTER_EXPECTED.items()
                      if (bool(r.get(k)) if isinstance(v, bool) else r.get(k)) != v})
    # ★⛔**2026-08-26 V4 반증**: 이 게이트가 보던 둘은 **건너뛴 VARPRO 를 못 잡는다**.
    #   · `varpro_steps`(run.py:530) = `init_result['varpro_steps']` = `:913` 이 config 에서
    #     읽은 값의 **echo** 다 (건너뛰어도 400 이다).
    #   · `varpro_fired`(run.py:531) = `bool(init_result['varpro'])` 인데 `varpro.py:201` 이
    #     **건너뛴 경우에도 truthy dict** 을 먼저 만들어 반환한다 ⟹ 항상 True 다.
    #   형제 배치 `B-18` 이 이 함정을 주석으로 적고 `vp_steps_exec`·`vp_skipped` 로 잡았는데
    #   이 배치만 빠져 있었다(= "같은 기능이 두 곳에 구현돼 한쪽만 계측" 부류).
    #   ⟹ `run_estimation` 은 실행 스텝 수를 내보내지 않으므로, **실행됐을 때만 채워지는**
    #      `varpro_obj_before/after`(`varpro.py:248-258` 의 `rec.update`, 건너뜀 반환 뒤에 있다)
    #      를 행에 싣고 그 존재를 요구한다. 목적함수가 실제로 내려갔는지도 함께 본다.
    vp_none = [r for r in ok if r.get('varpro_obj_before') is None
               or r.get('varpro_obj_after') is None]
    g('G-5c 계수기 되읽기 + ★VARPRO 실집행 증거 (⛔echo/truthy 아님)',
      bool(ok) and not bad_cnt and not vp_none,
      f'기대 {COUNTER_EXPECTED} · 이탈 키 {bad_cnt} · '
      f"steps {_srt({r.get('varpro_steps') for r in ok})} · "
      f'obj 미기록(=건너뜀) {len(vp_none)}/{len(ok)}')

    try:
        dep = deploy_constants()
        g('G-5d 배포 상수 트립와이어 (패키지 단일출처 == 이 배치의 핀)',
          dep == DEPLOY_EXPECTED, f'{dep} vs {DEPLOY_EXPECTED}')
    except Exception as exc:                                 # noqa: BLE001
        g('G-5d 배포 상수 트립와이어 (패키지 단일출처 == 이 배치의 핀)', False,
          f'{type(exc).__name__}: {exc}')

    routes = _srt({r.get('auto_selected') for r in ok})
    eff_routes = _srt({(r.get('effective') or {}).get('init_method') for r in ok})
    ens = _srt({r.get('ensemble_k') for r in ok})
    nsel = [r.get('auto_n_selected') for r in ok if r.get('auto_n_selected') is not None]
    disc = bool(nsel) and min(nsel) < AUTO_SWITCH_THRESH_PRE_UNIFY
    # ★2026-08-26 V4: `auto_real_gate` 를 행에 싣기만 하고 **아무도 읽지 않았다**.
    #   합성 auto 경로에서 이 값은 `sequential_update.py:756` 이 **False** 로 채운다
    #   (실측 게이트가 발화하면 True, auto 밖이면 None) — 1셀 실측으로 확인했다.
    #   ⟹ 전 행 `False` 를 요구한다. 이것이 *"실측 게이트로 새지 않았다"* 의 직접 증거다.
    gates_fired = [r.get('auto_real_gate') for r in ok
                   if r.get('auto_real_gate') is not False]
    g('G-6 라우터 learned 전건 · ensemble_k == 1 · 파생 init_method == 경로 · 실측게이트 미발화',
      bool(ok) and routes == [ROUTE_EXPECTED] and eff_routes == [ROUTE_EXPECTED]
      and ens == [1] and not gates_fired,
      f'auto_real_gate≠False {len(gates_fired)}/{len(ok)} · '
      f'auto_selected {routes} · 파생 init_method {eff_routes} · ensemble_k {ens} · '
      f"auto_n_selected 최소 {min(nsel) if nsel else None} ⟹ "
      f"임계 0 vs 배포전 {AUTO_SWITCH_THRESH_PRE_UNIFY} "
      f"{'판별됨(최소 n_selected < 8 인데 learned 경로)' if disc else '⚠구별 불가 — 이 표본에서 두 값이 같은 경로를 낸다'}")

    # ⛔2026-08-26 V4: `is not None` 만 보면 **`n_gaussians == 0` 이 통과한다** — 표현이
    #   하나도 없는 런이다. 그러면 `apply_varpro` 도 `no_model` 로 건너뛰고 스냅도 무의미하다.
    #   표현 절제 배치에서 그 행이 통과하면 절제 자체가 정의되지 않는다 ⟹ `> 0` 을 요구한다.
    g('G-7 원자 수 기록 (n_gaussians > 0 — ⛔0·None 통과 아님)',
      bool(ok) and all(isinstance(r.get('n_gaussians'), int)
                       and r['n_gaussians'] > 0 for r in ok),
      f"None {sum(1 for r in ok if r.get('n_gaussians') is None)} · "
      f"0 {sum(1 for r in ok if r.get('n_gaussians') == 0)} / {len(ok)} · "
      f"값 {_srt({r.get('n_gaussians') for r in ok})}")

    # ⛔2026-08-26 V4: 초판은 세 키가 **전 행 None 이어도** 집합 크기 1 이라 통과했다
    #   (`B-16` 의 `final_n` 부류). ⟹ 창 값의 존재를 먼저 요구한다.
    win = {}
    for r in ok:
        win.setdefault(r['episode'], set()).add(
            (r.get('tracking_frames'), r.get('scored_t_first'), r.get('scored_t_last')))
    bad_win = {e: v for e, v in win.items() if len(v) != 1}
    win_none = [r for r in ok if r.get('tracking_frames') is None
                or r.get('scored_t_first') is None or r.get('scored_t_last') is None]
    g('G-8 채점 창이 에피소드 안에서 9행 전부 동일 (⛔전 행 None 통과 아님)',
      bool(win) and not bad_win and not win_none,
      f'{len(win) - len(bad_win)}/{len(win)} 에피소드 일치 · 창 미기록 {len(win_none)}/{len(ok)}'
      + (f' · 불일치 예: {list(bad_win.items())[:1]}' if bad_win else ''))

    zsets = {}
    for r in ok:
        zsets.setdefault(r['episode'], set()).add(r.get('zero_nRMSE'))
    bad_z = {e: v for e, v in zsets.items() if len(v) != 1}
    same_window = bool(ok) and all(r.get('zero_n_frames') == r.get('tracking_frames')
                                   for r in ok)
    none_z = sum(1 for r in ok if r.get('zero_nRMSE') is None)
    g('G-9 ZERO 창 일치 · 에피소드 내 불변 · None 0행',
      bool(zsets) and not bad_z and same_window and none_z == 0,
      f'창 일치 {same_window} · 분산 에피소드 {len(bad_z)} · None {none_z}')

    # ── G-10 격자 재도출. ⚠**에피소드가 0개면 통과하지 않는다**(초판의 공허한 통과) ──
    eps_list = episodes()
    dk_bad, devs = [], []
    for e in eps_list:
        try:
            dkx, dky = grid_dk(e['path'])
        except Exception as exc:                             # noqa: BLE001
            dk_bad.append((e['episode'], repr(exc)))
            continue
        devs.append(abs(dkx - DK1))
        if abs(dkx - DK1) >= DK_REDERIVE_TOL or not dky > DK1:
            dk_bad.append((e['episode'], dkx, dky))
    g('G-10 격자 재도출 (|2π/Lx − DK1| < 1e-5 · 2π/Ly > DK1 = 등방 선택이 유리)',
      len(eps_list) == N_EP and not dk_bad,
      f'에피소드 {len(eps_list)}/{N_EP} · 이탈 {len(dk_bad)} · '
      f'★DK1 반올림 편차 최대 {max(devs):.3e} (상대 {max(devs) / DK1:.3e}, '
      f'허용오차의 {max(devs) / DK_REDERIVE_TOL:.4f}배)'
      if devs else f'에피소드 {len(eps_list)}/{N_EP} · 이탈 {len(dk_bad)}')

    # ── G-11 ★양성대조: SHIP-NOW 가 `B-11b`(08-23) 동결 행과 **비트 동일**인가 ──
    try:
        ref = b11b_ship()
    except Exception as exc:                                 # noqa: BLE001
        ref = {}
        g('★G-11 양성대조 기준 파일', False, f'{type(exc).__name__}: {exc}')
    ship = [r for r in ok if r['arm'] == 'SHIP-NOW']
    pc_hit = [r for r in ship if (r['episode'], r['est_seed']) in ref]
    pc_bad = [(r['episode'], r['est_seed'], r['avg_nRMSE'],
               ref[(r['episode'], r['est_seed'])])
              for r in pc_hit
              if abs(r['avg_nRMSE'] - ref[(r['episode'], r['est_seed'])])
              > POSCTRL_RTOL * abs(ref[(r['episode'], r['est_seed'])])]
    g('★G-11 양성대조: SHIP-NOW 가 B-11b 동결 행과 비트 동일',
      bool(pc_hit) and not pc_bad and len(pc_hit) == len(ship) and bool(ship),
      f'대조 {len(pc_hit)}/{len(ship)} 행 · 불일치 {len(pc_bad)}'
      + (f' 예: {pc_bad[:1]}' if pc_bad else ''))

    # ── G-12 ZERO 교차검증: 독립 계산값이 B-11b 의 SPEC25 행 기록과 맞는가 ──
    try:
        zref = b11b_zero()
    except Exception:                                        # noqa: BLE001
        zref = {}
    zpairs = [(e, s, zref[e]) for e in zsets for s in zsets[e]
              if e in zref and s is not None]
    z_bad = [(e, s, r) for e, s, r in zpairs if abs(s - r) > ZERO_RTOL * abs(r)]
    g('★G-12 ZERO 교차검증 (독립 계산 vs B-11b 기록)',
      bool(zpairs) and not z_bad and len(zpairs) == len(zsets),
      f'{len(zpairs) - len(z_bad)}/{len(zpairs)} 일치 (rtol {ZERO_RTOL})'
      + (f' 예: {z_bad[:1]}' if z_bad else ''))

    # ── G-13 ★표본의 비선형성 실현률 (m-9) — 자료에서 재도출해 핀과 대조 ──
    try:
        counts, _per = mesh_census(eps_list)
        g('★G-13 mesh_verdict 3-분할 = 사전등록 핀 (⛔표본을 "HOS M=3" 로 단일 서술 금지)',
          bool(eps_list) and counts == MESH_EXPECTED and sum(counts.values()) == N_EP,
          f'{counts} vs 핀 {MESH_EXPECTED} (n={sum(counts.values())}/{N_EP})')
    except Exception as exc:                                 # noqa: BLE001
        g('★G-13 mesh_verdict 3-분할 = 사전등록 핀', False, f'{type(exc).__name__}: {exc}')

    # ── N-1 / N-2 음성 대조 ────────────────────────────────────────────────
    n_used = agg['n_used']
    nominal = thresholds(n_used)[0] if n_used else None
    neg1 = cmp_zero(val, zero, good, 'SHIP-NOW')
    neg2 = cmp_zero(val, zero, good, 'KGRID1')
    neg1_ok = bool(nominal is not None and neg1['n'] and neg1['better'] >= nominal)
    neg2_ok = bool(nominal is not None and neg2['n'] and neg2['better'] >= nominal)
    g(f'N-1 음성대조 SHIP-NOW < ZERO ≥ {nominal}/{n_used} (미달 ⟹ PRIMARY 무효)',
      neg1_ok, f"{neg1['better']}/{neg1['n']} p={neg1['p_one_sided']}")
    g(f'N-2 음성대조 KGRID1 < ZERO ≥ {nominal}/{n_used} (미달 ⟹ 라벨 전환)',
      neg2_ok, f"{neg2['better']}/{neg2['n']} p={neg2['p_one_sided']}")
    return out


# ── 라벨 (결과 전 확정) ───────────────────────────────────────────────────────
def label_primary(w, n, neg1_ok, neg2_ok, n_tie=0):
    """PRIMARY 라벨. `neg1_ok` 는 무효화, `neg2_ok` 는 **읽기 전환**이다 (§7).

    `n_tie` = 두 팔의 `avg_nRMSE` 가 **정확히 같은** 에피소드 수. 동점은 승리로 세지 않고
    (strict `<`, `B-16` 상속) **산술적으로 문턱이 도달 불가**해지면 그 사실을 라벨에 적는다.
    """
    if n < MIN_N_USED:
        return ('SAMPLE-LOSS-UNDECIDED',
                f'유효 표본 {n} < {MIN_N_USED} ⟹ 문턱 재계산으로 결론을 사지 않는다')
    nominal, confirm, not_earn = thresholds(n)
    if n - n_tie < nominal:
        return ('UNDECIDABLE-TIES',
                f'동점 {n_tie}/{n} ⟹ 최대 도달 가능 승수 {n - n_tie} < 문턱 {nominal} '
                '= **산술적으로 판정 불가**. ⛔널을 "효과 없음" 으로 읽지 마라.')
    if not neg1_ok:
        return ('ARM-NO-SKILL-HERE',
                f'⛔`SHIP-NOW` 가 이 층에서 참 제로필드를 문턱({nominal}/{n})만큼 이기지 못한다 '
                '⟹ 뺄 기술이 없으므로 절제 비교는 무효다')
    p = sign_test_one_sided(w, n)
    if w >= nominal and not neg2_ok:
        return ('SNAP-DESTROYS-SPARSE-ARM',
                f'★{w}/{n}, p={p:.4g} — 스냅한 팔이 **참 제로필드조차 이기지 못한다** ⟹ '
                '재는 것은 두 작동 팔 사이의 *차* 가 아니라 **기술의 소멸**이다. '
                '⛔상대 중앙 %를 margin 으로 인용 금지(무기술 팔 대비 크기는 비교량이 아니다).')
    if w >= confirm:
        return ('CONTINUOUS-K-EARNS-SPARSE',
                f'★{w}/{n}, p={p:.4g} — 희소 층에서도 격자에 스냅하면 나빠진다 ⟹ '
                '연속 파수가 **두 층 모두에서** 이득의 담지자 쪽이다')
    if w >= nominal:
        return ('CONTINUOUS-K-EARNS-SPARSE-NOMINAL',
                f'⚠{w}/{n}, p={p:.4g} — 명목만(가족단위 α={FAMILY_ALPHA} 미달), 탐색적')
    if w <= not_earn:
        return ('CONTINUOUS-K-DOES-NOT-EARN-SPARSE',
                f'⛔{w}/{n} — ★희소에서는 격자에 스냅해도 나빠지지 않는다 ⟹ '
                '희소에서 이긴 것은 표현이 아니라 적합 절차다 (기여 ① 문안에 층 한정어가 필요하다)')
    return ('ATTRIBUTION-UNDECIDED-SPARSE',
            f'{w}/{n} — 문턱 사이. ★**연속 파수는 혼잡 층에서만 값을 한다** = 작동 경계이고, '
            '이것은 나쁜 소식이 아니다(PREREG §2.4 규칙 2).')


def label_dose(w, n, n_tie=0):
    """용량-반응.

    ★**희소 층에서는 동점이 구조적으로 생긴다** — 원자가 1–5개면 `dk` 격자와 `2dk` 격자가
    **같은 점으로 반올림**될 수 있다(`k/dk` 가 짝수 정수 근처면 두 격자의 최근접점이 일치).
    실제로 결과 0행 스모크(BF4 2comp seed1585, 원자 1개)에서 두 팔이 **비트 동일**했다.
    ⟹ 동점을 승리로 세지 않되(strict `<`), **문턱이 산술적으로 도달 불가**해지면 널을
    *"용량-반응 없음"* 으로 읽지 않는다.
    """
    if n < MIN_N_USED:
        return ('SAMPLE-LOSS-UNDECIDED', f'유효 표본 {n} < {MIN_N_USED}')
    nominal = thresholds(n)[0]
    if n - n_tie < nominal:
        return ('DOSE-UNDECIDABLE-TIES',
                f'동점 {n_tie}/{n} ⟹ 최대 도달 가능 승수 {n - n_tie} < 문턱 {nominal} '
                '= 산술적으로 판정 불가(희소 층의 구조적 동점).')
    p = sign_test_one_sided(w, n)
    return (('DOSE-RESPONSE',
             f'{w}/{n}(동점 {n_tie}), p={p:.4g} — 간격을 2배로 하면 더 나빠진다')
            if w >= nominal else
            ('NO-DOSE-RESPONSE', f'{w}/{n}(동점 {n_tie}), p={p:.4g}'))


# ─────────────────────────────────────────────────────────────────────────────
# ★런 전 게이트 자가진단 — **가짜 행으로 게이트가 실제로 발화하는지 확인한다**
# ─────────────────────────────────────────────────────────────────────────────
# 왜: 이 저장소가 반복해서 물린 것이 *"검사기가 검사하지 않는다"* 다
#     (W8 `G-4` 공허한 통과 · 2026-08-23 하루 세 번 · 초판 B-19 의 G-0/G-2/G-8/G-10).
#     ⟹ **게이트를 켜 보지 않고 배치를 돌리지 않는다.**
# ⛔이 함수는 결과 파일을 쓰지 않는다. 추정기를 부르지 않는다(런 0회).
# ⚠공개: 픽스처가 `B-11b` 동결 행을 쓰므로 **N-1 의 값(15/16)이 런 전에 도출 가능**하다.
#   PRIMARY·DOSE 는 픽스처가 임의로 만든 값이라 아무 정보도 담지 않는다. PREREG §10 참조.
def _fixture_rows():
    """전 게이트가 통과해야 하는 **합성 행 144개**. (추정기 미호출)"""
    ship, zref, eps = b11b_ship(), b11b_zero(), episodes()
    mult = {'SHIP-NOW': 1.0, 'KGRID1': 1.10, 'KGRID2': 1.20}
    rows = []
    for e in eps:
        name = e['episode']
        for arm in ARM_ORDER:
            for s in EST_SEEDS:
                base = ship[(name, s)]
                rows.append(dict(
                    arm=arm, est_seed=s, episode=name, bf=e['bf'],
                    ncomp=e['ncomp'], gen_seed=e['gen_seed'],
                    avg_nRMSE=base * mult[arm],
                    k_snap_dk=ARMS[arm]['k_snap_dk'],
                    k_snap_fired=(arm != 'SHIP-NOW'),
                    k_snap_cells=(None if arm == 'SHIP-NOW' else 3),
                    k_snap_shift_median=(None if arm == 'SHIP-NOW' else 0.006),
                    varpro_steps=400, varpro_fired=True, init_loss_n=0,
                    # ★V4: 실집행 증거(건너뛰면 `rec` 에 없어 None 이 된다)
                    varpro_obj_before=1.0e-2, varpro_obj_after=4.0e-3,
                    auto_selected='learned', auto_n_selected=4,
                    # ⛔V4 정정: 합성 auto 경로의 실제 값은 **False** 다(None 아님).
                    #   1셀 실측(BF4_2comp_seed1585, 시드 42)에서 확인 — 초판 픽스처가
                    #   런이 낼 수 없는 값을 적고 있었고 게이트가 그것을 읽지 않았다.
                    auto_real_gate=False, ensemble_k=1, n_gaussians=3,
                    tracking_frames=160, scored_t_first=14.1, scored_t_last=30.0,
                    episode_total_frames=301,
                    zero_nRMSE=zref[name], zero_n_frames=160,
                    effective=dict(lambda_l1=1e-4, tracking_mode='w_phi',
                                   tracking_lr=0.005, init_method='learned',
                                   real_estimator='bank_mmse', max_steepness=0.15,
                                   seed=s, depth=50.0,
                                   updater_class=UPDATER_CLASS_EXPECTED,
                                   primary_class='SequentialUpdater', n_captured=1,
                                   env_probe={k: None for k in ENV_PROBE_KEYS}),
                    cfg_intent={'⚠': '의도표 — 증거 아님'},
                    git_head='fixture', import_md5={}, wall_s=1.0))
    return rows


def _perturbations():
    """(게이트 접두어, 설명, 변형함수) — 각 변형은 그 게이트를 **FAIL 로 뒤집어야** 한다."""
    def _mut(fn):
        def apply(rows):
            import copy
            rs = copy.deepcopy(rows)
            fn(rs)
            return rs
        return apply

    def first(rs, arm):
        return next(r for r in rs if r['arm'] == arm)

    return [
        ('G-0', '오류 행 1개 추가',
         _mut(lambda rs: rs.append(dict(arm='KGRID1', episode=rs[0]['episode'],
                                        est_seed=42, error='boom')))),
        ('G-1', '한 셀(3행) 제거',
         _mut(lambda rs: [rs.remove(r) for r in
                          [x for x in rs if x['arm'] == 'KGRID2'][:3]])),
        ('G-2', '한 행의 est_seed 를 99 로',
         _mut(lambda rs: first(rs, 'KGRID1').__setitem__('est_seed', 99))),
        ('G-3', 'KGRID1 한 행의 발화를 False 로',
         _mut(lambda rs: first(rs, 'KGRID1').__setitem__('k_snap_fired', False))),
        ('G-4', 'KGRID1 한 행의 dk 를 바꿈',
         _mut(lambda rs: first(rs, 'KGRID1').__setitem__('k_snap_dk', 0.99))),
        ('G-5 ', '파생 lambda_l1 을 1e-3 으로',
         _mut(lambda rs: first(rs, 'SHIP-NOW')['effective'].__setitem__('lambda_l1', 1e-3))),
        ('G-5 ', '파생 seed 를 est_seed 와 어긋나게',
         _mut(lambda rs: first(rs, 'SHIP-NOW')['effective'].__setitem__('seed', 7))),
        ('G-5 ', 'S8_L1_TRACK_ONLY 환경변수 흔적',
         _mut(lambda rs: first(rs, 'SHIP-NOW')['effective']['env_probe']
              .__setitem__('S8_L1_TRACK_ONLY', '1'))),
        ('G-5 ', '포착된 updater 가 0개(훅 미설치)',
         _mut(lambda rs: first(rs, 'SHIP-NOW')['effective'].__setitem__('n_captured', 0))),
        ('G-5b', 'init_loss_n = 7',
         _mut(lambda rs: first(rs, 'SHIP-NOW').__setitem__('init_loss_n', 7))),
        ('G-5c', 'varpro_steps = 0',
         _mut(lambda rs: first(rs, 'SHIP-NOW').__setitem__('varpro_steps', 0))),
        # ★V4 추가: 러너가 실제로 낼 수 있는 **가장 위험한** 행 —
        #   VARPRO 가 건너뛰어졌는데 `varpro_steps`(echo)=400 · `varpro_fired`(truthy)=True
        #   가 그대로 남는 경우. 초판 게이트는 이것을 **통과시켰다.**
        ('G-5c', '★VARPRO 건너뜀 (echo 400 · truthy True 유지, obj 만 없음)',
         _mut(lambda rs: first(rs, 'SHIP-NOW').update(varpro_obj_before=None,
                                                      varpro_obj_after=None))),
        ('G-5b', '★init_loss_n 키가 사라짐(None) — 초판은 통과했다',
         _mut(lambda rs: [r.__setitem__('init_loss_n', None) for r in rs])),
        ('G-6', '★auto_real_gate = True (실측 게이트 발화) — 초판은 통과했다',
         _mut(lambda rs: first(rs, 'SHIP-NOW').__setitem__('auto_real_gate', True))),
        ('G-7', '★n_gaussians = 0 (표현이 하나도 없다) — 초판은 통과했다',
         _mut(lambda rs: first(rs, 'SHIP-NOW').__setitem__('n_gaussians', 0))),
        ('G-8', '★채점 창 3키가 전 행 None — 초판은 통과했다',
         _mut(lambda rs: [r.update(tracking_frames=None, scored_t_first=None,
                                   scored_t_last=None) for r in rs])),
        ('G-6', 'auto_selected = v2_konly',
         _mut(lambda rs: first(rs, 'SHIP-NOW').__setitem__('auto_selected', 'v2_konly'))),
        ('G-6', '파생 init_method 가 auto 로 남음(라우팅 미발화)',
         _mut(lambda rs: first(rs, 'SHIP-NOW')['effective']
              .__setitem__('init_method', 'auto'))),
        ('G-7', 'n_gaussians = None',
         _mut(lambda rs: first(rs, 'SHIP-NOW').__setitem__('n_gaussians', None))),
        ('G-8', '한 행의 채점 창을 다르게',
         _mut(lambda rs: first(rs, 'KGRID2').__setitem__('tracking_frames', 120))),
        ('G-9', '한 행의 zero_nRMSE 를 None 으로',
         _mut(lambda rs: first(rs, 'KGRID2').__setitem__('zero_nRMSE', None))),
        ('★G-11', 'SHIP-NOW 한 행을 1e-12 만큼 어긋나게',
         _mut(lambda rs: first(rs, 'SHIP-NOW').__setitem__(
             'avg_nRMSE', first(rs, 'SHIP-NOW')['avg_nRMSE'] + 1e-12))),
        ('★G-12', 'zero_nRMSE 를 rtol 밖으로',
         _mut(lambda rs: [r.__setitem__('zero_nRMSE', r['zero_nRMSE'] * 1.001)
                          for r in rs if r['episode'] == rs[0]['episode']])),
        ('N-1', 'SHIP-NOW 를 전부 ZERO 위로',
         _mut(lambda rs: [r.__setitem__('avg_nRMSE', r['zero_nRMSE'] * 2)
                          for r in rs if r['arm'] == 'SHIP-NOW'])),
        ('N-2', 'KGRID1 을 전부 ZERO 위로',
         _mut(lambda rs: [r.__setitem__('avg_nRMSE', r['zero_nRMSE'] * 2)
                          for r in rs if r['arm'] == 'KGRID1'])),
    ]


def selftest(verbose=True):
    """전 게이트 자가진단. `(통과여부, 보고문자열)`. ⛔파일을 쓰지 않는다."""
    log = []

    def say(s):
        log.append(s)
        if verbose:
            print(s, flush=True)

    fails = []
    base = _fixture_rows()
    res = gates(base)
    bad = [n for n, okk, _ in res if not okk]
    say(f'[1/4] 픽스처 {len(base)}행 · 게이트 {len(res)}개 — '
        f'통과 {len(res) - len(bad)}/{len(res)}')
    for n, okk, d in res:
        if not okk:
            say(f'      ⛔{n}: {d}')
    if bad:
        fails.append(f'픽스처에서 통과하지 못한 게이트: {bad}')

    # ── 2. 공허한 통과: rows=[] 에서 `ROWLESS_OK` 밖 게이트가 하나도 통과하면 안 된다 ──
    empty = gates([])
    leaked = [n for n, okk, _ in empty
              if okk and not any(n.startswith(p) or p in n for p in ROWLESS_OK)]
    say(f'[2/4] rows=[] — 통과 {sum(1 for _, o, _ in empty if o)}/{len(empty)} '
        f'(허용 {list(ROWLESS_OK)}) · 공허한 통과 {len(leaked)}')
    for n in leaked:
        say(f'      ⛔공허한 통과: {n}')
    if leaked:
        fails.append(f'0행에서 통과한 게이트: {leaked}')

    # ── 3. 변형마다 해당 게이트가 FAIL 로 뒤집히는지 ─────────────────────────
    n_flip = 0
    for prefix, desc, mut in _perturbations():
        r2 = gates(mut(base))
        hit = [(n, okk) for n, okk, _ in r2 if n.startswith(prefix)]
        if not hit:
            fails.append(f'{prefix}: 그 이름의 게이트가 없다 ({desc})')
            say(f'      ⛔{prefix} 게이트 없음 — {desc}')
            continue
        if all(okk for _, okk in hit):
            fails.append(f'{prefix}: 변형에도 통과 — {desc}')
            say(f'      ⛔{prefix} 발화 안 함 — {desc}')
        else:
            n_flip += 1
    say(f'[3/4] 변형 {len(_perturbations())}건 중 게이트가 발화한 것 {n_flip}')

    # ── 4. 자료측 게이트가 **자료 부재/틀린 핀**에서 통과하지 않는지 ──────────
    #    (`ROWLESS_OK` 셋은 행이 없어도 판정하므로 [2/4] 가 면제해 준다. 그러니 이 셋만은
    #     **자기 입력**이 비었을 때 통과하지 않음을 따로 확인해야 한다 — 초판 G-10 의 결함.)
    step4 = []
    saved_eps, saved_mesh = globals()['episodes'], globals()['MESH_EXPECTED']
    try:
        globals()['episodes'] = lambda: []
        r3 = gates(base)
        if any(okk for n, okk, _ in r3 if n.startswith('G-10')):
            step4.append('G-10: 에피소드 0개에서 통과(공허)')
        if any(okk for n, okk, _ in r3 if 'G-13' in n):
            step4.append('G-13: 에피소드 0개에서 통과(공허)')
        globals()['episodes'] = saved_eps
        globals()['MESH_EXPECTED'] = {'ADEQUATE': 16}
        if any(okk for n, okk, _ in gates(base) if 'G-13' in n):
            step4.append('G-13: 틀린 핀에도 통과')
    finally:
        globals()['episodes'] = saved_eps
        globals()['MESH_EXPECTED'] = saved_mesh
    fails += step4
    say(f'[4/4] 자료 부재·틀린 핀에서 G-10·G-13 이 통과하지 않는다: '
        f'{"OK" if not step4 else "⛔ " + "; ".join(step4)}')

    okk = not fails
    say(('✅ 게이트 자가진단 통과' if okk else '⛔ 게이트 자가진단 실패:\n  - '
         + '\n  - '.join(fails)))
    return okk, '\n'.join(log)


if __name__ == '__main__':
    ok_, _ = selftest()
    sys.exit(0 if ok_ else 1)
