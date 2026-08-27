r"""B-25 단일출처 — **관측 신뢰가중(`use_confidence`)의 절제** (2026-08-27). ⛔결과 0행에서 동결.

★러너·분석기·자가진단이 **이 파일 하나**를 부른다(팔·문턱·게이트·라벨 전부).

## 질문
***"참 표면 법선에서 나온 관측 가중이 합성 헤드라인을 얼마나 지탱하는가?"***

## ★왜 이 양이 특별한가 — **코드에서 확인한 비대칭** (추론 아님)
| | 합성 경로 | 실측 경로 |
|---|---|---|
| 법선 | `pipeline.py:225-227` `interpolate_normals(..., NX_gt, NY_gt, NZ_gt)` | **없음** |
| 입사각 | `lidar_simulator.py:340-342` `cos θ = −ray·n` | `real_data.py:200` `cos θ = \|dz\|/\|ray\|` |
| ⟹ 가중이 담은 것 | ★**참 파면의 기울기장** | **센서-점 기하뿐**(수면을 평면으로 가정) |

`NX_gt,NY_gt,NZ_gt` 의 출처는 `pipeline.py:192/194` = `hos_sim.evaluate_at` · `evaluate_field`
**= 참값 그 자체**. ⟹ 합성의 `confidence` 는 **추정기가 알 수 없어야 할 양의 함수**이고,
실측의 `confidence` 는 같은 이름의 **다른 양**이다(`real_data.py:179` 축자:
*"assumed vertical for water surface"*).

⛔**단 "실측 경로에 구조적으로 존재할 수 없다" 는 과장이다** — 실측에도 `confidence` 는 있다
(`real_data.py:588`). 정확한 문장은 ***"실측의 가중은 기울기 정보를 담을 수 없다"*** 이다.

## 소비처 — **배포 경로에서 실제로 읽히는 두 곳** (전수)
1. `sequential_update.py:562-563` → `_compute_loss` 의 가중
   · Phase 1 `:893` 는 `learned_init_iters=0` 이라 **0회** · **추적 루프 `:1390` 이 진짜 소비처**
2. ★`varpro.py:118` `sqrt_c = sqrt(clamp(conf))` → **(kx,ky) 하강**에 들어간다
   (`:96-97` 이 설계행렬과 관측을 둘 다 가중, 호출 `sequential_update.py:918`)
   ⟹ ★**질적으로 새 소비처**다(2026-08-22 배포). 기존 완화 측정은 이 경로 **이전**이다.
⛔소비처는 이 둘뿐이다(전수 확인): `learned_init.py` 의 `ls_fit_wphi(:309-323)` 는 **무가중**,
  `_adaptive_prune_grow` 도 `confidence` 를 **읽지 않는다**(`grep confidence` = :335,:337,:435,:562,:563,:918).

## ★★설계 결함 하나를 결과 전에 잡았다 — `use_confidence=False` 는 **2요인** 개입이다
`_compute_loss` 는 **가중합을 무가중 개수로 나눈다**(`:564-570`):
    conf 켬 : (Σ cⱼ rⱼ²)/n + λ·Σ|w|     conf 끔 : (Σ rⱼ²)/n + λ·Σ|w|
저장소가 이 사실을 이미 축자로 적어 뒀다(`:335-337`): *"the data term is scaled by
mean(confidence) (~0.86) while the L1 and smoothness penalties are not"*.
⟹ 스위치를 끄면 **정보만 사라지는 게 아니라 자료항이 1/c̄ 배로 커진다** = **유효 λ 가 c̄ 배로 약해진다.**
**이 코퍼스 실측**: c̄ = 0.8652–0.8787 (중앙 **0.8706**) ⟹ **1/c̄ ≈ 1.149** (약 13%).
⚠W7 이 λ 용량-반응을 쟀다(λ=0 널 1.009× · 1e-4 **1.516×** · 1e-3 3.665×) ⟹ **λ 축은 살아 있다.**
⟹ ⛔**2팔(SHIP vs NOCONF)만으로는 "정보의 기여" 를 잴 수 없다.**

★**해법 = 질량보존 팔 `FLATCONF`**: 프레임마다 `confidence` 를 **그 프레임 자신의 평균**으로 치환.
  · Σⱼ c̄_f = Σⱼ cⱼ **정확히 성립** ⟹ 자료항 규모·유효 λ **불변**
  · 점별 편차 = 0 ⟹ ★**기울기 정보만 제거**된다  (std(conf) 실측 중앙 0.2337 = 지울 것이 실재)
⟹ **PRIMARY 는 `SHIP` vs `FLATCONF`**(1요인) 이고 `NOCONF` 는 배포 스위치의 서술로 내린다.

## ⛔이 배치가 **주장하지 않는 것**
1. ⛔**외부 기준선 비교가 아니다** — `SPEC25`·LETKF 행 **0개**. `B-1`·`B-12` 문장을 대체·수정하지 않는다.
2. ⛔**배포 기본값을 바꾸지 않는다** — 어떤 결과가 나와도 `use_confidence` 기본값을 이 배치로 돌리지 않는다.
3. ⛔**실측 판정을 뒤집지 않는다** — 층이 다르다(실측 가중은 애초에 다른 양이다).
4. ⛔**동등성(equivalence)을 주장하지 않는다** — §문턱 참조. 널은 *"효과 없음"* 이 아니라
   *"n=48 에서 검출 안 됨(MDE 명시)"* 으로만 쓴다.
"""
import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B22 = os.path.abspath(os.path.join(HERE, '..', 'B22_PRIOR_ABLATION_2026-08-27'))
B19 = os.path.abspath(os.path.join(HERE, '..', 'B19_KGRID_SPARSE_2026-08-25'))
B12 = os.path.abspath(os.path.join(HERE, '..', 'B12_HOS_EXTEND_2026-08-23'))
B11 = os.path.abspath(os.path.join(HERE, '..', 'B11_HOS_BASELINE_2026-08-23'))
W15 = os.path.abspath(os.path.join(HERE, '..', 'W15_DEPLOY_VERIFY_2026-08-22'))
for _p in (HERE, ROOT, B22, B19, B12, B11, W15):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── ⛔손복사 0: 표본·시드·비교함수·되읽기를 **동결 배치에서 import** ──────────
from gates_b12 import ALL_SEEDS, N_EP                             # noqa: E402,F401
from gates_b11 import HOS, EST_SEEDS                              # noqa: E402,F401
from gates_b19 import (zero_nrmse, sign_test_one_sided, cmp_pair,  # noqa: E402,F401
                       cmp_zero, effective_from_updater, DERIVED_KEYS,
                       DERIVED_EXPECTED, ENV_PROBE_KEYS,
                       UPDATER_CLASS_EXPECTED, TP_FACTOR)
# ★표본과 `SHIP` 팔 정의는 `B-22` 에서 통째로 물려받는다 ⟹ **양성대조가 정의상 같은 팔**이다.
from gates_b22 import (episodes, MESH_EXPECTED, deploy_constants,  # noqa: E402,F401
                       DEPLOY_EXPECTED, arm_config as _b22_arm_config,
                       ROUTE_LEARNED, ENSEMBLE_K_LEARNED, INIT_ITERS_LEARNED,
                       CROWDED_MIN_NCOMP)

# ─────────────────────────────────────────────────────────────────────────────
# 팔 (3) — ⛔빈 dict 금지. 기저는 `B-22` 의 `SHIP`(=배포) 이고 팔마다 **한 가지**만 바꾼다.
# ─────────────────────────────────────────────────────────────────────────────
#   conf_mode='raw'  → 로더가 준 `confidence` 그대로
#   conf_mode='flat' → 프레임마다 자기 평균으로 치환 (질량 보존, 정보 제거)
CONF_RAW, CONF_FLAT = 'raw', 'flat'

ARMS = {
    # 배포 그대로 (= `B-22` 의 `SHIP`) — 양성대조 기준행
    'SHIP':     dict(use_confidence=True,  conf_mode=CONF_RAW),
    # 배포 스위치 하나 (⚠2요인: 정보 제거 + 자료항 1/c̄ 배)
    'NOCONF':   dict(use_confidence=False, conf_mode=CONF_RAW),
    # ★질량보존 절제 (1요인: 정보만 제거)
    'FLATCONF': dict(use_confidence=True,  conf_mode=CONF_FLAT),
}
ARM_ORDER = ['SHIP', 'NOCONF', 'FLATCONF']
CONF_ON_ARMS = ('SHIP', 'FLATCONF')          # `updater.use_confidence` 가 True 여야 하는 팔
CONF_OFF_ARMS = ('NOCONF',)                  # False 여야 하는 팔
FLAT_ARMS = ('FLATCONF',)                    # 자료 훅이 **발화해야** 하는 팔
RAW_ARMS = ('SHIP', 'NOCONF')                # 훅이 값을 바꾸면 **안 되는** 팔
# ★VARPRO 는 세 팔 전부에서 발화한다(배포 400 스텝). 가중만 다르다.
VP_WEIGHTING_EXPECTED = {'SHIP': 'confidence', 'FLATCONF': 'confidence',
                         'NOCONF': 'none'}
VP_FALLBACK_EXPECTED = {'SHIP': None, 'FLATCONF': None,
                        'NOCONF': 'use_confidence=False'}

# ★실측한 코퍼스 상수 — **핀(트립와이어)**이지 판정에 쓰는 값이 아니다.
#   48 에피소드 전건에서 잰 값(결과 0행에서 측정, 유효 프레임만):
CONF_MEAN_RANGE = (0.86, 0.89)      # 실측 0.8652–0.8787
CONF_STD_MIN = 0.15                 # 실측 0.2274–0.2391 ⟹ **지울 정보가 실재**함의 하한
CONF_MASS_RTOL = 1e-5               # FLATCONF 의 프레임별 Σc 보존 허용오차 (float32)


def arm_config(arm, est_seed):
    """★팔의 **완성된 패키지 설정** — `B-22` 의 `SHIP` 설정 + 이 팔이 바꾸는 한 키.

    ⛔리터럴 손복사 0: 기저를 `gates_b22.arm_config('SHIP')` 에서 받아 온다
    ⟹ 출하 기본값이 바뀌면 세 팔이 **함께** 따라가고 `G-5d` 트립와이어가 멈춘다.
    ⚠`conf_mode` 는 패키지 키가 아니다 — 러너의 **자료 훅** 인자이므로 여기 넣지 않는다.
    """
    cfg = dict(_b22_arm_config('SHIP'))          # atoms=None = 배포 인라인 제안
    cfg['use_confidence'] = ARMS[arm]['use_confidence']   # ← 이 배치의 유일한 패키지 개입
    cfg['seed'] = est_seed
    return cfg


def conf_mode(arm):
    return ARMS[arm]['conf_mode']


def effective_b25(upd, n_captured=None):
    """★`B-19` 의 객체 되읽기 + **이 배치가 개입하는 속성**(`use_confidence`).

    `DERIVED_KEYS` 에 `use_confidence` 가 없다(`B-19` 는 이 축을 안 건드렸다) ⟹ 여기서 더한다.
    ⛔설정 dict 을 다시 읽지 않는다 — `SequentialUpdater.__init__:435` 가 만든 **속성**이다.
    ⚠`MedianEnsembleUpdater` 는 미지 속성을 member 0 으로 전달한다(`:1895 __getattr__`)
      ⟹ 두 클래스 어느 쪽이 잡혀도 같은 값을 읽는다.
    """
    out = effective_from_updater(upd, n_captured=n_captured)
    if out is not None:
        out['use_confidence'] = getattr(upd, 'use_confidence', '<missing>')
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 양성대조 — `B-22`(08-27) 의 동결 `SHIP` 행. ⛔비트 동일 요구.
# ─────────────────────────────────────────────────────────────────────────────
POSCTRL_RTOL = 0.0                   # 0 = 비트 동일


def _b22_rows():
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(B22, 'results_b22*.jsonl'))):
        if os.path.basename(p).startswith('SMOKE'):
            continue
        with open(p) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'error' not in r:
                    out.append(r)
    return out


def b22_ship():
    """{(episode, est_seed): avg_nRMSE} — 이 배치의 `SHIP` 이 **비트 동일**해야 한다."""
    return {(r['episode'], r['est_seed']): float(r['avg_nRMSE'])
            for r in _b22_rows() if r.get('arm') == 'SHIP'}


# ─────────────────────────────────────────────────────────────────────────────
# 문턱 — ⛔**동등성 마진 Δ 를 쓰지 않는다** (`B-21` 의 BLOCKER 부류를 원천 차단)
# ─────────────────────────────────────────────────────────────────────────────
# ★사유: 부호-TOST 판정은 Δ 와 자료의 순서통계량의 포함관계와 **동치**라, 자료를 손에 쥔 뒤
#   Δ 를 고르면 판정을 고르는 것과 같다. ⟹ 이 배치는 **방향 검정 + 명시된 MDE** 만 쓴다.
#   널은 *"효과 없음이 증명됐다"* 가 아니라 ***"n=48 · 양측 α=0.05 에서 검출되지 않았다"*** 이다.
ALPHA_TWO_SIDED = 0.05
FAMILY_ALPHA_TWO_SIDED = 0.025       # 주장 가족 둘(PRIMARY · SECONDARY) 중 PRIMARY 몫
MIN_N_USED = 44                      # 표본 손실이 크면 SAMPLE-LOSS-UNDECIDED
NEG_INVALIDATING = ('SHIP',)         # 이 팔이 무기술이면 배치 **무효**


def thresholds(n=N_EP):
    """(NOMINAL_HI, CONFIRM_HI, NOMINAL_LO, CONFIRM_LO) — **양측**. 코드가 계산한다."""
    hi_n = next((w for w in range(n + 1)
                 if sign_test_one_sided(w, n) <= ALPHA_TWO_SIDED / 2), None)
    hi_c = next((w for w in range(n + 1)
                 if sign_test_one_sided(w, n) <= FAMILY_ALPHA_TWO_SIDED / 2), None)
    return hi_n, hi_c, (n - hi_n if hi_n is not None else None), \
        (n - hi_c if hi_c is not None else None)


def mde_note(n=N_EP):
    """★결과 전에 동결하는 **검출력 서술** — 널을 읽을 때 반드시 함께 인용한다."""
    hi_n, hi_c, lo_n, lo_c = thresholds(n)
    return (f'n={n} 부호검정(양측). NOMINAL 은 승수 ≥{hi_n} 또는 ≤{lo_n}, '
            f'CONFIRM 은 ≥{hi_c} 또는 ≤{lo_c}. '
            f'⟹ **검출 가능한 최소 효과는 승률 {hi_n}/{n}={hi_n / n:.3f}** 이고, '
            f'그보다 작은 일관된 효과는 이 표본에서 **검출되지 않는다**. '
            f'⛔널을 "효과 없음" 으로 읽지 마라 — "이 MDE 위에서 검출 안 됨" 이다.')


# ─────────────────────────────────────────────────────────────────────────────
# 집계 (★단일출처: 게이트·분석기가 **같은 함수**를 부른다)
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(rows):
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
        route_ok = all(r.get('auto_selected') == ROUTE_LEARNED for r in rs)
        conf_ok = all((r.get('effective') or {}).get('use_confidence')
                      == ARMS[r['arm']]['use_confidence'] for r in rs)
        full = all((a, e) in val for a in ARM_ORDER)
        if route_ok and conf_ok and full:
            good.append(e)
    zero = {r['episode']: r['zero_nRMSE'] for r in ok
            if r['episode'] in good and r.get('zero_nRMSE') is not None}
    return dict(ok=ok, cell=cell, val=val, eps_all=eps_all, good=good,
                zero=zero, n_used=len(good))


def comparisons(val, good):
    """이 배치의 **모든 짝 비교** — 라벨 함수와 분석기가 같은 dict 을 쓴다."""
    return dict(
        # ★PRIMARY — 1요인(정보만). 방향 미예측 ⟹ 양측.
        P=cmp_pair(val, good, 'SHIP', 'FLATCONF'),
        # SECONDARY — 배포 스위치(2요인) 와 순수 질량효과
        S1=cmp_pair(val, good, 'SHIP', 'NOCONF'),
        S2=cmp_pair(val, good, 'FLATCONF', 'NOCONF'),
    )


def negatives(val, zero, good):
    return {a: cmp_zero(val, zero, good, a) for a in ARM_ORDER}


# ─────────────────────────────────────────────────────────────────────────────
# 게이트 (★단일출처 — 분석기는 `gates(rows)` 를 **호출만** 한다)
# ⛔**공허한 통과 금지**: 결과 0행에서 참이 되는 게이트를 두지 않는다(W8 `G-4` 부류).
# ─────────────────────────────────────────────────────────────────────────────
ROWLESS_OK = ('G-5d', 'G-12')        # 행 없이도 판정 가능(패키지·자료만 읽는다)


def _srt(vals):
    return sorted(vals, key=lambda x: (x is None, repr(x)))


def gates(rows, agg=None):
    """`[(이름, 통과여부, 상세)]`. 러너 산출 전체(오류 행 포함)를 받는다."""
    if agg is None:
        agg = aggregate(rows)
    ok, cell, val = agg['ok'], agg['cell'], agg['val']
    eps_all, good, zero = agg['eps_all'], agg['good'], agg['zero']
    out = []

    def g(name, cond, detail):
        out.append((name, bool(cond), str(detail)))

    def rows_of(arm):
        return [r for r in ok if r['arm'] == arm]

    def eff(r, k):
        return (r.get('effective') or {}).get(k)

    g('G-0 오류 0행', bool(rows) and len(ok) == len(rows),
      f'{len(rows) - len(ok)} 행 오류 / {len(rows)} 행')

    g(f'G-1 표본 = {len(ARM_ORDER)}팔 × {N_EP} 에피소드',
      len(val) == len(ARM_ORDER) * N_EP and len(eps_all) == N_EP,
      f'{len(val)}/{len(ARM_ORDER) * N_EP} 셀 · 에피소드 {len(eps_all)}/{N_EP}')

    bad_seed = {k: sorted(x['est_seed'] for x in v) for k, v in cell.items()
                if sorted(x['est_seed'] for x in v) != sorted(EST_SEEDS)}
    g('G-2 셀마다 추정기 시드 3개 전건', bool(cell) and not bad_seed,
      f'{len(cell) - len(bad_seed)}/{len(cell)} 셀 정상'
      + (f' · 이탈 예: {list(bad_seed.items())[:2]}' if bad_seed else ''))

    # ── G-3 ★개입 되읽기 ①: 추정기 **객체**의 `use_confidence` ────────────────
    #    ⛔`cfg[k]==cfg[k]` 아니다 — 왼쪽은 `SequentialUpdater.__init__:435` 가 만든 속성,
    #      오른쪽은 이 파일의 팔 정의다(러너의 설정 dict 을 다시 읽지 않는다).
    uc = {a: _srt({eff(r, 'use_confidence') for r in rows_of(a)}) for a in ARM_ORDER}
    exp_uc = {a: [ARMS[a]['use_confidence']] for a in ARM_ORDER}
    g('G-3 ★객체 되읽기: updater.use_confidence 가 팔 정의와 일치',
      bool(ok) and uc == exp_uc, f'{uc} (기대 {exp_uc})')

    # ── G-4 ★개입 되읽기 ②: **소비처가 스스로 증언**한다 (varpro 반환 rec) ────
    #    `varpro.py:218-220` 이 `use_confidence=False` 를 보고 스스로 남기는 값이다.
    #    ⚠`run.py:530-535` 는 이 두 키를 **버린다** ⟹ 러너가 관측 훅으로 붙잡는다.
    vw = {a: _srt({r.get('vp_weighting') for r in rows_of(a)}) for a in ARM_ORDER}
    vf = {a: _srt({r.get('vp_weighting_fallback') for r in rows_of(a)})
          for a in ARM_ORDER}
    exp_vw = {a: [VP_WEIGHTING_EXPECTED[a]] for a in ARM_ORDER}
    exp_vf = {a: [VP_FALLBACK_EXPECTED[a]] for a in ARM_ORDER}
    g('G-4 ★소비처 증언: varpro 가 실제로 쓴 가중 (vp_weighting / fallback)',
      bool(ok) and vw == exp_vw and vf == exp_vf,
      f'weighting {vw} · fallback {vf}')

    # ── G-4b ★자료 훅이 팔대로 발화했나 (프레임이 스스로 증언) ────────────────
    #    FLATCONF: 프레임별 std == 0 **정확히** · Σc 보존
    #    RAW 팔  : std > 0 (지울 정보가 실재했다는 증거이기도 하다)
    flat_bad = [(r['arm'], r['episode'], r['est_seed'], r.get('conf_std_max'),
                 r.get('conf_mass_rel_err_max')) for r in ok if r['arm'] in FLAT_ARMS
                if not (r.get('conf_std_max') == 0.0
                        and r.get('conf_mass_rel_err_max') is not None
                        and r['conf_mass_rel_err_max'] <= CONF_MASS_RTOL)]
    raw_bad = [(r['arm'], r['episode'], r['est_seed'], r.get('conf_std_max'))
               for r in ok if r['arm'] in RAW_ARMS
               if not (r.get('conf_std_max') is not None
                       and r['conf_std_max'] >= CONF_STD_MIN)]
    g('G-4b ★자료 훅 발화: FLATCONF std≡0 + 질량보존 · RAW std>0',
      bool(ok) and not flat_bad and not raw_bad,
      f'flat 이탈 {len(flat_bad)} · raw 이탈 {len(raw_bad)}'
      + (f' · 예 {(flat_bad + raw_bad)[:2]}' if (flat_bad or raw_bad) else ''))

    # ── G-4c ★훅이 **모든 프레임**을 지났나 (부분 적용 = 조용한 제3의 팔) ─────
    hook_bad = [(r['arm'], r['episode'], r.get('conf_frames_seen'),
                 r.get('conf_frames_changed')) for r in ok
                if not (r.get('conf_frames_seen')
                        and r.get('conf_frames_changed')
                        == (r['conf_frames_seen'] if r['arm'] in FLAT_ARMS else 0))]
    g('G-4c ★훅 적용 범위: FLATCONF 는 전 프레임 치환 · RAW 는 0 프레임',
      bool(ok) and not hook_bad,
      f'이탈 {len(hook_bad)}/{len(ok)} 행' + (f' · 예 {hook_bad[:2]}' if hook_bad else ''))

    # ── G-5 ★양성대조: `SHIP` 이 `B-22` 동결 행과 **비트 동일** ───────────────
    ref = b22_ship()
    sh = rows_of('SHIP')
    miss = [(r['episode'], r['est_seed']) for r in sh
            if (r['episode'], r['est_seed']) not in ref]
    dev = [(r['episode'], r['est_seed'], r['avg_nRMSE'],
            ref[(r['episode'], r['est_seed'])]) for r in sh
           if (r['episode'], r['est_seed']) in ref
           and r['avg_nRMSE'] != ref[(r['episode'], r['est_seed'])]]
    g('G-5 ★양성대조: SHIP == B-22 동결 SHIP (비트 동일)',
      bool(sh) and not miss and not dev,
      f'대조 {len(sh) - len(miss)}/{len(sh)} 행 · 이탈 {len(dev)} · 참조없음 {len(miss)}'
      + (f' · 예 {dev[:2]}' if dev else ''))

    # ── G-5b ★반-no-op: 개입 팔이 SHIP 과 **비트 동일하면 안 된다** ───────────
    #    ⛔"설정을 넣었다 ≠ 설정이 먹었다"(W7). 훅·스위치가 조용히 무력화되면 여기서 잡힌다.
    noop = {}
    for a in ('NOCONF', 'FLATCONF'):
        same = [e for e in eps_all if (a, e) in val and ('SHIP', e) in val
                and val[(a, e)] == val[('SHIP', e)]]
        noop[a] = len(same)
    g('G-5b ★반-no-op: NOCONF·FLATCONF 가 SHIP 과 비트 동일한 셀 0개',
      bool(eps_all) and all(v == 0 for v in noop.values()), f'{noop} (기대 전부 0)')

    # ── G-5c 배포 파생 속성(λ·추적모드·라우팅·앙상블·VARPRO)이 배포값 그대로 ──
    bad_der = {k: _srt({eff(r, k) for r in ok}) for k, v in DERIVED_EXPECTED.items()
               if _srt({eff(r, k) for r in ok}) != [v]}
    route = _srt({r.get('auto_selected') for r in ok})
    vps = _srt({r.get('varpro_steps') for r in ok})
    vpf = _srt({bool(r.get('varpro_fired')) for r in ok})
    ek = _srt({r.get('ensemble_k') for r in ok})
    ii = _srt({r.get('init_iters') for r in ok})
    g('G-5c 배포 구성 불변 (λ·추적·라우팅·VARPRO·앙상블·반복)',
      bool(ok) and not bad_der and route == [ROUTE_LEARNED]
      and vps == [DEPLOY_EXPECTED['varpro_steps']] and vpf == [True]
      and ek == [ENSEMBLE_K_LEARNED] and ii == [INIT_ITERS_LEARNED],
      f'파생이탈 {bad_der} · route {route} · varpro {vps}/{vpf} · '
      f'ens {ek} · iters {ii}')

    # ── G-5d 배포 상수 트립와이어 (패키지 단일출처 == 이 배치가 전제한 값) ────
    dep = deploy_constants()
    mism = {k: (dep.get(k), v) for k, v in DEPLOY_EXPECTED.items() if dep.get(k) != v}
    g('G-5d 배포 상수 == 사전등록이 전제한 값', not mism, f'이탈 {mism or "없음"}')

    # ── G-6 채점 창이 세 팔에서 같다 (`B-13` 부류 재발 방지) ──────────────────
    win = {}
    for e in eps_all:
        for r in ok:
            if r['episode'] == e:
                win.setdefault(e, set()).add(
                    (r.get('tracking_frames'), r.get('scored_t_first'),
                     r.get('scored_t_last')))
    bad_win = {e: v for e, v in win.items() if len(v) != 1}
    g('G-6 채점 창이 팔 사이에서 동일', bool(win) and not bad_win,
      f'{len(win) - len(bad_win)}/{len(win)} 에피소드 일치'
      + (f' · 이탈 {list(bad_win)[:2]}' if bad_win else ''))

    # ── G-7 환경 프로브: 연구용 게이트가 꺼져 있다 ────────────────────────────
    envs = {k: _srt({(eff(r, 'env_probe') or {}).get(k) for r in ok})
            for k in ENV_PROBE_KEYS}
    g('G-7 연구 환경변수 전부 unset', bool(ok) and all(v == [None] for v in envs.values()),
      f'{envs}')

    # ── G-8 ZERO 가 참 제로필드에서 계산됐다 (⛔상수 1.0 금지) ────────────────
    zs = [r.get('zero_nRMSE') for r in ok if r.get('zero_nRMSE') is not None]
    g('G-8 ZERO = 참 제로필드 (상수 1.0 아님)',
      bool(zs) and len(zs) == len(ok) and len(set(zs)) > 1
      and not any(z == 1.0 for z in zs),
      f'{len(zs)}/{len(ok)} 행 · 고유값 {len(set(zs))} · 범위 '
      f'[{min(zs):.4f}, {max(zs):.4f}]' if zs else '값 없음')

    # ── G-9 에피소드마다 ZERO 가 팔 사이에서 동일 ────────────────────────────
    zz = {}
    for r in ok:
        zz.setdefault(r['episode'], set()).add(r.get('zero_nRMSE'))
    bad_z = {e: v for e, v in zz.items() if len(v) != 1}
    g('G-9 ZERO 가 팔 사이에서 동일 (자료 훅이 채점을 오염시키지 않았다)',
      bool(zz) and not bad_z, f'{len(zz) - len(bad_z)}/{len(zz)} 에피소드 일치')

    # ── G-10 임포트 경로 md5 가 배치 안에서 불변 (더러운 작업트리 탐지) ───────
    md5s = {json.dumps(r.get('import_md5'), sort_keys=True) for r in ok}
    g('G-10 임포트 경로 md5 불변', bool(ok) and len(md5s) == 1,
      f'고유 md5 묶음 {len(md5s)}')

    # ── G-11 유효 표본 ───────────────────────────────────────────────────────
    g(f'G-11 유효 에피소드 >= {MIN_N_USED}', agg['n_used'] >= MIN_N_USED,
      f'{agg["n_used"]}/{N_EP}')

    # ── G-12 표본의 비선형성 실현률 == 핀 (HDF5 에서 재도출) ─────────────────
    try:
        import h5py
        cen = {}
        for e in episodes():
            with h5py.File(e['path'], 'r') as h:
                v = h['config'].attrs.get('mesh_verdict')
                v = v.decode() if isinstance(v, bytes) else str(v)
            cen[v] = cen.get(v, 0) + 1
        g('G-12 표본 비선형성 실현률 == 핀', bool(cen) and cen == MESH_EXPECTED,
          f'{cen} (핀 {MESH_EXPECTED})')
    except Exception as ex:                                   # noqa: BLE001
        g('G-12 표본 비선형성 실현률 == 핀', False, f'읽기 실패 {ex!r}')

    # ── G-13 코퍼스 신뢰가중 통계가 핀 안에 (지울 정보가 실재한다) ───────────
    cm = [r.get('conf_mean_raw') for r in ok if r.get('conf_mean_raw') is not None]
    cs = [r.get('conf_std_raw') for r in ok if r.get('conf_std_raw') is not None]
    g('G-13 코퍼스 c̄·std 가 핀 안에 (개입할 대상이 실재)',
      bool(cm) and bool(cs) and len(cm) == len(ok)
      and CONF_MEAN_RANGE[0] <= min(cm) and max(cm) <= CONF_MEAN_RANGE[1]
      and min(cs) >= CONF_STD_MIN,
      f'c̄ [{min(cm):.4f}, {max(cm):.4f}] (핀 {CONF_MEAN_RANGE}) · '
      f'std min {min(cs):.4f} (핀 >= {CONF_STD_MIN})' if cm and cs else '값 없음')

    # ── N[arm] 음성대조: 각 팔이 참 제로필드보다 나은가 ──────────────────────
    # ⚠`thresholds` 는 표본이 너무 작으면 `None` 을 돌려준다(그 n 에서는 어떤 승수로도
    #   α 에 닿지 못한다). ⟹ **크래시도 공허한 통과도 아니라 FAIL** 이어야 한다.
    #   (자가진단 픽스처는 n=48 이라 이 가지를 못 밟는다 — 스모크 실행이 잡았다.)
    hi_n, _hi_c, lo_n, _lo_c = thresholds(len(good) or N_EP)
    for a in ARM_ORDER:
        c = cmp_zero(val, zero, good, a)
        cond = (hi_n is not None and c['n'] > 0 and c['better'] >= hi_n)
        g(f'N[{a}] {a} < ZERO (참 제로필드)', cond,
          f"{c['better']}/{c['n']} · p={c['p_one_sided']} · "
          f"rel {c['rel_median_pct']}" if c['n'] else '표본 0')

    return out


# ─────────────────────────────────────────────────────────────────────────────
# ★라벨 — **결과 전 확정**. 어느 결과가 나와도 무엇을 쓸지 여기 적혀 있다.
# ─────────────────────────────────────────────────────────────────────────────
L_NULL = 'GT-WEIGHT-NOT-DETECTED'
L_HURT = 'GT-WEIGHT-HURTS'
L_LEAK = 'GT-WEIGHT-EARNS'
L_VOID = 'VOID'
L_UNDEC = 'SAMPLE-LOSS-UNDECIDED'

# ★각 라벨이 **논문에 무엇을 하는가** (결과 전 확정, ⛔결과를 보고 고르지 않는다)
ACTIONS = {
    L_NULL: (
        '★*"참 표면 법선에서 나온 가중은 헤드라인을 지탱하지 않는다"* — 단 **검출 한계 안에서**.\n'
        '  ⟹ ① `B-18` 잔여 후보에서 이 축을 **뺀다**(sim/real 격차 설명에서 제외).\n'
        '  ⟹ ② 논문: 헤드라인이 GT 유래 가중에 기대지 않는다고 **적을 수 있다**.\n'
        '       ⛔단 반드시 MDE 를 함께 적는다(§문턱). *"효과가 없다"* 라고 쓰지 마라.\n'
        '  ⟹ ③ ⛔`use_confidence` 를 **기여로 주장하지 마라** — 절제해도 안 바뀌는 것은 기여가 아니다.'),
    L_HURT: (
        '★*"기하 신뢰가중은 오히려 해친다"* — 헤드라인은 무손상(오히려 과소평가).\n'
        '  ⟹ ① 논문에서 **"기하 인식 신뢰 가중" 을 기여·설계 결정에서 완전히 내린다**.\n'
        '  ⟹ ② `B-18` 잔여 후보에서 뺀다(GT 누출이 이득을 만들지 않았다).\n'
        '  ⟹ ③ ⛔배포 기본값을 이 배치로 바꾸지 않는다(별도 사전등록이 필요하다).'),
    L_LEAK: (
        '⛔★**가장 나쁜 경우 — 헤드라인 크기에 GT 누출 몫이 있다.**\n'
        '  ⟹ ① `SHIP` − `FLATCONF` 의 상대 중앙값을 ***"GT 유래 가중이 기여한 몫의 추정"***\n'
        '       으로 **논문 본문에 적는다**(범위 절 + 한계 절). ⛔각주로 숨기지 마라.\n'
        '  ⟹ ② 헤드라인 인용 시 **"이 중 X% 는 합성에서만 가능한 관측 가중에서 온다"** 를 동반한다.\n'
        '  ⟹ ③ 실측 비대칭을 **코드 인용과 함께** 적는다(`pipeline.py:225-227` vs `real_data.py:200`)\n'
        '       — 실측 가중은 같은 이름의 **다른 양**이라 이 몫을 옮길 수 없다.\n'
        '  ⟹ ④ `B-18` 잔여 공간에서 이 축을 **살아있는 설명으로 승격**한다.'),
    L_VOID: '⛔배치 무효 — `SHIP` 이 참 제로필드를 못 이긴다. 어떤 짝 비교도 읽지 않는다.',
    L_UNDEC: '⛔표본 손실 과다 — 판정하지 않는다.',
}


def label_primary(cmpx, negs, n_used):
    """PRIMARY 라벨 (양측). ⛔동등성 마진 없음 — 널은 *검출 실패*로만 읽는다."""
    def _no_skill(a):
        t = thresholds(negs[a]['n'])[0]
        return negs[a]['n'] == 0 or t is None or negs[a]['better'] < t

    if any(_no_skill(a) for a in NEG_INVALIDATING):
        return L_VOID, 'ZERO', None
    if n_used < MIN_N_USED:
        return L_UNDEC, 'N', None
    c = cmpx['P']
    hi_n, hi_c, lo_n, lo_c = thresholds(c['n'])
    if c['better'] >= hi_c:
        return L_LEAK, 'CONFIRMATORY', c
    if c['better'] >= hi_n:
        return L_LEAK, 'NOMINAL', c
    if c['better'] <= lo_c:
        return L_HURT, 'CONFIRMATORY', c
    if c['better'] <= lo_n:
        return L_HURT, 'NOMINAL', c
    return L_NULL, 'NOT-DETECTED', c


# ─────────────────────────────────────────────────────────────────────────────
# ★런 전 자가진단 — 가짜 행으로 게이트가 **실제로 발화하는지** 확인 (런 0회, 파일 0개)
# ⚠공개: 픽스처가 `B-22` 동결 `SHIP` 행을 쓰므로 `N[SHIP]` 값이 런 전에 도출 가능하다.
#   그것은 새 정보가 아니다(동결 배치에서 이미 연역된다). PRIMARY 는 픽스처의 **임의 배수**라
#   아무 정보도 담지 않는다.
# ─────────────────────────────────────────────────────────────────────────────
FIXTURE_MULT = {'SHIP': 1.0, 'NOCONF': 1.10, 'FLATCONF': 1.06}


def _fixture_rows():
    ship, eps = b22_ship(), episodes()
    zref = {}
    for r in _b22_rows():
        if r.get('zero_nRMSE') is not None:
            zref[r['episode']] = float(r['zero_nRMSE'])
    rows = []
    for e in eps:
        name = e['episode']
        for arm in ARM_ORDER:
            flat = arm in FLAT_ARMS
            for s in EST_SEEDS:
                rows.append(dict(
                    arm=arm, est_seed=s, episode=name, bf=e['bf'],
                    ncomp=e['ncomp'], gen_seed=e['gen_seed'],
                    avg_nRMSE=ship[(name, s)] * FIXTURE_MULT[arm],
                    # 소비처 증언
                    vp_weighting=VP_WEIGHTING_EXPECTED[arm],
                    vp_weighting_fallback=VP_FALLBACK_EXPECTED[arm],
                    # 자료 훅 계수기
                    conf_frames_seen=140, conf_frames_changed=(140 if flat else 0),
                    conf_std_max=(0.0 if flat else 0.2337),
                    conf_mean_raw=0.8706, conf_std_raw=0.2337,
                    conf_mass_rel_err_max=(0.0 if flat else None),
                    # 배포 계수기
                    auto_selected=ROUTE_LEARNED, auto_n_selected=18,
                    ensemble_k=ENSEMBLE_K_LEARNED, ensemble_member_n=[18],
                    n_gaussians=18, init_iters=INIT_ITERS_LEARNED, init_loss_n=0,
                    varpro_steps=DEPLOY_EXPECTED['varpro_steps'], varpro_fired=True,
                    tracking_frames=160, scored_t_first=14.1, scored_t_last=30.0,
                    episode_total_frames=301,
                    zero_nRMSE=zref.get(name, 0.9), zero_n_frames=160,
                    effective=dict(lambda_l1=1e-4, tracking_mode='w_phi',
                                   tracking_lr=0.005, init_method=ROUTE_LEARNED,
                                   real_estimator='bank_mmse', max_steepness=0.15,
                                   seed=s, depth=50.0,
                                   use_confidence=ARMS[arm]['use_confidence'],
                                   updater_class=UPDATER_CLASS_EXPECTED,
                                   primary_class='SequentialUpdater', n_captured=1,
                                   env_probe={k: None for k in ENV_PROBE_KEYS}),
                    cfg_intent={'⚠': '의도표 — 증거 아님'},
                    git_head='fixture', import_md5={'x': 'y'}, wall_s=1.0))
    return rows


def _perturbations():
    """(게이트 접두어, 설명, 변형) — 각 변형은 그 게이트를 **FAIL 로 뒤집어야** 한다."""
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
        ('G-0', '오류 행 1개', _mut(lambda rs: rs.append(dict(
            arm='SHIP', episode='x', est_seed=42, error='boom')))),
        # ⚠자가진단이 초판을 잡았다: **한 행**을 지우면 셀은 남아(중앙값 2개) G-1 이 안 걸린다.
        #   G-1 이 주장하는 것은 *셀* 의 수이므로 변형도 **셀 하나(시드 3개)** 를 지운다.
        ('G-1', '셀 하나(시드 3개 전부) 삭제', _mut(
            lambda rs: [rs.remove(r) for r in list(rs)
                        if r['arm'] == 'FLATCONF'
                        and r['episode'] == rs[0]['episode']])),
        ('G-2', '시드 하나를 중복으로', _mut(
            lambda rs: first(rs, 'NOCONF').__setitem__('est_seed', 43))),
        ('G-3', '★NOCONF 객체가 use_confidence=True 로 (스위치가 안 먹었다)', _mut(
            lambda rs: first(rs, 'NOCONF')['effective'].__setitem__(
                'use_confidence', True))),
        ('G-4', '★varpro 가 NOCONF 에서도 confidence 로 가중 (소비처 미도달)', _mut(
            lambda rs: first(rs, 'NOCONF').__setitem__('vp_weighting', 'confidence'))),
        ('G-4', '★fallback 이 SHIP 에서 발화 (교차 오염)', _mut(
            lambda rs: first(rs, 'SHIP').__setitem__(
                'vp_weighting_fallback', 'use_confidence=False'))),
        ('G-4b', '★FLATCONF 인데 std>0 (훅이 값을 안 바꿨다)', _mut(
            lambda rs: first(rs, 'FLATCONF').__setitem__('conf_std_max', 0.2))),
        ('G-4b', '★FLATCONF 가 질량을 안 지켰다', _mut(
            lambda rs: first(rs, 'FLATCONF').__setitem__(
                'conf_mass_rel_err_max', 0.3))),
        ('G-4b', '★RAW 팔의 std 가 0 (원본이 이미 평평 = 지울 정보가 없었다)', _mut(
            lambda rs: first(rs, 'SHIP').__setitem__('conf_std_max', 0.0))),
        ('G-4c', '★훅이 일부 프레임만 지났다', _mut(
            lambda rs: first(rs, 'FLATCONF').__setitem__('conf_frames_changed', 70))),
        ('G-4c', '★RAW 팔에서 훅이 값을 바꿨다', _mut(
            lambda rs: first(rs, 'NOCONF').__setitem__('conf_frames_changed', 140))),
        ('G-5 ', '★SHIP 이 동결 행과 다르다 (양성대조 실패)', _mut(
            lambda rs: first(rs, 'SHIP').__setitem__('avg_nRMSE', 0.12345))),
        ('G-5b', '★FLATCONF 가 SHIP 과 비트 동일 (개입이 no-op)', _mut(
            lambda rs: [r.__setitem__('avg_nRMSE', next(
                x['avg_nRMSE'] for x in rs
                if x['arm'] == 'SHIP' and x['episode'] == r['episode']
                and x['est_seed'] == r['est_seed']))
                for r in rs if r['arm'] == 'FLATCONF'])),
        ('G-5c', '★λ 가 배포값이 아니다', _mut(
            lambda rs: first(rs, 'SHIP')['effective'].__setitem__('lambda_l1', 1e-3))),
        ('G-5c', '★VARPRO 가 발화하지 않았다', _mut(
            lambda rs: first(rs, 'FLATCONF').__setitem__('varpro_fired', False))),
        ('G-6', '★한 팔이 다른 창에서 채점됐다', _mut(
            lambda rs: first(rs, 'NOCONF').__setitem__('tracking_frames', 120))),
        ('G-7', '★연구 환경변수가 켜져 있다', _mut(
            lambda rs: first(rs, 'SHIP')['effective']['env_probe'].__setitem__(
                'S8_L1_PHASE1_ONLY', '1'))),
        ('G-8', '★ZERO 가 상수 1.0', _mut(
            lambda rs: [r.__setitem__('zero_nRMSE', 1.0) for r in rs])),
        ('G-9', '★자료 훅이 채점 기준을 오염시켰다', _mut(
            lambda rs: first(rs, 'FLATCONF').__setitem__('zero_nRMSE', 0.5))),
        ('G-10', '★배치 중 임포트 경로가 바뀌었다', _mut(
            lambda rs: first(rs, 'SHIP').__setitem__('import_md5', {'x': 'CHANGED'}))),
        ('G-13', '★코퍼스 c̄ 가 핀 밖 (다른 코퍼스를 돌렸다)', _mut(
            lambda rs: first(rs, 'SHIP').__setitem__('conf_mean_raw', 0.5))),
        ('N[SHIP]', 'SHIP 을 전부 ZERO 위로', _mut(
            lambda rs: [r.__setitem__('avg_nRMSE', r['zero_nRMSE'] * 2)
                        for r in rs if r['arm'] == 'SHIP'])),
    ]


def selftest(verbose=True):
    """전 게이트 자가진단. `(통과여부, 보고)`. ⛔파일을 쓰지 않는다."""
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

    empty = gates([])
    leaked = [n for n, okk, _ in empty
              if okk and not any(p in n for p in ROWLESS_OK)]
    say(f'[2/4] rows=[] — 통과 {sum(1 for _, o, _ in empty if o)}/{len(empty)} '
        f'(허용 {list(ROWLESS_OK)}) · 공허한 통과 {len(leaked)}')
    for n in leaked:
        say(f'      ⛔공허한 통과: {n}')
    if leaked:
        fails.append(f'0행에서 통과한 게이트: {leaked}')

    n_flip = 0
    for prefix, desc, mut in _perturbations():
        r2 = gates(mut(base))
        hit = [(n, okk) for n, okk, _ in r2 if n.startswith(prefix)]
        if not hit:
            fails.append(f'{prefix}: 그 이름의 게이트가 없다 ({desc})')
            say(f'      ⛔{prefix} 게이트 없음 — {desc}')
        elif all(okk for _, okk in hit):
            fails.append(f'{prefix}: 변형에도 통과 — {desc}')
            say(f'      ⛔{prefix} 발화 안 함 — {desc}')
        else:
            n_flip += 1
    say(f'[3/4] 변형 {len(_perturbations())}건 중 게이트가 발화한 것 {n_flip}')

    # ── 4. 라벨 함수가 **세 결과를 전부** 낼 수 있는지 (결과 전 확인) ────────
    #    ⚠자가진단이 초판을 잡았다: `SHIP` 을 키우면 음성대조가 먼저 깨져 **VOID** 가 나온다
    #      (실측 여유가 좁다 — `SHIP<ZERO` 를 전건 유지하는 최대 배수가 0.67 이다).
    #      ⟹ 라벨 도달성은 **`FLATCONF` 쪽을 움직여** 확인한다(음성대조 무손상).
    agg = aggregate(base)
    negs = negatives(agg['val'], agg['zero'], agg['good'])
    seen = {}

    def _try(name, fn):
        import copy
        rs = copy.deepcopy(base)
        fn(rs)
        a2 = aggregate(rs)
        n2 = negatives(a2['val'], a2['zero'], a2['good'])
        lab, tier, c = label_primary(comparisons(a2['val'], a2['good']),
                                     n2, a2['n_used'])
        seen[lab] = f"{name} → {lab}/{tier} (better {c['better'] if c else '-'}/48)"
        return lab

    # FLATCONF 를 나쁘게 ⟹ SHIP 이 이긴다 ⟹ 누출
    _try('FLATCONF ×1.20', lambda rs: [r.__setitem__('avg_nRMSE', r['avg_nRMSE'] * 1.20)
                                       for r in rs if r['arm'] == 'FLATCONF'])
    # FLATCONF 를 좋게 ⟹ 가중이 해친다
    _try('FLATCONF ×0.90', lambda rs: [r.__setitem__('avg_nRMSE', r['avg_nRMSE'] * 0.90)
                                       for r in rs if r['arm'] == 'FLATCONF'])
    # 에피소드마다 부호를 번갈아 ⟹ 승수가 문턱 사이 ⟹ 검출 안 됨
    eporder = sorted({r['episode'] for r in base})
    half = {e: (0.90 if i % 2 == 0 else 1.10) for i, e in enumerate(eporder)}
    _try('FLATCONF 교대 ±10%',
         lambda rs: [r.__setitem__('avg_nRMSE', r['avg_nRMSE'] * half[r['episode']])
                     for r in rs if r['arm'] == 'FLATCONF'])
    # SHIP 을 ZERO 위로 ⟹ 무효
    _try('SHIP > ZERO', lambda rs: [r.__setitem__('avg_nRMSE', r['zero_nRMSE'] * 2)
                                    for r in rs if r['arm'] == 'SHIP'])
    lab0, tier0, _ = label_primary(comparisons(agg['val'], agg['good']),
                                   negs, agg['n_used'])
    for k, v in seen.items():
        say(f'      · {v}')
    seen = set(seen) | {lab0}
    need = {L_LEAK, L_HURT, L_NULL, L_VOID}
    say(f'[4/4] 라벨 함수가 낸 결과 {sorted(seen)} · 기저 픽스처 = {lab0}/{tier0}')
    if not need <= seen:
        fails.append(f'라벨 함수가 {sorted(need - seen)} 를 낼 수 없다')

    hi_n, hi_c, lo_n, lo_c = thresholds(N_EP)
    say(f'문턱(n={N_EP}, 양측): NOMINAL >={hi_n} 또는 <={lo_n} · '
        f'CONFIRM >={hi_c} 또는 <={lo_c}')
    say(mde_note(N_EP))
    ok_all = not fails
    say('✅자가진단 통과' if ok_all else f'⛔자가진단 실패 {len(fails)}건: {fails}')
    return ok_all, log


if __name__ == '__main__':
    sys.exit(0 if selftest()[0] else 1)
