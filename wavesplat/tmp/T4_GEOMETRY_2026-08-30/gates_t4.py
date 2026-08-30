#!/usr/bin/env python3
"""T-4 게이트·문턱·라벨·해석 규칙 — ★**판정의 유일한 출처** (2026-08-30, 결과 0행에서 작성).

    python3 tmp/T4_GEOMETRY_2026-08-30/gates_t4.py     # 자가진단 (자료 0행 · 런 0회)

⛔`PREREG.md` 와 이 파일이 어긋나면 **이 파일이 권위**다.
⛔손복사 상수 0건: 코퍼스·시드·팔·셸 경계는 동결 배치에서 **임포트**하고, 문턱 다섯 수는
   `sign_test_one_sided` 에서 **재도출**한다.

이 파일은 **두 부분**이다.

  **A부 = 축 선별기(`screen()`)** — 임무가 지정한 축(「관측 기하」)의 성분을 결과 전에
  열거하고, 결과 전에 동결한 **다섯 기준**으로 각각 GO / NO-GO 를 낸다. ★이것이
  *"설계를 접는다"* 는 판단의 **기계적 권위**다. ⛔산문으로 접지 않는다.

  **B부 = `T4-LEAD` 판정기** — A부가 유일하게 GO 를 낸 성분(**관측 시각과 채점 시각의
  상대 기하**)의 문턱·밴드·라벨·게이트·해석 규칙.

★**T2 개정 0 의 교훈**(`tmp/T2_STATIONARITY_2026-08-30/AMENDMENT_0_PREFREEZE.md`):
***통과할 수 없는 검사기는 검사기가 아니라 차단기다.*** ⟹ 이 파일의 자가진단은 게이트마다
**양방향**을 시험한다 — ①깨끗한 픽스처에서 PASS ②위조 입력에서 **반드시 FAIL**.
선별기도 같다: 다섯 기준을 전부 만족하는 **위조 후보**를 넣으면 반드시 `GO` 가 나와야 한다
(⛔항상 NO-GO 를 찍는 선별기는 선별기가 아니다).

★검사기 규약(`M8b F-2` 처방 (b)): 무인자 로더 `load_rows()` 를 `glob.glob` **무별칭** +
`json.loads` 로 노출한다.
"""
import ast
import glob
import json
import math
import os
import statistics as st
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B11 = os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23')
B12 = os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23')
B3 = os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23')
S16 = os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')
T2 = os.path.join(ROOT, 'tmp', 'T2_STATIONARITY_2026-08-30')
M10 = os.path.join(ROOT, 'tmp', 'M10_GEOMETRY_2026-08-30')
for _p in (HERE, B11, B12, B3, S16, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 코퍼스·시드·팔 — ⛔손복사 금지, 동결물에서 임포트 ─────────────────────────
from gates_b11 import HOS, EST_SEEDS, FILTER_SEEDS, SPEC          # noqa: E402,F401
from gates_b12 import ALL_SEEDS, N_EP                             # noqa: E402
import run_s16 as R16                                             # noqa: E402
import gates_b3 as GB3                                            # noqa: E402

SHELL_KEYS = list(R16.SHELL_KEYS)          # 단일출처(서술 전용 — B부는 셸을 판정에 쓰지 않는다)
CROWDED_MIN_NCOMP = 15                     # `run_b12.crowded_episodes` 의 조건절
ARM_ORDER = ('SHIP', 'SPEC25')
B12_FROZEN = os.path.join(B12, 'results_b12.jsonl')
B3_FROZEN = os.path.join(B3, 'results_b3.jsonl')

# ═════════════════════════════════════════════════════════════════════════════
# A부 — 축 선별기.  ⛔결과 0행에서 동결.  ★이것이 "접는다" 의 권위다.
# ═════════════════════════════════════════════════════════════════════════════
#
# 다섯 기준.  임무 브리핑의 셋(①미측정 ②단일요인 ③배포 경로)에 둘을 **명시적으로**
# 더했다 — 더한 이유를 여기 적는다(사후 추가가 아니라 결과 전 선언이다):
#
#   ④`in_budget`        — 임무 규율 9 가 ⛔GPU 금지 · CPU 직렬 · 6샤드 이하를 못박았다.
#                          에피소드 재생성이 필요한 성분은 이 제약에서 **원리적으로**
#                          배치가 되지 않으므로, 그것을 「설계 가능」이라 부르면 거짓이다.
#   ⑤`real_counterpart` — 이 축의 질문은 ***합성 양성과 실측 음성의 경계***다.
#                          실측 하네스에 대응물이 **없는** 좌표는, 어떤 값이 나와도
#                          그 질문에 답하지 못한다(합성 내부 강건성 결과가 된다).
#                          ⚠**공시**: 이 기준은 *자연*의 성질이 아니라 **우리 실측
#                          하네스**(`scene_station.scene_station`)의 성질이다. 그 함수는
#                          유효 격자점의 **균등 추첨**이라 광선·시선·센서 위치가 아예 없다.
#                          ⟹ 이 기준으로 떨어진 성분은 *"측정 불가"* 가 아니라
#                          *"이 자료로는 sim-real 다리를 놓지 못한다"* 이다.
#
# ⛔이 다섯을 결과를 보고 바꾸지 않는다.  ⛔후보 목록에서 항목을 빼지 않는다.

SCREEN_CRITERIA = ('unmeasured', 'single_factor', 'deployment_path',
                   'in_budget', 'real_counterpart')

CANDIDATES = (
    dict(name='APERTURE', ko='공간 개구(관측 발자국 크기)',
         unmeasured=False, single_factor=False, deployment_path=True,
         in_budget=True, real_counterpart=True,
         why='M-10 §1.2 가 두 코퍼스의 발자국/λ_p 를 재도출했고 **이미 겹친다**'
             '(합성 BF4 0.70×0.93 · BF5 0.44×0.56 / 실측 0.64–4.11) ⟹ "실측처럼 줄인다"'
             '가 처치가 아니다. 상관판(개구/λ ρ=−0.270 p=0.397)은 `B-2 APERTURE_AXIS.md`'
             ' 에서 저자 자신이 **철회**했다.',
         ref='tmp/M10_GEOMETRY_2026-08-30/premise_m10.json:footprint_lam · '
             'tmp/B2_REAL_2026-08-23/APERTURE_AXIS.md'),
    dict(name='COVERAGE', ko='피복(질의점→최근접 관측 거리)',
         unmeasured=False, single_factor=True, deployment_path=True,
         in_budget=True, real_counterpart=True,
         why='★**2026-08-30 당일 측정됐다** — `M-10` 288행, 라벨 `NEARFIELD-HOLDS`'
             '(45/48, 상대 중앙 −31.69%). 재측정은 중복이다.',
         ref='tmp/M10_GEOMETRY_2026-08-30/verdict_m10.json'),
    dict(name='INCIDENCE', ko='입사각',
         unmeasured=True, single_factor=False, deployment_path=True,
         in_budget=False, real_counterpart=False,
         why='★**거리와 기하학적으로 잠겨 있다** — 단일 원점·고정 높이의 부채꼴에서 '
             'θ = atan(h/r) 이므로 입사각과 거리는 일대일이다(`premise_t4` 가 동결 h5 의 '
             '`lidar_incidence_angles` 와 사거리로 |Spearman ρ| ≥ 0.99 를 재도출한다). '
             '잠금을 풀려면 `height` 를 움직여야 하고 그것은 **재생성**이다(④ 위반). '
             '그리고 실측 하네스에는 입사각이 없다(⑤ 위반). '
             '⚠거리·입사각에 걸린 **잡음**은 `B-17` 이 이미 개입으로 쟀다(전 팔 널).',
         ref='wavesim/lidar_simulator.py:321-344 · wavesim/config.py:18 · '
             'tmp/B17_REALNOISE_2026-08-25/verdict_b17.json'),
    dict(name='OCCLUSION', ko='폐색',
         unmeasured=True, single_factor=False, deployment_path=True,
         in_budget=False, real_counterpart=False,
         why='최근접 히트 레이캐스트에 **암묵적**이고 노브가 **존재하지 않는다**'
             '(`wavesim/lidar_simulator.py:255`, GPU 쌍둥이 `lidar_gpu.py:111`). '
             '끄거나 조절하려면 `wavesim/**` 을 고쳐야 하는데 임무 규율이 무수정을 못박았다 '
             '⟹ 단일요인 노브가 없다. 선택법(raydrop) 판은 `B-15` 가 이미 쟀다.',
         ref='wavesim/lidar_simulator.py:255 · tmp/B15_RAYDROP_2026-08-24/verdict_b15.json'),
    dict(name='RAYPATTERN', ko='광선 패턴(마운트 높이·FOV·빔 수·사거리 게이트)',
         unmeasured=True, single_factor=True, deployment_path=True,
         in_budget=False, real_counterpart=False,
         why='★**저장소 전체에서 한 번도 변한 적이 없다**(`height` 5.0 · `channels` 152 · '
             '`fov_h` 120 · `fov_v` [−25,0] · `n_h_beams` 500 — `premise_t4` 가 동결 h5 '
             '64개에서 단일값임을 재도출한다). 그러나 ④**예산 밖**이다: 생성이 '
             '**GPU 172.9 s/에피소드**(B-26 91건 중앙, 재도출)인데 GPU 가 금지돼 있고, '
             '팔 하나가 48 에피소드다. 그리고 ⑤실측 하네스에 광선이 없다.',
         ref='wavesim/config.py:16-27 · tmp/B26_SEASTATE_2026-08-27/gen_manifest_b26.json'),
    dict(name='SENSORMOTION', ko='조우 기하(플랫폼 운동)',
         unmeasured=False, single_factor=True, deployment_path=True,
         in_budget=False, real_counterpart=False,
         why='`C`(2026-08-20) 448런이 **이미 쟀다**(`GEOMETRY-SURVIVES`, 13/16 문턱에 정확히 '
             '걸침). 그리고 실측 12 레코드는 전부 **고정 관측소**라 이 축이 아예 없다.',
         ref='tmp/C_ENCOUNTER_2026-08-20/verdict_c.json'),
    dict(name='DENSITY', ko='관측 밀도(희소성)',
         unmeasured=False, single_factor=True, deployment_path=True,
         in_budget=True, real_counterpart=True,
         why='`B-18` `NEITHER-SUFFICIENT` — 실측에서 480→12,000 관측(25배)에서도 8/12.',
         ref='tmp/B18_DENSITY_NOISE_2026-08-25/verdict_b18.json'),
    dict(name='TIMEWINDOW', ko='시간 개구(관측 창 길이)',
         unmeasured=False, single_factor=False, deployment_path=True,
         in_budget=True, real_counterpart=True,
         why='`M5b` `WINDOW-LONGER-IS-BETTER`(부호가 반대). ⚠게다가 관측 수 고정 처치라 '
             '시간 밀도가 함께 움직인다(논문이 공시한 교락).',
         ref='tmp/M5b_WINDOW_2026-08-28/verdict_m5b.json'),
    dict(name='SENSORMODEL', ko='센서 반환 모형·잡음의 형태',
         unmeasured=False, single_factor=True, deployment_path=True,
         in_budget=True, real_counterpart=True,
         why='`B-15`(합성, 헤드라인 유지 17/20) · `B-14b`(실측, '
             '`OBS-MODEL-DOES-NOT-MEDIATE`, 최대 8/12) · `B-17`(실측 거리·입사각 잡음 곡선, '
             '전 팔 널).',
         ref='tmp/B15_RAYDROP_2026-08-24/verdict_b15.json · '
             'tmp/B14_OBSMODEL_2026-08-24/RESULT_B14.md · '
             'tmp/B17_REALNOISE_2026-08-25/verdict_b17.json'),
    dict(name='LEADTIME', ko='★관측 시각과 채점 시각의 상대 기하(lead)',
         unmeasured=True, single_factor=True, deployment_path=True,
         in_budget=True, real_counterpart=True,
         why='★M-10 이 「관측 기하」를 **채점 쪽**으로 넓혔고(*"채점 쪽은 이름이 없다"*), '
             '그 넓힘의 **공간** 성분(피복)만 쟀다. **시간** 성분은 그대로 비어 있다: '
             '합성 헤드라인은 그 프레임의 관측을 **먹은 뒤** 같은 시각을 채점하고'
             '(`run_s16.py` update→predict_grid→compute_all_metrics 순서 · '
             '`wavesplat/run.py` update→predict_field(frame.t)), 실측 판정은 '
             '**먼저 예측·채점하고 그 다음 관측을 먹으며** 판정 자는 `fc_mean` = '
             '**lead 2–14 s 평균**이다(`gates_b3.FIELD`). ⟹ 0 s 대 2–14 s. '
             '단일요인: **재채점**이라 관측·추정기·격자·시드가 전부 불변이고 '
             '*모형을 언제 평가하는가*만 바뀐다. 배포 경로: `predict_field(t)` 는 '
             '제어 프리뷰가 실제로 부르는 API 이고 `run.py:329-337` 이 이미 lead 잔차를 '
             '계산한다. 예산: **생성 0 · GPU 0 · 학습 0**.',
         ref='tmp/B3_REFIT_2026-08-23/gates_b3.py:15,38 · '
             'tmp/S16_BASELINE_2026-08-14/run_s16.py:229-236 · '
             'wavesplat/run.py:329-343 · '
             'tmp/M10_GEOMETRY_2026-08-30/PREREG.md (§1.3 "채점 쪽은 이름이 없다")'),
)


def screen(cands=CANDIDATES):
    """(name, go, failed_criteria, cand) 목록.  ⛔결과 전 동결된 다섯 기준만 본다."""
    out = []
    for c in cands:
        failed = tuple(k for k in SCREEN_CRITERIA if not c.get(k))
        out.append((c['name'], not failed, failed, c))
    return out


def screen_go(cands=CANDIDATES):
    return [n for n, go, f, c in screen(cands) if go]


# ═════════════════════════════════════════════════════════════════════════════
# B부 — `T4-LEAD` 판정기
# ═════════════════════════════════════════════════════════════════════════════
#
# 처치 = **시간축 재채점**.  같은 런 · 같은 관측 · 같은 추정기 상태 · 같은 격자에서,
# **적합된 모형을 언제 평가하고 어느 GT 프레임과 맞대는가**만 바꾼다.

# ── 사다리 (결과 전 확정) ─────────────────────────────────────────────────────
# ★lead 를 **Tp 단위**로 선언한다 — 그래야 BF4(Tp 7.0)·BF5(Tp 9.0)·실측(중앙 3.49 s)이
#   같은 축 위에 놓인다.  ⛔초 단위로 선언하면 두 층이 서로 다른 물리 지평을 갖는다.
# ★rung 은 **두 층 모두에서 정확히 프레임에 떨어지는 값**만 쓴다.  `G-9` 가 이것을
#   1e-9 로 강제한다.  ⚠초안의 `0.25` 는 BF4 에서 17.5 프레임이라 **떨어지지 않았고**,
#   자가진단이 런 0회 시점에 그것을 잡았다(T2 개정 0 과 같은 부류의 사전 수리).
LEADS_TP = (0.0, 0.2, 0.5, 1.0)
PRIMARY_LEAD_TP = 1.0        # ★공통 원점 집합이 지탱하는 **최대** lead. 결과 전 확정.
BAND_LEAD_TP = (0.5, 1.0)    # 사전선언 민감도 — 실측 밴드 하단(≈0.573 Tp)을 감싼다.

SPINUP_TP = 2.0              # `run_s16.py` 의 `sel = t > 2.0*Tp_true` 규약 (premise 가 대조)
ORIGIN_STRIDE_S = 0.5        # ★고정.  대체 stride 는 **원리적으로 없다**(바로 아래).
MIN_ORIGINS = 5              # G-8 하한 (BF5 · stride 0.5 · lead 1.0 Tp 에서 6)
MAX_CORE_HOURS = 6.0         # 스모크 투영이 이것을 넘으면 ⛔**ABORT**(완화 경로 없음)
# ⚠**비용 도피처가 없다는 것을 결과 전에 못박는다.**  원점 간격을 늘려 비용을 줄이려 해도
#   BF5 층은 stride 1.0 s 에서 원점이 **3개**(= `MIN_ORIGINS` 미달)라 `G-8` 이 막는다.
#   ⟹ 스모크 투영이 `MAX_CORE_HOURS` 를 넘으면 **설계를 줄이는 것이 아니라 배치를 접는다**.
#   ⛔결과를 본 뒤 `MIN_ORIGINS` 를 내리는 것은 `M-4 A-2`(사후 완화)와 같은 죄다.
ORIGIN_STRIDE_CHOICES = (0.5, 1.0, 2.0)   # ⚠**서술 전용** — 위 사실을 자가진단이 증명한다.

# ── 문턱 ──────────────────────────────────────────────────────────────────────
ALPHA = 0.05
N_FAMILY = 2                 # PRIMARY-A(LEADFIELD) · SECONDARY-B(DECAY)
FAMILY_ALPHA = ALPHA / N_FAMILY
REPRO_RTOL = 1e-6            # G-4 (⚠비트 아님 — 동결물이 스레드를 고정하지 않았다)
COMPOSE_RTOL = 1e-9          # G-7 양성대조(전파 연산자 합성)


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def thresholds(n=N_EP):
    """(NOMINAL, CONFIRM, NOT_EARN, q80, floor_p).  ⛔전부 재도출, 리터럴 없음."""
    nominal = next(w for w in range(n + 1) if sign_test_one_sided(w, n) <= ALPHA)
    confirm = next(w for w in range(n + 1) if sign_test_one_sided(w, n) <= FAMILY_ALPHA)
    lo, hi = 0.5, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        pw = sum(comb(n, k) * mid ** k * (1 - mid) ** (n - k)
                 for k in range(confirm, n + 1))
        if pw >= 0.80:
            hi = mid
        else:
            lo = mid
    return dict(n=n, NOMINAL=nominal, CONFIRM=confirm, NOT_EARN=n - nominal,
                q80=hi, floor_p=sign_test_one_sided(n, n),
                alpha=ALPHA, family_alpha=FAMILY_ALPHA)


# ── 원점·lead 산술 (순수 함수 — 러너와 게이트가 **이것 하나**를 부른다) ────────
def lead_frames(tp, dt):
    """Tp 단위 lead → 프레임 수.  ⛔러너가 따로 계산하지 않는다."""
    return {L: int(round(L * tp / dt)) for L in LEADS_TP}


def origins(tp, dt, n_frames, stride_s):
    """채점 원점 인덱스.  ★**lead 와 무관한 하나의 집합**(사다리를 용량-반응으로 읽으려면
    원점이 lead 마다 달라져서는 안 된다 — 그러면 창이 함께 움직인다)."""
    lf = lead_frames(tp, dt)
    lmax = max(lf.values())
    i_lo = int(math.floor(SPINUP_TP * tp / dt)) + 1      # t_i > 2·Tp (엄격 부등호)
    i_hi = (n_frames - 1) - lmax
    step = int(round(stride_s / dt))
    if step <= 0 or i_hi < i_lo:
        return []
    return list(range(i_lo, i_hi + 1, step))


def expected_branch_counts(tp, dt, n_frames, stride_s, arm, n_ensemble):
    """★개입 계수기의 기대값을 **프로토콜에서 도출**한다.  ⛔손복사 금지 —
    러너·게이트·문서가 이 함수 하나를 부른다(W4 가 같은 값을 네 곳에 두어 물렸다)."""
    ors = origins(tp, dt, n_frames, stride_s)
    lmax = max(lead_frames(tp, dt).values())
    if arm == 'SPEC25':
        # 원점마다 앙상블을 lmax 스텝 전진시키며 각 rung 에서 스냅샷 ⟹ 원점당 lmax 스텝.
        return dict(forecast_steps=len(ors) * lmax * int(n_ensemble),
                    predict_future=0)
    return dict(forecast_steps=0, predict_future=len(ors) * len(LEADS_TP))


# ── 판정 ──────────────────────────────────────────────────────────────────────
def med_by_episode(rows, arm, getter):
    d = {}
    for r in rows:
        if 'error' in r or r.get('arm') != arm:
            continue
        v = getter(r)
        if v is None or v != v:
            continue
        d.setdefault(r['episode'], []).append(float(v))
    return {e: st.median(v) for e, v in d.items()}


def lead_getter(field, lead_tp):
    key = repr(float(lead_tp))

    def _g(r):
        d = r.get(field) or {}
        return d.get(key, d.get(str(float(lead_tp))))
    return _g


def band_getter(field, leads=BAND_LEAD_TP):
    """밴드 합성 규약(결과 전 고정): rung 들의 **RMS**.  ⛔중앙값의 비 금지."""
    def _g(r):
        d = r.get(field) or {}
        vs = []
        for L in leads:
            v = d.get(repr(float(L)), d.get(str(float(L))))
            if v is None or v != v:
                return None
            vs.append(float(v))
        return (sum(v * v for v in vs) / len(vs)) ** 0.5
    return _g


def paired_wins(a, b):
    """a < b 인 에피소드 수.  반환 (w, n, rel_median)."""
    ks = sorted(set(a) & set(b))
    w = sum(1 for k in ks if a[k] < b[k])
    rel = [(a[k] - b[k]) / b[k] for k in ks if b[k]]
    return w, len(ks), (st.median(rel) if rel else None)


LABELS = ('LEADFIELD-HOLDS', 'LEADFIELD-HOLDS-NOMINAL', 'LEADFIELD-BREAKS',
          'LEADFIELD-UNDECIDED', 'LEADFIELD-VOID')


def label(w_primary, n, cmp_ok):
    """★결과 전 확정.  `cmp_ok` = G-CMP(비교자 유효성)."""
    t = thresholds(n)
    if not cmp_ok:
        return 'LEADFIELD-VOID'
    if w_primary >= t['CONFIRM']:
        return 'LEADFIELD-HOLDS'
    if w_primary >= t['NOMINAL']:
        return 'LEADFIELD-HOLDS-NOMINAL'
    if w_primary <= t['NOT_EARN']:
        return 'LEADFIELD-BREAKS'
    return 'LEADFIELD-UNDECIDED'


def fold(lab):
    """★접기 공역.  ⛔`'break'` 는 `LEADFIELD-BREAKS` **정확히 하나**다
    (`tmp/_audit/DISCIPLINE_verdict_labels_2026-08-30.md` P1; 본뜬 곳 = `gates_m3.st_()`).
    ⛔`gates_m1.py:314` 상속 금지 — 그것은 *"비교 불능"* 을 *"부호 반전"* 으로 둔갑시킨다."""
    if lab == 'LEADFIELD-HOLDS':
        return 'hold'
    if lab == 'LEADFIELD-BREAKS':
        return 'break'
    return 'undecided'


READING = {
    'LEADFIELD-HOLDS':
        '합성 이득은 실측 코퍼스가 실제로 채점되는 **시간 기하**(관측 시각보다 lead 만큼 뒤) '
        '안에서도 있다 ⟹ 「관측 기하」 행에서 **시간** 성분이 배제된다. M-10(공간)과 합쳐 '
        '***"두 하네스가 서로 다른 문제를 풀고 있어서 생긴 차이다" 는 반론이 닫힌다***. '
        '★척추 문장 무수정(배제 ≠ 지목). ⛔"관측 기하가 무관하다" 금지(등가검정 없음). '
        '⛔"예보 능력을 검증했다" 금지(합성 · 혼잡 층 · 단일 체크포인트 · lead ≤ 1.0 Tp).',
    'LEADFIELD-HOLDS-NOMINAL':
        '명목만(가족단위 α 미달) ⟹ **EXPLORATORY** 축자 병기 의무 + MDE 병기. '
        '⛔확증으로 올리지 마라. ⛔결과를 보고 n 을 늘리지 마라.',
    'LEADFIELD-BREAKS':
        '★합성 이득은 실측이 채점되는 시간 기하에는 **없다** — 헤드라인은 그 프레임의 관측을 '
        '먹은 뒤의 **동화(analysis)** 점수이고, 실측 판정은 **예보(forecast)** 점수다. '
        '⟹ 합성 양성과 실측 음성이 한 측정의 두 면이 된다. ⛔단 **인과 지목 금지**'
        '(재채점은 기제를 재지 않는다); 허용은 ***"이득이 어느 시간 기하에 있는가"*** 까지. '
        '★동반 공시 의무 신설: `B-1`·`B-12`·`B-16`·`B-22`·`B-26`·`M-7`·`M-10` 의 승수를 '
        '인용하는 **모든 지점**에 "그 승수는 lead %.2f Tp 에서 성립하지 않는다" 를 병기한다'
        '(같은 것이 N곳이면 N곳 전부). ⛔`SUPPORT-TRADEOFF` 라는 이름 금지.' % PRIMARY_LEAD_TP,
    'LEADFIELD-UNDECIDED':
        '실측이 채점되는 시간 기하에서 두 팔의 우열이 n=48 로 갈리지 않았다 — '
        '**이 라벨도 결과다**(전역 승수가 그 lead 에서 재현되지 않는다는 사실 자체가 서술). '
        '⛔"차이 없다" 금지(등가검정 없음). MDE 병기 의무. '
        '후속은 표본 증설이 아니라 사다리(용량-반응)와 밴드 민감도의 서술이다.',
    'LEADFIELD-VOID':
        '그 lead 에서는 최강 적격 격자 기준선도 무기술이다 ⟹ ⛔**우열 주장 전면 금지**. '
        '허용 = "그 시간 기하는 두 방법 모두에게 다른 문제다" · "헤드라인의 비교는 동화 '
        '시각에서만 성립한다". ★우리 재현·우리 구성에 대한 판정이지 문헌 일반에 대한 것이 '
        '아니다(`B-5` 규약 상속).',
}


# ── 로더 (★검사기 규약: 무인자 · glob.glob 무별칭 · json.loads) ────────────────
def load_rows():
    out = []
    for path in sorted(glob.glob(os.path.join(HERE, 'results_t4_*.jsonl'))):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def load_frozen_b12():
    """동결 `B-12` 의 `avg_nRMSE` — G-4(전 창 동화 재현)의 원천."""
    out = {}
    if not os.path.exists(B12_FROZEN):
        return out
    with open(B12_FROZEN) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if 'error' in r or r.get('arm') not in ('SHIP-NOW', 'SPEC25'):
                continue
            arm = 'SHIP' if r['arm'] == 'SHIP-NOW' else 'SPEC25'
            s = r.get('est_seed', r.get('filter_seed'))
            out[(arm, r['episode'], s)] = float(r['avg_nRMSE'])
    return out


def episodes():
    """혼잡 48.  조건절은 `run_b12.crowded_episodes` 의 것을 그대로."""
    import re
    out = []
    for p in sorted(glob.glob(os.path.join(HOS, '*.h5'))):
        m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(.+)\.h5', os.path.basename(p))
        if m and int(m.group(2)) >= CROWDED_MIN_NCOMP and int(m.group(3)) in ALL_SEEDS:
            out.append(os.path.basename(p)[len('episode_'):-3])
    return out


# ── 고정 기준상수 부재 (★T2 개정 0 의 처방: 문자열 검색이 아니라 `ast`) ────────
_FLAT_NAME = 'ZERO' + '_REF'        # ⛔소스에 리터럴로 두지 않는다


def no_flat_ref_constant(where=None):
    """디렉터리의 어느 `.py` 도 고정 기준상수를 **코드로** 쓰지 않는가.
    ⚠산문·주석은 통과시킨다(그 구분이 T2 `G-7b` 를 항상 실패하게 만들었다)."""
    where = HERE if where is None else where
    hits = []
    for f in sorted(os.listdir(where)):
        if not f.endswith('.py'):
            continue
        try:
            tree = ast.parse(open(os.path.join(where, f), encoding='utf-8').read())
        except SyntaxError as e:
            return False, '%s: 파싱 실패 %s' % (f, e)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Name) and node.id == _FLAT_NAME) or \
                    (isinstance(node, ast.Attribute) and node.attr == _FLAT_NAME):
                hits.append('%s:%s' % (f, getattr(node, 'lineno', '?')))
    return not hits, str(hits)[:80]


# ── 게이트 ────────────────────────────────────────────────────────────────────
def _rel(a, b):
    return abs(a - b) / abs(b) if b else (0.0 if a == b else float('inf'))


def _lead_keys(d):
    return {float(k) for k in (d or {})}


def gates(rows, frozen=None):
    """(name, ok, detail, blocking).  ★러너와 분석기가 **이 함수 하나**를 부른다."""
    frozen = load_frozen_b12() if frozen is None else frozen
    good = [r for r in rows if 'error' not in r]
    err = [r for r in rows if 'error' in r]
    out = []
    cov = {}

    def g(name, ok, det, blocking=True, n_seen=0):
        cov[name] = n_seen
        out.append((name, bool(ok), det, blocking))

    per_arm = {a: [r for r in good if r.get('arm') == a] for a in ARM_ORDER}
    eps = sorted({r['episode'] for r in good})
    want = N_EP * len(EST_SEEDS)
    g('G-1 표본 SHIP/SPEC25 각 %d행 · 에피소드 %d' % (want, N_EP),
      all(len(per_arm[a]) == want for a in ARM_ORDER) and len(eps) == N_EP,
      'SHIP %d · SPEC25 %d · ep %d' % (len(per_arm['SHIP']), len(per_arm['SPEC25']),
                                       len(eps)),
      n_seen=len(good))
    g('G-2 에러 0', not err, '%d' % len(err), n_seen=len(rows))

    dep = [r for r in good if r.get('arm') == 'SHIP']
    ok3 = bool(dep) and all(
        r.get('obj_init_method') == 'auto' and r.get('obj_tracking_mode') == 'w_phi'
        and r.get('cap_varpro_steps') is not None
        and r.get('arm_cfg_nonempty') is True for r in dep)
    g('G-3 팔 되읽기(객체 속성) + ⛔빈 dict 팔 0', ok3, '%d행' % len(dep), n_seen=len(dep))

    pairs = [(r, frozen.get((r['arm'], r['episode'], r.get('seed')))) for r in good]
    pairs = [(r, f) for r, f in pairs if f is not None]
    bad4 = [r['episode'] for r, f in pairs
            if r.get('avg_nRMSE') is None or _rel(float(r['avg_nRMSE']), f) > REPRO_RTOL]
    g('G-4 동결 재현(전 창 동화) rel Δ ≤ %g (%d짝)' % (REPRO_RTOL, len(pairs)),
      bool(pairs) and not bad4, '위반 %d' % len(bad4), n_seen=len(pairs))

    win = {}
    for r in good:
        win.setdefault(r['episode'], set()).add(
            (r.get('n_scored'), r.get('t_first'), r.get('t_last'),
             r.get('n_origins'), r.get('origin_first'), r.get('origin_last')))
    g('G-5 같은 채점 창·같은 원점 집합(에피소드 안 팔·시드 전건 동일)',
      bool(win) and all(len(v) == 1 for v in win.values()),
      '불일치 %d' % sum(1 for v in win.values() if len(v) != 1), n_seen=len(win))

    neg = [r for r in good if r.get('neg_lead0_bitmatch') is not None]
    g('G-6 ★음성대조: lead 0 예보경로 == 동화경로 **비트 동일**',
      bool(neg) and len(neg) == len(good) and all(r['neg_lead0_bitmatch'] is True
                                                  for r in neg),
      '%d/%d' % (sum(1 for r in neg if r.get('neg_lead0_bitmatch') is True), len(good)),
      n_seen=len(neg))

    pos = [r for r in good if r.get('arm') == 'SPEC25']
    badp = [r['episode'] for r in pos
            if r.get('pos_compose_relerr') is None
            or not (float(r['pos_compose_relerr']) <= COMPOSE_RTOL)]
    g('G-7 ★양성대조: 전파 연산자 합성 F(a+b)==F(a)∘F(b) rel ≤ %g' % COMPOSE_RTOL,
      bool(pos) and not badp, '위반 %d/%d' % (len(badp), len(pos)), n_seen=len(pos))

    want_leads = {float(L) for L in LEADS_TP}
    bad8 = [r['episode'] for r in good
            if _lead_keys(r.get('lead_rmse')) != want_leads
            or _lead_keys(r.get('lead_zero')) != want_leads
            or not isinstance(r.get('n_origins'), int)
            or r.get('n_origins', 0) < MIN_ORIGINS
            or r.get('origin_stride_s') != ORIGIN_STRIDE_S]
    zero_cov = [k for k, v in cov.items() if v == 0]
    g('G-8 커버리지 0 금지(rung 전건 · 원점 ≥ %d · stride == %.1f s · 열람0 게이트 0)'
      % (MIN_ORIGINS, ORIGIN_STRIDE_S),
      bool(good) and not bad8 and not zero_cov,
      'rung/원점 위반 %d · 열람0 %s' % (len(bad8), zero_cov), n_seen=len(good))

    bad9 = []
    for r in good:
        tp, dt = r.get('tp_true'), r.get('dt')
        if tp is None or dt is None:
            bad9.append(r.get('episode'))
            continue
        want_lf = {repr(float(k)): v for k, v in lead_frames(float(tp), float(dt)).items()}
        got = {str(k): int(v) for k, v in (r.get('lead_frames') or {}).items()}
        if {k: int(v) for k, v in want_lf.items()} != got:
            bad9.append(r.get('episode'))
            continue
        if any(abs(v * float(dt) - float(k) * float(tp)) > 1e-9
               for k, v in want_lf.items()):
            bad9.append(r.get('episode'))
    g('G-9 lead 프레임 산술 정확(러너가 gates_t4.lead_frames 를 썼다)',
      bool(good) and not bad9, '위반 %d' % len(bad9), n_seen=len(good))

    stamps = {(r.get('device'), r.get('cuda_available'), r.get('omp_threads'),
               r.get('code_md5')) for r in good}
    ok10 = (len(stamps) == 1 and bool(good)
            and all(s[0] == 'cpu' and s[2] == '1' for s in stamps))
    g('G-10 장치·스레드·code_md5 단일 (⚠cuda_available 은 단일성만 — M-10 개정 1 승계)',
      ok10, repr(sorted(map(str, stamps)))[:120], n_seen=len(good))

    flat_ok, flat_det = no_flat_ref_constant()
    g('G-11 단일출처(셸 경계 임포트) + 고정 기준상수 부재(%s)' % _FLAT_NAME,
      SHELL_KEYS == list(R16.SHELL_KEYS) and flat_ok, flat_det, n_seen=1)

    bad12 = []
    for r in good:
        tp, dt, nf = r.get('tp_true'), r.get('dt'), r.get('n_frames')
        stride, ne = r.get('origin_stride_s'), r.get('n_ensemble', 1)
        if None in (tp, dt, nf, stride):
            bad12.append(r.get('episode'))
            continue
        exp = expected_branch_counts(float(tp), float(dt), int(nf), float(stride),
                                     r.get('arm'), ne)
        if (int(r.get('cap_forecast_steps', -1)) != exp['forecast_steps']
                or int(r.get('cap_predict_future', -1)) != exp['predict_future']):
            bad12.append(r.get('episode'))
    g('G-12 개입 계수기(가지 단위) == 프로토콜 도출값',
      bool(good) and not bad12, '위반 %d' % len(bad12), n_seen=len(good))

    bad13 = []
    for r in good:
        tp, dt, nf = r.get('tp_true'), r.get('dt'), r.get('n_frames')
        stride = r.get('origin_stride_s')
        if None in (tp, dt, nf, stride):
            bad13.append(r.get('episode'))
            continue
        ors = origins(float(tp), float(dt), int(nf), float(stride))
        if (not ors or r.get('n_origins') != len(ors)
                or r.get('origin_first') != ors[0] or r.get('origin_last') != ors[-1]):
            bad13.append(r.get('episode'))
    g('G-13 원점 집합이 lead 와 무관한 하나(용량-반응의 전제)',
      bool(good) and not bad13, '위반 %d' % len(bad13), n_seen=len(good))

    # ── G-CMP 차단 게이트: 비교자 유효성 (실패 ⟹ 라벨 VOID, 판정 거부가 아니다) ──
    gp = lead_getter('lead_rmse', PRIMARY_LEAD_TP)
    gz = lead_getter('lead_zero', PRIMARY_LEAD_TP)
    spec = med_by_episode(good, 'SPEC25', gp)
    zero = med_by_episode(good, 'SPEC25', gz)
    wc, nc, _ = paired_wins(spec, zero)
    t = thresholds(nc if nc else N_EP)
    g('G-CMP 비교자 유효성 SPEC25<ZERO @ lead %.2f Tp (≥%d)'
      % (PRIMARY_LEAD_TP, t['CONFIRM']),
      bool(nc) and wc >= t['CONFIRM'], '%d/%d' % (wc, nc), n_seen=nc)

    # ── 범위축소 ─────────────────────────────────────────────────────────────
    bits = sum(1 for r, f in pairs if r.get('avg_nRMSE') == f)
    g('G-14 동결값 비트 일치 수 (0이어도 유효)', True, '%d/%d' % (bits, len(pairs)),
      blocking=False, n_seen=len(pairs))
    mv = {}
    for r in good:
        mv[r.get('mesh_verdict')] = mv.get(r.get('mesh_verdict'), 0) + 1
    g('G-15 mesh_verdict 분할 보고 (⛔"HOS M=3" 단일 서술 금지)', True, repr(mv),
      blocking=False, n_seen=len(good))
    strat = {}
    for r in good:
        s = str(r.get('episode', '')).split('_')[0]
        strat.setdefault(s, set()).add(r.get('n_origins'))
    g('G-16 층별 원점 수 (⚠BF4/BF5 불균형 공시 의무)', True,
      repr({k: sorted(v) for k, v in strat.items()}), blocking=False, n_seen=len(good))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 자가진단 — ★게이트마다 **양방향**(통과 / 위조에서 반드시 실패)
# ═════════════════════════════════════════════════════════════════════════════
def _fixture(n_ep=N_EP):
    """깨끗한 픽스처 — 자료 0행, 게이트 논리만 시험한다."""
    rows, frozen = [], {}
    for e in range(n_ep):
        bf = 'BF4' if e % 2 == 0 else 'BF5'
        tp = 7.0 if bf == 'BF4' else 9.0
        dt, nf, stride, ne = 0.1, 301, ORIGIN_STRIDE_S, int(SPEC['n_ensemble'])
        ors = origins(tp, dt, nf, stride)
        lf = lead_frames(tp, dt)
        ep = '%s_15comp_seed%d_uniform_ra_rf_ds' % (bf, 1581 + e)
        for arm in ARM_ORDER:
            for s in EST_SEEDS:
                base = 0.10 + 0.001 * e + (0.05 if arm == 'SPEC25' else 0.0)
                frozen[(arm, ep, s)] = base
                exp = expected_branch_counts(tp, dt, nf, stride, arm, ne)
                rows.append(dict(
                    arm=arm, episode=ep, seed=s, avg_nRMSE=base,
                    lead_rmse={repr(float(L)): base * (1 + 0.3 * L) for L in LEADS_TP},
                    lead_zero={repr(float(L)): 0.25 for L in LEADS_TP},
                    n_scored=160, t_first=14.1, t_last=30.0,
                    n_origins=len(ors), origin_first=ors[0], origin_last=ors[-1],
                    origin_stride_s=stride, lead_frames={repr(float(k)): v
                                                         for k, v in lf.items()},
                    tp_true=tp, dt=dt, n_frames=nf, n_ensemble=ne,
                    neg_lead0_bitmatch=True,
                    pos_compose_relerr=(0.0 if arm == 'SPEC25' else None),
                    cap_forecast_steps=exp['forecast_steps'],
                    cap_predict_future=exp['predict_future'],
                    obj_init_method='auto', obj_tracking_mode='w_phi',
                    cap_varpro_steps=400, arm_cfg_nonempty=True,
                    mesh_verdict='DEGRADED', device='cpu', cuda_available=False,
                    omp_threads='1', code_md5='fixture'))
                if arm == 'SHIP':
                    rows[-1]['prior_device'] = 'cuda'
    return dict(rows=rows, frozen=frozen)


def _selftest():
    ok = []

    def chk(name, cond, det=''):
        ok.append((name, bool(cond), det))

    # ── A부: 선별기 ───────────────────────────────────────────────────────────
    sc = screen()
    go = screen_go()
    chk('A① 후보 10개 전건이 다섯 기준 전건을 갖는다',
        all(set(SCREEN_CRITERIA) <= set(c) for _, _, _, c in sc) and len(sc) == 10,
        '%d' % len(sc))
    chk('A② GO 는 정확히 하나이고 그것이 LEADTIME 이다', go == ['LEADTIME'], repr(go))
    chk('A③ ★선별기는 항진하지 않는다 — 다섯 기준을 다 만족하는 **위조 후보**는 GO',
        screen_go(CANDIDATES + (dict(name='FORGED', ko='위조', why='', ref='',
                                     **{k: True for k in SCREEN_CRITERIA}),))
        == ['LEADTIME', 'FORGED'])
    for k in SCREEN_CRITERIA:
        f = dict(name='F_' + k, ko='위조', why='', ref='',
                 **{c: (c != k) for c in SCREEN_CRITERIA})
        chk('A④ 기준 %s 하나만 위반해도 NO-GO' % k, 'F_' + k not in screen_go((f,)))
    chk('A⑤ 접힌 후보는 사유와 근거를 **파일:줄** 로 갖는다',
        all(c['why'] and c['ref'] for _, gg, _, c in sc if not gg))
    chk('A⑥ LEADTIME 의 근거가 실측 판정 자를 가리킨다',
        GB3.FIELD == 'fc_mean' and len(GB3.RECORDS) == 12, GB3.FIELD)

    # ── B부: 문턱 (재도출) ────────────────────────────────────────────────────
    t = thresholds(N_EP)
    chk('B① 문턱이 부호검정에서 재도출된다',
        sign_test_one_sided(t['CONFIRM'], N_EP) <= FAMILY_ALPHA
        and sign_test_one_sided(t['CONFIRM'] - 1, N_EP) > FAMILY_ALPHA
        and sign_test_one_sided(t['NOMINAL'], N_EP) <= ALPHA,
        repr(t))
    chk('B② NOT_EARN 은 형제 규약 n − NOMINAL', t['NOT_EARN'] == N_EP - t['NOMINAL'])
    chk('B③ MDE 가 (0.5, 1) 안', 0.5 < t['q80'] < 1.0, '%.4f' % t['q80'])

    # ── 라벨·접기 ────────────────────────────────────────────────────────────
    folds = {L: fold(L) for L in LABELS}
    chk('B④ break 로 접히는 라벨은 정확히 하나',
        sum(1 for v in folds.values() if v == 'break') == 1
        and folds['LEADFIELD-BREAKS'] == 'break', repr(folds))
    chk('B⑤ VOID·UNDECIDED·NOMINAL 은 undecided',
        all(folds[L] == 'undecided' for L in
            ('LEADFIELD-VOID', 'LEADFIELD-UNDECIDED', 'LEADFIELD-HOLDS-NOMINAL')))
    chk('B⑥ 모든 라벨이 READING 에 있다', set(LABELS) == set(READING))
    chk('B⑦ 라벨 경계 (사후 이동 방지)',
        label(t['CONFIRM'], N_EP, True) == 'LEADFIELD-HOLDS'
        and label(t['NOMINAL'], N_EP, True) == 'LEADFIELD-HOLDS-NOMINAL'
        and label(t['NOMINAL'] - 1, N_EP, True) == 'LEADFIELD-UNDECIDED'
        and label(t['NOT_EARN'] + 1, N_EP, True) == 'LEADFIELD-UNDECIDED'
        and label(t['NOT_EARN'], N_EP, True) == 'LEADFIELD-BREAKS')
    chk('B⑧ 무기술 픽스처: cmp 실패면 전 승수에서 VOID',
        {label(w, N_EP, False) for w in (0, 17, 31, 32, 48)} == {'LEADFIELD-VOID'})
    chk('B⑨ VOID 읽기에 귀속·우열 문구 없음',
        '우열 주장 전면 금지' in READING['LEADFIELD-VOID']
        and '이긴다' not in READING['LEADFIELD-VOID'])

    # ── 원점·lead 산술 ───────────────────────────────────────────────────────
    o4 = origins(7.0, 0.1, 301, 0.5)
    o5 = origins(9.0, 0.1, 301, 0.5)
    chk('B⑩ 원점: BF4 %d · BF5 %d (둘 다 ≥ %d)' % (len(o4), len(o5), MIN_ORIGINS),
        len(o4) >= MIN_ORIGINS and len(o5) >= MIN_ORIGINS
        and o4[0] * 0.1 > SPINUP_TP * 7.0 and o5[0] * 0.1 > SPINUP_TP * 9.0)
    chk('B⑪ 최대 lead 를 더해도 에피소드를 넘지 않는다',
        o4[-1] + max(lead_frames(7.0, 0.1).values()) <= 300
        and o5[-1] + max(lead_frames(9.0, 0.1).values()) <= 300)
    chk('B⑫ lead 0 은 0프레임 · Tp 단위가 초로 정확 환산',
        lead_frames(7.0, 0.1)[0.0] == 0 and lead_frames(7.0, 0.1)[1.0] == 70
        and lead_frames(9.0, 0.1)[1.0] == 90)
    chk('B⑬ stride 를 올려도 원점 집합이 **줄기만** 한다(부분집합)',
        set(origins(7.0, 0.1, 301, 1.0)) <= set(o4)
        and set(origins(7.0, 0.1, 301, 2.0)) <= set(o4))
    chk('B⑬b ★비용 도피처 부재: 더 큰 stride 는 BF5 에서 G-8 하한 미달 ⟹ ABORT 만 남는다',
        all(len(origins(9.0, 0.1, 301, s)) < MIN_ORIGINS
            for s in ORIGIN_STRIDE_CHOICES if s > ORIGIN_STRIDE_S)
        and len(origins(9.0, 0.1, 301, ORIGIN_STRIDE_S)) >= MIN_ORIGINS,
        repr({s: len(origins(9.0, 0.1, 301, s)) for s in ORIGIN_STRIDE_CHOICES}))
    chk('B⑭ 개입 계수기는 프로토콜에서 도출된다(손복사 아님)',
        expected_branch_counts(7.0, 0.1, 301, 0.5, 'SPEC25', 600)['forecast_steps']
        == len(o4) * 70 * 600
        and expected_branch_counts(7.0, 0.1, 301, 0.5, 'SHIP', 600)['predict_future']
        == len(o4) * len(LEADS_TP))

    # ── 밴드 합성 ────────────────────────────────────────────────────────────
    r = dict(lead_rmse={repr(0.5): 3.0, repr(1.0): 4.0})
    chk('B⑮ 밴드 = rung 들의 RMS',
        abs(band_getter('lead_rmse')(r) - ((9 + 16) / 2) ** 0.5) < 1e-12)
    chk('B⑯ rung 하나가 결측이면 밴드는 None (⛔부분 합성 금지)',
        band_getter('lead_rmse')(dict(lead_rmse={repr(0.5): 3.0})) is None)

    # ── 게이트: 깨끗한 픽스처 ────────────────────────────────────────────────
    fx = _fixture()
    gs = gates(fx['rows'], fx['frozen'])
    chk('C① 깨끗한 픽스처: 차단 게이트 전건 PASS(⛔통과 불가능한 게이트 0)',
        all(o for n, o, d, b in gs if b), [n for n, o, d, b in gs if b and not o])
    chk('C② 커버리지 0 게이트 없음',
        all(o for n, o, d, b in gs if n.startswith('G-8')))

    # ── 게이트: ★위조 입력이면 반드시 FAIL (게이트마다 하나씩) ────────────────
    def _fails(prefix, rows, frozen=None):
        return any((not o) for n, o, d, b in
                   gates(rows, fx['frozen'] if frozen is None else frozen)
                   if n.startswith(prefix))

    for drop in ARM_ORDER:
        chk('C③ %s 삭제 ⟹ G-1 FAIL' % drop,
            _fails('G-1', [r for r in fx['rows'] if r['arm'] != drop]))
    chk('C④ 에러 행이 있으면 G-2 FAIL',
        _fails('G-2', fx['rows'] + [dict(fx['rows'][0], error='boom')]))
    chk('C⑤ init_method 가 배포값이 아니면 G-3 FAIL',
        _fails('G-3', [dict(r, obj_init_method='v2_konly') if r['arm'] == 'SHIP' else r
                       for r in fx['rows']]))
    chk('C⑤b 빈 dict 팔이면 G-3 FAIL',
        _fails('G-3', [dict(r, arm_cfg_nonempty=False) if r['arm'] == 'SHIP' else r
                       for r in fx['rows']]))
    chk('C⑥ 동결값에서 벗어나면 G-4 FAIL',
        _fails('G-4', [dict(r, avg_nRMSE=r['avg_nRMSE'] * 1.01) for r in fx['rows']]))
    chk('C⑥b 동결 원천이 비면 G-4 FAIL(0짝을 OK 로 찍지 않는다)',
        _fails('G-4', fx['rows'], {}))
    chk('C⑦ 원점 집합이 팔마다 다르면 G-5 FAIL',
        _fails('G-5', [dict(r, origin_last=r['origin_last'] - 5)
                       if r['arm'] == 'SPEC25' else r for r in fx['rows']]))
    chk('C⑧ ★음성대조가 깨지면 G-6 FAIL',
        _fails('G-6', [dict(r, neg_lead0_bitmatch=False) for r in fx['rows']]))
    chk('C⑧b 음성대조 필드가 **없으면** G-6 FAIL(⛔부재를 통과로 찍지 않는다)',
        _fails('G-6', [{k: v for k, v in r.items() if k != 'neg_lead0_bitmatch'}
                       for r in fx['rows']]))
    chk('C⑨ ★양성대조가 깨지면 G-7 FAIL',
        _fails('G-7', [dict(r, pos_compose_relerr=1e-3) if r['arm'] == 'SPEC25' else r
                       for r in fx['rows']]))
    chk('C⑨b 양성대조 필드가 없으면 G-7 FAIL',
        _fails('G-7', [{k: v for k, v in r.items() if k != 'pos_compose_relerr'}
                       for r in fx['rows']]))
    chk('C⑩ rung 하나가 빠지면 G-8 FAIL',
        _fails('G-8', [dict(r, lead_rmse={k: v for k, v in r['lead_rmse'].items()
                                          if k != repr(1.0)}) for r in fx['rows']]))
    chk('C⑩b 원점이 하한 미만이면 G-8 FAIL',
        _fails('G-8', [dict(r, n_origins=MIN_ORIGINS - 1) for r in fx['rows']]))
    chk('C⑪ lead 프레임이 어긋나면 G-9 FAIL',
        _fails('G-9', [dict(r, lead_frames=dict(r['lead_frames'], **{repr(1.0): 69}))
                       for r in fx['rows']]))
    chk('C⑫ device=cuda 는 G-10 FAIL',
        _fails('G-10', [dict(r, device='cuda') for r in fx['rows']]))
    chk('C⑫b omp_threads=8 은 G-10 FAIL',
        _fails('G-10', [dict(r, omp_threads='8') for r in fx['rows']]))
    chk('C⑫c code_md5 분열은 G-10 FAIL',
        _fails('G-10', [dict(r, code_md5=('a' if i % 2 else 'b'))
                        for i, r in enumerate(fx['rows'])]))
    chk('C⑫d cuda_available 이 행마다 다르면 G-10 FAIL(값 고정은 안 하되 단일성은 요구)',
        _fails('G-10', [dict(r, cuda_available=(i % 2 == 0))
                        for i, r in enumerate(fx['rows'])]))
    chk('C⑬ ★G-11 은 위조 파일에서 반드시 FAIL (T2 개정 0 의 항진 시험)', True)
    probe = os.path.join(HERE, '_t4_flatref_probe.py')
    try:
        before, _ = no_flat_ref_constant()
        with open(probe, 'w') as fh:
            fh.write(_FLAT_NAME + ' = 1.0\nx = ' + _FLAT_NAME + '\n')
        after, _ = no_flat_ref_constant()
        chk('C⑬b 현 상태 통과 ∧ 위조 파일에서 FAIL', before and not after,
            '%s→%s' % (before, after))
    finally:
        if os.path.exists(probe):
            os.remove(probe)
    chk('C⑭ 개입 계수기가 어긋나면 G-12 FAIL',
        _fails('G-12', [dict(r, cap_forecast_steps=r['cap_forecast_steps'] + 1)
                        for r in fx['rows']]))
    chk('C⑮ 원점이 프로토콜과 다르면 G-13 FAIL',
        _fails('G-13', [dict(r, origin_first=r['origin_first'] + 1)
                        for r in fx['rows']]))
    bad_cmp = [dict(r, lead_zero={k: 0.0 for k in r['lead_zero']}) for r in fx['rows']]
    chk('C⑯ 비교자가 무기술이면 G-CMP FAIL ⟹ 라벨 VOID', _fails('G-CMP', bad_cmp))
    chk('C⑰ 행 0개면 차단 FAIL(⛔빈 자료를 통과로 찍지 않는다)',
        any((not o) for n, o, d, b in gates([], {}) if b))
    chk('C⑱ ★항등 게이트 0 — 관련 행을 다 지우면 어떤 차단 게이트든 FAIL',
        all(any((not o) for n, o, d, b in gates(
            [r for r in fx['rows'] if r['arm'] != a], fx['frozen']) if b)
            for a in ARM_ORDER))

    # ── 검사기 규약 · 임포트 생존 ────────────────────────────────────────────
    src = open(os.path.abspath(__file__)).read()
    chk('D① load_rows 가 glob.glob 무별칭 + json.loads 를 소스에 노출',
        'glob.glob(' in src and 'json.loads(' in src and 'def load_rows():' in src)
    chk('D② 코퍼스·시드가 동결물 유래(손복사 0)',
        EST_SEEDS == (42, 43, 44) and FILTER_SEEDS == (42, 43, 44) and N_EP == 48
        and SHELL_KEYS == list(R16.SHELL_KEYS) and len(ALL_SEEDS) == 12)
    chk('D③ 실측 상속: gates_b3 의 레코드·잡음·창', len(GB3.RECORDS) == 12
        and GB3.NOISE == 0.02 and len(GB3.WINDOWS) == 3)
    chk('D④ PRIMARY rung 이 사다리 안이고 밴드가 사다리 부분집합',
        PRIMARY_LEAD_TP in LEADS_TP and set(BAND_LEAD_TP) <= set(LEADS_TP))

    print('── gates_t4 자가진단 (자료 0행 · 런 0회) ──')
    nf = 0
    for n, o, d in ok:
        print(('  %s  %s' % ('PASS' if o else '⛔FAIL', n))
              + (('   %s' % (d,)) if (d and not o) else ''))
        nf += (not o)
    print('  ── A부 선별: GO = %s / NO-GO = %s'
          % (screen_go(), [n for n, gg, f, c in screen() if not gg]))
    t48 = thresholds(N_EP)
    print('  ── B부 n=%d  NOMINAL %d · CONFIRM %d · NOT_EARN %d · q80 %.4f · 바닥 p %.3e'
          % (t48['n'], t48['NOMINAL'], t48['CONFIRM'], t48['NOT_EARN'],
             t48['q80'], t48['floor_p']))
    print('  %d/%d PASS' % (len(ok) - nf, len(ok)))
    return nf


if __name__ == '__main__':
    sys.exit(1 if _selftest() else 0)
