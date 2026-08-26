"""B-20 단일출처 — **재적합 교락 통제** (2026-08-27).

⛔이 파일이 팔·문턱·게이트·라벨의 **유일한 출처**다. `run_b20.py` 와 `analyze_b20.py` 는
여기서 import 하고 **호출만** 한다(B-19 규약 상속: *"같은 기능이 두 파일에 구현돼 한쪽만
계측"* 부류를 만들지 않는다).

## 왜 — `B-16`/`B-19` 의 절제 팔은 **두 가지가 동시에 바뀐다**
`sequential_update.py:921` 이 `_snap_k_to_grid` 를 부르고, 그 함수는(`:939-979`)
  (1) 파수를 간격 `dk` 격자에 **반올림**하고,
  (2) `(w, φ)` 를 `learned_init.ls_fit_wphi`(→ `tnse.fit_amplitudes_phases`,
      **ridge 0.5 · 무가중 · 정규방정식**)로 **다시 맞춘다**.
그런데 배포 팔 `SHIP-NOW` 의 `(w, φ)` 는 그 직전 VARPRO(`varpro.py`,
`VP_RIDGE_DEFAULT=1e-8` · `weighting='confidence'` · `clamp=True`)의 해다.
⟹ `KGRID1` 은 `SHIP-NOW` 대비 **파수 양자화 + 적합 절차 교체** 두 가지가 바뀐 팔이고,
   `B-16`(−38.61%)·`B-19`(−67.43%)의 **크기**는 그 둘의 합이다.

★**용량-반응**(`KGRID1` vs `KGRID2`)은 두 팔이 **같은** `ls_fit_wphi` 를 통과하므로 이 교락에서
자유롭다 ⟹ *"효과의 **존재** 는 안전하고 **크기** 만 오염된다"* 가 현재 방어선이다.
**이 배치는 그 방어선을 측정으로 바꾼다.**

## 개입 — 팔 하나만 새로 만든다
`NOOP-REFIT` = `k_snap_dk = DK0`. `DK0` 는 **손으로 고르지 않았다**: 아래 `derive_dk0()` 가
채점 격자(x 10–250 m · y −100–100 m) · 채점 창 끝(t=30 s) · 분산관계 dω/dk 에서
*"파수 섭동이 렌더된 파랑장을 **float32 정밀도에서조차** 바꿀 수 없는"* 최대 간격을 도출한다.
⟹ `NOOP-REFIT` 은 **재적합만 발화하는 팔**이고, `G-15`/`G-15b` 가 그것을 런에서 직접 잰다.

## 판정 둘 (가족단위 α = 0.05/2)
  **A** `SHIP-NOW < NOOP-REFIT` — 재적합 **단독** 이 해치는가 (교락 검출)
  **B** `NOOP-REFIT < KGRID1` — **적합 절차를 고정한 채** 파수 양자화가 해치는가
★**B 가 이 배치의 하중을 받는다** — A 가 어떻게 나오든 B 는 `B-16`/`B-19` 를 대체할
*교락 없는* 절제 대조이고, 그 크기가 **새 인용값**이 된다.
⟹ 상세한 결과 전 해석 규칙은 `PREREG.md` §7 에 있다(여기서 반복하지 않는다).

## ⚠검정이 아닌 결정: **어느 크기를 인용하는가**
`shift_pp = |MAG_CLEAN − MAG_MAIN|` 를 `PP_BAND` 와 비교한다. `PP_BAND` 도 손으로 고르지
않았다 — `B-12` 가 **같은 대조·같은 참값·같은 코퍼스**에서 시드 부분집합만 달리해 낸
−38.73% 와 −41.09% 의 차(**2.36 pp**)이고, 그 배치의 사전등록이 **결과 전에** 그 둘을
*"결론 동일, 크기 추정 폭만 좁아짐"* 으로 읽기로 정해 뒀다 ⟹ **저장소가 스스로 "결론을
바꾸지 않는다" 고 선언한 크기 차**다. `pp_band()` 가 그 verdict 파일에서 **재도출**한다.
"""
import json
import math
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B19 = os.path.abspath(os.path.join(HERE, '..', 'B19_KGRID_SPARSE_2026-08-25'))
B16 = os.path.abspath(os.path.join(HERE, '..', 'B16_KGRID_2026-08-24'))
B12 = os.path.abspath(os.path.join(HERE, '..', 'B12_HOS_EXTEND_2026-08-23'))
B11 = os.path.abspath(os.path.join(HERE, '..', 'B11_HOS_BASELINE_2026-08-23'))
for _p in (HERE, ROOT, B19, B16, B11):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 표본·되읽기 규약·ZERO·격자·mesh 는 **`B-19` 동결 모듈에서 import** (손복사 0) ────────
#    ⚠`gates_b19.py` 는 이미 돌아간 배치의 동결 파일이다 — **건드리지 않는다**.
#    그 md5 를 `FROZEN_md5.txt` 에 실어 드리프트를 검사한다.
import gates_b19 as B19G                                       # noqa: E402
from gates_b19 import (DK1, EST_SEEDS, MESH_EXPECTED,          # noqa: E402
                       DERIVED_KEYS, DERIVED_EXPECTED, COUNTER_EXPECTED,
                       ROUTE_EXPECTED, UPDATER_CLASS_EXPECTED, DEPLOY_EXPECTED,
                       NO_DERIVED_EVIDENCE, ENV_PROBE_KEYS,
                       effective_from_updater, deploy_constants,
                       zero_nrmse, grid_dk, mesh_census, episodes, N_EP)

B19_RESULT_GLOB = os.path.join(B19, 'results_b19*.jsonl')
B12_VERDICT = os.path.join(B12, 'verdict_b12.json')

# ─────────────────────────────────────────────────────────────────────────────
# `DK0` 도출 — ⛔손으로 고르지 않는다 (`G-17` 이 자료에서 재도출해 이 핀과 대조한다)
# ─────────────────────────────────────────────────────────────────────────────
# 요구: `NOOP-REFIT` 의 파수 섭동이 **렌더된 파랑장을 float32 정밀도에서조차 바꿀 수 없다**.
#   장은 η = Σ w·cos(kx·x + ky·y − ω·t + φ) 이고 ω = √(g|k|tanh(|k|d)) 이므로
#   격자 반올림(성분마다 |δk| ≤ dk/2, 벡터로 |δ**k**| ≤ dk/√2)이 만드는 최대 위상 오차는
#       Δψ_max = (dk/2)·(|x|_max + |y|_max) + max|dω/dk|·(dk/√2)·t_max
#   ⟹ Δψ_max ≤ (dk/2)·L,  L = (|x|max + |y|max) + √2·max|dω/dk|·t_max.
#   `Δψ_max ≤ 2⁻²³`(float32 상대 정밀도) 를 걸고 **그 아래 최대 10의 거듭제곱**을 고른다.
#   ★10의 거듭제곱으로 내리는 이유: 상한 바로 밑의 임의값보다 **재현·기술이 쉽고**, 아래쪽
#     수치 하한(`round(k/dk)` 가 2⁵³ 근처로 가는 것)에서 여전히 10^10 배 이상 떨어져 있다.
EPS32 = 2.0 ** -23                 # float32 상대 정밀도 = 위상 오차 허용 상한 [rad]
PHASE_BOUND = EPS32
CG_KMIN = 0.01                     # dω/dk 최대를 취할 |k| 하한(λ=628 m — 도메인 240 m 밖)
# ★핀 (결과 전 도출값). `G-17` 이 `derive_dk0()` 로 재도출해 대조한다.
DK0 = 1e-10
XMAX_PIN, YMAX_PIN, TMAX_PIN = 250.0, 100.0, 30.0
LEVER_PIN = 1185.9504435618978     # L (단위: dk/2 당 rad·m) — `derive_dk0()` 출력
DK0_BOUND_PIN = 2.0103586991838653e-10   # 2·EPS32 / L


def derive_dk0(eps=None, cg_kmin=CG_KMIN):
    """채점 격자·채점 창·분산관계에서 `DK0` 를 **자료에서 다시 낸다** (런 0회).

    반환 dict: xmax, ymax, tmax, cg_max, lever, bound, dk0(= 상한 아래 최대 10의 거듭제곱).
    ⛔에피소드가 하나도 없으면 `None` 을 돌려준다(코퍼스 부재를 통과로 읽지 않는다).
    """
    import h5py
    import numpy as np
    eps = episodes() if eps is None else eps
    if not eps:
        return None
    xmax = ymax = tmax = 0.0
    depths = set()
    for e in eps:
        with h5py.File(e['path'], 'r') as h:
            gx = h['frames/gt_grid_x'][:]
            gy = h['frames/gt_grid_y'][:]
            ts = h['frames/timestamps'][:]
            depths.add(float(h['config'].attrs['depth']))
        xmax = max(xmax, float(np.abs(gx).max()))
        ymax = max(ymax, float(np.abs(gy).max()))
        tmax = max(tmax, float(np.abs(ts).max()))
    if len(depths) != 1:
        return None
    depth = depths.pop()
    from wavesplat.model.dispersion import G as GRAV        # 단일출처 (dispersion.py:18)
    k = np.linspace(cg_kmin, 1.0, 200001)
    w = np.sqrt(GRAV * k * np.tanh(k * depth))
    cg_max = float(np.gradient(w, k).max())
    lever = (xmax + ymax) + math.sqrt(2.0) * cg_max * tmax
    bound = 2.0 * PHASE_BOUND / lever
    dk0 = 10.0 ** math.floor(math.log10(bound))
    return dict(xmax=xmax, ymax=ymax, tmax=tmax, depth=depth,
                cg_max=cg_max, lever=lever, bound=bound, dk0=dk0)


def pp_band():
    """`PP_BAND` 를 `B-12` 의 **동결 verdict** 에서 재도출한다 (손복사 0).

    `B-12` 는 같은 대조(`SHIP < SPEC25`)·같은 HOS 참값·같은 코퍼스에서 **시드 부분집합만**
    달리해 `new_seeds_only` −38.73% 와 `old_seeds_only` −41.09% 를 냈고, 그 배치의 사전등록이
    **결과 전에** 그 둘을 *"결론 동일, 크기 추정 폭만 좁아짐"* 으로 읽기로 정해 두었다.
    ⟹ 그 차이가 **저장소가 스스로 '결론을 바꾸지 않는다' 고 선언한 크기 차**다.
    """
    with open(B12_VERDICT) as fh:
        v = json.load(fh)
    return abs(float(v['new_seeds_only']['rel_median_pct'])
               - float(v['old_seeds_only']['rel_median_pct']))


PP_BAND_PIN = 2.3596978918107183   # G-17b 가 `pp_band()` 로 재도출해 대조한다

# ── 팔 ────────────────────────────────────────────────────────────────────────
# ⛔빈 dict 로 팔을 정의하지 않는다(2026-08-23 R2 F-7: `{}` 는 런타임 기본값 스냅샷).
# ⚠`SHIP-NOW` 를 **맨 앞**에 둔다 — 러너의 온라인 드리프트 검사가 첫 런에서 발화하도록.
ARMS = {'SHIP-NOW':   {'k_snap_dk': None},
        'NOOP-REFIT': {'k_snap_dk': DK0},
        'KGRID1':     {'k_snap_dk': DK1}}
ARM_ORDER = ['SHIP-NOW', 'NOOP-REFIT', 'KGRID1']
# ★`B-19` 와 **같은 구성으로 다시 도는** 두 팔 (양성대조 G-11 / G-11b)
POSCTRL_ARMS = ('SHIP-NOW', 'KGRID1')
POSCTRL_RTOL = 0.0                 # 0 = 비트 동일 요구
ZERO_RTOL = 0.0                    # ZERO 도 같은 함수로 계산 ⟹ 비트 동일이어야 한다

# ── 판정 문턱 (코드가 계산한다) ───────────────────────────────────────────────
ALPHA = 0.05
FAMILY_ALPHA = 0.05 / 2            # 주장 둘: A · B (음성대조는 게이트이지 주장이 아니다)
MIN_N_USED = 14
THRESH_PIN = (12, 13, 4)           # G-18 이 `thresholds(16)` 및 `gates_b19` 와 대조
BITID_MIN = 1.0                    # G-15b: float32 파수 파라미터 비트 동일 비율
DK_ROUND_TOL = 1e-6                # 반올림 상한 검사의 상대 여유


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def thresholds(n):
    """(NOMINAL, CONFIRM, NOT_EARN). 형제 규약(`B-12`·`B-19`): NOT_EARN = n − NOMINAL."""
    nominal = next((w for w in range(n + 1) if sign_test_one_sided(w, n) <= ALPHA), None)
    confirm = next((w for w in range(n + 1)
                    if sign_test_one_sided(w, n) <= FAMILY_ALPHA), None)
    return nominal, confirm, (n - nominal if nominal is not None else None)


# ⚠**A 의 검출 문턱은 가족단위 보정을 하지 않는다**(α=0.05 = NOMINAL).
#   보정은 A 의 **널 주장**(= "교락 없음")을 *쉽게* 만드는 방향이므로, 보수적 읽기를 위해
#   일부러 느슨한 문턱으로 검출한다. 가족단위 보정(CONFIRM)은 **양성 주장 둘** —
#   A 의 `HELPS` 가지(w_A ≤ n−CONFIRM)와 B — 에만 건다. (결과 전 확정)


# ── `B-19` 동결 행 (양성대조 · ZERO 교차검증 · MAG_MAIN 대조) ─────────────────
def _b19_rows():
    import glob
    out = []
    for p in sorted(glob.glob(B19_RESULT_GLOB)):
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'error' not in r:
                    out.append(r)
    return out


def b19_ref(arm):
    """`B-19` 의 동결 행: {(episode, est_seed): avg_nRMSE}."""
    return {(r['episode'], r['est_seed']): float(r['avg_nRMSE'])
            for r in _b19_rows() if r.get('arm') == arm}


def b19_zero():
    """`B-19` 가 기록한 참 제로필드 기준: {episode: zero_nRMSE}."""
    return {r['episode']: float(r['zero_nRMSE']) for r in _b19_rows()
            if r.get('zero_nRMSE') is not None}


# ─────────────────────────────────────────────────────────────────────────────
# 집계 · 비교 (★단일출처: 게이트와 분석기가 **같은 함수**를 부른다)
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(rows):
    """median-of-3 집계 + 유효 표본 선정 (`B-19.aggregate` 와 같은 규약)."""
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
    """짝 비교 `a < b`. 동점은 승리로 세지 않는다(strict `<`, `B-16`·`B-19` 상속)."""
    import statistics as st
    common = [e for e in eps if (a, e) in val and (b, e) in val]
    return dict(
        n=len(common), better=sum(1 for e in common if val[(a, e)] < val[(b, e)]),
        n_tie=sum(1 for e in common if val[(a, e)] == val[(b, e)]),
        p_one_sided=(sign_test_one_sided(sum(1 for e in common
                                             if val[(a, e)] < val[(b, e)]), len(common))
                     if common else None),
        # ★저장소가 실제로 인용하는 통계량과 **같은 형태**를 쓴다 (verdict 들의 rel_median_pct)
        rel_median_pct=(st.median([(val[(a, e)] - val[(b, e)]) / val[(b, e)] * 100
                                   for e in common]) if common else None),
        median_a=st.median([val[(a, e)] for e in common]) if common else None,
        median_b=st.median([val[(b, e)] for e in common]) if common else None)


def cmp_zero(val, zero, eps, a):
    """음성 대조 `a < ZERO` (참 제로필드; ⛔상수 1.0 아님)."""
    import statistics as st
    common = [e for e in eps if (a, e) in val and e in zero]
    w = sum(1 for e in common if val[(a, e)] < zero[e])
    return dict(n=len(common), better=w,
                p_one_sided=sign_test_one_sided(w, len(common)) if common else None,
                rel_median_pct=(st.median([(val[(a, e)] - zero[e]) / zero[e] * 100
                                           for e in common]) if common else None))


LN_GUARD = math.log(1.05)          # 담지자 몫 서술의 분모 가드 (|d_total| ≥ 5%)


def decompose(val, eps):
    """★로그 분해 — 에피소드마다 **정확히 가법**이다 (서술용, 검정 아님).

        d_total(e) = ln(K/S) = d_refit(e) + d_quant(e)
        d_refit(e) = ln(N/S)         (재적합 단독)
        d_quant(e) = ln(K/N)         (적합 절차를 고정한 파수 양자화)
    """
    import statistics as st
    S, N, K = 'SHIP-NOW', 'NOOP-REFIT', 'KGRID1'
    common = [e for e in eps if all((a, e) in val for a in (S, N, K))]
    if not common:
        return None
    d_tot = [math.log(val[(K, e)] / val[(S, e)]) for e in common]
    d_ref = [math.log(val[(N, e)] / val[(S, e)]) for e in common]
    d_qnt = [math.log(val[(K, e)] / val[(N, e)]) for e in common]
    med_tot, med_ref, med_qnt = st.median(d_tot), st.median(d_ref), st.median(d_qnt)
    keep = [i for i, d in enumerate(d_tot) if abs(d) >= LN_GUARD]
    return dict(
        n=len(common),
        median_d_total=med_tot, median_d_refit=med_ref, median_d_quant=med_qnt,
        additivity_max_abs_err=max(abs(t - (r + q))
                                   for t, r, q in zip(d_tot, d_ref, d_qnt)),
        share_of_medians=(med_ref / med_tot if abs(med_tot) >= LN_GUARD else None),
        median_of_shares=(st.median([d_ref[i] / d_tot[i] for i in keep])
                          if keep else None),
        n_guard_dropped=len(common) - len(keep))


def magnitudes(val, eps):
    """★인용값 결정에 쓰는 두 크기 + 그 차 (검정 아님, `PREREG.md` §7.3)."""
    main = cmp_pair(val, eps, 'SHIP-NOW', 'KGRID1')['rel_median_pct']
    clean = cmp_pair(val, eps, 'NOOP-REFIT', 'KGRID1')['rel_median_pct']
    if main is None or clean is None:
        return dict(mag_main=None, mag_clean=None, shift_pp=None,
                    pp_band=None, magnitude_unmoved=None)
    band = pp_band()
    return dict(mag_main=main, mag_clean=clean, shift_pp=abs(clean - main),
                pp_band=band, magnitude_unmoved=bool(abs(clean - main) <= band),
                # ⚠상대형은 **서술용**이다. 결과 전에 **엄격한 쪽(절대형)** 을 골랐다.
                pp_band_relative_variant=abs(main) * band / 40.19406999827358)


# ─────────────────────────────────────────────────────────────────────────────
# 게이트 (★단일출처 — 분석기는 `gates(rows)` 를 **호출만** 한다)
# ─────────────────────────────────────────────────────────────────────────────
# ⛔**공허한 통과 금지** — 결과 0행에서 참이 되는 게이트를 만들지 않는다.
#    행 없이도 판정할 수 있는 게이트는 아래 넷뿐이고(자료·패키지만 읽는다), 그 넷도
#    **자기 입력이 비면 통과하지 않는다.**
ROWLESS_OK = ('G-5d', 'G-10', 'G-13', 'G-17', 'G-18')


def _srt(vals):
    return sorted(vals, key=lambda x: (x is None, repr(x)))


def _snap(r):
    return r.get('snap_rec') or {}


def gates(rows, agg=None):
    """러너 산출 전체(오류 행 포함)를 받아 `[(이름, 통과여부, 상세)]` 를 낸다."""
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
      and all(fired.get(a) == tot.get(a) and tot.get(a)
              for a in ('NOOP-REFIT', 'KGRID1')),
      ' · '.join(f'{a} {fired.get(a)}/{tot.get(a)}' for a in ARM_ORDER))

    # ★G-3b: `_snap_k_to_grid` 가 **통제 팔에서도 정확히 한 번** 불렸다 = 코드 경로가 같다.
    #   (SHIP-NOW 는 `if not dk: return None` 로 즉시 빠져나온다 — 그것이 no-op 의 정의다.)
    calls = _srt({r.get('snap_calls') for r in ok})
    g('G-3b 스냅 호출 계수 == 1 전 행 (통제 팔 포함 · 앙상블 중복호출 아님)',
      bool(ok) and calls == [1], f'snap_calls {calls}')

    dks = {a: _srt({r.get('k_snap_dk') for r in ok if r['arm'] == a}) for a in ARM_ORDER}
    g('G-4 dk 되읽기 (계수기 `init_result.k_snap.dk`)',
      bool(ok) and dks.get('SHIP-NOW') == [None] and dks.get('NOOP-REFIT') == [DK0]
      and dks.get('KGRID1') == [DK1], str(dks))

    # ── G-5 객체 파생 속성 (⛔cfg 항등식 아님 — `gates_b19` §3.1 규약 그대로) ────
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

    g('G-5b 행동 지문 init_loss_n == 0 (⛔None 은 통과 아님)',
      bool(ok) and all(r.get('init_loss_n') is not None and r['init_loss_n'] == 0
                       for r in ok),
      f"None {sum(1 for r in ok if r.get('init_loss_n') is None)}/{len(ok)} · "
      f'값 {_srt({r.get("init_loss_n") for r in ok})}')

    bad_cnt = sorted({k for r in ok for k, v in COUNTER_EXPECTED.items()
                      if (bool(r.get(k)) if isinstance(v, bool) else r.get(k)) != v})
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
    gates_fired = [r for r in ok if r.get('auto_real_gate') is not False]
    g('G-6 라우터 learned 전건 · ensemble_k == 1 · 파생 init_method == 경로 · 실측게이트 미발화',
      bool(ok) and routes == [ROUTE_EXPECTED] and eff_routes == [ROUTE_EXPECTED]
      and ens == [1] and not gates_fired,
      f'auto_real_gate≠False {len(gates_fired)}/{len(ok)} · auto_selected {routes} · '
      f'파생 init_method {eff_routes} · ensemble_k {ens} · '
      f"auto_n_selected 최소 {min(nsel) if nsel else None}")

    # ★G-7: 원자 수 > 0, 그리고 스냅 팔에서 **계수기의 원자 수와 일치**한다.
    n_bad = [r for r in ok if not (isinstance(r.get('n_gaussians'), int)
                                   and r['n_gaussians'] > 0)]
    n_mis = [r for r in ok if r['arm'] != 'SHIP-NOW'
             and _snap(r).get('n_atoms') != r.get('n_gaussians')]
    g('G-7 원자 수 > 0 · 스냅 계수기 n_atoms == n_gaussians (⛔0·None 통과 아님)',
      bool(ok) and not n_bad and not n_mis,
      f'n_gaussians 불량 {len(n_bad)}/{len(ok)} · 계수기 불일치 {len(n_mis)} · '
      f"값 {_srt({r.get('n_gaussians') for r in ok})}")

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

    eps_list = episodes()
    dk_bad, devs = [], []
    for e in eps_list:
        try:
            dkx, dky = grid_dk(e['path'])
        except Exception as exc:                             # noqa: BLE001
            dk_bad.append((e['episode'], repr(exc)))
            continue
        devs.append(abs(dkx - DK1))
        if abs(dkx - DK1) >= B19G.DK_REDERIVE_TOL or not dky > DK1:
            dk_bad.append((e['episode'], dkx, dky))
    g('G-10 격자 재도출 (|2π/Lx − DK1| < 1e-5 · 2π/Ly > DK1)',
      len(eps_list) == N_EP and not dk_bad,
      (f'에피소드 {len(eps_list)}/{N_EP} · 이탈 {len(dk_bad)} · '
       f'DK1 반올림 편차 최대 {max(devs):.3e}') if devs else
      f'에피소드 {len(eps_list)}/{N_EP} · 이탈 {len(dk_bad)}')

    # ── G-11 / G-11b ★★양성대조: 두 팔이 `B-19` 동결 행과 **비트 동일** ─────────
    for tag, arm in (('G-11', 'SHIP-NOW'), ('G-11b', 'KGRID1')):
        try:
            ref = b19_ref(arm)
        except Exception as exc:                             # noqa: BLE001
            g(f'★{tag} 양성대조 {arm} == B-19 동결 행', False,
              f'{type(exc).__name__}: {exc}')
            continue
        mine = [r for r in ok if r['arm'] == arm]
        hit = [r for r in mine if (r['episode'], r['est_seed']) in ref]
        bad = [(r['episode'], r['est_seed'], r['avg_nRMSE'],
                ref[(r['episode'], r['est_seed'])]) for r in hit
               if abs(r['avg_nRMSE'] - ref[(r['episode'], r['est_seed'])])
               > POSCTRL_RTOL * abs(ref[(r['episode'], r['est_seed'])])]
        g(f'★{tag} 양성대조: {arm} 가 B-19 동결 행과 비트 동일',
          bool(hit) and not bad and len(hit) == len(mine) and bool(mine),
          f'대조 {len(hit)}/{len(mine)} 행 · 불일치 {len(bad)}'
          + (f' 예: {bad[:1]}' if bad else ''))

    try:
        zref = b19_zero()
    except Exception:                                        # noqa: BLE001
        zref = {}
    zpairs = [(e, s, zref[e]) for e in zsets for s in zsets[e]
              if e in zref and s is not None]
    z_bad = [(e, s, r) for e, s, r in zpairs if abs(s - r) > ZERO_RTOL * abs(r)]
    g('★G-12 ZERO 교차검증 (같은 함수 ⟹ B-19 기록과 비트 동일)',
      bool(zpairs) and not z_bad and len(zpairs) == len(zsets),
      f'{len(zpairs) - len(z_bad)}/{len(zpairs)} 일치 (rtol {ZERO_RTOL})'
      + (f' 예: {z_bad[:1]}' if z_bad else ''))

    try:
        counts, _per = mesh_census(eps_list)
        g('★G-13 mesh_verdict 3-분할 = 사전등록 핀 (⛔표본을 "HOS M=3" 로 단일 서술 금지)',
          bool(eps_list) and counts == MESH_EXPECTED and sum(counts.values()) == N_EP,
          f'{counts} vs 핀 {MESH_EXPECTED} (n={sum(counts.values())}/{N_EP})')
    except Exception as exc:                                 # noqa: BLE001
        g('★G-13 mesh_verdict 3-분할 = 사전등록 핀', False, f'{type(exc).__name__}: {exc}')

    # ── G-14 ★★개입 격리를 **파라미터 수준**에서 잰다 ──────────────────────────
    #    SHIP-NOW  : (w,φ) 불변 **그리고** 파수 비트 동일 = `_snap_k_to_grid` 가 진짜 no-op
    #    NOOP/KGRID: (w,φ) 변함 = 재적합이 실제로 발화
    ship = [r for r in ok if r['arm'] == 'SHIP-NOW']
    ship_bad = [r for r in ship if r.get('w_phi_changed') is not False
                or _snap(r).get('fired') is not None
                or r.get('k_bit_identical_frac') != 1.0]
    refit_arms = [r for r in ok if r['arm'] in ('NOOP-REFIT', 'KGRID1')]
    refit_bad = [r for r in refit_arms if r.get('w_phi_changed') is not True]
    g('★G-14 개입 격리(파라미터): SHIP no-op · 두 스냅 팔에서 (w,φ) 재적합 발화',
      bool(ship) and bool(refit_arms) and not ship_bad and not refit_bad,
      f'SHIP 이탈 {len(ship_bad)}/{len(ship)} · 스냅 팔 (w,φ) 미변화 {len(refit_bad)}/'
      f'{len(refit_arms)} · NOOP dw_max 중앙 '
      f"{_median_or_none([r.get('w_abs_change_max') for r in ok if r['arm'] == 'NOOP-REFIT'])} "
      f"· dphi_max 중앙 "
      f"{_median_or_none([r.get('phi_abs_change_max') for r in ok if r['arm'] == 'NOOP-REFIT'])}")

    # ── G-15 ★★`NOOP-REFIT` 의 파수 개입이 **렌더된 장에서 검출 불가** ──────────
    noop = [r for r in ok if r['arm'] == 'NOOP-REFIT']
    psi = [r.get('k_dpsi_max') for r in noop]
    psi_bad = [r for r in noop if r.get('k_dpsi_max') is None
               or r['k_dpsi_max'] > PHASE_BOUND]
    round_bad = [r for r in noop if r.get('k_dk_max') is None
                 or r['k_dk_max'] > DK0 / math.sqrt(2) * (1 + DK_ROUND_TOL)]
    g('★G-15 NOOP-REFIT 파수 섭동 ≤ float32 장 정밀도 (Δψ_max ≤ 2⁻²³) · 반올림 상한 준수',
      bool(noop) and not psi_bad and not round_bad,
      f'Δψ_max 최대 {max((p for p in psi if p is not None), default=None)} '
      f'(허용 {PHASE_BOUND:.4e}) · 위반 {len(psi_bad)}/{len(noop)} · '
      f'반올림 상한 위반 {len(round_bad)} · '
      f"최소 |k| {min((r.get('k_min_mag') for r in noop if r.get('k_min_mag') is not None), default=None)}")

    # ⚠**G-15b 는 실패해도 PRIMARY 를 무효화하지 않는다**(결과 전 확정, PREREG §7.4).
    #   판정의 근거는 G-15(물리적 검출 불가)이고, 이것은 **더 강한 형태**(파라미터 비트 동일)다.
    bit_bad = [r for r in noop if r.get('k_bit_identical_frac') is None
               or r['k_bit_identical_frac'] < BITID_MIN]
    g('★G-15b (강한 형태, 무효화 아님) NOOP-REFIT 파수가 float32 파라미터에서 비트 동일',
      bool(noop) and not bit_bad,
      f'비동일 행 {len(bit_bad)}/{len(noop)} · 비율 최소 '
      f"{min((r.get('k_bit_identical_frac') for r in noop if r.get('k_bit_identical_frac') is not None), default=None)}")

    # ── G-16 ★NOOP-REFIT 이 SHIP-NOW 와 **다르다**(팔이 no-op 이 아니다) ────────
    ties = [e for e in good if ('SHIP-NOW', e) in val and ('NOOP-REFIT', e) in val
            and val[('SHIP-NOW', e)] == val[('NOOP-REFIT', e)]]
    g('★G-16 NOOP-REFIT ≠ SHIP-NOW (동점 셀 0 — 재적합이 결과를 실제로 바꾼다)',
      bool(good) and not ties, f'동점 {len(ties)}/{len(good)} 에피소드'
      + (f' 예: {ties[:2]}' if ties else ''))

    # ── G-17 ★DK0 재도출 (자료 부재에서 통과하지 않는다) ────────────────────────
    try:
        d = derive_dk0()
        okd = (d is not None and d['dk0'] == DK0
               and abs(d['lever'] - LEVER_PIN) < 1e-6 * LEVER_PIN
               and abs(d['bound'] - DK0_BOUND_PIN) < 1e-9 * DK0_BOUND_PIN
               and (d['xmax'], d['ymax'], d['tmax']) == (XMAX_PIN, YMAX_PIN, TMAX_PIN)
               and DK0 <= d['bound'] and PHASE_BOUND == EPS32)
        g('★G-17 DK0 자료 재도출 == 핀 (⛔손으로 고른 값 아님)', okd,
          (f"재도출 dk0={d['dk0']:g} bound={d['bound']:.6e} lever={d['lever']:.6f} "
           f"cg_max={d['cg_max']:.4f} (x,y,t)=({d['xmax']},{d['ymax']},{d['tmax']}) "
           f"depth={d['depth']} · 핀 dk0={DK0:g} · Δψ_max(DK0)={(DK0 / 2) * d['lever']:.4e} rad "
           f"= {(DK0 / 2) * d['lever'] / EPS32:.4f}×eps32 · "
           f"Δψ_max(DK1)={(DK1 / 2) * d['lever']:.4f} rad") if d else '코퍼스 0개 ⟹ 판정 불가')
    except Exception as exc:                                 # noqa: BLE001
        g('★G-17 DK0 자료 재도출 == 핀', False, f'{type(exc).__name__}: {exc}')

    try:
        band = pp_band()
        g('★G-17b PP_BAND 재도출 == 핀 (B-12 동결 verdict 에서)',
          abs(band - PP_BAND_PIN) < 1e-12, f'{band!r} vs 핀 {PP_BAND_PIN!r}')
    except Exception as exc:                                 # noqa: BLE001
        g('★G-17b PP_BAND 재도출 == 핀', False, f'{type(exc).__name__}: {exc}')

    # ── G-18 ★문턱 자기일치 + 형제 배치와의 대조 ────────────────────────────────
    try:
        mine_t, sib_t = thresholds(N_EP), B19G.thresholds(N_EP)
        g('★G-18 문턱 == 핀 (12/13/4) 이고 형제 배치(B-19)와 같다',
          mine_t == THRESH_PIN and sib_t == THRESH_PIN
          and abs(FAMILY_ALPHA - 0.025) < 1e-15,
          f'{mine_t} vs 핀 {THRESH_PIN} vs B-19 {sib_t} · family_alpha {FAMILY_ALPHA}')
    except Exception as exc:                                 # noqa: BLE001
        g('★G-18 문턱 == 핀', False, f'{type(exc).__name__}: {exc}')

    # ── N-1 / N-2 음성 대조 ────────────────────────────────────────────────────
    n_used = agg['n_used']
    nominal = thresholds(n_used)[0] if n_used else None
    neg1 = cmp_zero(val, zero, good, 'SHIP-NOW')
    neg2 = cmp_zero(val, zero, good, 'NOOP-REFIT')
    neg1_ok = bool(nominal is not None and neg1['n'] and neg1['better'] >= nominal)
    neg2_ok = bool(nominal is not None and neg2['n'] and neg2['better'] >= nominal)
    g(f'N-1 음성대조 SHIP-NOW < ZERO ≥ {nominal}/{n_used} (미달 ⟹ A·B 무효)',
      neg1_ok, f"{neg1['better']}/{neg1['n']} p={neg1['p_one_sided']}")
    g(f'N-2 음성대조 NOOP-REFIT < ZERO ≥ {nominal}/{n_used} (미달 ⟹ B 라벨 전환)',
      neg2_ok, f"{neg2['better']}/{neg2['n']} p={neg2['p_one_sided']}")
    return out


def _median_or_none(vals):
    import statistics as st
    v = [x for x in vals if x is not None]
    return round(st.median(v), 6) if v else None


def gate_ok(gts, prefix):
    hit = [okk for n, okk, _ in gts if n.startswith(prefix) or prefix in n]
    return bool(hit) and all(hit)


# ─────────────────────────────────────────────────────────────────────────────
# 라벨 (★결과 전 확정 — `PREREG.md` §7 의 표와 1:1)
# ─────────────────────────────────────────────────────────────────────────────
def label_a(w_a, n, n_tie_a, neg1_ok, g15_ok, magnitude_unmoved):
    """PRIMARY-A: 재적합 **단독** 이 교락인가 + 어느 크기를 인용하는가."""
    nominal, confirm, _ = thresholds(n) if n else (None, None, None)
    if n < MIN_N_USED:
        return ('SAMPLE-LOSS-UNDECIDED',
                f'유효 표본 {n} < {MIN_N_USED} ⟹ 문턱 재계산으로 결론을 사지 않는다')
    if not g15_ok:
        return ('CONTROL-NOT-ISOLATED',
                '⛔`NOOP-REFIT` 의 파수 섭동이 렌더 장에서 검출 가능하다 ⟹ 이 팔은 '
                '재적합 통제가 아니다. 교락에 대해 아무것도 말하지 않는다.')
    if not neg1_ok:
        return ('ARM-NO-SKILL-HERE',
                f'⛔`SHIP-NOW` 가 참 제로필드를 문턱({nominal}/{n})만큼 이기지 못한다 '
                '⟹ 뺄 기술이 없으므로 절제·통제 비교가 무효다')
    if n_tie_a == n:
        return ('REFIT-IS-EXACTLY-A-NOOP',
                f'★{n}/{n} 셀이 `SHIP-NOW` 와 **비트 동일** ⟹ 재적합이 결과를 전혀 바꾸지 '
                '않는다 = 교락이 존재하지 않는다. (⚠G-16 이 이 상태를 FAIL 로 잡는다 — '
                '그 경우 배선을 먼저 의심하라.)')
    if n - n_tie_a < nominal:
        return ('UNDECIDABLE-TIES',
                f'동점 {n_tie_a}/{n} ⟹ 최대 도달 가능 승수 {n - n_tie_a} < 문턱 {nominal} '
                '= 산술적으로 판정 불가. ⛔널을 "교락 없음" 으로 읽지 마라.')
    p = sign_test_one_sided(w_a, n)
    if w_a <= n - confirm:
        direction = 'HELPS'
    elif w_a >= nominal:
        direction = 'HARMS'
    else:
        direction = 'NULL'
    tail = (f'{w_a}/{n}(동점 {n_tie_a}), p={p:.4g}, 크기 이동 '
            f'{"밴드 안" if magnitude_unmoved else "밴드 밖"}')
    if direction == 'HARMS' and not magnitude_unmoved:
        return ('REFIT-CONFOUNDS-MAGNITUDE',
                f'⛔{tail} — 재적합 단독이 해치고 인용값도 밴드 밖으로 움직인다 ⟹ '
                '`B-16`/`B-19` 의 **크기 인용 금지**. 살아남는 것은 용량-반응과 PRIMARY-B 이고, '
                '인용값은 **PRIMARY-B 의 크기로 갈아 끼운다**.')
    if direction == 'HARMS':
        return ('REFIT-DIRECTIONAL-ONLY',
                f'⚠{tail} — 재적합이 해치는 **방향**은 실재하나 인용값을 밴드 밖으로 옮기지 '
                '못한다 ⟹ 크기는 유지하되 논문에 **교락의 존재를 명시**하고 PRIMARY-B 를 병기한다.')
    if direction == 'NULL' and magnitude_unmoved:
        return ('REFIT-IS-NOT-THE-CARRIER',
                f'★{tail} — 재적합 단독은 느슨한 문턱(α={ALPHA})에서도 검출되지 않고 인용값도 '
                '밴드 안이다 ⟹ `B-16`/`B-19` 의 **크기까지 인용 가능**하고 절제 논증이 온전하다.')
    if direction == 'NULL':
        return ('REFIT-SHIFTS-MAGNITUDE-ONLY',
                f'⚠{tail} — 부호로는 검출되지 않으나 인용값이 밴드 밖으로 움직인다 ⟹ '
                '⛔옛 크기 인용을 **보류**하고 **PRIMARY-B 의 크기**를 쓴다.')
    if magnitude_unmoved:
        return ('REFIT-HELPS-IMMATERIAL',
                f'★{tail} — 재적합 단독은 오히려 **낫다**(주 대조가 손해를 과소평가) ⟹ '
                '옛 크기는 **보수적**이므로 인용 가능하다.')
    return ('REFIT-HELPS-MAIN-IS-CONSERVATIVE',
            f'★{tail} — 재적합이 낫고 그만큼 주 대조가 손해를 **과소평가**한다 ⟹ 옛 크기는 '
            '하한으로만 쓰고, 인용값은 **PRIMARY-B 의 크기**(더 큰 쪽)로 갈아 끼운다.')


def label_b(w_b, n, n_tie_b, neg1_ok, neg2_ok, g15_ok):
    """PRIMARY-B: **적합 절차를 고정한** 파수 양자화 절제 (교락 없는 대조)."""
    nominal, confirm, not_earn = thresholds(n) if n else (None, None, None)
    if n < MIN_N_USED:
        return ('SAMPLE-LOSS-UNDECIDED', f'유효 표본 {n} < {MIN_N_USED}')
    if not g15_ok:
        return ('CONTROL-NOT-ISOLATED',
                '⛔통제 팔이 격리되지 않았다 ⟹ 이 대조도 순수한 양자화 절제가 아니다.')
    if not neg1_ok:
        return ('ARM-NO-SKILL-HERE', '⛔`SHIP-NOW` 가 이 층에서 무기술 ⟹ 절제가 무의미하다.')
    if n - n_tie_b < nominal:
        return ('QUANT-UNDECIDABLE-TIES',
                f'동점 {n_tie_b}/{n} ⟹ 도달 가능 승수 {n - n_tie_b} < 문턱 {nominal}')
    p = sign_test_one_sided(w_b, n)
    suffix = '' if neg2_ok else '-NO-SKILL-BASE'
    tailtxt = ('' if neg2_ok else ' ⚠기준 팔 `NOOP-REFIT` 이 참 제로필드를 문턱만큼 '
               '이기지 못한다 ⟹ ⛔상대 중앙 %를 margin 으로 인용 금지.')
    if w_b >= confirm:
        return ('QUANTIZATION-EARNS-INDEPENDENTLY' + suffix,
                f'★{w_b}/{n}(동점 {n_tie_b}), p={p:.4g} — **적합 절차를 고정한 채로도** 파수를 '
                '격자에 스냅하면 나빠진다 ⟹ 교락 없는 절제로 기여 ①이 지탱된다.' + tailtxt)
    if w_b >= nominal:
        return ('QUANTIZATION-EARNS-NOMINAL' + suffix,
                f'⚠{w_b}/{n}(동점 {n_tie_b}), p={p:.4g} — 명목만(가족단위 α={FAMILY_ALPHA} '
                '미달), 탐색적.' + tailtxt)
    if w_b <= not_earn:
        return ('QUANTIZATION-DOES-NOT-EARN',
                f'⛔{w_b}/{n}(동점 {n_tie_b}), p={p:.4g} — 적합 절차를 고정하면 양자화가 오히려 '
                '**낫다** ⟹ `B-16`/`B-19` 가 잰 것은 표현이 아니라 **적합 절차**다. '
                '기여 ① 문안을 다시 써야 한다.')
    return ('QUANTIZATION-UNDECIDED',
            f'{w_b}/{n}(동점 {n_tie_b}), p={p:.4g} — 문턱 사이. n=16 의 검정력 한계이고, '
            '⛔이 결과를 보고 표본을 늘리지 않는다(`B-8b` 정지규칙).')


# ─────────────────────────────────────────────────────────────────────────────
# ★런 전 게이트 자가진단 — **가짜 행으로 게이트가 실제로 발화하는지 확인한다**
# ─────────────────────────────────────────────────────────────────────────────
# ⛔파일을 쓰지 않는다. 추정기를 부르지 않는다(런 0회).
# ⚠공개: 픽스처가 `B-19` 동결 행을 쓰므로 **N-1 의 값(15/16)과 MAG_MAIN(−67.43%)이 런 전에
#   도출 가능**하다 — 그러나 그 둘은 `B-19` 가 이미 공개한 수이고 G-11/G-11b 가 비트 동일을
#   사전등록하므로 **새 정보가 아니다**. PRIMARY-A·B 는 픽스처가 임의 배수로 만든 값이라
#   아무 정보도 담지 않는다. `PREREG.md` §10 참조.
def _fixture_rows(noop_mult=1.03):
    ship, kg, zref, eps = (b19_ref('SHIP-NOW'), b19_ref('KGRID1'),
                           b19_zero(), episodes())
    rows = []
    for e in eps:
        name = e['episode']
        for arm in ARM_ORDER:
            for s in EST_SEEDS:
                if arm == 'SHIP-NOW':
                    v, snap = ship[(name, s)], None
                elif arm == 'KGRID1':
                    v = kg[(name, s)]
                    snap = dict(dk=DK1, n_atoms=3, n_unique_cells=3,
                                shift_median=0.0101, shift_max=0.0180, fired=True)
                else:
                    v = ship[(name, s)] * noop_mult
                    snap = dict(dk=DK0, n_atoms=3, n_unique_cells=3,
                                shift_median=3.5e-11, shift_max=5.9e-11, fired=True)
                rows.append(dict(
                    arm=arm, est_seed=s, episode=name, bf=e['bf'],
                    ncomp=e['ncomp'], gen_seed=e['gen_seed'], avg_nRMSE=v,
                    snap_rec=snap, snap_calls=1,
                    k_snap_dk=(snap or {}).get('dk'),
                    k_snap_fired=bool(snap), k_snap_cells=(snap or {}).get('n_unique_cells'),
                    k_snap_shift_median=(snap or {}).get('shift_median'),
                    # ★파라미터 수준 개입 계측 (러너의 스냅 훅이 채운다)
                    w_phi_changed=(arm != 'SHIP-NOW'),
                    w_abs_change_max=(None if arm == 'SHIP-NOW' else 0.012),
                    phi_abs_change_max=(None if arm == 'SHIP-NOW' else 0.031),
                    k_bit_identical_frac=(1.0 if arm != 'KGRID1' else 0.0),
                    k_dk_max=(0.0 if arm == 'SHIP-NOW' else
                              (5.9e-11 if arm == 'NOOP-REFIT' else 0.0180)),
                    k_dpsi_max=(0.0 if arm == 'SHIP-NOW' else
                                (5.93e-08 if arm == 'NOOP-REFIT' else 15.5)),
                    k_min_mag=0.043,
                    varpro_steps=400, varpro_fired=True, init_loss_n=0,
                    varpro_obj_before=1.0e-2, varpro_obj_after=4.0e-3,
                    auto_selected='learned', auto_n_selected=4,
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
                          [x for x in rs if x['arm'] == 'KGRID1'][:3]])),
        ('G-2', '한 행의 est_seed 를 99 로',
         _mut(lambda rs: first(rs, 'KGRID1').__setitem__('est_seed', 99))),
        ('G-3 ', 'NOOP-REFIT 한 행의 발화를 False 로',
         _mut(lambda rs: first(rs, 'NOOP-REFIT').__setitem__('k_snap_fired', False))),
        ('G-3b', '★스냅 호출이 0회(훅이 안 걸렸거나 경로가 다르다)',
         _mut(lambda rs: first(rs, 'SHIP-NOW').__setitem__('snap_calls', 0))),
        ('G-3b', '★앙상블 중복호출(snap_calls=3)',
         _mut(lambda rs: first(rs, 'NOOP-REFIT').__setitem__('snap_calls', 3))),
        ('G-4', 'NOOP-REFIT 한 행의 dk 를 1e-6 으로(위원장 제안값)',
         _mut(lambda rs: first(rs, 'NOOP-REFIT').__setitem__('k_snap_dk', 1e-6))),
        ('G-5 ', '파생 lambda_l1 을 1e-3 으로',
         _mut(lambda rs: first(rs, 'SHIP-NOW')['effective'].__setitem__('lambda_l1', 1e-3))),
        ('G-5 ', '파생 seed 를 est_seed 와 어긋나게',
         _mut(lambda rs: first(rs, 'SHIP-NOW')['effective'].__setitem__('seed', 7))),
        ('G-5 ', 'S8_L1_TRACK_ONLY 환경변수 흔적',
         _mut(lambda rs: first(rs, 'SHIP-NOW')['effective']['env_probe']
              .__setitem__('S8_L1_TRACK_ONLY', '1'))),
        ('G-5 ', '포착된 updater 가 0개(훅 미설치)',
         _mut(lambda rs: first(rs, 'SHIP-NOW')['effective'].__setitem__('n_captured', 0))),
        ('G-5b', 'init_loss_n 키가 사라짐(None)',
         _mut(lambda rs: [r.__setitem__('init_loss_n', None) for r in rs])),
        ('G-5c', 'VARPRO 건너뜀 (echo 400 · truthy True 유지, obj 만 없음)',
         _mut(lambda rs: first(rs, 'SHIP-NOW').update(varpro_obj_before=None,
                                                      varpro_obj_after=None))),
        ('G-6', 'auto_selected = v2_konly',
         _mut(lambda rs: first(rs, 'SHIP-NOW').__setitem__('auto_selected', 'v2_konly'))),
        ('G-6', 'auto_real_gate = True (실측 게이트 발화)',
         _mut(lambda rs: first(rs, 'SHIP-NOW').__setitem__('auto_real_gate', True))),
        ('G-7', 'n_gaussians = 0 (표현이 하나도 없다)',
         _mut(lambda rs: first(rs, 'SHIP-NOW').__setitem__('n_gaussians', 0))),
        ('G-7', '★스냅 계수기 n_atoms 가 n_gaussians 와 불일치',
         _mut(lambda rs: first(rs, 'NOOP-REFIT')['snap_rec'].__setitem__('n_atoms', 99))),
        ('G-8', '채점 창 3키가 전 행 None',
         _mut(lambda rs: [r.update(tracking_frames=None, scored_t_first=None,
                                   scored_t_last=None) for r in rs])),
        ('G-9', '한 행의 zero_nRMSE 를 None 으로',
         _mut(lambda rs: first(rs, 'KGRID1').__setitem__('zero_nRMSE', None))),
        ('★G-11 ', 'SHIP-NOW 한 행을 1e-12 만큼 어긋나게',
         _mut(lambda rs: first(rs, 'SHIP-NOW').__setitem__(
             'avg_nRMSE', first(rs, 'SHIP-NOW')['avg_nRMSE'] + 1e-12))),
        ('★G-11b', '★KGRID1 한 행을 1e-12 만큼 어긋나게 (B-19 재현 실패)',
         _mut(lambda rs: first(rs, 'KGRID1').__setitem__(
             'avg_nRMSE', first(rs, 'KGRID1')['avg_nRMSE'] + 1e-12))),
        ('★G-12', 'zero_nRMSE 를 rtol 밖으로',
         _mut(lambda rs: [r.__setitem__('zero_nRMSE', r['zero_nRMSE'] * 1.000001)
                          for r in rs if r['episode'] == rs[0]['episode']])),
        ('★G-14', '★SHIP-NOW 에서 (w,φ) 가 바뀌었다 (no-op 이 아니다)',
         _mut(lambda rs: first(rs, 'SHIP-NOW').__setitem__('w_phi_changed', True))),
        ('★G-14', '★NOOP-REFIT 에서 (w,φ) 가 안 바뀌었다 (재적합 미발화)',
         _mut(lambda rs: first(rs, 'NOOP-REFIT').__setitem__('w_phi_changed', False))),
        ('★G-14', '★w_phi_changed 키가 전 행 None (계측 부재)',
         _mut(lambda rs: [r.__setitem__('w_phi_changed', None) for r in rs])),
        ('★G-15 ', '★NOOP-REFIT 의 Δψ_max 가 float32 정밀도를 넘음',
         _mut(lambda rs: first(rs, 'NOOP-REFIT').__setitem__('k_dpsi_max', 2e-7))),
        ('★G-15 ', '★Δψ_max 가 전 행 None (계측 부재)',
         _mut(lambda rs: [r.__setitem__('k_dpsi_max', None) for r in rs])),
        ('★G-15 ', '★반올림 상한 위반 (|Δk| > dk/√2)',
         _mut(lambda rs: first(rs, 'NOOP-REFIT').__setitem__('k_dk_max', 1e-8))),
        ('★G-15b', '★파수가 float32 에서 비트 동일하지 않다',
         _mut(lambda rs: first(rs, 'NOOP-REFIT')
              .__setitem__('k_bit_identical_frac', 0.5))),
        ('★G-16', '★NOOP-REFIT 이 SHIP-NOW 와 비트 동일(팔이 no-op)',
         _mut(lambda rs: [r.__setitem__('avg_nRMSE',
                                        next(x for x in rs if x['arm'] == 'SHIP-NOW'
                                             and x['episode'] == r['episode']
                                             and x['est_seed'] == r['est_seed'])['avg_nRMSE'])
                          for r in rs if r['arm'] == 'NOOP-REFIT'])),
        ('N-1', 'SHIP-NOW 를 전부 ZERO 위로',
         _mut(lambda rs: [r.__setitem__('avg_nRMSE', r['zero_nRMSE'] * 2)
                          for r in rs if r['arm'] == 'SHIP-NOW'])),
        ('N-2', 'NOOP-REFIT 를 전부 ZERO 위로',
         _mut(lambda rs: [r.__setitem__('avg_nRMSE', r['zero_nRMSE'] * 2)
                          for r in rs if r['arm'] == 'NOOP-REFIT'])),
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
    say(f'[1/5] 픽스처 {len(base)}행 · 게이트 {len(res)}개 — 통과 {len(res) - len(bad)}/{len(res)}')
    for n, okk, d in res:
        if not okk:
            say(f'      ⛔{n}: {d}')
    if bad:
        fails.append(f'픽스처에서 통과하지 못한 게이트: {bad}')

    empty = gates([])
    leaked = [n for n, okk, _ in empty
              if okk and not any(p in n for p in ROWLESS_OK)]
    say(f'[2/5] rows=[] — 통과 {sum(1 for _, o, _ in empty if o)}/{len(empty)} '
        f'(허용 {list(ROWLESS_OK)}) · 공허한 통과 {len(leaked)}')
    for n in leaked:
        say(f'      ⛔공허한 통과: {n}')
    if leaked:
        fails.append(f'0행에서 통과한 게이트: {leaked}')

    n_flip = 0
    for prefix, desc, mut in _perturbations():
        r2 = gates(mut(base))
        hit = [(n, okk) for n, okk, _ in r2 if n.startswith(prefix.strip())
               or prefix.strip() in n]
        if not hit:
            fails.append(f'{prefix}: 그 이름의 게이트가 없다 ({desc})')
            say(f'      ⛔{prefix} 게이트 없음 — {desc}')
            continue
        if all(okk for _, okk in hit):
            fails.append(f'{prefix}: 변형에도 통과 — {desc}')
            say(f'      ⛔{prefix} 발화 안 함 — {desc}')
        else:
            n_flip += 1
    say(f'[3/5] 변형 {len(_perturbations())}건 중 게이트가 발화한 것 {n_flip}')

    step4 = []
    saved_eps, saved_mesh = globals()['episodes'], globals()['MESH_EXPECTED']
    saved_pin = globals()['DK0_BOUND_PIN']
    try:
        globals()['episodes'] = lambda: []
        r3 = gates(base)
        for tag in ('G-10', 'G-13', 'G-17 '):
            if any(okk for n, okk, _ in r3 if tag.strip() in n and 'G-17b' not in n):
                step4.append(f'{tag}: 에피소드 0개에서 통과(공허)')
        globals()['episodes'] = saved_eps
        globals()['MESH_EXPECTED'] = {'ADEQUATE': 16}
        if any(okk for n, okk, _ in gates(base) if 'G-13' in n):
            step4.append('G-13: 틀린 핀에도 통과')
        globals()['MESH_EXPECTED'] = saved_mesh
        globals()['DK0_BOUND_PIN'] = 1.0
        if any(okk for n, okk, _ in gates(base) if 'G-17 ' in n):
            step4.append('G-17: 틀린 핀에도 통과')
    finally:
        globals()['episodes'] = saved_eps
        globals()['MESH_EXPECTED'] = saved_mesh
        globals()['DK0_BOUND_PIN'] = saved_pin
    fails += step4
    say(f'[4/5] 자료 부재·틀린 핀에서 G-10·G-13·G-17 이 통과하지 않는다: '
        f'{"OK" if not step4 else "⛔ " + "; ".join(step4)}')

    # ── 5. 라벨 함수가 결과 전 표와 1:1 인가 (분기 전수) ────────────────────────
    n = N_EP
    cases = [
        (label_a(16, n, 0, True, True, False)[0], 'REFIT-CONFOUNDS-MAGNITUDE'),
        (label_a(16, n, 0, True, True, True)[0], 'REFIT-DIRECTIONAL-ONLY'),
        (label_a(8, n, 0, True, True, True)[0], 'REFIT-IS-NOT-THE-CARRIER'),
        (label_a(8, n, 0, True, True, False)[0], 'REFIT-SHIFTS-MAGNITUDE-ONLY'),
        (label_a(0, n, 0, True, True, True)[0], 'REFIT-HELPS-IMMATERIAL'),
        (label_a(0, n, 0, True, True, False)[0], 'REFIT-HELPS-MAIN-IS-CONSERVATIVE'),
        (label_a(8, n, 16, True, True, True)[0], 'REFIT-IS-EXACTLY-A-NOOP'),
        (label_a(8, n, 6, True, True, True)[0], 'UNDECIDABLE-TIES'),
        (label_a(8, n, 0, False, True, True)[0], 'ARM-NO-SKILL-HERE'),
        (label_a(8, n, 0, True, False, True)[0], 'CONTROL-NOT-ISOLATED'),
        (label_a(8, 10, 0, True, True, True)[0], 'SAMPLE-LOSS-UNDECIDED'),
        (label_b(14, n, 0, True, True, True)[0], 'QUANTIZATION-EARNS-INDEPENDENTLY'),
        (label_b(14, n, 0, True, False, True)[0],
         'QUANTIZATION-EARNS-INDEPENDENTLY-NO-SKILL-BASE'),
        (label_b(12, n, 0, True, True, True)[0], 'QUANTIZATION-EARNS-NOMINAL'),
        (label_b(2, n, 0, True, True, True)[0], 'QUANTIZATION-DOES-NOT-EARN'),
        (label_b(8, n, 0, True, True, True)[0], 'QUANTIZATION-UNDECIDED'),
    ]
    bad_lab = [(got, want) for got, want in cases if got != want]
    say(f'[5/5] 라벨 분기 {len(cases)}건 전수 — 불일치 {len(bad_lab)}'
        + (f' {bad_lab}' if bad_lab else ''))
    if bad_lab:
        fails.append(f'라벨 분기 불일치: {bad_lab}')

    okk = not fails
    say(('✅ 게이트 자가진단 통과' if okk else '⛔ 게이트 자가진단 실패:\n  - '
         + '\n  - '.join(fails)))
    return okk, '\n'.join(log)


if __name__ == '__main__':
    if '--derive' in sys.argv:
        print(json.dumps(derive_dk0(), indent=1))
        print('PP_BAND =', pp_band())
        print('thresholds(16) =', thresholds(N_EP))
        sys.exit(0)
    ok_, _ = selftest()
    sys.exit(0 if ok_ else 1)
