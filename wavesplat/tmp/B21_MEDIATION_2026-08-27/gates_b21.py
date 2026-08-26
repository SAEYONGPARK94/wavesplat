"""B-21 단일출처 — **매개(mediation)를 한 배치에서 정식으로 잰다** (2026-08-27).

## 왜 이 파일이 있는가
2026-08-26 토론에서 위원장이 §0 에 **매개 주장**을 세웠다: *"우리의 연속 파수를 기준선의
격자 간격으로 양자화하면(`KGRID1`) 기준선(`SPEC25`) 대비 마진이 사라진다."*
그 수는 `B-19`(희소 절제)의 `KGRID1` 열과 `B-11b`(희소 기준선)의 `SPEC25` 열을
**서로 다른 배치에서 꺼내 붙여** 만든 것이고, ⛔**`B-19` 사전등록이 결과 전에 그 연산을
금지했다**(`tmp/B19_KGRID_SPARSE_2026-08-25/PREREG.md:417` *"외부 기준선 비교가 아니다.
이 배치에 `SPEC25`·LETKF 행은 0개다."* · `:419` 크기 직접 비교 금지).
⟹ ★**소수의견의 처방이 옳다: 세 팔을 한 배치에서 사전등록해 다시 재는 수밖에 없다.**
그것이 이 배치다.

⛔**이 배치는 `B-16`·`B-19`·`B-11b`·`B-12` 의 판정을 대체하거나 수정하지 않는다.**
그 배치들의 라벨·문턱·게이트·결과 파일은 **건드리지 않았다.**

## 무엇이 새로 측정되는가 (교차배치 결합이 낼 수 없는 것)
1. **한 배치·한 코퍼스·한 코드 상태**에서 다섯 팔이 동시에 돈다 ⟹ `git_head` + 임포트 md5 +
   ★**채점 창 일치(G-8)** 가 검사된다. `results/b1_full_comparison.json` 이 죽은 결함
   (두 팔이 **같은 프레임에서 채점된 적이 없다**)은 교차배치 결합으로는 **원리적으로**
   확인할 수 없다.
2. ★★**교락을 추론이 아니라 개입으로 잰다** — `FITSWAP` 팔(아래).
3. **`KGRID2` vs `SPEC25`** 는 어느 배치에도 없다. 매개 가설의 **방향성 예측**을 여기서 처음 건다.

## ★★교락 — 양쪽 심사가 1순위로 지목한 것
`sequential_update.py:921 → :939 _snap_k_to_grid` 는 **두 가지를 동시에 바꾼다**:
  (a) 파수를 간격 `dk` 격자에 반올림한다 (양자화)
  (b) `(w, φ)` 를 `learned_init.ls_fit_wphi`(**ridge 0.5 · 무가중**)로 **다시 적합**한다
      — 배포 `SHIP` 의 `(w, φ)` 는 VARPRO(`varpro.py`, `VP_RIDGE_DEFAULT=1e-8` ·
      `weighting='confidence'`)의 해다.
⟹ `KGRID1` 은 *"연속 파수를 뺀 팔"* 이 아니라 *"연속 파수를 빼고 적합 절차도 바꾼 팔"* 이다.

★**`FITSWAP` 이 그 (b) 를 단독으로 분리한다**: `k_snap_dk = 1e-12` 로 돌린다.
`round(kx/1e-12)*1e-12` 는 kx 를 상대오차 ~1e-16 으로 되돌려주므로 **양자화는 수치적으로
존재하지 않고**(1셀 실측 `shift_median = 1.77e-13` = `DK1` 의 **6.8e-12 배**), 남는 변화는
**(b) 뿐**이다. ⟹ ★***교락의 크기가 추정이 아니라 측정치가 된다.***
⚠**패키지를 고치지 않는다** — 이미 동결·배포된 `k_snap_dk` 노브를 그대로 쓴다.

⚠**`FITSWAP` 이 재는 것의 정확한 범위**: `_snap_k_to_grid` 는 끝에서
`_build_tracking_optimizer()` 를 **한 번 더** 부른다(`:969`). 그 함수는 독스트링이 적은 대로
*"creates a fresh Adam optimizer"* 이고(`:503-508`), 사이에 낀 VARPRO 는 **자기 지역
옵티마이저**를 쓴다(`varpro.py:126 opt = torch.optim.Adam([kx, ky], ...)`) ⟹ 두 번째 재구성은
상태가 비어 있어 **무해**하다. 그래도 안전하게 읽는다: `FITSWAP − SHIP` 은
**(재적합 + 옵티마이저 재구성)의 상한**이고, 상한이 작으면 결론에 충분하다.

## 팔 다섯 (⛔빈 dict 금지 · 배포 상수는 단일출처에서 읽어 명시)
| 팔 | 무엇 | 어떻게 |
|---|---|---|
| `SHIP`    | 배포 = 연속 파수 | `k_snap_dk=None` |
| `KGRID1`  | 표현 절제 (Δk) | `k_snap_dk=DK1` |
| `KGRID2`  | 용량-반응 (2Δk) | `k_snap_dk=DK2` |
| `FITSWAP` | ★교락 단독 | `k_snap_dk=1e-12` (= 양자화 없음, 재적합만) |
| `SPEC25`  | 외부 기준선 | `S16` 의 `run_s16.run_letkf(path,'SPEC25',filter_seed=s)` **그대로 import** |

## ⛔`SPEC25-FINE` 은 **없다** — 코드에서 확인하고 뺐다 (PREREG §5 에 전문)
세 단계로 확인했다(전부 실측):
 1. **노드 세분(40×30 → 79×59)은 Δk 를 세분하지 않는다.** 배경오차 모형의 빈 간격은
    `letkf_spec.py:177 dkx = 2π/(nx·dx)` 이고 `nx·dx = L·N/(N−1)` 이므로 노드를 두 배로
    해도 **0.025525 → 0.025849 (+1.3%)** 다. 바뀌는 것은 Nyquist(0.5105 → 1.0210)다.
 2. **Δk 를 정확히 반으로 만드는 유일한 구성은 정의역 2배**다(같은 dx, 79×59):
    **0.025525 → 0.012924**. 노드 수가 같아 **비용도 같다**.
 3. ⛔**그런데 둘 다 도달 불가능하다.** `run_s16.run_letkf:203-204` 가 해석 격자를
    **지역 변수**로 박아 두고 `:232` 가 **같은 지역 변수**로 채점 격자 보간기를 만든다
    ⟹ 필터의 격자만 바꾸면 그 자리에서 죽는다(실측 traceback:
    `ValueError: There are 30 points and 59 values in dimension 0`).
    구현하려면 **이미 돌아간 배치의 동결 파일을 고치거나** `run_letkf` 를 **복제**해야 하고,
    둘 다 이 저장소가 스스로 금지한 것이다(후자는 *"같은 기능이 두 파일에" * 결함 부류).
⟹ **팔에서 뺐다.** 대신 **도달 가능한 반쪽 대칭 검정**을 건다 = `KGRID2` vs `SPEC25`(아래 S).

## 판정 (전부 결과 전 확정 · 문턱은 `math.comb` 로 코드가 계산)
가족 = **주장 6개** (P1·P2 · S1·S2 · C1·C2). 음성대조·양성대조는 **게이트이지 주장이 아니다**
(`B-15`·`B-16`·`B-19` 규약 상속). **가족단위 α = 0.05/6**.
n=16 → NOMINAL 12 · CONFIRM 14 · n=48 → NOMINAL 31 · CONFIRM 33 (코드가 낸다).

**P (PRIMARY, 층별)** = `KGRID1` vs `SPEC25`
  · `w ≥ CONFIRM`          ⟹ ⛔`MEDIATION-REFUTED` (양자화해도 마진이 남는다)
  · `w ≤ n − CONFIRM`      ⟹ `OVERSHOOT-BELOW-BASELINE` (절제가 기준선 **아래로** 떨어뜨렸다)
                              ★이것은 매개가 **아니다** — 결과 전에 못 박는다.
  · 양쪽 다 아니고 **TOST 통과** ⟹ ★`MEDIATION-CONFIRMED`
  · 양쪽 다 아니고 TOST 실패 ⟹ `MEDIATION-UNDECIDED`
⚠**부호검정의 널만으로 *"사라진다"* 를 쓰지 않는다**(소수의견 D-2). 등가는 TOST 가 말한다.

**S (SECONDARY, 층별)** = `KGRID2` vs `SPEC25` — ★매개가 참이면 **방향성 예측**이 있다:
2Δk 는 기준선의 분해능보다 **거친** 표현이므로 공통 바닥 그림이 옳다면 우리가 **진다**.
  · `w2 ≤ n − CONFIRM` ⟹ `DOSE-CROSSES-BASELINE` (예측대로)
  · `w2 ≥ CONFIRM`     ⟹ `DOSE-DOES-NOT-CROSS`
  · 그 사이            ⟹ `DOSE-CROSS-UNDECIDED`

**C (교락, 층별)** = `FITSWAP` vs `SHIP`
  · TOST(Δ_conf) 통과 ⟹ `CONFOUND-NEGLIGIBLE`
  · TOST 실패 + 부호검정이 가족단위 α 에서 유의 + |상대 중앙| ≥ Δ_conf ⟹ `CONFOUND-MATERIAL`
  · 그 밖 ⟹ `CONFOUND-UNDECIDED`

**A (기술적 분해, ⛔검정 아님)** — 절제 효과 중 재적합이 차지하는 몫
`share = median(FITSWAP−SHIP) / median(KGRID1−SHIP)`. p 값을 붙이지 않고 라벨도 없다.

## TOST (동등성) — 등가 한계를 **결과 전에** 정한다
분포무관 **부호-TOST**: 짝 상대차 `d_e = (A_e − B_e)/B_e × 100` 에 대해
`#{d < +Δ} ≥ k_min(n, .05)` **그리고** `#{d > −Δ} ≥ k_min(n, .05)` 이면 등가.
(같은 `sign_test_one_sided` 를 쓴다 — 새 통계량을 만들지 않는다.)
· **Δ = 10.0 %** (P1·P2). 도출: 이 주장이 지우려는 **동결 마진**이 희소 −74.7%(`B-11b`) ·
  혼잡 −40.19%(`B-12`) 다. 10% 는 **둘 중 작은 것의 1/4 이하** ⟹ 등가는
  *"혼잡 마진의 75% 이상, 희소 마진의 86% 이상이 지워졌다"* 를 뜻한다. 두 층 공통(층별 조정 없음).
· **Δ_conf = 5.0 %** (C1·C2). 도출: 교락이 설명할 수 있어야 하는 절제 효과가
  `SHIP` 기준으로 혼잡 ≈ +63%(`B-16` −38.61%) · 희소 ≈ +207%(`B-19` −67.43%) 다.
  5% 는 **작은 쪽의 1/12 이하** ⟹ 등가는 *"재적합은 절제 효과의 8% 미만"* 을 뜻한다.
· TOST 는 **각 측 α=0.05**(표준 TOST). ⛔**Δ 를 어떤 관측된 차이에도 맞추지 않았다** —
  특히 위원장의 21/48·8/16 과 소수의견의 CI `[−13.4%, +11.7%]` 는 **Δ 도출에 쓰지 않았다**
  (그 CI 에 맞추면 순환이다). ★**이 사전등록을 쓰는 동안 동결 파일에서
  `KGRID1` vs `SPEC25` 를 계산하지 않았다.**
⚠**Δ=10% 에서 n=16 의 등가가 통과하려면 최소 8/16 이 밴드 안에** 있어야 한다
(위 4개 · 아래 4개까지만 허용). 희소의 시드간 CV 가 46.5% 인 층이므로 **실패가 흔할 것**이고,
그때 라벨은 `MEDIATION-UNDECIDED` 이지 ⛔`REFUTED` 가 아니다.

## ★★결과 전 해석 규칙 — **어느 결과가 나와도 무엇을 쓸지 미리 정한다**
1. `MEDIATION-CONFIRMED` ⟹ ⛔**novelty 로 읽지 마라.** 매개가 참이면 그것은
   *"두 추정기가 **공통 이산화 바닥**에 앉는다"* 는 뜻이고, 그것은 off-grid 문헌의
   **교과서 예측**이다(Rife & Boorstyn 1974 · Tang 2013 · Candès–Fernandez-Granda 2014).
   ⟹ 쓸 수 있는 문장은 ***"선점된 기제가 이 응용에서 갖는 지분의 계량"*** 이지
   *"새로운 기제의 발견"* 이 아니다. **이 규칙은 결과 전에 박는다.**
2. `MEDIATION-REFUTED` ⟹ 위원장 §0 의 매개 서사는 **이 표본에서 지탱되지 않는다**.
   ⛔그렇다고 *"기여 ①이 강화됐다"* 로 승격하지 않는다 — `B-16`·`B-19` 가 이미 낸 판정이
   그대로일 뿐이고, 이 배치는 **교차배치 결합을 정식 측정으로 바꾼 것**이다.
3. `MEDIATION-UNDECIDED` ⟹ ★**이것이 가장 흔할 것**이고 **정직한 결과**다.
   쓸 문장 = *"n=16/48 로는 등가를 세울 수 없다"*. ⛔*"차이가 없다"* 로 쓰지 마라.
4. `OVERSHOOT-BELOW-BASELINE` ⟹ 절제가 **공통 바닥보다 아래로** 갔다는 뜻이다.
   그 경우 P 의 널/등가 여부와 무관하게 **매개 주장은 지지되지 않는다**(바닥이 공통이 아니다).
5. `CONFOUND-MATERIAL` ⟹ ★**`B-16`·`B-19` 의 크기는 그만큼 오염돼 있다**. 그러나
   **존재는 오염되지 않는다** — 용량-반응(`KGRID1` vs `KGRID2`)은 두 팔이 **같은
   `ls_fit_wphi` 를 통과**하므로 이 교락에서 자유롭다(양쪽 심사가 합의한 논점).
   ⟹ 그 경우 논문은 **크기 인용을 내리고 존재만 유지**한다.
6. `CONFOUND-NEGLIGIBLE` ⟹ `KGRID1` 의 효과는 **양자화에 귀속 가능**해진다.
   ⛔단 그것은 *"이 코퍼스에서"* 이고, 소수의견의 지적(`k_snap_shift_median = 0.0104` 가
   측정된 분해능 한계 0.042–0.074 의 **1/4 이하**)은 **남는다** — 분해능보다 작은 섭동이
   큰 효과를 낸다는 사실 자체가 설명을 요구한다. **한계 절에 그대로 적는다.**
7. **층이 갈리면**(예: 혼잡 CONFIRMED · 희소 REFUTED) ⛔합산하지 않는다. 두 층을 **따로**
   적는다(`B-19` §9-2 · `S13` 이 여기서 사전등록을 이탈했다).

## ⛔주장하지 않는 것
· ⛔**`SPEC25-FINE` 을 못 돌렸으므로 대칭 검정의 절반은 비어 있다.** *"기준선을 세분해도
  안 좋아진다"* 는 문장을 **쓸 수 없다**(측정이 없다. 부재증명 아님).
· ⛔`B-1`·`B-12`·`B-16`·`B-19` 의 크기와 **직접 비교하지 않는다**(다른 팔 구성·다른 가족).
· ⛔`−38.58%` 소수 둘째 자리 인용 금지(통계적으로 구별 안 되는 세 팔이 −38.6/−41.0/−43.5).
· ⚠**체크포인트가 평가와 같은 분포**라는 단서는 이 배치가 건드리지 않는다.
· ⚠**희소 층에서는 기준선 자신이 무기술을 간신히 넘을 뿐**이다(`B-11b`: `SPEC25<ZERO`
  12/16 p=0.038). N-2 가 그것을 매 층 다시 잰다.
"""
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B19 = os.path.abspath(os.path.join(HERE, '..', 'B19_KGRID_SPARSE_2026-08-25'))
B16 = os.path.abspath(os.path.join(HERE, '..', 'B16_KGRID_2026-08-24'))
B12 = os.path.abspath(os.path.join(HERE, '..', 'B12_HOS_EXTEND_2026-08-23'))
B11 = os.path.abspath(os.path.join(HERE, '..', 'B11_HOS_BASELINE_2026-08-23'))
S16 = os.path.abspath(os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14'))
for _p in (HERE, ROOT, B19, B16, B12, B11, S16):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 단일출처 재사용 (⛔손복사 0 — 판정에 쓰는 것은 전부 import 한다) ──────────
#    ⚠`DK1`/`DK2` 가 `gates_b16.py:48-49` 의 **반올림 리터럴**이라는 사실은 `B-19` §4.1 이
#      이미 공개했고 그 편차(절대 6.12e-08)를 G-10 이 자료에서 재도출해 다시 잰다.
from gates_b19 import (DK1, DK2, SPARSE_MAX_NCOMP, TP_FACTOR,      # noqa: E402
                       DK_REDERIVE_TOL, DERIVED_KEYS, DERIVED_EXPECTED,
                       ROUTE_EXPECTED, UPDATER_CLASS_EXPECTED, COUNTER_EXPECTED,
                       DEPLOY_EXPECTED, ENV_PROBE_KEYS, NO_DERIVED_EVIDENCE,
                       AUTO_SWITCH_THRESH_PRE_UNIFY,
                       effective_from_updater, deploy_constants, zero_nrmse, grid_dk)
from gates_b11b import HOS, SPARSE_SEEDS                            # noqa: E402
from gates_b11 import EST_SEEDS                                     # noqa: E402
from gates_b12 import ALL_SEEDS as CROWDED_SEEDS                    # noqa: E402

FILTER_SEEDS = EST_SEEDS          # SPEC25 도 median-of-3 (B-11b·B-12 규약)

# ── 팔 ────────────────────────────────────────────────────────────────────────
# ⛔빈 dict 로 팔을 정의하지 않는다 (2026-08-23 R2 F-7 · gates_b7.py:46 사고).
FITSWAP_DK = 1e-12                # ★양자화가 수치적으로 없는 값 = 재적합만 남는다
ARMS = {'SHIP':    {'k_snap_dk': None},
        'KGRID1':  {'k_snap_dk': DK1},
        'KGRID2':  {'k_snap_dk': DK2},
        'FITSWAP': {'k_snap_dk': FITSWAP_DK}}
ARM_ORDER = ['SHIP', 'KGRID1', 'KGRID2', 'FITSWAP']
BASELINE_ARM = 'SPEC25'
ALL_ARMS = ARM_ORDER + [BASELINE_ARM]

# `SPEC25` 정의는 손으로 옮기지 않는다 — `run_s16.ARMS['SPEC25']` 가 권위다.
SPEC_EXPECTED = {'loc_radius': 25.0, 'n_ensemble': 600, 'background': 'jonswap',
                 'filter_class': 'LETKFGridSpec', 'nx': 40, 'ny': 30}

# ★개입 크기 게이트 — *"설정을 넣었다 ≠ 설정이 먹었다"*(W7) 를 팔마다 다르게 검사한다.
KGRID_SHIFT_MIN = 1e-4      # KGRID1/2 는 파수를 실제로 옮겨야 한다
FITSWAP_SHIFT_MAX = 1e-9    # FITSWAP 은 옮기면 안 된다 (= 양자화 없음의 직접 증거)

# ── 층 ────────────────────────────────────────────────────────────────────────
STRATA = ('sparse', 'crowded')
STRATUM_SEEDS = {'sparse': tuple(SPARSE_SEEDS), 'crowded': tuple(CROWDED_SEEDS)}
STRATUM_N = {'sparse': 16, 'crowded': 48}

# ── 통계 ──────────────────────────────────────────────────────────────────────
ALPHA = 0.05
N_CLAIMS = 6                       # P1 P2 S1 S2 C1 C2 (음성·양성대조는 게이트)
FAMILY_ALPHA = ALPHA / N_CLAIMS
TOST_ALPHA = 0.05                  # 표준 TOST: 각 측 0.05
TOST_DELTA_PCT = 10.0              # P1·P2 등가 한계 (도출은 모듈 독스트링)
TOST_DELTA_CONF_PCT = 5.0          # C1·C2 등가 한계
MIN_N_USED = {'sparse': 14, 'crowded': 42}   # 표본 손실이 이보다 크면 UNDECIDED
# ── ★양성대조 허용오차 — **팔 종류별로 다르다** (2026-08-27 배선 스모크가 잡았다) ──
#    · ablation 팔(torch 경로)은 `B-19` G-11 과 이 배치 스모크에서 **비트 동일**이었다
#      (`SHIP` 0.22227002618213496 · `KGRID1` 0.25662708717087906, 소수 17자리 일치).
#    · ⛔**`SPEC25` 는 비트 동일이 아니다**: 같은 셀에서 오늘 0.2857572313055884 vs
#      `B-11b` 동결 0.2857572313055882 = **2 ULP**(상대 7.0e-16). numpy/BLAS 축약 순서의
#      비결정성이고 `letkf_grid.py` 는 그동안 커밋된 적이 없다.
#    ⟹ **결과 전에** 기준선 팔의 허용오차를 `1e-12` 로 정한다 — 실측 편차의 약 1,400배이고
#      이 배치가 재는 가장 작은 양(Δ_conf = 5%)의 **2e-11 배**다. ⛔결과를 보고 늘리지 않는다.
#      게이트 상세에 **실측 최대 상대 편차**를 적는다(늘어나면 RESULT 에 드러난다).
POSCTRL_RTOL_ABL = 0.0             # 0 = 비트 동일 요구 (torch 경로)
POSCTRL_RTOL_SPEC = 1e-12          # 기준선 팔 (numpy/BLAS 축약 순서)
ZERO_RTOL = 1e-5


def sign_test_one_sided(w, n):
    if n <= 0:
        return None
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def k_min(n, alpha):
    """단측 부호검정에서 p ≤ alpha 가 되는 최소 승수."""
    if n <= 0:
        return None
    return next((w for w in range(n + 1) if sign_test_one_sided(w, n) <= alpha), None)


def thresholds(n):
    """(NOMINAL, CONFIRM) — NOMINAL = α=0.05, CONFIRM = 가족단위 α."""
    return k_min(n, ALPHA), k_min(n, FAMILY_ALPHA)


def sign_tost(diffs_pct, delta, alpha=TOST_ALPHA):
    """분포무관 부호-TOST. `diffs_pct` = 짝 상대차(%) 리스트.

    두 단측 검정을 **같은 부호검정**으로 돌린다:
      H01: median(d) ≥ +Δ  를 `#{d < +Δ}` 가 문턱을 넘으면 기각
      H02: median(d) ≤ −Δ  를 `#{d > −Δ}` 가 문턱을 넘으면 기각
    둘 다 기각되면 등가(= 중앙값이 (−Δ, +Δ) 안에 있다)를 α 에서 선언한다.
    ⛔새 통계량을 만들지 않는다 — 저장소가 이미 쓰는 부호검정 그대로다.
    """
    d = [x for x in diffs_pct if x is not None]
    n = len(d)
    if n == 0:
        return dict(n=0, delta=delta, passed=False, k=None,
                    below_hi=0, above_lo=0, p_hi=None, p_lo=None, n_inside=0)
    hi = sum(1 for x in d if x < delta)        # H01 기각 증거
    lo = sum(1 for x in d if x > -delta)       # H02 기각 증거
    k = k_min(n, alpha)
    return dict(n=n, delta=delta, k=k, below_hi=hi, above_lo=lo,
                p_hi=sign_test_one_sided(hi, n), p_lo=sign_test_one_sided(lo, n),
                n_inside=sum(1 for x in d if -delta < x < delta),
                passed=bool(k is not None and hi >= k and lo >= k))


# ── 표본 ──────────────────────────────────────────────────────────────────────
def episodes(stratum=None):
    """HOS 코퍼스에서 층별 에피소드. 조건절은 `B-19`/`B-16` 의 것을 그대로 쓴다.

    sparse  : `ncomp < 15`  그리고 seed ∈ `gates_b11b.SPARSE_SEEDS`  (16)
    crowded : `ncomp >= 15` 그리고 seed ∈ `gates_b12.ALL_SEEDS`      (48)
    """
    import glob
    import re
    out = []
    for p in sorted(glob.glob(os.path.join(HOS, '*.h5'))):
        m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(.+)\.h5', os.path.basename(p))
        if not m:
            continue
        nc, sd = int(m.group(2)), int(m.group(3))
        if nc < SPARSE_MAX_NCOMP and sd in STRATUM_SEEDS['sparse']:
            st = 'sparse'
        elif nc >= SPARSE_MAX_NCOMP and sd in STRATUM_SEEDS['crowded']:
            st = 'crowded'
        else:
            continue
        if stratum is not None and st != stratum:
            continue
        out.append(dict(path=p, stratum=st, bf=m.group(1), ncomp=nc, gen_seed=sd,
                        episode=os.path.basename(p)[len('episode_'):-3]))
    return out


# ── 관측 훅 (러너 배선. 판정 증거는 `effective_from_updater` 가 만든다) ──────
_CAPTURED = []


def install_capture():
    """`wavesplat.run.make_updater` 를 감싸 **실제로 만들어진 추정기 객체**를 붙잡는다.

    ⛔거동을 바꾸지 않는다: 진짜 `make_updater(config)` 의 반환값을 **그대로** 돌려준다
    (하위클래스·프록시·인자 변경 없음). 증명은 추론이 아니라 **G-11 비트 동일 양성대조**다.
    """
    import wavesplat.run as WR
    if getattr(WR.make_updater, '_b21_capture', False):
        return
    real = WR.make_updater

    def capturing(config):
        u = real(config)
        _CAPTURED.append(u)
        return u

    capturing._b21_capture = True
    capturing._b21_real = real
    WR.make_updater = capturing


# ── 양성대조 기준: 동결 행 ────────────────────────────────────────────────────
#   `FITSWAP` 은 **신규 팔이라 기준이 없다** — 결과 전에 그렇게 적는다(G-11 상세가 반복한다).
POSCTRL_SRC = {
    ('sparse', 'SHIP'):    (os.path.join(B19, 'results_b19_*.jsonl'), 'SHIP-NOW', 'est_seed'),
    ('sparse', 'KGRID1'):  (os.path.join(B19, 'results_b19_*.jsonl'), 'KGRID1', 'est_seed'),
    ('sparse', 'KGRID2'):  (os.path.join(B19, 'results_b19_*.jsonl'), 'KGRID2', 'est_seed'),
    ('sparse', 'SPEC25'):  (os.path.join(B11, 'results_b11b.jsonl'), 'SPEC25', 'filter_seed'),
    ('crowded', 'SHIP'):   (os.path.join(B16, 'results_b16_*.jsonl'), 'SHIP-NOW', 'est_seed'),
    ('crowded', 'KGRID1'): (os.path.join(B16, 'results_b16_*.jsonl'), 'KGRID1', 'est_seed'),
    ('crowded', 'KGRID2'): (os.path.join(B16, 'results_b16_*.jsonl'), 'KGRID2', 'est_seed'),
    ('crowded', 'SPEC25'): (os.path.join(B12, 'results_b12.jsonl'), 'SPEC25', 'filter_seed'),
}
NO_POSCTRL = ('FITSWAP',)     # 신규 팔 — 동결 기준이 존재하지 않는다


def _load_jsonl(pattern):
    import glob
    import json
    out = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'error' not in r:
                    out.append(r)
    return out


def posctrl_ref(stratum, arm):
    """{(episode, seed): avg_nRMSE} — 동결 배치의 행."""
    src = POSCTRL_SRC.get((stratum, arm))
    if src is None:
        return None
    pattern, ref_arm, seed_key = src
    return {(r['episode'], r[seed_key]): float(r['avg_nRMSE'])
            for r in _load_jsonl(pattern)
            if r.get('arm') == ref_arm and r.get(seed_key) is not None}


def posctrl_zero_ref(stratum):
    """동결 기준선 행의 `zero_nRMSE` (G-12 교차검증용)."""
    pattern, ref_arm, _ = POSCTRL_SRC[(stratum, 'SPEC25')]
    return {r['episode']: float(r['zero_nRMSE']) for r in _load_jsonl(pattern)
            if r.get('arm') == ref_arm and r.get('zero_nRMSE') is not None}


# ── 집계·비교 (★게이트와 분석기가 **같은 함수**를 부른다) ────────────────────
def aggregate(rows):
    """median-of-3 집계 + 유효 표본 선정. ⚠중복 구현 금지 부류를 만들지 않는다."""
    import statistics as st
    ok = [r for r in rows if 'error' not in r]
    cell = {}
    for r in ok:
        cell.setdefault((r['stratum'], r['arm'], r['episode']), []).append(r)
    val = {k: st.median([x['avg_nRMSE'] for x in v]) for k, v in cell.items()}
    zero, good, eps_all = {}, {}, {}
    for st_name in STRATA:
        eps = sorted({e for (s, _a, e) in val if s == st_name})
        eps_all[st_name] = eps
        g = []
        for e in eps:
            rs = [r for r in ok if r['stratum'] == st_name and r['episode'] == e]
            abl = [r for r in rs if r['arm'] in ARMS]
            route_ok = bool(abl) and all(r.get('auto_selected') == ROUTE_EXPECTED
                                         for r in abl)
            snap_ok = all(bool(r.get('k_snap_fired')) == (r['arm'] != 'SHIP') for r in abl)
            full = all((st_name, a, e) in val for a in ALL_ARMS)
            if route_ok and snap_ok and full:
                g.append(e)
        good[st_name] = g
        zs = {}
        for r in ok:
            if r['stratum'] == st_name and r['episode'] in g \
                    and r.get('zero_nRMSE') is not None:
                zs[r['episode']] = r['zero_nRMSE']
        zero[st_name] = zs
    return dict(ok=ok, cell=cell, val=val, eps_all=eps_all, good=good, zero=zero,
                n_used={s: len(good[s]) for s in STRATA})


def paired_diffs(val, eps, stratum, a, b):
    """짝 상대차 (%) — `(a − b)/b × 100`, 에피소드 순."""
    out = []
    for e in eps:
        ka, kb = (stratum, a, e), (stratum, b, e)
        if ka in val and kb in val and val[kb] != 0:
            out.append((val[ka] - val[kb]) / val[kb] * 100.0)
    return out


def cmp_pair(val, eps, stratum, a, b):
    """짝 부호검정 `a < b`. 동점은 승리로 세지 않는다(strict `<`, `B-16`·`B-19` 상속)."""
    import statistics as st
    common = [e for e in eps if (stratum, a, e) in val and (stratum, b, e) in val]
    w = sum(1 for e in common if val[(stratum, a, e)] < val[(stratum, b, e)])
    # ⚠**동점을 반대꼬리의 승리로 세지 않는다** — `n − better` 로 반대꼬리를 세면 동점이
    #   반대 방향 주장을 **부풀린다**. 두 꼬리 모두 strict 로 세어 동점은 어느 쪽도 아니게 한다
    #   (`B-16`·`B-19` 의 strict `<` 규약을 양 꼬리로 확장한 것이고, 희소 층에서 실제로
    #   비트 동일 동점이 관측된 적이 있다 — `B-19` §6.1).
    worse = sum(1 for e in common if val[(stratum, a, e)] > val[(stratum, b, e)])
    d = paired_diffs(val, common, stratum, a, b)
    n = len(common)
    return dict(n=n, better=w, worse=worse,
                n_tie=sum(1 for e in common if val[(stratum, a, e)] == val[(stratum, b, e)]),
                p_one_sided=sign_test_one_sided(w, n) if n else None,
                p_other_tail=sign_test_one_sided(worse, n) if n else None,
                rel_median_pct=st.median(d) if d else None,
                median_a=st.median([val[(stratum, a, e)] for e in common]) if common else None,
                median_b=st.median([val[(stratum, b, e)] for e in common]) if common else None,
                diffs_pct=d)


def cmp_zero(val, zero, eps, stratum, a):
    """음성 대조 `a < ZERO` (참 제로필드 기준. ⛔상수 1.0 아님)."""
    import statistics as st
    common = [e for e in eps if (stratum, a, e) in val and e in zero]
    w = sum(1 for e in common if val[(stratum, a, e)] < zero[e])
    return dict(n=len(common), better=w,
                p_one_sided=sign_test_one_sided(w, len(common)) if common else None,
                rel_median_pct=(st.median([(val[(stratum, a, e)] - zero[e]) / zero[e] * 100
                                           for e in common]) if common else None))


def attribution_share(val, eps, stratum):
    """★기술적 분해 (⛔검정 아님): 절제 효과 중 **재적합**이 차지하는 몫.

    `median_e(FITSWAP−SHIP) / median_e(KGRID1−SHIP)` — 절댓값 차의 중앙끼리의 비다.
    ⛔p 값도 라벨도 붙이지 않는다. 분모가 0 에 가까우면 `None` 을 돌려준다.
    """
    import statistics as st
    common = [e for e in eps if all((stratum, a, e) in val
                                    for a in ('SHIP', 'KGRID1', 'FITSWAP'))]
    if not common:
        return dict(n=0, share=None, num=None, den=None)
    num = st.median([val[(stratum, 'FITSWAP', e)] - val[(stratum, 'SHIP', e)]
                     for e in common])
    den = st.median([val[(stratum, 'KGRID1', e)] - val[(stratum, 'SHIP', e)]
                     for e in common])
    return dict(n=len(common), num=num, den=den,
                share=(num / den if den not in (0, None) and abs(den) > 1e-12 else None))


# ─────────────────────────────────────────────────────────────────────────────
# 게이트 (★단일출처 — 분석기는 `gates(rows)` 를 **호출만** 한다)
# ─────────────────────────────────────────────────────────────────────────────
# ⛔**공허한 통과 금지** — 결과 0행에서 참이 되는 게이트를 만들지 않는다
#    (W8 `G-4` 부류 · `gates_b18.py:223` · `B-19` §8.1). 아래 둘만이 행 없이도 판정할 수
#    있고(패키지·자료만 읽는다), 그 둘도 **자기 입력이 비면 통과하지 않는다.**
ROWLESS_OK = ('G-5d', 'G-10')


def _srt(vals):
    """`None` 이 섞인 값 집합도 정렬한다 (게이트 상세 출력용)."""
    return sorted(vals, key=lambda x: (x is None, repr(x)))


def _gname(name):
    """게이트 이름에서 강조 표식(`★`)을 벗긴 식별자.

    ⚠자가진단이 `name.startswith('G-8')` 로 대조했는데 실제 이름이 `'★G-8 …'` 이라
    **변형 10건이 조용히 미발화로 보고됐다**(2026-08-27 자가진단이 잡았다).
    ⟹ 이름 대조는 반드시 이 함수를 거친다.
    """
    return name.lstrip('★ ')


def gates(rows, agg=None):
    """러너 산출 전체(오류 행 포함)를 받아 `[(이름, 통과여부, 상세)]` 를 낸다."""
    if agg is None:
        agg = aggregate(rows)
    ok, val = agg['ok'], agg['val']
    out = []

    def g(name, cond, detail):
        out.append((name, bool(cond), str(detail)))

    abl = [r for r in ok if r['arm'] in ARMS]
    spec = [r for r in ok if r['arm'] == BASELINE_ARM]

    g('G-0 오류 0행', bool(rows) and len(ok) == len(rows),
      f'{len(rows) - len(ok)} 행 오류 / {len(rows)} 행')

    exp_cells = sum(len(ALL_ARMS) * STRATUM_N[s] for s in STRATA)
    got_eps = {s: len({e for (st, _a, e) in val if st == s}) for s in STRATA}
    g(f'G-1 표본 = 5팔 × (16+48) = {exp_cells} 셀',
      len(val) == exp_cells and all(got_eps[s] == STRATUM_N[s] for s in STRATA),
      f'{len(val)}/{exp_cells} 셀 · 에피소드 ' +
      ' · '.join(f'{s} {got_eps[s]}/{STRATUM_N[s]}' for s in STRATA))

    bad_seed = {}
    for k, v in agg['cell'].items():
        want = sorted(FILTER_SEEDS if k[1] == BASELINE_ARM else EST_SEEDS)
        got = sorted(x['seed'] for x in v)
        if got != want:
            bad_seed[k] = got
    g('G-2 셀마다 시드 3개 전건 (ablation=est · SPEC25=filter)',
      bool(agg['cell']) and not bad_seed,
      f'{len(agg["cell"]) - len(bad_seed)}/{len(agg["cell"])} 셀 정상'
      + (f' · 이탈 예: {list(bad_seed.items())[:2]}' if bad_seed else ''))

    fired = {a: sum(1 for r in abl if r['arm'] == a and r.get('k_snap_fired'))
             for a in ARM_ORDER}
    tot = {a: sum(1 for r in abl if r['arm'] == a) for a in ARM_ORDER}
    g('G-3 개입 발화 (SHIP 0건 · 나머지 셋 전건)',
      bool(abl) and tot.get('SHIP') and fired.get('SHIP') == 0
      and all(tot.get(a) and fired.get(a) == tot.get(a)
              for a in ('KGRID1', 'KGRID2', 'FITSWAP')),
      ' · '.join(f'{a} {fired.get(a)}/{tot.get(a)}' for a in ARM_ORDER))

    dks = {a: _srt({r.get('k_snap_dk') for r in abl if r['arm'] == a}) for a in ARM_ORDER}
    g('G-4 dk 되읽기 (계수기 `init_result.k_snap.dk`)',
      bool(abl) and dks.get('SHIP') == [None] and dks.get('KGRID1') == [DK1]
      and dks.get('KGRID2') == [DK2] and dks.get('FITSWAP') == [FITSWAP_DK],
      str(dks))

    # ── G-4b ★팔마다 **다른** 개입 크기를 요구한다 ──────────────────────────
    #    `FITSWAP` 이 "양자화가 아니다" 라는 주장은 **이 게이트가 증거**다.
    def _sh(a):
        return [r.get('k_snap_shift_median') for r in abl if r['arm'] == a]
    kg_bad = [(a, x) for a in ('KGRID1', 'KGRID2') for x in _sh(a)
              if x is None or x < KGRID_SHIFT_MIN]
    fs = _sh('FITSWAP')
    fs_bad = [x for x in fs if x is None or x > FITSWAP_SHIFT_MAX]
    g('★G-4b 개입 크기: KGRID1/2 는 파수를 옮기고 (≥1e-4) FITSWAP 은 옮기지 않는다 (≤1e-9)',
      bool(fs) and not kg_bad and not fs_bad,
      f'FITSWAP |Δk| 중앙 최대 {max([x for x in fs if x is not None], default=None)} '
      f'(허용 {FITSWAP_SHIFT_MAX}) · KGRID 이탈 {len(kg_bad)} · FITSWAP 이탈 {len(fs_bad)}')

    # ── G-5 객체 파생 속성 (⛔cfg 항등식 아님 — `B-19` BLOCKER 수리 규약) ────
    have = bool(abl) and all(isinstance(r.get('effective'), dict)
                             and all(k in r['effective'] for k in DERIVED_KEYS)
                             for r in abl)
    bad_der = _srt({k for r in abl for k, v in DERIVED_EXPECTED.items()
                    if (r.get('effective') or {}).get(k) != v})
    seed_ok = bool(abl) and all((r.get('effective') or {}).get('seed') == r['seed']
                                for r in abl)
    cls_ok = bool(abl) and all((r.get('effective') or {}).get('updater_class')
                               == UPDATER_CLASS_EXPECTED for r in abl)
    cap_ok = bool(abl) and all((r.get('effective') or {}).get('n_captured') == 1
                               for r in abl)
    env_ok = bool(abl) and all(all(v is None for v in
                                   ((r.get('effective') or {}).get('env_probe') or {}).values())
                               for r in abl)
    g('G-5 ★객체 파생 속성 일치 (⛔cfg 항등식 아님) + 시드·클래스·환경',
      have and not bad_der and seed_ok and cls_ok and cap_ok and env_ok,
      f'기대 {DERIVED_EXPECTED} · 이탈 키 {bad_der} · 키존재 {have} · seed 일치 {seed_ok} '
      f'· 클래스 {cls_ok} · 포착수 1 {cap_ok} · S8/S5 환경변수 미설정 {env_ok}')

    g('G-5b 행동 지문 init_loss_n == 0 (⛔None 은 통과 아님)',
      bool(abl) and all(r.get('init_loss_n') is not None and r['init_loss_n'] == 0
                        for r in abl),
      f"None {sum(1 for r in abl if r.get('init_loss_n') is None)}/{len(abl)} · "
      f'값 {_srt({r.get("init_loss_n") for r in abl})}')

    bad_cnt = _srt({k for r in abl for k, v in COUNTER_EXPECTED.items()
                    if (bool(r.get(k)) if isinstance(v, bool) else r.get(k)) != v})
    vp_none = [r for r in abl if r.get('varpro_obj_before') is None
               or r.get('varpro_obj_after') is None]
    g('G-5c 계수기 되읽기 + ★VARPRO 실집행 증거 (⛔echo/truthy 아님)',
      bool(abl) and not bad_cnt and not vp_none,
      f'기대 {COUNTER_EXPECTED} · 이탈 키 {bad_cnt} · '
      f"steps {_srt({r.get('varpro_steps') for r in abl})} · "
      f'obj 미기록(=건너뜀) {len(vp_none)}/{len(abl)}')

    try:
        dep = deploy_constants()
        g('G-5d 배포 상수 트립와이어 (패키지 단일출처 == 이 배치의 핀)',
          dep == DEPLOY_EXPECTED, f'{dep} vs {DEPLOY_EXPECTED}')
    except Exception as exc:                                 # noqa: BLE001
        g('G-5d 배포 상수 트립와이어 (패키지 단일출처 == 이 배치의 핀)', False,
          f'{type(exc).__name__}: {exc}')

    routes = _srt({r.get('auto_selected') for r in abl})
    eff_routes = _srt({(r.get('effective') or {}).get('init_method') for r in abl})
    ens = _srt({r.get('ensemble_k') for r in abl})
    nsel = [r.get('auto_n_selected') for r in abl if r.get('auto_n_selected') is not None]
    disc = bool(nsel) and min(nsel) < AUTO_SWITCH_THRESH_PRE_UNIFY
    gates_fired = [r for r in abl if r.get('auto_real_gate') is not False]
    g('G-6 라우터 learned 전건 · ensemble_k == 1 · 파생 init_method == 경로 · 실측게이트 미발화',
      bool(abl) and routes == [ROUTE_EXPECTED] and eff_routes == [ROUTE_EXPECTED]
      and ens == [1] and not gates_fired,
      f'auto_real_gate≠False {len(gates_fired)}/{len(abl)} · auto_selected {routes} · '
      f'파생 init_method {eff_routes} · ensemble_k {ens} · '
      f"auto_n_selected 최소 {min(nsel) if nsel else None} ⟹ 임계 0 vs 배포전 "
      f"{AUTO_SWITCH_THRESH_PRE_UNIFY} "
      f"{'판별됨' if disc else '⚠구별 불가 — 이 표본에서 두 값이 같은 경로를 낸다'}")

    g('G-7 원자 수 기록 (n_gaussians > 0 — ⛔0·None 통과 아님)',
      bool(abl) and all(isinstance(r.get('n_gaussians'), int) and r['n_gaussians'] > 0
                        for r in abl),
      f"None {sum(1 for r in abl if r.get('n_gaussians') is None)} · "
      f"0 {sum(1 for r in abl if r.get('n_gaussians') == 0)} / {len(abl)}")

    # ── G-8 ★★헤드라인 게이트: 다섯 팔이 **같은 프레임에서 채점됐는가** ─────
    #    `results/b1_full_comparison.json` 이 죽은 결함(WS 120–220 vs ETKF 301, 100/100
    #    에피소드 전부 불일치)을 **교차배치 결합으로는 원리적으로 확인할 수 없다.**
    win, win_none = {}, []
    for r in ok:
        if r.get('n_scored') is None or r.get('scored_t_first') is None \
                or r.get('scored_t_last') is None:
            win_none.append(r)
            continue
        win.setdefault((r['stratum'], r['episode']), set()).add(
            (r['n_scored'], round(float(r['scored_t_first']), 6),
             round(float(r['scored_t_last']), 6)))
    bad_win = {k: v for k, v in win.items() if len(v) != 1}
    zmis = [r for r in ok if r.get('zero_n_frames') is not None
            and r.get('n_scored') is not None and r['zero_n_frames'] != r['n_scored']]
    g('★G-8 채점 창이 **다섯 팔 전부** 에피소드 안에서 동일 + ZERO 창과도 동일',
      bool(win) and not bad_win and not win_none and not zmis,
      f'{len(win) - len(bad_win)}/{len(win)} 에피소드 일치 · 창 미기록 {len(win_none)} · '
      f'ZERO 창 불일치 {len(zmis)}'
      + (f' · 불일치 예: {list(bad_win.items())[:1]}' if bad_win else ''))

    zsets = {}
    for r in ok:
        zsets.setdefault((r['stratum'], r['episode']), set()).add(r.get('zero_nRMSE'))
    bad_z = {k: v for k, v in zsets.items() if len(v) != 1}
    none_z = sum(1 for r in ok if r.get('zero_nRMSE') is None)
    g('G-9 ZERO 에피소드 내 불변 · None 0행 (⛔상수 1.0 아님)',
      bool(zsets) and not bad_z and none_z == 0,
      f'분산 에피소드 {len(bad_z)}/{len(zsets)} · None {none_z}')

    # ── G-10 격자 재도출 (⚠에피소드가 0개면 통과하지 않는다) ────────────────
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
    n_exp = sum(STRATUM_N.values())
    g('G-10 격자 재도출 (|2π/Lx − DK1| < 1e-5 · 2π/Ly > DK1)',
      len(eps_list) == n_exp and not dk_bad and bool(devs),
      f'에피소드 {len(eps_list)}/{n_exp} · 이탈 {len(dk_bad)}'
      + (f' · ★DK1 반올림 편차 최대 {max(devs):.3e} '
         f'(허용오차의 {max(devs) / DK_REDERIVE_TOL:.4f}배)' if devs else ''))

    # ── G-11 ★★양성대조: 재실행 행이 동결 행과 **비트 동일**인가 ────────────
    pc_tot, pc_bad, pc_missing, pc_dev = 0, [], [], []
    for st_name in STRATA:
        for arm in ALL_ARMS:
            if arm in NO_POSCTRL:
                continue
            ref = posctrl_ref(st_name, arm) or {}
            mine = [r for r in ok if r['stratum'] == st_name and r['arm'] == arm]
            for r in mine:
                key = (r['episode'], r['seed'])
                if key not in ref:
                    pc_missing.append((st_name, arm) + key)
                    continue
                pc_tot += 1
                rtol = (POSCTRL_RTOL_SPEC if arm == BASELINE_ARM else POSCTRL_RTOL_ABL)
                dev = (abs(r['avg_nRMSE'] - ref[key]) / abs(ref[key])
                       if ref[key] else float('inf'))
                pc_dev.append((arm, dev))
                if dev > rtol:
                    pc_bad.append((st_name, arm, r['episode'], r['seed'],
                                   r['avg_nRMSE'], ref[key], dev))
    n_pc_exp = sum(STRATUM_N[s] * 3 for s in STRATA) * (len(ALL_ARMS) - len(NO_POSCTRL))
    g('★G-11 양성대조: 재실행 4팔이 동결 행(B-19·B-16·B-11b·B-12)과 비트 동일',
      pc_tot == n_pc_exp and not pc_bad and not pc_missing,
      f'대조 {pc_tot}/{n_pc_exp} 행 · 불일치 {len(pc_bad)} · 기준 없음 {len(pc_missing)} · '
      f'허용 ablation {POSCTRL_RTOL_ABL} / SPEC25 {POSCTRL_RTOL_SPEC} · ★실측 최대 상대편차 '
      f'ablation {max([d for a, d in pc_dev if a != BASELINE_ARM], default=None)} / '
      f'SPEC25 {max([d for a, d in pc_dev if a == BASELINE_ARM], default=None)} · '
      f'⚠`FITSWAP` 은 **신규 팔이라 동결 기준이 존재하지 않는다**(대조 0행, 결과 전 선언)'
      + (f' · 예: {pc_bad[:1]}' if pc_bad else ''))

    # ── G-12 ZERO 교차검증: 우리 독립 계산 vs 동결 기준선 행의 기록 ──────────
    z_pairs, z_bad = 0, []
    for st_name in STRATA:
        try:
            zref = posctrl_zero_ref(st_name)
        except Exception:                                    # noqa: BLE001
            zref = {}
        for (s, e), vs in zsets.items():
            if s != st_name or e not in zref:
                continue
            for v in vs:
                if v is None:
                    continue
                z_pairs += 1
                if abs(v - zref[e]) > ZERO_RTOL * abs(zref[e]):
                    z_bad.append((s, e, v, zref[e]))
    g('★G-12 ZERO 교차검증 (독립 계산 vs 동결 기준선 행)',
      z_pairs >= n_exp and not z_bad,
      f'{z_pairs - len(z_bad)}/{z_pairs} 일치 (rtol {ZERO_RTOL})'
      + (f' 예: {z_bad[:1]}' if z_bad else ''))

    # ── G-13 SPEC25 되읽기 — **객체 유래** effective_config ──────────────────
    sp_bad = []
    for r in spec:
        ec = r.get('effective_config') or {}
        for k, v in SPEC_EXPECTED.items():
            got = ec.get(k, r.get(k))
            if isinstance(v, float):
                bad = got is None or abs(float(got) - v) > 1e-9
            else:
                bad = got != v
            if bad:
                sp_bad.append((r['episode'], r['seed'], k, got, v))
                break
    g('★G-13 SPEC25 되읽기 (객체 유래: loc 25 · ens 600 · jonswap · 40×30 · LETKFGridSpec)',
      bool(spec) and not sp_bad,
      f'{len(spec) - len(sp_bad)}/{len(spec)} 행 정상 · 기대 {SPEC_EXPECTED}'
      + (f' · 예: {sp_bad[:1]}' if sp_bad else ''))

    # ── 음성 대조 (층별) — 게이트이지 주장이 아니다 ─────────────────────────
    #    ★게이트와 라벨이 **같은 함수**(`negative_ok`)를 부른다. 문턱 비교를 두 곳에 쓰면
    #    이 저장소가 반복해서 물린 *"같은 기능이 두 파일에 구현돼 한쪽만 계측"* 부류가 된다.
    for st_name in STRATA:
        n_used = agg['n_used'][st_name]
        neg = negative_ok(agg, st_name)
        nominal = thresholds(n_used)[0] if n_used else None
        for tag, key, why in (
                ('N-1', 'n1', '마진이 존재해야 매개를 잴 수 있다'),
                ('N-2', 'n2', '기준선이 무기술이면 우열 비교가 뜻을 잃는다'),
                ('N-3', 'n3', '우리 팔이 무기술이면 절제가 뜻을 잃는다')):
            res = neg['detail'][key]
            g(f'{tag}[{st_name}] 음성대조 ≥ {nominal}/{n_used} ({why})',
              neg[key], f"{res['better']}/{res['n']} p={res['p_one_sided']}")
    return out


def negative_ok(agg, stratum):
    """N-1·N-2·N-3 통과 여부 (라벨 함수가 쓴다). 게이트 목록과 **같은 함수**로 낸다."""
    val, good, zero = agg['val'], agg['good'][stratum], agg['zero'][stratum]
    n_used = agg['n_used'][stratum]
    nominal = thresholds(n_used)[0] if n_used else None
    n1 = cmp_pair(val, good, stratum, 'SHIP', BASELINE_ARM)
    n2 = cmp_zero(val, zero, good, stratum, BASELINE_ARM)
    n3 = cmp_zero(val, zero, good, stratum, 'SHIP')
    det = dict(n1=n1, n2=n2, n3=n3)
    if nominal is None:      # ⚠유효 표본 0 ⟹ **통과가 아니다**(공허한 통과 금지)
        return dict(n1=False, n2=False, n3=False, detail=det)
    return dict(n1=bool(n1['n'] and n1['better'] >= nominal),
                n2=bool(n2['n'] and n2['better'] >= nominal),
                n3=bool(n3['n'] and n3['better'] >= nominal),
                detail=det)


# ── 라벨 (결과 전 확정) ───────────────────────────────────────────────────────
def label_primary(stratum, prim, tost, neg):
    """P1/P2 — `KGRID1` vs `SPEC25`."""
    n, w = prim['n'], prim['better']
    if n < MIN_N_USED[stratum]:
        return ('SAMPLE-LOSS-UNDECIDED',
                f'유효 표본 {n} < {MIN_N_USED[stratum]} ⟹ 문턱 재계산으로 결론을 사지 않는다')
    if not neg['n2']:
        return ('BASELINE-NO-SKILL-HERE',
                '⛔기준선이 이 층에서 참 제로필드를 문턱만큼 이기지 못한다 ⟹ '
                '우열 비교가 뜻을 잃는다 (PRIMARY 무효)')
    if not neg['n3']:
        return ('ARM-NO-SKILL-HERE',
                '⛔`SHIP` 이 이 층에서 참 제로필드를 문턱만큼 이기지 못한다 ⟹ '
                '뺄 기술이 없다 (PRIMARY 무효)')
    if not neg['n1']:
        return ('NO-MARGIN-HERE',
                '⛔`SHIP < SPEC25` 가 재현되지 않는다 ⟹ **매개할 마진 자체가 없다** '
                '(PRIMARY 무효). ⚠이것은 `B-11b`/`B-12` 판정의 반박이 아니라 '
                '이 배치의 재현 실패다 — G-11 을 먼저 읽어라.')
    _nominal, confirm = thresholds(n)
    p = sign_test_one_sided(w, n)
    if w >= confirm:
        return ('MEDIATION-REFUTED',
                f'⛔{w}/{n}, p={p:.4g} — 파수를 기준선 격자에 양자화해도 `KGRID1` 이 여전히 '
                '기준선을 이긴다 ⟹ 위원장 §0 의 매개 서사는 이 표본에서 지탱되지 않는다. '
                '⛔*"기여 ①이 강화됐다"* 로 승격하지 마라(결과 전 규칙 2).')
    if prim['worse'] >= confirm:
        return ('OVERSHOOT-BELOW-BASELINE',
                f'★{w}/{n} (진 에피소드 {prim["worse"]}, 동점 {prim["n_tie"]}), '
                f'반대꼬리 p={prim["p_other_tail"]:.4g} — 절제가 우리를 기준선 '
                '**아래로** 떨어뜨렸다. ⛔이것은 매개가 **아니다**(바닥이 공통이 아니라 '
                '우리가 그 아래로 내려갔다). 결과 전 규칙 4.')
    if tost['passed']:
        return ('MEDIATION-CONFIRMED',
                f'★{w}/{n} 부호검정 널 + TOST(Δ={tost["delta"]}%) 통과 '
                f'(밴드 안 {tost["n_inside"]}/{tost["n"]}, hi {tost["below_hi"]} · '
                f'lo {tost["above_lo"]} ≥ {tost["k"]}) ⟹ 연속 파수를 기준선의 격자 간격으로 '
                '양자화하면 마진이 닫힌다. ⛔**novelty 로 읽지 마라** — 공통 이산화 바닥은 '
                'off-grid 문헌의 교과서 예측이다(Rife&Boorstyn 1974 · Tang 2013 · '
                'Candès–FG 2014). 쓸 수 있는 문장은 *"선점된 기제의 지분 계량"* 이다.')
    return ('MEDIATION-UNDECIDED',
            f'{w}/{n} 부호검정 널 + TOST(Δ={tost["delta"]}%) 실패 '
            f'(밴드 안 {tost["n_inside"]}/{tost["n"]}) ⟹ **등가를 세울 수 없다**. '
            '⛔*"차이가 없다"*·*"마진이 사라진다"* 로 쓰지 마라(결과 전 규칙 3).')


def label_dose(stratum, dose, neg):
    """S1/S2 — `KGRID2` vs `SPEC25`. 매개가 참이면 우리가 **져야** 한다."""
    n, w = dose['n'], dose['better']
    if n < MIN_N_USED[stratum] or not (neg['n2'] and neg['n3']):
        return ('DOSE-CROSS-UNDECIDED', f'표본/음성대조 미충족 (n={n})')
    _nominal, confirm = thresholds(n)
    if dose['worse'] >= confirm:
        return ('DOSE-CROSSES-BASELINE',
                f'★{w}/{n} (진 에피소드 {dose["worse"]}, 동점 {dose["n_tie"]}), '
                f'반대꼬리 p={dose["p_other_tail"]:.4g} — 간격을 2배로 하면 '
                '기준선 **아래로** 내려간다 ⟹ 공통 바닥 그림의 방향성 예측이 맞았다.')
    if w >= confirm:
        return ('DOSE-DOES-NOT-CROSS',
                f'{w}/{n}, p={sign_test_one_sided(w, n):.4g} — 2Δk 로 거칠게 해도 '
                '기준선을 이긴다 ⟹ 기준선의 바닥은 Δk 가 아니다.')
    return ('DOSE-CROSS-UNDECIDED', f'{w}/{n} — 문턱 사이.')


def label_confound(stratum, conf, tost_c, share):
    """C1/C2 — `FITSWAP` vs `SHIP`. ★교락을 추론이 아니라 측정으로 만든다."""
    n = conf['n']
    if n < MIN_N_USED[stratum]:
        return ('CONFOUND-UNDECIDED', f'유효 표본 {n} < {MIN_N_USED[stratum]}')
    if tost_c['passed']:
        return ('CONFOUND-NEGLIGIBLE',
                f'★재적합 단독 효과가 ±{tost_c["delta"]}% 안에 있다 '
                f'(밴드 안 {tost_c["n_inside"]}/{n}, 상대 중앙 {conf["rel_median_pct"]:.3f}%, '
                f'절제 효과 중 몫 {share["share"]}) ⟹ `KGRID1` 의 효과는 **양자화에 귀속 '
                '가능**하다. ⛔단 분해능 이하 섭동(0.0104 vs 0.042–0.074)의 수수께끼는 남는다.')
    _nominal, confirm = thresholds(n)
    w = conf['better']
    sig = (w >= confirm) or (conf['worse'] >= confirm)
    big = conf['rel_median_pct'] is not None and abs(conf['rel_median_pct']) >= tost_c['delta']
    if sig and big:
        return ('CONFOUND-MATERIAL',
                f'⛔재적합 단독으로 상대 중앙 {conf["rel_median_pct"]:.3f}% '
                f'({w}/{n}) ⟹ ★**`B-16`·`B-19` 의 크기는 그만큼 오염돼 있다.** '
                '⚠**존재는 오염되지 않는다** — 용량-반응은 두 팔이 같은 `ls_fit_wphi` 를 '
                '통과하므로 이 교락에서 자유롭다. ⟹ 논문은 크기 인용을 내리고 존재만 유지한다.')
    return ('CONFOUND-UNDECIDED',
            f'{w}/{n}, 상대 중앙 {conf["rel_median_pct"]}% — 등가도 유의차도 세우지 못했다.')


# ─────────────────────────────────────────────────────────────────────────────
# ★런 전 게이트 자가진단 — 게이트를 켜 보지 않고 배치를 돌리지 않는다
# ─────────────────────────────────────────────────────────────────────────────
def _fixture_rows():
    """가짜 행 320개. `SHIP`/`KGRID1`/`KGRID2`/`SPEC25` 는 **동결 값 그대로**,
    `FITSWAP` 만 합성한다(신규 팔이라 기준이 없다 — G-11 이 그것을 선언한다).

    ⚠**`FITSWAP` 값은 `SHIP × 1.001` 이라는 임의 배수**다 ⟹ C1/C2 와 A 는 이 픽스처에서
    **아무 정보도 담지 않는다.** P/S 도 마찬가지로 동결 값의 산술일 뿐이다.
    """
    rows = []
    eps = episodes()
    if not eps:
        return rows
    refs = {(s, a): (posctrl_ref(s, a) or {}) for s in STRATA for a in ALL_ARMS
            if a not in NO_POSCTRL}
    zrefs = {s: posctrl_zero_ref(s) for s in STRATA}
    for e in eps:
        s = e['stratum']
        for arm in ALL_ARMS:
            seeds = FILTER_SEEDS if arm == BASELINE_ARM else EST_SEEDS
            for sd in seeds:
                if arm == 'FITSWAP':
                    base = refs[(s, 'SHIP')].get((e['episode'], sd))
                    v = None if base is None else base * 1.001
                else:
                    v = refs[(s, arm)].get((e['episode'], sd))
                if v is None:
                    return []          # 기준이 비면 픽스처를 만들지 않는다
                rows.append(_fixture_row(e, arm, sd, v, zrefs[s].get(e['episode'])))
    return rows


def _fixture_row(e, arm, seed, val, zero):
    dk = ARMS[arm]['k_snap_dk'] if arm in ARMS else None
    row = dict(arm=arm, stratum=e['stratum'], episode=e['episode'], bf=e['bf'],
               ncomp=e['ncomp'], gen_seed=e['gen_seed'],
               seed_kind='filter' if arm == BASELINE_ARM else 'est', seed=seed,
               avg_nRMSE=val, zero_nRMSE=zero, zero_n_frames=160,
               n_scored=160, scored_t_first=14.1, scored_t_last=30.0,
               Hs_true=1.2, Tp_true=7.0, wall_s=1.0)
    if arm == BASELINE_ARM:
        row['effective_config'] = dict(SPEC_EXPECTED, filter_class='LETKFGridSpec')
        row.update(loc_radius=25.0, n_ensemble=600, background='jonswap',
                   filter_class='LETKFGridSpec')
        return row
    shift = (None if dk is None else
             (1.7e-13 if arm == 'FITSWAP' else 6.0e-3))
    row.update(k_snap_dk=dk, k_snap_fired=dk is not None, k_snap_cells=1,
               k_snap_shift_median=shift,
               effective={**{k: DERIVED_EXPECTED.get(k) for k in DERIVED_KEYS},
                          'init_method': ROUTE_EXPECTED, 'seed': seed,
                          'max_steepness': 0.3, 'depth': 50.0,
                          'updater_class': UPDATER_CLASS_EXPECTED,
                          'primary_class': 'SequentialUpdater', 'n_captured': 1,
                          'env_probe': {k: None for k in ENV_PROBE_KEYS}},
               varpro_steps=400, varpro_fired=True,
               varpro_obj_before=1.4e-3, varpro_obj_after=1.1e-3,
               init_loss_n=0, auto_selected=ROUTE_EXPECTED, auto_n_selected=4,
               auto_real_gate=False, ensemble_k=1, n_gaussians=3,
               tracking_frames=160, episode_total_frames=301)
    return row


_MUTATIONS = [
    ('G-0', lambda R: R + [dict(arm='SHIP', stratum='sparse', episode='x',
                                seed=42, error='boom')]),
    ('G-1', lambda R: R[:-3]),
    ('G-2', lambda R: [dict(r, seed=99) if r is R[0] else r for r in R]),
    ('G-3', lambda R: [dict(r, k_snap_fired=False) if r['arm'] == 'KGRID1' else r
                       for r in R]),
    ('G-4', lambda R: [dict(r, k_snap_dk=0.09) if r['arm'] == 'KGRID2' else r for r in R]),
    ('G-4b', lambda R: [dict(r, k_snap_shift_median=6e-3) if r['arm'] == 'FITSWAP' else r
                        for r in R]),
    ('G-4b', lambda R: [dict(r, k_snap_shift_median=1e-9) if r['arm'] == 'KGRID1' else r
                        for r in R]),
    ('G-5', lambda R: [dict(r, effective={**r['effective'], 'lambda_l1': 1e-3})
                       if r['arm'] in ARMS else r for r in R]),
    ('G-5', lambda R: [dict(r, effective={**r['effective'], 'seed': 999})
                       if r['arm'] in ARMS else r for r in R]),
    ('G-5', lambda R: [dict(r, effective={**r['effective'], 'n_captured': 0})
                       if r['arm'] in ARMS else r for r in R]),
    ('G-5', lambda R: [dict(r, effective={**r['effective'],
                                          'env_probe': {'S8_L1_TRACK_ONLY': '1'}})
                       if r['arm'] in ARMS else r for r in R]),
    ('G-5', lambda R: [dict(r, effective={**r['effective'],
                                          'updater_class': 'SequentialUpdater'})
                       if r['arm'] in ARMS else r for r in R]),
    ('G-5b', lambda R: [dict(r, init_loss_n=None) if r['arm'] in ARMS else r for r in R]),
    ('G-5b', lambda R: [dict(r, init_loss_n=7) if r['arm'] in ARMS else r for r in R]),
    ('G-5c', lambda R: [dict(r, varpro_steps=0) if r['arm'] in ARMS else r for r in R]),
    ('G-5c', lambda R: [dict(r, varpro_obj_before=None) if r['arm'] in ARMS else r
                        for r in R]),
    ('G-6', lambda R: [dict(r, auto_selected='v2_konly') if r['arm'] in ARMS else r
                       for r in R]),
    ('G-6', lambda R: [dict(r, effective={**r['effective'], 'init_method': 'auto'})
                       if r['arm'] in ARMS else r for r in R]),
    ('G-6', lambda R: [dict(r, auto_real_gate=None) if r['arm'] in ARMS else r for r in R]),
    ('G-7', lambda R: [dict(r, n_gaussians=0) if r['arm'] in ARMS else r for r in R]),
    ('G-8', lambda R: [dict(r, n_scored=301) if r['arm'] == BASELINE_ARM else r
                       for r in R]),
    ('G-8', lambda R: [dict(r, n_scored=None) if r['arm'] == 'SHIP' else r for r in R]),
    ('G-8', lambda R: [dict(r, zero_n_frames=301) for r in R]),
    ('G-9', lambda R: [dict(r, zero_nRMSE=None) if r is R[0] else r for r in R]),
    ('G-9', lambda R: [dict(r, zero_nRMSE=(r['zero_nRMSE'] or 0) + 0.01) if r is R[0]
                       else r for r in R]),
    ('G-11', lambda R: [dict(r, avg_nRMSE=r['avg_nRMSE'] * (1 + 1e-12))
                        if r['arm'] == 'KGRID1' else r for r in R]),
    ('G-11', lambda R: [dict(r, avg_nRMSE=r['avg_nRMSE'] * 1.05)
                        if r['arm'] == BASELINE_ARM else r for r in R]),
    ('G-11', lambda R: [dict(r, avg_nRMSE=r['avg_nRMSE'] * (1 + 1e-11))
                        if r['arm'] == BASELINE_ARM else r for r in R]),
    ('G-12', lambda R: [dict(r, zero_nRMSE=r['zero_nRMSE'] * 1.01) for r in R]),
    ('G-13', lambda R: [dict(r, effective_config={**r['effective_config'], 'nx': 80})
                        if r['arm'] == BASELINE_ARM else r for r in R]),
    ('G-13', lambda R: [dict(r, effective_config={**r['effective_config'],
                                                  'loc_radius': 50.0})
                        if r['arm'] == BASELINE_ARM else r for r in R]),
]


def selftest(verbose=True):
    """추정기를 부르지 않고(런 0회 · 파일 0개) 게이트가 실제로 발화하는지 확인한다."""
    def _p(*a):
        if verbose:
            print(*a, flush=True)
    okall = True

    rows = _fixture_rows()
    if not rows:
        _p('⛔픽스처를 만들 수 없다 (동결 기준 행 부재) ⟹ 자가진단 실패')
        return False, {}
    res = gates(rows)
    npass = sum(1 for _n, c, _d in res if c)
    _p(f'[1/4] 픽스처 {len(rows)}행 전 게이트: {npass}/{len(res)} PASS')
    for n, c, d in res:
        if not c:
            _p(f'      ⛔{n} — {d[:160]}')
    okall &= (npass == len(res))

    empty = gates([])
    leak = [n for n, c, _d in empty if c and not _gname(n).startswith(ROWLESS_OK)]
    _p(f'[2/4] rows=[] 공허한 통과: {len(leak)} (기대 0) '
       f'· 행 없이 판정 가능한 게이트 {ROWLESS_OK}')
    for n in leak:
        _p(f'      ⛔공허한 통과: {n}')
    okall &= not leak

    fired, missed = 0, []
    for tag, mut in _MUTATIONS:
        try:
            mr = gates(mut([dict(r) for r in rows]))
        except Exception as exc:                             # noqa: BLE001
            missed.append((tag, f'분석기가 죽었다: {type(exc).__name__}: {exc}'))
            continue
        hit = [n for n, c, _d in mr if not c and _gname(n).startswith(tag)]
        if hit:
            fired += 1
        else:
            missed.append((tag, '해당 게이트가 FAIL 로 뒤집히지 않았다'))
    _p(f'[3/4] 변형 {len(_MUTATIONS)}건: {fired} 발화')
    for t, why in missed:
        _p(f'      ⛔{t} — {why}')
    okall &= (fired == len(_MUTATIONS))

    # [4/4] 자료·핀이 틀리면 자료 게이트가 통과하지 않는가
    import copy
    saved = dict(POSCTRL_SRC)
    try:
        POSCTRL_SRC[('sparse', 'SPEC25')] = ('/nonexistent/*.jsonl', 'SPEC25', 'filter_seed')
        bad = gates(rows)
        pc = [c for n, c, _d in bad if _gname(n).startswith('G-11')]
        okall &= not any(pc)
        _p(f'[4/4] 기준 파일 부재에서 G-11 통과? {any(pc)} (기대 False)')
    finally:
        POSCTRL_SRC.clear()
        POSCTRL_SRC.update(saved)
        del copy

    _p('★자가진단 ' + ('전부 통과' if okall else '⛔실패'))
    return okall, dict(n_fixture=len(rows), n_gates=len(res), n_pass=npass,
                       n_mutations=len(_MUTATIONS), n_fired=fired)


if __name__ == '__main__':
    ok, _s = selftest()
    sys.exit(0 if ok else 1)
