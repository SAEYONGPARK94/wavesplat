# 기여-증거 감사 (CONTRIB_AUDIT) — 2026-08-12

**감사관**: 기여-증거 감사관 (서브에이전트). **런 0회** — 전부 기존 원자료 재도출.
**열람 범위 = 주장 범위**: 아래에 명시한 파일만 열었다. 열지 않은 파일에 대해서는 아무 말도 하지 않는다.

**⚠기여는 8개가 아니라 7개다.** `thesis/thesis_main.tex:485-527` 의 `\begin{enumerate}` 안
`\item` 을 셌다: **7개**(`grep -c` = 7, 개별 위치 아래 각 절 머리에 기재). 과제 지시의 "8개"는
오산이며, 지시가 나열한 7개가 원문과 일치한다.

**인용금지 준수**: `reference_citation_bans.md`(75항목) 를 먼저 열었다. 아래 어느 판정도
금지 수치를 근거로 쓰지 않았다. 저촉 항목은 각 절에 `⛔` 로 표시했다.

---

## 요약표

| # | 기여 | 판정 | 한 줄 근거 |
|---|---|---|---|
| 1 | 격자 비의존 스펙트럼 프레임워크 | **ALIVE (탐색적)** | 출하 `auto` 26/30 p=5.95e-05 (내가 재도출). 어제 자료로는 널이었다 |
| 2a | 자유도 5N→4N | **ALIVE** | 산술·코드 사실 |
| 2b | 분산제약의 정확도/안정성 기여 | **UNMEASURED** | 논문 자신이 p=0.69 · n=50 파일은 **반대 방향** |
| 2c | "프레임당 200→10 스텝" | **UNMEASURED** | 200-스텝 추적 팔이 저장소에 **존재한 적 없다** |
| 3 | 기하 신뢰 가중치 | **DEAD (정확도로서)** | n=50 서 끄는 쪽이 **48/50 더 좋다** |
| 4 | 적응형 N | **정확도 DEAD · 기제 WEAKENED** | Δ=+0.0003, 8/12, p=0.68. 논문은 이미 기여에서 제외 |
| 5 | 사전 지정 6-지표 패널 | **ALIVE** | 패널이 예고한 실패모드가 **양쪽 팔에서 실제로 관측됨** |
| 6 | 공정한 ETKF 기준선 (24 config) | **WEAKENED** | 스윕 4축 승자는 맞는데 **지배 파라미터가 스윕 밖** |
| 7 | 하이드로포일 FF 제어 | **DEAD / UNANSWERABLE** | 프리뷰 8/8 이 격자 밖·사각지대 안. E2E 는 추정 후 kx 부호를 뒤집는다 |

---

## #1 격자 비의존 스펙트럼 Gaussian 파면 추정 프레임워크 (`thesis_main.tex:485-499`)

**주장(원문)**: "파도장을 $N$개의 평면파 집합으로 직접 파라미터화하고 gradient optimization으로
희소 LiDAR 관측에 적합시킨다 … 격자 보간 없이 희소·불규칙 관측에서 직접 추정하며, 임의의 공간
위치에서 파면을 평가할 수 있다."

**증거 (구조적 주장)** — **ALIVE, 코드 사실**. `wavesplat/run.py:213-216` `_predict_at(u,x,y,t)` 가
임의 `(x,y,t)` 텐서를 모델에 직접 먹인다. 격자 보간 단계가 없다. 이건 측정이 아니라 열람으로 확정된다.

**증거 (정확도 우위) — 어제와 오늘이 다르다.**

*어제까지(2026-06-24 공개 파일)*: `results/100ep_ws_letkf.json` (커밋 2026-04-29).
`tmp/BLANKSLATE_2026-08-12/letkf_framematch.json` 이 프레임 집합을 수리하고 재도출했다 —
WS 0.2294 vs LETKF 0.2247, **50/50 p=1.000**(수리 전 51/49 p=0.9204). 층화하면 ncomp≤5 **50/10**
p=1.62e-07 승 · ncomp≥15 **0/40** p=3.64e-12 패. ⟹ 어제 자료만으로는 **전역 널 + 레짐 절벽**.

*오늘* — **내가 독립 재도출했다**. 자료: `tmp/BLANKSLATE_2026-08-12/shipping_default.json`(무경합
60런) 의 `shipping_default` 팔 30 에피소드 vs `letkf_perframe.json` 을 `t > 2·Tp` 로 프레임 정합한
LETKF. 내 재계산:

| 팔 | 중앙 nRMSE | vs 프레임정합 LETKF | p(양측 부호) |
|---|---|---|---|
| 공개 파일 WS 열 | 0.2335 | 15/30 | 1.000 |
| `v2_konly` 오늘 재런 | 0.1640 | 18/30 | 0.362 |
| **출하 기본값 `auto`** | **0.1181** | **26/30** | **5.95e-05** |

성분수 층화(내 재계산): ncomp≤5 auto **15/18** p=0.0075(공개 팔과 동일) · ncomp≥15 auto
**11/12** p=0.0063 인데 공개 팔은 **0/12** p=0.0005. **VERDICT_R1 §1 · VERDICT_R2 §R2-1 의
수치를 전건 재현했다** (26/30 · 11/12 포함). 레짐 절벽은 **방법의 성질이 아니라 6월 24일 구성의
성질**이었다.

**판정: `ALIVE`, 단 탐색적.** 반드시 병기해야 하는 제한 4건:
1. **n=30, 사전등록 없음** (`BLANKSLATE/RESULT.md` 머리말이 배치 전체를 탐색적으로 선언).
2. **표준 코퍼스** — 2026-08-07 생성기 독립성 결함이 걸린다.
3. **`auto` 라우터는 선택된 객체** — `wavesplat/optimizer/learned_init.py:41-48` 이 스스로
   "v1 was picked by the FROZEN tie-break (crowded paired 11/16) … p=0.105 … must not be written
   anywhere" 라고 적어 놨다. 1차 확증40 은 `NO_DEPLOY` 였다. winner's curse 노출.
4. **기준선이 썩었다** — LETKF 는 6/24 이후 커밋 0. 그리고 오늘 프로브가 **4/4 에피소드에서 더
   강한 LETKF 를 찾았다**(#6 참조).

**인용금지 저촉**: ⛔"WS 0.223 / ETKF 0.319" · ⛔"13.4 ms / 132.6 ms" · ⛔"~10× 빠르다" ·
⛔**"정직한 문장은 레짐 경계다"**(08-12 오늘 금지) 를 이 기여의 근거로 쓸 수 없다.
비용은 중앙값+IQR 로만: 초기화 24.0 s [18.5–68.1], 추적 7.2 ms/frame [6.3–17.4]
(`BLANKSLATE/RESULT.md` BS-1b).

---

## #2 분산관계 하드 제약 (`thesis_main.tex:500-505`)

**주장(원문)**: "자유 파라미터를 $5N \to 4N$ 으로 감소시키고 물리적 유효성을 보장한다.
이 제약은 추적 단계에서 위상 예측의 warm start를 제공하여 **프레임당 최적화 스텝을 200회에서
10회로 감소시킨다**."

### 2a. 5N→4N — `ALIVE`
ω 를 `√(g|k|tanh(|k|d))` 로 구속하면 성분당 자유 파라미터가 5개에서 4개가 된다. 산술이고 코드
사실이다 (`configs/estimation.yaml:47` `use_dispersion: true`).

### 2b. 정확도/안정성 기여 — `UNMEASURED` (그리고 유일한 대표본은 반대 방향)
저장소의 대조 팔은 `A1_noDisp` 다 (`scripts/run_ablation_bf45.py:35`,
`scripts/run_ablation_seeded.py`). **내가 두 파일을 시드-중앙값으로 재집계했다**:

| 파일 | n | Full 이 더 좋음 | mean(ablated − Full) | 부호검정 p |
|---|---|---|---|---|
| `results/ablation_bf45.json` | 50 | **13/50** | **−0.02637** (제거가 더 좋음) | 0.0009 |
| `results/ablation_seeded.json` | 6 | 2/6 | −0.03016 | 0.6875 |

논문 자신의 재측정(2026-08-03, 교정 트래커, n=6, `paper/wavesplat.tex:1574-1588`)은 반대 부호를
얻었으나 **4/6, p=0.69** 이고, 논문이 스스로 "large effect size, insufficient power — **not** that
the constraint is demonstrated to contribute" 라고 적었다. 두 파일(4월, 교정 전)의 레짐 분할은
논문에 따르면 재현되지 않는다(`wavesplat.tex:1618-1622`).

⟹ **어느 표본에서도 이 제약의 정확도 기여가 유의하게 나온 적이 없고, 가장 큰 표본(n=50)은
반대 부호다.** 물리적 보존 논거(전파 가능성·좌표 변환 가능성)는 측정과 독립적으로 유효하며
논문이 `wavesplat.tex:1623-1643` 에 이미 그렇게 써 놨다 — **그것이 정직한 형태다.**

**인용금지 저촉**: ⛔**"분산 하드제약 유의"(p=0.69) 는 08-03 자로 이미 금지 목록에 있다.**
학위논문이 이를 "물리적 유효성을 보장한다" 로만 쓰면 저촉이 아니지만, 정확도 이득으로 확장하면 저촉.

### 2c. "프레임당 200회 → 10회" — `UNMEASURED`
**200-스텝 추적 팔은 저장소에 존재한 적이 없다.** 근거:
- `200` 의 출처는 **초기화** 예산이다: `configs/estimation.yaml:58`
  `max_iters_per_stage: [200, 200, 500]`. 학위논문 자신도 `:1214`, `:3691`, `:3839` 에서 200 을
  **초기화 스텝**으로 표기한다.
- `10` 은 **추적** 예산이다: `configs/estimation.yaml:14` · `wavesplat/run.py:32`
  `tracking_steps: 10`.
- 저장소 전체에서 `tracking_steps` 에 대입된 값을 전수 grep 했다: **10 · 5 · 3 · 1 · 0** 뿐이고
  **200 은 0회**. 대조 팔로 실제 실행된 것은 `tmp/rx1_tracker_2026-08-05/CON_probe_arms.py:28`
  의 `C-steps1: tracking_steps=1` 정도다.

⟹ "200→10" 은 **서로 다른 두 단계의 예산을 나란히 적은 설계 서술**이지, 통제된 감축 측정이 아니다.
warm start 가 추적 예산을 200 에서 10 으로 줄였다고 말하려면 **분산제약 없이 추적을 200 스텝
돌린 팔**이 있어야 하는데 없다. 학위논문 `:2419` 의 "$\sim$10 스텝 vs. $\sim$200 스텝" 도 같은 결함.

---

## #3 기하 신뢰 가중치 (`thesis_main.tex:506-510`)

**주장(원문)**: "LiDAR 관측의 입사각·거리 의존 품질을 직접 모델링하여 … 이분산성을 반영한다."
(주의: 원문은 **정확도 향상을 명시적으로 주장하지 않는다** — 기여 목록에 실린 것 자체가 암묵적
주장이다.)

**대조는 존재한다**: `A2_noConf` (`run_ablation_bf45.py:38`, `run_ablation_seeded.py`),
그리고 `tmp/tracker_debate_2026-08-05/PRO_diag_shrink.py:42` 의 `D3-noconf`.

**내 재도출** (시드 중앙값 → 짝 부호검정):

| 파일 | n | Full 이 더 좋음 | mean(ablated − Full) | 부호검정 p |
|---|---|---|---|---|
| `results/ablation_bf45.json` | **50** | **2/50** | **−0.02428** | **≈3e-12** |
| `results/ablation_seeded.json` | 6 | 2/6 | +0.02057 | 0.6875 |

**신뢰 가중치를 끄면 50개 중 48개에서 더 좋다.** 논문의 교정 후 재측정(n=6)도 같은 방향이다:
"removing confidence weighting changes it by $-0.002$ --- marginally *lower* without --- and is
worse on only 1 of 6 ($p=0.22$)" (`paper/wavesplat.tex:1577-1578`), 그림 캡션은
"**The confidence ablation is null**" (`:1595`).

**판정: 정확도 기여로서는 `DEAD`.** 두 개의 독립 표본이 있고 **둘 다 제거 쪽**이며, 그중 큰 쪽은
n=50 에서 48/50 이다. 이분산성을 *모델링한다*는 서술 자체는 코드 사실로 참이지만,
**그 모델링이 무엇을 벌어 주는지에 대한 양성 증거는 저장소에 0건**이다.

⚠**한계 명시**: n=50 파일은 2026-04-10 커밋이라 교정 전 트래커·구 메시다. 그러므로
"−0.024 만큼 나쁘다"는 **크기**는 인용하지 말 것. 인용 가능한 것은 **부호와 카운트**, 그리고
"교정 후 표본에서도 뒤집히지 않았다"는 사실이다.

---

## #4 적응형 성분 수 N 제어 (`thesis_main.tex:511-515`) — **두 문서의 모순**

**주장(원문)**: "에너지 기반 가지치기와 잔차 기반 성장을 통해 단순 너울($N=1$)부터 복잡한
다성분 해상($N=21$)까지 해상 복잡도에 자동 적응한다."

**논문은 정확히 반대로 적었다**. `paper/wavesplat.tex:270-277`:
> "We deliberately do *not* list it as a contribution: measured under the corrected tracker with
> a median-of-3 seed protocol, its accuracy benefit is null."

**원자료로 판정한다.** `paper/wavesplat.tex:1457-1533` 의 `tab:adaptive` — 12개 사전지정
에피소드 × 3 시드(42/123/456) 중앙값, Nyquist 적정 메시, 교정 트래커, 재측정 2026-08-03:

- 평균 Δ = **+0.0003 nRMSE**, 적응 팔 우세 **8/12**, **Wilcoxon p=0.68 · 부호검정 p=0.39**.
- **시드 산포 중앙값 0.010 nRMSE = 평균 효과의 약 30배** ⟹ "못 잡았다"가 아니라
  **"잴 가치가 있는 크기의 이득이 없다"**.
- 부호가 레짐 의존: 희소 8개 전부 개선, 혼잡 4개 중 3개 악화(BF4 15comp −0.005 · BF4 30comp
  −0.011 · BF5 30comp −0.007).

**기제 주장은 절반만 산다.** 같은 표가 `N_a` 를 인쇄한다: 1성분에서 2로 가지치기, BF5 에서 20–21로
성장 ⟹ "N=1..21 범위" 는 맞다. **그러나 혼잡에서는 반대로 간다** — `N_true`=15/30 인데 `N_a`=**9–10**
으로 **내려가고** 그 대가를 정확도로 치른다(`wavesplat.tex:1517-1522`). 즉 **"복잡도에 자동
적응한다"는 주장은 복잡한 쪽에서 정확히 실패한다.**

**판정: 정확도 = `DEAD`(측정된 널) · 기제 = `WEAKENED`(희소에서만 작동).**
**모순의 판정**: 원자료가 논문 편을 든다. 학위논문 기여 #4 는 **철회 대상**이다.
살아 있는 형태는 논문이 이미 쓴 운용 형태뿐 — "희소에서 같은 정확도를 `N_a`=2 로 달성 =
프레임당 추적 상태 **7.5배 축소**" (`wavesplat.tex:1530-1533`).

**인용금지 저촉**: ⛔"버그 수정이 적응형-N 이득을 제거했다"(3중 변경 교락, 08-03 금지) —
논문도 `:1620-1622` 에서 같은 이유로 귀속을 거부한다. 구 수치 **+6.6%** 는 인용 불가.
메모리 인덱스의 "적응형-N NULL(08-03)" 과 일치.

---

## #5 사전 지정 6-지표 평가 패널 (`thesis_main.tex:516-520`)

**주장(원문)**: "nRMSE 단일 지표로는 걸러낼 수 없는 zero-predictor를 진단하는 6개 지표
(nRMSE, VR, $\rho$, SS, Dir $R^2$, $H_s$ error)를 **사전 지정**하여 공정한 비교를 수행한다."

**증거 1 — 사전지정이 시간순으로 확인된다.** `wavesplat/evaluation/metrics.py:1-20` 이 패널을
정의하고 "defined as the **pre-registered metric panel** for the upcoming full-dataset
re-evaluation. They MUST be applied uniformly to all methods … not cherry-picked" 라고 적는다.
git: 이 문구가 들어온 커밋은 **2026-04-08 `7e4665f`**, 그것을 쓰는 100 에피소드 러너는
**2026-04-29 `23548e2`**. ⟹ **패널이 런보다 3주 앞선다.** 이 저장소에서 사전지정이
타임스탬프로 확인되는 몇 안 되는 사례다.

**증거 2 — 패널이 예고한 실패모드가 실제로 잡혔다. 양쪽 팔에서.**
`results/100ep_ws_letkf.json` 100 에피소드에서 내가 재도출:

| | SS>0 | 중앙 VR | 중앙 ρ |
|---|---|---|---|
| WaveSplat(공개 팔) | 53/100 | 0.669 | 0.456 |
| LETKF | 90/100 | **0.110** | 0.365 |

**LETKF 는 nRMSE 로는 이기면서 분산의 89%를 잃는다**(VR=0.110 = 과도 평활, 사실상
zero-predictor 쪽) — 이것이 패널이 존재 이유로 내건 바로 그 실패모드다. 동시에 패널은
**WaveSplat 자신의 공개 팔이 제로필드를 못 넘는다**는 것도 잡았다(SS>0 53/100;
오늘 `BLANKSLATE/RESULT.md` BS-1a 가 54/46 p=0.484 로 독립 재도출).

**판정: `ALIVE`.** 7개 중 **자기 자신에게 불리한 결과를 산출한 유일한 기여**이고, 그것이
바로 그 가치다.

⚠정확히 쓸 것: 6개 중 **VR·ρ·SS 3개가 "reviewer-defense" 추가분**이고 nRMSE·Hs_err·Dir_R² 는
이전부터 있었다(`metrics.py:5-8`). "6개를 사전 지정했다"는 묶음 표현이다.

---

## #6 공정한 ETKF 기준선 — 24 config 스윕 (`thesis_main.tex:521-524`)

**주장(원문)**: "24개 설정 하이퍼파라미터 스윕을 통해 최적화된 ETKF를 기준선으로 제공하여,
임의의 파라미터 선택에 따른 불이익을 방지한다."

**증거 1 — 스윕은 실재하고 승자는 실제로 배포됐다(4축에 한해).**
`results/etkf_tuning_sweep.json` = 정확히 **24 config × 12 에피소드**
(`scripts/run_etkf_tuning_sweep.py:47-52`: grid 2 × ensemble 2 × mult_inf 3 × add_inf 2).
내가 `_mean_nRMSE` 로 정렬한 승자 = **`g40x30_ne100_mi1.0_ai0.005`** (0.35842).
헤드라인 러너 `scripts/run_100ep_letkf_and_ws_metrics.py:64-70` 은 `40,30` · `n_ensemble=100` ·
`mult_inflation=1.0` · `add_inflation=0.005` — **네 스칼라가 정확히 일치한다.** 이 부분은 참이다.

**증거 2 — 그러나 지배 파라미터가 스윕 밖이다. 세 겹으로.**

(a) **추정기 클래스가 다르다.** 스윕은 `ETKFGrid` 를 돌렸고(`run_etkf_tuning_sweep.py:27,62`),
헤드라인은 **`LETKFGrid(..., loc_radius=50.0)`** 를 돌린다(`run_100ep...:18,68-70`).
**국소화(localization)는 스윕에 없다.** 그리고 그 한 축이 스윕 전체보다 크다 — 같은 에피소드에서
스윕 승자 대 배포 LETKF:

| 에피소드 | 스윕 승자(ETKFGrid) | 배포(LETKFGrid, loc50) |
|---|---|---|
| BF2_1comp | 0.4360 | **0.2568** |
| BF3_5comp | 0.3549 | **0.2384** |
| BF4_15comp | 0.3791 | **0.2295** |
| BF5_30comp | 0.4124 | **0.2634** |

(b) **승자가 4축 중 3축에서 경계값이다** — `ne=100`(최대) · `mult_inf=1.0`(최소) ·
`add_inf=0.005`(최소). **스윕이 자기 최적점을 감싸지 못한다.**

(c) **미스윕 노브가 둘 더 있다** — `wavesplat/baselines/letkf_grid.py:48-50` 의
`max_obs=40` 과 `obs_noise=0.05`.

**증거 3 — 오늘 프로브가 4/4 에피소드에서 더 강한 기준선을 찾았다.**
`tmp/DEBATE_blankslate_2026-08-12/r2_letkf_{strength,round2,round3}.json`(에피소드당 24 팔),
프레임정합 nRMSE 기준 내 재집계:

| 에피소드 | published | 최선 팔 | 상대 |
|---|---|---|---|
| BF2_1comp | 0.2578 | `noise0.2` 0.2517 | −2.4% |
| BF3_5comp | 0.2354 | `ne600_l25_n02` 0.2234 | −5.1% |
| BF4_15comp | 0.2294 | `ne1200` 0.2153 | −6.1% |
| BF5_30comp | 0.2485 | `ne1200` 0.2332 | −6.2% |

최선 팔들은 전부 **`loc10`/`loc25`/`ne600`/`ne1200`** — **스윕 범위 밖**이다.

**판정: `WEAKENED`.** "24 config 스윕을 했고 그 승자를 썼다"는 **네 축에 한해 참**이고 재현된다.
"따라서 기준선은 공정하다/최적화됐다"는 **미지지** — 배포 기준선의 지배 파라미터가 한 번도
스윕되지 않았고, 오늘 4/4 에피소드에서 더 나은 설정이 발견됐다.
VERDICT_R2 §R2-3 의 처방("`LETKF-TUNED` 를 넷째 팔로 넣지 않으면 심사에서 죽는다")과 정합.

---

## #7 파랑 예측 기반 하이드로포일 2DOF feedforward 제어 (`thesis_main.tex:525-527`)

**주장(원문)**: "\WS{}의 파도장 추정 결과를 활용하는 downstream 제어 응용을 제시한다.
Dual-actuator feedforward 구조, IIR 저역통과 필터 설계, 추정 오차 허용 범위 및 최적 미리보기
거리 분석을 포함한다."

**증거 A — 기하: 채점 지점이 격자 밖이자 사각지대 안이다. 런 0회, 설정 파일 산술.**
- 프리뷰점 (`wavesplat/run.py:322-327`): `x_p = x_foil + U·L`,
  `x_foil ∈ {+2.0, −2.5}`, `L ∈ {0.1,0.2,0.3,0.5}`, `U = 11.3`(`hydrofoil/config.py:41`)
  ⟹ **최대 `2.0 + 11.3×0.5 = 7.65 m`**, 스텐실 다리 최외곽 `9.65 m`.
- 시뮬 격자 시작 (`configs/default.yaml:30`): `x_range: [10.0, 250.0]`.
- 사각지대 반경 (`default.yaml:6,10`): `5.0 / tan(25°) = **10.7225 m**`.

⟹ **8/8 프리뷰점과 32/32 스텐실 다리가 격자 밖이며 동시에 사각지대 안이다.** 내가 산술로 확인했다.
`thesis_main.tex:3014-3015`(및 `:3577`)은 바로 그 영역을 "**LiDAR 정밀도가 가장 높은 영역**"이라고
서술한다 — 반환이 0인 곳이다. **심사위원이 계산기 한 번으로 뒤집는다.**

**증거 B — α_wave 세 번의 널에 기하학적 이유가 생겼다.**
저장소의 유일한 제어 물리량 `α_wave`(`run.py:314-338`, `hydrofoil/controller.py:12,135`)는
S2(양 레짐 널) · S7(17/32 양 레짐 널) · S8(혼잡 13/32 · 희소 18/32,
`tmp/S8_deploy_confirm_2026-08-10/RESULT.md:73`) 에서 **세 번 널**이다. 증거 A 가 그 이유다 —
**성능 문제가 아니라 거치 기하 문제.**

**증거 C — E2E 표의 추정은 자기가 보고하는 기하에서 돌지 않았다.**
`hydrofoil/run_phase3_e2e.py:114-117`:
> `# Flip to head sea: episode waves go +x, vessel goes +x → following sea`
> `# Negate kx to make waves come FROM ahead → head sea (where FF is effective)`
> `gt_components = _flip_to_head_sea(gt_components)`
> `ws_components = _flip_to_head_sea(ws_components)`

**추정을 마친 뒤** GT 와 WaveSplat 추정 **양쪽의 kx 부호를 뒤집는다.** 에피소드는
`direction_deg=0`(파가 센서에서 **멀어진다**)이고, 저장소가 스스로 `scripts/d3_h6_v2_e2e.py:15`
에 "**AWAY from the sensor --- anti-causal**"라고 적어 놨다. ⟹ `thesis_main.tex:3053-3070` 의
E2E 표(WS/GT 96%/92%/37%)는 **following sea 에서 얻은 추정치를 사후 부호반전해 head-sea
제어기에 먹인 것**이다. 게다가 **에피소드 3개 · 단일 시드 · 널 팔 없음**.

**증거 D — 관측성 게이트는 이 코퍼스에서 `UNANSWERABLE` 이다(P2 자기감사).**
`p2_observability_gate.json`(24 ep × 61 frame = 1,464 프레임)이 15/16 프리뷰점 0 프레임,
U=11.3 에서 8/8 을 0 으로 냈지만, `wavesim/pipeline.py:160-161` 이 레이캐스트 메시를 gt_grid 와
같은 `x_range` 로 만들어 **x<10 m 에는 수면 자체가 없다** ⟹ 과잉결정.
**교락 없이 사는 부분은 해석적 부등식뿐**: FOV 만으로 8점 중 **7점**이 배제되고, 나머지 1점
(7.65 m)은 최대 파고에서 **24 중 4 에피소드**에서만 도달.

**판정: `DEAD` (검증된 응용으로서) / 추정 측 요구는 이 코퍼스에서 `UNANSWERABLE`.**
제어 코드·LP 필터 스윕·민감도 표는 **산출물로서 존재**하지만(그 부분은 살아 있다),
**WaveSplat 의 추정을 제어 이득에 연결하는 모든 문장**은 ①기하상 불가하거나 ②세 번 널이거나
③사후 부호반전 위에 서 있다. VERDICT_R2 §R2-4 의 구제 경로(`direction_deg=180`,
τ_fill 0.21–2.31 s)는 **계획이지 측정이 아니다.**

**인용금지 저촉 (심각)**: `thesis_main.tex:3033-3050` 지연 예산표가 ⛔**"Intel i7-12700H"**
(실제 AMD 7800X3D) 와 ⛔**"13.4 ms"** 를 인쇄한다 — **둘 다 금지 목록(08-05)**.
⟹ **이 표는 통째로 인용 불가.** 또한 ⛔"제어가 좋아진다"(α_wave 양 레짐 널, S8) ·
⛔E-D `0_5` 셸을 "배 근처"로 · ⛔정확도 배치(`direction_deg=0`)와 제어 배치(`180`)의 교차인용.

---

## 기여 목록을 다시 쓴다면

**빠지는 것**: #2b(분산제약의 정확도) · #2c(200→10) · #3(신뢰 가중치) · #4(적응형 N 을 이득으로) ·
#7(제어를 검증된 응용으로). #6 은 격하.
**분산 하드제약은 설계 원소로 남긴다** — 물리적 전파 가능성 논거는 측정과 독립이고
논문이 이미 그 형태로 써 놨다(`wavesplat.tex:1623-1643`). 기여 목록에서만 뺀다.

### 살아 있는 증거만으로 쓸 수 있는 기여 4개

**C1. 격자 비의존 스펙트럼 추정기가, 출하 구성에서, 튜닝된 LETKF 를 모든 성분수에서 이긴다.**
근거: `tmp/BLANKSLATE_2026-08-12/shipping_default.json` + `letkf_perframe.json`
(내 재도출: 26/30, p=5.95e-05; ncomp≥15 11/12, p=0.0063). 구조는 `wavesplat/run.py:213-216`.
**필수 라벨**: 탐색적 · n=30 · 사전등록 없음 · 라우터 in-sample · 기준선 미갱신.
`S12`(배포 구성의 첫 사전등록 LETKF 대결) 전에는 헤드라인으로 쓰지 말 것.

**C2. zero-predictor 실패모드를 잡아내는 사전 지정 6-지표 패널 — 우리 자신의 팔 포함.**
근거: `wavesplat/evaluation/metrics.py:1-20`(커밋 `7e4665f` 2026-04-08, 런보다 3주 앞섬) ·
`results/100ep_ws_letkf.json`(LETKF VR 0.110 vs WS 0.669; WS SS>0 53/100) ·
`tmp/BLANKSLATE_2026-08-12/letkf_framematch.json`. **저장소에서 사전지정이 타임스탬프로
확인되는 사례이고, 자기에게 불리한 결과를 낸 사례다.**

**C3. LiDAR 구동 하이드로포일 FF 제어의 기하학적 실현가능 경계 (음성, 부등식).**
근거: `wavesplat/run.py:322-327` (프리뷰 최대 7.65 m) · `configs/default.yaml:30`(격자 10.0 m) ·
`configs/default.yaml:6,10`(사각지대 10.72 m) · `hydrofoil/config.py:41`(U=11.3) ·
`tmp/DEBATE_blankslate_2026-08-12/p2_observability_gate.json`.
**통계적 음성은 표본이 바뀌면 뒤집히고 기하학적 음성은 안 뒤집힌다** — 실을 음성은 이쪽이다.
이 하나가 α_wave 세 번의 널을 설명한다.

**C4. 온라인 모델 차수 선택의 측정된 널과 그 잡음 바닥.**
근거: `paper/wavesplat.tex:1457-1533` (Δ=+0.0003 nRMSE, 8/12, Wilcoxon p=0.68, 시드 산포 0.010 =
효과의 30배). 살아 있는 양성 형태는 **운용 이득뿐** — 희소에서 동일 정확도를 `N_a`=2 로
달성(프레임당 추적 상태 7.5배 축소, `wavesplat.tex:1530-1533`).

### (조건부) C5 — 사전등록을 통과한 유일한 배포 변경
`lambda_l1` 1e-3 → 1e-4. 근거: `tmp/S9_lr_x_l1_2026-08-11/RESULT.md`(판정 `BOTH`), 배포 커밋
`40e22f6`, 세 곳 반영(`configs/estimation.yaml:12` · `run.py:30` ·
`optimizer/sequential_update.py:40`). **46일 30배치 중 사전등록을 통과해 배포된 유일한 노브.**
⛔단 "−33% 기대" 금지(크기 불안정, E1) · ⛔S7/S8 재실행 시 기준선이 처치값을 받는 문제 미해결 ·
`/TWO-LOCI` 미해결. **기여로 실으려면 "방법"이 아니라 "배포 규율"의 사례로 실을 것.**

---

## 감사 범위 (열람 목록)
`thesis/thesis_main.tex`(발췌: 440-560, 3005-3075, 3014, 헤더 목록) ·
`paper/wavesplat.tex`(255-300, 1457-1650) ·
`tmp/DEBATE_blankslate_2026-08-12/{VERDICT_R1.md, VERDICT_R2.md, r2_letkf_*.json}` ·
`tmp/BLANKSLATE_2026-08-12/{RESULT.md, shipping_default.json, letkf_perframe.json,
letkf_framematch.json}` ·
`results/{100ep_ws_letkf.json, etkf_tuning_sweep.json, ablation_bf45.json, ablation_seeded.json,
three_contributions.json}` ·
`scripts/{run_100ep_letkf_and_ws_metrics.py, run_etkf_tuning_sweep.py, run_ablation_bf45.py,
d3_h6_v2_e2e.py}` ·
`wavesplat/{run.py, evaluation/metrics.py, baselines/letkf_grid.py, baselines/etkf_grid.py,
optimizer/learned_init.py}` ·
`hydrofoil/{run_phase3_e2e.py, config.py}` · `configs/{estimation.yaml, default.yaml}` ·
`tmp/S8_deploy_confirm_2026-08-10/RESULT.md` ·
memory `MEMORY.md` · `reference_citation_bans.md` · `project_wp_e2e_2026-07-26.md`.

**열지 않은 것 = 주장하지 않은 것**: S2/S7 원자료(메모리·S8 RESULT 인용으로만 사용),
`hydrofoil/run_lp_filter_sensitivity.py` 산출물, 제5장 수조 계획, `results/direction*` 계열.
