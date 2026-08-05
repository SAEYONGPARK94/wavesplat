# PREREG — S1 성분 예산 스윕 / S2 오라클 (kx,ky) 초기화
작성 2026-08-05, **런 착수 전**. 이 문서는 결과를 본 뒤 수정하지 않는다.
수정이 필요하면 아래에 `## AMENDMENT (시각)` 절을 추가하고 사유를 적는다.

---

## 0. 무엇을 가르려는가

혼잡 레짐(성분 ≥15)에서 WaveSplat 이 LETKF 에 진다. 이 열세가

* **(H_impl) 구현 결함** — 탐색이 원자를 충분히/제대로 놓지 못한다. 예산을 늘리거나 배치를
  참값으로 심으면 회복된다.
* **(H_repr) 표현·추정의 원리적 한계** — 배치를 참값으로 심어도 남는다.

둘 중 무엇인지 가른다. S1 이 예산 축, S2 가 배치 축이다.

### 사전 MEASURED 사실 (이번 실험의 근거, 재측정 없음)
* `configs/estimation.yaml` `model.n_gaussians: [3,8,15]`, `adaptive.max_n: 30`,
  `adaptive.prune_threshold: 0.10`.
* 초기화 coarse-to-fine 은 15에서 멈춘다. 30까지는 `_adaptive_prune_grow` 의 grow 만 갈 수 있고,
  그 grow 는 초기화 끝에서 **1회**만 돈다 (`sequential_update.py` initialize 말미; 추적 중 호출 0).
* 혼잡 36런 관측 `final_n` 최대 22 / 중앙 ~19 ⟹ 성분 예산이 소진된 적이 없다.

---

## 1. 공통 설계

### 1.1 에피소드 (n = 4, 고정)
`data_finemesh_2026-07-23/` 의
`episode_BF4_15comp_seed0_uniform.h5`, `episode_BF4_30comp_seed0_uniform.h5`,
`episode_BF5_15comp_seed0_uniform.h5`, `episode_BF5_30comp_seed0_uniform.h5`.
GT 성분수 N_gt = 15 / 30 / 15 / 30 (h5 `config/wave_components`, shape 확인 완료).

### 1.2 시드
`seed ∈ {42, 123, 456}`, 팔마다 3런, **median-of-3**.
시드 42 는 `tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl` 와 겹치므로 비트 동일 검증 앵커로도 쓴다.

### 1.3 기본 설정 (모든 팔 공통, WS-P1)
`init_method` **명시** (팔마다 아래 표) · `use_dispersion=True` · `use_confidence=True` ·
`adaptive_n=True` (S2-T4 제외) · `tracking_window=3` · `tracking_mode='w_phi'` ·
`prune_threshold=0.10` (S2-T2/T4 제외) · `grow_residual_threshold=0.05` ·
`TRACK_PHASE_PREDICT=False` (현행 기본값, 2026-07-26 flip).

### 1.4 총 gradient step 수를 팔 간 고정한다 (교란 차단)
`SequentialUpdater.initialize` 는 `for iteration in range(sum(max_iters_per_stage))` 로
**항상 정확히 sum(max_iters) 스텝**을 돈다 (스테이지 조기 advance 시 남은 스텝은 마지막
스테이지가 흡수). 따라서 모든 팔에서 `sum(max_iters_per_stage) = 900` 으로 **고정**한다.
현행 `[200,200,500]` 이 900 이므로 대조 팔은 무변경이다.
⟹ "성분을 더 줬다"와 "gradient step 을 더 줬다"가 섞이지 않는다.

**남는 교란(사전 공개)**: 스테이지를 추가하면 N=15 구간이 받는 스텝 수가 줄어든다
(3-스테이지에선 N=15 가 잔여 전부, 4-스테이지에선 최대 200). 총량 고정과 동시에는 제거 불가.
S1 이 음성으로 나올 경우 이 교란은 "예산을 늘려도 안 낫다"를 **약화**시키는 방향이므로,
음성 결론은 이 교란을 명시해 보고한다.

---

## 2. 지표 패널 · 열화 방향 · 집계

| 지표 | 열화 방향 | 비고 |
|---|---|---|
| `nRMSE` | **클수록 나쁨** | **PRIMARY** |
| `Variance_Ratio` | 1.0 에서 멀수록 나쁨 → 파생량 `|VR−1|` 로 판정, 원값도 기록 | WS 는 under-dispersive (혼잡 0.68–0.84) |
| `Pearson_rho` | 작을수록 나쁨 | |
| `Skill_Score` | 작을수록 나쁨 | zero-predictor 기준 |
| `Directional_R2` | 작을수록 나쁨 | |
| `Hs_error` | 클수록 나쁨 | 이미 절대값 정의 |
| `spec.detection_rate` | 작을수록 나쁨 | `batchC_ws_phasefix.py::spectral_accuracy` 그대로 이식 |
| `spec.energy_detection_rate` | 작을수록 나쁨 | |
| `spec.k_mag_rel_error_mean` / `omega_rel_error_mean` / `amp_rel_error_mean` / `direction_error_deg_mean` | 클수록 나쁨 | matched 성분 한정 |
| `final_n` (`N`) | 방향 없음 | **진단량**, 판정 근거 아님 |
| `elapsed_ms`, `wall_s` | 방향 없음 | ⚠ **기록만 하고 인용 금지** — 이 박스의 시계 결함, 재측정 예정 |

### 2.1 집계 함수 (고정)
1. 팔 × 에피소드 마다 3시드의 **중앙값**을 취해 per-episode 값을 만든다 (지표별 독립 중앙값).
2. 처치 − 대조의 **짝별 차**를 에피소드마다 계산한다.
3. 보고 통계 = **짝별 차의 중앙값** (n=4). 평균은 보고하지 않는다.
4. 시드 잡음자 = 에피소드 내 3시드의 **range(max−min)**; 그 4에피소드 중앙값을
   `seed_range_median` 으로 함께 보고한다.

### 2.2 층
* 층 정의 = `n_comp ∈ {15, 30}`, 각 층 n=2.
* **PRIMARY 는 4개 뭉친 값**. 층별 값은 **기술통계로만** 보고하고, 층 간 차이를 기제로
  승격하지 않는다 (2026-08-04 규율: 층 정의 변수 통제 전 인과 주장 금지).

---

## 3. 검정 · MDE — α=0.05 는 이 n 에서 원리적으로 불가능

* 짝 수 n=4. 양측 정확 부호검정의 **바닥 p = 2·(1/2)^4 = 0.125**.
  Wilcoxon signed-rank 도 n=4 양측 바닥 p = 0.125.
* ⟹ **효과 크기가 무한대라도 p<0.05 는 나올 수 없다.** 이 실험은 유의성 실험이 아니다.
* 보고 검정: **양측 정확 부호검정** (n=4), p 값은 {0.125, 0.625, 1.0} 중 하나만 나온다.
  그대로 적되 "유의" 라는 단어를 쓰지 않는다.
* **MDE(부호 축)** = "4/4 일관된 부호", 그 때 p=0.125. 그보다 약한 어떤 것도 부호로 구별 불가.
* **MDE(크기 축)** = 시드 잡음자보다 큰 효과. 즉 |짝별 차 중앙값| > `seed_range_median` 일 때만
  "잡음보다 큰 효과"라고 쓴다. 그렇지 않으면 NULL 로 쓰지 않고 **"검정력 미달"** 로 쓴다.

### 3.1 그래서 무엇을 결론할 수 있는가 (사전 확정)
결론은 유의성이 아니라 **LETKF 격차의 회복 비율**로 낸다.

에피소드 e 에 대해 (nRMSE, median-of-3):
```
G_e   = nRMSE_control(e) − nRMSE_LETKF(e)      # LETKF 격차 (양수면 WS 패배)
R_e   = (nRMSE_control(e) − nRMSE_treat(e)) / G_e   # 회복 비율
resid_e = (nRMSE_treat(e) − nRMSE_LETKF(e)) / G_e   # = 1 − R_e, 남은 격차 비율
```
`nRMSE_LETKF` 는 **동결 참조값** (`tmp/rerun_2026-07-25/batchA_baselines.jsonl`, seed 42,
동일 finemesh 데이터, 재실행하지 않음):

| episode | LETKF nRMSE |
|---|---|
| BF4_15comp | 0.2282 |
| BF4_30comp | 0.2347 |
| BF5_15comp | 0.2298 |
| BF5_30comp | 0.2651 |

**사전 확정 판정 규칙** (PRIMARY = nRMSE, control = 각 실험의 지정 대조):

| 판정 | 조건 |
|---|---|
| **IMPLEMENTATION-LIMITED** | median(R_e) ≥ 0.50 **그리고** R_e > 0 이 4/4 |
| **REPRESENTATION-LIMITED** | median(R_e) ≤ 0.20 **그리고** R_e < 0.50 이 4/4 |
| **INDETERMINATE** | 그 외 전부 |

⚠ 어느 판정도 p 값을 근거로 하지 않는다. 근거는 크기와 부호 일관성이며,
n=4 라는 사실을 판정문에 함께 적는다.

---

## 4. 실험 S1 — 성분 예산 스윕

**질문**: 스테이지 천장을 GT 성분수 이상으로 올리면 혼잡 오차가 회복되는가?

전 팔 `init_method='v2_konly'` **명시**.

| 팔 | `n_gaussians` (stages) | `loss_thresholds` | `max_iters_per_stage` | Σ | `max_n` |
|---|---|---|---|---|---|
| **S1-A0** (대조, 현행) | [3, 8, 15] | [0.1, 0.05, null] | [200, 200, 500] | 900 | 30 |
| S1-A1 | [3, 8, 15, 22] | [0.1, 0.05, 0.05, null] | [200, 200, 200, 300] | 900 | 30 |
| S1-A2 | [3, 8, 15, 30] | [0.1, 0.05, 0.05, null] | [200, 200, 200, 300] | 900 | 30 |
| S1-A3 | [3, 8, 15, 30, 40] | [0.1, 0.05, 0.05, 0.05, null] | [200, 200, 200, 200, 100] | 900 | **40** |

* `max_n = max(30, max(stages))` — A3 만 40. (A0–A2 는 프로덕션 값 30 그대로.)
  이유: `_adaptive_prune_grow` 는 `n >= max_n` 이면 grow 를 건너뛴다. A3 에서 40 을 두면
  가지치기가 40 밑으로 깎았을 때만 grow 가 산다.
* 대조 = **S1-A0**. 회복 비율 R_e 의 control 도 S1-A0.
* 런 수: 4팔 × 4에피소드 × 3시드 = **48**.

**사전 예측 (ARGUED, 판정 근거 아님)**: 관측된 `final_n` 이 한 번도 상한에 닿은 적 없으므로
S1 은 NULL 에 가까울 것이다. 그래도 돌리는 이유는 "닿은 적 없다"가 "천장이 무관하다"를
함의하지 않기 때문이다 — 스테이지 사다리는 grow 와 달리 **초기화 탐색의 후보 수**를 바꾼다.

---

## 5. 실험 S2 — 오라클 (kx, ky) 초기화 ★

**질문**: 참 파수를 심고 진폭·위상만 추정해도 혼잡 오차가 남는가?

### 5.1 오라클 경로 정의 (연구 전용, 구현 완료)
`wavesplat/optimizer/initializer.py::initialize_oracle` + `init_method='oracle'` 분기.
GT 는 에피소드 h5 `config/wave_components` (열 `[A, omega, kx, ky, phi, Q]`)에서
**kx, ky 두 열만** 읽는다. `w`, `phi` 는 `v2_konly` 와 **글자 그대로 동일**하게
`w = 0.01` 균일, `phi ~ U(0, 2π)` (동일 torch 전역 RNG). ⟹ 단일요인 = 파수 배치.

**N 과 N_gt 가 다를 때의 규칙 (사전 확정)**
* `N < N_gt` : 진폭 A 상위 N 개 (= 에너지 A² 상위 N 개).
  `wavesplat/baselines/oracle_baseline.py` 의 기존 top-N 규칙과 동일.
* `N = N_gt` : 전부, 저장 순서 그대로. ← **반드시 포함하는 팔**
* `N > N_gt` : 참 성분 전부 + 나머지는 `RandomState(seed).randn()` 랜덤 파수로 패딩
  (= `initialize_from_frames_v2` 의 패딩과 동일 법칙).

### 5.2 팔

| 팔 | `init_method` | stages | `max_iters` | Σ | `prune_threshold` | `adaptive_n` | `init_k_lr_scale` |
|---|---|---|---|---|---|---|---|
| **S2-C0** (현행 형상) | v2_konly | [3,8,15] | [200,200,500] | 900 | 0.10 | True | 1.0 |
| **S2-C1** (일치 대조) | v2_konly | [N_gt] | [900] | 900 | 0.10 | True | 1.0 |
| S2-T1 | **oracle** | [N_gt] | [900] | 900 | 0.10 | True | 1.0 |
| **S2-T1f** ★ | **oracle** | [N_gt] | [900] | 900 | 0.10 | True | **0.0 (k 동결)** |
| S2-T2 | **oracle** | [N_gt] | [900] | 900 | **0.0 (가지치기 없음)** | True | 1.0 |
| S2-T4 | **oracle** | [N_gt] | [900] | 900 | — | **False** | **0.0 (k 동결)** |

* `S2-C0` = `S1-A0` 과 동일 설정 ⟹ **런을 재사용**한다 (중복 실행 없음).
* `loss_thresholds=[null]` (단일 스테이지), `max_iters_per_stage=[900]` 로 총 스텝 900 고정.
* `init_k_lr_scale` 은 이번에 추가한 연구 전용 노브: 초기화 단계의 (kx,ky) 학습률 배수.
  기본 1.0 = 기존 동작 비트 동일. 0.0 = 파수 동결 ⟹ **"진폭·위상만 추정"의 문자 그대로의 팔**.
  추적 단계는 이미 `tracking_mode='w_phi'` 로 kx/ky 동결이므로, 0.0 은 그 동결을 초기화까지
  뒤로 확장한 것이다.
* **PRIMARY 처치 = S2-T1f**, **PRIMARY 대조 = S2-C1**.
  (T1f vs C1 은 스케줄·스텝수·w/phi 처리가 모두 같고 파수 배치만 다르다... 단
  T1f 는 파수 동결이 추가된다. 그래서 **T1 vs C1 = 순수 단일요인**, T1f vs T1 = 동결의 효과.
  세 값 모두 보고한다.)
* 회복 비율 R_e 는 **control = S2-C0 (현행 형상)** 기준으로도 계산해 "출하 구성 대비 LETKF
  격차의 몇 %가 남는가"를 낸다. 이게 caller 가 요구한 숫자다.
* 런 수: 5팔(C0 재사용 제외) × 4에피소드 × 3시드 = **60**.

### 5.3 S2 가 축을 가르는 방식 (사전 확정 해석)
* R_e(T1f vs C0) 가 크고 4/4 → **H_impl**. 남은 일은 탐색·배치를 고치는 것.
* R_e 가 작고 4/4 → **H_repr**. 원자 배치를 완벽히 줘도 남는 오차 = 표현/추정의 한계.
  이 경우 남은 오차의 절대 크기와 LETKF 대비 위치를 헤드라인 숫자로 낸다.
* T2 (가지치기 없음) − T1 의 차이는 "배치가 완벽할 때 가지치기가 여전히 해로운가"에만 답한다.
  이건 부수 질문이며 주 판정에 넣지 않는다.

---

## 6. 비트 동일 검증 (런 전 필수 게이트)

`S1-A0 / seed 42` 4에피소드가 `tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl` 의 동일
(에피소드, 시드=42) 레코드와 **nRMSE·N·detection_rate 전부 일치**해야 한다.

동결 기대값:

| episode | nRMSE | N | detection_rate |
|---|---|---|---|
| BF4_15comp | 0.2720 | 10 | 0.400 |
| BF4_30comp | 0.3015 | 19 | 0.467 |
| BF5_15comp | 0.2117 | 22 | 0.467 |
| BF5_30comp | 0.2031 | 20 | 0.367 |

허용 오차: nRMSE 1e-6 (부동소수 표기 반올림 한정), N·detection_rate 는 정확 일치.
**불일치하면 멈추고 보고한다.** 이번 세션의 추가 코드가 기존 경로를 건드렸다는 뜻이다.

---

## 7. 산출물

* `tmp/s1s2_2026-08-05/PREREG.md` (이 문서)
* `tmp/s1s2_2026-08-05/run_s1s2.py` (드라이버)
* `tmp/s1s2_2026-08-05/results.jsonl` (전 런 원자료, 패널 전체 + spec + 타이밍)
* `tmp/s1s2_2026-08-05/RESULT.md` (판정)

## 8. 금지

* 프로덕션 기본값 변경 금지 (`configs/*.yaml`, `DEFAULT_CONFIG` 무변경).
* `paper/` 접근 금지 (2026-08-04 상시 지시).
* 타이밍 인용 금지.
* 사후 팔 추가로 판정 뒤집기 금지. 추가 팔은 EXPLORATORY 라벨로만.

---

## AMENDMENT 1 (2026-08-05, **런 종료 후** — 사후 기록, 판정 변경 아님)

**발견**: §3.1 의 회복 비율 `R_e = (ctrl − treat)/G_e`, `G_e = nRMSE_ctrl − nRMSE_LETKF` 는
`G_e ≤ 0` 인 에피소드에서 **정의가 무너진다**. 실측 결과 4개 중 **2개(BF5_15comp, BF5_30comp)에서
S1-A0 이 이미 LETKF 를 이겼다** (0.2157 < 0.2298, 0.2464 < 0.2651). 사전등록 당시 나는 "혼잡 4개
전부에서 WS 가 LETKF 에 진다"를 암묵 전제했고, 그 전제가 **틀렸다**.

**조치 (사후, 명시)**
1. 사전등록 판정 규칙의 **문자 그대로의 출력도 그대로 싣는다** (숨기지 않는다).
2. PRIMARY 대조인 **S2-T1f vs S2-C1 은 4/4 모두 `G_e > 0`** 이라 규칙이 정상 작동한다.
   ⟹ **S2 의 주 판정은 사전등록 그대로 유효**하다.
3. `vs S1-A0` (2차 프레이밍) 에는 조건 잘 잡힌 **pooled 형태**를 병기한다:
   `R_pool = (mean₄ ctrl − mean₄ treat) / (mean₄ ctrl − mean₄ LETKF)`.
   이건 **사후 정의**이므로 EXPLORATORY 라벨로만 쓴다.
4. S1 의 `vs S1-A0` 회복비 판정은 2/4 에피소드에서 무의미하므로 **폐기**하고,
   S1 은 §2.1 의 짝별 차 + §3 의 크기 MDE(시드 range 대비)로만 판정한다.

## AMENDMENT 2 (2026-08-05, 런 종료 후) — EXPLORATORY 팔 2개 추가

사전등록 팔이 전부 끝난 뒤, S2-T1f 의 `init_nrmse` 중앙 **0.0629** 가 추적 평균 **0.1671** 과
크게 어긋나는 것을 보고 추가. `tracking_lr = 0.0` (Adam lr 0 ⟹ 추적이 무연산, 분산관계로 전진만)
* `X-T1f-notrack` = S2-T1f + 추적 동결
* `X-A0-notrack` = S1-A0 + 추적 동결
**EXPLORATORY 전용.** S1/S2 판정에 넣지 않는다. `tracking_steps=0` 은 `update()` 에서
`loss` 미정의로 크래시하므로 쓸 수 없어 `tracking_lr=0.0` 을 쓴다.
