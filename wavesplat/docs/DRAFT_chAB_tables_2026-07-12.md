# Ch.A / Ch.B 수치 표 초안 (논문 이식용) — 2026-07-12

> Opus 집필 에이전트 초안 (CPU 전용). **커밋 금지.** 로그 `tmp/opus_chAB.log`.
> 목적: 3-챕터 골격(`PAPER_LEARNING_CHAPTER_outline_2026-07-11.md`)의
> **Ch.A(주어진 센서의 검출 법칙·작동영역·규약 반전)** 와
> **Ch.B(무학습 배포 레시피·벤치마크 v2)** 의 본문 수치 표를 논문에 바로
> 얹을 수 있는 형태로 정리한다.
>
> **규율:**
> - 모든 수치는 표 아래 **[출처]** 줄에 `results/direction3/*.json` 경로·키·
>   재도출값을 명기. JSON에서 재도출 불가한 수치는 표에 **미확보**로 표기.
> - 문서(아크 md)와 JSON 재도출이 다르면 그 사실을 명기하고 **JSON 채택**.
> - 톤: "최초/유일/SOTA/winning" 금지. rho는 pooled correlation이며 forecast
>   skill 아님(상시지시 07-05 혼용 금지).
> - **중복 회피:** 실측 학습-vs-무학습 *집계* 및 커버리지 매트릭스는
>   `DRAFT_writing_items_2026-07-12.md` B-10에 이미 있음. 본 문서의 실측 표
>   (B-7)는 **무학습(hand) 배포 팔의 레코드별 분해**만 제시하고 학습 비교는
>   B-10을 상호참조한다.

---

# Ch.A — 검출 법칙이 작동영역을 정의한다 (규약 반전)

무대: 8 seas(seed 0–7), head-sea 180°, σ_η ≈ 1 m, 선형 GT, 동일 naive MMSE
추정기, 관측은 OBS_CAP = 5,000 서브샘플(공간 분포 보존) — 따라서 표의 rho는
**바닥값**이다. 지표: rho_fan(부채꼴 nowcast), rho_ann(환형 ROI 장 상관),
rho_slp(경사 장 상관), rho_enc(조우 트랙), r90(관측 반경 커버리지 90분위), n_raw
(에피소드당 원시 복귀 수).

## Table A1 — 고정-임계 검출 법칙 T-스윕 (센서 물리로 잰 작동영역)

**캡션 초안 (국문).** 표 A1. Falcon 링크버짓 보정 고정-임계(T)를 낮출수록 관측
복귀 수(n_raw)와 커버리지 반경(r90)이 단조 증가하며, 부채꼴·경사·조우 채널의
상관이 함께 부활한다. 매뉴얼 사양(10 % 표적 250 m에서 50 % 검출)의 등가 임계
**T_m50 = A_m·ln2 = 0.0173** 에서 r90 = 73 m, 조우 rho = 0.61로 포화 어깨에
도달한다(환형 채널은 전 구간 0.93로 이미 포화, 부채꼴은 원거리 감쇠 A∝1/r²로
0.6대에서 포화). 모든 값은 8-sea 평균이며 관측 5 k 서브샘플의 바닥값이다.

**Caption (EN).** Table A1. Lowering the spec-calibrated fixed detection
threshold T monotonically increases the number of returns (n_raw) and the
coverage radius (r90); fan/slope/encounter channels revive together. At the
manual-equivalent threshold T_m50 = A_m·ln2 = 0.0173 (50 % detection of a
10 % target at 250 m) the operating point reaches r90 = 73 m and encounter
rho = 0.61. Eight-sea means; a floor value under a 5 k observation subsample.

| T | n_raw | r90 (m) | rho_fan | rho_ann | rho_slp | rho_enc |
|---|---|---|---|---|---|---|
| 0.3 | 58,285 | 43.0 | −0.004 | 0.944 | 0.732 | 0.210 |
| 0.1 | 74,618 | 54.6 | 0.363 | 0.919 | 0.844 | 0.578 |
| 0.03 | 85,260 | 67.0 | 0.558 | 0.927 | 0.863 | 0.552 |
| **0.0173 = T_m50** | **88,088** | **72.9** | **0.613** | **0.931** | **0.867** | **0.606** |
| 0.01 | 89,946 | 75.8 | 0.619 | 0.940 | 0.885 | 0.599 |
| 0.003 | 91,917 | 79.8 | 0.662 | 0.930 | 0.890 | 0.597 |
| 0.00263 = T_m90 | 92,025 | 79.7 | 0.649 | 0.930 | 0.883 | 0.590 |
| 0.001 | 92,537 | 81.3 | 0.641 | 0.927 | 0.888 | 0.586 |

**[출처]** `results/direction3/h3_fixedT_operating_domain.json` → `agg[<T>]`
키 (`rho_fan, rho_ann, rho_slp, rho_enc, n_raw, r90`). 8-sea 평균, `n_sea`=8.
상수: `A_manual`=0.025, `obs_cap`=5000, `T_grid`=[0.3,0.1,0.03,0.0173,0.01,
0.003,0.00263,0.001]. 임계식 재도출: `wavesim/raydrop_threshold.py` L87
`T_MANUAL_50 = 0.10/(250/125)**2 * ln2 = 0.017329` (A_m = 0.025). 전 셀
재도출 일치.

---

## Table A2 — 규약 반전: 같은 예산의 top-k 50 vs MCAR vs 센서 법칙

**캡션 초안 (국문).** 표 A2. 구 벤치마크의 top-k 50점 규약(프레임당 50점, 12
프레임 = 600 복귀)은 조우 채널을 사망시킨다(rho_enc 0.10). 같은 600점 예산을
공간에 균일 살포(MCAR)하면 조우 rho가 0.86으로 회복되므로 **관측 예산 자체는
무죄이며 죽인 것은 점의 배치**다. 센서의 실제 고정-임계 법칙(T_m50)은 예산을
88 k로 열어 커버리지 r90을 20 m → 73 m로 넓히고 조우 rho를 0.10 → 0.61로
부활시킨다(단 6/8 바다; §A3 이봉성 참조).

**Caption (EN).** Table A2. The legacy top-k-50 convention (50 pts/frame ×
12 frames = 600 returns) kills the encounter channel (rho_enc 0.10). Spreading
the *same* 600-point budget uniformly (MCAR) recovers encounter rho to 0.86 —
the budget is innocent; placement did the killing. The sensor's true fixed-T
law (T_m50) opens the budget to 88 k, widens coverage r90 from 20 m to 73 m,
and revives encounter rho from 0.10 to 0.61 (6/8 seas; see §A3 bimodality).

| 작동점 | 복귀/에피소드 | r90 (m) | 장 rho (η) | 경사 rho | 조우 rho |
|---|---|---|---|---|---|
| top-k 50 (규약) | 600 | 19.7 | 0.941 | 0.516 | **0.104** |
| MCAR 600 (예산 대조) | 600 | 87.9 | 0.871 | 0.843 | **0.860** |
| **T_m50 (센서 법칙)** | **88,088** | **72.9** | 0.931\* | 0.867\* | **0.606** |

\* T_m50 행의 "장 rho(η)"는 환형 rho_ann(0.931), "경사 rho"는 rho_slp(0.867)로
매핑(A1과 동일). top-k 50 / MCAR 행은 H1 실험의 rho_eta_inf / rho_slp_inf /
rho_enc를 사용(전장 η·경사·조우). 두 실험의 ROI 정의가 완전히 동일하지 않아
채널 대응은 정성적이며, 정량 대비는 조우 rho(양 실험 공통 정의)에서만 엄밀하다.

**[출처]**
- top-k 50: `h1_mask_destination_main.json` (arm=main, dir=180, amp_scale=3.33,
  n_sea=16) → `summary.rho_eta_inf.naive`=0.9409, `summary.rho_slp_inf.naive`
  =0.5161, `summary.rho_enc.naive`=0.1044 (t 0.70). r90 = rows 평균 19.72,
  n_amp(복귀)=600.
- MCAR 600: `h1_mask_destination_mcar.json` (arm=mcar, dir=180, n_sea=4) →
  `summary.rho_eta_inf.naive`=0.8712, `rho_slp_inf.naive`=0.8427,
  `rho_enc.naive`=0.8602 (t 11.61). r90 = rows 평균 87.87.
- T_m50: `h3_fixedT_operating_domain.json` `agg["0.0173"]` (A1과 동일).
- **불일치·미확보:**
  1. **결측률(92 % → 4.8 %)**: H3 아크 문서의 결측률 컬럼은 이 JSON들에서
     재도출 불가. H1 rows 정의(n_amp 600 / n_miss ≈ 87 k)로는 top-k 50 결측률이
     **99.3 %**이며 문서의 "92 %"와 다르다 — 정의가 다른 문서-전용 수치.
     **표에는 미포함**, 커버리지는 r90·n_raw로만 보고(JSON 채택).
  2. **top-k 50의 "부채꼴 사망"**: H1 summary에 rho_fan 없음(rho_eta_inf/
     slp_inf/enc만) — 부채꼴 컬럼은 이 JSON에 미확보. G0b/F2의 정성 판정
     "사망"은 각주로만.

---

## Table A3 — T_m50 조우 채널의 이봉성 (작동영역 정직 고지)

**캡션 초안 (국문).** 표 A3. T_m50 센서 법칙에서 조우 rho는 6/8 바다가
0.82–0.99이나 2개 바다(seed 1·2)가 음수다. 같은 두 바다가 임계를 더 낮춘
T = 0.001에서도 음수(−0.38 / −0.50)로 남아 서브샘플 잡음이 아닌 **바다의
성질**임을 확인한다. 이 이봉성은 Ch.B의 배포 레시피(잡음바닥 + 층화)에서
완전히 해소되므로(§B2), 정보의 벽이 아니라 **추정기 병리**로 귀속된다
(자기수정 #57).

**Caption (EN).** Table A3. Under the T_m50 sensor law, encounter rho is
0.82–0.99 for 6/8 seas but negative for two (seeds 1, 2). Those same two
remain negative (−0.38 / −0.50) even at a much lower threshold T = 0.001,
confirming this is a property of the sea, not subsample noise. The bimodality
is fully resolved by the Ch.B deployment recipe (§B2), so it is attributed to
estimator pathology, not an information wall (self-correction #57).

| seed | rho_enc @ T_m50 | rho_enc @ T=0.001 | r90 (m) | rho_slp | rho_fan |
|---|---|---|---|---|---|
| 0 | 0.985 | 0.950 | 72.5 | 0.967 | 0.758 |
| 1 | **−0.387** | **−0.380** | 78.3 | 0.934 | 0.560 |
| 2 | **−0.191** | **−0.495** | 72.0 | 0.835 | 0.465 |
| 3 | 0.873 | 0.880 | 69.0 | 0.811 | 0.356 |
| 4 | 0.823 | 0.890 | 73.4 | 0.812 | 0.626 |
| 5 | 0.906 | 0.943 | 75.2 | 0.712 | 0.720 |
| 6 | 0.976 | 0.973 | 71.4 | 0.977 | 0.720 |
| 7 | 0.867 | 0.924 | 71.6 | 0.886 | 0.697 |
| **평균** | **0.606** | 0.586 | 72.9 | 0.867 | 0.613 |

**[출처]** `h3_fixedT_operating_domain.json` → `seas[i].rows` 중 `T`=0.0173 및
0.001 행의 `rho_enc`, `r90`, `slp[1]`, `fan[1]`. 평균은 `agg["0.0173"]`와 일치
(rho_enc 0.6064). 문서 §2의 {0.99,−0.39,−0.19,0.87,0.82,0.91,0.98,0.87}과 전
seed 재도출 일치.

---

# Ch.B — 무학습 배포 레시피가 조우예보를 달성한다

## Table B1 — 무학습 배포 레시피 정의 (파라미터 원장)

**캡션 초안 (국문).** 표 B1. 학습을 전혀 쓰지 않는 배포 레시피의 세 구성요소와
확정 파라미터. (i) 센서 물리에 충실한 고정-임계 검출, (ii) 범위-층화 2,000점
서브샘플(100 ms 계산 예산 목표), (iii) 오지정-인지 잡음바닥(EM 미사용, 평평한
prior, 잡음 분산을 센서 잡음이 아닌 모델 오차 수준으로 팽창). smult = 10은
손맞춤이며 원리적 버전(s2 자기보정)은 향후 과제(§B3 각주).

**Caption (EN).** Table B1. The three components and fixed parameters of the
no-learning deployment recipe: (i) spec-faithful fixed-threshold detection,
(ii) range-stratified 2,000-point subsample (targeting a 100 ms compute
budget), (iii) misspecification-aware noise floor (no EM, flat prior, noise
variance inflated to a model-error rather than sensor-noise level).

| 구성요소 | 파라미터 | 값 | 근거 |
|---|---|---|---|
| 고정-임계 검출 | T = T_MANUAL_50 | 0.017329 (= A_m·ln2, A_m=0.025) | 매뉴얼 250 m@10 % 50 % 검출 등가 |
| 관측 서브샘플 | 범위-층화 점 수 | 2,000 | 층화가 rand보다 조우 rho↑ (§B2) |
| 잡음바닥 | EM 사용 | 아니오 (no_em) | val-잔차 선택이 바닥을 죽임 (§B3) |
| 잡음바닥 | pexp (prior 기울기) | 1 (평평) | 고-k 질량 hik_frac 억제 |
| 잡음바닥 | smult (잡음 배율) | 10 → s2 = 0.25 m² | 대역 밖+절단 에너지 ≈ 오지정 수준 |
| 윈도우 | 12-프레임 / ~20 s | ending t=22 s | 조우-인과 버퍼 (≠ 지연) |

**[출처]** T_MANUAL_50 식·값: `wavesim/raydrop_threshold.py` L83–94
(`T_MANUAL_50 = 0.10/(250./125.)**2*np.log(2.)` = 0.017329, A_m = 0.025).
층화 2k / no_em / pexp=1 / smult=10 / s2=0.25: `h5_em_pathology.json`
`arm="floor"` 행 (`pexp`=1.0, `smult`=10.0, `s2`=0.25) 및 아크 문서
`H5_EM_PATHOLOGY_AND_RECIPE_2026-07-11.md` §1. 윈도우 정의:
`a2_dir180_eval.json` `obs_recipe` ("range-stratified ~2000 pts, 12-frame/
~20 s window ending t=22 s"). **미확보:** "100 ms 계산 예산 내"는 이 JSON들에
직접 없음 — 2k-점 solve 타이밍은 스트리밍 축(과거 40 ms/frame, 07-06)에서
유래한 설계 목표로 표기하고, `deploy_speed.json`은 다른 추정기(F1 404 ms/
RELAX 1054 ms)라 이 레시피 타이밍의 근거로 인용 금지.

---

## Table B2 — 배포 레시피 사다리 (8-sea head-sea 조우 rho)

**캡션 초안 (국문).** 표 B2. 네 단계 무학습 사다리. 센서 법칙(rand 5 k, EM)의
조우 rho 0.61(6/8 생존, 최악 −0.39)에서, (a) 범위-층화 2 k로 0.87(7/8), (b)
오지정 잡음바닥 추가로 rand 2 k 0.93(8/8), (c) 층화 + 바닥 결합으로 **0.96
(8/8, 최악 +0.90)** 에 도달한다. §A3의 이봉성(음수 2 바다)이 완전히 해소된다.

**Caption (EN).** Table B2. A four-rung no-learning ladder. From the sensor law
(rand 5 k, EM) at encounter rho 0.61 (6/8 survive, worst −0.39), stratification
raises it to 0.87 (7/8), the misspecification noise floor to 0.93 (8/8), and
their combination to **0.96 (8/8, worst +0.90)**, fully resolving the §A3
bimodality.

| 레시피 (8 seas) | 조우 rho 평균 | 최악 | ≥ 0.7 |
|---|---|---|---|
| 센서 법칙 (rand 5 k, EM) | 0.606 | −0.387 | 6/8 |
| 범위-층화 2 k (EM) | 0.873 | +0.517 | 7/8 |
| 잡음바닥 (floor, rand 2 k) | 0.934 | +0.700 | 8/8 |
| **★ 잡음바닥 + 층화 2 k** | **0.961** | **+0.900** | **8/8** |

**[출처]**
- 센서 법칙 rand 5 k EM: `h3_fixedT_operating_domain.json` `agg["0.0173"]`
  (rho_enc 0.6064) + `seas` per-sea (최악 −0.387, ≥0.7 = 6/8; §A3). [주의:
  `h4_selection_ladder.json` `agg["rand@5000"].enc`는 별도 런으로 0.607(최악
  −0.299, 5/8) — 문서·표는 T_m50 정의 런(h3)을 채택.]
- 범위-층화 2 k EM: `h4_selection_ladder.json` `agg["strat_r@2000"].enc`
  =0.8734; `seas`에서 최악 0.5173, ≥0.7 = 7/8. (문서 값 0.873/+0.52/7/8과
  정확 일치.)
- floor rand 2 k: `h5_combo_recipe.json` `arm="floor_rand2k"` → 8 seed 평균
  0.9337, 최악 0.700, 8/8.
- floor + 층화 2 k: `h5_combo_recipe.json` `arm="floor_strat2k"` → 평균
  0.9612, 최악 0.900, 8/8.
- 전 셀 재도출 일치(문서 §1 사다리와 일치, rand 5 k min은 h3 채택).

---

## Table B3 — n-독약 기제: 잔차-선택이 대용량에서 바닥을 죽인다

**캡션 초안 (국문).** 표 B3. held-out z-잔차로 (pexp, smult)를 고르는 생산
경로(em ≡ no_em; val 조기중단이 EM 갱신을 사실상 폐기)는 관측 점 수 n을 2 k →
20 k로 늘리면 조우 rho가 0.45 → 0.40로 오히려 나빠진다. n이 크면 그리드가
잡음바닥을 축소(smult 3→1)하고 prior를 평평하게(pexp→0.5) 만들어 고-k 질량
(hik_frac)을 0.29 → 0.44로 키우므로, 창-안 적합은 좋아지되 시간 외삽(조우
미래)이 폭주한다. 고정 바닥(smult=10, s2=0.25)은 n=2 k에서 0.45 → 0.88로
구조하나 n=20 k에서는 여전히 층화가 필요(0.64, 최악 −0.13). "점을 더 넣으려면
더 세게 정칙화하라." (GP 오지정/nugget-inflation 문헌 정합.)

**Caption (EN).** Table B3. The n-poison mechanism. Selecting (pexp, smult) by
held-out z-residual (em ≡ no_em, since early stopping discards EM updates)
*degrades* encounter rho from 0.45 to 0.40 as observation count n grows from
2 k to 20 k: larger n drives the grid to shrink the noise floor (smult 3→1)
and flatten the prior (pexp→0.5), raising high-k mass (hik_frac 0.29→0.44), so
in-window fit improves but temporal extrapolation blows up. A fixed floor
(smult=10) rescues at n=2 k (0.45→0.88) but at n=20 k still needs stratification.

| 팔 (4 seas) | n = 2,000 조우 rho (최악) | n = 20,000 조우 rho (최악) | hik_frac @ 20k |
|---|---|---|---|
| em (생산 경로) | 0.454 (−0.141) | 0.401 (−0.417) | 0.442 |
| no_em | 0.479 (−0.144) | 0.408 (−0.417) | 0.378 |
| **floor** (no_em, pexp=1, smult=10) | **0.882 (+0.671)** | 0.644 (−0.133) | 0.250 |

**[출처]** `h5_em_pathology.json` (list 24 = 4 seed × 2 n × 3 arm). enc 평균/
최악을 (n, arm)별 재도출: em/no_em/floor @ n=2000 = 0.4541/0.4791/0.8819
(최악 −0.141/−0.144/+0.671), @ n=20000 = 0.4014/0.4083/0.6445 (최악 −0.417/
−0.417/−0.133). hik_frac @ n=20000 = em 0.442, no_em 0.378, floor 0.250;
@ n=2000 no_em = 0.293. 문서 §0 표(+0.454/+0.479/+0.882 등)와 전 셀 일치.

---

## Table B4 — 벤치마크 v2 (55 dir0 fixedT 에피소드) BF별 채점

> ### ⚠ 2026-07-13 정정 (E-2/E-3b) — BF2 행은 바다가 아니라 시뮬레이터를 쟀다
>
> 구 캡션은 BF2 1-성분을 "**장파**·단일 성분"이라 부르며 하한으로 제시했다.
> **두 군데가 틀렸다.** (1) BF2 1comp는 장파가 아니라 **λ = 8.2 m 단파**다
> (`jonswap_components`의 n=1 빈-중심이 1.75·ωp에 놓임 → T = 0.57·Tp; E-2b).
> (2) 그 8.2 m 파를 레이캐스트 메시(dx 6.2 m)가 **파장당 1.2점 = Nyquist
> 미만**으로 렌더해, LiDAR 반환에 파 신호가 아예 없었다 (E-1/E-2).
>
> 적합한 메시(180×150, dx 1.34 m)로 같은 5개 바다를 재렌더해 **동일 레시피로
> 짝지어 재채점**한 결과 (`e3b_bf2_rescore.json`; 관측점 cap도 저장본과 같은
> 8000/frame으로 맞춤 — 메시만 움직인 개입):
>
> | BF2 1comp (5시드) | ann | fan | enc |
> |---|---|---|---|
> | 조악 메시 (구 발표값) | 0.180 | 0.184 | 0.348 |
> | 미세 메시 | 0.644 | 0.362 | 0.504 |
> | **짝지음 차 ± SE** | **+0.464 ± 0.148** (t 3.1) | +0.178 ± 0.120 (t 1.5) | +0.157 ± 0.259 (t 0.6) |
>
> **n = 5에서 확증되는 것은 근접장(ann)뿐이다.** fan·enc의 상승은 잡음 안이고
> (enc는 5시드 중 2개가 오히려 악화, 범위 −0.53~+0.93), 조악 enc의 재도출값
> (0.348)이 발표값(0.168)의 2배인 것도 이 지표가 창-표본 추출에 극도로
> 민감함을 보여준다. **논문에는 ann만 회복으로 단언하고, fan·enc는 산포와
> 함께 제시할 것.**
>
> 그럼에도 **구 BF2 행의 철회는 유효하다** — 그 수치는 Nyquist 미만 렌더러를
> 통과해 측정됐으므로 무엇을 재든 바다의 속성이 아니다. 또한 **"100 %
> 아티팩트"는 과장**이다(E-1b 표현 정정): 적합한 메시에서도 BF2는 벤치마크
> 최약체다(fan 0.36 vs BF3–BF5 0.85–0.93). 남은 약세의 원인으로 **관측 샘플링
> 대비 파장**(8.2 m 파를 15–200 m 부채꼴에서)을 제안하지만, 이는 **가설이며
> 미검정**이다 — ann(0.64) > fan(0.36)이라는 약한 정황뿐. 직접 검정(부채꼴
> 밀도 상향 또는 근거리 창)은 미착수 항목으로 등록.
>
> (주의: E-1b가 보고한 "완전 회복 0.98–1.00"은 **cap 50/frame** 미세메시
> 에피소드에서 A-7 하이브리드로 잰 값이라 이 표와 직접 비교 불가 — 추정기도
> 관측량도 다르다. 재확인 대기 항목.)
>
> 표의 **전체(55) 행도 BF2 5개를 포함하므로 재채점 대상**이다(미착수 —
> BF3–BF5는 피크가 적합 범위이나 BF3 5comp는 분산의 77 %가 미해상 대역:
> E-2 표 참조). 아래 수치는 **정정 전 원본**으로 감사 이력을 위해 보존한다.

**캡션 초안 (국문, 정정 반영).** 표 B4. 고정-T 에피소드 55개(dir0, BF2–BF5,
성분 수 1–30)에 배포 레시피를 적용한 채점. 부채꼴(fan) rho는 전체 평균
0.84(48/55 ≥ 0.7), 조우(enc) rho는 0.72(39/55 ≥ 0.7). BF3–BF5에서 부채꼴이
0.85–0.93로 포화한다. **BF2 1-성분 행(구 "하한 0.19")은 철회한다** — 그
작동점은 레이캐스트 메시가 파장(8.2 m)을 표현하지 못하는 구간이었다(파장당
1.2점, Nyquist 미만). 미세 메시에서 **근접장은 회복된다(ann 0.18 → 0.64,
t 3.1)**; 부채꼴·조우의 상승(fan 0.36, enc 0.50)은 **n = 5에서 유의하지 않아
회복을 단언하지 않는다**. 고성분(BF4 15/30comp)에서 조우 rho가 떨어지는 것은
성분 수-대-개구 한계다.

**Caption (EN, corrected).** Table B4. Scoring the deployment recipe on 55
fixed-T episodes (dir0, BF2–BF5, 1–30 components). Fan rho averages 0.84
(48/55 ≥ 0.7); encounter rho 0.72 (39/55 ≥ 0.7); fan saturates at 0.85–0.93 for
BF3–BF5. **The BF2 single-component row (formerly "the 0.19 lower bound") is
withdrawn**: at that operating point the raycast mesh could not represent the
sea's 8.2 m wavelength (1.2 points per wavelength, below Nyquist). On a fine
mesh the near field recovers (ann 0.18 → 0.64, t = 3.1); the fan and encounter
gains (0.36, 0.50) are **not significant at n = 5** and are reported with their
spread rather than claimed as recoveries.

### ★표 B4 (최종판, E-8 재채점 — 55 에피소드 전수, 3팔 짝지음)

`e8_bench55_rescore.json`. **검증 헤드라인은 희소 팔로 인용한다**(E-9 규약:
학습은 데이터 마음껏, 검증은 원래 가정한 희소 예산). 2,000점 팔은 "낙관적
방사모델의 경우"로 병기.

| 팔 | ann | fan (≥0.7) | enc (≥0.7) |
|---|---|---|---|
| 조악 메시 6.2 m, 2,000점 *(구 발표값 재현)* | 0.835 | 0.840 (48/55) | 0.734 (36/55) |
| 적합 메시 1.34 m, 2,000점 | 0.884 | 0.858 (49/55) | 0.733 (38/55) |
| **적합 메시, 600점 = 희소(원래 가정)** | **0.833** | **0.828 (49/55)** | **0.763 (41/55)** |

조악 팔이 발표값(fan 0.841 / ann 0.839 / enc 0.723)을 재현한다 — 채점기 정합 확인.

**BF별 fan (조악 → 적합 → 희소)**

| BF | n | 조악 | 적합 | 희소 | 적합-조악 이동 |
|---|---|---|---|---|---|
| BF2 | 5 | 0.184 | 0.362 | 0.355 | **+0.178 ± 0.120** |
| BF3 | 15 | 0.844 | 0.849 | 0.826 | +0.005 ± 0.004 |
| BF4 | 25 | 0.935 | 0.935 | 0.900 | +0.000 ± 0.001 |
| BF5 | 10 | 0.926 | 0.925 | 0.889 | −0.000 ± 0.001 |

**세 가지 결론 (전부 정직하게)**:
1. **메시 아티팩트는 BF2만 위협했다.** BF3–BF5(55 중 50 에피소드)는 분산의
   19–22 %가 미해상이고 연산자 오차가 잡음의 2.4배였는데도 **fan rho가 사실상
   안 움직인다**(+0.005 / +0.000 / −0.000). 즉 벤치마크 헤드라인은 한 번도
   위험하지 않았고, **위험했던 건 그 최악 행 하나뿐**이었다. (E-3의 Δ 이동
   +0.013은 두 팔의 *차이*에 대한 것이지 절대 rho가 아니다 — 모순 아님.)
2. **★희소(600점)에서 헤드라인 지표(조우 예보)가 살아남는다.** 55 에피소드
   짝지음 검정 (희소 − 2,000점):

   | 지표 | 차이 ± SE | t | 개선된 에피소드 | 판정 |
   |---|---|---|---|---|
   | ann (근접장) | −0.0507 ± 0.0095 | −5.37 | 4/55 | 유의하게 **손해** |
   | fan (부채꼴) | −0.0293 ± 0.0029 | −10.17 | 3/55 | 유의하게 **손해** |
   | **enc (조우 예보)** | **+0.0295 ± 0.0293** | **+1.01** | 23/55 | **불변(잡음 내)** |

   즉 관측 예산을 1/3로 줄이면 **장(場) 정확도는 유의하게 잃지만
   (ann −0.05, fan −0.03), 제어에 실제로 쓰이는 조우 예보는 통계적으로
   변하지 않는다.** 논문의 작동성 주장은 희소 예산에서 그대로 유효하다.
3. **⚠ 철회**: 초안에 "희소가 조우를 오히려 **개선**한다(0.733 → 0.763)"고
   썼으나 **짝지음 검정에서 잡음 안**이다(t 1.01, 개선 23/55 = 절반 미만).
   평균 상승은 일부 에피소드의 큰 개선이 끌어올린 것. **"개선"이라 쓰지 말고
   "불변"으로 인용할 것.** (기제 추정도 함께 폐기 — 검정할 효과가 없다.)

그림: `fig_bench55_2026-07-13.png`.

**(참고) 구 발표 표 — 감사 이력용 보존**

| BF | n_ep | fan rho (≥0.7) | ann rho | enc rho (≥0.7) |
|---|---|---|---|---|
| ~~BF2~~ (철회) | 5 | ~~0.194 (0/5)~~ | ~~0.223~~ | ~~0.168 (1/5)~~ |
| BF3 | 15 | 0.845 (13/15) | 0.870 | 0.770 (12/15) |
| BF4 | 25 | 0.934 (25/25) | 0.930 | 0.766 (18/25) |
| BF5 | 10 | 0.923 (10/10) | 0.875 | 0.823 (8/10) |
| **전체** | **55** | **0.841 (48/55)** | **0.839** | **0.723 (39/55)** |

보조 (성분 수 분해, 참고): BF3_2comp fan 0.97 / BF3_5comp 0.88 (enc 0.64);
BF4_1comp 0.96 / BF4_15comp enc 0.60 / BF4_30comp enc 0.55; 부채꼴은 성분 수에
둔감, 조우는 고성분에서 하락.

**[출처]** `h6_v2_e2e.json` (list 55, 각 항목 `file, n_win, ann, fan, enc, r0`;
r0=10.14 고정). BF별 평균·≥0.7 카운트 재도출: BF2(5) fan 0.194/0.223/enc 0.168,
BF3(15) 0.845/0.870/0.770, BF4(25) 0.934/0.930/0.766, BF5(10) 0.923/0.875/0.823.
전체 fan 0.8407(48/55), ann 0.8394, enc 0.7232(39/55). 파일명 전부 `_fixedT.h5`,
dir180 없음(재도출 확인). 핸드오프 §1.2 "fan 전 벤치마크 +0.84, BF2 1comp 0.19
하한"과 일치(BF2_1comp fan 0.194). **미확보:** dir180 head-sea 10-에피소드
벤치마크 채점은 이 JSON에 없음 — `a2_dir180_eval.json`은 2 에피소드(BF3)
방향-불변 대조뿐(§B7 각주). 배포 레시피의 계산 시간(창당 ms)은 h6에 미포함.

---

## Table B5 — HOS M = 3 비선형 재현 (조우 rho, 4 seeds × Hs)

**캡션 초안 (국문).** 표 B5. 배포 레시피 헤드라인이 완전 비선형(HOS M = 3)
해면에서 재현됨을 확인. head-sea, 조우 rho: Hs 1.2 m에서 평균 0.92, 가파른 Hs
3.0 m에서 평균 0.91. "주어진 센서로 조우 rho ~0.9"는 선형 가정에 의존하지
않는다. (seed 3의 0.76–0.80은 비선형이 아닌 바다-특이로, §A3·B6의 선형 쌍둥이
에서도 낮음.)

**Caption (EN).** Table B5. The recipe headline reproduces on fully nonlinear
(HOS M = 3) surfaces. Head-sea encounter rho: mean 0.92 at Hs 1.2 m, 0.91 at
steep Hs 3.0 m. "Encounter rho ~0.9 with a given sensor" does not rely on the
linear assumption.

| | seed 0 | seed 1 | seed 2 | seed 3 | 평균 |
|---|---|---|---|---|---|
| Hs 1.2 m | 0.937 | 0.954 | 0.992 | 0.801 | **0.921** |
| Hs 3.0 m (가파름) | 0.930 | 0.952 | 0.980 | 0.761 | **0.906** |

**[출처]** `h7_hos_arena.json` (list 16 = 4 seed × 2 Hs × 2 M) 중 `M`=3 행의
`enc`. Hs 1.2: 0.9366/0.9543/0.9915/0.8009 (평균 0.9208); Hs 3.0: 0.9297/
0.9516/0.9804/0.7606 (평균 0.9056). 문서 H7 §0 표(0.94/0.95/0.99/0.80→0.92;
0.93/0.95/0.98/0.76→0.91)와 전 seed 일치. **캐비엇(#58):** 같은 JSON의 `M`=1
행이 Hs 3.0에서 enc=NaN (HOS M=1은 순수 선형 아님 — steep에서 자유표면 BC
부분 비선형이 NaN 폭주 후 평평한 바다로 위장; sig=0.0). 선형 기준은 §B6의
위상회전 쌍둥이로 교체.

---

## Table B6 — HOS 학습 가능 상금 상한 (선형 쌍둥이 − M3, Hs 3.0 m)

**캡션 초안 (국문).** 표 B6. 같은 초기 위상의 선형 쌍둥이(a₀·e^{−iωt}, 부호
검증 rho 0.995)와 M3의 skill 차이 = 비선형성이 만드는 학습 가능 상금의 상한.
Hs 3.0 m에서 환형 −0.007, 부채꼴 +0.012, 조우 +0.023(t 1.3, 유의성 미달).
표고를 직접 주는 LiDAR에서는 가파른 바다도 "보이는 대로 적합"이 거의 최적 —
시뮬 비선형에는 학습이 먹을 방이 사실상 없다(Ehlers급 이득의 주원천은 비선형
파가 아닌 나쁜 관측 연산자라는 예측의 정면 확인).

**Caption (EN).** Table B6. The skill gap between M3 and a same-initial-phase
linear twin (a₀·e^{−iωt}, sign-check rho 0.995) upper-bounds the learnable
prize from nonlinearity. At Hs 3.0 m: annular −0.007, fan +0.012, encounter
+0.023 (t 1.3, not significant). For an elevation-direct LiDAR there is
essentially no room for learning in simulated nonlinearity.

| 채널 (twin − M3), Hs 3.0 m | 방 (평균) | 표준편차 |
|---|---|---|
| 환형 (ann) | −0.007 | 0.010 |
| 부채꼴 (fan) | +0.012 | 0.004 |
| 조우 (enc) | +0.023 | 0.018 (t 1.3) |

**[출처]** `h7b_linear_twin.json` (`sign`=−1.0; `rows` = 4 seed @ Hs 3.0) 의
ann/fan/enc를 `h7_hos_arena.json` M=3 Hs 3.0 행과 seed별 차감 재도출:
enc room = +0.0010/+0.0030/+0.0116/+0.0757 → 평균 +0.0228; ann room 평균
−0.0071; fan room 평균 +0.0119. 문서 H7 §0-② Hs 3.0 열(−0.007/+0.012/+0.023)과
일치. **미확보:** Hs 1.2 m 열(문서: 환형 −0.008, 부채꼴 +0.002, 조우 −0.001)은
`h7b_linear_twin.json`에 Hs 1.2 쌍둥이 행이 없어 재도출 불가 — 문서-전용,
**표에 미포함**(Hs 3.0만 JSON 재도출). 등록 예측(0.03–0.08)의 절반 이하로
G-HOS 게이트(≥0.01, t≥3) 미달이라는 판정은 유지.

---

## Table B7 — 실측 60창 무학습(hand) 배포 팔, 레코드별 (Ch.B 배포 검증)

**캡션 초안 (국문).** 표 B7. 무학습 배포 팔(hand-MMSE, 학습 0)을 5개 실측
스테레오 레코드 × 12창 = 60창에 zero-shot 적용한 조우-트랙·전장 상관. 조우
트랙 전체 평균 0.53(28/60 창 ≥ 0.7), 전장 평균 0.55. 폭풍 너울(LJ18g)이 조우
트랙 최고(0.63), 근거리 소패치(AA14)가 최저(0.37). (학습 팔과의 창-짝 비교 및
커버리지 매트릭스는 `DRAFT_writing_items` B-10 참조 — 본 표는 무학습 팔만.)

**Caption (EN).** Table B7. The no-learning deployment arm (hand-MMSE, zero
learning) applied zero-shot to 5 real stereo records × 12 windows = 60 windows.
Encounter-track rho averages 0.53 (28/60 windows ≥ 0.7); whole-field rho 0.55.
(The paired learned comparison and coverage matrix are in `DRAFT_writing_items`
B-10; this table isolates the no-learning arm.)

| 레코드 | 조우 트랙 rho (hand) | ≥ 0.7 | 전장 rho (hand) | 진폭비 (hand) |
|---|---|---|---|---|
| AA14 | 0.374 | 4/12 | 0.507 | 1.030 |
| AA15a | 0.546 | 5/12 | 0.697 | 0.886 |
| AA15b | 0.550 | 6/12 | 0.663 | 0.862 |
| YS17 | 0.562 | 6/12 | 0.494 | 0.712 |
| LJ18g (폭풍) | 0.627 | 7/12 | 0.388 | 0.942 |
| **전체 (60창)** | **0.532** | **28/60** | **0.550** | 0.887 |

**[출처]**
- 조우 트랙 hand: `gate_promotion_eval.json` (list 60, key `hand`) 레코드별
  평균 재도출 = AA14 0.3742, AA15a 0.5464, AA15b 0.5499, YS17 0.5624,
  LJ18g 0.6272; 전체 0.5320, ≥0.7 = 28/60. (전체 0.5320은 B-10 층③ hand와
  동일 팔.)
- 전장 hand + 진폭비: `field_wide_eval.json` (list 60, key `field_hand`,
  `amp_hand`) 레코드별 = AA14 0.507/1.030, AA15a 0.697/0.886, AA15b 0.663/
  0.862, YS17 0.494/0.712, LJ18g 0.388/0.942; 전체 field_hand 0.5497(=B-10
  층② hand), amp_hand 0.887.
- **중복 회피:** 학습(head/learned) 비교·이득(t·CI)·커버리지 매트릭스는
  `DRAFT_writing_items_2026-07-12.md` B-10/B-13에 이미 있음 — 여기서는 무학습
  배포 팔의 레코드별 분해만 신규 제시.

---

# 부록 — 재확인 수치 원장 (감사용)

| # | 수치 | 재도출 값 | 출처 | 상태 |
|---|---|---|---|---|
| A1 | T_m50 행 (n_raw/r90/fan/ann/slp/enc) | 88088 / 72.9 / 0.613 / 0.931 / 0.867 / 0.606 | `h3...json` agg["0.0173"] | 일치 |
| A1 | T_m50 임계값 | 0.017329 (A_m 0.025·ln2) | `raydrop_threshold.py` L87 | 일치 |
| A2 | top-k 50 (r90/η/slp/enc) | 19.7 / 0.941 / 0.516 / 0.104 | `h1_..._main.json` summary+rows | 일치 |
| A2 | MCAR 600 (r90/η/slp/enc) | 87.9 / 0.871 / 0.843 / 0.860 | `h1_..._mcar.json` | 일치 |
| A2 | 결측률 92 %/4.8 % | — | JSON 미포함 | **미확보** (문서-전용; H1 정의로는 99.3 %) |
| A2 | top-k 50 부채꼴 "사망" | — | H1 summary에 rho_fan 없음 | **미확보** (정성 각주) |
| A3 | 조우 이봉성 8-seed | 0.985/−0.387/−0.191/0.873/0.823/0.906/0.976/0.867 | `h3...json` seas rows | 일치 |
| B1 | 레시피 파라미터 (T/2k/no_em/pexp1/smult10/s2 0.25) | 0.017329 / 2000 / — / 1 / 10 / 0.25 | `raydrop_threshold.py`, `h5_em_pathology.json` floor | 일치 |
| B1 | 100 ms 계산 예산 | — | h5/h6에 미포함 | **미확보** (설계 목표; deploy_speed는 타 추정기) |
| B2 | 사다리 (rand5k/strat2k/floor2k/floor+strat) | 0.606 / 0.873 / 0.934 / 0.961 | h3 + h4 + h5_combo | 일치 |
| B2 | rand 5 k min/생존 | −0.387, 6/8 (h3) vs −0.299, 5/8 (h4) | h3 T_m50 채택 | **경미 불일치**(별도 런; JSON=h3 채택) |
| B3 | n-독약 enc (em/no_em/floor × 2k/20k) | 0.454/0.479/0.882; 0.401/0.408/0.644 | `h5_em_pathology.json` | 일치 |
| B3 | hik_frac (floor 0.25, no_em 0.29→0.44) | 0.250 / 0.293→0.442 | 동상 | 일치 |
| B4 | 벤치마크 v2 전체 fan/ann/enc | 0.841(48/55) / 0.839 / 0.723(39/55) | `h6_v2_e2e.json` | 일치 |
| B4 | BF2 1comp 하한 | fan 0.194 | 동상 | 일치 (핸드오프 "0.19") |
| B5 | HOS M3 조우 rho (Hs1.2/3.0) | 0.921 / 0.906 | `h7_hos_arena.json` M=3 | 일치 |
| B5 | M1 Hs 3.0 NaN | NaN (sig 0) | 동상 | 일치 (#58) |
| B6 | HOS 방 Hs 3.0 (ann/fan/enc) | −0.007 / +0.012 / +0.023 | `h7b_linear_twin.json` − h7 M3 | 일치 |
| B6 | HOS 방 Hs 1.2 열 | — | h7b에 Hs 1.2 쌍둥이 없음 | **미확보** (문서-전용; 표 미포함) |
| B7 | 실측 60창 hand 조우/전장 | 0.532(28/60) / 0.550 | `gate_promotion_eval`, `field_wide_eval` | 일치 |

## 미확보 목록 (JSON 재도출 불가 → 표에 미포함/각주 처리)
1. **결측률 92 % → 4.8 %** (Ch.A): H3 아크 문서 컬럼. h3 JSON에 미포함이며
   H1 정의(600/87 k)로는 99.3 %. 커버리지는 r90·n_raw로 대체 보고.
2. **top-k 50 부채꼴 rho** (Ch.A): H1 summary에 rho_fan 없음. "사망"은 정성.
3. **100 ms 계산 예산·2k-solve 타이밍** (Ch.B B1): h5/h6에 미포함.
   `deploy_speed.json`은 F1/RELAX/DPS(타 추정기)라 근거로 부적합.
4. **HOS 학습-방 Hs 1.2 열** (Ch.B B6): `h7b_linear_twin.json`이 Hs 3.0만
   보유. Hs 1.2 쌍둥이 미저장.
5. **dir180 head-sea 10-에피소드 벤치마크** (Ch.B B4): 미채점. 존재하는
   `a2_dir180_eval.json`은 2 에피소드(BF3) 방향-불변 대조뿐.

## 불일치 목록 (문서 vs JSON — JSON 채택)
1. **B2 rand 5 k 행**: 문서 사다리(0.61/min −0.39/6/8)는 h3 T_m50 런에서
   재도출 일치. `h4_selection_ladder.json` `rand@5000`은 별도 런으로
   0.607/min −0.299/5-8 — 표는 정의가 일치하는 **h3(T_m50) 채택**.
2. **결측률**(위 미확보 1과 동일): 문서 92 % vs H1 정의 99.3 % — 표 미포함.

## 검증에 사용한 JSON (경로 확정)
`results/direction3/` 하위: `h3_fixedT_operating_domain.json`,
`h1_mask_destination_main.json`, `h1_mask_destination_mcar.json`,
`h4_selection_ladder.json`, `h5_em_pathology.json`, `h5_combo_recipe.json`,
`h6_v2_e2e.json`, `h7_hos_arena.json`, `h7b_linear_twin.json`,
`gate_promotion_eval.json`, `field_wide_eval.json`, `a2_dir180_eval.json`,
`deploy_speed.json`. 임계식: `wavesim/raydrop_threshold.py`.
