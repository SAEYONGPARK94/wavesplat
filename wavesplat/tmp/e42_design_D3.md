# E-42 Design D3 — Regime-Bayes Model Averaging (RB-BMA): 사후분포를 출력하고, 가설 단위로 커밋한다

Designer 3 / 3. 2026-07-17. 근거 코드 직접 판독: `tmp/e41_regime_prize_ladder.py`,
`results/direction3/e41_regime_prize_ladder.json`, `tmp/e41_evidence_detector.py`,
`tmp/e37_bimodal_ladder.py`, `tmp/e29_gennet_C.py` (`maps6`, `beamform_power`,
`solve_with_grid`), `scripts/d3_prior_net_train.py` (incumbent 손실).
기제 프로브 실행: `tmp/e42_probe_D3.py` (si=200..205, 채점 이력 없는 시드, n=6 스모크 — §7).

---

## 1. 가설류 한 문장

**입력 (6ch beamform maps + raw (P, z)) → 출력 = 저차원 joint regime 가설공간
θ=(d1,a1,d2,a2,w) 위의 이산 사후분포** — 동결된 dense joint grid 위의 categorical,
가중 = exact type-II Gaussian evidence × (학습된 per-sea temperature/abstention head β);
**훈련신호 = proper scoring rule** (proposal은 참 θ에 대한 NLL — sim에서 라벨 공짜;
trust head는 1-차원 출력에 대한 task-score, 채점과 disjoint한 교정 바다에서 동결);
**최종 추정 = 가설 단위 BMA**: γ̂ = Σ_k w_k(z)·γ(θ_k) (primary), û = Σ_k w_k(z)·MMSE(z; γ(θ_k)) (secondary).

- "beamform map → CNN → 고정뱅크 γ → task loss"가 **아니다**: 출력 파라미터화가
  1904-atom γ가 아닌 5-dim joint θ의 분포이고, 손실이 task-loss-through-γ가 아닌
  proper scoring이며, 결정규칙이 단일 solve가 아닌 z-의존 가중 혼합이다.
- E-28 비반복 체크: 앵커 warm-start 없음(구조적 floor = uniform-BMA ≈ m2 ≈ cur, §5),
  informativeness 스칼라 조건화 없음(evidence는 z 자체의 우도), 고정뱅크 γ 회귀 없음.

---

## 2. 설계를 구속하는 사실 (브리프 + 원자료 재확인)

`e41_regime_prize_ladder.json` X 아레나 (n=32)에서 직접 재확인:

| 양 | 값 | 함의 |
|---|---|---|
| oracle − cur_grid | +0.0982 ± 0.0204 | 상금 |
| m2 − cur_grid | −0.0049 ± 0.0237 | **incumbent = 정직한 all-marginal 수준** |
| oracle − m2 | +0.1030 ± 0.0213 | 상금 = "smear → joint-pointy" 승격의 값 |
| sum(MAIN) = +0.048 vs oracle−m2 = +0.103 | | **절반 이상이 상호작용** |
| d1 MAIN +0.0501 (유일 유의) | | d1은 tail 현상 (si=19: mu_err 158°, M_d1−m2 = +0.43) |
| real − oracle | +0.2122 | realization 축 — 닫힘, 건드리지 않음 |

추가로 코드에서 확정한 두 사실:

1. **`beamform_power`는 dispersion-제약 시간축 steering을 이미 쓴다**
   (`d3_sa_nomp.py:320-340`, `−ω(|k|)·t` 항 포함). 즉 incumbent의 입력에는 ring
   정보가 이미 들어 있다. 갭의 원인은 입력이 아니다.
2. **incumbent 손실 = log-γ에 대한 masked SmoothL1 회귀** (`d3_prior_net_train.py`).
   결정론적 per-pixel 회귀의 손실-최적해는 사후가 다봉일 때 **per-pixel 조건부
   중앙값 = 주변부 smear**다. "incumbent ≈ m2"(위 표)는 버그도 학습 부족도 아니고
   **그 클래스의 Bayes-최적 도달점**이다. E-39(in-distribution 갭 잔존, cross-Tp
   증강 40-50% 있었음)가 데이터 레버를 독립적으로 반증한 것과 정합.

---

## 3. 왜 incumbent 클래스는 구조적으로 못 하는가

incumbent가 구현하는 추정기 클래스: **z ↦ A(γ(map(z)))·z** — map이 정하는 단일
γ로 한 번 solve. regime 불확실성 하의 참 생성모델은 **Gaussian mixture prior**
(각 성분 = jointly-pointy γ(θ))이고, 그 Bayes 추정기는

  E[u|z] = Σ_k p(θ_k|z) · E[u|z, θ_k]  =  Σ_k w_k(z) · A(γ(θ_k)) · z

즉 **z-의존 가중의 가설-단위 커밋**이다. 단일 γ로 이를 재현하려면 moment-matched
단일 Gaussian으로 mixture를 대체해야 하는데, 그 대체 비용이 정확히 E-41이 측정한
oracle − m2 = +0.103이다. 두 방향의 반증을 모두 만족:

- **더 학습/더 데이터로 못 감** (fact 5, E-39): 클래스의 최적해 자체가 smear.
- **조건화 스칼라로 못 감** (fact 8, E-28): 문제는 γ 생성의 조건화가 아니라
  γ "하나만" 내는 출력 파라미터화 + 단일 solve 결정규칙.

본 설계가 여는 것: 출력이 분포(categorical over joint θ)이면 proper scoring의
최적해는 **사후분포 그 자체**다 — 다봉이면 다봉을 내고, 데이터가 판별해 주면
집중한다. 집중된 성분은 완전한 joint 가설이므로 **(d1×d2×ring×w) 결합이 성분
내부에 통째로 보존**된다 — 상호작용 >50% 사실에 대한 정면 응답 (§6).

---

## 4. 설계 본체

### 4.1 Core (학습 없음 — 이것만으로 승부 가능해야 함)

1. **동결 joint 후보 grid** (stagewise freezing 금지 — e41 detector의 wrong-lobe
   lock 원천 제거):
   - d1: mu_hat 기준 **전역** ±180° (step 10°, refinement ±4° step 2°) — d1 tail
     rescue는 국소 ±40° 탐색으론 불가능 (e41 detector의 d1 탐색창이 mu_hat±40°라
     mu_err 100°+ 바다에서 원리적으로 실패; 프로브 si200에서 전역 grid가 구조함)
   - a1: {0.90, 0.95, 1.00, 1.05, 1.10, 1.20}
   - d2: d1 ± [45°, 165°] step 10° (양측), refinement ±4°
   - a2: **0.55–1.85 step 0.10** (아레나 참 범위 [0.51, 1.78] — 프로브에서 이보다
     조악한 grid가 4/6 바다의 ring을 놓침, §7)
   - w: {0.10, 0.20, 0.30, 0.40, 0.50} + **1-lobe 가설(w=0)을 같은 집합에 포함**
     (검출 임계 자체를 제거 — 단일모드는 사후가 w=0 성분에 질량을 주는 것으로 처리)
   - 조합 ~3–6만. **계산은 문제 아님**: K = (1−w)·K_lobe1 + w·K_lobe2 + σ²I 로
     공분산이 lobe-단위 선형이므로 unique lobe별 K_l만 사전계산(수백 개 n×n)하면
     조합당 O(n²) 어셈블 + O(n³/3) Cholesky. 프로브 실측 1944 evals ≈ **1초/sea**
     (4080S, `EvidenceEngine` 재사용) → 6만 evals ≈ 30초/sea, n=64 아레나 ≈ 30분.
2. **Exact evidence**: `EvidenceEngine` 그대로 — N(0, Φ diag(γ) Φᵀ + (3σ₀)²I),
   γ-scale var(z), smult=3 동결. 70s 시간축이 dispersion으로 ring을 우도 안에서
   직접 고정한다 (e41이 확립: ring 3–7% 해결).
3. **가중**: w_k ∝ exp(β·logE_k) × p(θ_k). p(θ) = 아레나 생성분포(균일) —
   **net의 p(θ|map)을 prior로 곱하지 않는다** (evidence가 이미 z를 썼으므로
   이중 사용; 학습된 뾰족 prior가 틀리면 PERM 재판). β는 §5.
4. **결정**: primary = γ-BMA(가중 γ 혼합 후 1회 solve — 사후 집중 시 jointly-pointy
   γ와 동일). secondary = solve-BMA(가설별 field 혼합 = 엄밀한 Bayes 사후평균).
   프로브에서 둘의 분리 없음(§7) → 단순한 쪽을 primary로.

### 4.2 학습 레이어 (분리 게이트 — core가 실패하면 이것들은 의미 없음)

- **L1. Trust/temperature head (반-si204 장치, 본 설계의 학습 핵심)**:
  입력 = (maps6 요약특징, evidence 프로파일 통계: ΔlogE, 사후 엔트로피, top-2 가설
  간 각도/ring 간격, mu_hat–top1 정합도) → 출력 = per-sea β 또는 α∈[0,1]
  (γ̂ = α·γ_BMA + (1−α)·γ_m2). **1-차원 출력이라 구조적으로 과적합 어려움**
  (SmultHead 선례). 훈련 = task-score, 교정 바다에서만, 채점 전 동결.
  타깃 병리 = 프로브 si204 (엔트로피 0.16으로 붕괴했는데 틀림, −0.24) 및
  corr(d2_err, ΔlogE)=+0.419.
  정직 주석: E-28의 informativeness 스칼라(~0 기여)와 가족 유사성이 있다. 차이는
  조건화 대상 — γ 생성이 아니라 **가설 커밋의 강도**. 그래도 ~0 기여로 죽을 수
  있음을 사전 인정하고 별도 게이트로 판정 (§8).
- **L2. Proposal MDN (선택)**: maps6 → 연속 θ 후보 (grid 밖 refinement + 다봉 d1
  제안). NLL on 참 θ, 8–16k 합성 바다, 30–50k step, 3 seed. grid가 이미 싸므로
  아레나 내 기여는 작을 것으로 예상 — 배포(100ms) 경로용 distillation의 씨앗.
- **L3. Distillation head (배포용, 아레나 채점 대상 아님)**: evidence 사후를 KL로
  흉내내는 categorical net — 오프라인 evidence 계산을 100ms 추론으로 증류.

### 4.3 이 설계의 구조적 안전판

β→0 (혹은 α→0) 극한 = uniform-BMA/m2 ≈ cur (m2−cur = −0.005). 즉 **교정이
정직하면 실패 모드가 '동률'로 퇴화**한다 — e37 ens2_blind의 −0.059 같은 손실
모드(100% 발화 + 66° 오차의 뾰족 커밋)가 구조적으로 없다. 단서: 이 안전판은 β
교정이 채점과 disjoint한 바다에서 동결될 때만 유효하다.

---

## 5. PERM / "+0.419 confidently-wrong" 정면 대응

1. **argmax 전면 금지** — 모든 arm이 BMA. (fit2_point류는 진단 전용.)
2. **+0.419의 원인 분해 가설**: (a) stagewise freezing의 wrong-lobe lock,
   (b) 상관 잔차의 유효표본수 ≪ 400에서 오는 우도 과신, (c) family 미세오지정.
   (a)는 joint grid로 제거 — 프로브에서 catastrophic-mu 바다의 d2err 1.3°(si200),
   4.9°(si201)로 부분 확인. (b)는 β로 평탄화. **(c)는 β로 못 고친다** — 편향은
   온도의 문제가 아님. 그래서 사전등록 진단(§8)에 "집중-바다 부분군에서
   sign(RB−m2)"를 넣고, 음수면 설계 사망 선언.
3. **프로브가 병리를 실증**: si204 — 사후 엔트로피 0.16(사실상 단일 가설), top
   가설의 d2는 참과 3.4°인데 primary lobe 배치가 틀려 −0.24 손실. 즉 이 설계의
   PERM-손실은 이론적 걱정이 아니라 **관측된 현상**이고, L1 head와 grid 정밀화가
   그 위에 하중을 받는다. 이 사례를 교정 진단의 canonical example로 등록.
4. **단일모드 do-no-harm**: w=0 가설이 후보집합에 있으므로 검출 임계가 없고,
   작은-w nuisance lobe는 field 공간에서 1-lobe와 거의 동일 — e41의 TAU 병리
   (ΔlogE 통계의 오염)는 **verdict 통계의 실패였지 추정기의 실패가 아니다**.
   그래도 unimodal 음성 n=24에서 RB − cur ≥ −0.01을 게이트로 둔다.

---

## 6. 상호작용 >50% 정면 대응

- 각 mixture 성분이 **완전한 joint (d1,a1,d2,a2,w)** — marginal 사다리(M_p)가
  못 담는 결합을 성분 커밋이 정의상 전부 담는다. oracle은 사후 → δ 극한의 성분.
- 캡의 정직한 산식: 회수량 ≈ Σ_sea [사후가 참-근방에 두는 질량] × [그 바다의
  headroom] − [잘못 집중한 질량의 PERM 벌점]. d1만 맞추는 설계의 +0.050 캡을
  넘는 경로는 참-근방 **joint** 질량뿐이다. e41 detector 진단(ring 3–7%, d1
  22.6°, 사후-hoc d2 15°)은 그 질량이 실재함을 시사하되, 프로브 si203/204는
  그것이 균일하지 않음을 보여준다 — 그래서 목표를 +0.098 전액이 아닌
  **+0.04~0.06 (상금의 40–60%)**로 사전 설정한다.

---

## 7. 기제 프로브 결과 (n=6 스모크 — 결론 아님, 방향타)

`tmp/e42_probe_D3.py`, si=200..205 (채점/교정 이력 없는 시드), joint grid 1944
후보/sea (의도적으로 조악한 ring grid), β=1, ~1초/sea:

```
si    muE    topD2e  near  H     cur     uni     γ-BMA   s-BMA   oracle
200   150.8    1.3   0.85  3.35  +0.412  +0.407  +0.584  +0.565  +0.641   ← 전역 d1 grid가 catastrophic-mu 구조
201    20.0    4.9   0.00* 3.28  +0.567  +0.533  +0.586  +0.588  +0.639
202     8.9   20.2   0.00* 3.26  +0.242  +0.300  +0.362  +0.356  +0.327
203    40.3   66.8   0.00* 1.76  +0.701  +0.488  +0.610  +0.617  +0.702   ← headroom 없는 바다에서 −0.09 손실
204   145.6    3.4   0.00* 0.16  +0.417  +0.196  +0.173  +0.174  +0.426   ← confidently-wrong 실증 (−0.24)
205    10.9   25.3   0.00* 4.11  −0.057  +0.156  +0.165  +0.155  +0.090
mean                             +0.380  +0.347  +0.414  +0.409  +0.471
```
\* near-mass 0은 프로브 a2 grid {0.75..1.6}가 참 범위 [0.51,1.78]을 못 덮은 양자화
탓 — 본 설계 grid(0.55–1.85 step 0.1)의 필요성을 역으로 확인.

판독 (스모크 한정): ① joint 전역 grid는 detector가 원리적으로 못 하던 d1-tail
rescue를 한다. ② confidently-wrong은 이 설계에도 실재(1/6) — L1 head가 장식이
아님. ③ solve-BMA ≈ γ-BMA — 표현력 주장의 하중은 "joint 커밋 + evidence 가중"에
있지 혼합 수준에 있지 않다. ④ 조악한 grid로도 mean +0.033 — 방향은 맞다.
⑤ 계산 예산 걱정 소멸 (1초/sea).

---

## 8. 사전등록 (동결 대상)

**아레나**: `build_sea_crosstp` si=100..163 (**n=64, 신선 시드** — si=0..31은
+0.098이 측정된 그 바다이고 이번 세션의 설계 반복 전체가 그것을 봐 왔다; 참고
비교로만 병행 보고). unimodal 음성 `build_sea_bimodal(2000+j, unimodal=True)`
n=24. **비선형 leg** (상시지시): bound-wave 주입 cross-Tp n=32, exploratory 라벨
(family에 구속파 arm 없음 — 결과가 나쁘면 "linear-only" 한정 명시).

**동결물**: 후보 grid (§4.1), β 또는 L1 head (교정 바다 = crosstp 500..523 양성
+ bimodal 3000..3023 음성 — e41과 동일, 채점과 disjoint), smult=3, 모델 prior
균일. 스모크는 si≥900만.

**Arms**: cur_grid / m2 (e41 레시피) / U-BMA (uniform 가중, 컨트롤) /
**RB-γ (primary)** / RB-solve / RB+L1 (trust head) / oracle.

**판정 (paired, n=64, 예상 SE ≈ 0.014)**:
- **성공**: RB-γ − cur_grid ≥ **+0.04**, 95% CI 0 배제, win ≥ 60%.
- **사망**: RB-γ − cur_grid < +0.02 또는 CI 0 포함. (n=32의 MDE ±0.06으로는
  +0.03~0.05대 효과를 못 가른다 — n=64가 최소.)
- 기제 게이트 (전부 필수):
  (i) RB-γ − m2 CI 0 배제 — smear 재측정이 아님의 증명;
  (ii) RB-γ − U-BMA > 0 CI 0 배제 — evidence 가중이 하중을 받음;
  (iii) unimodal RB-γ − cur ≥ −0.01;
  (iv) 양성에서 corr(d2_err, 사후 집중도) ≤ 0 (교정 성립);
  (v) 사후-집중 부분군(엔트로피 하위 1/3)에서 mean(RB−m2) ≥ 0 — 음수면
  우도 편향(§5-2c) 확정, **설계 사망을 스스로 선언**.
- L1 head 게이트 (분리): (RB+L1) − RB-γ ≥ +0.01 & CI 0 배제. 실패 시 "학습
  레이어 무기여, core만 생존"으로 정직 보고 (E-28 선례에 비춰 실패 확률 낮지 않음).
- L2 MDN 게이트 (분리, 선택): 학습 30k+ step·3 seed·val 수렴 확인 후에만 판정.

**계산 예산**: core 아레나 전체 ≈ 30–60분 (GPU). L1 head 학습 ≈ 1h.
L2 MDN ≈ 2–4h. 전부 4080S 단일 카드 반나절 내.

---

## 9. 자기 반박 — 예상 사망 경로 (확률 순)

1. **상금-모호성 동시성 (가장 유력)**: 상금이 몰린 바다에서만 사후가 평평/틀리면
   회수 ≈ 0으로 동률. 프로브 si203-204가 이 방향의 냄새 (headroom 없는 곳에서
   손실, catastrophic 바다 절반만 구조). 진단: per-sea corr(사후 엔트로피,
   oracle−cur headroom) 큰 양수 + 회수 < +0.02 → **regime 축도 400pt 추론-한계**
   결론으로 이관하고 CRB/식별성 계산 (docs §6-2의 처방)이 후속.
2. **우도 시스템 편향이 joint grid에서도 생존**: +0.419가 freezing 아티팩트가
   아니라 family/이산화/상관잔차의 편향이면 β는 못 고친다. 게이트 (v)가 잡는다.
   이 경우 RB는 m2 아래로 내려가고 설계는 죽는다 — 정직하게: 프로브 si204는
   freezing이 없는데도 confidently-wrong이었다. **이미 1/6 반례가 있다.**
3. L1/L2 무기여 (E-28 재판): core만 남으면 "학습 추정기" 서사는 죽고 "결정규칙
   교체" 서사만 남는다 — 과학적으로는 여전히 상금 회수지만 박사 골격에는
   "학습이 필요 없었다"로 기록해야 한다.
4. 아레나-family 근친: family가 아레나 생성분포와 동일 구조(스프레드 30 고정,
   JONSWAP γ=3.3)라 "known climatology" 가정이 실측에선 무너진다. Y-아레나
   (open bimodal: 스프레드/γ per-lobe 자유)와 실측 2레코드 이전은 본 사전등록
   밖의 후속 — 여기서 이긴다고 실측 회수를 주장하지 않는다.

---

## 10. Part B — `e41_blind_recovery.py`에 대한 의견 (오케스트레이터 실행)

**총평**: 실험 자체가 내 설계의 core(학습 없음)와 거의 동일물이다. **먼저
돌려라** — 그 결과가 내 설계의 L1/L2가 필요한지(부분 회수), 불필요한지(전액
회수), 전부 무의미한지(0 회수)를 가른다.

**Arms**:
- `fit1` (1-lobe BMA, 무임계): 포함. d1-tail rescue만의 분리 측정기 —
  `fit_bma − fit1`이 "2번째 시스템 + joint"의 순기여가 된다.
- `fit2_hard`: **동결 TAU=14.64로만**. post-hoc TAU=3.5는 채점 바다의 진단을 보고
  고른 값 — 쓰는 순간 오염. 사실상 UNQUALIFIED가 확정이므로 이 arm은 "검출-게이트
  접근의 부고 기록"용으로만 가치가 있다.
- `fit_bma` (**primary로 지명**): 임계 없음 — 검출기 자격 문제를 통째로 우회.
  단 **두 가지 구조 결함을 고치고 돌릴 것**:
  (i) `fit_two_lobe`의 **stagewise freezing** — lobe1을 stage-1 argmax에 고정한 채
  d2를 탐색하므로 wrong-lobe lock이 후보집합 자체에서 참을 배제한다 (최대 false
  negative 원천). joint grid로 교체 (프로브 실측 1초/sea — 비용 반론 없음).
  (ii) d1 탐색창 mu_hat±40° — d1 MAIN +0.050은 mu_err 100°+ tail에서 온다
  (e41 rows si=19 등). 전역 d1 필수.
- **추가 컨트롤 (필수)**:
  (a) **m2** — 같은 바다에서 이미 계산됨 (e41 JSON). cur ≈ m2이므로 "cur를
  이겼다"만으로는 부족하고 **m2를 이겨야 정보 회수**다. 이것이 E-37 재발(자기
  검출기의 harm을 정보로 오독)의 1차 방어선.
  (b) **U-BMA** (같은 후보집합, 균일 가중) — evidence 가중의 순기여 분리.
  (c) unimodal 음성 n≥24에서 do-no-harm ≥ −0.01.
  (d) 가능하면 solve-level BMA 변형 — 현 `fit_bma`는 γ-혼합이라 사후가 퍼질 때
  구조적으로 m2로 붕괴 (false negative 방향). 단 내 프로브에선 분리 없음 —
  낮은 우선순위.

**목표/임계**: `ens2_blind ≥ +0.03`의 +0.03은 유지 가능하지만 **n=32는 검정력
미달** (SE 0.020 → t 1.5). **n=64로** (si=0..31 + 32..63, 또는 신선 100..163
병행 — si=0..31은 세션 전체가 관찰해 온 바다라 신선 시드 병행이 정직하다).
사전등록 제안: primary = `fit_bma − cur_grid ≥ +0.03` & CI 0 배제 & win ≥ 60%
**AND `fit_bma − m2` CI 0 배제**. 부분군(d1-rescue tail)은 GT를 쓰지 않는 지표
(맵 각도 프로파일의 2봉성 또는 사후 엔트로피)로만 정의하고 exploratory 라벨.

**최대 false positive 경로**: ① arm 다중성 (fit1/fit2/fit_bma × cur_grid/cur_dep
× n=32) — primary arm·primary 대비 사전 지명으로 차단. ② 하이퍼 오염 — TAU=3.5,
그리고 이번 세션 진단을 본 뒤 grid 범위를 조정하는 것 전부. grid는 아레나 생성
범위(문서화된 regime 지식)로만 정당화하고 채점 전 동결. ③ si=0..31 단독 사용 —
설계 반복이 이 32개에 적합해 온 누적 효과.

**최대 false negative 경로**: ① stagewise freezing (위) — 이거 하나로 실험 전체가
"detector-limited"로 끝날 수 있다. ② β=1 과신 우도가 speckle-fit 가설에 질량을
몰빵 (PERM 손실) → "정보 없음"으로 오판. 교정 바다 (500+/3000+)에서 β 하나를
동결해 넣는 것은 GT-peek이 아니다 — smult=3과 같은 지위. ③ ring grid가 참 범위
[0.51,1.78]을 못 덮는 것 (내 프로브가 직접 밟은 함정; 0.55–1.85 step 0.1이면 됨.
끝단 0.51–0.55는 못 덮음 — step 유지하되 0.50까지 확장 권고).

**판정 매트릭스 제안**: fit_bma가 (m2 대비 CI 0 배제로) +0.03↑ = regime 축은
추론 가능, 학습은 장식 → 내 설계의 L1/L2 게이트로 넘어감 / +0.01~0.03 = 부분
회수, 교정·tail이 병목 → L1 head가 정확히 그 자리 / ≈0인데 게이트 (iv)(v) 통과 =
정보 한계 → CRB 계산으로 종결하고 "개구는 천장을 사고 회수를 못 산다"의 regime-축
확장판을 쓴다.
