# E-42 Design D2 — AMP-MMSE: Amortized Mixture-Posterior MMSE
**Designer 2 of 3 (독립 경쟁 설계). 2026-07-17.**
대상: cross-Tp crossing seas의 oracle − cur_grid = +0.0982 ± 0.0204 (n=32, 26/32).

---

## 0. 가설류 한 문장 (hypothesis class)

> **입력** z의 6ch `maps6` ⊕ 1-lobe explain-away **residual maps** 6ch →
> **출력** 저차원 2-시스템 sea 파라미터 θ = (d1, a1; m; Δd2, a2, w)에 대한 **이산 joint posterior q(θ|·)**
> (출력 파라미터화 = γ가 아니라 θ의 확률분포) →
> **추론** q가 제안한 K개 joint 가설 {γ_h = lobe_grid(θ_h)}를 **정확 Gaussian evidence로 tempered
> reweight**하여 per-hypothesis MMSE **field들의 mixture** Σ w_h(z)·MMSE(γ_h; z)를 출력.
> **학습 신호** = 시뮬 (θ, z) 쌍에 대한 **NLL(cross-entropy)** — task loss 아님.

incumbent(빔포밍맵→CNN→고정뱅크 γ 1개→task loss)와 겹치는 것은 solver와 bank뿐이며, 그것들은
**가설 내부의 도구**로 쓰인다. 출력 객체·학습 신호·입력 통계·추론 구조 4개가 전부 다르다 (§2).

---

## 1. 근거 사실과의 정합 (fact-by-fact)

| Established fact | 이 설계에서의 역할 |
|---|---|
| F1 realization 축 폐쇄 | θ는 6차원 ensemble 파라미터만. 성분/각도 복원 일절 없음 |
| F2 PERM −0.077 (sharp-wrong 유해) | commitment가 z-의존 evidence 가중으로만 발생; flat하면 smear로 후퇴 (§4) |
| F3 개구 폐쇄 | 관측 사양 불변, 400pt/70s 그대로 |
| F5 E-39 데이터 갭 아님 | 같은 클래스 재학습이 아님 — 클래스 교체. 학습 데이터 요구량은 incumbent와 동급 |
| F6 E-41: prize = d1 + interaction, incumbent = m2 수준 | §2·§3의 출발점. joint 가설이 MAIN+interaction을 함께 겨냥 |
| F7 evidence fit: d1 tail-rescue 13/24, ring 3–7%, corr(d2_err,ΔlogE)=+0.419 | evidence는 **가중 기계로만** 사용(문턱 없음), tempering + NLL-보정으로 +0.419 병리 방어 (§4) |
| F8 E-28 collapse | §2 체크리스트로 명시 회피 |

### 새 프로브 2건 (이 설계 세션에서 직접 실행, 수 분)

**P-1. prize 바다에서 mu_hat(빔포밍 d1)이 오염되어 있다** (`e41_regime_prize_ladder.json` X_rows 재분석):
```
mu_err quartiles = [14.6, 27.2, 42.5, 95.0]°,  >45°가 8/32, >90°가 4/32
corr(mu_err, MAIN_d1) = +0.517
oracle − cur | mu_err>45° = +0.135 (n=8)  vs  ≤45° = +0.086 (n=24)
```
⟹ **E-41의 "d1이 유일한 유의 MAIN"의 기제**: 두 번째 시스템이 전역 빔포밍 피크를 끌어당겨
d1 읽기 자체를 오염시킨다. d1을 고치려면 lobe-2를 동시에 explain-away 해야 한다 —
**이것이 ">절반이 interaction"의 물리적 실체**다. d1만 겨냥하는 설계(+0.050 cap)와
joint 추론 설계의 차이가 여기서 갈린다.

**P-2. explain-away residual map**: blind 1-lobe MMSE fit(mu_hat, universal radial, smult=3)의
잔차 e = z − fit에 대한 `maps6(P, e, kp)`의 lobe-2 capture(prof(d2)/prof(d1), pm 채널):
8바다에서 stage-1 lock이 괜찮은 경우(mu_err<35°) 1.6–3.2× 노출 개선, stage-1이 틀리면(>60°) 무효.
⟹ residual 통계는 **유용한 보조 입력이되 단독으론 불충분** — stage-1 의존성이 있으므로
조건화는 반드시 joint여야 하고(P-1과 정합), 최종 무게는 evidence reweighting이 진다.

---

## 2. 왜 incumbent가 구조적으로 못 하는가 (E-28 회피 논증)

incumbent의 클래스 = {z ↦ W(γ(BF(z)))·z : γ는 per-atom matched-filter 맵의 결정적 함수}.
γ 하나가 정해지면 추정기는 **z에 선형인 고정 필터 1개**다.

1. **결정 구조의 벽.** prize 바다들에서 θ-posterior는 다봉이다(±방향 모호, lobe-2가 lobe-1
   sidelobe에 가림, d1 오염 — P-1). task loss 아래에서 단일-γ 망의 위험최소해는 posterior smear다:
   sharp-wrong은 PERM(−0.077)이 보여주듯 smear보다 크게 손해라서, 학습이 잘 될수록 m2로 간다.
   **F6이 이를 실측**: m2 − cur = −0.005 ± 0.024, incumbent는 이미 honest all-marginal 수준.
   즉 incumbent의 실패는 학습 부족이 아니라 **그 클래스의 Bayes-최적점이 m2라서**다.
   E-39(더 많은 cross-Tp 데이터 무효)가 독립적으로 같은 결론.
2. **이 설계의 출력은 그 클래스 밖이다.** Σ w_h(z)·MMSE(γ_h;z)는 mixture-of-Gaussians prior 하의
   exact posterior mean이며, 가중 w_h가 z의 함수 → **비선형 추정기**. posterior가 옳은 joint mode에
   집중하는 바다에서만 sharp commit, 아니면 smear — 단일 γ로는 표현 불가능한 "per-sea calibrated
   commitment"가 클래스의 정의 자체에 들어 있다.
3. **입력 통계의 벽.** maps6는 dispersion-constrained이지만 **per-atom marginal power**다. 희소
   1D-ish 궤적의 PSF sidelobe 아래에서 두 시스템이 간섭하는 joint 구조(explaining-away)는 exact
   Gaussian evidence가 공짜로 계산하고(400×400 chol), CNN은 궤적-의존 PSF의 deconvolution을
   근사 학습해야 한다 — F1이 확립했듯 net은 marginal deconvolution까지는 배웠고(2.44×) 그 가치는
   0이었다. 남은 것이 joint 항이고, 그건 evidence의 서식지다 (F7의 wrong-lobe rescue가 실측 증거).
4. **E-28 collapse 체크리스트**: 출력 객체(θ-posterior ≠ γ) ✗공유, 학습 신호(NLL ≠ task loss)
   ✗공유, 입력(residual maps 추가) ✗동일, 앵커(warm-start 없음 — 학습은 회귀가 아니라 posterior
   추정) ✗공유. informativeness **스칼라** 조건화는 일절 안 씀(E-28에서 ~0 확정) — 조건화는
   per-hypothesis evidence라는 **z-의존 벡터**로 한다. E-28이 시험한 적 없는 축이다.

**대칭 질문 — "그럼 net 없이 evidence만으로 되는 것 아닌가?"** 정직한 답: 그럴 수 있고,
그 경우 학습부는 죽는다. A5 ablation(§5)이 이를 사전등록으로 직접 판정한다. net의 존재 이유는
(a) full-circle d1 × joint(Δd2,a2,w) 커버리지를 K=128로 압축하는 calibrated proposal,
(b) **+0.419 병리의 교정** — evidence는 misspecification 아래에서 sharp-wrong을 선호할 수 있는데,
NLL-학습된 q는 "이 맵 패턴에서 참 d2가 실제로 어디 있었는가"의 빈도 정보를 가중에 싣는다.

---

## 3. 아키텍처·학습·추론

**가설 공간** (X 아레나 기준 6-dim; Y 확장 시 spread/γ 추가):
d1 ∈ full circle(mu_hat 상대 72 bins — wrong-lobe rescue를 표현 가능하게 **±40° 잘림 금지**, e41
detector의 구조적 결함 교정), a1 ∈ 6 bins, m ∈ {1-sys, 2-sys}, (Δd2 × a2) = **joint 364-way head**
(26 signed-offset bins ±[45°,170°] × 14 ring bins [0.55,1.85]), w ∈ 6 bins.
γ_h는 기존 `lobe_grid`(e41)로 생성 — 코드 재사용, 캐시.

**망**: PriorNet trunk 재사용(12ch stem) → GAP → heads. d2 heads는 (Δd2,a2)를 **joint softmax**로
(marginal factorization이 interaction을 죽이는 걸 방지). rotation augmentation(문제의 회전
등변성)으로 표본효율 확보.

**학습 데이터**: `build_sea_openbi`식 넓은 generator(unimodal 30% / equal-Tp bimodal 20% /
cross-Tp 50%; Tp1~U(5.5,9.5), spread~U(15,45), r2~U(0.15,0.5), U~U(0,1.5)cp) 20k 샘플.
θ 라벨은 generator에서 공짜. 생성 비용 실측 기반 sea당 ~0.3–0.6 s(maps6 ×2 + 1-lobe solve)
→ ~2–3 h, 1회 캐시.

**학습**: NLL(각 head cross-entropy 합), ≥40k steps, batch 256, best-val-NLL, 수렴곡선 필수,
**3 seeds**(train-enough + 소표본 단정 금지 규율). ~1–2 h/seed on 4080S.

**추론** (바다당 ~2 s):
1. top-K=128 joint 가설을 q에서 beam-enumerate. **항상 포함**: h0 = q-marginal smear(m2급 앵커),
   h_cur = incumbent의 γ(do-no-harm 앵커), h1 = 1-lobe best.
2. `EvidenceEngine`(e41, smult=3 그대로)으로 L_h 배치 계산.
3. w_h ∝ q(θ_h)^α · exp(β·L_h). (α, β)는 held-out **sim** 2k에서 태스크 스코어 최대화로 고르는
   전역 스칼라 2개 — per-sea 튜닝 없음, 사전등록된 프로토콜.
4. 출력 = Σ w_h · MMSE(γ_h; z) — **field-space mixture** (grid-space mixing은 γ 평균 = 단일-γ
   smear로 회귀하므로 부차 arm으로만).

---

## 4. PERM / +0.419 / interaction 정면 대응

- **Interaction**: 가설이 (d1,Δd2,a2,w) **joint**이므로 posterior가 옳은 joint mode에 집중하면
  MAIN(+0.048)과 interaction(+0.055)을 함께 획득. P-1이 보여주듯 interaction의 실체 = "d1 수리가
  lobe-2 explain-away를 전제" — joint 가설 스코어링이 정확히 그 축이다. d1-only 설계의 +0.050
  cap에 걸리지 않는다.
- **PERM(sharp-wrong 유해)**: 3중 방어. (i) NLL은 proper scoring rule — confidently-wrong이
  학습 목적함수에서 직접 벌점(task loss는 smear로 도피 가능, NLL은 불가). (ii) evidence tempering
  β≤1 + q^α 혼합 — evidence 단독의 과신(+0.419) 완화. (iii) h0/h_cur 앵커가 혼합에 상존 →
  가중이 flat하면 출력이 m2/incumbent 수준으로 바닥.
- **사전등록 anti-PERM 진단**(스코어와 독립): committed 바다들(max w_h > 0.5)에서
  corr(d2_err, max w_h) ≤ 0 이어야 함. 양수(+0.419 재현)면 스코어와 무관하게 설계 **flagged**.

---

## 5. 사전등록 (falsifiable, 실행 전 동결)

**아레나**: 신선 X 바다 `build_sea_crosstp(si)`, si=100..195 (**n=96**; 기존 분석에 쓰인
0–31 / 500–523 / 3000–3023과 disjoint). 컨트롤: unimodal n=48(`build_sea_bimodal(2000+j,
unimodal=True)`), open-bimodal Y n=48. paired SE 예상 ≈ 0.020·√(32/96) ≈ **0.012** → MDE ≈ 0.024.

**Arms**: cur_grid(기준) / m2 / ens1 / **AMP**(본 설계) / A3b = grid-space mixing 변형 /
A4 = β=0(proposal-only) / A5 = α=0·uniform-prior evidence-only BMA(같은 K) / oracle(천장).

**판정** (전부 paired, n=96):
- **PRIMARY**: mean(AMP − cur_grid) ≥ **+0.04**, 95% CI 0 배제, win ≥ 60%. (+0.098의 ~40% 회수.
  목표를 +0.04로 잡은 산수: commit 가능 비율 ~50–60% × per-sea 상금 ~0.09 × 포착률 ~0.7.)
- SECONDARY: AMP − m2 CI 0 배제(“smear를 이겼는가” — cur만 이기는 건 무의미, cur≈m2이므로);
  far-Tp(>25%) 층화 보고; capture fraction (AMP−cur)/(oracle−cur) 보고.
- DO-NO-HARM: unimodal & Y에서 mean(AMP − cur) ≥ −0.01.
- CALIBRATION GATES(스코어링 **전**, held-out sim 2k): m-head ECE ≤ 0.05; (Δd2,a2) 80%
  credible set coverage ∈ [0.72, 0.88]; mu_err>45° 부분군에서 d1-head mode가 참 ±15° 안 ≥ 50%
  (rescue 능력의 사전 확인).
- **KILL 조건(내 설계의 사망 선언)**:
  (K1) PRIMARY CI 상한 < +0.02 → **사망** (수렴·3-seed 확인 후).
  (K2) calibration gate 실패(충분학습 후) → 설계 사망, 스코어링은 exploratory 라벨로만.
  (K3) AMP − A5 의 CI가 0을 포함 → **학습부 사망**: amortizer는 dead weight, 성과는 evidence
  기계의 것 — "학습 추정기" 주장 철회하고 part-B 파이프라인으로 귀속.
  (K4) anti-PERM 진단 양(+)이면 스코어 무관 flagged.

**컴퓨트**: 데이터 2–3 h(1회) + 학습 1–2 h × 3 seeds + eval ~30 min. 4080S 하루 내. ✓

---

## 6. 자기 반박 — 실패 모드 (사전 명시)

1. **"Calibration이 상금을 먹는다"** (최대 위험). 정직한 posterior가 대부분의 바다에서 flat →
   출력이 smear로 후퇴 → 이득 ≈ 0 ≈ incumbent. E-41의 honest TPR 33%가 직접 경고다. PERM의
   교훈은 양날: 벌을 피하려면 확신 못 할 때 안 걸어야 하는데, 걸 수 있는 바다가 적으면 기대
   이득이 +0.02–0.03에 머물러 K1으로 죽는다. 이 트레이드는 회피 불가능하고, 그래서 목표를
   +0.098이 아니라 +0.04로 사전등록했다.
2. **Family misspecification의 양방향 위험.** X 아레나에서 `lobe_grid` 패밀리(JONSWAP radial ×
   wrapped-normal, spread 30)는 generator와 **사실상 well-specified** — 이기면 "패밀리가 정답을
   내장했다"는 비판이 정당하다(→ Y 아레나 do-no-harm + spread-free 변형 보고로 부분 방어, 완전
   방어 불가). 거꾸로 Y/실측에선 misspecification이 +0.419 병리를 재점화 → β 최적이 0으로 가며
   실패 모드 1로 붕괴. **이 설계의 승리는 X 아레나 한정 주장만 licensed.**
3. (부차) type-II ML의 SBL 병리 — evidence가 z-과적합 γ(특히 h_cur)를 선호할 수 있음; tempering이
   유일 방어이고 α,β 2-dof가 sim-val에 과적합할 표면적은 작지만 0은 아님.
4. (부차) 10° bin 해상도로 d2를 25° 이내에 못 꽂을 수 있음 — evidence local refine(e41 코드)로
   완화 가능하되 refine은 evidence 기반이라 +0.419에 노출. refine 없는 버전을 기본으로 사전등록.

---

## 7. Part B — `e41_blind_recovery.py`에 대한 의견 (오케스트레이터 실행분)

**7.1 Arms.**
- **PRIMARY로 fit_bma(문턱 무관)를 지금 지정하라.** 스크립트가 아직 없으므로 지금 쓰는 docstring이
  곧 사전등록이다. fit2_hard는 동결 TAU=14.64에서 TPR 33% → 사실상 fit1과 동치가 되어 정보가 없다.
  fit2_hard는 arm으로 유지하되(원 docstring의 명명 존중) "detector-limited" 라벨을 미리 박아라.
  **TAU=3.5는 절대 금지** — calibration 데이터를 보고 고른 post-hoc이다.
- **fit_bma_field arm을 추가하라** (같은 가중, grid 혼합 대신 **field 혼합**): 현 `_bma`는
  grid-space 혼합 = 공분산 평균 = **단일-γ smear로 회귀** — incumbent 클래스 안이다. field-space
  혼합이 mixture prior의 exact posterior mean이고 joint commitment를 보존한다. 이 두 arm의 차이
  자체가 "클래스 경계가 상금을 가르는가"의 직접 측정이라 공짜 정보다.
- fit2_point는 유지 — PERM-스타일 진단(point ≪ bma면 hedging이 일하고 있다는 증거).
- **컨트롤**: cur_grid, **m2**(honest smear — 진짜 null), **ens1**(mu_hat 1-lobe): fit1 ≈ fit_bma면
  성과는 "lobe-2 회수"가 아니라 "evidence가 d1/ring1을 수리"한 것 — 주장을 그에 맞게 좁혀야 한다.
  ens2_oracle(천장), 그리고 **unimodal 블록 n≥24에서 전 arm의 harm 측정**(E-37의 핵심 교훈:
  이게 없으면 X 아레나 100% crossing 구성이 발화 남발의 비용을 숨긴다).

**7.2 목표/문턱.** +0.03 bar 자체는 유지할 만하다(tail-rescue 경제학: 회수 가능분이 d1 수리
+ ring이고 d2는 중앙값 40.9°라 PERM 손실로 일부 반납 — 정직한 기대치 +0.02~+0.05). 그러나
**n=32는 부적정**: SE≈0.020 → +0.03 참효과의 검출이 반반이다. **신선 바다 si=100.. n=96**으로
(SE≈0.012), 그리고 "≥ +0.03 & CI 0 배제 & unimodal harm ≥ −0.01"의 3중 판정을 권한다.
층화 보고(far-Tp>25%)를 secondary로.

**7.3 E-37 재발 방지 컨트롤.**
- per-sea gain을 d2_err·dlogE에 회귀해 보고: "corr(gain, dlogE) < 0"이 나오면 evidence 과신이
  해를 만들고 있다는 뜻 — 그걸 정보 부재로 오독하지 말 것(E-37의 정확한 실수).
- accepted-but-wrong(d2_err>45°) 바다의 조건부 평균 gain을 따로 보고 — 검출기 해악의 직접 계량.
- 같은 32바다(si 0–31) 재사용 금지: 그 바다들로 +0.098도, E-41 분해도, 모든 통찰도 만들어졌다.
  forking paths의 교과서 조건이다. 신선 si로.

**7.4 최대 위험.**
- **False positive**: (i) si 0–31 재사용 + 어떤 형태든 calibration 데이터를 본 뒤의 knob 변경
  (TAU, 탐색 grid, w floor). (ii) fit1의 d1/ring1 수리 이득을 "crossing-sea 정보 회수"로 과대
  포장 — ens1/fit1 컨트롤로만 분리 가능.
- **False negative**: (i) **stage-1 d1 탐색이 mu_hat±40°로 잘려 있음** — P-1이 보여주듯 prize
  바다의 8/32이 mu_err>45°, 4/32은 >90°인데 fit1은 이를 구조적으로 못 고치고, fit2의 rescue는
  swap(46%)이라는 우연 경로에 의존한다. full-circle stage-1이 맞는 수리지만, 그건 **v2 detector
  선언 + 신선 calibration**이어야지 조용한 tweak이면 사전등록 위반이다. (ii) grid-space `_bma`가
  joint commitment를 씻어내는 것 — fit_bma_field arm이 방어. (iii) 동결 TAU가 fit2_hard를
  무의미하게 만드는 것 — 이미 알고 있으니 결과 해석에서 fit2_hard 음성은 "정보 부재"의 증거로
  쓰면 안 된다.

**7.5 내 설계와의 관계.** 이 recovery 실험은 내 A5 ablation(evidence-only)의 사전판이다.
fit_bma(_field)가 +0.05 이상을 회수하면 내 설계의 학습부는 시작 전에 절반쯤 죽는다 — 그 경우
남는 질문은 "amortized proposal이 full-circle 커버리지와 +0.419 교정으로 잔여분을 더 먹는가"뿐이고,
그 답이 K3 기준으로 판정된다. 반대로 fit_bma가 0 근처면 evidence 가중 단독으론 부족하다는 뜻이라
NLL-보정 proposal의 존재 이유가 강해진다. 어느 쪽이든 실험은 내 설계 이전에 돌릴 가치가 있다.

---

## 8. 산출물·참조
- 본 문서: `tmp/e42_design_D2.md`
- 프로브 P-1/P-2: 본 세션 인라인 실행 (원자료 `results/direction3/e41_regime_prize_ladder.json`,
  코드 조각은 §1에 기술 — 재현 ~2 min)
- 재사용 코드: `tmp/e41_regime_prize_ladder.py`(lobe_grid), `tmp/e41_evidence_detector.py`
  (EvidenceEngine), `tmp/e29_gate0_ladder.py`(solve_with_grid), `scripts/d3_prior_net_train.py`
  (PriorNet trunk), `tmp/e37_bimodal_ladder.py`(아레나)
- 캐비엇: 전부 시뮬. X 아레나에서 lobe_grid 패밀리는 generator와 정합(§6-2) — 승리 주장은
  X 한정. 실물 LiDAR 미관측.
