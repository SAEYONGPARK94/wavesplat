# Direction II — 검증 로그 (매 단계: 설계문서 대조 + 박사 novelty/contribution)

설계문서: [docs/direction2_diffusion_prior_design.md](../../docs/direction2_diffusion_prior_design.md)

---

## Step 1 — 스펙트럼 표현 + GP-marginal-likelihood forward (sanity)

**날짜:** 2026-06-26

### A. 설계문서 대조 (md §1, §4-Step1)
- 계획: "위상을 GP로 주변화 → S에 대한 marginal likelihood `p(z|S)=N(z;0,K_S+σ²I)`가
  미분가능 forward. Step1 = 이 forward의 sanity." → **수행 완료.**
- 판정 기준(암묵): 정답 S가 틀린 S보다 높은 likelihood여야 posterior가 작동. → **충족.**

### B. 결과 (BF4_5c, 300 관측점, window=10프레임)
| 스펙트럼 | log-likelihood | Δ(정답−해당) |
|---|---|---|
| 정답 S_true | **+437.2** | — |
| 30° 회전 | +383.7 | **+53.6** |
| 90° 회전 | −605.9 | +1043 |
| 랜덤 | −22193 | +22630 |
| 빈(=prior없음) | −22556 | +22993 |

### C. 핵심 발견 (단순 PASS 이상)
**likelihood의 방향 민감도가 비대칭이다:**
- 큰 오차(90°)·구조붕괴(랜덤)는 **파국적 페널티**(−1000~−22000) → likelihood가 *거친 구조*는 강하게 식별.
- **그러나 작은 오차(30°)는 +53.6 / 300관측 ≈ 0.18 nat/obs 로 거의 평평** → 정답 근방에서 likelihood가 **flat**.
- 이것이 **aperture problem의 정량적 실체**: 데이터만으로는 미세 방향이 약하게 제약됨.
  → WaveSplat(per-episode MLE)이 방향에서 막히는 이유, 그리고 **prior가 채워야 할 정확한 공백**을 수치로 보여줌.

### D. 설계 대비 보완 필요 (다음 단계로)
- numpy forward는 미분 불가 → posterior sampling용 **torch 미분가능 버전**(∇_S log p(z|S)) Step3에서 구현 필요(설계 §1 명시).
- K_S 비용: active-cell 합으로 처리했으나 dense noisy S(diffusion 중간상태)에선 무거움 → 저랭크/subsample 최적화 Step3 과제(설계 §5와 일치).

### E. 박사 novelty / contribution 검토
- **이 실험 자체는 기여가 아님**(sanity). 그러나 두 가지 박사급 함의:
  1. **정식화의 타당성 입증**: "위상 주변화 GP-marginal로 스펙트럼 역문제를 posterior로 세운다"는
     설계가 작동함을 확인 → 방향 II의 토대 성립.
  2. **30° vs 90° 비대칭 = Direction I(CRB)과의 다리.** 정답 근방 likelihood 곡률 ≈ Fisher 정보 ≈ CRB⁻¹.
     즉 이 평평함이 **방향 추정의 근본한계**를 가리킴. → "데이터 likelihood는 거친 구조엔 sharp,
     미세 방향엔 flat; 따라서 (II)학습된 prior와 (I)근본한계 분석이 둘 다 필요"라는 **논문 서사**를 데이터로 뒷받침.
- **정직성:** 아직 "혁신"이 아니라 "혁신의 전제 검증". novelty는 Step2(방향스펙트럼 diffusion prior)와
  Step3(물리 posterior가 oracle gap을 좁히는가)에서 판가름.

### F. 판정
✅ **PASS — Step 2(diffusion prior 학습)로 진행.** 단, 위 D의 torch forward·최적화는 Step3 전 필수.

---

## Step 2b + Step 3 — diffusion prior + 물리 posterior (핵심 GO/NO-GO)

**날짜:** 2026-06-26

### A. 설계문서 대조 (md §1 정식화, §4 Step2/3)
- Step2b: diffusion prior 학습 → 생성 스펙트럼이 실제와 일치(energy 0.52 vs 0.55, occupancy
  0.067 vs 0.070, 시각적으로 peak+방향호 재현). **PASS.**
- Step3: PnP(likelihood gradient + diffusion 투영)로 S 추정 → eval. **수행 완료.**

### B. 결과 (field nRMSE, 1-step forecast, 3 에피소드)
| | MLE(prior 없음) | **PnP(diffusion prior)** | grid-oracle | exact-continuous |
|---|---|---|---|---|
| 평균 | **1.000 (완전실패)** | **0.294** | 0.282 | **~0.06** |

진단 (격자 양자화 분리):
- exact 연속 성분 → 0.044/0.075 (oracle-GP 0.07 수준)
- 64×64 격자 oracle → 0.265/0.291 (**격자손실 +0.22**)

### C. 핵심 결론 (두 갈래로 명확)
1. **prior 개념은 작동한다 (GO):** MLE 1.000 → PnP 0.294. likelihood만으로는 ill-posed(완전실패)인
   방향문제를 **학습된 prior가 해소**. PnP가 grid-oracle(0.282)에 도달 → "격자가 허용하는 만큼
   복원". T-NSE(amortization 붕괴)와 정반대 — **prior-learning + per-episode 물리 posterior는 작동**.
2. **격자 표현이 치명적 병목 (NO-GO on grid):** 완벽한 grid-oracle조차 0.28 (exact 0.06의 5배).
   64×64(Δk≈0.034)의 방향/파수 양자화가 정밀도를 파괴. WaveSplat의 grid-free(0.20)보다 나쁨.

### D. 박사 novelty / contribution 검토
- **검증된 기여 (positive):** "방향스펙트럼 diffusion prior + 위상주변화 GP-marginal posterior"가
  aperture-limited 역문제를 푸는 **개념적 타당성 입증** + amortization(T-NSE)과의 대비.
  → 이건 도메인 최초이고 발표 가치 있음.
- **진단 기여:** *왜 naive 격자가 실패하는가*를 정량화(+0.22 격자손실). 이게 다음 방법을 규정.
- **진짜 novel 방법은 합성에 있다:** Direction II(prior가 거친 방향분포 제공) + Direction I
  (off-grid/atomic-norm이 연속 정밀도 제공)는 **대안이 아니라 상보적**. prior로 방향 초기화 →
  off-grid LS 정밀화. 이 hybrid가 oracle gap을 닫을 후보. → 박사 핵심 방법으로 승격 가능.
- **Step1의 30° flat 발견과 일관:** likelihood가 미세방향에 flat → prior가 거친 방향을 잡고,
  off-grid가 미세 정밀화. 세 조각(I/II + likelihood 구조)이 한 서사로 수렴.

### E. 판정
✅ **개념 GO / 격자 NO-GO.** 다음 = **Step 4: grid-free 합성**(diffusion prior 방향 초기화 →
off-grid 연속 성분 정밀화). 이게 Direction I과 II를 묶는 박사 핵심 기여 후보.

---

## Step 4 — grid-free 합성 (prior init → off-grid refine)

**날짜:** 2026-06-26

### 결과 (field nRMSE, 3 에피소드 평균)
| init \ refine | no refine | reg refine(best) |
|---|---|---|
| no-prior(grid-MLE) | 1.000 (실패) | 1.000 |
| **diffusion prior** | 0.294 | **0.278** (거의 불변) |
| oracle 방향 | **0.090** | 0.104 |
| (참고) exact 연속 | 0.090 | — |

### 핵심 결론 — 깨끗한 NO-GO, 정밀 진단
- **off-grid refinement가 oracle gap을 못 닫는다:** prior 0.294 → refine 0.278 (5%만 개선). oracle(0.09)에 한참 못 미침.
- **과적합은 trust-region으로 해결되나(oracle 0.167→0.104), prior-init은 애초에 안 움직임.**
- **근본 원인 = Step1의 "30° flat".** 정답 근방 likelihood가 평평(0.18 nat/obs) → local 최적화가 받을 gradient 신호가 없음. prior가 거친 위치를 줘도 미세 정밀화 불가.

### 통합 통찰 (모든 단계 수렴) — 박사 핵심 서사
1. **데이터 자체에 미세 방향정보가 없다(aperture problem).** Step1(30° flat) = 낮은 Fisher 정보 = 높은 CRB. prior·refine 어떤 영리한 추론도 *없는 정보*를 만들 순 없음.
2. **유일한 탈출구 = 유효 개구 확장(synthetic aperture) = 시간(여러 프레임) 활용.** 이게 정확히 **ST-DF-GP(W프레임 융합, 0.205)** 가 single-window(0.28)보다 잘 되는 이유.
3. → 박사 기여는 "마법 추정기"가 아니라 **(I) 근본한계(CRB) 규명 + (II) 학습 prior의 견고성·한계 + 결론적으로 (winning) space-time 합성이 정답임을 이론·실증으로 입증.**

### 박사 novelty / contribution
- **음성결과도 강한 기여:** "single-window에선 prior로도 한계 — 정보가 없어서"를 CRB로 증명하면, ST-DF-GP의 시간융합이 *필연*임을 보이는 이론적 정당화. 이게 Direction I을 박사 1순위로 끌어올림.
- 정직: Step4 naive 합성은 실패. 그러나 실패가 "왜 시간융합이 답인가"를 규정 → 연구 서사 완성.

### 판정
✅ **개념(prior) GO / 격자·naive-합성 NO-GO / 근본원인 = 단일윈도우 정보부족.**
→ 다음 우선순위 전환: **Direction I (CRB + synthetic-aperture 이론)** 으로 "시간융합이 정답"을 증명.
