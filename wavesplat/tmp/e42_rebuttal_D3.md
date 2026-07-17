# E-42 반박 (D3) — kill의 산술은 맞고, kill의 추론은 틀렸다: "exact posterior"는 계산된 적이 없다

D3. 2026-07-17. 새 측정 3건: `tmp/e42_rebuttal_probe_D3.py` (snapped-oracle, n=64),
`tmp/e42_rebuttal_probe2_D3.py` (evidence rank of truth, n=16),
JSON 조건부 분석 (`results/direction3/e42_blind_recovery.json` 원자료).
산출: `results/direction3/e42_rebuttal_snap_D3.json`.

## 0. 인정하는 것 (분쟁 없음)

- 사전등록 pooled 판정 **DEAD는 유효**: fit_bmaF − cur = −0.010, CI 상한 < +0.02.
- 내 n=6 프로브의 +0.033은 n=64에서 죽었다. 소표본 낙관 — 내 잘못.
- 내 실패모드 ①(상금-모호성 동시성)과 si204(confidently-wrong)는 예언적이었다.
- **공격 2(비-Gaussianity)는 내 생존로가 아니다 — 스스로 기각한다.** 결정론적 진폭
  + 유한 방향 표집의 비-Gaussian 보정이 담는 정보는 **realization-축 정보**다
  (닫힘: RL@∞SNR −0.0003). 그걸로 이기는 net은 W7식 시뮬 지문 조회다. 범주 오류.

## 1. 그러나 — 채점 데이터 자체가 kill의 서사를 반박한다

원자료 조건부 분석 (공통-finite n=62):

| 측정 | 값 | 함의 |
|---|---|---|
| **내 사전등록 자진사망 조항** (집중 부분군 bmaF−m2 < 0?) | dlogE 상위 1/3: **+0.0348** (13/21), 하위 1/3: −0.0214 | **미발화.** 우도-편향 사망이 아니라 "평평-사후 바다의 드래그" 프로파일 |
| +0.419 병리 (corr(d2_err, dlogE)) | **+0.078, p=0.54 — 소멸** | 내 §5-2a 가설대로 v1 탐색 아티팩트. "confidently-wrong 우도" 서사 기각 |
| U_BMA − m2 | **+0.0202** ≈ bmaF−m2 (+0.0185) | **U_BMA는 smear 컨트롤이 아니다** — evidence가 선별한 top-48+48(≈MAP 근방)의 균일평균. "evidence==uniform ⟹ 온도축 사망" 추론은 컨트롤이 공허해서 성립 불가 |
| d2를 맞춘 바다(≤20°, 52%)에서 bmaF−m2 | +0.025 / good-joint(n=25) +0.040 / 집중∩정합(n=15) +0.057 | 커밋은 맞을 때 돈이 된다 — 정보벽 예측과 불일치 |
| 같은 부분군에서 oracle−bmaF | **+0.051** | 맞은 fit조차 0.05를 흘린다 → 어디서? §2가 답 |

## 2. 핵심 반박 — 세 측정치

**(a) 격자 천장은 열려 있다.** 참 (d1,a1,d2,a2,w)를 v2 blind 격자(10°×0.1×W_GRID)로
반올림한 snapped-oracle: **snap − oracle = −0.0075 ± 0.0036** (양자화 비용 ≈ 0),
**snap − cur = +0.0624 ± 0.0134 (46/64)**. 상금의 87%가 blind 기계가 도달 가능한
격자점 위에 산다. "PERM at the family level"이 아니다 — PERM은 틀린 뾰족함의
벌점이고, 여기선 **맞는 뾰족함이 격자 위에 존재하는데 기계가 안 집었다**:
**snap − bmaF = +0.0790**.

**(b) 채점된 기계는 그 격자점을 평가한 적이 거의 없다.** stage-B 후보는 **1-lobe
evidence top-6 seed**에 조건화된다. 16개 바다 직접 측정: **참 lobe-1이 top-6에 든
바다 = 1/16** (참 lobe-1의 1-lobe evidence rank: median ≈ 30/504, 최악 326).
교차 바다에서 최선의 *단일*-lobe 설명은 타협 방향/링이지 참 lobe-1이 아니므로
이 seed 기준은 **교차 바다에 대해 체계적으로 편향된 ranker**다. F2는 greedy
freeze를 없앤 게 아니라 argmax-freeze를 같은 greedy 기준의 top-6-freeze로 바꿨다.
(fit1의 유의한 손실 −0.059도 같은 기제로 설명된다.)

**(c) 참은 evidence-경쟁적이다.** 참 5-tuple을 같은 `JointEngine`으로 직접 평가:
**logE(탐색 최대) − logE(snapped 참) median = +1.66 nats** (q25 +0.59, q75 +3.07),
12/16이 ≤3 nats, **2/16은 음수 — 참이 탐색이 평가한 전부를 능가** (X104:
**+14.1 nats**). 즉 우도 표면엔 정보가 있다. 그것을 버린 것은 탐색+가중 레이어다.

**결론**: "z ~ N(0,ΦγΦᵀ+σ²I)가 exact ⟹ evidence 사후 = 사후 ⟹ 학습 불요 ⟹ 벽"
의 연쇄에서 두 번째 고리가 끊긴다. **이 실험이 계산한 것은 family 사후가 아니라
greedy-seeded 부분집합의 사후다.** "detector-limited" 라벨(실험 스스로 사전등록한
단서)이 작동 조항이고, 그 limit의 정체는 top-6 seeding으로 특정된다.

## 3. Coordinator의 4개 공격에 대한 답

1. **내 kill 조항**: 발화하지 않음 (+0.0348, §1). 자진사망 의무 없음.
2. **exact likelihood**: 기술적으로 거짓(CLT 근사 + 51×51 뱅크-양자화 공분산 —
   Δω·T 결어긋남은 이 레포가 문서화한 기제)이나, CLT 몫은 realization 축이라
   **생존로로 쓰지 않는다** (§0). 뱅크 몫은 (c)의 +1.66 nats 잡음 바닥에 기여
   가능성만 기록.
3. **온도축**: U_BMA로는 bracket되지 않았다(공허 컨트롤, §1). 그러나 정직하게 —
   tertile 산술상 완벽한 per-sea 게이트조차 이 파이프라인 위에선 pooled ≈ cur
   동률이 상한이다. **온도는 답이 아니다. 탐색이 답이다.** 내 설계의 L1(trust
   head)은 이 데이터로 사실상 죽었다 — 인정.
4. **CRB 이관**: **시기상조.** 지금 CRB를 계산하면 탐색-유발 갭을 정보 벽으로
   오라벨한다 — E-38에서 "닫힌 headroom"이 뱅크 양자화였던 것과 같은 오류 클래스.
   CRB(Slepian-Bangs, FIM_ij = ½tr(K⁻¹∂_iK K⁻¹∂_jK), 5-dim, 바다당 수 초 — 계산
   가능함은 맞다)는 §4의 측정이 실패한 **뒤에** 올바른 종결이다.

## 4. 생존 판정 + 단일 저렴 측정 (사전등록)

**죽은 것**: 내 설계의 L1 온도 head(위), n=6 프로브의 낙관, 그리고 "top-K seeded
evidence BMA"라는 구현체. **아직 반증되지 않은 것**: "family 사후를 실제로
계산하면 격자 천장 +0.0624의 상당분이 회수된다"는 core 주장 — 이 실험은 그것을
시험하지 않았다.

**단일 측정** (오케스트레이터에게, ~1 h GPU, 학습 없음):
- **seed를 top-6 → top-64로** (참 lobe-1 커버리지 1/16 → 예상 ≥13/16; 대칭성으로
  전수 504도 ~3h에 가능). 같은 바다 si=100..163, 같은 solver, 같은 TAU 불요
  (threshold-free BMA만). arms: bmaF_full / cur / m2 / snap(천장 참조).
- **사전등록**: bmaF_full − cur ≥ **+0.03**, CI 0 배제, win ≥ 60%, 그리고
  진단으로 truth-in-candidate-set 커버리지 ≥ 80% 확인.
- **내 사망 선언 조건**: 커버리지가 ≥80%로 확인되고도 bmaF_full − cur < +0.02
  또는 CI가 0 포함이면 — 참이 후보에 있는데도 사후가 못 고르는 것 = 진짜
  모델하 정보 한계(realization-과적합 우세). 그때는 **내가 틀렸고 coordinator의
  벽이 맞다**. CRB 계산으로 종결하고 나는 서명한다.

## 5. 학습의 자리 (재정의 — 원래 브리프의 질문에 대한 수정 답)

"exact evidence가 있으니 학습 불요"는 **후보집합이 주어졌을 때**만 참이다. 이번
kill이 실제로 증명한 것은 **proposal이 병목**이라는 것 — 1-lobe evidence는 교차
바다에서 증명된 나쁜 ranker(참 rank median 30/504)다. 학습의 남은 자리는 γ 회귀도
온도도 아니고 **amortized proposal**(맵 → lobe-쌍 후보 ranking; 라벨 공짜, top-64
전수 평가가 기준선이라 net은 "같은 회수율을 1초에"가 목표)이다. 단 §4가 먼저다:
전수-격자 사후가 못 벌면 proposal을 배울 대상 자체가 없다.
