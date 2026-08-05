# 설계 — kp-robust B-2 (센서-실현 재설계) 2026-07-22

> 사용자 "설계해줘". M-3/M-3G 후속: B-2 우위(+0.139)가 eval-time oracle kp 의존, beamform kp선 +0.022.
> [[project_execution_2026-07-22]]. 규율: MEASURED/ARGUED, 실험 전 사전등록.

## 1. 진단 (코드-정확, MEASURED)
전 파이프라인이 **단일 스칼라 kp로 정규화**됨:
- **입력**: `beamform_power(P,z,DEPTH, KREL*kp_e, NGRID)` — 파워맵 격자 물리범위 = [−2.5kp, 2.5kp]. kp 편향 시
  진짜 피크가 정규화 프레임서 엉뚱한 위치·맵 왜곡.
- **bank(출력)**: `kxb = kxn * kp_e` (`d3_prior_net_taskft.py:56`) — 51×51 정규화격자를 kp로 스케일.
  kp +37% 편향(M-3 최난셀 Tp13)이면 bank가 고파수로 밀림.
- **EM**(`net_gamma_mmse use_em`)은 **γ(prior 분산)만 정련, kp는 고정** ⟹ 편향 kp 미교정.
⟹ **kp 하나가 입력정규화+bank 둘 다 지배** = beamform kp 편향(장파·소개구서 심함)이 파이프라인 전체를 망침.

## 2. ★핵심 미묘점 (설계 방향을 가르는 질문, ARGUED)
M-3: **편향 kp에서 L·N이 회복**(0.06→0.51), T는 소폭 하락 ⟹ T−L 축소. 즉 **B-2 우위는 부분적으로 "정확한 kp가
window-inside 팔의 over-fit을 유발"한 데서 온 아티팩트**일 수 있음. ⟹ **"kp-robust"의 두 목표가 갈림**:
- **(a) T−L 갭 복원** (chapter B 주장): kp-robustness가 갭을 **복원할지 오히려 없앨지 불명** — 갭이 kp-아티팩트면
  복원 불가(억지 금지).
- **(b) 센서-실현 추정 자체 개선** (spine A/제품): 편향 kp 하 **모든 팔** 개선 — 프레이밍 독립, 더 가치있고 정직.
⟹ **권고 = (b)를 1차 목표로.** (a)는 sweep(§4 step0)이 판별.

## 3. 후보 접근 (비용/가치)
| # | 접근 | 무엇 | (a)갭복원 | (b)추정개선 | 비용 |
|---|---|---|---|---|---|
| 1 | **kp-jitter 증강** | 훈련 시 kp_train = kp_e·(1+ε), ε~beamform 오차분포(±30%) → 스케일-robust prior 학습 | ? | 중 | 저(재훈련) |
| 2 | **kp-EM 정련** | 초기 beamform kp → 성분적합 → 지배파수 재추정 반복(현 EM에 kp축 추가) | ? | 중~고 | 저(무재훈련) |
| 3 | **multi-kp 앙상블** | kp×{0.7,1,1.4} 병렬추론 후 held-out 적합으로 선택/평균 | 낮 | 중 | 중(3× 추론) |
| 4 | **절대 wide 격자** | kp-정규화 제거, 고정 절대 k-격자(해상범위 커버) | ? | ? | 고(재설계·일반화 손실 위험) |
| 5 | **kp-불확실성 인지 bank** | beamform 신뢰도 낮을 때 넓은/soft prior(불확실성 전파) | 낮 | 중 | 중 |
- **1순위 = #2 kp-EM**(무재훈련·즉시·물리적으로 정론: 성분적합이 beamform보다 kp 잘 잡을 수 있음, 특히 장파).
  단 under-det 장파선 적합 자체가 편향돼 수렴 안 할 위험(ARGUED).
- **2순위 = #1 kp-jitter 재훈련**(갭복원 여부 결정적 시험, (a)·(b) 동시).

## 4. 실험 순서 (사전등록 골격)
- **Step 0 kp-민감도 sweep** (무재훈련): 실패창 T−L을 kp=m·참피크서 측정. `tmp/kpsens_sweep.py`. **⚠ 1 sea만
  돌고 중단**(net_gamma_mmse per-call 비용 5×mult로 ~60min/full, 아래 (a)-path 우선순위 강등에 따라 보류).
  **1-sea 프리뷰(Tp9_Hs4, 과소해석 금지)**: m=1.0(참) T−L −0.009 / m=0.7 +0.147 / m=1.3 +0.029 — **참-kp서 갭≈0,
  이 milder sea(hand .414)는 갭 없음** ⟹ 집계 +0.139는 **최난셀 소수 지배** 시사(1 sea라 ARGUED). full 11-sea는
  (a)-path 재개 시에만.
- **Step 1 kp-EM 정련**(무재훈련·값쌈): use_em에 kp-재추정 추가, M-3 아레나 재평가. **(b) 검정**: 장파서 kp 정확도↑·
  전팔 skill↑? 최난셀 Tp13 kp 0.033→0.024 근접?
- **Step 2 kp-jitter 재훈련**(Step0가 gradual일 때만): T/L/N 재훈련(±30% kp jitter), beamform-eval. **(a) 결정**:
  T−L 회복(+0.08↑)이면 갭 salvageable, 유지(+0.02)면 갭=kp-아티팩트 확정.
- 각 단계 사전등록·이중검증(fable). **억지 금지**: (a) 복원이 안 되면 "B-2는 센서-현실서 marginal"을 정직 채택.

## 5. 정직한 메타 (fable 방법론 리뷰와 접점)
- kp 취약성은 **"물리-앵커 파이프라인의 가장 약한 단(2026-07-05 current-arc verdict: beamform-argmax kp가 최취약)"**
  의 재확인. 근본 회피 = **kp에 안 걸리는 표현**(fable 리뷰 대상: attention set-encoder·neural operator·kp-free
  파라미터화). ⟹ 본 설계(#1~5)는 **기존 파이프라인 내 패치**, fable 리뷰는 **파이프라인 교체** 여부 — 상보.
- **⚠ 스코프 정직**: 이 모든 것은 chapter B(loss-swap) 구제 시도. **spine A(envelope)는 무관·이미 강함(C8).**
  B가 salvage 안 돼도 thesis 손상 없음 — 오히려 "센서-현실 한계"의 정직한 기록.
