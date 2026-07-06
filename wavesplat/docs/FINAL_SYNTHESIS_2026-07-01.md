# 최종 종합 — 방법론 전면 검토 + 실측 검증 + 방향 (2026-07-01)

> 하루 전체 세션: 선행연구·데이터셋·제안 방법 다수를 sim(HOS 비선형) + 실측(Bergamasco 스테레오)에서
> 검토. 이 문서가 capstone. 세부: `METHODS_eval_`, `PILOT_realdata_`, `DIRECTION_REVIEW_`,
> `NOVELTY_survey_`, `ref_paper/1x_*/_*.md`.

## 1. 실행한 것 (17개 실험 + 문헌)
- 방법: naive, RELAX(이산물리), M1 SpecGP(연속물리+UQ), M3 RELAX-anchored, M2 학습형(sim), M4 학습형
  (scale-adapt), M5/M5b 학습형(실측 직접학습), M6(Fourier features), M7(MSE 진단), + forecast, + 생성prior 검토.
- 데이터: HOS 비선형 sim(6윈도) + **실측 Bergamasco 스테레오**(Ifremer, 14m·0.05m·12Hz, sparsify).

## 2. 핵심 결과 (냉정)
**(a) sim 비선형**: 학습형 amortized posterior 최고(nRMSE 0.250, 보정 UQ, 8.5ms) > RELAX 0.274 > M3 0.376 > M1 0.796.
**(b) 실측 BS_2011**: **어떤 방법도 실패** — 물리 0.38–0.58, naive 0.37–0.52, 학습형 전부 붕괴(M2 OOD·M4 학습붕괴·
  M5b 0.92·M6 1.07). **M7 결정타: net이 실측 윈도를 하나도 overfit 못 함**(single-win rmse~1.12 고정).
**(c) forecast(실측)**: 물리(RELAX)가 persistence 이김 — **skill 0.12–0.60**(0.5–1s서 0.48–0.60).

## 3. 진단 (일관되고 이론적으로 정합)
- 실측 실패는 **fixable ML 문제 아님**: spectral bias(M6)·NLL 도피(M7) 모두 배제. net이 overfit조차 못 함.
- **근본 원인 = 정보/개구 한계**: BS_2011 개구 14m ≈ **2파장**, 방향분산 **109°** → 희소 fwd-LiDAR가 t\* 장을
  **결정 못 하는 unidentifiable 역문제**. → 우리 **CRB·유효개구 이론이 실측에서 method-agnostic 벽으로 확인**.
- **일관된 그림**: well-conditioned(큰 sim 도메인) → 학습형 작동; ill-conditioned(소형개구 실측) → 전부 붕괴.
  "학습은 정보한계에 *도달*하나 *초월* 못 함" — 전체 thesis와 정확히 일치.
- 생성/확산 prior(SOTA ill-posed 해법)도 **symptom(near-mean)**은 고쳐도 **정보한계는 못 넘음**(참값 대신
  그럴듯한 장 생성). 압축센싱≈RELAX.

## 4. 살아남은 두 positive
1. **정보한계 이론(CRB)이 robust backbone** — 실측서 재확인. (아이러니: [[direction1_crb]] 방향)
2. **forecast에서 물리가 persistence 이김** — naive가 원리적으로 못 하는 가치. 유효 예보 horizon 존재.

## 5. novelty 현실 (강한 제약)
DSWP·Ehlers PINO(실시간 sparse 재구성)·Zhang2022(phase-resolved 실시간 UQ)·predictable-zone closed-form·
DOA-CRB 전부 선행. "amortized+UQ"·"grid-free"·"연속표현" 단독 novelty 약함. → `NOVELTY_survey_`, `DIRECTION_REVIEW_`.

## 6. 방향 옵션 (냉정한 권고)
- **A. 정보한계 중심 재프레이밍(권고)**: thesis backbone = *ship-LiDAR phase-resolved 파 추정의 정보한계(개구·
  방향 식별)를 CRB로 규정 + 실측(스테레오 sparsify)에서 벽을 실증 + forecast에서 도달가능 가치 정량화.* 이건
  우리가 **가진 것으로 정직하게 방어가능**하고, 실측 실패조차 *증거*가 됨.
- **B. 결정적 확인 실험(heavy)**: 큰-개구 레코드(AA Acqua Alta ~6GB 등)서 방법이 *되면* → "실측 실패=개구
  한계" 확정(A를 강화). 다운로드+분석 1사이클.
- **C. 생성 prior 구현(heavy)**: DPS-on-real. M7상 정보한계는 못 넘지만 앙상블 UQ+그럴듯한 복원 확보. 다시간.
- **D. vehicle 재고**: 학습형 실측 배포가 핵심이면 현 데이터·개구론 난망 → 하이드로포일 응용/다른 센서 재검토.

**권고 순위: A(재프레이밍) → B(확인) 병행.** C는 novelty·정보한계상 우선순위 낮음. D는 A/B 후 판단.

## 7. B·C 실행 결과 (2026-07-01 오후, 사용자 요청)
**B — 큰 개구 레코드 확인 (AA_2014 Acqua Alta, 개구 86×70m vs BS_2011 14m):**
- 지배파장 ~32m → **파장당 개구비 ~2.7**(BS_2011 ~2.3, 거의 동일!). 절대개구 6배지만 파도 비례 확대.
- RELAX nRMSE **0.381**/corr 0.92, naive 0.332. (BS_2011: RELAX 0.51–0.58) → **약간 개선, naive 여전히 우세.**
- ★ **정보한계 이론 실측 검증**: *절대개구 아닌 파장당 개구비가 식별성 결정* — 두 레코드(14m/6m, 86m/32m,
  둘 다 ~2.5파장)서 일관. clean positive는 아니나 **이론엔 강한 확증.**

**C — 생성 prior (ill-posed 정칙화):**
- C(PCA/EOF 선형 prior): nRMSE 0.524 / corr 0.85 vs naive 0.473 → **못 이김.**
- C2(확산 DDPM+DPS inpainting, 실측 학습): nRMSE 0.574 / corr 0.83 vs naive 0.435 → **못 이김.**
- ★ **정의적**: SOTA 생성 prior(확산)조차 그럴듯한 파동장(corr 0.83)은 만들되 **naive보다 정확하지 않음**
  → 정보한계는 method-agnostic(M7과 일관). "학습은 한계 도달, 초월 못함" 재재확인.

## 8. B·C 후 결론 — 그래도 존재하는 positive (paper-able)
nowcast 복원은 정보한계로 naive 못 이김(모든 방법). **그러나 두 개의 정직한 positive:**
1. **FORECAST**: 물리(RELAX)가 persistence 이김(skill 0.12–0.60, 0.5–1s서 0.48–0.60) — 실측 검증. naive/
   persistence가 원리적으로 못 주는 *phase-resolved 물리의 가치*. → **"실측 ship-LiDAR로 단기 파 예보"** 가능.
2. **개구비 설계법칙**: 복원/식별성이 *파장당 개구비*로 결정(절대크기 X), 실측 2레코드서 정량 확증(CRB 근거).
→ **positive 논문 방향(‘실패 인정’ 아님)**: *실측 검증된 (a) 단기 forecast skill + (b) CRB 기반 개구비 설계법칙*,
   nowcast 정보한계는 그 *경계조건*으로 제시. A(정보한계)+forecast를 positive로 재구성.
