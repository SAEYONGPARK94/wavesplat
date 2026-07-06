# 개선 방향 연구 검토 (2026-07-02)

> 현황: grid-spectral 자기지도 = 학습됨(label-free)이나 field corr **0.490 최종 plateau** (RELAX 0.947 미달, 그리드 한계).
> novelty: "label-free physics-informed 파랑재구성"은 PINO(2025)·Ehlers(2023)·PINN(2025)에 선점.
> → 기술적 개선(성능 gap) + 전략적 개선(novelty 재positioning) 둘 다 검토.

## A. 기술적 개선 — 0.47→RELAX gap 좁히기
우리 문제 = **dispersion-구속 3D(공간k+시간ω) sparse spectral 추정** = DOA/array-processing과 수학적 동형.
그쪽 툴킷이 잘 정립(빌려올 수 있음):
- **A1. Deep-unrolled gridless sparse recovery** ★ — RELAX/ISTA 반복을 trainable 층으로 언롤 + off-grid refine
  층. DOA서 검증(적은 층 수렴, grid mismatch 해소, 속도↑). → 그리드 한계(0.47 plateau) 돌파 + RELAX보다 빠름.
  (Deep Unfolded Gridless DOA via Atomic Norm; unsupervised gridless DOA.)
- **A2. Coarse-to-fine / peak refine** — net 그리드 스펙트럼 → peak 추출 → 미분가능 연속 refine (= neural-guided
  RELAX를 자기지도로). 이미 supervised 버전 보유(component_distill), self-sup로 전환.
- **A3. Atomic-norm 연속 sparse** — 처음부터 gridless (grid mismatch 원천 제거).

## B. Novelty 재positioning — PINO/Ehlers/DOA-unrolling이 안 하는 것
ML 기법 자체(unrolling·gridless·self-sup)는 기존 → **기법이 아니라 아래가 진짜 기여여야:**
- **B1. Dispersion-COUPLED 구조** — 파랑은 ω=√(gk·tanh) 로 공간-시간 결합(DOA엔 없음). 파랑 특이 구조. (약~중)
- **B2. ★정보한계 결합** — 학습/언롤이 **언제 도움되고 언제 개구-한계 벽에 막히나**(CRB·predictable-zone).
  "학습은 정보한계를 saturate하나 벽 너머는 어떤 방법도 불가." PINO/Ehlers/DOA-unrolling 아무도 이 프레임 없음.
  **최강 차별점** (내 CRB 백본 commit b7a6c2e, [[direction1_crb]]).
- **B3. 보정 UQ** — RELAX·PINO는 점추정. per-추정 calibrated UQ가 gap. (접근 2 RELAX-front UQ.)
- **B4. Forecast operability** — 실측 operability 법칙(개구비≥0.7λ, skill 0.7~0.9). 배치 관점 기여.

## 권장 (냉정)
기술 툴킷(A1)은 빌려올 수 있어 성능 gap은 좁혀지나 **그 자체론 novelty 강화 안 됨**(기존 기법).
→ 논문 기여의 축을 **"새 ML 방법"이 아니라 B2(정보한계)+B4(실측 operability)+B3(UQ)** 로:
  **"학습된 amortized 추정기가 정보한계를 어디까지 saturate하고 어디서 개구 벽에 막히나 — 그리고 배치가능
  forecast operability"** = vehicle는 deep-unrolled self-sup amortized RELAX, 기여는 이론(한계)+응용(operability).
이게 PINO/Ehlers(방법 논문)와 겹치지 않는 정직한 각도. [[project_direction_2026-07]] P1(센서niche+Bergamasco)과 정합.

## A1 실증 결과 (2026-07-02) — 자기지도 neural-guided RELAX
grid net(자기지도) peak 추출 → physics refine(obs-fit, 라벨없음). **refine이 그리드 한계 크게 돌파.**
| NC/MO | grid-only | A1-v1 | A1-v2(촘촘/혼합) | RELAX |
|---|---|---|---|---|
| 4/600 | 0.49 | 0.76 | **0.80** | 0.99 |
| 6/600 | 0.43 | 0.84 | **0.89** | 0.90 |
| 6/300 | 0.43 | 0.71 | **0.82** | 0.98 |
| 8/300 | 0.41 | 0.73 | **0.57** | 0.93 |
| 12/300 | 0.50 | 0.58 | **0.59** | 0.75 |
| 12/150 | 0.49 | 0.53 | **0.46** | 0.67 |
| **평균** | 0.46 | 0.69 | **0.69** | **0.87** |
- **판정**: A1은 6-comp서 RELAX 근접(0.89 vs 0.90)이나 **8+/sparse 어려운 레짐서 덜 robust** → **자기지도 학습기가
  RELAX를 robust하게 못 넘음** (평균 0.69 < 0.87). refine이 detection 병목을 못 메움(RELAX adaptive greedy가 우세).
- **결정 규칙 적용**(못 넘음): **RELAX = 추정 본체 확정**(실측 0.6~0.83 검증). 학습 기여는 **접근2(UQ/forecast)+B2(정보한계 이론)** 로.
- 남은 학습 레버 = **deep-unrolled RELAX**(greedy를 학습층으로)지만 novelty 선점+ROI 불확실 → 우선순위 낮음.
