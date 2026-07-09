# ref_paper 추가본 — 2026-07-01 (놓쳤던 클러스터 보강)

> 기존 lit review(2026-04)에서 빠졌던 영역을 검색·정독·저장. 두 핵심 family를 구분해야 함.

## A. 위상분해 장 vs 통계/스펙트럼 — family 구분 (중요)
우리는 **(II) 위상분해 결정론적 장 복원/예측(DSWP)** family. 아래 (I)은 *adjacent*(인용해 구분만).
- **(I) 통계/스펙트럼 추정 — "wave buoy analogy"(선박운동→방향스펙트럼)**: Nielsen 계열, FPSO ANN/TBE,
  adversarial nets(2022), B-spline 방향스펙트럼(OE 2024), Kalman 실시간(Sci.Rep. 2023), 정상선박운동(2004).
  → *방향 스펙트럼*(통계)만 줌, 위상분해 장 아님. 기존 review S1(Nielsen 2008)만 있었음 → **이 분야 전체를
  인용해 "우리는 phase-resolved field, 저들은 statistical spectrum"으로 구분 필요.** (paywall, 서지로만 기록)
- **(II) 우리 family**: 이미 정리됨(Nouguier·Ehlers·Wang&Pan·Qi·Desmars·Fisher·Zhang2022 등).

## B. sparse→field 재구성 ML 계보 (우리 amortized posterior의 직접 사촌 — 과밀)
저장(`10_sparse2field_ML/`):
- **arxiv_2411.12948** — Attention 기반 sparse tsunameter→full-field 쓰나미파 복원 (2024).
- **arxiv_2603.13077** — Rooftop wind sparse 센서 복원, **결정론→생성(UQ)** 비교 (2025).
- **arxiv_2302.09748** — NAS 앙상블로 flow 재구성·예보 **UQ** (2023).
- (기보유) Senseiver(NeurIPS22, 실측 SST SOTA; `01_NP_variants/`,`06_DA/`), ConvCNP/TNP/LBANP, FieldFormer.
- 추가 비저장(서지): flow-field autoencoder+cross-attention(PoF 2025), differentiable sensor placement(2312.09176).
→ **결론: "sparse 관측→전장 + attention + UQ/생성"은 확립·과밀 ML 계보. 우리 방법은 그 일원 → ML novelty ≈ 0
재확인.** 차별은 오직 *응용 conjunction*(phase-resolved ocean + 분산물리 + 희소 oblique ship-LiDAR + 보정UQ + 실측).

## C. 센서-운동 (우리 niche 직접 관련)
- 부유 LiDAR **운동보정** deterministic (WES 10:2791, 2025) — offshore wind이지만 "이동 플랫폼 LiDAR 보정"
  방법론. 우리 niche(선박운동+LiDAR)의 참고. (서지)
- 선박탑재 X-band radar에 선박운동+강수 통합(OE 2025).

## D. 유용한 framing (문헌이 말하는 미해결 병목)
- **DA 동시 파+선박(2504.13943)**: *"실시간 phase-resolved 예측의 핵심 미해결 = 희소/간접 측정에서 초기조건
  복원(IC reconstruction)"*. → 우리 work이 정확히 이 병목을 침. (positioning 문구로 활용)

## 종합 함의
- ML 방법 novelty 낮음(B), 통계-스펙트럼 분야와 구분 필요(A), 우리 niche의 센서-운동 참고 확보(C),
  positioning 문구 확보(D). **결론 불변**: 방어선 = 희소 oblique ship-LiDAR 위상분해 + 보정UQ + 실측 검증.
