# 2026-07-01 추가 선행연구 — 기존 lit review에서 누락된 핵심 (UQ·경쟁자)

> 기존 `docs/WaveSplat_literature_review.md`(2026-04)는 탄탄하나 **UQ 클러스터가 통째로 비어 있었음.**
> 아래는 그 공백을 메우는 추가본. PDF 받은 것은 이 폴더에 저장, 못 받은 것은 URL만 기록.

## ★ 가장 중요 (UQ 직격 — 기존 review에 없음)
- **Zhang, Belmont et al. (2022) "Phase-resolved real-time ocean wave prediction with quantified
  uncertainty based on variational Bayesian machine learning", *Applied Energy* 324.**
  → phase-resolved **실시간 예측 + 불확실성 정량화(예보값·예보영역 모두)**, **predictable zone
  최대 74.6% 확장**, 선형파이론 대비 −55.4%·결정론 ML 대비 −11.7%. **우리 "빠른 posterior + 보정 UQ"
  novelty를 정면 점유.** PDF 저장: `zhang2022_varBayes_UQ_appliedenergy.pdf` (CC-BY, Warwick WRAP).
  (자동 텍스트추출 실패 → 팀이 정독 필요. 확인할 것: 센서 1D/2D·calibration 방식·실측 여부.)

## 추가 경쟁자/관련
- **DA 동시 파+선박운동 예보 (arXiv 2504.13943, 2025)** — PDF 저장 `da_wave_ship_2504.13943.pdf`.
  자료동화로 phase-resolved 파 + 선박운동 동시 예보. (제어 응용과 가장 근접.)
- **S-band coherent radar deterministic reconstruction+prediction, condition-number regularized LS
  (Remote Sensing 16(22):4147, 2024)** — MDPI open. 다운로드 차단(Cloudflare) → URL만:
  https://www.mdpi.com/2072-4292/16/22/4147 . coherent 레이더로 space-time 속도 → phase-resolved 계수.
- **"Real-time phase-resolved ocean wave prediction in directional wave fields: enhanced algorithm
  and experimental validation"** (RG 369475098) — **2D 방향장 + 실험검증**까지 존재. → 우리 "directional"
  단독 novelty 약화. (저자/출처 정독으로 확정 필요.)
- **Hessner et al. (2020) "Wave-by-Wave Forecasting via Assimilation of Marine Radar Data", JTECH** —
  기존 review R1과 동일(이미 인지).

## 기존 review가 이미 커버 (재확인 — 팀이 알고 있었음)
Nouguier 2014(flash-LiDAR, ship, 직접 precursor=W1/L1), Jiang&Li 2024(ETKF optical, primary baseline=E3),
Ehlers 2024 U-Net/FNO(M1)·Ehlers 2025 PINO(M2, =arXiv 2508.03315, `07_Wave_competitors/ehlers_2025b_pino.pdf`),
Wang&Pan EnKF 2021/2022(E1/E2), Qi 2023 JFM(W3)·Qi 2025 multi-scale FNO radar(R3), Fisher 2021 sparse
buoy predictable zone(W4), Xue 2025 LiDAR point cloud(L4), Ma 2023 airborne LiDAR directional(L5).
→ **결론: 내가 앞 세션에 "발견"이라 한 PINO/Ehlers는 신규가 아니었음. 진짜 신규 = Zhang2022 UQ.**
