# WaveSplat & Hydrofoil 논문 — 개선 및 추가 작업 항목

> 4인 페르소나 토론 (Prof.K / Prof.E / Rev.2 / Saeyong) 2차 검토 결과 반영.
> 2026-04-07 초판 → 2026-04-07 2차 업데이트.

---

## 0. 이번 리뷰에서 수정 완료된 항목

| # | 문제 | 수정 내용 |
|---|------|----------|
| 1 | WaveSplat Abstract: "N=1 for simple swells" 과장 (BF2에서 N_a=11) | "N=2 for two-component swells, N=18-21 for complex seas" + failure mode 언급 |
| 2 | bjerkaas2022 bib key 오류 (실제 저자 Collins et al.) | → collins2022로 수정 (양쪽 논문 모두) |
| 3 | Hydrofoil Abstract over-claim: "93-103% recovery" 일반화 | BF3/BF4 구분 + "37% in BF4 due to directional misestimation" 명시 |
| 4 | Hydrofoil comfort metric 불일치 | "BF3 beam sea" 명시 + a_RMS < 0.315 m/s² 조건 |
| 5 | perez2005 → perez2012 bib year 오류 | 수정 완료 |
| 6 | Ehlers+Klein FNO (2024), Desmars JFM (2023) 미인용 | Related Work에 추가 + 차별화 논의 |
| 7 | Ablation A1 해석 부족 (Rev.2 지적) | w_phi tracking의 ω 고정 메커니즘 상세 설명 |

---

## 1. 수치 검증 (논문 제출 전 필수)

| 항목 | 현재 논문 수치 | JSON 데이터 | 확인 필요 |
|------|---------------|-------------|-----------|
| WS avg nRMSE | Abstract: 0.211, 통계검증: 0.230±0.084 | p1_comparison.json | 두 수치 관계 명확화 (12ep 단일 seed vs 50 paired runs) |
| GPR 비교 | "23.5% worse" (§5.1) | gpr_comparison.json | 비교 기준 통일 필요: (0.276-0.233)/0.233=18.5% vs /0.276=15.6% |
| ETKF-fine | Discussion: "0.421" | etkf_comparison.json | 정확한 평균값 재확인 |
| Model mismatch | "< 0.25%" (§3.1) | ablation A1 결과 | 이론적 0.25% vs 관측 ~2% 차이 설명 보강 필요 |

---

## 2. 추가 가능한 Figure/Table (높은 임팩트)

### 2-1. 수렴 곡선 Figure ★★★
- **데이터**: `extended_analysis.json` → `convergence` (frame-by-frame nRMSE)
- **내용**: 시간축 nRMSE 곡선 (BF3-2comp, BF4-15comp, BF5-30comp)
- **가치**: 논문에 수렴 figure 완전히 부재 — 핵심 시각화

### 2-2. Near/Far-field RMSE 분리 ★★
- **데이터**: `etkf_comparison.json` → `avg_RMSE_far`
- **내용**: RMSE_near (10-80m) vs RMSE_far (80-250m) 비교
- **가치**: LiDAR 거리별 성능 — WaveSplat far-field 우수성 확인

### 2-3. 100 Episode Boxplot ★★
- **필요**: 전체 100 episode 재실행 or multi-seed 결과
- **가치**: 12ep 대표 선정 selection bias 해소

### 2-4. Hs 추정 정확도 ★
- **데이터**: `ablation_seeded.json` → `avg_Hs_error`
- **가치**: Ocean Engineering 리뷰어에게 Hs는 핵심 실용 지표

---

## 3. 추가 분석 (중간 임팩트)

### 3-1. BF4-5 Ablation ★★★ (최우선)
- **현재**: BF3 6ep에서만 수행 → A1이 full보다 좋은 결과
- **필요**: BF4, BF5 episode에서 dispersion 제거 ablation 재실행
- **가치**: dispersion constraint 논증의 **결정적 근거**
- **예상**: BF4-5에서 spectral crowding → unconstrained ω 추정 실패 → nRMSE 역전

### 3-2. T_converge 통계
- **데이터**: p1_comparison.json (일부 null)
- **가치**: "within first 10% frames" 주장의 정량적 근거

### 3-3. ETKF-fine vs ETKF-medium 정식 비교 테이블
- **데이터**: etkf_comparison.json에 둘 다 존재
- **가치**: "grid resolution이 아닌 ensemble 방법론 한계" 논증 — Discussion에 이미 언급했으나 테이블 없음

### 3-4. Sliding-window GPR Baseline
- **필요**: 새 코드 + 전체 episode
- **가치**: "GPR baseline 불공정" 리뷰어 방어
- **구현**: 3-frame window Mat​é​rn GPR, temporal kernel은 dispersion 기반

### 3-5. Bias vs Variance 분리
- **현재**: nRMSE만 보고 (point estimate)
- **필요**: multi-seed 결과에서 systematic bias vs random variance 분리
- **가치**: 추정 오류의 특성 이해 (temporal correlation, spectral leakage)

---

## 4. Hydrofoil 논문 추가 분석

### 4-1. 더 많은 E2E Episode ★★★
- **현재**: 3개 (BF3-2comp, BF3-5comp, BF4-5comp)
- **필요**: BF3-15comp, BF4-15comp, BF4-30comp
- **가치**: WS/GT=37% (BF4)가 outlier vs systematic 판별

### 4-2. Direction Error 정량화 ★★
- **현재**: fig_component_accuracy 정성적
- **필요**: episode별 k_x sign error rate 계산
- **가치**: "directional > amplitude accuracy" 정량적 근거

### 4-3. LP Filter Sensitivity Table ★★
- **현재**: α=0.05 경험적 선택, α=0.15만 언급
- **필요**: α ∈ {0.01, 0.03, 0.05, 0.10, 0.15, 0.20} × 4 scenarios
- **가치**: filter 설계의 정량적 정당화 (run_sensitivity.py 확장)

### 4-4. Comfort Metric 상세 테이블
- **현재**: ISO 2631 등급만 언급
- **필요**: 각 scenario별 PD-only vs PD+FF의 a_RMS 절대값 + ISO 등급
- **가치**: "Slight discomfort → Comfortable" 주장의 정확한 scenario 매핑

### 4-5. 60-120s 장기 시뮬레이션
- **필요**: WaveSplat 긴 episode + hydrofoil sim 연장
- **가치**: ISO 2631 평가 신뢰도, tracking stability 검증

### 4-6. Actuator Dynamics 모델
- **필요**: 1st-order lag (τ=20-50ms) 추가
- **가치**: "ideal actuator" 가정의 robustness 확인

---

## 5. 선행 연구 추가 검토 항목

### 이번에 추가한 references
- Ehlers, Klein, Desmars et al. (2024) — FNO/U-Net phase-resolved reconstruction (Ocean Eng.)
- Desmars, Hartmann et al. (2023) — Nonlinear deterministic wave reconstruction (JFM)
- Qi et al. (2018) — Nonlinear phase-resolved reconstruction (JFM)

### 추가 검토 필요한 논문들 (검색에서 발견)
| 논문 | 관련성 | 조치 |
|------|--------|------|
| Ehlers et al. "Uncertainty-Guided Prediction Horizon" (2025, JMSE) | phase-resolved prediction의 prediction horizon 한계 | WaveSplat discussion에서 인용 가능 |
| Physics-Informed Neural Operators for wavefield reconstruction (arXiv 2508.03315, 2025) | PINO 접근 — training-free physics constraint | WaveSplat과 직접 비교 대상 |
| Qi et al. "Phase-resolved wave prediction with linear wave theory and PINNs" (2023, Applied Energy) | linear wave + PINN hybrid | WaveSplat의 pure physics 접근과 대비 |
| Monohull seakeeping with/without hydrofoil (2025, JMSE 13(10)) | CFD hydrofoil 효과 16-33% motion reduction | Hydrofoil intro에 인용 가능 |
| Navier 30 electric hydrofoil (NewAtlas, 2024) | 상용 전기 hydrofoil 사례 | Hydrofoil intro에 citation (non-academic) |

---

## 6. 논문 구조적 약점 (4인 토론 합의)

### WaveSplat
| 약점 | 심각도 | 해결 방안 |
|------|--------|----------|
| Ablation A1: dispersion 제거가 더 좋은 결과 | 높음 | BF4-5 ablation으로 역전 보여야 (§3-1) |
| 실데이터 검증 부재 | 높음 | limitation 인정 (완료) + 수조 실험 계획 언급 (완료) |
| GPR baseline 약함 | 중간 | sliding-window GPR 추가 (§3-4) or limitation 강화 (완료) |
| Uncertainty quantification 부재 | 낮음 | future work 언급 (완료) |

### Hydrofoil
| 약점 | 심각도 | 해결 방안 |
|------|--------|----------|
| BF4 E2E 37% degradation | 높음 | abstract 수정 (완료) + direction error 정량화 (§4-2) |
| Comfort metric scenario 불명확 | 중간 | 수정 완료 — BF3 beam sea 명시 |
| LP filter 경험적 tuning | 중간 | sensitivity table 추가 (§4-3) |
| 30s 짧은 시뮬레이션 | 중간 | 60-120s 확장 (§4-5) |
| 2-DOF 모델 단순화 | 낮음 | 물리적 정당화 완료 + 3-DOF future work |

---

## 7. 우선순위 (추천 실행 순서)

### 즉시 가능 (JSON 데이터만으로)
1. **수치 검증** — GPR 비교%, ETKF-fine 값, nRMSE 기준 통일
2. **수렴 곡선 Figure** — convergence JSON에서 생성
3. **Near/far RMSE 테이블** — etkf_comparison JSON에서 추출
4. **Comfort metric 상세 테이블** — hydrofoil 결과에서 추출

### 데이터 확보 후
5. **BF4-5 Ablation** — dispersion 논증의 핵심 (최우선)
6. **LP Filter Sensitivity Table** — run_sensitivity.py 확장
7. **E2E episode 추가** — hydrofoil 통계 강화
8. **Direction error 정량화** — k_x sign error rate

### 코드 개발 필요
9. **Sliding-window GPR** — temporal GPR baseline
10. **장기 시뮬레이션** — 60-120s episodes
11. **Actuator dynamics** — 1st-order lag 모델

---

*Updated: 2026-04-07 (2차 리뷰)*
*4인 페르소나: Prof.K(novelty/선행연구), Prof.E(수치정확도/comfort metric), Rev.2(ablation/baseline공정성), Saeyong(방어/구현)*
