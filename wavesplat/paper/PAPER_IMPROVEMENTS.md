# WaveSplat & Hydrofoil 논문 — 데이터 기반 개선 가능 항목

> 전체 100 episode 데이터셋이 확보되면 검토/추가할 수 있는 분석들.
> 현재 3개 episode + 7개 result JSON으로는 figure 재생성 및 새 분석 실행 불가.

---

## 1. 수치 검증 (논문 정확도)

| 항목 | 현재 논문 수치 | JSON 데이터 | 확인 필요 |
|------|---------------|-------------|-----------|
| WS avg nRMSE | Abstract: 0.211, 통계검증: 0.230±0.084 | p1_comparison.json 기준 | 두 수치의 관계 명확화 (12ep 단일 seed vs 50 paired runs) |
| GPR 비교 | "23.5% worse" | gpr_comparison.json | (0.276-0.233)/0.233=18.5% vs (0.276-0.233)/0.276=15.6% — 기준 통일 필요 |
| ETKF-fine 결과 | Discussion에서 0.421 언급 | etkf_comparison.json에 ETKF-fine 존재 | 정확한 값 재확인 |

---

## 2. 추가 가능한 Figure/Table (높은 임팩트)

### 2-1. 수렴 곡선 Figure
- **데이터**: `extended_analysis.json` → `convergence` 섹션에 frame-by-frame nRMSE 존재
- **내용**: 시간축 nRMSE 수렴 곡선 (BF3-2comp, BF4-15comp, BF5-30comp)
- **가치**: 현재 논문에 수렴 figure 없음 — "converges within first 10% of frames" 주장의 근거

### 2-2. Near-field vs Far-field RMSE 분리
- **데이터**: `etkf_comparison.json`에 `avg_RMSE_far` 키 존재
- **내용**: RMSE_near (10-80m) vs RMSE_far (80-250m) 비교 테이블
- **가치**: LiDAR 거리별 성능 차이 — WaveSplat이 far-field에서도 우수한지 확인

### 2-3. 100 Episode 분포 (Boxplot/Violin)
- **필요**: 전체 100 episode 재실행 또는 multi-seed 결과
- **내용**: episode별 nRMSE 분포를 boxplot으로 시각화
- **가치**: 12ep 대표 선정의 selection bias 해소

### 2-4. Hs 추정 정확도
- **데이터**: `ablation_seeded.json`에 `avg_Hs_error` 존재
- **내용**: Hs_error 테이블 추가 (secondary metric)
- **가치**: 해양공학 리뷰어에게 Hs 정확도는 중요한 실용적 지표

---

## 3. 추가 가능한 분석 (중간 임팩트)

### 3-1. BF4-5 Ablation (Dispersion Constraint)
- **현재**: Ablation은 BF3 6ep에서만 수행
- **필요**: BF4, BF5 episode에서 A1 (dispersion 제거) 재실행
- **가치**: dispersion constraint가 고해상도에서 필수적이라는 주장의 **결정적 근거**
- **예상**: BF4-5에서 A1 제거 시 tracking 불안정 → nRMSE 역전

### 3-2. T_converge 통계 테이블
- **데이터**: p1_comparison.json에 T_converge 값들 존재 (일부 null)
- **내용**: episode별 수렴 시간 통계 (mean, std, null 비율)
- **가치**: "within first 10% frames" 주장의 정량적 뒷받침

### 3-3. ETKF-fine vs ETKF-medium 비교
- **데이터**: etkf_comparison.json에 둘 다 존재
- **내용**: grid resolution 증가가 ETKF 성능에 미치는 영향
- **가치**: "grid resolution이 문제가 아닌 ensemble 방법론 자체의 한계" 논증

### 3-4. Sliding-window GPR Baseline
- **필요**: 새 코드 작성 + 전체 episode 재실행
- **내용**: temporal GPR (3-frame window) vs single-frame GPR
- **가치**: "GPR baseline이 불공정하게 약하다"는 리뷰어 지적 방어

---

## 4. Hydrofoil 논문 추가 분석

### 4-1. 더 많은 E2E Episode
- **현재**: 3개 episode (BF3-2comp, BF3-5comp, BF4-5comp)
- **필요**: BF3-15comp, BF4-15comp, BF4-30comp 등
- **가치**: WS/GT 통계적 분포 — 37% (BF4)가 outlier인지 systematic인지

### 4-2. Direction Error 정량화
- **현재**: fig_component_accuracy에서 정성적 표시
- **필요**: k_x sign error rate를 episode별로 계산
- **가치**: "directional accuracy > amplitude accuracy" 주장의 정량적 근거

### 4-3. 60-120s 장기 시뮬레이션
- **현재**: 30s episodes
- **필요**: WaveSplat 긴 episode + hydrofoil sim 연장
- **가치**: 통계적 안정성, ISO 2631 평가의 신뢰도

### 4-4. Actuator Dynamics 모델
- **현재**: ideal actuator (instantaneous)
- **필요**: 1st-order lag ($\tau$ = 20-50ms) 추가
- **가치**: 실용성 논증 — 실제 actuator bandwidth에서 성능 유지 확인

---

## 5. 우선순위 (추천 실행 순서)

1. **수치 검증** — 논문 제출 전 필수, JSON 데이터만으로 가능
2. **수렴 곡선 Figure** — JSON에 데이터 있고, 논문에 빠진 핵심 시각화
3. **Near/far RMSE** — JSON에 데이터 있고, 공간 분석 보강
4. **BF4-5 Ablation** — 데이터+torch 필요, dispersion 논증의 핵심
5. **E2E episode 추가** — 데이터 필요, hydrofoil 논문 통계 강화
6. **장기 시뮬레이션** — 데이터 생성 + 재실행, 통계적 신뢰도
7. **Sliding-window GPR** — 새 코드 필요, 리뷰어 방어

---

*Generated: 2026-04-07*
*Context: 4인 페르소나 토론 (Prof.K / Prof.E / Rev.2 / Saeyong) 결과 반영*
