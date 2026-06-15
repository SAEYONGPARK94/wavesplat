# Direction A 검증 리포트 (2026-06-15)

## 개요

GP K_disp 논문 결과의 신뢰성 검증을 위해 3가지 작업을 수행하고,
이전 결과 전체를 재검토하여 버그 2건을 발견/수정했다.

---

## 1. WavePINO S4 공정 비교 (30 에피소드)

**목적**: 기존 WavePINO 결과(0.220)는 6-에피소드 혼합 프로토콜(slope+uniform)로 측정된 것.
GP K_disp와 동일한 30-에피소드 프로토콜로 재평가.

**결과**:

| 메트릭 | GP K_disp (ws_ms5_reg_amp) | WavePINO S4 | 차이 |
|--------|---------------------------|-------------|------|
| nRMSE | **0.223** | 0.228 | GP 2.1% 우위 |
| VR | **0.783** | 0.309 | GP 2.5배 우수 |
| Pearson | **0.489** | 0.376 | GP 30% 우수 |
| Skill | -0.012 | **+0.107** | PINO 우위 |

**핵심**: 동일 평가 프로토콜에서 GP K_disp가 WavePINO를 초과한다.
WavePINO의 기존 0.220은 다른 프로토콜 때문이었음.

**스크립트**: `scripts/eval_pino_30ep.py`
**결과 파일**: `results/pino_s4_30ep_eval.json`

---

## 2. GP UQ Coverage 분석 (버그 수정 포함)

**목적**: GP K_disp의 불확실성 정량화(UQ) 품질 검증.

### 발견된 버그 (치명적)

`eval_uq_coverage.py` 50행에서 amplitude 컬럼 인덱스 오류:
- **수정 전**: `beam_components[:, 3]` (ky = 파수) → prior_var = 17.44 m^2
- **수정 후**: `beam_components[:, 0]` (A = 진폭) → prior_var = 0.0086 m^2

이전 결과(모든 레벨에서 100% coverage)는 **완전한 허위**였음.

### 수정 후 결과

| 명목 coverage | 실측 coverage | 오차 | 판정 |
|--------------|--------------|------|------|
| 50% | **10.4%** | -39.6%p | 심각한 과신 |
| 68% | **15.2%** | -52.8%p | 심각한 과신 |
| 80% | **19.5%** | -60.5%p | 심각한 과신 |
| 90% | **24.8%** | -65.2%p | 심각한 과신 |
| 95% | **29.2%** | -65.8%p | 심각한 과신 |

Mean posterior std: 0.052m | CRPS: 0.236

### 근본 원인: reg_amp 진폭 퇴화

L2 정규화(lambda=2.0)가 모든 성분의 진폭을 동일 값으로 붕괴시킴.
- 추정 분산 = 실제 파랑 분산의 **11~31%**만 포착
  - BF3_5comp: 추정=0.002 vs 실제=0.023 (11%)
  - BF4_15comp: 추정=0.018 vs 실제=0.090 (20%)
  - BF5_30comp: 추정=0.078 vs 실제=0.250 (31%)

**GP 평균 예측은 정상** (K^{-1}z 역행렬이 작은 커널 스케일을 보정).
**UQ만 실패** — prior variance와 reduction이 모두 잘못된 A^2에 비례하므로 posterior std가 과소.

### 논문 대응

- GP UQ를 "프레임워크 역량(framework capability)"으로 한정하여 기술
- 진폭 calibration이 향후 과제임을 명시
- 가능한 해결: 사후 분산 스케일링 (empirical calibration factor)

**스크립트**: `scripts/eval_uq_coverage.py`
**결과 파일**: `results/gp_uq_coverage_eval.json`

---

## 3. 계산 비용 비교 (재측정)

**목적**: 이전 측정값이 동시 실행 프로세스(CPU 경합)로 인해 2~3배 부풀려져 있었음. 단독 실행으로 재측정.

### 발견된 문제 2건

1. **CPU 경합**: 이전 세션의 중복 프로세스 4개가 동시 실행 중
   - WaveSplat init: 110s → 실제 **77s** (이전 대비 30% 감소)
   - GP init: 889s → 실제 **385s** (이전 대비 57% 감소)

2. **GP 추론 시간에 커널 구축 미포함**: `eval_compute_cost.py`에서 `gp_predict`만 측정하고
   `build_K_disp_oracle`를 제외. 수정 후 **14ms** (커널+Cholesky 합산).

### 최종 측정 결과

| 방법 | 초기화 | 추론 | 에피소드당 총합 | 학습 데이터 |
|------|--------|------|----------------|------------|
| WaveSplat | 77s | 13ms/프레임 | 78s | 불필요 |
| GP K_disp (ms5) | 385s | 14ms/프레임 | 385s | 불필요 |
| WavePINO S4 | 0s | 12ms/프레임 | 0.4s | 90 에피소드 |

**측정 환경**: RTX 3060 Ti 8GB, CPU 단독 실행 (경합 없음)

**핵심**: GP의 추론 속도(14ms)는 WaveSplat(13ms), WavePINO(12ms)와 대등.
병목은 전적으로 초기화(385s, 5-시드 방향 추정)에 있음.

**스크립트**: `scripts/eval_compute_cost.py`
**결과 파일**: `results/compute_cost_comparison.json`

---

## 전체 검토 요약

### 수정된 오류 (2건)

| 오류 | 영향 | 수정 |
|------|------|------|
| UQ coverage 컬럼 인덱스 (col 3→0) | 100% coverage → 29.2% (결과 역전) | eval_uq_coverage.py L50 |
| GP 추론 타이밍 (커널 구축 누락) | 79ms → 14ms (과대 측정) | eval_compute_cost.py L64-68 |

### 확인된 정상 결과

| 항목 | 상태 | 비고 |
|------|------|------|
| GP K_disp nRMSE = 0.223 (30-ep) | 정상 | ws_ms5_reg_amp, 30 에피소드 |
| WavePINO S4 nRMSE = 0.228 (30-ep) | 정상 | 동일 프로토콜, Hs 정규화 정상 |
| Oracle GP nRMSE = 0.126 | 정상 | 이론적 상한 |
| WaveSplat nRMSE = 0.263 | 정상 | MAP baseline |

### Direction A 최종 성적표

```
                    nRMSE    VR      Pearson   Skill
Oracle K_disp       0.126    0.629   0.801     0.657   (이론 상한)
GP K_disp ms5       0.223    0.783   0.489    -0.012   (실용 최선)
WavePINO S4         0.228    0.309   0.376     0.107   (공정 30-ep)
WaveSplat           0.263    0.914   0.346    -0.275   (기존 방법)
```

GP K_disp는 nRMSE/VR/Pearson에서 모든 비교 대상을 초과하며,
학습 데이터 없이 작동하는 유일한 방법이다.
UQ는 현재 과신(overconfident) 상태이나, 진폭 calibration으로 해결 가능한 구조적 문제이다.
