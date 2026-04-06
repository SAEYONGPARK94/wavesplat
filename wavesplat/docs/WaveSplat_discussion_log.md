# WaveSplat 연구 설계 토론 전체 기록

## 참여자 | 토론 기간: 2026.03.27~29

- **Prof. K (지도교수)**: Novelty와 물리적 의미에 강한 집착. 시뮬레이터 불신. Top venue 지향.
- **Prof. E (외부위원, 해양공학)**: 주제에 회의적. 실용성/물리적 타당성 의심. 구체적 수치로 논증.
- **Rev. 2 (학회 리뷰어)**: 기술적 정확성 집착. 기존 연구 차별점 정밀 추적.
- **Saeyong (박사과정)**: Avikus 실무 경험. 구현 가능성/일정 고려. 비판 수용 및 개선.

---

# Part I: 연구 주제 및 방법론 진화

## 1. 초기 아이디어: Physics-informed 3DGS

### 출발점
- 선박 탑재 LiDAR로 수면 포인트가 극도로 sparse → 3D Gaussian Splatting으로 dense wave field 복원
- 3DGS의 sparse → dense 복원 능력 + wave physics prior 결합

### 8라운드 검토 과정에서 발견된 문제

**라운드 1~3: 3DGS 적용 가능성 검토**
- 3DGS 핵심은 multi-view 이미지의 photometric loss → LiDAR-only에서는 이 loss가 없음
- Tile-based rasterization, SH color 모두 불필요
- 남는 것은 "Gaussian basis + gradient descent"뿐 → 이것은 3DGS가 아님

**라운드 4~5: FluidGS와의 차별화**
- FluidGS(ACM MM 2025)가 이미 "physics-informed 3DGS for fluids" 수행
- WaveSplat이 Navier-Stokes를 wave equation으로만 바꾸면 → application이지 contribution이 아님
- Prof. K: "FluidGS의 물리만 바꾸면 그건 contribution이 아니라 application이야"

**라운드 6~8: "From Points to Waves" 발견**
- Coastal Eng. (2024): ETKF + sparse stereo 3D points → wave field 추정
- Dispersion relation을 예측 모델에 사용, fuzzy logic으로 outlier 처리
- 핵심 한계: sparse 포인트를 **uniform grid에 보간 후** ETKF에 입력
- 극도로 sparse한 LiDAR에서는 grid 보간 자체가 불가능 (포인트 없는 cell이 대다수)

### [결정] 3DGS → Grid-free Spectral Gaussian Estimation 전환

**전환 논리**:
1. 3DGS의 고유 메커니즘을 다 제거하면 3DGS라 부를 수 없음
2. FluidGS와 차별화 불가
3. "From Points to Waves"의 grid 보간 한계를 극복하는 것이 진짜 novelty
4. Spectral domain에서 Gaussian mixture → "Spectral Gaussian" 표현이 물리적으로 자연스러움

---

## 2. Spectral Gaussian Estimation 구체화

### 4인 토론: "왜 Spectral Gaussian인가?"

**Prof. K**: "Spectral 도메인에서 Gaussian mixture로 wave spectrum을 표현하는 게 기존에 있는가?"

**Saeyong**: "파랑 스펙트럼을 parametric 모델(JONSWAP, PM)로 표현하는 건 있지만, non-parametric Gaussian mixture로 표현하는 건 없습니다. Parametric은 단일 첨두만 표현 가능하지만, Gaussian mixture는 이중 첨두(wind sea + swell), 비대칭 스펙트럼도 유연하게 표현 가능합니다."

**Prof. E**: "Discrete spectral components vs. Spectral Gaussian의 실질적 차이가 뭐냐?"

**Saeyong**: 세 가지 차이:
1. **연속 표현**: spectral leakage 방지 (첨두가 grid 사이에 있어도 정확)
2. **적응적 해상도**: 에너지 집중 대역에 narrow, 분산 대역에 wide Gaussian
3. **Multi-modal 표현**: 데이터에서 자동으로 모드 발견

**Prof. E**: "고주파 f⁻⁵ tail을 Gaussian으로 표현하려면 많은 Gaussian이 필요하다."

**Saeyong**: "고주파 tail의 에너지는 전체의 5~10%. 파고에 기여하는 주요 성분(0.05~0.3Hz, 에너지 90%+)에 집중하면 됩니다."

### [결정] "Gaussian Splatting" 명칭 제거

**Prof. E**: "Spectral 공간에서 Gaussian mixture를 fitting하는 것은 mixture model estimation이지, Gaussian Splatting이 아니다. 학술 논문에서 느슨한 유추로 유명 기법의 이름을 붙이면 신뢰도가 떨어진다."

**Prof. K**: "Visibility를 위해 misleading한 이름을 쓰면 rebuttal에서 더 고생한다."

**결정**: 논문 제목 — "WaveSplat: Grid-free Wave Field Estimation from Sparse Ship-mounted LiDAR via Spectral Gaussian Mixture"

### 확정된 방법론 구성요소

1. **Spectral Gaussian Representation**: 파랑 스펙트럼을 (k, ω) 공간의 Gaussian mixture로 표현
2. **Dispersion-constrained Parameterization**: 각 Gaussian의 중심이 dispersion shell 위에 제약 → ωᵢ = √(g|kᵢ|tanh(|kᵢ|d))로 자유 파라미터 2개(kx, ky)로 축소
3. **Sequential Update**: 매 프레임 predict(위상 진화) → update(관측 반영)
4. **Confidence-weighted Observation Model**: 입사각/거리 기반 LiDAR 포인트 신뢰도

### 확정된 Contributions

- **C1**: Grid-free spectral estimation framework (기존 ETKF의 grid 보간 한계 극복)
- **C2**: Dispersion-constrained spectral Gaussian parameterization (물리를 구조에 내장)
- **C3**: Fresnel-aware confidence model (LiDAR 수면 관측 특화)
- **C4**: 2nd order bound harmonic correction (선택적, 후속 연구 가능)

---

# Part II: 시뮬레이션 전략

## 3. Isaac Sim → Pure Python 전환

### 교수의 근본 요구

**Prof. K**: "Isaac Sim이 잘 만들어져 있다는 건 이유가 안 돼. 모든 물리가 교과서/논문에서 유도되어야 해. 시뮬레이션의 각 구성요소가 알려진 물리 법칙을 올바르게 구현하고 있다는 것을 독립적으로 보여줘야 해."

### Isaac Sim을 쓸 수 있는 논리 (검토됨)

Isaac Sim의 RTX ray tracing을 기하학 가속기로만 사용하고, 모든 물리(Fresnel, dropout, noise)는 우리 코드에서 구현하는 방식이 가능. 하지만:

- Isaac Sim 초중급자가 환경 구축에 4~6주 소요 (vs Pure Python 2~3주)
- 데이터셋 생성 속도 차이 18분 (RTX 22초 vs Open3D 19분) → 무시 가능
- 해양공학 reviewer에게 Isaac Sim은 낯선 도구 → 설득력 저하
- 재현성: pip install vs Isaac Sim + RTX GPU
- 디버깅: print/pdb vs Isaac Sim 내부 상태 추적

### [결정] Pure Python으로 확정

**WaveSim 아키텍처**:
```
wavesim/
├── wave_generator.py        # Gerstner wave (NumPy)
├── lidar_simulator.py       # Ray casting (Open3D)
├── raydrop_uniform.py       # Model A: diffuse scattering
├── raydrop_slope.py         # Model B: specular backscatter
├── gt_generator.py          # Gerstner 수식 직접 평가
├── noise_model.py           # Gaussian noise
├── data_writer.py           # HDF5
└── config.yaml
```

### Trust Chain (최종 합의)

모든 구성요소가 교과서/논문 기반 물리 법칙에서 유도되고 unit test로 독립 검증:

- Gerstner wave → Gerstner (1802), Dean & Dalrymple (1991)
- JONSWAP → Hasselmann et al. (1973)
- Q 제약 → Tessendorf (2001), Finch (2004)
- Ray intersection → Möller-Trumbore (1997)
- Fresnel → Hecht (2017)
- Cox-Munk → Cox & Munk (1954)
- Sensor noise → LiDAR datasheet

## 4. Calibration-free 접근

### 토론 과정

**이전 방안**: NEUBOAT 데이터로 raydrop 모델 calibration
**문제점**:
- NEUBOAT LiDAR는 장애물 감지용, 수면 관측용이 아님 → overfitting 위험
- 특정 LiDAR/조건에 종속 → 일반화 한계
- Calibration 데이터의 해상 조건을 정확히 모름

**최종 방안**: Parametric sweep
- Return probability를 고정하지 않고 N_pts = 5~500 sweep
- "N_pts ≥ X이면 RMSE < Y"라는 일반적 결론
- NEUBOAT 데이터는 곡선 위의 "reality anchor"로만 사용 (1~2문장)

**Rev. 2**: "SNR을 sweep하면서 BER 곡선을 그리듯, N_pts를 sweep하면서 RMSE 곡선을 그리는 것. 아무도 SNR 모델이 정확하냐고 묻지 않는다."

## 5. 시뮬레이션의 "충분히 완벽한" 기준

### 반드시 정확해야 하는 것 (Must-have)

| 요소 | 구현 | 비용 |
|------|------|------|
| LiDAR 빔의 기하학적 hit 위치 | Open3D ray casting | 1주 |
| Return/dropout 패턴 (두 모델) | Fresnel + slope 기반 | 1주 |
| 측정 노이즈 | Gaussian noise | 반나절 |
| GT 정확성 | Gerstner 수식 직접 평가 | 1일 |
| Q 파라미터 제약 | ΣQᵢωᵢAᵢ ≤ 1 | 코드 한 줄 |
| Full dispersion | ω²=gk·tanh(kd) | 코드 한 줄 |

### 확실히 스킵하는 것

시각적 렌더링, 바람/조류/안개/비, 비선형(HOS), 쇄파, 6-DOF, MDL material, 물보라

---

# Part III: LiDAR 수면 반사 물리

## 6. Return 메커니즘 토론

### NEUBOAT 관측 사실

**확인된 것**: 파도가 있는 실해역에서 전방 LiDAR에 수면 return이 간헐적으로 찍힘
**미확인**: 정수면에서 찍히는지 (테스트한 적 없음)

### 두 가지 메커니즘 후보

**Model A — Diffuse scattering (capillary wave 기반)**
```python
P_return = η × R_fresnel(θ_i) × cos(θ_i) / distance²
```
- 수면 미세 거칠기가 빛을 산란
- 풍속 의존 (Cox-Munk)
- 정수면에서도 약간의 return 가능
- Capillary wave가 주요 산란원

**Model B — Specular backscatter (gravity wave slope 기반)**
```python
반사 방향 r = d - 2(d·n)n
if |r - (-d)| < threshold: return
else: drop
```
- 파봉 경사면에서 수면 법선이 빔 방향과 정렬될 때만 return
- 파의 존재에 결정적 의존
- 정수면에서는 return 불가
- Capillary wave 불필요

### 물리적 분석

전방 100m 관측 시 (설치 높이 5m):
```
Grazing angle = arctan(5/100) = 2.86°
Specular backscatter를 위한 필요 slope = tan(2.86°) = 0.050
BF4의 파 경사 ε ≈ 0.05 → 파봉 근처에서 조건 성립!
```

거리별 필요 slope:
```
R=200m → slope 0.025 (BF3에서도 가능)
R=100m → slope 0.050 (BF4 파봉에서 성립)
R=50m  → slope 0.100 (BF4 최대 경사 근처)
R=30m  → slope 0.167 (BF5 이상 필요)
```

### 방향 의존성 발견

**Prof. K**: "slope_required는 빔 방향 성분의 slope. 파 전파 방향과 빔 방향이 일치할 때만 성립."

→ Head sea(파가 센서 방향으로 전파) → return 최대
→ Beam sea(횡방향) → return 최소
→ 논문에서 분석할 수 있는 흥미로운 물리적 현상

### [결정] 두 모델 병렬 구현

- 어느 메커니즘인지는 NEUBOAT 데이터 분석 또는 수조 실험에서 확인
- 그 전까지 두 모델을 모두 구현
- 두 모델의 비교 자체가 ablation study의 한 축
- 가능한 결과 시나리오 모두 논문에 가치 있음

## 7. Return Probability 추정

### Prof. E의 계산

```
32채널 LiDAR, 수평 FOV 120°, 0.2° resolution
→ 프레임당 ~19,200 빔
→ 수면 도달 ~9,600 빔
→ Return rate ~1% 가정 시 → 프레임당 ~96개 포인트
→ 0.5%면 48개, 2%면 192개
```

**주의**: 이 1%의 근거가 불충분. 정확한 값은 이론 계산만으로는 불가능.
→ **Parametric sweep으로 해소** (N_pts = 5~500)

### 포인트 분포 패턴

- 근거리(30~80m)에 집중, 원거리(150m+)에서는 거의 없음
- Model B에서는 파봉 센서 향 경사면에 편향 집중
- 이 편향이 오히려 **wave field 복원에 유리**할 수 있음 (파봉 = 가장 중요한 feature)

---

# Part IV: 비선형 파랑 및 물리적 충실도

## 8. 비선형 파랑의 영향

### Beaufort별 평가

| BF | ε | 2차 파봉 보정 | Hydrofoil 관련 | 판정 |
|----|---|-------------|--------------|------|
| 3 | 0.03 | 1~2cm | Operational | 무시 가능 |
| 4 | 0.05 | 5~10cm | Operational | 관리 가능 |
| 5 | 0.05~0.07 | 10~20cm | 한계 | 2차 보정 필요 |
| 6+ | >0.06 | >20cm | 운항 중단 | 범위 외 |

### 핵심 발견

**Hydrofoil operational envelope(Hs<1.5m, BF3~4)과 선형 모델 유효 범위가 겹침**
→ 적용 범위 한정이 실용적으로도 자연스러움

### 비선형이 WaveSplat에 미치는 구체적 영향

1. **Dispersion relation 깨짐**: Bound harmonics는 ω²=gk·tanh(kd)를 따르지 않음. Dispersion 하드 제약이 bound harmonics를 체계적으로 무시 → 파봉 과소추정
2. **Crest sharpening**: 2차 Stokes 효과로 파봉이 선형 예측보다 5~15% 높음
3. **관측 편향**: 비선형 파봉에서 LiDAR 포인트가 편향 분포

### [결정] 비선형 대응 전략

- **구현**: 선형 복원만 (1차)
- **후속 연구**: 2차 bound harmonic correction, HOS 교차 검증
- **논문**: "non-breaking waves, BF3~5" 적용 범위 명시
- **쇄파**: 절대 다루지 않음 (범위 외)

## 9. Gerstner Wave 중첩의 물리적 충실도

### "중력에 의한 crest 붕괴" 문제

**Saeyong의 원래 질문**: "시뮬레이션에서는 중력 영향으로 인한 crest 붕괴를 구현할 수 없는데, 이건 물리적 오류 아닌가?"

**Prof. K의 분석**:
- Gerstner wave 중첩에서 여러 성분의 crest가 중첩되면 높이가 무한히 올라갈 수 있음
- 실제 물리에서는 이때 에너지 cascade + 쇄파로 높이가 제한됨
- Gerstner 중첩에는 이 메커니즘이 없음

**Prof. E의 정량적 평가**:
- BF4에서 30성분 중첩 시, 국소 경사가 Miche 한계(H/L=1/7) 초과 확률 < 1%
- 성분 수를 30개로 제한하면 고주파 tail이 잘려 비현실적 경사 발생 확률 극히 낮음
- Q 파라미터 제약이 particle crossing 방지

### Gerstner wave의 정확한 물리적 위치

- Gerstner wave는 **rotational** exact solution (irrotational이 아님)
- Stokes wave(irrotational)와 1차에서 동일, 2차 이상에서 차이
- 차이의 크기: O(ε²) ≈ 0.25% (BF4) → 무시 가능

### [결정] 물리적 충실도 계층 분석

논문 Discussion에 포함:
- Level 0: 개별 Gerstner = exact solution → 문제 없음
- Level 1: 중첩 = approximate → ε≪1에서 유효
- Level 2: 경사 한계 초과 가능 → Q 제약으로 방지
- Level 3: Wave-wave interaction 없음 → 후속 연구
- Level 4: 쇄파 없음 → 범위 외

**시뮬레이션 성능 = 실해역 상한** (bound harmonics 없어 더 쉬운 조건)

## 10. 물리적/이론적 오류 검토 (9개 이슈)

### 심각도 높음 (2건)

**이슈 #4: Particle Crossing**
- Gerstner 중첩에서 Jacobian ≤ 0이면 multi-valued surface 발생
- 충분조건: ΣQᵢ·kᵢ·Aᵢ < 1
- BF3~4에서 Σ(ωᵢAᵢ) ≈ 0.3~0.5 → 여유 충분
- **대응**: Q 제약을 코드에 반드시 구현

**이슈 #8: Return Probability 불확실**
- 프레임당 포인트 수의 정확한 예측 불가
- **대응**: Parametric sweep (N_pts = 5~500)

### 심각도 중간 (5건)

- #1: Grazing/incidence angle 혼용 → 용어 통일 (θ_g = π/2 - θ_i)
- #5: Spectral Gaussian 역변환은 narrow 조건에서만 해석적 → 조건 명시
- #6: 2차 전달함수의 방향 의존성 → 2D 구현 시 full 전달함수 (후속)
- #7: "깊은 물" 가정 과도 → **full dispersion(tanh) 항상 사용**
- 해류 영향 → limitation으로 기술

### 심각도 낮음 (2건)

- #2: 높은 반사율 ≠ 높은 return → 수식적 유도 보완
- #3: Gerstner는 rotational → 논문에 명시
- #9: 시간 전파 비선형 오차 → BF4에서 8초는 안전

---

# Part V: 실험 전략

## 11. 수조 실험

### 수조 제원

110m × 8m × 3.5m, 일방향 조파기, 팬 없음, wave gauge 없음, 4일×3시간 가용

### 역할의 진화

**초기 계획**: GT 확보 + 알고리즘 검증 + return 측정 (4일 풀 활용)
**중간 수정**: Return characterization만 (1일)
**현재 상태**: 당장 불가, 후속에 수행

### Capillary wave 없음의 의미

- **이전 우려**: capillary 없으면 return이 안 나올 수 있음
- **새로운 해석**: NEUBOAT에서 파도 시 return이 나온 것은 gravity wave slope에 의한 specular backscatter일 가능성 → capillary 없어도 **파만 있으면 return 가능**
- 수조 결과 = 실해역의 **하한** (conservative test)

### [결정] 수조는 논문 1 투고를 지연시키지 않음

- 논문 1: 시뮬레이션 + NEUBOAT 데이터로 투고
- 수조: revision 단계 또는 학위 논문에서 수행

## 12. 실선 시험

### 가용 횟수: 1~2회

### [결정] Calibration이 아니라 Validation에 사용

**Calibration에 쓰면**: 시뮬레이션 파라미터 맞추기. 논문 가치 낮음.
**Validation에 쓰면**: "실제로도 작동한다"는 결정적 증거. 논문 가치 높음.

### Validation 방법 (GT 없이)

1. **Prediction consistency**: 현재 복원 → 다음 프레임 예측 → 실제 관측과 비교
2. **해양 부이 비교**: 복원된 Hs/Tp와 인근 관측소 데이터 대조
3. **선박 운동 정합성**: 복원된 wave field의 선박 위치 파고와 IMU heave 비교

### [결정] 실선은 시뮬레이션 완성 후 (M12~M13)

시뮬레이션 결과가 convincing하면 실선 제안. 안 되면 시뮬레이션 only로 투고.

## 13. NEUBOAT 데이터 활용

### 역할 변화

**이전**: Calibration 용 → **현재**: Reality anchor (1~2문장)

### 분석 내용

- 수면 포인트 추출 (높이/거리 필터 + 구조물 마스킹)
- 프레임당 수면 포인트 수 분포
- 거리별 밀도
- (가능하면) 기상 조건 매칭

### [결정] NEUBOAT 분석과 WaveSim 개발 병렬 진행

WaveSim은 두 raydrop 모델을 모두 구현, NEUBOAT 결과에 따라 main 모델 선택.

---

# Part VI: 논문 구조 및 타임라인

## 14. 논문 구성

### 논문 1 (핵심, 12개월)

**제목**: "WaveSplat: Grid-free Wave Field Estimation from Sparse Ship-mounted LiDAR via Spectral Gaussian Mixture"

**구조**:
```
Section 1: Introduction
Section 2: Problem Formulation (LiDAR 수면 반사 물리)
Section 3: WaveSplat Algorithm (spectral Gaussian estimation)
Section 4: WaveSim (Pure Python, trust chain)
Section 5: Experiments
  5.1: Progressive complexity (1→30 성분)
  5.2: Sparsity sensitivity (N_pts sweep)
  5.3: Baseline 비교 (GPR, ETKF-grid)
  5.4: Observation model 비교 (uniform vs slope-dependent)
  5.5: Ablation
  5.6: NEUBOAT reality anchor
Section 6: Discussion
Section 7: Conclusion
```

**타겟**: Ocean Engineering (IF ~5) 또는 IROS 2027

### 논문 2 (확장, 4개월)

**제목**: "Wave-Preview Model Predictive Control for Hydrofoil Vessels Using LiDAR-based Wave Field Estimation"

**내용**: WaveSplat(black box) → 시간 전파 → Hydrofoil MPC → closed-loop 검증
**타겟**: IROS 또는 Control Engineering Practice

### 학위 논문 (통합, 6개월)

```
Ch.1: 서론
Ch.2: 선행연구
Ch.3: LiDAR 수면 관측 물리 + WaveSim [논문 1 확장]
Ch.4: Spectral Gaussian Estimation [논문 1 핵심]
Ch.5: 시뮬레이션 검증 [논문 1 실험]
Ch.6: 비선형 보정 + 선박 운동 [미발표]
Ch.7: 실해역 검증 [미발표, 수조/실선]
Ch.8: Hydrofoil MPC [논문 2]
Ch.9: 결론
```

## 15. 범위 한정

### 구현 (Must)

- WaveSim (Pure Python): wave_generator, lidar_simulator, raydrop 2종, gt_generator, noise, data_writer
- Spectral Gaussian Estimation (PyTorch)
- Sequential update, confidence model
- Progressive complexity (1→30 성분)
- Sparsity sweep (N_pts = 5~500)
- Baseline 2종 (GPR, ETKF-grid)
- Observation model A/B 비교
- NEUBOAT 데이터 분석

### 스킵 (Future work)

- Adaptive spectral densification
- 2차 bound harmonics
- HOS 교차 검증
- 6-DOF 선박 운동 (정지+직진까지만)
- Hydrofoil MPC (논문 2)
- Cramér-Rao bound
- Isaac Sim

## 16. 타임라인

| Phase | 기간 | Track A (데이터) | Track B (개발) |
|-------|------|----------------|---------------|
| 1 | M1~M2 | NEUBOAT 분석 | WaveSim (두 raydrop 모델 포함) |
| 2 | M3~M4 | → main 모델 선택 | WaveSim 검증 + 데이터셋 생성 |
| 3 | M5~M8 | — | WaveSplat 알고리즘 + 실험 |
| 4 | M9~M11 | — | 논문 1 작성/투고 |
| (선택) | M12 | 수조 또는 실선 | revision 반영 |
| 5 | M14~M17 | — | 논문 2 (MPC) |
| 6 | M18~M24 | 수조/실선 | 학위 논문 |

---

# Part VII: 핵심 원칙 및 체크리스트

## 17. 개발 8원칙

1. **모든 물리는 first principles에서 유도한다.** "라이브러리가 해준다"는 근거가 아니다.
2. **모든 구성요소는 독립적으로 검증한다.** Unit test로 해석해와 비교.
3. **Full dispersion relation(tanh)을 항상 사용한다.** 깊은 물 근사 금지.
4. **Q 파라미터 제약(ΣQᵢωᵢAᵢ ≤ 1)을 항상 강제한다.**
5. **Parametric sweep으로 결론을 일반화한다.** 특정 값에 종속된 주장 금지.
6. **두 가지 raydrop 모델을 모두 유지한다.** 데이터가 답을 준다.
7. **논문 1의 투고를 수조/실선 때문에 지연시키지 않는다.**
8. **코드와 데이터를 공개한다.** 재현성이 신뢰도의 핵심.

## 18. 4인 최종 합의 체크리스트

- [x] 3DGS 명칭 제거, Spectral Gaussian Estimation으로 확정
- [x] Isaac Sim 불사용, Pure Python WaveSim
- [x] Calibration-free, parametric sweep 기반
- [x] NEUBOAT 데이터는 reality anchor (1~2문장)
- [x] 두 가지 raydrop 모델 병렬 구현 (uniform + slope-dependent)
- [x] 수조/실선은 논문 1 투고를 지연시키지 않음
- [x] 비선형 보정은 후속 연구
- [x] 적용 범위 BF3~5, non-breaking waves
- [x] Q 파라미터 제약 필수 구현
- [x] Full dispersion(tanh) 항상 사용
- [x] Grazing/incidence angle 용어 통일
- [x] 시뮬레이션 성능 = 실해역 상한이라는 점 논문에 명시
- [x] 코드 + 데이터셋 공개
- [x] 논문 1에 집중 — 이것이 나오면 나머지는 따라옴

---

# Part VIII: 미해결 질문 (향후 토론 주제)

## 19. 열린 질문들

1. **Spectral Gaussian의 수렴성**: Non-convex 최적화에서 local minima에 얼마나 자주 빠지는가? Coarse-to-fine이 충분한가?
2. **Temporal accumulation의 최적 window**: 몇 초/몇 프레임을 누적하는 것이 최적인가?
3. **Spectral Gaussian 수 N의 자동 결정**: 고정 N vs adaptive densification의 trade-off
4. **2D 방향 스펙트럼 추정**: 1D(주파수)에서 2D(주파수+방향)로 확장 시 추가 난이도
5. **실시간 성능**: PyTorch 기반 최적화가 프레임당 50ms 이내에 수렴하는가?
6. **수조에서 실제로 return이 나오는가**: 정수면 vs 파도 조건 분리 실험 필요
7. **NEUBOAT 데이터에서 수면 포인트가 충분히 분리 가능한가**: 노이즈, 물보라 등과의 구분
8. **Baseline(ETKF-grid)의 공정한 구현**: "From Points to Waves" 논문의 재구현 정확도
