# WaveSplat + Hydrofoil Feedforward — 발표용 종합 학습자료

> 박사 논문 2편(Ocean Engineering 투고)을 "기초부터 응용까지" 한 번에 읽고
> 이해하여 PPT를 만들고 교수님께 발표할 수 있도록 구성한 통합 가이드.
>
> 작성일: 2026-04-08

---

## 목차

**Part A — 배경 지식 (기초)**
1. 해양파 물리 기초
2. 선박이 파도 위에서 겪는 문제
3. 파면(wave field) 추정이 왜 필요한가
4. 기존 접근법과 그 한계

**Part B — 논문 1: WaveSplat (파면 추정)**
5. 문제 정의와 목표
6. 핵심 아이디어: Grid-free Spectral Gaussian
7. 알고리즘 단계별 설명
8. 실험 결과 핵심 수치
9. 경쟁 기법 대비 우위

**Part C — 논문 2: Hydrofoil Feedforward Control**
10. 하이드로포일 선박의 운동 방정식
11. Wave-preview feedforward 제어 구조
12. 시뮬레이션 결과와 해석
13. 두 논문의 연결 고리

**Part D — 발표 전략**
14. PPT 슬라이드 구성 제안 (20장)
15. 예상 질문과 모범 답변
16. 논문 학습 체크리스트

---

# Part A — 배경 지식

## 1. 해양파 물리 기초

### 1.1 파도는 파장(λ), 주파수(ω), 방향(θ)을 가진 여러 성분의 합

바다 표면의 높이 η(x,y,t)는 무수히 많은 사인파(단색파)의 선형 중첩으로 표현됩니다:

$$
\eta(x,y,t) = \sum_{i=1}^{N} a_i \cos(k_{x,i}x + k_{y,i}y - \omega_i t + \varphi_i)
$$

- **a_i**: 진폭 (m)
- **k_i = (k_x, k_y)**: 파수 벡터 (rad/m), 크기 |k|=2π/λ, 방향=진행방향
- **ω_i**: 각주파수 (rad/s), 주기 T=2π/ω
- **φ_i**: 위상

### 1.2 분산 관계 (Dispersion Relation) — 절대 중요

물리학적으로 ω와 |k|는 독립이 아니라 **분산 관계**로 묶여 있습니다:

$$
\omega = \sqrt{g|k|\tanh(|k|d)}
$$

- g = 9.81 m/s² (중력)
- d = 수심 (m); 깊은 바다에서는 tanh≈1 → ω=√(g|k|)

**의미**: 파장이 길수록 주기도 길다 (큰 파도는 느리게 움직이는 듯 보임).
WaveSplat은 이 관계를 **hard constraint**로 사용해 추정 파라미터 공간을 축소합니다.

### 1.3 JONSWAP 스펙트럼

실제 바다는 "일정한 바람이 일정 시간 분 일정 거리" 조건에서 관측된
**JONSWAP 스펙트럼**을 따릅니다. Beaufort 등급(BF2~BF5)에 대응해
유의파고 H_s와 피크 주기 T_p가 결정됩니다:

| BF | H_s (m) | T_p (s) | 설명 |
|---|---|---|---|
| 2 | 0.3 | 3.8 | 약한 바람 |
| 3 | 0.6 | 4.8 | 가벼운 너울 |
| 4 | 1.0 | 5.6 | 중간 |
| 5 | 2.0 | 7.0 | 거친 바다 |

### 1.4 파도의 경사도 (Steepness) ε

$$
\varepsilon = \pi H_s / \lambda
$$

- ε < 0.1 → 선형 파 (linear wave) 가정 타당
- ε > 0.1 → 비선형 효과, Stokes 파 이론 필요

WaveSplat이 테스트한 BF2–5 범위는 ε ≈ 0.05–0.07로 **선형 모델이 유효**합니다.

---

## 2. 선박이 파도 위에서 겪는 문제

### 2.1 6자유도 운동 (6 DOF)
선박은 파도를 만나면 heave(상하), pitch(종동), roll(횡동) 등 6개 축으로 흔들립니다.
특히 **하이드로포일(수중익)** 선박은 빠른 속도(20+ knots)로 달리면서
파도의 수직 가진에 민감합니다.

### 2.2 피드백 제어의 한계
기존 제어는 **측정된 운동**(가속도, 각도)을 보고 보정합니다. 즉,
"이미 흔들리기 시작한 다음" 반응하므로 고주파 파도 외란에 대해
지연이 생기고 승객 쾌적성(ISO 2631)이 악화됩니다.

### 2.3 해결책: Wave Preview (파도 미리보기)
만약 **선박 전방 30–100 m의 파면**을 1–5초 앞서 알 수 있다면,
피드포워드 제어기가 "곧 올 외란"을 선행 보상할 수 있습니다.
→ 이것이 두 논문이 함께 풀고자 하는 **통합 파이프라인**입니다.

---

## 3. 파면 추정이 왜 필요한가

### 3.1 센서: LiDAR on ship deck
선박 갑판에 장착된 스캐닝 LiDAR는 1초에 수만~수십만 개의 점을
해수면에서 반사시켜 받습니다. 이 점들은

- **희소(sparse)**: 거리에 따라 간격이 벌어지고 일부 영역은 관측 불가
- **불규칙(irregular)**: 격자가 아닌 방사형 패턴
- **노이즈 있음**: 파도 경사, 스프레이 등의 교란

### 3.2 목표
이 희소한 점들로부터 **전체 파면 η(x,y,t)**를 시간에 따라 복원하고,
나아가 "수 초 뒤" 파면을 예측해야 합니다.

---

## 4. 기존 접근법과 한계

| 방법 | 원리 | 한계 |
|---|---|---|
| **격자 보간 + FFT** | 관측점을 격자로 보간 후 공간 FFT | 희소/불규칙 관측에서 보간 오차 큼, 경계 영향 |
| **EnKF (Ensemble Kalman Filter)** | 여러 "멤버"를 굴려 통계적 동화 | 계산 비용 큼, 격자 기반 |
| **GPR (Gaussian Process Regression)** | 커널 기반 연속 보간 | O(N³) 복잡도, 1000점만 넘어도 실시간 불가 |
| **HOS (Higher-Order Spectral)** | 비선형 파 물리 기반 | 초기조건/주기경계 필요, 실시간 어려움 |
| **신경망 (Ehlers 2024 등)** | 데이터 드리븐 | 학습 데이터 외 일반화 약함, 물리 보장 X |

**공통 한계**: (i) 격자 의존, (ii) 시간 추적 불안정, (iii) 물리(분산관계) 미적용.

---

# Part B — 논문 1: WaveSplat

## 5. 문제 정의

**입력**: 시간별 LiDAR 포인트 클라우드 {(x_n, y_n, z_n, t_n)}
**출력**: 시간에 따라 진화하는 파면 η̂(x,y,t)의 추정
**제약**: 실시간(≤30 ms/프레임), 임의 공간 해상도 평가 가능, 물리 일관성

## 6. 핵심 아이디어 — Grid-free Spectral Gaussian

WaveSplat은 파면을 **N개의 "Spectral Gaussian 컴포넌트"**의 합으로 표현합니다:

$$
\hat\eta(x,y,t) = \sum_{i=1}^{N} a_i \cos(k_{x,i} x + k_{y,i} y - \omega_i t + \varphi_i)
$$

각 컴포넌트는 파라미터 θ_i = (a_i, k_{x,i}, k_{y,i}, φ_i)를 가지며,
**ω_i는 분산 관계로 k_i로부터 자동 계산**됩니다 (자유 파라미터 X).

### 왜 "Splat"인가?
3D Gaussian Splatting처럼 **명시적 원소(컴포넌트)**를 직접 최적화하되,
격자나 볼륨 대신 **스펙트럼 공간**에서 각 원소가 "splat"됩니다.
→ Grid-free + Explicit parametric form.

### 핵심 차별점
- **격자 없음**: 관측점에서 직접 forward model 계산
- **분산 관계 hard constraint**: 비물리적 해 원천 차단
- **Temporal tracking**: 프레임 간 컴포넌트 ID 유지 (w_phi mode)
- **Adaptive N**: 에너지 기반 pruning + 잔차 triggered growth

## 7. 알고리즘 단계별 설명

### Step 1: Initialization (Beamforming)
첫 프레임에서 (k_x, k_y)의 거친 격자를 만들고 각 (k_x, k_y)에 대해
관측 LiDAR 점을 투영해 **beamforming power**를 계산합니다.
상위 N개 피크를 초기 컴포넌트로 선택합니다.

- **v2_konly 방식**: 빔포밍은 k만 결정, ω와 φ는 후속 최적화에 위임
- 평균 18% tracking 성능 개선 달성

### Step 2: Optimization (Coarse-to-Fine)
각 컴포넌트의 (a, k_x, k_y, φ)를 Adam 옵티마이저로 최적화:

$$
\mathcal{L} = \sum_n (\hat\eta(x_n,y_n,t_n) - z_n)^2 + \lambda_{\text{prior}} \|a\|_2^2
$$

분산 관계는 ω를 미분 가능한 함수로 대체하여 gradient flow를 유지합니다.

### Step 3: Temporal Tracking (w_phi mode)
새 프레임이 오면 **(k_x, k_y)는 고정**하고 (ω, φ)만 업데이트합니다.
- 이유: k는 파도 전체의 정체성, ω·φ만 시간에 따라 바뀜
- 이전 버전은 모든 파라미터를 업데이트해 overfitting 발생 → w_phi로 해결

### Step 4: Adaptive N
- **Pruning**: 에너지 하위 컴포넌트 제거 (|a_i|²가 임계 이하)
- **Growth**: 잔차가 임계 넘으면 빔포밍으로 새 컴포넌트 추가
- **Safety rollback**: 추가 후 NRMSE 악화 시 되돌림

### Step 5: Continuous Evaluation
최적화된 파라미터 θ로 임의 (x,y,t)에서 η̂ 평가.
**격자 의존 없음** → 하이드로포일 전방 100 m 지점의 파고 예측 즉시 가능.

## 8. 실험 결과 (핵심 수치)

12개 에피소드 (BF2-5, uniform/slope LiDAR, 5 seeds) 기준:

| 방법 | 평균 nRMSE | 런타임 (ms/frame) |
|---|---|---|
| **WaveSplat** | **0.233** | **13.4** |
| ETKF (grid) | 0.450 | 31.7 |
| GPR | 0.276 | >200 |

- **nRMSE**: 48% 개선 vs ETKF, 16% 개선 vs GPR
- **실시간성**: 73.7 Hz 동작 → LiDAR 10 Hz 대비 여유 있음
- **메모리**: O(N·D) (N=컴포넌트, D=파라미터 수). 격자 기반 대비 100×↓

## 9. 경쟁 기법 대비 우위 (novelty 방어)

Prof.K persona 리뷰 대응:

1. **vs Drazen (2008) 단일 프레임 sparse method**: WaveSplat은
   temporal tracking을 통해 **프레임 간 컴포넌트 정체성 유지** — 이는
   연속 예측 및 피드포워드 control에 필수적.
2. **vs 신경망 기반 (Ehlers 2024)**: 물리 제약(분산관계)으로 OOD 일반화.
3. **vs HOS**: 초기조건 불필요, 관측 도착과 동시에 동작.

---

# Part C — 논문 2: Hydrofoil Feedforward

## 10. 2DOF 운동 모델

하이드로포일 선박의 heave(z)-pitch(θ) 연성 운동:

$$
M\ddot z + C_z \dot z + K_z z = F_{\text{wave}}(t) + F_{\text{foil}}(\delta_f, \delta_r)
$$

$$
I_\theta \ddot\theta + C_\theta\dot\theta + K_\theta\theta = M_{\text{wave}}(t) + M_{\text{foil}}(\delta_f, \delta_r)
$$

- **M, I_θ**: 질량/관성
- **δ_f, δ_r**: 전·후 foil 플랩 각도 (제어 입력)
- **F_wave, M_wave**: 파도에 의한 외란 (WaveSplat이 추정!)

### 대표 파라미터 (일반적 소형 하이드로포일)
- LOA 8.5 m, beam 2.4 m, 1700 kg, 22 knots
- 탠덤 foil 구성 (전방 inverted-T, 후방 T-foil)

## 11. Feedforward 제어 구조

**기존 피드백 제어**:
```
y(t) → K_fb → u_fb(t)
```
만 있을 때 고주파 파도에 응답 지연.

**제안: FF + FB 합성**:
```
η̂(x_preview, t+τ) → K_ff → u_ff(t)
y(t) → K_fb → u_fb(t)
u(t) = u_ff(t) + u_fb(t)
```

- x_preview: 선박 전방 ~50 m 지점
- τ: preview time (0.5–3 s)
- K_ff: 파도 외란 모델 기반 역동역학

### 저역통과 필터
WaveSplat 추정에는 잡음이 있으므로 IIR LPF를 적용:

$$
f_c = \alpha/(2\pi\Delta t) \approx 0.8 \text{ Hz}
$$

- α = 0.05, Δt = 0.01 s
- BF3 head sea 22 knots에서 최대 encounter frequency 0.5 Hz 상회 → 신호 보존, 잡음 억제

## 12. 시뮬레이션 결과

| Scenario | FB only (m/s² RMS) | FF+FB (m/s² RMS) | 개선율 |
|---|---|---|---|
| BF3 head | 0.42 | 0.21 | **50%** |
| BF4 head | 0.78 | 0.45 | 42% |
| BF4 bow | 0.65 | 0.40 | 38% |

- **Optimal preview**: τ ≈ 1.0–1.5 s (너무 길면 추정 오차 누적)
- **한계**: nRMSE 50% 이상 추정 오차에서는 FF 이득 소실

## 13. 두 논문의 연결 고리

```
[LiDAR] → [WaveSplat 추정] → [전방 파면 예측] → [Feedforward 제어] → [승객 쾌적성↑]
   (sensing)   (Paper 1)          (interface)         (Paper 2)
```

**박사학위 통합 기여**:
- Paper 1: sensing → information (파면 추정의 물리/알고리즘 혁신)
- Paper 2: information → action (정보의 제어 적용과 이득 증명)
- 합쳐서: **"grid-free spectral estimation으로 하이드로포일 승차감을 실제 개선한다"**

---

# Part D — 발표 전략

## 14. PPT 20장 구성안

| # | 제목 | 내용 |
|---|---|---|
| 1 | Title | 제목, 저자, 소속, 날짜 |
| 2 | Motivation | 하이드로포일 승차감 문제, preview 필요성 |
| 3 | Big Picture | 두 논문 통합 파이프라인 (그림) |
| 4 | Wave Physics 101 | η = Σa cos(kx-ωt+φ), 분산관계 |
| 5 | LiDAR observation | 희소·불규칙 포인트 시각화 |
| 6 | Prior Art & Gap | 표: 격자/EnKF/GPR/NN의 한계 |
| 7 | **Paper 1: Problem** | 입력/출력/제약 |
| 8 | Key Idea: Grid-free Spectral Gaussian | 수식 + 직관 |
| 9 | Algorithm Pipeline | init → opt → track → adapt |
| 10 | Temporal tracking (w_phi) | 왜 k 고정이 핵심인가 |
| 11 | Results: accuracy | 0.233 vs 0.450 vs 0.276 (bar chart) |
| 12 | Results: speed | 13.4 ms (실시간) |
| 13 | Qualitative Figure | 파면 복원 heatmap |
| 14 | **Paper 2: Problem** | 하이드로포일 2DOF, FF 필요성 |
| 15 | FF+FB Architecture | 블록다이어그램 |
| 16 | Filter Design | f_c 공식과 선택 근거 |
| 17 | Results: acceleration RMS | 50% 개선 시나리오 |
| 18 | Preview time sensitivity | τ vs nRMSE 그래프 |
| 19 | **Integrated Contribution** | 두 논문 합친 임팩트 |
| 20 | Limitations & Future Work | 선형 가정, vessel param, nonlinear 확장 |

## 15. 예상 질문 & 답변

**Q1. "Grid-free라고 하는데 결국 컴포넌트 N개를 정해야 하지 않나?"**
A. N은 adaptive입니다. 에너지 기반 pruning과 잔차 triggered growth가
N을 시나리오별(BF2 단일 성분 → BF5 30 성분)로 자동 조절합니다.

**Q2. "분산관계 hard constraint가 과적합을 막는다는 근거?"**
A. Ablation에서 soft(penalty) vs hard 비교 결과 hard가 BF4-5에서 평균
14% nRMSE 향상. 논문 Table III 참조.

**Q3. "Drazen 2008과 뭐가 다른가?"**
A. Drazen은 단일 snapshot sparse 재구성만. WaveSplat은 frame-to-frame
컴포넌트 동일성을 유지 → 피드포워드 제어에 필수.

**Q4. "선박 제원이 공개된 모델인가?"**
A. 공개된 hull form이 아닌 "대표적 소형 전기 하이드로포일"의 전형적
파라미터입니다. 논문 Limitation에 명시했고, FF 아키텍처는 vessel
specific 동역학이 아닌 파도 외란에 작용하므로 정성적 결론은 일반화 가능.

**Q5. "비선형 파도에 대해선?"**
A. BF2-5 범위 ε≈0.05-0.07에서 모델 오차 <0.5%. BF6+ (ε>0.1)는 향후
확장 과제이며 HOS 또는 Stokes 2nd-order forward model로 교체 가능한
모듈식 구조.

**Q6. "실제 선박에 구현했나?"**
A. 본 연구는 시뮬레이션 기반 proof-of-concept. 실측 검증은 Avikus의
다음 단계 개발 계획.

## 16. 논문 학습 체크리스트

발표 전 반드시 숙지해야 할 것:

- [ ] 분산 관계 유도 (얕은물/깊은물 극한 포함)
- [ ] JONSWAP 스펙트럼 모양과 주요 파라미터
- [ ] LiDAR 포인트가 왜 grid-free 접근을 필요로 하는가
- [ ] Spectral Gaussian 파라미터 (a, k_x, k_y, φ)의 물리적 의미
- [ ] Adam optimizer 원리 (단순 설명 가능 수준)
- [ ] Beamforming 직관 (상관을 통한 방향성 집중)
- [ ] ETKF와 WaveSplat의 근본적 차이 (ensemble vs parametric)
- [ ] GPR의 O(N³) 병목과 sparse GP 대안
- [ ] Heave-pitch 연성의 물리적 직관
- [ ] FF vs FB 제어의 차이와 합성 이유
- [ ] f_c = α/(2πΔt) 필터 공식 유도
- [ ] ISO 2631 승차감 지표 (간단히)
- [ ] 논문 핵심 수치 **외워두기**: nRMSE 0.233 / 48% 개선 / 13.4 ms / 50% accel reduction

---

## 부록: 주요 수식 한눈에

**파면 모델**
$$\eta(x,y,t) = \sum_i a_i \cos(k_{x,i}x + k_{y,i}y - \omega_i t + \varphi_i)$$

**분산관계 (deep water → finite depth)**
$$\omega = \sqrt{g|k|\tanh(|k|d)}, \quad |k|=\sqrt{k_x^2+k_y^2}$$

**WaveSplat 손실**
$$\mathcal{L} = \sum_n \big(\hat\eta(x_n,y_n,t_n) - z_n\big)^2 + \lambda\|a\|_2^2$$

**nRMSE**
$$\text{nRMSE} = \frac{\sqrt{\frac{1}{N}\sum (\hat\eta-\eta)^2}}{\sigma_\eta}$$

**FF 제어 합성**
$$u(t) = K_{ff}\,\hat\eta(x_{\text{prev}}, t+\tau) + K_{fb}\,y(t)$$

**IIR LPF cutoff**
$$f_c = \frac{\alpha}{2\pi\Delta t}$$

**파 경사도**
$$\varepsilon = \pi H_s / \lambda$$

---

*이 문서를 읽은 뒤 PPT 초안 작성 → 논문 해당 섹션 대조 → 예상질문 리허설 순으로
진행하면 2~3일 내에 발표 준비가 완료됩니다. 각 섹션별 세부 내용은 논문 본문과
`docs/PRESENTATION_SUMMARY.md`(기존 한글 요약)를 교차 참조하세요.*
