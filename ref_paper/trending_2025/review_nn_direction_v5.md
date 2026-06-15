# NN Direction Fact-Based Review v5 — 논문 본문 기반 검토

**Date**: 2026-05-19
**읽은 논문**: KalmanNet (Revach 2022), Neural ODE Mixture (Alvarez-Lopez 2025),
              NLML (LANL 2026), PFT-PINN (Ehlers 2025), LWT-PINN (Li 2024)

---

## 1. 논문별 핵심 사실 (본문 기반)

### 1.1 KalmanNet (Revach et al., IEEE TSP 2022)

**아키텍처**: 3 GRU + 7 FC layers, ~25K params (Architecture #2)
- GRU_Q (hidden=m^2): process noise 추적
- GRU_Sigma (hidden=m^2): predicted covariance 추적
- GRU_S (hidden=n^2): innovation covariance 추적
- FC2: Kalman Gain K_t 출력 (dim = m*n)
- Input features: observation diff, innovation, evolution diff, update diff (4개, 모두 L2-normalized)

**학습**: MSE on state + L2 regularization, Adam lr=1e-3, 1000 epochs, batch=20

**결과**:
- Linear 2x2: KF와 동일 (-21.90 vs -21.91 dB)
- Nonlinear, model mismatch: **EKF 대비 11.56 dB gain**
- Lorenz 3D, nonlinear h: **EKF 대비 11.83 dB gain**
- 계산 시간: EKF(5.4s) > KalmanNet(4.7s) > MB-RNN(2.3s) — matrix inversion 없어서 빠름

**한계**:
- GT state 필요 (supervised only)
- f(.), h(.) 함수 필요 (알려진 모델 전제)
- 고정 state dimension (adaptive N 시 재설계 필요)
- 최대 테스트: m=10 (10D state)

**우리 문제 적용**:
- Per-component: m=2 (w, phi), n=1 → 매우 작은 네트워크
- 컴포넌트 간 coupling은 residual 재계산으로 처리
- GT는 simulation에서 생성 가능
- Transition: phi += omega*dt (알려진 물리)
- Observation: z = Sigma w_i cos(...) (알려진 forward model)

### 1.2 Neural ODE Mixture (Alvarez-Lopez 2025)

**핵심**: Gaussian mixture weights alpha(t)만 Neural ODE로 진화. 평균/분산은 고정.

**우리 문제와의 차이 (본문 확인 후)**:
- 그들: probability simplex 위 weights (Sigma alpha = 1, alpha >= 0)
- 우리: unconstrained amplitudes w + angles phi — 완전히 다른 constraint geometry
- 그들: MMD loss (분포 매칭) — 우리: MSE loss (field reconstruction)
- **예측 기능 없음** — retrospective smoothing만 수행
- **기술적 전이 최소** — 개념적 유사성만 존재

### 1.3 NLML (LANL 2026)

**아키텍처**: MLP 4 hidden x 3072 neurons, LeakyReLU, **48.9M params**
- Input: 50 freq x 36 dir = 1800 spectral bins + CG(50) + depth(1) = 1851
- Output: Snl (1800)
- Loss: variance-weighted L1

**Snl의 물리적 의미**:
- 4-wave resonant interaction (Hasselmann 1962)
- 스펙트럼 형상 진화의 주요 메커니즘
- **시간 스케일: 시간~일 단위** (hours to days)

**결과**: WRT 대비 136x 속도, DIA 대비 2x 정확도

**우리 문제에 대한 의미 (중요한 발견)**:
- Snl은 **시간~일** 단위에서 작동 → **10-60초 예측에는 Snl 무관**
- NLML은 dense spectral grid (1800 bins) → 우리 parametric representation (15 comp)과 불일치
- **존재 증명으로서만 가치**: NN이 비선형 wave-wave interaction을 학습할 수 있다

### 1.4 Ehlers PFT-PINN (2025) / LWT-PINN (Li 2024)

**Ehlers**: 4 hidden x 200 neurons x 2 networks (eta + Phi), Fourier feature embedding
- 완전 비선형 PFT (Laplace + KFSBC + DFSBC + Bottom BC) — soft constraints
- **~30분 training per instance** — 실시간 불가
- 1D+t only
- 예측 horizon: 1 Tp (~1.2s)

**LWT-PINN**: WaveSplat과 유사한 explicit spectral decomposition
- 주파수 고정, 진폭+위상 최적화
- 1D, linear only
- 0.13s 계산, near real-time
- 예측: 13.7s high-precision

**WaveSplat 대비**: 2D+t, 13ms, real-time → PINN 접근 대비 압도적 우위

---

## 2. 중요한 팩트 기반 재평가

### 2.1 Neural ODE Prediction 방향의 재검토

이전 제안: "Neural ODE가 Snl 등 spectral source terms를 학습하여 30-60초 예측 개선"

**NLML 논문이 밝힌 사실**: Snl 시간 스케일 = 시간~일.
→ 10-60초 예측에서 Snl은 **무시 가능**.

**그러나**, 이것이 Neural ODE를 무효화하지는 않음. Phase-resolved 예측에서 관련 비선형 효과는 Snl과 다름:

| 비선형 효과 | 시간 스케일 | Snl인가? | 10-60초 관련? |
|---|---|---|---|
| Bound harmonics (Stokes) | 즉시 | No | Yes (but 해석적) |
| 비선형 주파수 보정 | ~wave period | No | Yes (O(epsilon^2)) |
| Wave group modulation | ~10-100s | No | **Yes** |
| Phase coherence loss | ~30-60s | Partial | **Yes** |
| Spectral peak migration | hours | Yes (Snl) | No |
| Energy redistribution | hours-days | Yes (Snl) | No |

**10-60초 예측에서 Neural ODE가 배울 수 있는 것**:
- Wave group modulation (amplitude envelope 변화)
- Phase coherence 손실 (선형 모델이 과신하는 부분)
- 이들은 Snl이 아닌, **finite-bandwidth spectral interference** 효과

**정직한 평가**: 이 효과들의 크기는 아직 모름. HOS vs linear prediction 비교 실험이 필요.

### 2.2 KalmanNet 방향의 강화된 근거

KalmanNet이 WaveSplat tracking에 적합한 이유 (논문 본문 기반):

1. **Model mismatch handling**: WaveSplat의 linear model은 현실과 mismatch.
   KalmanNet은 모델 mismatch 시 EKF 대비 **11.56 dB gain** 달성.

2. **비선형 observation model**: h(x) = Sigma w_i cos(...) 는 비선형.
   KalmanNet은 비선형 h에서 **11.83 dB gain** (Lorenz 실험).

3. **계산 효율**: matrix inversion 없음 → EKF보다 빠름.
   현재 Adam 10 steps → KalmanNet 1 forward pass.

4. **Uncertainty**: Kalman framework = 자연스러운 posterior covariance.

5. **Per-component 적용**: m=2, n=1 → 매우 작은 네트워크 (~수천 params).

---

## 3. Novelty 검증

### 3.1 KalmanNet + Wave Estimation

- "KalmanNet ocean wave" → **0편**
- "learned Kalman filter wave estimation" → **0편**
- "neural Kalman spectral tracking" → **0편**
- 가장 가까운: KalmanNet NCLT localization (차량 위치 추적) — 완전히 다른 도메인

**Novelty 확인**: KalmanNet을 wave spectral tracking에 적용한 연구 없음.

### 3.2 Neural ODE + Wave Spectral Prediction

- "neural ODE wave prediction" → **0편** (spectral params 레벨)
- "neural ODE spectral evolution ocean" → **0편**
- 가장 가까운: 1D-FNO-ResNet (Ocean Eng 2024) — spatial domain, not parametric

**Novelty 확인**: Neural ODE로 spectral wave params를 예측한 연구 없음.

### 3.3 경쟁자/선점 위험

| 그룹 | 방향 | 위협? |
|---|---|---|
| Ehlers (Hamburg) | PINN/PINO for wave reconstruction | No — 실시간 불가, 1D only |
| LANL (NLML) | NN for Snl in WW3 | No — dense grid, hours timescale |
| Shlezinger (KalmanNet) | learned Kalman for signal processing | **Possible** — 하지만 wave 미적용 |
| Alvarez-Lopez | Neural ODE for distributions | No — no forecasting, different problem |

**선점 위험**: 낮음. Wave spectral estimation/prediction은 이 그룹들의 관심 밖.

---

## 4. 방법론 검토: 우리 문제에 맞는가?

### 4.1 KalmanNet for WaveSplat Tracking

**현재 tracking** (sequential_update.py:321-373):
```python
# Prediction
self.model.phi.add_(omega * dt)
# Update: Adam 10 steps on MSE loss
for step in range(self.tracking_steps):
    self.optimizer.zero_grad()
    loss = self._compute_loss([frame])
    loss.backward()
    self.optimizer.step()
```

**KalmanNet 대체 설계**:
```python
# Prediction (동일)
phi_pred = phi + omega * dt
w_pred = w  # 선형 가정

# Observation
z_pred = forward_model(w_pred, kx, ky, phi_pred, frame)
innovation = z_obs - z_pred

# KalmanNet Update (Adam 10 steps 대체)
K_t = kalmannet(innovation, prev_features)  # single forward pass
state_update = K_t @ innovation_features
w_new = w_pred + state_update[:N]
phi_new = phi_pred + state_update[N:]
```

**구현 이슈**:
- Observation dim (M=500-2000 LiDAR points) >> state dim (2N=30)
  → innovation을 그대로 쓸 수 없음. Feature extraction 필요.
  → 해결: innovation을 per-component projection으로 압축
    `innov_i = <z_residual, cos(theta_i)>, <z_residual, sin(theta_i)>`
    → n=2 per component. 이것은 linear projection과 동일.

- **결론: per-component KalmanNet, m=2, n=2가 가장 자연스러움**
  → 각 component i: state=(w_i, phi_i), obs=(cos_proj, sin_proj)
  → Component 간 coupling: residual 재계산으로 처리

**Training data**: wavesim에서 GT spectral params trajectory 생성 (수천 episode)

### 4.2 Neural ODE for Wave Prediction

**설계**:
```python
# State: theta = (w_1, phi_1, ..., w_N, phi_N)
# Known dynamics:
def f_physics(theta, t):
    dphi = omega  # dispersion
    dw = 0        # constant amplitude
    return (dw, dphi)

# Learned correction:
def f_nn(theta, t, nn_params):
    return mlp(theta, t)  # small MLP

# Combined ODE:
def f_combined(theta, t):
    return f_physics(theta, t) + f_nn(theta, t, nn_params)

# Integration:
theta_future = odeint(f_combined, theta_0, t_span)
```

**Training data**: HOS simulator → 비선형 evolution GT
- HOS는 이미 완성 (wavesim/hos_simulator.py)
- Run HOS for many episodes → extract spectral params at each dt
- Train: minimize |theta_predicted - theta_HOS|^2

**검증 필요 사항 (구현 전에 확인해야 할 것)**:
1. HOS vs linear prediction gap이 10-60초에서 얼마나 큰가?
   → 이것이 Neural ODE가 학습할 "신호"의 크기를 결정
   → Gap이 작으면 (예: nRMSE 차이 < 5%), Neural ODE의 가치 의문

2. Phase coherence time이 실제로 얼마인가?
   → 30초 이상에서 phase가 의미 없으면 prediction 자체가 무의미

---

## 5. 최종 판단

### KalmanNet for Tracking: 추천

| 기준 | 평가 | 근거 (논문 기반) |
|------|------|-----------------|
| Novelty | **확인됨** | wave estimation 적용 0편 |
| 기술적 적합성 | **높음** | m=2, n=2 per component, 논문에서 m=10까지 검증 |
| 개선 기대치 | **중-상** | model mismatch 시 11+ dB gain (논문 결과) |
| 구현 가능성 | **높음** | PyTorch 코드 공개, per-component 설계 자연스러움 |
| 추가 가치 | **uncertainty** | Kalman posterior covariance → 안전 시스템 활용 |

### Neural ODE for Prediction: 조건부 추천

| 기준 | 평가 | 근거 (논문 기반) |
|------|------|-----------------|
| Novelty | **확인됨** | spectral wave prediction 적용 0편 |
| 기술적 적합성 | **불확실** | Snl은 무관 (시간~일 스케일), phase-resolved 효과 크기 미지 |
| 개선 기대치 | **미확인** | HOS vs linear gap 실험 필요 |
| 구현 가능성 | **높음** | torchdiffeq, HOS 데이터 즉시 사용 |
| 선행 검증 필요 | **필수** | HOS-linear gap 정량화 먼저 |

### 추천 진행 순서

```
Step 1: HOS vs linear prediction gap 실험 (1-2일)
  → HOS 시뮬레이션 실행 → spectral params 추출
  → Linear prediction (phi += omega*dt) vs HOS GT 비교
  → 10s, 30s, 60s에서 gap 측정
  → Gap < 5%면: Neural ODE 가치 약함 → KalmanNet만 진행
  → Gap > 10%면: Neural ODE 가치 있음 → 둘 다 진행

Step 2: KalmanNet prototype (1-2주)
  → Per-component KalmanNet (m=2, n=2)
  → wavesim training data 생성
  → WaveSplat tracking 대체 실험

Step 3 (conditional): Neural ODE prototype
  → HOS training data
  → Physics-residual Neural ODE
  → Linear prediction baseline 대비 비교
```
