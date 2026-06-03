# KalmanNet × WaveSplat Tracking — Pilot Design (2026-06-03)

## 0. 배경

검토 의견 *"큰 갈래"* 에 대응, vanilla TNP 실패 (test nRMSE 0.693) 이후 *paradigm 전환* 의 1순위 후보.

### Strategy A — *KalmanNet 단독* (Init 그대로)
- **Init**: WaveSplat v2_konly (기존 그대로, nRMSE 0.488 → 0.141 검증됨)
- **Tracking**: KalmanNet (Feng Wang 2025 ETKF 의 진화)

### 왜 Init 그대로?
1. *추정오차 (20%) ≫ 비선형 효과 (5%)* — 추정오차 줄이는 게 우선
2. Direction B 의 nonlinear init 시도 (Stokes2, Neural corrector) 모두 *추정오차에 묻혀 실패*
3. **순서**: Tracking 추정오차 ↓ (KalmanNet) → 비선형 효과 visible → 비선형 보정 (Stage 2)

---

## 1. KalmanNet 핵심 (Revach et al. 2022, IEEE TSP)

### Classical Kalman Filter
```
Predict:  x_t|t-1 = f(x_{t-1}|t-1)
          P_t|t-1 = F·P·Fᵀ + Q          ← Q 필요
Update:   K_t = P_t|t-1·Hᵀ(H·P·Hᵀ + R)⁻¹  ← R, matrix inversion 필요
          x_t|t = x_t|t-1 + K_t·(y_t - h(x_t|t-1))
```

### KalmanNet 변형
- **f(·), h(·)**: *partial knowledge 로 그대로 사용* (우리 경우 dispersion advance + LiDAR forward)
- **Q, R**: *unknown, RNN 이 흡수*
- **K_t**: *작은 RNN 으로 학습* → matrix inversion bypass, nonlinearity 자동 처리

### 핵심 4가지 features (논문 III-A)
| Feature | 의미 | WaveSplat 매핑 |
|---|---|---|
| F1: y_t - y_t|t-1 | Observation innovation | LiDAR η_obs - η_pred at obs points |
| F2: y_t - y_{t-1}|t-1 | Forward observation diff | t-direction 차분 |
| F3: x_t|t-1 - x_{t-1}|t-1 | State update diff | (Δw, Δφ) from previous step |
| F4: x_t|t-1 - x_{t-1}|t-2 | Forward state evolution | Phase advance term |

---

## 2. WaveSplat Tracking 의 현재 구조

### `sequential_update.py:update()` (line 334-386)
```python
def update(self, frame):
    # Step 1: Phase prediction (kinematic preview)
    omega = self.model._omega()
    self.model.phi.add_(omega * dt)        # φ ← φ + ω·Δt
    
    # Step 2: Buffer update (3-frame sliding window)
    self._frame_buffer.append(frame)
    
    # Step 3: Update via Adam gradient descent
    for step in range(10):                  # ~13 ms / frame
        loss = self._compute_loss(buffer)
        loss.backward()
        adam.step()                          # implicit Kalman gain
```

여기서 *Adam optimizer = implicit Kalman gain* (적응형 step size).

### State Space Model 매핑

```
State x_t = {(w_n, φ_n)}_{n=1..N}    (k_x, k_y, ω fixed from init)
            → dim = 2N, N ∈ {1, 2, 5, 15, 20, 30}
            → padded to N_max=30 → state dim 60

Dynamics:   x_t|t-1 = f(x_{t-1})
              w 그대로
              φ_n ← φ_n + ω_n · Δt     ← analytical, dispersion-based

Observation: y_t = h(x_t) + v_t
              η_pred(x_i, y_i, t_i) = Σ_n w_n cos(k_xn·x + k_yn·y - ω_n·t + φ_n)
              at ~50 LiDAR points
              dim = ~50 (variable, padded to 71 max)
```

---

## 3. KalmanNet × WaveSplat Architecture

### KalmanNet update step

```python
def kalmannet_update(model, frame, hidden_state):
    # === Step 1: Predict (그대로) ===
    omega = model._omega()
    model.phi.add_(omega * dt)
    
    # === Step 2: Innovation features ===
    eta_pred = forward_lidar(model, frame.x_obs, frame.y_obs, frame.t)
    innovation = frame.eta_obs - eta_pred       # ← residual at obs points
    
    # === Step 3: KalmanNet RNN gain ===
    state_features = compute_state_features(model)  # F3, F4
    obs_features   = innovation                       # F1
    
    delta_state, hidden_state = kalmannet(
        innovation_features = obs_features,
        state_features      = state_features,
        hidden              = hidden_state,
    )
    # delta_state: (60,) = (Δw[N_max], Δφ[N_max]) with mask
    
    # === Step 4: Apply state update ===
    n = model.n_gaussians
    model.w.add_(delta_state[:n])
    model.phi.add_(delta_state[N_max:N_max+n])
    
    return hidden_state
```

### Architecture detail

```
Input encoder (FC layers):
  - Innovation (~50 dim, padded) → 64 dim embedding
  - State features (60 dim) → 64 dim embedding
  
GRU layer:
  - Hidden state: 128 dim
  - Single layer (또는 2 layer)
  
Output decoder (FC layers):
  - 128 → 60 (Δw, Δφ for N_max=30)
  - 마지막 layer 출력에 mask 적용 (active components 만)
  
Total params: ~50K (TNP 의 1.16M 보다 훨씬 작음)
```

### Loss function

```python
# Training mode: trajectory loss
loss = Σ_t || x_t|t - x_t^GT ||²
     = Σ_t (||Δw||² + ||Δφ||²)

# 또는 observation likelihood:
loss = Σ_t || y_t - h(x_t|t) ||²
```

---

## 4. Dataset 설계

### Task = tracking sequence

```
Per episode (301 frames):
  - 첫 ~30 frames: Init phase (skip)
  - 나머지 ~270 frames: tracking sequences
  - Subsequence length: 20 frames (KalmanNet 학습에 충분)
  - Stride: 10 frames → ~27 sequences/episode

110 episodes × 27 = ~3000 training sequences
```

### Per-sequence data

```
Input:
  - 20 frames of LiDAR observations (~50 pts each)
  - Init state from WaveSplat init (k_x, k_y fixed; w, φ initial)
  
Target:
  - GT (w, φ) at each frame (from spectral_gaussian fit to GT field)
  
또는 simpler:
  - GT η field at LiDAR points (observation likelihood)
```

### Train/Val/Test split

```
Episode-level split (data leakage 방지):
  Train: 70 episodes (BF2-BF5 main sweep, seed 0-3)
  Val:   20 episodes (seed 4 of BF2-BF5)
  Test:  20 episodes (BF6-BF7 + held-out)
```

---

## 5. Implementation Plan

### Day 1-2: Dataset + Init coupling
- `scripts/kalmannet_dataset.py` — HDF5 → tracking sequences
- WaveSplat init 실행 후 그 state 를 KalmanNet 시작점으로
- Sanity: 한 episode 에서 init → tracking sequence 추출

### Day 3-4: KalmanNet model
- `wavesplat/optimizer/kalmannet.py` — GRU-based gain module
- Architecture: FC encoder → GRU → FC decoder
- Forward pass + backward pass 검증

### Day 5: Sanity test
- `scripts/kalmannet_sanity.py` — forward + step + loss decrease
- 1 episode 의 1 sequence 로 overfit 시도

### Day 6-7: 결과 보고서 + git commit

---

## 6. Go/No-Go Gates

### Stage 1 (Sanity): Day 5
- [ ] Forward pass OK
- [ ] Backward + Adam step 가능
- [ ] Single sequence overfit (NLL ↓)

### Stage 2 (Single-episode overfit): Future
- [ ] 1 episode 의 모든 sequence 로 overfit
- [ ] Tracking nRMSE < 0.1 도달 가능?

### Stage 3 (Generalization): Future
- [ ] 70 ep train → 20 ep test
- [ ] Test nRMSE < 0.20 ?
- [ ] Feng Wang ETKF (~0.32) 명확 우위?

---

## 7. 예상 Challenges

| 위험 | 가능성 | Mitigation |
|---|---|---|
| Variable N (1~30 components) 처리 | HIGH | N_max padding + mask |
| Init quality 의존성 | MED | v2_konly 검증된 init 사용 |
| Sequence length 학습 안정성 | MED | 짧은 sequence (20 frames) + GRU 표준 |
| GPU 부담 | LOW | 50K params, GPU 큰 부담 없음 |
| Observation 비균일 (50 pts variable) | MED | Padding + mask |

---

## 8. Expected Outcomes

### 최선 (Stage 3 통과):
- Test nRMSE 0.15-0.18 (Feng Wang 0.32, WaveSplat 0.22 능가)
- *Sequential learning + dispersion + sparse* 동시 성공
- PhD paper의 main contribution

### 중간 (Stage 2 만 통과):
- 1 episode overfit OK, generalization 약함
- *방법 자체는 valid*, scaling 또는 architecture 개선 필요

### 최악 (Stage 1 부터 fail):
- 학습 자체 어려움 — sparse observation 으로 60-dim state 학습 불충분
- Fallback: DKL-GP 또는 ConvCNP hybrid

---

## 9. 코드 파일

| 파일 | 역할 |
|---|---|
| `wavesplat/optimizer/kalmannet.py` | KalmanNet GRU 모듈 |
| `scripts/kalmannet_dataset.py` | Tracking sequence generator |
| `scripts/kalmannet_sanity.py` | Stage 1 sanity test |
| `docs/kalmannet_pilot_design_20260603.md` | 본 설계 문서 |
| `docs/kalmannet_stage1_report_20260603.md` | Stage 1 결과 (작성 예정) |
