# HNN for Ocean Wave Spectral Evolution — Fact-Based Synthesis Report

**Date**: 2026-05-19
**Purpose**: 참조 논문 원문 검토 후 novelty/방법론 최종 판단
**Verdict**: **조건부 추천 — 방향 재정의 필요**

---

## 1. 검토한 논문 요약

| Paper | Core Idea | Scale | Key Limitation |
|-------|-----------|-------|----------------|
| Greydanus 2019 (HNN) | MLP로 H 학습, Hamilton eq.으로 dynamics 유도 | 2-12D (ideal spring~three-body) | 12D에서 이미 어려움 |
| sPHNN (NeurIPS 2025) | Port-Hamiltonian + FICNN(convex H) | **40D** (partial obs.) | H가 convex여야 함 |
| Zakharov eq. (arXiv 2401.03539) | 해양파 Hamiltonian 정식화 | 이론적 무한 | finite depth 커널 singularity |
| NLML (LANL 2026) | black-box DNN으로 Snl 학습 | WAVEWATCH III scale | Hamiltonian 구조 없음 |
| Symplectic Neural Operators (May 2026) | Hamiltonian PDE용 neural operator | PDE scale | 매우 최근 — scoop 위험 |

## 2. WaveSplat 현재 시스템 팩트 (코드 기반)

```
Forward model:  z(x,y,t) = Σᵢ wᵢ·cos(kxᵢ·x + kyᵢ·y - ωᵢ·t + φᵢ)
Dispersion:     ωᵢ = √(g·|kᵢ|·tanh(|kᵢ|·d))     ← hard constraint
Tracking:       φᵢ += ωᵢ·Δt  →  Adam 10 steps (w, φ only)
State space:    N=15 components × (w, kx, ky, φ) = 60D (tracking시 30D)
Observation:    매 프레임 LiDAR 점군 → gradient update
```

## 3. 핵심 팩트 체크 결과

### 3.1 비선형 효과 크기: O(ε²) ≈ 0.25-1%

Zakharov 이론에 따른 비선형 주파수 보정:
```
Ωᵢ = ωᵢ + Tᵢᵢᵢᵢ|bᵢ|² + 2Σⱼ≠ᵢ Tᵢⱼᵢⱼ|bⱼ|²
```

전형적인 해양파 steepness ε = Hs·k/2 ≈ 0.05-0.10에서:
- 주파수 보정 = O(ε²) ≈ **0.25-1%**
- 현재 WaveSplat nRMSE = 22.3% → 비선형 보정의 영향 << tracking noise

**결론**: 비선형 효과는 현재 estimation 오차 대비 무시 가능할 정도로 작다.

### 3.2 비선형 보정은 해석적으로 이미 알려져 있음

Zakharov 커널 Tᵢⱼₖₗ은 해석적 공식이 존재:
- Deep water: Krasitskii (1994) 커널
- Finite depth: 활발한 연구 중 (singularity 문제)

→ **NN이 학습할 "미지의" 물리가 사실상 없다.** 알려진 공식을 직접 구현하면 된다.

### 3.3 HNN 스케일링 문제

| Method | Max tested dim | WaveSplat state dim | Feasible? |
|--------|---------------|---------------------|-----------|
| Standard HNN | 12D | 30-60D | **No** |
| RO-HNN | 600D (via 3-10 latent) | 30-60D | Maybe (lossy reduction) |
| sPHNN | 40D | 30D (w,φ only) | **Marginal** |

sPHNN가 유일하게 규모가 맞지만:
- **convex H 가정**: H_linear = Σ ωᵢ|bᵢ|²는 convex. 그러나 H_NL (4-wave interaction)의 convexity는 보장 안 됨
- Partial observation 처리는 가능 (augmented latent)

### 3.4 에너지 보존의 실질적 가치

WaveSplat tracking loop:
```
매 프레임 (0.5-1.0초 간격):
  1. φ += ω·Δt          (물리적 예측)
  2. Adam 10 steps       (관측 데이터로 보정)
```

**관측이 매 프레임 들어오므로**, NN의 에너지 보존이 drift를 방지할 시간적 여유가 거의 없다.
Adam gradient step이 이미 관측 기반 보정을 수행.

→ **estimation 문제에서 Hamiltonian 구조의 실질적 이득 = 미미**

### 3.5 Novelty 확인

- "HNN for ocean wave spectral evolution" 관련 논문 = **0편** (confirmed)
- 가장 가까운 경쟁: NLML (black-box DNN, no Hamiltonian), Symplectic Neural Operators (PDE level, 2026 May)
- Novelty는 존재하지만, **novelty가 있다고 해서 좋은 연구 방향은 아님**

## 4. 정직한 판단

### HNN-WaveSplat (원래 제안)에 대한 평가

| 기준 | 점수 | 이유 |
|------|------|------|
| Novelty | ★★★★☆ | 기존 논문 0편, 확실한 novelty |
| 물리적 의미 | ★★★☆☆ | Hamiltonian 구조는 아름답지만 실질 효과 미미 |
| 실용적 개선 | ★☆☆☆☆ | 비선형 효과 0.25-1% vs 현재 오차 22% |
| 기술적 실현 가능성 | ★★☆☆☆ | 30-60D에서 sPHNN만 가능, convexity 미보장 |
| 논문 스토리 | ★★★☆☆ | "왜 NN이 필요한가?"에 대한 답변이 약함 |

**총평**: Novelty는 있으나, "해석적으로 이미 알려진 0.25-1% 비선형 보정을 왜 NN으로 학습하는가?"에 대한 설득력 있는 답이 없다.

### 근본적 문제

> **"관측이 매 프레임 들어오는 estimation 문제에서,
> 프레임 간 temporal evolution의 정확성 개선은
> 관측 데이터 자체의 보정력에 비해 미미하다."**

이것은 HNN의 문제가 아니라, **estimation 문제 자체의 특성**이다.

## 5. 대안 제안: Estimation → Prediction 확장

HNN/Hamiltonian 구조가 **진짜 의미 있는** 곳: **관측 없이 미래를 예측할 때**

### 5A. WaveSplat-Predict: 단기 파랑 예측

```
Phase 1: WaveSplat estimation (현재 시스템, 관측 기반)
  → spectral params {wᵢ, kxᵢ, kyᵢ, φᵢ} 추정

Phase 2: Forward prediction (관측 없음, 10-60초 미래)
  → Hamiltonian evolution으로 파고장 예측
  → 비선형 효과가 누적되므로 HNN 가치 증가
```

**왜 prediction에서는 HNN이 의미 있는가:**
- 관측 보정 없음 → energy conservation이 drift 방지에 핵심
- 30-60초 예측 시 비선형 효과 누적 → O(ε²·T) 만큼 커짐
- 실용적 가치: 선박 운항 중 "앞으로 30초 후 파고" 예측 → 안전 시스템

**논문 스토리:**
> "LiDAR로 추정한 spectral params를 Hamiltonian 구조로 시간 진화시켜
> 단기 파랑장을 예측한다. Energy conservation이 비물리적 발산을 방지하고,
> 비선형 상호작용을 학습하여 linear prediction 대비 개선한다."

### 5B. 비선형 WaveSplat (Stokes 2nd order) — NN 없이

```
z(x,y,t) = Σᵢ wᵢ·cos(θᵢ) + ΣᵢΣⱼ Bᵢⱼ·cos(θᵢ ± θⱼ)
```

- Bᵢⱼ는 wᵢ, wⱼ, kᵢ, kⱼ, depth로부터 해석적으로 계산
- **추가 free parameter 없음** (bound harmonics)
- Steep wave (BF5-6)에서 estimation 개선 기대
- NN 불필요, 순수 물리 확장
- 논문 한 편 분량 (estimation 개선 + ablation)

### 추천 조합

```
PhD 구조:
  Chapter 1: WaveSplat (현재, estimation)        — OCEANS 2026
  Chapter 2: Stokes 2nd order 확장 (estimation)  — 추가 저널 논문
  Chapter 3: Hamiltonian prediction (HNN/sPHNN)  — 핵심 PhD 기여
```

- Chapter 2는 NN 없이 순수 물리 → 빠르게 완료 가능
- Chapter 3에서 HNN이 진짜 의미 있는 문제(prediction)에 적용
- Prediction 문제에서는 "왜 NN인가?"에 대한 답이 명확:
  - 해석적 Zakharov 커널은 finite depth에서 singularity
  - NN이 effective kernel을 data-driven으로 학습
  - Energy conservation으로 장시간 안정성 보장

## 6. 최종 결론

| | 원래 방향 (HNN-Estimation) | 수정 방향 (HNN-Prediction) |
|---|---|---|
| NN 필요성 | 약함 (해석적 해 존재) | **강함** (finite depth 커널 미해결) |
| 물리적 이득 | 0.25-1% | **누적되어 의미 있음** |
| 실용적 가치 | 미미 | **선박 안전 시스템** |
| 논문 스토리 | 약함 | **강함** |
| 기술적 난이도 | 높음 (30-60D HNN) | 동일 |

**결론: HNN을 estimation에 붙이는 것은 추천하지 않는다.**
**대신 WaveSplat estimation → Hamiltonian prediction 파이프라인으로 방향을 재정의하면,
novelty + 물리적 의미 + 실용성이 모두 살아난다.**

당장 구현할 것: **Stokes 2nd order 확장** (Chapter 2, NN 없음, 빠르게 결과 확인)
이후 구현할 것: **Hamiltonian prediction** (Chapter 3, HNN/sPHNN 활용)
