# WaveSplat + Hydrofoil Feedforward — 발표용 종합 학습자료

> 박사 논문 2편(Ocean Engineering 투고)을 "기초부터 응용까지" 한 번에 읽고
> 이해하여 PPT를 만들고 교수님께 발표할 수 있도록 구성한 통합 가이드.
>
> 작성일: 2026-04-08 (최초)
> **최종 업데이트: 2026-04-10** — 논문이 full 100-ep / 6-metric panel / tuned ETKF sweep /
> dispersion residual / Cohen's d / 40p OE 원고로 갱신됨. 과거 나오던 "48% / 50 paired /
> 31.7 ms / 0.233·0.450·0.276" 수치는 구버전이며, **최신 수치는 아래 § WaveSplat 최신 수치
> 표 및 [PRESENTATION_SUMMARY.md](PRESENTATION_SUMMARY.md)를 우선**하세요.

---

## ⚡ WaveSplat 수치

> ⛔**2026-08-13 감사 — 이 표의 헤드라인 수치는 전량 철회됨. 외우지 말 것.**
> (금지 유지 · 3라운드에서 **사유 교체**)
> `0.223 / 0.319 / 30.1%` · `13.4 ms` · `132.6 ms` · `~10×` · `84/100` 인용 금지.
>
> ⚠**1라운드가 적은 "결과 파일이 저장소에 존재하지 않는다" 는 거짓이었다**(그 감사관은
> `results/` 를 한 파일도 열지 않았다). 네 수치는 `results/b1_full_comparison.json`
> (**git 추적**, 커밋 `e57528c`, 100 ep, `etkf_config` 내장)에서 **전부 재도출된다**:
> `WaveSplat.avg_nRMSE` **0.2229** · `ETKF.avg_nRMSE` **0.3189** ·
> `WaveSplat.avg_inference_ms` **13.3760** · `ETKF.avg_inference_ms` **132.6387**.
>
> **금지가 유지되는 진짜 사유 3가지**:
> 1. **프레임 불일치** — `WaveSplat.n_frames` **120–220** vs `ETKF`·`GPR-1` **301**,
>    **100/100 에피소드 전부에서 WS 가 더 적다** ⟹ 두 팔이 **같은 프레임에서 채점된 적이 없다**.
>    ⟹ ⛔그 파일의 **`84/100` 승패비도 인용 금지**. (S13 배치는 통과 — 불일치 0/128.)
> 2. **기준선 선택** — 그 ETKF 는 `grid 40×30 · ne=100 · mult_inflation 1.0`. 같은 `WaveSplat`
>    열을 튜닝 LETKF 와 붙이면 아래 표대로 **널**이다.
> 3. **지연은 프로세스 간 비교** — 정직값은 동일 프로세스·1스레드 **1.5–2.6×**.
>
> 근거: `reference_citation_bans.md`, `tmp/AUDIT_recheck_2026-08-13/`

**같은 `WaveSplat` 열을 튜닝 LETKF 와 붙인 재도출** (`results/100ep_ws_letkf.json`, 100 에피소드):

| 항목 | 값 | 비고 |
|------|-----|------|
| nRMSE (100 ep) | WS **0.2229** / LETKF **0.2278** | **51/49, p = 0.9204 → 널 (차이 없음)** |
| Speed | 동일 프로세스·1스레드에서 **1.5–2.6×** | CPU = **AMD Ryzen 7 7800X3D**. 단일 숫자 대신 중앙값+IQR 병기 |
| Out-of-sample 승패 | S13 704런: 혼잡 **56/64** · 희소 **63/64** | 라우터 미노출 시드 |
| 단위시험 | **204 passed, 0 failed, 0 skipped** | 85 wavesim + 119 wavesplat (2026-08-13 실측; 신규 8 = `test_shipping_defaults.py`) |

⛔**RETRACTED (2026-08-13 감사)** — 원 표에 있던 다음 값은 재실행 전 인용 불가:
`nRMSE 0.223/0.319/0.261 (30.1% 개선)` · `Speed 13.4/132.6/>200 ms (~10×)` ·
`Skill Score +0.034/−0.702/−0.140` · `Variance Ratio 0.66/0.90/0.065` ·
`Pearson ρ 0.45/0.11/0.19` · `Dir R² −0.25/+0.07/+0.03` ·
`Wilcoxon p=1.1×10⁻¹⁵ (d=1.16)` · `p=4.7×10⁻⁴ (d=0.42)` · `Raydrop 20/20 wins` ·
`LETKF 12ep 비교 0.211/0.238/0.267 (6/7 지표 1위)` · `LETKF Dir R² 5.8×` ·
`LETKF Speed 13.2 vs 115.7 ms (8.8×)` · `SS Bootstrap CI`.
⚠**사유는 "재현 불가" 가 아니다**(3라운드 교체) — 100-에피소드 계열
(`nRMSE`·`Speed`·`Skill Score`·`Variance Ratio`·`Pearson ρ`·`Dir R²`)은
`results/b1_full_comparison.json` 에서 **재도출되며 프레임 불일치·기준선 선택 결함을 물려받고**,
지연은 **프로세스 간 비교**다. `Wilcoxon p/d`·`Raydrop 20/20`·`LETKF 12ep` 계열만 그 파일에서
**그대로 재현되지 않는**(구버전 12-에피소드) 값이다. 근거: `reference_citation_bans.md`

유지되는 항목(위 3사유와 무관): Dispersion drift (A1) `R̄=0.51 ± 0.12 vs Full=0` ·
ETKF sweep `24 configs, best 0.358 / worst 1.187 (3.3×)` ·
Sparsity `< 8% degrade at 20% obs density` · Noise `±13% under σ=0.10 m`.

---

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

- **`v2_konly` 방식**: 빔포밍은 k만 결정, ω와 φ는 후속 최적화에 위임.
  ⛔단 `v2_konly` 는 **출하 기본값이 아니다** — 2026-07-26 flip 이후 기본값은 GT-free 라우터
  **`auto`** (`wavesplat/run.py:43` · `configs/estimation.yaml:24`). 명시 지정 시에만 쓰인다.
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

> ⛔**RETRACTED (2026-08-13 감사, 3라운드 사유 교체)** — 이 표에 있던
> `WS 0.223 / 13.4` · `ETKF 0.319 / 132.6` · `GPR 0.261 / >200`,
> 그리고 "nRMSE 30.1% 개선, ~10× 속도 우위", "73.7 Hz 동작" 은 인용 금지다.
> ⚠단 사유는 "재현되지 않는다" 가 **아니다** — 이 값들은 `results/b1_full_comparison.json`
> (커밋 `e57528c`, 100 ep)에서 재도출된다. 금지 사유는 ①**프레임 불일치**
> (`WaveSplat.n_frames` 120–220 vs `ETKF`·`GPR-1` 301, **100/100 에피소드에서 WS 가 더 적음**
> ⟹ `84/100` 승패비도 금지) ②**기준선 선택**(`grid 40×30 · ne=100 · mult_inflation 1.0`)
> ③**프로세스 간 지연 비교**다. 근거: `reference_citation_bans.md`

**같은 `WaveSplat` 열을 튜닝 LETKF 와 붙인 재도출** (`results/100ep_ws_letkf.json`, 100 ep):

| 방법 | 평균 nRMSE | 판정 |
|---|---|---|
| WaveSplat | 0.2229 | — |
| LETKF | 0.2278 | **51/49, p = 0.9204 → 널** |

- **nRMSE**: ⛔"30.1% 개선" 은 근거 없음 — 이 표본에서 차이는 **통계적으로 널**이다.
- **지연**: 동일 프로세스·1스레드에서 **1.5–2.6×** (CPU = AMD Ryzen 7 7800X3D).
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

| # | 제목 | 핵심 수치 / 그림 | 출처 |
|---|------|----------------|------|
| 1 | Title | OE 투고, 저자, 날짜 | — |
| 2 | Motivation | 하이드로포일 승차감 ↔ wave preview 필요성 | Intro §I-A |
| 3 | Big Picture | 두 논문 통합 파이프라인 + one-liner | Part D |
| 4 | Wave Physics 101 | η=Σa cos(kx−ωt+φ), **ω²=g\|k\|tanh(\|k\|d)** | Part A |
| 5 | LiDAR observation | scatter, "~50 pts/frame in 200×150 m" | graphical_abstract Panel 1 |
| 6 | Prior Art & Gap | ETKF/GPR/NN 한계 표 | §II |
| 7 | **Paper 1: Key Idea** | z=Σwᵢ cos(...), dispersion hard, **5N→4N** | §III-A |
| 8 | Pipeline | init(beamform→C2F→prune→grow) + track(predict+update) | Alg table |
| 9 | Why freeze k (w_phi) | 단일 프레임 ill-conditioned, 200→10 steps | §III-D |
| 10 | Dataset + Pre-registered metrics | 100 ep, BF2~5, U/S, **6-metric panel** 정의 | §IV |
| 11 | Tuned ETKF fairness | 24-config sweep, best 0.358 / worst 1.187 (3.3×) | Table ETKF sweep |
| 12 | **Headline: 6-metric Table** | **SS: WS +0.034 유일 양수**, VR/ρ로 ETKF/GPR 실패 모드 진단 | Table II |
| 13 | Speed & Statistics | ⛔RETRACTED — `13.4 vs 132.6 ms (~10×)` · `p=1.1e−15` · `d=1.16`. 재측정 후 "1스레드 1.5–2.6×" 로 재작성 | §V-B,C |
| 14 | A1 dispersion ablation (Q4) | A1 nRMSE 16% 낮지만 **R̄=0.51, 0.10 Hz off** → 물리 유효성의 대가 | §V-E |
| 15 | Honest limits | VR=0.66 under-fit, **Dir R²=−0.25** → hydrofoil BF4-head 연결 | §V-A |
| 16 | **Paper 2: Hydrofoil FF** | 2DOF 문제, FF+FB, f_c=0.8 Hz | Hydro §III |
| 17 | LP α sensitivity (Q8) | BF4 Head: α=**0.01** optimal(+13.7%), α=0.05에서 −1.1% | Hydro Table |
| 18 | Accel RMS results | BF3 head 50%, BF4 head 42%, BF4 bow 38% | Hydro §V |
| 19 | **Integrated Contribution** | sensing → info → action, Dir R² ↔ BF4-head pairing | Both |
| 20 | Limitations & Future | sim-only, Dir R² fix, real LiDAR, Stokes-2, current | Both Limits |

**보조 슬라이드 (Q&A 대비용)**:
- A1: Hyperparameter sensitivity (±5%)
- A2: Noise robustness (±13% under σ=0.10 m)
- A3: ETKF sweep 전체 24-config 랭킹
- A4: Dispersion residual 공식 + 10-run 통계
- A5: Sparsity 표 (20~100%)
- A6: Raydrop 표 — ⛔`20/20 wins` 수치는 철회됨(구버전 12-에피소드 계열)
- A7: Per-episode Table I (12 eps)

## 15. 예상 질문 & 답변

**Q1. "Grid-free라고 하는데 결국 컴포넌트 N개를 정해야 하지 않나?"**
A. N은 adaptive입니다. 에너지 기반 pruning과 잔차 triggered growth가
N을 시나리오별(BF2 단일 성분 → BF5 30 성분)로 자동 조절합니다.

**Q2. "분산관계 hard constraint가 nRMSE를 악화시키는데 왜 쓰나?"** (최신 Q4)
A. 네, A1 (no dispersion) ablation은 실제로 16% 더 낮은 nRMSE를 보입니다.
하지만 A1에서 학습된 (k, ω) 쌍의 **분산 manifold 이탈량 R̄ = 0.51 ± 0.12**
(≈0.10 Hz off)이고, 50% 벗어난 ω로는 φ←φ+ωΔt 기반 phase propagation이
1초 내에 de-cohere합니다. 즉 A1의 16%는 **nRMSE를 위해 물리적 유효성을
포기한 대가**이며, hydrofoil feedforward처럼 1초 후 파면을 예측해야 하는
downstream에 쓸 수 없습니다. 논문 §V-E의 Q4 evidence 참조.

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
- [ ] 논문 핵심 수치 — ⛔**2026-08-13 감사: 아래 구 목록은 외우면 안 된다.**
  - ⛔RETRACTED: `nRMSE WS 0.223 / ETKF 0.319 → 30.1% 개선` ·
    `Speed WS 13.4 ms / ETKF 132.6 ms → ~10× 빠름` · `84/100 승` ·
    `SS +0.034 (only positive)` · `Wilcoxon p=1.1e−15 (Cohen's d=1.16)`.
    ⚠사유(3라운드 교체): 이 값들은 `results/b1_full_comparison.json` 에 **있다** —
    금지 이유는 **프레임 불일치**(WS 120–220 vs ETKF/GPR 301 프레임, 100/100 에피소드) ·
    **기준선 선택**(튜닝 LETKF 로 바꾸면 널) · **프로세스 간 지연 비교**다.
    근거: `reference_citation_bans.md`
  - ✅검증된 값: **WS 0.2229 vs LETKF 0.2278, 51/49, p = 0.9204 (널)**,
    지연 **1스레드 1.5–2.6×** (AMD Ryzen 7 7800X3D), 단위시험 **204 passed**
  - Dispersion drift: A1 R̄=0.51±0.12 vs Full=0 (유효)
  - Hydrofoil 가속도 저감: BF3 head 50%, BF4 head 42%, BF4 bow 38%

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

---

# Part E — 학습·이해를 위해 해야 할 것 (발표 방어력 강화)

> 논문 내용 자체는 Part A~D에서 설명되었습니다.
> 이 Part E는 **교수님 발표에서 질문이 들어왔을 때 흔들리지 않기 위해**
> 추가로 읽고, 손으로 해보고, 외워야 할 것을 정리합니다.

## E-0. 2026-04-10 업데이트 이후 새로 방어해야 할 질문 3개

최신 논문(full 100 ep / 6-metric panel / tuned ETKF sweep / dispersion residual)
이후 추가로 대비해야 할 질문:

**Q-A. "Skill Score가 cherry-picked metric이 아닌가? WaveSplat에게 유리한 걸 고른 게 아닌가?"**
A. SS는 **사전 등록(pre-registered)** 6개 메트릭 중 하나이고, 선택 시점이 실험 전입니다.
증거로 우리는 같은 panel 위에서 Dir R²=−0.25라는 **우리 자신의 약점**을 동시에 보고합니다
(§V-A 말미). Cherry-picking이면 이 약점은 숨겼을 것입니다. 더 중요한 건 SS의 해석:
`SS = 1 − MSE(ẑ)/MSE(0)`에서 ẑ=0일 때 SS=0이므로, **SS ≤ 0은 "zero predictor보다
나쁘다"는 뜻**. ETKF/GPR이 여기서 음수라는 사실은 이들이 zero prediction에도 지는
estimator라는 진단적 증거입니다.

**Q-B. "Tuned ETKF의 Nₑ=100이 너무 큰 것 아닌가? 공정한가?"**
A. 24-config sweep의 **top-10이 모두 Nₑ=100**이고, 40×30 grid가 80×60보다 일관되게
우수했습니다 (후자는 ensemble이 undersize). 즉 Nₑ=100이 ETKF 자체에 가장 유리한
설정입니다.
⛔RETRACTED (2026-08-13 감사, 3라운드 사유 교체) — 원문의 `Nₑ=50 pre-tuning 31.7 ms` 와
`nRMSE 0.436→0.319`. 지연은 **프로세스 간 비교**(스레드 미고정 포함)이고, `0.319` 는
**재현되지 않는 분모가 아니라** `results/b1_full_comparison.json` 의 값(`0.3189`)으로서
**프레임 불일치**(WS 120–220 vs ETKF 301 프레임, 100/100 에피소드)와 **기준선 선택**의
산물이다. 근거: `reference_citation_bans.md`
"ETKF 에게 유리한 방향으로 튜닝한 뒤 비교했다" 는 **설계 서술은 유효**하고, S13 에서
기준선을 더 강화했을 때 오히려 우리 보고값이 나빠졌다는 사실이 이를 뒷받침한다
(공개 LETKF 였다면 63/64, 튜닝 후 56/64).

**Q-C. "A1 dispersion ablation에서 `R̄=0.51`이 정확히 무슨 의미인가?"**
A. A1에서 학습된 각 성분 (kᵢ, ωᵢ)의 **진폭 가중 상대 편차**입니다:
`R̄ = Σ|wᵢ|·|ωᵢ−ωᵢ^disp|/(ωᵢ^disp · Σ|wⱼ|)`, ωᵢ^disp=√(g|kᵢ|tanh(|kᵢ|d)).
A1은 R̄=0.51±0.12 (절대값 ~0.10 Hz), Full은 0 by construction. 50% 벗어난 성분은
`Δφ = ω·Δt`로 1초 propagate하면 `Δφ_true·(1±0.5)`로 벌어져 phase 맞춤이 불가능해
frame 간 tracking이 무너집니다. 코드는 `scripts/run_q4_dispersion_residual.py`,
데이터는 `results/q4_dispersion_residual.json`에 있습니다.

---

## E-1. 읽어야 할 자료 (우선순위 순)

전체 읽지 말고 **Abstract → Figures → Conclusion** 순으로 1편당 30분.

| 우선순위 | 자료 | 읽을 부분 | 방어하는 질문 |
|---|---|---|---|
| ★★★ | **Holthuijsen (2007)** *Waves in Oceanic and Coastal Waters* | Ch.3 (선형파), Ch.5 (스펙트럼) | "분산관계 유도?", "JONSWAP이 뭐냐?" |
| ★★★ | **Faltinsen (2005)** *Hydrodynamics of High-Speed Marine Vehicles* | Ch.2 (운동), Ch.4 (수중익) | "heave/pitch 연성?", "foil lift?" |
| ★★ | **Desmars et al. (2023)** *Ocean Engineering* | Abstract + Method | "가장 가까운 기존 기법과 뭐가 다르냐?" |
| ★★ | **Ehlers et al. (2024)** *Ocean Engineering* | Method | "왜 딥러닝을 쓰지 않았나?" |
| ★★ | **Desmars et al. (2024)** *JFM* | Abstract + Conclusion | "nonlinear 확장 가능성?" |
| ★ | **Kingma & Ba (2015)** Adam | 1 페이지 요약 | "옵티마이저 선택 이유?" |
| ★ | **Capon (1969)** Beamforming | Section 1 | "빔포밍 원리?" |

## E-2. 손으로 반드시 해볼 것 3가지

- [ ] **분산관계 직접 유도** (A4 한 장)
  - Laplace eq → free-surface BC (kinematic + dynamic) → ω² = g|k|tanh(|k|d)
  - 얕은물/깊은물 극한 각각 유도
  - **교수 질문 "보드에 써봐"에 즉시 대응 가능해야 함**
- [ ] **HDF5 에피소드 파일을 직접 열어 LiDAR 포인트 scatter plot**
  - `HANDS_ON.md`의 snippet 활용
  - 한 프레임에 몇 점, 어느 영역 커버, 어떻게 희소한지 눈으로 확인
- [ ] **한 에피소드 끝까지 돌려보고 핵심 수치 확인**
  - ⛔RETRACTED — `nRMSE 0.223, 13.4 ms, 30.1% 개선, ~10× 속도` 를 암기하지 말 것.
    근거: `reference_citation_bans.md`
  - ✅확인할 값: **WS 0.2229 vs LETKF 0.2278 (51/49, p = 0.9204 = 널)**,
    지연은 **1스레드 1.5–2.6×**

## E-3. 엘리베이터 피치 암기 (3가지 버전)

> ⛔**2026-08-13 감사 — 아래 세 피치는 모두 철회된 수치를 담고 있다.**
> `13 ms` · `30% 낮춤` · `0.223 vs 0.319` · `10배` · `13.4 vs 132.6 ms` ·
> `p=1.1e−15` · `d=1.16` · "Skill Score 가 양수인 유일한 방법".
> ⚠**사유(3라운드 교체)**: 이 값들은 `results/b1_full_comparison.json` 에 **실재한다** —
> 금지 이유는 **프레임 불일치**(WS 120–220 vs ETKF/GPR 301 프레임, 100/100 에피소드) ·
> **기준선 선택**(튜닝 LETKF 로 바꾸면 널) · **프로세스 간 지연 비교**다.
> 발표 전 아래 피치를 다시 쓸 것. 근거: `reference_citation_bans.md`

**30초 버전 (한 문장)** — ⛔RETRACTED
> "LiDAR 희소 관측으로부터 격자 없이 분산관계가 hard constraint로 걸린 spectral
> Gaussian 집합을 직접 최적화하여 13 ms 안에 파면을 복원하고, 100 에피소드 기준
> tuned ETKF 대비 nRMSE를 30% 낮추면서 사전 등록된 6개 메트릭 중 Skill Score가
> 양수인 유일한 방법이 되었다."

**2분 버전 (기술적, 2026-04-10)** — ⛔RETRACTED
> "기존 파면 추정은 격자 기반이라 희소·불규칙 LiDAR 관측에 적합하지 않다. 우리는
> 파면을 N개의 spectral Gaussian으로 직접 파라미터화하고, 분산관계
> ω=√(g|k|tanh(|k|d))를 hard constraint로 걸어 자유도를 5N에서 4N으로 줄인다.
> 프레임 간에는 k를 고정한 채 (w, φ)만 업데이트하는 w_phi tracking으로 overfitting을
> 막고, 에너지 기반 adaptive N으로 복잡도를 N=1~21로 자동 조절한다. 100 에피소드,
> 100 paired runs, 24-config ETKF hyperparameter sweep 위에서 tuned ETKF 대비
> nRMSE 30.1% 감소(0.223 vs 0.319), 약 10배 빠른 13.4 vs 132.6 ms/frame, 그리고
> 사전 등록된 6-metric 패널 중 Skill Score가 양수인 유일한 방법이다
> (Wilcoxon p=1.1e−15, Cohen's d=1.16, large effect). 우리는 또한 Dir R²=−0.25
> 라는 자체 약점을 같은 패널 위에서 투명하게 보고하며, no-dispersion ablation이
> nRMSE는 16% 낮지만 (k,ω) 쌍이 dispersion manifold에서 R̄=0.51 벗어나 downstream
> 예측에 쓸 수 없음을 정량적으로 증명한다."

**5분 버전**
> 위 2분 버전 + 배경 동기(하이드로포일 승차감 문제) + 경쟁 기법의 한계
> (격자 보간 오차, EnKF 비용, GPR O(N³)) + 한계점 및 future work 언급.

## E-4. 반드시 외울 수치 & 사실 목록 (2026-04-10 최신)

**WaveSplat 논문 (full 100 episodes)**

> ⛔**2026-08-13 감사 — 아래 ⛔ 항목은 외우면 안 된다.**
> ⚠**금지 사유는 "결과 파일이 없다" 가 아니다**(1라운드 오진) — 100-에피소드 값들은
> `results/b1_full_comparison.json`(커밋 `e57528c`)에서 재도출된다. 금지 사유는
> **프레임 불일치**(`WaveSplat.n_frames` 120–220 vs `ETKF`·`GPR-1` 301, 100/100 에피소드
> ⟹ `84/100` 승패비도 금지) · **기준선 선택** · **프로세스 간 지연 비교**다.

- [ ] **nRMSE**: ⛔RETRACTED — `WS 0.223 / ETKF 0.319 / GPR 0.261 → 30.1% 개선`.
      ✅검증된 값 = **WS 0.2229 vs LETKF 0.2278, 51/49, p = 0.9204 → 널**
- [ ] **Runtime**: ⛔RETRACTED — `13.4 ms / 132.6 ms (median 110) / >200 ms → ~10×`.
      ✅허용 = **동일 프로세스·1스레드 1.5–2.6×**, CPU = **AMD Ryzen 7 7800X3D**
- [ ] **Skill Score**: ⛔RETRACTED — `+0.034 유일 양수 / −0.702 / −0.140`
- [ ] **Variance Ratio**: ⛔RETRACTED — `0.66 / 0.90 / 0.065`
- [ ] **Pearson ρ**: ⛔RETRACTED — `0.45 / 0.11 / 0.19`
- [ ] **Dir R²**: ⛔RETRACTED — `WS −0.25`
- [ ] **통계**: ⛔RETRACTED — `Wilcoxon p=1.1×10⁻¹⁵, Cohen's d=1.16 (large)` ·
            `p=4.7×10⁻⁴, d=0.42`. 재도출 전 인용 불가
- [ ] **Out-of-sample 승패 (✅유효)**: S13 704런 = 혼잡 **56/64** · 희소 **63/64**
- [ ] **단위시험 (✅유효)**: **204 passed, 0 failed, 0 skipped** (85 wavesim + 119 wavesplat)
- [ ] **Dispersion residual (A1)**: R̄=**0.51±0.12** (≈0.10 Hz off) vs Full=0 by construction
- [ ] **ETKF sweep**: 24 configs, best 0.358 / worst 1.187 (3.3× spread),
            winner = 40×30, **nₑ=100**, α=1.00, σ=0.005
- [ ] **Raydrop**: ⛔RETRACTED — `20/20 wins` (구버전 12-에피소드 계열)
- [ ] **Noise**: ±13% nRMSE under σ=0.10 m (5× baseline)
- [ ] **Sparsity**: < 8% degrade at 20% observation density
- [ ] **Adaptive N range**: 1 (single swell) ~ 21 (complex sea)
- [ ] **논문 길이**: **40 p** (Ocean Engineering 권장 상한)

**Hydrofoil 논문**
- [ ] **가속도 저감**: BF3 head sea 50%, BF4 head 42%, BF4 bow 38%
- [ ] **LP filter α 민감도**: BF4 Head는 **α=0.01이 최적(+13.7%)**, α=0.05에서 −1.1% 퇴보
        — honest reporting (컴패니언 논문의 Q8)
- [ ] **WaveSplat Dir R² −0.25 ↔ BF4-Head degradation 연결** (Q5)
- [ ] **Optimal preview time**: 1.0–1.5 s
- [ ] **선박 제원**: LOA 8.5 m, 1700 kg, 22 knots
- [ ] **필터 f_c**: 0.8 Hz (α=0.05, Δt=0.01 s)

**공통**
- [ ] **BF 범위**: 2–5 (ε ≈ 0.05–0.07)
- [ ] **테스트 수**: 57 (39 wavesim + 18 wavesplat, 모두 pass)
- [ ] **제출 자료**: highlights, cover_letter.pdf, graphical_abstract.pdf 모두
        [paper/submission/](../paper/submission/)에 존재

## E-5. 약점/한계 미리 자백하기 (방어 > 은폐)

발표 중 한계를 먼저 언급하면 리뷰어가 공격할 여지가 줄어듭니다.
아래를 limitation 슬라이드에 **반드시 포함**:

1. 선형 파 가정 (BF6+ 비선형 영역 미검증)
2. 단일 선박 구성 (특정 hull-foil 형상은 다를 수 있음)
3. 시뮬레이션 기반 (실측 검증은 future work)
4. Single LiDAR (multi-sensor fusion 미탐색)
5. Directional spreading 단순화

## E-6. 3일 발표 준비 스케줄

```
[Day 1 오전] STUDY_GUIDE.md Part A, B 정독 (2h)
[Day 1 오후] HANDS_ON Stage 1, 2 실행 + Part E 읽기 자료 ★★★ 2개 (3h)
             분산관계 직접 유도 (1h)

[Day 2 오전] STUDY_GUIDE.md Part C, D 정독 (1.5h)
             Part E 읽기 자료 ★★ (1.5h)
[Day 2 오후] HANDS_ON Stage 3 실행 (1h)
             PPT 슬라이드 1~10 작성 (3h)

[Day 3 오전] PPT 슬라이드 11~20 작성 (2.5h)
[Day 3 오후] E-4 수치 암기 (30m)
             E-3 엘리베이터 피치 3가지 버전 외우기 (1h)
             D-15 예상 질문 리허설 (1h)
             타이머 보며 발표 연습 2회 (1h)
```

---

## E-7. 논문 보강(B 적용) 후 추가 학습 항목

`noise robustness ablation`, `hyperparameter sensitivity`, `latency budget`이
논문에 새로 추가됨에 따라 발표 시 받을 수 있는 질문도 3개 늘었습니다.
이 섹션은 **그 새로운 질문들을 방어하기 위해 추가로 공부할 것**을 정리합니다.

### E-7.1. 읽을 자료 (각 30분)

| 우선 | 키워드/자료 | 왜 필요한가 |
|---|---|---|
| ★★ | **Ge, Huang, Jin, Yuan (2015)** "Escaping from saddle points — online stochastic gradient for tensor decomposition" | σ=0.02에서 **nRMSE가 오히려 개선**되는 현상의 이론적 근거 |
| ★★ | **Neelakantan et al. (2015)** "Adding gradient noise improves learning for very deep networks" | Regularization by noise 직관 형성 |
| ★ | **Sea-based LiDAR driver latency 실측 리포트** (Velodyne/Ouster/Seyond 공식 문서 아무거나) | Latency budget의 "0.1 ms LiDAR acquisition"이 optimistic인지 설명 |
| ★ | **ISO 2631-1 (2008)** 발췌본 | "Comfortable" 경계값(0.315 m/s²)과 가속도 RMS 평가 기준 |

### E-7.2. 개념 한 줄 정리 (암기)

1. **Regularization-by-noise (stochastic perturbation)**
   > "작은 Gaussian 노이즈는 non-convex 손실 함수의 saddle point나 mild local minimum을
   > 벗어나게 하는 **확률적 탈출 효과**를 준다. 노이즈 크기가 작을수록 signal-preserving,
   > 너무 크면 SNR 자체가 깨진다. WaveSplat은 σ=0.02(baseline 1배)에서 탈출 효과가
   > 유효하다가 σ=0.10에서 사라진다."

2. **Latency budget의 scope**
   > "14.6 ms는 **알고리즘/제어기 연산 budget**이며, LiDAR driver overhead(2~5 ms),
   > OS scheduling jitter(수백 μs), 네트워크 전송 지연은 별도다. 그래도 100 ms cycle의
   > 15%로 여유가 충분해 deployment에 무리가 없다."

3. **Adaptive N noise 흡수 메커니즘**
   > "노이즈가 잔차(residual)를 키우면 grow trigger가 발동해 새 컴포넌트를 추가한다.
   > 새 컴포넌트는 낮은 에너지로 수렴하므로 pruning threshold(|a|² < τ)에서 제거된다.
   > 이 과정에서 safety rollback이 nRMSE 악화를 감지하면 되돌리므로 노이즈가
   > 추정 파면에 직접 새겨지지 않는다."

### E-7.3. 새로 암기할 수치 (E-4 보강)

- [ ] **Noise robustness**: σ 5배 증가에도 nRMSE 변화 **±13% 이내**
- [ ] **Noise regularization**: σ=0.02에서 BF3 5-comp nRMSE 0.168→0.149 (개선)
- [ ] **Latency budget 합계**: ⛔RETRACTED (2026-08-13 감사) — `14.6 ms` 는 항목 `WaveSplat
      13.4` 를 포함하고 있고 그 13.4 ms 가 금지 수치다(커밋 `77b8123` 이후 짝이 안 맞음).
      추정기 항목을 재측정하기 전 예산 합계와 마진을 인용하지 말 것.
      근거: `reference_citation_bans.md`
- [ ] **Budget margin**: ⛔RETRACTED — `85.4%` 는 위 무효 합계에서 파생된 값이다.
- [ ] **Hyperparameter sensitivity**: 하나를 ±1 decade 변경 시 nRMSE 변화 **±5% 이내**

### E-7.4. Part D 예상질문 추가 (Q7 ~ Q9)

**Q7. "σ=0.02에서 nRMSE가 오히려 떨어지는 게 말이 되나? 결과를 의심할 수밖에 없다."**
A. 타당한 의심이지만, 이는 non-convex 최적화에서 잘 알려진
**stochastic perturbation effect**입니다. 작은 노이즈가 deterministic 초기화가
갇혀 있던 mild local minimum을 벗어나게 해주는 현상으로, Ge et al. (2015)와
Neelakantan et al. (2015)에서 이론·실험적으로 입증되었습니다. σ=0.10으로 갈수록
이 효과는 사라지고 SNR 저하가 지배적이 되어 nRMSE가 다시 상승합니다. 논문 Table 8에서
이 반전이 관찰됩니다.

**Q8. "Latency budget의 'LiDAR acquisition 0.1 ms'는 현실적인가?"**
A. 해당 숫자는 **한 frame의 LiDAR 주기(100 ms) 대비 논리적 최소 readout 시간**이며,
실제 상용 LiDAR driver는 보통 2~5 ms의 overhead가 있습니다. 하지만 이는 본 연구의
scope(알고리즘/제어기 연산)에서 분리되어야 하는 **deployment 단계 이슈**이고, 전체
15 ms 예산에 추가되더라도 85%→80% 정도의 margin만 감소할 뿐 실시간성 결론은 바뀌지
않습니다.

**Q9. "왜 ETKF에도 같은 noise injection 실험을 안 했나?"**
A. 본 ablation의 목적은 **WaveSplat 내부의 adaptive N 메커니즘이 노이즈에 어떻게
반응하는가**를 정량화하는 것이었습니다. WaveSplat vs ETKF의 일반 비교는 Section 5.1의
12 에피소드 main comparison에서 이미 다뤘고, ETKF는 기본 관측 모델이 다르기 때문에
noise variance를 주입하는 의미 자체가 다릅니다. Revision에서 요구된다면 공정한 비교를
위한 noise-matched ETKF 설정을 추가할 수 있습니다.

### E-7.5. 손으로 해볼 것 추가

- [ ] **Noise ablation 직접 재현**: `python3 wavesplat/scripts/run_noise_ablation.py`
      실행하여 결과 표(`results/noise_ablation.json`)를 눈으로 확인
- [ ] **Latency budget 재계산**: 각 stage 시간을 임의로 바꿔서 "언제 실시간성이
      깨지는가" (→ WaveSplat stage가 40 ms 넘어가면 budget 초과)를 확인
- [ ] **논문 Section 5.7, 6.6, Table 8, Fig 9 정독**: 추가된 부분이 본인의 말로
      설명되는지 체크


