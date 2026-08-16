# WaveSplat: 교수님 발표 자료 (Ocean Engineering 제출 버전)

> ⛔**2026-08-13 감사 — 이 문서의 헤드라인 수치는 전량 철회됨 (금지 유지 · 3라운드에서 사유 교체).**
> `0.223 / 0.319 / 30.1%` · `13.4 ms` · `132.6 ms` · `~10×` · `84/100` 인용 금지.
>
> ⚠**1라운드가 적은 "결과 파일이 저장소에 존재하지 않는다" 는 거짓이었다** — 그 감사관은
> `results/` 를 **한 파일도 열지 않았다**(범위 없는 부재 주장). 네 수치는
> `results/b1_full_comparison.json`(**git 추적**, 커밋 `e57528c`, 100 에피소드, `etkf_config` 내장)
> 에서 **전부 재도출된다**: `WaveSplat.avg_nRMSE` **0.2229** · `ETKF.avg_nRMSE` **0.3189** ·
> `WaveSplat.avg_inference_ms` **13.3760** · `ETKF.avg_inference_ms` **132.6387**.
>
> **금지가 유지되는 진짜 사유 3가지**:
> 1. **프레임 불일치** — 같은 파일에서 `WaveSplat.n_frames` **120–220** vs `ETKF`·`GPR-1` **301**,
>    **100/100 에피소드 전부에서 WS 가 더 적다** ⟹ 두 팔이 **같은 프레임에서 채점된 적이 없다**.
>    ⟹ ⛔그 파일의 **`84/100` 승패비도 인용 금지**. (S13 배치는 이 점검 통과 — 불일치 0/128.)
> 2. **기준선 선택** — 그 ETKF 는 `grid 40×30 · ne=100 · mult_inflation 1.0`. **같은 `WaveSplat`
>    열**을 튜닝 LETKF 와 붙이면 `results/100ep_ws_letkf.json` 에서
>    **0.2229 vs 0.2278, 51/49, p = 0.9204 → 널(차이 없음)**.
> 3. **지연은 프로세스 간 비교** — 허용되는 서술은 **동일 프로세스·1스레드 1.5–2.6×**
>    (CPU = **AMD Ryzen 7 7800X3D**).
>
> 아래 본문의 ⛔RETRACTED 표시를 확인하고, 이 문서를 발표에 쓰기 전 전면 재실행할 것.
> 근거: `reference_citation_bans.md`, `tmp/AUDIT_recheck_2026-08-13/`

> **최근 업데이트**: 2026-04-29 (⛔수치는 2026-08-13 감사로 무효)
> **저자**: 박세용 (Avikus / HD Hyundai)
> **논문 제목**: *WaveSplat: Grid-Free Spectral Wave Field Estimation from Sparse Shipboard LiDAR*
> **대상 저널**: Ocean Engineering (Elsevier) — 40p 원고, highlights/cover letter/graphical abstract 준비 완료
> **관련 파일**: [wavesplat.tex](../paper/wavesplat.tex), [cover_letter.tex](../paper/submission/cover_letter.tex), [graphical_abstract.pdf](../paper/submission/graphical_abstract.pdf)

---

## 0. TL;DR (한 문장)

⛔**RETRACTED (2026-08-13 감사, 3라운드 사유 교체)** — 원문 TL;DR 은
"tuned ETKF 대비 nRMSE 30.1% 낮추고 약 10× 빠르게 (13.4 vs 132.6 ms/frame)" 였다.
⚠**"재현되지 않는다" 가 아니다** — `results/b1_full_comparison.json` 에서 그대로 나온다.
금지 사유는 ①**프레임 불일치**(WS 120–220 vs ETKF/GPR 301 프레임, 100/100 에피소드)
②**기준선 선택**(같은 WS 열을 튜닝 LETKF 와 붙이면 널) ③**프로세스 간 지연 비교**다.
근거: `reference_citation_bans.md`

허용되는 서술: "LiDAR 희소 관측에 **격자를 쓰지 않고** 분산관계 하드제약이 걸린 spectral
Gaussian 집합을 직접 최적화하여 파면을 복원한다. 100 에피소드 재도출에서
**WS 0.2229 vs LETKF 0.2278 = 51/49, p = 0.9204 (널)**, 지연은 동일 프로세스·1스레드에서
**1.5–2.6×**."

⛔"Skill Score가 양수인 유일한 방법" 을 포함한 6-메트릭 패널 수치는 위 무효 분모에서
파생됐다 — 재실행 전 인용 금지.

---

## 1. 연구 배경 및 동기

### 1.1 문제 정의

선박 탑재 scanning LiDAR는 전방 해수면을 10 Hz로 스캔하여 프레임당 ~50점의
3D 관측을 생성합니다. 이 관측은:

- **Sparse**: 200 × 150 m 관측 영역에서 프레임당 50~200점
- **Heterogeneous**: 근거리는 조밀, 원거리는 희박 (고정 각도 분해능)
- **Noisy**: 거리에 따른 beam divergence, 입사각에 따른 specular dropout

이러한 관측으로부터 **연속적이고 물리적으로 타당한 파도장**을 실시간 복원하는
것이 목표입니다.

### 1.2 기존 방법의 한계 (본 논문 §II 기준)

| 방법 | 근본 한계 |
|------|----------|
| **ETKF-grid** (Desmars 2020, Huhn 2022) | Sparse 관측을 격자에 보간 → aliasing, 고정 해상도, O(Nₑ·G²·logG²) 비용 |
| **단일 스냅샷 Sparse 스펙트럼** (Drazen 2008) | 프레임 간 컴포넌트 동일성 없음 → 시간 추적 불가 |
| **GPR** (RBF/Matérn) | O(M³), 시간 모델 없음, 단순 공간 보간 |
| **PINN / FNO / NN-ETKF** (Zhang 2022, Ehlers 2024, Desmars 2024) | 학습 분포 의존, LiDAR 불규칙 샘플링과 궁합 나쁨, 블랙박스 |

### 1.3 핵심 아이디어

파도장을 **N개의 plane wave (spectral component) 집합**으로 직접 파라미터화:

```
z(x, y, t) = Σᵢ wᵢ · cos(kx,ᵢ · x + ky,ᵢ · y − ωᵢ · t + φᵢ)
```

- **grid-free**: 공간 격자 없음 → aliasing 없음
- **dispersion hard constraint**: `ωᵢ = √(g · |kᵢ| · tanh(|kᵢ|·d))` — 자유 파라미터에서 제거 (5N → 4N)
- **adaptive N**: 에너지 기반 pruning + 잔차 기반 growth
- **Geometry-aware confidence weighting**: 입사각/거리 의존 관측 가중치 (Lambertian × inverse range)

---

## 2. 방법론 상세

### 2.1 Forward Model

```
z(x, y, t) = Σᵢ wᵢ cos(kx,ᵢ x + ky,ᵢ y − ωᵢ t + φᵢ)
ωᵢ = √(g |kᵢ| tanh(|kᵢ| d))     ← hard constraint (every forward eval)
```

파라미터: `{wᵢ, kx,ᵢ, ky,ᵢ, φᵢ}` → **4N**개 (ω는 k로부터 파생)

### 2.2 Observation Model (C3: Geometry-aware Confidence)

```
cⱼ = cos(θⱼ) / (dⱼ / d_ref)     d_ref = 125 m
L = (1/M) Σⱼ cⱼ (zⱼ − ẑⱼ)² + λ₁ Σ|wᵢ| + λs Σ(w_{σ(i)} − w_{σ(i−1)})²
```

- θⱼ: beam incidence angle, dⱼ: range
- λ₁ = 1e−3 (sparsity), λs = 1e−4 (spectral smoothness)

### 2.3 Phase 1: Initialization (T_init = 2Tₚ)

1. **Dispersion-aware beamforming** (⛔`v2_konly` 는 출하 기본값이 아니다 — 2026-07-26 flip
   이후 기본값은 라우터 `auto`, `run.py:43`·`estimation.yaml:24`): Cartesian 81×81 grid
   (kx, ky ∈ [−4.0, 4.0] rad/m, Δk ≈ 0.10 rad/m), 각 k에서
   `ω = √(g|k|tanh(|k|d))` 고정, 관측 시계열과 phasor 상관.
   Fine refinement: 41×41 local grid (Δk=0.01, ±0.2 rad/m)
2. Greedy peak extraction (0.15 rad/m Cartesian suppression)
3. **Coarse-to-fine** Adam optimization: N=3 → 8 → 15 (200 iters/stage, lr=5e−3)
4. **Energy-based pruning**: 누적 에너지 90% 잔존, RMSE 130% 초과 시 rollback
5. **Residual growth**: 잔차 RMS > 0.05 m이면 잔차에 beamforming → 성분 추가 (Nmax = 30)

### 2.4 Phase 2: Tracking (w_phi mode, 10 Hz)

```
Predict: φᵢ ← φᵢ + ωᵢ · Δt       (dispersion-exact phase propagation)
Update:  K=10 Adam steps (lr=1e−3) on Kw=3 sliding-frame window
         — only (wᵢ, φᵢ); kx, ky are frozen
```

**Why freeze k?** 단일 프레임에서 kx/ky는 ill-conditioned (공간 aliasing).
Multi-frame init이 ~20 프레임에 걸쳐 평균화하므로 고정하는 편이 안정적이며,
`φ ← φ + ω·Δt`가 강한 warm start가 되어 **200 → 10 gradient step**으로 감소.

---

## 3. 실험 설계

### 3.1 데이터셋

| 파라미터 | 값 |
|----------|-----|
| 총 에피소드 | **100** (5 seeds × 10 configs × 2 raydrop) |
| Sea state | BF2~BF5 (Hs 0.3~2.0 m, Tₚ 3.8~9 s) |
| 성분 수 | 1, 2, 5, 15, 30 |
| Raydrop | uniform (diffuse), slope (specular) |
| Duration | 301 frames (30 s @ 10 Hz) |
| GT grid | 80 × 60 spatial |
| LiDAR | Seyond Falcon K1 spec (1550 nm, 152 ch, 120°×25° FOV) |
| 수심 | 50 m |

Wave generation: JONSWAP 스펙트럼 + Gerstner (nonlinear) → linear model로 추정 시험

### 3.2 Baselines

| 방법 | 설정 | 비고 |
|------|------|------|
| **ETKF (tuned)** | 24-config 스윕 최적값: g=40×30, nₑ=**100**, α=1.00, σ=0.005 | ⛔RETRACTED (2026-08-13 감사) — `0.319 (100-ep)` 는 `results/b1_full_comparison.json` 에 **있다**(0.3189). 금지 사유는 부재가 아니라 **프레임 불일치**(WS 120–220 vs ETKF 301) + **기준선 선택**(튜닝 LETKF 로 바꾸면 널) |
| **GPR-1** (single-frame) | Matérn ν=5/2, ℓ=20 m, σ=0.02 m | 시간 모델 없음 |
| **GPR-sliding** (3-frame) | 위 + 3-frame sliding window | 통계 검증용 강한 GPR 베이스라인 |

**ETKF 스윕 결과 (24 configs → 3.3× spread)**:
| 순위 | g | nₑ | α | σ | mean nRMSE |
|------|---|----|----|---|------------|
| **Best (tuned)** | 40×30 | 100 | 1.00 | 0.005 | **0.358** |
| 2nd | 40×30 | 100 | 1.02 | 0.005 | 0.403 |
| 3rd | 40×30 | 50 | 1.00 | 0.005 | 0.416 |
| 22nd | 80×60 | 50 | 1.00 | 0.020 | 0.986 |
| Worst | 80×60 | 50 | 1.05 | 0.020 | 1.187 |

→ 전체 top-10이 모두 σ=0.005 공유, 40×30 격자가 80×60보다 일관 우위
(관측이 적어 미세 격자는 ensemble이 부족).

### 3.3 Metrics: **Pre-specified 6-metric panel**

nRMSE 하나로는 "zero-predictor"를 걸러낼 수 없기 때문에 아래 6개를 **사전 지정**:

| Metric | 정의 | 목표 | 진단 가치 |
|--------|-----|------|-----------|
| **nRMSE** ↓ | RMSE / Hs | 낮을수록 | primary |
| **VR** → 1 | Var(ẑ)/Var(z) | 1 | 진폭 보존; ≪1이면 under-fit |
| **ρ** ↑ | Pearson(ẑ, z) | 1 | 공간 형상 일치 |
| **SS** ↑ | 1 − MSE(ẑ)/MSE(0) | > 0 | zero-predictor보다 나은가 |
| **Dir R²** ↑ | 2D wavenumber 스펙트럼 회귀 | 1 | 방향성 일치 |
| **Hs err** ↓ | \|Hs_est − Hs_gt\|/Hs_gt | 0 | 유의파고 보존 |

---

## 4. 실험 결과

### 4.1 Full 100-episode 6-metric Evaluation (**primary result**)

> ⛔**RETRACTED (2026-08-13 감사, 3라운드 사유 교체)** — 이 6-메트릭 표(WS `0.223` /
> ETKF `0.319` / GPR `0.261` 와 그 SS·VR·ρ·Dir R² 열)는 **인용 금지 상태 그대로다. 단 사유가
> 바뀌었다.** ⚠"어느 결과 파일에서도 재현되지 않는다" 는 **거짓**이었다 — 이 표는
> `results/b1_full_comparison.json`(커밋 `e57528c`, 100 ep)에서 재도출된다
> (`WaveSplat.avg_nRMSE 0.2229` · `ETKF.avg_nRMSE 0.3189`). 금지 사유는 셋이다:
> ①**프레임 불일치** — 같은 파일에서 `WaveSplat.n_frames` **120–220** vs `ETKF`·`GPR-1` **301**,
> **100/100 에피소드에서 WS 가 더 적다** ⟹ 같은 프레임에서 채점된 적이 없다
> (⟹ 그 파일의 `84/100` 승패비도 금지). ②**기준선 선택** — 그 ETKF 는
> `grid 40×30 · ne=100 · mult_inflation 1.0`. ③**지연은 프로세스 간 비교**.
> 근거: `reference_citation_bans.md`
>
> **같은 `WaveSplat` 열을 튜닝 LETKF 와 붙인 재도출** (`results/100ep_ws_letkf.json`, 100 ep):
>
> | Method | nRMSE ↓ | 판정 |
> |---|---|---|
> | WaveSplat | 0.2229 | — |
> | LETKF | 0.2278 | **51/49, p = 0.9204 → 널** |
>
> GPR 은 재도출에서 **0.276** 만 확인됐다(`0.261` 아님).

**핵심 관찰** ⛔(아래 4개 관찰은 위 무효 표에 근거한다 — 재실행 전 인용 금지):
1. **Skill Score: WaveSplat만 양수 (+0.034)**. ETKF와 GPR 모두 zero-predictor보다 나쁨.
2. **GPR은 거의 zero predictor**: VR=0.065 → 진폭의 6.5%만 재현.
3. **ETKF는 variance-correct but shape-wrong**: VR=0.898 but ρ=0.109. 에너지는
   맞지만 공간 구조가 틀림.
4. **WaveSplat의 한계도 노출**:
   - VR=0.66 → 진폭이 평균 ~34% 과소추정 (pruning의 대가)
   - Dir R²=−0.25 → **방향 성분은 chance 미만** (중요한 한계, hydrofoil 컴패니언 논문 BF4-head 악화의 원인)
   - SS=+0.034 → 양수이지만 marginal

### 4.2 Sea State 별 (GPR의 zero-predictor 지문)

> ⛔**RETRACTED (2026-08-13 감사, 3라운드 사유 교체)** — 이 BF별 표의 `Avg` 행
> (`0.223` / `0.319` / `0.261`)은 위 §4.1 과 **같은 `b1_full_comparison.json` 산출**이므로
> **프레임 불일치·기준선 선택 사유를 그대로 물려받는다**(= "재현 불가"가 아니다).
> 표 전체를 재실행 전 인용 금지. 근거: `reference_citation_bans.md`

| BF | WS | ETKF(tuned) | GPR-1 | Nₑₚ |
|----|-----|-----|-----|-----|
| BF2 | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 10 |
| BF3 | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 30 |
| BF4 | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 40 |
| BF5 | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 20 |
| **Avg** | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 100 |

<!-- 이력 보존 (2026-08-13 감사에서 철회된 원본 표 — 인용 금지):
| BF2 | 0.302 | 0.369 | 0.258 | 10 |
| BF3 | 0.160 | 0.319 | 0.259 | 30 |
| BF4 | 0.221 | 0.315 | 0.263 | 40 |
| BF5 | 0.283 | 0.302 | 0.262 | 20 |
| Avg | 0.223 | 0.319 | 0.261 | 100 |
-->

→ GPR의 nRMSE는 BF2~5에 걸쳐 0.258~0.263으로 **거의 불변**. 이는 ẑ≈0일 때
RMSE∝Hs이고 Hs로 정규화하면 상쇄되는 "zero-predictor"의 진단적 지문.

### 4.3 Per-episode breakdown (12 대표 에피소드, Table I)

> ⛔**RETRACTED (2026-08-13 감사, 3라운드 회귀검증에서 추가 적발)** — 이 표는 §4.1·§4.2 와
> 같은 **구버전 12-에피소드 실행**이고, 아래 불릿이 이미 철회한 `0.211 vs 0.358` · `0.436` 을
> **표 본문에 그대로 살려두고 있었다**. 지연 열(`ms/f`)은 **16스레드 unpinned 아티팩트**다.
> 같은 라운드가 `README.md`(`0.211 vs 0.358 (41.1% better)`)와
> `docs/CHANGELOG.md`(`0.211 vs ETKF 0.436 (+51.5%)`)에서 정확히 이 수치를 금지했다.
> 근거: `reference_citation_bans.md` · `tmp/AUDIT_recheck_2026-08-13/R3_regression.md §M-4`

| Episode | WS nRMSE | WS ms/f | ETKF(tuned) nRMSE | ETKF ms/f | Nₐ |
|---------|----------|---------|--------|--------|-----|
| BF2 1c  | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 11 |
| BF3 1c  | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 2 |
| BF3 2c (U) | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 2 |
| BF3 2c (S) | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 15 |
| BF3 5c (U) | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 4 |
| BF3 5c (S) | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 15 |
| BF4 1c  | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 1 |
| BF4 5c  | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 3 |
| BF4 15c | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 19 |
| BF4 30c | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 18 |
| BF5 15c | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 21 |
| BF5 30c | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | 19 |
| **Avg** | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | — |

<!-- 이력 보존 (2026-08-13 3라운드 감사에서 철회된 원본 표 — 인용 금지):
| BF2 1c  | 0.315 | 15.1 | 0.436 | 154 | 11 |
| BF3 1c  | 0.217 | 12.7 | 0.322 | 154 | 2 |
| BF3 2c (U) | 0.052 | 22.2 | 0.349 | 635 | 2 |
| BF3 2c (S) | 0.061 | 13.7 | 0.305 | 120 | 15 |
| BF3 5c (U) | 0.167 | 12.8 | 0.355 | 125 | 4 |
| BF3 5c (S) | 0.152 | 13.2 | 0.320 | 177 | 15 |
| BF4 1c  | 0.171 | 10.3 | 0.331 | 98  | 1 |
| BF4 5c  | 0.200 | 12.7 | 0.352 | 106 | 3 |
| BF4 15c | 0.302 | 15.9 | 0.379 | 113 | 19 |
| BF4 30c | 0.314 | 13.6 | 0.376 | 121 | 18 |
| BF5 15c | 0.297 | 15.0 | 0.364 | 98  | 21 |
| BF5 30c | 0.290 | 13.3 | 0.412 | 93  | 19 |
| Avg | 0.211 | 14.2 | 0.358 | 166 | — |
-->

- ⛔RETRACTED (2026-08-13 감사, 3라운드에서 **사유 분리**) — `12/12 승` · `best-case 8.5×`
  는 **구버전 12-에피소드 실행**이다(사유 유지). 반면 `100-ep 평균 132.6 ms (median 110 ms)` ·
  `WS 13.4 ms` · `~10× speed advantage` 는 `results/b1_full_comparison.json` 에 **실재한다**
  (`132.6387` / `13.3760`) — 금지 사유는 "재현 불가"가 아니라 **프로세스 간 비교**
  (스레드 미고정 포함)다. 근거: `reference_citation_bans.md`
- 허용되는 지연 서술: **동일 프로세스·1스레드에서 1.5–2.6×** (CPU = AMD Ryzen 7 7800X3D).
  단일 숫자 대신 **중앙값 + IQR 병기**(`shipping_default.json`).
- 허용되는 승패 서술: S13 704런 out-of-sample = 혼잡 **56/64**, 희소 **63/64**.

### 4.4 Multi-seed 통계 검증 (100 paired runs)

> ⛔**RETRACTED (2026-08-13 감사, 3라운드 사유 교체)** — 이 절의 통계량 전량
> (`Wilcoxon p = 1.1×10⁻¹⁵`, `Cohen's d = 1.16 (large)`, 95% CI `[0.085, 0.119]`)은
> 인용 불가다. ⚠사유는 "분모가 저장소에 없다" 가 **아니다** — 분모는
> `results/b1_full_comparison.json` 에 있다(`ETKF.avg_nRMSE 0.3189`). 진짜 사유는
> ①그 분모가 **프레임 불일치**(WS 120–220 vs ETKF 301 프레임, 100/100 에피소드)에서 나왔고
> ②**기준선 선택**의 산물이며 — 같은 100 에피소드를 튜닝 LETKF 로 재도출하면
> **51/49, p = 0.9204 = 널** — ③이 절의 `p`·`d` 값 자체는 그 파일에서 **그대로 재현되지도
> 않는다**는 것이다. 재도출 전 인용 불가. 근거: `reference_citation_bans.md`

| 비교 | mean ± std | Wilcoxon W | p-value | **Cohen's d (paired)** | 95% CI |
|------|------------|------------|---------|------------------------|--------|
| WaveSplat | ⛔RETRACTED | — | — | — | — |
| vs ETKF (tuned) | — | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED |
| vs GPR sliding | — | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED | ⛔RETRACTED |

- 100 paired runs = 10 configs × 5 seeds × 2 raydrop (설계는 유효, 수치는 무효)

### 4.5 Ablation Study (A1: no dispersion, A2: no confidence)

BF3 6 에피소드 × 3 seeds:

| Config | avg nRMSE | vs Full |
|--------|-----------|---------|
| Full | 0.188 | baseline |
| A1 (no dispersion) | **0.158** | **nRMSE 16% 낮음** — 단, 아래 주의 |
| A2 (no confidence) | 0.215 | 14.5% 악화 |

BF4–BF5 50 eps × 3 seeds 확장:
- A1 = 0.241 ± 0.068 vs Full = 0.266 ± 0.047 → 9.4% nRMSE "advantage"

**A1이 nRMSE는 더 낮지만 물리적으로는 쓸 수 없다 (Q4 분산잔차 측정)**:

분산 manifold에서 얼마나 벗어나는가? 10 paired runs에서:
```
R̄ = Σ |wᵢ| · |ωᵢ − √(g|kᵢ|tanh(|kᵢ|d))| / (ωᵢ · Σ|wⱼ|)

A1_noDisp:  R̄ = 0.51 ± 0.12   (절대값 ≈ 0.10 Hz off dispersion)
Full:       R̄ ≡ 0               (by construction)
```

해석: A1은 linear forward model ↔ nonlinear Gerstner 불일치를 (k, ω)가
manifold에서 벗어나도록 함으로써 흡수. 결과적으로 **1초 뒤 phase prediction이
de-cohere**하여 시간 추적, 공간 프레임 변환, downstream wave forecast에 재사용 불가.
즉 A1의 16%는 **물리적 유효성을 버리고 얻은 nRMSE**다.

### 4.6 Spectral Parameter Estimation Accuracy (NEW)

WaveSplat의 고유 기능: 개별 스펙트럼 성분의 물리 파라미터를 추출.
헝가리안 매칭으로 추정 vs GT 성분 대응 후 오차 평가.

| N_comp | Detection | Energy Det. | |k| Rel Error | Dir Error [°] |
|--------|-----------|-------------|---------------|---------------|
| 1      | 100%      | 100%        | 0.005         | 0.1           |
| 2      | 50%       | **96.9%**   | 0.001         | 0.2           |
| 5      | 60%       | **93.7%**   | 0.001         | 0.1           |
| 15     | 50%       | 66.7%       | 0.595         | 26.2          |
| 30     | 38%       | 49.1%       | 0.714         | 26.1          |

**핵심**: ≤5 comp에서 에너지 기준 93%+ 포착, |k| 오차 <0.5%, 방향 <0.2°.
LETKF/ETKF는 이런 decomposition 자체가 불가능.
진폭 오차는 크지만(0.17~8.2), 이는 tracking 중 Adam momentum에 의한
진폭 감쇠 때문이며 kx/ky/ω/direction은 초기화에서 결정되어 정확.

### 4.7 Sparse Observation Robustness: WS vs LETKF (NEW)

관측 점 수를 10~100개로 변화시키며 4개 에피소드에서 비교.

| n_pts | WS nRMSE | LK nRMSE | WS SS   | LK SS   | WS wins |
|-------|----------|----------|---------|---------|---------|
| 10    | **0.191**| 0.239    | +0.293  | +0.090  | 3/4     |
| 20    | **0.176**| 0.237    | +0.378  | +0.105  | 3/4     |
| 50    | **0.181**| 0.234    | +0.332  | +0.124  | 3/4     |
| 100   | **0.181**| 0.234    | +0.327  | +0.124  | 3/4     |

**핵심**: WaveSplat은 10pts에서도 nRMSE 0.191 유지 (LETKF 0.239).
n_pts에 거의 무관한 robustness. 실무적 가치: LiDAR 포인트 손실(차폐/비/안개)
상황에서 결정적 이점.
예외: BF4 15comp에서만 LETKF 우위 (고복잡도 영역).

### 4.8 Deterministic Forecast (NEW)

추정 파라미터로 미래 파도장을 분산관계 기반 forward propagation.
추정: t=[0,20s], 예측: t=[20,30s]. 5개 에피소드 평균.

| Δt     | nRMSE | SS     | ρ     | VR    |
|--------|-------|--------|-------|-------|
| +0.5s  | 0.218 | +0.141 | 0.323 | 0.325 |
| +1.0s  | 0.218 | +0.140 | 0.322 | 0.327 |
| +5.0s  | 0.218 | +0.135 | 0.328 | 0.342 |
| +10.0s | 0.219 | +0.128 | 0.316 | 0.331 |

**핵심**: 10초 후에도 nRMSE 사실상 일정(0.218→0.219), SS>0 유지.
BF3 2comp는 10초 후에도 SS=+0.953, ρ=0.983.
LETKF는 스펙트럼 파라미터를 추출할 수 없으므로 forecast 자체가 불가능.
→ Hydrofoil feedforward(5~10s preview)의 직접적 기반.

### 4.9 Raydrop Sensitivity (10 paired eps × U/S)

| | WS U | WS S | WS Δ% | ETKF U | ETKF S | ETKF Δ% |
|---|---|---|---|---|---|---|
| **Avg** | 0.232 | **0.211** | −9.1 | 0.450 | 0.377 | −16.2 |

→ ⛔RETRACTED (2026-08-13 감사) — `WaveSplat 20/20 승`. 이 수치는 구버전 12-에피소드
계열 실행(#9)이고 라우팅 교락이 있다. 어느 현행 결과 파일에서도 재현되지 않는다.
근거: `reference_citation_bans.md`
Slope 이 양쪽 다 돕고 ETKF 가 더 많이 개선된다는 **부호** 관찰만 유지한다.

### 4.7 Robustness: Sparsity & Noise

**Sparsity (5 eps, 20~100%)**:
- BF4~5 복잡 해상: 20% (~10점)에서도 < 8% 성능 저하
- 예외: BF3 2c — 100%에서만 N=2 해상 가능 (20%에서는 N=1 fallback)

**Noise (3 eps, σ ∈ {0, 0.02, 0.05, 0.10} m)**:
- σ=0.10 m (5× baseline)에서도 nRMSE 변화 ±13% 이내
- Mechanism: adaptive N이 저에너지 성분에 노이즈 흡수 → pruning이 제거

### 4.8 Hyperparameter Sensitivity

한 번의 coarse sweep (BF3 5c)으로 결정된 값을 모든 에피소드/시드/raydrop에 **고정**:
```
lr = 0.01,  λ₁ = 1e−3,  τ_prune = 0.10,  τ_grow = 0.05
```
각 파라미터를 ±2× 변경해도 nRMSE는 **±5% 이내**. 전체 100 runs가 하나의 설정으로
재현된다는 사실 자체가 "낮은 민감도"의 강한 증거.

---

## 5. 계산 복잡도

| 방법 | Per-frame | 파라미터/상태 | 실측 |
|------|-----------|-------------|------|
| **WaveSplat** | O(K · Kw · M · N) | 4N ≤ 120 | ⛔RETRACTED |
| ETKF (tuned) | O(Nₑ G² log G² + Nₑ³ + Nₑ² M) | Nₑ · G² = 100·1200 | ⛔RETRACTED |
| GPR | O(M³) | M² | ⛔RETRACTED |

⛔**RETRACTED (2026-08-13 감사, 3라운드 사유 교체)** — `13.4 ms` · `132.6 ms (median 110)` ·
`>200 ms` · `31.7 ms` · `~10× 빠름`. ⚠이 값들은 `results/b1_full_comparison.json` 에
**실재한다**(`13.3760` / `132.6387`) — 금지 사유는 부재가 아니라 **프로세스 간 비교**
(스레드 미고정 포함)이고, 커밋 `77b8123` 이후 짝도 맞지 않는다.
근거: `reference_citation_bans.md`
허용: **동일 프로세스·1스레드에서 1.5–2.6×**, CPU = **AMD Ryzen 7 7800X3D**,
단일 숫자 대신 **중앙값 + IQR 병기**(`shipping_default.json`).
복잡도 열(`O(·)`)은 해석적 유도이므로 유효하다.

### 5.7 강화 베이스라인 비교 (12 에피소드)

LETKF(Gaspari-Cohn localization, loc_radius=50m)와 dispersion-aware GPR(3-frame)을 추가 비교:

| 지표 | WaveSplat | LETKF(loc50) | ETKF(tuned) | GPR-1 | GPR-disp(3f) |
|------|-----------|-------------|-------------|-------|-------------|
| nRMSE ↓ | **0.211** | 0.238 | 0.358 | 0.273 | 0.267 |
| VR → 1 | **0.574** | 0.114 | 1.214 | 0.049 | 0.077 |
| ρ ↑ | **0.460** | 0.346 | 0.097 | 0.163 | 0.215 |
| SS ↑ | **+0.162** | +0.128 | −1.020 | −0.159 | −0.106 |
| Dir R² ↑ | **0.431** | 0.074 | 0.045 | 0.013 | 0.026 |
| Hs err ↓ | 0.282 | 0.668 | **0.242** | 0.798 | 0.741 |
| Speed ↓ | **13.2ms** | 115.7ms | 103.9ms | 208.8ms | 473.6ms |

**WaveSplat이 7개 중 6개 지표에서 1위**. Dir R² 5.8× 우위, 속도 8.8× 우위.

SS Bootstrap 95% CI: WaveSplat [−0.15, +0.46] (0 포함), LETKF [+0.08, +0.17] (0 미포함).
→ LETKF는 일관 positive skill, WaveSplat은 저복잡도에서 강하고 고복잡도(≥15comp)에서 약함.

LETKF loc_radius sweep (20/50/100/200m): 20m과 50m이 거의 동일 (nRMSE 0.234), 200m은 급감 (0.279).
→ loc_radius=50m은 near-optimal 선택.

---

## 6. 기여 정리

### 6.1 Novelty
1. **Grid-free spectral representation**: 3D GS(Kerbl 2023)의 아이디어를 해양 파면 추정에 최초 적용
2. **Dispersion as hard constraint**: `ωᵢ = f(|kᵢ|, d)`, 파라미터 5N → 4N
3. **Adaptive model order**: energy pruning + residual growth + safety rollback
4. **Geometry-aware confidence weighting**: LiDAR 관측 품질 직접 모델링 (Lambertian × inverse range; 정확한 Fresnel은 raydrop에서만)
5. **Pre-specified 6-metric evaluation**: zero-predictor 구분을 위한 panel
6. **24-config ETKF hyperparameter sweep**: 공정한 tuned baseline
7. **Dispersion manifold drift quantification**: A1 ablation의 "자유도" 비용 정량화
8. **Spectral decomposition**: ≤5 comp에서 |k| 오차 <0.5%, 방향 <0.2° — 에너지 93%+ 포착
9. **Extreme sparse robustness**: 10pts에서도 nRMSE 0.191 유지 (LETKF 0.239)
10. **Deterministic forecast**: 분산관계 기반 10초 예측, SS>0 유지

### 6.2 Honest Limitations (§V-A 끝 단락)
1. **VR = 0.66** → 진폭 34% 과소 — **근본 원인 규명**: Adam momentum이 phase tracking 중 w의 부호 진동을 감쇠시킴
2. **SS = +0.034** → 양수지만 marginal
3. **Dir R² = −0.25** → 방향성이 chance 미만. 하이드로포일 BF4-head에서 이 한계가 직접 관찰됨
4. **시뮬레이션 전용**, 선형 모델, Nₜᵣᵤₑ ≥ 30 한계, current 미추정
5. **진폭 보정 시도**: post-hoc LS refit은 dense observation에서만 효과적, sparse LiDAR(~50pts)에서는 공간 커버리지 부족으로 무효

---

## 7. 핵심 수치 요약 (한 장 슬라이드용)

```
┌───────────────────────────────────────────────────────────────┐
│              WaveSplat — 6-metric 전체 결과                   │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ⛔ RETRACTED (2026-08-13 감사)                               │
│     nRMSE / Skill Score / VR / ρ / Dir R² / Speed 6-지표 값,  │
│     Wilcoxon p, Cohen's d, 20/20 raydrop wins 전량 무효.      │
│                                                               │
│  ⚠ 사유 교체 (3라운드): 값은 저장소에 있다 —                  │
│     results/b1_full_comparison.json (커밋 e57528c, 100 ep)    │
│     ETKF 0.3189 / WS 0.2229 / 13.3760 / 132.6387 ms.          │
│     금지 사유 = 프레임 불일치(WS 120-220 vs ETKF 301,         │
│     100/100 ep) + 기준선 선택 + 프로세스 간 지연 비교.        │
│     -> 그 파일의 84/100 승패비도 인용 금지.                   │
│                                                               │
│  같은 WS 열 vs 튜닝 LETKF (results/100ep_ws_letkf.json):      │
│    nRMSE : WS 0.2229  vs  LETKF 0.2278                        │
│    판정   : 51/49, p = 0.9204  →  NULL (차이 없음)            │
│    지연   : 동일 프로세스·1스레드 1.5–2.6×                    │
│             CPU = AMD Ryzen 7 7800X3D                         │
│    승패   : S13 704런 OOS = 혼잡 56/64 · 희소 63/64          │
│                                                               │
│  근거: reference_citation_bans.md                             │
│        tmp/AUDIT_recheck_2026-08-13/A3_citation_bans.md       │
│                                                               │
│  vs Stronger Baselines (12 ep):                               │
│    ⛔RETRACTED — 12-에피소드 구버전 실행                       │
│    Dir R²: WS 0.431 vs LETKF 0.074 (5.8×)                    │
│    Speed: WS 13.2ms vs LETKF 115.7ms (8.8×)                  │
│                                                               │
│  Spectral Decomposition (≤5 comp):                            │
│    Energy detection: >93%                                     │
│    |k| error: <0.5%, Direction: <0.2°                         │
│                                                               │
│  Sparse Robustness (WS vs LETKF):                            │
│    10 pts: WS 0.191 vs LETKF 0.239 (3/4 wins)               │
│                                                               │
│  Deterministic Forecast:                                      │
│    t+10s: nRMSE=0.219, SS=+0.128 (constant)                  │
│    BF3 2c: SS=+0.953, ρ=0.983 at t+10s                       │
│                                                               │
│  Dispersion manifold drift (A1 vs Full):                      │
│    A1:  R̄ = 0.51 ± 0.12  (0.10 Hz off)  → nRMSE 낮지만      │
│    Full: R̄ ≡ 0             (by construction)  물리적으로 사용 가능 │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## 8. 슬라이드 구성 제안 (OE 발표 기준 20장)

| # | 제목 | 핵심 그림/수식/표 | 발표 팁 |
|---|------|-----------------|---------|
| 1 | Title | 제목, 저자, OE 투고 마크 | 10초 |
| 2 | Motivation | Hs/Tp→hydrofoil 승차감, preview 필요 | 1편 컴패니언 논문 언급 |
| 3 | Problem | LiDAR 희소 관측 scatter (graphical_abstract의 Panel 1) | "50점/frame from 200×150 m" |
| 4 | Prior Art | §II 표: ETKF / GPR / NN 한계 | 격자-free라는 우리 키워드 강조 |
| 5 | Big Idea | `z = Σwᵢ cos(...)`, `ω=√(g|k|tanh(|k|d))` | "5N → 4N" 숫자 강조 |
| 6 | Pipeline | Init(beamform→C2F→prune→grow) + Track(predict+update) | Table 알고리즘 요약 참고 |
| 7 | Why freeze k | Phase 2 의 w_phi mode 그림, "200→10 steps" | overfit 방지 직관 |
| 8 | Dataset | 100 ep, BF2~5, U/S, 5 seeds | "pre-specified" 단어 |
| 9 | Metrics | 6-metric 패널 정의 표 + SS=1−MSE(ẑ)/MSE(0) | zero-predictor 개념 |
|10 | ETKF fairness | 24-config sweep 표 (best/worst, 3.3× spread) | "tuned baseline" 어필 |
|11 | **§5.1 Six-metric table** | 핵심표 (WS SS +0.034 only) | 가장 중요한 슬라이드 |
|12 | Sea-state 지문 | GPR 0.258~0.263 table | zero-predictor 진단 |
|13 | Per-episode Table I | 12 eps bar chart (fig1_comparison) | BF3 2c 8.5× 강조 |
|14 | Speed | ⛔RETRACTED — `13.4 vs 132.6 ms` 재측정 필요 | 1스레드 1.5–2.6× 로 재작성 |
|15 | Multi-seed 통계 | ⛔RETRACTED — p·Cohen's d 재도출 필요 | "d=1.16 large" 인용 금지 |
|16 | A1 dispersion drift | R̄=0.51 vs 0, "16% nRMSE는 물리 포기 비용" | Q4 핵심 |
|17 | Raydrop + Sparsity | ⛔RETRACTED — `20/20 wins` 구버전 | ≤8% degrade 만 |
|18 | Honest limits | VR 0.66, Dir R² −0.25, hydrofoil BF4-head 연계 | 약점 먼저 꺼내기 |
|19 | Contributions | 7개 번호 + OE 적합성 | highlights 그대로 |
|20 | Future work | real LiDAR, current, Stokes-2, 논문 2 연계 | 닫기 |

**보조 슬라이드 (질문 대비)**:
- Hyperparameter sensitivity (±5%)
- Noise robustness (σ up to 0.10)
- ETKF sweep full ranking
- Dispersion residual 공식과 10-run 통계

---

## 9. 예상 질문 & 모범 답변 (업데이트)

**Q1. "SS만 양수라는 게 정말 novel한 결과인가? Cherry-picked metric 아닌가?"**
A. SS는 **사전 지정**된 6개 중 하나로, 본 논문을 시작할 때 선언했고 코드에 박혀 있습니다.
선택된 후 하나만 보여주는 cherry-pick이 아니라, 6개 모두 공개하며 우리 자신의 약점(Dir R² −0.25)
도 같은 panel에서 함께 드러냅니다. GPR의 VR=0.065가 **zero-predictor 진단**의 결정적 증거입니다.

**Q2. "ETKF를 tuning 안 해서 이긴 것 아닌가?"**
A. 24-config 스윕을 수행했습니다(grid ∈ {40×30, 80×60}, Nₑ ∈ {50, 100},
α ∈ {1.00, 1.02, 1.05}, σ ∈ {0.005, 0.020}). 스윕 결과 3.3× spread가 있고,
**best configuration (40×30, Nₑ=100, α=1.00, σ=0.005, nRMSE=0.358)을 전체 비교에
사용**했습니다. Worst와 best의 격차가 커서 tuning 필요성도 동시에 입증됩니다.

**Q3. "A1 (no dispersion)이 16% nRMSE 좋은데 dispersion이 왜 필요한가?"** (Q4 핵심)
A. 네, A1이 nRMSE는 낮습니다. 하지만 **(kᵢ, ωᵢ)가 dispersion manifold에서 벗어남**:
`R̄ = 0.51 ± 0.12`, 절대값 ~0.10 Hz. 50% 벗어난 성분은 **1초 뒤 phase prediction이
de-cohere**해서 tracking도 downstream forecast도 불가능합니다. 즉 A1의 16%는
**nRMSE를 위해 물리적 유효성을 포기한 대가**이고, hydrofoil preview controller처럼
1초 뒤 파면을 예측해야 하는 downstream에는 쓸 수 없습니다.

**Q4. "Dir R² = −0.25면 방향이 틀린 것 아닌가?"**
A. 맞습니다. 본 논문은 이 한계를 §V-A 끝에서 명시적으로 보고합니다. LiDAR
관측 swath 위에서는 정확하지만 **파수 공간에서의 방향 할당은 평균적으로
chance 미만**이고, 이는 컴패니언 hydrofoil 논문의 BF4-head 시나리오에서
feedforward gain이 반전되는 현상으로 직접 관찰됩니다. 이 한계를 숨기지 않고
drive로 명시한 것이 본 논문의 "pre-specified panel" 철학입니다.

**Q5. "왜 OCEANS가 아니라 Ocean Engineering인가?"**
A. OCEANS는 6–8 페이지 conference이고, 본 결과(6-metric panel + 100-ep
통계 + 24-config ETKF sweep + dispersion residual + raydrop + sparsity +
noise robustness)를 담기엔 부족합니다. 40p OE 원고가 적정 분량입니다.
Desmars 2020, Huhn 2022 등 가장 가까운 선행연구도 OE 게재.

**Q6. "100 paired runs의 'paired' 의미는?"**
A. 같은 (episode, seed, raydrop) 3종 조건 아래에서 WaveSplat과 ETKF를
동일 관측으로 돌린 뒤 per-run nRMSE를 **쌍으로** 얻어, Wilcoxon signed-rank
test와 paired Cohen's d = mean_diff / std_diff를 계산. 환경 변동성이
상쇄되어 효과 크기 추정이 더 robust합니다.
⛔RETRACTED (2026-08-13 감사, 3라운드 사유 교체) — `Cohen's d=1.16 = "large effect"` 는
**프레임 불일치·기준선 선택 결함이 있는 분모**(`b1_full_comparison.json` 의 `ETKF 0.3189`)
위에서 계산됐고, 값 자체도 그 파일에서 **그대로 재현되지 않는다**. 재도출 전 인용 불가.
근거: `reference_citation_bans.md`

**Q7. "GPR이 왜 그렇게 많이 한지는? Hyper tuning 했나?"**
A. 단일 프레임 GPR은 표준 Matérn ν=5/2, ℓ=20 m, σ=0.02 m로 설정했고,
추가로 3-frame sliding-window GPR로 VR=0.11까지 올린 **더 강한 버전**을
Wilcoxon 검증에 사용했습니다. 그래도 여전히 near-zero predictor입니다
(WS와의 Cohen's d = 0.42, 통계적으로 robust하게 WS가 우위).

**Q8. "Real LiDAR 검증은?"** (가장 치명적 질문)
A. 본 논문은 **시뮬레이션 전용 proof-of-concept**입니다 — 이 점을 Limitations
첫 항목에서 명시적으로 자백합니다. WaveSim 환경은 beam divergence,
raydrop, 거리 의존 noise를 모두 모델링하지만 real LiDAR는 multi-path,
spray 등 추가 편향이 있을 수 있습니다. SNU 조파수조 검증 → Avikus 선박
실증이 다음 단계입니다.

---

## 10. 발표자 체크리스트 (외울 수치)

> ⛔**2026-08-13 감사 — 이 "외울 수치" 목록의 헤드라인 항목은 외우면 안 된다.**
> 아래 ⛔ 표시된 항목은 재실행 전 발표에서 인용 금지.
> ⚠**금지 사유는 "결과 파일이 없다" 가 아니다**(그건 1라운드 오진) — 값들은
> `results/b1_full_comparison.json` 에 있고, 금지 사유는 **프레임 불일치**(WS 120–220 vs
> ETKF/GPR 301 프레임, 100/100 에피소드) · **기준선 선택** · **프로세스 간 지연 비교**다.

- [ ] **nRMSE (full 100-ep)**: ⛔RETRACTED — `0.223 / 0.319 / 0.261 → 30.1%`.
      검증된 재도출 = **WS 0.2229 vs LETKF 0.2278, 51/49, p = 0.9204 (널)**
- [ ] **Speed**: ⛔RETRACTED — `13.4 ms / 132.6 ms (median 110) → ~10×`.
      허용 = **동일 프로세스·1스레드 1.5–2.6×**, CPU **AMD Ryzen 7 7800X3D**
- [ ] **6-metric SS**: ⛔RETRACTED — `+0.034 / −0.702 / −0.140`
      (같은 `b1_full_comparison.json` 산출 ⟹ 프레임 불일치·기준선 선택 상속)
- [ ] **Variance Ratio**: ⛔RETRACTED — `WS 0.66, ETKF 0.90, GPR 0.065` (동상)
- [ ] **Dir R²**: ⛔RETRACTED — `WS −0.25` (동상)
- [ ] **Wilcoxon**: ⛔RETRACTED — `p=1.1e−15 (d=1.16)` · `p=4.7e−4 (d=0.42)`.
      재도출 전 인용 불가
- [ ] **Dispersion residual (Q4)**: A1 R̄=0.51±0.12, Full=0
- [ ] **ETKF sweep**: 24 configs, best 0.358, worst 1.187 (3.3×)
- [ ] **Raydrop**: ⛔RETRACTED — `20/20 wins` (구버전 12-에피소드 계열)
- [ ] **Noise**: ±13% under 5× baseline (σ=0.10 m)
- [ ] **Sparsity**: <8% degrade at 20% observation density
- [ ] **Adaptive N 범위**: N=1 (single swell) ~ N=21 (complex sea)
- [ ] **논문 길이**: 40p (OE 권장 상한)

---

## 11. 제출 상태 (2026-04-10)

- [x] `wavesplat.tex` — 40p 최종
- [x] `submission/cover_letter.tex` + PDF
- [x] `submission/graphical_abstract.pdf` + md 스펙
- [x] Highlights (frontmatter에 포함)
- [x] CRediT / Declaration / Data availability
- [ ] 공동저자 final sign-off
- [ ] Elsevier Editorial Manager 업로드
- [ ] Reviewer 4명 suggestion 이메일/소속 재확인

---

*이 문서는 논문(`paper/wavesplat.tex`)의 2026-04-10 최종 버전을 기준으로 작성되었습니다.*

⛔**2026-08-13 감사 정정** — 위 문장의 "Table I~IV 와 1:1 매칭" 은 더 이상 안심의 근거가
아니다. **논문의 그 표들 자체가 `b1_full_comparison.json` 의 분모(`0.3189`)를 쓰고 있고, 그
분모는 프레임 불일치(WS 120–220 vs ETKF/GPR 301 프레임, 100/100 에피소드)와 기준선 선택의
산물이다**(논문 원고는 사용자 지시로 편집 정지 중이라 미수정). 이 문서를 발표에 쓰기 전
§4·§5·§7·§10 을 재실행할 것.
근거: `reference_citation_bans.md`, `tmp/AUDIT_recheck_2026-08-13/`
