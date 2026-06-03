# Direction B — Broad Paradigm Survey (2026-06-01)

검토 의견 *"큰 갈래가 필요하다"* 에 대한 *모든 가능한 방안* 의 광범위 탐색. 7개 카테고리 × 약 6편/카테고리 = **42편 신규** + **기존 19편** = **총 61편** PDF 다운로드 후 abstract + 일부 본문 검토.

저장: `/home/saeyong/reserch/wavesplat/ref_paper/{01..07}/`

---

## Executive Summary

### Top 3 추천 (우선순위)

| 순위 | 후보 | 카테고리 | 점수 | 예상 작업 | 강점 | 약점 |
|---|---|---|---|---|---|---|
| **1** | **TE-TNP (Ashman 2024)** | NP variants | 4.5/5 | 4-5일 | Vanilla TNP 실패 원인 *이론적 해결*, official impl | Pure attention paradigm 제한 |
| **2** | **KalmanNet (Revach 2022)** | DA | 4/5 | 2-3주 | WaveSplat tracking *자연 확장*, ETKF 차별 명확, real-time | Init 별도 필요 |
| **3** | **Sound Field DKL + SM kernel** | DKL-GP | 4/5 | 5-7일 | Wave eq + sparse 검증, Bayesian UQ, 교수님 narrative | O(N³) scale |

### 핵심 통찰

1. **Vanilla TNP 실패의 진단 정확**: 7개 paper agent 가 모두 *spatial inductive bias 부족* 을 vanilla TNP 의 핵심 약점으로 지적. *TE-TNP* 는 정확히 *이 문제를 이론적으로 해결* (proven: stationary process ↔ TE prediction map)
2. **Real-time 제약 (13 ms) 의 영향력**: INR (per-scene), Diffusion (50+ NFE) 후보 다수가 *real-time 제약으로 탈락*. *Consistency distillation* 또는 *amortized* 접근만 생존
3. **DPPS 실패는 paradigm 한계 아님**: *Sparse gradient (6.5% coverage)* + *isotropic prior* 가 원인. *training-time physics + cross-attention conditioning* 으로 우회 가능
4. **2D directional + LiDAR + 13ms + grid-free Gaussian = 우리만 unique**: 2024-2026 신규 위협 없음 확인 (Ehlers 2025b PINO 는 *1D* 만)

---

## 카테고리별 요약

### Category 1: NP variants (`ref_paper/01_NP_variants/`)
*Papers*: ConvCNP, LBANP, BSA-TNP, **TE-TNP**, Gridded TNP, Senseiver tsunami (6편)

**핵심 발견**: Vanilla TNP 의 spatial bias 부족이 *진단 정확*. 후보:
- **TE-TNP** (4.5/5) — *cambridge-mlg/tetnp* official PyTorch. TE-MHSA/TE-MHCA 로 *translation equivariance 직접 구성*. ICML 2024 실험에서 ConvCNP 능가
- **ConvCNP** (4/5) — SetConv + CNN, 가장 simple baseline. Zero-shot OOD 검증
- **BSA-TNP** (4/5) — *G-invariant attention bias* (spatial + temporal translation 동시). 2026 AISTATS, JAX impl
- **Senseiver tsunami** (3.5/5) — *직접 비교 가능 prior art* (sparse buoy → wavefield)
- LBANP (2/5) — scaling motivation 불필요 (50 pts/frame)

### Category 2: INR (`ref_paper/02_INR/`)
*Papers*: SIREN, Fourier Features, **MMGN**, **STRIDE**, NIF, CoordX (6편)

**핵심 발견**: Per-scene optimization 이 *real-time 13 ms 제약* 과 충돌. 후보:
- **MMGN** (Luo ICLR 2024, 4/5) — *factorized Gabor INR*. SST 39% 개선 검증, WaveSplat spectral Gaussian 과 *철학적 동일*. Dispersion 자연 통합
- **STRIDE** (Tong 2026, 4/5) — LSTM/Mamba 시간 encoder + FMMNN INR. *Real-time 가능한 유일한 INR variant*. Shallow water 검증
- SIREN (3/5) — wave equation 명시적, 그러나 per-scene → real-time 불가
- NIF (3/5) — hypernetwork, interpretability 손실

### Category 3: Diffusion / Flow Matching (`ref_paper/03_Diffusion/`)
*Papers*: DPS, PiRD, Spatially-Aware (Zhuang), CoCoGen, EDM, Flow Matching, **Consistency Models**, Song SDE, LADiBI (9편)

**핵심 발견**: *Standard DPS-family 는 dead end* (DPPS 실패와 동일 fail mode). 후보:
- **PiRD + Consistency distillation** (3.5+/5) — LF→HF residual diffusion + PINN loss + 1-step inference 결합. Real-time + sparse + PDE *3가지 동시* 해결
- **Spatially-Aware Diffusion** (Zhuang 2024, 3.5/5) — cross-attention 으로 sparse conditioning, moving sensor 처리. 가장 가까운 setting
- 한계: *Generative prior 의 가치는 noisy regime* — noiseless 에선 deterministic optimizer 가 강함

### Category 4: DKL-GP (`ref_paper/04_DKL_GP/`)
*Papers*: Wilson 2016 DKL, **Sound Field DKL** ★, PI-DKL, SVGP, SVDKL, **Spectral Mixture kernel** ★ (6편)

**핵심 발견**: *Sound Field DKL* 이 *가장 가까운 prior art* — sparse sensors + wave eq regularization. 검증된 framework. 후보:
- **Sound Field DKL + SM kernel** (4/5) — μ_q ↔ (kx, ky), ω_q = dispersion 으로 hard constraint encoding. Author code 존재
- Spectral Mixture kernel 단독 (3.5/5) — safe fallback baseline
- **narrative 강점**: 교수님 Bayesian opt 배경 (UUV hull 2024) 과 자연 연결

### Category 5: Operator Learning (`ref_paper/05_Operator/`)
*Papers*: FNO, PINO, DeepONet, **Geo-FNO** ★, Multipole GNO, MeshGraphNet (6편)

**핵심 발견**: 일반 FNO/PINO 는 *grid 비호환* 으로 Direction B 에서 이미 실패. *Geo-FNO* 만 sparse point cloud 직접 처리. 후보:
- **Geo-FNO + PINO dispersion loss** (3.5/5) — point cloud 직접 입력, deformation → uniform latent grid → FNO. Dispersion 자연 통합
- 한계: 50 sparse pts 로 deformation 학습이 험난 (논문은 보통 1000+ 노드)
- **Ehlers 2025b PINO 와 차별성**: 1D vs 2D, single radar vs multi-pose LiDAR

### Category 6: Data Assimilation (`ref_paper/06_DA/`)
*Papers*: **KalmanNet** ★, **LEVDA** ★, VIRSO, FieldFormer, Senseiver tsunami, Score-based DA (6편)

**핵심 발견**: *Vanilla TNP 의 amortized 한계* 를 DA 가 *직접 해결* (sequential update). 후보:
- **KalmanNet** (4/5) — Kalman gain 만 NN. *WaveSplat tracking 모드의 자연 확장*. Real-time 13ms 유지, *Feng Wang 2025 ETKF 와 narrative 명확 차별* ("Neural Kalman gain for dispersion-constrained tracking")
- **LEVDA** (4/5, 2026) — latent 4DEnVar + neural surrogate, joint state+param. Multi-frame smoothing 으로 정확도 우위. 단 작업 4-6주
- Score-based DA (3/5) — DPPS 의 정교한 후속, real-time 약함

### Category 7: Wave-specific 경쟁자 (`ref_paper/07_Wave_competitors/` + 기존)
*Papers (추가)*: Ehlers 2025b PINO, SHRED, SWH sparse 2025

**핵심 발견 — 위협 분석**:
- **Ehlers 2025b PINO (arxiv 2508.03315)** = *단일 최대 위협*. Same timing (2025), same speed (~14 ms), same physics-informed framing. **분리선 4축**: *1D vs 2D directional*, *buoy/radar vs LiDAR*, *soft BC vs hard dispersion*, *dense grid vs grid-free Gaussian*
- **Feng Wang 2025** = ETKF baseline anchor. *stereo vision + grid* vs 우리 *LiDAR + grid-free* 차별
- **Wave-characteristics LiDAR 2025** (Ocean Eng) — *동일 센서 도메인* 이지만 *spectral statistics only* (phase-resolved 아님) → 명시적 차별 필요
- **결론**: *2D directional + ship LiDAR + 13 ms + grid-free Gaussian + hard dispersion* 5축 교집합에서 *우리만 unique*

---

## 후보 비교 매트릭스

### 정량 비교 (5점 척도)

| 후보 | 카테고리 | Spatial bias | Real-time | Sparse OK | Dispersion 통합 | Baseline 깰 가능 | 작업 |
|---|---|---|---|---|---|---|---|
| **TE-TNP** | NP | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ (DPPS-style) | 4.5/5 | 4-5일 |
| ConvCNP | NP | ★★★★ | ★★★★★ | ★★★ | ★★★★ | 4/5 | 4-5일 |
| BSA-TNP | NP | ★★★★★ | ★★★★ | ★★★★★ | ★★★★ | 4/5 | 7-10일 (PyTorch port) |
| Gridded TNP | NP | ★★★★ | ★★★★ | ★★★★ | ★★★★ | 4/5 | 7-10일 |
| **MMGN-INR** | INR | ★★★★ | ★★★ (amortized 시) | ★★★★★ | ★★★★★ | 4/5 | 10-14일 |
| STRIDE | INR | ★★★★ | ★★★★ | ★★★★ | ★★★ | 4/5 | 14-21일 |
| **PiRD+Consistency** | Diffusion | ★★★ | ★★★ (distill) | ★★★★ | ★★★★★ | 3.5/5+ | 21-28일 |
| Spatially-Aware Diff | Diffusion | ★★★★ | ★★ | ★★★★★ | ★★★ | 3.5/5 | 21일 |
| **Sound Field DKL+SM** | DKL-GP | ★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | 4/5 | 5-7일 |
| SM kernel only | DKL-GP | ★★★ | ★★★★★ | ★★★★★ | ★★★★★ | 3.5/5 | 2-3일 |
| **Geo-FNO+PINO** | Operator | ★★★★★ | ★★★★ | ★★★ | ★★★★★ | 3.5/5 | 5-6주 |
| **KalmanNet** | DA | ★★★ | ★★★★★ | ★★★★★ | ★★★★ | 4/5 | 2-3주 |
| **LEVDA** | DA | ★★★★ | ★★★ | ★★★★★ | ★★★★★ | 4/5 | 4-6주 |

### Narrative 강점 (PhD 논문 관점)

| 후보 | Novelty | 교수님 연결 | Direction B 진단과 일치 | 1순위 |
|---|---|---|---|---|
| TE-TNP | NP for ocean wave first | ML/inference 강점 | *Vanilla TNP 실패 직접 해결* | ★★★★★ |
| KalmanNet | Neural Kalman for waves | DA/Bayesian 가깝 | *Amortized vs sequential* | ★★★★★ |
| Sound Field DKL | DKL for ocean dispersion | *Bayesian opt 배경 연결* | Safe & principled | ★★★★ |
| MMGN | Factorized INR for waves | INR 새 영역 | *Spectral Gaussian 의 INR 변형* | ★★★★ |
| PiRD+Consistency | Diffusion + physics + real-time | Generative 새 영역 | 가장 야심찬 contribution | ★★★ |
| Geo-FNO+PINO | Operator for sparse waves | 2D 확장 (Ehlers 1D) | *Ehlers 2025b 와 직접 차별* | ★★★ |

---

## 최종 추천 — 단계적 진행 계획

### Phase 1: Quick Win (1주, 가장 안전)
**목표**: WaveSplat baseline (0.223) 깰 가능성 *가장 높은 후보 2개* 빠르게 시도

```
Day 1-2: TE-TNP pilot
  - cambridge-mlg/tetnp clone
  - WaveSplat dataset adapter (기존 코드 재사용)
  - vanilla TNP 와 동일 train/test split 으로 head-to-head
  - Sanity test (single episode overfit)

Day 3-4: Sound Field DKL + SM kernel pilot
  - SM kernel + dispersion lock-in baseline (단순)
  - github.com/davstrom99/dkpde 참고
  - 50 pts/frame exact GP — sub-ms inference

Day 5-7: 결과 비교 + 의사결정
  - TE-TNP nRMSE vs DKL-GP nRMSE vs WaveSplat 0.223
  - 어느 쪽이 promising 한지 결정
```

**예상 outcome**:
- *최선*: TE-TNP 가 0.223 근접/돌파 → 본격 진행
- *대안*: DKL-GP 가 0.18-0.21 달성 → safer paper
- *최악*: 둘 다 0.30+ → Phase 2 로

### Phase 2: KalmanNet (2-3주, narrative 가장 명확)
Phase 1 결과 무관, 또는 Phase 1 실패 시 진행.

```
Week 1: WaveSplat tracking 모드의 closed-form gain → GRU 로 swap
  - Dynamics f(·) = dispersion phase advance (기존)
  - Observation h(·) = forward LiDAR (기존)
  - Gain K_t = GRU(history)

Week 2: 110 episodes 학습 + tracking nRMSE 측정
  - Init 은 v2_konly 유지 (현 WaveSplat init)
  - Sequence length 5-10 frames

Week 3: ETKF (Feng Wang 2025) 와 직접 비교
  - 같은 dataset, 같은 init
  - nRMSE + speed 정량
```

**Narrative**: *"Neural Kalman gain for dispersion-constrained spectral Gaussian wave tracking — a hybrid model-based/data-driven extension that retains 13ms real-time."* Feng Wang 2025 ETKF baseline 의 *자연스러운 진화*.

### Phase 3: Long-term (옵션)
Phase 1+2 에서 *뚜렷한 winner* 나오면 그 방향으로 *paper 작성*. 안 나오면:
- LEVDA (4-6주) — 가장 정교, smoothing + joint param
- Geo-FNO+PINO (5-6주) — Ehlers 2025b 와 직접 차별
- MMGN-INR (10-14일) — spectral Gaussian 의 INR 변형

---

## 탈락한 후보들 (이유 명시)

| 후보 | 탈락 이유 |
|---|---|
| Vanilla FNO / DeepONet | Grid 비호환 (이미 Direction B 실패) |
| DPS / DPPS standard | Sparse gradient = 0 (DPPS 이미 negligible) |
| SIREN per-scene | Real-time 13 ms 불가 |
| LADiBI (잘못된 DDIS) | Text-to-image domain, ocean wave 무관 |
| VIRSO | Static reconstruction, no temporal/dispersion |
| FieldFormer | Local-only = 문제 정의 후퇴 |
| LBANP | Scaling motivation 불필요 (50 pts) |
| MeshGraphNet | Forward simulator, our task = inverse |
| CoordX | Speedup only, no accuracy/physics |
| EnKF-Net | KalmanNet 의 변종 (redundant) |

---

## 다음 단계 의사결정

**즉시 실행 가능** (Phase 1 시작):
1. **TE-TNP pilot** — 4-5일
2. **Sound Field DKL + SM kernel pilot** — 5-7일

**시작 전 확인 필요**:
- *GPU 환경* — Stage 1 TNP pilot 에서 확인했듯 학습은 GPU 필수
- *작업 우선순위* — 본 paper (OCEANS 2026) 작업과 병행 vs 분리

**Paper narrative 측면**:
- Phase 1 winner 가 *TE-TNP* 면 → *"NP for ocean wave estimation, first 2D directional + sparse LiDAR"*
- Phase 1 winner 가 *DKL-GP* 면 → *"Bayesian wave field inference with dispersion-aware deep kernel"*
- Phase 2 KalmanNet → *"Neural extension of EnKF baselines for real-time wave tracking"*

3가지 모두 *strong PhD contribution* + Direction B 검토 의견 *"큰 갈래"* 에 대응.

---

## 첨부 — Paper 다운로드 인벤토리

총 **61편** (`/home/saeyong/reserch/wavesplat/ref_paper/`):

| 카테고리 | 파일 수 | 핵심 paper |
|---|---|---|
| 기존 (루트) | 19 | Desmars × 3, Ehlers × 3, Wang Pan × 2, Klein, Feng Wang, Hunt, Tessendorf, Kingma, Möller-Trumbore, Tang, Vof-EnKF, etc. |
| 01_NP_variants | 6 | ConvCNP, LBANP, BSA-TNP, TE-TNP, Gridded TNP, Senseiver tsunami |
| 02_INR | 6 | SIREN, Fourier Features, MMGN, STRIDE, NIF, CoordX |
| 03_Diffusion | 9 | DPS, PiRD, Spatially-Aware, CoCoGen, EDM, Flow Matching, Consistency, Song SDE, LADiBI |
| 04_DKL_GP | 6 | Wilson DKL, Sound Field DKL, PI-DKL, SVDKL, SM kernel, etc. |
| 05_Operator | 6 | FNO, PINO, DeepONet, Geo-FNO, Multipole GNO, MeshGraphNet |
| 06_DA | 6 | KalmanNet, LEVDA, VIRSO, FieldFormer, Senseiver tsunami, Score-based DA |
| 07_Wave_competitors | 3 | Ehlers 2025b PINO, SHRED, SWH sparse 2025 |
