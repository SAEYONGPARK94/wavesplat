# AAETR / Schieler 원문 검증 + 손상 PDF 복구 (2026-08-05)

> **한 줄 결론**
> ★**오늘 Transformer 검토의 헤드라인 — "'최초'는 미검증이 아니라 거짓이다" — 은 원문으로도 유지된다.**
> AAETR 본문은 FINDING.md 가 초록에서 추정했던 **아키텍처·손실·좌표입력을 축자로 확인**해 준다(§1).
> **단 두 개의 보조 서술은 정정해야 한다**: (i) *"임의 기하 지원"* 은 AAETR 이 **한 번도 주장하지 않는다**(ARGUED 였음),
> (ii) *"차원만 1D angle"* 은 **과소 기술**이다 — 미지수 개수·위상·시간축·물리제약·정련단계에서 **7가지가 다르다**(§3).
> 그리고 **선점이 한 겹 더 깊다는 새 사실**: "학습 초기화 → 2차 Newton 정련" 은 Schieler(2022/24)가 아니라
> ★**Barthelme & Utschick, IEEE TSP 2021** 이 먼저다 — Schieler 가 §I 에서 **자기 선행연구로 명시 인용**한다(§2.3).
>
> **판정: AAETR = 부분 선점**(구조 동일 · 문제 다름). **인용된 축(1b)에 한해서는 완전 선점.**
> **Schieler(1c) = 완전 선점**(축자 문구 실재 + 동일 Gauss-Newton 정련 + 동일 런타임 논증).

**표기 규율**: `[M]` = 논문 본문에서 읽은 것(절·식 번호 표기), `[A]` = 내 추론.
**원자료 provenance**: `aaetr_2408.09362.pdf` md5 `190c95afd3ebd948f302918e8903858a` (8쪽, arXiv:2408.09362**v1**, 2024-08-18, cs.CV) ·
`schieler_2211.04846.pdf` md5 `8f220be72839f5521b134511aff1d95b` (5쪽, arXiv:2211.04846**v4**, 2023-12-19, eess.SP) ·
텍스트 추출본 `aaetr.txt` / `schieler.txt` 동봉(PyMuPDF).

---

## 1. AAETR 원문 (임무 1)

**서지** `[M]` Zhaoxuan Zhu, Chulong Chen, Bo Yang (**Motional, Inc.**), *"Angle of Arrival Estimation with Transformer:
A Sparse and Gridless Method with Zero-Shot Capability"*, arXiv:2408.09362v1. 8쪽(참고문헌 포함), **부록 없음**.

### 1.1 무엇을 추정하는가 — **1차원 방위각 + 진폭 + 신뢰도. 위상 없음.**

`[M]` §AAETR 문제정의(3쪽):
> "The output of the algorithm are the predicted angle θ̃ ∈ R^{M×1}, magnitude α̃ ∈ R^{M×1} and confidence of p̃ ∈ R^{M×1}
> of the detection targets, where M is the maximal number of detection targets as a hyperparameter."

`[M]` M = **80 learnable queries**(§Training Details, 5쪽), embedding dim 128, 총 **1.6M 파라미터**.

`[M]` ★**추정 대상은 단일 (range, Doppler) 빈 안의 방위각 하나뿐이다.** 평가 설정이 *"target counts ... from 2 to 10
**per range-Doppler bin**"*(§Dynamic Range and Scenario Configuration, 4쪽)이고, 파이프라인상 위치가
*"the Range-Doppler (RD) cube undergoes DOA ... to generate the Range-Angle-Doppler (RAD) cube"*(§Introduction, 1쪽)이다.
⟹ **range·Doppler 두 축은 이미 FFT 격자로 처리된 뒤**이고, AAETR 은 **남은 한 축만 gridless 로 푼다.**

`[M]` 신호모델 식 (5): 조향벡터 `a_m(θ_m) = α_m [e^{-jωτ_1,m(θ_m)}, ..., e^{-jωτ_K,m(θ_m)}]^T` 에서
**ω 는 반송파 각주파수(기지값)** 이지 추정 대상이 아니다. ⟹ **타깃당 미지수 = θ 1개 (+ nuisance 진폭 α).**

`[A]` 우리와의 대비: 우리는 성분당 **(kx, ky, A, φ) 4개**를 추정하고 ω 는 분산관계로 **묶인다**.
**AAETR 에는 위상 φ 를 추정하는 head 자체가 없다**(출력 3-tuple 에 없음). 위상 없는 출력은 **장(場)을 시간 전진**시킬 수 없다.

### 1.2 입력 — **고정 48소자 가상배열의 단일 스냅샷 복소벡터 + 소자 3D 좌표(PE)**

`[M]` §AAETR(3쪽):
> "The input to the algorithm is the measured data vector y ∈ C^{K×1}, and the virtual array element positions
> x ∈ R^{K×3} is also available, where K is the number of virtual array elements."
> "The real and imaginary parts of the input measured data vector y ... and the virtual array element positions x
> are first projected to high dimensional embedding space ȳ ∈ R^{K×D} and x̄ ∈ R^{K×D} and subsequently fed into
> **6 transformer encoder blocks as the input and the positional encoding**, respectively."

`[M]` K = **48**(6 Tx × 8 Rx sparse MIMO, §Synthetic Dataset 4쪽). **단일 스냅샷**(single snapshot; §Related Works 에서
multi-snapshot 요구를 MUSIC 의 결점으로 지목). **시간축 토큰 없음.**

`[M]` ★**"arbitrary" / "geometry" 라는 단어는 본문에 0회 등장한다**(전문 grep). 학습·평가·실측 전부 **동일한 사내 LRR 배열 하나**다.
⟹ **정정**: 오늘 FINDING.md 표 1b 의 *"array element 3D 좌표를 측정값과 함께 입력 토큰으로 ⟹ **임의 기하 지원**"* 에서
앞부분은 `[M]` 로 확인되지만 **뒷부분("⟹ 임의 기하 지원")은 논문의 주장이 아니라 우리 추론**이다. `[A]` 로 강등할 것.
(구조적으로 가능은 하지만 **논문이 시연도 주장도 하지 않았다.**)

### 1.3 물리 제약 — **아키텍처 안에는 없다. 물리는 전부 시뮬레이터에 있다.**

`[M]` 전문 grep: `constraint` **0회**, `dispersion` **0회**, `manifold` **0회**, `projection layer` **0회**.
`[M]` 물리(식 4–6, 조향벡터·far-field 가정)는 **§Synthetic Dataset 안**에 있고, 용도가 명시적으로 학습데이터 생성이다:
> "we propose to use a **physics-based high-fidelity simulation pipeline to generate ground truth data** efficiently in a large scale."

⟹ `[M]` **네트워크는 (y, x) → (θ̃, α̃, p̃) 의 순수 블랙박스**다. 우리 설계의 3번 항목
(`docs/learned_offgrid_design.md` §Architecture-3: *"Dispersion hard constraint: ω = √(g|k|·tanh(|k|d)). **ω is never free.**"*)
에 **대응물이 없다**. 미분가능 render head(같은 문서 §Architecture-4)에 해당하는 것도 없다.

### 1.4 주장 범위 — **종단(end-to-end). 정련(polish) 단계 없음.**

`[M]` 전문 grep: `refine` **0회**, `Newton` **0회**, `polish` **0회**, `iterative refinement` **0회**(정련을 뜻하는 어떤 단어도 없음).
`[M]` §Conclusion: *"AAETR requires fewer tunable hyperparameters and is **end-to-end trainable**, making it well-suited
for integration into deep learning radar perception pipelines ... a promising **drop-in module**."*
`[M]` 실측 배포시 후처리는 **정련이 아니라 재격자화**다: *"As AAETR is inherently gridless, **summation splatting**
(Niklaus and Liu 2020) is employed to convert prediction targets to RAD data cubes"*(§Zero Shot on Real Data, 6쪽).

⟹ ★**우리 저장소 기록과 정반대의 주장이다.** 우리 6개 구현은 전부 영 예측자 바닥(0.248/0.248/0.2566/0.2596/0.28)이었고
바닥을 깬 건 **뒤에 붙은 Gauss-Newton polish**(0.2596 → 0.1099)였다. AAETR 은 **망 단독으로 IAA 를 이긴다고 주장**한다.
`[A]` 이 차이가 "AAETR 이 우리 실패를 반증한다"는 뜻은 **아니다** — 문제(미지수 1개 vs 4개)·학습규모(§1.5)·평가지표(§1.6)가 전부 다르다.
`[A]` 오히려 **가장 유용한 대조**는 이것이다: **미지수 1개 문제에서는 망 단독이 되고, 미지수 4개 + 위상 문제에서는 안 됐다.**

### 1.5 학습 규모 — `[M]` **63M 샘플**, batch 32,768, **16× A10 GPU**, 온더플라이 생성.
`[A]` 우리 100 에피소드와 **5~6 자릿수** 차이. `[M]` 저자들도 이 무한 데이터 공급이
*"critical for the zero-shot transfer"* 라고 명시한다.

### 1.6 평가 — `[M]` **검출 문제로 평가**한다. TP = GT 로부터 **±0.5°** 이내 + 신뢰도 임계 초과.
지표는 **precision-recall / max-F1 / 각도·크기 L1 오차**(§Comprehensive Evaluation Method). 조건: 2–10 targets/bin, SNR 15–35 dB, dynamic range 13 dB.
`[M]` ★**실측 데이터에서의 주장은 정성적일 뿐이다**: *"quantitative evaluation is challenging due to the **lack of ground truth**
in these scenarios"*(§Zero Shot on Real Data, 7쪽). 증거는 BEV 그림 3장 + 라이다 대조 + "건물 뒤 검출은 전부 FP" 논증.
`[A]` ⟹ AAETR 의 sim-to-real 주장은 **정량 이전 단계**다. 우리 CDIP support / N5 층화 같은 **분포이동의 정량화**는 이 논문에 없다.

---

## 2. Schieler 원문 (임무 2)

**서지** `[M]` S. Schieler, S. Semper, R. Faramarzahangari, M. Döbereiner, C. Schneider, R. Thomä,
*"Grid-free Harmonic Retrieval and Model Order Selection using Deep Convolutional Neural Networks"*, arXiv:2211.04846v4 (EuCAP 2024).
⚠ **저자 6인**이다 — FINDING.md 의 *"Schieler, Semper & Thomä"* 는 축약 표기이므로 논문 인용 시 정정 필요.

### 2.1 축자 문구 — **실재한다(초록). 본문에는 같은 뜻의 문장이 3개 더 있다.**

`[M]` **초록 마지막 문장(축자)**:
> "We also compare the performance to other harmonic retrieval methods and **integrate it into an existing maximum
> likelihood estimator for efficient initialization of a gradient-based iteration**."

`[M]` 본문의 동치 문장들:
- §I Introduction: *"Motivated by the approach in [11], we also use the estimates directly to **warm-start a second-order
  gradient iteration** and show that the initial guesses allow convergence with high probability."*
- §IV Results: *"When using the estimates provided by our approach to **initialize (8)**, the estimation accuracy for higher
  SNRs improves significantly from only 10 iterations of (8). ... **we can initialize the iteration much more efficiently
  with a single forward from the network** but still accurate enough for the gradient iteration to converge."*
- §V Conclusion: *"the estimates of up to 20 paths can be used to **warm-start a second-order gradient iteration of the
  highly non-convex likelihood function**."*

### 2.2 맥락이 우리 것과 같은가 — **정련의 정체·비용논증까지 같다.**

`[M]` 식 (8)은 *"essentially a **second-order Gauss-Newton scheme**, since F is the Fisher-Information matrix and J is the
Jacobian matrix of the negative log-likelihood"*, 목적함수 식 (9) `λ(γ,τ,α) = (1/σ²)‖Y − S(γ,τ,α)‖²_F`.
⟹ `[A]` **우리 polish 와 같은 대상(비선형 최소제곱)에 같은 알고리즘(Gauss-Newton)** 이다.

`[M]` 런타임 논증도 같다: DFT peak search **3 ms** · 제안 망 **19 ms** · 망 + GN 10회 **60 ms** · ML(RIMAX) **11.9 s**.
> "our approach addresses applications requiring **fast fixed-clock estimates**, where the **accuracy-runtime trade-off can be
> regulated by the number of gradient iterations**."

`[A]` 이 문장은 우리 척추("제어-호환 ≤100 ms")의 논증 구조와 **같다** — 08-04 백지 재검토가 지연 헤드라인을 죽인 것과 별개로,
**"정련 반복수로 정확도-지연을 조절한다"는 서술 자체가 이미 선행문헌에 있다.**

### 2.3 ★새 발견 — **선점이 한 겹 더 깊다(2021)**

`[M]` Schieler §I 이 자기 선행연구로 명시 인용:
> "the work in **[11]** combines a regression task solved by a DNN with gradient steps on the likelihood function of the data
> to perform DoA estimation. **The grid-free estimates from the DNN are used as initial guesses for a second-order Newton method.**
> This approach combines the best of two worlds: fast, robust, but approximate initial estimates combined with an iterative
> high-resolution method allowing quadratic convergence, but only after successful initialization provided by the DNN."

`[M]` **[11] = A. Barthelme and W. Utschick, "A Machine Learning Approach to DoA Estimation and Model Order Selection for
Antenna Arrays With Subarray Sampling," IEEE Transactions on Signal Processing, 2021, DOI 10.1109/TSP.2021.3081047.**

⟹ ★**"학습 초기화 → 고전 정련" 축의 최선행은 Schieler(2022/24)가 아니라 Barthelme & Utschick(2021)이다.**
`[M]` Schieler 는 자기 기여를 *"Especially compared to [11], this is an **improvement in terms of parameter dimension,
quantity, and data complexity**"* 라고 **증분(增分)으로 스스로 규정**한다.
`[A]` ⟹ 이 축에서 우리가 할 수 있는 최선의 진술도 **"파라미터 차원·개수·데이터 복잡도에서의 증분"** 형태뿐이다. 개념 신규성은 없다.

### 2.4 그러나 아키텍처는 우리 것도 AAETR 것도 아니다

`[M]` **CNN 이지 transformer 가 아니다**(§III, Fig. 1 = Conv2D/BatchNorm/ReLU 스택).
`[M]` **Hungarian / set prediction 없음** — 대신 **YOLO 식 셀-상대 인코딩**: 파라미터 공간을 I·J 셀로 나누고
셀당 경로수 μ 와 중심으로부터의 편차 (Δτ, Δα) 를 회귀. *"Our encoding is a task-specific modification of [12]"*,
`[12] = Redmon et al., YOLO, CVPR 2016`. 저자들 스스로 **"quasi-grid-free"** 라 부른다(초록).
`[M]` **입력은 규칙격자 텐서**: `Y ∈ C^{Nf×Nt}` → 8개 윈도우 → 2D-DFT → 4개 실수 사상(Re, Im, log|·|, ∠) →
CNN 입력 `Y₂ ∈ R^{4·8 × Nf × Nt}`(§III-A). ⟹ **점군도 아니고 좌표 토큰도 아니다.**
`[M]` 추정 대상: 지연 τ·도플러 α (**2D**) + **모델 차수**, 최대 **20 경로**, 단일 스냅샷. 복소 가중 γ 는 망이 추정하지 않음.
`[M]` **τ 와 α 사이에 물리적 구속이 없다** — 우리 ω = f(k) 같은 **하드 제약의 대응물이 없다**.

---

## 3. ★ 우리와의 차이 전수 (AAETR 기준)

| # | 축 | AAETR `[M]` | 우리(`docs/learned_offgrid_design.md` §Architecture, 저장소 재확인) | 판정 |
|---|---|---|---|---|
| 0 | **아키텍처** | transformer **enc 6 + dec 6**, dim 128, **learnable query 80**, cross-attention, FFN head | set encoder(self-attn) + **DETR decoder, N_q=32 learnable query** | ★**동일** |
| 0' | **손실** | *"A modified set prediction loss in (Carion et al. 2020)"*, **bipartite matching** 식 (1)(2)(3), L1(angle)+L1(magnitude)+log p | Chamfer/**Hungarian** over (kx,ky) + field MSE + obs MSE | ★**동일 계열** |
| 0'' | **입력 형식** | 측정값 + **좌표를 positional encoding 으로** 토큰화 | per-point token = MLP(normalize[x,y,t,z]) | ★**동일 착상** |
| 1 | **미지수/타깃** | **θ (1개)** + nuisance 진폭. **위상 없음** | **kx, ky, A, φ (4개)**, ω 는 분산관계로 종속 | **다름** |
| 2 | **주파수** | 식 (5)의 ω = **기지 반송파** | ω = **추정량**(하드 제약으로 k 에 묶임) | **다름** |
| 3 | **물리 제약** | **아키텍처에 없음**(물리는 시뮬레이터에만) | **분산 하드제약 + 파라미터 없는 미분가능 render head** | **다름** |
| 4 | **입력 지지집합** | **고정** 48 가상소자, 매 샘플 동일 기하, **단일 스냅샷** | **불규칙·이동**하는 LiDAR 점군 (x,y,t,z), 프레임당 ~50점, 2.2·Tp 창 | **다름** |
| 5 | **시간축** | **없음**(공간 전용, 스냅샷 1개) | **있음** — t 채널이 synthetic aperture 를 나른다 | **다름** |
| 6 | **문제 축 수** | **3축 중 1축만** gridless (range·Doppler 는 상류 FFT 격자) | 전 파라미터가 off-grid | **다름** |
| 7 | **출력의 운명** | **summation splatting 으로 RAD 격자에 되박음** | 격자로 되돌리지 않음 — 추정치 자체가 산출물 | **다름** |
| 8 | **정련 단계** | **없음**(refine/Newton/polish 0회) | **Gauss-Newton polish 가 유일하게 바닥을 깬 팔**(0.2596→0.1099) | **다름**(⟹ Schieler/Barthelme 축) |
| 9 | **예보** | **없음** — 정지 장면 추정 | **리드타임 예보**가 척추 | **다름** |
| 10 | **평가** | 검출 PR / max-F1 / ±0.5° TP / L1 | 장(場) nRMSE / skill score | **다름** |
| 11 | **실측 증거** | **정성적만**(*"lack of ground truth"*) | GT 있는 실측 창에서 정량 | **다름** |
| 12 | **학습 규모** | **63M** 샘플, 16 GPU | 100 에피소드 | **다름** |

⟹ ★**"차원만 1D" 는 틀렸다.** 정확한 진술은 다음이다:
> **[M] 아키텍처(0)·손실(0')·좌표입력 착상(0'')은 AAETR 이 완전히 선점했다.
> [M] 그러나 문제는 다르다 — 미지수 1개 vs 4개, 위상 없음, 시간축 없음, 물리 하드제약 없음,
> 3축 중 1축만 gridless, 정련 없음, 예보 없음.**

---

## 4. 손상 PDF 복구 (임무 3)

`[M]` **원본 식별 확정**: 손상 파일의 압축 스트림을 inflate 한 결과 *"**grid encoder**, which moves point-wise data
representations onto a structured grid of **pseudo-tokens**"* 등 본문 문구가 나왔고, 이는
**Ashman, Diaconu, Langezaal, Weller, Turner, "Gridded Transformer Neural Processes for Large Unstructured
Spatio-Temporal Data", arXiv:2410.06731** (ICML 2025) 와 일치.

| | 손상본(교체 전) | 신규본(교체 후) |
|---|---|---|
| 크기 | **2,097,152 B(정확히 2 MiB)** | **14,506,621 B** |
| md5 | `edb2c9e1c18f45d67f3cc3987fb99c97` | ★**`88d6484745ca77650d45fbe0d96ef825`** |
| 페이지 | **0**(fitz 가 열지 못함 / `%%EOF`·xref 없음, pypdf `PdfStreamError`) | ★**31**(1쪽·31쪽 텍스트 추출 정상, 20쪽 Fig.13 정상) |
| 출처 | 불명(2 MiB 에서 잘린 다운로드) | `https://arxiv.org/pdf/2410.06731v2` (curl) |

`[M]` **교체 완료**: `/home/vexit/wavesplat/wavesplat/ref_paper/01_NP_variants/GriddedTNP_Ashman2024.pdf`.
`[A]` 잘린 위치가 **정확히 2 MiB** 인 것은 다운로더의 크기 상한 아티팩트로 보인다.
`[M]` **후속 점검 완료**: `ref_paper/` 전체 **PDF 73개**를 훑어 1/2/4 MiB **정각** 크기의 파일이 **더 없음**을 확인
(교체 후 기준) ⟹ **같은 절단 사고의 재발본은 없다**. ⚠단 정각이 아닌 절단은 이 검사로 못 잡는다.

---

## 5. ★ 선점 강도 판정 (임무 4)

### 5.1 축별 판정

| 축 | 판정 | 근거(원문 확인) |
|---|---|---|
| **1b. attention + learnable query + Hungarian + gridless + 좌표 입력** | ★**완전 선점** | AAETR §AAETR / §Training Details / 식 (1)(2)(3) 전부 축자 확인. **오늘 FINDING 의 판정이 본문으로도 유지**된다. |
| **1c. 학습 초기화 → 고전 optimizer 정제** | ★**완전 선점, 그리고 더 오래됨** | Schieler 초록 축자 + 본문 3곳 + 식 (8)(9)가 **Gauss-Newton**임을 명시. ★**최선행은 Barthelme & Utschick, IEEE TSP 2021** (Schieler [11]). |
| **전체 명제**("attention set-to-set 이 파랑 성분 집합을 한 번의 forward 로 뱉고 optimizer 가 정제한다") | **부분 선점** | 각 **성분**은 전부 선점됐고, **결합체를 그대로 담은 논문은 확인되지 않았다**. 남는 건 §3 표의 1–12 = **application delta**. |
| **AAETR 단독** | ★**부분 선점**(구조 동일·문제 다름) | §3 표. 인용된 축(1b)에 한해서는 완전 선점. |

### 5.2 헤드라인은 유지되는가 — **유지된다.**

`[M]` 오늘 Transformer 검토의 *"'최초'는 미검증이 아니라 거짓이다"* 는 **뒤집히지 않는다.**
금지 대상이던 주장들(attention set-to-set 으로 물리 파라미터 집합 출력 / 학습 초기화 → 고전 정련 / 성분집합 prior)은
**전부 원문 축자로 선점이 확인**되었다. `[A]` 오히려 §2.3 의 **2021년 선행** 발견으로 **선점이 1년 더 깊어졌다.**

`[A]` **그러나 세 가지를 정정·추가해야 한다**(뒤집힘 아님, 정밀화):
1. **정정** — *"임의 기하 지원"* 은 AAETR 의 주장이 아니다(§1.2). `[M]`→`[A]` 강등.
2. **정정** — *"차원만 1D angle"* 은 과소 기술이다. §3 의 12개 차이로 대체.
3. **추가** — *"Schieler, Semper & Thomä"* → **저자 6인**. 그리고 **1c 축의 최선행은 Barthelme & Utschick 2021**.

### 5.3 살아남는 진술 (⛔금지문구를 건드리지 않는 형태)

`[A]` 다음은 원문 대조 후에도 말할 수 있다:
- **"DETR 식 집합예측을, 위상까지 포함한 결맞음(coherent) 파랑 성분에 분산관계 하드제약과 미분가능 render head 를
  얹어 적용한 사례는 확인된 범위에서 없었다"** — 단 ⛔**"최초"라는 단어는 여전히 금지**(§6 미확인 목록이 넓다).
- **"미지수 1개(방위각) 문제에서는 학습 단독이 고전 초해상을 이겼고(AAETR), 미지수 4개+위상 문제에서 우리 6개 구현은
  영 예측자 바닥에 머물렀다"** — `[A]` **문제 난이도 대조로서의 가치**가 이번 확인의 실질 소득이다.
- ⛔**"학습 초기화 → 정련"의 신규성은 완전히 사망**(2021·2022·2024 3중).

---

## 6. 확인 못 한 것 (열람 범위 = 주장 범위)

- **Barthelme & Utschick (IEEE TSP 2021) 원문 미열람** — Schieler §I 의 **서술을 통한 간접 확인**뿐. 유료장벽 가능.
- **Schieler 저널판 arXiv:2506.23621 (IJMWT 17:892–899, 2025) 미열람** — EuCAP 판(v4)만 읽음.
- **AAETR 은 v1 만 존재 확인** — 게재본(학회/저널) 여부 미확인. **부록 없음**은 확인됨(8쪽 = 본문 7 + 참고문헌 1).
- **미열람 유지**: SlotFlow(2511.23228) · arXiv:2601.04581(Spectral point transformer) · NMRTrans(2602.10158) ·
  JOMAE 147(2) 022002 · JMSE 13(11) 2196. ⟹ **오늘 FINDING §5.2 의 2b·2c "공백"은 여전히 검증되지 않았다.**
- **GriddedTNP 신규본은 페이지 무결성만 확인**(1·20·31쪽 텍스트 추출 정상). **내용 재검토는 하지 않음** ⟹
  오늘 FINDING §5.1 의 GriddedTNP 관련 수치(Swin-TNP 127 ms @ bs=8 등)는 **손상본 2–19쪽 기반 그대로**이고,
  1쪽·20쪽 이후를 반영한 재확인은 **미실시**.
- **새 런 0건** — 이 문서의 모든 수치는 원문 인용 또는 저장소 기존 기록이다.
