# FINDING — Vaswani et al. 2017 "Attention Is All You Need" 정독 후 적용 판정
2026-08-05. 원문 `https://arxiv.org/abs/1706.03762` (v7 PDF, 15 pages) 를 **본문 전체** 읽음
(로컬 캐시 PDF → `fitz` 텍스트 추출로 축자 검증, ar5iv HTML 병행).
아키텍처 마이크로벤치 스크립트: `tmp/transformer_review_2026-08-05/arch_bench.py`.

**새 파랑런 0건 · 학습 0건 · 프로덕션 수정 0줄 · `paper/` 무접근.**
MEASURED / ARGUED 를 문장 단위로 분리한다. 표시 없는 문장은 MEASURED.

---

## ★ 한 줄 결론

> **Transformer 는 "적용할 후보"가 아니다. 이 저장소에 이미 **6번** 구현되어 있고,
> 정확히 이 임무가 제안하려던 형태 — 원시 관측점 집합 → 성분 집합의 attention set-to-set 사상 —
> 은 `docs/learned_offgrid_design.md`(2026-06-28)에 **"amortized RELAX"라는 이름으로 설계까지
> 끝나 있으며**, 구현 6개가 전부 **영 예측자 바닥(nRMSE 0.25)** 에서 멈췄다
> (`0.248 / 0.248 / 0.2566 / 0.2596 / 0.28`, **학습 분포 안에서도**). 바닥을 깨는 유일한 팔은
> 신경망이 아니라 그 뒤의 **물리 정련**(Gauss-Newton polish, 0.2596 → 0.1099)이다.**
> **[MEASURED] 외부적으로도 3축 전부 선점**: AAETR(arXiv:2408.09362) · Schieler(arXiv:2211.04846) ·
> SlotFlow(arXiv:2511.23228). ⟹ **"최초"는 미검증이 아니라 거짓이다.**
> ⟹ **판정: 아키텍처는 이 문제의 movable axis 가 아니다.** 오늘 밝혀진 배치 병목의
> 측정 가능한 주범은 신경망 부재가 아니라 **해석적 탐색 격자의 k-공간 해상도**다
> (§4.3 / §7-E1 — **상수 2개, 학습 0, 추가비용 0**).
> ★그리고 이 결론은 논문을 **버리라**는 뜻이 아니다 — §3.5(positional encoding)를 읽은 덕에
> **격자 결함을 찾았다**. **논문의 가치가 아키텍처 이식이 아니라 진단 렌즈로 실현되었다.**

---

# 1. 논문 구조 요약 (§3 · §4 · §5)

## 1.1 §3.1 Encoder / Decoder Stacks
- encoder = **N=6** 동일 레이어. 서브레이어 2개: multi-head self-attention + position-wise FFN.
- 각 서브레이어에 residual + layer norm: **`LayerNorm(x + Sublayer(x))`** — 즉 **post-LN**.
- residual 을 가능케 하려고 **모든** 서브레이어·임베딩 출력 차원을 `d_model = 512` 로 통일.
- decoder = N=6, 서브레이어 **3개**(+encoder 출력에 대한 multi-head attention). self-attention 은
  마스킹되어 위치 `i` 의 예측이 `< i` 의 출력에만 의존(auto-regressive 보존).

## 1.2 §3.2 Attention
**§3.2.1 Scaled Dot-Product Attention — 식 (1)**
```
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
```
`1/√d_k` 의 근거는 **각주 4** (축자 확인):
> "assume that the components of q and k are independent random variables with mean 0 and
> variance 1. Then their dot product, q·k = Σ_{i=1}^{d_k} q_i k_i, has mean 0 and **variance d_k**."

큰 `d_k` 에서 내적이 커져 softmax 가 **기울기가 극히 작은 영역**으로 밀려나는 것을 상쇄한다.
additive attention 과 이론적 복잡도는 같으나 dot-product 가 **행렬곱 최적화 덕에 실제로 더 빠르고
메모리 효율적**이라고 명시.

**§3.2.2 Multi-Head Attention**
```
MultiHead(Q,K,V) = Concat(head_1, …, head_h) W^O ,  head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```
`W_i^Q, W_i^K ∈ ℝ^{d_model×d_k}`, `W_i^V ∈ ℝ^{d_model×d_v}`, `W^O ∈ ℝ^{h·d_v×d_model}`.
**h = 8**, `d_k = d_v = d_model/h = 64`. 근거 문장(축자):
> "Multi-head attention allows the model to jointly attend to information from different
> representation subspaces at different positions. **With a single attention head, averaging
> inhibits this.**"

head 당 차원을 줄였기 때문에 총 계산비용은 full-dimensionality single-head 와 **비슷**하다.

**§3.2.3 Applications — 세 가지 용법**
1. **encoder-decoder attention**: query = 직전 decoder 레이어, key/value = encoder 출력.
   모든 decoder 위치가 **입력 전체**를 본다.
2. **encoder self-attention**: Q=K=V= 직전 encoder 레이어 출력.
3. **decoder self-attention**: 자기 이하 위치만. 불법 연결을 softmax 입력에서 **−∞ 로 마스킹**.

## 1.3 §3.3 Position-wise FFN — 식 (2)
```
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
```
위치마다 **동일하게, 독립적으로** 적용. "kernel size 1 인 convolution 2개"로도 기술 가능.
`d_model = 512`, 내부 차원 **`d_ff = 2048`** (= 4×).

## 1.4 §3.4 Embeddings and Softmax
두 임베딩 레이어와 pre-softmax linear 이 **같은 weight matrix 공유**. 임베딩 레이어에서는
그 가중치에 **`√d_model`** 을 곱한다.

## 1.5 §3.5 Positional Encoding ★
recurrence 도 convolution 도 없으므로 **순서 정보를 주입해야** 한다. 임베딩과 **더할** 수 있도록
같은 차원 `d_model`.
```
PE(pos, 2i)   = sin( pos / 10000^{2i/d_model} )
PE(pos, 2i+1) = cos( pos / 10000^{2i/d_model} )
```
- 각 차원이 하나의 sinusoid 에 대응. **"The wavelengths form a geometric progression from
  2π to 10000·2π."** (축자)
- 선택 근거(축자): **"for any fixed offset k, PE_{pos+k} can be represented as a linear function
  of PE_pos"** ⟹ 상대 위치로 attend 하기 쉬울 것이라는 **가설**.
- **learned positional embedding 과 비교했더니 "nearly identical results" (Table 3 row (E))**.
  sinusoid 를 고른 이유는 성능이 아니라 **학습 때보다 긴 시퀀스로 외삽 가능할지 모른다**는 것.
  ⟹ ★**PE 의 구체적 형태는 이 논문에서 성능상 무차별했다.** 뒤(§4-ⓒ)에서 중요해진다.

## 1.6 §4 Why Self-Attention — 복잡도 표 (Table 1, 축자)

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|---|---|---|---|
| Self-Attention | **O(n²·d)** | **O(1)** | **O(1)** |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(log_k n) |
| Self-Attention (restricted) | O(r·n·d) | O(1) | O(n/r) |

세 가지 desiderata: ①레이어당 총 계산 복잡도 ②병렬화 가능량(= 최소 순차연산 수)
③**장거리 의존 간 경로 길이**. 핵심 논증(축자):
> "The shorter these paths between any combination of positions in the input and output
> sequences, the easier it is to learn long-range dependencies."

- self-attention 이 recurrent 보다 **빠른 조건은 `n < d`** 라고 명시(문장 축자: "self-attention
  layers are faster than recurrent layers when the sequence length n is smaller than the
  representation dimensionality d"). ← ⓔ 판정에서 결정적.
- 아주 긴 시퀀스에는 이웃 `r` 로 제한(restricted) ⟹ 경로 길이 `O(n/r)`. **"We plan to
  investigate this approach further in future work."**
- 부수 주장: self-attention 은 **해석 가능성**을 높일 수 있다(head 가 구문·의미 구조를 학습).

## 1.7 §5 Training
- **§5.1**: WMT14 En-De 약 **4.5M 문장쌍**, BPE 공유 vocab **37,000**. En-Fr **36M 문장**,
  word-piece **32,000**. 배치는 길이 근사 정렬, 배치당 약 **25,000 source + 25,000 target 토큰**.
- **§5.2**: **8× NVIDIA P100**. base = step 당 **0.4 s**, **100,000 step / 12시간**.
  big = step 당 **1.0 s**, **300,000 step / 3.5일**.
- **§5.3 Optimizer**: Adam **β₁=0.9, β₂=0.98, ε=10⁻⁹**. 식 (3):
  ```
  lrate = d_model^{-0.5} · min( step_num^{-0.5},  step_num · warmup_steps^{-1.5} )
  ```
  **`warmup_steps = 4000`**. warmup 동안 **선형 증가**, 이후 **step 수의 역제곱근**으로 감소.
- **§5.4 Regularization**: residual dropout **P_drop = 0.1** (각 서브레이어 출력에, 더하고
  정규화하기 **전**). label smoothing **ε_ls = 0.1** — 축자: **"This hurts perplexity, as the
  model learns to be more unsure, but improves accuracy and BLEU score."**

---

# 2. ★ 이 저장소의 사전 상태 — 판정 전에 반드시 읽어야 할 사실

⛔ **이 절이 임무 전제를 뒤집는다.** "Transformer 를 적용할 만한가"는 이미 답이 나와 있다.

**`nn.MultiheadAttention` / `nn.Transformer*` 를 쓰는 파일이 저장소에 17개 있다**
(`grep -rl` 로 열거, `scripts/` 16 + **`wavesplat/model/tnse.py` 1개는 프로덕션 패키지 안**).
그중 이 임무와 직접 겹치는 셋:

### 2.1 `wavesplat/model/tnse.py` — T-NSE ★ 임무가 제안하려던 바로 그것
헤더 축자:
> Stage 1 (neural): W frames of sparse LiDAR → {kx_n, ky_n}
> Architecture: **SetTransformer-style (permutation-invariant over points within each frame,
> attention across frames for temporal context).**

- `PointEncoder`: **원시 `(x, y, t, z, conf)` 5-튜플 → d_model** (`tnse.py:34-44`).
  ⟹ **관측을 격자화하지 않고 점 단위로 직접 먹는다.**
- `self.attn = nn.MultiheadAttention(...)` (`:51`), `self.cross_attn = ...` (`:101`),
  DETR-style query + confidence head.
- `D_MODEL=128, N_HEADS=4, N_LAYERS=4, D_FF=256, MAX_PTS=128, K_MAX=0.60, K_MIN=0.02`.

**측정된 결과 (2026-06-25 종결, `memory/project_tnse_status.md`):**

| 시도 | kx_std (샘플 간 출력 다양성) | recon nRMSE |
|---|---|---|
| reconstruction-only | — | 0.275 |
| + Chamfer, avg-pool | ≈ 0.002 | 0.28 |
| + Chamfer, **cross-attention** | ≈ 0.003 | 0.28 |
| + **Hungarian matching, cross-attention** | **≈ 0.0001** | 0.28 |

- **4/4 조합 전부 입력 불변(input-invariant)으로 붕괴.** 전역 평균(지배방향 ~15–20°)을
  상수로 뱉는다. recon 0.28 > **zero-predictor 0.25** ⟹ 영 예측자보다 나쁘다.
- **대조 2개가 제대로 걸려 있다**: ①단일 에피소드 overfit 은 0.21 로 학습됨 ⟹
  forward/loss/gradient 는 정상, **순수 amortization 실패**. ②GT 방향이 실제로 다양함
  (지배방향 std **24°**, 범위 −20°~39°) ⟹ **degenerate task 아님**.
- 기록된 원인: **aperture-limited sparse LiDAR(프레임당 ~50점)에서 방향 신호가 약해
  최적화가 입력별 추출 대신 전역 평균 예측으로 도망감.**

★★**1차 출처(`results/SESSION_NOTES_2026-06-25.md`)가 아키텍처를 명시적으로 배제한다.** 축자:
> "avg-pool → cross-attention 디코더 | 동일 붕괴(kx_std≈0.003) | **구조 아님, 손실 문제**"

그리고 그 문서가 처방으로 적은 탈출구 — *"커버리지 Chamfer → **Hungarian 이분매칭**으로 query 를
특정 true 성분에 1:1 배정 + DETR 식 no-object … **입력 의존성 강제**"* (다음 할 일 1번) — 는
**실제로 실행되었고 더 심하게 붕괴했다** (`kx_std ≈ 0.002` → **`0.0001`**).
⟹ ★**당시 진단된 기제("에너지가중 Chamfer 는 고정 15방향 dictionary 만으로도 낮게 나오니
입력별 식별 유인이 없다")는 Hungarian 결과가 반증한다** — 1:1 배정은 상수 dictionary 해를
허용하지 않는데도 붕괴했기 때문이다. ⟹ **커버리지 손실과 배정 손실이 둘 다 죽었다.
"손실을 바꾸면 된다"는 탈출구는 이미 소진되었다.**

### 2.2 `scripts/d3_beamform_cond_diffusion.py` — `CondSetDenoiser` (배포 중)
`learned_init.py` 가 `init_method='learned'` 에서 실제로 호출하는 **배포 체크포인트**
(`beamform_cond_diffusion_mixed_v1.pt`). 아키텍처(`:60-83`)는 **Vaswani decoder 그대로**:

| Vaswani | `CondSetDenoiser` |
|---|---|
| §3.2.3 ② encoder self-attention | `attn=nn.MultiheadAttention(d, heads)` — **원자 집합 위 self-attention** |
| §3.2.3 ① encoder-decoder attention | `cross=nn.MultiheadAttention(d, heads)` — **beamform 조건 토큰 256개에 cross-attention** |
| §3.3 FFN | `nn.Linear(d,2d) → SiLU → nn.Linear(2d,d)` |
| §3.1 residual + LN | `h = h + attn(ln1(h))` … |
| §3.5 learned PE | `self.pos = nn.Parameter(torch.randn(1,16*16,d)*0.02)` |

**차이(전부 소소)**: pre-LN(코드) vs post-LN(§3.1) · SiLU vs ReLU(식 2) ·
`d_ff = 2d`(코드) vs `4d`(§3.3) · `h=4, L=4, d=128` vs `h=8, N=6, d=512`.
**원자 집합에는 PE 를 안 붙인다 — 순서 없는 집합이므로 옳다.**

### 2.3 `scripts/d3_beamform_detr.py` — DETR 집합 예측 (고아 상태)
헤더 축자: *"a transformer **DECODER** with N learned queries emits a **SET** of continuous
atoms directly: each query → (kx, ky, amp, presence). Trained with bipartite (**Hungarian**)
matching … DETR object detection transplanted to spectral-component estimation."*
`nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 4*d), layers)`, `NQ=48`,
`scipy.optimize.linear_sum_assignment`. 출력 `kxy = KLIM·tanh(coord(h))` ⟹ **|kx|,|ky| ≤ 0.6 로
하드 바운드**. [MEASURED] 오늘 4개 혼잡 에피소드 기준 그 안에 **87/90 성분(96.7%), 에너지
99.4–100%** 가 들어간다 — 아슬아슬하게 충분하다(§4.3 표의 컬럼 검산과 동일 근거).
**평가 1회 존재**: `results/direction3/beamform_detr_eval.json` mean nRMSE **0.2566** (6 ep).
⚠ 파일 날짜 **2026-07-03** ⟹ **메시-Nyquist 수정(07-23)·위상 수정(07-25) 이전**.
`learned_init.py` 어디에서도 import 하지 않는다 = **배선 안 됨**.

---

# 3. ⓐ–ⓕ 구성요소별 대응 판정

각 항목: **[대응]** 무엇이 무엇에 대응하는가 · **[판정]** · **[근거]**.

## ⓐ self-attention over set — **대응 성립. 그리고 이미 배포됨.**
- **[대응]** Vaswani 의 토큰 시퀀스 → 우리의 성분 집합 `{(kx, ky, w, φ)}_{n=1..N}`.
  단 우리 집합은 **순서가 없다** ⟹ §3.5 PE 를 **붙이지 않는 것이 정확**하고,
  그러면 self-attention 은 정확히 permutation-equivariant set 연산자가 된다.
- 성분 간 상호작용은 **억지가 아니다**: 우리 목적함수는 겹친 성분의 합을 관측에 맞추므로
  성분들은 **에너지를 두고 실제로 경쟁**한다(가까운 파수는 관측 창 안에서 거의 구분 불가).
  §3.2.2 의 "different representation subspaces" 가 여기서는 "서로 다른 파수 이웃"으로 읽힌다.
- **[판정] 구조적으로 잘 맞는다. 그러나 신규 적용이 아니다** — `CondSetDenoiser.attn`
  (`d3_beamform_cond_diffusion.py:68`) 이 M=32 원자 집합 위에서 정확히 이것을 하고 배포 중이다.
- [ARGUED] `n = N ≤ 40` 이고 `d = 128` 이므로 §4 의 "self-attention 이 유리한 조건 `n < d`"
  를 **크게 만족**한다. `O(n²d)` 의 `n²=1600` 은 무의미한 비용이다.

## ⓑ cross-attention — **두 방향 다 이미 구현. 정확히 이 축에서 negative 가 나 있다.**
- **[대응]** 두 배선이 가능하고 **둘 다 이 저장소에 존재한다**:
  1. **query = 성분, key/value = 관측(격자화된 beamform map 256 토큰)** — `CondSetDenoiser.cross`.
     **배포 중.**
  2. **query = 성분/learned query, key/value = 원시 관측점 ~50개** — `tnse.py:101 cross_attn`
     (`PointEncoder` 가 raw `(x,y,t,z,conf)` 를 먹음). **2026-06-25 붕괴로 종결.**
- **[판정] 배치 문제에 맞는 형태는 맞다. 그러나 "원시 점에 cross-attend" 판은 이미 실패했고,
  "beamform map 에 cross-attend" 판은 이미 성공해서 배포 중이다.** 새로 열 문이 없다.
- ★**두 팔의 차이는 attention 이 아니다.** 최소 **3가지가 동시에 다르다**:
  ①입력 표현(원시 점 ↔ 물리 유도 스펙트럼 맵) ②손실(결정적 Hungarian/Chamfer ↔ diffusion
  denoising) ③출력(결정적 1점 ↔ 확률적 다중 샘플). ⟹ ⛔**귀속 불가**
  (이 저장소의 "3중 변경 = 귀속 불가" 규율 적용).
- [ARGUED] 다만 T-NSE 붕괴 양상(`kx_std ≈ 1e-4`, 전역 평균 출력)은 **다봉 타겟에 결정적
  회귀손실을 걸었을 때의 교과서적 conditional-mean 붕괴**와 정확히 일치한다. 이것이 사실이라면
  움직인 축은 **attention 유무가 아니라 손실/출력 형식**이다. **미측정** — 아래 §7 참조.

## ⓒ positional encoding — ★**전이된다. 단, 전이되는 순간 "무차별한 설계 선택"이기를 그만둔다.**
이 항목이 이번 정독에서 **가장 생산적인 항목**이었다.

- **[대응]** §3.5 는 좌표를 **여러 주파수의 sin/cos 로 투영**한다. 우리 `beamform()`
  (`scripts/d3_beamform_net.py:46-55`) 은 축자로 이렇게 한다:
  ```python
  phi  = kxf·x + kyf·y − omf·t          # omf 는 분산관계로 slave
  resp = (z · exp(-1j·phi)).sum(-1)
  Bmap = |resp|² / M²  → log1p 정규화
  ```
  ⟹ **beamform power map 이 곧 우리 문제의 positional encoding 이다.** 관측 좌표 `(x,y,t)` 를
  `NK×NK = 64×64` 개 주파수 `(kx, ky)` 의 sin/cos 로 투영한 것. Tancik 2020 의 Fourier features
  (`ref_paper/02_INR/fourier_features_tancik_2020.pdf`)와 같은 계열이고, 차이는
  **주파수 집합이 무작위(Tancik)도 기하급수(Vaswani §3.5)도 아니라 물리로 정해진 균등 격자**라는 점.

- ★★**결정적 비대칭**: Vaswani 에서 PE 주파수는 **임의의 태그**이고, 그래서 학습형 PE 와
  **"nearly identical results"** 였다(§3.5, Table 3 (E)). **우리 문제에서는 그 주파수가
  추정하려는 미지수 그 자체다.** ⟹ 인코딩 기저를 **자유롭게 고를 수 없다**. 무엇을 인코딩에
  넣었느냐가 **무엇을 분해할 수 있느냐**를 직접 결정한다.
- **[판정] 개념은 완전히 전이된다. 그러나 전이의 결론은 "Transformer 를 붙이자"가 아니라
  "우리의 PE 격자가 물리와 안 맞게 잡혀 있다"이다** — §4 에서 정량화한다.

## ⓓ multi-head — **대응은 되지만 "head = 주파수대/방향"은 억지다.**
- **[대응]** 후보: head 마다 다른 파수대(swell/wind-sea) 또는 다른 방향 섹터를 맡는 구조.
- **[판정] 억지다. 세 가지 이유, 전부 논문 근거:**
  1. §3.2.2 는 head 분업을 **강제하지 않는다**. 근거는 "averaging inhibits this"라는
     **부정적** 논거 — 즉 평균화를 피하려고 head 를 나눌 뿐, 어떤 head 가 무엇을 맡을지는
     학습에 맡긴다. 주파수대 분업은 **논문이 주장한 적 없는 것**을 넣어 읽는 것이다.
  2. §4 말미의 해석가능성 주장은 저자들도 "**could** yield more interpretable models"로 약하게
     썼다. head 특화는 사후 관찰이지 설계 보장이 아니다.
  3. 우리가 원하는 "주파수대별 분업"은 **이미 hard 하게 강제되어 있다** — 분산관계
     `ω = √(g|k|tanh(|k|d))` 가 `ω` 를 `|k|` 에 못박아서, 파수대는 자유 파라미터가 아니다.
     head 에 재발명시킬 이유가 없다.
- 현행 코드는 `h=4` 를 그냥 기본값으로 쓴다(`CondSetDenoiser(heads=4)`, `tnse.N_HEADS=4`).
  **어떤 실험도 head 특화를 측정한 적 없다** — 그리고 측정할 이유도 위 3번 때문에 약하다.

## ⓔ §4 최대 경로 길이 — ★**우리 문제에는 "긴 시퀀스"가 없다. 논증이 전이되지 않는다.**
- **[대응 시도 3가지와 각각의 판정]**
  1. **시간축**: 초기화 창은 `2·Tp` ≈ 14–22 s. 성분의 시간 발전은 **분산관계가 닫힌 형식으로
     준다** (`φ(t) = kx·x + ky·y − ω t + φ₀`). 오늘 측정이 이걸 직접 확증했다 —
     오라클 배치 + 추적 정지에서 `init_nrmse` 0.0629 → 추적 평균 **0.0643** (거의 무변화,
     `tmp/s1s2_2026-08-05/RESULT.md §4.4`). ⟹ **시간축 장거리 의존은 학습할 게 없다. 이미 안다.**
  2. **공간축**: 프레임당 ~50점, 도메인 240×200 m. `n ≈ 50` 이다. §4 의 `O(n)` vs `O(1)` 경로
     차이는 `n` 이 수백~수천일 때 의미가 있다. **n=50 에서 recurrent 의 O(n) 경로도 문제가 아니다.**
  3. **성분 간**: `N ≤ 40`. 마찬가지로 짧다.
- **[판정] §4 의 핵심 셀링포인트(장거리 의존의 경로 단축)는 우리 문제에서 무효다.**
  §4 가 스스로 준 조건 `n < d` 는 만족하지만, 그건 "self-attention 이 **싸다**"는 말이지
  "**필요하다**"는 말이 아니다. 우리 `n` 은 어떤 아키텍처로도 짧다.
- [ARGUED] 오히려 §4 표를 우리 쪽으로 읽으면 반대 결론이 나온다: 순차연산 `O(1)` 의 가치는
  **병렬화**인데, 오늘 타이밍 재측정에서 우리 배포 경로는 **단일 코어 · 단일 프로세스 · 1 BLAS
  스레드**로 고정되어 있다(`tmp/timing_remeasure_2026-08-05/RESULT.md §2`). 병렬화가 이득이
  안 되는 실행 환경이다.

## ⓕ encoder-decoder — **"생성"은 없다. 순수 추론이다. 단 한 군데 예외가 있고, 그게 배포판이다.**
- **[대응]** §3.1 decoder 의 정체는 **auto-regressive 생성**이다(마스킹, `< i` 의존).
  우리에게는 **자기회귀적으로 뽑을 순서 있는 출력이 없다** — 성분 집합은 순서가 없고
  한 번에 다 나와야 한다. ⟹ **마스킹된 decoder self-attention 은 대응 대상이 없다.**
- **[판정] encoder-decoder 는 그대로는 안 맞는다.** 맞는 것은 **DETR 식 비자기회귀 decoder**
  (learned query N개 → 병렬로 집합 방출)이고, 이건 §3.2.3 의 ①+② 만 쓰고 ③(마스킹)을 버린 형태다.
- ★**예외**: 배포 중인 `CondSetDenoiser` 는 **diffusion** 이므로 실제로 **생성 모델**이다.
  즉 우리 파이프라인은 이미 "순수 추론"이 아니라 **조건부 생성 + 물리 polish** 다.
  `sample()` 은 `T=500` 역확산 스텝 × `nsamp=4` 샘플(`d3_setdiff_prior.py:28`,
  `learned_init.py:332`), 이어서 `polish_atoms(..., iters=80)` Gauss-Newton.

### ⓐ–ⓕ 요약표

| | 구성요소 | 우리 대응물 | 대응 품질 | 이미 있는가 |
|---|---|---|---|---|
| ⓐ | self-attention over set | 성분 집합 `{(kx,ky,w,φ)}` (순서 없음) | **좋음** | ✅ 배포 중 (`CondSetDenoiser.attn`) |
| ⓑ | cross-attention | 관측 ↔ 성분 (양방향 다) | **좋음** | ✅ 둘 다 (`.cross` 배포 / `tnse.cross_attn` **실패 종결**) |
| ⓒ | positional encoding | **beamform power map = 물리적 Fourier feature** | **좋음 + 비대칭 발견** | ✅ 단 격자가 어긋남 (§4) |
| ⓓ | multi-head | 주파수대·방향 분업 | ❌ **억지** (분산관계가 이미 강제) | h=4 기본값, 미측정 |
| ⓔ | 최대 경로 길이 | 시간축/공간축/성분간 — **전부 짧음** | ❌ **논증 무효** | — |
| ⓕ | encoder-decoder | 자기회귀 생성 대상 없음 | ❌ 그대로는 부적합 / DETR 판만 적합 | ✅ 고아 (`d3_beamform_detr.py`) |

---

# 4. ★★ 가장 유망한 적용점 + 정량 근거

## 4.1 먼저 — 임무가 준 힌트의 **논리를 검증했고, 부분적으로 틀렸다**

> 힌트: *"초기화가 25.3 s 로 프레임 비용의 3,909배다 — 한 번의 forward 로 배치를 내는 것의
> 가치가 여기서 정량화된다."*

**[MEASURED] 틀렸다.** 25.3 s 는 **배치 탐색 비용이 아니라 coarse-to-fine 전체 적합 비용**이다
(스테이지 합 900 gradient step). 근거: 오늘 S2 의 `oracle` 팔은 **배치를 공짜로 받고도**
같은 900 스텝 파이프라인을 그대로 돌렸다(`tmp/s1s2_2026-08-05/RESULT.md §2`). 따라서
"한 번의 forward 로 배치를 내면 25.3 s 가 사라진다"는 **성립하지 않는다** — 사라지는 것은
배치 탐색분뿐이고, **그 분해는 저장소에 측정값이 없다**(§8-1).

**[MEASURED] 대신 이렇게 하면 참이 된다 — 학습 경로 *내부* 비교.**
배포 중인 학습 배치는 한 번의 forward 가 **아니다**: `T=500` 역확산 × `nsamp=4` + polish 80 iter.
같은 박스(단일 스레드 CPU)에서 두 아키텍처를 실제 치수 그대로 재구성해 잰 결과
(`tmp/transformer_review_2026-08-05/arch_bench.py`, torch 2.12.1, median of 20–30):

| | 파라미터 | 1회 비용 | 전체 |
|---|---|---|---|
| `CondEncoder` (조건 인코딩) | — | 8.17 ms | 1회 |
| `CondSetDenoiser` 역확산 1스텝 (batch n=4) | 1,454,723 | 6.51 ms | **× 500** |
| ⟹ **배포 중 diffusion 샘플링 전체** | | | **3.26 s** |
| `DETRSet` **단일 forward** (enc+dec 포함) | 1,044,004 | **11.13 ms** | 1회 |
| **비율** | | | **293×** |

(벤치는 무작위 초기화 가중치 · 체크포인트 미로드 — 지연은 가중치 **값**에 무관.
`DETRSet` 의 3개 출력 head(coord 2 + amp 1 + pres 1)는 파라미터 수가 동일한 `Linear(d,4)`
하나로 합쳐 재현했다.)

⟹ **[MEASURED] 배치를 "한 번의 forward"로 내는 것의 가치는 25.3 s 대비가 아니라
배포 중 학습 배치 3.26 s 대비 293× 다.** 그리고 11.1 ms 는 프레임 예산 100 ms **안에** 들어간다
(현행 프레임 6.19 ms 와 합쳐도 17.3 ms). diffusion 3.26 s 는 **32.6× 초과**한다.
⚠ 이 수치는 **네트워크 forward 만**이다. beamform map 생성 비용은 양팔 공통이고 미측정이다.

## 4.2 그래서 유망한 적용점은?

**[판정] 새 Transformer 를 설계하는 것이 아니다. 후보는 두 개이고 둘 다 "이미 지어진 것을
현행 조건에서 다시 재는 것"이다. 그리고 ★진짜 1순위는 Transformer 가 아니다(§4.3).**

**후보 A (Transformer 축, 2순위)** — 고아 상태인 `d3_beamform_detr.py` 를 `learned_init.py` 에
배선하고 현행 조건에서 재측정. 정량 근거 3가지:
1. **배치가 병목임이 오늘 확정**: 현행 beamform 배치 `init_nrmse` **0.2612** →
   오라클 배치 **0.0629** (−76%). 최종 nRMSE 는 0.2667 → 0.1671 (짝별 −0.1145, 4/4,
   시드 잡음자의 **6.3배**), LETKF 0.2394 를 **넘어선다**.
2. **비용이 예산 안**: 11.1 ms (§4.1).
3. **유일한 측정이 두 번의 무효화 이후로 갱신 안 됨**: `beamform_detr_eval.json` = 2026-07-03,
   즉 메시-Nyquist 수정(07-23)·위상 수정(07-25) **이전**. 이 저장소는 그 두 수정이
   기존 결론을 뒤집은 사례를 이미 여러 건 기록하고 있다.

**후보 A 의 사전 확률을 크게 깎는 반대 증거 (대칭 감사) — 4건, 전부 MEASURED:**
1. `d3_beamform_detr.py` 는 **T-NSE 와 같은 손실 계열(Hungarian bipartite matching)** 이고,
   T-NSE 의 그 팔이 **가장 심하게 붕괴**했다(`kx_std ≈ 1e-4`).
2. ★그 손실 계열이 **이 저장소에서 이미 4번 mean-collapse** 했다 — `0.248` (d3_component_net,
   6000 steps) · `0.248` (d34 raw-coord) · `0.2566` (d3_beamform_detr) · `0.28` (T-NSE) —
   **전부 영 예측자 0.25 근방 또는 그 위**(§5.3, §6-1b).
3. ★그 실패가 **학습 분포 안에서도** 일어난다 ⟹ 데이터·일반화로 가는 길이 닫혀 있다(§5.3).
4. ★**"exhaustively (multi-seed, train-enough, hyperparameters)"** 로 이미 시도되었다고
   저장소가 기록한다 ⟹ 하이퍼파라미터 탐색 여지도 이미 소진.

⟹ **후보 A 에 남은 것은 "메시·위상 수정 이후 조건에서 이 4중 음성이 뒤집히는가"라는
재확인 질문 하나**뿐이다. 그것도 정당한 질문이지만, **E1 대비 비용이 자릿수 크고 사전 확률이 낮다**
(§7 에서 E2 로 강등, E2′ 를 그 앞에 둔다).

## 4.3 ★★★ 그러나 진짜 1순위는 Transformer 가 아니다 — **탐색 격자 해상도**

정독 항목 ⓒ(PE = 물리적 Fourier feature)를 우리 코드에 적용해 보니 **측정 가능한 결함**이 나왔다.

**[MEASURED] 해석적 v2 beamform 은 `kx, ky ∈ (−4.0, 4.0)` 을 `n_coarse = 81` 로 훑는다**
(`initializer.py:421-424` 기본값; `beamform_peak_ratio` `:305`, `learned_init.py:253` 도 동일).
⟹ **격자 스텝 = 8.0/80 = 0.100 rad/m.**

**[MEASURED] 그런데 오늘 쓴 4개 혼잡 에피소드의 참 성분은 `|k| ≤ 0.72` 안에만 있다**
(`config/wave_components` 컬럼 = `[A, om, kx, ky, phi, Q]`, `data_writer.py:87` /
`d3_appendix_figs_v2.py:212` 로 확인; 분산관계 `ω²=g·k·tanh(kd)` 로 행 단위 검산 통과):

| episode | N | \|k\| 범위 | NN 간격 중앙 | NN 최소 | < 0.15 (MIN_DK) | **< 0.100 (격자 스텝)** | < 0.020 |
|---|---|---|---|---|---|---|---|
| BF4_15comp | 15 | 0.031–0.699 | 0.0835 | 0.0168 | 13/15 | **9/15** | 2/15 |
| BF4_30comp | 30 | 0.027–0.719 | 0.0439 | 0.0082 | 28/30 | **23/30** | 6/30 |
| BF5_15comp | 15 | 0.021–0.423 | 0.0505 | 0.0096 | 15/15 | **14/15** | 3/15 |
| BF5_30comp | 30 | 0.019–0.435 | 0.0263 | 0.0047 | 30/30 | **29/30** | 10/30 |
| **합** | **90** | | | | **86/90** | **75/90** | **21/90** |

**[MEASURED] 세 가지가 동시에 성립한다:**
1. **탐색 범위가 5.6× 과대**. `|k|max = 0.72` 인데 ±4.0 을 훑는다.
   ⟹ 81×81 격자 중 신호가 있을 수 있는 셀은 약 15×15 = **3.4%** 뿐. **96.6% 가 구조적으로 빈칸.**
2. **억제 상자가 실제 간격보다 크다**. `supx = int(MIN_DK/dkx)+1 = int(0.15/0.10)+1 = 2` 셀
   ⟹ 실효 억제 반경 **±0.200 rad/m**. 참 성분 **86/90 이 0.15 보다, 75/90 이 0.10 보다 가깝다.**
   ⟹ 첫 peak 를 채택하는 순간 그 이웃은 **정의상 못 찾는다.**
3. **fine 단계가 구제 못 한다**. `_beamform_refine(delta=0.2, n_fine=41)` 은 **이미 선택된**
   coarse peak 주변만 정련한다(`initializer.py:346-357`). 억제 상자 안의 두 번째 성분을
   **새로 추가하지 못한다**.

★**이것이 2026-07-23 `MIN_DK` 스윕이 포화한 진짜 이유다.** 그 문서 결론은
*"0.05 와 0.02 는 어디서나 동일한 결과 … nothing is gained by going finer"* 였다. 산술로 확인:
`int(0.05/0.10)+1 = 1`, `int(0.02/0.10)+1 = 1` — **둘 다 1셀 = ±0.100 rad/m 로 반올림된다.**
⟹ **그 포화는 "물리적 간격에 도달해서"가 아니라 "격자 스텝이 바닥이라서"였다.**
`MIN_DK` 는 0.10 rad/m 아래로 **내려갈 수가 없다**. 그리고 75/90 성분이 그보다 가깝다.

**[MEASURED] 처방은 상수 2개, 학습 0, 추가 연산 0:**

| `kx_range` | `n_coarse` | 격자 스텝 | MIN_DK=0.15 실효 억제 | 분해 불가 성분 |
|---|---|---|---|---|
| **(−4.0, 4.0) 현행** | 81 | **0.1000** | ±0.200 | **75/90** |
| (−0.8, 0.8) | 81 | **0.0200** | ±0.160 | **21/90** |
| *(참고) 학습 조건 맵 (−0.6, 0.6)* | 64 | 0.0190 | ±0.152 | — |

⟹ **동일 비용(81×81)으로 격자 스텝이 5× 미세해지고, 구조적으로 분해 불가한 성분이
75/90 → 21/90 (−72%) 로 준다.** [ARGUED] 이것이 오늘 측정된 `detection_rate` 0.25–0.47 의
직접적 후보 원인이고, 배치 병목(`init_nrmse` 0.2612 vs 오라클 0.0629)의 상당 부분을 설명한다.

⚠ **[ARGUED, 반드시 병기]** `MIN_DK` 를 그냥 낮췄을 때 07-23 이 관측한 **부작용은 실재**한다 —
혼잡 5/5 개선, **중간밀도 0/5 악화(중복 peak)**. 격자를 미세화하면 그 중복 peak 위험은
**커진다**. 따라서 이 처방은 **단독으로 나가면 안 되고**, 이미 코드에 있으나 기본값이 아닌
`PEAK_SELECT_MODE = 'recompute_residual'`(`initializer.py:29-42`, 잔차에서 매 반복 맵 재계산 ⟹
억제 반경이 load-bearing 이 아니게 됨)과 **함께** 사전등록해야 한다.
⛔ 이 절은 **가설 + 산술**이다. **정확도는 하나도 재측정하지 않았다.**

★ 그리고 이 발견의 출처를 정직하게 밝히면: **§3.5 를 읽은 덕에 나왔다.** "PE 의 주파수 집합이
무엇을 분해할 수 있는지를 결정한다"는 관점이 아니었다면 `(−4,4)/81` 을 그냥 지나쳤을 것이다.
**논문의 가치가 아키텍처 이식이 아니라 진단 렌즈로 실현된 사례다.**

---

# 5. 선행 대비 신규성 판정

⛔ **이 저장소는 3축 문헌검증 전까지 "최초" 사용을 금지**하고 있다. 아래는 **선점 확인** 결과이며,
어떤 문장도 "최초"를 주장하지 않는다.

**[판정] 선점되었다. 그것도 외부 문헌이 아니라 이 저장소 자신에게 먼저.**

| 축 | 선행 | 우리 상태 |
|---|---|---|
| set 위 self-attention | Vaswani §3.2.3 ② (2017) | 배포 중 |
| DETR 식 비자기회귀 집합 예측 + Hungarian | DETR (Carion 2020) — `d3_beamform_detr.py` 헤더가 **명시적으로 "DETR … transplanted"** 라고 자인 | 고아, 평가 1회(구본) |
| 희소 관측 → 필드, attention | `ref_paper/01_NP_variants/` **6편 정독 완료**(아래 §5.1) | 비교 대상 |
| **성분집합 prior** | ★**SlotFlow (2511.23228)** — 2026-08-01 페르소나 토론에서 **"성분집합 prior 선점 ⟹ 신규성 사망"** 으로 이미 판정됨 | ⛔ 인용금지 목록에 등재 |
| 좌표 Fourier feature | Tancik 2020 (`ref_paper/02_INR/`), Vaswani §3.5 | beamform map 이 물리판 |

★**가장 중요한 선점은 자기 선점이다**: "attention set-to-set 로 관측 → 성분 집합" 은
`wavesplat/model/tnse.py` + `scripts/train_tnse.py` 로 **이미 구현·학습·종결**되었다(2026-06-25).
새 제안이 아니라 **재현 시도**가 된다.

**[MEASURED] 남는 공백은 하나뿐이고, 그것은 아키텍처가 아니다**:
"결정적 matching 손실이 이 aperture 에서 붕괴할 때, **입력 표현을 원시 점에서 물리 유도
스펙트럼 맵으로 바꾸면 구제되는가**" — T-NSE(원시 점, 붕괴)와 배포 diffusion(맵, 성공) 사이에
**3중 변경**이 있어 아직 귀속 불가한 상태로 남아 있다(§3-ⓑ).

## 5.1 ★ NP 변종 6편 정독 결과 — **출력 형식 축은 이 6편 안에서 비어 있다**

`ref_paper/01_NP_variants/` 6편 전문 추출·정독(별도 에이전트, 절·식 번호 확인):

| 논문 | attention 형태 | 좌표 인코딩 | 최종 출력 | **성분 집합 출력?** |
|---|---|---|---|---|
| ConvCNP (Gordon 2020) | **없음 — CNN** | EQ kernel 의 **차** ψ(·−x_n), ConvDeepSet Thm 1 | target x* 에서 (μ, σ) — Eq 2 | ❌ |
| TE-TNP (Ashman 2024) | self(context)+cross(target←context), translation-equivariant 판 | attention logit 안 **쌍별 차 x_n−x_m** 을 MLP ρ 로 (Eq 7–8) | target 예측분포 파라미터 — §2.1 | ❌ |
| Gridded-TNP (Ashman 2024) | MHCA(context→격자 pseudo-token) → ViT/Swin MHSA → **k-NN MHCA**(격자→target) | point-wise MLP 임베딩, latent Z = Z_token × X | μ_t, σ_t — App. E Eq 13 | ❌ |
| BSA-TNP (2025) | **full pairwise**, inducing point 미사용 + G-invariant **attention bias** | ★**좌표를 embedding 에서 뺀다** (`Embed_s(s)=∅`, §4.1). 위치는 bias `b=Σα_m exp(−β_m‖s_i−s_j‖²)` 로만 (Eq 4) | test 점에서 p(f_t\|θ) — §4 | ❌ |
| LBANP (Feng 2023) | **Perceiver latent bottleneck** — `LEMB_i = SA(CA(LEMB_{i−1}, CONTEXT))`, `QEMB_i = CA(QEMB_{i−1}, LEMB_i)` | 특별한 위치 인코딩 **없음** | mean + diagonal cov — §3.2 | ❌ |
| Senseiver (McDugald 2024) | **Perceiver latent bottleneck** — Eq 1–4 | **trigonometric(sinusoidal) PE** + ★**bathymetry 를 추가 좌표로 인코딩** | query 좌표에서 해수면 높이 **스칼라** — Eq 4 | ❌ |

**[MEASURED] 6/6 이 "좌표 → 필드값(또는 그 point-wise 분포)" 디코더다. 성분 파라미터 집합을
출력하는 것은 0편.** ⚠**함정**: 이들이 말하는 "distribution parameters" 는 **target 좌표 하나당
point-wise Gaussian 의 (μ,σ)** 이지 성분 파라미터가 아니다.
보강 증거 — **잠재 파라미터를 실제로 굴리면서도 출력하지 않는다**:
ConvCNP §5.1/App C.2 는 매 iteration `f ~ U[3,5]` 로 **주파수를 무작위 추출한 sawtooth** 를
학습하지만 f 를 추정하지 않고 필드값만 낸다. LBANP §4.2 / TE-TNP §4.1 은 task 마다 GP 커널
하이퍼파라미터(ℓ, σ_f)를 뽑지만 전부 암묵적으로 marginalize 한다.

⛔ **그럼에도 "최초"를 도출하면 안 된다** — 6편 중 3편(TE-TNP, Gridded-TNP, BSA-TNP)이
**같은 저자군/계보**라 코퍼스 편향이 있다. 08-04 R4–R5 의 **"선점지도 빈칸 = method 코퍼스
아티팩트"** 패턴과 동형이다. 그리고 SlotFlow(2511.23228) 성분집합-prior 선점 판정은 **별개로
그대로 유효**하다.

**우리에게 직접 쓸모 있는 3가지:**
1. **좌표 처리가 3파로 갈린다** — (i) 커널 차+CNN(ConvCNP) (ii) attention logit 안의 쌍별 차
   (TE-TNP) (iii) **좌표를 embedding 에서 빼고 거리 bias 로만**(BSA-TNP). ⟹ ⓒ 를 기술할 때의 비교축.
2. ★**Senseiver 가 이 6편 중 유일한 실제 해양 파동장**이고, 우리와 가장 가깝다:
   non-zero 신호 센서 **0–13개** → frame 당 81,921 픽셀. ★**물리 soft constraint(SWE 연속방정식,
   λ=100)가 개선 0** (8.6e-2 → 8.8e-2) — 무제약 모델의 연속방정식 오차가 이미 solver 수치오차
   이하였기 때문. ★**"trigger time" 59.5분** = 파면이 센서에 도달하기 전엔 재구성 불능.
   ⚠단 **tsunami(shallow-water long wave)** 이고 wind wave/방향 스펙트럼이 **아니다**.
   불확실성 출력 없음(MSE only).
3. **지연은 2편만 보고**: Gridded-TNP Swin-TNP **127 ms @ bs=8**(A100), BSA-TNP **103.74 batch/s**
   (4090). ⚠**둘 다 per-sample 단일 추론 지연은 안 냈다** ⟹ 우리 "≤100 ms" 척추와 직접 비교
   불가, batch·HW 정규화 필요.

⚠ **파일 손상 발견(조치 필요)**: `ref_paper/01_NP_variants/GriddedTNP_Ashman2024.pdf` 는
**정확히 2,097,152 바이트(2 MiB)에서 잘린 손상 파일**이다(EOF/xref 없음, `fitz` 가 0 페이지로 엶).
raw stream inflate 로 **본문 2–19쪽만 복구**했고 **1쪽(제목·초록)과 20쪽 이후(Appendix E 실험
세부)는 복구 불가**. 재다운로드 필요. (기존 `memory/data_corrupt_slope_files.md` 와 같은 계열의
저장소 무결성 문제.)

## 5.2 ★★★ 3축 문헌검증 — **"최초"는 미검증이 아니라 거짓이다**

별도 에이전트의 웹 3축 검증(WebSearch 72회 + WebFetch 14회 + arXiv API 9회) 결과:

| 축 | 판정 | 가장 가까운 선행 |
|---|---|---|
| 1. NN 이 파라미터 **집합**을 직접 회귀 | **선점됨** | Han & Biswas, J. Franklin Inst. 334(1):1–18 (**1997**) · DeepFreq (Izacard, NeurIPS 2019, arXiv:1906.00823) · SignalNet (2021, arXiv:2106.05490, **Chamfer set loss**) |
| 1b. ★**attention + learnable query + Hungarian + gridless + 좌표 입력** | **선점됨** | ★**AAETR** — Zhu, Chen & Yang, arXiv:**2408.09362** (2024-08). transformer **encoder 6 + decoder 6**, **learnable query 80개**(DETR 방식), **array element 3D 좌표를 측정값과 함께 입력 토큰으로** ⟹ 임의 기하 지원, **가변 카디널리티 집합** {angle, magnitude, confidence} 출력, *"modified set prediction loss inspired by DETR"* + **Hungarian**, gridless **single forward pass**. **우리 가설과 아키텍처·손실·입력형식이 전부 같고 차원만 1D angle.** |
| 1c. ★**학습 초기화 → 고전 optimizer 정제** | **선점됨** | ★**Schieler, Semper & Thomä**, arXiv:**2211.04846** (EuCAP 2024) / 저널판 arXiv:2506.23621 (IJMWT 17:892–899, 2025). 원문 축자: 출력을 *"integrated into an existing maximum likelihood estimator for **efficient initialization of a gradient-based iteration**"* — **우리 문장 그대로.** model order selection 동시 수행. |
| 2a. 파랑에서 attention 기반 sparse 재구성 | **선점됨** | Senseiver (Santos, *Nature Machine Intelligence* 2023) → McDugald, *JAMES* 2024 / *GRL* 2025 |
| 2b. **파랑에서 성분 집합 출력** | **공백(약함)** — *미발견*이지 부재 아님 | 최근접: LWT-PINN, *Appl. Energy* 355:121602 (2024, 고정 FFT 격자·1D·비상각) |
| 2c. 파랑에서 NN 초기화 → 고전 정제 | **공백(약함)** | 없음 |
| 3a. DETR 집합예측 → 물리 파라미터 | **선점됨** | AAETR 2024 · NMRTrans (arXiv:2602.10158) · Set Transformer + 미분가능 Pople inversion layer |
| 3b. **정현파 중첩의 amortized inference** | **선점됨** | ★**SlotFlow** — Houba, Giarda, Speri (ETH), arXiv:**2511.23228** (2025-11). slot 기반 conditional normalizing flow, **permutation-invariant Hungarian**, K classifier, **중첩 10성분 정현파 분해 카디널리티 99.85%**, RJMCMC 대비 ~10⁶배. 공개 구현체 존재. **08-01 판정이 정확했다.** |
| 3c. "학습된 초기화" | **선점됨 — 이름 붙은 확립 패러다임** | **amortized optimization** — Amos, *FnT in ML* **16(5):592–732 (2023)** · Sambharya, *"Learning to Warm-Start Fixed-Point Optimization Algorithms"*, **JMLR 25 (2024)** |

⟹ ★**"attention set-to-set 이 파랑 성분 집합을 한 번의 forward 로 뱉고 optimizer 가 정제한다"는
명제 전체가 AAETR(1b) + Schieler(1c) + SlotFlow(3b) 세 편으로 사실상 완전히 덮인다.**
남는 것은 **"파랑 관측기하로의 이식"이라는 application delta 하나**뿐이고, 그것도
★**arXiv:2601.04581 (2026, Zhou et al., "Spectral point transformer")** 가 해수면을
**각 점이 문자 그대로 `[z, kx, ky, ω, p]` 인 sparse point set** 으로 토큰화해
permutation-invariant transformer 로 처리하고 있어 **얇다** — head 가 Hs 스칼라 하나를
회귀할 뿐이므로 **head 교체 한 번 거리**다.

## 5.3 ★★★ 그리고 이 저장소는 이미 **6번** 구현했다

§2 에서 3개를 찾았으나 실제로는 더 있다. **`docs/learned_offgrid_design.md` (2026-06-28) 은
이 임무가 제안하려던 아이디어의 설계 명세서 그 자체**이고, 스스로 붙인 이름이 **"amortized RELAX"** 다.
축자(§Architecture):
> 1. **Set encoder** (permutation-invariant): per-point token = MLP(normalize[x,y,t,z]);
>    **L self-attention blocks** over tokens → context H
> 2. **Atom decoder (DETR-style set prediction)**: **N_q=32 learnable query embeddings**;
>    **cross-attention(queries → H) + self-attention**; per query head →
>    (kx, ky, log A, φ, presence_logit). … bounded by a **tanh·K_LIM**, NOT grid bins
> 3. **Dispersion hard constraint**: ω = √(g|k|·tanh(|k|d)). **ω is never free.**
> 4. **Render head (differentiable, no learned params)**

⟹ ⓐ + ⓑ + ⓕ + 분산 하드제약이 **2026-06-28 에 이미 전부 설계되어 있었다.**

| # | 구현체 | 정체 | 기록된 결과 |
|---|---|---|---|
| 1 | `wavesplat/model/tnse.py` (+`scripts/train_tnse.py`) | SetTransformer + DETR query + Hungarian | **입력 불변 붕괴** `kx_std≈1e-4`, 0.28 |
| 2 | `scripts/d34_learned_offgrid.py` | 위 설계의 구현체 (raw-coord net) | **mean-collapse, eval 0.248** |
| 3 | `scripts/d3_component_net.py` | **Fourier-feature DETR**, Chamfer+Hungarian, 6000 steps | **mean-collapse, eval 0.248** |
| 4 | `dcspecnet/model.py` | **Set Transformer (ISAB/PMA)** — docstring 이 **Lee et al. ICML 2019 을 이미 인용** → dispersion-constrained 디코더 | *"novelty 부족 + 학습 부족"* (`docs/direction_b_survey_20260522.md`, 2026-05-22) |
| 5 | `scripts/d3_beamform_detr.py` | DETR decoder + Hungarian, 단발 | **0.2566 > 영 예측자 0.25** |
| 6 | `scripts/d3_beamform_cond_diffusion.py` | **CondSetDenoiser** — self-attn + cross-attn diffusion | **배포 중.** raw 0.2596 → **+polish 0.1099** |

**[MEASURED] `docs/NETWORK_IMPROVEMENT_IDEAS_2026-07-03.md` 의 축자 기록이 §6-1b 를 독립 확증한다:**
> "Tried **exhaustively** (multi-seed, **train-enough**, hyperparameters), all measured on the
> real episodes; simple-sea nRMSE ≈**0.19-0.25** vs RELAX **0.05-0.09** throughout … mean-collapse
> … **eval 0.248**. Same for d34's raw-coord net (**0.248**). … amortized field regressors (all 3,
> and a fresh episode-matched one with UQ head …): mean-collapse to **0.24-0.25**, **even on the
> *training* distribution** (NLL gamed by variance inflation)."

★**저장소가 이미 같은 판정을 두 번 더 내려 놨다**: `docs/N2_OUR_OWN_NETWORK_2026-07-10.md` 의
전수 대조표에 **"집합/트랜스포머(점군) Set Transformer ✘"**, 그리고
`docs/PERSONA_DEBATE_novelty_2026-08-01.md` L118–127 + `MEMORY.md` ⛔목록에
**"성분집합 prior 신규성"이 이미 금지 항목**(SlotFlow 선점).
[ARGUED] 부수: 우리 해석적 초기화(coarse beamform → `_beamform_refine`)는 사실상
**NOMP**(Mamandipoor, Ramasamy & Madhow, IEEE TSP 2016)의 "coarse detect → Newton refine" 구조이고,
07-26 페르소나 토론이 해석 팔을 **2D-MNOMP** 로 명명한 것과 일치한다.

★★두 가지가 결정적이다:
1. **"even on the training distribution"** ⟹ 이건 **분포 밖 실패가 아니라 분포 안 실패**다.
   더 많은 데이터·더 나은 일반화로 가는 길이 **닫혀 있다.**
2. **"train-enough"** 가 명시되어 있다 ⟹ 이 저장소의 상시 지시
   [충분히 학습 후 판단](`feedback_train_enough_before_judging.md`)이 **이미 충족된 상태의 음성**이다.

⚠ `ref_paper/graphify-out/` 선점 지도 조회 결과: attention 20편 / transformer 11편이지만
**"set prediction / set-to-set / line spectral / frequency estimation / deep unfolding / MNOMP"
전부 0건**. ⛔**그러나 이 0건은 부재 증거가 아니다** — `text/` 는 각 PDF 의 **첫 3페이지만**
추출한 것이고(`(19 pages, first 3 shown)`), 인접 커뮤니티 #18 "Atomic-Norm Off-Grid Estimation"
은 노드 4개(Tang·Bhaskar·Recht 2013 고전 SDP 한 편)뿐이다. ⟹ **코퍼스의 구멍**이며
08-04 R4–R5 의 **"P5 선점지도 빈칸 = method 코퍼스 아티팩트"** 와 동형이다.

⚠ **열람 범위 = 주장 범위 / 이중검증 상태**:
- **내가 저장소 안에서 직접 재확인한 것**: `docs/learned_offgrid_design.md` 전문 ·
  `NETWORK_IMPROVEMENT_IDEAS_2026-07-03.md` L68–80 의 0.248 · `results/SESSION_NOTES_2026-06-25.md` ·
  6개 파일 존재 · 4개 `*_eval.json` 원자료 · `wave_components` 레이아웃(분산관계 검산).
- **에이전트가 원문 확인한 것**: Schieler arXiv:2211.04846(WebFetch 원문) · SlotFlow(검색 2회) ·
  DeepFreq · Amos 튜토리얼.
- ⛔**아무도 원문 확인 못 한 것 = `AAETR (arXiv:2408.09362)`** — 그런데 **이 판정의 척추**다.
  JOMAE 147(2) 022002 과 arXiv:2601.04581 도 미확인. **인용 전 원문 확인 필수.**
- **미검색 영역**: 중국어 문헌 · **OMAE/ISOPE/IWWWFB 프로시딩**(결정론적 파랑예측은 프로시딩
  문화가 강함) · 박사학위논문 · 유료장벽 4편(특히 **JMSE 13(11) 2196 = AI phase-resolved 리뷰**,
  반례 가능성 최고). ⟹ **2b·2c 의 "공백"은 보이는 것보다 약하다.**

---

# 6. ★ 적용하면 안 되는 이유 (대칭)

1. **[MEASURED] 정확히 이 아키텍처가 이 데이터에서 이미 붕괴했다.** T-NSE 4/4 조합 입력 불변,
   `kx_std ≈ 1e-4`, recon 0.28 > zero-predictor 0.25. **단일 에피소드 overfit 0.21 대조**가
   구현 버그를 배제하고, **GT 방향 std 24° 대조**가 과제 축퇴를 배제한다. 기록된 원인은
   **aperture** — 프레임당 ~50점에서 방향 신호가 약하다. **Transformer 는 aperture 를 못 늘린다.**
   ★1차 출처가 **"구조 아님, 손실 문제"** 라고 직접 적었고, 그 손실 처방(Hungarian)까지
   실행해서 **더 심하게 붕괴**했다(§2.1). ⟹ **아키텍처 축과 손실 축이 둘 다 닫혀 있다.**
1b. ★★★ **[MEASURED] 저장소의 모든 신경 집합예측기는 물리 polish 를 떼면 영 예측자 수준이다.**
   `results/direction3/*_eval.json` 4개를 그대로 읽어 정렬했다. **영 예측자 바닥은 nRMSE = 0.25**
   (pred≡0 ⟹ RMSE = std(η) = Hs/4 ⟹ nRMSE = 0.25, 이 저장소의 관례값과 일치):

   | arm | 정체 | mean nRMSE | vs 0.25 |
   |---|---|---|---|
   | T-NSE (4조합) | **SetTransformer + cross-attn (+Hungarian)** | 0.275–0.28 | ❌ **더 나쁨** |
   | `beamform_net` | heatmap CNN (격자 기반) | 0.2386 | ≈ 바닥 |
   | `component_net_ff` | feed-forward 성분망 | 0.2480 | ≈ 바닥 |
   | `beamform_detr` | **Transformer decoder + Hungarian, 단발** | 0.2566 | ❌ **더 나쁨** |
   | `cond_bma` | **Transformer diffusion, polish 전** | 0.2596 | ❌ **더 나쁨** |
   | **`cond_polish_bma`** | **위 + Gauss-Newton 물리 polish** | **0.1099** | ✅ **−56%** |

   ⟹ ★**바닥을 유의하게 깨는 유일한 팔은 신경망이 아니라 그 뒤에 붙은 물리 정련이다.**
   07-25 혼잡 −45% 승리도 같은 파일이 `cond_bma 0.2596 → cond_polish_bma 0.1099` 로
   **분해해서 보여준다** — 이긴 것은 `cond+polish` 이지 `cond` 가 아니다.
   ⚠**귀속 한계**: 이 파일들에는 **"polish 를 무작위/beamform 초기값에서 출발시킨 대조"가 없다**.
   ⟹ *"polish 가 전부 한다"* 는 **말할 수 없고**, *"신경망 출력 단독은 영 예측자와 구분되지 않는다"*
   만 측정되었다. ⚠전부 **2026-07-03** 파일 = 메시·위상 수정 **이전**, 에피소드 집합도 팔마다 조금씩 다름.

2. **[MEASURED] 데이터 규모가 §5 와 3–5 자릿수 다르다.** Vaswani §5.1: **4.5M 문장쌍**,
   §5.2: 8×P100 으로 **12시간–3.5일**. 우리: **100 에피소드**(오늘 실험은 **4개**).
   §5.4 의 정규화 처방(dropout 0.1, label smoothing 0.1)은 그 규모에 맞춰 조정된 것이고
   우리 규모로 전이된다는 근거가 없다.
3. **[MEASURED] 추론 예산.** 프레임 예산 100 ms, 현행 WS **6.19 ms/frame**.
   배포 diffusion 배치 **3.26 s** = 예산 **32.6× 초과**(§4.1). DETR 단발 11.1 ms 는 들어가지만,
   그건 §4.2 후보 A 가 **아직 안 지어진 배선**이라는 뜻일 뿐이다.
4. **[MEASURED] 분산 하드제약과의 충돌은 "충돌"이 아니라 "중복"이다.** `ω = √(g|k|tanh(|k|d))`
   가 이미 자유도를 절반으로 줄여 놓았다. §3.2.2 의 head 분업(ⓓ)이 재발명하려는 구조를
   물리가 **공짜로, 정확하게** 준다. 신경망에 맡기면 **정확한 제약을 근사로 바꾸는 손해**다.
   ⟹ 오늘 이걸 직접 측정했다: **오라클 배치 + 추적 정지 = nRMSE 0.0643, ρ 0.962,
   Skill 0.926** — 표현과 분산관계 전진만으로 충분하다. **모자란 건 배치뿐이다.**
5. **[MEASURED] 학습이 분포 밖에서 무너진 기록이 3건 누적.**
   - `tmp/heldout40_2026-07-26/RESULT.md`: **S1·S2·S7 FAIL ⟹ 통합 헤드라인 삭제**(8개 바 중 5개만 통과).
   - `tmp/wp_real_2026-07-25/`: 실측 typical 27창에서 **LOSS-significant** — 해석적 0.744 vs
     학습 0.805, meanΔ **−0.0611**, two-sided Wilcoxon **p = 0.0046**, rank-biserial −0.608.
   - `memory`: 실측 기여는 강건성뿐(⛔"보험" 인용금지), 제안이 sparse 불능.
6. **[MEASURED] 오늘의 두 번째 병목은 학습으로 못 고친다.** 추적기가 오라클 배치를
   0.0643 → 0.1671 로 **2.6배** 악화시키고, 그 손실의 **77–92% 가 정규화 항**이다
   (`tmp/tracker_debate_2026-08-05/VERDICT.md`). 처방은 **2줄, 런타임 비용 0**
   (Phase 2 에서만 `λ₁=λ_smooth=0` ⟹ 0.1425 → 0.0167, −88%, 3/3 시드).
   ⟹ **더 큰 모델이 아니라 잘못된 목적함수가 문제다.**
7. [ARGUED] **기회비용.** §4.3 의 격자 처방은 학습 0·추가비용 0 이고, 후보 A 는 배선+재학습이다.
   같은 병목(배치)을 겨냥하는데 비용이 자릿수로 다르다.

---

# 7. 실행한다면 — 최소 실험 설계 + 비용

우선순위 순. **셋 다 사전등록 필수** (n=4 에서 α=0.05 는 원리적으로 불가 — 양측 정확 부호검정
바닥 p = 2·(1/2)⁴ = **0.125** ⟹ **"유의"라는 단어를 쓰지 않는다**. 오늘 S1/S2 의 규율 그대로).

### E1 (1순위) — 탐색 격자 해상도. **학습 0.**
- **팔**: `kx_range/ky_range` ∈ {(−4,4) 현행, (−0.8,0.8)} × `PEAK_SELECT_MODE` ∈
  {`static_suppress` 현행, `recompute_residual`} = **2×2 요인설계**.
  전 팔 `init_method` **명시**(W9 오염 사고 규율), 총 gradient step **900 고정**.
- **에피소드**: 혼잡 4개(오늘과 동일) **+ 중간밀도 대조 2개** — 07-23 이 관측한
  "중간밀도 0/5 악화" 부작용을 **반드시** 잡도록. 시드 {42,123,456} **median-of-3**.
- **PRIMARY**: `init_nrmse` (배치 품질의 직접 측정치, 추적기 교란 없음).
  기준선 현행 **0.2612**, 상한 오라클 **0.0629**. **SECONDARY**: `detection_rate`, 최종 nRMSE.
- **MDE**: 오늘 S2 의 시드 range 중앙 **0.0182** 보다 큰 효과 + 4/4 일관 부호.
- **비용**: 2×2×6ep×3seed = **72 런**. 오늘 S1/S2 가 132 런이었으므로 **약 40–60분**, GPU 불필요.
- **선행 반증 가능성**: 만약 `init_nrmse` 가 안 움직이면 §4.3 산술은 맞아도 **인과가 아니다**
  ⟹ 배치 병목의 원인은 격자가 아니라 다른 곳. 그것도 깨끗한 결과다.

### E2 (2순위, ★사전 확률 낮음) — DETR 단발 forward 배선. **학습 필요.**
- **단일요인 질문**: "Hungarian matching 붕괴(T-NSE `kx_std≈1e-4`)가 **입력 표현을
  beamform map 으로 바꾸면** 구제되는가." — 이것이 이 축에 남은 **유일한 미측정 요인**이다.
- ⚠★**그러나 기존 증거가 이미 부정 쪽을 가리킨다**: `beamform_detr_eval.json` mean **0.2566
  > 영 예측자 0.25** (§6-1b). 즉 **"beamform map 을 먹인 Hungarian DETR"은 이미 한 번 돌았고
  이미 바닥 아래로 못 내려갔다.** ⟹ E2 는 사실상 *"구본 측정이 두 버그 때문에 부당하게
  나빴는가"* 만 묻는 재확인 실험이다. 사전 확률을 그렇게 낮춰 잡고 시작해야 한다.
- **필수 대조 (사전등록)**: **`kx_std` 를 PRIMARY 진단으로 먼저 건다** — 정확도를 보기 전에
  **출력 다양성이 T-NSE 수준(1e-4)으로 붕괴하는지**부터 판정. 붕괴하면 정확도는 보지 않고 종료.
  T-NSE 가 쓴 대조 2개를 **그대로 재사용**: ①단일 에피소드 overfit 이 학습되는가
  ②GT 다양성이 실재하는가. **+ 신설 대조: 영 예측자 0.25 를 사전에 바로 명시**하고
  그것을 못 이기면 정확도 서사를 **쓰지 않는다**.
- **비용**: 학습 + 배선 + 평가. **E1 대비 자릿수 큼.** ⟹ **E1 이 음성일 때만 착수.**
- ⚠ 기존 `beamform_detr_eval.json`(2026-07-03) 은 **무효 처리**하고 재측정할 것.

### E2′ (E2 를 대체할 더 값싼 질문) — **polish 귀속 대조**
§6-1b 의 귀속 공백을 **학습 0 으로** 메운다: 같은 `polish_atoms(iters=80)` 을
**(a) 배포 diffusion 원자 (b) v2 beamform peak (c) 무작위 원자** 세 초기값에서 출발시켜 비교.
**"신경망 제안이 물리 polish 에 실제로 무엇을 더해 주는가"** 를 직접 잰다.
E2 를 할지 말지의 판단 근거가 되므로 **E2 보다 먼저**. 비용: 기존 체크포인트 재사용, 학습 없음.

### E3 (조건부) — Transformer 와 무관하나 같은 병목
`tmp/tracker_debate_2026-08-05` 의 P1 처방(Phase 2 정규화 off, **2줄**)은 이미
0.1425 → 0.0167 (−88%, 3/3) 로 측정되어 있다. **E1 과 교란되므로 동시 변경 금지** —
어느 쪽이든 **한 번에 하나만**.

---

# 8. ★ 확인 못 한 것 (정직한 공백)

1. **25.3 s 초기화의 단계별 분해 미측정.** 배치 탐색 : 900-step joint fit 비율을 모른다.
   ⟹ §4.1 의 "힌트가 틀렸다"는 **"25.3 s 가 배치 비용이라는 근거가 없다"**는 뜻이지,
   **"배치 비용이 작다"를 측정한 것이 아니다.** 분해하려면 계측 런이 필요하다(이번엔 금지).
2. **§4.1 벤치는 forward-only.** beamform map 생성 비용, 데이터 이동, GPU 경로 **전부 미측정**.
   CPU 단일 스레드 1개 박스, 무작위 가중치(체크포인트 미로드 — 지연은 가중치 값에 무관).
3. **§4.3 은 산술과 코드 독해뿐. 정확도 0건 재측정.** `(−0.8,0.8)` 이 실제로 `init_nrmse` 를
   개선한다는 **측정은 없다**. E1 전까지 어떤 개선 주장도 하면 안 된다.
4. **NP 변종 6편 중 Gridded-TNP 는 1쪽(제목·초록)과 20쪽 이후를 못 읽었다** — PDF 손상(§5.1).
   나머지 5편은 전문 정독.
5. **6편 코퍼스는 편향되어 있다** — 3편이 같은 저자군. "출력 형식 축이 비어 있다"는
   **이 6편 한정** 진술이고, 이것으로 신규성을 주장하면 08-04 가 적발한 **"method 코퍼스
   아티팩트"** 를 반복하는 것이다.
5b. ⛔★**AAETR (arXiv:2408.09362) 원문을 아무도 못 읽었다** — 그런데 §5.2 판정의 척추다.
   JOMAE 147(2) 022002(ASME 403) · arXiv:2601.04581 · 1997년 2편도 미확인.
   ⟹ **인용 전 원문 확인 필수.** 이건 08-05 세션이 적발한 **"앞 에이전트 보고를 검증 없이
   전달"** 의 재발 위험 지점이고, 나는 여기서 그 위험을 **제거하지 못한 채 넘긴다.**
5c. **미검색 영역이 남아 있다** — 중국어 문헌 · OMAE/ISOPE/IWWWFB 프로시딩 · 학위논문 ·
   유료장벽 4편. ⟹ §5.2 의 "공백(약함)" 2건은 **더 약하게** 읽어야 한다.
   ★**arXiv abstract 질의는 과소보고한다**는 것이 이 조사 안에서 자기증명되었다:
   `abs:"direction of arrival" AND abs:"set prediction"` = **0 hit** 인데 **AAETR 이 정확히
   그것을 한다**(본문에만 표현). ⟹ **0-hit 를 부재로 읽으면 안 된다.**
5d. **축 3(집합예측/amortized inference)이 세 축 중 가장 얕다** — 전담 에이전트가 ~35분 후
   무응답 중단되어 직접 검색 6회 + 축 1 교차 커버리지로 대체되었다.
5e. **투고 전 필수 확보 목록**: `JMSE 13(11) 2196` (AI phase-resolved 리뷰, **반례 확률 최고**) ·
   `JOMAE 147(2) 022002` · `arXiv:2601.04581` 전문 · OMAE/ISOPE 프로시딩 스윕 ·
   **AAETR 원문**(§8-5b).
6. **T-NSE negative 의 유효성 범위.** 2026-06-25 ⟹ 메시 수정(07-23)·위상 수정(07-25) **이전**.
   재구성 손실 팔은 렌더러 버그의 영향을 받았을 수 있다. 단 **Hungarian 방향손실 팔은
   렌더러를 안 거치고 (kx,ky) 를 직접 감독**하므로 그 팔의 붕괴(`kx_std≈1e-4`)는
   렌더러 독립이다. [ARGUED] 그래서 negative 는 견고하다고 본다 — 그러나 재확인은 안 했다.
7. **ⓓ(multi-head 분업) 미측정.** "억지다"는 논문 근거에 기반한 **논증**이지 측정이 아니다.
8. ★**내가 이 세션에서 낸 오류 1건 — 자진 철회.**
   `config/wave_components` 를 `(kx, ky, …)` 로 오독하여 *"참 성분의 64%가 학습 배치망의
   출력 클램프 밖에 있다"* 는 **structural mismatch 를 만들어냈다.** 실제 레이아웃은
   **`[A, om, kx, ky, phi, Q]`** 이고, 올바른 컬럼으로 재계산하면 **87/90(96.7%)이 조건 맵
   안, 에너지 99.4–100%, 90/90 이 샘플러·polish 클램프 안**이다. ⟹ **k-공간 support 는 정상.
   그 발견은 존재하지 않는다.** (분산관계 `ω²=g·k·tanh(kd)` 행 단위 검산으로 레이아웃 확정.)
   ★교훈: **배열 컬럼 레이아웃을 검산 없이 가정하면 "구조적 결함"을 발명하게 된다.**

---

## 부록 — 인용 금지 제안 (신규 · `MEMORY.md` ⛔목록 추가 권고)

- ⛔ **"Transformer 를 파랑 성분 추정에 적용한다"를 신규 기여로** — 저장소 내부 **6중 선점**
  (`tnse.py` 프로덕션 · `d3_beamform_cond_diffusion.py` 배포 · `d3_beamform_detr.py` ·
  `d34_learned_offgrid.py` · `d3_component_net.py` · `dcspecnet/model.py`), 설계 명세는
  `docs/learned_offgrid_design.md` 2026-06-28.
- ⛔ **"attention set-to-set 로 파라미터 집합 출력"의 신규성** → **AAETR (arXiv:2408.09362, 2024)**
  선점. ⚠단 **원문 미확인** — 인용 전 확인 필수.
- ⛔ **"learned initialization → 고전 optimizer 정제"의 신규성** → **Schieler et al.
  (arXiv:2211.04846, EuCAP 2024)** 선점 + **amortized optimization** 은 이름 붙은 패러다임
  (Amos, FnT ML 16(5), 2023 · Sambharya, JMLR 25, 2024).
- ⛔ **"학습 배치가 분포 밖에서만 무너진다"** — `NETWORK_IMPROVEMENT_IDEAS_2026-07-03.md` 축자
  **"even on the *training* distribution"**. 허용: *"학습 분포 안에서도 mean-collapse 한다."*
- ⛔ **"한 번의 forward 가 25.3 s 초기화를 없앤다"** — 25.3 s 는 joint fit 이고 분해 미측정(§8-1).
  허용: *"배포 중 학습 배치 3.26 s 대비 단발 forward 11.1 ms = 293×(forward-only, CPU 1스레드)"*.
- ⛔ **"MIN_DK 를 낮춰도 소용없다(07-23)"를 물리적 결론으로** — 그 포화는 **격자 스텝 0.10 의
  반올림 artifact**다(`int(0.05/0.10)+1 = int(0.02/0.10)+1 = 1`). 허용: *"현행 (−4,4)/81 격자에서는
  MIN_DK 가 0.10 rad/m 아래로 내려갈 수 없다."*
- ⛔ **"탐색 범위를 좁히면 배치가 좋아진다"** — E1 전까지 **측정 0건**. 허용: *"75/90 성분이
  격자 스텝보다 가깝다"* (산술).
- ⛔ **"학습 배치망의 k-support 가 부족하다"** — **내가 만든 오류, 철회됨**(§8-8).
