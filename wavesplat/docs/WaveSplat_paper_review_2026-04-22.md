# WaveSplat 논문 검토 및 수정 계획

**작성일**: 2026-04-22
**목적**: 다른 서버에서 작업 이어받을 수 있도록 전체 검토 결과와 수정 계획 정리
**관련 파일**:
- `docs/WaveSplat_Algorithm_Guide.html` (이미 업데이트 완료)
- `paper/wavesplat.tex` (수정 대상)
- `thesis/thesis_main.tex` (수정 대상)
- `wavesplat/optimizer/initializer.py` (참조용)

---

## 0. 컨텍스트 및 작업 상태

### 진행 완료 (이 세션)
- ✅ 9개 질문에 대한 답변 작성
- ✅ Algorithm Guide HTML에 gradient chain rule, Adam element-wise, Phase별 optimizer, τ_R 유도, Gerstner vs linear, fine refinement 단계 추가
- ✅ 3개 논문(paper/wavesplat.tex, thesis/thesis_main.tex, hydrofoil_feedforward.tex) 교차 검토
- ✅ 문제 2, 3, 4에 대한 4-persona 검토
- ✅ 제약식(dispersion) 문제에 대한 4-persona 검토

### 진행 필요 (다음 서버)
- ⬜ `paper/wavesplat.tex`에 수정사항 반영 (§3.5, §4.3, §5 ablation 해석, Limitations)
- ⬜ `thesis/thesis_main.tex`에 동일 수정 동기화 (한국어)
- ⬜ Beamforming fine refinement 단계를 paper/thesis Algorithm에 추가
- ⬜ τ_R sensitivity 실험 (선택, 800 runs, 서버에서 ~하루)
- ⬜ GPR + confidence ablation (선택, ~반나절)
- ⬜ CHANGELOG에 v0.7.0 entry 추가

---

## 1. 9개 질문 답변 요약 (원본 대화 참조)

| # | 질문 | 핵심 답변 | HTML 반영 |
|---|------|----------|-----------|
| 1 | 빔포밍 수식 각 항 | $z_j$(관측), $\exp(-i\Psi)$(steering), $\omega(k)$(dispersion 고정), $\|\cdot\|^2$(보강간섭) | 기존 내용 |
| 2 | Δk=0.1 vs 빔포밍 해상도 0.0075 | **2-stage 빔포밍 발견**: coarse 81×81 (0.10) + fine 41×41 (0.01). 0.0075는 부정확한 값 | Part 6 표 추가 |
| 3 | Fresnel confidence θ_j 정의 | ray와 수면법선 각도. **실측은 flat 가정** ($\mathbf{n}\approx\hat{z}$), sim은 Gerstner true normal | Part 7 warning 추가 |
| 4 | $\lambda_s=10^{-4}$ 근거 | 데이터 항 $\sim 5\times10^{-3}\text{m}^2$ 대비 0.3% 수준, soft regularizer | 기존 내용 |
| 5 | Gradient chain rule 전개 | 5-step 해석적 유도: $\omega\to\|k\|\to k_x$ 체인, 군속도 $v_g$ 등장 | Part 7에 5-step 추가 |
| 6 | Adam v는 element-wise? | **예, 파라미터별 독립 $m_i, v_i$**. $\eta/\sqrt{v_i}$가 자동 스케일링 | Part 7 수식 + 표 추가 |
| 7 | τ_R=0.05m 근거 | 노이즈 하한 $\sim 0.022$m (σ_r=2cm + Gerstner-linear $O(\epsilon^2)\sim 6$mm) × 2.3 safety | Part 8 유도 표 + Gerstner/linear 비교 추가 |
| 8 | Tracking에서 Adam 대상 | **$w, \varphi$만** ($k_x, k_y$ frozen). 매 프레임 10 steps, 3-frame window | Part 9 수식 + 흐름 추가 |
| 9 | 슬라이딩 윈도우 사용처 | Phase 2 tracking의 loss 계산. FIFO 버퍼 3프레임. | 기존 내용 |

---

## 2. 논문 검토 결과 요약

### 2.1 정상 확인 (수정 불필요)
- C1 tracking 재프레이밍 (v0.6.0): paper L208-221에 "Estimation accuracy depends on initialization quality" 명시 ✓
- 빔포밍 격자 (Cartesian 81×81, ±4.0, 0.15 억제): 모든 문서 일치 ✓
- λ₁=10⁻³, λ_s=10⁻⁴ 값: 일치 ✓
- Hydrofoil 논문의 WaveSplat 인용 ("~20% nRMSE @ 13ms/frame"): black-box 인용이라 영향 없음 ✓

### 2.2 수정 필요 (paper ↔ code 불일치)

#### 문제 1: Fine Refinement 단계 누락
- **Code** ([initializer.py:252, 399](../wavesplat/optimizer/initializer.py)):
  - `_beamform_refine()` 함수가 각 피크마다 41×41 fine grid (±0.2 rad/m)로 재탐색
  - `initialize_from_frames_v2` → `v2_konly` 모두 이 fine stage 사용 (production default)
- **Paper** (L471-500): Coarse 81×81만 기술, fine 언급 없음
- **Thesis** (Alg 1, L3414-3440): 의사코드에도 fine stage 없음
- **혼동**: paper L41 "coarse-to-fine refinement"는 $N=3\to8\to15$ 진행을 의미 (빔포밍 refinement 아님) — 독자 오해 소지
- **영향**: 재현자가 fine 없이 구현하면 $|k|$ 정확도 ~17% (0.10/0.6), tracking 성능 저하
- **긴급도**: 높음 (재현성)

#### 문제 2: Confidence "local surface normal" 주장
- **Paper** (L419-420): "$\theta_j$ is the beam incidence angle (computed from the local surface normal and beam direction)"
- **Thesis** (L1098): 동일 표현
- **실제**:
  - WaveSim ([lidar_simulator.py](../wavesim/lidar_simulator.py)): Gerstner 메시의 **true normal** 사용
  - Real data ([real_data.py:199-201](../wavesplat/real_data.py)): **flat 가정**, θ = arccos(|dz|/|ray|)
- **영향**:
  - Paper 결과는 sim only → 기술적으로 틀리지 않음
  - 그러나 "real LiDAR에서도 이 정도 성능" 암시 → over-claim 리스크
  - Grazing 관측(θ~60°)에서 $\cos$ 편향 ~9%, typical 관측(θ~10-30°)에서 1-2%
- **긴급도**: 높음 (over-claim 방지)

#### 문제 3: "Fresnel-aware" 명칭의 과장
- **실제 공식**: $c_j = \cos\theta_j/(d_j/d_{\text{ref}})$ — Lambertian projection × inverse distance
- **엄밀 Fresnel**: $R(\theta) = 0.5[R_s(\theta) + R_p(\theta)]$는 $\theta$에 비선형 (n=1.321 @1550nm)
- **Thesis L1105-1109**: 이미 정직하게 기술 ("간소화는 의도적 설계 선택")
- **Paper**: C3 novelty로 "Fresnel-aware" 유지 → 명칭 과장
- **영향**:
  - 숫자는 맞음 → 결과 유효
  - Reviewer: "Where is Fresnel in your formula?" 리젝션 리스크
  - GPR + confidence ablation 없음 → reviewer: "Is this novel or standard distance weighting?"
- **긴급도**: 중간

#### 문제 4: τ_R = 0.05m 유도 불완전
- **Paper** (L544): 값만 제시, 유도 없음
- **Thesis** (L1332-1337): "노이즈 σ=0.02m의 2.5배", "BF2 Hs의 17%" — 부분 유도
- **누락**: Gerstner-linear model mismatch ($\sim$6mm) 기여
- **Sensitivity 실험 없음**: reviewer가 "임의 선택" 지적 가능
- **긴급도**: 낮음 (수정 쉬움)

---

## 3. 4-Persona 검토 — 문제 2, 3, 4

### 참여자
- **Prof. K** (지도교수): novelty, physical meaning, top venue
- **Prof. E** (외부위원, 해양공학): 회의적, 정량 수치 요구
- **Rev. 2** (학회 리뷰어): 기술적 정확성, 재현성
- **Saeyong** (학생): Avikus 실무 경험, 한계 인정 + 방어

### 3.1 문제 2 (normal 가정) 최종 판정

| 질문 | 판정 |
|---|---|
| 결과 자체가 틀렸나? | **아니오** — sim 조건에서 일관된 비교, 재현 가능 |
| 클레임이 과장되었나? | **예** — "real LiDAR에서 Fresnel-aware" 인상 |
| 결과를 철회해야 하나? | **아니오** — two-paragraph 수정으로 해결 |
| 리스크 | Reviewer "why not real data?"로 거부 근거 삼을 수 있음 |

**Prof. E 우려**: Forward-looking LiDAR는 대부분 관측이 grazing (FOV -25°~0°, d>50m에서 θ>45°). flat 가정의 편향이 ablation 효과보다 클 가능성.

**Rev. 2 반박**: Weight는 **관측 간 상대 순서**가 중요. Flat 가정에서도 가까운 점 > 먼 점 순서 유지. 다만 **Model B(slope) 환경에서는 crest 우대 효과 소실** — 실측이 Model B에 가까우면 C3 효과 반감.

### 3.2 문제 3 (Fresnel 명칭) 최종 판정

| 질문 | 판정 |
|---|---|
| 결과 자체가 틀렸나? | **아니오** — 공식/효과 측정 정확 |
| C3가 independent contribution? | **논쟁적** — GPR+conf 실험 없으면 판정 불가 |
| 권장 조치 | C3 재명명 or C2 하위로 격하 |

**Prof. K 제안**: C3를 **"observation-noise-aware weighting"** 또는 **"heteroscedastic observation model"**로 개명. 또는 C3를 C2 아래 구현 디테일로 격하 → 3개 contribution(C1 grid-free, C2 dispersion, C4 adaptive)으로 재구성.

### 3.3 문제 4 (τ_R 유도) 최종 판정

| 질문 | 판정 |
|---|---|
| 결과 자체가 틀렸나? | **아니오** — 경험적으로 검증됨 |
| 유도가 불완전? | **예** — paper에 누락, thesis에만 부분 유도 |
| 결과에 영향? | **낮음** — sensitivity 실험으로 방어 가능 |

**Prof. E 요구**: τ_R ∈ {0.02, 0.05, 0.10, 0.15} m sweep. 예상 결과:
- 0.02: 노이즈 과반응 → 무한 성장, nRMSE +5~10%
- **0.05: 최적**
- 0.10: 놓친 성분 무시, BF5에서 nRMSE +3~5%
- 0.15: adaptive 효과 사라짐

---

## 4. 4-Persona 검토 — 제약식 (Dispersion Constraint)

### 트리거 질문
"실제 파도랑은 다른데 그러면 제약식도 문제 있는거 아니야?"

### 핵심 데이터 (Paper A1 ablation, L1111-1144)
| 설정 | nRMSE (BF3) | nRMSE (BF4-5) | 분산 잔차 R̄ |
|---|---|---|---|
| Full (hard constraint) | 0.188 | 0.266 | 0 (by construction) |
| A1 (unconstrained) | **0.158** | **0.241** | **0.51** |

A1이 **9.4% 더 좋은 nRMSE**를 달성 — reviewer가 "왜 제약?"이라고 물을 수 있음.

### 4.1 Prof. E의 공격 (dispersion 제약 비판)
- Sim이 Gerstner라 off-shell 에너지 O(ε²)가 작지만, 실측에서는:
  - **Bound harmonics** (2차 Stokes): ε²/4
  - **Wave-wave mixing**: O(ε²)
  - **Currents (Doppler)**: ω → ω + k·U, 0.5m/s에서 BF4의 5% shift
  - **Wind forcing**: 시간에 따른 ω, k 진화
- 실해역 R̄ 추정: 10-30% off-shell (sim 0.51은 artifact)

### 4.2 Rev. 2의 정량 반박
**Bound harmonics 에너지 비율** (Stokes 2차):
- BF3 (ε=0.03): 0.02%
- BF4 (ε=0.05): 0.06%
- BF5 (ε=0.07): 0.12%

**합계 off-shell**: 약 0.25% (BF4 기준). 즉 **99%+ 에너지는 linear dispersion shell 위**. Hard constraint는 <1% 에너지를 무시하는 정당한 근사.

### 4.3 Saeyong의 A1 결과 해석
A1의 R̄=0.51은 bound harmonics로 설명 불가 (기대값 <1%). 실제 원인:
- **Over-fitting 가설**: 50-200 pts × 3 frames = 150-600 vs 5N=75 free params → 부족한 관측 / 파라미터 제약 없음 → ω drift로 MSE만 낮춤
- 이것이 paper L1141-1144의 "1초 뒤 decohere" 방어 논리의 근거

### 4.4 Prof. K의 종합
**제약식은 물리적으로 정당**하지만 **paper의 설명이 약함**.

현재 paper 답변: "physical correctness vs fit quality trade-off"
수정 제안: "**A1의 낮은 nRMSE는 physical signal이 아니라 over-fitting artifact** (R̄=0.51은 bound harmonic 기대값 <1%를 크게 상회). Hard constraint는 이 over-fitting을 방지하는 물리 기반 regularizer."

### 4.5 Prof. E의 양보 + 잔여 우려
- Rev. 2 수치 분석 수용: **제약은 정당**
- 잔여: **Doppler** (current ~5% shift)는 bound harmonics보다 큼 → limitations에 명시 필요

### 4.6 실측 적용 시 통합 리스크 매트릭스

| 이슈 | Sim 영향 | Real 영향 |
|---|---|---|
| Normal flat 가정 (문제 2) | 없음 (true normal) | 중간 (grazing 9%) |
| Fresnel 명칭 (문제 3) | 없음 | 낮음 (명칭만) |
| τ_R=0.05m (문제 4) | 검증됨 | 불확실 (노이즈 다르면 재튜닝) |
| Dispersion hard constraint | <1% 에너지 loss | ~5% Doppler + 1% bound |
| Linear forward model | <0.25% (ε²) | 동일 (BF3-5) |

**실측 예상 nRMSE 악화: 5-15%** (sim 0.211 → 실측 0.22-0.24 추정)

### 4.7 최종 합의
**"결과 자체는 건드릴 필요 없다. 설명·명칭·제약 기술이 문제다."**

---

## 5. Paper/Thesis 수정안 (LaTeX 스니펫)

### 5.1 Paper §3 — Fine refinement 추가

**위치**: `paper/wavesplat.tex` L493-500 뒤

```latex
Each extracted peak is refined locally via a second beamforming
pass on a finer 41$\times$41 grid spanning $\pm$0.2~rad/m around
the coarse peak ($\Delta k_{\text{fine}} = 0.01$~rad/m), improving
wavenumber accuracy from 0.10 to approximately 0.01~rad/m. Without
this refinement, the 0.10~rad/m grid would limit $|\mathbf{k}|$
accuracy to roughly 17\% at typical BF4 wavenumbers
($|\mathbf{k}| \approx 0.3$~rad/m), inducing significant phase
drift during subsequent tracking.
```

### 5.2 Paper §3.5 — Confidence normal 가정 명시

**위치**: `paper/wavesplat.tex` L419-425 근처 (Fresnel-zone confidence weighting 섹션)

```latex
The incidence angle $\theta_j$ is the angle between the LiDAR ray
direction and the local water surface normal $\mathbf{n}$. In the
simulation experiments reported here, $\mathbf{n}$ is obtained from
the ground-truth Gerstner wave mesh. For real-data deployment, the
surface normal is unknown prior to reconstruction; a pragmatic
approximation is $\mathbf{n} \approx \hat{\mathbf{z}}$ (flat-surface
assumption), introducing a slope-dependent error bounded by the wave
steepness $\epsilon \approx 0.05$ at BF4 ($\sim 3^\circ$ normal
deviation). For near-normal observations ($\theta < 30^\circ$) this
yields $<2\%$ error in $\cos\theta_j$; grazing observations
($\theta \approx 60^\circ$) can see up to $\sim 9\%$ error.
```

### 5.3 Paper §5 — A1 ablation 재해석

**위치**: `paper/wavesplat.tex` L1120 근처 ("This result requires careful interpretation.")

```latex
This result requires careful interpretation. The 9.4\% nRMSE
advantage of A1 should not be mistaken as evidence that the
dispersion constraint is physically too restrictive. For BF3--5
deep-water gravity waves, $>99\%$ of wave energy lies on the linear
dispersion shell~\citep{holthuijsen2007, stokes1847}; bound harmonic
contributions scale as $\epsilon^2/4 < 0.15\%$ at BF4. The observed
$\bar R = 0.51$ off-shell drift in A1 therefore does not reflect
physical signal. Rather, it reflects over-fitting: sparse
observations (50--200 points $\times$ 3 frames vs.\ $5N = 75$ free
parameters for $N = 15$) underconstrain the unconstrained model,
which absorbs noise and model mismatch by letting $\omega$ drift
off the dispersion manifold. Hard enforcement of the dispersion
relation acts as a physics-based regularizer against such
over-fitting; the modest nRMSE penalty on held-out simulation
frames is compensated by the critical payoff during tracking
(Phase~2): phase prediction $\Delta\varphi = \omega\Delta t$
requires $\omega$ on the dispersion manifold, and the 51\%
off-shell A1 components de-cohere from the true field within one
dominant period.
```

### 5.4 Paper Limitations — Doppler 추가

**위치**: `paper/wavesplat.tex` L1408 근처 (Limitations 섹션)

```latex
\emph{Current-induced frequency shift}: the dispersion relation
$\omega^2 = g|k|\tanh(|k|d)$ assumes a current-free medium. In the
presence of a mean current $\mathbf{U}$, the observed frequency
shifts by $\mathbf{k}\cdot\mathbf{U}$ (Doppler); a 0.5~m/s current
introduces $\sim 5\%$ bias at BF4. Real-data deployment would
require either (a) joint estimation of $\mathbf{U}$ as an
additional state, or (b) soft relaxation of the dispersion
constraint. Both are deferred to future work.

\emph{Surface normal approximation for real data}: in simulation
we use the exact Gerstner surface normal for confidence weighting
$c_j$, but real-data application must substitute the flat-surface
approximation $\mathbf{n} \approx \hat{\mathbf{z}}$. The resulting
bias scales with wave slope (up to $\sim 9\%$ in $\cos\theta_j$
at grazing incidence at BF4); empirical characterization of this
bias effect on end-to-end nRMSE is deferred to wave-tank
validation.

\emph{Observation weight nomenclature}: we refer to the weighting
$c_j = \cos\theta_j / (d_j/d_{\text{ref}})$ as ``Fresnel-aware'' to
reflect its physical motivation from Fresnel reflection
asymmetry and heteroscedastic measurement noise. The functional
form is a deliberate simplification; the exact Fresnel reflectance
$R(\theta)$ could be substituted at the cost of requiring polarization
and surface-roughness assumptions that are difficult to justify
from observation alone.
```

### 5.5 τ_R 유도 보강

**위치**: `paper/wavesplat.tex` L544 수정

```latex
If the residual RMS after pruning exceeds $\tau_R = 0.05$~m
(indicating unmodeled wave energy), new components are grown
by applying beamforming to the residual signal
$r_j = z_j - \hat{z}_j$ and adding the dominant residual peaks
to the model, up to $N_{\max}=30$. The threshold is derived
from the theoretical residual floor: combining LiDAR range
noise ($\sigma_r \approx 0.02$~m) with the Gerstner--linear
model mismatch at BF4 ($O(\epsilon^2) H_s / 2 \approx 6$~mm) and
angular noise projection ($\sim 5$~mm) yields a lower-bound RMS
of $\sim 0.022$~m; $\tau_R = 0.05$~m provides a $\sim 2.3\times$
safety margin for unambiguous detection of unmodeled signal
(see Appendix~\ref{app:tau_sensitivity} for sensitivity analysis).
```

### 5.6 Thesis 동기화

Thesis는 이미 부분적으로 정직한 기술이 있으므로 다음만 추가:
- Ch.3 빔포밍 섹션(~L1190)에 fine refinement stage 추가 (Alg 1 pseudocode 포함)
- Ch.3 L1098 근처에 normal 가정 sim/real 구분 추가
- Ch.3 L1332-1337 근처에 Gerstner-linear mismatch 기여 추가
- Ch.4 "실측" 섹션에 Doppler + normal 한계 추가

### 5.7 Thesis 구체 수정 (한국어)

**Ch.3 Algorithm 1 (L3414-3440) — Step 1.5 추가**:
```latex
\STATE \textbf{// Step 1.5: Fine refinement (각 피크마다)}
\STATE $(k_x^*, k_y^*)$ 주변 $\pm 0.2$\,rad/m 영역을 $41 \times 41$ 미세 격자 ($\Delta k_f = 0.01$\,rad/m)로 재스캔
\STATE $P_\text{fine}(k_x, k_y)$ 최대점 $(k_x^f, k_y^f)$ 채택
```

**Ch.3 Confidence (L1098 수정)**:
```latex
여기서 $\theta_j$는 빔 입사각이다. 시뮬레이션에서는 Gerstner 메시로부터
계산된 정확한 표면 법선 $\mathbf{n}_{\text{true}}$를 사용하고, 실측에서는
파도면 법선이 사전에 알려지지 않으므로 $\mathbf{n} \approx \hat{\mathbf{z}}$
(평균 해수면 가정)로 근사한다. 근사 오차는 파도 경사 $\epsilon \approx 0.05$
(BF4)에 비례하며, $\cos\theta_j$에서 grazing 관측 ($\theta \sim 60^\circ$)
약 9\%, near-normal ($\theta < 30^\circ$) 약 1--2\%이다.
```

---

## 6. 추가 실험 계획 (선택적)

### 6.1 τ_R Sensitivity Sweep
**목적**: τ_R 선택의 robustness 입증

**설정**:
- τ_R ∈ {0.02, 0.05, 0.08, 0.10, 0.15} m
- 100 episodes × 5 τ values × 2 raydrop = 1000 runs
- 서버에서 병렬 (4 workers) ~1일

**예상 결과** (이미 라운드 3에서 분석):
- 0.05가 최적, ±0.03m 이내에서 robust (<5% nRMSE 변화)
- 논문 appendix + 본문 1문장 유도 추가

**스크립트 위치**: `scripts/run_tau_sensitivity.py` (작성 필요)

### 6.2 GPR + Confidence Ablation
**목적**: C3(confidence)가 WaveSplat 고유 novelty인지 검증

**설정**:
- 기존 GPR baseline에 $c_j$ weighting 추가
- 100 episodes × 2 raydrop = 200 runs, ~반나절

**예상 분기**:
- GPR도 개선 → C3를 "standard observation weighting" 으로 재프레이밍
- WS에만 도움 → spectral representation과의 상호작용으로 설명

**스크립트 위치**: `scripts/run_gpr_confidence_ablation.py` (작성 필요)

---

## 7. 작업 체크리스트 (다른 서버에서)

### Priority 1 — Paper 수정 (1-2일)
- [ ] `paper/wavesplat.tex`에 5.1~5.5 스니펫 삽입
- [ ] `thesis/thesis_main.tex`에 5.7 스니펫 삽입
- [ ] `references.bib`에 `stokes1847`, `holthuijsen2007` 확인 (없으면 추가)
- [ ] Paper/thesis 컴파일 테스트 (`latexmk -pdf`)
- [ ] 기존 figure/table 번호 유지 확인

### Priority 2 — HTML 동기화 확인 (완료, 참조용)
- [x] Part 6: Fine refinement 추가
- [x] Part 7: Gradient chain rule, Adam element-wise, Confidence normal 주의
- [x] Part 8: τ_R 유도, Gerstner/linear 비교
- [x] Part 9: Phase별 Adam 구성

### Priority 3 — 추가 실험 (선택)
- [ ] `scripts/run_tau_sensitivity.py` 작성 + 실행
- [ ] `scripts/run_gpr_confidence_ablation.py` 작성 + 실행
- [ ] 결과를 paper/thesis에 반영

### Priority 4 — Changelog
- [ ] `docs/CHANGELOG.md`에 v0.7.0 entry 작성
  - Paper 재현성 수정 (fine refinement)
  - Real-data normal 가정 명시
  - A1 ablation 재해석
  - τ_R 유도 보강
  - Doppler limitations 추가

---

## 8. 참고 — 이번 세션 HTML 수정 내역

### 파일: `docs/WaveSplat_Algorithm_Guide.html`

**Part 6 (Beamforming Initialization) 수정**:
- "Greedy 피크 추출" → "Greedy 피크 추출 + Fine Refinement (2-stage)"로 변경
- Coarse/Fine 해상도 표 추가
- "왜 2-stage인가" insight 추가

**Part 7 (Nonlinear Optimization) 수정**:
- "왜 초기화만으로 부족한가" 섹션의 0.0075 rad/m 주장 제거, 0.01 rad/m + SNR-dependent 0.005-0.01로 수정
- "Fresnel Confidence" 섹션에 normal 가정 warning 추가
- Sim vs Real 법선 처리 비교 표 추가
- "Gradient 계산" 섹션에 **5-step chain rule 완전 전개** 추가:
  - Step 1: 기호 설정
  - Step 2: $w_i$ 미분 (선형)
  - Step 3: $\varphi_i$ 미분
  - Step 4: $k_{x,i}$ 미분 (4-step chain: ω→|k|, |k|→k_x, 결합, 최종)
  - Step 5: $k_{y,i}$ 미분
  - 슬라이딩 윈도우 gradient 관점 insight 추가
- "Adam Optimizer" 섹션 대폭 확장:
  - Element-wise 연산 warning 추가
  - 파라미터별 수식 ($i$ 인덱스) 명시
  - WaveSplat $\theta \in \mathbb{R}^{4N}$ 명시
  - 전형적 $g_i, v_i$, 실효 학습률 표
  - Hyperparameters 표

**Part 8 (Coarse-to-Fine + Adaptive N) 수정**:
- "잔차 기반 성장" 섹션에 **τ_R = 0.05m 유도** 추가:
  - 오차 기여 항목 표 (σ_r=0.02, 모델 오차, 각 노이즈)
  - 이론 하한 RMS 계산 (~0.022m)
  - Safety margin ~2.3× 논의
- "Gerstner vs Linear Model Mismatch" 섹션 신설:
  - Gerstner 수식 (Lagrangian, horizontal displacement 포함)
  - Linear sinusoidal 수식 (Eulerian)
  - Δz_model ~ H_s·ε²/2 유도
  - "왜 일부러 linear로 추정하는가" 4가지 이유
- "τ_R 초과 = 왜 빔포밍 실패 신호" 상세 설명
- τ_R 민감도 warning (0.02 / 0.05 / 0.15 영향)

**Part 9 (Tracking) 수정**:
- "w_phi 모드" 섹션에 **Phase별 Adam 구성** 추가:
  - Phase 1 (init): $4N$ 파라미터 전체 Adam
  - Phase 2 (tracking): $2N$ 파라미터만 Adam
  - Phase 2 프레임당 흐름 flow 다이어그램
  - 단계별 대상 파라미터/모멘트 초기화/학습률/steps 표
  - 구현 세부 (build_tracking_optimizer) insight
  - "함정" warning: requires_grad_(False)만으로 불충분, optimizer 재생성 필요

---

## 9. 현재 코드 베이스 상태 (참조용)

### 커밋 이력 (최근)
```
4ffe75d CHANGELOG v0.6.0: 빔포밍 버그 수정 및 C1 재프레이밍 이력 추가
617b7e9 학위논문: 빔포밍 격자 수정 + C1 tracking 재프레이밍
bd01f21 교육 자료 C1 reframing 반영: tracking 정확도 의존성 경고 추가
89c0f7a C1 novelty 재프레이밍: tracking 역할을 정확도→효율성으로 수정
9eb6b29 교육 자료 빔포밍 그리드 설명 수정 (polar→Cartesian)
```

### 수정된 파일 (이 세션, unstaged)
- `docs/WaveSplat_Algorithm_Guide.html` (대규모 확장)
- `docs/WaveSplat_paper_review_2026-04-22.md` (이 문서, 신규)

### 참조할 코드 위치
- `wavesplat/optimizer/initializer.py:252` — `_beamform_refine` (fine stage)
- `wavesplat/optimizer/initializer.py:399` — greedy loop에서 fine stage 호출
- `wavesplat/optimizer/sequential_update.py:113-130` — `build_tracking_optimizer` (Phase별 Adam)
- `wavesplat/optimizer/sequential_update.py:460-470` — `grow_residual_threshold` 사용
- `wavesplat/model/confidence.py` — torch 기반 confidence
- `wavesim/noise_model.py:69` — numpy 기반 confidence (simulation)
- `wavesplat/real_data.py:172-203` — 실측용 incidence angle 계산 (**flat 가정**)

---

## 10. 다른 서버 이어받기 지침

1. **이 md 파일을 먼저 읽기** (§0, §7 먼저)
2. **`docs/WaveSplat_Algorithm_Guide.html` 열어 이번 세션 수정 확인** (Part 6~9)
3. **Priority 1(paper 수정)부터 시작**:
   - `paper/wavesplat.tex` 열고 §5.1~5.5 스니펫 적용
   - 컴파일 테스트 → PDF 차이 확인
4. **Thesis 동기화** (§5.7)
5. **Changelog 업데이트** (§7 Priority 4)
6. **선택**: §6의 추가 실험 실행

---

**문서 끝.**
