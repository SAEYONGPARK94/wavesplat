# Barthelme & Utschick 선점 주장 원문 검증

> **한 줄 판정: 부분 선점(PARTIALLY PREEMPTED). 단 내가 받은 논문 제목은 실재하지 않는다.**
>
> ① **제목 "A Machine Learning Approach to Threshold Mitigation in Statistical Signal Processing" 는 존재하지 않는 논문이다.**
> Barthelme 의 DBLP 전체 기록(11편, 2014–2021)에 "threshold" 를 제목에 가진 논문은 **0편**.
> 실재 앵커는 **"A Machine Learning Approach to DoA Estimation and Model Order Selection for
> Antenna Arrays With Subarray Sampling," IEEE TSP, vol. 69, pp. 3075–3087, 2021**.
> ★**단 원 출처(`tmp/aaetr_review_2026-08-05/FINDING.md` L139–140)는 제목·DOI 를 정확히 적고 있었다** ⟹
> 오염은 **원 에이전트가 아니라 나에게 오는 중계 과정**에서 발생. **선점 주장 자체는 옳은 논문에 붙어 있었다.**
>
> ② **선점되는 것**: "학습 초기화 → 모델 기반 반복 정련" 축은 **완전 선점**, 그것도 동일한 *동기문*·동일한 *실증 결론*·동일한 *복잡도 논증*까지 함께.
> ③ **선점되지 않는 것**: 이 논문은 **set-proposal 이 아니다** — 식 (13)의 순열불변 손실을 **명시적으로 폐기**하고 정렬-고정순서 회귀로 간다.
> ⟹ ★**"learned set-proposal → optimizer" 를 한 덩어리로 선점당했다는 서술은 부정확**하다. 두 개의 서로 다른 계보로 **쪼개야** 한다.

**표기 규율**(sibling FINDING 과 동일): `[M]` = 원문에서 읽은 것(절·식 번호 표기), `[A]` = 내 추론.

**원자료 provenance**: `barthelme_utschick_tsp2021_2009.12858.pdf`, md5 `ca20cfebc9d5ad7f0140b99b62d55506`,
13쪽, arXiv:2009.12858, PyMuPDF 추출본 `paper_text.txt`.

---

## 1. 식별 (identification)

### 1.1 주장에 붙은 제목은 실재하지 않는다 `[M]`

| 항목 | 내용 |
|---|---|
| 주장된 제목 | "A Machine Learning Approach to **Threshold Mitigation** in Statistical Signal Processing" |
| 검색 결과 | 정확구절 웹검색 **일치 0건**. DBLP `pid/152/9890`(Andreas Barthelme) **전 11편 중 "threshold" 제목 0편** |
| 판정 | ★**존재하지 않는 논문** |

`[A]` 오염의 그럴듯한 씨앗: 같은 저자의 **WSA 2020, "An MSE Approximation for Grid-Based Maximum
Likelihood Direction-of-Arrival Estimators"** (Barthelme, Kunz, Utschick). grid-based MLE 의 MSE 근사는
전형적으로 **threshold region**(SNR 문턱 아래 outlier 지배 영역)을 다룬다. 다만 이건 **추측이고, 나는 그 논문을 읽지 않았다.**

### 1.2 선점 주장이 실제로 서 있어야 할 논문 `[M]`

> **A. Barthelme and W. Utschick, "A Machine Learning Approach to DoA Estimation and Model Order
> Selection for Antenna Arrays With Subarray Sampling," *IEEE Transactions on Signal Processing*,
> vol. 69, pp. 3075–3087, 2021. DOI `10.1109/TSP.2021.3081047`. preprint **arXiv:2009.12858**.**

`[M]` 이것이 옳은 앵커인 근거는 **우리 레포 안에 이미 있다**: `tmp/aaetr_review_2026-08-05/FINDING.md` L139–140 이
Schieler 의 참고문헌 [11] 을 **정확한 제목과 DOI 로** 기록하고 있고, Schieler 는 §I 에서 이를 자기 선행연구로
인용하며 자기 기여를 *"improvement in terms of parameter dimension, quantity, and data complexity"* 라고
**증분으로 자기규정**한다.

### 1.3 혼동하기 쉬운 형제 논문들 — 어느 것도 앵커가 아니다

| # | 논문 | 신경망 출력 | 후단 | 앵커인가 |
|---|---|---|---|---|
| **TSP 2021** (앵커) | DoA Estimation and Model Order Selection … Subarray Sampling | **DoA 스칼라 L개** | **SML 우도 블록좌표 경사상승** | ★**예** |
| SPL 2021 | DoA Estimation Using NN-Based **Covariance Matrix Reconstruction**, vol. 28, pp. 783–787, DOI `10.1109/LSP.2021.3072564` | **공분산 행렬** | MUSIC | 아니오 `[M, 초록 수준만]` |
| ICASSP 2020 / arXiv 1910.09284 | Model Order Selection in DoA Scenarios via Cross-entropy … (Barthelme, **Wiesmayr**, Utschick) | 모델차수 분류 | — | 아니오 |
| WSA 2020 | An MSE Approximation for Grid-Based ML DoA Estimators (Barthelme, **Kunz**, Utschick) | — (해석적) | — | 아니오 |

★**SPL 2021 은 "신경망이 공분산을 뱉는다"는 축**이라 우리와 **오히려 더 멀다.** 과제가 물은 (a)축의
"a covariance?" 는 **이 형제 논문의 성질이지 앵커의 성질이 아니다.**

---

## 2. 앵커 논문이 실제로 하는 일 `[M]` (전문 통독)

- **§II 시스템 모델**, 식 (1): 안테나 M개, RF 체인 W<M, 스위칭 네트워크로 K개 부배열을 **순차 샘플링**.
  협대역 원거리장, 조향행렬 `A(θ)`. 초점은 **L ≥ W** (소스 수가 동시표본 안테나 수 이상) 영역.
- **§III-B MCENet(순수 데이터 기반)**:
  - **입력** = K개 부배열 **표본공분산행렬**의 실수 파라미터를 쌓은 벡터, 차원 `KW²`. **스냅샷 수 N 과 무관.**
  - **구조** = 완전연결 feedforward, `Nh=4`, `Nu=2048`, ReLU (Table II).
  - **출력** = **L개의 실수** = DoA 추정치 `θ̂`. ⟹ 출력층 폭이 곧 **모델 차수. L 은 망마다 고정.**
  - **손실** = mean cyclic error, 식 (12).
  - ★**순열 처리, 식 (13)**: 순열행렬 Π 에 대한 최소화를 정의해놓고, *"if the network is fed with sorted
    labels, it converges to a point where the optimal permutation matrix Π is constant … we can even
    **omit the minimization over all permutations**"* ⟹ **순열불변 집합손실을 명시적으로 폐기**하고
    **정렬 고정순서 회귀**로 간다. (원문 "permutation" 4회, **"Hungarian" 0회**.)
- **§III-C 하이브리드**, 식 (14)–(19) + **Algorithm 1**:
  - `SSR / GLS / MCENet` 중 하나로 **초기 θ̂[0]** → **SML 로그우도에 대한 블록좌표 경사상승**(θ, R_s, σ²_η 를
    번갈아 1스텝씩, 각자 step size). 정지조건 `while L(c[t]) > L(c[t-1])`.
  - **중간 단계 필수**: 망은 θ̂ 만 주고 nuisance (R_s, σ²_η) 를 안 주므로, **고정 θ̂ 에서 GLS 문제 (9)를 풀어** 조달.
  - ★**동기문 축자**: *"by utilizing a purely data-based method in a scenario, where an appropriate
    stochastic model exists, **we ignore a significant amount of information about the problem at hand.
    Hence, we propose the combination of NN based initialization and model aware gradient steps** on the
    SML likelihood function to improve the DoA estimate."*
  - ★**결론 축자 (§V)**: *"a **combination of NN based initialization and model-based gradient steps was
    crucial** to achieve competitive DoA estimates"*.
  - ★**off-grid 축자 (§III-C)**: *"The NN method … **does not suffer from this grid mismatch problem due to
    its formulation as a regression problem**"*, 그리고 모델기반 초기화에 대해서는 *"the consecutive gradient
    steps **alleviate the grid mismatch problem** that is inherent to any grid-based approach"*.
- **§III-D 실증**: MCENet 단독은 non-outlier 영역에서 모델기반에 **못 미친다**; 하이브리드가 되어야 Genie ML 에
  근접. *"The MCENet alone cannot provide estimates that can compete with a model-based approach for
  non-outlier realizations."* ⟹ ★**학습은 outlier robustness 를, 모델기반 정련은 정확도를 산다** (trade-off 를 원문이 명시).
- **§IV 모델 차수 = 별도 망 CovNet**: `Nh=3`, `Nu=1024`, 출력 `Lmax+1` 뉴런 + **softmax**, cross-entropy 식 (23),
  사후확률 추정 ⟹ MAP 근사. `Lmax=3`.
  ★**§IV 축자**: *"With an inaccurate estimate of the model order, we base our algorithms on the wrong
  stochastic model or **choose the wrong NN, which has been trained on mismatched data**."*
  ⟹ **L 로 인덱싱된 신경망 뱅크**. 차수는 분류기가 고르고, DoA 망 자신은 차수를 못 바꾼다.
- **훈련**: MCENet 128×10⁶ 표본 지도학습, CovNet 별도 cross-entropy. **Algorithm 1 은 추론 시점 절차이고,
  경사상승 단을 통과하는 결합 목적함수는 논문 어디에도 없다** ⟹ **모듈식**. (원문 "end-to-end" 1회는
  *선행연구를* 그렇게 부르는 대목.)
- **평가 세팅**: UCA `M=9`, `R/λ=1`, `K=4` 부배열 × `W=3`, `L=3`, `N=10`, Monte Carlo. Table III 계산시간
  MCENet 5.4 s vs SSR 127.2 s vs GLS 1486.2 s (1000 realization).
- **부재 확인** (`grep -a`, 이진오판 교정 후): **"dispersion" 0회 · "forecast" 0회 · "measured/real data" 0회**.
  §III-D/§IV-C 통독으로도 **전부 합성 Monte Carlo, 실측 데이터 0건**.

---

## 3. 나란히 놓기 (side-by-side)

| 축 | **Barthelme & Utschick, TSP 2021** `[M]` | **WaveSplat** `[M]` (코드 확인) | 판정 |
|---|---|---|---|
| **(a) 망의 출력** | **L개 실수 스칼라**(DoA 각도). 출력층 폭 = L 고정. **정렬된 고정순서 벡터** — 순열불변 손실을 식 (13)에서 **명시 폐기**. 격자 아님, 공분산 아님 | **가변 길이 원자 집합** (kx, ky, energy). `learned_init.py:121` `propose_atoms` → nsamp=4 posterior sample → `pool_and_select:165` 에너지 문턱 + dedup + `max_n=30` 캡 | **다름** — 우리는 집합, 저들은 **고정순서 벡터** |
| **(b) 성분 개수** | **망 자체는 고정**(L = 출력층 폭). 별도 분류기 **CovNet**(softmax over `Lmax+1`, `Lmax=3`)이 추정하고 **L별 신경망 뱅크에서 골라 쓴다** | **추정기 내부에서 추론 시점에 변한다** — `run.py:55–57` `prune_threshold=0.10` / `grow_residual_threshold=0.05` / `max_n=30`, 동일 모델을 가지치기·성장 | **다름**(구조적). ⚠단 adaptive-N 은 현재 **NULL**(p=0.677) ⟹ **이득으로 주장 금지** |
| **(c) 2단계 정체 / 물리제약 여부** | **SML 로그우도 블록좌표 경사상승**(식 14–19, Alg. 1). **물리제약 있음** — 조향행렬 `a(θ)` 가 배열 물리. 단 nuisance (R_s, σ²_η) 를 **GLS 로 따로 조달**해야 초기화가 완성됨 | **Adam 으로 가중 MSE 최소화**(`sequential_update.py:232`), 파라미터 (w, kx, ky, φ), ω 는 자유변수가 아님. 관측 잔차 직접 적합(우도 아님), nuisance 조달 단계 **없음** | **유사**(둘 다 1차 경사, 모델기반 목적함수) / **세부 다름** |
| **(d) 훈련 end-to-end vs 모듈식** | **모듈식** — MCENet·CovNet 각각 지도학습, 경사상승 단을 통과하는 결합 목적 **없음** | **모듈식** — e2e fine-tune 은 **FAIL(null, medΔ −0.0007, p=0.30)** 로 종결, "modular = product" 확정 | ★**같음** |
| **(e) 도메인** | **배열신호처리 / 방향탐지**. 합성 Monte Carlo **전부**, 실측 0건 | **지구물리 / 해양파**. 선박탑재 LiDAR, 실측 스테레오 13레코드 + CDIP 25,426 해상상태 대조 | **다름** |
| **(f) 분산관계 하드제약의 대응물** | ★**없다. 원문 "dispersion" 0회.** 협대역 가정 ⟹ **주파수가 하나(반송파)뿐**이라 ω–k 결합이라는 것이 **존재할 수 없다**. 미지수는 각도 θ 뿐 | `spectral_gaussian.py:97–101` → `dispersion.py:33` `ω = √(g·|k|·tanh(|k|d))`. **페널티 아니라 하드제약** — ω 는 파라미터가 아니고 (kx,ky) 의 함수(`use_dispersion=False` 인 A1 절제에서만 `omega_free` 등장) | ★**대응물 없음** |
| **(g) off-grid / 연속 파라미터** | ★**예, 그리고 명시적으로 그렇게 논증한다** — 회귀 정식화라 격자 불일치가 없고, 후속 경사스텝이 격자기반 초기화의 grid mismatch 를 해소(§III-C) | 예 — 전 파라미터 연속 | ★**같음** |
| **(h) 센서 기하** | **고정 UCA**, 기하 기지(旣知), M=9. 시변성은 **스위칭 네트워크의 부배열 전환**뿐 — 플랫폼 운동 없음, synthetic aperture 없음 | **이동하는 희소 센서** — 선박 운동이 개구를 나른다 | **다름** |
| **(+) 시간축·예보** | ★**없음.** 스냅샷은 공분산 추정용 i.i.d. 실현. **"forecast" 0회**, 상태 전파 없음 | **리드타임 예보가 척추**. 위상 φ 를 절대시각으로 추정하고 전방 전파 | **다름** |
| **(+) 위상** | **없음** — 공분산 기반 2차 통계, 위상 주변화. SSR 은 아예 *"non-coherent processing, i.e., phase offsets between the subarrays are not estimated"* | **결맞음(coherent)** — 성분별 진폭 w 와 **절대위상 φ** 를 추정 | **다름** |

---

## 4. 판정 — **부분 선점 (PARTIALLY PREEMPTED)**

### 4.1 죽는 주장 (완전 선점) `[M]`

1. ★**"학습 초기화 → 모델기반/물리인지 반복 정련" 이라는 결합 구조 자체.**
   개념만이 아니라 **동기문·실증결론·복잡도논증이 셋 다 겹친다**:
   - 동기: *"appropriate stochastic model 이 있는데 순수 데이터 기반만 쓰면 정보를 버린다"* — 우리가 쓰는 논거와 동일.
   - 실증: *"NN alone cannot compete … the combination was **crucial**"* — 우리 Gauss-Newton polish 서사와 동일.
   - 복잡도: NN 추론이 모델기반보다 10–200배 싸다는 Table III 논증 — 우리 지연 논증과 동일 형태.
   ⟹ **"우리가 처음"은 이 축에서 미검증이 아니라 거짓.** 그리고 Schieler(2022/24)보다 **더 이르다.**
2. ★**"학습단은 robustness 를, 모델기반 정련단은 정확도를 산다"는 역할분담 서사** — §III-D 가 명시적으로 그 trade-off 를
   말하고 cost function 설계로 설명까지 한다. 우리의 "학습 init 이 혼잡 레짐에서 돕는다" 프레이밍과 **같은 구조**.
3. **신경망 기반 모델차수 선택을 사후확률 분류기로 놓는 것**, 그리고 **차수 추정이 틀리면 잘못된 망을 고르게 된다**는 지적.
4. **회귀 정식화로 off-grid 연속 파라미터를 내고, 후속 경사스텝으로 격자 불일치를 없앤다**는 논증.

### 4.2 살아남는 주장 (이 논문에 대해서만) `[M]` 근거 + `[A]` 범위판단

| # | 살아남는 것 | 근거 | 주의 |
|---|---|---|---|
| S1 | ★**분산관계 하드제약** `ω=√(g|k|tanh(|k|d))` 에 **대응물이 없다** | 원문 "dispersion" **0회**; 협대역 단일 반송파라 ω–k 결합이 **원리적으로 존재 불가** | `[A]` 이건 *이 논문 대비* 신규일 뿐, **해양파 모델링에서는 표준** ⟹ ⛔"novel" 로 승격 금지 |
| S2 | ★**결맞음 위상 + 시간축 + 예보** | 저들은 공분산 2차통계(위상 주변화), SSR 은 *"non-coherent"* 명시; "forecast" 0회 | 견고함 |
| S3 | ★**추정기 내부의 가변 cardinality**(뱅크 없이, 물리모델 잔차/에너지 기준) | 저들은 `Lmax=3` softmax + **L별 신경망 뱅크** | ⚠**구조적 차이일 뿐**. adaptive-N 은 **NULL(p=0.677)** ⟹ ⛔"기여"로 쓰지 말 것(이미 Limitations 로 강등됨) |
| S4 | **이동 희소 센서 / 운동이 나르는 개구** | 저들은 고정 UCA, 기하 기지 | 견고함 |
| S5 | **실측 데이터** | 저들은 **합성 Monte Carlo 전부, 실측 0건** | 견고함 |
| S6 | ★**set-proposal 아키텍처는 이 논문이 선점하지 않는다** | 식 (13)에서 순열불변 손실을 **명시 폐기**, "Hungarian" 0회 | ⚠**다른 계보(AAETR/DETR)에는 선점당함** — sibling FINDING 참조 |

### 4.3 앞선 판정이 어긋난 지점 — **뒤집힘 아니고 범위 정정 2건** `[A]`

1. ★**"learned *set-proposal* → optimizer 가 한 겹 더 깊이 선점됐다"는 한 덩어리 서술은 부정확하다.**
   이 논문은 **set predictor 가 아니다**. 정확한 분해는:
   - **"학습 초기화 → 모델기반 정련"** ⟹ **Barthelme & Utschick 2021** (최선행)
   - **"순열불변 집합예측(attention + learnable query + Hungarian)"** ⟹ **AAETR/DETR 계보** (별도 출처)
   ⟹ **하나의 선점이 아니라 서로 독립인 두 개의 선점.** 한 문장으로 묶으면 각각의 범위를 과대·과소 진술하게 된다.
2. **"IEEE TSP 2021" 이라는 venue·연도는 맞았고, 제목만 오염됐다.** 원 출처는 정확했으므로
   ⟹ ★**교훈은 "원 에이전트가 틀렸다"가 아니라 "중계 단계에서 서지정보가 조용히 변조됐다"** 이다.
   이 프로젝트의 누적 실패양상 **(a) 에이전트 보고 무검증 전달** 의 **서지 판본**.

### 4.4 열람 범위 = 주장 범위 (하지 못한 것)

- 형제 논문 **SPL 2021 은 초록·서지 수준만** 확인(전문 미열람). "공분산을 뱉는다 → MUSIC" 이상은 주장하지 않는다.
- **WSA 2020**(threshold 씨앗 추정)은 **읽지 않았다**. §1.1 의 오염 기원 설명은 **`[A]` 추측**이다.
- Barthelme 외 저자의 광범위한 hybrid-init 문헌은 **훑지 않았다** ⟹ ⛔**"Barthelme 가 최선행이다"** 라고 단정 금지.
  말할 수 있는 것은 **"Schieler 가 자기 선행으로 지목한 것이 Barthelme 2021 이고, 그것이 우리가 확인한 범위에서 가장 이르다"** 까지.
- IEEE Xplore 최종본 대신 **arXiv 판본**을 읽었다. 절·식 번호는 arXiv 판 기준.

---

## 5. 우리 코퍼스 점검 (READ ONLY — 아무것도 수정하지 않음)

| 점검 | 결과 |
|---|---|
| `ref_paper/` 에 있는가 | ★**없다.** 전체 18파일 확인, Barthelme/Utschick **0건** |
| `paper/wavesplat.tex` 가 인용하는가 | ★**아니다.** `paper/references.bib`(39 entry) 에도 **없다** |
| 레포 어디엔가 있는가 | `tmp/aaetr_review_2026-08-05/{FINDING.md, schieler.txt}` **두 파일뿐** — 즉 **다른 에이전트의 작업디렉터리에만** 존재 |
| 자연스러운 인용 자리 | `paper/wavesplat.tex` **L406–415** — MUSIC/ESPRIT/Capon 을 논하고 곧바로 *"WaveSplat leverages beamforming for \emph{initialization} and refines the estimates by gradient optimization with physics constraints"* 라고 쓰는 대목. ★**우리 2단계 서사를 진술하는 바로 그 문장이, 그 서사의 최선행을 인용하지 않고 있다** |

⚠ **경로 함정 기록**: 논문은 `/home/vexit/wavesplat/paper/` 가 **아니라**
`/home/vexit/wavesplat/wavesplat/paper/` 에 있다. 전자로 grep 하면 **조용히 0건**이 나온다(내가 한 번 물렸다).

⛔ **`paper/` 이하는 상시 지시(2026-08-04)에 따라 편집 금지.** 아래 §6 은 **제안 목록이지 수행한 변경이 아니다.**

---

## 6. 문구를 바꿔야 하는 우리 주장 목록

> 전부 **제안**이다. 논문 편집 금지 지시가 풀린 뒤 적용할 것.

| # | 위치 | 현재 | 바꿔야 할 이유 / 방향 |
|---|---|---|---|
| 1 | 서지 인용 전반 | "Barthelme & Utschick, *A Machine Learning Approach to **Threshold Mitigation** in Statistical Signal Processing*, TSP 2021" | ★**존재하지 않는 논문.** 정확히: *"A Machine Learning Approach to **DoA Estimation and Model Order Selection for Antenna Arrays With Subarray Sampling**," IEEE TSP, **69**, 3075–3087, 2021, DOI `10.1109/TSP.2021.3081047`* |
| 2 | novelty 판정문 | "**learned set-proposal → optimizer polish** 가 한 겹 더 깊이 선점" | ★**두 개로 분해.** Barthelme = "학습 초기화 → 모델기반 정련"(set 아님). set 예측은 AAETR/DETR 계보. 한 문장으로 묶지 말 것 |
| 3 | 2단계 파이프라인 서술 | "학습 초기화 + 물리 optimizer 정련" 을 **구조적 기여**로 제시 | ★**기여에서 강등.** 2021년에 동일 구조 + 동일 동기문 + 동일 실증결론 존재. 살아남는 것은 **"어떤 물리를 정련단에 넣었는가"(분산 하드제약·결맞음 위상·시간축)** 뿐 |
| 4 | "학습이 혼잡/어려운 레짐의 robustness 를 산다" 서사 | 우리 발견처럼 서술 | §III-D 가 동일 trade-off 를 명시 ⟹ **"우리 도메인에서 재현됨"** 으로 격하 |
| 5 | `paper/wavesplat.tex` L406–415 | beamforming init + gradient refinement 를 인용 없이 진술 | ★**선행연구 인용 추가 필요**(Barthelme 2021). 현재 **가장 눈에 띄는 인용 공백** |
| 6 | adaptive-N 관련 서술 | — | S3(가변 cardinality)은 **이 논문 대비 구조적 차이로는 유효**하지만 **NULL(p=0.677)** 이므로 ⛔이득 주장 금지. 기존 Limitations 강등과 **모순 없음** |
| 7 | off-grid / 격자없음 프레이밍 | "격자 불일치를 피한다"를 차별점처럼 | §III-C 가 회귀 정식화 + 후속 경사스텝으로 **같은 논증**을 한다 ⟹ 차별점 아님 |
| 8 | 모델차수 선택 서술 | — | "신경망 기반 차수 선택" 자체가 선점(CovNet). 우리 차별점은 **분류기 없이 물리잔차로 추론시점에 바꾼다**는 *기제*뿐 |

### ⛔ 신규 인용금지 (제안)

- **"A Machine Learning Approach to Threshold Mitigation …" 을 서지로 인용하는 것** — 실재하지 않음.
- **"학습 초기화 → 물리 정련 결합이 우리 기여"** — 2021년 선점(동기문까지 동일).
- **"Barthelme 가 이 축의 최선행이다"** — 우리 열람범위 안에서만 참. 광범위 hybrid-init 문헌 미조사.
- **"Barthelme 가 set-proposal 을 선점했다"** — **거짓**. 식 (13)에서 순열불변 손실을 명시 폐기했다.
