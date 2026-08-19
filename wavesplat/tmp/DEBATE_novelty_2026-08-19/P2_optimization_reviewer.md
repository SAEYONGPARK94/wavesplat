# P2 — 최적화·신호처리 심사위원 재판정 (런 0회, 2026-08-19)

**질문**: *"진폭·위상을 해석적으로 소거하고 파수에서만 하강하는 것(VARPRO)을 격자없는 다성분 파랑
재구성에 적용한 것"* 이 최적화/신호처리 분야에서 발표 가능한 기여인가?

**규율**: 모든 문헌 주장은 WebSearch/WebFetch 로 직접 검증했고 미확인은 `UNVERIFIED` 로 적었다.
`MEASURED`(저장소 실측 재확인) / `ARGUED`(내 논증) 를 분리했다. **런 0회.**
**열람 범위** = W4/W5 `results.jsonl` 원자료 재집계 · W4/W5 `PREREG`·`RESULT`·`AUDIT_*` ·
`varpro.py`·`run_w4.py`·`run_w5.py`·`arms_config_w5.py`·`recon/W2_upper_bound.md`·`recon/C2_varpro.md` ·
S3 `RESULT.md` · 웹 검색 9건 + WebFetch 7건. **그 밖은 안 봤다.**

---

## 0. 판정 요약 (맨 위)

| | 판정 |
|---|---|
| **Q1 — 방법론 기여로 성립하는가** | ⛔**NO (IEEE TSP / ICASSP / SPARS)** · ✅**조건부 YES (JOE / Ocean Eng / EUSIPCO 응용트랙)** — 논문 **한 절**이지 **한 편**이 아니다 |
| **Q5 — "당연하다" 논증이 있는가** | ★★★**있다. 그리고 내가 처음 브리핑을 받았을 때 예상한 것보다 더 강하다.** 최우선 산출. |

### Q5 의 두 문장

**(1) 문헌 쪽** — 우리가 보고하는 세 수치(더 정확 −25.3% · 더 빠름 2.05× · 스텝 720<820)는
**Golub–Pereyra 2003 종설의 초록에 이미 문장으로 있다**:

> *"...results in a **better-conditioned problem**. **The same optimization method applied to the original
> and reduced problems will always converge faster for the latter.**"*
> — Golub & Pereyra, *Inverse Problems* **19** (2003) R1, 초록 (WebFetch 축자, `VERIFIED`)

★그리고 **우리 비교는 그 문장의 전제를 실제로 만족한다.** `varpro.py:209-211` 이 축소 문제에
**Adam** 을 쓰고 배포 팔도 **Adam** 을 쓴다 ⟹ *"the same optimization method applied to the original
and reduced problems"*. **나는 처음에 "우리는 최적화기도 바꿨으니 그 전제를 안 만족한다"고 방어선을
세웠는데, 코드를 읽고 그 방어선을 철회한다.** (내 정정 1)

**(2) 우리 자료 쪽** — ★★★**"기제: decorrelation 이 절반이 됐다" 는 독립 측정이 아니라
헤드라인의 재표현이다.** 아래 §3.1 에서 산술로 보인다. **이것이 이 문서의 가장 날카로운 산출이다.**

---

## 1. Q1 — "알려진 방법 + 새 문제 + 기제 측정" 이 기여가 되는 조건

### 1.1 성문 규칙 (검증됨)

`VERIFIED` — IEEE SPS **Guidelines for Associate Editors** (WebFetch). 즉시기각 사유:

> *"limited novelty (e.g., a straightforward combination of theories and algorithms that are well
> established and are repeated on a known scenario **with no new experimental data nor new application**)"*

★**이 문언은 연언이다** — "확립된 것 + 알려진 시나리오 + **새 자료도 새 응용도 없음**".
새 응용이 있으면 즉시기각 절에서 빠진다. ⟹ **우리는 즉시기각은 아니다.**

`VERIFIED` — **Guidelines for Reviewers**: 채택 요건 = **NOVELTY** + **APPROPRIATENESS** 둘.
⚠`UNVERIFIED` — 검색 스니펫에 도는 *"trivial application of an existing technique in a different context
may be rejected"* 는 **두 공식 페이지 어디에서도 축자 확인 실패**(양쪽 직접 열람). **인용 금지.**

### 1.2 성립 조건 — `ARGUED`

| 조건 | 내용 | 우리 |
|---|---|---|
| **C1 알고리즘 개조** | 문제 구조 때문에 A 를 고쳐야 했고 그 개조가 A 로 되먹여진다 | ⛔**미충족** |
| **C2 이론 확장** | A 의 보장이 새 문제에서 다른 형태로 성립함을 증명 | ⛔**미충족**(증명 0건) |
| **C3 예측 반증** | A 가 예측과 다르게 행동하고 원인을 규명 | ⚠**부분** — W4 `P-2`(§5.4) 하나 |
| **C4 레짐 상전이** | 이득이 문제 파라미터(분리도)의 함수로 측정되고 기존 이론 문턱과 대조 | ⛔**미측정** ⟹ §4 실험 B |

### 1.3 판정

- ⛔**IEEE TSP / ICASSP / SPARS: 불성립.** C1–C4 중 **0.5개**. TSP 문턱은 신호처리 자체의 진보다.
- ✅**응용 벤치: 조건부 성립.** 존재증명 3건(전부 `VERIFIED` 서지):
  1. **LiDAR 파형 분해 VARPRO** — Wang et al., *Complexity* 2020, doi:10.1155/2020/6726139.
     ★**"LiDAR + 성분 합 + 진폭(선형)/위치·폭(비선형) 분리"** = 표면적으로 우리와 가장 가까운 선례.
  2. **대기 원격탐사 일반화 VARPRO** — arXiv:2401.02301 / *Mathematics* 11:2839 (DLR).
  3. **Sparse VARPRO in Robotic Perception** — arXiv:2512.07969 (2025-12, Papalia·Yang·Everett 등).
  ⟹ **"VARPRO 를 X 에 적용했다" 는 2020–2025 에 이미 여러 도메인에서 소진된 논문 틀이다.**

★**정직한 문장**: *"이 결과는 파랑 재구성 논문의 **방법 절 + 절제 실험**으로는 강하고,
신호처리 방법론 논문의 **주 기여**로는 약하다."*

---

## 2. Q2 — "이득이 전부 초기화에서 나고 추적은 오히려 까먹는다" 는 새로운가?

### 2.1 ⛔**먼저: 그 문장의 앞 절반은 발견이 아니라 개입 설계의 항등식이다** (내 정정 2)

`MEASURED` — `varpro.py:364-397` 은 **`SequentialUpdater._initialize_learned` 하나만**
몽키패치한다(원본 호출 후 VARPRO 를 그 출력에 적용하는 wrapper). `wavesplat/**` 은 무수정.
⟹ ★**개입이 초기화에만 존재하므로 이득이 초기화에서 나는 것 말고는 물리적으로 불가능하다.**
"이득이 전부 초기화에서 난다" 는 **측정 결과가 아니라 패치 지점의 재진술**이다.

⚠**추가로**: W4 `RESULT.md` 에는 초기화-대-추적 문장이 **아예 없다**(`추적`/`tracking` 0회, `Phase-1` 만 1회).
그 진술은 `recon/C2_varpro.md:244-247` 에 있고, 거기서는 **정반대로** 적혀 있다 —
> ⛔*"추적기와의 상호작용은 **미측정** … 온라인에서 이득이 부분적으로 씻길 수 있다."*

### 2.2 뒤 절반("추적이 까먹는다")도 규약 의존적이다 (내 정정 3)

`MEASURED` — 내가 두 `results.jsonl` 을 직접 재집계한 **raw median**:

| 팔 | `init_nrmse` | `avg_nRMSE` | 추적 후 변화 |
|---|---|---|---|
| `L-SHIP` | 0.1494 | 0.1576 | **+5.5%** |
| `L-VP-ONLY` | 0.1055 | 0.1105 | **+4.7%** |

⟹ ★**추적은 두 팔을 거의 같은 비율로 악화시킨다.** 브리핑의 *"배포 0.1507→0.1578 악화, VARPRO
0.1056→0.1058 중립"* 은 **짝-med3 규약에서만 나오는 대비**다. **두 규약을 함께 인쇄하기 전에는
"추적이 배포만 까먹는다" 를 쓰면 안 된다.**

### 2.3 문헌 선례

`VERIFIED` — **"초기화가 지배하고 정련은 국소 다듬기" 는 비볼록 신호처리의 표준 서사다.**
스펙트럼 초기화 + 국소 정련(Wirtinger flow 계열) 전 문헌이 이 분해를 한다 —
초기화가 **수렴 영역(basin of attraction) 안**에 드느냐가 성패를 가른다
(Lu & Li, arXiv:1702.06435 스펙트럼 초기화 상전이). VARPRO 문헌도 *"fewer iteration steps and
less initial point guessing"*, *"reduced number of local minimizers"* 를 반복한다.
⟹ ⛔**"초기화가 전부다" 자체는 새롭지 않다. 그 분야의 기본 가정이다.**

### 2.4 남는 것

"좋은 초기화 뒤에 붙은 정련이 해를 **악화**시킨다" 의 명시적 보고는 분리형 최소제곱 문헌에서
**내 열람 범위 안에서 못 찾았다**(`UNVERIFIED` — **부재증명 아님**).
그러나 이것은 **VARPRO 에 대한 발견이 아니라 우리 추적기가 잘못 설정됐다는 진단**이고,
저장소는 그 축을 **이미 닫았다**(Rx-7 = *"5%에서 벌고 95%에서 잃는 2.9배 나쁜 거래"*).
⟹ **논문 기여로 승격 불가. 배포 결정으로는 중요(추적 단계를 끄거나 고쳐야 한다).**

---

## 3. Q3 — decorrelation 절반은 이론적으로 예측되는가? `2511.01234` 와 연결되는가?

### 3.1 ★★★**먼저: 그것은 독립 측정이 아니다** — 산술로 보인다 (내 정정 4, 최우선)

`MEASURED` — `decorr_mean` 의 정의(`run_w4.py:383-384`, W5 는 `run_w5.py:240` 으로 그대로 호출):

```
m = MSE / Var(gt) = (1 − ρ²) + (√VR − ρ)²  =  decorr + shrink        # 정확한 직교분해
decorr_mean = mean_frames(1 − ρ²)
```

그리고 `nRMSE = RMSE / Hs`, `Hs ≈ 4σ` ⟹ **`m ≈ 16·nRMSE²`**.
저장소는 이미 **`shrink` 가 `m` 의 0.9–6%뿐**임을 측정했다(S3 `RESULT.md:59`; W4/W5 팔에서 4–10%).
⟹ ★**`decorr ≈ m ≈ 16·nRMSE²`, 즉 `decorr` 는 헤드라인 지표의 단조변환(제곱)에 불과하다.**

**수치로 확인**(내 재집계):

| | `L-SHIP` | `L-VP-ONLY` | 비 |
|---|---|---|---|
| `avg_nRMSE` | 0.1576 | 0.1105 | 0.701 |
| **`nRMSE` 제곱비 (예측)** | — | — | **0.492 (−50.8%)** |
| **`decorr_mean` (실측)** | 0.3851 | 0.1848 | **0.480 (−52.0%)** |
| `shrink_mean` | 0.0221 | 0.0082 | 0.371 |
| `decorr+shrink` vs `16·nRMSE²` | 0.4072 vs 0.3974 | 0.1930 vs 0.1954 | 오차 2.5%/1.2% |

⟹ ★★★**"decorrelation 이 −50.5% 됐다" 는 "nRMSE 가 −25.3% 됐다" 를 제곱해서 말한 것이다.
예측치 −50.8% 와 실측 −52.0% 의 차이는 1.2%p.**
⛔**따라서 `decorr` 감소를 "기제(mechanism)" 로 제시하면 안 된다.** 기제라면
**헤드라인이 같은데도 달라지는 양**이어야 하는데, 이 양은 헤드라인에 대수적으로 묶여 있다.
✅**살아 있는 것은 `shrink` 쪽 뿐이다** — `shrink` 비 0.371 이 `decorr` 비 0.480 보다 **더 많이** 줄었다.
즉 **진폭 축소 오차가 위상 오차보다 더 개선됐다**. 그러나 `shrink` 는 `m` 의 5% 이하라 **크기가 없다.**

### 3.2 `arXiv:2511.01234` — 검증 결과, 그리고 **연결은 약하다**

`VERIFIED` (WebFetch): *"A Saddle Point Remedy: Power of Variable Elimination in Non-convex
Optimization"*, **Gan·Chen·Yi·Yang, 2025-11-03, cs.LG/stat.ML**. Hessian inertia + Schur 여인자로
**"축소 지형의 국소 최대 ↔ 원 문제의 안장점" 1:1 대응** 증명. 검증 도메인은
**비볼록 행렬분해 · 2파라미터 신경망 · 심층 ResNet 뿐 — 물리 역문제 없음**(브리핑 정확).

⛔**그 정리는 "어느 오차 성분이 줄어드는지" 에 대해 아무 말도 하지 않는다.** §3.1 을 합치면
"안장 탈출 → 목적함수 하락 → `m` 하락 → `decorr` 하락" 은 **정리의 예측이 아니라 항등식 사슬**이다.
⟹ ★**`2511.01234` 를 기제의 이론 근거로 인용하면 심사위원이 "그 정리는 그 말을 안 한다"로 깎는다.**

### 3.3 ★★★**그런데 훨씬 더 가까운 논문이 이미 있다** — 두 번째 산출

`VERIFIED` (WebFetch, arxiv.org/abs/2605.00585):

> **"Local Geometry of Least Squares for Unmixing Signals with Parameter-Dependent Dictionaries"**
> Santos Michelena · Maxime Ferreira Da Costa · José Picheral — **2026-05-01**, **eess.SP; math.NA**
> **unmixing metric**(선형·비선형 파라미터의 서로 다른 역할·민감도를 담는 거리) 도입, 수렴 보장 확립,
> **variable projection 을 기하학적으로 분석 — 최적 선형 파라미터의 다양체 위로 최적화를 제한하는 것**,
> **PSF unmixing** 에 적용, **parametric coherence** 도입, 그리고
> ★***"support separation directly controls both the size of the convergence region and the stability
> of recovery."***

★**우리 상황과 축자적으로 겹친다**: parameter-dependent dictionary + 선형 계수 = 우리의
`w·cos(θ+φ) = a·cosθ + b·sinθ`(`k` 만 비선형) · parametric coherence = 우리 원자 간 상관 ·
support separation = 우리 분리도 · **PSF unmixing = 물리 역문제**.
⟹ ⛔**"물리 역문제에는 아직 이론이 없다" 는 방어선이 2026-05 에 무너졌다** (브리핑 전제 정정 1).

**두 방향으로 작동한다.**
- ✅**유리**: 이론 근거를 공짜로 얻는다. 그리고 ★**§4 실험 B 의 예측을 만들어 준다**(분리도 → 수렴 영역).
- ⛔**불리**: **기제 novelty 는 3개월 전 eess.SP 에 선점됐다.** 남는 것은
  *"이론이 예측한 것을, 그 이론의 레짐 가정이 실제로 성립하는 물리 문제에서 측정했다"* 뿐이다.

⚠**범위**: 나는 `2605.00585` 의 **초록·서지만** 확인했고 **본문 미독**(부록에 basin 반경 분석이 있다는
2차 요약만 봄 — `UNVERIFIED`). **인용 전에 본문 필독.**

---

## 4. Q4 — 무엇을 더 재면 신호처리 논문이 되는가

★진단: 현재 처치는 **4요인 묶음**이다. `L-VP-ONLY` 는 `L-SHIP` 대비 아래를 **동시에** 바꾼다.
(코드 실독으로 확인, `MEASURED`)

| 요인 | 근거 | 문헌이 "당연" 하다고 말하는가 |
|---|---|---|
| **(i) 진폭·위상의 닫힌형 소거** | `varpro.py:174-187` 정규방정식+ridge 1e-8 | ★**그렇다**(Golub–Pereyra 초록) |
| **(ii) 소거를 통과하는 도함수(`dc/dK` 항)** | 독스트링 `:49-56` — **Kaufman 이 버리는 항을 autograd 가 나른다** | ⚠**아니다** — 여기가 비자명한 절반 |
| **(iii) ℓ₁ 벌점 처리 변경** | ★**VARPRO 목적함수에 `lambda_l1` 이 없다**(파일 전체 grep 0회) | ⛔**요인이지 기여가 아니다** |
| **(iv) 성장분기 발화 3배** | W5 `RESULT.md` §4 — VP 팔 141/192 vs 배포측 47/192 | ⛔**요인이지 기여가 아니다** |

⚠**(iii) 이 특히 위험하다**: `L-VP-ONLY` 는 `learned_init_iters:0` 이라 **λ 가 붙은 결합 500스텝을
통째로 제거**하고 **벌점 없는 VP 400스텝**으로 갈아끼운다 ⟹ **처치는 "VARPRO" 가 아니라
"VARPRO + λ 제거" 다.** (W3 이 배포 λ=1e-4 ≈ λ=0 을 널로 재긴 했으나 **다른 팔에서**다.)

### ★실험 A (최우선, 가장 쌈) — **ALS 팔로 (i) 과 (ii) 를 가른다**

| | |
|---|---|
| 팔 | **`L-ALS`**: 매 스텝 `(a,b)` 를 닫힌형으로 풀되(= (i)), **그 해를 상수로 detach** 하고 `k` 1스텝 하강 |
| 구현 | `varpro.py:180` 의 `c` 에 `.detach()` 한 줄 — ★**`dc/dK` 경로만 끊는다.** 나머지 전부 동일 |
| 사는 것 | ★**`L-VP-ONLY < L-ALS` 이면 이득의 원천이 "닫힌형 풀이"가 아니라 <br>**"소거를 통과하는 도함수"** 다. 그것은 Golub–Pereyra 의 **비자명한 절반**이고 <br>`2605.00585` 의 "최적 선형 다양체로 제한" 기하와 직접 맞물린다** |
| ⛔반대 | `L-ALS ≈ L-VP-ONLY` 이면 **이득은 (i) 뿐 ⟹ Q5 의 "당연하다" 확정, 축을 닫아야 한다** |
| 동반 필수 | **`L-VP-ONLY-λ` 팔**(VP 목적함수에 λ=1e-4 를 IRLS/근접연산으로 넣거나, 대칭으로 배포 팔의 λ 를 0 으로) ⟹ 요인 (iii) 제거 |
| 비용 | 2팔 × 192런 ≈ **W5(640런/8워커 35–45분) 외삽으로 25–30분** + 코드 반나절 |

⟹ ★**이 실험이 "논문이 되느냐"를 한 번에 가른다. 무엇보다 먼저.**

### ★실험 B — **분리도 상전이 곡선** (C4 를 사는 유일한 길)

| | |
|---|---|
| 설계 | 성분 분리도를 **Rayleigh 단위로 스윕**(0.5 / 0.8 / 1.0 / 1.3 / 2.0 / 4.0 R), 각 수준에서 `L-VP-ONLY` vs `L-SHIP` (+ 실험 A 의 `L-ALS`) |
| 예측 | `2605.00585`: *"support separation 이 수렴 영역 크기와 복원 안정성을 지배"* ⟹ ★**분리도가 크면 이득 소멸, 초해상으로 갈수록 이득 증가** 하는 단조 곡선 |
| 사는 것 | ★**측정된 상전이 = C4 충족.** "우리 이득은 초해상 레짐에서만 존재한다"는 **범위 문장** + **2026-05 이론의 첫 물리-역문제 검증**. **널이어도 정보**(이론 미지지) |
| 비용 | 6수준 × 32ep × 2~3팔 × 3시드 = **1,150–1,730런 ≈ 1–2.5시간** + 생성기 분리도 제어 |
| ⚠전제 | **먼저 §6-C4 의 Rayleigh 수치 문제를 고쳐야 한다** |

⛔**실험 A 없이 B 만 하면 "당연한 것의 곡선"이 된다.**

---

## 5. ⛔ Q5 — 죽이는 각도 (최우선 산출)

### 5.1 주 논증 — **"2003년 종설의 초록이 이미 그렇게 말한다"**

| 우리 측정 | 문헌이 이미 말한 것 |
|---|---|
| **2.05× 빠름** (29.7 s → 14.5 s) | *"will always converge faster"* |
| **스텝 720 < 820** | *"reduces the dimension of the parameter space"* |
| **−25.3% 더 정확** | *"better-conditioned"* ⟹ 같은 예산에서 더 낮은 해 |

★**그리고 그 문장의 전제("같은 최적화기")를 우리가 만족한다** — 양 팔 모두 Adam(`varpro.py:209-211`).
⟹ 내가 세우려던 방어선(§0)이 코드 실독으로 무너졌다.

★기준선은 **결합 파라미터화 위의 Adam(1차 경사)** 이다. 분리형 문제에서 **1차 결합 하강은
문헌이 열등하다고 아는 바로 그 팔**이다(`VERIFIED` — VARPRO 대 ALS 비교에서
*"techniques based on alternating least squares have significant disadvantages"*; ALS 조차 결합 1차보다 낫다).
⟹ ★★**"강한 방법이 약한 기준선을 이겼는데, 그 우열은 1973년부터 알려져 있었다."**

### 5.2 보조 논증 다섯 개 (심사위원이 실제로 쓸 것)

1. ⛔**기제 주장이 헤드라인의 제곱이다**(§3.1). *"decorrelation −50.5%"* 는 *"nRMSE −25.3%"* 와
   **1.2%p 안에서 같은 수**다. ★**심사위원이 이걸 발견하면 "기제 측정" 이라는 두 번째 기둥이 사라진다.**
2. ⛔**"이득이 초기화에서 난다" 는 패치 지점의 재진술이다**(§2.1) — 개입이 초기화에만 있다.
3. ⛔**`P-2` 널(결합 적합 불필요)도 예측된다.** VP 가 더 잘 조건화된 축소판이면 그 위에 원 지형의
   하강을 얹어 더 얻을 게 없다. `JOINTFIT-REDUNDANT` 는 **정리의 따름정리**에 가깝다.
4. ⛔**파랑 문헌 쪽 절반도 이미 있다.** `VERIFIED` — Desmars et al., *JFM* **975**(2023) **AMBM** 은
   **파수를 고정해 놓고 진폭·위상을 최소제곱(LM, 명시적 야코비안)으로 푼다**.
   ⟹ ★**"진폭·위상이 선형" 이라는 구조는 파랑 재구성 문헌이 이미 쓰고 있다.** 우리가 더한 것은
   **그 위에서 `k` 를 하강시킨 것**이고, 그 표준 도구가 VARPRO 다. ⟹ **양쪽 절반이 다 기성품이고 우리는 접합부다.**
5. ⛔**정규화된 VARPRO 도 성숙한 하위분야다.** `VERIFIED` 서지 — Tikhonov VARPRO
   (*Complexity* 2019; *Axioms* 10:196; IEEE 8360468), **ℓ₁ VARPRO = VPAL**
   (*J. Comput. Appl. Math.*, S0168927423001800), 정규화 분리형 **역문제** 수렴해석(arXiv:2402.08568),
   **arXiv:2601.05224 (2026-01)** *"Variable Projection Methods for Solving Regularized Separable
   Inverse Problems"*. ⟹ **우리 문제의 정확한 형태(ℓ₁ + 분리형 + 역문제)에 전용 알고리즘 계열이 이미 있다.**
   그리고 **우리 구현은 ℓ₁ 을 그냥 떨어뜨렸다**(§4-(iii)) — 심사위원은 정확히 그 자리를 찌른다.
6. ⛔**가장 가까운 선례가 하필 LiDAR 다** — Wang et al., *Complexity* 2020. 심사위원이 이 한 편만
   찾아도 *"LiDAR 신호에 VARPRO 적용은 6년 전"* 이 된다.

### 5.3 ★**"당연하다" 를 깰 수 있는 유일한 자리**

Golub–Pereyra 의 문장은 **(i) 차원축소·조건수**에 대한 것이다. 그것은
**(ii) `dc/dK` 도함수 항이 얼마나 버느냐**를 **적분하지 않는다** — Kaufman 근사가 널리 쓰이는 이유가
바로 그 항을 버려도 대개 괜찮기 때문이고, **우리는 그 항을 살린 구현**이다(`varpro.py` 독스트링).
⟹ ★**`L-ALS`(= `c.detach()`) 대비 유의한 이득이 있으면 그것은 문헌이 예측하지 않은 크기다.**
**그 실험 없이는 최적화/신호처리 벤치에 내면 안 된다.**

### 5.4 ⚠**과잉 킬 금지** — 살아 있는 것 둘

- ✅**공학적 가치는 온전하다** — **더 정확(−25.3%, 58/64, p=4.5e-12) · 더 빠름(2.05×) · 더 쌈(720<820) ·
  9셸 전부 통과**. **배포 결정으로는 완결적**이다. **novelty 와 배포 가치를 혼동하지 마라.**
- ✅**W4 `P-2`(결합 적합을 그냥 빼면 +23.3% 나빠진다)는 문헌이 예측하지 않는다.**
  "결합 하강은 무용"이 **아니라** "VP 가 있을 때만 무용"이라는 **조건부 구조**는
  Golub–Pereyra 의 단조 서사보다 한 겹 복잡하다. ⟹ **C3 의 유일한 후보.** 다만 §5.2-3 의
  반론(따름정리다)을 이기려면 이론적 진술이 필요하고, 우리에겐 없다.

---

## 6. 정정 (규율상 반드시 보고)

| # | 정정 |
|---|---|
| **C1** | ★★★**"decorrelation −50.5% 가 기제다" 철회 권고** — `decorr = 1−ρ² ≈ 16·nRMSE²` 이고 헤드라인 제곱비 −50.8% 와 **1.2%p 차**. **독립 측정이 아니다**(§3.1) |
| **C2** | ★**"이득이 전부 초기화에서 난다" 는 발견이 아니다** — `varpro.py` 가 `_initialize_learned` 만 패치한다(§2.1). 게다가 W4 `RESULT.md` 에 그 문장이 없고, `recon/C2_varpro.md:244` 는 **추적기 상호작용을 "미측정"** 이라 적었다 |
| **C3** | ⚠**"추적은 VARPRO 만 안 까먹는다" 는 규약 의존** — raw median 으로는 배포 +5.5%, VARPRO +4.7% 로 **거의 같다**(§2.2) |
| **C4** | ⚠**"1.29 Rayleigh / 0.07 Rayleigh" 전제가 무르다** — 출처 `recon/W2_upper_bound.md` §3.4 는 `[M]` 태그가 **자기선언**이고 **재현 스크립트가 `recon/` 에 없다**. 같은 배치의 독립 측정(`AUDIT_NEUTRAL.md:214`)은 **분리 중앙 0.0569 = 1.63 Rayleigh**(다른 코퍼스). 그리고 **0.07 은 "≈0.5 rad @200 m" 라는 근거 미기재 문턱**에서 나온 `[A]` 다. ⟹ ★**"초해상 영역" 은 유지되지만(1.29~1.63 모두 <2R), 실험 B 전에 이 표를 재산출해야 한다** |
| **C5** | ⛔**"VARPRO 는 `scipy` 에도 사실상 표준" 은 거짓** — `VERIFIED`, `scipy` 에 VARPRO 구현이 **없다**. arXiv:2401.02301 이 *"the only modern implementation of a VP algorithm is a MATLAB code by O'Leary and Rust [2013]"* 라 적는다. 실재하는 것은 그 MATLAB 코드와 서드파티 Rust crate(`geo-ant/varpro`, **Kaufman 근사**). ⟹ **"표준 라이브러리에 있을 만큼 당연" 논증은 못 쓴다**(§5.1 주 킬은 이것과 무관하게 성립) |
| **C6** | ⚠**`2511.01234` 를 기제 근거로 쓰지 마라**(§3.2). 쓸 것은 **`2605.00585`** 이고 **본문 미독이라 아직 인용 금지** |
| **C7** | ⚠**`verdict_w5.json` 의 `"iut": [55,64]` 는 IUT 가 아니다** — `analyze_w5.py` 에서 `w,n` 이 세 번 재바인딩돼 **마지막 셸(`obs:150_inf`)** 이 직렬화됐다. 참 IUT 는 **63/64**(W5 `RESULT.md:12`). **판정 문자열은 무손상**(불리언이 덮어쓰기 전에 묶임). ⚠**정적 분석 결과이고 나는 분석기를 실행하지 않았다** |
| **C8** | ⚠**`varpro.py` 독스트링의 "FULL Golub–Pereyra" 는 미세한 과장** — 실제로는 **ridge 1e-8 로 정규화된** 축소 목적함수의 GP 경사다. 자기시험(autograd vs FD)은 **같은 ridged 목적함수**를 비교하므로 이 구분을 잡지 못한다. 수치적으로는 무시할 만하다 |

---

## 7. ⛔ 이 문서가 주장하지 않는 것

- ⛔**"VARPRO 결과가 틀렸다" 고 말하지 않았다.** 원자료를 재집계해 **P-1·P-2·속도·decorr·상한비를 전부 재확인**했다.
- ⛔**"배포하지 마라" 고 말하지 않았다.** 배포 가치와 novelty 는 별개다(§5.4).
- ⛔**부재증명 금지** — §2.4 의 *"역효과 보고를 못 찾았다"* 는 **내 열람 범위 안**의 진술이다.
- **범위 상속**: 합성 · 선형 GT · `direction_deg=0` · 혼잡 층 · `init_method='learned'` 고정.
  W4/W5 사전등록의 인용금지(⛔전역 헤드라인 · ⛔희소 · ⛔"벽을 닫았다"(상한 대비 ×81~×90) ·
  ⛔`ORA-K-CVX` 비를 솔버 이득으로) 는 **전부 그대로 유효**하다.

---

## 8. 검증한 문헌 (전건 WebSearch/WebFetch)

| 출처 | 확인 방식 | 상태 |
|---|---|---|
| **Golub & Pereyra, *Inverse Problems* 19 (2003) R1** | WebFetch 초록 축자 | `VERIFIED` — §5.1 킬 문장 |
| O'Leary & Rust, *COAP* **54**:579 (2013) | WebSearch 서지 | `VERIFIED` |
| Ruhe & Wedin (1980) | 2차 요약만 | ⚠`PARTIAL` — 직접 미확인 |
| **arXiv:2511.01234** (Gan·Chen·Yi·Yang, 2025-11) | WebFetch 초록 축자 | `VERIFIED` — 검증 도메인 ML 한정 |
| **arXiv:2605.00585** (Michelena·Ferreira Da Costa·Picheral, 2026-05, eess.SP) | WebFetch 초록 | `VERIFIED` 서지 / ⚠**본문 미독** |
| arXiv:2512.07969 Sparse VARPRO in Robotic Perception (2025-12) | WebFetch | `VERIFIED` 서지 |
| Wang et al., *Complexity* 2020 — **LiDAR 파형 VARPRO** | WebSearch | `VERIFIED` 서지 |
| arXiv:2401.02301 / *Mathematics* 11:2839 (DLR) | WebSearch | `VERIFIED` — **scipy 부재 근거** |
| VPAL, ℓ₁ 정규화 VARPRO (*JCAM*, S0168927423001800) | WebSearch | `VERIFIED` 서지 |
| arXiv:2402.08568 정규화 분리형 역문제 수렴해석 | WebSearch | `VERIFIED` 서지 |
| arXiv:2601.05224 (2026-01) 정규화 분리형 역문제 VP | WebSearch | `VERIFIED` 서지 |
| Fu et al. Tikhonov VARPRO (*Complexity* 2019 · *Axioms* 10:196 · IEEE 8360468) | WebSearch | `VERIFIED` 서지 |
| Denoyelle et al., Sliding Frank–Wolfe (*Inverse Problems* 2019) | WebSearch | `VERIFIED` 서지 (W4 가 이미 SFW 폐기) |
| **Desmars et al., *JFM* 975 (2023) — AMBM** | WebFetch 논문 페이지 | `VERIFIED` — **파수 고정, 진폭·위상만 LS, k 최적화 없음** |
| IEEE SPS Guidelines for Reviewers / for Associate Editors | WebFetch 양쪽 | `VERIFIED` — 즉시기각 문언 축자 |
| *"trivial application ... may be rejected"* | WebFetch 2회 실패 | ⛔`UNVERIFIED` — **인용 금지** |
| Lu & Li, arXiv:1702.06435 스펙트럼 초기화 상전이 | WebSearch | `VERIFIED` 서지 |
