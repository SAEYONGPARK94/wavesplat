# R3-2 · novelty 심사 — **Wang 이 방향을 지목한 뒤에 무엇이 남는가**

> 라운드 3 · 2026-08-22 밤 · **추정 런 0회** · 산출 단일 파일
> 규율: `BRIEF.md` + `BRIEF2.md` + `BRIEF3.md`. `MEASURED`/`ARGUED`/`UNVERIFIED` 문장별 분리.
> ★지시대로 **`R2_5_literature.md` 를 열기 전에 원문 PDF 를 먼저 추출해 읽었고**, 그 다음에 대조했다.

---

## 판정 요약 (맨 위)

### A. 핵심 질문에 대한 답

| | |
|---|---|
| **Wang 2025 가 우리 방향을 지목했는가** | ★**부분적으로 그렇다. 그러나 "명시 지목" 은 과장이다.** 마지막 문장 하나의 **네 절 중 둘만** 우리와 일치한다(§2.3) |
| **그 뒤에 남는 novelty** | ★**표현(ⓐ)도 경계(ⓑ)도 원장(ⓓ)도 못 산다. 유일하게 사는 것은 ⓒ 의 절반** — *"벌점이 얼마를 물어냈는지를 개입으로 회계한 것"* |
| **⛔그러나 ⓒ 의 선언된 이유(*"차원 불일치"*)는 선점됐다** | 통계·PINN 문헌에서 **3중 선점**(§4.2). 게다가 ★**우리 스스로 그 이유를 재도출하면 숫자가 안 맞는다**(§4.4) |

### B. 후보별 처형표

| | 후보 | 판정 | 사유 (한 줄) |
|---|---|---|---|
| **ⓐ** | 격자 없는 스펙트럼 표현을 순차 추적에 | ⛔**`DEAD — 3중 선점 + 자기모순`** | 표현 자체 = off-grid line spectral estimation(2013 이전) · 순차화 = Pascoal 2017·Kim 2020(**Wang 이 직접 인용**) · ★**우리 실측 경로는 격자다**(`bank_mmse` `meshgrid`) |
| **ⓑ** | 경계의 개입 측정 | ⚠**`ALIVE-BUT-THIN` (기여이나 novelty 아님)** | 개입 설계 1건(HOSGATE 짝 GT)만 방법론적으로 신선. 나머지는 **내부 노브 대조** ⟹ 08-15 감사가 이미 죽인 부류 |
| **ⓒ** | 벽의 원인 분해 (`W7·W9·W10`) | ★**`ALIVE — 단 절반만`** | **기제는 선점**(lasso λ∝σ, 1949–2012 / PINN 무차원화 2021–) · ★**측정(개입 귀속 %)은 문헌에 없다** |
| **ⓓ** | 정직한 음성의 원장 | ⛔**`NOT-A-CONTRIBUTION` (학위 서술로는 유효)** | 논문 기여로 팔 수 없다. 인용 가능한 venue 가 ML 워크숍 하나뿐이고 해양공학에 없다 |
| **ⓔ**(내가 추가) | ★**√N 벌점 스케일링이 레짐 비대칭을 낳는다** | ★**`NEW — 미측정, 런 16회`** | §4.5. **아무도 쓰지 않았고 유도 두 줄 + 이미 있는 자료로 확인 가능** |

### C. ★내가 이번 라운드에 새로 만든 것 (전부 런 0회)

| # | 발견 | 등급 |
|---|---|---|
| **N-1** | ★★**`W9`/`W10` 이 "차원 불일치" 라 부른 것 중 `λ_smooth` 는 **차원이 맞는다**. `Σ(Δw)²` 는 `[m²]` 다 | ★★ `MEASURED`(코드) |
| **N-2** | ★★★**벌점의 1차 최적점 변위는 성분당 `λ = 1e-4 m = 0.1 mm` 다** ⟹ 측정된 `1.52배` 를 **벌점의 편향으로 설명할 수 없다**. `W9`/`W10` 의 회계는 *"벌점 최적점의 편향"* 과 *"옵티마이저가 그 최적점에 못 간다"* 를 **가르지 않는다** | ★★★ `ARGUED`(측정 입력) |
| **N-3** | ★★**`SCALE_INVARIANT_L1` 은 잘못된 정규화자를 쓴다** — 신호 RMS(0.2567 m)로 나누는데 lasso 이론이 요구하는 것은 **잡음 척도**(5.87 mm)다. **43.7배 과보정** ⟹ 고쳐도 벌점이 목적함수의 **55%** 로 남는다(계산) ⟹ **47% 만 사라진 것이 예측된다** | ★★ `ARGUED`(측정 입력) |
| **N-4** | ★★★**고정 해상에서 `Σ|A| ∝ √N`** — BF4 에서 GT 장 RMS 는 N 과 무관하게 0.295–0.301 m 인데 `Σ|A|` 는 **0.60(N=2) → 2.28(N=30) = 3.8배** ⟹ **벌점만 성분수를 탄다. 벽이 혼잡 전용인 이유의 후보** | ★★★ `MEASURED` |
| **N-5** | ★**논문의 자기비하 문장이 저장소 기록과 어긋난다** — *"독립적 착상이 아니라 그 방향의 구현"*. 표현은 `2026-04-06` 커밋, Wang 은 `2026-05-08` 에 저장소에 들어왔다 | ★★ `MEASURED`(git) |
| **N-6** | ⛔**우리 Wang 2025 기준선은 무기술이다 — 내가 12셀 전건 재집계** (0.2462–0.2757, 중앙 0.2502 = 제로필드 0.25) ⟹ ⛔*"최근접 경쟁자를 이겼다"* 불가. **`WS<Wang` 은 7/12 뿐** | ★★★ `MEASURED`(재집계) |
| **N-7** | ⛔**`R2-4` 의 `PASS-WITH-REVISION` 사다리가 오늘 논문 수리로 닫히지 않았다** — `R-1`(`체크포인트` 0회) · `R-3`(`Bergamasco`/`Acqua`/`Yellow` 각 0회) · `R-4`(2025–26 서지 **2편** < 요구 5편) | ★★ `MEASURED`(grep) |
| **N-8** | ⚠**셸 분해가 두 개인데 비교자가 다르다** — `W15`(학습 vs 배포 해석적 = **내부 노브**) vs `S16`(방법 vs 외부 기준선). ⛔**섞어서 "SUPPORT-TRADEOFF 아니다" 로 결론내면 08-15 감사가 죽인 오류의 재발** | ★★ `ARGUED` |

---

## §0 열람 범위 (= 주장 범위)

### 직접 연 1차 문헌 (내가 열었다)

| 자료 | 어떻게 | 범위 |
|---|---|---|
| ★**Wang et al. (2025)** *Coastal Eng.* **197**:104690 | 로컬 `ref_paper/feng_wang_2025.pdf` → **PyMuPDF 로 12쪽 전문 추출**(내 스크래치패드) | ★**전문 텍스트**. 결론·§2.1–2.2·§3.2·서론·참고문헌 **축자 인용** |
| ICBINB 워크숍 | WebSearch | 존재·연혁만 |
| off-grid/atomic norm 계열 | WebSearch(2회) | **요약 수준**. 개별 논문 본문 미열람 |
| lasso 척도 문헌 | ⚠**위임 조사** (내 하위 에이전트) — BRT 2009 · sqrt-lasso 2011 · scaled lasso 2012 · glmnet · Kendall 2018 · PINN 무차원화 | 축자 인용은 **에이전트 검증**. 내가 재확인한 것 **0건** ⟹ `[위임-VERIFIED]` |

### 직접 연 저장소 파일 (코드·원자료 — 내가 열고 재집계)

- **코드**: `wavesplat/optimizer/sequential_update.py`(155–190 · 360–485 · 565–600) · `wavesplat/model/spectral_gaussian.py`(1–80) · `wavesplat/optimizer/bank_mmse.py`(50–70 · 150–160)
- **원자료 재집계(내가 계산)**: `results/wang2025_comparison.json`(**12셀 전건**) · `tmp/HOSGATE_2026-08-13/results.jsonl`(**192행 전건**) · `tmp/W9_.../results.jsonl`(**48행 전건**) · `tmp/W7_.../results.jsonl`(**48행 전건**) · `data_indep_crowded_2026-08-08/` 및 `data*/` HDF5 **4개**(GT 장 RMS · 관측 RMS)
- **문서**: `BRIEF{,2,3}.md` · `RESULT.md` · `R2_5_literature.md`(§판정표·§0·§1·§2) · `R2_4_committee_revote.md`(§3.2 사다리 · grep) · `W7`/`W9`/`W10`/`W15`/`W17` 의 `PREREG`/`RESULT`/`AMENDMENT` · `STATUS_2026-08-21/STATUS.md`(1–45) · `reference_citation_bans.md`(선두 60행 + grep) · `thesis/thesis_main.tex`(505–620 · 950–1000 + 전수 grep) · `thesis/thesis_refs.bib`
- **git**: `spectral_gaussian.py`·`etkf_wang2025.py`·`wang2025_comparison.json` 의 **추가 커밋 일자**

### ⛔안 연 것 (판정 시 `[인용]` 표시)

- 라운드 1·2 산출 **14편 중 11편의 전문**(`RESULT.md` 와 grep 으로만 접근).
- **Bergamasco 2021 WASSfast 원문**(⚠서지도 미확인 — §4.3 참조) · **Pascoal 2017** · **Kim 2020 학위논문** · **Tang–Bhaskar–Recht 2013**(로컬 PDF 있으나 미열람) · **Desmars 2023 본문** · **DSGP-Net 본문**.
- `S13`·`S16`·`S17`·`B`·`C`·`W3`·`W4`·`W5` 의 **원자료**(BRIEF·STATUS 수치를 `[인용]`).
- `W17` 결과(런 진행 중, 20행만 존재 — **판정 안 함**).

---

## §1 08-14 / 08-19 / 라운드 1·2 대비 **무엇이 바뀌었나**

| | 그때 | ★지금 (R3-2) |
|---|---|---|
| Wang 2025 의 무게 | 08-19·R1: *"가장 가까운 경쟁자"*(서지만) → R2-5: *"**우리 방법을 future work 로 지명**"* | ⚠**둘 다 정밀하지 않다.** 내가 문장을 **절 단위로 쪼개** 판정했다 — **2/4 일치, 1/4 불일치(공분산 전파), 1/4 미지정(격자성)**. §2.3 |
| ⓐ(격자없음) | R1·R2: *"격자없음은 아직 우리 것"*(L-4) | ⛔**`DEAD`.** 세 겹으로 죽는다 — off-grid CS 문헌 · Wang 이 인용한 Pascoal/Kim 순차 성분 상태 · ★**우리 실측 경로가 격자다** |
| ⓒ(벽의 회계) | `W9`·`W10` 이 47/9/16/28% 를 냈다. **기제 = 차원 불일치** | ★**측정은 산다, 기제는 못 산다.** 그리고 ★**기제를 재도출하면 숫자가 안 맞는다**(N-2) |
| `λ_smooth` | `W9 RESULT §4`: *"여전히 `[m]` 단위"* | ⛔**거짓.** `Σ(Δw)²` 는 `[m²]` = **데이터항과 차원 일치**(N-1) |
| 논문의 위치 문장 | *"독립적 착상이 아니라 그 방향의 구현"* | ⛔**저장소 기록과 어긋난다**(N-5). 표현이 먼저 들어왔다 |
| `R2-4` `FAIL-FOR-NOW` | 오늘 논문 4곳 수리 | ⛔**사다리 5칸 중 3칸이 여전히 비었다**(N-7) ⟹ **등급 안 움직인다** |
| ⓔ(√N) | — | ★**새 축.** 아무도 안 봤고 유도 두 줄이다(N-4·§4.5) |

---

## §2 ★★★Wang et al. (2025) 원문 — **내가 직접 추출해 읽었다**

**서지** (PDF 표제부·각주 축자, `VERIFIED`):
Feng Wang, Qidan Zhu\*, Chengtao Cai, Xiaoyu Wang, Renjie Qiao,
*"From points to waves: Fast ocean wave spatial–temporal fields estimation using ensemble transform Kalman filter with optical measurement"*,
**Coastal Engineering 197 (2025) 104690**. Harbin Engineering University.
Received 8 Aug 2024 / Accepted 12 Dec 2024 / **Available online 18 December 2024**.

### 2.1 ★문제의 문장 — **축자** (결론 마지막 문장, PDF p.11)

> "**Future work could focus on exploring appropriate entry frequencies as state vectors to reduce state dimensionality, while preserving prior correlations through spectral–spatial transformation in the mode of direct sparse assimilation.**"

`VERIFIED` — 내가 PyMuPDF 로 추출한 12쪽 전문에서 직접 읽었다. R2-5 의 인용과 **한 글자도 다르지 않다**(독립 재확인).

### 2.2 ★그 문장을 읽으려면 **먼저 두 곳을 봐야 한다** (R2-5 가 안 붙인 맥락)

**(a) `entry frequencies` 는 Wang 이 만든 말이 아니라 그가 인용하는 계보의 말이다.** 서론 축자(p.2):

> "researchers have explored wave field estimation using KF. For instance, **Pascoal et al. (2017) and Kim (2020)** proposed Kalman filter-based approaches for sea state estimation, but using multiple ships concurrently as buoys. **They constructed the state vectors from wave components (paired real and imaginary spectrum parts) evolving independently, typically using tens of entry frequencies.**"

참고문헌 축자: *"Pascoal, R., Perera, L.P., Soares, C.G., 2017. Estimation of directional sea spectra from ship motions in sea trials. Ocean Eng. 132, 126–137."* · *"Kim, H., 2020. Real-time Monitoring by Using Kalman Filter in the Ocean (Ph.D. thesis). Texas A&M University."*

⟹ ★**Wang 의 "future work" 는 새 표현의 제안이 아니라 *이미 존재하는 상태 매개화(Pascoal 계보)를 스테레오 희소점 문제로 가져오자*는 제안이다.**

**(b) Wang 자신의 상태는 스펙트럼이 아니라 고도 격자다.** §3.2 축자:

> "There are two main reasons for adopting **elevation rather than spectrum as the state**. Firstly, minor numerical perturbations in the frequency domain can lead to significant errors when reconstructing elevation within the filtering. Secondly, using elevation simplifies the observation matrix **H** to an identity matrix… **the state dimension is 16384 × 1 or 65536 × 1**."

⟹ 그 문장은 **자기 설계의 약점(상태 차원 6.5만)에 대한 자기 처방**이다.

### 2.3 ★★★판정 — *"우리 방향을 명시 지목했다"* 는 **과장이다**

문장을 절로 쪼개 우리와 대조한다 (`MEASURED` 대조, 코드 열람 기반):

| # | Wang 의 절 (축자) | WaveSplat 은? | 일치 |
|---|---|---|---|
| 1 | *"appropriate entry frequencies as state vectors"* | N개 성분을 상태로 둔다 | ✅**일치** |
| 2 | *"to reduce state dimensionality"* | 동기가 다르다 — 우리는 **분해능**(격자 아래 파수), 그는 **앙상블 크기 대비 상태 차원** | ⚠**부분** |
| 3 | *"preserving prior correlations through spectral–spatial transformation"* | ⛔**우리는 공분산을 전파하지 않는다.** 결정론적 NLS/VARPRO이고 필터가 아니다 | ⛔**불일치** |
| 4 | *"in the mode of direct sparse assimilation"* | 희소점에 직접 적합한다 | ✅**일치** |
| — | **`entry`(= 이산 스펙트럼의 항목) vs 연속 파수** | 우리 `kx,ky` 는 `nn.Parameter` = **연속 자유변수**(`spectral_gaussian.py:58-59`) | ❓**Wang 이 말하지 않았다** |

★**정확한 문장**:
> *"Wang et al.(2025)은 결론에서 **상태 차원을 줄인 스펙트럼 상태 + 희소점 직접 동화**를 향후 과제로 적었다. 이는 본 연구가 택한 네 설계 중 **둘과 일치**한다. 그러나 그들은 (i) 파수를 **연속 자유변수로 둘 것**을 말하지 않았고 — 그들이 쓴 `entry frequencies` 는 그들이 인용한 Pascoal 계보의 **이산 항목**을 가리킨다 — (ii) 그 상태를 **칼만 공분산 전파 안에** 두자는 것이어서 본 연구의 결정론적 변수투영과는 다른 틀이다."*

⛔**따라서 논문 `thesis_main.tex:966` 의 *"독립적 착상이 아니라 그 방향에 대한 하나의 구현"* 은 두 번 부정확하다**:
1. **지목 범위가 넓게 인용됐다**(위 표 — 2/4 일치).
2. ★**저장소 기록이 반대를 가리킨다**(`MEASURED`, git): `wavesplat/model/spectral_gaussian.py` 는 커밋 **`dc00878` (2026-04-06)** 에 들어왔고, Wang 2025 가 저장소에 나타난 최초 시점은 `baselines/etkf_wang2025.py` + `results/wang2025_comparison.json` 이 함께 추가된 **`aa7da86` (2026-05-08)** 이다 ⟹ **표현이 약 4주 앞선다.**
   ⚠**우선권(priority)은 여전히 Wang 에게 있다**(2024-12-18 온라인). 다투는 것은 우선권이 아니라 **인과 서술**이다.
   ⚠**부재증명이다** — 저자가 그 전에 초록을 읽었을 가능성을 배제하지 못한다.
   ⚠⛔**R2-5 의 *"PDF 가 저장소에 두 달간 있었다(mtime 2026-06-24)"* 는 근거로 못 쓴다** — `ref_paper/` 의 **PDF 거의 전부가 같은 초(`Jun 24 00:28`)** 다 = **일괄 복사 타임스탬프**이지 취득일이 아니다.

★**권고 대체 문언**:
> *"본 연구의 표현은 해당 문헌과 독립적으로 구현되었으나(저장소 기록: 표현 2026-04-06, 해당 문헌의 저장소 편입 2026-05-08), **우선권은 Wang et al.(2025)에 있다**. 본 연구는 그들이 향후 과제로 적은 두 요소(축소된 스펙트럼 상태·희소점 직접 동화)를 공유하며, 파수를 연속 자유변수로 둔 점과 필터가 아닌 결정론적 변수투영을 쓴 점에서 갈린다."*

### 2.4 ★그 논문에서 **우리에게 유리한 것** (전부 내가 축자 확인)

| 축 | 축자 | 함의 |
|---|---|---|
| 표현 | *"sparse 3D points **interpolated to a uniform grid** as measurements"* / 상태 `16384×1` 또는 `65536×1` | 완전한 격자 |
| 플랫폼 | *"deployed on the deck of **a vessel on position keeping**"* | **정지 기하** |
| 예보 | 예측단계는 **한 스텝**(Eq. 2 의 위상 회전). 논문에 `predictable zone`·예보 지평 **없음** | 예보 축이 비어 있다 |
| ★**비용함수 비판** | *"the optimization problem in Eq. (4) and (7) employ a **fixed weight matrix W** as a frequency domain filter … and uses a **scalar α̇ to balance prior information and observed data**"* / 결론: *"distinguishing it from traditional constrained optimization methods with **fixed weight matrices**"* | ★**§4 의 문헌 고리** |

⚠**내 하위 조사 에이전트는 이 두 축자를 확인하지 못했다**(ScienceDirect 403). ★**나는 로컬 PDF 로 확인했다** ⟹ **이것이 "1차출처를 직접 열어라" 규율이 이번 라운드에 실제로 산 값이다.**

---

## §3 후보 전수 처형

### ⓐ *"격자 없는 스펙트럼 표현을 순차 추적에 쓴 것"* ⟹ ⛔**`DEAD`, 그러나 그 안에서 한 조각이 산다**

#### ⓐ-1 표현 자체 = **선점** (분야 밖, 10년 이상)

off-grid / gridless line spectral estimation 은 basis mismatch 를 피하려는 표준 동기이고
정전(canonical) 문헌은 **Tang, Bhaskar, Shah & Recht, *Compressed Sensing Off the Grid*, IEEE TIT 2013** 이다
(⚠**저장소 `ref_paper/tang_bhaskar_recht_2013.pdf` 에 이미 있다 — 우리가 보유만 하고 인용하지 않았다**).
`[위임-VERIFIED + 내 WebSearch 요약]`

#### ⓐ-2 **순차화도 선점** — Wang 이 직접 인용한 계보

**Pascoal & Guedes Soares (2009)** *Ocean Eng.* 36(6–7):477–488 이래로, **상태 = 수십 개 이산 파랑 성분의 복소진폭**인
칼만 추정기가 존재한다. `[위임-VERIFIED-partial]` · Wang 2025 서론이 **축자로 그렇게 요약한다**(§2.2a, 내가 확인).

#### ⓐ-3 ★★★**결정타 — 배포 추적기는 파수를 얼린다** (`MEASURED`, 내가 코드에서 확인)

`configs/estimation.yaml:43` · `run.py:44` 출하 기본값 **`tracking_mode: 'w_phi'`**,
`sequential_update.py:397-399`:

```
elif mode == 'w_phi':
    self.model.kx.requires_grad_(False)
    self.model.ky.requires_grad_(False)
```

⟹ ★**순차 추적 단계에서 `kx,ky` 는 고정이고 진폭·위상만 갱신된다.**
이는 **Desmars et al. (2023) JFM 975 A8 §2.4** 의 AMBM 과 **형태가 같다** —
> "the wavenumbers of the wave components are not defined the same way for each method: they are implicitly defined by the grid size and resolution for HOS–OCM, and **arbitrarily chosen for AMBM**." `[위임-VERIFIED, Cambridge Core 전문]`

그리고 **Qi, Wu, Liu, Kim, Yue (2018) JFM 838** 의 HOR 과도 같다 —
> "HOR **optimizes the amplitude and phase** of L free wave components…" (⚠`free` 는 *자유파 vs 구속파* 의 뜻이지 자유 파수가 아니다) `[위임-VERIFIED]`

⟹ ★**"격자 없는 표현을 순차 추적에 썼다" 는 문장은 배포 시스템에 대해 거짓이다.
연속 파수는 오직 초기화 창에서만 자유변수다.**

#### ⓐ-4 **실측 경로는 아예 격자다** (`MEASURED`, 내가 코드에서 확인)

`sequential_update.py:569-590` 의 `REAL-DOMAIN GATE` 가 GT 없는 입력(=모든 실측)을 `bank_mmse` 로 보낸다.
`bank_mmse.py:60-61` 은 `torch.linspace` + `torch.meshgrid` 로 **파수 격자**를 만들고, `:158` 도 `np.meshgrid` 다
(2단 zoom 이 있으나 여전히 격자 탐색이다). ⟹ ⛔**제목의 *grid-free* 는 합성 경로 한정이다.**
(라운드 1 B2 의 소견을 **내가 독립 확인**했다.)

#### ⓐ-5 ✅**그래도 한 조각이 산다** — 내 첫 판단을 정정한다

내 하위 조사가 **해양파 분야에서 연속 파수를 자료로부터 추정하는 재구성 논문을 찾지 못했다**.
가장 가까운 두 개는 오히려 **틈을 증언한다**:

- **Jiang, Kalisch, Benoit, Holand, Sprenger (2026)**, arXiv:2605.25074 §5 축자 `[위임-VERIFIED, arXiv HTML]`:
  > "Although the use of a discrete Fourier basis may introduce **off-grid effects, such as spectral leakage and frequency drift**, the results indicate that it **remains a suitable and physically meaningful representation** for sparse wave reconstruction."
  ⟹ ★**분야가 off-grid 문제를 알면서 격자를 유지하기로 명시 선택했다.** 이것이 인용 가능한 최상의 "틈" 문장이다.
- **Desmars 2023** 은 파수를 *"arbitrarily chosen"* = **사용자가 미리 고정**한다(위 축자).

⚠**단 두 개의 유보**(둘 다 반드시 논문에 적을 것):
1. **Plant, Holland & Haller (2008)** IEEE TGRS 46(9):2644–2658 및 그 계보(**cBathy**, Holman 2013)가
   **해양파에서 연속 파수를 비선형 역산으로 추정한다** — 목적은 수심 역산(주파수당 지배 파수 1개)이라
   N-원자 장 재구성이 아니지만, **인용하고 구별하지 않으면 심사에서 맞는다**. `[위임-UNVERIFIED — IEEE 차단]`
2. 기법 자체는 1970년대–2013 신호처리 ⟹ **발명이 아니라 이식**이다(저장소가 VARPRO=1973 으로 이미 겪었다).

★**ⓐ 의 생존 형태(유일하게 방어 가능한 문언)**:
> *"파수를 격자에도 사전 지정에도 묶지 않고 **초기화 창에서 자료로부터 연속값으로 추정**하는 것.
> 추적 단계는 파수를 고정하므로 그 부분은 선행 연구와 같은 형태다."*
⛔**"순차 추적"·"grid-free 시스템"·"실측" 은 이 문장에 붙일 수 없다.**

### ⓑ *"경계의 개입 측정"* ⟹ ⚠**`ALIVE-BUT-THIN` — 기여이나 novelty 는 얇다**

#### ⓑ-1 ⛔선점: "어디까지 되는가" 는 **predictable zone 문헌이 이미 정의했다**

**Desmars et al. (2023)** JFM 975 A8 §4.1 축자 `[위임-VERIFIED, Cambridge Core 전문]`:
> "The prediction accuracy is quantified over the **theoretically accessible prediction zone** corresponding to the spatio-temporal domain within which pieces of information about the reconstructed wave components, **travelling at the group velocity** c_g=∂ω/∂k, **overlap**."

**Qi, Wu, Liu, Kim, Yue (2018)** JFM 838 축자 `[위임-VERIFIED]`:
> "For all the cases we consider, HOR **converges to the underlying wave field within** a nonlinear spatial-temporal predictable zone P_NL…"

계보: Naaijen, Trulsen & Blondel-Couprie (2014) *Int. Shipbuild. Prog.* 61(3–4):203–223 ·
Qi et al. (2018) *Wave Motion* 77:195–213 · Lee et al. (2024) *Proc. IMechE M* (**이동 측정까지 포함**).

⟹ ⛔**"경계를 그렸다" 는 novelty 가 아니다. 그 분야의 창립 결과다.**

#### ⓑ-2 ✅그럼에도 남는 **한 줄** — 그러나 크기가 문제다

문헌은 경계를 **정보/운동학**으로 설명하고(**군속도 도달**), 지대 **안에서는 수렴을 단언**한다.
우리 측정은 지대 **안에서** 참값에서 출발해도 해가 이탈함을 보인다:
`W7`/`W9` 혼잡 n=16 (내가 두 배치 원자료로 **독립 재도출**):

| λ_l1 | 창-안 잔차 > 진실 | 짝 중앙비 |
|---|---|---|
| 0 (`W7` 기준팔) | — | — |
| ★**1e-4 (출하)** | **16/16** | ★**1.516268** (`W9`) |
| 1e-3 (S4·S5) | 16/16 | **3.4980** (`W7` 원자료) |

⟹ ★**정직한 문장** = *"predictable zone 은 **필요조건이지 충분조건이 아니다** — 정보가 도달한 영역 안에서도 솔버가 진실보다 나쁜 곳에서 멈춘다."*

⛔**그러나 크기가 작다.** 두 벌점을 다 고치면 비가 **1.146** 으로 내려간다(`W10`) ⟹
**문헌과 다투는 순수 잔여는 ≈15%** 다. ⛔*"26.9배"·"3.66배" 는 λ=1e-3 조건이라 인용 금지**(`reference_citation_bans.md`, 08-15 정정).

#### ⓑ-3 ★방법론으로는 **하나만** 신선하다 — 짝 GT 개입

`HOSGATE` (내가 **192행 전건 재집계**, `results.jsonl`):

| GT | 층 | `SHIP<LETKF+` | 상대 중앙 | `SHIP` 절대 중앙 |
|---|---|---|---|---|
| LIN | 혼잡 | 14/16 | −28.9% | 0.1557 |
| ★**HOS M=3** | 혼잡 | **14/16** | **−25.4%** | 0.1586 |
| LIN | 희소 | 16/16 | −90.9% | **0.0195** |
| ★**HOS M=3** | 희소 | 15/16 | −72.3% | ★**0.0700** |

★**비선형화 비용이 팔마다 다르다**: `SHIP` 희소 **0.0195 → 0.0700 = 3.59배**,
`LETKF_PLUS` **+0.5%**, `ZERO` **−0.0%** ⟹ **"희소 이득 = 표현 아티팩트" 가 추론이 아니라 개입 측정이다.**
⟹ ★**같은 성분 집합 위에서 참값 생성기의 비선형성만 토글해 이득의 귀속을 가르는 설계**가 ⓑ 에서 유일하게 방법론적으로 팔 만한 조각이다.

⚠**나머지는 전부 내부 노브 대조**다(08-15 감사가 `SUPPORT-TRADEOFF` 10건 전수에서 이미 죽인 부류).
⚠**N-8**: `W15` 의 9셸 평탄(2.18–2.52배)은 **학습 vs 배포 해석적 = 내부 대조**이고, `S16` 의 셸 분해가
**방법 vs 외부 기준선**이다. **둘을 섞어 *"SUPPORT-TRADEOFF 아니다"* 로 결론내면 같은 오류의 재발**이다.

### ⓓ *"정직한 음성의 원장"* ⟹ ⛔**`NOT-A-CONTRIBUTION`** (학위 서술로는 유효)

- **논문 기여로 팔 수 없다.** 해양공학에 음성결과 venue 가 없다.
- ML 쪽 **ICBINB**(*I Can't Believe It's Not Better*, NeurIPS 워크숍 계열, 2020·2022·2023 확인)가 유일한
  형식적 창구이나 도메인이 안 맞고, **2026 에 확인되는 것은 생물 분기(ICBINB-BIO)** 다. `[내 WebSearch — VERIFIED 존재, 2026 일반분기 UNVERIFIED]`
- ★**단 학위논문 제6장/부록으로는 강하다**, 그리고 `기여 ⑥`(현 목록의 6번)에 이미 그렇게 적혀 있다.
- ⛔**"26축 사망" 같은 총량 표현은 그 자체가 인용 위험이다** — `reference_citation_bans.md` 가 **금지 목록의 총량 표현**을
  이미 금지했고(같은 계수-규칙 문제), 축 계수도 문서마다 달라진 전력이 있다. **개별 축을 날짜 태그로 가리켜라.**
---

## §4 ★★★ⓒ 심층 — *"손실함수가 [m] 와 [m²] 를 더한다"* 는 **일반 결함인가 우리 버그인가**

### 4.1 무엇이 실제로 코드에 있는가 (`MEASURED` — `sequential_update.py:430-478`, 내가 읽었다)

```
total_loss = Σ_obs w·(z_pred − z_obs)²  /  n_obs          # [m²], n_obs 로 나눈 평균
total_loss += lambda_l1     * Σ_i |w_i|                    # [m]  ← 차원 불일치
total_loss += lambda_smooth * Σ_i (w_{i+1} − w_i)²         # [m²] ← 차원 일치
```

★**N-1 (내 정정)**: `W9 RESULT §4` 는 *"`lambda_smooth`(1e-4) 가 **여전히 [m] 단위**다"* 라고 적었다.
⛔**거짓이다.** `w` 는 진폭 `[m]`(`spectral_gaussian.py:38` 독스트링 축자 *"amplitudes / weights [m]"*)이므로
`Σ(Δw)²` 는 `[m²]` = **데이터 항과 같은 차원**이고 `lambda_smooth` 는 **무차원**이다.
⟹ ⛔`W10 RESULT §1` 의 *"두 벌점의 **단위**를 함께 고치면 72%"* 도 **틀린 라벨**이다.
`λ_smooth=0` 은 **단위 수리가 아니라 항의 제거**다.
✅**논문 본문(`thesis_main.tex:539-541`)은 이미 정확하다** — 47% 만 차원 불일치로 귀속하고 9%·16% 는 중립 서술이다.
⟹ **고칠 곳은 `W9`/`W10` 의 `RESULT.md` 와 `BRIEF3 §5` 문언이다.**
⚠**내 하위 조사 에이전트가 같은 결론에 독립 도달했다**(*"λ_s is already dimensionally consistent … only λ₁ drifts"*).

### 4.2 ⛔**기제는 선점됐다 — 3중** `[위임-VERIFIED]`

| 층 | 축자 | 함의 |
|---|---|---|
| **통계** | Bickel, Ritov & Tsybakov (2009) *Ann. Statist.* 37(4):1705 — *"the Lasso estimator … with **r = Aσ√(log M / n)**"* | λ 를 **σ 에 비례**하도록 정의하는 것이 정전이다. `√(log M/n)` 이 **mean-vs-sum 부기까지** 이미 담는다 |
| **통계** | Belloni, Chernozhukov & Wang (2011) *Biometrika* 98(4):791 — *"The method is **pivotal** in that it neither relies on the knowledge of the standard deviation σ nor does it need to pre-estimate σ."* | ★**이 논문이 존재하는 이유가 바로 그 결함이다** |
| **통계** | Sun & Zhang (2012) *Biometrika* 99(4):879 — *"iteratively estimating the noise level via the mean residual square and **scaling the penalty in proportion to the estimated noise level**"* | 우리의 "수리"와 같은 계열, 14년 전 |
| **소프트웨어** | `glmnet` 문서 — *"`glmnet` **standardizes y to have unit variance** … before computing its lambda sequence"* / scikit-learn `Lasso` 목적함수에 **`1/(2·n_samples)`** 가 박혀 있다 | 관례가 이미 처방이다 |
| **멀티태스크** | Kendall, Gal & Cipolla (CVPR 2018) — *"the optimal weighting of each task is dependent on the **measurement scale (e.g. meters, centimetres or millimetres)**"* | ★**우리 문장과 축자로 가장 가깝다.** 2018, 피인용 1만+ |
| **PINN** | Wang, Sankaran & Perdikaris (arXiv:2308.08468) — *"the scale of PDE residuals can still vastly differ from the scale of the latent solution function, leading to a considerable discrepancy in the scale of different loss terms"* / NDAWL-PINN (2025) — *"The idea of non-dimensionalization is to **eliminate the units of each loss term**, thus making the summation in the loss function more reasonable."* | ★**"손실항을 무차원화하라" 는 이미 표준 조언**이고 **정량 이득(50–100배, Wang/Teng/Perdikaris 2021 SISC)까지 보고돼 있다** |

⟹ ⛔**"손실함수가 [m] 와 [m²] 를 더한다" 를 *발견* 으로 팔 수 없다.**

### 4.3 ★해양파 분야는 어떤가 — **결함을 피하고 있다, 논문을 쓰지 않고**

- **가장 가까운 동종 목적함수** = Jiang et al. (2026) arXiv:2605.25074, 희소 자료로부터 방향 스펙트럼 CS 복원.
  목적함수: `min ½‖CΨs−y‖²_F + λ_e‖s‖₁ + λ_g Σ‖s_i‖₂ + μ(…)`. 축자 `[위임-VERIFIED]`:
  > "Both parameters λ_e and λ_g act on the shrinkage intensity … and **after normalization, they can be treated as dimensionless coefficients** selected empirically"

  ⟹ ★**그들은 신호를 먼저 정규화하고 계수를 무차원으로 만든다.** 우리가 안 한 그 한 줄이다.
- **WASSfast**: ⚠**서지 정정** — `Computers & Geosciences` **147**:104666 (2021) 이고
  ⛔**`Coastal Engineering` 이 아니다**(`BRIEF`/`RESULT` 계열 서술을 확인할 것). `[위임-VERIFIED via Crossref]`
  목적함수에 **L1 항이 없다**(제약/연제약 스펙트럼 적합). 배포 `settings.cfg` 의 `alpha=16.0` 은
  **가우시안 창의 형상 모수**이지 가산 벌점 가중치가 아니다. `[위임-VERIFIED, GitLab 원본]`
  ⚠**단 Wang 2025 의 서술(내가 축자 확인)은 *"uses a scalar α̇ to balance prior information and observed data"* 라고 적는다**
  ⟹ **둘이 같은 기호인지 다른 모수인지 미해결**. ⛔**어느 쪽도 단정하지 말 것.**
- ⟹ **분야에 L1 벌점을 쓰면서 λ 를 정규화하지 않는 관행이 있다는 증거를 찾지 못했다.**
  ⟹ ⛔***"일반적 결함이면 그게 가장 큰 novelty" 라는 가정의 전제가 성립하지 않는다.***
  현재 증거로 이것은 **우리 구현의 결함**이고, 분야는 이미 (조용히) 피하고 있다.

### 4.4 ★★★그런데 **우리 자신의 기제 설명이 숫자와 안 맞는다** (N-2, 내가 재도출)

`W9 PREREG §0` 의 기제: *"진실 지점에서 벌점이 목적함수의 86%"*.
**그 지분이 크다는 것과 해가 이동한다는 것은 다른 명제다.** 1차 계산을 해 보면:

성분 하나의 진폭 `A` 만 움직일 때, 데이터 항 `= ½(A−A₀)² + σ²` 이므로 `∂/∂A = (A−A₀)`.
벌점의 하위기울기는 `λ` 이므로 **벌점 최적점은 `Â = A₀ − λ`** 다.

| 양 | 값 | 출처 |
|---|---|---|
| λ_l1 (출하) | **1e-4** | `sequential_update.py:40` |
| ⟹ 성분당 진폭 변위 | ★**1e-4 m = 0.1 mm** | 위 유도 |
| 진실 진폭 (BF4·15성분) | ≈ **0.11 m** | 내 계산 (§4.5) |
| ⟹ **상대 수축** | ★**0.09%** | — |
| 벌점 최적점의 창-안 잔차 증가 | `½(1e-4)²·15 = 7.5e-8 m²` vs 잡음 바닥 **3.45e-5 m²** ⟹ **비 ≈ 1.000** | — |
| **측정된 비** | ★**1.5163** | `W9`(내가 재도출) |

⟹ ★★★**"벌점이 만든 새 최적점"으로는 1.52배를 설명할 수 없다**(성분이 서로 거의 직교하다는 가정 하에).
**남는 설명은 둘뿐이다**: (i) 관측 연산자가 심하게 **불량조건**이라 벌점 수축이 저곡률 방향으로 증폭된다,
(ii) **옵티마이저가 그 벌점 최적점에 도달하지 못한다**(Adam, **학습률 스케줄 없음** — `_build_optimizer` 에
스케줄러 **0개**, `MEASURED`).
⟹ ⛔**`W9`/`W10` 의 회계(47%/9%/16%/28%)는 이 둘을 가르지 않는다.** 절제는 *"벌점이 방아쇠다"* 를 사고,
*"벌점의 **차원**이 원인이다"* 를 **사지 않는다.**
★**그리고 이것은 저장소 자신의 `W3` 판정(*"벽은 벌점이 아니라 옵티마이저"*)·`S4` 라벨(`OPTIMIZER-FAILURE`)과 같은 방향이다.**

★**N-3 — 그리고 "수리" 자체가 잘못된 정규화자를 쓴다**:
`SCALE_INVARIANT_L1=True` 는 `Σ|w|` 에 **관측 RMS** 를 곱한다(`:463-468`). 그러나 §4.2 의 lasso 이론이 요구하는 것은
**잡음 척도** `σ` 다. 내가 잰 두 값:

| 양 | 값 | 어떻게 |
|---|---|---|
| 관측 RMS `RMS(z_obs)` | **0.2567 m** | 내가 HDF5 재집계 (`BF4_15comp_seed20`, 유효 15,051점) |
| 잡음 척도 `σ = √(진실 지점 평균제곱잔차)` | **5.87e-3 m** | `W9` `ORA-WPHI-0STEP` `avg_obs_loss` = **3.452e-5 m²** |
| ★**비** | ★**43.7배 과보정** | — |

⟹ 진실 지점에서 벌점이 차지하는 지분 (내 계산, `Σ|A|=1.648 m`):
**출하 82.6% → `SI` 수리 후 55.0% → 잡음 척도 정규화 시 2.7%**.
★**즉 "수리" 후에도 벌점이 목적함수의 절반 이상을 쥐고 있다** ⟹ **47% 만 사라진 것이 예측된다.**
⟹ ★**`W17` 의 세 팔(`STEPS10`·`LR10X`·`NOCLAMP`)에 이 요인이 없다.** **팔 하나(잡음 척도 정규화) 16런**이면 갈린다.

### 4.5 ★★★N-4 — **`Σ|A| ∝ √N` 이 레짐 비대칭을 만든다** (내가 잰 것, 새 축 ⓔ)

`gt_wave_field` 를 4개 에피소드에서 직접 읽었다 (`MEASURED`):

| 에피소드 | N | GT 장 RMS | `Σ|A| = √(2·RMS²·N)` |
|---|---|---|---|
| BF4_2comp_seed20 | 2 | 0.3000 | **0.600 m** |
| BF4_5comp_seed0 | 5 | 0.2999 | 0.948 m |
| BF4_15comp_seed20 | 15 | 0.3008 | **1.648 m** |
| BF4_30comp_seed20 | 30 | 0.2946 | **2.282 m** |

★**같은 BF4 에서 해상 에너지(장 RMS)는 N 과 무관하게 0.295–0.301 m 로 일정한데, `Σ|A|` 는 3.8배 벌어진다.**
(⚠`W9 PREREG` 의 *"`Σ|A|` 1.19–2.77 m"* 와 정합 — 내가 독립 유도했다.)

⟹ ★**데이터 항은 N-불변인데 L1 벌점만 `√N` 으로 자란다.**
진실 지점 벌점 지분: **N=2 → 63.5% · N=15 → 82.7% · N=30 → 86.9%**(내 계산).
⟹ ★★**"벽이 혼잡 전용이다" 라는 저장소의 최장수 관측에 대한, 처음으로 유도 가능한 후보 기제.**
⚠`ARGUED` — 지분 차이(63% vs 87%)가 판정 차이(널 vs 16/16)를 낳는지는 **재지 않았다**.
★**시험**: `λ_l1` 을 `√N` 으로 나누는 팔 하나. 혼잡·희소 각 16런 = **32런**. 사전등록 가능.

### 4.6 ⟹ **ⓒ 의 최종 판정**

| 하위 주장 | 판정 |
|---|---|
| *"λ 가 단위를 갖고 해상 척도를 탄다"* | ⛔**완전 선점**(BRT 2009 · sqrt-lasso 2011 · scaled lasso 2012 · glmnet · Kendall 2018) |
| *"손실항은 무차원화해야 한다"* | ⛔**완전 선점**(PINN, 2021–, 정량 이득까지) |
| *"이것이 파랑 재구성 분야의 일반 결함이다"* | ⛔**증거 없음** — 가장 가까운 논문(Jiang 2026)은 **정규화 후 무차원**이라 명시. **우리 구현의 결함**이다 |
| ★*"벌점 항이 물어낸 오차 몫을 **개입 절제로 귀속**했다"* | ✅**문헌에 없다** — 위임 조사가 어느 역문제 문헌에서도 *"오차의 N%가 벌점 탓"* 형태의 귀속을 찾지 못했다. PINN 은 **반대 부호**(고치니 50–100배)로만 보고한다 |
| ★*"그 귀속이 원인을 확정한다"* | ⛔**아니다** — §4.4. 절제는 **방아쇠**를 사고 **기제**를 못 산다 |

★**ⓒ 의 생존 문언**:
> *"희소 관측 위상분해 파랑 역문제에서, **벌점 항이 오차의 몇 %를 물어내는지를 개입 절제로 귀속**했다
> (혼잡 n=16, 진실-초기화 기준, 창-안 잔차비 1.516 → `λ_l1` 척도 수리 1.275 → 두 항 제거 1.146).
> 기제(λ 의 척도 의존성)는 통계학의 고전적 결과이며 본 연구가 발견한 것이 아니다.
> 본 연구가 더하는 것은 **그 고전적 결함이 이 문제에서 실제로 얼마를 물어내는지의 측정**이다."*
⛔**"차원 불일치가 원인이다" 라고 쓰지 마라** — §4.4 가 그 문장을 아직 사지 못했다.
---

## §5 ★★★정직한 novelty 문장 — **절마다 배치·n·효과크기를 붙였다**

⛔**규칙**: 근거 없는 절은 **지웠다**. 아래 문단에 남은 모든 절에는 괄호 안에 배치·표본·크기가 있다.

> 본 연구는 **희소·불균질 3차원 점군으로부터의 위상분해 파랑장 추정**에서, 파수를 격자에도 사전 지정에도
> 묶지 않고 **초기화 창에서 자료로부터 연속값으로 추정**하는 표현을 채택하고, 그 표현이 **어디까지 유효하며
> 그 경계를 무엇이 만드는지**를 개입 실험으로 확정한다.
>
> **(1)** 혼잡 다성분 해상에서 무기술 기준(제로필드) 대비 오차가 낮다는 **부호**는 서로 독립인 세 표본에서
> 재현된다 — out-of-sample 코퍼스 `S13` **56/64**, 추정기 시드 median-of-3 `S13b` **61/64 (−28.2%)**,
> 그리고 비선형(HOS M=3) 참값 위의 재검정 `HOSGATE` **혼잡 14/16 (−25.4%, n=16)**.
> *(내가 `HOSGATE` 192행을 재집계해 마지막 항을 독립 확인했다. 앞 둘은 `[인용]`.)*
> ⛔**크기는 주장하지 않는다** — 외부 기준선 대비 크기는 `S16` 이 `UNDECIDED` 로 남겼다.
>
> **(2)** 반대로 **희소 해상의 큰 상대 이득은 표현 아티팩트**다. 같은 성분 집합 위에서 참값 생성기의
> 비선형성만 토글하면 제안 방법의 절대오차만 **0.0195 → 0.0700 (3.59배)** 로 오르고,
> 두 기준선은 **`LETKF+` +0.5% · `ZERO` −0.0%** 로 무감하다 (`HOSGATE`, 희소 n=16, **내가 재집계**).
> ⟹ 이득의 귀속을 **개입으로** 가른 사례다.
>
> **(3)** 지지집합 **안**에서, 참 파수·진폭·위상을 초기값으로 주어도 해는 그 점에 머무르지 않는다
> (`W9`, 혼잡 n=16, 창-안 잔차 짝 중앙비 **1.5163**, **16/16**). 이 이탈은 정규화 계수에 **용량–반응**한다
> (`λ=0` → **8/16 널·1.009** `[인용 W7]`; `λ=1e-4` → **16/16·1.5163**; `λ=1e-3` → **16/16·3.4980**
> — 뒤 두 값은 **내가 `W9`·`W7` 원자료로 재도출**).
> ⟹ ★**predictable-zone 문헌이 정보 도달로 설명하는 경계 안쪽에도, 솔버가 만드는 잔여 경계가 있다.**
> ⛔**크기가 작다** — 두 벌점 항을 제거하면 비가 **1.146** 이 된다(`W10` `[인용]`) ⟹ 순수 잔여 ≈**15%**.
>
> **(4)** 그 이탈의 **회계**: `λ_l1` 척도 수리가 **47%**, 두 벌점 항 전체가 **72%** 를 없앤다
> (`W9`·`W10`, 각 n=16 `[인용]`). ⛔**단 이것은 방아쇠의 귀속이지 기제의 확정이 아니다** — 1차 계산상
> 벌점 최적점의 진폭 변위는 성분당 **0.1 mm(0.09%)** 에 불과해 관측된 1.52배를 낼 수 없다(§4.4).
>
> **(5)** 처방으로 **변수 투영**(Golub–Pereyra 1973)을 이 문제에 적용하면, 배포 라우터를 그대로 둔
> 배정 의도(ITT) 비교에서 혼잡 층이 **55/60, 짝 중앙 −25.28%, p=5.19e-12** 로 개선된다
> (`B`, 1,152런 `[인용]`). ⛔**방법의 신규성이 아니라 적용과 측정이 기여다.**

★**⛔이 문단이 명시적으로 **주장하지 않는** 것** (심사 대비, 전부 근거 있는 부재):

| | 왜 못 쓰는가 |
|---|---|
| *"순차 추적에 격자없는 표현을 썼다"* | 출하 `tracking_mode='w_phi'` 가 **`kx,ky` 를 얼린다**(§3ⓐ-3). 추적 형태는 Pascoal 2009·Desmars 2023 과 같다 |
| *"grid-free 시스템"* | **실측 경로는 `bank_mmse` 파수 격자**다(§3ⓐ-4) |
| *"실측을 했다"* | 논문 본문에 `Bergamasco`·`Acqua`·`Yellow` **각 0회**(내 grep) |
| *"이동 플랫폼 / 조우 예보"* | `C` 는 `exploratory`(라운드 1), 시뮬레이터에 heave/pitch/roll 없음 |
| *"최근접 경쟁자를 이겼다"* | ★**우리 Wang 2025 구현이 무기술이다** — 12셀 nRMSE **0.2462–0.2757 (중앙 0.2502 ≈ 제로필드 0.25)**, `WS<Wang` **7/12** (**내가 전건 재집계**) |
| *"손실함수 차원 불일치가 일반 결함이다"* | Jiang 2026 은 **정규화 후 무차원**이라 명시(§4.3) |
| *"음성 결과 원장이 기여다"* | 해양공학에 venue 없음(§3ⓓ) |

### 5.2 ★어디에 낼 수 있는가 — **구체적으로**

| 순위 | Venue | 무엇을 싣는가 | 지금 가능한가 | 막는 것 |
|---|---|---|---|---|
| ★**1** | **OCEANS 2026 (MTS/IEEE)** — 저장소의 원래 표적 | (1)+(3)+(5) 를 **시스템+진단** 논문으로 | ✅**지금 가능** | 없음(합성 전용이 허용되는 체급). ⚠제목에서 *"선박 탑재 LiDAR"* 를 빼야 한다 |
| ★**2** | **Applied Ocean Research** 또는 **J. Atmos. Ocean. Technol.(AMS JTECH)** — 계측·방법 노트 | ★**ⓒ 단독 논문**: *"How much does the regulariser cost? An intervention accounting for a sparse phase-resolved wave inverse problem"* | ⚠**거의 가능** | §4.4 를 닫아야 한다(= `W17` + **잡음 척도 팔 16런**). 그전에는 원인 문장을 못 쓴다 |
| **3** | **Ocean Engineering** (현 초고의 표적) | 전체 시스템 | ⛔**아니다** | 실측 0쪽 · 외부 기준선 크기 `UNDECIDED` · 2025–26 서지 **2편**(§N-7) |
| **4** | ★**Coastal Engineering** (= Wang 의 지면) | Wang 과의 정면 비교 | ⛔**아니다, 그러나 가장 값싼 잠금해제** | ★**공개 자료를 이미 갖고 있다**(`data/real_bergamasco/` 13개 `.nc`, Acqua Alta·Yellow Sea 포함 `[인용 R2-5 L-10]`). 막는 것은 **우리 Wang 포팅이 무기술**이라는 것 |
| **5** | ICBINB (NeurIPS 워크숍 계열) | ⓓ 원장 | ⛔ | 도메인 불일치 · 2026 확인 분기는 생물 |
| — | **학위논문** | ⓐ(좁힌 형태)+ⓑ+ⓒ+ⓓ 전부 | ✅ | `R2-4` 사다리 3칸(§N-7) |

⚠**시간 압박 (`[위임-VERIFIED]` 서지)**: **Collins et al. (2026)** *Reviews of Geophysics*, DOI 10.1029/2025RG000888,
*"Measuring Ocean Surface Waves"* 가 축자로 *"Remote sensing techniques based on radar and **LiDAR** … widely deployed from land, **ships**, aircraft…"* 라 적었다
⟹ **2026 년 기준 리뷰가 이미 선박 LiDAR 를 표준 범주로 다룬다.** 어떤 원고든 이것을 위치지어야 한다.
그리고 **Wang & Pan (2025)** arXiv:2504.13943 (EnKF-HOS, 자유부유 선박) · **Yao et al. (2025)** *Phys. Rev. Fluids* 10:094801 (선박 레이더, 정지/이동) ·
**DSGP-Net (2026)** *Ocean Eng.* 356:125086 (**같은 경쟁 그룹**) 가 모두 2025–26 에 있다.

---

## §6 ★내 정정·철회 그리고 주 세션의 오류

### 6.1 ★내 자신의 정정 (이 문서 안에서 내가 바꾼 것)

| # | 처음 생각 | 무엇이 바꿨나 | 최종 |
|---|---|---|---|
| **C-1** | ⓐ(격자없음)를 *"off-grid CS 로 완전 선점"* 으로 죽이려 했다 | 위임 조사가 **해양파 분야에서 연속 파수를 자료로 추정하는 재구성 논문을 못 찾았고**, Jiang 2026 이 *"격자를 유지하기로 선택"* 을 축자로 적는다 | ⚠**"분야 밖 선점 + 분야 안 공백"** 으로 정정. **죽는 진짜 이유는 문헌이 아니라 우리 코드**(추적기가 파수를 얼린다) |
| **C-2** | ⓒ 를 *"차원 불일치 = 최대 novelty"* 로 밀려 했다(과제 지시의 가설) | §4.2 3중 선점 + §4.3 분야가 이미 회피 + §4.4 **우리 기제 계산이 안 맞는다** | ⛔**기제는 죽고 측정만 산다** |
| **C-3** | `HOSGATE` 혼잡을 `SHIP<ZERO` 로 집계해 **16/16 −35.5%** 를 얻고 원장과 어긋난다고 생각했다 | 비교자가 `LETKF_PLUS` 임을 확인 | ✅**14/16 −25.4% 전건 재현.** 원장이 옳았다 |
| **C-4** | `avg_obs_loss` 로 `Σ|w|` 를 역산해 *"진폭이 5–14배 부풀었다"* 는 결론에 갈 뻔했다 | `avg_obs_loss ≈ all_resid² + 벌점` 임을 확인(0STEP 팔에서 비 0.97) ⟹ **역산이 창-안이 아니라 전-구간이라 무효** | ⛔**철회.** ★**대신 남는 처방**: `Σ|w|` 는 **어느 결과 파일에도 기록되지 않는다** — 벌점을 원인으로 지목한 세 배치가 **벌점 자신의 값을 한 번도 안 적었다** |

### 6.2 ⛔주 세션(오케스트레이터)의 오류 — 오늘 자백한 것 **외에**

| # | 오류 | 등급 | 근거 |
|---|---|---|---|
| **O-1** | ★★**`W9 RESULT §4`: *"`lambda_smooth` 가 여전히 `[m]` 단위"*** ⟹ **거짓**. `Σ(Δw)²` 는 `[m²]`. 따라서 `W10 RESULT §1` 의 *"두 벌점의 **단위**를 함께 고치면 72%"* 도 **틀린 라벨**이고, `BRIEF3 §5` 가 그 문언을 그대로 나른다 | ★★ | `MEASURED` 코드 |
| **O-2** | ★★★**회계가 *"벌점이 방아쇠"* 와 *"차원이 원인"* 을 가르지 않는다** — 1차 계산상 벌점 최적점 변위는 **0.1 mm** (§4.4) | ★★★ | `ARGUED`(측정 입력) |
| **O-3** | ★★**`SCALE_INVARIANT_L1` 이 신호 RMS 로 정규화한다 — 이론이 요구하는 것은 잡음 척도. 43.7배 과보정** ⟹ 수리 후에도 벌점이 목적함수의 55% (§4.4 N-3). **`W17` 세 팔에 이 요인이 없다** | ★★ | `MEASURED` 두 값 + 계산 |
| **O-4** | ★**논문 `:966` 의 자기비하가 저장소 기록과 어긋난다** — 표현 `2026-04-06` vs Wang 저장소 편입 `2026-05-08` (§2.3) | ★★ | `MEASURED` git |
| **O-5** | ★**`R2-4` 사다리가 오늘 수리로 닫히지 않았다** — `R-1` 기준의 `체크포인트` **0회**(`checkpoint`·`사전학습`도 0), `R-3` 의 `Bergamasco`/`Acqua`/`Yellow` **각 0회**, `R-4` 의 2025–26 서지 **2편**(요구 5) ⟹ ⛔**`FAIL-FOR-NOW` 는 안 움직인다** | ★★ | `MEASURED` grep |
| **O-6** | ⚠**초기화 벽시계가 세 문서에서 세 쌍의 수로 적혀 있다** — `W5 RESULT`: **29.7 → 14.5 s** · `BRIEF §4 #7`: **26.81 → 8.47 s (64/64)** · `thesis_main.tex:549`: **22.5 → 9.5 s (60/64)**. 팔이 다를 수 있으나 **어느 문서도 그렇게 말하지 않는다** ⟹ ★**단일출처화 대상**(저장소가 이 결함 부류에 다섯 번 물렸다고 `BRIEF.md` 가 적었다) | ★★ | `MEASURED` 세 파일 |
| **O-7** | ⚠**셸 분해 둘의 비교자가 다르다**(N-8) — `W15` = 내부 노브, `S16` = 외부 기준선. `BRIEF3 §3` 은 `W15` 를 *"저장소 최초 셸 분해"* 라 적는데 `S16` 이 **방법 vs 기준선 n=64** 로 먼저 했다 | ★ | `ARGUED` + `[인용 RESULT/MEMORY]` |
| **O-8** | ⚠**서지 오류: WASSfast 는 `Computers & Geosciences` 147:104666 이다**(`Coastal Engineering` 아님) | ★ | `[위임-VERIFIED Crossref]` |
| **O-9** | ⚠**`R2-5` 의 *"PDF 가 저장소에 두 달간 있었다(mtime)"* 는 근거가 안 된다** — `ref_paper/` 전체가 같은 초의 일괄 복사 타임스탬프다 | ★ | `MEASURED` `ls` |

---

## §7 ★다음에 할 것 — **가장 싼 것부터** (전부 사전등록 가능)

| # | 무엇 | 비용 | 무엇을 산다 |
|---|---|---|---|
| **P-1** | ★**`Σ|w|` 와 벌점 항의 값을 결과 파일에 기록하라** | ★**코드 한 줄** | 벌점을 원인으로 지목한 세 배치가 **벌점 자신의 값을 안 적었다**(C-4). `RMS_near` 와 같은 부류 |
| **P-2** | ★★**잡음 척도 정규화 팔** (`λ_eff = λ·σ̂`, `σ̂` = 잔차 RMS) | **16런** | §4.4 의 43.7배 과보정 가설. `W17` 세 팔이 못 가르는 요인 |
| **P-3** | ★**`λ_l1 / √N` 팔** (혼잡·희소 각 16) | **32런** | ⓔ — 레짐 비대칭의 첫 유도 가능한 기제(§4.5) |
| **P-4** | ★**Proximal/ISTA 로 L1 을 처리하는 팔**(하위기울기 Adam 대신) | **16런** | §4.4 의 (ii) — *"벌점 최적점에 도달 못 함"* 을 가른다 |
| **P-5** | ⛔**우리 Wang 2025 포팅을 고치거나 결과 파일을 폐기하라** | 진단 | 현재 12/12 무기술 = **비교에 쓸 수 없다**(N-6). 고치면 §5.2 의 **Venue 4 가 열린다** |
| **P-6** | **`R2-4` 사다리 3칸**(`체크포인트` 서술 · 실측 경계 절에 데이터셋 이름 · 2025–26 서지 5편) | **런 0회** | `FAIL-FOR-NOW` → `PASS-WITH-REVISION` |
| **P-7** | **Qi et al. (2018) *Wave Motion* 77:195–213 원문 확보** | 조회 | ★척추(경계 논지)의 **최중요 참고문헌인데 초록이 출판사 차단**이다. MIT DSpace 1721.1/111739 가 유력 경로 `[위임]` |
| **P-8** | **Tang–Bhaskar–Shah–Recht (2013) 인용** | 런 0회 | ★**로컬에 PDF 를 갖고 있으면서 인용 0회**다. 심사에서 반드시 나온다 |

---

## §8 **D-1 권고** (BRIEF §6 의무 절)

**라운드 1·2 의 결정을 유지한다 — 그러나 문언을 한 번 더 좁혀야 한다.**

현행:
> 범위 = *"희소·불균질 LiDAR 로부터의 실시간 파도장 추정"*(플랫폼 중립) ·
> 검증 = 정지 확증 + 이동 경계 · 실측 절은 고정 구조물 12 레코드로 따로.

★**내 수정 제안 두 곳** (둘 다 이번 라운드 측정에서 나온다):

1. ⛔**`LiDAR` 를 빼라.** 논문 본문에 실측이 0쪽이고 실측 12레코드는 **스테레오**다.
   그리고 **실측 경로는 제안 표현을 쓰지 않는다**(`bank_mmse` 격자). **오늘 바뀐 제목
   (*"희소 3차원 점군으로부터의 실시간 파랑장 추정"*)이 이미 옳다 — 그것을 범위 문장에도 반영하라.**
2. ⚠**`실시간 … 추정` 에서 `순차 추적`을 novelty 로 읽히게 두지 마라.**
   출하 추적기는 파수를 얼리므로 **Pascoal 2009 / Desmars 2023 과 같은 형태**다(§3ⓐ-3).
   ★**범위 문장은 "추정", 기여 문장은 "초기화 창의 연속 파수 추정" 으로 층을 나눌 것.**

★**권고 문언**:
> **범위** = *"희소·불균질 3차원 점군으로부터의 실시간 파랑장 추정"* (플랫폼·센서 중립)
> **기여의 자리** = **초기화 창**(연속 파수 추정 + 변수 투영 + 벌점 회계)
> **검증** = 정지 기하 확증 · 이동 기하는 `exploratory` 경계 · 실측은 **경계 서술 절**(제안 표현 미사용을 명시)

⚠**(b)(항주 선박 조우 예보)를 되살릴 수 있는 조건은 이번 라운드에 **더 나빠졌다** —
`Yao et al. (2025)` PRF 10:094801(선박 레이더, 정지/이동) · `DSGP-Net (2026)`(**같은 경쟁 그룹**, 이동선 명시 주장) ·
`Wang & Pan (2025)` arXiv:2504.13943(자유부유 선박) 이 전부 2025–26 에 있다.

---

## §9 한 줄 결론

> ★**Wang 이 지목한 것은 우리의 절반이고, 우리가 그 절반에서 그를 앞선 곳은 *초기화 창* 하나다.
> 격자없음은 추적기가 파수를 얼려서, 실측 경로가 격자여서 두 번 죽는다.
> 경계 측정은 문헌이 이미 그린 지대를 다시 그린 것이고, 새로운 것은 그 지대 *안*의 잔여 15% 뿐이다.
> 벌점 회계는 살아 있으나 그 이유(척도 의존성)는 2009–2012 통계학의 것이고, 게다가 우리 계산으로는 숫자가 맞지 않는다.
> ⟹ 남는 정직한 문장은 하나다 — "우리는 이 문제에서 벌점이 얼마를 물어내는지를 개입으로 처음 쟀다."
> 그것은 논문 한 편(AOR/JTECH)이지 학위논문 한 권이 아니다. 학위논문을 채우는 것은 novelty 가 아니라 §5 의 (1)–(5) 전부다.**

**끝.** 산출 = 이 파일 하나. 추정 런 **0회**.
