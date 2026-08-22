# R2-5 · 문헌 정밀조사 — **원문을 끝까지 읽었다**

> 라운드 2 · 2026-08-22 · 런 0회 · 산출 단일 파일
> 규율: `BRIEF.md` + `BRIEF2.md`. `MEASURED`/`ARGUED`/`UNVERIFIED` 문장별 분리. 열람범위=주장범위.

---

## 판정 요약 (맨 위)

| # | 항목 | 판정 | 근거 등급 |
|---|---|---|---|
| **L-1** | ★★★**Wang et al. 2025 원문 PDF 가 08-19 부터 저장소 안에 있었다** — `ref_paper/feng_wang_2025.pdf` (12쪽, 파일 mtime **2026-06-24**). 라운드 1의 A1·B2 는 초록도 못 봤다고 적었는데(`PARTIAL`), **전문이 로컬에 있었다.** 내가 12쪽 전부 추출해 읽었다 | ★★★ | `MEASURED` (전문 추출) |
| **L-2** | ★★★**Wang 2025 의 마지막 문단이 우리 방법을 "future work" 로 지명한다** — 축자: *"Future work could focus on exploring **appropriate entry frequencies as state vectors to reduce state dimensionality**, while preserving prior correlations through **spectral–spatial transformation in the mode of direct sparse assimilation**."* ⟹ **격자 대신 소수 스펙트럼 성분을 상태로 두고 희소점을 직접 동화** = WaveSplat 의 정의 그 자체 | ★★★ | `VERIFIED` 축자 |
| **L-3** | ★★★**저장소에 이미 Wang 2025 기준선 구현과 결과 파일이 있고, 그 팔은 무기술이다** — `wavesplat/baselines/etkf_wang2025.py` + `results/wang2025_comparison.json`. 내가 재집계: `ETKF-Wang2025` nRMSE 가 **12/12 셀에서 0.2462–0.2757 (중앙 0.2502)** = **제로필드 해석값 0.25 에 못 박혀 있다.** ⟹ ⛔**이 파일로 "최근접 경쟁자를 이겼다" 를 쓸 수 없다** | ★★★ | `MEASURED` 재집계 |
| **L-4** | ★★**DSGP-Net 은 LiDAR 도 격자없음도 아니다** — 스테레오 카메라 + **영상좌표계 dense height map(CNN)** = 완전한 격자 표현. 학습자료는 **Godot 합성**, 검증은 **Acqua Alta·Yellow Sea 실측**. **단일 프레임 공간 보간**이라 **예보 지평이 없다** | ★★ | 서지 `VERIFIED` / 본문 `UNVERIFIED-축자` (SD 403·CAPTCHA) |
| **L-5** | ★★**Wang 그룹의 2025 현장 자료는 "이동선" 이 아니다** — 축자: *"the stereo camera is deployed on the deck of **a vessel on position keeping**"*, *"Throughout the measurement process, **the vessel remained stable on the sea surface**."* ⟹ B2-8 의 *"같은 그룹이 이동선까지"* 는 **2026 DSGP-Net 한정**이고 2025 는 **정지 기하**다 | ★★ | `VERIFIED` 축자 |
| **L-6** | ★★**Wang 2025 공개 코드는 핵심이 빠져 있다** — GitHub `fengHEU/Wave-ETKF` 실재(GPL-3.0)하나 README 가 **칼만필터 본체는 저작권 등록 대기로 비공개**라고 적는다. **재현 불가** | ★★ | `VERIFIED` (리포 열람) |
| **L-7** | ★★★**AOR 173:105173 은 우리 문제가 아니다** — *"Bayesian Neural Fields"* 는 **위성고도계+부이로 유의파고(Hs) 장을 재구성**하는 논문이다(북티레니아해, 9개 위성, 한 달치 장을 **52 s**). ⟹ A1 의 *"grid-free 파랑장 재구성이 우리 표적 저널에 실렸다"* 는 **서지는 맞고 위협 판정은 과대**다. **위상분해 축에서 겹침 0** | ★★★ | `VERIFIED` 초록 전문 |
| **L-8** | ★**A3 의 Kim 2023 판정을 1차출처로 독립 재확인 — 전건 일치** | ★★ | `VERIFIED` (Zenodo PDF 17쪽 내가 직접 내려받아 추출) |
| **L-9** | ⛔**원장 오류 발견: "Desmars JFM 975 A8 = 합성 전용" 은 거짓** — 초록 축자: *"The suitability of the algorithms for severe wave conditions in finite depth and using non-perfect observations is assessed through **wave tank experiments**."* | ★★ | `VERIFIED` 축자 (로컬 PDF) |
| **L-10** | ★★★**우리는 경쟁자가 쓴 바로 그 공개 데이터를 이미 디스크에 갖고 있다** — `data/real_bergamasco/` 에 **AA_2014·AA_2015·YS_2017 포함 13개 `.nc`**. Wang 2025 는 `AA 03`, DSGP-Net 은 **Acqua Alta + Yellow Sea** 를 썼다. **정면 비교가 자료 취득 없이 가능하다** | ★★★ | `MEASURED` (파일 열거) |
| **L-11** | ★**남은 틈은 실재한다. 단 좁고, 세 조건의 교집합이다** — §7 | — | `ARGUED` |

---

## §0 열람 범위 (= 주장 범위)

### 직접 연 1차 출처 (전문 또는 초록 전문)

| 자료 | 어떻게 | 범위 |
|---|---|---|
| **Wang et al. (2025)** *Coastal Eng.* 197:104690 | 로컬 PDF `wavesplat/ref_paper/feng_wang_2025.pdf` → `pypdf` 로 **12쪽 전문 텍스트 추출**(61,010자) | ★**전문** (초록·서론·방법·표 2/3·현장절·결론·CRediT) |
| **Kim et al. (2023)** *Ocean Eng.* 276:114212 | Zenodo `records/7891109` (CC-BY) PDF **내가 직접 `curl` 로 내려받아** `pypdf` 추출, **17쪽**(76,027자) | ★**전문** (초록·Eq.(20)–(22)·§4 prediction zone·표 2·LIDAR 절) |
| **Desmars et al. (2023)** *JFM* 975 A8 | 로컬 PDF `ref_paper/desmars2023_jfm.pdf` (40쪽) | **1–2쪽만**(서지·초록 전문). 본문 미열람 |
| **Liu, Tsao, Manuel & Heo (2026)** *AOR* 173:105173 | Crossref `works/DOI` + OpenAlex `abstract_inverted_index` 복원 | ★**초록 전문 + 서지 + 소속** |
| **Golub & Pereyra (1973/2003)** | Crossref `works/DOI` 직조회 (2003) + 서지 검색 (1973) | **서지만** |
| **Wang, Wang & Zhu (2026)** DSGP-Net *Ocean Eng.* 356:125086 | Crossref + OpenAlex + Semantic Scholar **서지 3중 조회 성공**. 본문·초록은 **ScienceDirect 가 HTTP 403 / CAPTCHA 로 거부**(직접·`r.jina.ai`·`colab.ws`·`scilit`·OUCI **5경로 전부 실패**) | ⚠**서지 `VERIFIED` / 초록 `UNVERIFIED-축자`** |
| **`fengHEU/Wave-ETKF`** GitHub | WebFetch (README) | 리포 존재·라이선스·공개범위 |
| **Liu et al. (2025)** arXiv 2512.06024 | arXiv 초록 | 초록만 |

### 직접 연 저장소 파일 (코드·자료)

`ref_paper/` 목록 · `wavesplat/baselines/etkf_wang2025.py`(1–45행) · `results/wang2025_comparison.json`(**전건 재집계**) · `data/real_bergamasco/` 파일 열거 · `memory/reference_bergamasco_stereo_dataset.md` · `tmp/STATUS_2026-08-21/STATUS.md`(1–70행) · `BRIEF.md`·`BRIEF2.md`·`RESULT.md`(140–175행) · `A1_ocean_reviewer.md`(325–352행 + DSGP grep) · `A3_control_reviewer.md`(grep) · `B2_external_examiner.md`(grep) · `reference_citation_bans.md`(선두 + 문헌 grep).

### ⛔안 연 것 (판정 시 `[인용]`)

- **DSGP-Net 본문·초록 축자** — 5경로 전부 차단.
- **Yao et al. 2025 PRF 10:094801**, **Sensors 26(3):993**, **Çelik 2026**, **Ehlers 2508.03315 전문** — A1 §5 가 이미 검증했고 내 우선순위 밖.
- **Wang 2025 의 수식 (1)–(36)** — 읽었으나 재도출은 안 했다.
- **Desmars 2023 본문**, **Kim 2024 (Ocean Eng. 313:119316)**.
- 라운드 1의 A2·B1·B3·C1·C2·D1·D2 **전문** (문헌 grep 만).

---

## §1 08-14 / 08-19 / **라운드 1** 대비 무엇이 바뀌었나

| | 그때 | ★지금 |
|---|---|---|
| Wang 2025 | 08-19·라운드 1: *"가장 가까운 경쟁자"* — **서지만 확인**, 초록 `PARTIAL` | ★★★**전문을 읽었다. 저장소에 PDF 가 두 달간 있었다.** 결론이 **양쪽으로** 바뀐다: 위협은 줄고(격자·정지·코드 반쪽), 압박은 커진다(**그들 future work 가 우리 방법**) |
| 우리 vs Wang 비교 | *"우리가 못한 것(실측 검증)을 그들이 했다"* (B2-8) | ★★★**우리는 이미 Wang 기준선을 구현·실행했고 그 결과가 무기술이다.** "안 했다"가 아니라 **"했는데 못 쓴다"** |
| DSGP-Net | A1: *"이동선을 명시 주장, 우리와 완전히 같지는 않다"* | ★**표현축이 추가로 죽는다** — 격자(영상좌표 dense map) + 스테레오 + 단일프레임. **"격자없음"은 아직 우리 것** |
| AOR BayesNF | A1: ⛔*"grid-free 를 차별점으로 쓰기 전에 반드시 읽어야 한다"* | ★★**읽었다. 위협 아님** — Hs/위성고도계/월단위. **경보 하향** |
| Kim 2023 `1.8Tp` | A3 가 Zenodo 로 검증 | ✅**내가 같은 1차출처를 독립 재취득해 전건 재확인**(축자·Eq.(21)·표 2 수치·"turbine" 문구 전부 일치) |
| Desmars 2023 | MEMORY: *"1-D · 정지 센서 · **합성 전용**"* | ⛔**"합성 전용" 반증** — 초록이 wave tank experiments 를 명시 |
| W7·W8 | — | **내 판정을 움직이지 않는다**(§8) |

---

## §2 ★★★Wang et al. (2025) — **전문 정독** `VERIFIED`

**서지** (Crossref + OpenAlex + PDF 표제부 3중 일치):
Feng Wang ᵃ, Qidan Zhu ᵃ\*, Chengtao Cai ᵃᵇ, Xiaoyu Wang ᵃ, Renjie Qiao ᵃ,
*"From points to waves: Fast ocean wave spatial–temporal fields estimation using ensemble transform Kalman filter with optical measurement"*, **Coastal Engineering 197 (2025) 104690**.
소속: **Harbin Engineering University**. Received 8 Aug 2024 / Accepted 12 Dec 2024 / Available online **18 Dec 2024**.

### 2.1 초록 축자 (전문)

> "Accurate spatial–temporal wave measurement is vital for ocean engineering applications. Although stereo vision shows great potential in this field, performing dense reconstruction requires processing vast amounts of pixel data, which reduces the efficiency of stereo image matching and subsequent point cloud processing. Recently, the paradigm of fusing sparse 3D points with predictions has emerged as a promising solution that balances accuracy and efficiency, yet requires an optimization framework capable of handling both state estimation and robust outlier filtering. Therefore, this study proposes a Kalman filter (KF)-based method for ocean wave field estimation, aiming to improve efficiency through recursion and to remove outliers and interpolation errors in measurements. **The method leverages linear gravity wave dispersion relations for prediction, with sparse 3D points interpolated to a uniform grid as measurements.** … Both CPU and GPU implementations were evaluated on published and field-collected datasets, demonstrating superior performance in efficiency, accuracy, and robustness compared to existing methods under the same paradigm."

### 2.2 ★우리에게 유리한 것 — **전부 축자 확인**

| 축 | 축자 | 우리 위치 |
|---|---|---|
| **표현** | *"sparse 3D points **interpolated to a uniform grid** as measurements"* · *"the proposed method regards **the entire interpolated grid as the measurements**"* · 상태 차원 *"fixed at **16384 × 1** with 64 sub-states"*(=128×128) | ★**완전한 격자.** 우리는 성분 매개 |
| **센서** | 스테레오(binocular). 현장은 **ZED2i**, *"a maximum effective measurement distance of approximately **20 m**"* | ★**개구가 20 m 다.** LiDAR(Falcon)는 자릿수 위 |
| **플랫폼** | *"the stereo camera is deployed on the deck of **a vessel on position keeping**"* / *"the vessel **remained stable** on the sea surface"* | ★**정지 기하.** "이동선"은 그들 2025 에 없다 |
| **예보** | 칼만 예측단계 = **한 스텝**. 논문 어디에도 예보 지평·prediction zone 없음. 제목·결론 모두 *"estimation"·"reconstruction"* | ★**예보 축이 비어 있다** |
| **속도** | 표 3 단계별 (dense ratio 0.100 / 0.005): Image Preprocessing **76/76 ms**, Point Cloud Generation **185/86**, Gridding Interpolation **69/6**, State Ensemble Prediction **18/18**, Fuzzy Logic **32/32**, ETKF State Analysis **45/42** ⟹ ★**합계 425 ms / 260 ms per frame** (i5-10400 + GTX 2060, 축자 *"Nvidia GeForce GTX 2060"*). WASSfast 대비 *"3–4 times faster on CPU and 2 times faster on GPU"* | ⚠**비교 시 CPU/GPU 세대와 프로세스 경계를 반드시 적을 것** (저장소가 여기서 두 번 물렸다) |
| **코드** | GitHub `fengHEU/Wave-ETKF`, GPL-3.0. ★README: **칼만필터 본체는 저작권 등록 대기로 비공개**, WASSFAST 유래 모듈만 공개 | ★**그들도 완전 재현 불가** |

### 2.3 ★★★그들이 스스로 남긴 틈 — **결론 마지막 문장 축자**

> "**Future work could focus on exploring appropriate entry frequencies as state vectors to reduce state dimensionality, while preserving prior correlations through spectral–spatial transformation in the mode of direct sparse assimilation.**"

`ARGUED` — 이 문장이 서술하는 것을 항목별로 옮기면:
- *"entry frequencies as state vectors"* = **격자 대신 소수 스펙트럼 성분을 상태로**,
- *"reduce state dimensionality"* = 16,384 → O(10),
- *"direct sparse assimilation"* = **보간 격자를 거치지 않고 희소점을 직접**.

⟹ **WaveSplat 의 설계 결정 세 개가 그대로다.** 이것은 두 방향으로 작동한다:
- ✅**유리** — 우리 방향이 최근접 경쟁자가 지목한 **명시적 공백**이다. 논문 서론에 **인용 가능한 한 문장**이 생겼다.
- ⛔**불리** — 2024-12 에 **공개적으로 제안된** 아이디어다. *"우리가 처음 생각했다"* 는 못 쓴다. 살 수 있는 것은 **"그들이 지목한 공백을 우리가 실행하고 측정했다"** 뿐이다.

### 2.4 그들이 이미 한 것 — 우리가 못 한 것

- 검증 자료: *"a public stereo image dataset of ocean waves (**Guimarães et al., 2020**)"* = `AA 03` + 자체 현장(ZED2i).
- 즉 ★**우리 `data/real_bergamasco/` 와 동일 출처**(§6).

---

## §3 ★★DSGP-Net (2026) — 서지 `VERIFIED` / 본문 `UNVERIFIED-축자`

**서지 3중 조회 일치** (Crossref `works/DOI` · OpenAlex · Semantic Scholar `CorpusId 287064833`):
**Feng Wang, Xiaoyu Wang, Qidan Zhu** (전원 Harbin Engineering University),
*"DSGP-Net: A dual-space guided propagation network for ocean wave height estimation from sparse stereo measurements"*,
***Ocean Engineering* 356, 125086**, DOI `10.1016/j.oceaneng.2026.125086`.
⚠**날짜가 출처마다 다르다**: OpenAlex `publication_date = 2026-03-31`, Crossref/S2 `2026-05` — 온라인 3월 / 지면 5월로 읽는 것이 자연스럽다 `ARGUED`. 참고문헌 **26편**.

★★**저자 ID 가 Wang 2025 와 겹친다** (S2 `authorId` 2323135823 / 2336274103 / 2336286294 세 명 모두) ⟹ **같은 그룹의 후속작이 `VERIFIED` 로 확정**된다. A1 의 판정과 일치.

### 3.1 *"uniquely enables … moving vessels"* — 실재하는가

⚠**나는 출판사 페이지를 열지 못했다** (ScienceDirect 403 · CAPTCHA; `r.jina.ai`·`colab.ws`·`scilit`·OUCI·Unpaywall 우회 **전부 실패**). 대신 **독립적으로 던진 4회의 검색이 같은 문장을 두 번 동일하게 반환**했다:

> "DSGP-Net achieves **14–23% RMSE reduction over state-of-the-art methods on fixed platforms and uniquely enables accurate reconstruction on moving vessels**, achieving real-time performance."

**등급: `LIKELY-VERBATIM (UNVERIFIED)`.** ⟹ ★**논문에 인용할 때는 반드시 사서 확인해야 한다.** *"A1 이 찾은 문구"* 는 **실재할 가능성이 높다는 것까지가 내가 산 것**이다. ⛔**축자 인용으로 쓰지 마라.**

### 3.2 대신 확인된 것 (검색이 반복 반환한 실질 내용, `UNVERIFIED-축자` / 내용 `ARGUED`)

| 축 | 내용 | 우리와의 관계 |
|---|---|---|
| **문제** | *"single-frame sparse-to-dense reconstruction"* — 한 프레임의 **공간 보간** | ★**예보 지평 없음** — Wang 2025 와 같은 한계 |
| **센서** | **스테레오 카메라** + 카메라 영상을 contextual guidance 로 | ⛔LiDAR 아님 |
| **표현** | 1단 SGP-Net = **영상좌표계 dense height map**, 2단 DSGP-Net = image space ↔ world space **cross-space fusion** | ★★**격자다.** CNN dense map. **"격자없음"은 여전히 우리 것** |
| **학습자료** | **Godot Engine 물리기반 파랑 시뮬레이션**, 200 batch × 64 frame = **12,800 샘플** | ⚠**합성으로 학습** — 우리의 학습-분포 취약성과 **같은 구조** |
| **검증** | **실측**: Acqua Alta **−14.4%**, Yellow Sea **−23.3%** (vs WASSfast-CNN). 추가로 fixed platform 정량·정성, 스펙트럼, ablation, **moving vessel platform 실험** | ★**실측 검증을 했다** — 우리가 못 한 칸 |
| **이동선 실험이 실측인가 합성인가** | ⚠**확인 실패** | `UNVERIFIED` — ★**이 한 칸이 "이동선 축이 점유됐는가"를 가른다.** 사서 확인할 것 |
| **동기 축자(반복 반환)** | *"sparse feature tracking achieves real-time performance but produces measurements covering only **5–20% of the surface**"* | ⚠우리 희소/혼잡 레짐 정의와 대조할 것 |

---

## §4 ★★★AOR 173:105173 — **위협 판정 하향** `VERIFIED`

**서지** (Crossref 직조회): Ding Peng Liu(National Taiwan Univ.), Wen-Huai Tsao(UNC Wilmington), Lance Manuel(UT Austin), **Taemin Heo**(홍익대, 교신), *"Probabilistic wave field reconstruction via Bayesian Neural Fields"*, ***Applied Ocean Research* 173, 105173** (2026-08; OpenAlex `2026-07-17`).

**초록 전문** (OpenAlex `abstract_inverted_index` 복원) 요지 — 축자 핵심구:

> "…reconstructs continuous, high-resolution **significant wave height** fields with rigorous uncertainty quantification, **using only spatiotemporal coordinates as inputs**. The framework trained on heterogeneous constellation of **nine satellite missions** covering the **North Tyrrhenian Sea** … On independent buoy record, BayesNF attains mean absolute error of **0.35 m** root mean square error (RMSE) of **0.50 m**, reducing RMSE by approximately 26% relative to operational CMEMS L4 product … The framework generates **full-month**, high-resolution regional wave field **in under a minute (roughly 52 s on a single TPU v6e)**."

### ★판정

| 축 | BayesNF | 우리 |
|---|---|---|
| 대상량 | **유의파고 Hs** (통계량) | **위상분해 η(x,y,t)** |
| 센서 | 위성고도계 9종 + 부이 | 선박탑재 LiDAR |
| 시공간 규모 | 지역해 · **한 달** | 수십~수백 m · **초** |
| 입력 | *"only spatiotemporal coordinates"* (좌표→값 MLP) | 희소 3D 점 + 분산관계 |
| 물리 | **없음** (*"with no meteorological forcing"* = 순수 자료구동) | 분산관계 하드제약 |

⟹ ★**겹치는 것은 "neural **field** = 좌표 기반 연속 표현" 이라는 단어 하나뿐이다.**
- ✅**"위상분해 격자없음" 은 이 논문에 점유되지 않았다.**
- ⛔**단 A1 의 경고의 절반은 남는다**: *"격자없음(grid-free) 자체"* 는 이제 **우리 표적 저널의 일상어**다. ⟹ ★**"우리는 격자를 안 쓴다" 를 novelty 문장으로 세우면 리뷰어가 이 논문을 든다.** 살 수 있는 문장은 **"격자없음"이 아니라 "격자없음 × 분산관계 하드제약 × 순차 추적"의 교집합**이다.
- ⚠**Neural field 와 우리 spectral Gaussian 의 차이도 적어야 한다**: BayesNF 는 **좌표→값 신경망**(물리 없음, 파라미터 O(10⁵–10⁶)), 우리는 **파동 성분 O(10)** 의 해석적 매개화. ★**"해석 가능한 소수 파라미터"가 진짜 차이**이고 그건 §2.3 이 지목한 공백과 **같은 문장**이다 `ARGUED`.

---

## §5 ★Kim et al. (2023) — **A3 판정 독립 검증: 전건 일치** `VERIFIED`

**나는 A3 문서를 읽기 전에 원문을 먼저 취득했다** (과제 지시대로). 경로: ScienceDirect 403 → HAL Anubis 차단 → **HAL API 로 서지 확보** → Zenodo `records/7891109` (CC-BY) 확인 → **`curl` 로 PDF 직접 내려받아 17쪽 추출**. 그 다음에 A3 를 열어 대조했다.

### 5.1 A3 의 축자 인용 — **한 글자도 어긋나지 않는다**

내 추출본 1006–1013행:

> "The real-time system generally has a total numerical time 𝑇ₜₒₜₐₗ, composed of times for nowcast 𝑇ₙₒwcₐₛₜ and forecasts 𝑇fₒᵣₑcₐₛₜ, which must be smaller than the length of the practical prediction zone in time **at the location of the turbine** (i.e., 𝑇ₜₒₜₐₗ < 𝑡′ₘₐₓ = 𝑡ₘₐₓ − 𝑡ᵣ, **see Eq. (21)**). The length of the practical prediction zone from the latest time of nowcast 𝑡ᵣ until WG21 falls outside the prediction zone 𝑡ₘₐₓ **is about 1.8𝑇𝑝 = 21.6 s at full scale.** … **All the computations were run on an Intel(R) Core(TM) i7-8700 CPU 3.20 GHz.**"

✅**`1.8·Tp` 는 원문 축자다.** A3 도 E 배치도 옳았다.

### 5.2 ★A3 의 핵심 주장 — **Eq.(21) 을 내 눈으로 보고 확인**

내 추출본 810–824행:

> 𝑡′ₘₐₓ = min[ (𝑥 − 𝑥_D)/(𝑐_g,max) · cos(𝜃_B/2) , (𝑥 − 𝑥_B + 𝑦′_B)/(𝑐_g,max) · cos 45° ]

⟹ ✅**`t′_max` 는 오로지 (평가위치 𝑥, 센서배열 모서리 𝑥_D·𝑥_B·𝑦′_B, 최대군속도 𝑐_g,max, 방향폭 𝜃_B) 의 함수다.**
**`1.8` 이라는 수는 이 식에 그들의 WG21 기하를 넣은 결과값이지 보편 규칙이 아니다.** A3 판정 **지지**.

### 5.3 추가로 내가 확인한 것 (A3 가 안 적은 것)

| 항목 | 축자 |
|---|---|
| 검증 종류 | *"we outline a **tank-scale experimental campaign** conducted to mimic the measurements of a LIDAR in a real configuration"* ⟹ ★**수조. 실해역 아님. 그리고 실제 LiDAR 도 아니다** — *"a network of **wave gauges** mimicking the measurements of a LIDAR"* |
| 표 2 (Case A @ WG21, 예보지평 ≈21.6 s) | LWT `T_total` **1.7 s** · ICWM Previous **95.7** / Simplified **7.6** · LWT-CDR Previous **52.1** / Simplified **3.0**. `T_forecast` 는 전부 **< 0.01 s** |
| 관심점 | *"at the location of **the turbine**"*, *"a prediction horizon suitable for the control strategies of **the wind turbines**"* — 센서는 회전 구조물 탑재, **관심점은 고정** |

⟹ ★**A3 의 §4.3 재도출(우리 개구 → 0.7–5.7 s)의 전제는 원문에 있다.** 나는 그 재도출의 **입력(우리 개구·c_g)** 은 검증하지 않았다 — 그건 A3 의 `MEASURED` 이고 내 범위 밖이다 `[인용]`.

⚠**한 가지 A3 가 덜 말한 유리한 점**: Kim 의 `T_forecast < 0.01 s` 는 **나우캐스트(초기화)가 예산의 전부**라는 뜻이다. 우리 구조도 같다(초기화 8.47 s + 프레임 8.90 ms, `STATUS §1` 7행). ⟹ ★**"초기화가 예산을 먹는다"는 우리만의 병이 아니라 이 문제 계열의 공통 구조**이고, Kim 은 그것을 **95.7 → 7.6 s (12.6×)** 로 줄여 통과시켰다. 우리는 **26.81 → 8.47 s (3.19×)** 다 `[인용: STATUS §1 7행]`. ★**같은 종류의 기여이고, 같은 자로 잴 수 있다.**

---

## §6 ★★★저장소 안에 이미 있던 것 — **아무도 안 봤다**

### 6.1 Wang 2025 기준선이 **구현·실행돼 있고, 무기술이다** `MEASURED`

`wavesplat/baselines/etkf_wang2025.py` 독스트링 축자:

> "ETKF baseline **faithful to Wang et al. 2025** (Coastal Eng 197, 104690). … 1. Interpolated measurements: sparse 3D points -> Delaunay interp -> grid (H=I) 2. Fuzzy logic: adaptive R … 3. Ensemble size: 300 (optimal per Table 2) 4. Sub-state localization"

⟹ ✅**구현은 원문에 충실하다** (내가 §2 에서 읽은 네 요소와 일치. 앙상블 300 = 그들 표 2 최적값도 맞다).

**결과 파일 `results/wang2025_comparison.json` 전건 재집계** (파일 mtime 2026-06-24, 12 에피소드):

| 에피소드 | WaveSplat | ETKF(ours,Ne=100) | **ETKF-Wang2025** |
|---|---|---|---|
| BF2_1comp | 0.3150 | 0.4797 | **0.2534** |
| BF3_1comp | 0.2174 | 0.3357 | **0.2501** |
| BF3_2comp uniform | 0.0524 | 0.3793 | **0.2497** |
| BF3_2comp slope | 0.0607 | 0.3168 | **0.2480** |
| BF3_5comp uniform | 0.1675 | 0.3790 | **0.2504** |
| BF3_5comp slope | 0.1516 | 0.3331 | **0.2462** |
| BF4_1comp | 0.1706 | 0.3529 | **0.2499** |
| BF4_5comp | 0.1996 | 0.3781 | **0.2519** |
| BF4_15comp | 0.3018 | 0.4022 | **0.2490** |
| BF4_30comp | 0.3142 | 0.4026 | **0.2552** |
| BF5_15comp | 0.2966 | 0.3882 | **0.2518** |
| BF5_30comp | 0.2897 | 0.4368 | **0.2757** |

★★★**`ETKF-Wang2025` 열의 폭이 0.0295 (0.2462–0.2757), 중앙 0.2502.**
`ARGUED` — nRMSE = RMSE/Hs 이고 선형해에서 RMS(η) = Hs/4 이므로 **제로필드 예측의 nRMSE 는 해석적으로 정확히 0.25** 다. 이 팔은 **12/12 셀에서 그 값에 못 박혀 있다** = **입력을 전혀 쓰지 않은 것과 구별되지 않는다.**
(같은 진단이 원장에 이미 있다 — *"`results/ablation_bf45.json` … Full 팔 중앙 0.2757 이 제로필드 ≈0.25 를 못 이긴다"*.)

⟹ **⛔인용 금지 3건 (내가 새로 제안):**
1. ⛔**`results/wang2025_comparison.json` 로 "최근접 경쟁자를 이겼다" 금지** — 기준선이 무기술이다. 게다가 WaveSplat 이 그 무기술 팔을 이긴 것도 **7/12** 뿐이다.
2. ⛔**같은 파일의 `avg_inference_ms` 금지** — WaveSplat 열이 **13.37–14.10 ms** 로 ⛔금지된 *"13.4 ms"* 계보이고, Wang 열(중앙 **623 ms**)은 **프로세스 간 비교**다(저장소가 이미 두 번 물린 결함).
3. ⛔**`scripts/generate_labmeeting_2026_05_13.py:552` 의 발표 문구 금지** — *"우리 `etkf_wang2025.py` 가 이 논문을 충실히 재현"* 이라 적는데, **구현은 충실해도 그 팔이 아무것도 추정하지 않는다.** 충실성 주장은 **결과로 반증**된다.

⚠**정직한 문장** = ***"우리는 Wang et al. 2025 의 재구현을 시도했고, 그 팔이 우리 합성 우주에서 제로필드와 구별되지 않았다. 원인 규명 전까지 이 비교는 어느 방향으로도 인용할 수 없다."***

### 6.2 ★★★경쟁자의 검증 자료를 우리가 이미 갖고 있다 `MEASURED`

`data/real_bergamasco/` 파일 열거 (내가 직접 `find`):

```
AA_2014/ 1  AA_2015/ 2  BS_2011/ 4  BS_2013/ 3  LJ_2018/ 2  YS_2017/ 1   = .nc 13개
```
(원논문 기준 **12 레코드** — LJ Grid1/Grid2 가 동일 시각. `STATUS` 08-22 정정과 정합.)

출처는 **Guimarães, P.V. et al. (2020), *Sci. Data* 7:145** — ★**Wang 2025 가 축자로 쓴 바로 그 데이터셋**(*"a public stereo image dataset of ocean waves (Guimarães et al., 2020)"*, sequence `AA 03`)이고, ★**DSGP-Net 이 숫자를 낸 두 사이트(Acqua Alta, Yellow Sea)가 둘 다 우리 디스크에 있다.**

⟹ ★★★**"실측 검증이 없다"(B2-1) 와 "경쟁자가 우리 데이터로 검증했다"(B2-8) 는 사실 같은 문제의 두 얼굴이고, 자료 취득이 필요 없다.** 막는 것은 자료가 아니라 **파이프라인**(스테레오 점군 → 우리 관측연산자)과 **정규화 규약**(⚠MEMORY: 합성 nRMSE 분모 `Hs`, 실측 분모 `σ`, 비 3.045–4.278 — **Hs 단위로 환산해야 비교 가능**).

---

## §7 ★★★경쟁자 대비 우리 위치표

| | **문제** | **센서** | **표현** | **검증** | **실시간** | **이동 플랫폼** | **코드공개** |
|---|---|---|---|---|---|---|---|
| **Wang et al. 2025** *Coastal Eng.* 197:104690 | 위상분해 시공간 장 **추정**(예보 없음) | 스테레오(ZED2i, 유효 **~20 m**) | ★**균일격자** 128²/256² (상태 16,384) | ★**합성 없음 — 공개 실측(AA 03) + 자체 현장** | ⚠**260–425 ms/frame** (i5-10400+GTX2060) | ⛔**정지**(*"vessel on position keeping"*) | ⚠**부분** — GPL-3.0 리포 있으나 **KF 본체 비공개** |
| **DSGP-Net 2026** *Ocean Eng.* 356:125086 | **단일프레임 sparse→dense 높이 보간**(예보 없음) | 스테레오 + 카메라 영상 guidance | ★**격자**(영상좌표 dense map → world) | **학습 = Godot 합성 / 검증 = 실측 AA·YS** | *"real-time"* 주장 `UNVERIFIED-축자` | ★**주장 있음** (*"uniquely enables … moving vessels"*, `LIKELY-VERBATIM`) ⚠실측인지 `UNVERIFIED` | ⚠미확인 |
| **Kim et al. 2023** *Ocean Eng.* 276:114212 | ★**위상분해 나우캐스트 + 예보**(prediction zone) | **모의 LiDAR** = 수조 wave gauge 망 | 파동성분(격자없음) | ⛔**수조 실험**(실해역 아님) | ★**통과 판정 있음** `T_total < 1.8Tp = 21.6 s`; Simplified **3.0–7.6 s** | ⛔센서는 회전구조물, **관심점은 고정 터빈** | ⚠데이터만(Zenodo 7689781) |
| **Liu et al. 2026 (BayesNF)** *AOR* 173:105173 | ⛔**유의파고 Hs 장**(위상분해 아님) | 위성고도계 9종 + 부이 | ★**격자없음**(좌표→값 신경장) | **실측**(SWOT·부이 독립검증) | 한 달치 장 **52 s** (TPU v6e) | 해당 없음 | ⚠미확인 |
| **Desmars et al. 2023** *JFM* 975 A8 | 비선형 재구성+예보 | **합성 원격관측**(레이더 모델) | 파동성분 | **합성 + ★수조 실험**(⛔"합성 전용" 반증) | ⚠미확인 | ⛔**1-D · 정지** | ⚠미확인 |
| ★**우리 (WaveSplat, 2026-08 배포)** | ★**위상분해 순차 추적/나우캐스트** (⛔"예보 검증" 금지) | **선박탑재 LiDAR** (⚠**합성만** — 실측 13파일은 **고정 스테레오**) | ★★**격자없음 · 해석적 스펙트럼 성분 O(10)** | ⛔**합성 전용**(선형 GT + HOS M=3). 실측 재구성 결과 **논문에 0쪽** | 초기화 **8.47 s** · 프레임 **8.90 ms** (⚠Kim Eq.(21) 우리 기하 재도출 예산 0.7–5.7 s **미달**) | ⚠`C` 배치 **`exploratory`** (게이트 4개 미구현) | ⚠공개 안 됨 |

*우리 행의 근거: `BRIEF §4` 표 + `STATUS_2026-08-21 §1` 7행 + 08-22 정정 헤더. ⛔`STATUS §1` 5·8·9행은 08-22 에 무효화됨 — 나는 그것들을 안 썼다.*

---

## §8 ★남은 틈이 실재하는가 — **내 판정**

### ✅**실재한다. 단 하나이고, 세 조건의 교집합이며, 넓지 않다.**

> ★**"희소·불균질 LiDAR 점군으로부터, 격자를 거치지 않고, 분산관계를 강제한 소수 파동성분을 순차적으로 추적하는 것."**

**살아남는 이유 — 각 경쟁자가 어디서 갈라지는지:**

| 갈림점 | 누가 여기서 갈라지는가 |
|---|---|
| **격자를 안 쓴다** | Wang 2025(균일격자 16,384) · DSGP-Net(영상 dense map) — ★**둘 다 격자다** |
| **분산관계를 강제한 물리 매개화** | BayesNF(*"only spatiotemporal coordinates"*, 물리 없음) · DSGP-Net(단일프레임, 시간 정보 불필요) |
| **순차 추적(시간축 재적합)** | DSGP-Net(*"without requiring temporal information"*) — ★단일프레임이라 시간축이 아예 없다 |
| **LiDAR** | Wang·DSGP-Net(스테레오, 20 m) · Kim(수조 wave gauge 모의) · BayesNF(위성) — ★**실제 LiDAR 로 잰 위상분해 연구가 이 목록에 0편** |

### ⛔**그러나 틈을 좁히는 사실 네 가지 — 전부 이번 조사에서 나왔다**

1. ★★★**Wang 2025 가 이 틈을 스스로 지목했다**(§2.3). ⟹ *"아무도 생각 못 했다"* 는 죽는다. 살아남는 문장은 **"제안된 공백을 실행하고 측정했다"** 뿐이고, 그러려면 **측정이 그들 자로 서야 한다**.
2. ★★**같은 그룹이 2024-12 → 2026-03 으로 이 공간을 체계적으로 걷고 있다.** 다음 편이 정확히 이 틈일 확률이 낮지 않다 `ARGUED`.
3. ⛔**"LiDAR" 라는 갈림점을 우리가 아직 못 산다** — 우리 LiDAR 는 **합성뿐**이고, 우리가 가진 실측은 **경쟁자와 같은 스테레오**다. ⟹ 현재로선 **센서 축이 논문에서 "시뮬레이션 가정"** 이다.
4. ⛔**"격자없음" 단독은 이제 표적 저널의 일상어**다(§4).

### ★그래서 지금 쓸 수 있는 정직한 novelty 문장 (내 제안)

> **"희소 스테레오/LiDAR 점군으로부터 균일격자로의 보간 없이, 분산관계로 구속된 소수의 파동성분을 상태로 두고 순차 갱신하는 추정기. Wang et al. (2025) 가 향후과제로 지목한 *'entry frequencies as state vectors … direct sparse assimilation'* 을 구현하고, 격자 기반 앙상블 동화 대비 혼잡 레짐에서 부호 수준의 우위와 3.19배 빠른 초기화를 측정했다."**

⚠**이 문장이 사는 조건 3개**: ① 크기 대신 **부호**만 주장(`STATUS §2` 준수) · ② **혼잡 층 한정** 명시 · ③ **합성 한정** 명시.
⛔**이 문장으로도 못 사는 것**: *"이동선"*, *"조우 예보"*, *"실시간 통과"*, *"정확도 크기"*.

---

## §9 ★Golub & Pereyra 인용 문장 — **정직한 표현 제안**

**서지 `VERIFIED`:**
- Golub, G.H. & Pereyra, V. (1973), *"The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate"*, **SIAM J. Numer. Anal. 10(2), 413–432**.
- Golub, G. & Pereyra, V. (2003), *"Separable nonlinear least squares: the variable projection method and its applications"*, **Inverse Problems 19(2), R1–R26**, DOI `10.1088/0266-5611/19/2/201` (Crossref 직조회).

**맥락**: `98883a8` 로 `wavesplat/optimizer/varpro.py` 배포, 출하 기본값 `varpro_steps=400 · learned_init_iters=0` (`BRIEF2 §(2)`). ⛔`W6` 이 포락선 정리로 **VARPRO ≡ ALS 경사**임을 증명 ⟹ **기제 novelty 없음**.

### 제안 문장 (논문 §Method 용)

> **"파랑장 추정 문제는 파동수·주파수에 대해 비선형이고 성분 진폭·위상에 대해 선형인 분리형 비선형최소제곱(separable NLS)이다. 우리는 선형 변수를 폐형으로 소거하는 고전적 variable-projection 기법(Golub & Pereyra, 1973; 2003)을 이 문제에 적용한다. 알고리즘 자체는 새롭지 않다 — 기여는 (i) 이 문제가 분리형임을 밝히고, (ii) 그 구조를 쓰지 않는 배포 옵티마이저 대비 개선을 사전등록된 조건에서 측정한 것이다."**

★**왜 이 표현인가** (검증 결과에 근거한 세 가지):
1. ⛔**"우리는 VARPRO 를 제안한다" 금지** — 1973년 기법이고 `W6` 이 우리 구현이 ALS 와 수학적으로 동치임을 보였다.
2. ✅**"분리형임을 밝힌 것" 은 살아 있다** — 내가 읽은 경쟁자 중 **파랑 재구성을 separable NLS 로 진술한 논문이 없다**(Wang 2025 = 칼만/MAP 프레이밍 · Kim 2023 = 최적화 동화 · DSGP-Net = CNN). ⚠단 **내 열람 범위 안에서만** — 전수 조사가 아니다 `ARGUED`.
3. ⛔**"3.19배 빨라졌다" 를 novelty 로 쓰지 마라** — 그것은 **공학적 이득**이고, 인용 시 `W5·B`(64/64) 근거와 **혼잡·합성 한정**을 함께 적어야 한다.

---

## §10 ★내 정정·철회 (필수 절)

| # | 무엇 | 정정 |
|---|---|---|
| **정-1** | ★**과제 지시가 나에게 준 전제 하나가 틀렸다** — *"B2 가 AOR 173:105173 을 보고했다"*. 실제로 **`105173` 문자열은 `A1_ocean_reviewer.md:339` 와 `RESULT.md:152` 에만 있고 B2 에는 0회**다. **A1 의 발견이다.** (내가 grep 으로 확인) |
| **정-2** | ⚠**나는 DSGP-Net 초록을 축자로 확인하지 못했다.** 5경로 전부 차단. ⟹ ★**§3 의 표현·검증·이동선 판정은 전부 `UNVERIFIED-축자`** 이고, 논문에 쓰려면 **사서 읽어야 한다.** 나는 그 한계를 §3.1 에 명시했고, ⛔**"확인했다"고 말하지 않는다.** |
| **정-3** | ⚠**§6.1 의 "제로필드 0.25" 는 `ARGUED` 이지 `MEASURED` 가 아니다** — 나는 그 12 에피소드의 **실제 제로필드 팔을 돌리지 않았다**(런 0회). 0.25 는 nRMSE=RMSE/Hs · RMS(η)=Hs/4 의 **해석적 값**이고, 원장의 독립 서술과 일치한다는 것까지가 근거다. ⟹ ★**처방: 그 12 셀에 ZERO 팔을 붙이는 것은 런이 필요하고, 그때까지 판정은 *"제로필드와 구별되지 않는다"* 이지 *"제로필드와 같다"* 가 아니다.** |
| **정-4** | ⛔**원장 정정 제안**: `MEMORY.md` RESUME 의 *"Desmars JFM 975 A8 은 1-D · 정지 센서 · **합성 전용**이다"* 에서 **"합성 전용"을 삭제**할 것. 초록 축자가 wave tank experiments 를 명시한다. **"1-D"·"정지 센서" 는 유지**(unidirectional waves, 원격 레이더 모델). |
| **정-5** | ⚠**나는 A3 의 §4.3 재도출(0.7–5.7 s)을 검증하지 않았다.** 나는 **Kim 쪽 절반**(원문 축자·Eq.(21)의 기하 의존성·표 2)만 샀다. **우리 개구·c_g 입력은 A3 의 `MEASURED` 이고 내 범위 밖**이다. ⟹ ⛔**"R2-5 가 0.7–5.7 s 를 확인했다" 고 쓰지 마라.** |
| **정-6** | ⚠**§2.2 의 "425 / 260 ms per frame" 은 내가 표 3 의 6단계를 더한 값이고, 원문이 그 합계를 적지는 않는다.** 단계값은 축자, **합산은 내 산술**이다 `MEASURED+ARGUED`. 또한 그들의 `WASSfast 대비 3–4×/2×` 는 **다른 표(Fig. 6)** 근거다. ⟹ 인용 시 **"단계 합"임을 명시**할 것. |

---

## §11 ★주 세션(오케스트레이터)의 추가 오류 — `BRIEF2 §3` 요구

| # | 오류 | 근거 |
|---|---|---|
| **O-1** | ★★★**`BRIEF §3`·`RESULT §7` 이 "새로 발견된 경쟁자" 로 Wang 2025 를 올렸지만, 그 논문 PDF 는 `ref_paper/feng_wang_2025.pdf` 에 **2026-06-24 부터** 있었고, 그 논문을 재현한 기준선 코드 `baselines/etkf_wang2025.py` 와 결과 `results/wang2025_comparison.json` 도 저장소에 있었다.** ⟹ **"새로 발견"이 아니라 "잃어버렸다 되찾은 것"** 이다. ★이것은 `STATUS` 08-22 헤더가 이미 자백한 ***"원장이 이미 잰 것을 잃어버렸다"*** (`D-17`/`D-18`) 와 **정확히 같은 결함의 세 번째 사례**다 |
| **O-2** | ★★**`RESULT §7` 이 AOR 105173 을 *"grid-free 파랑장 재구성"* 이라 요약했다** — 초록을 열면 **유의파고 · 위성고도계 · 월단위**다. ⟹ **A1 의 신중한 문언**(*"읽어야 한다"*)이 **종합 단계에서 단정으로 굳었다.** ★저장소의 상습 결함 *"요약이 측정보다 넓다"* 의 문헌판 |
| **O-3** | ⚠**`BRIEF §3` 의 *"가장 가까운 경쟁자: Wang et al. 2025"* 라는 한 줄이 라운드 1 열 편 전체의 문헌 프레임을 고정했다.** A1 만이 저자 추적으로 DSGP-Net 을 찾았고, **나머지 아홉은 브리핑을 그대로 받았다.** ⟹ ★**처방: 브리핑에 문헌 주장을 넣을 때는 `VERIFIED` 등급과 열람범위를 함께 적을 것** (지금 `BRIEF §3`·`§4` 에는 없다) |
| **O-4** | ⚠**`scripts/generate_labmeeting_2026_05_13.py:552`** 가 2026-05 발표자료에 *"우리 `etkf_wang2025.py` 가 이 논문을 충실히 재현"* 이라 적었는데, **같은 저장소의 결과 파일이 그 팔을 무기술로 보여준다.** 발표 문언이 결과와 대조되지 않았다 |

---

## §12 W7 · W8 이 내 판정을 움직이는가 — **`BRIEF2 §2` 요구**

⛔**움직이지 않는다.** 이유를 명시한다:

- **W7**(λ 교차, 창-안 벽의 존재/크기)과 **W8**(VARPRO 이식 비트동일)은 **전부 내부 솔버 축**이다. 내 §7 위치표의 여섯 열 중 어느 칸도 λ 나 이식으로 바뀌지 않는다.
- **단 한 곳에서 간접적으로 닿는다**: **W8 SECONDARY 가 `PORT-OK-EFFECT-NULL`**(11/16, −11.78%, p=0.105) 이므로, §9 의 VARPRO 인용 문장에서 ⛔**"배포로 개선했다"를 쓸 수 없다.** 내가 §9 에 쓴 문장은 **`W5·B` 의 초기화 3.19배**만 근거로 삼고 배포 전후 정확도 개선은 주장하지 않는다 — ✅**W8 과 정합한다.**
- **W7 이 `STATUS §1` 5행(척추)을 흔든 것**은 §7 표의 우리 행에 반영돼 있다: 나는 척추 문장을 **위치표에 쓰지 않았다**(08-22 정정 헤더가 무효화했으므로).

---

## §13 라운드 1 열 편 사이의 **문헌 축 모순** — `BRIEF2 §1` 요구

| 모순 | 어느 쪽이 옳은가 |
|---|---|
| **B2-8**: *"Wang 그룹(2025 + 2026)이 … 검증했고, **2026년 논문은 이동선까지** 다룬다"* ↔ **A1-G**: *"DSGP-Net **만** 이동선"* | ★**A1 이 옳다.** Wang 2025 는 축자 *"vessel on position keeping"* = **정지**. B2 의 묶음 서술이 2025 에 이동선 속성을 흘렸다 (§2.2) |
| **B2 §5.2**: *"Wang 2025 = ★★★공개 + 자체 현장 실측"* ↔ **B2-8**: *"우리가 못한 것(실측 검증)을 했다"* | ✅**둘 다 옳지만 불완전**하다 — **우리도 시도했다**(§6.1). 정확한 문장은 *"우리는 실측 검증을 안 한 게 아니라, 그 비교가 무기술 기준선 위에 섰다"* |
| **A1 §4.3**: AOR 105173 ⛔*"grid-free 를 차별점으로 쓰기 전에 반드시 읽어야 한다"* ↔ **RESULT §7**: *"grid-free 파랑장 재구성이 실렸다"* | ★**A1 의 조건부가 옳고 RESULT 의 단정이 틀렸다** (§4, O-2) |
| **A3-6**: *"`1.8Tp` 는 원문 축자이나 보편 규칙 아님"* ↔ **`STATUS §1` 9행**: *"실시간 문턱 63/64 통과"* | ★**A3 가 옳다.** 나는 Eq.(21) 을 직접 보고 **기하 의존성을 확인**했다 (§5.2) |

---

## §14 ★D-1 권고 (마지막 절 — 필수)

**라운드 1 결정** = *"희소·불균질 LiDAR 로부터의 실시간 파도장 추정"*(플랫폼 중립) · 검증 = 정지 확증 + 이동 경계 · 실측 절은 고정 구조물 12 레코드.

### ✅**나는 이 결정을 지지한다. 단 문헌 근거로 두 곳을 조인다.**

**조임 1 — "실시간" 을 문장에서 빼거나 조건부로 만들 것.** `ARGUED`
- Kim 2023 이 **이 계열의 실시간 자를 이미 정의하고 통과 판정까지 냈다**(Eq.(21) · `T_total < t′_max` · Simplified 3.0–7.6 s, §5). 그 자를 우리 기하로 재도출하면 우리는 **미달**이다`[인용: A3 §4.3]`.
- Wang 2025 도 *"fast"* 라 부르지만 **260–425 ms/frame** 이고 **"real-time" 을 초록 결론에 안 쓴다**(축자 확인: *"superior performance in efficiency, accuracy, and robustness"*).
- ⟹ ★**권고 문언**: *"실시간 파도장 추정"* → **"순차(recursive) 파도장 추정"** 또는 **"프레임당 O(10 ms) 갱신을 갖는 …"**. **실시간 주장은 §Discussion 에서 Kim 의 자와 함께 조건부로만.**

**조임 2 — "LiDAR" 를 단수로 못 박지 말 것.** `ARGUED`
- ⛔우리 LiDAR 는 합성뿐이고, 실측은 **스테레오**다(§6.2). 제목이 "LiDAR" 이면 리뷰어가 **실측 절과 제목의 불일치**를 즉시 본다(B2-1 이 이미 지적한 것의 문헌판).
- ✅**반대로 유리한 점**: 우리가 가진 실측이 **경쟁자가 쓴 바로 그 데이터셋**이라 *"희소 3D 점군"* 으로 문언을 잡으면 **제목·실측·경쟁자 비교가 한 줄에 선다.**
- ⟹ ★**권고 문언**: *"희소·불균질 **LiDAR**"* → **"희소·불균질 3D 점군(선박탑재 LiDAR·스테레오)"**.

### ★그리고 **결정이 아니라 실행** 으로 넘겨야 할 것 (우선순위)

| # | 무엇 | 왜 지금인가 | 런 |
|---|---|---|---|
| **1** | ★★★**`results/wang2025_comparison.json` 의 Wang 팔이 왜 무기술인지 규명 — 또는 그 파일을 인용금지 목록에 올리고 봉인** | 최근접 경쟁자에 대한 **유일한 비교**가 지금 어느 방향으로도 못 쓰인다 (§6.1) | ⚠필요 |
| **2** | ★★★**AA_2014/AA_2015/YS_2017 위에서 우리 추정기를 돌린다** — 자료 취득 0, 경쟁자 3편과 **같은 사이트** | B2-1(논문에 실측 0쪽) + B2-8(경쟁자가 우리 데이터로 검증) 을 **한 번에** 닫는다 (§6.2) | ⚠필요 |
| **3** | ★★**DSGP-Net 원문을 산다** (§3.1 의 `UNVERIFIED` 3칸: 축자 문구 · 이동선 실험이 실측인지 · real-time 정의) | *"이동선 축이 점유됐는가"* 가 여기 걸려 있다 | 0 |
| **4** | ★**서론에 Wang 2025 future-work 문장을 인용한다** (§2.3) | **최근접 경쟁자가 우리 방향을 지목한 문장**이다 — 이보다 강한 서론 근거가 지금 저장소에 없다 | 0 |
| **5** | ⛔**원장 정정 2건 전파** — Desmars "합성 전용" 삭제(정-4) · `RESULT §7` AOR 행 수정(O-2) | | 0 |

---

*작성: R2-5 (문헌 정밀조사자) · 2026-08-22 · 런 0회 · 열람 범위는 §0 에 전부 열거함.*
