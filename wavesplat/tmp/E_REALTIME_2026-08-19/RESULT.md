# E · 실시간 주장 정식화 — `RESULT`

**런 0회.** 기록 재집계(W5·W4 원자료, bankmmse 포트, confirm2 사전등록) + 문헌 1차 검증(WebFetch).
⚠**WebSearch 예산이 세션 시작 시점에 이미 소진**(200/200)돼 검색은 못 했다. 검증은 **직접 URL WebFetch**
(arXiv HTML, Cambridge PDF 원문, OpenAlex/Crossref API)로만 했고, 확인 못 한 칸은 전부 `UNVERIFIED` 다.

---

## 0. 한 줄

★**바뀐 것은 "얼마나 빠른가"가 아니라 "어느 문턱을 넘었는가"다.** 이 하위문헌이 **표로 검정하는**
실시간 기준은 **"동화 시간이 실용 예보구간(≈1.8·Tp) 안에 들어와야 한다"**(Kim+ 2023, *Ocean Eng.*
276:114212)이고, 배포 팔은 **0/64 실패**, 오늘의 `L-VP-ONLY` 는 **63/64 통과**다.
(Ehlers 의 산문 기준 "파도보다 빨라야"로 재도 결과는 같다: 0/64 → 63/64.)

⛔**정정 2 (이 배치 배경 문구의 오류)**: *"우리가 그 틈을 처음 산다"* 는 **쓸 수 없다.**
Kim+ 2023 이 **하드웨어·기준·표를 이미 냈다**(i7-8700; nowcast 3.0–95.7 s; forecast <0.01 s).
★**우리 8.47 s 는 그 표에서 `Simplified` 대역(3.0–7.6 s)의 바로 옆이고, 배포 26.81 s 는
그들이 "실시간 불충족"이라 판정한 `Previous`(52.1/95.7 s) 쪽 자리였다.**

⛔**그리고 그 수는 합성·혼잡·`learned` 고정에서 잰 것이고, 실측 입력은 코드가 다른 경로로 보낸다.
같은 팔을 실측 창에 실제로 태운 유일한 기록은 초기화 39.2–53.9 s 다.**

---

## 1. 문헌이 요구하는 "실시간" — 무엇을 어떻게 재는가 `VERIFIED`

### 1.1 축자 인용 (전부 원문 fetch 로 확인)

**Ehlers, Stender & Hoffmann, arXiv:2508.03315** (*Bridging ocean wave physics and deep learning:
Physics-informed neural operators for nonlinear wavefield reconstruction in real-time*)
— `https://arxiv.org/abs/2508.03315`, `https://arxiv.org/html/2508.03315v1`

> ★**내 정정 시도 1건 = 철회(내가 틀렸다).** 나는 arXiv abs 에 `Journal reference:` 줄이 없고
> 제출 이력이 **v1(2025-08-05) 하나**, 분류 **cs.LG**, Comments `"13 pages, 7 figures"` 뿐인 것을 보고
> 저장소의 *"(Phys. Fluids 2025)"* 표기를 `UNVERIFIED` 로 깎으려 했다. **Crossref 로 확인하니 저장소가
> 맞다**: **Physics of Fluids 37, 107119 (2025-10-01), doi:10.1063/5.0294655**, 저자 Ehlers·Stender·
> Hoffmann, 제목 일치. **arXiv 쪽이 갱신되지 않았을 뿐이다.**
> ⟹ ★**교훈(재발)**: **부재 증거로 정정하지 마라.** `arXiv 에 없다` 는 `출판되지 않았다` 가 아니다.
> 게재 확인은 **Crossref/OpenAlex 로 이중 확인**해야 한다. ✅**정식 인용 = Phys. Fluids 37, 107119 (2025).**

1. **문제 진술** (초록, 축자):
   > "Accurate real-time prediction of phase-resolved ocean wave fields remains a critical yet largely
   > unsolved problem, primarily due to **the absence of practical data assimilation methods** for
   > reconstructing initial conditions from sparse or indirect wave measurements."

2. ★**실시간의 정의** (축자) — **이 하위문헌에서 확인된 유일한 명시적 기준**:
   > "**Real-time capability requires wave computations to be faster than the physical waves**, while
   > modelling nonlinearity is crucial to capture key dynamics such as wave–wave interactions,
   > nonlinear dispersion, and rogue waves."

3. ★**Desmars 를 깎는 기준** (축자):
   > "Although the first method enables effective parallelization, **its real-time capability is not
   > demonstrated.**"
   ⟹ ★**깎는 기준은 "초기화 시간"도 "프레임률"도 "예보 리드"도 아니다 — `증명의 부재` 다.**
   Desmars 는 **병렬화 가능성을 논증**했을 뿐 **어떤 시간 수치도 보고하지 않았고**(§1.2 에서 원문
   전수 확인), Ehlers 는 정확히 그 점을 지적한다. **비교 가능한 수를 낸 쪽이 이긴다는 뜻이다.**

4. **자기 수치**: `"The PINO 𝓜A achieves a reconstruction time of approximately 0.014 s per sample"`,
   하드웨어 `"NVIDIA GeForce RTX 3090 GPU"`(학습 기준 명시), 자료
   `"Since real-world measurements lack ground truth wave elevation data for direct comparison, this
   study validates the PINO approach using synthetic data."`
   ⟹ ★**Ehlers 도 실측 검증은 없다.** 그리고 0.014 s 는 **재구성(초기조건) 1샘플**이지 순차 추적이 아니다.

**Desmars, Hartmann, Behrendt, Hoffmann & Klein (2023)**, *Nonlinear deterministic reconstruction and
prediction of remotely measured ocean surface waves*, **J. Fluid Mech. 975, A8**, doi:10.1017/jfm.2023.841
— 원문 PDF 40쪽 전문을 받아 전수 검색(`fitz`)했다.

- ★**벽시계/CPU 시간 = 0건.** `CPU`·`wall`·`GPU`·`processor`·`Intel`·`AMD`·`minutes`·`hours`·`expensive`
  **전부 0 히트**. 보고하는 것은 **점근 비용**(`O(Np·Ns)`, `O(Np log Ns)`)과 **병렬화 가능성**뿐:
  > "no sequential time propagation of the solution is required, **making possible the time
  > parallelization** of the model evaluation in the optimization process."
- **동화 구간(assimilation period)**: `Ta = 3·Tp` — 합성 사례 `Tp = 10 s` ⟹ 30 s, 실험 사례
  `Tp = 16.7 s` ⟹ **"A similar assimilation period Ta = 3Tp (here ∼50 s) is used"**.
  시간 이산 `Nt = 48·Ta/Tp = 144` (`Δt ≈ 0.21 s`).
- **자료**: `"High-fidelity synthetic wave data are first used, before applying the methods to
  experimental wave tank data."` ⟹ **합성 HOS + 수조**. **외해 실측 0건.**
- **예보 리드**: 절대 시간이 아니라 **이론적 `prediction zone`**(군속도 겹침 영역)으로만 정의.
  리모트 센서 일반론에서 `"limited to measurement distances of ∼100 m, restraining the prediction
  horizon to a few characteristic wave periods."`

### 1.1b ★★★**이 배치의 전제가 틀렸다 — 이 하위문헌에는 이미 정식 실시간 기준과 표가 있다** `VERIFIED`

**Kim, Ducrozet, Bonnefoy, Leroy & Perignon (2023)**, *Real-time phase-resolved ocean wave prediction in
directional wave fields: **Enhanced algorithm and experimental validation***, **Ocean Engineering 276,
114212**, doi:10.1016/j.oceaneng.2023.114212 — **오픈액세스 전문 확보**(`https://hal.science/hal-04327969v1/document`, 18쪽 전수 검색).
⚠**저자는 Desmars 가 아니다**(같은 École Centrale Nantes / LHEEA 연구실의 후속 논문이다).

이 논문은 우리 문제 설정과 **거의 같다**: 선박/터빈 탑재 **광학 센서(LiDAR) 모사** 관측, 희소·비균질
샘플링, **동화(nowcast) + 전파(forecast)** 2단계, 방향 확산.

★**그리고 실시간 기준을 조작적으로 정의하고 표로 검정한다** (축자):
> "Table 2 Comparison of numerical cost and stability between the assimilation methods for Case A at
> WG21 where **the practical prediction zone in time 𝑡max − 𝑡r ≈ 21.6 s at full scale**."
> "It is confirmed that **the time duration for the forecast appears to be negligible in comparison to
> the time duration for the nowcast as well as the practical time prediction zone.**"
> "**with the previous assimilation, LWT-CDR cannot satisfy the real-time constraints**, while the
> simplified one renders not only LWT-CDR but also ICWM to **accomplish the real-time prediction in time**."
> "All the computations were run on an **Intel(R) Core(TM) i7-8700 CPU 3.20 GHz**."

**Table 2 (Case A, 축자 수치)** — `𝑡max − 𝑡r ≈ 1.8·Tp = 21.6 s`(전척 `Tp = 12 s`)가 **예산**이다:

| 모델 / 동화 | `T_nowcast` (s) | `T_forecast` (s) | `T_total` (s) | 반복수 | 수렴확률 (`Ns=400`) | 실시간 판정 |
|---|---|---|---|---|---|---|
| LWT | 1.1 | < 0.01 | 1.7 | – | – | ✅ |
| ICWM **Previous** (= Desmars+ 2020 동화) | **95.7** | < 0.01 | **95.7** | 22.7 | 83.5% (334) | ⛔ |
| ICWM **Simplified** | **7.6** | < 0.01 | 7.6 | 5.8 | 98.5% (394) | ✅ |
| LWT-CDR **Previous** (= Desmars+ 2020 동화) | **52.1** | < 0.01 | 52.1 | 15.9 | 88% (352) | ⛔ (축자로 "cannot satisfy") |
| LWT-CDR **Simplified** | **3.0** | < 0.01 | 3.0 | 3.0 | 100% (400) | ✅ |

**설정**: ECN 수조 50×30×5 m, **기하 1:40 · Froude 시간 1:√40**, PM 스펙트럼 전척 `Tp = 12 s`,
`Hs = 7 m`, `Lp = 225 m`, `Hs/Lp = 3.1%`, 심해 `kp·d ≈ 5.6`. Case A = 확산 `s=15`, `Jθ=11` 레이 ×
`Jr=20` 게이지, `Nθ=9`. 같은 파랑장 **47회 재현**으로 반복성 확인. **외해 실측은 아니다(수조).**

> ⛔**정정 2 (내 전제 붕괴 · 이 배치 배경 문구의 오류).**
> 배경은 *"Ehlers 가 Desmars 를 깎았고 저장소는 그 틈을 못 샀다"* 였고, 나는 §1.1 에서 Desmars JFM
> 975 A8 에 **시간 수치가 0건**임을 확인해 그 틀을 강화했다. **그러나 그 결론을 하위문헌 전체로
> 확장하면 거짓이다**: 같은 연구실의 Kim et al. (2023)이 **하드웨어를 명시하고, 실시간 기준을
> 예보구간으로 정의하고, 동화 시간을 표로 보고하며, Desmars 2020 동화가 그 기준을 못 넘는다는 것까지
> 정량화**했다. ⟹ ★**"우리가 이 축에서 비교 가능한 수를 처음 낸다" 는 문장은 쓸 수 없다.**
> ⟹ ★**그리고 이게 좋은 소식이다**: **비교할 자가 생겼다.** 아래 §2.1 에 그 자로 우리 수를 걸었다.

### 1.2 표 — 이 하위문헌이 보고하는 시간 수치

| 논문 | 초기화 / 동화 시간 | 프레임·갱신 시간 | 예보 리드 | 하드웨어 | 실측 여부 | 명시한 "실시간" 기준 |
|---|---|---|---|---|---|---|
| **Ehlers+ 2025** (*Phys. Fluids* 37:107119; arXiv:2508.03315) | — (재구성이 곧 초기화) | ★**0.014 s / 샘플** | 보고 없음 (재구성 중심) | RTX 3090 (학습 기준 명시) | ❌ **합성만** (축자로 인정) | ★**"파도보다 빨라야"** |
| **Desmars+ 2023** (JFM 975 A8) | ★**보고 0건** (`Ta = 3Tp` ≈ 30–50 s 의 **자료** 구간만) | 보고 0건 | `prediction zone` (이론), 리모트 일반론 "a few wave periods" | ★**보고 0건** | ❌ 합성 HOS + **수조** | 명시 없음 (병렬화 가능성만 논증) |
| ★★**Kim, Ducrozet, Bonnefoy, Leroy & Perignon 2023** (*Ocean Eng.* 276:114212) | ★**nowcast 1.1 / 3.0 / 7.6 / 52.1 / 95.7 s** (모델×동화별) | ★**forecast < 0.01 s** | 실용 예보구간 **`1.8·Tp` = 21.6 s**(전척) | ★**Intel i7-8700 3.20 GHz** | ❌ **수조 1:40** (LiDAR **모사**) | ★★**`T_total` < 예보구간(≈1.8·Tp)** |
| ⚠**Desmars+ 2020 동화** (자기 보고 아님 — **Kim+ 2023 이 대신 측정**) | **52.1 s / 95.7 s** | < 0.01 s | 상동 | i7-8700 | 수조 | ⛔축자 **"cannot satisfy the real-time constraints"** |
| **Qi 2017 MIT 박사논문** (프록시 — **JFM 838 아님**) | ★**"O(1) hours"** (3D HOR 전체) | `UNVERIFIED` | `UNVERIFIED` | ★**16코어 Microway 클러스터** | 합성 + **수조**(TAMU 36.1 m) | 없음(실시간 주장 0건) |
| **Qi, Wu, Liu, Kim & Yue 2018** (JFM 838:544) | `UNVERIFIED` (유료·OA 없음) | `UNVERIFIED` | `UNVERIFIED` (동일 그룹 2018b 가 `predictable zone` 정식화) | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` |
| **Mérigaud & Ringwood 2019** (*IEEE JOE*) | 예측행렬 **5–10 min** (30분마다 1회) | "**a simple matrix multiplication, which is not computationally demanding**" | **GoF>90% ≈50 s**, **GoF≈80% ≈80 s** | ★**3.50 GHz 8코어 Intel** | ❌ 스펙트럼 기반 **해석적** | 행렬 갱신 주기 = 해황 변화 주기(**"every 30 min"**) |
| ★**Mérigaud, Herterich, Flanagan, Ringwood & Dias 2018** (*IFAC-PapersOnLine* 51-29:232) | 행렬 **1회**(수치 없음) | 상동 | ⚠**"not appropriate … beyond a few seconds ahead"** (ADCP 최대 이격 40 m) | `UNVERIFIED` | ✅★**실측 ADCP** (Inis Meáin, 아일랜드) | ★축자 **"the only operation to be carried out in real time is the matrix multiplication"** |
| **Peña-Sanchez, Mérigaud & Ringwood 2018** (*IEEE JOE*) | `UNVERIFIED` | `UNVERIFIED` | "several wave periods" (초 단위 없음) | `UNVERIFIED` | ✅ 합성 + **실측** 병행 | 온라인 필터 지연 `T_delay` 언급만 |
| **Wang, Zhu, Cai, Wang & Qiao 2025** (*Coastal Eng.* **197:104690**) ✅서지 확인 | `UNVERIFIED` (유료, OA 0건) | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` |
| **Wang & Pan 2025** (arXiv:2504.13943, 파랑+선체운동 동시예보) | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` (결과가 `t/Tp` 단위) | `UNVERIFIED` | ❌ **합성** (축자) | 실시간 주장 0건 |
| **Yao, Zhang, Zhou, Song & Toffoli 2025** (*Phys. Rev. Fluids*, **선박 탑재 레이더**) | `UNVERIFIED` (유료) | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` |
| **Fan+ 2026** (*IEEE TGRS*, **선박 탑재 X-band+운동 융합**) | `UNVERIFIED` (유료) | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` | `UNVERIFIED` |
| — | | | | | | |
| ★**우리 `L-SHIP`(배포)** | **26.81 s** (med3 중앙) | **9.09 ms** | 미측정(②c) | 7800X3D CPU 1스레드 | ❌ 합성·혼잡 | (사내) 갱신 ≤100 ms |
| ★**우리 `L-VP-ONLY`(오늘)** | ★**8.47 s** | **8.90 ms** | 미측정(②c) | 동일 | ❌ 합성·혼잡 | 동일 |
| ⚠**우리 실측 경로**(`bank_mmse`) | **0.7 / 0.8 / 1.3 s** (창 3개) | ⚠**9.2 / 45.1 / 959.3 ms** | lead 14–20 s 채점됨 | 동일 | ✅ **스테레오 실측** | 동일 |
| ⛔**우리 `learned` 를 실측에 강제** | ⛔**39.2 s / 53.9 s** | 0.1 / 2.9 ms | — | 동일 | ✅ 실측 1창 | — |

### 1.3 서지 확인만 된 것 (시간 수치는 못 얻음) `VERIFIED(서지) / UNVERIFIED(수치)`

- ★**Wang, Zhu, Cai, Wang & Qiao (2025)**, *From points to waves: **Fast** ocean wave spatial–temporal
  fields estimation using **ensemble transform Kalman filter** with optical measurement*,
  **Coastal Engineering 197, 104690**, doi:`10.1016/j.coastaleng.2024.104690` (2025-04).
  ⟹ ★**우리 기준선 계열(ETKF)과 정확히 같은 도구로 "Fast" 를 표방하는 2025 논문이다.**
  본문 비공개(OA 없음, S2 초록 없음) ⟹ **시간 수치 `UNVERIFIED`**. **최우선 확보 대상.**
- ⚠**Wright, Craske & Karmpadakis (2026)**, *Real-time phase-resolved wave prediction over planar
  coastal bathymetries using U-Net convolutional neural networks*, **Coastal Engineering 209, 105044**,
  doi:`10.1016/j.coastaleng.2026.105044` (2026-07). ⟹ **제목에 "Real-time" 이 박힌 2026 논문**.
  저장소의 **"차별점 소멸" 경고를 한 건 더 보탠다.** 시간 수치 `UNVERIFIED`.
- **Qi, Wu, Liu & Yue (2018b)**, *Predictable zone for phase-resolved reconstruction and forecast of
  irregular waves*, **Wave Motion 77, 195–213** — Desmars 원문 참고문헌에서 확인.
  ⟹ 이 하위문헌에서 **리드는 "초"가 아니라 `predictable zone` 으로 보고되는 것이 표준**임을 뒷받침.

> ⚠**표의 `UNVERIFIED` 행들은 "값이 없다"가 아니라 "이 세션이 확인하지 못했다"** 이다.
> WebSearch 예산 소진 + Crossref/S2 rate-limit(429) 이 사유다. **이 4행이 채워지기 전에는
> "우리가 문헌 대비 빠르다"류의 비교 문장을 쓸 수 없다.** (병렬 확인 에이전트 결과 미도착.)

---

## 2. 우리 수가 그 정의를 만족하는가 — 항목별 판정

### 2.0 세 개의 서로 다른 자
- ★★**문헌 자 A (Kim+ 2023 — 조작적이고, 표로 검정된 것)**: **`T_nowcast + T_forecast` < 실용 예보구간
  `t_max − t_r ≈ 1.8·Tp`.** ⟹ ★**이 축에서 우리가 쓸 자는 이것이다** — 임의 문턱이 아니고, 같은
  문제 설정(선박 탑재 광학, 희소, 동화+전파)에서 이미 통과/실패 판정에 쓰인 자다.
- **문헌 자 B (Ehlers)**: *계산이 파도보다 빨라야 한다* = **소비한 자료의 물리 시간보다 계산 시간이 짧아야**.
  (산문 기준. 우리 구현으로는 `init_time_s < 2·Tp` = 초기창 자료 길이.)
- **사내 자 C (동결)**: `tmp/confirm2_2026-07-27/PREREG_CONFIRM2.md:120` — 축자
  > "I3 (분리, C4): (a) **갱신 경로 하드 게이트 — m3 프레임당 ≤ 100ms**: 19.7ms 평균 / 20.8ms 최악
  > 측정 PASS (OMP=1·워커 1, 8 lane, 5.1× 헤드룸). (b) **초기화 경로는 1회성 비용으로 예산 밖** …"
  ⟹ ★**저장소는 스스로 "초기화는 100 ms 예산 밖"이라고 동결해 놓았다.** 그러니 **8.47 s 로 통과했다고
  말할 게이트가 사내에는 애초에 없다** — 오늘 새로 넘은 문턱은 **문헌 쪽 자**다.
- ⚠두 자가 갱신 경로에서는 **우연히 일치**한다: LiDAR `scan_rate = 10.0 Hz` ⟹ 프레임 주기 = **100 ms**.
  즉 "프레임당 ≤100 ms" = "갱신이 파도보다 빠르다".

### 2.1 판정표 `MEASURED`

집계: W5 원자료(`tmp/W5_WHY_2026-08-19/results.jsonl` 640행 + W4 재사용 896행 = 1,536행),
에피소드 64 × 시드 3 의 **median-of-3**, `stratum=crowded` 전건, `init_method='learned'` 고정,
`device='cpu'`, `OMP/OPENBLAS/MKL_NUM_THREADS=1`, 워커 8.

| 항목 | 자 | 배포 `L-SHIP` | 오늘 `L-VP-ONLY` | 판정 |
|---|---|---|---|---|
| ★★**동화가 예보구간 안에 끝나는가** (`init_time_s < 1.8·Tp`) | **문헌 A (Kim+ 2023)** | 비 **1.768** (**0/64**) | ★비 **0.582** (**63/64**) | ⛔**미충족** → ✅**충족(63/64)** |
| **초기화가 파도보다 빠른가** (`init_time_s < 2·Tp`, 초기창 = `tp_factor=2.0` ⟹ `data_loader.py:116`) | 문헌(Ehlers) | 비 **1.592** (0/64) | ★비 **0.524** (**63/64**, 최악 1.008) | ⛔**미충족** → ✅**충족(63/64)** |
| **갱신이 프레임 주기 안인가** (`elapsed_ms_mean` ≤ 100 ms) | 사내 동결 + 문헌 | 9.09 ms (max med3 10.54, max 행 12.27) | 8.90 ms (max med3 **10.12**, max 행 **10.93**) | ✅**충족** (헤드룸 **11.2×**, 에피소드평균 최댓값 기준 **9.9×**) |
| **갱신 최악 프레임** ≤ 100 ms | 사내(하드 게이트) | ⚠**미측정** | ⚠**미측정** | ⚠**해당없음-미측정** (§3.3) |
| **예보 리드**(기여 ②c) | 문헌 | 미측정 | 미측정 | ⛔**미충족(미측정)** |
| **실측 입력에서의 위 전부** | 문헌 | 경로 자체가 다름 | 경로 자체가 다름 | ⛔**해당없음 → 미충족** (§3.2) |

**모자란 양(미충족 항목)**
- **배포 초기화**: 문헌 A 자(`1.8·Tp`) 대비 **1.768배**(0/64) ⟹ 통과하려면 **≥43% 단축** 필요.
  문헌 B 자(`2·Tp`) 대비 **1.592배** ⟹ **≥37% 단축**. **두 자 모두에서 전건 실패**다.
- **오늘 팔**: **3.19배 단축**(64/64 더 빠름)해 **두 자 모두 63/64 통과**. 실패한 1건은 문헌 A 자로
  비 **1.120**(12% 초과), 문헌 B 자로 **1.008**(0.8% 초과) — **같은 에피소드 하나**다.
  ⚠**`Tp=7 s` 층에서만 실패**(31/32)하고 `Tp=9 s` 층은 **32/32** ⟹ **예산이 짧을수록 아슬아슬하다.**
  ⟹ ★**더 짧은 `Tp`(더 거친 바다가 아니라 더 잔 바다)가 이 축의 최악 조건**이라는 뜻이고, 이건
  **미탐색**이다(코퍼스에 `Tp ∈ {7, 9}` 두 값뿐).
- **예보 리드**: 저장소에 리드-리졸브 정확도 곡선이 실측 경로엔 있으나(`rolling_refit.json` lead 14–20 s)
  **오늘의 팔에는 0건**. 즉 "실시간 순차 추적"의 *예보* 절반이 이 배치엔 없다.

---

## 3. ⚠정직 검사 — 이 수를 그대로 쓰면 안 되는 이유 전부

### 3.1 하드웨어 격차 `ARGUED`
- 측정기: **AMD Ryzen 7 7800X3D**(8코어/16스레드, 96 MB 3D V-Cache), `device='cpu'`.
  ★**GPU 는 쓰지 않았다** — RTX 4080 Super 는 이 수에 기여하지 않는다(이식성엔 유리).
- ⚠**8개 워커가 동시에 돌았다**(`WORKERS=8` 기본, `run_w5.py:684`)**, 스레드는 1로 고정.**
  ⟹ 8코어에 8프로세스 = **경합 상태의 수**다. 방향은 **보수적일 가능성이 크지만**(전용 실행이면 더 빠름)
  **X3D 의 거대 L3 를 8프로세스가 나눠 쓴 상태**라 부호를 단정할 수 없다. **전용 1잡 재측정 없음.**
- ⛔**선박 탑재 계산기 측정 0건.** 저장소에 Jetson/임베디드/선박용 산업 PC 벤치마크가 **하나도 없다**.
  7800X3D 는 2023년 데스크톱 게이밍 CPU 이고, 전형적 선박 탑재 x86 산업 PC 대비 단일코어 성능이
  높은 축이다. ⟹ ★**"8.5 s" 는 하드웨어 명시 없이는 무의미하다.** Ehlers 가 RTX 3090 을 적었듯
  **우리도 CPU 모델·스레드·동시부하를 문장 안에 넣어야 한다.**
- ⚠체크포인트 로드가 `_load_cond` 에서 **(ckpt, device) 캐시**된다(`learned_init.py:95-102`).
  배치는 워커당 1회만 지불 ⟹ **640행 중 사실상 8행에만 포함**. 콜드스타트 배포는 매번 지불한다
  (`results/direction3/beamform_cond_diffusion_mixed_v1.pt` = **5.86 MB**, 로드 시간 **미측정**).

### 3.2 ★★★자료·경로 격차 — **실측 입력은 이 코드에 도달하지 않는다** `MEASURED`
- `optimizer/sequential_update.py:460-508` **REAL-DOMAIN GATE**: `gt_field` 부재(= 모든 실측 프론트엔드)면
  라우터를 **우회**해 `bank_mmse` 로 보낸다. 축자 주석: *"the deployment authorisation covers SYNTHETIC
  inputs only"*. ⟹ ★**오늘 잰 `learned`+VARPRO 경로는 실측에서 실행되지 않는다.**
- **실측 배포 경로의 실제 수** (`tmp/bankmmse_port_2026-07-29/deployed_path_verify.json`, `real_gate=true`):

  | 창 | 라우트 | 초기화 | `predict_field` ms/frame | nowcast nRMSE |
  |---|---|---|---|---|
  | AA15a_w5 | `bank_mmse` | **1.3 s** | 45.1 | 0.4885 |
  | AA15b_w3 | `bank_mmse` | **0.7 s** | 9.2 | 0.6486 |
  | YS17_w0 | `bank_mmse` | **0.8 s** | ⛔**959.3** | 0.7228 |

  ★**초기화는 실측 경로가 10배 이상 빠르다**(0.7–1.3 s vs 8.47 s) — 닫힌형 선형 적합이라서다.
  ⛔**그러나 렌더링이 한 레코드에서 100 ms 예산을 9.6배 넘긴다.** 원인 미분리(질의격자 크기 후보,
  레코드별로 다름). ⚠**이 ms 는 W5 의 ms 와 다른 양이다**(§3.4).
- **같은 실측 창에 `learned` 팔을 강제**하면(`real_domain_gate=False`, 연구용 탈출구):
  `tmp/P1_real_path_2026-08-11/AUDIT_NEUTRAL.md §4.1` **초기화 39.2 s / 53.9 s** (λ=1e-3 / 1e-4).
  ⟹ ★**오늘의 경로를 실측에 태운 유일한 기록은 8.5 s 가 아니라 39–54 s 다**(4.6–6.4배).
  VARPRO 를 그 위에 얹으면 얼마가 되는지는 **미측정**.
- **층 격차**: W5 전 팔이 `stratum='crowded'` **전건**. 희소 0행. 라우터가 아니라 `learned` **고정**.
  배포 라우터는 혼잡의 90.6% 만 이 팔로 보낸다(저장소 기록) ⟹ 나머지 9.4% 는 이 수에 없다.
- **합성 격차**: 코퍼스 `data_s8_crowded_2026-08-10`, `Tp ∈ {7, 9} s`, 301 프레임 = 30.1 s.
  ⛔저장소 확정 사실: **합성 GT 가 추정기 모델식과 100% 동일**(선형 중첩) ⟹ 초기화가 푸는 문제가
  실측보다 구조적으로 쉽다. HOS 게이트는 **정확도 라인에서만** 통과했고 **비용 라인은 미검**.

### 3.3 계량 자체의 결함 `MEASURED`
- ★**꼬리가 없다.** `elapsed_ms_mean = np.mean(...)` (`run_w4.py:392-394, 434`) — 에피소드별
  **평균**만 기록한다. **최댓값·p99 는 어디에도 없다.** 100 ms 는 **프레임 마감(deadline)** 인데
  우리가 가진 건 평균뿐이다. ⚠게다가 `L-VP-ONLY` 는 **성장 분기 발화가 3배**(141/192 vs 47/192,
  W5 §4) ⟹ **스파이크가 생길 구조적 이유가 늘었는데** 그 프레임의 지연은 안 잡힌다.
- ★**갱신 시간이 "쓸 수 있는 장(field)" 을 포함하지 않는다.** `run.py:299` 는 `update()` 의
  `elapsed_ms` 만 적고, `run.py:277` 의 `predict_field(...)` 는 **계측되지 않는다.**
  ⟹ **8.9 ms 는 동화 비용이지 예보 산출 비용이 아니다.** (실측 경로에서 예산을 터뜨린 959 ms 가
  바로 이 미계측 단계다.)
- ★**`wall_s` 잔차가 팔마다 갈린다** `ARGUED`: `wall − init − Σupdate` 를 채점 프레임으로 나누면
  `L-SHIP` **14.5 ms/f**, `L-ITER0` 15.0 vs **`L-VP` 31.2, `L-VP-ONLY` 31.1**.
  원자수로는 설명 안 된다(`L-ITER0` 도 19개). ⟹ **VARPRO 팔에 귀속처 없는 ~16 ms/프레임(≈2.6 s/런)이
  있다.** 결론(2.05배 총 벽시계 우위)은 안 뒤집히지만, ★**"프레임당 8.9 ms" 가 그 팔이 프레임당 쓰는
  전부가 아니다.** 초기화 밖 실측 총합은 대략 **배포 ~24 ms/f vs 오늘 팔 ~40 ms/f** ⟹ 100 ms 대비
  헤드룸이 **11배가 아니라 2.5배**로 줄어든다. (⚠채점(GT 대조) 비용 포함 = 배포엔 없는 일. **분리 미측정**.)
- **비결정성**: `deterministic: false`. 시드 3개 median-of-3 로 완화했으나 시간은 층·시드 간
  변동이 크다(초기화 max 행 14.23 s vs 중앙 8.47 s = **1.68배**).

### 3.4 ⚠**같은 이름의 다른 양** (이 저장소가 반복해서 물린 함정)
| 이름 | W5(합성) | bankmmse(실측) |
|---|---|---|
| "초기화" | 2·Tp 창 위 **경사하강 720스텝 + 학습 제안** | 40 s 창 위 **닫힌형 3멤버 적합** |
| "ms/frame" | `update()` **동화만** | `predict_field()` **격자 렌더링만** |
⟹ ⛔**두 열을 같은 표의 같은 칸에 넣어 비교하지 마라.** 위 §1.2 표에서 행을 분리한 이유다.

### 3.5 초기화는 일회성인가 — **경로마다 답이 다르다** `MEASURED`
- **합성/`learned` 경로**: `run.py:169` 가 `initialize()` 를 **1회** 호출하고 이후 `update()` 루프뿐.
  재초기화 트리거 **없음** ⟹ ✅**일회성**. (단 추적 중 성장 분기가 원자를 추가한다 — 부분 재적합.)
- **실측/`bank_mmse` 경로**: `_bank_update` 가 `bank_refit_every_s` 로 **재적합**한다.
  **출하 기본값은 `None` = 재적합 없음**(`sequential_update.py:824`).
  ⚠**그런데 저장소 자신의 측정이 재적합이 더 낫다고 말한다** (`rolling_refit.json`, 9창):
  sustained-mean nRMSE **AA15a 1.080 → 0.822 · AA15b 0.802 → 0.680 · YS17 0.954 → 0.878**
  (frozen → refit@2 s), **3/3 레코드 개선**. 사전등록 `PROTOCOL_ROLLING_REFIT.md` 의 `ADOPT-CANDIDATE`
  규칙(최적 대비 0.01 이내 중 가장 싼 주기)을 적용하면 **R\* = 2 s**(YS17 만 5 s 로도 충분).
  **비용**: 3멤버 **순차** 적합 **0.78–1.52 s**(`fit_wall_s_m3`, `run_rolling_refit.py:125-128` 로 확인).
  ⟹ ★**2 s 주기 = 2 s 마다 0.78–1.52 s = 코어 1개의 39–76% 를 상시 점유**(3코어 병렬 시 ~13–25%).
  ⟹ ⛔**"초기화는 1회성 비용" 은 실측 경로에서 정직한 문장이 아니다** — 정확도를 위해선 상시 재적합이
  필요하고, 그 비용은 **예산 밖으로 밀어놓은 항목이 아니라 상시 부하**다.

### 3.6 범위 격차 (문헌 대비)
- ⛔**리드 없음**: 두 문헌 모두 최종 목표는 **예보**다. Desmars 는 `prediction zone`, Ehlers 는
  "real-time forecasting". 우리 오늘 수치는 **재구성/추적**만이고 **기여 ②c(리드) 미측정**.
- ⛔**정지 센서**: 저장소 감사 기록 — `ship_x=ship_y=0` 하드코딩, `ship_speed_kts` dead config,
  195 에피소드 전건. 조우 기하 미측정. ⟹ Desmars 가 문제 설정으로 삼은 **"전진하는 배 주위"** 를
  우리는 아직 시뮬레이션하지 않았다.
- ⚠**차별점의 시한**: 저장소 문헌조사가 이미 잡았다 — 선박 탑재 이동 재구성 논문이 2025–2026 에
  다수(예: `Sensors 26(3):993` 선박 탑재 스테레오) ⟹ **"선박 탑재" 자체는 곧 novelty 가 아니다.**

---

## 4. ★살 수 있는 문장 / 못 사는 문장

### ✅ 살 수 있다

> **S1 (권장 헤드라인 — 문헌의 자로).** *"Kim et al. (2023)이 이 문제에 쓴 실시간 기준 — 동화 시간이
> 실용 예보구간(`≈1.8·Tp`) 안에 들어올 것 — 을 그대로 적용하면, 합성 혼잡 층에서 배포 구성은
> **0/64** 로 실패하고 오늘 구성은 **63/64** 로 통과한다(중앙 비 1.77 → 0.58). Ehlers 의 산문 기준
> ('계산이 파도보다 빨라야')으로 재도 판정은 같다(0/64 → 63/64)."*
> · **근거**: `MEASURED`(W5 median-of-3, `init_time_s`) × `VERIFIED`(Kim+ 2023 Table 2 문맥 축자).
> · **왜 사는가**: **문턱을 우리가 고르지 않았다.** 두 개의 독립적인 문헌 자에서 **판정이 일치**한다.
> · **반례/한계**: 1 에피소드는 비 1.12(Ehlers 자로는 1.008)로 실패. 합성·혼잡·`learned` 고정. 8워커 경합.
>   ⚠**Kim 의 Tp = 12 s 라 예산이 21.6 s, 우리 Tp = 7/9 s 라 예산이 12.6/16.2 s** — 우리 쪽이 더 빡빡한
>   예산에서 통과했다는 점은 유리하지만, **파랑 조건이 다르므로 "더 잘했다"로 번역하지 마라.**

> **S1b (같은 사실의 정직한 뒷면 — 반드시 함께 쓸 것).** *"절대 시간으로 보면 우리 8.5 s 는
> 돌파가 아니다. Kim et al. 은 2023년에 i7-8700 에서 동화 3.0 s(LWT-CDR-Simplified)·7.6 s(ICWM-Simplified)
> 를 이미 보고했다. 우리가 넘은 것은 **문턱**이지 **최속**이 아니다."*
> · **근거**: `VERIFIED` Table 2. · **왜 필요한가**: 이걸 빼면 첫 리뷰어가 붙인다.

> **S2.** *"프레임당 동화 비용의 에피소드 평균은 중앙 8.9 ms, 에피소드 평균의 최댓값 10.1 ms 로,
> 10 Hz LiDAR 의 100 ms 프레임 주기 안이다(중앙 기준 11.2배, 에피소드 평균의 최댓값 기준 9.9배 여유).
> 단 최악 **프레임** 지연은 측정되지 않았다."*
> · **근거**: `MEASURED` + 사내 동결 게이트(`PREREG_CONFIRM2.md:120`).
> · **반드시 붙일 것**: "동화 비용이며 장 렌더링은 포함하지 않는다"(§3.3).

> **S3.** *"초기화를 3.19배 줄이면서 정확도가 25.3% 좋아졌다(64/64 더 빠름, 58/64 더 정확)."*
> · **근거**: `MEASURED`, W5 P-1. · **한계**: 혼잡 층 한정, `L-VP-ONLY` 는 최소 4요인 묶음(W5 §4).

> **S4 (문헌 배치 — 좁혀서).** *"Desmars et al. (JFM 975 A8)은 벽시계·CPU 시간을 하나도 보고하지 않고
> 점근 비용과 병렬화 가능성만 논증하며, Ehlers et al. 은 바로 그 점을 들어 'real-time capability is not
> demonstrated' 라 평한다. 반면 같은 연구실의 Kim et al. (2023)은 하드웨어를 명시하고 동화 시간을 표로
> 보고하며, 그 표에서 Desmars et al. (2020)의 동화가 실시간 기준을 넘지 못함을 정량화한다."*
> · **근거**: `VERIFIED` — JFM 원문 40쪽 전수 검색(시간 관련 키워드 전부 0 히트) + Ehlers 축자 + Kim 축자.
> · ⛔**한계(중요)**: 이 문장을 **"문헌 전체가 수치를 안 낸다"** 로 확장하면 **거짓**이다(§1.1b 정정 2).
>   그리고 §1.2 의 `UNVERIFIED` 6행은 아직 안 열렸다.

> **S5 (정직한 음성 — 이게 우리 강점 서사와 맞는다).** *"우리 실측 경로는 초기화가 1 s 미만이지만,
> 세 창 중 하나에서 장 렌더링이 959 ms/frame 으로 100 ms 예산을 9.6배 넘겼다. 즉 우리도 아직
> 실시간을 입증하지 못했다 — 다만 어디가 막혔는지는 수로 지목할 수 있다."*
> · **근거**: `MEASURED`, `deployed_path_verify.json`.

### ⛔ 못 산다

> **N1.** ⛔*"우리는 실시간 재구성을 입증했다."*
> · **반례 셋**: (a) 실측 입력은 `REAL-DOMAIN GATE` 가 다른 경로로 보내므로 **오늘 잰 코드가
>   실측에서 실행되지 않는다**; (b) 같은 팔을 실측에 강제한 유일한 기록이 **초기화 39.2–53.9 s**;
>   (c) 실측 배포 경로의 렌더링이 **959 ms/frame** 인 창이 있다.

> **N2.** ⛔*"초기화 8.5 초."* (조건 없이)
> · **반례**: 층(혼잡)·경로(`learned` 고정)·하드웨어(7800X3D CPU 1스레드, 8워커 경합)·코퍼스(합성,
>   GT 가 모델식과 동일)를 떼면 **재현 불가능한 수**다. 최소한 이 넷을 문장 안에 넣어야 한다.

> **N3.** ⛔*"프레임당 8.9 ms 이므로 100 ms 제약에 11배 여유가 있다."*
> · **반례**: 그 8.9 ms 는 **동화만**이다. 잔차 산술로 보면 초기화 밖 프레임당 총합은 **~40 ms**
>   (배포 ~24 ms)이고 여유는 **~2.5배**다(`ARGUED`, 채점 비용 미분리). **최악 프레임은 미측정**.

> **N4.** ⛔*"초기화는 1회성 비용이므로 예산 밖이다."* (실측 경로에 대해)
> · **반례**: 저장소 자신의 `rolling_refit` 이 **2 s 주기 재적합이 3/3 레코드에서 더 낫다**고 쟀고,
>   비용은 **코어 1개의 39–76% 상시 점유**다. (합성 경로에 대해서는 이 문장이 참 — §3.5.)

> **N5.** ⛔*"Ehlers 가 지적한 공백(실시간 순차 추적)을 우리가 메운다."*
> · **반례**: Ehlers 가 깎는 기준은 **증명의 부재**이고, 우리는 (a) **예보 리드 미측정(②c)**,
>   (b) **실측 무증명**, (c) **정지 센서 가정**이다. 즉 **깎이는 쪽에 우리도 서 있다.**
>   ★그러나 **N5 를 S1+S4 로 바꾸면 살아난다** — "메운다"가 아니라 **"이 축에서 비교 가능한 수를
>   처음 낸다"**. Desmars 가 0개, Ehlers 가 재구성 1개를 냈고, 우리는 **초기화·갱신 2개를 낸다.**

> **N6.** ⛔*"우리 초기화가 Desmars/Ehlers 보다 빠르다/느리다."*
> · **반례**: Desmars JFM 은 **수치가 없어 비교 불가**, Ehlers 의 0.014 s 는 **GPU·1샘플 재구성**으로
>   우리 CPU·순차추적과 **다른 양**이다. §1.2 의 `UNVERIFIED` 6행은 아직 안 열렸다.
>   ⚠**Kim+ 2023 과는 "비교 가능"에 가장 근접하지만**(같은 CPU 급, 같은 2단계 구조, 같은 기준)
>   **파랑 조건(Tp 12 vs 7/9 s)·관측 기하·GT 성질이 다르다** ⟹ 순위 문장 말고 **"같은 대역"** 까지만.

> **N7.** ⛔*"이 축에서 비교 가능한 수를 우리가 처음 낸다."* / ⛔*"문헌에 실시간 순차 추적의 틈이 있고
> 우리가 그것을 산다."*
> · **반례**: **Kim et al. 2023 이 이미 냈다** — 기준·하드웨어·표·통과판정까지(§1.1b).
>   게다가 그들은 **수조 실험 검증**까지 붙였고 우리는 **합성뿐**이다.
> · ★**살릴 수 있는 잔여 틈**: Kim 은 **한 표면 샘플의 재구성 비용**을 재고 `T_forecast < 0.01 s` 라 적을 뿐,
>   **프레임마다 갱신되는 순차 추적을 지속 부하로 측정하지 않는다.** 우리는 그걸 잰다(8.9 ms/frame,
>   160 프레임 연속). ⟹ ⚠**단 이건 "우리가 잰 것"이지 "그들이 못 하는 것"이 아니다.**
>   문장은 **"지속 갱신 비용을 프레임 단위로 보고한다"** 까지만 안전하다.

---

## 5. 기여 ②c 를 살리려면 — 최소 추가 측정 목록

우선순위 순. 비용은 W5 실측 처리량(에피소드·시드당 벽시계 중앙 14.6–29.8 s, 워커 8)에서 외삽.

| # | 측정 | 왜 필수인가 | 최소 설계 | 예상 비용 |
|---|---|---|---|---|
| **M1** | ★**최악 프레임 지연**(max·p99) 기록 | 100 ms 는 **마감**이지 평균이 아니다. 현재 하드 게이트가 **평균으로 채점되고 있다** | `run_w4.py:392` 옆에 `elapsed_ms_max`·`p99` 한 줄 추가 후 재집계 | **코드 1–2줄** + W5 재실행 1회 (~1.5 h @8워커) |
| **M2** | ★**`predict_field` 계측** | 예보 산출 비용이 **어느 경로에서도 예산에 안 들어가 있다**. 실측 경로가 여기서 터졌다 | `run.py:277` 을 타이밍으로 감싸 `predict_ms` 기록 | 코드 1줄 + M1 과 동일 런에 합승 |
| **M3** | ★★**리드-리졸브 정확도 곡선**(②c 본체) | 지금 "실시간 추적"의 **예보 절반이 0행**이다 | `L-VP-ONLY` vs `L-SHIP` vs `ZERO`, 리드 τ ∈ {0,1,…,3·Tp} s, 혼잡 64 ep × 3 시드, **사전등록 필수** | 채점만 추가 ⟹ 런 재사용 시 **~2–3 h**; 신규 런이면 ~4 h |
| **M4** | ★**전용 1잡 타이밍 재측정** | 현 수는 **8워커 경합**. 배포 수는 전용 실행 | 동일 팔, `WORKERS=1`, 8 에피소드 × 3 시드 | **~20 분** |
| **M5** | ★★**실측 경로에 오늘의 팔을 태우기** | 오늘 수치의 **실측 대응물이 39–54 s 짜리 하나뿐** | `real_domain_gate=False` + VARPRO 로 실측 13 레코드 초기화·갱신·렌더 시간만 (정확도 판정 아님, **탈출구임을 명시**) | **~1 h**, 런 13 |
| **M6** | **`bank_mmse` 렌더링 959 ms 원인 분리** | 실측 경로의 유일한 예산 위반 | 격자 크기 × 원자수 × 레코드 3요인 분해 | **~30 분** |
| **M7** | **콜드스타트 비용** (체크포인트 로드 + 첫 호출 JIT) | 캐시 덕에 **640행 중 8행에만 포함**됨 | 새 프로세스 1회 × 5 반복 | **~5 분** |
| **M8** | ⚠**대표 선박 탑재 하드웨어 1종** | "8.5 s" 의 이식성 주장 근거가 **0건** | 저사양 x86 1대(또는 CPU 주파수·코어 제한 프록시)에서 M1·M2 반복 | 하드웨어 확보에 의존, 프록시면 **~40 분** |
| **M9** | **문헌 표 §1.2 의 `UNVERIFIED` 6행 채우기** (최우선 = **Wang+ 2025 ETKF** — 우리 기준선과 **같은 도구**) | 비교 문장을 쓰려면 필수 | 도서관/저자 요청. ⚠**WebFetch 는 ScienceDirect 403 · 10 MB 상한**, `curl` 은 제한 없음(하위 에이전트 실측) | 런 0회 |
| **M10** | ★**보고 형식을 Kim+ 2023 Table 2 에 맞추기** — `T_nowcast` / `T_forecast` / `T_total` / 반복수 / ★**수렴확률** | ★**같은 표로 읽히면 리뷰어가 직접 비교한다.** 특히 **수렴확률**은 우리가 한 번도 보고한 적 없는 열이고, 추정기 분산·시드 CV 를 정직하게 담을 자리다 | 기존 행에서 파생 + 발산/실패 카운터 1개 | 코드 1–2줄, **런 0회 재집계** |
| **M12** | ★**`Tp` 축 확장** — 현재 코퍼스는 `Tp ∈ {7, 9} s` 둘뿐인데 **실패한 1건이 짧은 쪽에서 나왔다**(31/32 vs 32/32) | 예산이 `1.8·Tp` 라 **`Tp` 가 짧아질수록 자동으로 빡빡해진다**. 이 축의 최악 조건이 미탐색 | `Tp = 5 s` 층 16 ep × 3 시드 추가 | **~30 분** |
| **M11** | ★**`T_total < 1.8·Tp` 를 사전등록 게이트로 동결** | 지금은 **내가 사후에 건 자**다. 다음 배치 전에 얼려야 이 저장소가 반복해서 물린 "사후 규약" 실패를 피한다 | PREREG 한 줄 | 0 |

**게이트 제안** (사전등록에 넣을 것):
- **G-1**: M1 이 나오기 전에는 ⛔"100 ms 제약을 만족한다"를 **평균으로 주장하지 않는다**.
- **G-2**: M2·M6 이 나오기 전에는 ⛔"프레임당 비용" 을 **동화만으로 인용하지 않는다**.
- **G-3**: M5 가 나오기 전에는 ⛔실시간 문장에 **"실측" 이라는 단어를 붙이지 않는다**.
- **G-4**: M3 없이는 ⛔기여 ②c 를 **"측정됨"으로 표기하지 않는다**(현재 상태 = **미측정**).

---

## 6. 열람 범위 (= 주장 범위)

**연 것**: `tmp/W5_WHY_2026-08-19/{RESULT.md, verdict_w5.json, results.jsonl(640), analyze_w5.py,
run_w5.py}` + `tmp/W4_SOLVER_2026-08-19/run_w4.py`(재사용 896행 경유) · `tmp/bankmmse_port_2026-07-29/
{deployed_path_verify, rolling_refit, blind_vs_frozen, m3_field}.json, {verify_deployed_path,
run_rolling_refit}.py, PROTOCOL_ROLLING_REFIT.md` · `tmp/P1_real_path_2026-08-11/AUDIT_NEUTRAL.md §4.1` ·
`tmp/confirm2_2026-07-27/PREREG_CONFIRM2.md §5` · `wavesplat/{run.py, data_loader.py, real_data.py,
optimizer/sequential_update.py, optimizer/learned_init.py}` · 에피소드 HDF5 1건(스키마 확인).
**문헌 (직접 열람)**: arXiv:2508.03315 (abs + html 전문 질의 3회) · **JFM 975 A8 PDF 원문 40쪽 전수
텍스트 검색** · ★**Ocean Eng. 276:114212 (Kim+ 2023) PDF 원문 18쪽 전수 검색** (HAL 오픈액세스) ·
Crossref/OpenAlex/Semantic Scholar(서지 확인, S2 는 429 로 부분 실패).
**문헌 (병렬 검증 에이전트 경유)**: Qi 2017 MIT 박사논문 · Mérigaud & Ringwood 2019 · Mérigaud+ 2018
IFAC · Peña-Sanchez+ 2018 · Wang & Pan arXiv:2504.13943 — **인용문과 URL 을 함께 받았고 위 표에 그대로
전재**했다. ⚠**내가 직접 원문을 열지 않은 행이다**(2차 출처 표시).
**안 연 것**: Wang+ 2025 *Coastal Eng.* 197:104690(유료·OA 0건) · Qi+ 2018 JFM 838(유료) ·
Yao+ 2025 PRF · Fan+ 2026 TGRS(둘 다 유료) ⟹ §1.2 의 `UNVERIFIED` 행. W5 감사 문서 3종. 희소 층 자료(없음).
**런 0회.** 어떤 추정도 실행하지 않았고 어떤 파일도 수정하지 않았다(이 문서 제외).
