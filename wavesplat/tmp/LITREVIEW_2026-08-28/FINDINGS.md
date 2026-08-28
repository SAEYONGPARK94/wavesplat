# 문헌 실사 (임무 B) — FINDINGS
접근일: **2026-08-28**. 도구: WebSearch/WebFetch(Crossref·OpenAlex·Semantic Scholar API·Google Patents·PMC) + **저장소 보유 원문 PDF**.
규율: 각 항목에 **열람 범위**를 명시한다(전문 / 초록만 / 메타데이터만 / 2차 서술만).

---

## 0. 총평 — 심사 지적("원문 미열람 4라운드")에 대한 답

어제 종합자의 지적 *"저장소 밖 문헌은 전부 [미확인]"* 은 **절반만 맞았다**:
**Wang 2025 원문 PDF 는 저장소 안에 있었다** (`ref_paper/feng_wang_2025.pdf`, 12쪽 + 전문 추출
`tmp/litproj_2026-08-05/text/feng_wang_2025.txt` 1,365행). 오늘 그 전문과 결론을 실제로 열었고,
Bergamasco 계열·신규 2025–26 문헌·특허 축을 웹에서 확인했다. 결과:

- **저장소 기록과 원문의 불일치 = 2건** (§1-c 순차성 서술, §2-a 자료 출처 이름).
- **저장소 기록이 원문으로 확증된 것 = 4건** (Wang 결론 인용 · 정지 유지 선박 · 실측 12레코드/4고정관측소/스테레오 · DSGP-Net 3주장).
- **선점 발견 = 표현·알고리즘 조합 수준에서는 0건, 응용 공간 수준에서는 특허 2건 + DSGP-Net**.

---

## 1. Wang et al. 2025 — 최근접 경쟁자 ★열람 범위: **전문**

**서지 확정** (Crossref, 접근 2026-08-28):
Feng Wang, Qidan Zhu, Chengtao Cai, Xiaoyu Wang, Renjie Qiao,
"From points to waves: Fast ocean wave spatial–temporal fields estimation using ensemble
transform Kalman filter with optical measurement", *Coastal Engineering* **197 (2025) 104690**,
DOI 10.1016/j.coastaleng.2024.104690. ⟹ 저장소의 "197:104690" **정확**
(웹 검색 요약의 "104659" 는 오류였다 — Crossref 가 권위).
원문: 저장소 보유 PDF(`ref_paper/feng_wang_2025.pdf`)의 전문 추출본
(`tmp/litproj_2026-08-05/text/feng_wang_2025.txt`)을 열람. 소속: Harbin Engineering University.

### (a) 정확히 무엇을 했나
- **상태 = 격자 위 고도장**: *"the high dimensionality (O(10^4) in this study) of the state vector
  flattened from the 2D wave elevation grid"* — 스펙트럼이 아니라 **평탄화된 고도 격자**가 상태다.
- **예측 = WASSfast 의 분산관계 스펙트럼 전파를 채택**: *"The proposed method adopts WASSfast's
  prediction model"*.
- **관측 = 희소 3D 점을 균일 격자로 보간한 것**: *"interpolates sparse 3D points to a uniform grid
  of wave elevations as observations"*. 이상치는 퍼지 논리로 제거.
- **자료**: 공개 = **Guimarães et al. 2020 의 AA 03 레코드 하나**(*"the public Acqua Alta stereo data
  set (AA 03)"*) — **우리 실측 12레코드 중 하나와 동일 자료**. 실측 = 2024-03,
  **ZED2i 스테레오를 정지 유지 선박 갑판에**(*"on the deck of a vessel on position keeping"*,
  *"the vessel remained stable on the sea surface"*) ⟹ 저장소 bib note 의 "정지 유지 선박" **확증**.
- **하드웨어**: Intel i5-10400 + GTX 2060 노트북.

### (b) 실시간/순차 추적을 했는가 — **했다 (프레임 재귀)**
ETKF 는 정의상 프레임마다 예측–갱신을 반복하는 **순차 필터**다. 단계별 시간(Table 3, dense ratio
0.005): 전처리 76 + 점군 86 + 보간 6 + 앙상블 예측 18 + 퍼지 32 + ETKF 분석 42 ≈ **260 ms/프레임**
(≈3.8 Hz). 고밀도(0.100)에서는 ≈425 ms 이고 *"at the higher dense ratio its computational efficiency
is lower than CNN mode of WASSfast"* 라고 스스로 적었다. 예보(lead) 평가는 없고 **재구성(nowcast)만** 평가.

### (c) ⛔저장소 기록과의 불일치 — **"순차 추적이 문헌의 틈" 은 성립하지 않는다**
MEMORY 08-15 묶음: *"문헌에 틈은 「실시간 순차 추적」 하나"* — **원문과 어긋난다.**
Wang 2025 는 재귀 필터고(위), WASSfast 도 초록이 *"**continuously update** the 3D surface and 2D
wavenumber spectrum estimations"* 라 적는다(§2-b). **순차성·실시간은 이 계열의 공통 속성이지 우리
차별점이 아니다.** 살아있는 차별점은 **표현**이다: ①연속(비격자) 파수 원자 ②희소 성분 매개화
③학습된 사전 ④작동 경계의 측정. 현행 thesis 의 Wang 대비 문장("차이는 표현이다")은 **옳게 서 있다** —
낡은 것은 메모리 쪽이다.

### (d) ★"결론에서 우리 접근 방향을 명시 지목" — **사실 (축자 확보)**
결론 마지막 문장 축자:
> *"Future work could focus on exploring appropriate entry frequencies as state vectors to reduce
> state dimensionality, while preserving prior correlations through spectral–spatial transformation
> in the mode of direct sparse assimilation."*

= **주파수 상태벡터 + 스펙트럼–공간 변환 + 직접 희소 동화** — 우리 접근의 방향과 일치. 저장소 bib
note 의 요약과 thesis 본문 서술은 **정확**하다. ⚠단 이 문장이 지목하는 것은 **표현의 방향**이지
**학습된 사전**이나 **연속 파수**가 아니다 — novelty 문안에서 그 둘까지 지목받은 것처럼 쓰지 말 것.

### (e) ★신규 발견 2건 (저장소 기록에 없던 것)
1. **Wang 은 직접 희소 동화를 이미 실험했다** — 격자 상태에 희소 관측을 그대로 동화하는 팔을
   측정했고 *"direct assimilation of sparse measurements in ETKF shows limited improvement at
   original sparse locations"* 로 **보간 관측보다 나쁘다**고 결론냈다. ⟹ 우리 위치지음은
   *"그들이 향후 과제로 미룬 것"* 이 아니라 **"그들이 격자 상태로 시도해 실패를 측정한 것을
   스펙트럼 상태로 구현한 것"** 이 더 정확하고 더 강하다.
2. **공개 코드가 반쪽이다** — https://github.com/fengHEU/Wave-ETKF 존재(GPL-3.0, 커밋 2개)하나
   **ETKF 핵심은 저작권 등록 대기로 비공개**(*"full open-source release considered after copyright
   registration"*). ⟹ `B2_WANGFIX` 재현 실패 서사에 단서 추가 가능: **참조 구현 대조가 원천적으로
   불가능했다**(우리 재현이 자기검정에서 죽은 것과 별개의, 문서화 가능한 사실).

---

## 2. Bergamasco 계열 — 실측 자료의 출처 ★열람 범위: 자료 논문 **전문**(PMC) · WASSfast **초록 축자**

### (a) ⛔저장소 기록과의 불일치 — **자료 출처의 이름이 틀렸다 (메모리/임무문만; thesis 는 옳다)**
임무문·메모리의 *"Bergamasco et al. 2021 = 우리 실측 자료의 출처"* 는 **부정확**하다.
- 실제 자료 논문 = **Guimarães, P.V., Ardhuin, F., Bergamasco, F., Leckler, F., Filipot, J.-F.,
  Shim, J.-S., Dulov, V., Benetazzo, A., "A data set of sea surface stereo images to resolve
  space-time wave fields", *Scientific Data* 7:145 (2020)**, DOI 10.1038/s41597-020-0492-9
  (전문: https://pmc.ncbi.nlm.nih.gov/articles/PMC7228941/, 접근 2026-08-28). Bergamasco 는 3저자.
- **Bergamasco et al. 2021** = WASSfast **방법** 논문(*Computers & Geosciences* 147:104666).
- ✅`thesis_refs.bib` 는 이미 옳게 분리해 인용한다(`Guimaraes2020Dataset` = 자료,
  `Bergamasco2021WASSfast` = 방법). 고칠 곳은 **메모리 인덱스와 구두 서술 습관**이다
  (`data/real_bergamasco/` 디렉터리명이 오독을 계속 재생산한다).

### (b) ✅자료 구성은 저장소 서술과 **일치** (원문 확증)
원문 축자: *"Seven recorded at the Marine Hydrophysical Institute platform in the Black Sea; three
recorded from the Acqua Alta oceanographic research tower in the Adriatic Sea; one recorded at the
Sogheongho Ocean Research Station in the Yellow Sea; one recorded at La Jument lighthouse in the
Iroise Sea."* ⟹ **12 레코드 · 관측소 4곳(BS·AA·YS·LJ) · 전부 고정 구조물 · 이동/선박 0건**.
센서 = **스테레오**: *"Two BM-500GE JAI cameras"*, 2048x2456 5MP, 8-bit. 해상상태 범위:
BS Hs 0.30–0.55 / AA 1.33–1.99 / YS 1.94 / **LJ 10.03 m (Tp 13.33 s)**.
⟹ CLAUDE.md 의 "실측 12 레코드, 고정 구조물 4곳" · thesis 의 "전부 고정 관측소·스테레오" **확증**.
(⚠LJ01 Hs 10.03 은 08-12 금지목록의 "사용 전 확인" 단서가 여전히 붙는다 — 여기서는 원논문 값의 전재만.)

### (c) WASSfast (Bergamasco et al. 2021) — 초록 축자 확보 (Semantic Scholar API, 접근 2026-08-28)
> *"we propose a new method to directly estimate the sea surface spectrum **over time** from a
> sequence of stereo frames. We exploit the frequency dispersion relation of gravity waves and the
> (Non-uniform) Fast Fourier Transform to **continuously update** the 3D surface and 2D wavenumber
> spectrum estimations given a sparse set of matching features"* — 성능 *"more than an order of
> magnitude"* 빠름 (thesis 의 "약 10배" 서술과 일치).

⛔**thesis 의 차별화 문장 하나가 이 초록과 긴장한다**: `thesis_main.tex` §최근접 선행
*"WASSfast 는 … 프레임별 재구성을 목표로 한다. 본 연구는 … 프레임 간 순차 추적을 수행한다"* —
WASSfast 도 초록 스스로 **시간 축 연속 갱신**을 명시한다. 출력이 프레임별 격자 고도장인 것은 맞지만,
**"순차 추적" 은 대비축이 될 수 없다**. 대비는 ①연속 파수 원자(WASSfast 는 FFT/NFFT 파수 격자 —
⚠이 격자 서술은 초록+Wang 2025 의 2차 서술 기반, WASSfast 본문은 미열람) ②희소 성분 매개화
③학습된 사전으로 한정할 것. **수리 권고: 해당 문단의 대비 문장 1곳.**
2차 서술(Wang 2025 전문에서): WASSfast 는 예측 스펙트럼과 희소 점 고도의 균형을 **제약 최소화**로
풀고, ill-conditioned 행렬 탓에 반복 경사하강을 쓴다. CNN 모드(Pistellato et al. 2021, 256x256 격자
고정)는 희소→조밀 직접 사상이다.

---

## 3. 2025–2026 신규 문헌 — 선점 탐색 ★핵심 질문: 우리를 선점한 것이 있는가

### (a) **DSGP-Net (2026)** — 가장 가까운 신규 위협 ★열람 범위: 메타데이터(Crossref)+초록 요약(검색)
Feng Wang, Xiaoyu Wang, Qidan Zhu (같은 Harbin 그룹), "DSGP-Net: A dual-space guided propagation
network for ocean wave height estimation from sparse stereo measurements", *Ocean Engineering*
**356 (2026) 125086**, DOI 10.1016/j.oceaneng.2026.125086.
- 검색 요약 기준: 2단계 CNN(영상 좌표 희소 고도 + 카메라 영상 → 신뢰 인지 조밀 표현 → 이중 공간
  융합), 고정 플랫폼에서 SOTA 대비 RMSE **14–23%** 개선, **이동 선박**에서 재구성, **실시간** 주장,
  희소 피복 5–20%. ⟹ 저장소 bib note("희소 측정 + 이동 선박 + 실시간을 전부 주장") **일치 확인**.
- **선점 판정**: **응용 공간(희소 스테레오→조밀, 이동 플랫폼, 실시간)은 선점. 표현은 아니다** —
  CNN 조밀 완성이지 스펙트럼 상태·순차 스펙트럼 추적·학습된 스펙트럼 사전이 아니다. 오히려 초록이
  *"temporal sequences … fail on moving platforms"* 라며 **시간축 방법에 반대로 위치**를 잡는다.
- ⚠**시간 압박 신호**: Wang 2025 결론의 방향(스펙트럼 상태 + 직접 희소 동화)을 낸 그룹이 2026 에
  이미 후속을 냈다. 그들이 스펙트럼 상태 버전을 내면 우리 위치지음의 반이 사라진다.

### (b) 조합 수준 선점 — **발견 0건** (탐색 범위 명시)
검색 8회(쿼리: Gaussian splatting waves / LiDAR 파랑 재구성 / gridless·off-grid 파수 / learned prior
phase-resolved / wave packet·spectral Gaussian 표현 / DSGP-Net / WASSfast / 특허 2회)에서
**"연속 파수 스펙트럼 원자 + 학습된 사전 + 희소 3D 점 순차 추적"의 조합을 가진 논문은 나오지
않았다**. ⚠부재 증명이 아니다 — 열람 범위는 위 쿼리의 1페이지 결과와 개별 확인 항목뿐이다.
개별 확인(전부 비선점 판정):
- **Ehlers 계열**(2305.11913 U-Net/FNO 희소→조밀; PINN DA, PRF 10:094901 2025; PINO 2508.03315) —
  격자/신경장 기반, 저장소 기보유(`ref_paper/`).
- **Çelik 2026** (*Ocean Eng.* 349:124166, OpenAlex 초록): 부이 관측망 SWH **통계장** 완성 —
  위상해상 아님.
- **Jebari et al. 2025** (*Ocean Eng.* 120958, OpenAlex 초록): 부유 플랫폼 스테레오 **계측** 합성
  환경 검증 — 추정 표현 아님.
- **Gaussian splatting 계열**(WaterGS·OceanSplat 등 2025–26): 전부 **수중 장면** 재구성 — 해면
  파랑장 아님.
- **LiDAR 파랑**(*Ocean Eng.* S0029801825026782, Pingtan): 국소 곡면 적합으로 파랑 **특성 분석** —
  위상해상 장 추정 아님.
- **선박 스테레오 관측**(*Sensors* 26(3):993, 2026): 관측 실험 — 추정 방법 아님.

### (c) ⚠표현 수준의 고전 선행 — "최초의 연속 파수 표현" 을 막는 논문
**Charnotskii, "Sparse Spectrum Model of the Sea Surface", OMAE 2011** (초록만):
*"each surface realization contains a finite, possibly random, number of sinusoidal components with
random frequency, phase and amplitude"* — 성분 수가 격자가 아니라 해상상태로 결정되는 **연속 파수
희소 사인 성분 모델**. 시뮬레이션/통계 모델이지 추정기가 아니므로 우리를 죽이지 않지만,
⛔**"연속 파수 희소 표현 자체가 신규" 라는 문장은 이 선행 앞에서 못 쓴다**
(금지목록의 기존 항목 "성분집합 prior 신규성 금지" 와 정합).

---

## 4. 특허 축 — US 12,054,225 대체 사례 ★열람 범위: Google Patents 페이지 2건 열람 + 검색

✅**대체 사례를 찾았다. 응용 공간의 진짜 선행 특허가 존재한다**:
- **US 10,677,891** "Systems and methods for wave sensing and ship motion forecasting using
  multiple radars" + **US 10,684,353** "… with scrolling forecast displays" — **Applied Physical
  Sciences Corp** (= FutureWaves), 우선일 **2016-09-16**, ONR 계약 지원. EP 대응
  **EP3512768A2** 열람: 선박 레이더에서 **이산 주파수·방향별 진폭+위상의 파 성분**을 회귀로 추정해
  위상해상 해면을 만들고 **1–3 s 주기로 갱신**하며 선체 운동을 결정론적으로 예보.
  ⟹ **"선박 원격 센서 → 위상해상 파랑장 → 운동 예보" 응용 공간은 2016 년에 특허로 선점돼 있다.**
  단 센서는 레이더(점군 아님), 표현은 이산 성분, **학습된 사전·연속 파수·희소 점군 동화 청구 없음**
  (분산관계·희소 측정 언급 없음 — EP 문서 기준).
- US 8,494,697 B2 (Boeing, 2011, 열람): 자율 "wave boat" 함대의 물리 센서로 모선 운동 예측 —
  파랑장 재구성 청구 **없음**.
- US 2012/0130569 · US 2014/0114509 (검색만): 3D 해면 영상→파면 검출·DSWP 계열.
- 재확인: **US 12,054,225 는 파도장 특허가 아니다**(Candela, 수중익선 제어 — 08-19 판정 유지,
  오늘 재열람은 안 함).

---

## 5. novelty 판단에 미치는 영향 (종합)

1. **"무조건 옹호 0표" 는 원문 열람 후에도 유지된다** — 순차성·실시간·희소 점군 동화·응용 공간
   전부 선행이 있다(WASSfast·Wang 2025·FutureWaves 특허·DSGP-Net).
2. **살아남는 novelty 는 조합이다**: 연속 파수 희소 스펙트럼 원자(단 표현 자체는 Charnotskii 가
   선행) + **학습된 사전**(이 계열 어디에도 없음) + **작동 경계의 측정**(우리만 실측 음성을 공시).
   Wang 2025 결론이 이 방향을 축자로 호명한다는 위치지음은 **원문으로 확증**됐고, §1-e-1 의
   신규 사실(그들이 격자-상태 직접 동화를 재보고 실패를 측정했다)로 **더 강하게** 쓸 수 있다.
3. **수리할 곳 2곳**: ①thesis WASSfast 문단의 "순차 추적" 대비 문장(§2-c) ②메모리/인덱스의
   "Bergamasco 2021 = 자료 출처" 서술(§2-a). 임무문 소거표는 무관(문헌은 판정에 안 들어감).
4. **시간 압박**: Harbin 그룹이 같은 문제에서 연 1편 이상을 내고 있고(2025 CE → 2026 OE),
   Wang 2025 의 향후 과제가 곧 우리 접근이다. **선점당하기 전에 내는 것 자체가 리스크 관리다.**

## 출처 URL (접근일 전부 2026-08-28)
- Wang 2025: 저장소 PDF `ref_paper/feng_wang_2025.pdf`(전문) · Crossref DOI 10.1016/j.coastaleng.2024.104690 · https://www.sciencedirect.com/science/article/abs/pii/S0378383924002382 (403, 미열람) · 코드 https://github.com/fengHEU/Wave-ETKF
- Guimarães 2020: https://pmc.ncbi.nlm.nih.gov/articles/PMC7228941/ (전문) · DOI 10.1038/s41597-020-0492-9
- WASSfast: Semantic Scholar API DOI 10.1016/j.cageo.2020.104666 (초록 축자) · https://sites.google.com/unive.it/wass/software/wassfast
- DSGP-Net: Crossref DOI 10.1016/j.oceaneng.2026.125086 · https://www.sciencedirect.com/science/article/abs/pii/S0029801826009200 (미열람)
- Charnotskii 2011: https://asmedigitalcollection.asme.org/OMAE/proceedings-pdf/OMAE2011/44380/773/4575170/773_1.pdf (초록 수준)
- 특허: https://patents.google.com/patent/EP3512768A2/en (열람) · https://patents.google.com/patent/US8494697B2/en (열람) · US 10,677,891 · US 10,684,353 (검색 확인)
- 기타: OpenAlex (Çelik 10.1016/j.oceaneng.2025.124166 · Jebari 10.1016/j.oceaneng.2025.120958)
