# A1 — 선행연구·선점(prior art) 심사 소견서 (2026-08-24)

> **지위**: 나는 이 연구의 novelty 주장을 **문헌 대비로 죽이는** 쪽에 선다.
> **런 0회.** 읽고, 로컬 PDF 를 추출하고, 웹으로 서지를 검증했다.
> ⚠**선행 토론과 중복 금지 지시를 지켰다** — 아래 §0.3 에 *"내가 되풀이하지 않은 것"* 을 명시한다.

---

## 1. 판정 한 줄

> ### ⛔`PRIOR-ART-FAIL / GAP-IS-NARROWER-THAN-PRINTED`
> **문헌 대비로 남은 novelty 는 한 단어(*연속 파수 원자*)와 한 형태(*개입으로 측정된 작동 경계*)뿐이고,
> 원고에는 `US 12,054,225` 부류의 인용 오류가 최소 3건 살아 있으며 — 그중 하나(`Desmars = ETKF`)는
> 제2장 기준선 계보 전체를 지탱한다.**

⚠**단서**: 내 `FAIL` 은 **문헌 처리**에 대한 것이지 **측정**에 대한 것이 아니다.
`B-12` 의 산술에는 내가 댈 흠이 없다(내가 재도출하지 않았으므로 `[인용]`; B2 §10.1 이 독립 재도출로 전건 일치).

---

## 0. 열람 범위 (= 주장 범위)

### 0.1 ✅직접 열고 확인한 것
- **논문**: `thesis/thesis_main.tex` **제1장 전문**(`:462-825`) · **제2장 전문**(`:826-1232`) ·
  `:1878-1890`(prune--grow) · 전수 정규식 스캔(키워드 24종 · citation key 14종 계수).
- **서지**: `thesis/thesis_refs.bib` **전문**(382행, `@` 항목 34건).
- **로컬 1차 PDF (내가 `pypdf` 로 추출)**: `wavesplat/ref_paper/feng_wang_2025.pdf`(표제부 + 결론 축자) ·
  `wang_pan_2021_jfm.pdf`(표제부) · `wang_pan_2022_jfm.pdf`(표제부) · `tang_bhaskar_recht_2013.pdf`(표제부).
- **저장소 기록**: `scripts/generate_labmeeting_2026_05_13.py:540-560, :1528` ·
  `/home/vexit/wavesplat/ref_paper/literature_survey_cs_spectral.md`(§1·§2·§3·§9·§10 + 참고문헌) ·
  `memory/reference_bergamasco_stereo_dataset.md` 전문 ·
  `tmp/S16_BASELINE_2026-08-14/{run_s16.py:66,210-211, PREREG.md:92}` ·
  `tmp/STATUS_2026-08-23/STATUS.md` 전문 · `CLAUDE.md` · `MEMORY.md`(RESUME 2개 + 08-22~24) ·
  `memory/reference_citation_bans.md`(선두 + 문헌 grep).
- **선행 토론**: `DEBATE_NOVELTY_PHD_2026-08-21/B2_external_examiner.md` **전문(715행)** ·
  `R2_5_literature.md` **전문** · `R3_2_novelty.md`(판정표 + §0) · `B3_advisor.md`(문헌 grep).
- **웹 검증(`VERIFIED`)**: Desmars 2020 *Ocean Eng.* 212:107659 · Desmars 2023 *JFM* 975 A8(초록 전문 + 방법) ·
  Ehlers *Ocean Eng.* 288:116059(U-Net/FNO) · Wang & Pan 2021 *JFM* 918 A19(초록) · Wang & Pan 2025 arXiv:2504.13943(초록) ·
  Bergamasco et al. 2021 *Computers & Geosciences* 147:104666(서지 + WASS 프로젝트 페이지 축자) ·
  Pistellato et al. 2021 *Remote Sensing*(요약).

### 0.2 ⛔열지 않은 것 (판정 시 `[인용]` / `UNVERIFIED`)
- **논문 제3–6장**(제3장 `:1878` 주변만) ⟹ 제어·수조·실측 절 문언에 대한 판정 없음.
- **Bergamasco 2021 C&G 본문·초록 축자** — ScienceDirect 접근 실패. §3 의 *"NUFFT 균일 격자"* 구분은
  **키워드·프로젝트 페이지 기반 `ARGUED`** 이지 축자가 아니다.
- **DSGP-Net 본문**(R2_5 가 5경로 실패를 기록) · **Desmars 2020/2023 본문**(초록·서지만).
- `tmp/B*` 원자료 — 나는 **재집계를 하지 않았다**. 수치는 전부 `[인용]`.

### 0.3 ★내가 **되풀이하지 않은** 선행 지적 (지시 준수 증거)
DSGP-Net 약어 오기(B2 §5.3 — ✅**이미 수리됨**, `thesis_refs.bib:342-355` · `thesis_main.tex:1205-1217`) ·
VARPRO 선점(B2 §5.1 — 논문이 이미 인정, `:1225-1228`) · Kim 2023 `1.8T_p` 처리(A3·R2_5 §5 — 논문 처리 옳음, `:1218-1223`) ·
Wang 2025 미래과제 문장의 절 단위 분해(R3_2 §2.3) · AOR BayesNF 위협 하향(R2_5 §4) ·
`etkf_wang2025.py` 팔의 무기술(R2_5 §6.1) · Wang&Pan 2025·Yao 2025 의 이동선 점유(B3 §5).
⟹ **아래는 전부 그 뒤에 남아 있던 것이다.**

---

## 2. Q1 — 2026-08-24 시점에 **남은 novelty** 와 *"누가 이미 했는가"*

각 줄 = **한 문장 주장 + 선점자 + 등급**. ⛔=죽음, ⚠=조건부, ★=살아 있음.

| # | 논문이 주장하는(또는 주장할 수 있는) novelty | **누가 이미 했는가** | 판정 |
|---|---|---|---|
| **N-1** | 격자 없이 희소 3D 점군에서 파랑장을 **실시간** 복원 | ★★★**Bergamasco et al. (2021)** *Computers & Geosciences* **147**:104666 (WASSfast) — WASS 공식 페이지 축자: *"exploits the **linear dispersion relation and sparse feature triangulation** to resolve sea-surface elevation **in real-time**"*; C&G 키워드에 *non-uniform fast Fourier transform · 3D spectrum* | ⛔**DEAD** |
| **N-2** | 희소 점을 **다중 프레임에 걸쳐 분산관계로 묶어** 표면을 복원 | ★★**Pistellato et al. (2021)** *Remote Sensing* 13:3780 — *"**embed the linear dispersion relation into the model itself to physically relate the sparse points observed at different times**"*(연속 3프레임) | ⛔**DEAD** |
| **N-3** | 희소 3D 광학 점군의 **앙상블 칼만 동화** | **Feng Wang et al. (2025)** *Coastal Eng.* 197:104690 — ✅**논문이 이미 인정**(`:1191-1201`) | ⛔**DEAD**(자백됨) |
| **N-4** | ★논문이 *"문헌에서 주장하는 **유일한 틈**"* 이라 못 박은 **시간 축(순차 추적·전방 예보)** (`:1216-1217`) | ⛔**선점 3중**: **Wang & Pan (2021)** *JFM* **918**:A19 (EnKF--HOS, 순차 동화 + *"arbitrarily long forecast time with sequentially assimilated data"*, **실측 레이더 검증**) · **Wang & Pan (2022)** *JFM* **949**:A31 · **Desmars et al. (2023)** *JFM* 975:A8 (HOS--OCM 예보) · **Kim et al. (2023)** | ⛔**DEAD** — ★그 문장은 **DSGP-Net 한 편**에 대해서만 참이다 |
| **N-5** | 연속(off-grid) 파수를 상태로 두는 희소 스펙트럼 추정 | **Tang, Bhaskar, Shah & Recht (2013)** *"Compressed Sensing Off the Grid"* · **Candès & Fernández-Granda (2014)** · **Yang & Chu (2019)** 2D decoupled ANM — ★**앞의 것은 PDF 가 `wavesplat/ref_paper/tang_bhaskar_recht_2013.pdf` 에 2026-06-24 부터 있다** | ⛔**DEAD**(방법으로서) |
| **N-6** | 변수 투영(VARPRO)으로 진폭·위상 소거 | Golub & Pereyra (1973) — ✅논문이 인정(`:1225-1228`) | ⛔**DEAD**(자백됨) |
| **N-7** | 파랑장 추정 → **선체 운동 예보/제어** 통합 | **Wang & Pan (2025)** arXiv:2504.13943 (EnKF--HOS--CMI, 파랑+선체운동 **동시 예보**) `[선행 지적: B3 §5]` | ⛔**DEAD** |
| **N-8** | ⚠**분산관계를 원자 사전(atom dictionary)에 하드 제약으로 넣은 2D 연속 파수 추정** | ⚠**내 열람 범위에서 선점 문헌을 찾지 못했다.** ANM 계열은 물리 제약이 없고(`literature_survey_cs_spectral.md §3.2` 가 *"Adding dispersion constraint to ANM is non-trivial"* 이라 적는다), WASSfast 는 NUFFT **균일 파수 격자**로 보인다(`ARGUED`) | ★**ALIVE — 단 "한 단어"** |
| **N-9** | ★**자기 방법의 작동 경계를 개입 실험으로 확정한 것**(기여 ①③④⑥) | ⚠**경쟁자 3편 중 어느 것도 자기 방법의 실패 경계를 개입으로 재지 않았다** — 단 이는 내가 읽은 범위(Wang 2025 전문 · Bergamasco 2021 초록/페이지 · DSGP-Net 서지)에서의 관찰이고 **부재 증명이 아니다** | ★**ALIVE — 형태로서** |

### ★내 계수
- **문헌 대비로 죽은 축 = 7** (N-1~N-7).
- **살아 있는 것 = 2**, 그리고 그 둘은 **성질이 다르다**:
  - **N-8** = *알고리즘 차별점*이지만 **한 단어**다(*연속* 파수 vs *격자* 파수). 논문은 이것을 그렇게 좁게 쓰지 않는다.
  - **N-9** = *방법론적 기여*이고, **B2 §11(c)·R3_2 ⓑ 와 같은 결론**이다. 나는 여기에 문헌 근거 하나를 더한다(§5).

⟹ ★***"격자 비의존"·"희소 3D 점군"·"실시간"·"분산관계 제약"·"순차 추적" 은 개별로도 조합으로도 못 산다.
2021 년에 — 우리 실측 자료를 만든 바로 그 그룹이 — 그 조합을 발표했다.***

---

## 3. Q2 — Wang et al. 2025 와의 관계: **그 위치지음이 심사에서 버티는가**

### 3.1 ✅**사실관계는 버틴다 — 내가 로컬 PDF 에서 다시 확인했다**

`wavesplat/ref_paper/feng_wang_2025.pdf` 결론 마지막 문장, **내 `pypdf` 추출 축자**:
> *"Future work could focus on exploring appropriate entry frequencies as state vectors to reduce state
> dimensionality, while preserving prior correlations through spectral--spatial transformation in the mode
> of direct sparse assimilation."*

논문 `:1193-1201` 의 처리 — *"부분적으로만 겹친다"*, *"본 연구가 그 문헌을 따라 착상했다는 서술은 사실이 아니며"*,
*"발표 우선권은 그들에게 있다"* — 는 **정직하고 정확하다.** 이 축에서 나는 반대하지 않는다.

### 3.2 ⛔**그러나 위치지음이 문서에 통합되지 않았다 — 이것이 심사에서 깨진다**

**`MEASURED`(내 계수, `thesis_main.tex` 전수 grep)**:

| citation key | 본문 등장 횟수 | 등장 위치 |
|---|---|---|
| `Wang2025PointsToWaves` | **1** | `:1191` |
| `DSGPNet2026` | **1** | `:1205` |
| `Kim2023RealTime` | **1** | `:1218` |
| `GolubPereyra2003` | **1** | `:1227` |
| `Desmars2020` | 4 | `:562, :564, :936, :3313` |
| `Desmars2023` | 6 | `:562, :565, :589, :939, :1087, :3313` |

⟹ ★***2025–26 문헌 갱신 전체가 `:1184-1232` 한 절 안에 갇혀 있고, 각 1회씩만 인용된다.***
`bib` 의 `year >= 2025` 항목은 **2건**(내 grep)이고, 이는 `R2-4` 가 요구한 5편에 여전히 미달한다`[인용: R3_2 N-7]`.
제2장 §격자 기반(`:930-942`)과 §격자 비의존(`:1043-1105`)은 **갱신 전 상태 그대로**다.

⟹ **심사위원이 순서대로 읽으면**: §2.2 에서 *"기준선 계보 = Desmars"* 를 배우고, 40쪽 뒤 §최근접 선행에서
*"사실 같은 문제·같은 기준선 계열을 먼저 발표한 사람은 Wang 이다"* 를 읽는다. **두 서술이 서로를 모른다.**

### 3.3 ⛔⛔**그리고 위치지음의 전제가 틀렸다 — "최근접 선행 연구" 절이 최근접자를 담고 있지 않다**

Wang 2025 자신이 **자기 incumbent 를 명시한다**(내 추출, 결론 축자):
> *"The recursive uncertainty propagation chain in ETKF significantly enhances computational efficiency
> compared to the constrained optimization framework of **WASSfast (PU)**"* ·
> *"at the higher dense ratio its computational efficiency is lower than **CNN mode of WASSfast**"*

그 **WASSfast** 가 §2 N-1/N-2 의 Bergamasco/Pistellato (2021) 이고, **우리와 Wang 보다 가깝다**:
Wang 은 *"sparse 3D points **interpolated to a uniform grid** as measurements"*(초록 축자)로 **보간 격자**를 쓰는데,
WASSfast 는 **희소 점 + 분산관계를 직접** 쓴다 — 즉 ***우리가 Wang 과 갈린다고 적은 바로 그 축에서 WASSfast 는 우리 쪽에 있다.***

⟹ ★**Q2 답**: *"동시대 독립 연구 · 우선권은 그들"* 이라는 **문장 자체는 버틴다.**
⛔**그러나 그 절이 하는 일 — "최근접 선행을 세우고 우리 위치를 잡는다" — 은 실패한다.**
**최근접자가 빠져 있고**, 그 사실을 **경쟁자 자신의 결론이 두 줄로 알려준다.**
심사에서 이것은 *"틀린 위치지음"* 이 아니라 ***"불완전한 문헌 조사"*** 로 채점되며, 후자가 더 무겁다.

### 3.4 ⚠**내 진영에 불리한 정정 — 원장과 논문이 어긋난다**

내 과제 브리핑과 `MEMORY.md` RESUME(2026-08-22 항목)는 논문이 *"'독립 착상'이 아니라 '그 방향의 구현'으로
위치지음"* 했다고 적는다. ⛔**논문은 그 반대로 쓰여 있다** — `:1197-1201` 이 *"따라 착상했다는 서술은 **사실이
아니며** … 같은 방향을 **독립적으로** 취한 동시대 연구"* 라 적고, 그 옆에 `[범위, 2026-08-22 정정]` 표시가 있다.
⟹ ★**원장이 논문보다 낡았다.** 이는 오늘 `STATUS_2026-08-23/STATUS.md:185-189` 이 스스로 자백한
*"원장의 '다음' 항목이 완료를 따라가지 못했다 … 결함은 측정이 아니라 **기록의 지연**"* 의 **문헌 축 재발**이다.

---

## 4. Q3 — `US 12,054,225` 부류의 오류가 **현재 원고에 남아 있는가**: ⛔**최소 3건, 그중 하나는 하중 부재**

> **부류의 정의**(저장소가 두 번 물린 것): ***인용된 문헌이 실제로 하는 일과, 논문이 그 문헌에 귀속시킨 일이 다르다.***

### ★E-1 ⛔⛔⛔**`Desmars 2020/2023 = ETKF` 가 거짓이다 — 제2장 기준선 계보 전체가 여기 서 있다**

**논문의 서술** (`MEASURED`, 축자):
- `:562` `Ensemble Transform Kalman Filter (ETKF)~\citep{Desmars2020, Desmars2023}는 …`
- `:564` `\citet{Desmars2020}이 X-band 레이더로부터의 ETKF 기반 재구성을 제안하였으며,`
- `:936` (제2장, 같은 문장 반복)
- `:939-940` `\citet{Desmars2023}이 **이 프레임워크에** 비선형 확장을 적용하여 …`
- `:941-942` `본 연구의 ETKF-grid baseline은 **위 레이더 기반 정식화를 LiDAR 관측에 이식한 결과이다.**`
- `:589`·`:1087` `비선형 ETKF 확장~\citep{Desmars2023}`

**실제** (`VERIFIED`):
1. **Desmars et al. (2020)** *Ocean Eng.* 212:107659 제목 = *"Experimental and numerical assessment of
   deterministic nonlinear ocean waves prediction algorithms using **non-uniformly sampled wave gauges**"*
   — **논문 자신의 `bib:44-47` 이 이 제목을 그대로 싣는다.** 내용은 **비선형성이 다른 파랑 모델 3종의 예측 성능 비교**다.
   ⟹ **X-band 레이더도 아니고 ETKF 도 아니다.**
2. **Desmars et al. (2023)** *JFM* 975:A8 — 내가 Cambridge 초록 페이지에서 확인:
   방법은 **HOS--OCM(HOS--observation coupling method)** 이고 잔차를 **LGMRES 로 최소화하는 변분/최적화**다.
   ⟹ **앙상블 칼만 필터가 아니다.** (초록은 *"only the third-order HOS solution predicts …"* 로 끝난다.)
3. ★**앙상블 칼만을 위상 분해 파랑에 처음 결합한 것은 Wang & Pan (2021)** *JFM* 918:A19
   (`wavesplat/ref_paper/wang_pan_2021_jfm.pdf` 표제부, 내가 추출: *"Phase-resolved ocean wave forecast with
   ensemble-based data assimilation", Guangyao Wang & Yulin Pan*).

★★★**결정적 — 저장소 자신의 기록이 이미 옳게 적어 두었다**(`MEASURED`, 축자):
`scripts/generate_labmeeting_2026_05_13.py:1528`
> *"**Desmars family (2018, 2020, 2023) 는 LIDAR 와 optical 센서 + variational + HOS-OCM 의 발전 흐름**이고,
> Wang-Pan / Wang family (2021, 2022, 2025) 는 EnKF-HOS 의 발전입니다"*

같은 파일 `:551`
> *"EnKF … 원래 기상학 데이터 동화에서 출발한 기법인데 **Wang & Pan 2021 이 wave estimation 에 처음 적용**했습니다."*

⟹ ★***2026-05-13 에 옳게 적은 계보가, 학위논문에서 뒤바뀌었다.*** 그리고 **검증 비용은 0이었다** —
`desmars_2018_hal.pdf`·`desmars2020_hal.pdf`·`desmars2023_jfm.pdf` 세 편이 전부
`wavesplat/ref_paper/` 에 **2026-06-24 부터** 있다(내 `ls -la`).

⛔**파급**: `:941-942` 는 **존재하지 않는 정식화를 이식했다**고 적는다. 논문의 주 기준선(`ETKF-grid`)에
**유효한 문헌 출처가 없다.** 그리고 실제 출처가 될 자격이 있는 유일한 문헌이 **Wang 2025 = 최근접 경쟁자**다
— 그것을 §2.2 가 아니라 §최근접 선행에 따로 두었으므로, **경쟁자가 기준선의 원조라는 사실이 논문 어디에도 없다.**

### ★E-2 ⛔⛔**최근접 경쟁자의 논문 제목이 틀렸다** — DSGP-Net 오기와 **정확히 같은 부류의 이웃 항목**

| | |
|---|---|
| `thesis_refs.bib:331-332` | *"From points to waves: **sea surface elevation reconstruction with an ensemble transform Kalman filter and optical measurement**"* |
| **로컬 PDF 1쪽 축자**(내 추출) | *"From points to waves: **Fast ocean wave spatial–temporal fields estimation using ensemble transform Kalman filter with optical measurement**"* |

또한 `author = {Wang, Feng and others}` — 실제는 **5인**이고 교신저자는 **Qidan Zhu**(PDF 표제부).
⟹ ★***2026-08-23 에 DSGP-Net 제목을 고치면서(`bib:352-355` 에 정정 주석까지 달면서) 바로 위 11줄의 항목은
열지 않았다.*** PDF 는 `2026-06-24` 부터 디스크에 있었다. **B2 §5.2 도 올바른 제목을 웹에서 인용했으나 `bib` 와
대조하지 않았다** — 즉 이 오류는 **세 차례의 심사를 통과했다.**

### ★E-3 ⚠**`Cummins2017` — `bib` 자신의 각주가 본문을 부인한다**

- 본문 `:1174-1175`: *"\citet{Cummins2017}이 레이더/LiDAR 기반 파도 미리보기가 **단기 선박 운동 예측을
  개선함을 보였다**"*
- `bib:291` 제목: *"The role of wave prediction in the control of **wave energy converters**"*
- `bib:296` 각주: *"Methodology **applicable to** ship motion prediction."*

⟹ **파력 발전기 논문을 선박 운동 실증으로 승격**시켰고, **`bib` 각주가 그 승격을 스스로 부인한다.**
(⚠나는 이 논문 원문을 열지 않았다 — 판정 근거는 **논문 내부의 자기모순**이지 원문 대조가 아니다.)

### ★E-4 ⚠**무조건 부재 주장** — 제2장이 제1장의 완충을 잃었다

- `:566` (제1장): *"… 적용한 사례는 **본 연구의 알기로** 보고된 바가 없으며"* ✅완충 있음
- `:940` (제2장): *"선박 탑재 LiDAR 관측에 ETKF를 직접 적용한 사례는 **보고된 바가 없으며**"* ⛔완충 없음

저장소 규율이 명시적으로 금지하는 **부재 증명** 형태다. 그리고 E-1 때문에 이 문장은 **틀린 전제 위에 서 있다**.

### ★E-5 ⚠**실측 자료의 출처를 인용하지 않는다**

`memory/reference_bergamasco_stereo_dataset.md` 가 출처를 명시한다:
**Guimarães, P.V. et al. (2020),** *"A data set of sea surface stereo images to resolve space-time wave fields"*,
**Scientific Data 7:145**, DOI `10.1038/s41597-020-0492-9`.

**논문 전수 grep(내 계수)**: `Guimar` **0** · `Bergamasco` **0** · `Acqua` **0** · `Scientific Data` **0**.
논문은 이 자료를 12곳에서 *"고정 관측소 스테레오 12 레코드"* 라고만 부른다(`:296, :653, :776, :1781, :4557` 등).

⟹ ⛔**공개 데이터셋 인용 규범 위반**이고, 동시에 **문헌 처리의 손실**이다 —
그 자료는 **Wang 2025 가 쓴 바로 그 공개 데이터셋**(*"a public stereo image dataset of ocean waves
(Guimarães et al., 2020)"*, `[인용: R2_5 §2.4]`)이고 **DSGP-Net 이 숫자를 낸 두 사이트**다`[인용: R2_5 §3]`.
★그리고 데이터셋 저자 **Bergamasco 는 §2 N-1 의 WASSfast 저자와 같은 사람**이다.
⟹ ***논문의 실측 폴더 이름(`data/real_bergamasco/`)이 자기가 인용하지 않은 최근접 선행자의 이름이다.***

### ★E-6 ⚠**prune--grow 를 3D Gaussian Splatting 에 귀속**

`:1878-1882`: *"이 prune-grow 순환은 3D Gaussian Splatting~\citep{Kerbl2023}의 densification--pruning 전략과
개념적으로 유사하다."* 그런데 바로 위 `:1870-1874` 가 기술하는 것은
*"잔차 신호에 **빔포밍을 재적용**하여 **주도적 잔차 피크를 모델에 추가**"* — 이는 **CLEAN(1974) / matching pursuit /
RELAX** 의 정의 그 자체다. 논문 전수 grep: `RELAX` **0** · `Stoica` **0** · `matching pursuit` **0** · `CLEAN` **0**.
⚠**크기는 작다** — 적응형 N 은 이미 정확도 기여로 철회됐다(`:806-807`). ⛔**그러나 방향이 나쁘다**:
50년 된 배열신호처리 알고리즘을 2023년 그래픽스 논문에 귀속시키면 심사위원은 **저자가 그 계보를 모른다**고 읽는다.

### ★E-7 ⚠**§격자 비의존 절에 인용이 0건이다**

`:1048-1054`: *"고전적 스펙트럼 추정 방법(FFT, Welch, **MUSIC**, **ESPRIT**)은 … **Capon** 의 minimum-variance
빔포밍과 MUSIC 은 …"* — **`\citep` 이 한 개도 없다.** 이 방법군은 논문의 **알고리즘상 최근접 이웃**인데
6줄로, 무인용으로 기각된다. 그리고 그 절이 인용하는 유일한 희소 표현 문헌은 **Candès 2006**(`:1104`, 격자 위 CS)
— ★**격자 비의존을 주장하는 절이 격자 기반 CS 만 인용한다.**

### ★E-8 ★★★**부류의 진짜 원인 — "원장이 이미 잰 것을 잃어버렸다" 의 세 번째 재발**

`/home/vexit/wavesplat/ref_paper/literature_survey_cs_spectral.md`, **작성일 `2026-05-21`**, 내가 열었다:
- §3 = *"Sparse Spectral Estimation (**Gridless / Atomic Norm**)"* — Tang/Bhaskar/Recht 2013 · Yang & Chu 2019 · SBL.
- §5.2 = ★***"Gallego, Yezzi et al. (2021) — Real-Time Optical Wave Estimation via Dispersion"***, 그 문서의 축자 평가:
  > *"**HIGHLY RELEVANT** — same idea of using **sparse 3D points + dispersion relation** to estimate wave spectrum."*
- §10 권장 독서 5번 = *"**Gallego/Yezzi 2021 — sparse points + dispersion for wave estimation (most similar input setup)**"*.

⟹ ★***저장소는 2026-05-21 에 "입력 설정이 가장 유사한 선행 연구" 를 이미 특정해 두었고, 그것이 학위논문에
0회 인용된다.*** 같은 문서가 지목한 **Harley & Moura 2013 · Bayindir 2016/2018 · Laface 2017/2022 ·
Candès & Fernández-Granda 2014** 도 전부 0회다.

**디스크에 있으나 `bib` 에 없는 1차 PDF**(내 `ls -la`, 전부 mtime `2026-06-24`):
`wang_pan_2021_jfm.pdf` · `wang_pan_2022_jfm.pdf` · `tang_bhaskar_recht_2013.pdf` · `klein2020_fluids.pdf` ·
`ehlers_2025.pdf` · `pinns_2025_da.pdf` · `vof_enkf_2025.pdf` · `desmars_2018_hal.pdf` ·
`07_Wave_competitors/{ehlers_2025b_pino,shred_williams_2023,swh_sparse_2025}.pdf`.

⟹ ★**`E-1`~`E-7` 은 서로 다른 실수가 아니라 **하나의 결함의 일곱 얼굴**이다:
***저장소는 문헌을 조사했고, 그 조사 결과가 논문에 전파되지 않았다.***
이는 `MEMORY.md` 가 이미 이름 붙인 부류(*"원장이 이미 잰 것을 잃어버렸다"* — `D-17`/`D-18`, Wang PDF 2개월 방치)의
**세 번째 사례**이며, 이번에는 **학위논문 본문**에서 일어났다.

---

## 5. Q4 — 실측 음성 + 외부 기준선 비교 불가 상태에서 **합성만으로 novelty 가 서는가**

> ### ⛔**서지 않는다. 단 사유는 "합성이라서" 가 아니다 — 사유는 셋이고, 전부 문헌 축이다.**

**사유 ①: 합성 우위의 상대가 문헌 방법이 아니다.**
`B-12` 의 비교자 `SPEC25` 는 자체 구현 `letkf_spec.LETKFGridSpec` 이고 **JONSWAP 배경 · 국소화 25 m ·
앙상블 600**(`tmp/S16_BASELINE_2026-08-14/run_s16.py:66`, `PREREG.md:92`)이며,
참 `Hs`·`Tp` 를 오라클로 받는다(`run_s16.py:210-211`)`[인용: B2 §4.2]`.
✅**오라클을 준 것은 우리에게 불리한 설계이고 나는 그것을 인정한다.**
⛔**그러나 `SPEC25` 를 인용할 문헌이 없다.** 논문이 살 수 있는 문장은
*"우리가 만든 최강 격자 기준선을 이겼다"* 이지 *"문헌의 방법을 이겼다"* 가 아니다.

**사유 ②: 문헌에 추적 가능한 유일한 기준선이 합성에서조차 무기술이다.**
`baselines/etkf_wang2025.py` 팔은 `results/wang2025_comparison.json` 12셀에서 **0.2462--0.2757**
= 제로필드 해석값 0.25 에 못 박혀 있다`[인용: R2_5 §6.1, 내가 재집계하지 않음]`.
실측에서는 **두 종 다 자기검정 실패**(`STATUS_2026-08-23/STATUS.md:37-39`, `B-5` **0/12**).
⟹ ***합성에도 실측에도 "문헌 방법 대비" 행이 0개다.***

**사유 ③ ★★★: 이 분야의 평가 규범이 실측이고, 그 자료를 우리가 갖고 있다.**
Wang 2025 = 공개 `AA 03` + 자체 현장 · DSGP-Net = Acqua Alta + Yellow Sea · Bergamasco 2021 = WASS 계열 실측.
**그 자료가 `data/real_bergamasco/`(= Guimarães 2020) 에 있다**(`memory/reference_bergamasco_stereo_dataset.md`).
⟹ ⛔***"실측 외부 기준선 비교 불가" 는 문헌의 성질이 아니라 우리 재현의 성질이다.***
기여 ⑥ 이 그것을 *"측정된 한계"*(`:751-753`)로 승격하는 것은 **문헌 대비로는 성립하지 않는다** —
같은 레코드 위의 공개 수치가 **세 편** 존재하고, 논문은 그 데이터셋을 **인용조차 하지 않는다**(E-5).

### ✅**단 하나의 형태로는 선다 — 그리고 나는 그것을 죽이지 못했다**

novelty 를 **방법**이 아니라 ***"측정된 작동 경계"***(N-9)로 두면:
- 합성 개입 실험(`B-12` 43/48, 음성대조 `SPEC25<ZERO` 47/48`[인용]`)이 경계의 **한쪽 끝**을 정당하게 지탱하고,
- 실측 음성(`B-7`/`B-8`, `STATUS.md:32-35`)이 **다른 쪽 끝**을 지탱하며,
- 이 두 끝은 **문헌 대비로 선점되지 않았다**(§2 N-9).

⟹ ★**이것은 B2 §11 의 권고 (c) 와 같은 결론이고, 나는 문헌 근거를 하나 더한다**:
***경계 논문은 "우리 시뮬레이터의 경계" 가 아니라 "이 문제의 경계" 를 그려야 하고,
이 문제의 좌표계는 2020 년 이후 Acqua Alta / Yellow Sea 다.***
그 좌표계 위에 우리 점을 하나도 찍지 않은 채 경계를 선언하면, 심사위원은 그것을
*"측정된 경계"* 가 아니라 *"우리 코드의 경계"* 로 읽는다.

---

## 6. 내 진영에 **불리한 것** (필수 절 — 먼저 적는다)

1. ⛔★**나는 Bergamasco 2021 C&G 본문을 읽지 못했다.** 내 N-1 킬의 근거는
   ①C&G 서지·키워드(NUFFT · 3D spectrum) ②WASS 공식 페이지 축자 ③Pistellato 2021 요약 ④Wang 2025 결론의
   WASSfast 언급 — **넷 다 2차 또는 요약이다.**
   ⟹ ★**"WASSfast 의 상태는 NUFFT 균일 파수 격자다" 는 `ARGUED` 이지 `MEASURED` 가 아니다.**
   그 구분이 무너지면 N-8(연속 파수)마저 죽고, 반대로 그 구분이 확인되면 **저자에게 유리해진다.**
   ⛔**논문에 쓰기 전에 반드시 원문을 사서 읽어야 한다.** 이것은 이 저장소가 두 번 물린 부류(`US 12,054,225`,
   DSGP-Net)를 **내가 반복하지 않기 위한 조건**이다.
2. ⚠**N-4 킬은 "선언 문언" 을 죽이지 교집합 전체를 죽이지 않는다.** Wang & Pan 2021/2022 는
   **HOS 스펙트럼 격자 + 조밀 레이더 관측**이고, 희소 3D 점군이 아니다. 내가 죽인 것은
   *"시간 축이 문헌에서 남은 **유일한** 틈"* 이라는 **전칭**이다.
3. ⚠**N-8 에 대한 선점 문헌을 나는 찾지 못했다.** 이는 **부재 증명이 아니다.**
   `literature_survey_cs_spectral.md §3.2` 가 *"Adding dispersion constraint to ANM is non-trivial"* 이라 적는 것은
   **미해결이라는 뜻이지 미발표라는 뜻이 아니다.**
4. ⚠**E-3(Cummins) 은 원문 대조가 아니라 논문 내부 모순이다.** 원문이 실제로 선박 운동을 다룰 가능성을 배제하지 않는다.
5. ✅**저자에게 유리한 사실을 적는다**: DSGP-Net 제목 오기는 **이미 수리됐고**(`bib:352-355` 에 정정 사유까지 기록),
   VARPRO·Kim `1.8T_p`·Wang 우선권은 **논문이 먼저 자백했다**. ★**즉 저자는 이 부류를 인지하고 있고,
   실제로 고친 이력이 있다.** 내 `FAIL` 은 *"안 고친다"* 가 아니라 ***"한 번에 이웃 항목까지 훑지 않는다"*** 에 대한 것이다.
6. ⚠**나는 제3–6장을 읽지 않았다.** 제어·수조·실측 절의 문헌 처리에 대해 나는 **판정할 자격이 없다.**
7. ⚠**나는 `B-*` 원자료를 재집계하지 않았다.** 이 문서의 모든 수치는 `[인용]` 이다.

---

## 7. 구체적 처방 — **무엇을 하면 내 판정이 바뀌는가**

⚠**결과를 보고 벌칙을 만들지 않기 위해, 조건을 지금 사전등록한다.**

### ★A. `FAIL` → `MAJOR REVISION` 조건 (전부 **런 0회**, 반나절)

| # | 처방 | **측정 가능한 통과 기준** |
|---|---|---|
| **P-1** | ★★★**Desmars 계보 문장 6곳 교체** (`:562, :564, :589, :936, :939-942, :1087`) — ETKF 를 Desmars 에서 떼고, 격자 앙상블 칼만 계보를 **Wang & Pan 2021 (*JFM* 918:A19) → Wang & Pan 2022 (*JFM* 949:A31) → Feng Wang et al. 2025 (*Coastal Eng.* 197:104690)** 로 다시 잡는다. Desmars 는 **변분/HOS--OCM 계보**로 별도 문단 | 각 문장이 **로컬 PDF**(`ref_paper/wang_pan_2021_jfm.pdf` 등)에서 축자 확인되고, `:941-942` 의 *"위 레이더 기반 정식화를 이식"* 이 **삭제**된다 |
| **P-2** | ★★**`Wang2025PointsToWaves` 제목·저자 정정** | `bib:331-332` 문자열 == `feng_wang_2025.pdf` 1쪽 표제 문자열; `author` 에 5인 전원 |
| **P-3** | ★★★**§최근접 선행 연구에 WASSfast 문단 신설** — Bergamasco et al. 2021 *C&G* 147:104666 + Pistellato et al. 2021 *Remote Sensing* 13:3780. **갈림점을 한 단어로 적는다**: 우리 = **연속 파수 원자**, 그들 = **NUFFT 파수 격자** | ⚠**원문 확보 후에만.** 그리고 이 문단이 들어가면 `:1216-1217` 의 *"유일한 틈 = 시간 축"* 이 **자동으로 무효**가 되므로 **함께 교체**한다 |
| **P-4** | ★★**자료 출처 인용** — Guimarães et al. (2020) *Sci. Data* 7:145, DOI `10.1038/s41597-020-0492-9` 를 실측 절과 제6장에 인용 + *"경쟁자 3편이 같은 사이트에서 숫자를 보고했다"* 한 문장 병기 | 논문 grep `Guimar` ≥ 2. 그리고 기여 ⑥ 문언이 *"비교 불가"* → ***"우리 재현으로는 비교 불가이며, 같은 자료의 공개 수치는 존재한다"*** 로 바뀐다 |
| **P-5** | ⚠**부재 주장 완충** `:940` 에 *"본 연구의 알기로"* 추가 (또는 E-1 수리 시 문장 자체 삭제) | 논문의 무완충 부재 주장 **0건** |
| **P-6** | ⚠**off-grid 계보 3편 + CLEAN 계보 인용** (`Tang, Bhaskar, Shah & Recht 2013` [로컬 PDF 보유] · `Candès & Fernández-Granda 2014` · `Yang & Chu 2019`; prune--grow 는 CLEAN/matching pursuit 로 계보 정정, 3DGS 는 **유추로 강등**) | `:1043-1054` 의 `\citep` 개수 0 → ≥3 |
| **P-7** | ⚠**`Cummins2017` 문장을 `bib` 각주 수준으로 하향** (`:1174-1175`) | 본문이 *"보였다"* → *"방법론적으로 적용 가능하다"* |

★**P-1~P-5 가 되면 나는 `MAJOR REVISION` 을 던진다.**

### ★★B. `MAJOR REVISION` → `PASS` 를 다투는 조건 (**런 필요**, 판정 이동 가능)

| # | 처방 | 통과 기준 |
|---|---|---|
| **P-8** | ★★★**`AA_2014` / `AA_2015` / `YS_2017` 에서 우리 추정기를 돌려, Wang 2025·DSGP-Net·Bergamasco 2021 이 숫자를 낸 **같은 사이트**에 우리 점을 찍는다.** 자료 취득 0 | ⚠**규약 환산을 사전등록할 것**(합성 nRMSE 분모 `Hs`, 실측 분모 `σ`, 실측비 **3.045--4.278** — `memory/reference_bergamasco_stereo_dataset.md`). ★**결과가 음성이어도 판정은 올라간다** — 음성이 *"우리 코드의 경계"* 가 아니라 ***"이 문제의 경계"*** 가 되기 때문이다 |
| **P-9** | ★**문헌 정합 검사기**(방벽 4번째) — `bib` 의 각 항목에 `method=` · `sensor=` · `venue-verified=` 필드를 강제하고, 본문의 `\citet{X}` 서술문과 대조해 불일치를 **경고로 승격** | 오늘 이 검사기를 돌리면 **최소 3건**(E-1·E-2·E-3)이 걸린다. 통과 기준 = **제1·2장 citation 전건에 필드 존재 + 불일치 0** |

⚠**P-9 의 근거**: 오늘 `STATUS.md:120-134` 가 이름 붙인 부류 — *"검사기가 검사하지 않는다"* · *"침묵을 경고로 승격하라"* —
가 **문헌 축에는 아직 적용되지 않았다.** 방벽 3종(`check_batch_artifacts` · `find_handcopied_refs` · `check_thresholds`)은
**측정만 검사하고 인용은 검사하지 않는다.** ⟹ ★***E-1 이 세 라운드의 심사를 통과한 이유가 정확히 그것이다.***

---

## 8. 소견 다섯 줄

1. ★★★***2021 년에, 우리 실측 자료를 만든 그룹이, 희소 3D 점 + 분산관계 + 실시간을 발표했다.
   그 논문은 우리 디스크의 문헌 조사서가 "입력 설정이 가장 유사하다" 고 지목했고, 학위논문에 0회 인용된다.***
2. ★★★***제2장은 자기 기준선을 ETKF 를 쓰지 않는 논문에 귀속시킨다. 저장소는 2026-05-13 에 그것을 옳게 적었다.***
3. ★★***최근접 경쟁자의 제목이 `bib` 에서 틀렸다 — DSGP-Net 오기를 고친 커밋에서 11줄 위 항목을 안 열었다.***
4. ⚠***"실측 외부 기준선 비교 불가" 는 문헌의 성질이 아니라 우리 재현의 성질이다. 같은 레코드의 공개 수치가 셋 있다.***
5. ★★★***그래도 죽지 않는 것이 하나 있다 — 경쟁자 셋 중 누구도 자기 방법의 실패 경계를 개입으로 재지 않았다.
   그것이 이 학위의 유일한 척추이고, 그 척추는 합성에서만 서 있어서는 안 된다. 좌표계는 이미 우리 디스크에 있다.***

---

### 부록 — 내가 `VERIFIED`/`MEASURED` 로 표시한 것의 재현 명령

```bash
cd /home/vexit/wavesplat/wavesplat
# E-2: Wang 2025 제목 대조
python3 -c "from pypdf import PdfReader; print(PdfReader('ref_paper/feng_wang_2025.pdf').pages[0].extract_text()[:400])"
sed -n '329,340p' thesis/thesis_refs.bib
# E-1: Desmars 계보 (논문 vs 자기 bib vs 자기 발표자료)
sed -n '560,570p;936,942p' thesis/thesis_main.tex ; sed -n '42,54p;201,214p' thesis/thesis_refs.bib
sed -n '551p;1528p' scripts/generate_labmeeting_2026_05_13.py
# E-8: 조사됐으나 전파되지 않은 문헌
ls -la ref_paper/*.pdf ; grep -n "Gallego\|Tang, Bhaskar\|HIGHLY RELEVANT" /home/vexit/wavesplat/ref_paper/literature_survey_cs_spectral.md
# §3.2: 2025-26 문헌이 한 절에 갇혀 있음
for k in Wang2025PointsToWaves DSGPNet2026 Kim2023RealTime GolubPereyra2003; do echo -n "$k "; grep -o "$k" thesis/thesis_main.tex | wc -l; done
# E-5: 자료 출처 미인용
grep -c "Guimar\|Bergamasco\|Acqua" thesis/thesis_main.tex
# E-7: 격자 비의존 절의 인용 0건
sed -n '1043,1055p' thesis/thesis_main.tex
```

**Sources (문헌 검증)**
- [Experimental and numerical assessment of deterministic nonlinear ocean waves prediction algorithms using non-uniformly sampled wave gauges — *Ocean Eng.* 212:107659](https://www.sciencedirect.com/science/article/abs/pii/S0029801820306557)
- [Nonlinear deterministic reconstruction and prediction of remotely measured ocean surface waves — *JFM* 975:A8 (HOS–OCM, LGMRES)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/nonlinear-deterministic-reconstruction-and-prediction-of-remotely-measured-ocean-surface-waves/0D492ABAE4B99782612F5CE4AC890B6D)
- [Phase-resolved ocean wave forecast with ensemble-based data assimilation — *JFM* 918:A19 (Wang & Pan 2021)](https://arxiv.org/abs/2012.00677)
- [Data Assimilation-based Simultaneous Phase-Resolved Ocean Wave and Ship Motion Forecast (Wang & Pan 2025)](https://arxiv.org/abs/2504.13943)
- [Toward real-time optical estimation of ocean waves' space-time fields — *Computers & Geosciences* 147:104666](https://www.sciencedirect.com/science/article/abs/pii/S0098300420306385)
- [WASSfast — "exploits the linear dispersion relation and sparse feature triangulation … in real-time"](https://sites.google.com/unive.it/wass/software/wassfast)
- [Machine learning for phase-resolved reconstruction … (Ehlers et al., U-Net/FNO) — *Ocean Eng.* 288:116059](https://www.sciencedirect.com/science/article/pii/S0029801823024435)
- [From points to waves — *Coastal Engineering* 197:104690](https://www.sciencedirect.com/science/article/abs/pii/S0378383924002382)

*작성: A1 (선행연구·선점 심사관) · 2026-08-24 · 추정 런 0회 · 열람 범위는 §0 에 전부 열거함.*
