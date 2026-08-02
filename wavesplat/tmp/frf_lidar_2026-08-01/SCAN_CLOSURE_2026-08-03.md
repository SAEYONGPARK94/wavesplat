# 실측 소스 스캔 — 종결 (2026-08-03)

2026-08-02 00:58에 **중단된 `projects/` 크롤**을 이어받아 완주하고, 같은 문서(`SOURCE_SCAN_2026-08-02.md`)가
남긴 미확인 항목 2개(NASA ATM 대양 비행 · CDIP/FOWD 실사용 가능성)를 닫는다.

⛔ 이 문서는 어떤 성능 주장도 하지 않는다. 전부 카탈로그 열람과 변수 헤더 직독이다.

---

## 0. 한 줄 결론

**서버 전체를 컬렉션 수준에서 소진했다. 세 번째로 놓친 데이터셋은 없다** —
`WaveShape/lidarWaveScans`(30파일·3일)가 chldata 서버 전체에서 유일한 2D 위상분해 표면-LiDAR
산출물이다. 그리고 **미확인 2건은 둘 다 "닫힘"이 아니라 "다른 문이 열림"으로 끝났다**:
ATM은 대양 파랑을 실제로 쟀지만 **공개 아카이브가 없고**, CDIP은 FOWD를 거치지 않고
**원본이 그대로 열려 있으며 우리가 원한 것보다 더 가까운 변수를 준다.**

---

## 1. 끊긴 크롤 이어받기 — 그리고 그 크롤의 구조적 한계

`crawl_projects2.py`를 재시작했다(`crawl2.log`). 그러나 재시작 후 **크롤러 자체의 맹점**을 찾았다:

```python
for h, t in rs:
    if YEAR.match(t):
        walk(...); break      # ← 연도 디렉터리는 '첫 번째 하나만' 내려간다
```

⟹ 이 크롤은 **연도축에 대해 원리적으로 비-소진적(non-exhaustive)** 이다. 완주해도
"전부 봤다"의 근거가 될 수 없다. 그래서 크롤 완주를 기다리는 대신 **컬렉션 수준 얕은 소진 스캔**을
별도로 돌려 그것을 판정 근거로 삼았다.

**추기(같은 세션, 나중)**: 깊은 크롤은 결국 **완주했다** — `nodes visited: 386`,
638행, **fetch 실패 0**. (앞서 "미완주"로 적었던 것은 대기 루프가 자기 자신을 매치해
종료를 감지하지 못한 탓이다. 사실을 정정한다.) 완주분을 `analyze_projects_tree.py`로
분류하면 **독립적으로 같은 답**이 나온다:

| 버킷 | 히트 |
|---|---|
| **3D 표면 LiDAR** | **`lidarWaveScans` 단 1개**(30 파일) |
| 스테레오/영상 | `argusNorthhead`(318) · `Delilah/Video` |
| 레이더 | **0** |
| 이동 플랫폼 | **0** |

⟹ 판정 근거는 여전히 얕은 소진 스캔(§2)이지만, **깊은 크롤이 독립 계기로 같은 결론에 도달**했다.

## 2. 소진 스캔 결과 — 서버 전체

### 2.1 `frf/` 최상위 6트리 (이전 조사는 3개만 봤다)

| 트리 | 내용 | 2D 위상분해 수면? |
|---|---|---|
| `oceanography` | (기존 조사) `lidarWaveGauge×5`·`lidarHydrodynamics`·`lidarWaveRunup` | ❌ 공선 1D |
| `projects` | 12개 프로젝트 — §2.2 | ⭐ WaveShape만 |
| `geomorphology` | DEMs · altimeter · elevationTransects | ❌ 해저 |
| `meteorology` | barometer · metGauges · wind | ❌ |
| `survey_temp` | 1,646개 측량 CSV (1974–) | ❌ 해저·해빈 |
| `argus_temp` | **비어 있음** | ❌ |

### 2.2 `frf/projects/` = 12개 전부 열람

`CurrituckSound` · `DareCounty` · `Delilah` · `Dunex` · `Publications` · `SandyDuck` ·
**`WaveShape`** · `Wilmington` · `argusNorthhead` · `bathyduck` · `duck94` · `specialProducts`

- `argusNorthhead` 318파일 = Argus **영상**(강도지 표고 아님)
- `bathyduck` = cBathy(영상기반 수심역산)·survey
- `Delilah`(1990)·`SandyDuck`(1997)·`duck94` = 압력·유속 배열 시대, 3D 표면 스캐너 없음
- `Dunex` = UW_drifters 단독 · `specialProducts` = `8m-arrayIG` 단독
- ⟹ **`WaveShape` 하나만 남는다.**

### 2.3 ★ 서버 루트는 FRF가 전부가 아니었다 — 8개 컬렉션 추가 확인

`https://chldata.erdc.dren.mil/thredds/catalog/catalog.xml` 최상위는 **9개 컬렉션**이고
`FRF data`는 그중 하나다. 이전 조사는 이 층을 본 적이 없다. 전부 열었다:

| 컬렉션 | 내용 | 판정 |
|---|---|---|
| **UAS data** | `FRF/PHANTOM4_PRO_RGB`(5일)·`FRF/X8_GOPRO_APPLANIX`(4일, 2017) + `GROUND_CONTROL`(GCP·LAS) | ❌ **드론 RGB 사진측량**(SfM). LAS 점군은 지상기준점 공동등록용, 수면 아님 |
| Buoy data | MEDS · NDBC | ❌ 1D 점계측 |
| WIS data | Atlantic/Pacific/GoM/Alaska/GreatLakes | ❌ **후측(hindcast) 모델**, 관측 아님 |
| CMTB Data | integratedBathyProduct · morphModels · waveModels | ❌ 모델 |
| DIF / CSAT / CIRP / EWN | 문서·zip·아틀라스·성능지표 | ❌ |

⚠ **이건 같은 실패 양식의 세 번째다** — ①`oceanography/waves/` 미확인 → ②`projects/` 미확인 →
③**서버 루트 미확인**. 앞의 둘은 주장을 낸 뒤에 정정했고, 이번엔 주장 전에 잡았다.
교훈은 "한 층 위를 봐라"가 아니라 **"열람 범위를 주장의 범위와 같이 적어라"** 이다.

## 3. `WaveShape` 정밀 열람 — TRIAGE 수치 확증 + 신규 사실 2건

`WaveShape` 하위 계기 9종:
`AquatecABS` · `BB3_opticalTurbidity` · `ECONTU_opticalTurbidity` · `LISST_ABS` ·
`SurveyTransect` · `SuspendedSediment` · `TritonADV` · `VectorADV` · **`lidarWaveScans`**

### 3.1 LiDAR 표본은 확증됐다 (부풀지 않음)

`lidarWaveScans` 카탈로그 직독: **30 파일 / 2.79 GB / 3일**
(2017-09-07 ×10, 09-08 ×6, 10-25 ×14). TRIAGE의 수치와 **일치**.

⟹ 그러나 **캠페인 자체는 16일**이다(ABS/`SurveyTransect`는 2016-11-18 ~ 2017-10-26의 16개 날짜).
**LiDAR는 그중 3일에만 실렸다.** 표본 3일은 열람 누락이 아니라 **배치의 사실**이다.
TRIAGE §W4의 "계절 다양성 없음"은 **강화된다**(반박이 아니라).

### 3.2 ★신규 — 같은 날의 동반 계기

| 계기 | 날짜 | LiDAR 3일과 겹침 |
|---|---|---|
| `SurveyTransect` | 16일 전부 | **3/3 겹침** |
| `TritonADV` | 10일(2017-09-13 ~ 10-26) | **1/3**(10-25만) |
| `VectorADV` | 3일(2016-11 ~ 2017-04) | 0/3 |

두 가지가 따라온다:
1. **`SurveyTransect`가 3일 전부에 수심 단면을 준다.** 우리 분산 하드제약
   ω=√(g·k·tanh(k·d))는 **d를 요구하는데**, 쇄파대에서 d는 상수가 아니다.
   FRF LiDAR 위에서 무엇을 하든 **d는 가정이 아니라 측정으로 넣을 수 있다.** TRIAGE에 없던 사실.
2. **ADV는 3일 중 1일(2017-10-25, 14파일)만 겹친다.** 유속 기반 독립 GT 교차검증은
   **가능하지만 1세션 한정**이다. "가능하다"로 일반화하면 안 된다.

## 4. 미확인 항목 ① — NASA ATM 대양 비행: **존재하나 공개 아카이브 없음**

- ATM의 공식 용도 목록에 **"measurement of sea-surface elevation and ocean wave characteristics"** 가 있다.
- 실제로 Hwang·Krabill·Wright·Swift·Walsh가 ATM으로 **3D 해면 지형 → 방향 파수 스펙트럼**을 냈다
  (SWADE 등). ⟹ **"이동 항공 플랫폼 + 2D 개구 + 심해"는 이미 1990년대에 측정된 적이 있다.**
- 그러나 NSIDC가 배포하는 ATM 산출물(`ILATM1B`/`ILATM2` 등)은 **극지 빙권 전용**이고,
  대양 파랑 캠페인의 **공개 배포 경로는 찾지 못했다.** 논문만 있다.

⟹ ATM은 "레짐 불일치로 기각"이 아니라 **Scripps MASS·ERDC와 같은 연락 트랙**으로 재분류된다.
그리고 논문 문구에 제약이 생긴다:

> ⛔ **"이런 데이터는 존재하지 않는다"는 쓸 수 없다. 쓸 수 있는 것은
> "공개 아카이브가 없다(no public archive)"뿐이다.**

✅ 부수 확인: `hwang2000`은 **이미 `references.bib`에 있고 본문 L171에서 인용된다.**
즉 선점 사고가 아니라 **문구 정밀도 문제**다.

## 5. 미확인 항목 ② — CDIP/FOWD: **FOWD를 우회하고 CDIP 원본이 더 낫다**

전 커밋(`014c5c9`)은 FOWD 채택을 권고하면서 스스로 한계를 달았다 —
*"부이 대역폭은 주파수영역 1D … 방향 부이의 a1/b1/a2/b2 → 방향 분산이 더 가깝지만"*.
**그 '더 가까운 것'이 그냥 열려 있다.** 실물 확인:

| 확인 | 결과 |
|---|---|
| FOWD 처리본 배포 경로 | **GitHub 저장소에 다운로드 URL/DOI 없음** — 처리 툴킷만. 총량 1.1 TB |
| CDIP THREDDS 아카이브 | **열림** — `thredds.cdip.ucsd.edu`, 스테이션 **194개** |
| 배포 단위 | 스테이션 × 배치(deployment) netCDF (예: `100p1_d01.nc` … 20개) |
| **변수(DDS 직독)** | `waveA1Value` `waveA2Value` `waveB1Value` `waveB2Value` **`waveSpread`** `waveBandwidth` `waveEnergyDensity` `waveDp` `waveHs` `waveTp` |
| 배열 형상 | `waveSpread[waveTime=50972][waveFrequency=64]` — **단일 배치 하나가 50,972 반시간 레코드** |
| 전송 | OPeNDAP 부분전송 가능(`waveSpread` 단일 배치 ≈ 13 MB). **1.1 TB 받을 필요 없음** |

⟹ 세 가지가 바뀐다:
1. **대리변수가 한 단계 가까워진다.** FOWD의 1D 대역폭이 아니라 **주파수별 방향 확산
   `waveSpread`(와 a1/b1/a2/b2 원본)** 를 직접 쓴다.
2. **비용이 무너진다.** 1.1 TB 다운로드 → OPeNDAP 부분전송 수십 MB.
3. **검정력 규모가 달라진다.** 현재 실측 판정 단위 n=5(레코드)와 대비해
   **배치 하나가 5×10⁴ 해상상태**다.

⚠ **한계는 그대로 유효하다.** `waveSpread`는 **단일점 부이의 방향 확산**이고, 우리
`n_components`는 **분리 가능한 방향 성분의 개수**다. 같지 않다.
⟹ **"혼잡도 재정의"는 여전히 불가.** 정당한 용도는 **합성 생성기 파라미터의 지지집합(support)을
실측 분포에 맞추는 것**뿐이며, 사전등록에 못박아야 한다(전 커밋의 제약을 그대로 승계).

---

## 6. 이 스캔이 논문 문구에 부과하는 정정

현재 `paper/wavesplat.tex`의 Data availability(L2423–)는 **코드·시뮬 데이터 공개 약속만** 있고
니치 진술이 없다. 스캔 자산을 넣을 때 다음 세 문구 제약이 동시에 걸린다:

1. "존재하지 않는다" ❌ → **"공개 아카이브가 없다"** ✅ (§4, ATM/SWADE 때문)
2. 조사 범위를 **명시**해야 한다 — 어떤 서버·어떤 컬렉션·어느 층까지 (§2.3의 3연속 실패 때문)
3. FRF LiDAR 3일은 **"배치의 사실"** 로 쓸 수 있다(열람 한계가 아니라). (§3.1)

## 7. 근거 파일

- `projects_tree.txt` / `crawl2.log` — 재개한 깊은 크롤(미완주, §1)
- `lidarWaveScans_catalog.xml` — `lidarWaveScans` 카탈로그 원본 30 데이터셋
- `analyze_projects_tree.py` — 크롤 산출 분류기
- 선행: `TRIAGE.md` · `SOURCE_SCAN_2026-08-02.md` · `REVIEW_FABLE_2026-08-02.md`
