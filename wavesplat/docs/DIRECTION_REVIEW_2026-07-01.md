# 냉정한 방향 검토 — 선행연구·데이터셋 종합 후 (2026-07-01)

> 작업: 기존 lit review + ref_paper 정독, 추가 선행연구 검색, 공개 데이터셋 발굴, 방향 재판정.
> 톤: 낙관 금지, fact 기반. 관련: `NOVELTY_survey_amortized_lidar_2026-07-01.md`,
> `ref_paper/08_UQ_wave/_NEW_PAPERS_2026-07-01.md`, `ref_paper/09_datasets_CRB/_DATASETS_2026-07-01.md`.

## 1. 문헌 지형 (팀이 이미 알던 것 + 신규)
이 분야(원격탐사 phase-resolved 파 재구성+예보=DSWP)는 **성숙·과밀**. 기존 review가 이미 핵심을
커버: Nouguier 2014(ship flash-LiDAR, 직접 precursor), Jiang&Li 2024(ETKF optical, primary baseline),
Ehlers 2024/2025(U-Net/FNO + PINO sparse 재구성), Wang&Pan EnKF, Qi 2023/2025, Fisher 2021(sparse
buoy predictable zone), Xue 2025·Ma 2023(LiDAR 점군→파라미터/방향스펙트럼).
**신규로 추가된 핵심 = Zhang 2022**(변분베이즈 phase-resolved 실시간 **UQ** + predictable zone 74.6%↑) —
기존 review에 UQ 클러스터가 통째로 비어 있어 놓쳤던, 우리 UQ novelty의 정면 경쟁자.

## 2. 우리 두 헤드라인의 냉정한 위치
**(H1) "Grid-free spectral Gaussian"** (원 WaveSplat, 2026-04 novelty): 차별점은 Jiang&Li(격자보간)
대비 "grid-free". 그러나 Ehlers/PINO/NP(ConvCNP·Senseiver)·INR·neural operator 전부 grid-free 연속표현.
→ **"grid-free" 단독은 더 이상 깨끗한 차별점 아님**(우리 ref_paper 자체가 grid-free 방법으로 가득).
**(H2) "Amortized posterior + 보정 UQ"** (최근 pivot): Zhang 2022가 phase-resolved 실시간 UQ를 점유,
amortized inference(NP 계열)는 확립된 ML. → **증분적.**
**판정: 두 헤드라인 모두 깨끗한 박사 novelty 아님.**

## 3. 그래도 살아있는 가장 방어가능한 gap (냉정)
교차 결과, *덜 점유된* 단 하나의 교집합:
> **실측·희소·경사·단일 스캐닝 ship-LiDAR(+플랫폼 운동·dropout) 라는 특정 센서 레짐에서, LiDAR
> 관측물리를 내장한 dropout-aware *보정* 배포 posterior로 phase-resolved 장을 복원하고, *실데이터*로
> 검증한다.**
- 왜 덜 점유됐나: 선행은 면적 레이더/dense 스테레오(10⁴m², 15–20Hz)/buoy망/항공·flash-LiDAR(dense)·
  연안 LiDAR가 대부분. **이동 선박의 극희소 단일 스캐닝 개구 + dropout을 역문제로** 다룬 건 거의 없음.
  Nouguier 2014가 가장 근접하나 dense flash-frame·최소자승·UQ 없음·dropout 없음·시뮬.
- **위험(정직)**: 좁고, 닫히는 중(Xue 2025 등 ship/coastal LiDAR 점군 연구 증가). 메서드(네트워크/학습)
  변경은 이 niche에 *봉사하는 수단*일 뿐 독립 novelty 아님 — 앞서 확인.

## 4. 데이터셋 (질문 2 답)
- **희소 ship-LiDAR phase-resolved 공개셋: 없음.** 우리 110ep도 전부 시뮬·정지.
- **★ Bergamasco 2020 스테레오 4-D 파고장(Scientific Data, 오픈)**: 실측 phase-resolved 공간장 →
  *희소 oblique+dropout*으로 서브샘플하면 **LiDAR 없이도 실데이터 sim-to-real 검증** 가능. 최대 약점
  (sim-only) 해소의 현실적 첫 발. Caveat: 고정 플랫폼(선박운동 없음)·스테레오 자체오차.
- 보조: CDIP/Sofar Spotter(점·스펙트럼 — 해상태 현실성/방향 검증), One Ocean Expedition(실측 선박운동+
  altimeter 단일점), SAR(통계). 상세: `ref_paper/09_datasets_CRB/_DATASETS_2026-07-01.md`.

## 5. 방향 옵션 & GO/NO-GO (냉정)
- **P1 (추천) — 센서-niche + 실데이터:** 위 §3으로 좁히고 **Bergamasco sparsify로 실데이터 검증**을
  make-or-break로. 기여 = (센서 레짐 + dropout-aware 보정 배포 posterior + *실측* 검증). 분야가 가장
  결여한 것(이 센서의 실데이터)을 친다. 위험: 좁음·경쟁 접근 중.
- **P2 — 한계/벤치마크 기여:** 이 센서의 정보한계(CRB; predictable-zone·DOA 선행 *인정* 하에 정제) +
  현실 연속·비선형·희소에서 방법들이 어디서 깨지는지 정직한 벤치. 위험: 이론 부분 점유, 증분 평가 위험.
- **P3 — vehicle 재고:** 진짜 가치가 하이드로포일 제어면 제어 중심 — 그러나 *하드웨어 제어 입증* 필요
  (sim-only 박사 불가, 사용자 확인). 하드웨어 없으면 사실상 사망.

**권고:** **P1 우선**, P2를 이론적 받침으로 병행. **GO/NO-GO 게이트** = (a) 3편 정독으로 §3 niche가
진짜 비었는지 확정(directional+experimental / Zhang2022 본문 / predictable-zone closed-form),
(b) **Bergamasco sparsify 파일럿**에서 우리 추정기가 실데이터에서 의미 있는 복원을 내는지. 둘 다 통과면
P1로 재설계, 하나라도 실패면 vehicle 재고를 정면으로 논의.

## 6. 정직한 메타노트
- 직전 세션들의 "winning/deployment-ideal" 톤은 문헌 대비 과대였음(정정 완료). 또한 내가 "발견"이라던
  PINO/Ehlers는 팀 ref_paper에 이미 있었음 — **기존 자료를 먼저 읽지 않은 내 실수.**
- 현 단계 결론은 단정 아님. 위 GO/NO-GO 두 게이트 통과 전까지 novelty는 "위험·미확정".
