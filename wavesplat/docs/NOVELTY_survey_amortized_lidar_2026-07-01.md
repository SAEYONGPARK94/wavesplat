# 선행연구 조사 & 냉정한 Novelty 평가 — amortized inference × LiDAR 파 추정 (2026-07-01)

> 사용자 요청(신뢰 회복 맥락): 단정 금지, 실제 문헌 검색 기반. 결론은 **낙관적이지 않다.**
> 이전 세션의 "deployment-ideal / winning / 0.219" 톤은 **문헌 대비 과대**였음을 명시 정정한다.

## 1. 문헌이 보여주는 것 (각 구성요소가 이미 점유됨)
**(a) 원격탐사 기반 phase-resolved 파 재구성+예보 (DSWP) — 성숙한 분야**
- FutureWaves (레이더 기반 실시간 선박운동 예보, 상용), 파형 레이더 sea-trial, X-band 레이더 재구성,
  mixed space-time 레이더, DLR. → "원격탐사로 phase-resolved 파 재구성+수십초 예보"는 신규 아님.
- **LiDAR도 이미**: Scripps 2D 스캐닝 LiDAR 해상 계측(타워 12 m), flash-LiDAR 비선형 재구성(2014).
  → "LiDAR로 파 재구성" 자체가 신규 아님.

**(b) ML phase-resolved 재구성/예보 — 2022–2025 과밀**
- **PINO 실시간 sparse 비선형 재구성** (arXiv 2508.03315): buoy/radar sparse → 실시간 phase-resolved
  *비선형* 재구성, physics-informed, GT 불필요. **우리 "실시간 sparse 재구성"과 정면.**
- U-Net+FNO 재구성(2305.11913), multi-scale FNO 예보(PoF 2025), attention 예보(ASME 2025),
  video prediction(2024), 단파정 DL(2022), AI-방법 평가(JMSE 2025).

**(c) amortized inference / conditional neural process + UQ — 확립된 ML**
- amortized 리뷰(2404.12484)에 SST GP 공간장 예시, sound-field CINN, UC-NeRF, turbulence CNF.
  → **방법(set-transformer + Gaussian NLL)은 교과서**(CNP, Garnelo 2018 계열). ML novelty ≈ 0.

**(d) ★ phase-resolved 파 예보의 UQ — 이미 존재 (내가 기대던 차별점이 약함)**
- "deterministic phase-resolved 예보의 예측오차 확률적 정량화"(Ocean Eng 2018).
- **변분 베이즈로 phase-resolved 예보 + 예보값·예보영역 불확실성 정량화** (직격).
- SG-MCMC+DL 파 예측 UQ, operator learning conformal calibrated UQ, coverage(PICP) test 관행.
  → "빠른 posterior + 보정 UQ"는 **엔지니어링 기여**지 정의적 novelty 아님.

**(e) CRB/Fisher 정보 — 파 분야선 빠른 검색서 직접 선행 안 보임**
- 검색 결과 일반 CRB 이론·중력파·레이더만; *sparse ship-LiDAR phase-resolved 파의 정보한계*
  특정 선행은 미발견. → **가장 유망한 진짜-novelty 후보** (단, 어레이처리/방향스펙트럼 추정 CRB
  문헌 대비 심층 확인 필요 — 빠른 검색의 부재 ≠ 증명된 novelty).

## 2. 냉정한 Novelty 판정
- **헤드라인 "amortized inference를 LiDAR 파 추정에 적용 (+보정 UQ)"는 조합으로서 대부분 선점됨.**
  재구성·실시간·sparse·ML예보·amortized·심지어 phase-resolved UQ까지 각각 선행 존재.
- 속도(ms)+UQ만으로는 Ocean Eng reviewer(DSWP 전문가)를 못 넘김. sim-only는 더 불리(분야엔 sea-trial 존재).

## 3. 그래도 방어가능한 기여가 살 수 있는 곳 (각각 caveat)
1. **(최강 후보) 정보한계 이론 + 한계도달 deployable 추정기** — sparse 단일개구 ship-LiDAR에서
   *무엇이 근본적으로 복원 가능한가*(방향 병목·유효개구·predictable zone)를 CRB로 규정하고, 그 한계에
   근접하는 배포가능 추정기를 제시. 응용 ML 논문들은 근본한계를 안 다룸. *아이러니하게 사용자가
   비중을 낮춘 d1 방향.* → 방향스펙트럼 CRB(어레이) 문헌 대비 심층 확인 필수.
2. **희소·경사입사·단일플랫폼 ship-LiDAR 레짐** — 대다수는 레이더(면적)/dense flash-LiDAR/buoy망.
   ~74 pts/frame·oblique·dropout의 극희소 단일개구는 별개 난이도. (단 flash-LiDAR·Scripps 존재 →
   "우리 레짐이 진짜 더 어렵다/다르다"를 입증해야.)
3. **벤치 편향 방법론 결과** — 이산성분 벤치가 고전 추정기를 과대평가(연속스펙트럼서 역전). 정직·유용하나
   thesis 본체로는 얇음.

## 4. 페르소나 패널 (적대적, strawman 아님)
- **Reviewer 2 (해양공학/DSWP)**: "DSWP가 2010년대부터 레이더·LiDAR로 phase-resolved 재구성+예보,
  sea-trial까지 했다. PINO가 sparse 비선형 실시간 재구성을 한다. 변분베이즈가 phase-resolved 예보에
  UQ까지 준다. 'transformer 써서 빠르다' 외에 뭐가 새로운가? 게다가 sim-only다."
- **ML reviewer (NeurIPS/ICLR)**: "CNP + heteroscedastic NLL은 표준. amortized posterior UQ도 표준.
  바닐라 set-transformer. ML 기여 ≈ 0 — 응용논문이면 응용 novelty가 캐리해야 하는데 거기도 과밀."
- **지도교수(실용·박사관점)**: "어떤 선행도 주장 못 하는 *한 문장*이 뭔가? 못 대면 아직 thesis가 아니다.
  빠른 surrogate 말고, 아무도 규정 못 한 근본한계나, 이 센서로 분야가 못 가진 실측결과를 가져와라."
- **정직한 self/devil's advocate**: 기여가 "알려진 방법 + 알려진 문제 + 더 빠름 + sim-only"로 붕괴할 위험.

**패널 수렴**: 살아남으려면 (a) 이 센서의 *근본한계 이론*(덜 점유, 방어가능) 그리고/또는 (b) sparse
ship-LiDAR + 배포가능 posterior 파이프라인이 *실제 데이터(수조/sea-trial)*에서 된다는 결과. 속도·UQ
단독으론 Reviewer 2에서 죽는다.

## 5. 정직한 권고 & 다음 검증
- **현 헤드라인은 novelty 위험이 크다.** "amortized posterior가 본체"라는 직전 합의도 재고 필요.
- 가장 방어가능한 축: **정보한계 이론(CRB) + 한계도달 배포추정기 + (가능하면) 실측 검증**.
- **다음 must-do (조사 심화)**: 최근접 5편 전문 정독 후 novelty 재판정 —
  PINO(2508.03315), U-Net/FNO 재구성(2305.11913), 변분베이즈 UQ phase-resolved 예보,
  ensemble-DA(2012.00677), DA 동시 파+선박운동(2504.13943); + 방향스펙트럼 CRB(어레이처리) 대조.
- 결론을 *내가 단정하지 않는다.* 위 정독 전까지 novelty는 "위험, 미확정"으로 둔다.

## 6. 전문 정독 후 갱신 (2026-07-01, 탈출선 평가)
**정독 확정 사실:**
- **PINO(2508.03315)**: amortized 14 ms, physics-informed(HOSM 1056, GT 불필요). 저자 명시 한계 =
  **1D 단방향만(2D=future work) · UQ 없음 · 실데이터 없음 · buoy/radar(LiDAR 아님).**
- **Zhang 2022(Applied Energy, S0306261922010042)**: phase-resolved 실시간 예측 + **UQ(예보값·예보영역)
  + predictable zone 최대 74.6% 확장**(변분베이즈). → "빠른 예측+UQ+predictable zone" 점유.
- **Predictable zone**: 고정/이동/면적 센서 **closed-form 이론 존재**(Yue 그룹), horizon ~1–3 파주기.
- **DOA CRB**: 어레이처리 교과서. "방향 병목을 CRB로"는 DOA-CRB의 파 적용 → 수학 novelty 약함.
- **directional + experimental validation** 논문 존재(RG 369475098). 단파정 3D-FFT 재구성·예측도 존재.

**탈출선 사실평가:**
1. 정보한계/predictable-zone 이론 → ❌ (closed-form + DOA-CRB 선점). **★ 직전에 "최강 후보"라 한 것 정정·다운그레이드.**
2. 2D 방향장 → ❌ (실험검증까지 점유).
3. 빠른 amortized + 보정 UQ → △ 얇음(Zhang 점유, 델타=ms+명시 calibration).
4. **희소·경사·단일개구 *스캐닝 ship-LiDAR* 관측물리(dropout/incidence/운동)를 역문제에 내장 +
   dropout-aware *보정* posterior** → ⭕ **유일하게 덜 점유** (선행 LiDAR는 깨끗한 샘플러로 취급).

**"네트워크/학습 변경으로 탈출되나?" → 사실상 ❌.** 아키텍처·학습 교체 자체는 novelty 아님(강한
차별자는 전부 문제수준인데 점유됨). 메서드 튜닝은 *센서-niche에 봉사하는 수단*으로만 의미:
SBI(dropout forward model)·incidence 조건부 인코더·dropout 하 calibration 유지.

**살아있는 단 하나의 형상(확정 아님):** "단일·희소·경사 스캐닝 ship-LiDAR 레짐에서 LiDAR 관측물리를
내장한 dropout-aware 보정 posterior로 phase-resolved 복원 + 실측(수조) 검증." 그마저 LiDAR 파 연구
증가로 닫히는 중.

**메타결론(정직):** "amortized posterior가 본체"는 못 살릴 가능성 큼. 살리려면 위 센서-niche로 좁히고
실데이터를 붙이거나, *이 문제가 박사 vehicle로 맞는지 자체를 재고*. 다음: overlap 못박을 3편 정독
(directional+실험검증 / Yue predictable-zone closed-form / Zhang2022 본문).
