# R4 — novelty 3축 문헌검증 + 판정 (2026-08-05)

명제 4: *"남은 novelty 는 방법이 아니라 평가 프로토콜(추정기 유효영역을 외부 관측
기후값으로 층화)이며, 그것은 박사 novelty 로 충분하다."*

| 에이전트 | 판정 |
|---|---|
| PRO | 찬성 — **단 자기 입장 철회**: 프로토콜(절차)은 기여 단위가 아니고, 남는 건 **측정 2문장 + 레시피** |
| CON | **반대(중강)** — 입장 변경: *"개명이므로 무가치"* → **"표준이므로 의무"** |
| NEUTRAL | **반대** (⚠4연속 "명제가 잘못" 판정을 **거부**: "4연속은 판정이 아니라 의식") |

⟹ **novelty 축은 사실상 만장일치로 부정.** 이 프로그램이 2026-08-01 이후 유일하게 열려
있다고 믿었던 축이 닫혔다. **P5 3축 문헌검증 = 완료, 결과는 선점 확인.**

---

## 1. MEASURED — 개념은 최소 5개 분야에서 표준명을 갖는다

| # | 표준명 | 근거 |
|---|---|---|
| 1 | **conditional verification / stratification into quasi-homogeneous subsets** | ★결정적. CAWCR/WMO 검증 지침이 *"pooling 은 비균질 표본에서 성능 변동을 가리고 가장 흔한 레짐으로 편향된다 … 관측 강도·레짐별로 층화하라"* 를 **명시 권고**(Hamill & Juras 2005; Jolliffe & Stephenson 2003 교과서). GMD 2025 는 이를 **Simpson's paradox** 로 명명 |
| 2 | **applicability domain** | **OECD QSAR 검증 제3원칙 = 2004년부터 규제 의무** — "모델은 자기 유효영역을 데이터 분포 기준으로 정의해야 한다" |
| 3 | **transportability / external validity** | Pearl & Bareinboim (*Stat. Sci.* 2014) selection diagram · Degtiar & Rose 리뷰 |
| 4 | **covariate / subpopulation shift, benchmark representativeness** | WILDS · OoD-Bench · full-spectrum OOD · Raji et al. NeurIPS 2021 D&B (construct validity) |
| 5 | **sim-to-real gap 정량화** | SEDD · DoGSS-PCL — ★**LiDAR 점군 modality 까지 동일** |

**해양·파랑 분야 내부 선례 (이게 가장 아프다)**
- ★**DTIC, *Directional Validation of Wave Predictions***: hindcast 방향확산을 **피크주파수 상대
  4개 대역으로 층화 검증**하고, **"관측에 parametric/data-adaptive 모델을 적용하지 않는다"** 를
  설계원칙으로 명시 ⟹ **우리의 "양팔 동일 추정기"조차 기성품**이다.
- ★**FOWD (Häfner 2021)**: "ML 에 필요한 대표적 실측 분포"를 위해 **CDIP 전 카탈로그를 처리**
  ⟹ 우리의 24/194 스테이션 사용은 그것의 부분집합이다.
- **IEC TS 62600-100 / DNV**: WEC 성능을 **Hs–Tp scatter diagram 발생확률로 가중**
  ⟹ 외부 관측 기후값 조건화는 OE 실무에서 **수십 년째 표준**.
- Ewans 1998(세계 확산 측정) · WMO Wave Measurement Evaluation · 부이 representativeness error.
- *Sailing synthetic seas* (Coastal Eng. 2022): **합성 벤치마크를 10년 부이와 Hs–Tm 평면에서 대조** = 직접 선례.

## 2. ★P5 의 "선점 지도 빈칸"은 증거가 아니었다
graphify 72편 코퍼스는 **method 논문 집합**이므로 **구조적으로 검증방법론 논문을 담을 수 없다.**
"빈칸"은 발견이 아니라 **코퍼스 구성의 아티팩트**였다. (⟹ 선점 지도의 부재를 novelty 근거로
쓰는 것은 코퍼스 범위를 먼저 진술해야 한다 = "열람 범위 = 주장 범위" 규율의 문헌 버전.)

## 3. (a)/(b)/(c) 분해 판정 (NEUTRAL, PRO·CON 동의)

| | 내용 | 판정 | 강도 |
|---|---|---|---|
| **(a)** 벤치마크가 비대표적일 수 있다는 **착상** | 3개 이상 독립 분야에서 규범화 | **OLD** | **0** |
| **(b)** **도구**: 양팔 동일 σ₁ 재도출 + 자기결과 층화 | 부품 전부 OLD(σ₁ 은 1988 교과서, 층화는 Hamill&Juras, 양팔동일은 DTIC 설계원칙) | **조립만 신선** | **Methods 절의 성실성 = 필요조건**, 박사 기여 불가 |
| **(c)** **발견**: 40/100 이 실측 최소값보다 좁고 그런 실측 0건 · 결합피복 28.0% · 미피복 축은 Tp · 겹침대역서 LETKF 19/24 | 남이 낼 수 없다(우리 생성기의 속성) | **NEW** | **장(chapter) 1개 + 저널 1편.** 단 **자기감사이지 분야감사가 아니다** |

## 4. ⛔ "최초"는 이제 미검증이 아니라 **거짓**
쓸 수 있는 최대치 = 검색 범위를 명시한 존재-부재 1문장:
> "To our knowledge — search scope: 8 targeted queries across wave-model validation, buoy
> directional-moment climatology, forecast-verification stratification, and
> dataset-shift/applicability-domain literature, plus a 72-paper method corpus — no prior study
> of **sparse-sensor phase-resolved** wave-field reconstruction has reported what fraction of its
> own evaluation sea states lies outside the observed distribution of a buoy climatology, nor
> stratified its reported skill by that external variable. The stratification principle itself is
> established (Hamill & Juras 2005; OECD applicability domain; IEC TS 62600-100)."

## 5. ★그래서 무엇이 박사 novelty 후보인가 — 3인이 같은 곳을 가리켰다

**(B) 표현-불일치 식별가능성 (representation-mismatch identifiability)** — **방법 수준 주장**.
> 유한 이산 성분집합은 방향확산에 **성분을 지불해야** 하고, 국소화 격자 필터는 그렇지 않다.

근거가 이미 모여 있다: n_comp 통제 후 **유일 생존 설명변수가 확산**(부분 ρ +0.460, p=0.0028) ·
유의한 방향은 **WS↔확산 +0.503 (p=0.024)** 이고 LETKF 쪽은 null · 우리 표현에 **연속 D(θ;f) 가 없다**.
**시험이 이미 명세돼 있고 오늘 실행 가능해졌다**(§6).
성공 = 방법 novelty, 실패 = 강한 음성 정리. ⚠ 기제 방향은 **미검정** 표기 필수.

**(A) applicability domain 을 서술 → 작동으로**: (Hs, Tp, σ₁) 발생지지 × 셀별 스킬을 공개
인공물로 내고, 그 위에 **GT-free in/out-of-domain 판별기**를 사전등록해 배포 팔 교체 규칙으로 검정.
GT-free 배포규칙은 이미 **2회 FAIL** ⟹ 실제 미해결 문제이고 양성·음성 모두 출판 가능.

## 6. ★실행됨 — (B) 의 시험이 오늘 가능해졌다
`direction_spread_deg` 배선 완료(§`R3_FINAL` 실행 1). 명목 30→60° 만으로 실현 확산이
n=5/15/30 에서 **33.7 / 44.8 / 41.6°** ⟹ **CDIP 실측 중앙 35.3° 를 성분수 고정 상태로 관통**.
⟹ 확산이 **관측된 공변량에서 조작된 처치로** 바뀐다 ⟹ 현재 부분상관으로는 불가했던
**인과 주장이 가능**해진다. 12ep 교차설계 = 다음 실행 항목.

## 7. NEUTRAL 이 제시한, (c) 를 승격시키는 무비용 작업 (P-B 보다 먼저)
> 위상분해 재구성 문헌 N편(Wang&Pan, Ehlers, Zhang 2022, radar DSWP 계열)의 **공표된 시험
> 해상상태 Hs/Tp/확산을 추출해 같은 CDIP 지지집합에 투사**한다.
> **문헌 다수가 같은 off-support 대역** ⟹ (c) 가 자기감사 → **분야감사로 승격**(박사 기여).
> **문헌은 on-support 인데 우리만 아니면** ⟹ (c) 는 **우리 생성기의 결함 보고로 격하**.
> 새 런 0건, ~3시간. **이것이 P-B on-support 귀속보다 먼저 와야 한다.**

## 8. ⛔ 신규 인용 금지 (22–24)
22. **"평가 프로토콜이 novelty"** — 5개 표준명 확인. 허용: *"기성 규범(conditional verification /
    applicability domain)을 이 서브필드에 이식했고, 그 결과 얻은 측정이 기여다"*.
23. ★**"CDIP 최소 확산 12.18° 를 바다의 물리적 하한으로"** 인용 금지 — 우리 표현에 연속
    D(θ;f) 가 없으므로 두 팔의 σ₁ 은 **같은 추정기이지만 같은 피적분 대상이 아니다**.
    부호는 견고, **임계값의 물리적 해석은 불가**. 허용: *"이 추정기로 24 스테이션 25,426
    레코드에서 **관측된** 최소값"*.
24. **"추정기 유효영역"** 이라는 표현 자체 — ⛔#8 방향 독해 유발.
    허용: **"합성 평가집합 내 유효영역"**.
25. **"선점 지도에 빈칸이 있다"** — 코퍼스가 method 논문 집합이라 검증방법론을 담지 못한다.
    코퍼스 범위를 먼저 진술하지 않은 부재 주장 금지.

## 9. ⚠ 토론 자체에 대한 NEUTRAL 의 경고 (받아들임)
4라운드 / 새 런 **0건** / 문서 7편. 철회 1건은 한 라운드 동안 "settled" 로 동결된 뒤에야 잡혔다.
> *"지금 벌어지고 있는 일이 verdict 제조다 — 다음 산출물은 판정문이 아니어야 한다."*

⟹ **더 이상 토론 라운드를 열지 않는다.** 다음 산출물은 §6 교차설계 또는 §7 문헌투사 =
**측정**이어야 한다.
