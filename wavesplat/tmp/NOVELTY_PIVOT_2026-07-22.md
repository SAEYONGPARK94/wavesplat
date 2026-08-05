# Novelty 축 재고 (옵션 C 피벗) — 더 강한 각도 탐색 2026-07-22

> 사용자 "C부터 해보자"(현 CRB-envelope 축이 modest ⟹ 노벨티 재고). 정직 전제: 학습≈해석 prior(정확도 tie),
> 방법 novelty 0, 필드 crowded(Ehlers·Desmars·Zhao·Nantes/Hamburg). litcheck 인지([[feedback_litcheck_before_novelty]]).

## 0. 냉정한 출발점 — 무엇이 강한 노벨티를 막나
- **필드가 붐빔**: 추정 방법·predictable zone·경험 UQ 다 선점. **어떤 각도도 "landmark" 되기 어려움**(구조적 제약).
- **정직 positive가 얇음**: 학습 tie·blind≈net·아크마다 washout. **강한 positive를 못 만든 게 데이터의 성질.**
- ⟹ **피벗의 목표 = "덜 붐비고 + 우리만의 자산을 쓰고 + NAOE-native"인 각도. "약함"을 완전히 없애진 못함(정직).**

## 1. 우리만의 진짜 자산 (남들이 안 가진 것)
1. **엄밀 CRB 기계** (field·θ·envelope) — phase-resolved 파장에 이 깊이로 구축한 그룹 드묾.
2. **모든 방법이 같은 천장 근처** 실증 (학습≈해석≈ETKF ~0.8, HOS-검증) — **필드-교정적 발견.**
3. **예외적 자기감사·inverse-crime 자적발** — ML-for-physics서 드문 성숙도.
4. **estimator-free 운용성 envelope** (경험UQ 원리 불가) — HOS서 경험UQ 이김(modest).
5. HOS+실측 스테레오+CRB 통합 인프라.

## 2. 피벗 후보 (각각: 무엇·왜 강한가·선점위험·실행)
### C1 — **"정보-한계지 방법-한계 아니다" 달성가능성 이론** (음성을 헤드라인으로)
- **무엇**: 필드의 경쟁 추정기(해석·학습·ETKF)가 **모두 같은 CRB 정보천장 근처**임을 증명 + estimator-free 운용성
  envelope + 천장을 옮기는 유일 레버(센서 개구/기하·외부정보 공급) 규명. **"방법 개선 그만, 센서를 설계하라."**
- **왜 강한가**: 오늘 데이터가 정확히 이걸 지지(학습≈해석≈천장). **필드-교정 이론적 주장 + 실용 결론.** 음성을
  약점 아닌 **중심 발견**으로. 자기감사 성숙도가 이 주장의 신뢰성 뒷받침.
- **선점위험**: 중(predictable-zone/정보한계 논의 존재, but "전 방법 천장 근처 엄밀증명 + CRB envelope"은 미점유).
- **급소(정직)**: E-52는 "estimator-limited + 작은 정보잔여"지 순수 "정보-한계" 아님(CRB Gaussian-근사, GUARD2 미납).
  ⟹ **"정보-한계" 주장하려면 CRB 이론 강화 필수**(random-phase CRB·achievability). 실행: E-51/52 확장 + HOS.
- **판정**: **정직·필드-교정적이나 CRB 강화 없이는 "약한 음성 PhD"로 읽힐 위험.** 강화하면 방어가능.

### C2 — **제어-결합 결정-관련 예보** (NAOE-native, 덜 붐빔)
- **무엇**: 운용성 envelope(언제 믿나)을 **선박운동 제어 결정에 폐루프 결합** — 예보 신뢰도로 제어모드 전환
  (공격적↔보수적). 헤드라인 = "정보이론 운용성 경계가 제어 의사결정을 문 여닫는다."
- **왜 강한가**: 필드는 추정서 멈춤. **제어-결합 + 정보이론 operability = 진짜 gap**(NAOE 청중 직결). estimator-free
  envelope의 "결정 도구" 용도가 경험UQ 대비 delta.
- **선점위험**: 저(추정 crowded이나 제어-결합+CRB operability는 미점유). 단 FutureWaves 등 sea-trial 시스템 존재.
- **급소**: **제어 시연 필요**(sim). 오늘 B-2 제어전달은 약함/경계선. **제어 이득을 sim서 실증해야**(HOS 선박운동 모델).
- **판정**: **가장 NAOE-native·덜 선점. 단 제어 데모가 관문**(현재 미보유). 성공하면 positive-framed 강한 각도.

### C3 — **센서/궤적 설계 최적화** (estimator-free CRB의 설계 용도)
- **무엇**: CRB이 센서스펙 폐형 ⟹ **센서 배치/사양/궤적을 운용성 envelope 최대화로 최적화** = naval architect 설계도구.
- **왜 강한가**: 추정 아니라 **설계** = 다른 문제류·덜 붐빔. estimator-free가 핵심(데이터·추정기 없이 설계).
- **선점위험**: 중(active sensing/D-optimal 문헌 있음; 오늘 궤적정책은 realistic서 죽음).
- **급소**: "CRB를 설계에 적용"이 incremental로 읽힐 수. 궤적축은 오늘 washout(realistic). **정적 센서-설계 sweep은 생존 가능.**
- **판정**: NAOE-native이나 novelty가 "응용"에 가까움. C2보다 약함.

## 3. ★추천 (정직)
- **1순위 = C2(제어-결합)**: 가장 NAOE-native·덜 선점·positive-framed. **관문 = HOS 선박운동 sim서 "envelope-게이트
  제어"가 naive 제어를 이김 실증.** 이게 되면 현 modest 자산(envelope)이 **결정-관련 강한 헤드라인**으로 승격.
- **2순위 = C1(정보-한계 이론)**: 가장 정직·필드-교정적. **관문 = CRB 강화(random-phase·achievability)로 "정보-한계"
  주장 봉인.** 음성 헤드라인이라 심사 리스크 있으나 엄밀하면 방어가능.
- **C1+C2 결합이 최강**: "정보-한계 이론(왜 방법 개선 무의미) + 그 한계 안에서 제어-결합 운용성(무엇을 해야)."
- **⚠정직 경고**: 어느 피벗도 crowded 필드·sim-heavy·thin-positive 제약을 완전 극복 못 함. **최선 현실 = modest→
  solid(landmark 아님).** 강한 novelty 원하면 **실물 센서 데이터 확보가 유일한 판도 변경자**(모든 각도 공통).

## 4. 다음 (사용자 결정)
- **C2 착수**: HOS 선박운동 sim + envelope-게이트 제어 vs naive 비교 설계·실행 (제어 데모 = 관문).
- **C1 착수**: CRB 강화(random-phase/achievability) + "전 방법 천장 근처" 엄밀화.
- 또는 두 관문 다 열어 결합.
