# E-54 다채널 이식 — 초안 (Opus 입장, fable 토론용) 2026-07-18

> E-48 조건② / 학위 관건: 5요소 회계 프로토콜이 regime을 넘어 해류·개구 채널로 이식되나?
> **규율 적용**([[feedback_direction_flip_discipline]]): 각 factor를 MEASURED / ARGUED / UNDERPOWERED /
> [from-memory·미검증]으로 태그. 첫 읽기 **의도적 과소주장.** 값은 대부분 3~22일 된 메모리 = 재검증 필요.

## 5요소 (regime 예시서 정의)
value(공급되면 과제개선) · recoverability(blind 추론 가능?) · supply(외부공급·등급/환율) ·
achievability(실현 공급 존재?) · info-limit(회수실패가 정보냐 추정기냐, CRB 판별).

## 채널별 이식 상태 (태그 포함)
| factor | **regime** (예시, MEASURED) | **current/Uc** | **aperture/개구** |
|---|---|---|---|
| value | +0.072 field / +0.043 조우(n=957) ✅MEASURED | 해류 Doppler가 sim-실측 갭 [from-mem, ARGUED] | 이득 크기 지배(ρ+0.69, 18/18) ✅MEASURED(E-16) |
| recoverability | blind≈net ✅(arena X·Y) | blind 5/5 실패 [from-mem, **UNDERPOWERED n=5**] | N/A (외부-info 채널 아님) |
| supply | 부이 ±5° 필요·±25° 지불 ✅ | 외부 Uc [ARGUED 구조유추] | N/A |
| achievability | 부이급 부재→사산 ✅ | **radar/ADCP 해류 제품 실재** → regime보다 나을 수 [ARGUED, 미측정] | N/A |
| info-limit | CRB 결과 B(추정기-급) ✅ | ❌ 미측정(해류 CRB 없음) | ✅ direction1_crb(D_eff·T_wall·정보벽) |

## ★핵심 구조 통찰 (첫 읽기, ARGUED)
**채널이 두 種類다:**
1. **외부-정보 채널**(regime·current): 정보를 blind 추론 못 해 **공급**해야 함 → 5요소 회계 대상.
2. **기하/CRB 축**(aperture): **공급 불가**(센서+궤적이 고정). 이건 채널이 아니라 **정보벽을 세우는
   백본**(=Direction-I `d1_crb.py`) — 채널들이 그것에 **대해** 측정되는 기준.
⟹ "다채널 이식"은 **두 가지를 섞고 있었음.** 정직한 일반화 = 5요소 회계는 **외부-정보 채널**에
적용(regime→current), aperture는 **별개의 CRB 백본**(이미 존재).

## ★정직한 판정 (과소주장)
- 5요소 회계가 **완전 계측된 채널 = regime 1개뿐.**
- **current = 구조 유추 + recoverability 1건(n=5, UNDERPOWERED)** — value/supply-등급/achievability/CRB
  **전부 홈.** ⟹ "이식됨"은 **ARGUED이지 MEASURED 아님.** (fable 07-18 경고가 옳았음: "재료완비" 낙관.)
- aperture = **채널 아님, CRB 백본**(Direction-I) → 5요소 프레임에 억지로 넣는 건 **범주 스트레치.**
- ⟹ **학위-급 "일반 프로토콜" 주장은 현재 데이터로 미성립.** 정직한 현 상태 = **"한 채널(regime) 심층
  회계 + 프레임워크 제안 + 두 번째 채널(current) 구조 스케치(미계측)"**. 이건 "검증된 다채널 프로토콜"보다
  **modest.**

## ★current이 오히려 더 강할 가능성 (양성 실마리, ARGUED·미측정)
regime의 배포가 사산한 이유 = 부이급 입력 부재. **current은 다르다: radar/ADCP 해류 제품이 실재**
(운용 해양학서 흔함). ⟹ **current은 achievability를 통과할 수 있다** = regime이 못 한 "배포 가능한
외부-공급 채널"의 실례가 될 수 있음. 단 **value·등급·CRB 미측정** — 이걸 재려면 실험 필요(재서술 아님).

## 다음 (규율대로 — 이식 前 사전등록)
1. **current 채널 dose 사다리**(regime E-43 유사): 참 Uc 공급 → value 측정; 등급별 환율; blind 재실행(n≥96).
   achievability = radar/ADCP 등급이 지불구간인지. **홈이라 재서술 아닌 실험.**
2. **aperture는 채널 회계가 아니라 CRB 백본으로 재배치** — Direction-I를 프레임의 "정보벽 축"으로 명시.
3. 개구 baseline-특이성(E-22/E-27: net-vs-well-specified-MMSE 한정)을 **이식 전 사전등록**(fable 지목).

## 토론 질문 (fable에게)
Q1. current "이식"이 진짜인가, 아니면 n=5 위의 희망인가? UNDERPOWERED면 "이식됨" 주장 자체 보류?
Q2. aperture=백본(채널 아님) 재프레임이 정당한가, 아니면 프로토콜 일반성을 못 보이니 범주로 도망친 건가?
Q3. "외부-정보 채널 프로토콜"(regime+current)이 진짜 통일 스코프인가, 아니면 사후 협소화인가?
Q4. current-achievability 양성 실마리(radar/ADCP)가 thesis의 배포 사산을 부분 구제하나, 아니면 또 다른
    "합성서 생존·실측서 죽음" 반복 위험인가?

---

# ★fable 반박 + 화해 (최종, 2026-07-18) — 내 초안이 양방향으로 틀렸음

**내 초안 오류 (규율4 대칭감사가 잡음 — 둘 다 인정):**
- **과잉 비관**: ①"blind 5/5 = UNDERPOWERED n=5"는 **종류 오분류.** 실제 = **식별성 논증**(단일 sparse 트랙은
  (트랙파수, 조우주파수)만 관측 → 해류 시프트 비식별, evidence 면이 평탄 아닌 **편향**) + **5 목적함수 전멸** +
  **oracle 대조 통과**(Uc+kp 기지 → track 0.294 ≈ 무해류 0.311 완전복원). 정직 태그 = **레거시-MEASURED(07-05
  아레나, 재검증 필요).** ②"value 전부 홈" 거짓 — 07-05 dose 실재(0.15 m/s hand 전멸, 0.3 m/s net 75%↓, oracle 복원).
- **과잉 낙관(누락)**: achievability 희망 섹션이 **내 프로젝트 최강 측정치 누락** — **kp_known 조건선 해류 손실
  미미(−0.002~−0.04)**, 주 해악통로 = kp/대역배치(beamform Doppler 취약)지 분산관계 아님 ⟹ **값싼 kp_hint가
  해악 대부분 회수 ⟹ 외부 Uc 한계환율이 이미 판정바 이하일 수 = current은 achievability 아니라 VALUE서 죽을 수.**
- 사소: "해류 Doppler=sim-실측 갭"은 **진단**이지 공급-**가치** 아님(다른 양); aperture ρ+0.69만 인용(07-14 ρ+0.41 p0.02도 존재).

**Q1~Q5 화해:**
- Q1: "이식됨 보류" 유지하나 근거 교정 — 정직 라벨 = **"레거시-MEASURED(타 아레나) + 현 아레나 미이식."**
- Q2: aperture=백본 재프레임 **정당**(ex-ante 판별: 궤적외생·센서고정은 07-05 잠긴 제약; 5요소 중 supply·
  recoverability가 aperture엔 **파싱 불가** = 범주오류 = 채널 아님). 단 ★통찰로 팔면 과대 — 실은 v2 §9 "재료완비"의
  **오산 정정**(비용: 채널 3→2). **과소 정정**: CRB 판별기 **2클래스 시연**(성분급 d1_crb + 파티션급 E-51/52) =
  소박한 일반성 증거.
- Q3: 협소화 맞으나 **독립 실체 있음** — 통일 클래스 = "sparse 트랙서 비식별이나 보조계기 측정가능한 생성모델
  파라미터"(regime·current·**수심 d**[해도공급, achievability 자명]이 3번째 자연 멤버). 단 완전계측 1개로 "프로토콜"
  명명은 시기상조 → **"회계 프레임워크: 1채널 완전 + 2채널째 사전등록."**
- Q4 (★문헌 접지): **X-band 선박레이더**(파랑-유효 해류 = 종류일치) **0.04~0.07 m/s**(Lund 2020), 드리프터
  RMSE **4 cm/s**(Lund 2018); **HF radar** 6~19 cm/s(공칭 ~7). **★regime과 부호 반대**: regime 필요±5° vs 가용
  ±21.9°=**4.4× 나쁨**; current kill-dose 0.15 m/s vs 가용 0.04~0.07=**2~3.75× 좋음.** 유병률도 반대(연안 해류
  0.1~0.5 m/s **편재** vs cross-Tp 희소). **BUT**: 필요등급=value-유지지 kill-dose 아님(곡선 미측정); kp_hint 반대신호;
  실측 anchor(측정 Uc 주입) = regime 4중 사망 자리, 미실행. **판정: achievability는 regime보다 구조적 강함(접지)
  — 구제 여부는 VALUE 다리가 결정, 거기 유일 기존 측정은 비관 방향.**
- Q5 (★가장 중요): **"이식=채널 생존"은 내 암묵 오류.** 회계 프로토콜 이식 기준 = **채널 생존 아니라 5요소가
  검정력 있게 판정 완료.** current이 "kp수정후 value 미미 + achievability 통과"로 나면 실패 아니라 **프로토콜
  두 번째 성공**(regime=achievability사망 / current=value사망 = **프레임워크가 실제 판별함**) + 배포지침("radar 말고
  kp_hint 고쳐라"). 내 초안은 이 등식오류로 학위관건을 실제보다 어렵게 그림.

## ★최종 판정 (화해): 다채널 일반성 = **부분(PARTIAL)** — 미성립 아님
"1채널 완전계측(regime) + 1채널 레거시-측정(current, 타 아레나·재검증 필요·value 반대신호 포함) + CRB 판별기
2클래스 시연." **내 "미성립"은 반 단계 과소, "이식됨"은 명백 과대.** "부분→2채널 판정완료" 거리 = **연구 리스크
아니라 사전등록 실험 1개.**

---

# E-55 사전등록 — current 채널 dose 사다리 (frozen, fable 제안 + 규율 적용)
> 이식 기준 = **5요소 검정력 판정완료**(채널 생존 아님). value사망도 프로토콜 성공으로 계상.
- **아레나**: 현 regime 아레나에 `ω → ω + k·Uc` 추가 (기존 `d3_sa_uspeed.py` 기반, 현 아레나로 이식).
- **검정력(규율1)**: ≥3 동질블록, SE ≤ 0.010.
- **팔/판정**:
  (a) `value_raw` = oracle-Uc 상금 (참 Uc 공급 gain)
  (b) **`value_marginal | kp_hint`** ← **일차 위협가설·주 판정** (kp 힌트 준 뒤에도 외부 Uc가 가치 있나)
  (c) **등급 사다리** {0, 4, 7, 15 cm/s} — 실계기(X-band 4~7·HF ~7·kill 15)에 앵커
  (d) `blind` 1종(GP-evidence) 검정력 재실행
  (e) **FIM에 Uc 열 추가** → 정보벽 vs 효율벽 판별(E-51/52 방식)
- **킬 기준(사전명기)**: `value_marginal(4~7 cm/s) < 0.03` → "current = value-사망 채널"로 기입 **(프로토콜 성공
  계상)**. `value_marginal ≥ 0.03 + achievability 통과` → "배포가능 외부-공급 채널"(regime이 못 한 것).
- **규율**: 발견≠발표(복제블록만), 새통계 격리(FIM-Uc 기존거동 확인), 대칭감사, MEASURED/ARGUED 태그.
