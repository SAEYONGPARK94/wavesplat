# E-50 방향 점검 — CRB가 올바른 keystone인가? 무엇을 확립하나? (2026-07-18)

> 사용자 지시: "방향 점검하고 crb로 가줘." E-49(killer-② 해소) 후 keystone ③(5-dim θ CRB) 착수 전 점검.
> 목적: CRB가 (a) 올바른 도구인지 (b) E-43c와 모순 없이 무엇을 확립/반증하는지 (c) E-19 함정을 피하는지
> **실행 전에** 못박고, 사전등록 판정 경계를 정한다.

## 1. CRB가 해야 할 일
E-49가 음성을 **추정기-급**으로 확정: "이 고정-shape blind 전수열거는 배포 net을 어느 아레나서도 못 이김."
학위 승격엔 이를 **정보-급**으로 올려야 함 — "어떤 unbiased estimator도 이 관측 기하서 파티션 θ를 못 짚음."
유일 장치 = θ=(d1,a1,d2,a2,w)의 **Cramér–Rao Bound** (관측 기하·잡음 given).

## 2. ★핵심 긴장 — E-43c와의 모순 위험 (반드시 직면)
E-43c: 우도가 **참 노드를 정확히 짚음** (true-node 백분위 **0.996**, +36 nats vs permuted, node-oracle +0.104).
⟹ **FIM이 비퇴화 = θ가 우도상 식별된다.** 그렇다면 naive CRB(θ)는 **작을** 개연이 크고, 그건 "정보는 있다"
= **info-wall을 오히려 반증**한다.
- **정직한 예상**: d1(우세방향)은 **tight**, **d2/w(2차 파티션)가 관건.** E-49의 blind≈net도 이걸 시사 —
  둘 다 d1을 잡고, 느슨한 d2/w는 field corr 가치를 안 더함(oracle prize +0.069가 거기 잠김).
- ⟹ CRB의 **진짜 질문**: **√CRB(d2)·√CRB(w)가 prior 폭에 필적하나(=2차 파티션이 정보한계)** vs
  **prior보다 훨씬 좁나(=식별됨 ⟹ blind≈0은 정보 아닌 결정/사후기하 문제).**

## 3. ★CRB는 "이기는 실험"이 아니라 "판별 실험" — 두 결과 다 정직·논문화 가능
| 결과 | 조건 | 함의 |
|---|---|---|
| **A. info-limit** | √CRB(d2) ≈ prior 폭 (관측이 d2에 정보 거의 못 더함) | **정보-급 폐쇄 ✓** — blind≈net을 CRB로 기제 설명. 학위 승격. |
| **B. decision-limit** | √CRB(d2) ≪ prior 폭 (d2 식별됨) | blind≈0은 **결정/사후기하** 문제지 정보 아님 → thesis는 **추정기-급 유지**(E-43c와 정합). 정직 강등이나 여전히 논문 기여(왜 식별되는데 회수 못 하나 = 다봉 사후 조건부평균 손실). |

**핵심**: CRB는 A이길 바라고 돌리는 게 아니라 **A/B를 가르려고** 돌린다. 어느 쪽이든 전진 —
결과를 원하는 방향으로 미리 정하지 않는다(arc의 forking-path 반성).

## 4. 모델 주의 (E-45) — random-phase 진짜 모델 vs Gaussian-근사
진짜 파동 = **결정적 진폭·랜덤 위상**(비-Gaussian); blind 우도(JointEngine) = **앙상블-Gaussian 근사**
`z ~ N(0, C(θ))`, `C(θ)=Φ diag(γ(θ)) Φ^T + σ²I`. CRB는 가정 모델에 의존:
- **1차 = Gaussian-근사 CRB** (추정기들이 실제 쓰는 모델의 한계 = estimator류에 정합). 계산가능:
  `FIM_ij = ½ tr(C⁻¹ ∂ᵢC C⁻¹ ∂ⱼC)`, `∂ᵢC = Φ diag(∂ᵢγ) Φ^T`, `∂ᵢγ`는 lobe_grid의 유한차분.
  인프라 존재(JointEngine.__init__의 Φ·C 구성 재사용).
- **진짜 random-phase CRB** = 근본 한계(더 어려움) → **후속**. 1차 결과를 이 캐비엇과 함께 보고.

## 5. E-19 함정 회피 (필수 가드)
E-19 실패 = 스칼라 informativeness proxy가 **개구비와 공선**(개구 통제 시 partial p0.14 = 예측력 소멸).
신 CRB는 **θ-파티션 전(全) FIM**이라 성격이 다르나, **같은 함정 재발 가능**:
- **가드 1**: 개구비(관측 span/λ)를 공변량으로 저장, √CRB(d2)의 **개구-부분상관** 검정. 공선이면 "CRB=개구
  재명명" 경고(E-19 재판).
- **가드 2**: CRB를 **field-recoverability와 연결** — CRB(θ)가 예측하는 achievable field corr을 E-49의
  실측 blind/oracle과 대조(CRB가 실측 achievability를 예측하면 진짜 bound, 아니면 model-artifact).

## 6. 설계 (사전등록 예정 — 다음 단계서 코드+독립감사)
- **FIM(θ=d1,a1,d2,a2,w)** per-sea, Gaussian-근사, **arena X와 arena Y 둘 다** fresh n=64.
- 보고: **√CRB per-param** (특히 √CRB(d2)[deg], √CRB(w)) **vs prior 폭** (d2 분리 U(60,150)°, w).
- **개구-부분상관** (가드1), **CRB→achievability 연결** (가드2, E-49 oracle/blind 대조).
- 사전등록 A/B 경계값 freeze **전에 arena X 기존 데이터서 CRB 거동 확인**(E-49 rule4 실수 재발 방지 —
  새 통계는 기존 아레나 거동 먼저 본다).
- 검증: 독립 재도출 + fable (E-49와 동일 3중 규율).

## 7. 판정 — CRB는 올바른 keystone (단 "판별 실험"으로 정직 프레임)
E-49가 "폐쇄는 추정기-급"이라 못박았으니 CRB가 **정보-급 여부를 가르는** 유일한 다음 관문이 맞다. 단
**A(승격)를 보장하지 않으며**, E-43c 때문에 **B(추정기-급 유지)도 실질 가능**하다. 둘 다 정직한 학위 재료.
⟹ **진행**: 이 설계로 CRB 구현 → smoke → 독립감사 → 실행 → 3중검증. (E-19처럼 개구-공선으로 무너지면
그것도 결과 — 재명명 경고와 함께 보고.)

---

# E-51 결과 (n=64) + 독립 FIM 감사 (2026-07-18)

스크립트 `tmp/e51_crb.py`, 원자료 `results/direction3/e51_crb.json`, 감사 스크립트 `tmp/e51_audit.py`.

## 결과 (arena X 주, well-specified) — median √CRB
```
d1 (우세방향)  11.1°           prior~104°(U0-360)   tight ✓ (GUARD2 기대 충족)
d2 (2차방향)   16.0°           prior std 26°        0.62×  ← 경계선
w  (에너지비)  0.146           prior std 0.087      1.68×  ← CRB>prior, 91%(58/64) sea>prior [ROBUST]
Tp2           0.57s           prior std 1.31s      0.44×
FIM cond median 31 (well-conditioned = θ 국소식별 = E-43c 정합; near-singular은 sep 작을 때만)
```

## 독립 감사 (opus, 신선) — **수학·코드 버그 없음 / 해석 강등**
- **build_C ↔ JointEngine 공분산 일치**(★핵심): logdet 5자리·상대오차 4e-7. `K=Mtot·(vz/stot)+σ²I`,
  원자([g,g])·정규화(NB-sum)·scale 정확 재현. **PASS.**
- **FIM 공식** `½tr(C⁻¹∂ₖC C⁻¹∂ₗC)` 정확(zero-mean이라 완전), **PD·대칭 확인**. **EPS 0.5×/2× 불변=수렴.**
  medX 전부 재현, NaN 0. **PASS.**
- **GUARD1 (E-19 재판) PASS**: aperture vs √CRB(d2) raw −0.28, **부분상관 −0.30**(sep·r2·Tp2·cond 통제서도
  유지, 붕괴 안 함) = **개구 재명명 아님.** 단 aperture는 약한 다수동인 중 하나(cond +0.44·sep −0.36·r2 −0.34가
  더 강함) ⟹ CRB(d2)는 진짜 multivariate.

## ★정직한 판정 (감사 반영) — keystone ③ **부분 이행**
- **A-partial (모델-클래스 info-limit, 2차 파티션)**: **robust 신호 = w**(미식별, 1.68×prior, 91% sea).
  **d2는 경계선**(0.62, 문턱 0.6을 0.43°차로 겨우 넘음 = 동전던지기 ⟹ 헤드라인 금지). **d1 tight.**
  ⟹ "2차 파티션(특히 에너지비 w)이 정보-제약, 우세방향은 잘 결정" = blind≈net 기제 후보.
- **⚠ 스코프 (인용 전 필수)**:
  1. **Gaussian-근사 CRB** = 이 모델 쓰는 추정기(blind·net 둘 다)의 한계 ⟹ blind≈net 설명엔 적절하나
     **진짜 random-phase 근본 정보벽 아님**(후속). "정보-급 폐쇄"는 **모델-클래스 정보한계**로 스코핑.
  2. **GUARD2 (achievability) 미이행 = 필수 미납**: CRB(θ)→field-corr cap을 E-49 실측 blind/oracle과
     대조해야 "θ 부정밀이 blind≈net을 만든다"가 성립. prize가 2차 lobe에 +0.069를 걸어놨으므로(field가 θ에
     민감) "w/d2 부정밀→field 캡"은 자명하지 않음. **θ→field 선형화(또는 θ~N(true,CRB) 몬테카를로 field)
     로 예측 achievable corr을 E-49와 대조**가 다음 필수 단계.
- **⟹ keystone ③ 상태**: θ-CRB **계산·검증 완료**(2차 파티션 w 정보-제약 확립). 그러나 **정보-급 폐쇄 확정
  인용엔 (a) GUARD2 achievability (b) random-phase CRB 두 조각 미납.** E-51은 필요조건이지 충분조건 아님.

---

# E-52 GUARD2 achievability (n=64) + 독립 검증 — ★keystone③ STRONG form 기각, 판별=결과 B

스크립트 `tmp/e52_crb_achievability.py`, 원자료 `results/direction3/e52_crb_achievability.json`.
θ̂~N(θ_true,CRB) 몬테카를로(M=32)로 field 만들어 corr = "CRB-efficient plug-in 추정기의 field". arena X si=700-763.

## 결과 (재도출 정확 일치)
```
oracle_reg − cur         +0.0856* CI[+0.060,+0.111]  med +0.071
crb_limited − cur        +0.0455* CI[+0.024,+0.068]  med +0.033  win 40/64  ← CRB 천장이 net 위
oracle_reg − crb_limited +0.0401* CI[+0.030,+0.051]  med +0.030  win 59/64  ← 진짜 정보한계 잔여
사전등록 판정 = NOT CONFIRMED.
```

## ★독립 검증 (opus) — crb_limited는 천장(ceiling)이지 achievable 아님
- **crb_limited는 correct-family·unbiased·efficient plug-in의 상한(ceiling)으로 유효**하나, verdict 문자열
  "beats the net"은 **realizable 추정기가 이긴다로 읽혀 과대.** 낙관편향 3중(전부 천장 올림): ①correct 2-lobe
  shape 부여(net엔 없음) ②unbiased 부여(실제론 w 미식별 1.68×prior → prior로 shrink → corr↓) ③Gaussian-근사.
- **★E-49와 결정적 대비**: 실제 blind(FULL) = net **+0.012 (CI∋0)**, oracle −0.057. 가상 crb_limited = net
  **+0.046**, oracle −0.040. ⟹ **실제 blind는 CRB 천장에 도달 못 함(~+0.034 아래)** = +0.046은 **efficiency-gap/
  ceiling이지 회수된 headroom 아님.** (오목성 인공팽창은 배제 — oracle−crb=+0.040>0로 천장이 oracle 아래.)

## ★정직 종합 (E-51+E-52) — keystone③ 재판정
파티션 field 회수 = **두 벽**:
- **(i) 진짜 θ-정보벽 = oracle − crb_limited = +0.040** (2차 파티션 w 미식별, **모델-클래스**).
- **(ii) crb_limited − net = +0.046 = 정보 아니라 efficiency-gap/천장** — 실제 blind(E-49 FULL≈net)가 도달 못 함.

⟹ **keystone③ STRONG form("정보가 net에서 캡한다 = 음성을 정보-급으로 승격") = 기각.** 정보벽은 net이
아니라 **oracle−0.040(=crb_limited)**에 앉음 = 관측은 net/blind가 뽑는 것보다 +0.046 더 informative.
⟹ **방향점검의 판별 = 결과 B(decision/estimator-limit)** 쪽. E-43c(우도가 θ 짚음)·검사(prosecution)의
"정보 아닌 결정 문제" 우려가 **CRB로 확증됨.**

**단 음성 척추는 생존**: E-49의 blind≈net(realizable 추정기 사실)은 E-52가 **뒤집지 못함**(실제 blind가
천장 미도달). 바뀌는 건 **"정보-급 승격 야망"뿐** — thesis는 **추정기-급 폐쇄(E-49)로 유지**되고, 정직한
프레임 = *"모델-클래스 CRB가 oracle−0.040서 캡(2차 파티션 w), net→천장 +0.046은 추정기 비효율(실제 blind도
회수 실패)"*.

**캐비엇**: Gaussian-근사 CRB라 이 천장(crb_limited)도 낙관적 → random-phase CRB 후속 필요(천장 더 낮을 수).

## keystone③ 최종 상태
- ❌ **STRONG form(정보-급 폐쇄, 음성을 net에서 정보-급 승격) = 기각** (CRB 천장이 net 위).
- ✅ **음성 척추(blind≈net, 추정기-급, E-49) = 생존**(E-52 미반증).
- ✅ **작은 진짜 정보 잔여 +0.040**(2차 파티션 w, 모델-클래스) = 정직히 인용가능.
- ⟹ 논문 프레임: "정보의 벽"이 아니라 **"추정기-급 폐쇄 + 작은 모델-클래스 정보 잔여"**. 학위 승격 장치로서의
  CRB(음성→정보-급)는 **실패**; 대신 CRB가 **판별해준 것(결과 B)**이 정직한 기여.
