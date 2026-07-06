# 방향 결정 — 2026-07-02 세션 종합

> 사용자 출발 질문: "추정도 안 되는데 forecast가 되는 게 박사논문으로 적절?" + "우리 데이터/방법이 현실과
> 너무 다른가?" → 하루치 검토·실측분석으로 답을 냄.

## 확립된 사실 (전부 이번 세션 실증)
1. **코드 clean** (버그 0), sim 약점=방향구조(비선형 아님).
2. **nowcast 단일프레임 복원 = 물리 ≈ naive** (정보/개구비 한계). 헤드라인 될 수 없음.
3. **forecast + unobserved 외삽 = 물리가 명확히 이김** (BS: skill 0.76@1s, 외삽 margin +0.078, 둘 다 naive
   원리적 불가). ← **진짜 positive.**
4. **개구비-파장이 binding 변수**: La Jument 0.11λ는 forecast·외삽마저 실패, BS ~1λ는 성공. = **설계 법칙.**
5. 방법 fix(적응 klim)는 짧은 파 실측에 유용, CV 모델차수는 중립. `estimator_improved.py` 하드닝.
6. **amortized NN + UQ**(d56/d58): 8.5ms·보정 UQ·forecast>persistence는 되나, **point 정확도 0.25=trivial
   붕괴**(feedforward-collapse). UQ·speed·forecast는 자산, 재구성 정확도는 정보벽에 막힘.

## 권장 방향 (headline 피벗)
**"희소 ship-LiDAR로부터 실시간 단기 phase-resolved 파 FORECAST — 개구비 식별성 경계(CRB 법칙) + 보정 UQ."**

- **Headline positive**: dispersion 기반 forecast가 persistence를 operational lead(0.5~3s)서 이기고, 관측
  없는 영역으로 외삽 — **실측 스테레오서 검증**. interp/naive가 원리적으로 못 하는 것.
- **경계 법칙(핵심 기여)**: forecast·외삽은 **개구비 ≳1λ**서만 작동(La Jument 0.11λ=전부 실패). 얼마의
  개구가 필요한지 = **설계 법칙**. 이게 "왜 nowcast가 tied인지"(정보벽)도 설명.
- **배치 레이어**: amortized NN = ms 추론 + 보정 UQ(불식별 케이스 flag). 정확도 한계는 정직히 같은 정보벽.
- **정직한 경계**: nowcast 정보한계는 *실패가 아니라 boundary condition*.

## 박사논문 적절성 (사용자 질문 직답)
"추정이 안 됨"이 아니라 **"단일프레임 재구성은 정보한계(증명가능한 경계)이고, 그 안에서 forecast·외삽이
interp가 못 하는 값을 준다"** — boundary 규명(CRB) + 달성가능한 것 입증(forecast/외삽) + 배치가능(fast+UQ).
**coherent·honest 박사 서사.** 어제 역설은 역설이 아니라 정보이론적으로 당연.

## 리스크 & 차별
- novelty: DSWP forecast(Ehlers·Desmars) 존재 → **차별 = 희소 oblique ship-LiDAR + CRB 개구비 operability
  법칙 + 실측 검증 + 보정 UQ의 conjunction** ([[NOVELTY_survey]] 방어선과 일치).
- 확증 필요: B를 **다레코드(A 다운로드분)로 확장** + operability 곡선(개구비 vs forecast skill) 정량화.

## 다음 실행
1. A 완료 → B를 AA·YS·다BS 레코드로 확장 → **개구비 vs forecast-skill operability 곡선**(설계 법칙 실측).
2. amortized posterior의 collapse 완화(재구성 정확도↑) — 되면 fast+UQ+forecast 배치 완성, 안 되면 UQ/forecast로 positioning.
3. paper: headline을 nowcast→forecast+operability+UQ로 피벗.
