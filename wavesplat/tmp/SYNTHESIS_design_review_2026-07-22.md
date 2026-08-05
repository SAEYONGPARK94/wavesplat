# 종합 — kp-robust 설계 + 학습방법 ground-up 리뷰 (Opus×fable, 2026-07-22)

> 사용자 "설계해줘 + 다른 네트워크/학습법 바닥부터 검토". Opus=kp-robust 설계, fable=방법 지형 리뷰.
> 권위: `tmp/DESIGN_kp_robust_2026-07-22.md`(설계) + `tmp/GROUNDUP_methods_review_fable_2026-07-22.md`(리뷰).

## A. kp-robust B-2 설계 (Opus) — 요지
- **병목(MEASURED)**: 단일 스칼라 kp가 **입력 정규화 + bank 격자** 둘 다 지배. beamform kp 편향(장파·소개구서 심함)이
  전 파이프라인 열화. 현 EM은 γ만 정련·kp 미교정.
- **★미묘점**: B-2 우위가 부분적으로 **"정확한 kp가 window-inside 팔 over-fit 유발"의 아티팩트**일 수 있음
  (M-3: 편향 kp서 L·N 회복→갭 축소). ⟹ 목표 분리: **(a) T−L 갭 복원**(chapter B, 아티팩트면 불가) vs
  **(b) 센서-실현 추정 개선**(더 가치·정직). **권고=(b) 우선.**
- **후보(우선순위)**: ①kp-EM 정련(무재훈련·물리정론, 1순위) ②kp-jitter 재훈련(갭복원 결정시험) ③multi-kp 앙상블
  ④절대격자 ⑤불확실성-인지 bank. staged 사전등록.
- **정직**: spine A(envelope) 무관·이미 강함. B가 salvage 안 돼도 thesis 무손상.

## B. 학습방법 ground-up 리뷰 (fable, ★자체정정 반영 최종본) — 요지
- **~40개 아키텍처 이미 시도.** SUCCESS(RELAX·net-MMSE·beamform-detect·**d56 fast amortized posterior**·궤적정책) /
  FAIL(raw좌표회귀·T-NSE·DETR·점군-native attention·FNO·자기지도인코더·track-critic ≥15건) / PARTIAL(DPS·SmultHead
  [상수환원]·E-28[원결론 07-17 기각]).
- **패턴(핵심 진단)**: FAIL=전부 "raw 표현→point-estimation"(다봉 posterior MSE최적=평균=mean-collapse).
  SUCCESS=전부 "네트워크는 저차원 물리-정규화 prior만, 재구성은 닫힌형 물리 solve". ⟹ **표현이 아니라 목적함수/분해가 관건.**
- **★set-diffusion 정정(fable 원문 재확인)**: "DPS가 RELAX 전승(0.208)"은 **초기 d43 관측-기아 수치**. 최종 엄격검증
  (d51/52 F1-F4): 비학습 dense-ridge **0.233** ≈ DPS **0.234 = 동률**, "~0.233=연속스펙트럼 정보한계, 아무도 못 넘음."
  ⟹ **정확도로 최선 baseline을 유의하게 이긴 유일 사례 = set-diffusion 아니라 d56 fast amortized posterior**
  (0.222 vs F1 0.234, p<0.01, 8ms, **단 실시간-사이클 제약 안에서만**; 긴 창 허용시 RELAX 재추월 0.240).
- **★3번째 범주(중요)**: 조우예보 0.10→0.90~0.99는 정보/표현 아니라 **검증-프로토콜 버그(n-독약: 창-안 held-out으로
  잡음바닥 하이퍼 고르면 창-밖 열화)** — 고치니 **학습 없이** 뛰었음. ⟹ **"학습 필요"의 상당수가 깨진 baseline.**
  **B-2/M-3의 창-안 vs 창-밖 교훈과 정확히 같은 축** = 대칭경고(새 net 전에 "이게 진짜 한계냐 깨진 baseline이냐" 물어라).
- **레짐별 갈림**: well-det=**진짜 정보벽**(≥15확증) / under-det=**정량 알고리즘 갭**(FIM 0.71~0.93 vs RELAX 0.27~0.54,
  학습 이기는 곳) / **센서 궤적=유일하게 벽을 옮김**(정책 greedy ×2.5~4.4).
- **제안(실제 우선순위 1→2→3→5→4)**: ①**궤적정책 완결**(최우선) ②**SBI/NPE로 keystone③ CRB Gaussian-근사 검증**
  (GUARD2 미납 보강, spine A 직결) ③k-grid CNN equivariance ⑤coarse-beamform 조건 attention(반증위험 최고)
  ④**[하향] set-diffusion UQ-형태만 distill**(정확도 우위 없음 정정 ⟹ 다봉-UQ 값만, 먼저 다봉성 진단·음성이면 폐기).
- **★재고금지(이중선점 신규확인)**: **FNO/DeepONet**(F4 내부死 + Ehlers OE2023이 U-Net vs FNO 벤치해 FNO승 이미 보고) ·
  **PINO/PDE-residual 자기지도**(같은 Ehlers 그룹 2025-08 arXiv 2508.03315 발표 = 봉인) · SIREN·점군 transformer·track-critic.

## C. ★교차 종합 (전략 권고)
1. **두 트랙이 같은 진단으로 수렴**: kp-취약성 = "물리-앵커의 가장 약한 단"(2026-07-05 verdict) = fable의 "SUCCESS
   레시피의 물리-정규화 앵커가 강점이자 약점". ⟹ kp-robust 패치(A)는 **앵커 하드닝**, fable의 ③⑤(equivariance·
   attention)는 **앵커 자체 교체 여부** — 상보. 근본 회피는 후자.
2. **⟹ 전략 재배치**: chapter B(loss-swap) salvage보다 **fable #1(궤적정책 완결)이 최고가치 학습방향** — 유일하게
   **정보벽을 옮기는** 축이고(×2.5~4.4), thesis(센서-인지 운용성)와 직결. **movable axis에 학습노력 집중이 정직·고효율.**
3. **kp-robust는 값싼 것만**: kp-EM(무재훈련) 1건은 (b) 개선 여지 있어 시도 가치. kp-jitter 재훈련은 (a) 갭이
   아티팩트일 위험이라 sweep(full 11-sea) 결과 보고 결정. **B salvage에 큰 재훈련 투자 = 비권장**(아티팩트-한계).
4. **정직 프레임**: well-det는 벽(네트워크 그만), under-det+센싱설계는 여지 ⟹ **학습의 자리 = under-det 알고리즘
   갭 + 궤적정책.** 이게 A척추(정보 대 효율 판별)와 정합적 = "학습이 이기는 곳을 정보이론이 예측".

## D. 다음 (사용자 결정 대기)
- **추천 1순위**: fable #1 궤적정책 완결(연속화·HOS·실추정기 결합) — 설계·검증 착수 가능.
- **값싼 병행**: kp-EM 정련 1건(무재훈련, (b) 검정).
- **보류**: chapter B kp-jitter 재훈련(sweep full 결과 선행), set-diffusion flow-matching(방법 기성, 응용).
