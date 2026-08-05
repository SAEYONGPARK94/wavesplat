# M-3G prereg — E-B2K 센서-kp 재훈련 GATE (2026-07-22, 결과 전 동결)

> M-3(eval-side, oracle-훈련 헤드)가 FLAG(T−L +0.139→+0.022): oracle-kp 의존. 이는 **train/test 불일치**
> 가능성(헤드가 oracle-kp에 튜닝)을 배제 못 함. **GATE = 훈련도 센서-kp로 하면 T−L이 회복하나?**
> [[project_execution_2026-07-22]]·M3_RESULT. 원래 E-B2K 워크플로(fable red-team→실행→fable 감사) 준용.

## 설계
- **유일 변경 = 훈련 손실의 kp**: `d3_b2deep_train.py`의 `sample_loss`가 쓰는 `float(kp_e)`(oracle 스펙트럼
  피크)를 **beamform kp**(scene의 P,z에서 2단 beamform_top, M-3 eval과 동일 recipe: klim=2.0 n=120 →
  klim=3.5·kp n=120)로 교체. 나머지(arch·init·lr·steps 12000·seeds·data bwtask·σ₀=nfrac·Hs) **전부 E-B2K와 동일.**
- kp_beam은 8000 훈련 scene에 대해 **1회 사전계산·캐시**(`tmp/bwtask_kp_beam.npy`), 9회 재훈련 재사용.
- **평가 = M-3 센서-아레나**(kp_hint=None/beamform), 새 checkpoint `b2deep_{arm}_s{s}_beam.pt` 9개.
- 판정 = M-3와 동일: 동결 11 실패바다(hand=oracle 고정, 집합 불변) T−L/T−N, 바다-군집 t + wild, 단측.

## ★사전 판정 규칙 (동결) — 세 가설 판별
| 결과 | 해석 | B-2 헤드라인 |
|---|---|---|
| **T−L ≥ +0.08 AND ≥8/11 AND wild p<0.05** | (b) M-3 FLAG는 train/test 불일치였음, **센서-훈련이 회복** | **PASS: B-2 센서-실현 생존** |
| **T−L ≤ +0.05 (M-3 수준 유지)** | (a) **진짜 oracle-kp 의존** | **FAIL: B-2 센서-실현 불가** |
| 중간(+0.05~+0.08) | 부분 회복 | PARTIAL |
- 최난셀 Tp13_Hs5도 병기(oracle T−L +0.613, M-3 +0.056 대비).
- 검정력: k=11(M-3와 동일 집합). MDE 등 M-3와 동일.

## 사전 예측 (내 것, 반증 대비 명시)
M-3서 이미 내 "생존" 예측이 틀렸으므로 이번엔 **중립**: (b) 회복도 (a) 유지도 유의미. 개인 사전확률 ~55% (b)회복
(smult 지문 T 3.2 vs L/N 1.8이 kp와 무관한 훈련신호축이라 재훈련이 복원할 여지) / 45% (a)유지.

## 캐비엇
- σ₀는 E-B2K 원본대로(nfrac·Hs) 유지 = kp 변수만 격리(σ₀는 M-3서도 미변경 ⟹ kp gate에 confound 아님).
- beamform kp는 훈련·평가 **동일 recipe**(2단 n=120)라 train/eval 일관.
- **VOID**(훈련 퇴행): 시드간 smult std>0.5 시(E-B2K 규칙 준용).
- 이중검증: fable 설계 red-team(사전) + 결과 독립감사(사후).
