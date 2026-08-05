# M-3 prereg — E-B2K 센서-입력(F1) 완화, 1단계=eval-side kp-beamform (2026-07-22, 동결)

> F1(E60K §5): E-B2K 훈련·평가가 실제로는 **oracle kp**(참 스펙트럼 피크)를 받음 ⟹ "센서-실현" 주장 불가.
> HOS 아레나서 유일 oracle 입력 = `kp = sea['kp']`(NOISE는 고정 센서-스펙). σ₀도 실측아레나 관례 0.05·Hs는
> 별개; HOS는 NOISE 고정이라 kp만 oracle. [[project_e60_b2_headline_audit_2026-07-19]] 후속.

## 질문 (1단계, eval-side)
동결 11바다 실패창(hand<0.5, E-15 oracle-hand로 **고정**)에서, 추정기가 kp를 **600 센서점에서 beamform**
(kp_hint=None → net_gamma_mmse 내부 2단 beamform_top)하면 T−L(+0.139)·T−N(+0.107) 우위가 **생존**하는가?

## 방법 (동결)
- 동결 아레나(e60k_hos_arena.json)와 **동일 rng 스트림**(900000+7919·si+31·r)으로 draw 재현 → 관측 bit-동일.
- **hand는 동결값(oracle-kp) 재사용** → 실패창/11바다 집합 불변(사전등록 집합서만 판정, peeking 방지).
- 9팔(T/L/N×s0/s1/s2)만 **kp_hint=None**으로 재평가. 나머지 전부 동일.
- 주 통계 = 동결과 동일: 실패창(11바다) T−L·T−N, 바다-군집 t + wild, 단측.

## ★사전 예측 (arm-symmetry — 결과 전 기입)
세 팔이 **같은** 센서-kp를 받으므로 짝지음 Δ는 대체로 생존해야 함(F1이 "판정 무영향(팔-대칭)"이라 명시).
- **생존 = SUPPORT**: T−L ≥ +0.08 AND 부호 ≥8/11 AND wild p<0.05 ⟹ 결과가 oracle-kp 아티팩트 아님(eval-side 센서-강건).
- **붕괴 = FLAG**: T−L < +0.05 or 부호 뒤집힘 ⟹ oracle-kp 의존 존재(진짜 발견, 헤드라인 재검토).
- 그 외 PARTIAL.

## 스코프 (인용 의무)
- **이건 부분 완화**: eval-side만. **완전 "센서-실현"은 훈련도 센서-kp여야**(d3_b2deep_train의 kp_e→beamform
  재훈련 9ckpt, ~1h) = 2단계(별도, 사용자 go 후). 단 head 입력(beamform power map X)은 **이미 센서-유래**라
  oracle은 grid-centering kp뿐 ⟹ eval-side 생존이면 F1 노출 대부분 해소.
- 이중검증: 독립 재도출 + (헤드라인-인접이므로) fable 재확인 요청.
