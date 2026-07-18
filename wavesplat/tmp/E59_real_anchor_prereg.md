# E-59 사전등록 — §8 실측 anchor (조건③, 옵션 A) 2026-07-18

> 사용자 "A부터 해보자, fable과 경쟁적으로". 옵션 A = 스테레오 실측서 envelope↔skill 재현 확인.
> 규율 전면([[feedback_direction_flip_discipline]]·[[feedback_dual_verification_default]]): 실행 前 fable
> 적대 red-team, 사전등록 kill, 대칭감사, MEASURED/ARGUED. **fable이 이 prereg를 먼저 깬다.**

## 질문 (좁게)
실제 스테레오 파랑에서 **CRB 운용성 envelope이 sparse-obs 예보가 실측 표면과 맞는 곳을 rank-예측**하나 —
**합성 §8의 부호/크기를 matched 개구비서 재현**하나? (예측 우위 재검 아님 — 부호·envelope 유효성 재현만.)

## 데이터 접지 (detrend 후, 실측)
| rec | patch(m) | Hs_det | λ_p | kp | spread | **aper/λ** | valid% | 신뢰 |
|---|---|---|---|---|---|---|---|---|
| AA15a | 65×85 | 1.53 | 32 | 0.194 | 32° | **2.00** | 65 | ✅ primary |
| AA14 | 70×86 | 1.14 | 27 | 0.231 | 40° | **2.58** | 43 | ✅ primary (valid 낮음) |
| YS17 | 220×240 | 1.54 | 47 | 0.135 | 36° | **4.72** | 57 | ✅ primary (well-det단) |
| LJ18g | 250×300 | **22.1**(비물리) | 301 | 0.021 | 38° | 0.83 | 100 | ⛔ exploratory-only |
- **★핵심 접지**: 신뢰 실측은 aper/λ **2.0~4.7 = 합성 modest(5comp kp0.208≈AA15a) 레짐**. **강-envelope 深-under-det
  레짐(aper/λ<1)엔 신뢰 실측 없음**(LJ18g Hs22 비물리·sub-λ 패치·[[project_e42_e43_regime_external_2026-07-17]]
  단일의존). ⟹ **실측 anchor는 modest 레짐 부호 재현이 1차 목표**; 강-레짐은 실측 불가(합성 한정 캐비엇).
- 실측 spread 32~40° = 합성보다 넓은 다방향(더 어려움). NaN 구멍 35~57%.

## 설계 (frozen)
- rec = {AA15a, AA14, YS17} primary. LJ18g = 별도 exploratory(1차 판정 제외).
- **detrend**: 프레임별 평면(a+bx+cy) 제거.
- **obs 윈도우 슬라이딩**: t_now ∈ 여러 위치(rec당 ≥4 윈도우) → 윈도우별 Spearman → median+CI(에피소드류 반복).
- **sparse obs**: obs 윈도우 [t_now−W, t_now] 프레임서 valid 픽셀 무작위 부분표본 N_obs≈1500(dense 금지=너무 쉬움/
  tautology). 잡음 2cm(Falcon) 추가.
- **θ̂ (FIM 전용)**: kp(FFT peak) 중심 k-그리드 ring[0.35,1.75]kp, dense-obs ridge-LS 진폭 → 유의 성분 유지 =
  비오라클 θ̂. FIM=(1/σ²)JᵀJ(sparse-obs 기하), Finv.
- **질의 q** = grid × lead{0,2,4,6s}(rec 길이/fps 상한). t_now+lead.
- **σ_CRB(q)** = √(J_qᵀ Finv J_q)/Hs. **σ_actual(q)** = |k-그리드 ridge-LS forecast(sparse obs) − Z_stereo(q)|/Hs
  (**θ-free = tautology 없음**; Z_stereo=미래 프레임 bilinear). **σ_emp(q)** = √(Φ_q Ainv Φ_qᵀ)σ/Hs. **dist** = 최근접 obs.
- 위상-평균 없음(실측은 단일 실현) — 대신 윈도우 다수로 안정화. (합성 R-평균과 다름 = 캐비엇.)

## 검증 통계 + 판정 (frozen)
윈도우×rec별: **Spearman(σ_CRB,σ_actual)** [주] · partial|dist · emp · dist.
1. **SUPPORT(anchor 성립)**: median Spearman(CRB,act) **>0 (CI 하한>0)** AND partial|dist>0 AND matched 개구비서
   합성 modest(~0.2~0.4)와 부호·크기 일관 ⟹ **envelope↔skill 실측 재현**(스테레오·비이동 캐비엇 병기) = 조건③ 충족.
2. **FAIL(실측 미재현)**: median ≤0 또는 partial 흡수(거리) ⟹ 실측 파랑서 envelope 구조 없음 ⟹ **헤드라인 합성-only
   후퇴**(옵션 C). fable에겐 이게 kill.
3. **PARTIAL**: 0~0.2 또는 rec간 이질 큼 ⟹ CI 보고, 라벨 강제 금지(규율6).

## 대칭 감사 (사전 — 규율4)
- **과대**: 거리 confound(partial 필수)·dense-obs tautology(sparse 강제)·θ̂-tautology(σ_actual θ-free라 회피)·
  k-그리드 오라클(비오라클 ring).
- **과소(비관 리스크)**: 실측 짧음/잡음→θ̂ 불안정→CRB 잡음→ρ 하락(합성 estimator-limited 재현, envelope 무효 아님);
  스테레오 GT 자체 오차(σ_actual에 잡음 주입); NaN 구멍; 넓은 방향 spread(합성보다 어려움); **단일 실현(위상평균 없음)**.
- **★최대 캐비엇**: 실측=스테레오 유래, **실물 이동 LiDAR 미관측**([[project_observation_operator_audit_2026-07-13]]).
  "실측 anchor"는 = 스테레오-정지-표면 재현이지 센서-이동 재현 아님. 절대 "LiDAR 검증됨" 주장 금지.

## fable 적대 red-team 요청 (실행 前)
①이 설계에 치명적 결함? ②θ̂-FIM이 σ_actual과 은닉 tautology? ③sparse 표본/detrend가 결과를 만들거나 없앨 경로?
④kill 기준이 정직한가(FAIL 트리거가 실제 발동 가능한가)? ⑤matched-개구비 비교가 정당한가? ⑥더 정직한 대안 설계?

## 구현 순서
1. `tmp/e59_real_anchor.py`: loader+detrend+sparse+θ̂+CRB+ridge-forecast+stats.
2. smoke(1 rec·2 윈도우) → **fable red-team 반영** → primary 3 rec × ≥4 윈도우 → fable 결과 감사 → 화해.
