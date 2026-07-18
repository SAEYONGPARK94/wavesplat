# E-59 §8 실측 anchor (조건③, 옵션 A) — fable 경쟁심사 2라운드 + 최종 = QUALIFIED PARTIAL 2026-07-18

> 사용자 "A부터 해보자 fable과 경쟁적으로". 프로세스 = 사전등록 → **fable red-team(설계)** → C1-C7 v2 →
> **fable 결과 감사** → 버그 2개 수정 → 재판정. 규율([[feedback_direction_flip_discipline]]·
> [[feedback_dual_verification_default]]) 전면. 스크립트 `tmp/e59v2_real_anchor.py`, 원자료
> `results/direction3/e59v2_real_anchor.json`, prereg `tmp/E59_real_anchor_prereg_v2.md`.

## 프로세스 (경쟁심사가 양방향 오류를 다 잡음)
1. **fable red-team(설계) = "실행 불가, FATAL 3"**: ①kill 발동불가(리드풀링 t²·거리가드 dynamic range 0)
   ②공유기저 tautology(θ̂가 예보 k-그리드) ③단일실현 감쇠(완벽 envelope도 R=1서 ρ≈0.175). **내가 ③ 산술
   독립검증**(rel₁=0.289, R12/R96 예측 0.297/0.322≈실측). → C1-C7 반영 v2(strip-obs·within-lead·scramble
   null·spatial-RMS·operator-detrend·ring-clip·twin·blocked-boot).
2. **1차 실측 = 내가 FAIL 선언(오류)**: Δρ +0.147 CI[−0.06,+0.30]∋0.
3. **fable 결과 감사 = "FAIL 미인가"**: 버그 2개 발견(둘 다 Δ 하향편향) + FAIL 라벨이 규율 위반.

## fable가 잡은 버그 2개 (수정 완료, 원자 검증)
- **Bug A (마스크 비대칭)**: null이 `np.ones_like(valid)`로 전점 평가 vs real은 valid-마스크(구멍 검열) →
  Δ가 서로 다른 q-집합 차이(MNAR 구멍). 수정=null도 동일 valid 마스크.
- **Bug B (surrogate 분산 결손)**: scramble이 NaN을 0으로 채워 Parseval상 분산=valid_frac×truth(0.66~0.81×) →
  null이 forecast-geometry로 재가중 → **ρ_null 부풀고 Δ 눌림, 저-valid 레코드서 최악.** 수정=surrogate를
  nanstd(truth|valid)로 재스케일. 둘 다 **내 FAIL 방향으로 편향**했음.

## 버그수정 후 결과 (frozen ridge 1e-3)
| | crb-act | null | **Δρ(real−null)** | blocked-CI |
|---|---|---|---|---|
| TWIN(합성 model-class) | +0.796 | (음수) | **+0.946** | [+0.472,+1.105] |
| REAL(스테레오 AA15a/AA14/YS17) | +0.707 | +0.617 | **+0.085** | **[+0.022,+0.113]** |
- **★버그수정이 CI를 0 제외로 뒤집음**: 레코드별 Δ가 일치로 수렴(AA15a +0.022·AA14 +0.082·YS17 +0.092;
  버그판은 −0.07~+0.31 이질). **Δρ = 작지만 3레코드 일관 양수, CI 하한 +0.022>0.**
- crb≈dist≈emp≈0.7(전부 기하). 실측이 model-class 천장의 **~8~19%**만 회수(twin Δ 0.11~1.16).

## 강건성 (fable 통제 반영)
- **★RIDGE_REL 민감(fable가 지목한 진짜 knob; 내 topK 스윕은 무효—forecast는 topK 무관)**: ridge 1e-3→1e-2→1e-1서
  null +0.62→+0.21→−0.12, **Δ +0.085→+0.346→+0.497**(CI 전부 0 제외). frozen 1e-3은 **out-of-strip forecast
  폭주로 null 오염 = Δ+0.085는 보수적 하한**; forecast-안정 ridge면 Δ+0.35~0.50. ⛔라벨 위해 ridge 선택 금지—1e-3 보고.
- **레코드 자격(ring_lo≤0.6=자격, 작은패치는 ring-clip이 model대역 잠식)**: YS17만 QUALIFIED(ring_lo0.35,
  Δ+0.092 CI[+0.028,+0.301]); AA15a(0.88)·AA14(0.68) band-crippled. **실측 스테레오 작은 패치는 under-det
  레짐을 원천적으로 못 시험**(super-patch 성분 배제 or edge 아티팩트) → YS17(대패치 4.7λ)만 진짜 aperture room.
- **⚠ twin이 linear·in-band = 천장 과대**([[feedback_validation_nonlinearity]] HOS 지시 위반) → real/twin 비율
  **과소평가**(공정 HOS twin이면 real이 상대적으로 더 좋아 보임). ~8~19%는 상대성능 하한.

## 최종 판정 (규율: 대칭·power-gate·goalpost 동결)
- **내 1차 FAIL 철회**(버그+미인가 라벨; CI∋0=INCONCLUSIVE지 FAIL 아님, power-gate=라벨 금지).
- **조건③ = QUALIFIED PARTIAL (약-양성, 미-refuted, 미-clean-validated)**: CRB envelope이 실측 스테레오서
  **작지만 통계적으로 견고한 coherent-skill 신호**(Δρ≥+0.085 CI 0제외, ridge↑서 +0.5까지·3레코드 일관), 단
  model-class 천장의 ~8~19%, 예측력 대부분 기하. **정보-우위 아님·정보-0도 아님.**
- **2단 해석 유지**: Level 1(운용) envelope이 실측 skill 위치 예측(crb+0.71, emp 동급) = 양성; Level 2(novelty)
  기하 너머 정보 = **작지만 0 초과**(버그수정 후). E-58(CRB=emp 예측동률)과 정합 = 방어가능 가치 = estimator-
  독립+판별(정보-우위 아님).

## 캐비엇·인용
- **최대 캐비엇 불변**: 실측=스테레오 유래, **실물 이동 LiDAR 미관측**; strip=정지기하(이동센서 아님).
- estimator-독립 헤드라인은 이 시험 **무증거**(테스트 추정기=CRB의 ML twin) — cross-estimator 별건(C8).
- **⛔인용금지**: "실측이 envelope 반증"·"실측 정보내용=0"·"실측 검증됨(+0.707 raw)"·"§8/조건③ clean PASS"·
  ridge 골라 큰 Δ 인용. **인용가능**: 버그수정 후 Δρ+0.085 CI[+0.022,+0.113](frozen·보수), ridge↑서 +0.5,
  실측 model-class ~8~19%, YS17 QUALIFIED Δ+0.092, twin linear=천장과대→비율하한, 스테레오 작은패치 under-det 미시험.
## ★후속(magnitude calibration, fable 통제#4) — 운용 주장 RESCUE (2026-07-18)
rank-Spearman이 못 보는 축: σ_CRB이 실측 오차 **LEVEL**을 예측하나? query를 σ_CRB decile로 bin → bin별 median
σ_actual. **REAL 3레코드 전부 강한 단조 level-calibration**: level-ρ **AA15a +1.00·AA14 +0.92·YS17 +0.89**,
저→고 decile서 오차 **5~6× range**. TWIN은 12~51× range(level-ρ 0.96~1.00). ⟹ **envelope이 실측서 오차 위치·
크기를 강하게 예측(=운용성 지도 주장 성립)**, 단 model-class 대비 **dynamic range 압축**(실측 오차 floor=모델
mismatch), 순서는 안 깨짐. **⚠caveat: level-calibration도 일부 기하(거리)** — "정보내용" 증명 아님, but 운용
주장(어디/얼마 오차)엔 충분. `RAW_SINK` 훅 추가(엔진).

## ★최종 종합 (2단 축, 화해 완료)
- **운용 축(calibration)**: 실측서 **강함**(level-ρ 0.89~1.00, 5~6× range) = envelope이 실측 오차 예측 = 운용성 지도 성립.
- **정보 축(Δρ vs null)**: **작지만 0초과**(+0.085 CI[+.022,+.113], ridge↑서 +0.5). 기하 너머 정보 증분은 작음.
- ⟹ **조건③ = 운용-지지·정보-우위 미주장** = **estimator-독립+판별** 가치축과 완전정합(정보-우위 아님).
  E-58(합성 CRB=emp)·E-59(실측 CRB≈기하 rank·but level 강함) 일관. 약-양성에서 **운용-지지**로 상향(정직).
- **다음(선택)**: 공정 HOS twin·≥4λ 추가레코드(census 8중 3)·forecast-안정 ridge. 헤드라인 합성-스코프+조건③
  "운용-calibration 실측 성립, 정보증분 작음" 각주.
