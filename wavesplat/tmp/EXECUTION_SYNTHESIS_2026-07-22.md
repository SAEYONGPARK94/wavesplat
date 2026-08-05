# 실행 종합 — 브레인스톰 열린실마리 전건 수행 (Opus×fable, 2026-07-22)

> 사용자 지시: "추천대로 + 열린 실마리 전부 수행, fable과 분담." 실험은 규율대로(사전등록→실행→이중검증→
> MEASURED/ARGUED). Opus=헤드라인-결정 실험, fable=분석·측정·문서정합. [[project_brainstorm_next_2026-07-22]].

## 결과 매트릭스
| 작업 | 담당 | 판정 | 권위 파일 |
|---|---|---|---|
| M-1 kp↔hand 통합 | Opus | 방향 5/5 일치·미봉인(ARGUED) | tmp/M1_RESULT_2026-07-22.md |
| **C8 cross-estimator** | Opus | **스코프-한정 SUPPORT / 무조건 REFUTE** | (본문 §C8) tmp/C8_prereg |
| M-3 센서-kp(F1) | Opus | **FLAG: +0.139→+0.022(beamform eval)** | tmp/M3_RESULT_2026-07-22.md |
| M-3G 재훈련 GATE | Opus+fable | **VOID(no-op)·이중검증: 훈련이 이미 beamform** | tmp/M3G_RESULT_2026-07-22.md |
| M-4 ≥4λ 실측 | Opus | **확대 불가(YS17 유일)·음성** | tmp/M4_ge4lambda_finding |
| F-1 Gaussianity | fable | MEASURED(§7 보정=ARGUED 유지) | tmp/F1_gaussianity |
| F-2 CRB 엔진 무결성 | fable | **§4=1엔진 참 / §4↔§5=별개(재서술)** | tmp/F2_crb_engine_integrity |
| F-3 다채널 조건② | fable | **CLOSED(문서정합만)** | tmp/F3_multichannel_reconcile |
| F-4 B-2 기제 | fable | 정밀화(T만 배포-질의 채점) | tmp/F4_b2_mechanism |
| F-5 level-calibration | fable | **MEASURED 재도출** | tmp/F5_level_calibration_reframe |

## C8 — cross-estimator envelope test (헤드라인 결정)
**질문**: σ_CRB이 자기 ML twin(dense ridge) 아닌 다른 추정기 클래스 오차도 예측? = "estimator-독립" 유일 직접시험.
sandbox=e58(연산자 격리), 추정기 3클래스: E_ridge(dense 선형)·E_relax(sparse 비선형 파라메트릭)·E_gp(nonparametric 단일모드).

### 잘-결정(5comp, n=16) — 완료
| 추정기 | crb-act median | partial\|dist | skill | competence |
|---|---|---|---|---|
| ridge | +0.286 CI[.248,.338] | +0.132 | 0.333 | competent |
| **relax** | **+0.612 CI[.587,.641]** | **+0.385** | 0.225 | competent |
| gp | −0.016 CI[−.048,.013] | −0.059 | 0.240 | competent |
- agree ridge-relax +0.339 · ridge-gp +0.150 · relax-gp +0.169.
- **동결 판정 = REFUTE/PARTIAL**(gp REFUTE + agree<0.4) — **무조건 "estimator-독립"은 실패**.
- **해석(MEASURED)**: envelope이 **relax(다른 클래스)를 ridge보다 더 잘 예측**(+0.61 vs +0.29, 거리보정 후에도)
  ⟹ **"ML twin만 예측" 반증**. 그러나 단일모드 GP(정보버림)는 미예측 ⟹ **정직한 형태 = "센서 파수정보를
  resolve하는 추정기 클래스엔 독립(ridge·relax), 정보버리는 단일모드엔 아님"**(스코프 한정). E-59 미검증 갭 해소:
  **헤드라인 "estimator-독립"은 이 한정형으로만 인용 가능**, 무조건형 인용금지.

### under-det(15comp, n=20) — 완료 ★
| 추정기 | crb-act | partial\|dist | LOO |
|---|---|---|---|
| ridge | **+0.719** | **+0.541** | [.719,.719] |
| relax | **+0.680** | **+0.574** | [.664,.697] |
| gp | +0.157 | +0.065 | [.133,.181] |
- agree ridge-relax **+0.551**. regime 대비(under>well) MW p: ridge 0.0000·relax 0.0022·gp 0.0000.
- **★한정형 SUPPORT 확정**: under-det서 ridge·relax(다른 클래스 2개) 강예측+상호동의 = **estimator-독립 봉인
  (운용상 중요 레짐)**. gp 약예측 = 스코프("resolve 클래스").
- **이중검증 2/2 완료**: 독립집계+LOO+MW / fable 독립 재구현 CONFIRM(자체 OMP, CRB↔ridge+0.556·↔OMP+0.749·
  상호+0.506, 상호동의 내 +0.55와 일치). 권위 `tmp/C8_RESULT_2026-07-22.md`·`tmp/F6_c8_verify_2026-07-22.md`.

## M-3 — 센서-kp(F1) = **FLAG (완료)**
동결 11 실패바다를 센서 beamform kp로 재평가: **T−L +0.139→+0.022(~6× 축소, 경계선 wild .043, 7/11)·
T−N 소멸(+0.007 n.s.)**. 최난셀 Tp13_Hs5 T−L +0.613→+0.056(beamform kp +37% 편향, L·N 회복). **내 사전예측
(arm-symmetry 생존) 반증** — F1 "판정무영향" 전제 거짓, B-2 극적우위는 상당부분 oracle-kp 덕. SELF-CHECK
bit-identical. 캐비엇: eval-side(train/test 불일치) ⟹ 완전판정=센서-kp 재훈련 필수. ⟹ **B 챕터 헤드라인 아님
확정·재훈련 관문 필수(A 척추 무영향).** `tmp/M3_RESULT_2026-07-22.md`.

## ★종합 헤드라인 영향 (최종, 전건 완료)
- **A 척추(envelope) = 강화**: C8이 "estimator-독립"을 **한정형으로 봉인**(relax·ridge, 다른 클래스 2개, under-det서
  강·이중검증) + gp 스코프. = E-59 최대 미검증 갭 해소. **무조건형 인용 금지**(gp 반례). F-2가 §4↔§5 통합서술 수정 요구.
- **B 챕터(loss-swap) = 약화·정직 캐비엇 (강한 정정)**: M-1(통합축 방향-일치·미봉인) + **★M-3/M-3G**:
  훈련 kp는 처음부터 beamform(생성코드 확인·이중검증) ⟹ **E-B2K +0.139는 eval서 참-피크(oracle) 먹인 값=운용
  불가·인용금지**, **beamform-eval 값 M-3 +0.022(경계선)가 sensor-realized 정답**, 재훈련 gate=no-op(회복불가).
  ⟹ **B 헤드라인 아님 확정 + E-B2K 아레나를 M-3(beamform)로 재판정해야 정직.** A와 독립.
- **운용성 claim = 강**: F-5 level-calibration MEASURED. M-4 = 실측 under-det는 YS17 상한(하드 한계, 이동 LiDAR만 경로).
- **⟹ 순수익 = 논문 무결성↑**: 강한 자산(A) 근거 획득, 약한 자산(B) 정직 캐비엇. 남은 건 실험 아니라
  **문서청소 6건 + 집필 + 지도교수 합의.**
- **문서청소 7건**: ①"estimator-독립"→한정형(C8) ②§4↔§5 엔진 재서술(F-2) ③다채널 조건② 문서정합 4건(F-3)
  ④§8 level-calibration 재구성(F-5) ⑤"census ≥4λ 3개" 인용정정(M-4) ⑥RAW_SINK 훅 버그(F-3·F-5)
  ⑦**★B-2 "+0.139" 인용금지 → M-3 "+0.022"(beamform-eval)로 대체·F1 "훈련=oracle kp" 정정(M-3G).**
