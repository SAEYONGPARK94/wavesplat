# E-58 §8 독립감사 결과 — envelope skill 검증 PARTIAL 정련 2026-07-18

> 사용자 요청(Discord): "§8 PARTIAL 독립감사 이어서 + 헤드라인 정련". 규율 전면 적용
> ([[feedback_direction_flip_discipline]], [[feedback_dual_verification_default]]): 사전등록 감사·
> 검정력 게이트·대칭감사·MEASURED/ARGUED 분리·goalpost 동결(PASS 바 사후변경 금지).
> 스크립트 `tmp/e58_audit_rederive.py`(원자료 재도출)·`tmp/e58b_audit.py`(감사 실행).
> 원자료 `results/direction3/e58b_audit.json`. 기존 결과 `results/direction3/e58_envelope_validation.json`.

## 감사 질문 (2개)
1. **[핵심] estimator 효율이 +0.33을 과소평가했나?** ridge-LS 비효율/유한-R 몬테카를로 잡음이 CRB↔actual
   상관을 눌렀나, 아니면 well-determined 레짐의 진짜 천장인가?
2. **헤드라인 정련**: "예측 우위"(emp-UQ 대비)가 죽었으므로 남는 가치축이 무엇인가?

## STEP 1 — 원자료 재도출 (기존 24-ep JSON, 무-계산)
요약 검증됨: CRB-act med **+0.329**, partial|dist **+0.210**, emp **+0.332**, dist **+0.317**.
- **★강한 이질성 발견**: 5comp(kp0.208, well-det) CRB-act med **+0.299** / 15comp(kp0.096, under-det) **+0.756**.
  - partial|dist: 5comp +0.181(min −0.02, 95% >0) / 15comp +0.594(min +0.45, 전부 >0, 거리 초과 생존).
  - CRB vs dist: 15comp에서 CRB가 거리 **4/4** 초과(+0.141); 5comp에선 근소(+0.019, 13/20).
  - **Spearman(comp, ρ) = +0.646 (p=6e-4)**, Spearman(kp, ρ) = −0.526 (p=8e-3).

## STEP 2 — 감사 실행 (n=36 신선, seed0 고정, tautology 가드: k-그리드 ring/NK 고정·비오라클)

### PART A — under-determination dose-response (헤드라인 축)
| 레짐 | n | kp | CRB-act | partial\|dist | emp-UQ | dist |
|---|---|---|---|---|---|---|
| 2comp | 6 | 0.168 | +0.524 | +0.339 | +0.266 | +0.382 |
| 5comp | 10 | 0.208 | +0.307 | +0.146 | +0.302 | +0.291 |
| 15comp | 10 | 0.096 | +0.742 | +0.566 | +0.711 | +0.585 |
| 30comp | 10 | 0.089 | +0.777 | +0.634 | +0.776 | +0.632 |

- **Spearman(kp, ρ) = −0.874 (p=3e-12, n=36)** ← 주축. **Spearman(ncomp, ρ) = +0.770 (p=4e-8)**.
  comp축은 2comp에서 비단조(2comp>5comp)이나 **kp축은 깔끔**(2comp kp0.168이 5·15 사이 → ρ도 사이).
  ⟹ **진짜 축 = kp(우세파장 대비 개구 = under-determination)**, [[project_kp_floor_audit_2026-07-14]] 개구비축 정합.
- **★emp-UQ 동률이 전 레짐서 유지**: 5comp 0.307≈0.302, 15comp 0.742≈0.711, 30comp 0.777≈0.776.
  ⟹ **"예측 우위" 어디서도 없음(견고)**. CRB 가치축 = 예측력 아니라 estimator-독립+판별.

### PART B — "estimator 효율이 +0.33 과소평가했나?" → **답: 아니오 (3중 확인)**
- **(b1) 유한-R 감쇠 무시가능**: σ_actual 신뢰도 rel=**0.975**, σ_CRB rel=0.999. ρ@R12 0.287→ρ@R96 0.321.
  **분리반분 감쇠보정 ρ = 0.326 ≈ 관측** (5comp). 15comp: 0.741→0.763, 보정 0.769. **몬테카를로 잡음이
  더 강한 상관을 숨기고 있지 않음.**
- **(b2) ridge-bias 무관**: ρ가 λ∈{1e-4,1e-3,1e-2}서 평평 — 5comp 0.317/0.320/0.336, 15comp 0.768/0.760/0.800.
  **더/덜 효율적 추정기로 바꿔도 안 움직임.**
- ⟹ **modest 5comp ρ≈0.33 = well-determined 레짐의 진짜 천장, 측정 아티팩트 아님.**

### STEP 3 — 천장 해석 검증: well-det는 skill map이 평평(예측할 게 적음)
σ_actual 공간 분산도(CoD=IQR/med): 5comp **0.470** vs 15comp **0.883** vs 30comp **0.846** (~2×).
⟹ well-det 레짐은 skill이 공간적으로 균일 → CRB·emp·dist 모두 modest ρ서 동률. **envelope 실패가 아니라
"운용성 균일하게 높음, 경고할 게 적음"을 올바로 보고**. envelope은 skill이 실제 변하는 under-det서 값을 함.

## 종합 판정 (규율: goalpost 동결)
- **§8은 사전등록 예측바(ρ≥0.5 AND ≥emp-UQ)로는 여전히 PARTIAL** — 사후 재정의로 PASS 선언 안 함.
- **감사가 PARTIAL의 의미를 정련**:
  1. sub-0.5 median은 **pooled** 수치 — well-det 지배. 분해 시 0.31(well-det)→0.78(under-det).
     **≥0.5 바는 under-determined 레짐(운용상 관건)서 통과**, well-det서 미달.
  2. well-det 미달은 **estimator 비효율 아님**(감사 배제) — **예측할 spatial 구조가 적은 진짜 천장**(flat map).
     실패 모드 아니라 envelope의 정직한 "경고할 것 없음".
  3. emp-UQ 동률은 **견고·예상됨** — CRB 가치를 "더 나은 예측"→**"estimator-독립+판별"**로 재배치
     (Zhao 대비 difference-first, 경험 UQ 원리적 불가; §5 E-52 판별 = ARGUED, §8서 재측정 아님).

## ★정련 헤드라인 (제안)
> "센서 사양에서 도출되는 (Cramér–Rao) 운용성 envelope은 phase-resolved 예보 skill의 시공간 경계를
> **추정기 자신의 경험적 UQ만큼** 예측한다 — 단 **어떤 추정기도 돌리지 않고 설계시점에**, 그리고
> **정보한계 vs 효율갭을 판별하는** 능력과 함께. 그 예측력은 **관측 under-determination(kp/개구비)에
> 비례**해 커지며 — 운용성이 실제 위험한 레짐서 강하다(ρ 0.52→0.78, p=3e-12)."

- **은퇴시킬 주장**: "CRB가 경험 UQ보다 skill 예측 잘함"(거짓, 전 레짐 동률). "§8 clean PASS".
- **retreat 여부(NAOE 판단)**: 헤드라인이 회계/진단으로 후퇴하지 **않음** — "추정기-독립 설계시점 운용성
  envelope, 운용성 위험 레짐서 강함"은 **방법 헤드라인**(단순 진단 아님). NAOE 얼굴(운용성 지도) 유지.
  단 "예측 우위" 없이 "estimator-독립+판별+under-det 스케일링"으로 서므로, fable 최대리스크는 **일부
  현실화됐으나 재프레임으로 생존** — 봉인이 아니라 재특정. 지도교수 합의 필요(difference-first 축).

## 캐비엇 (대칭감사·스코프)
- **kp 축 교란**: under-determination이 comp수·kp·해상(BF3 vs BF4: 2/5comp=BF3, 15/30=BF4)과 공변.
  kp는 물리 요약(개구/파장)이고 partial|dist도 함께 상승(0.15→0.63)이라 순수 거리/family 교란 아니나,
  세 요인 완전 분리는 이 데이터로 불가. **MEASURED = kp와의 단조 관계; ARGUED = 인과가 개구비.**
- 모사 센서 연산자·Gaussian-잡음 obs·nowcast/단리드. 이동-센서 실측 미검(조건③ 별건).
- emp-UQ baseline = ridge 사후분산(Zhao VB-NN의 **대리**, ARGUED). 판별(정보vs효율) = §5 E-52 전이(ARGUED).
- **인용 금지**: "§8 PASS", "CRB>emp-UQ 예측", "under-det 스케일링이 인과적으로 개구비", "봉인됨".
- **인용 가능**: +0.33 estimator-비의존(감사 3중), envelope↔skill under-det 스케일(kp ρ−0.874 p3e-12),
  emp-UQ 전레짐 동률, well-det 천장=flat-map(CoD 0.47 vs 0.88).

## 부록 — 조건④ (§7 random-phase Gaussianity) MEASURED (2026-07-18)
CRB(E-51/52)는 ensemble-Gaussian 근사, 진짜 필드=결정진폭·random-phase(비-Gaussian). §8 레짐서 필드
Gaussianity 경험 확인(R=300 위상실현/에피소드, nq=40): Shapiro-W med **5comp 0.991 / 15comp 0.995 /
30comp 0.995**(CLT로 comp수↑서 개선), 초과첨도 med −0.58→−0.33→−0.19(약 sub-Gaussian, random-phase
정현파 합 예상거동). **★envelope이 강한 under-det 레짐(15/30comp)이 가장 Gaussian ⟹ 감사 강-레짐 결과가
CRB 근사 가장 탄탄.** 추가로 **§8은 Spearman(rank) 판정이라 절대-CRB의 Gaussian-근사 오차는 ranking 무영향.**
⟹ random-phase 캐비엇 ARGUED→MEASURED, 결론 불변(THESIS_FRAMING §7 예상대로). 인용가능=필드 near-Gaussian·
강-레짐 최-Gaussian·rank-robust. 인용금지="exact random-phase CRB 불필요 증명"(주변분포 확인이지 보정크기
정량보장 아님 — THESIS_FRAMING §7 self-critique 유지).
