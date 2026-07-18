# E-58 사전등록 — envelope skill 검증 (§8 KEYSTONE, CONDITIONAL→PASS 관건) 2026-07-18

> fable 박사심사 조건①(단일 최대 리스크). 질문: **CRB-앵커 운용성 envelope이 실제 예보 skill의 시공간
> 경계를 (경험 UQ보다 잘/대등하게) 예측하는가?** 통과 시 헤드라인 봉인(CONDITIONAL→PASS). 규율 전면 적용
> ([[feedback_direction_flip_discipline]]): 사전등록·검정력 게이트·MEASURED/ARGUED·대칭감사.

## 기존 자산 (재사용 — 바퀴 재발명 금지)
`scripts/d1_crb.py`: `fisher`(FIM), `jacobian`, `pointwise_field_crb`(σ_CRB(x,t*)/Hs = CRB 국소 하한),
`load_ep`(에피소드), `collect_obs`(obs [0,t_now]), `predict`(참 장), `field_nrmse_crb`. **CRB envelope = 존재.**
`scripts/d1_predictable_zone.py`: `pointwise_field_crb`(σ_CRB 맵), `cg_cone`(기하 baseline).
⟹ **§8이 추가할 것 = (A) 실제 skill 맵 σ_actual (B) 경험-UQ baseline σ_emp (C) 검증 통계.**

## 설계 (frozen)
- **아레나**: wavesim LiDAR 에피소드 `data/episode_BF3_*·BF4_*.h5`(이동센서·희소·시공간 = envelope에 맞음).
  에피소드 n≥12(BF3/BF4 mix), t_now = 레코드 중간.
- **질의점 q=(xq,yq,t_now+lead)**: 공간격자(gx,gy) × lead {0,2,4,6 s}. 에피소드당 수백 q.
- **σ_CRB(q)** [기존]: FIM(obs [0,t_now]) → Finv; `jacobian(θ,q)` g; σ_CRB=√(gᵀFinv g)/Hs. **추정기-독립 하한.**
- **σ_actual(q)** [신규, 실제 skill]: 배포형 추정기 = k-그리드 ridge-LS 스펙트럴 예보(참 파수 미사용).
  obs(x,y,t,z)에 Φ_obs c≈z 적합(ridge λ) → 예보 field(q)=Φ_q c. err(q)=|field(q)−predict(θ,q)|/Hs.
  **다중 잡음실현 평균으로 안정화**(단일실현 잡음 제거).
- **σ_emp(q)** [신규, 경험 UQ baseline = Zhao 계열 대리]: ridge-LS 사후분산 σ_emp(q)=√(Φ_q Cov(c) Φ_qᵀ)/Hs,
  Cov(c)=σ²(ΦᵀΦ+λI)⁻¹. **추정기-종속.**
- **기하 baseline**: footprint 거리 / c_g-cone 포함여부(순수 기하 예측자).

## 검증 통계 + 판정 (frozen)
에피소드별 q 전체에서:
- **Spearman(σ_CRB, σ_actual)** ← 주 판정 (CRB가 실제 skill 경계를 예측하나)
- Spearman(σ_emp, σ_actual) (경험 UQ가 예측하나) · Spearman(거리, σ_actual) (기하 confound)
- **AUC**: σ_CRB<τ 가 σ_actual<τ' (고-skill 영역) 분류. 에피소드별 → median±분포.

**판정 규칙 (frozen, 규율1 SE 게이트):**
1. **PASS(envelope 유효)**: median Spearman(σ_CRB,σ_actual) **≥ 0.5** (에피소드 하한 CI>0) AND CRB가 **거리
   baseline 초과**(부분상관 — 순수 기하 confound 아님) AND CRB가 경험 UQ와 **대등 이상**(ρ_CRB ≥ ρ_emp − 0.1).
   ⟹ headline 봉인, CONDITIONAL→PASS 조건① 통과.
2. **FAIL(envelope 무효)**: median Spearman < 0.3 또는 거리 baseline이 CRB 흡수(부분상관 붕괴) ⟹ headline이
   회계/진단으로 후퇴(=NAOE 부적합) ⟹ **방향 재고**(fable: 이게 단일 최대 리스크).
3. **PARTIAL**: 0.3~0.5 또는 경험 UQ가 CRB 크게 초과 ⟹ CI 그대로 보고, 라벨 강제 금지(규율6).

## 대칭 감사 (사전 명기 — 규율4)
- **과대 위험**: σ_CRB·σ_actual 둘 다 footprint 거리에 구동되면 Spearman이 **기하 confound로 부풀려짐**
  → 거리 부분상관 필수. σ_actual이 오라클-파수면 σ_CRB와 **동어반복** → **k-그리드(비오라클) 필수.**
- **과소 위험**: ridge-LS 추정기가 CRB보다 훨씬 비효율이면 σ_actual이 σ_CRB를 안 따를 수 — 그건 envelope
  무효가 아니라 **추정기 미달**(경험 UQ도 같은 추정기라 대등 비교는 유효). estimator 품질 caveat 병기.
- **경험 UQ가 이기면**: 그것도 정직 보고 — CRB의 가치는 "예측력"이 아니라 "추정기-독립·설계역산·정보/효율
  판별"이라는 **다른 축**임을 명시(단 예측력서도 대등해야 headline 방어).

## 검정력·스코프
- 에피소드 n≥12, 에피소드별 Spearman의 median + wild-bootstrap CI. SE 게이트(규율1): median CI 하한이 판정
  경계(0.5/0.3)를 명확히 가를 것.
- 스코프: 모사 센서 연산자. 이동-센서 실측 미검(조건③ 별건). Gaussian 잡음 CRB.

## 구현 순서 (다음)
1. `tmp/e58_envelope_validation.py`: load_ep+CRB(재사용) + ridge-LS 예보 추정기 + σ_emp + Spearman/AUC/부분상관.
2. smoke(에피소드 2·q 소수) → 코드검증(σ_CRB grade0 sanity·추정기 오라클극한 체크).
3. **실행 前 독립감사**(팔 공정성·k-그리드 비오라클·거리 confound) → n≥12 실행 → 재도출+감사(3중검증).
