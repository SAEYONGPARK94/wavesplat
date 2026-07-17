# E-46 — HOS 비선형 레그: 살아남은 헤드라인의 HOS 재현 (상시지시 빚 청산)

**Status: RUNNING — 결과 섹션은 본 런 완료 후 채워짐 (이 파일은 사전등록+설계 부분이 먼저 동결된 초안)**

## 0. 왜 이 실험인가

상시지시: **"검증 데이터셋엔 항상 HOS 비선형 포함, 헤드라인은 HOS서 재현"**.
2-에이전트 감사 결과, E-29~E-45 아크 전체가 이 지시를 위반 — 살아남은 헤드라인이
비선형성 아래서 단 한 번도 재현된 적 없음. E-46이 그 빚을 갚는다.

살아남은 헤드라인 (post-audit, 전부 **선형** JONSWAP 중첩 아레나):

| 항목 | 선형 아레나 값 |
|---|---|
| prize `oracle − cur` (field) | **+0.0729 ± 0.0034** (6 blocks, Q p=0.24) |
| 배포 functional 전이 `enc10 orc − cur` | **+0.0431 ± 0.0077** (n=957, Wilcoxon 3.6e-08, ratio 0.60) |
| blind 추론 | ≈ 0 (사전등록 재현, CI가 +0.03 bar 배제) |
| **★buoy-grade 외부공급** (d±5°, Tp±5%, r2±0.05) | **+0.0349 CI[+0.0084,+0.0615]** |
| WW3/forecast-grade 공급 | **DEAD** (frozen-axis −0.0092 CI∋0; phantom −0.0158 vs −0.01 bar) |
| e45 phantom 참조 (선형, n=240) | fld ph_buoy−cur **−0.0074** CI[−0.0194,+0.0045] PASS / fld ph_ww3−cur **−0.0208** CI[−0.0338,−0.0078] 유의한 HARM |

핵심 질문: **buoy-grade 헤드라인이 HOS 비선형성에서 생존하는가, 아니면
선형-중첩 아티팩트인가?** null도 완전히 정당한 결과다 — 그 경우 헤드라인은
선형 바다로 스코프 축소.

## 1. 무엇을 지었나

`tmp/e46_hos_repro.py` — `build_sea_crosstp`의 정직한 HOS 아날로그.

- **아레나 X_H (HOS cross-Tp)**: 파라미터/트랙/노이즈 draw는 `build_sea_crosstp`와
  **동일 rng 스트림(9700+31·si), 동일 호출 순서** (Tp1=7, Hs~U(1.5,3),
  r2~U(0.2,0.5), d2=d1±U(60,150)°, Tp2=7·U(0.75,1.4), spread 30°,
  U~U(0,1.5)·cp, nfrac~U(0.02,0.09), N_OBS=400/70 s, 개구 side=2λ₁,
  GT=1200 held-out 픽셀 @t=70 s). 표면만 교체: **M=3 HOS 진화**
  (레포 `wavesim/hos_simulator.py` 그대로; dealiasing ON, factor
  ceil((M+1)/2)=2 — 문서화된 "M=1 dealiasing-off NaN 폭발" 경로는 사용 안 함).
- **HOS 수치**: 주기 도메인 8λ₁×8λ₁ (611 m), 64×64 스펙트럼 모드
  (dk=kp/8, kmax=4kp, 피크 링에서 각분해능 7.2°), depth 50 m, RK4 dt=0.15 s
  (ω_max·dt=0.27 — F1/E-16 전례 DT=0.10과 동일 마진), spin-up 3Tp=21 s.
- **IC**: `initialize_jonswap` 2회(파티션별 Hs√(1−r2)@Tp1,d1 seed 4000+si /
  Hs√r2@Tp2,d2 seed 5000+si), eta_hat·phi_s_hat 합산(선형 IC — 구속 고조파는
  spin-up 동안 자연 발생).
- **관측**: 같은 `sample_obs` 트랙의 시공간 점에서 sim을 시간전진시키며 η_HOS
  점샘플 + 동일 σ₀ 노이즈 벡터. kp는 `two_stage_kp`로 데이터에서 blind 추정.
- **★선형 쌍둥이 X_L**: **같은 IC 모드**를 정확히 선형 진화
  (eta_hat(t)=eta_hat(0)·e^{−iωt} — 같은 이산화의 선형 극한), **같은 노이즈
  벡터** → per-sea, per-draw 페어링된 물리 컨트롤.
  - **하니스 게이트를 쌍둥이에 건다**: twin fld orc−cur ∈ [0.04, 0.11]
    (e44 anchor band). 이렇게 해야 "HOS에서 상금이 죽음"이 버그와 구별되는
    **물리 결과**로 관측 가능.
  - **비선형성 검사도 대비(contrast)로**: 같은 위상·같은 스펙트럼의 쌍둥이
    대비 skewness — 단언이 아니라 알려진-음성 컨트롤 내장 (코디네이터의
    스크린 FPR 경고 반영; support-matched: 두 field 모두 같은 611 m 전체
    도메인 그리드에서 계산).
- **ARMS** (솔버 동일: `solve_with_grid`, SMULT=3, free arm, GT peek 없음;
  θ-draw는 H/L 두 세계에서 재사용 = draw-paired):
  - `cur` incumbent net (10^net_gamma_map)
  - `m2` honest all-marginal — **MANDATORY** (cur≈m2인 아크이므로 m2를 못
    이기면 아무것도 회수 못 함)
  - `orc` 참 θ 공급 (`supplied_grid`)
  - `buoy` buoy-grade θ (d±5°, Tp2±5%, r2±0.05), R=5 draws/sea, 페어링 전
    per-sea 평균 (E-14 감사 규칙)
  - unimodal 블록 (e45 arms 그대로, R=3): `uni_orc`, `uni_buoy`, `ph_buoy`,
    `ph_ww3` — **존재하지 않는 2번 lobe 공급**(phantom do-no-harm)
- **FUNCTIONALS**: field corr @Pq (primary/anchor) + encounter corr
  (`track_future` 외삽, enc10=0–10 s / enc05=0–5 s, 40점)

## 2. 사전등록 (스코어링 전 동결 — 스크립트 docstring과 동일)

- **시드**: 신선 블록 crossTp si=5000..5191 (n=192), unimodal si=6000..6063
  (n=64). 소각 블록(0..31, 100..163, 200..263, 300..347, 400..447, 900..915,
  1000..1959, unimodal 3000+/3200+/3300+/3400+/3500+/4000+) 회피. smoke
  si=4990..4993/6990..6991은 격리(크래시/NaN/타이밍만; 수치 튜닝 금지).
- **급준도**: 아레나 자체 분포 kp·Hs/4 ∈ [0.031, 0.062] (λ₁=76.4 m에서
  Hs 1.5–3 m). M=3 안정 + 측정가능한 skewness(2차 이론 ~3kpσ ≈ 0.09–0.19).
- **MDE (t-검정, α=.05, 검정력 80%, n=192)**:
  - field buoy−cur: 가정 per-sea paired sd 0.10 (e43 선형 이력) → **MDE ±0.020**;
    sd 1.5× 팽창(0.15)에도 MDE 0.030 < 목표 +0.035. → **주장 가능 레벨**.
  - encounter orc−cur: sd 0.238 (e44) → MDE 0.048 vs 기대 +0.043 →
    **SECONDARY** (~74% 검정력). buoy-enc 기대 ~+0.02 → **PILOT** (CI만 보고).
  - unimodal n=64, R=3: fld phantom CI 반폭 ~±0.020 — mean bar(−0.01)용으로
    적정, 그렇게 공시.
- **게이트/바**:
  - 비선형성: mean skew(HOS@T) ≥ +0.05 **그리고** 쌍둥이 대비 ≥ +0.04.
    실패 → "위장 선형 런", 판정 없음. 제외(NaN) > 5% → 하니스 문제, 판정 없음.
  - HARNESS GATE: twin fld orc−cur mean ∈ [0.04, 0.11]. 실패 → 아무것도 안 읽음.
  - **ANCHOR-H**: HOS fld orc−cur ≥ +0.03, t-CI 0 배제 → 상금 생존.
    twin 게이트 통과 + ANCHOR-H CI 상한 < +0.02 → **상금 자체가 선형 아티팩트**.
  - **PASS**: fld buoy−cur ≥ +0.03 & t-CI 0 배제 & fld buoy−m2 t-CI 0 배제
    & phantom do-no-harm(ph_buoy−cur mean ≥ −0.01, fld와 enc10 모두).
  - **KILL**: fld buoy−cur 95% t-CI 상한 < +0.02 → 헤드라인 비선형 생존 실패
    → 선형 바다로 스코프.
  - 그 사이 → INCONCLUSIVE + 달성 MDE 보고.
- **통계**: t-기반 CI만 (`stats.t.interval`; **1.96 금지** — e41:356 버그).
  결정론적 정수 시드만 (**Python hash() 금지** — E-28-B 실격 사유). R-draw는
  페어링 전 per-sea 평균. enc10 orc−cur에 Wilcoxon 병기. 탐색적 판독(라벨
  명시): draw-paired dd = (arm−cur)_H − (arm−cur)_L.

## 3. 메시 적정성에 관하여

`mesh_adequacy.py`는 **raycast 삼각 메시**의 검사기다. 이 아레나의 관측 연산자는
점-샘플링(스펙트럼 표면 직접 평가)이라 raycast 메시가 존재하지 않음 — 해당 검사는
**적용 불가**이고, 그 대신 HOS 스펙트럼-그리드 적정성을 검사한다:
- kmax = 4kp (자유파 대역 k2 ≤ 1.78kp + spread 커버; 2차 구속 고조파 ~2k도
  대부분 커버, 그 너머는 절단 — dealiasing이 folding은 차단)
- 스펙트럼 꼬리 pile-up 진단: k > 0.85·kmax 에너지 비율, t=0 vs t=T
- Hs drift |Hs(T)−Hs_target|/Hs_target (에너지 보존)
- NaN/비유한 상태 제외 카운트

## 4. 결과 (본 런 완료 후 기입)

TBD

## 5. 판정 (본 런 완료 후 기입)

TBD
