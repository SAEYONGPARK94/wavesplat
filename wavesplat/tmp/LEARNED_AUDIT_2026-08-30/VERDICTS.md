# 학습 경로 판정 전수 분류 — 2026-08-30

> **사용자 결정(오늘)**: *"Bank mmse 방식말고 학습에 대해서만 쭉 검토해줘 연구는 학습위주로 결과를 낼거야"*
> ⟹ 학위 주제 = **WavePrior = 학습된 스펙트럼 사전**. `bank_mmse` 는 **기여가 아니라 앵커·비교자**.
>
> **런 0회**(동결 결과 파일·소스만 읽음) · **git commit 0회** · **패키지 수정 0줄**.

---

## 0. 열람 범위 (= 주장 범위)

- **`tmp/*/verdict*.json` 90개**의 라벨을 **전수 추출**했다(§2 표의 뼈대).
- **팔 정의(`ARMS`)를 소스에서 직접 연 배치 = 58개**:
  `B-1`·`B-2`(+`b2c`~`b2f`)·`B-3`(+`b3b`·`b3c`·`b4`)·`B-5`·`B-6`(+`b6b`·`b6c`·`b6d`)·`B-7`(+`b7b`)·
  `B-8`(+`b8b`)·`B-9`(+`b9b`)·`B-10`·`B-11`·`B-11b`·`B-12`·`B-13`·`B-14`·`B-14b`·`B-15`·`B-16`·`B-17`·
  `B-18`·`B-19`·`B-20`·`B-20b`·`B-21`·`B-22`·`B-25`·`B-26`·`B_DEPLOY`·`C_ENCOUNTER`·
  `W-3`~`W-17`(15개)·`S-13`·`S-13b`·`S-16`·`S-17`·`HOSGATE`·`X-1`·`X-2`·
  `M-1`~`M-10`(11개, `M5`/`M5b` 별도)·`T-2`·`T-3`·`T-4`·`prereg_heldout_2026-08-01`
- **라우팅 소스**: `wavesplat/optimizer/sequential_update.py:415-450,670-770,1112,1175-1188` ·
  `wavesplat/run.py:38-90` · `wavesplat/optimizer/learned_init.py:1-30`
- **논문**: `thesis/thesis_main.tex`(구조·기여 목록 1158–1930·§sec:init-bank 3418–3470·§sec:ladder
  4053–4160·결론 8953–8990·제5장 8187–8215) · `paper/wavesplat.tex`(배너·초록·절 목록·
  Proposed approach·Conclusion) · `paper/encounter_forecast/`
- **금지목록** `memory/reference_citation_bans.md` 머리말 + 08-26~08-30 묶음 전문
- **현황판** `tmp/STATUS_2026-08-28/STATUS.md` §0·§1·§11.4

⚠**연 적 없는 것**: 2026-07 이전 레거시 배치(`rerun_*`·`confirm*`·`heldout*`·`unified_*`·
`blind_auto_real` 등) 다수 · `results/` 96개 JSON · `scripts/` 413개.

---

## 1. 분류의 법칙 — **산문이 아니라 팔 정의**

`sequential_update.py:_initialize_inner`(**:684-757**)가 전부를 결정한다:

| 입력 | `init_method` | `real_domain_gate` | 실제로 도는 것 |
|---|---|---|---|
| **합성**(GT 필드 있음) | `'auto'` | (무관) | 라우터 → 임계에 따라 `learned` **또는** `v2_konly` |
| **실측**(GT 필드 없음) | `'auto'` | `True`(기본) | **라우터 우회 → `bank_mmse`**(`:749-752`) |
| 실측 | `'auto'` | `False` | 라우터 → `learned` |
| 아무거나 | `'learned'` 명시 | (무관) | `learned`, **게이트 분기 미진입** |
| 아무거나 | `'v2_konly'` 명시 | (무관) | 해석적 빔포밍+코스투파인 (**학습도 bank 도 아님**) |

★**이름 충돌 3종** — `SHIP`/`DEPLOYED`/`{}` 가 배치마다 다른 것을 가리킨다:

| 실체 | 대표 배치 |
|---|---|
| **LEARNED**(합성 + `auto`, 임계 0) | `B-1`·`B-11`~`B-12`·`B-15`·`B-16`·`B-19`~`B-26`·`M-1`·`M-3`·`M-3b`·`M-10`·`W-15`·`W-16` |
| **BANK**(실측 + `auto` + 게이트 on) | `B-3`·`B-6`·`B-9`·`M-5`/`M-5b`의 `SHIP`·`M-6`·`M-7`의 `BANK`·`M-8`/`M-8b`의 `BANK*`·**`T2-STAT`의 `SHIP-*`**·`T-3`의 `SHIP-*`·`W-12`의 `DEPLOYED`·`prereg_heldout` |
| **LEARNED 강제**(실측 + `real_domain_gate=False`) | `B-2`계열·`B-7`·`B-10`·`B-14`/`B-14b`·`B-17`·`B-18`·**`M-4` 전층**·`M-5b`의 `PROP`·`M-7 E1`·`M-8`의 `PROP*`·`T2-STAT`의 `PROP-*` |

★**bank 전용 노브**(하나라도 세팅하면 무조건 bank): `bank_refit_every_s` · `bank_refit_async` ·
`bank_dk_frac` · `bank_k_set` · `real_bank_ensemble_k` (`sequential_update.py:432-440` 강제 검사).

★★**`B-20`의 "REFIT" 은 bank 재적합이 아니다** — `_snap_k_to_grid`(`sequential_update.py:939-979`)
안의 **`(w,φ)` LS 재적합**이다(`gates_b20.py:8-12` 축자). `bank_refit_*` 키가 그 배치에 **0회** 등장한다.

---

## 2. 분류표

### 2.1 (가) 학습 경로 판정 — 합성

| 배치 | 라벨 | 근거 팔 정의 (파일:줄) | 라우팅 증거 |
|---|---|---|---|
| `B-1`(=`W16` 자료) | `SHIP-BEATS-BEST-CROWDED` | `'SHIP': {}` — `gates_w16.py:17` | `auto_selected='learned'` **384/384** |
| `B-11`/`B-11b` | `HEADLINE-SURVIVES-NONLINEAR-FRESH` / `SPARSE-ADVANTAGE-SURVIVES` | `dict(DEFAULT_CONFIG)+seed` — `run_b11.py:33` | `auto_switch_thresh=0` 전행 |
| `B-12` | `HEADLINE-SURVIVES (CONFIRMATORY, n=48)` | `run_b12.py:34` 동일 | 〃 |
| `B-15` | `HEADLINE-SURVIVES-CORRECT-SENSOR` | `run_b15.py:55-56` | 〃 |
| `B-16` | `CONTINUOUS-K-EARNS` + `DOSE-RESPONSE` | `ARMS={'SHIP-NOW':{'k_snap_dk':None},'KGRID1':…,'KGRID2':…}` — `gates_b16.py:50-53` | 〃 |
| `B-19` | `CONTINUOUS-K-EARNS-SPARSE` | `gates_b19.py:114-117` | ★`G-6`: `auto_selected['learned']`·`ensemble_k[1]`·`auto_real_gate≠False 0/144` |
| `B-20` | `REFIT-IS-NOT-THE-CARRIER` + `QUANTIZATION-EARNS-INDEPENDENTLY` | `'NOOP-REFIT':{'k_snap_dk':1e-10}` — `gates_b20.py:147-150` | `G-6` 144/144 |
| `B-20b` | `QUANTIZATION-EARNS-INDEPENDENTLY-CROWDED` | `'FITSWAP':{'k_snap_dk':1e-12}` — `gates_b21.py:179-183` | `H-10` |
| `B-21` | `MEDIATION-UNDECIDED` ×2 · `CONFOUND-NEGLIGIBLE` ×2 | `gates_b21.py:180-186` | `G-6` 768/768 |
| **`B-22`** | ★**`PRIOR-EARNS`** | 8팔, `gates_b22.py:147-163`; `ANA-SHIP` 만 `auto_switch_thresh=999999`(=`v2_konly`) | 7팔 learned |
| `B-25` | `GT-WEIGHT-HURTS` | `gates_b25.py:90-98` (`use_confidence`·`conf_mode`) | `G-5c` |
| `B-26` | `SEASTATE :: SS-REAL = HOLDS` | `run_b12.run_ship` import — `gates_b26.py:83-84` | `G-15` 4코너 전부 learned |
| `B_DEPLOY` | `DEPLOY-CONFIRMED` | `'A-SHIP': delta={}` — `run_b.py:52` | ⚠**혼합** learned 313 / v2_konly 71 (임계 8) |
| `C_ENCOUNTER` | `GEOMETRY-SURVIVES` | `'SHIP':{'init_method':'auto'}` — `run_c.py:32-34` | `('SHIP','learned') 192/192` |
| `W-4` | `REMEDY-CONFIRMED / OPTIMIZER-EARNS` | `arms_config.py:61` `'init_method':'learned'` 명시 | 1,216행 전부 learned |
| `W-5` | `JOINTFIT-REDUNDANT` | `arms_config_w5.py:42-60` | 640행 전부 learned |
| `W-8` | `PORT-OK-EFFECT-NULL` | `'PKG-VP': {}` — `gates_w8.py:6` | ⚠**혼합** 15/1 |
| `W-11` | `ROUTER-UNNECESSARY` | `'ROUTED':{}` vs `'FORCE-LEARNED':{'init_method':'learned'}` — `gates_w11.py:16-20` | ⚠`ROUTED` = learned 61 / v2_konly **71** |
| `W-13` | `SPARSE-NULL` | `'LEARNED':{'init_method':'learned'}` vs `'ANALYTIC':{'init_method':'v2_konly'}` — `gates_w13.py:15-18` | 명시, 라우터 미작동 384/384 |
| `W-14` | `LEARNED-WINS-CROWDED` | `gates_w14.py:15-18` (W13 과 동일) | 〃 |
| `W-15` | `UNIFY-REGRESSES` | `'POST':{}` · `'PRE':{'auto_switch_thresh':8}` · `'ALL-ANALYTIC':{…999999}` — `gates_w15.py:17-21` | `G-1` POST **384/384 learned** |
| `W-16` | `HEADLINE-HOLDS-BOTH` | `'SHIP': {}` — `gates_w16.py:17` | 384/384 learned |
| `S-13` | (JSON 라벨 없음) `PASS`·`PASS` | `'SHIP': {}` — `run_s13.py:59` | ⚠**혼합** learned 112 / v2_konly **16** |
| `S-13b` | (라벨 없음) `PASS` | `cfg['seed']=est_seed` 만 다름 — `run_s13b.py:63-66` | ⚠**혼합** 120/8 · 100/**28**, `route_unstable=31` |
| `HOSGATE` | `ROBUST` | `dict(DEFAULT_CONFIG)` — `run_hosgate.py:95-97` | ⚠**혼합** learned 50 / v2_konly **14** |
| `M-1` | `HOS=HOLDS · LIN=HOLDS` | `run_b12.run_ship` — `run_m1.py:84` | ⚠이름의 "REAL" 은 **해상상태 코너**(합성) |
| `M-3` | `DIR180=BASELINE-NO-SKILL/SHIP-EARNS` · `SPR5=UNDECIDED` · ⛔`DEP12=UNMEASURABLE` | `run_m3.py:89` | `cap_auto_selected='learned'` |
| `M-3b` | `C1·C2·C3 = HOLDS` | `gates_m3b.py:94-131` (CKPT50-*) | 되읽기 |
| `M-10` | `NEARFIELD-HOLDS` | `run_m10.py:322` | `G-3` `obj_init_method=='auto'` 강제 |
| `_audit/verdict_t2_recheck` | (재도출) | `B-16`·`W15/W16`·`S13` 재집계 | — |

### 2.2 (가) 학습 경로 판정 — **실측**(게이트를 명시 우회한 것)

| 배치 | 라벨 | 팔 정의 (파일:줄) |
|---|---|---|
| `B-2` | `REAL-UNDECIDED` | `'PROP':{'real_domain_gate':False,'init_method':'learned'}` — `gates_b2.py:25` |
| `B-2c`/`B-2d`/`B-2e`/`B-2f` | `PARTIAL` ×3 + 라벨없음 | `cfg.update(G.ARMS['PROP'])` — `run_b2c.py:46` 외 |
| `B-7`/`B-7b` | `PROP-NO-REAL-SKILL` / `UNDECIDED` | `'PROP-TRACK':{…False,'learned'}` — `gates_b7.py:45` |
| `B-8`/`B-8b` | `SPLIT-UNDECIDED` / `SPLIT-STILL-UNDECIDED` | `PROP-TRACK`(learned) vs `V2K-TRACK`(`v2_konly`) — `gates_b8b.py:42-43` |
| `B-10` | `THREE-INITS-NO-SKILL` | `V2`·`V2K`·`PROP` 세 초기화 전부 게이트 우회 — `gates_b10.py:40` |
| `B-14`/`B-14b` | `OBS-MODEL-DOES-NOT-MEDIATE` (+`REPRESENTATION-ADEQUATE-NOMINAL`) | `PROP_CFG` — **`gates_b14.py:35`**(`# ⛔빈 dict 로 팔을 정의하지 않는다`) |
| `B-17` | `STRATUM-SURVIVES` (+ 전 팔 `REALNOISE-NULL`) | `ESTIMATOR_CFG` — `gates_b17.py:191-197`; `INERT_KEYS` 로 자기 공시(`:223-231`) |
| `B-18` | `NEITHER-SUFFICIENT` | `ARM_PROP = dict(PROP_CFG, **SHIPPED_2026_08_22)` — `gates_b18.py:72-74` |
| `W-12` | **`REAL-UNIFY-LOSES-SKILL`** | `'FORCE-LEARNED':{'real_domain_gate':False,'init_method':'learned'}` — `gates_w12.py:18` |
| `M-4` | `CAPACITY-HURTS` **[EXPLORATORY]** | ★`cfg['init_method']='learned'; cfg['real_domain_gate']=False` — **`gates_m4.py:157-158`** (실측층 포함 **bank 팔 0개**) |
| `M-5b` | `WINDOW-LONGER-IS-BETTER` | `'PROP':{…False,'learned'}` — `gates_m5b.py:206`; ★**`PRIMARY_ARM='PROP'`**(`:209`) |
| `M-7`(E1) | `NEITHER-MOVES-REAL · DATA=EXPLORATORY` | `run_m7.py:219-220`(5칸+`SHIPPED`) |
| `M-7`(E2/E3) | `DISTRIBUTION-SHIFT-SURVIVES` / `E3=REPORTED` | 합성 `run_b12.run_ship` + ckpt 주입 |
| `M-8`의 `PROP*` 5팔 | ⛔판정 없음(영구) | `_LEARN = dict(init_method='learned', real_domain_gate=False)` — `gates_m8.py:132` |
| `M-8b` | `NO-SINGLE-LINK-DETECTED (n=12, q80=0.931)` | `M-8` 상속(자체 `ARMS` 0건) |
| **`T2-STAT`**의 `PROP-*` | **`T2-STAT :: PROP-SKILL-UNDECIDED`** | `_PROP = dict(real_domain_gate=False, init_method='learned', …)` — `gates_t2.py:71`; ★**판정 주어가 `PROP`** |

### 2.3 (나) bank 경로 판정 — ⟹ **연구 서사에서 내려야 할 것**

| 배치 | 라벨 | 팔 정의 (파일:줄) |
|---|---|---|
| **`B-3`** | **`REFIT-HELPS (CONFIRMATORY)`** | `'FROZEN':{'bank_refit_every_s':None}` — `gates_b3.py:35` |
| `B-3b` | `ANY-REFIT-HELPS` | `{'bank_refit_every_s':1/2/5/10}` — `gates_b3b.py:43` |
| `B-3c` | `B3-REPLICATES (CONFIRMATORY)` | 동일 |
| `B-4` | `PROPAGATION-EARNS` ⛔**철회됨** | `'PERSIST':{'bank_refit_every_s':None}` — `gates_b4.py:38` |
| `FINAL6` | 인용 최종형 3문장 | `ARMS=('FROZEN','REFIT2')` — `final6.py:41` |
| `B-6` | `FRAME-BUDGET-FAIL` | `gates_b6.py:1` 축자 *"`bank_refit_every_s=2` 를 배포해도 되는가"* |
| **`B-6b`** | **`RENDER-LINEAR`** (=137 µs/점) | `gates_b6b.py:16` 축자 *"`bank_mmse` 의 렌더(`_bank_predict_grid`)를 … 잰다"* |
| `B-6c`/`B-6d` | `CONTROL-FRAME-OK` ×2 | `CANDIDATE = recommended_real_refit_config()` — `gates_b6d.py:44` |
| `B-9`/`B-9b` | `ASYNC-FRAME-OK / KEEPS-GAIN` | `{'bank_refit_every_s':5.0,'bank_refit_async':True}` — `gates_b9b.py:53` |
| `M-6` | `EFFECT-ALIVE-UPTO-0.5x [EXPLORATORY]` | `ASYNC_CFG`/`FROZEN_CFG` — `gates_m6.py:43-48` (**bank-only 노브 전용**) |
| `M-9` | ⛔**결과 없음**(`mode:"pre"`) | `bank_refit_*` — `gates_m9.py:98-102` |
| `X-1` | (verdict 없음, 탐색적) | ★`fit_bank_mmse(...)` **직접 호출** — `probe.py:20,49` (**합성 위의 bank**) |
| `prereg_heldout_2026-08-01` | `CHEAP-ALTERNATIVE` | `ARM_SPEC` A·B·D·E = `est='bank'` — `run_prereg_heldout.py:49-53` |
| `M-5`(VOID) | ⛔판정 없음 | `'SHIP': dict(…, real_estimator='bank_mmse')` — `gates_m5.py:182-183` |
| `T-3` | ⛔**스모크만, 본런 없음** | `'SHIP-K3'/'SHIP-K1M3'` `PATH='bank'` — `gates_t3.py:178-182` |
| **배치 안 bank 팔**(판정은 (가)) | — | `B-7`의 `'SHIP-TRACK':{}`(`gates_b7.py:46`) · `M-5b`의 `SHIP` · `M-7`의 `BANK` · `M-8/8b`의 `BANK`~`MMSE-PROP` 7팔 · **`T2-STAT`의 `SHIP-FROZEN`/`SHIP-REFIT2`**(`gates_t2.py:74-75`, 되읽기 `init_method_readback='bank_mmse'` 72/72) · `W-12`의 `DEPLOYED` |

### 2.4 (다) 둘 다 아님 / 무관

`B-5`(`NO-VALID-COMPETITOR`, LETKF 재현) · `B-13`(`FRAMES-MATCH`, 계측) ·
`W-3`(oracle λ, ⚠부차만 혼합 `SHIP`) · `W-6`(`CONTRAST-DOES-NOT-EXIST`, **런 0회 수학**) ·
`W-7`·`W-9`·`W-10`·`W-17`(전부 `init_method_readback='oracle'`) ·
`S-16`(LETKF 5팔) · `S-17`(LETKF 사다리) · `M-2`(★**조율 대상이 LETKF `loc_radius`**) ·
`X-2`(`v2_konly`+앙상블) · `T-4`(⛔**러너 부재, 런 0회**).

---

## 3. Q1 — ★학습 경로에 대해 **인용 가능한 양성 판정**

⚠**전부 합성이다. 실측 양성은 0건**(§4).
⚠⚠**먼저 읽어야 할 단서**: 아래 §3.4 의 **라우팅 오염**과 §3.5 의 **비교자 결함**.

### 3.1 최상위 — 사전·표현 자체의 절제 (기여 ①의 담지자)

| # | 판정 | 수치 · 문턱 | ⛔동반 의무 |
|---|---|---|---|
| **A1** | `B-22` **`PRIOR-EARNS` / `L2`** | `SHIP`(학습 패키지) < `ANA-SHIP`(해석적 패키지) **45/48 · −36.34% · p=6.56e-11 · 동점 0**. 두 팔 모두 참 제로필드를 이김(`SHIP` 48/48 · `ANA-SHIP` **32/48** p=0.0147) | ★**오늘 % 를 margin 으로 쓸 수 있는 유일한 성분.** ⚠우변 기술이 **얇다**(32 = 확증 문턱과 정확히 같음, −11.93%) · ⛔`ANA`/`ANA-N` % 금지(`COLLAPSE` 수식어; 15/48·13/48) · ⛔`S2a`/`S2b` % 금지 · ⛔*"학습 사전이 해석적 파랑 추정 일반을 이긴다"* 확대 금지 · ⚠**혼잡 한정** |
| **A2** | `B-22` `LEARN-NOVP < ZERO` | **41/48 · p=3.12e-07 · −14.05%** | ★*"VARPRO 없이 사전 단독으로도 기술이 남는다"* — 해석적 원자 네 팔은 VP on/off **전부 무기술**(15·13·8·9/48) |
| **A3** | `B-16` **`CONTINUOUS-K-EARNS`** | **46/48 · −38.61% · p=4.18e-12**, 용량-반응 **47/48 · p=1.74e-13** | ⚠**크기 귀속에 `B-21` `C2` `CONFOUND-NEGLIGIBLE` 병기 필수**(TOST PASS, 밴드 안 45/48, 몫 −0.34%) · ⛔`B-20` 라벨문 축자 인용 금지 · ⛔*"기준선 격자 간격에 맞췄다"* 금지(2.5% 차) |
| **A4** | `B-19` **`CONTINUOUS-K-EARNS-SPARSE`** | **14/16 · −67.43% · p=0.00209**(가족 α=0.025 통과), 용량-반응 14/16(동점 1) | ⚠크기 귀속 = `B-20` A(4/16 p=0.9894) + B(**14/16 −67.575%**) · ⚠`N-2` `KGRID1<ZERO` 12/16 = **문턱 정확히** · ⚠mesh 3분할 공시 · ⛔외부 기준선 행 0개 |
| **A5** | `B-20b` `FITSWAP<KGRID1`(혼잡) | **48/48 · p=3.55e-15 · −38.840%**, `shift_pp` 0.233 < 밴드 2.360 | ⛔*"독립 확증"·"완전 blind"·"복제"* 금지 · ⛔`B-20` 희소 14/16 과 합산 금지(dk 1e-12 vs 1e-10) |

### 3.2 헤드라인 (외부 기준선 대비)

| # | 판정 | 수치 | ⛔동반 의무 |
|---|---|---|---|
| **B1** | `B-1` `SHIP-BEATS-BEST-CROWDED` | 혼잡 **62/64 · −38.58% · p=1.13e-16** vs `SPEC25` | ⚠**의무 병기 3종**(verdict 자신이 적음): ①합성·선형 GT ②에피소드가 체크포인트 학습 분포와 겹침 ③*"강화된"* 이 아니라 *"우리가 가진 최강"* · ⛔희소·전체합산 금지 · ⚠게이트/분석기가 **결과 뒤 작성**(`gates_b1.py:9-11` 자기고백) |
| **B2** | `B-12` `HEADLINE-SURVIVES (CONFIRMATORY, n=48)` | **43/48 · −40.19% · p=6.84e-09**; **신규 시드만 29/32 p=1.28e-06**; 음성대조 47/48 | ⛔*"더 강해졌다"* 금지 · ⛔`B-1`·`B-11` 크기와 직접 비교 금지 |
| **B3** | `B-11` (HOS M=3 · 신선 시드) | **14/16 · −41.09% · p=0.00209**, 음성대조 16/16 | — |
| **B4** | `B-11b` (희소) | **15/16 · −74.69% · p=2.59e-04** | ⚠**희소 기준선 자신이 무기술을 간신히 넘는다**(12/16 p=0.038 vs 혼잡 16/16) ⟹ **같은 무게로 읽지 마라** · ⛔희소 크기 인용 금지 |
| **B5** | `B-15` `HEADLINE-SURVIVES-CORRECT-SENSOR` | slope **17/20 · −42.61% · p=0.00129**, uniform 16/20; 음성대조 **양쪽 20/20** | 센서 교락 폐쇄 |
| **B6** | `B-26` `SS-REAL` | **24/24 · −42.42% · p=6.0e-08**; `SHIP<ZERO` **24/24 −45.46%**; 2×2 = (hold,hold,hold) | ⛔★**F1**: *"`SS-REAL` 칸이 가장 강하다"* **전면 금지**(절대 정확도는 **+16.1% 반대**) · ⚠**F2 "닫힘" 공시**(`M-2`) **+ 5% 기술 바닥 미달 공시**(`SPEC25` −3.96%) · ⚠선형 GT 단서 · ⚠**§7-2 게이트 불일치** |
| **B7** | `M-1` | LIN **22/24 −43.76% p=1.79e-05** · HOS **21/24 −33.65% p=1.39e-04**, 음성대조 24/24 | ★**선형 GT 단서가 처음 떨어지는 칸** · ⛔`B-1`/`B-11`/`B-12`/`B-22`/`B-26` 과 **크기 직접 비교 금지** · ⚠**성분 540 중 270(50.0%) 절단** ⟹ *"완전한 M=3"* 금지 · ⚠동결 드리프트 1건 공시 |
| **B8** | `M-3` `DIR180` | `SHIP<ZERO` **13/16 · −32.21% · p=0.0106** | ⛔**기준선이 무기술(11/16) ⟹ 우열 주장 금지**; `SHIP<ZERO` 로만 · ⚠반대 관찰 병기(지지집합 **안**인 `SS-LOWH` 가 최약 칸) |
| **B9** | `M-10` `NEARFIELD-HOLDS` | `obs:0_5` **45/48 · −31.69% · p=6.56e-11**; `SHIP<ZERO` 48/48 | ⛔*"관측 기하가 무관하다"* 금지 · ⚠blind 아님 · ⛔**BF5 층 2/12 가 밴드 밖** ⟹ 층 없이 쓰지 말 것 · ⚠`obs:` 는 배-거리 아님 |
| **B10** | `M-3b` | C1 **16/16 p=1.53e-05** · C2 16/16 −60.15% · C3 **15/16 −49.89% p=2.59e-04** | ⛔C3 는 **비대칭 대조**(기준선만 완전 수리) · ⛔`M-3 DEP12` 수치를 인용 가능하게 만들지 않음 |
| **B11** | `W-16` `HEADLINE-HOLDS-BOTH` | 혼잡 `SHIP<ZERO` **64/64 · −49.47% · p=5.4e-20** | ⛔**희소 −91.89% 인용 금지**(=금지된 *"희소 −93%"*) · ⛔**기준선 행 0개** — *"기술이 있는가"* 이지 *"우월한가"* 가 아니다 |

### 3.3 옵티마이저·배포 처방 (학습 팔 안)

- ✅`W-4` **`OPTIMIZER-EARNS`**: `L-VP < L-SHIP` **52/64 · −17.53% · p=2.28e-07**;
  `L-ITER0` 7/64 **+23.34%** ⟹ *"`varpro_steps=0` 하나로는 배포 전 거동이 안 된다"*.
  ★1,216행 전부 `init_method_resolved='learned'` = **경로 순수**.
- ✅`W-5` **`JOINTFIT-REDUNDANT`**: `L-VP-ONLY < L-SHIP` **58/64 · −25.28% · p=4.52e-12**;
  vs `L-VP` **36/64 p=0.382 = 널** ⟹ 결합 적합 잉여. 벽시계 29.7 s → **14.5 s**.
  ⚠`RESULT.md:48-52` 자기신고: `L-VP-ONLY` 는 **단일요인이 아니라 묶음**(성장 발화 47/192→141/192).
- ✅`B_DEPLOY` **`DEPLOY-CONFIRMED`**: 혼잡 ITT **55/60 · −25.28% · p=5.19e-12**, IUT 63/64.
  ⚠희소 **26/51 = 널** · ⚠`S-2`(셸) **미측정** · ⚠라우터 혼합(90.6% learned).
- ✅`C_ENCOUNTER` `GEOMETRY-SURVIVES`: `SHIP<ZERO` **13/16 −17.56% p=0.0106**(문턱 13, 여유 **0**).
  ★`S-1` **실패**: `V2K<ZERO` **6/16 +17.75%** ⟹ *"기하가 아니라 체크포인트가 살렸다"*.
  ⛔이동 vs 정지 **0/16 +87.83%**.

### 3.4 ⚠⚠**라우팅 오염 — 인용 전에 반드시 확인할 것**

★**2026-08-22 13:58(`d6738922`) 이전의 `{}`/`auto` 팔은 라우터가 실제로 두 갈래로 갈렸다.**
저장소가 `SHIP`/`ROUTED`/`PKG-VP` 라 부른 팔의 **실측 라우팅**:

| 배치 | 팔 | learned | v2_konly |
|---|---|---|---|
| `S-13`(08-13) | `SHIP` | 112 | **16** |
| `S-13b`(08-13) | `SHIP@43` / `SHIP@44` | 120 / 100 | **8 / 28** (`route_unstable=31`) |
| `HOSGATE`(08-13) | `SHIP` | 50 | **14** |
| `W-3`(08-16) | `SHIP` | 130 | **14** |
| `W-8`(08-22 10:16) | `PKG-VP` | 15 | **1** |
| `W-11`(08-22 12:30) | `ROUTED` | 61 | **71** |

⟹ ⛔**`S-13`(56/64·63/64) · `S-13b`(61/64) · `HOSGATE`(14/16) · `S-16`/`S-17` 이 차용한 `SHIP` 열은
"학습 경로의 성적" 이 아니라 "임계 8 라우터의 성적" 이다.**
오늘 재실행하면 128/128 이 learned 로 간다 ⟹ **다른 팔**.
★**학습 위주 서사에서 인용해야 할 것은 08-22 이후 배치**(`B-1`·`B-11`~`B-26`·`W-15`·`W-16`·`M-*`)다.

### 3.5 ⚠⚠**비교자 결함 — `W-13`/`W-14` 의 `ANALYTIC` 은 배포 해석적이 아니었다**

`make_updater` 독스트링 축자: *"`init_method='auto'` gets the median-of-k analytic ensemble
wrapper; every explicit `init_method` … gets the plain `SequentialUpdater`"*
⟹ `W-13`/`W-14` 의 `ANALYTIC` 192런은 **앙상블 없이** 돌았다 = **학습 팔에 유리하게 약화된 비교자**.
`W14/RESULT.md:3-11` 이 합산 헤드라인 `90/128`·`−38.49%`·`3.4배` 를 **철회**했다.
⚠**`W13/RESULT.md` 에는 그 배너가 없다**(`AMENDMENT_2.md:24-25` 만 지시).

★**앙상블을 포함한 옳은 비교는 `W-15 §2` 하나뿐이다**:

| 층 | `POST`(학습) vs `ALL-ANALYTIC`(배포 해석적 + 앙상블 k=3) |
|---|---|
| **혼잡** | ✅**55/64 · −44.84% · p=3.5e-09** |
| **희소** | ⚠**28/64 · +21.42% · p=0.38 = 널** |

- 제로필드 이김: `POST` 127/128 vs `ALL-ANALYTIC` **112/128**(무기술 16 vs 1).
- 벽시계 13.8 s vs **149.6 s = 10.8배**(⛔*"3.4배"* 는 약한 비교자 대비였다).
- ⟹ ⛔**`W-14` 의 `63/64 · −50.98%` 와 `W-13` 의 `+49.24%` 는 둘 다 약화된 비교자에 대한 수다.**
  ⟹ ★**정확한 문장**: *"혼잡에서 학습이 배포 해석적을 55/64(−44.84%) 이기고, **희소에서는 갈리지
  않는다(28/64, 널)**."*
- ⛔`W-15` 의 *"9셸 전부 2.2–2.5배로 평평"* 은 **층 합산 아티팩트, 인용 금지**.
  옳은 인용 = 혼잡 2.20×→1.65× 단조 감소 · **희소 9셸 전부 널이고 비가 전부 1 미만(해석적이 낫다)**.

### 3.6 방어(negative) 성격의 양성

- ✅`B-25` **`GT-WEIGHT-HURTS`**: 참 법선 유래 신뢰가중을 질량보존 절제하면 배포 팔이 **13/48**
  (+18.23% 손해), `S2`(질량/λ 채널) **21/48 널** ⟹ ★**GT 누출 의혹이 반대 방향으로 닫힌다**
  (헤드라인은 **과소평가** 방향). ⛔blind 아님 · ⛔배포 기본값 변경 금지 · ⛔보정 크기 합성 금지 ·
  ⛔기제 서술 금지 · ⛔**실측 적용 금지**(실측 `confidence ≡ 1` = 구조적 무연산).
- ✅`M-7 E2` **`DISTRIBUTION-SHIFT-SURVIVES`**(등급 `PREREGISTERED`, 주어 = `S-1`):
  학습 분포를 실측 앵커 쪽으로 의도적으로 이동시켜도 합성 헤드라인 유지 —
  `DATA` **43/48** · `BOTH` **43/48**(문턱 34), 음성대조 47/48
  ⟹ ML 심사 사유 ③(*"체크포인트=평가 분포"*)이 **약해진다**(⛔닫히지 않았다).
  ⛔`per_cell.confirmatory=true` 를 *"`DATA` 축 확증"* 으로 읽기 금지 · ⛔E2 상대 % 인용 금지 ·
  ⚠코퍼스가 동결된 것이지 **시드 축이 신선한 것이 아니다** · ⚠학습 시드 잡음만으로 |Δ| 중앙 10.21%.
- ✅`W-11` `ROUTER-UNNECESSARY`: 라우터 제거 비용 중앙 **+0.00%**(16/44, p=0.976)
  ⟹ 임계 0 통일의 근거. ⚠표적 선택이 **in-sample** 자기신고 · ★손해 >50% 9 에피소드가 **전부 희소**.

---

## 4. Q2 — ★학습 경로에 대한 **음성 / 미결** 판정

### 4.1 실측 — ★**양성 0건**

| 배치 | 팔 | 결과 | 판정 |
|---|---|---|---|
| `B-2` | `PROP` | 참 제로필드 대비 **5/12**(구 기준 7/12) | `REAL-UNDECIDED` |
| `B-2c` | `LAM-BASE`/`SCALE`/`ZERO` | **5/12 · 6/12 · 6/12** | `PARTIAL` — ★**벌점을 완전히 없애도 동일** |
| `B-2d` | `MAXN-30/10/4` | **5/12 · 5/12 · 5/12** | `PARTIAL` |
| `B-2e` | `GROW-ABS/SCALED` | **5/12 · 5/12** | `PARTIAL` |
| `B-7` | `PROP-TRACK` | lead 1–3 **12/36** · 4–14 **14/36** | **`PROP-NO-REAL-SKILL`** |
| `B-8b` | `PROP-TRACK` / `V2K-TRACK` | **21/72**(rec 2/12) / **29/72**(rec 4/12) | ★짝 비교에서 **학습 제거 쪽이 51/72 −15.05% p=2.7e-04 로 유의하게 덜 나쁘다** |
| `B-10` | `V2` / `V2K` / `PROP` | **14/36 · 14/36 · 12/36** | `THREE-INITS-NO-SKILL` ⚠문턱 18 이 널 중앙값(`AMENDMENT_R3`) |
| `B-14b` | `PROP` × 관측 모형 4종 × σ 3수준 | 최대 **8/12** | `OBS-MODEL-DOES-NOT-MEDIATE` (⛔`B-14` 축 A **무효**) |
| `B-17` | `PROP` × 잡음 모형 4팔 | 개선 최대 **7/12**(p≥0.387) | 전 팔 `REALNOISE-NULL` |
| `B-18` | `PROP` × (밀도, 잡음) 2×2 | **8 · 8 · 4 · 8 /12**(문턱 10) | `NEITHER-SUFFICIENT` |
| `W-12` | `FORCE-LEARNED` | 배포(bank) 대비 **1/5, +31.77%**; ★**lead-mean 제로필드 0/5** | **`REAL-UNIFY-LOSES-SKILL`** |
| `M-4` | 최대 용량 팔 | **0/12**(반대 12/12, p=2.44e-04) | `CAPACITY-HURTS` **[EXPLORATORY 강등]** |
| `M-5b` | `PROP` 창 단축 | **2/12**(+31.30%), 용량-반응 7→4→2 | `WINDOW-LONGER-IS-BETTER` |
| `M-7 E1` | 학습 분포·손실 2×2 | 손실 **7/12**(p=0.387) · 분포 **9/12**(p=0.073) · 교호 **6/12**, Holm 기각 **0**, MDE **11/12** | **`NEITHER-MOVES-REAL`** |
| `M-8b` | `PROP` 끝단 | bank 가 **11/12 (p=0.0032, +52.7%)** 로 이김; 6고리 Holm 생존 **∅** | `NO-SINGLE-LINK-DETECTED (n=12, q80=0.931)` |
| **`T2-STAT`** | `PROP-REFIT2` | 참 제로필드 대비 **14/36 (p=0.9338, 비 1.204)**; vs `PROP-FROZEN` **13/36 (+4.04%)** | **`PROP-SKILL-UNDECIDED`** |

★**한 문장(인용 가능)**: ***"제안 표현(학습 경로)은 실측 12 레코드에서, 시험한 어떤 팔 구성으로도
참 제로필드를 6/12 넘게 이긴 적이 없다."***
(근거 = `B-2`·`B-2c`~`B-2e` 의 `beats_zero_true` 열 · `B-7`·`B-8b`·`B-10`·`B-14b`·`B-17`·`B-18`·
`M-4`·`M-5b`·`M-7 E1`·`T2-STAT`. ⛔**사이트 4곳뿐 ⟹ 사이트 수준 일반화 금지**.)

★★**그리고 `T2-STAT` 이 마지막 변호를 좁혔다** — bank 의 유일한 예보-지평 기술 담지자(주기적 재적합)를
**바이트까지 같은 관측 창·같은 시각**으로 제안 표현에 이식했는데 **부호로 갈리지 않았다**
(같은 36셀: 배포 bank **36/36 −23.96%** vs 제안 표현 **13/36 +4.04%**; 참 제로필드 비는
동결 1.099 → 재적합 **1.204** 로 **오히려 올라간다**).
⛔단 *"음성이다"* 로 읽지 마라 — `NOT_EARN(≤12)` 에 **두 셀·한 셀 차이**로 못 닿았다 ⟹ **「검출 불가」**.
★정확한 문장 = ***"「우리 팔에 공정한 기회를 준 적이 없다」는 변호가 닫히지는 않았고 약해졌다."***

### 4.2 합성에서의 음성·미결

- ⚠**희소 층에서 학습 vs 배포 해석적은 널**(`W-15 §2`: **28/64, +21.42%, p=0.38**)
  ⟹ ⛔*"학습이 모든 층에서 이긴다"* 금지. 희소의 `SPEC25` 대비 우위(`B-11b`)는 **학습에 귀속되지 않는다**.
- `B-21` **`MEDIATION-UNDECIDED` 양 층** — ⛔*"차이 없다"·"마진이 사라진다"·"매개가 없다"* **전부 금지**
  (TOST 양 층 실패; ⚠혼잡 **18 에피소드가 +10% 초과 열세**).
- `M-8b` PRIMARY **`PRIOR-ATOMS-UNDECIDED`** 8/12(단일검정 MDE q=0.869)
  — ⛔*"학습된 원자가 규칙 격자를 이긴다/진다/차이 없다"* **전부** 금지.
- `M-3` `SPR5` `UNDECIDED`(10/16) · ⛔`DEP12` **`UNMEASURABLE`**(처치 미집행, 세 수 인용 금지).
- `M-4` — **2026-08-30 `EXPLORATORY` 강등**. ⛔기여 지위 금지 · ⛔후속 우선순위 근거 금지.
- `W-8` `PORT-OK-EFFECT-NULL`: `PKG-VP < PRE` **11/16 p=0.105 = 문턱 13 미달**.
- ⛔**분산 하드제약(기여 #2b)** = `UNDECIDED`. 허용 = *"아직 측정되지 않았다"*.
- ⛔**적응형 성분 수** = 정확도 기여로 **철회**(널) · ⛔**기하 신뢰 가중** = `DEAD` + `B-25` 로 해로운 쪽.
- ⛔**`W-17`**: *"벽의 X% 가 무엇"* 이라는 **회계 자체를 인용하지 마라**(차원 불일치 47% 철회 승계).
  살아남는 것은 부호 셋 — 스텝을 늘리면 **더 나빠지고**(1.3740, 16/16), 클램프는 **완전 무관**(0%),
  ★*"미수렴이 아니라 목적함수의 최적점이 참값이 아니다"*.

---

## 5. Q3 — ★★**bank 판정을 서사에서 내리면 무엇이 무너지는가**

### 5.1 무너지는 것의 정체

★**실측에 관한 저장소의 *모든 양성*은 bank 다.** `run.py:72-75` 가 스스로 축자로 적는다 —
*"`bank_mmse` = the research estimator **behind every published real-sea number**."*

**네 덩어리**가 주어를 잃는다:

1. **전방 시간 전개 곡선 전체** (`B-3`·`B-3b`·`B-3c`·`B-4`·`FINAL6`)
   - ⚠**질문의 전제 정정**: `B-3` `REFIT-HELPS` 는 **32/36 이 아니라 36/36**
     (`verdict_b3.json.paired.better=36`, **−23.96%**, p=1.46e-11).
     **32/36 은 `B-4` 의 `FROZEN` vs `PERSIST`**(`verdict_b4.json.frozen_vs_persist.better=32`)다.
     둘 다 bank 이므로 분류는 같으나 **서로 다른 판정의 수**다.
   - 인용 최종형 3문장(`FINAL6`)도 전부 bank: (a) lead 1–3 **11/12 −19.92% p=0.0032** ·
     (b) lead 4–14 **5/12 p=0.806** · (c) 2 s 재적합 lead 4–14 **11/12 −18.42% p=0.0032**.
   - `B-3b`: 주기 1·2·5·10 s **전부** 문턱 통과(36·36·35·35 / 36) = *"조율된 상수가 아니다"*.
2. **나우캐스트/비동기/실시간 예산** (`B-6`·`B-6b`·`B-6c`·`B-6d`·`B-9`·`B-9b`·`M-6`)
   - `B-9b` **33/36 · −10.39% · p=1.14e-07**, 비동기 최악 프레임 **2.824 ms**(동기 774 ms).
   - ★**`B-6b` 의 137 µs/점(132.13–140.07)은 `bank_mmse._bank_predict_grid` 를 잰 것이다**
     (`gates_b6b.py:16` 축자) ⟹ **실시간 예산 규칙 전체가 bank 계측**이다.
   - `M-6` 은 그 이득이 **0.5× 배속에서만 가족단위 생존**(31/36)임을 이미 깎아 놓았다.
3. **작동하는 참조 사다리** (`M-8`·`M-8b`) — 참조 끝단이 bank(`BANK<PROP` **11/12**).
   `M-8` 은 애초에 **판정 없음(영구)**이고 살아있는 것은 `M-8b` 뿐이다.
4. **판별자·맥락 문장** — `M-5b` 의 `SHIP` **11/12 −13.13%**, `M-7 E1` 의 `BANK` **10/12**,
   `T2-STAT` 의 `SHIP` **36/36 −23.96%**. 전부 bank 팔.

### 5.2 ★**"학습 경로만으로 쓴 실측 절"은 어떻게 되는가**

> ★★**전면 음성이다. 남는 양성은 0개다.**

- 남는 것: `B-2`(5/12) · `B-7`(12/36) · `B-8b`(21/72) · `B-10`(12/36) · `B-14b`(8/12) ·
  `B-17`(7/12) · `B-18`(8/12) · `W-12`(1/5) · `M-4`(0/12) · `M-5b`(2/12) · `M-7`(Holm 0) ·
  `T2-STAT`(14/36) — **전부 음성 또는 미결**.
- 남는 **양성 형태의 문장**은 **음성 자체의 품질**에 대한 것뿐이다:
  · *"개입 부재 때문이 아니다"*(추적 갱신 **36/36** 실행, 성분 **16–30개** 생성)
  · *"체크포인트 교체로는 안 된다"*(`B-8b` V2K 도 무기술; 짝 비교는 학습 제거 쪽이 **51/72 덜 나쁘다**)
  · *"원인 후보 여섯이 개입으로 배제됐다"*(잡음 형태 `B-14b`/`B-17` · 센서 모형 `B-15` ·
    관측 희소성 `B-18` · 파랑 조건 `B-26` · 창 길이/시간축 `M-5b` · 표현 용량 `M-4`[탐색])
  · ★*"관측 기하의 **피복** 성분도 배제됐다"*(`M-10`, 합성 이득이 실측이 점유하는 `obs:0_5` 안에도 있다)
  · ★*"bank 의 기술 담지자를 이식해도 옮겨오지 않는다"*(`T2-STAT`, ⚠「검출 불가」로만)
- ★**소거의 끝에는 후보 목록이 아니라 공백이 남는다**(남은 미측정 넷: 관측 기하의 **잡음 의존** ·
  정상성 가정 · 스테레오 재구성면 오차[도달 불가] · 사이트 구성[검정력 벽]).
- ⟹ ★**학습 위주 서사에서 실측 절은 「측정된 음성 + 소거표 + 한계」로만 성립한다.
  정직하고 방어 가능하지만, 양성 결과는 하나도 없다.**

### 5.3 주어를 잃는 문장 — **파일:줄**

| # | 위치 | 문장(요지) | 상태 |
|---|---|---|---|
| **T1** | `thesis/thesis_main.tex:1691-1710` | ★**기여 항목 ②** *"실측 자료에서의 전방 시간 전개 — 유효 지평의 측정"*: (a) 11/12 −19.9% (b) 5/12 (c) 11/12 | ★**기여 목록에서 통째로 빠져야 한다.** 제목이 이미 *"(격자 기반 실측 경로에 대한 것이며 제안 표현에 대한 것이 아니다)"* 라고 자백 |
| **T2** | `thesis_main.tex:1848-1886` | 기여 항목 ⑦ *"실시간 예산의 환원"* — 137 µs/점 · 1.6 ms · 720점 · 제어 8점 | ★**렌더 항이 bank 계측**(`gates_b6b.py:16`) ⟹ 절반이 주어를 잃는다 |
| **T3** | `thesis_main.tex:605` · `:975`(영문) · `:3411` · `:9164-9165` | 같은 예산 규칙의 4개 미러 | 동상 |
| **T4** | `thesis_main.tex:3607-3611` | *"배포 실측 경로(bank-MMSE)의 전방 전개는 1–3 s 에서 기술을 얻으나…"* | 라벨은 정확. **서사에서 내리면 이 항목이 사라진다** |
| **T5** | `thesis_main.tex:3626-3646` | 그림 `fig:real_leadcurve`(`figures_ws/fig10_real_leadcurve.pdf`) | ★**bank 그림 — 실측 절의 유일한 그림** |
| **T6** | `thesis_main.tex:4053-4160` (§`sec:ladder`) | *"**기술이 있는 팔이 이미 있다: 배포 실측 경로(bank-MMSE)다**"*(`:4057-4058`) · 11/12 +52.7%(`:4061`) · 끝점 행(`:4097`) · MDE(`:4113`·`:4157`) | ★**절 전체의 참조 끝단이 bank.** 내리면 *"작동하는 참조"* 프레임 자체가 없어진다 |
| **T7** | `thesis_main.tex:3927-3930` | `M-5b` 판별자 *"배포 실측 경로에는 이득이다(11/12, −13.1%)"* | bank 팔 |
| **T8** | `thesis_main.tex:1759-1762` · `:1769-1789` | 기여 ③ 안에 삽입된 `M-5b` 판별자 + `M-8b` 사다리 문단 | bank 참조 의존 |
| **T9** | `thesis_main.tex:3749` | 소거표 행 *"★같은 처치가 배포 실측 경로에는 이득 11/12 = 양성 판별 관찰"* | bank |
| **T10** | `thesis_main.tex:8972-8974` | 결론 다섯 항목 중 하나 *"실측 게이트가 사용하는 격자 기반 경로의 전방 시간 전개는…"* | ★**결론 항목 하나가 통째로 bank** |
| **T11** | `thesis_main.tex:501-503` · `:858-861`(영문) · `:1341-1346` · `:6366` · `:9103-9107` | 초록·서론·결론의 *"작동하는 참조가 제안 표현을 11/12 로 이긴다"* 미러 **5곳** | bank 참조 |
| **T12** | `thesis_main.tex:6379-6381` | *"⚠맥락: 같은 자료에서 실제 배포 실측 경로는 **10/12** 다"* | bank(`M-7 E1`) |
| **T13** | `thesis_main.tex:3418-3470` (§`sec:init-bank`) | 실측 경로 = bank-MMSE 설명 전체 | ★**남겨야 한다** — *"왜 실측 절이 음성인가"* 의 근거. 단 **기여가 아니라 배경**으로 |
| **T14** | `thesis_main.tex:8187-8215` | 제5장 수조 설계 경고(게이트 발화 → bank) | 남긴다(설계 제약) |

⚠**주의**: T1·T4·T6·T10 은 **이미 "bank" 라고 정직하게 라벨돼 있다.**
무너지는 것은 *정확성*이 아니라 **기여로서의 지위**다.

---

## 6. Q4 — 논문에서 `bank_mmse` 가 「우리 방법」처럼 읽히는 곳

### 6.1 `thesis/thesis_main.tex` — ★**라벨은 이미 정확하다. 문제는 *지위* 다**

전수 확인 결과 **bank 를 "제안 표현"이라고 잘못 적은 문장은 발견되지 않았다**
(`:3427` 축자: *"따라서 본 논문이 제안하는 격자 비의존 스펙트럼 표현은 실측 입력에 관여하지 않는다"*).
고칠 대상은 아래다 — ⛔**고치지 않았다**.

| # | 파일:줄 | 문제 | 처방(제안) |
|---|---|---|---|
| **P1** | `thesis_main.tex:1691` | ★bank 결과가 **기여 목록의 한 항목**으로 서 있다 | 기여 목록 밖으로(배경·비교자 절) |
| **P2** | `thesis_main.tex:1850-1853` | 예산 규칙의 렌더 항이 bank 계측인데 **그 사실이 그 항목에 없다** | *"질의 렌더는 실측 경로(bank-MMSE) 렌더러에서 측정"* 명시, 또는 학습 경로 렌더러로 재측정 |
| **P3** | `thesis_main.tex:605` · `:975` · `:3411` · `:9164` | 같은 누락의 4개 미러 | 동시 수정(⚠「N곳 전부」의 N 에 **주석 포함**) |
| **P4** | `thesis_main.tex:8972` | 결론 5항목 중 1항목이 bank | 항목 수 재조정(⚠과거 *"셋→넷→다섯"* 개수 불일치 반복 지적) |
| **P5** | `thesis_main.tex:4053-4058` | §`sec:ladder` 이 bank 를 *"작동하는 참조"* 로 세운다 — 학습 위주 서사에서는 **"우리 방법이 진다"** 로 읽힌다 | 절 제목·도입에 **비교자(앵커) 지위** 명시 |
| **P6** | `thesis_main.tex:3418` `\subsection{실측 입력 경로: bank-MMSE}` | **제3장(방법 장) 안**에 있다 ⟹ *"우리 파이프라인의 한 모드"* 로 읽힌다 | 제3장 밖(관련 연구/실험 설정/부록)으로 이동 검토 |
| **P7** | `thesis_main.tex:1443-1690`(기여 ①) 중 헤드라인 계보 | `S-13`/`S-13b`/`HOSGATE` 유래 수를 인용하는 곳이 있으면 **§3.4 라우팅 오염** 단서가 필요 | 인용 계보를 08-22 이후 배치로 한정 |

### 6.2 `paper/wavesplat.tex` — ★**반대 문제**: bank 0회, **학습도 0회**

- `bank`/`MMSE`/`mmse` **0회**, `WavePrior` **0회**.
- `:266-300` **Proposed approach** 와 `:635` **Two-phase estimation** 이
  *"initialization via dispersion-aware beamforming with coarse-to-fine refinement"*
  = **`v2_konly`(해석적)** 를 「우리 방법」으로 서술한다.
  ⟹ ★**2026-08-22 경로 통일 이후 배포되는 합성 경로가 아니다.**
- `:2928-2960` **Conclusion** 이 **철회된 헤드라인 수치**(65.5%, 0.115 vs 0.332, p=2.5e-16, d=1.30)를
  여전히 담고 있다(부분 철회 주석만).
- `:16-37` 에 ⛔`DO NOT SUBMIT` 배너.
- ⟹ **수정 대상**: `paper/wavesplat.tex:266` · `:635` · `:2928`. ⛔이 감사에서는 **고치지 않았다**.

### 6.3 `paper/encounter_forecast/` — ★**전체가 bank 논문이다**

- `encounter_forecast.tex:70-71` 축자: *"markdown §3 (완문): **bank MMSE** / RLS equivalence …"*
- `supplementary/docs/DRAFT_main_paper_v0.md:178-191`
  *"**3.1 Bank MMSE without identification.** The field is represented on a fixed bank of
  ~1,900 wavenumber cells …"*
- ⟹ **이 원고의 "우리 추정기" = bank-MMSE.** 학습 위주 연구 방향에서 **주제 밖**이 된다.
  ⚠현재 상태 = 절 스텁만 있는 골격(2026-07-06).
- ⚠금지목록 머리말: **"bank-MMSE/refit 정확도 novelty" 인용 금지**(지연은 가능).

---

## 7. ⚠확신하지 못하는 지점 · 부수 발견

1. ⚠**`T-4_GEOMETRY_2026-08-30` 의 `SHIP` 팔** — `run_t4.py` 가 **존재하지 않는다**.
   `gates_t4.py:408-410`·`:483` 이 `run_b12` 상속을 **의도**로만 기록한다 ⟹ **미확인**.
   같은 이유로 `T-3`(스모크만, 본런 결과 없음)·`M-9`(`mode:"pre"`, 수리 미착지)는 **판정 없음**.
2. ★**`B-26` 게이트 불일치(신규 발견)** — `RESULT.md:6-7`(작성 08-28 07:07)은 *"게이트 29/29"* 라
   적는데, 현재 `verdict_b26.json`(mtime **2026-08-30 23:51**)은 **`gates_pass: 28 / 29`** 이고
   `G-0 시드 원장 충돌 0` 이 **FAIL(`충돌 [1641]`)** 이다. 그 뒤 다른 곳(`M-2`)이 seed 1641 을
   소모해 재실행에서 뒤집힌 것으로 보인다. **판정량과 무관한 위생 게이트**이나
   ⚠**"29/29" 를 인용하면 현재 산출물과 어긋난다.**
3. ⚠**기여 ⑦의 `t_upd = 1.6 ms`** — `analyze_b6b.py:24` 가 `verdict_b6.json` 중앙값을 쓰는데
   `B-6` 자체가 bank 하네스다. bank 초기화 후 성분 수가 **0**이므로 그 1.6 ms 가
   학습 경로의 추적 비용을 대표하는지 **미확인**.
4. ⚠**`W-12/RESULT.md:105-106` 의 커밋 오귀속** — 임계 flip 을 `36e3eb7`(12:36)로 적었으나
   실제는 **`d6738922`(13:58)**. 결론(런이 flip 전)은 **불변**.
5. ⚠**`W-8` 의 `G-4`(라우팅 일치)는 공허한 통과** — 러너가 `init_method_used`/`route` 라는
   **없는 키**를 읽어 48행 전부 `None`. 옳은 키는 `auto_selected`.
6. ⚠**`W-13/RESULT.md` 에 철회 배너가 없다** — `W-14` 는 붙었는데 `W-13` 은 `AMENDMENT_2.md:24-25`
   에만 지시가 있다. 같은 비교자 결함이 걸리므로 **`W-13` 의 `ANALYTIC` 열도 인용 불가**.
7. ⚠**2026-07 이전 레거시 배치**(`rerun_*`·`confirm*`·`heldout*`·`unified_*`·`blind_auto_real`
   ·`bankmmse_port` 등)의 팔 정의는 **열지 않았다**. 이 문서는 그것들을 분류하지 않는다.
8. ⚠**총량 표현 회피**: `verdict*.json` **90개**의 라벨을 전수 추출했고, **58개 배치**의
   `ARMS` 를 소스에서 직접 확인했다. 그 밖의 분류는 하지 않았다.
