# `T5-ENS` 사전등록 — 앙상블 교락의 예보 전이 (실측)

> **v2 (2026-09-01).** ⛔**v1 은 동결되지 않았고 폐기됐다** — 적대적 심사 3인(Opus·Fable·Opus)이
> **BLOCKER 8 · MAJOR 11 · MINOR 6** 을 냈고, 그중 셋이 설계를 죽였다. 개정 내역 = `AMENDMENT_0_PREFREEZE.md`.
> **동결 시점 = 결과 0행 · 추정기 0회.** 이 문서의 모든 수는 `gates_t5.py` 가 재도출한다(손복사 0).

---

## §0 질문 — 한 문장

> ***"나우캐스트에서 측정된 앙상블 이득(29/36 · −8.65%)이 **예보 지평으로 전이되는가?**"***

★**왜 이것이 「부호 재검」의 옳은 형태인가**: 저장소의 실측 예보 판정(`B-7`·`B-7b`·`W-12`)은
전부 **앙상블 없는 PROP** 의 성적이다. 그 교락이 판정을 뒤집으려면 **처치가 예보에서 힘이 있어야 한다.**
⟹ 처치의 힘을 직접 재는 것이 짝 비교를 다시 채점하는 것보다 **검정력이 높고 노출이 없다**(§2.3·§3).

---

## §1 미측정임의 전수 근거 (런 0회)

| # | 주장 | 근거 (실측 확인) |
|---|---|---|
| P1 | 패키지에 **학습 표현을 median-of-k 로 감싸는 경로가 없다** | `sequential_update.py:2028-2031`(래퍼는 `init_method=='auto'` 에만) · `:1940-1942`(`route != 'v2_konly'` ⟹ k=1) · `:1956-1957`(멤버를 `v2_konly` 로 하드 세팅) |
| P2 | bank 경로는 앙상블이 **켜져 있다** | `sequential_update.py:324 DEFAULT_REAL_BANK_ENSEMBLE_K = 3` · 적용 `:1134-1135` · `run.py:104` |
| P3 | ⟹ 실측 짝 판정에서 **PROP=무앙상블 · BANK=유앙상블** 이 구조적 확정 | P1 ∧ P2 |
| P4 | **예보 지평에서 앙상블을 준 PROP 은 한 번도 측정된 적이 없다** | `T3` 는 lead 0 단독(`gates_t3.py:82 LEAD=0.0`, `update()` 0회) · `B-7`/`T2`/`W-12` 의 PROP 팔은 전부 단일 멤버 |
| P5 | 멤버 노브 `learned_diffusion_seed` 는 **패키지 정규 config 키** | **`learned_init.py:353-354`** (`dseed = int(config.get('learned_diffusion_seed', …))`) · `sequential_update.py:1023` |
| P6 | ★**`analytic_ensemble_k` 는 learned 경로에서 무연산** | `make_updater` 조건이 `init_method == 'auto'`(`sequential_update.py:2028-2031`) ⟹ `init_method='learned'` 은 래퍼를 **받지 못한다**. ⟹ 앵커 팔(`T3 _PROP`)이 그 키를 명시한 것과 우리가 명시하는 것은 **동작상 등가**다(§4.2 는 그래도 명시한다 — 앵커 팔과 키 단위로 같게 두려고) |
| P7 | ★**`config['device']` 가 제안 경로에 도달하지 않는다** | `learned_init.py:355` 의 `propose_atoms(frames, depth, ckpt=ckpt, nsamp=nsamp, diffusion_seed=dseed)` 에 **`device` 인자가 없다** ⟹ `propose_atoms` 가 `:133-134` 에서 스스로 고른다(`'cuda' if torch.cuda.is_available() else 'cpu'`) |

★**P1 의 귀결**: 여기서 재는 팔은 **연구 팔**이다. ⛔**어떤 결과도 「배포 권고」가 아니다.**
★**P7 의 귀결**: **장치는 앵커가 정한다**(§7.1 `G-DEVICE`, `M8` 처방④).

---

## §2 이미 잰 것 · ★**노출 공시 (전수)**

### 2.1 나우캐스트에서 앙상블은 PROP 에 듣는다 (확증, `T3-NOW`)
`PROP-V400M3.e_med3 < .e_m0` = **29/36 · p=1.56e-04 · rel 중앙 −8.65%**.
★**이것은 이 배치의 「전제」이지 「엿본 답」이 아니다** — §0 의 질문이 *"그것이 전이되는가"* 이므로
그 값을 모르면 질문 자체가 성립하지 않는다(선례: `B-12` 가 `B-11` 을 확장할 때와 같은 구조).

### 2.2 bank 쪽 앙상블 효과는 실측에서 **≈0**
`M8-L1` 2/12 (동점 5, rel 0.0%) · `T3 E2` 15/36 (rel 0.00%, 퇴화 20/36).
⟹ 교락의 이빨은 BANK 쪽 이득이 아니라 **PROP 쪽 결손**이다.

### 2.3 ⛔⛔**노출 — 나는 나우캐스트 답을 이미 봤고, 그것이 NEAR 밴드를 8~9할 결정한다**

**(a) 내가 본 것 (2026-09-01, 사전등록 전, 동결 `T3` 자료 · 런 0회)**

| 비교 (lead 0, n=36) | 계수 | rel 중앙 |
|---|---|---|
| 비대칭(역사적 상태) `SHIP-K3.e_m0 < PROP-V400M3.e_m0` | 33/36 | −25.75% |
| 대칭 `SHIP-K3.e_m0 < PROP-V400M3.e_med3` | **31/36** | **−15.15%** |
| ★**끝점** `PROP-V400M3` vs 참 제로필드: 단일 **17/36** → median-of-3 **21/36** | | |

⛔**v1 은 마지막 줄을 공시하지 않았다** — 심사 A 가 잡았다.

**(b) 그 노출이 무엇을 오염시키는가 (심사 A 측정, 내가 재확인)**

| | Pearson(lead 0, NEAR 밴드) |
|---|---|
| **`PROP`** | **+0.864** (`B-7`·`B-7b` 양쪽 동일) |
| `SHIP` | +0.023 |

부호 일치: 끝점 **29/36 (81%)** · 짝 **32/36 (89%)**.
⟹ ★**NEAR 밴드의 끝점·짝 비교는 내가 이미 본 자료로 8~9할 결정돼 있다.**
⟹ **그 둘을 확증에서 뺀다**(§4.6 은 `EXPLORATORY`).

**(c) 왜 PRIMARY(전이)는 오염되지 않는가**
전이량은 **`e_med3` 와 `e_m0` 의 차분**이다. 위 상관은 두 값에 **공통으로** 실리는 초기화 오차의
상관이고, 차분에서 상쇄된다. ⚠**단 이것은 논증이지 측정이 아니다** — 예보 앙상블 차분은
어느 동결 자료에도 없다(P4). ⟹ **공시로 남기고 확증 등급은 유지한다.**

### 2.4 이 배치가 겨누는 판정 (전수)

| 판정 | 노출 유형 | 이 배치가 닫는가 |
|---|---|---|
| `B-2` · `M-8b` 끝점 | 나우캐스트 | ⛔아니오 — §2.3(a) 가 노출시켰다(Stage A 탐색적) |
| **`B-7` SEC-B · `B-7b` 헤드라인 · `W-12`** | **예보** | ✅**PRIMARY 가 처치의 힘을 재어 간접적으로** (§5) |
| `M-5b` PRIMARY · `M-8b` PRIMARY · `T2` SEC-A | ⛔교락 없음(짝 양쪽 동일 조건) | 해당 없음 |
| `B-8b` | ⛔bank 팔 없음 | 해당 없음 |

---

## §3 ★비대칭 — 결과 전에 적는다 (⛔**검정력 표 포함**)

**결과 전 예측**: 전이는 **일어나되 약해진다**. 나우캐스트 q=0.806 → 예보 NEAR q≈0.65–0.75.

★**v1 이 죽은 사유를 여기 박아둔다**: v1 의 PRIMARY 는 자기 예측(q=0.444) 아래에서
**P(라벨 없음 = UNDECIDED) = 87.5% · P(확증) = 0.07%** 였다. **예측을 적고도 그 예측 하의
검정력을 계산하지 않았다.** ⟹ ★**규율 신설: 사전등록은 「결과 전 예측」과 「그 예측 하의 검정력 표」를
함께 실어야 한다.**

**v2 PRIMARY 의 검정력 (n=36, CONFIRM=25)**

| 참 승률 가정 | P(≥25 확증) | P(≤11 역방향) |
|---|---|---|
| q=0.806 (나우캐스트 전이 100%) | **0.9651** | 0.0000 |
| q=0.70 (부분 전이 — **내 예측 중앙**) | **0.6091** | 0.0000 |
| q=0.60 (약한 전이) | 0.1620 | 0.0003 |
| q=0.50 (전이 없음) | 0.0144 | 0.0144 |

⟹ **내 예측 하에서 확증 확률 61%**. ⛔v1 의 0.07% 와 비교하라.
⟹ ★**그리고 음성(전이 없음)도 값이 있다** — §5.

**⛔결과를 보고 표본을 늘리지 않는다**(`B-8b` 선례). n=36 고정.

---

## §4 설계

### 4.1 처치는 단일 요인
PRIMARY·SEC-A 의 처치는 **`n_members` 1↔3** 하나뿐이고, 멤버는 `learned_diffusion_seed` 만 가른다
(`seed`=42 고정). `T3 member_overrides`(`gates_t3.py:218`)의 축자 상속.

### 4.2 팔 — **전 키 명시 · 빈 dict 0개**

⛔**빈 dict 로 팔을 정의하지 않는다**(`B-7` 사고: 런타임 스냅샷이 **당일 철회된** `bank_refit_every_s=5.0` 을 먹었다).

```
_BASE = dict(
    # ── 배포 삼종 + 표현 ─────────────────────────────
    varpro_steps=400, learned_init_iters=0, auto_switch_thresh=0,
    lambda_l1=1e-4, k_snap_dk=None, adaptive_n=True, max_n=30,
    # ── ★추적(=이 배치의 처치 기계. v1 은 이 셋을 빠뜨렸다) ──
    tracking_mode='w_phi', tracking_steps=10, tracking_lr=0.005, tracking_window=3,
    # ── bank ────────────────────────────────────────
    real_estimator='bank_mmse', bank_refit_every_s=None, bank_refit_async=False,
    bank_memory_s=40.0, bank_max_obs=600,
    # ── 기타 ────────────────────────────────────────
    use_confidence=True, seed=42, device='cpu',
    analytic_ensemble_k=3, real_bank_ensemble_k=3)
```
`depth` 는 셀마다 `scene_station` 이 준다(러너가 주입, 행에 싣는다).

| 팔 | config | 멤버 | MEMKEY | 앵커 |
|---|---|---|---|---|
| `PROP-M3` | `dict(_BASE, real_domain_gate=False, init_method='learned')` | `(0,1000,2000)` | `learned_diffusion_seed` | ★`B-7 PROP-TRACK` **전 곡선** |
| `BANK-K3` | `dict(_BASE, real_domain_gate=True, init_method='auto')` | `(0,)` | `None` | `T3 SHIP-K3.e_m0` (lead 0) |

- `PROP-M3` **멤버 0** = 무앙상블 PROP ⟹ **PRIMARY 의 대조군이 공짜**다.
- ⛔`BANK-K3` 는 **출하 기본값**이고 `B-7 SHIP-TRACK`(빈 dict)과 **같은 팔이 아니다**(§10).

### 4.3 사다리 · 채점 규약 (⛔**연산자를 문면에 동결한다**)

leads = `0..14 s` (1 s 간격, 15개). 밴드 = **NEAR `[1,3]`** · **FAR `[4,14]`**.

★**멤버 합성 규약 (v1 미선언 — 심사 B 가 잡았다)**:
> **lead 마다 ① 멤버들의 예측 장을 점별 median → ② 그 장 하나를 σ정규화 RMSE 로 채점 →
> ③ 밴드 안 lead 점수를 산술평균.**
> ⛔*"점수의 median"* 이 **아니다.** 두 수는 다르다.

근거: `T3 median_field`(`run_t3.py:113`) + `score`(`:128`) 의 축자 상속.
패키지 선례 = `MedianEnsembleUpdater.predict_field`(`sequential_update.py:2013-2019`, 임의 `t` 에서 멤버 median).
⚠**`T3 G-MIRROR`(하네스 median == 패키지 median)는 bank·lead0 에서만 증명됐고 learned·lead>0 에는
패키지 대응물이 없다(P1)** ⟹ **상속하지 않는다.** 대신 `G-MEDIAN-RULE`(§7.1)이 연산자를 직접 검사한다.

채점 순서는 `B-7` 축자 상속: **①먼저 예측·채점 → ②그 시각 관측 투입**(`run_b7.py:67-86`), lead 0 무투입.
⛔**미래 관측은 채점에 쓰지 않는다**(`scene_station.py:74` 계약) — `G-ANCHOR-CURVE` 가 비트로 강제한다.

### 4.4 원점 · 표본
- 셀 = **12 레코드 × 창 {1,3,5} = 36** (`gates_b3.py:30-34` 임포트, 재선언 0)
- `SEED=42` · `NOISE=0.02` — 동상
- ★**새 에피소드 시드 소모 = 0** ⟹ 시드 원장 `[1758–1818]` 에서 아무것도 청구하지 않는다
  (`B-26 G-0` 류 사후 충돌 원천 차단)
- 기준 = **참 제로필드**(`zero_ref.zero_ref_perlead`, 캐시 `B3_REFIT/true_zero_perlead.json`).
  ⛔**고정 상수 기준 금지**(레코드별 0.3375–1.1559).

### 4.5 문턱 (코드 재도출 — `gates_t5.thresholds()`)

| | 값 |
|---|---|
| n | 36 |
| ALPHA | 0.05 |
| **N_FAMILY** | **2** (PRIMARY · SEC-A — ⛔v1 의 4 에서 줄였다) |
| ★**Holm 사다리** | **`HOLM_MIN_WINS = [25, 24]`** (v1 은 "Holm" 이라 적고 26 을 고정해 **w=25 에 구멍**이 났다) |
| **CONFIRM** | **25** (p=0.014408 ≤ α/2) |
| **NOMINAL** | **24** (p=0.032623 ≤ α) |
| ★**NOT_EARN** | **11** = n − CONFIRM (⛔v1 의 12 는 **양성에만 가족 보정을 걸고 음성엔 안 건** 비대칭이었다) |
| ★**MDE q80** | **0.742** @ CONFIRM — ⛔**판정문 병기 의무** |

### 4.6 판정 · 라벨 결정표 (결과 전 고정 · ★**공역 전건 정의**)

등급은 `T3 _grade`(`gates_t3.py:452-465`) 규약을 그대로 쓴다 —
**Holm 사다리 생존 여부**가 등급을 정하지, 고정 문턱이 정하지 않는다.

**PRIMARY** — `PROP-M3.band_near_med3 < PROP-M3.band_near_m0` (n=36)
⚠**채점 필드를 명시한다**(v1 미명시, 심사 B): 좌변 `band_near_med3`, 우변 `band_near_m0`.

| 조건 | 라벨 |
|---|---|
| Holm 사다리 **생존** | ★`T5-ENS :: ENSEMBLE-TRANSFERS (CONFIRMATORY)` |
| w ≥ NOMINAL(24), Holm **탈락** | `T5-ENS :: ENSEMBLE-TRANSFERS [EXPLORATORY]` |
| NOT_EARN(11) < w < NOMINAL(24) | `T5-ENS :: TRANSFER-UNDECIDED` |
| w ≤ NOT_EARN(11) | `T5-ENS :: TRANSFER-REVERSED [EXPLORATORY]` ⚠**역방향 꼬리** |
| 차단 게이트 FAIL | ⛔**라벨 없음** → §4.6b |

★**역방향 꼬리 규율**(심사 A): `w ≤ 11` 은 *"앙상블이 예보에서 **해친다**"* 는 발견이지만
가족 α 는 **단측 사전등록 방향에만** 썼다 ⟹ ⛔**확증 승격 금지**, 등급 `EXPLORATORY` 고정
(선례 `T3 PREREG:302`).

**SEC-A** — 같은 검정, 밴드 **FAR**. 라벨은 위와 같되 주어가 `band_far_*`.

**⛔확증 가족 밖 (전부 `EXPLORATORY` 고정 · 결과와 무관)**
- `EX-1` 짝 `BANK-K3 < PROP-M3` (NEAR·FAR) — ⚠**§2.3(b) 노출 89%**
- `EX-2` 끝점 `PROP-M3.med3` vs 참 제로필드 (NEAR·FAR) — ⚠**노출 81%**
- `EX-3` 끝점 `BANK-K3` vs 참 제로필드 — ★**음성 대조**(`F1` 교훈: 비교자에 기술이 없으면
  짝 승수는 기준선 붕괴의 산물이다). ⛔**공시 전용**, 판정 아님

**★모순 조합의 읽기 (결과 전 고정 — 심사 A)**
| 조합 | 읽기 |
|---|---|
| PRIMARY 확증 ∧ `EX-2` 널 | *"앙상블은 예보에서 오차를 줄이지만 제로필드 문턱을 넘기진 못한다"* — 교락에 이빨은 있으나 척추는 안 바뀐다 |
| PRIMARY 널 ∧ `EX-1` 부호 유지 | ★*"처치가 예보에서 무력하다 ⟹ 앙상블 교락은 예보 판정을 뒤집을 수 없다"* (§5) |
| PRIMARY 널 ∧ `EX-1` 부호 반전 | ⛔**주장 금지** — 처치가 무력한데 부호가 갈렸다면 그것은 앙상블이 아닌 다른 것이다. **미해결로 기록** |
| PRIMARY 역방향 | ⛔확증 금지. `T3` 나우캐스트와 **방향이 반대**임을 공시하고 미해결로 남긴다 |

### 4.6b ★앵커 불일치 시의 처분 (결과 전 고정 — 심사 C)
⛔`M8` 은 앵커 실패로 **라벨 없이 영구 종료**됐다. 그 출구는 실제로는 **결과를 보지 않은 채 고른
승계 재분석**(`M8b`)이었는데 v1 은 그 경로를 등록하지 않았다.

1. 앵커 게이트가 FAIL 하면 **PRIMARY 계수를 열지 않는다**(분석기가 그 자리에서 멈춘다).
2. 진단은 **계수를 보지 않는 것만** 허용: ①멤버 0 을 `CUDA_VISIBLE_DEVICES=""` 로 재현 ②장면 상수 항등
   ③`ckpt_sha` 대조.
3. 진단이 원인을 **장치**로 특정하면 `H-ANCHOR`(위생)로 강등하고 배치는 **그 공시를 달고** 진행한다.
4. 특정하지 못하면 라벨 없이 닫고 **승계 배치**(`T5b`)로 간다. ⛔게이트 수정·재실행 금지.

### 4.7 Stage A (런 0회 · 등급 `EXPLORATORY` **고정**)
`stageA_reagg.py` 가 동결 `T3` 자료에서 §2.3(a) 표를 다시 만들고 어긋나면 `AssertionError`.
★**산출물 규약**(심사 C): **결과 파일을 쓰지 않는다.** `verdict_stageA.json` 에
`source_results: ["../T3_NOWCAST_2026-08-30/results_t3.jsonl"]` 를 싣는다
(`check_batch_artifacts.py:179-181` 재집계 규약).
⛔승격 없음 · ⛔`B-2`/`M-8b` 판정 갱신 없음 — 산출은 **공시 한 줄**이다.

---

## §5 널이면 무엇을 쓰는가 (`gates_t5.READING`)

- **PRIMARY 널(TRANSFER-UNDECIDED)** ⟹
  ***"나우캐스트에서 측정된 앙상블 이득이 예보 지평으로 전이되는 것은 검출되지 않았다
  (n=36, MDE q80=0.742)."***
  ★**그리고 그 귀결이 이 배치의 값이다**: 처치가 예보에서 힘이 없다면 **앙상블 교락은
  `B-7`·`B-7b`·`W-12` 의 부호를 뒤집을 수 없다** ⟹ ***교락이 「측정된 공시 한 줄」로 내려간다.***
  ⛔*"차이 없다"*·*"전이되지 않는다"* 금지 — **「검출 불가」**로만.
- **PRIMARY 확증** ⟹ 교락에 이빨이 있다 ⟹ `EX-1`/`EX-2` 를 **탐색적으로** 읽고, 확증은
  **새 사전등록**으로 넘긴다(⛔이 배치에서 승격 금지).
- ⛔어떤 결과도 **배포 권고**가 아니다(§1 P1).
- ⛔**사이트 4곳** ⟹ 사이트 수준 일반화 금지.

---

## §6 공시 (판정문 동반 의무)

1. **연구 팔** — 패키지에 학습 앙상블 경로 없음(P1).
2. **blind 아님** — §3 에 결과 전 예측 + 검정력 표.
3. **노출** — §2.3 (a)(b)(c) 전건. ⛔*"노출은 나우캐스트뿐"* 은 v1 의 거짓 문장이었다.
4. `BANK-K3` ≠ `B-7 SHIP-TRACK`(빈 dict, 철회된 refit 5.0).
5. **MDE q80 = 0.742** 병기.
6. **장치**: `cuda_available=True` 환경. `config['device']` 는 제안 경로에 **도달하지 않는다**(P7)
   ⟹ 앵커가 장치를 정한다.
7. **사이트 4곳 · 레코드 12개.**

---

## §7 게이트

### 7.1 차단 게이트 (FAIL ⟹ 라벨 없음 → §4.6b)

| 게이트 | 검사 | 심사 지적 반영 |
|---|---|---|
| `G-APPROVAL` | env `T5_APPROVED=1` **∧** `--approved` 둘 다 · 행 `approved=True` 전건 | |
| `G-ARM` | 팔 config **되읽기 == 의도** (⛔의도표 자기대조 금지, `S13b` 교훈) | A-MINOR |
| `G-DEPLOY-PIN` | 출하 기본값 트립와이어 — ★**`tracking_steps`·`tracking_lr`·`tracking_window` 포함** | B-4·C-5 |
| `G-MEMBER` | `seed_readback`·`diffusion_seed_readback` == `member_overrides()` 의도, 전건 | |
| ★`G-MEDIAN-RULE` | 합성 연산자가 **「장 median → 채점」** 임을 합성 픽스처로 검사(«점수 median» 을 넣으면 FAIL) | B-3 |
| ★`G-ANCHOR-CURVE` | `PROP-M3` **멤버 0 의 15-lead 곡선 전건**이 동결 `B-7 PROP-TRACK.per_lead` 와 **비트 동일**(36셀×15lead) | B-7(공짜 앵커) |
| ★`G-ANCHOR-BANK` | `BANK-K3` lead 0 == 동결 `T3 SHIP-K3.e_m0` 비트 (36/36) | B-7 |
| ★`G-SCENE` | **(a)** 행 대조는 `T3` 에 **실재하는** `sigma`·`npx` 로 한정 **(b)** `depth`·`n_obs` 는 **원천 항등**으로 — 동결 md5 아래서 `scene_station(record,win,NOISE)` 12레코드 재호출해 대조. ⛔v1 은 `T3` 행에 **없는** `depth`·`n_obs` 를 읽어 `M8` 을 문자 그대로 재연할 뻔했다 | ★B-1·C-1 |
| ★`G-ROUTE` | 팔별 `init_method_readback`·`real_estimator_readback`·`real_gate`·`ens_k_readback` 전건 == 의도. ⛔`v2_konly` 로 새는 것을 잡는다(class fallback 은 **LIVE**) | C-4 |
| ★`G-DEVICE` | `device`·`cuda_available`·`omp_threads`·`prior_device`·`ckpt_sha` 가 **전 행 단일 스탬프** | C-4 |
| ★`G-REFIT0` | `refit_calls == 0` 전건 (`bank_refit_every_s=None` 이 실제로 먹었는가) | C-4 |
| `G-INTERVENTION` | `PROP-M3` 의 `track_steps > 0` **멤버별 전건**. 0 이면 ⛔`PROP-NO-INTERVENTION` | B-8(멤버별 목록) |
| ★`G-FLAT` | **`.py` 만 · `ast.Name`/`Attribute` 만 · 토큰은 `_FLAT_NAME='ZERO'+'_REF'` 로 조립(리터럴 금지) · 산문 통과 · ★`FROZEN_md5.txt` 에 적힌 파일만 훑는다**(⛔`os.listdir` 금지 — 나중 놓인 포렌식 스크립트가 **소급 FAIL** 시킨다) | ★C-3 |
| `G-ROWS` | 팔별 셀 커버리지 36/36 · 0행 가드 |

### 7.2 범위 축소 게이트 (FAIL ⟹ 판정은 살고 주장 범위가 준다)
`G-DEGEN`(멤버 해시 distinct — 퇴화 셀 공시) · `G-LEADCOV`(밴드별 lead 커버리지) ·
`H-SEED`(시드 원장 — ★**`blocked_by` 에 넣지 않는다**, `B-26 G-0` 소급 차단 방지)

### 7.3 ★게이트 자가진단 (동결 **전** · `gates_t5.selftest()`)
`T3` 선례(`gates_t3.py:967-1100`)를 그대로 쓴다:
1. 합성 픽스처로 **현 상태 전건 PASS**
2. ★게이트별 `FORGERIES` 표 — **위조 1건 주입 시 반드시 FAIL** (항진 금지)
3. ★**`uncovered == []`** — 위조가 없는 차단 게이트를 열거해 0 임을 단언
4. ★**팔 삭제 민감도** — 어느 팔의 행을 지워도 반드시 어떤 차단 게이트가 FAIL (**항등 게이트 0**)
5. ★**0행 가드** — 행 0개면 `label is None` ∧ 카운터 `None` ∧ `blocked_by` 비지 않음
6. ★**라벨 공역** — `w ∈ [0,36] × {surv, ¬surv}` **전 조합에서 라벨이 정의됨**을 단언 (v1 의 w=25 구멍)
7. `check_gate_refs` · `check_gate_blindspots` 를 동결 전 실행, ★**5b 출력의 「검사된 배치 목록」에
   `T5` 가 있을 것**(건너뜀이면 실패) ⟹ `gates(rows, premise=None)` 시그니처 고정

---

## §8 러너 계약 — 행 스키마 (결과 전 동결)

```
batch arm record win approved n_members member_offsets
per_lead_med per_lead_m0 per_lead_members
band_near_med3 band_near_m0 band_far_med3 band_far_m0
zero_per_lead zero_near zero_far
field_md5_members_by_lead field_md5_med_by_lead
track_steps_members refit_calls init_fit_calls n_atoms_members n_fut_obs
seed_readback diffusion_seed_readback init_method_readback real_gate
real_estimator_readback ens_k_readback arm_cfg_nonempty deploy_pin_ok
anchor_curve_bitmatch scene_match
device cuda_available omp_threads prior_device ckpt_sha n_selected_members
depth sigma npx n_obs code_md5 git_head wall_s
```
★**H7 처방 반영**: `ckpt_sha`·`n_selected_members`·`prior_device`·`cuda_available` — `T3` 행엔 없었다.
★**`depth`·`n_obs` 를 싣는다** — `M8` 처방②(*"장면 상수를 행에 싣지 않은 자료는 비트 앵커로 쓸 수 없다"*)
를 **다음 배치를 위해** 지킨다. ⚠그 규칙상 `T3` 자신은 `depth` 가 없어 앵커 자격이 약하다 ⟹ §7.1 `G-SCENE`(b).

**실행 순서**: `selftest` → `smoke`(⚠**가장 비싼 팔 `PROP-M3` + 게이트 발화 셀 포함**) → 동결 →
본런(샤드) → `analyze_t5.py` → `check_batch_artifacts.py` → `run_all.sh`

---

## §9 비용 (⛔v1 은 **1.6× 과소추정**했다 — 심사 B)

`B-7` 의 6.71 s 는 **스레드 미고정** 환경 값이다. 앵커가 강제하는 환경은 `OMP_NUM_THREADS=1`
(`T3` 144행 전건)이고, 거기서 `PROP` 초기화+예측만 **9.7–10.1 s/런**이었다.

| 팔 | 런/셀 | 추정 셀당 | 셀 | 합 |
|---|---|---|---|---|
| `PROP-M3` | 3 | ~35 s | 36 | **≈ 21 분** |
| `BANK-K3` | 1 | ~3.2 s | 36 | ≈ 2 분 |
| 합 | **144 런 / 72 행** | | | **≈ 23 분** (단일 프로세스) |

샤드 3~6. ⚠**추정이다 — 스모크가 확정한다.**

---

## §10 ⛔주장하지 않는 것

- ⛔**배포 권고 아님**(P1).
- ⛔`B-7`·`B-7b`·`W-12` 의 **공표 계수를 이 배치 계수로 교체하지 않는다** — `BANK-K3` 는 다른 팔이고
  `W-12` 는 프로토콜이 다르다. 이 배치는 **처치의 힘**을 재지 **판정을 갱신하지 않는다**.
- ⛔`T3` 의 −8.65% 를 이 배치에 전이하지 않는다(크기 비교 금지).
- ⛔**역방향 꼬리 확증 승격 금지**(§4.6).
- ⛔나우캐스트 확증 금지(§2.3 노출).
- ⛔*"학습 표현 특유의 효과"* 금지 — **`v2_konly`+앙상블 팔이 없다**(심사 B-8; 예산상 제외).
- ⛔사이트 수준 일반화 금지.

---

## §11 동결 (`FROZEN_md5.txt` — **덧붙이기**, ⛔덮어쓰기 금지)

★**규칙**: `gates*` · `run*` · `analyze*` 로 시작하는 `.py` 는 **전부** 동결(스모크 스크립트 포함 —
`check_batch_artifacts._judge_files` 의 `JUDGE` 정규식이 잡는다).

- 배치 내: `PREREG.md` · `AMENDMENT_0_PREFREEZE.md` · `gates_t5.py` · `run_t5.py` · `analyze_t5.py` ·
  `premise_t5.py` · `premise_t5.json` · `stageA_reagg.py` · ★`APPROVAL_T5_2026-09-01.md`
- `ast` 임포트 전건: `gates_b3.py` · `scene_station.py` · `zero_ref.py` · `make_gif_bank_deployed.py` ·
  `gates_b7.py` · `gates_t3.py`
- 앵커 자료: `B7_PROP_TRACK_2026-08-23/results_b7.jsonl` · `T3_NOWCAST_2026-08-30/results_t3*.jsonl` ·
  `B3_REFIT_2026-08-23/true_zero_perlead.json`
- 배포 경로: `sequential_update.py` · `learned_init.py` · `run.py` · `bank_mmse.py`

⛔**동결 대상 아님**: `RESULT.md` · `verdict*.json` · `results*.jsonl` · `*.log`

## §12 현재 상태
- 결과 0행 · 추정기 0회 · 패키지 수정 0줄 · git commit 0회
- 방벽 6종(`run_all.sh`) 출력을 동결 시점 1회 · 본런 후 1회 첨부
- ⚠`APPROVAL_T5` 미작성 시 러너가 **기동하지 않는다**(`G-APPROVAL`)
