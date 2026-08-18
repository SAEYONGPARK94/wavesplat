# PREREG 초안 — `S17 · WHITE25 / WHITEHS50` **게이트 절 (§G) + 동결 사슬 (§F) + 계측 (§I)**

> **이 파일의 지위**: T2-B(게이트·계측 위생) 가 **런 0회**로 작성한 **PREREG 삽입용 완성 초안**이다.
> 본 파일 자체는 사전등록이 아니다 — `PREREG.md` 에 §G/§F/§I 로 **그대로 붙여 넣고**,
> 그 `PREREG.md` 의 md5 를 `FROZEN_md5.txt` 에 적은 시점부터 동결이다.
> **작성 시점에 이 배치의 판정 자료(`results.jsonl`)는 존재하지 않는다**(§F-2 가 증명 절차를 정한다).
>
> **열람 범위 = 주장 범위.** 이 초안이 근거로 연 파일:
> `tmp/S16_BASELINE_2026-08-14/{PREREG.md, AMENDMENT_1.md, RESULT.md, NOTES.md(발췌), SCRIPTS_md5.txt,
> FROZEN_md5.txt, code_fingerprint.json, isotropy_probe.json, letkf_spec.py, run_s16.py(발췌),
> analyze_s16.py(발췌), results.jsonl(집계만)}` ·
> `tmp/W3_SOLVER_2026-08-15/{PREREG.md, AMENDMENT_2.md, NOTES.md}` ·
> `tmp/S15_DIRECTION_2026-08-14/{PREREG.md §7·§9, SMOKE.md, chain.sh}` ·
> `wavesplat/baselines/letkf_grid.py` · `memory/project_w3_solver_2026-08-16.md:74-75`.
> ⛔`analyze_s16.py` 전문, `run_s16.py` 전문은 **열지 않았다**(발췌 grep + 지정 구간만).

---

## §G-0. 이 배치가 재는 것과, 게이트가 지켜야 하는 것

**팔 (신규 2개, 전부 `letkf_spec.LETKFGridSpec`)**

| 팔 | `background` | `white_amp` | `loc_radius` | 오라클 | 신규 런 |
|---|---|---|---|---|---|
| **`WHITE25`** | `white` | **0.1 m**(재고 결함 그대로) | **25** | **없음** | 128ep × 3시드 = 384 |
| **`WHITEHS50`** | `white` | **`Hs_true/4`**(0.30 / 0.50 m) | 50 | **`Hs` 오라클** | 128ep × 3시드 = 384 |

**재사용 (신규 런 0회)** — `tmp/S16_BASELINE_2026-08-14/results.jsonl`
**1,920행**(`SPEC25`·`SPEC50`·`SPEC100`·`WHITER50`·`WHITE50` × 128ep × 시드 42/43/44, **에러 0**,
`code_md5` **1종 `5bba1a480cba1ac14c97f237a0b9777d`**, `background_diag` **1,536행 비null**
= `WHITE50`(재고 클래스) 384행만 null — 본 초안이 직접 재집계함) +
`SHIP_med3` 384행(S13 128 + S13b 256, S16 §2 경로 그대로).

**단일요인 대조 3개 (요인이 정확히 하나씩만 다르다)**

| 이름 | 대조 | 다른 키 (선언) | 상태 |
|---|---|---|---|
| **L (국소화)** | `WHITE25` vs `WHITER50` | `loc_radius` {25, 50} | **신규** |
| **A (스프레드)** | `WHITEHS50` vs `WHITER50` | `white_amp` {Hs/4, 0.1} | **신규** |
| **S (구조)** | `SPEC50` vs `WHITEHS50` | `background` {jonswap, white} | **신규**(S16 A-2 가 이월한 바로 그 대조) |
| (기지) S+A | `SPEC50` vs `WHITER50` | `background` + `white_amp` | S16 R-2 = **28/64, med(rel) +3.5%** (`RESULT.md:` §2 표) |

⟹ **게이트가 지켜야 하는 것은 하나로 요약된다**: *S16 이 `loc` 을 안 맞춰 "수리 성공 59/64" 를 찍었다가
맞추자 28/64 로 뒤집힌 사건*(`S16/RESULT.md` §3.2)이 **이 배치에서 재발하지 않는 것**.
그 재발을 막는 게이트는 **G-3(요인 정합)** 과 **G-9(조작 검사)** 이고, 나머지는 재사용·동결·기록 위생이다.

**비용 (사전 기록)**: S16 실측 1,920런 = 21.62 h ⟹ **≈40.5 s/런** ⟹ 신규 768런 **≈8.6 h**(순차·단독 점유).
⚠**시드 3개는 선택이 아니라 정합 요구다**(G-17): S16 팔들이 median-of-3 로 집계돼 있으므로 신규 팔을
단일 시드로 두면 **집계 규약이 팔마다 다른 대조**가 된다 = B2 결함의 재발.

---

## §G. 게이트 (19개) — **각 게이트는 "실패할 수 있는가" 를 함께 못 박는다**

> 표기: **시점** = `PRE`(런 0회) · `SMOKE`(2 에피소드, `smoke_results.jsonl` **분리 저장**) ·
> `RUN`(전량 실행 중) · `POST`(전량 후, **판정 인쇄 전**).
> ★ = **이번 배치 고유**(S16/W3/S15 에 없던 것 또는 실질적으로 새로 세운 것).

### A. 런 전 (`PRE`) — 런 0회로 통과해야 스모크로 간다

| ID | 검사 대상 | 통과 기준 (숫자) | 실패 시 | 근거 |
|---|---|---|---|---|
| **G-1** | 표본 무결성 | `data_s9_sparse_2026-08-11` + `data_s8_crowded_2026-08-10` + `data_s9_crowded_2026-08-11` 에서 **128 에피소드**(혼잡 64 / 희소 64), 접미사 전부 `uniform_ra_rf_ds`, 중복 0, **키 집합이 S16 `results.jsonl` 의 128키와 완전 일치**(대칭차 0) | **중단** | S16 `PREREG.md:242`(G-1) · `run_s16.py:47-49,92-104` |
| **G-2** | 수입 인벤토리 | S16 `results.jsonl`: **1,920행 · 5팔 × 128ep × 시드{42,43,44} 각 128 · `error` 행 0 · `code_md5` 1종** · `REQUIRED_LETKF` 전 키 비null(단 `background_diag` 는 `WHITE50` 384행 null 허용, 아래 G-15 참조). `SHIP` 수입 **384행**, 전 행 `effective_config.lambda_l1 == 1e-4` · `seed ∈ {42,43,44}` | **중단** | S16 `PREREG.md:243-245`(G-2·G-4) · 본 초안 재집계 |
| **G-3** ★ | **상수·요인 정합 (cross-batch)** — 신규 팔의 `effective_config`(**객체 되읽기**, `letkf_spec.py:276-283`)를 S16 대응 팔의 것과 키별 비교 | 각 대조에서 **다른 키가 §G-0 표의 선언 키 정확히 1개**. 나머지 전 키 일치: `n_ensemble=600` · `mult_inflation=1.0` · `add_inflation=0.005`(+`_add_inf`) · `obs_noise` · `max_obs=40` · `depth` · `dt` · `max_eta` · `nx=40` · `ny=30` · `dx` · `dy` · `g` · `filter_class='LETKFGridSpec'` · `direction_model='isotropic'` · `filter_seed` | **해당 대조 판정 금지** (`UNMATCHED-CONTRAST`) — 다른 대조는 진행 | S16 `AMENDMENT_1` A-2 (B2) · `RESULT.md` §3.2 (59→28) |
| **G-4** ★ | **코드 지문 2층** — ① 전체 지문(고정 87파일, 2회 계산해 동일: `run_s16.py:111-131`, A-5) ② **LETKF 임포트 경로 부분집합** `wavesplat/baselines/letkf_grid.py` · `wavesplat/data_loader.py` · `wavesplat/evaluation/metrics.py` · `wavesplat/evaluate.py` · `wavesim/data_writer.py` 의 md5 가 S16 `code_fingerprint.json` 값과 **전건 일치** | ②는 **5/5 일치 필수**. ①은 **S16 digest 와 불일치가 정상**(HEAD `5f0846c`→`554dd78` 사이 `configs/estimation.yaml`·`optimizer/sequential_update.py` 2파일이 바뀌었고 **둘 다 주석/독스트링 전용**임을 diff 로 확인함) ⟹ **전체 digest 동일성을 재사용의 근거로 쓰지 않는다** | ② 실패 ⟹ **중단**(재사용 무효) · ① 불일치는 **기록만** | 본 초안 실측(오늘 5/5 SAME) · S15 `PREREG.md:302`(G-6 `import_md5`) |
| **G-5** | **동결 사슬 + 판정 행 0** (§F 가 절차) | `results.jsonl` **부재** · `PREREG.md` md5 == `FROZEN_md5.txt` 기재값 · 스크립트 4종 md5 == `SCRIPTS_md5.txt` · **런 전 git 커밋 존재** | **중단** | W3 `NOTES.md`(20:47 `196050e`) · S16 `FROZEN_md5.txt` |
| **G-6** | 시드 원장·생성 0 | 신규 `.h5` **0개**(실행 전/후 파일목록 diff) · **추정기 시드 소모 0**(`SHIP` 은 수입) · 필터 시드는 {42,43,44} 로 S16 과 동일(원장 소모 아님) | **중단** | S16 `PREREG.md:251`(G-10) · W3 `PREREG.md` G-9 |

### B. 스모크 (`SMOKE`) — 2 에피소드 `BF4_15comp_seed32` · `BF4_2comp_seed48`, **판정 자료 아님**

| ID | 검사 대상 | 통과 기준 (숫자) | 실패 시 | 근거 |
|---|---|---|---|---|
| **G-7** | **재사용 정당성 = 비트 재현** ① 오늘 코드로 `WHITER50@42` 2ep 재실행 vs S16 저장값 ② `SHIP@42` 재실행 vs S13 저장값 | **둘 다 `\|Δ\| = 0.000e+00`** (`avg_nRMSE` · 9셸 전부) | ① 실패 ⟹ **중단**(S16 열 재사용 불가) · ② 실패 ⟹ `SHIP` 수입 금지, 이 배치는 **기준선 3요인 분해만** 수행하고 `SHIP` 이 든 대조는 `NOT-RUN` | S16 `PREREG.md:246`(G-5) · S16 `RESULT.md` §4(G-5b 128/128 = 0) |
| **G-8** ★ | **누출 프로브 = 오라클 채널 인벤토리** (§I-2 가 구현) — 필터 **생성자 입력**의 `Hs_bg`/`Tp_bg` 만 교란(채점기의 `Hs_true` 는 **불변**) | **음성 3건**: `WHITE25` 는 `Hs×1.37` 에서 `\|Δ\|=0`, `Tp×0.83` 에서 `\|Δ\|=0`; `WHITEHS50` 은 `Tp×0.83` 에서 `\|Δ\|=0`. **양성 1건**: `WHITEHS50` 은 `Hs×1.37` 에서 `\|Δ\| > 1e-6` (**같은 게이트가 자기 무의미를 검출한다**) | 어느 하나라도 실패 ⟹ **중단** (팔이 선언하지 않은 GT 를 받거나, 선언한 GT 를 안 받는다) | W3 `PREREG.md` G-5(누출 프로브) · S16 `AMENDMENT_1` A-1(선언 안 한 오라클) · `letkf_spec.py:192-199` |
| **G-9** ★ | **배경 조작 검사 4항** (`background_diagnostics()`, `letkf_spec.py:226-267`) | **(a) 스프레드**: `\|std(η)/목표 − 1\| ≤ 1%` — `WHITE25` 목표 **0.1**, `WHITEHS50` 목표 **Hs/4 = 0.30(BF4)/0.50(BF5)**. **(b) 구조 부재**: `\|lag1_corr_x\|, \|lag1_corr_y\| ≤ 0.05` (백색 팔은 **결함을 보존해야** 대조가 성립한다). **(c) 모드수**: 백색 팔 `n_modes_nonzero == 0`. **(d) 등방성(승계)**: `\|corr(η̇,∂η/∂x)\|, \|corr(η̇,∂η/∂y)\| ≤ 0.15` | (a)(b)(c) 실패 ⟹ **중단**. (d) 실패 ⟹ **중단**(단 §G-등방성 판정 참조 — 백색에서는 **자명 통과**라 정보가 없다) | `letkf_spec.py:85,192-194,200-211` · S16 `isotropy_probe.json` · `AMENDMENT_1` A-1 |
| **G-10** ★ | **관측 thinning 지문** (§I-3) — 프레임별 thinning 인덱스를 이어붙여 md5(`thin_md5`) | **동일 (에피소드, filter_seed)** 에서 `WHITE25` = `WHITEHS50` = `WHITER50` = `SPEC50`(재실행분) 의 `thin_md5` **완전 일치**. **양성 대조**: 재고 `WHITE50` 은 **불일치해야 한다**(재고는 `add_inflation>0` 이라 `letkf_grid.py:164-165` 가 전역 스트림을 매 프레임 소비, spec 팔은 `letkf_spec.py:152-153` 이 0 으로 두어 전역 스트림을 **thinning 만** 소비) | 일치 실패 ⟹ **중단**(대조에 관측집합 교락) · 양성 대조 실패 ⟹ **기록**(재고 경로 이해가 틀렸다는 뜻, RESULT 에 명시) | `letkf_grid.py:116-121,164-165` · `letkf_spec.py:142-153` · S16 `PREREG.md:106-109` |
| **G-11** | 필터 시드 민감도 재측정 | 신규 2팔 × 2ep × 시드{42,43,44} 의 `avg_nRMSE` **상대 표준편차**를 인쇄. **문턱 2%**(S16 G-6 승계) | **>2% 여도 시드 수는 이미 3 으로 고정**(G-17 정합 요구) ⟹ **중단 없음**. 대신 §G-크기규율 발동: 그 팔이 든 대조의 **크기 인용 금지**, **부호만** 사용 | S16 `PREREG.md:247`(G-6) · S16 `SCRIPTS_md5.txt`(에스컬레이션 발동 8.216%/13.549%) |
| **G-12** | **분석기 자기봉인**(드라이런) | `analyze_s17.py` 가 ① 자기 md5 를 판정 로그 **첫 블록에 인쇄** ② `assert` 로 입력 경로가 `results.jsonl` 로 끝남 ③ 문턱·팔 풀·대조 목록이 **동결값과 일치**함을 assert ④ 합성 행으로 **전 분기**(부호검정·Holm·PROVISIONAL 도장·`UNMATCHED-CONTRAST`·`DEGENERATE`) 실행 확인 | **중단** | W3 `AMENDMENT_2` 중립 F-1/U-2/W-6 · U-3 · S16 `RESULT.md` §1(합성 2시나리오 사전 검증) |

### C. 런 중 (`RUN`)

| ID | 검사 대상 | 통과 기준 | 실패 시 | 근거 |
|---|---|---|---|---|
| **G-13** ★ | **진행 감시 = 파일 행 수**(§I-4) | 감시자는 `wc -l results.jsonl` 만 본다. **`pgrep -f <러너>` 금지**(자기 명령줄 매칭). 완료 판정 = **행 수 == 768** ∧ 러너가 마지막에 쓰는 `DONE.marker` 존재. 정체 = 행 수가 **>15분**(≈ 중앙 40.5 s/런 의 20배) 불변 | 정체 ⟹ 경보만(자동 kill 없음). 완료 오판 ⟹ 설계상 불가(두 조건 AND) | `memory/project_w3_solver_2026-08-16.md:74-75` (**Phase A 종료를 8시간 늦게 발견**) |
| **G-14** | 재개·중복·에러 | 행 키 `(episode, arm, filter_seed)` 로 재개. 종료 시 **키별 성공행 정확히 1개**, `error` 키 보유 행 **0** | 에러 행 >0 ⟹ 재시도 1회 후에도 남으면 그 셀을 `MISSING` 으로 인쇄하고 **PROVISIONAL** | S15 `PREREG.md §10` · W3 `PREREG.md` G-10 |

### D. 런 후 · 판정 인쇄 전 (`POST`)

| ID | 검사 대상 | 통과 기준 (숫자) | 실패 시 | 근거 |
|---|---|---|---|---|
| **G-15** | **필드 계약 (DCC)** — §I-1 목록 | 신규 768행이 §I-1 의 **공통 키 + spec 팔 추가 키**를 전부 갖고 `None`/`NaN` 아님. **셸 키 집합 9개가 팔 간 동일**. 수입 1,920행은 S16 계약으로 검사(단 `WHITE50` 의 `background_diag=null` 은 **사전 선언된 예외**) | **판정 인쇄 없이 종료** (`SCHEMA-FAIL`) | S15 `PREREG.md:248-268`(§7, G-8) · W3 `PREREG.md §6`(DCC) · S16 `PREREG.md:111-127` |
| **G-16** ★ | **제로필드·프레임 마스크 정합 (배치 내 + 배치 간)** | ① 같은 에피소드에서 신규 2팔·3시드의 `zero_nRMSE` **max\|Δ\| < 1e-12** ② **S16 저장 `zero_nRMSE` 와도 max\|Δ\| < 1e-12** ③ `n_match`·`n_all` 이 S16 값과 **정수 일치**(프레임 마스크 `t>2Tp` 동일) ④ `zero_shell` 9키도 ①②를 만족 | **중단** — 채점 자·프레임 창이 배치 간에 달라졌다는 뜻이고, 그러면 S16 열과의 어떤 대조도 무효 | S16 `PREREG.md:250`(G-9) · W3 G-8 · S15 `SMOKE.md`(G-9 가 NEP-50 float32 로 1.1e-08 를 잡은 전례) |
| **G-17** ★ | **집계 정합 (aggregation parity)** | 대조에 들어가는 **모든 팔이 같은 규약**: 필터 시드 {42,43,44} **3/3 완비**, 팔×에피소드 값 = **median-of-3**(셸도 셸키별 median). 어느 팔이든 시드 결손 ⟹ 전 판정에 **`PROVISIONAL` 도장** | 결손 ⟹ `PROVISIONAL`(인용 금지), 규약 불일치 ⟹ **해당 대조 판정 금지** | S16 `analyze_s16.py:39-59,215-216` · S16 `SCRIPTS_md5.txt`(median-of-3 승격) |
| **G-18** | **퇴화 필요조건** | 각 신규 팔이 해당 층에서 `ZERO` 를 이기는 에피소드 **≥41/64**. 미달 팔은 **`DEGENERATE`** 로 표시 | 그 팔이 든 **우월성 문장 금지**(수치는 인쇄). ⚠`WHITEHS50` 은 스프레드를 3–5배 올리므로 **실제로 발화할 수 있다** | W3 `PREREG.md §4.3` · S16 `PREREG.md:257`(R-1) |
| **G-19** ★ | **요인 가법성(상호작용) 검사** | `log(1+r_S) + log(1+r_A)` 와 `log(1+r_{S+A})`(= S16 R-2 의 **+3.5%**) 의 차가 **≤ log(1.05)**. `r` 은 `med(rel)` | 초과 ⟹ **"3요인을 분리했다" 문장 금지**, `INTERACTION` 라벨을 붙이고 각 대조를 **개별 사실**로만 보고 | S16 `AMENDMENT_1` A-2(2요인 잔존) · `RESULT.md:199-201`(이월 1·2) |

---

## §G-등방성. **등방성 게이트는 백색 배경에서 자명 통과한다 — 정보가 0 이다** (요구된 판정)

**판정: `VACUOUS` — 이번 배치의 처치(`WHITE25`·`WHITEHS50`)에 대해 등방성 게이트는 위음성 게이트다.**

근거 (전부 코드/저장값):

1. **실패 모드가 백색 분기에 존재하지 않는다.** S16 B1 이 잡은 결함은
   `letkf_spec.py:200-203` 의 **jonswap 분기**에서 `rfft2` 반평면(`kx≥0`) × **고정 ω 부호**가
   모든 모드를 +x 로 전파시킨 것이다. 백색 분기는 `letkf_spec.py:192-194`
   `X = rng.standard_normal((n, n_state)) * amp` 한 줄이고 **ω 도 k 도 들어가지 않는다**
   ⟹ η 과 η̇ 가 **구성상 독립**이라 `corr(η̇, ∂η/∂x)` 는 **몬테카를로 0** 이다.
2. **저장값이 그렇다.** `isotropy_probe.json` 의 `WHITER50`:
   `corr_detadot_dx = −0.00041` · `corr_detadot_dy = −0.00080` · **`n_modes_nonzero = 0`**
   — 문턱 0.15 대비 **여유 180배**. S16 전량 1,920행에서도 max\|corr\| 0.0080/0.0072.
3. ⟹ 이 게이트는 **배치가 옳든 그르든 통과**한다. 저장소가 이미 대가를 치른 부류
   (`SUPPORT-TRADEOFF` 게이트가 λ=0 도 통과시켜 무효화된 사건)와 **같은 형태**다.

**처분 (동결)**

- **G-9(d) 는 남긴다 — 단 "코드경로 회귀 증인" 으로만 라벨한다.** 남기는 이유는 하나:
  누군가 백색 팔을 실수로 jonswap 분기로 라우팅하면(예: `background` 오타, `white_amp` 미전달로
  `Hs_bg/4` 폴백) **이 게이트만이 그것을 봄**. ⛔**"등방성 PASS" 를 수리 품질의 증거로 인용 금지.**
- **대체 게이트 = G-9(a)(b)(c) 3항** — 이것이 이번 배치의 **진짜 조작 검사**다:
  - **(a) 스프레드**가 `WHITEHS50` 의 **처치 그 자체**다(A 대조의 유일 요인). 배관이 안 되면
    `std(η)` 가 0.1 로 남는다 = 목표 대비 **−67%~−80%** ⟹ ±1% 문턱이 **확실히 잡는다**.
    (S16 실측 오차: 백색 0.16%, 스펙트럼 0.11% ⟹ 문턱 여유 6배, **원리적 통과 불가 아님**.)
  - **(b) 구조 부재**가 S 대조의 성립 조건이다. 백색 lag1 실측 **−0.0012 / +0.0002** vs
    스펙트럼 **0.834 / 0.815** ⟹ 문턱 0.05 는 **양쪽에서 16배 이상 떨어져 있다**.
  - **(c) `n_modes_nonzero == 0`** 은 (a)(b)를 못 속이는 독립 증인이다(클래스 분기 직접 관측).
- **추가 대체 = G-8 양성 대조.** 등방성 게이트가 S16 에서 실패한 진짜 이유는
  *"게이트가 자기가 인쇄하는 성질을 검사하지 못했다"*(`AMENDMENT_1` A-1: 기존 진단 4개가 **ω 부호에 불변**).
  G-8 은 **음성 3건 + 양성 1건**을 동시에 요구해 **게이트의 무의미성 자체를 검출**한다.

---

## §G-크기규율 (게이트가 만드는 서술 제약)

1. **G-11 이 2% 를 넘긴 팔** ⟹ 그 대조는 **부호만** 쓴다(크기 인용 금지).
2. **G-18 `DEGENERATE`** ⟹ 그 팔이 든 우월성 문장 금지.
3. **G-19 `INTERACTION`** ⟹ "3요인 분리" 금지.
4. **G-3 `UNMATCHED-CONTRAST`** ⟹ 그 대조는 **인쇄만** 되고 판정에 못 들어간다.
5. 크기는 언제나 **3규약 병기**(`med(rel)` · `rel(med)` · `rel(mean)`).
6. ⚠**A 대조의 스프레드는 `Hs` 오라클로 설정된다** — 허용 문장은
   *"참 `Hs` 로 스프레드를 맞춘 백색 배경"* 이고, **GT-free 대응물(`hs_obs4std`, 전 행 기록됨)은 미측정**이다.
   (원하면 `WHITEOBS50`(= `white_amp = hs_obs4std/4`) 을 **지금 팔로 추가**해야 한다 —
   결과를 본 뒤 추가는 금지(S16 `PREREG.md:326` §11-4 승계).)

---

## §F. 동결 사슬 — **순서와 커밋 파일 목록**

### F-1. 순서 (이 순서를 어기면 동결이 아니다)

```
① PREREG.md 작성 완료 (이 §G/§F/§I 포함)          ← 이후 본문 수정 금지
② PRERUN_STATE.json 생성                          ← 판정 행 0 증명 (F-2)
③ FROZEN_md5.txt      : PREREG.md 의 md5 + 동결시각(KST) + git HEAD + 성격 표기
④ SCRIPTS_md5.txt     : run_s17.py · letkf_spec.py(사본 or 참조) · analyze_s17.py ·
                        dryrun_analyzer.py · watch_s17.sh 의 md5 + 시각
⑤ git add -f tmp/S17_WHITE25_2026-08-18/ && git commit   ← ★런 전 커밋 (경로가 gitignore 대상)
⑥ 스모크 (G-7…G-12)  → smoke_results.jsonl (판정 파일과 분리)
⑦ 필요 시 AMENDMENT_1.md (PREREG 는 수정하지 않는다) → md5 를 ③에 추가 기재 → 재커밋
⑧ 전량 실행 → results.jsonl
⑨ analyze_s17.py → verdict_s17.json (분석기가 자기 md5 를 첫 줄에 인쇄)
```

- **성격 표기 의무**: `FROZEN_md5.txt` 에 ⛔*"preregistered"* 대신
  **`self-attested freeze + git-anchored (commit <sha>)`** 라고 적는다.
  근거: W3 `PREREG.md` 서두(`tmp/*/PREREG.md` **50개 중 `FROZEN_md5.txt` 7개**) ·
  W3 `NOTES.md`(20:47 `196050e` 로 처음 git-anchored 달성). **S17 은 그 수준을 승계한다.**
- ⚠**커밋은 사용자 승인 사항이다**(저장소 규율: 지시 없이 커밋/푸시 금지).
  승인 전에는 ⑤를 **"커밋 대기"** 로 표시하고, 동결 성격은 `self-attested` 로만 적는다.

### F-2. **판정 행이 0개임을 증명하는 방법** (`PRERUN_STATE.json`)

동결 시점에 다음을 **한 파일에 기록하고 ⑤ 커밋에 포함**한다:

| 항목 | 값 |
|---|---|
| `results_jsonl_exists` | `false` (`os.path.exists` 실측) |
| `results_jsonl_stat` | `null` (존재 시 `st_size`·`st_mtime`·행 수·md5 를 **전부** 적고, 0행이 아니면 **배치 중단**) |
| `smoke_jsonl_exists` | `false` / 또는 존재 시 행 수 + md5 |
| `dir_listing_sha256` | 배치 디렉터리 전체 `find -printf "%p %s\n" \| sort` 의 sha256 |
| `git_head`, `git_dirty_n`, `git_dirty_paths_in_pkg` | `wavesplat/**`·`wavesim/**`·`configs/**` 안의 더러운 경로 **목록** |
| `code_fingerprint` | 87파일 digest(2회 계산 동일) + **LETKF 임포트 경로 5파일 md5** |
| `s16_reuse_inventory` | S16 `results.jsonl` 의 md5 · 행 수 1920 · 팔별/시드별 카운트 |

**증명의 논리**: ⑤ 커밋의 `git show --stat` 에 `results.jsonl` 이 **없고**, `PRERUN_STATE.json` 이
"부재" 를 기록하며, 커밋 이후 생성된 파일은 `git status` 로 사후 식별 가능하다
⟹ **"동결 시점에 판정 자료 0행" 이 커밋 그래프에 고정된다.**
⚠**여전히 self-attested 다**(외부 타임스탬프 권위 없음) — 커밋은 *순서*를 고정할 뿐 *시각*을 증명하지 않는다.
이 한계를 `FROZEN_md5.txt` 에 그대로 적는다.

### F-3. 커밋에 포함할 파일 목록 (⑤ 시점)

```
tmp/S17_WHITE25_2026-08-18/
  PREREG.md                 FROZEN_md5.txt          SCRIPTS_md5.txt
  PRERUN_STATE.json         code_fingerprint.json
  run_s17.py                letkf_spec_s17.py(또는 S16 사본 + md5 기재)
  analyze_s17.py            dryrun_analyzer.py      watch_s17.sh
  NOTES.md                  audit/*.md (런 전 감사 산출물)
  PREREG_DRAFT_gates.md (이 파일 — 초안 이력)
⛔포함하지 않는다: results.jsonl · smoke_*.jsonl · provisional/ · *_read/
   (⑧ 이후 별도 커밋에서, "판정 자료" 라고 명시해 추가한다)
```

### F-4. 잠정 판독 격리 (S16 승계)

- Phase 중간 판독은 `provisional/` 에만 쓴다. **`analyze_s17.py` 는 그 디렉터리를 읽지 않는다**
  (G-12 가 입력 경로를 assert). 근거: S16 `NOTES.md:253,275,328`(`provisional/`·`phase1_read/`·`phase2a_read/`).
- 잠정 판독 문서 첫 줄에 **`⛔PROVISIONAL — 인용 금지`** 를 박는다.

---

## §I. 계측 사양 (게이트가 요구하는 것을 만드는 코드)

### I-1. **필드 계약** — `REQUIRED_LETKF` 승계 + 이번 배치 추가분

**신규 팔 공통 (768행 전부, `None`/`NaN` 금지)** — S16 `run_s16.py:76-81` 승계:

```
episode, arm, stratum, bf, ncomp, seed, wall_s,
avg_nRMSE, zero_nRMSE, n_match, n_all, ms_median, n_obs_median,
Hs, Tp, depth, hs_obs4std,
shell_rmse{obs: 9키}, zero_shell{obs: 9키}, shell_frac{obs: 9키},
effective_config{★객체 되읽기}, loc_radius, n_ensemble, background,
filter_class, filter_seed,
git_head, git_dirty_n, code_md5, code_files_n
```

**★S17 추가 (없으면 `SCHEMA-FAIL`)**

| 필드 | 왜 필요한가 | 게이트 |
|---|---|---|
| `background_diag{...}` | S16 은 이 키를 `REQUIRED` 에 **넣지 않았다**(`run_s16.py:76-81` 에 부재) — 그래서 `WHITE50` 384행이 null 인 채로 통과했다. 신규 2팔은 **spec 클래스이므로 비null 필수** | G-9·G-15 |
| `white_amp`, `eta_target` | **A 대조의 처치 값 그 자체**. 의도가 아니라 **객체에서** 읽는다 | G-3·G-9(a) |
| `thin_md5`, `n_thinned_frames` | 관측 thinning 지문(§I-3) | G-10 |
| `import_md5` | 임포트 경로 파일 md5 — `git_head` 는 **더러운 작업트리를 못 잡는다** | G-4 |
| `letkf_path_md5{5키}` | LETKF 임포트 경로 5파일의 개별 md5(배치 간 비교용) | G-4 |
| `batch`, `arm_intent`, `contrast_tags[]` | 행이 어느 대조에 속하는지 **행 자신이 말한다**(분석기의 사후 라벨 금지) | G-3·G-17 |
| `seed_role` | `filter_seed` 인지 추정기 시드인지 — S13 이 시드 축을 서술에서 뒤섞은 전례 | G-17 |
| `attempt`, `error`(있으면) | 재시도·에러 회계 | G-14 |

**수입 행 계약**: S16 1,920행은 위 목록 중 **S16 시점 계약분만** 요구한다(사후 소급 불가).
⟹ ★**`thin_md5` 는 S16 행에 없다** ⟹ G-10 의 배치 간 대조는 **스모크에서 재실행한 `WHITER50`/`SPEC50`
2 에피소드**로만 성립한다. 이 한계를 PREREG 에 적고, RESULT 에서 넘겨짚지 않는다.

### I-2. **누출 프로브 구현** (G-8)

```python
# 런 0회 아님 — 스모크 안에서 2 에피소드 × 4셀 (≈6분)
# 교란은 "필터 생성자에 들어가는 값" 에만 건다. 채점기(compute_all_metrics)의
# loader.Hs_true 는 절대 건드리지 않는다 (건드리면 nRMSE 정규화가 바뀌어 프로브가 무의미해진다).
CELLS = [  # (arm, hs_mult, tp_mult, expect)
    ('WHITE25',   1.37, 1.00, 'IDENTICAL'),   # 음성: 이 팔은 Hs 를 안 쓴다
    ('WHITE25',   1.00, 0.83, 'IDENTICAL'),   # 음성: Tp 도 안 쓴다
    ('WHITEHS50', 1.00, 0.83, 'IDENTICAL'),   # 음성: Tp 는 안 쓴다 (스프레드만 Hs 사용)
    ('WHITEHS50', 1.37, 1.00, 'DIFFERENT'),   # ★양성 대조: Hs 는 반드시 결과를 바꾼다
]
# 판정: IDENTICAL 셀은 avg_nRMSE 와 9셸 전부 |Δ| == 0.0 (비트),
#       DIFFERENT 셀은 |Δ(avg_nRMSE)| > 1e-6.
# 근거: letkf_spec.py:192-199 (백색 분기는 white_amp 만 쓴다),
#       run_s16.py:212-214 (spec 팔은 background 와 무관하게 Hs_bg/Tp_bg 를 받는다
#       => 값이 '전달되지만 쓰이지 않음' 을 실측으로 확인해야 한다).
```

⚠**이 프로브가 없으면**: `WHITE25` 는 생성자에서 `Hs_true`·`Tp_true` 를 **실제로 받는다**
(`run_s16.py:212-214` 경로를 그대로 쓰면). "안 쓴다" 는 **코드 읽기의 주장**일 뿐이고,
S16 B1 이 정확히 그 층위(코드가 선언과 다름)에서 죽었다. ⟹ **프로브가 필수다.**

### I-3. **thinning 지문 구현** (G-10)

```python
# LETKFGrid._thin_observations 를 얇게 감싼다 (패키지 무수정; run_s17.py 안에서 래핑).
# letkf_grid.py:116-121 -- M > max_obs 일 때만 np.random.choice 가 도는 것에 유의.
h = hashlib.md5()
def _thin(self, ox, oy, oz, _orig=LETKFGrid._thin_observations):
    M = len(ox); out = _orig(self, ox, oy, oz)
    if M > self.max_obs:
        # 선택된 x 좌표 자체를 지문에 넣는다(인덱스는 원본 순서에 의존)
        h.update(np.ascontiguousarray(out[0], dtype=np.float64).tobytes())
        _thin.n += 1
    return out
# 사전 예측(동결): spec 클래스 팔들은 전역 스트림을 thinning 만 소비하므로
#   thin_md5 가 전부 같아야 한다 (letkf_spec.py:142-153 이 __init__ 의 전역 draw 를 되돌리고
#   add_inflation 을 0 으로 둔다). 재고 WHITE50 은 letkf_grid.py:88 의 init draw + :164-165 의
#   매 프레임 randn 때문에 달라야 한다.
```

### I-4. **감시 스크립트 사양** (G-13) — 자기 매칭 금지

```bash
#!/usr/bin/env bash
# watch_s17.sh — 진행은 오직 "파일 상태" 로 판단한다.
# ⛔금지: pgrep -f run_s17.py  (감시자 자신의 명령줄이 매칭된다 — W3 가 이것으로 8시간을 잃었다)
set -u
D=/home/vexit/wavesplat/wavesplat/tmp/S17_WHITE25_2026-08-18
R="$D/results.jsonl"; EXPECT=768; STALL=900   # 15분 = 중앙 40.5 s/런 의 약 20배
prev=-1; last_change=$(date +%s)
while :; do
  n=$( [ -f "$R" ] && wc -l < "$R" || echo 0 )
  now=$(date +%s)
  [ "$n" -ne "$prev" ] && { prev=$n; last_change=$now; }
  echo "$(date +%F' '%T) rows=$n/$EXPECT stall=$((now-last_change))s" >> "$D/WATCH.log"
  # 완료는 두 조건의 AND (행 수만으로 끝났다고 말하지 않는다)
  if [ "$n" -ge "$EXPECT" ] && [ -f "$D/DONE.marker" ]; then
      echo "COMPLETE rows=$n" >> "$D/WATCH.log"; break; fi
  if [ $((now-last_change)) -gt "$STALL" ]; then
      echo "*** STALL ${STALL}s at rows=$n ***" >> "$D/WATCH.log"; fi
  sleep 120
done
```

- 러너는 **잡 완료마다 1행 append** 하고, 정상 종료 마지막 줄에서 `DONE.marker` 를 쓴다.
- 프로세스 생사 확인이 꼭 필요하면 러너가 남긴 `run.pid` 로 **`kill -0 $(cat run.pid)`** 만 쓴다
  (패턴 매칭 금지).
- 감시자는 **절대 `results.jsonl` 에 쓰지 않는다.** 로그는 `WATCH.log` 뿐이다.

---

## §G-검토. 게이트 자체의 자기 시험 (이 절도 사전등록에 포함한다)

| 위험 | 어느 게이트에 있나 | 처리 |
|---|---|---|
| **원리적 통과 불가**(S6 형) | G-9(a) ±1% · G-9(b) 0.05 · G-19 log(1.05) | 전부 **S16 실측값으로 여유 확인**: (a) 실측 오차 0.16% ⟹ 여유 6배, (b) 백색 실측 0.0012 ⟹ 여유 40배, (c) `n_modes_nonzero` 는 0/629 로 이산. G-19 만 **사전 실측이 없다** ⟹ 초과 시 **중단이 아니라 라벨**로 처리(설계상 죽일 수 없게 했다) |
| **위음성**(`SUPPORT-TRADEOFF` 형) | **G-9(d) 등방성** | **`VACUOUS` 로 선언**(§G-등방성). 정보 있는 대체 = G-9(a)(b)(c) + G-8 양성 대조 |
| **자기 무의미 검출 불가** | G-8 | **양성 대조 셀을 넣어** 게이트가 "아무것도 안 보는 상태" 를 스스로 검출 |
| **게이트가 판정 자료를 오염** | G-7·G-8·G-9·G-10·G-11 | 전부 `smoke_results.jsonl` 로 분리, 분석기는 `results.jsonl` 만 읽음(G-12 assert) |
| **하드 스톱의 대가** | G-7①·G-16·G-15 | 실패 시 **8.6 h 뒤 판정 거부**가 될 수 있음을 **사전 고지**. 그래도 하드로 두는 이유: 이 셋이 무너지면 **S16 열과의 대조 자체가 성립하지 않기** 때문(재사용 배치의 본질) |
| **소프트 게이트의 남용** | G-4① · G-11 · G-18 · G-19 | 전부 **문장 제약**으로 귀결시켜(§G-크기규율) "인쇄만 하고 넘어가는" 경로를 없앰 |

---

## §G-요약 (PREREG 표지에 넣을 한 문단)

> S17 은 **런 0회 게이트 6개(G-1…G-6) · 스모크 게이트 6개(G-7…G-12) · 런 중 2개(G-13·G-14) ·
> 런 후 5개(G-15…G-19) = 총 19개**로 봉인된다.
> 이번 배치 고유 게이트는 **G-3(상수·요인 정합) · G-4②(LETKF 임포트 경로 비트 동일) ·
> G-8(오라클 채널 교란 프로브, 음성 3 + **양성 1**) · G-9(a)(b)(c)(스프레드·구조부재·모드수 조작검사) ·
> G-10(thinning 지문) · G-13(행 수 기반 감시) · G-16(배치 간 제로필드·프레임 마스크 정합) ·
> G-17(집계 정합) · G-19(요인 가법성)** 이다.
> ⛔**등방성 게이트(G-9d)는 백색 배경에서 자명 통과 = 정보 0** 이므로 **코드경로 회귀 증인으로만**
> 남기고, 처치의 실질 조작 검사는 **스프레드(±1%)와 구조 부재(lag1 ≤ 0.05)** 가 맡는다.
