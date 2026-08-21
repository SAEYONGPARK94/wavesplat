# V-3 — 적대적 검증: A4 의 λ-오염 범위 주장 (MAJOR M-2 / M-3 / C-2)

**역할**: refuter. 목표는 A4 의 주장을 **무너뜨리는 것**. 불확실하면 반증 쪽.
**방식**: 런 0회. git 이력 · 원자료 `results.jsonl` · 파일 mtime · 코드 직독 + 파이썬 재도출.
**열람 범위**(주장 범위 = 이 목록):
`wavesplat/wavesplat/{run.py, optimizer/sequential_update.py}` · `wavesplat/configs/*.yaml` ·
`tmp/rx2_init_2026-08-05/run_rx2b.py` · `tmp/S{1,4,5,6,7,8,9,11,13}*/run_*.py` + `analyze_s6.py` ·
`tmp/BLANKSLATE_2026-08-12/{bs1_shipping_default.py, bs1_recompute.py, bs3_ship_rings.py}` ·
`tmp/DEBATE_blankslate_2026-08-12/{r2_auto_rest70.py, r2_phase_attrib.py, r2_p1_support_probe.py}` ·
`scripts/run_100ep_letkf_and_ws_metrics.py` ·
원자료 `results.jsonl` 5개(S5·S6·S7·S8·S9·S13) + `results/100ep_ws_letkf.json` +
`tmp/BLANKSLATE_2026-08-12/shipping_default.json` + `tmp/DEBATE_blankslate_2026-08-12/r2_phase_attrib.json` ·
git 커밋 `40e22f6` · `5f0846c` · `97e733a` · `8e3ca72` · `b987da4` · `9a36f93` · `8d2bba8`.

---

## 0. 기준 시각 — "배포"가 언제 일어났는가 (MEASURED)

| 사건 | 시각 | 근거 |
|---|---|---|
| `lambda_l1` 1e-3 → 1e-4 커밋 | **2026-08-11 17:19:03 +0900** | `git show 40e22f6` (`configs/estimation.yaml` `-  lambda_l1: 0.001` / `+  lambda_l1: 0.0001`, `run.py` DEFAULT_CONFIG, yaml 폴백) |
| `LAMBDA_L1_DEFAULT` 단일출처화 | 2026-08-12 00:30:33 | `git show 5f0846c` → `sequential_update.py:40 LAMBDA_L1_DEFAULT = 1e-4` |
| 배포 시점의 **하한** 증거 | S9 데이터 완료 15:35 → 커밋 17:19 | `run_s9.py` 자체 주석(5f0846c 에서 추가): *"이 배치가 돌 때 λ 를 명시하지 않은 셀들은 폴백에서 **1e-3** 을 받았고 … **우연히 옳았다**"* |

**현행 실효값 재도출 (파이썬 실행, MEASURED)**
```
LAMBDA_L1_DEFAULT                = 0.0001
DEFAULT_CONFIG['lambda_l1']      = 0.0001
DEFAULT_CONFIG['init_method']    = 'auto'
SequentialUpdater({}).lambda_l1  = 0.0001
SequentialUpdater({}).init_method= 'v2_konly'     ← DEFAULT_CONFIG 와 불일치
dict(DEFAULT_CONFIG) | {'init_method':'auto','analytic_ensemble_k':3} → lambda_l1 = 0.0001
```

★**기제 귀속 정정**: `run.py:139-141` 이 `cfg = DEFAULT_CONFIG.copy(); cfg.update(config)` 이므로,
`run_estimation` 경로를 타는 **모든** 배치 러너에서 λ 를 실제로 주입하는 곳은
`sequential_update.py:173` 의 폴백이 **아니라** `run.py:30 DEFAULT_CONFIG['lambda_l1']` 이다.
(두 값은 지금 같으므로 결론은 안 바뀌지만, "sequential_update 폴백 때문"이라는 서술은 부정확.)

---

## 1. 핵심 표 — 배치 × 실행시점 × 배포전후 × 팔별 λ 명시 × 판정

`results.jsonl` mtime = 배치 완료 시각. **배포 = 2026-08-11 17:19**.

| 배치 | 완료 시각 (mtime) | 배포 전/후 | 처치 팔 λ | 기준선 팔 λ | 원래 런 판정 | 재실행 판정 |
|---|---|---|---|---|---|---|
| `rerun_2026-07-24` (λ 스윕) | 2026-07-24 | **전** | `sweepjobs.py:39` 명시 | 축 기준값 0.001 명시 | ✅ 유효 | ⚠혼합 |
| `p1_penalty_2026-08-05` | 2026-08-05 15:05 | **전** | `run_p1.py:45` λ=0 명시 | `BASE` λ 미명시 | ✅ **유효** | ⛔ 무효 |
| `rx1_tracker_2026-08-05` | 2026-08-05 21:37 | **전** | `run_rx1.py:84,90,100,110` λ=0 | 기준선 미명시 | ✅ **유효** | ⛔ 무효 |
| **S1** `S1_indep_sparse` | 2026-08-08 | **전** | 없음 | 없음 | ✅ 유효 | ⚠균일이동(대조 보존) |
| **S2 / S3 / S4** | 2026-08-08 | **전** | 없음 | 없음 | ✅ 유효 | ⚠균일이동(대조 보존) |
| **S5** `S5_window_loss` | **2026-08-08 21:37:44** | **전** | `run_s5.py:30,36,37` λ=0.0 명시 | `AUTO-NT`/`DUP` **미명시** → 실효 **1e-3** | ✅ **유효** | ⛔ 대조 1e-3↔0 → 1e-4↔0 축소 |
| **S6** `S6_l1_sweep` | **2026-08-09 01:26:33** | **전** | `run_s6.py:26,29` 3e-4/1e-4/0 명시 | `AUTO`(EXTRA 없음)·`AUTO-NT`(trk만) **미명시** → 실효 **1e-3** | ✅ **유효** | ⛔ `AUTO-NT` ≡ `AUTO-NT-L1-1e4` 붕괴 |
| **S7** `S7_confirm_n32` | **2026-08-10 13:13:42** | **전** | `run_s7.py:70-72` 1e-4/3e-4 명시 | `AUTO`·`AUTO-NT`·`AUTO-NT-DUP` **미명시** → 실효 **1e-3** | ✅ **유효** | ⛔ `AUTO` ≡ `AUTO-L1-1e4` 붕괴 |
| **S8** `S8_deploy_confirm` | **2026-08-11 01:09:18** | **전** | `run_s8.py:67-71` 1e-4/3e-4/0 명시 | `AUTO`·`AUTO-DUP2`·`AUTO-SEED-NULL`·`AUTO-NT`·`AUTO-TRK-QTR` **미명시** → 실효 **1e-3** | ✅ **유효** | ⛔ 처치·널·복제 전부 붕괴 |
| **S9** `S9_lr_x_l1` | **2026-08-11 15:35:36** | **전 (커밋 1h44m 前)** | `AUTO-L1`·`AUTO-QTR-L1` 명시 | **런 당시 `AUTO`·`AUTO-NT`·`AUTO-QTR`·`AUTO-SEED-NULL`·`AUTO-DUP` 미명시** → 실효 1e-3 | ✅ **유효** | ✅ **깨끗** — `5f0846c` 가 **전 셀 λ 명시**로 수리 |
| **X2** `X2_no_learned` | 2026-08-11 | 전후 경계 | `run_x2.py:27` λ=1e-4 명시 | S9 저장분 재사용 | ⚠ 교차배치 | — |
| **S10** `S10_selector` | 2026-08-12 | 후 | 추정기 호출 **0회** | n/a | ✅ 무관 | ✅ 무관 |
| **S11** `S11_null_corrected` | 2026-08-12 (jsonl **없음**) | 후 | `run_s11.py:57-59` 3 셀 명시 | S9 저장분 재사용 | ✅ **무관(런 0회)** | ✅ 깨끗 |
| **BLANKSLATE** `bs3_ship_rings.py` | 2026-08-12 16:31 | **후** | `:92,94,96` 3 셀 전부 명시 | 동상 | ✅ **깨끗** | ✅ 깨끗 |
| **BLANKSLATE** `bs1_shipping_default.py` | **2026-08-12 18:28** | **후** | `:36 'shipping_default': {}` → 1e-4 | **`:37 'v2_konly_100ep'` λ 미명시 → 1e-4** | ⛔**오염** (아래 §5) | ⛔ |
| **DEBATE** `r2_auto_rest70.py` | 2026-08-12 | **후** | `config=None` → 1e-4 (λ 를 **기록은** 함, `:32,53`) | 비교 상대 = 2026-06-24 공개 컬럼 **λ=1e-3** | ⛔**교락** (아래 §5) | ⛔ |
| **DEBATE** `r2_phase_attrib.py` | 2026-08-12 | **후** | `:39 'auto_phaseTrue': {}` → 1e-4 | `:40 'v2k_phaseTrue'` λ 미명시 → 1e-4, **"2026-06-24 behaviour" 라 선언** | ⛔**오염** (아래 §5) | ⛔ |
| **DEBATE** `r2_p1_support_probe.py` | 2026-08-12 | 후 | `:190-191` 명시 | 동상 | ✅ 깨끗 | ✅ |
| **S13** `S13_oos` | 2026-08-13 01:19 | **후** | `SHIP_L1E3` 1e-3 명시 | `ship_config()=dict(DEFAULT_CONFIG)` → λ **실효값이 cfg 에 실려** 기록 | ✅ **깨끗** | ✅ 깨끗 |
| **AUDIT_full** | 2026-08-12 | 후 | 추정기 호출 0회 | n/a | ✅ 무관 | ✅ 무관 |

---

## 2. ★결정적 반증 — 원자료가 "기준선이 처치값을 안 받았다"를 **직접** 증명한다

이 하네스는 **비트 결정적**이다: 바이트 복제 셀(`AUTO-DUP`/`AUTO-DUP2`/`AUTO-NT-DUP`)은
짝 Δ 가 **정확히 0**이다. 따라서 *만약* 기준선이 처치 λ 를 받았다면 처치−기준선 짝 Δ 도
**정확히 0** 이어야 한다. 재도출 결과(원자료 `results.jsonl`, `nRMSE` 짝 차):

| 배치 | 쌍 | n | Δ≠0 개수 | max\|Δ\| | 판정 |
|---|---|---|---|---|---|
| S5 | `AUTO-NT` − `AUTO-NT-NOL1` | 96 | **96/96** | 1.124e-01 | λ 갈림 ⟹ 미오염 |
| S6 | `AUTO-NT` − `AUTO-NT-DUP` | 96 | **0/96** | 0.000e+00 | 결정성 기준선 |
| S6 | `AUTO-NT` − `AUTO-NT-L1-1e4` | 96 | **96/96** | 1.046e-01 | λ 갈림 ⟹ 미오염 |
| S6 | `AUTO` − `AUTO-NT-L1-1e4` | 96 | **96/96** | 1.280e-01 | (게다가 `tracking_lr` 도 다름) |
| S7 | `AUTO-NT` − `AUTO-NT-DUP` | 192 | **0/192** | 0.000e+00 | 결정성 기준선 |
| S7 | `AUTO` − `AUTO-L1-1e4` | 192 | **192/192** | 1.378e-01 | λ 갈림 ⟹ 미오염 |
| S8 | `AUTO` − `AUTO-DUP2` | 192 | **0/192** | 0.000e+00 | 결정성 기준선 |
| S8 | `AUTO` − `AUTO-L1-1e4` | 192 | **192/192** | 7.977e-02 | λ 갈림 ⟹ 미오염 |
| S9 | `AUTO` − `AUTO-DUP` | 288 | **0/288** | 0.000e+00 | 결정성 기준선 |
| S9 | `AUTO` − `AUTO-L1` | 288 | **288/288** | 1.409e-01 | λ 갈림 ⟹ 미오염 |

⟹ **S5·S6·S7·S8·S9 의 기록된 결과는 오염되지 않았다.** 이건 시각 대조(§0)와 **독립인** 두 번째 확인이다.

부가 확인(`results.jsonl` 의 `cfg` 필드): 처치 셀은 λ 가 **실제로 기록돼 있다**
(S6 `AUTO-NT-L1-1e4`→`0.0001`, `-3e4`→`0.0003`, `-0`→`0.0`; S7/S8/S9 동일).
미기록은 λ 를 명시하지 **않은** 셀에 한정된다.

---

## 3. (a) "뿌리 = `run_rx2b.py:61`, 상속 러너 **20개**, 전부 λ 미명시" → **REFUTED (3중으로)**

**(a-1) 개수가 틀렸다.** `rg -l "import run_rx2b" --glob '*.py'` = **12개**. (20 은 `rg -l BASE_CFG` 의 개수인데,
그 20 중 16개는 run_rx2b 를 **import 하지 않고 자기 `BASE_CFG` 를 복붙**했고, 2개는
`wiring_check.py`(자기 배치 러너의 `R.BASE_CFG` 참조), 2개는 **주석/print 문자열일 뿐**이며,
`run_rx2b.py` **자신**도 그 20에 포함돼 있다. 즉 "20" 은 두 개의 서로 다른 집합을 겹쳐 센 값이다.)

**(a-2) "전부 λ 미명시" 가 틀렸다.** 12개 중 `lambda_l1` 문자열이 **0회**인 것은
`S1/run_s1.py` · `S2/run_s2.py` · `S3/run_s3.py` · `S4/run_s4.py` **4개뿐**이다.
나머지 8개 — `S5` · `S6` · `S7` · `S8` · `S9/run_s9.py` · `S11/run_s11.py` ·
`BLANKSLATE/bs3_ship_rings.py` · `DEBATE/r2_p1_support_probe.py` — 는 **전부 λ 를 명시한다**.
A4 가 "20개 전부 0회 등장 (grep 실측)" 이라며 나열한 목록 자체에
`S5/run_s5.py`·`S6/run_s6.py` 가 빠져 있고(같은 문서 M-3 에서는 이 둘이 λ 를 명시한다고 쓴다),
`S1~S4` 만 실제로 조건을 만족한다 = **자기 문서 내부 모순**.

**(a-3) 뿌리가 틀렸다.** `run_rx2b.py` 는 커밋 `9a36f93` **2026-08-06 00:23:45** 생성.
그보다 앞서 `run_rx2.py`(`b987da4`, 08-06 00:15), `run_e2.py`(`53552f5`, 08-05 23:57),
`run_e1.py:65`(`8e3ca72`, **08-05 19:38**) 가 같은 `BASE_CFG` 를 갖고 있고,
`-S"BASE_CFG"` 로 추적한 **최초 도입은 `97e733a`(2026-07-23 18:28)** 의
`tmp/variance_verify_fable.py` 다. `run_rx2b.py:61` 은 뿌리가 아니라 **계보 중간의 한 사본**이다
(그 자신도 주석에 *"== tmp/s1s2_2026-08-05 arm S1-A0"* 이라 출처를 밝힌다).

**살아남는 부분**: `run_rx2b.py:61-64 BASE_CFG` 에 `lambda_l1` 이 없다는 사실 자체와,
`run_rx2b.py:108 'AUTO'` / `run_s4.py:43-44 'AUTO-NT'` 에도 없다는 사실은 **정확**하다.

---

## 4. (b) "λ-처치 배치는 S5·S6·S7·S8 넷" → **PARTIALLY-REFUTED**

- **넷이 아니다.** 같은 기준(한 팔은 λ 명시, 다른 팔은 미명시)을 적용하면
  **`rx1_tracker_2026-08-05`**(`run_rx1.py:84,90,100,110` λ=0 vs 미명시 기준선)과
  **`p1_penalty_2026-08-05`**(`run_p1.py:45 'P1-both'` vs `BASE`)도 같은 상태다.
  **최소 6개** — 그리고 **S9 도 런 당시엔 같은 상태였다**(기준선 5셀 미명시, `results.jsonl` 실측). 즉 7개.
- **"과소보고" 프레임이 절반만 맞다.** CLAUDE.md:63 이 말하는 것은 *"현재 이 상태 = **재실행** 무효"* 이고,
  그건 **미래형 위험**이다. A4 의 표는 이를 "⛔치명적 / ⚠오염" 으로 **현재형 판정처럼** 적어
  이미 기록된 S5–S8 결과가 손상된 것처럼 읽히게 한다. §1·§2 가 그건 사실이 아님을 보인다.
- **살아남는 부분**: S5·S6 을 명단에 넣어야 한다는 지적 자체는 **옳다**(재실행 관점에서).

---

## 5. (c) "S9·S11·S13·BLANKSLATE 는 깨끗" → **PARTIALLY-REFUTED (BLANKSLATE·DEBATE 에서 실패)**

**S13 = 깨끗 (확인).** `run_s13.py:53-55 ship_config()` 가 `dict(DEFAULT_CONFIG)` 를 **전개**하고,
`:167-168` 이 그 위에 arm override 를 얹고, `:181-182` 가 **`effective_config` 를 통째로 기록**한다.
원자료 재도출: `SHIP`/`SHIP_NOCONF`/`SHIP_NODISP` = `lambda_l1 0.0001`, `SHIP_L1E3` = `0.001`,
전 448행 `git_head = 5f0846cc`. `:109-116` G-1 게이트가 `SequentialUpdater(cfg).lambda_l1 == cfg['lambda_l1']` 를
**런 전에** 강제한다. ⟹ 이 배치는 실제로 이 문제의 **유일한 완전 해법**이다.

**S9 = 깨끗하되 A4 의 이유는 아니다.** A4 는 "`run_s9.py:78-96`, `AUTO` 포함 전 셀 명시" 라 쓴다.
그 코드는 **런이 끝난 뒤**(`5f0846c`, 08-12 00:30) 들어왔다. S9 `results.jsonl`(08-11 15:35) 을 재도출하면
`AUTO`·`AUTO-NT`·`AUTO-QTR`·`AUTO-SEED-NULL`·`AUTO-DUP` 의 `cfg` 에 `lambda_l1` 이 **없다**.
S9 가 유효한 진짜 이유는 **런이 배포 전이었기 때문**이고, 그건 `run_s9.py` 주석이 스스로 그렇게 말한다.

**S11 = 무관(A4 의 이유와 다름).** `S11_null_corrected_2026-08-12/` 에 `results.jsonl` 이 **없다**.
MEMORY 도 "`learned_init_iters` 축은 **런 전에 사망**" 이라 기록한다. λ 명시 여부와 무관하게 **런 0회**다.

**★BLANKSLATE = 깨끗하지 않다 (A4 가 놓친 실제 오염, 그리고 이건 배포 *후*다).**
```
tmp/BLANKSLATE_2026-08-12/bs1_shipping_default.py:37-39
    'v2_konly_100ep': {'init_method': 'v2_konly',  # what 100ep_ws_letkf.json ran
                       'use_dispersion': True, 'use_confidence': True,
                       'adaptive_n': True, 'tracking_window': 3, 'seed': 42},
```
`lambda_l1` 이 **없다** ⟹ `DEFAULT_CONFIG` 에서 **1e-4** 를 받는다.
그런데 이 팔이 재현한다고 **선언한** 원본은
`scripts/run_100ep_letkf_and_ws_metrics.py:39-46` 이고 거기에도 λ 가 없어
**당시 폴백 1e-3** 을 받았다(`results/100ep_ws_letkf.json` mtime **2026-06-24 00:28**, 배포 7주 전).
이 팔은 2026-08-12 **18:28** 에 돌았다 = 배포 **후**.
⟹ *"what 100ep_ws_letkf.json ran"* 이라는 주석은 **거짓**이다. λ 가 10배 다르다.

**★DEBATE_blankslate = 같은 결함이 헤드라인 경로에 있다.**
- `r2_phase_attrib.py:31` 은 `SU.TRACK_PHASE_PREDICT = True  # <-- the 2026-06-24 behaviour` 로
  7주 전 구성을 **복원했다고 선언**하는데, `:39-42` 두 팔 모두 λ 미명시 ⟹ **둘 다 1e-4**.
  즉 복원 대상(1e-3)과 **λ 가 다르다**. 재도출(공개 컬럼 30ep 대조, MEASURED):
  `v2k_phaseTrue` 중앙 **0.2372** vs 공개 **0.2335**, **비트일치 0/30**, max\|Δ\| **8.407e-02**,
  나쁨 14 / 좋음 16 → MEMORY 의 "16/30 널" 이 바로 이 대조다.
- `r2_auto_rest70.py:47` `run_estimation(p, config=None)` ⟹ `auto` + **1e-4**.
  비교 상대는 2026-06-24 공개 컬럼(**v2_konly + 1e-3**).
  ⟹ **"출하 auto 85/100"** 는 `init_method` 단독 대비가 아니라 **`init_method` × λ 2요인 묶음**이다.
  (변명 가능한 점: `:32,53` 이 `lambda_l1` 을 **결과에 기록은 한다** ⟹ 사후 식별 가능. 반면
  `bs1_shipping_default.py`·`r2_phase_attrib.py` 는 기록조차 안 한다.)
- `r2_p1_support_probe.py:190-191`, `bs3_ship_rings.py:92-96` 는 **명시 = 깨끗**.

**단, 널 재도출은 무사하다**: `bs1_recompute.py:235` 는 `results/100ep_ws_letkf.json` 을
**`json.load` 만** 한다(재실행 없음) ⟹ "감사 §1 널 51/49→50/50" 은 λ 와 무관.

**MEASURED vs ARGUED 구분**: λ 가 다르다는 것은 MEASURED. 위 재현 불일치(max\|Δ\| 0.084)의 **어느 정도가 λ 탓인지**는
`TRACK_PHASE_PREDICT` 외 7주치 커밋과 교락돼 **미분리 = ARGUED**.

---

## 6. (d) "`run_s6.py:31` 의 `AUTO`/`AUTO-NT` 가 `AUTO-NT-L1-1e4` 와 같은 셀이 된다" → **PARTIALLY-REFUTED**

- **`AUTO` 는 절대 같은 셀이 안 된다.** `AUTO` 는 EXTRA 가 없어 `tracking_lr` = 출하 기본 **0.005**,
  `AUTO-NT-L1-1e4` 는 `run_s6.py:29` 로 **`tracking_lr: 0.0`**. λ 와 무관하게 **다른 셀**이다.
  원자료도 96/96 Δ≠0, max\|Δ\| 1.280e-01. ⟹ 이 절반은 **REFUTED**.
- **`AUTO-NT` 는 맞다 — 단 미래형으로만.** `run_s4.py:43 'AUTO-NT': {'tracking_lr': 0.0}` 이므로
  배포 후 재실행하면 λ 가 양쪽 1e-4 로 같아져 **바이트 동일**해진다.
  하지만 **기록된 S6 데이터에서는 96/96 Δ≠0, max\|Δ\| 1.046e-01** 로 명백히 다른 셀이다
  (같은 파일의 `AUTO-NT` vs `AUTO-NT-DUP` 는 **0/96**, 즉 하네스는 비트 결정적).
  ⟹ **현재형 서술("된다")은 REFUTED, 조건부 서술("재실행하면 된다")은 SURVIVES.**

---

## 7. (e) "`analyze_s6.py:115` 가 λ 미기록 셀을 1e-3 으로 **잘못** 라벨링하고 게이트가 그 값으로 통과" → **REFUTED (기록된 데이터에 대해)**

```
tmp/S6_l1_sweep_2026-08-08/analyze_s6.py:115   lam[r['cell']].add(c.get('lambda_l1', 0.001))
tmp/S6_l1_sweep_2026-08-08/analyze_s6.py:118-120  want[BASE]=want[DUP]=want[SHIP]=1e-3
```
기본값 `0.001` 이 적용되는 셀은 정확히 `AUTO`·`AUTO-NT`·`AUTO-NT-DUP` 셋이고(재도출 확인),
**S6 런(2026-08-09 01:26)은 배포(08-11 17:19)보다 2일 앞선다** ⟹ 당시 실효 폴백 = **1e-3**.
즉 이 라벨은 **참**이고 `§4.2` 게이트는 **옳게** 통과했다. `want` 표와 실효값이 일치한다.
⟹ "잘못 라벨링" · "게이트가 스스로 속는다" 는 **이 배치·이 데이터에 대해 REFUTED**.

**살아남는 부분(잠재 위험)**: 이 분석기를 **오늘** 재실행하면 같은 라인이 `1e-4` 인 셀을 `1e-3` 이라 인쇄하고
`want` 와 맞아 **통과한다**. 즉 결함은 "지금 거짓말하고 있다"가 아니라 "미래에 거짓말할 준비가 돼 있다"다.
(E1 이 이미 같은 실패양식을 `S9/gate.py` 에서 잡았다 — "게이트가 실효값이 아니라 **의도표**를 인쇄".)

---

## 8. (f) `init_method` 3중 기본값의 도달성 → **PARTIALLY-REFUTED / 심각도 하향**

A4 인용 지점(현행 줄번호로 재확인): `run.py:43 'auto'` · `run.py:99 'v2'` · `sequential_update.py:181 'v2_konly'`.

| 지점 | 도달 조건 | 판정 |
|---|---|---|
| `run.py:43` DEFAULT_CONFIG `'auto'` | `run_estimation` 전 경로 | ✅ **항상 도달 = 실효 기본값** |
| `sequential_update.py:181` `'v2_konly'` | `SequentialUpdater(cfg)` 를 **직접** 생성하고 cfg 에 키가 없을 때 | ⚠**배치 러너 경로에서는 DEAD**. `run.py:139-141` 이 `DEFAULT_CONFIG.copy()` 를 먼저 깔기 때문에 `init_method` 는 **항상** 존재한다. 저장소 내 직접 생성처(`scripts/pilot_iterative_refine.py:146,287` · `hydrofoil/*` · 테스트)는 **전부 `init_method` 를 명시**하거나 상위에서 채워 넣는다(확인). 실측 도달 호출자 **0건** |
| `run.py:99` `'v2'` | `--config <yaml>` 로 로드한 yaml 에 `optimizer:` 절이 있고 그 안에 `init_method` 가 **없을** 때 | ⚠**저장소 내 도달 불가**. `configs/` 3개 중 `optimizer:` 절을 가진 건 `estimation.yaml` 뿐이고 `:24` 에 `init_method: auto` 가 **있다**. `default.yaml`·`server.yaml` 은 `optimizer:` 절 자체가 **없다**(wavesim 설정) |

⟹ **`init_method` 는 `lambda_l1` 과 같은 등급의 함정이 아니다.** λ 는 실제로 값이 바뀌었고 폴백이
그 변화를 실어 날랐지만, `init_method` 의 두 하위 폴백은 **현재 저장소에서 실행 경로에 도달하지 않는다**.
심각도 **MAJOR → MINOR(위생)** 로 하향 권고.

★**대신 진짜 불일치가 하나 있다**: `SequentialUpdater({}).init_method == 'v2_konly'` 인데
`DEFAULT_CONFIG['init_method'] == 'auto'` 다. 같은 "기본값" 이라는 이름의 서로 다른 두 값이
한 저장소에 공존한다 ⟹ λ 처럼 **단일출처화**(`INIT_METHOD_DEFAULT`) 하는 게 맞다.
그리고 `sequential_update.py:159` docstring 은 아직 **`lambda_l1: … (default 0.001)`** 이라 쓴다(A4 C-1, 유효).
덧붙여 `sequential_update.py:426` · `:967` 에 **λ=0.001 이 하드코딩**돼 있다(S8 `P1ONLY`/`TRKONLY` 프로브의
"Phase-1 을 배포전 값으로 고정" 용도) — 배포 후 이 상수의 의미가 "출하 기본값" 에서 "구 기본값" 으로
조용히 바뀌었다. **주석에 그 사실이 없다.**

---

## 9. 최종 판정

| 주장 | 판정 | 한 줄 이유 |
|---|---|---|
| **(a)** 뿌리 `run_rx2b.py:61`, 상속 러너 20개 전부 λ 미명시 | **REFUTED** | import 하는 러너는 **12개**이고 그중 **8개가 λ 를 명시**한다(0회는 S1–S4 넷뿐); 최초 `BASE_CFG` 는 `97e733a` 2026-07-23 `variance_verify_fable.py` 로 `run_rx2b` 보다 2주 앞선다 |
| **(b)** λ-처치 배치 = S5·S6·S7·S8 넷 | **PARTIALLY-REFUTED** | S5·S6 추가는 옳으나 **넷이 아니라 최소 7개**(rx1·p1_penalty·S9-런당시)이고, 넷 전부 **배포 전 실행 = 기록된 결과는 무손상** |
| **(c)** S9·S11·S13·BLANKSLATE 는 깨끗 | **PARTIALLY-REFUTED** | S13 만 설계로 깨끗; S9 는 **런 시각** 덕이지 코드 덕이 아니고(수리는 런 이후 `5f0846c`), S11 은 **런 0회**, **BLANKSLATE `bs1_shipping_default.py:37` 과 DEBATE `r2_phase_attrib.py:39-42`·`r2_auto_rest70.py:47` 은 배포 *후* λ=1e-4 로 돌면서 λ=1e-3 공개 컬럼을 재현했다고 선언 = 실제 오염** |
| **(d)** `AUTO`/`AUTO-NT` 가 `AUTO-NT-L1-1e4` 와 같은 셀 | **PARTIALLY-REFUTED** | `AUTO` 는 `tracking_lr` 0.005 vs 0.0 로 **λ 와 무관하게 다른 셀**; `AUTO-NT` 는 **재실행 시에만** 붕괴하고 기록된 S6 은 96/96 Δ≠0(복제 셀은 0/96) |
| **(e)** `analyze_s6.py:115` 가 잘못 라벨링, 게이트가 속는다 | **REFUTED** | S6 런(08-09)이 배포(08-11)보다 앞서므로 **1e-3 이 참값**이고 게이트는 옳게 통과했다 — 결함은 현재형이 아니라 **재실행 시 잠재** |
| **(f)** `init_method` 도 같은 3중 폴백 함정 | **PARTIALLY-REFUTED / 심각도 하향** | `run.py:99`(shipped yaml 전부가 `init_method` 를 갖거나 `optimizer:` 절이 없음)와 `sequential_update.py:181`(`run.py:139` 이 DEFAULT_CONFIG 를 먼저 병합) **둘 다 실행 경로에서 도달 불가**; 남는 실질 문제는 `'auto'` vs `'v2_konly'` **두 기본값의 공존** |

### 최종 오염 배치 목록 (3분류)

**① 이미 무효 (배포 후 실행 + 팔 간 λ 불일치·오라벨)**
- `tmp/BLANKSLATE_2026-08-12/bs1_shipping_default.py` — `v2_konly_100ep` 팔이 λ=1e-4 인데 "λ=1e-3 인 100ep 재현" 이라 주석 (2026-08-12 18:28)
- `tmp/DEBATE_blankslate_2026-08-12/r2_phase_attrib.py` — "2026-06-24 behaviour" 복원 선언인데 두 팔 다 λ=1e-4 (재현 실패 실측: 0/30 비트일치, max|Δ| 8.4e-02)
- `tmp/DEBATE_blankslate_2026-08-12/r2_auto_rest70.py` — `auto`+1e-4 를 `v2_konly`+1e-3 공개 컬럼과 대조 ⟹ **"출하 auto 85/100" 은 2요인 묶음** (λ 는 기록돼 있어 사후 분리 가능)

**② 재실행 시 무효 (원래 런은 배포 전이라 **유효**)**
- `S5_window_loss_2026-08-08` · `S6_l1_sweep_2026-08-08` · `S7_confirm_n32_2026-08-10` · `S8_deploy_confirm_2026-08-10`
- `rx1_tracker_2026-08-05` · `p1_penalty_2026-08-05` · `rerun_2026-07-24`
- (파생 분석기 동반 무효: `S6/analyze_s6.py:115`, `S7/analyze_s7.py`, `S8/analyze_s8.py` 의 λ 기본값 라벨)

**③ 깨끗**
- `S13_oos_2026-08-12` (설계로 깨끗 — `effective_config` 기록 + G-1 게이트)
- `S9_lr_x_l1_2026-08-11` (원래 런은 시각으로 깨끗, 러너는 `5f0846c` 로 수리돼 재실행도 깨끗)
- `BLANKSLATE/bs3_ship_rings.py` · `DEBATE/r2_p1_support_probe.py` (전 셀 λ 명시)
- `S1`~`S4` (어느 팔도 λ 를 안 건드림 ⟹ 재실행 시 **균일 이동**, 팔 간 대조 보존. 단 작동점은 이동)
- `S10_selector` · `S11_null_corrected` · `AUDIT_full` (추정기 호출 0회)

### `CLAUDE.md:63` 대체 문장 (1개)

```
  ⚠**기본값을 바꾸면 λ 미명시 팔은 조용히 새 값을 받는다** — 주입 지점은 `run.py:30`
  `DEFAULT_CONFIG['lambda_l1']`(= `sequential_update.py:40 LAMBDA_L1_DEFAULT`)이고,
  `run_estimation` 이 이를 먼저 병합하므로 러너가 명시하지 않으면 **기준선이 처치값을 받는다**.
  **배포 전(≤2026-08-11 17:19)에 실행된 배치는 무손상**(원자료로 확인: 처치−기준선 짝 Δ 가
  S5 96/96 · S6 96/96 · S7 192/192 · S8 192/192 · S9 288/288 에서 ≠0 인 반면 바이트 복제 셀은
  정확히 0) — 무효가 되는 것은 **재실행·확장뿐**이며 대상은 `S5·S6·S7·S8` + `rx1`·`p1_penalty`
  (파생 분석기 `analyze_s{6,7,8}.py` 의 λ 기본값 라벨 포함). `S9` 는 커밋 `5f0846c` 로 전 셀
  λ 명시 수리 완료. **본보기는 `S13`**: `ship_config()=dict(DEFAULT_CONFIG)` 전개 + 행마다
  `effective_config` 기록 + 런 전 G-1 게이트(`run_s13.py:109-116`). ⛔배포 후 실행된
  `BLANKSLATE/bs1_shipping_default.py:37`·`DEBATE/r2_phase_attrib.py:39-42`·`r2_auto_rest70.py:47`
  는 λ=1e-4 로 돌면서 λ=1e-3 인 2026-06-24 공개 컬럼을 재현/대조했다 = **λ 교락, 재현 주장 인용 금지**.
```
