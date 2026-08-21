# AUDIT-A — 기록·주장 감사 (2026-08-12)

**범위** = `/home/vexit/.claude/projects/-home-vexit-wavesplat/memory/` (116 파일) + 이를 매 세션
같이 싣는 `/home/vexit/wavesplat/CLAUDE.md`.
**방법** = 모든 판정을 원자료(`results.jsonl` 재도출 · `git show` · 실제 `.py`/`.yaml`)로 대조.
요약을 요약으로 확인한 항목은 **하나도 판정에 쓰지 않았다**(그런 것은 UNVERIFIED 로 내렸다).

- **BLOCKER 5건** (거짓으로 확인, 즉시 수정 필요)
- **WARN 8건** (과장·범위 초과·상충)
- **UNVERIFIED 6건**

★가장 중요한 구조적 발견: **오늘 λ 오류는 개별 실수가 아니라 전파 경로의 결함이다.**
`MEMORY.md` 는 오늘 **13:53** 에 갱신됐는데 거짓 문장(25행)이 살아남았고,
`reference_citation_bans.md` 는 **10:25** 에 멈춰 있어 **13:52 의 정정을 못 받았다.**
`MEMORY.md:34` 는 그 ban 파일을 *"전체 목록 … 주장을 쓰기 전에 반드시 열어볼 것"* 이라고
지정한다 ⟹ **규율을 그대로 따르는 미래 세션은 정정을 지나쳐 25행의 거짓으로 간다.**

---

# BLOCKER

## B1. `MEMORY.md:25` — "두 파일 함께 고쳐야 배포 **(미수정)**" = 거짓

| | |
|---|---|
| **주장** | `MEMORY.md:25` — *"⛔두 파일 함께 고쳐야 배포(**미수정**)"* |
| **원자료 반증** | `wavesplat/configs/estimation.yaml:12` = `lambda_l1: 0.0001` · `wavesplat/wavesplat/run.py:30` = `'lambda_l1': LAMBDA_L1_DEFAULT` · `wavesplat/wavesplat/optimizer/sequential_update.py:40` = `LAMBDA_L1_DEFAULT = 1e-4` · `run.py:94` yaml 로더 폴백이 `DEFAULT_CONFIG['lambda_l1']` 를 참조. `git show 40e22f6`(2026-08-11 17:19) 커밋 본문 = *"## 적용 (3곳 — 'yaml 한 줄'이 아니었다)"*. `git show 5f0846c`(08-12 00:30) 이 단일출처화 완료 |
| **올바른 값** | **λ=1e-4 는 2026-08-11 커밋 `40e22f6` 로 배포 완료.** 08-12 `5f0846c` 로 파이썬 쪽 중복 제거(단일 출처 `sequential_update.py:40` + yaml 미러). 잔존 = `scripts/` **9곳**이 아직 1e-3 |
| **악성도** | `MEMORY.md` 는 **오늘 13:53 에 수정됐는데도** 이 줄이 남았다. 같은 파일 46행(E1)이 *"배포 직후 기준선 5셀이 처치값을 받아…"* 라고 배포를 전제하므로 **인덱스가 자기모순** |

## B2. `project_s9_prereg_2026-08-11.md:31` — 거짓의 발원지가 **미정정**

| | |
|---|---|
| **주장** | 31행 *"⛔배포하려면 **두 파일을 함께** 고쳐야 한다(`estimation.yaml:12` + `run.py:26`). **미수정 상태.**"* |
| **원자료 반증** | B1 과 동일. 게다가 **같은 파일 33–38행(§0a)** 이 *"적용 후 실측 GIF 를 만들다 의심이 들어…"* 라며 **적용을 전제**한다 ⟹ 문서 내부 모순 |
| **올바른 값** | 31행은 **PREREG 시점(런 직후, 배포 전)의 상태**였고 같은 날 배포로 무효화됐다. 파일 mtime `2026-08-11T17:36` = 배포 커밋(17:19) 직후에 쓰였는데 갱신되지 않았다 |
| **주의** | `project_persona_debate_2026-08-12.md:59` 는 *"이 파일이 오류의 출처다"* 라고 적었으나 **그 출처 파일을 고치지 않았다** ⟹ 정정이 인용만 되고 소스에 안 내려갔다 |

## B3. `reference_citation_bans.md` 에 **λ 정정이 없다** — 규율이 정정을 우회시킨다

| | |
|---|---|
| **주장** | `MEMORY.md:34` — *"[전체 목록 → reference_citation_bans.md] — 누적 40여 건, **전부 유효**. 주장을 쓰기 전에 **반드시 열어볼 것**"* |
| **원자료 반증** | `grep -n "미배포" memory/*.md` ⟹ 유일한 정정은 `project_b1_precondition_debate_2026-08-12.md:70` 한 곳뿐. `reference_citation_bans.md` mtime = **2026-08-12T10:25**, 정정 발생 = **13:52**(`project_persona_debate_2026-08-12.md` mtime) ⟹ **ban 파일은 오늘의 최대 오류를 못 받았다** |
| **올바른 값** | ban 파일 08-12 절에 추가 필요: ⛔**"λ 미배포"**(배포됨, `40e22f6`) · ⛔**bank-MMSE "~2배"**(B4) · 그리고 `run.py:26` 위치 표기 갱신(W1) |
| **악성도** | *"주장 전 반드시 열어볼 것"* 이 가리키는 파일이 **가장 늦게 갱신되는 파일**이다. 오늘 오류의 무저항 전파 경로가 정확히 이것 |

## B4. `MEMORY.md:95` — bank-MMSE **"~2배"** 를 nowcast 로 **면허**한다 (철회된 수치)

| | |
|---|---|
| **주장** | `MEMORY.md:95` *"⚠"~2배"는 **nowcast 기준**"* · 동일 문장 `project_bankmmse_port_2026-07-29.md:41` |
| **원자료 반증** | `tmp/bankmmse_port_2026-07-29/m3_field.json` 을 직접 열어 재도출: bank nowcast **0.4982 / 0.6657 / 0.7180**. v2_konly = 0.858 / 0.927 / 1.048 (`tmp/confirm2_2026-07-27/real_gif_info.json` 계열). 비 = **1.723 / 1.392 / 1.460** ⟹ 중앙 **1.46배**, 최대 1.72배, **2배 도달 창 0개**. 07-29 문서 자신의 표(0.489/0.649/0.723)로 재도출해도 1.754/1.428/1.449 = **여전히 0개** |
| **올바른 값** | **중앙 1.46배 (창별 1.72/1.39/1.46). "2배" 는 어떤 지표·어떤 프로토콜에서도 성립하지 않는다.** |
| **악성도** | 캐비엇이 *"nowcast 기준이면 ~2배는 괜찮다"* 로 읽힌다 — **철회가 일어난 바로 그 지표를 면허하고 있다**. 코드는 이미 고쳐졌다: `sequential_update.py:459-470` 과 `run.py:48-52` 가 둘 다 *"Do not cite 2x"* 를 명시. **메모리만 안 고쳐졌고 ban 파일에도 없다** |

## B5. `CLAUDE.md` — 매 세션 로드되는 **지시 파일**이 출하 기본값을 틀리게 적고 금지 수치를 싣는다

`CLAUDE.md` 는 *"IMPORTANT: These instructions OVERRIDE any default behavior"* 로 실린다.
헤더가 `## Current State (**2026-04-07**)` = **4개월 전 동결**.

| 줄 | 주장 | 원자료 | 올바른 값 |
|---|---|---|---|
| 53 | *"`init_method: v2_konly` — beamforming for kx/ky"* 를 **Important Design Decision** 으로 | `run.py:41` `'init_method': 'auto'` · `configs/estimation.yaml:24` `init_method: auto` · `MEMORY.md:104` (07-26 사용자 결정 flip) | **출하 기본값은 `auto`** (라우터 th=8, `sequential_update.py:696`; guard 은퇴 `:753`). ★**λ 와 완전히 같은 유형의 오류** — 지시 파일이 잘못된 배포 상태를 단언 |
| 49 | *"Speed: WS 13.4ms, ETKF 132.6ms (tuned, Ne=100)"* | `reference_citation_bans.md:21` 이 ⛔"132.6 ms 가 ETKF 비용" · ⛔"~10× 빠르다" 를 **금지** | **인용 금지 수치가 지시 파일에 살아 있다.** (같은 수치가 `paper/wavesplat.tex:53,104,761,783` 에도 남아 있으나 그쪽은 편집 정지 중 = 기지) |
| 46 | *"**57 tests** all passing (39 wavesim + 18 wavesplat)"* | `python3 -m pytest wavesim/tests/ wavesplat/tests/ --collect-only -q` ⟹ **196 collected** | **196** (S9·5f0846c 커밋 본문도 "196 테스트 통과") |
| 17/19/21 | *"scripts/ 11개"* · *"results/ 7 JSON"* · *"data/ 100 episodes"* | `ls` 실측 | **413 / 96 / 110** |
| 48 | *"Key result: WS avg nRMSE 0.223 vs ETKF 0.319 vs GPR 0.276"* | `paper/wavesplat.tex:1890` 의 값과 동일 계열. 07-24 논문 표 재실행에서 **주장 3건이 뒤집혔고**(`MEMORY.md:106`) 08-04 이후 전파 정지 | 출처 표기 없이 헤드라인으로 실려 있음 ⟹ 최소한 **"07-24 이후 미재검증"** 표시 필요 |

---

# WARN

**W1. ban 의 λ 위치 표기가 낡았다.** `reference_citation_bans.md:38`·`:40` — *"기본값은
`configs/estimation.yaml:12` **와** `wavesplat/run.py:26` 두 곳"*. 실제 `run.py:26` 은 **주석**이고
값은 `run.py:30` 의 **import**(`LAMBDA_L1_DEFAULT`), 진짜 정의는 `sequential_update.py:40`.
⟹ 허용 문장 *"두 파일에 중복 기재"* 의 **전제가 08-12 `5f0846c` 로 해소**됐다.
현재 정확한 서술 = **"파이썬 단일 출처(`sequential_update.py:40`) + yaml 미러(`estimation.yaml:12`)
+ `scripts/` 9곳 stale 1e-3"**.

**W2. ban 의 `scripts/` 개수가 과소.** `:38` *"scripts/ 7곳 하드코딩"* → 실측 **9곳/8파일**:
`run_noise_ablation.py:92` · `plot_beamforming_3d.py:203` · `generate_presentation_videos.py:68` ·
`gp_disp_pilot.py:551` · `evaluate_prediction.py:43` · `run_extended_analysis.py:105` ·
`pilot_iterative_refine.py:144`·`:285` · `generate_tracking_animation.py:36`.
`project_e1_effect_size_2026-08-11.md:49` 의 **"9곳"이 옳다** ⟹ ban 파일이 E1 정정을 못 받았다.

**W3. `MEMORY.md:34` "누적 40여 건"** → 실측 **⛔ 마커 75개 / 불릿 20줄**. 약 2배 과소.
주의 예산을 "40"으로 잡은 세션은 훑고 지나간다.

**W4. 시드 잔여 하한이 문서 간 상충.** `MEMORY.md:24` *"시드 잔여 **[1584–1818]**"* vs
`project_s11_null_corrected_2026-08-12.md:56` *"**시드 1581–1583 미사용** 4중 확인"*.
전수 스캔(`tmp/*/results*.jsonl` 20개 파일의 `seed` 필드): **실제 사용 최대 = 1580**(S9),
1581–1583 은 **어디에도 없다**. 두 규약(사전등록에 이름이 오르면 소각 vs 실행돼야 소각)이
같은 날 기록에 공존 ⟹ 하나로 못 박을 것.

**W5. `MEMORY.md:41` 의 "희소 53/64" 는 이득이 아니라 적중률이다.**
원자료 재도출(S9 `results.jsonl`, 널 = `AUTO` vs `AUTO-SEED-NULL`):
- *"`all_resid` 규칙이 nRMSE 가 더 낮은 팔을 골랐는가"* = **희소 53/64 · 혼잡 22/32** ⟹ **정확히 재현** ✅
- 그러나 *"규칙의 짝 이득"* = **26 개선 / 12 악화 / 26 동점, 중앙 상대 0.00%**
⟹ **53/64 는 처치의 41/14 와 같은 통계량이 아니다.** 이미 금지된
*"−38.4% vs −33.0%"*(다른 통계량 비교)와 **같은 오류 유형**이므로 병기 의무를 ban 에 추가 권장.

**W6. `MEMORY.md:27` "λ 단독 널(21/32)" 은 혼잡 전용.** 원자료 재도출(S7):
`AUTO-NT-L1-1e4 − AUTO-NT` = **혼잡 21/32 p=0.1102 (−2.75%)** · **희소 22/32 p=0.0501 (−21.74%)**.
21/32 만 읽으면 희소를 같은 판정으로 오귀속한다. (`S7 RESULT.md` 원문과 **전건 일치** ✅)

**W7. 같은 문장의 정정이 두 파일에서 다르다.** `reference_citation_bans.md:24`
*"'교차 = 20–30 m'(→ **20–50 m 안**)"* vs `MEMORY.md:63` *"**~30 m 안만** 이득"*.
E-D 원자료 재도출(864런, `AUTO − AUTO-NOTRACK` 셸별 짝 중앙):
`0_5` −0.1417(48/48) · `5_10` −0.0812 · `10_20` −0.0428 · `20_30` **−0.0059(31/48)** ·
`30_50` **+0.0177(12/48)** · `50_75` +0.0403 · `100_150` +0.0577.
⟹ 교차는 **20–30 셸과 30–50 셸 사이**. `MEMORY.md` 쪽이 더 정확하고 ban 쪽 대체문이 느슨하다.
(면적 분수 재도출: 30 m 안 = **16.0%**, 밖 = **84.0%** ⟹ "밖 84%" ✅ 정확)

**W8. bank-MMSE 3창 수치가 두 프로토콜로 병존, 미조정.**
`project_bankmmse_port_2026-07-29.md:32-34` 표 = 0.489/0.649/0.723(1 s 리드 e2e) vs
`sequential_update.py:459` 주석 = 0.498/0.666/0.718(`m3_field`). 어느 쪽도 2배를 못 만들지만
(B4), **어느 것이 인용 표준인지 메모리에 없다.**

---

# UNVERIFIED (확인 불가 — 이유 명시)

**U1. 시드 상한 "1818".** 디스크에 시드 원장 파일이 없다(`find -iname "*seed*ledger*"` 등 0건).
하한만 반증 가능했다(W4). 상한 근거는 **미확인**.

**U2. "초기화 51–58s→0.7–1.3s"**(`MEMORY.md:95`).
v2_konly 쪽 **57.7 s 는 확인**(`tmp/confirm2_2026-07-27/real_gif_info.json` `init_seconds: 57.7`).
bank 쪽 0.7–1.3 s 는 원자료에서 **못 찾음** — `m3_field.json` 에는 `seconds_total`
3.4 / 2.1 / 21.2 만 있고 초기화 분해가 없다.

**U3. `reference_bergamasco_stereo_dataset.md:30` "LJ01 Hs 10.0 / Tp 13.3".**
`project_b3_regime_position_2026-08-12.md:56` 이 스스로 *"쓰기 전에 확인할 것"* 이라 적었다.
나는 `.nc` 원본을 열지 않았다(수 GB, 이 감사 범위 밖). B-3 원자료(`b3_result.json`)에서
확인된 것은 기하뿐: LJ Grid2 valid_box **161.50 × 183.50 m**, LJ Grid1 **40.0 × 40.0 m**.
**Hs 는 여전히 미확인 — 그대로 열려 있다.**

**U4. S11 §2 의 "PRO Fisher 단측 p=0.0641".** `null_corrected.json` 에는
`p_switch = 0.0031314`(내 재도출과 일치 ✅)만 있고 **보정 후 Fisher 값이 저장돼 있지 않다.**
`AUDIT_PRO.md` 서술에만 존재 ⟹ 원자료 재도출 불가.

**U5. 외부 문헌에 기대는 ban 들.** Katende arXiv 2606.07153 · Sprunck 2510.27663 ·
Qi et al. Wave Motion 77:195–213 · Barthelme&Utschick TSP 2021 · Gomes&Selman AIJ 126:43–62.
**네트워크를 쓰지 않았다** ⟹ 검증 안 함. (Qi 는 이미 `project_b1_…:71` 에서 자체 플래그됨)

**U6. 2026-08-06 이전 ban 대부분.** `reference_citation_bans.md:14-25`(07월~08-05 계열)은
개별 원자료 재도출을 하지 않았다. **"문제없음"이라고 말하지 않는다 — 안 봤다.**

---

# ✅ 원자료로 재도출해 **일치**를 확인한 것 (반대 증거로 남김)

메모리의 수치 자체는 대체로 매우 정확했다. 실패는 **인덱스·ban·CLAUDE.md 에 집중**돼 있다.

| 주장 | 출처 | 내 독립 재도출 |
|---|---|---|
| S9 PRIMARY 희소 **50/64, −33.04%, p 7.07e-06** | `MEMORY.md:25` | `50/64, −33.040%, p=7.0695e-06` ✅ |
| S9 혼잡 **27/32 −9.86%** · 널 **33/64 p=0.90 / 18/32 p=0.60** | 〃 | `27/32 −9.857%` · `33/64 p=0.9007` · `18/32 p=0.5966` ✅ |
| S9 `all_resid` **62/64 −9.17% · 29/32 −11.94%** | `project_s9:18` | `62/64 −9.169%` · `29/32 −11.936%` ✅ |
| S8 혼잡 **27/32 −13.50%**, 희소 **22/32 p=0.0501** | `MEMORY.md:26` | 동일 ✅ |
| S8 **9셸 전부 개선·악화 셸 0개** | 〃 | 18셀(2레짐×9셸) 전부 음, 개선 21–29/32 ✅ |
| S7 λ 단독 **21/32 p=0.11018** · 묶음 **28/32** · 배포단일 **26/32 −11.69% / 23/32 −29.31%** | `S7 RESULT.md` | 전건 일치 ✅ |
| S7 게이트 바닥 p **4.66e-10** | `MEMORY.md:52` | `AUTO-NT-DUP` 0/32 = `4.657e-10` ✅ |
| E1 3표본 표 (−11.69/−13.50/−9.86 · −29.31/−16.78/−33.04) | `project_e1:19-20` | 전건 일치 ✅ |
| E-D 셸 `0_5` **−0.1417** · `30_50` **+0.0177** · `100_150` **+0.0577** · 밖 **84%** | `MEMORY.md:63` | 전건 일치 (면적 84.0%) ✅ |
| K6 `ORACLE−ORACLE-K` **−0.0533, 14/14** · PRIMARY **p=0.79** | `MEMORY.md:69` | `−0.053288, 14/14, p=1.22e-4` · `p=0.79053` ✅ |
| B-1 `ORACLE−AUTO` **−0.0640, 14/14, p=1.22e-4** | `project_b1:44` | `−0.063974, 14/14, p=1.22e-4` ✅ |
| S3 decorr 분해 **0.5205/0.0382 · 0.5268/0.0095 · 0.4829/0.0045 · 0.0005/0.0001**, ρ≈**0.72 / 0.9997** | `project_s3:33-38` | 에피소드-중앙 규약으로 **전건 일치** ✅ |
| S11 짝맞춤 희소 **41/14, p=3.55e-04, −35.80%** | `MEMORY.md:5` | `null_corrected.json` `C_matched` 와 일치 ✅ |
| S11 "독립확인 p=0.0031" | `project_s11:26` | `p_switch = 0.0031314` ✅ |
| S10 널 적중 **희소 53/64 · 혼잡 22/32** | `MEMORY.md:41` | 전건 일치(단 W5 참조) ✅ |
| B-3 개구 AUC **0.596/0.612/0.598/0.667/0.746/0.823/0.843** | `MEMORY.md:38` | 전건 일치 ✅ |
| B-3 기제 "희소 불변(9.07→9.53) / 혼잡만 상승(9.37→15.36)" | 〃 | 전건 일치 ✅ |
| B-3 실측 최대 **162×183 m**, 필요 195 m | `MEMORY.md:18` | `b3_result.json` `valid_box [161.50, 183.50]`, `need [195,195]` ✅ |
| B-1 `thesis_main.tex` **3,992줄 · CRB 0회 · Fisher 0회 · 개구 0회** | `project_b1:78` | `wc -l` 3992, grep 전부 0 ✅ |
| B-1 `d72_platform_aperture.py:50` = **L=60 m·seed=1** | `project_b1:24,46` | `def realistic_sea(..., L=60.0, ..., seed=1)` ✅ |
| B-1 `direction_deg` 불일치 (정확도 0 / 제어 180) | `project_b1:62-63` | `wave_generator.py:138 direction_deg=0.0`(합성 기본) vs `d3_e2e_stream_ff.py:103`·`d3_d17_ctrl_closedloop.py:197`·`d3_e2e_velff.py:48` = `180.0` ✅ |
| B-1 `pipeline.py:142 ship_x=0.0` 하드코딩 | `project_persona:66` | 정확히 그 줄 ✅ |
| "11,000런 → 실제 **16,327**" | `project_persona:60` | `tmp/*/results*.jsonl` 전 행 합 = **16327** ✅ |
| "56 s 에 1100 m" → **633 m** 정정 | `project_persona_debate_2026-07-26.md:56` | 11.3×56 = 632.8 ✅ 정정 반영됨 |
| 하드웨어 = AMD 7800X3D + RTX 4080 Super | `MEMORY.md:137` | `/proc/cpuinfo`·`nvidia-smi` ✅ ⟹ ban 의 "Intel i7-12700H 금지" **여전히 옳다** |
| 배포 노브 전반 (`auto` th=8 · guard 은퇴 · `bank_mmse` · `analytic_ensemble_k=3` · `TRACK_PHASE_PREDICT=False` · `tracking_mode=w_phi` · `adaptive_n=true`) | 여러 곳 | `sequential_update.py:696/753/83`, `run.py:41/54/60`, `estimation.yaml` 전부 ✅ |
| `learned_init_iters` 가 yaml 화이트리스트·`DEFAULT_CONFIG` 에 **없다** | `project_s11:29` | `sequential_update.py:642` 의 `config.get` 뿐, 두 곳에 부재 ✅ |
| `obs_resid_rms_oos` 가 `run.py:308` 에 **계산되나 미집계** | `MEMORY.md:41` | `run.py:308` 정확, 계산은 `update()` 이전(`:253-256`) ✅ |
| `initializer.py:593` = "오라클은 (kx,ky)만" | `MEMORY.md:59` | docstring 정확 ✅ |

---

# 감사 범위 선언 (열람 범위 = 주장 범위)

## 실제로 연 것
- **메모리 전문 읽음(9)**: `MEMORY.md`(141행 전체) · `reference_citation_bans.md` ·
  `project_s11_null_corrected_2026-08-12.md` · `project_s10_selector_2026-08-12.md` ·
  `project_s9_prereg_2026-08-11.md` · `project_e1_effect_size_2026-08-11.md` ·
  `project_b1_precondition_debate_2026-08-12.md` · `project_b3_regime_position_2026-08-12.md` ·
  `project_persona_debate_2026-08-12.md` · `project_s3_place_x_track_2026-08-08.md` ·
  `project_bankmmse_port_2026-07-29.md` · `reference_bergamasco_stereo_dataset.md` ·
  `reference_agreed_operating_point.md` · `server_environment.md`
- **grep 로만 본 것(전 116파일)**: `미배포|배포|lambda_l1|2배` 등 표적 검색.
- **원자료 재도출**: S9(2016행) · S8(1920) · S7(1152) · S3(480) · E-D(864) · K6(384) ·
  S11 `null_corrected.json` · S10 `DECISION.md` + S9 원자료로 규칙 재구성 · B-3
  `b3_result.json`/`aperture_curve.json` · `m3_field.json` · `real_gif_info.json` ·
  `audit_neutral_rederived.json`. **모두 내가 새로 짠 코드로 계산**(채점기 재사용 안 함).
- **코드/설정**: `configs/{default,estimation,server}.yaml` 전문 · `wavesplat/run.py`(1–100, 240–320) ·
  `optimizer/sequential_update.py`(40, 83, 440–500, 642, 696, 753, 938–955) ·
  `optimizer/initializer.py:585-600` · `wavesim/wave_generator.py:130-170,325-340` ·
  `wavesim/pipeline.py:138-146,54,257` · `wavesim/lidar_gpu.py:15-22` ·
  `scripts/d72_platform_aperture.py:45-55` · `scripts/d3_*.py` 3개 · `thesis/thesis_main.tex`(계수만) ·
  `paper/wavesplat.tex`(grep) · `/home/vexit/wavesplat/CLAUDE.md` 전문.
- **git**: `log --oneline -40` · `show 40e22f6`(전문+diff) · `show 5f0846c --stat` ·
  `log -S"lambda_l1"` · `log -S"LAMBDA_L1_DEFAULT"`.
- **실행**: `pytest --collect-only`(196) · `wc -l` 전 `results*.jsonl`(16327) ·
  `/proc/cpuinfo` · `nvidia-smi`.

## **안 본 것 — 여기에 대해 "문제없음"이라고 말하지 않는다**
1. **project 메모리 100여 개를 전문 읽지 않았다.** 07-22 이전 파일은 **한 줄도 안 열었다**.
   `direction3_gridfree_prior.md`(43 KB) · `project_waveprior_promotion_2026-07-24.md`(73 KB) ·
   `project_novelty_phd_verdict_2026-07-15.md`(36 KB) 등 대형 문서 **미열람**.
2. **`reference_citation_bans.md:14-25`(08-06 이전 ban ~40건)을 개별 원자료로 재도출하지 않았다.**
   "잘못 금지된 것"을 찾는 임무의 이 부분은 **미완**이다. 확인한 잘못-금지는 **0건이지만
   그건 안 봤기 때문일 수 있다.**
3. **`.nc` 실측 원본과 `.h5` 에피소드를 열지 않았다** ⟹ Hs·Tp·수심 등 데이터셋 파라미터 전부 미검증.
4. **`paper/wavesplat.tex` 1,500행을 감사하지 않았다**(grep 만) — 편집 정지 대상이라 우선순위 밖.
5. **네트워크 미사용** ⟹ 선점 문헌 5건 전부 미확인(U5).
6. **feedback_*.md 12개 중 3개만 열었다.**
7. `wavesim/` 15개 파일 중 **4개만** 열었다(`wave_generator`·`pipeline`·`lidar_gpu` + 목록).
   오늘 메인 세션이 물린 자리이므로 명시한다 — **나도 전수는 안 봤다.**

## 권고 (우선순위)
1. `MEMORY.md:25` 를 **"배포 완료(`40e22f6`, 3곳→08-12 단일출처)"** 로 교체. (B1)
2. `project_s9_prereg_2026-08-11.md:31` 에 정정 블록 삽입. (B2)
3. `reference_citation_bans.md` 에 **08-12 절 신설**: λ 미배포 · bank "~2배" · `run.py:26` 위치 ·
   `scripts/` 9곳 · W5 병기 의무. (B3·B4·W1·W2·W5)
4. `MEMORY.md:95` 와 `project_bankmmse_port_2026-07-29.md:41` 의 "~2배" 캐비엇 삭제 → **"중앙 1.46배"**. (B4)
5. **`CLAUDE.md` 전면 갱신** — 특히 53행 `init_method` 와 49행 금지 수치. (B5)
6. 시드 규약 한 줄로 못 박기(사전등록 명시 = 소각 여부). (W4)
