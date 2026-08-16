# AUDIT — `SUPPORT-TRADEOFF` 재현의 **팔 정체** 전수 확인 (런 0회)

**날짜** 2026-08-15 · **범위** 아래 §0 에 명시 · **추정기/LiDAR 호출 0** · 저장소 무수정(이 파일 외).
**목적** — 2026-08-14 토론에서 P4 가 `ARGUED` 로 남긴 **주장 A** 를 원자료로 판정한다:

> *"기존 `SUPPORT-TRADEOFF` 7회 재현은 전부 WaveSplat 내부 처치 대조였고 method-vs-baseline 은
> 한 번도 없었다. 따라서 정확한 명제는 '우리 노브를 돌리면 근거리를 팔아 먼바다를 산다' 이지
> '우리 방법의 이득이 먼바다에 있다' 가 아니다."*

---

## 0. 열람 범위 = 주장 범위

**전수 grep**(`SUPPORT-TRADEOFF|SUPPORT_TRADEOFF`) 를 `memory/**` · 저장소 `**/*.{md,py,json}` 에 돌려
언급 배치를 모았고, 임무가 준 후보 목록(`Rx6·Rx7·E-D·S11·S7·K6·E-B·B-1`)을 **그대로 믿지 않고**
2026-08-12 선행 재계수(`tmp/DEBATE_blankslate_2026-08-12/R2_P1_metrology.md:206-233`)의 8 사례표와
대조했다. 실제로 연 파일:

- 러너/셀 정의: `tmp/ED_lead_shell_2026-08-07/run_ed.py`(B3·Rx4·K6·Rx5·Rx6·Rx7·E-sweep 셀이 **한 파일에 누적**),
  `tmp/S11_null_corrected_2026-08-12/run_s11.py`, `scripts/d74_closed_loop_trajectory.py`,
  `scripts/run_b1_full_comparison.py`, `tmp/S15_DIRECTION_2026-08-14/run_s15.py`
- 채점기: `analyze_{eb,ed,rx7,s7}.py`, `S11/{null_corrected,pro_rederive}.py`, `Rx6/audit_pro_mech.py`
- 원자료: 각 배치 `results*.jsonl` 의 **팔 카운트 재집계**(아래 수치는 전부 재도출),
  `results/{b1_full_comparison,etkf_comparison,100ep_ws_letkf,gpr_comparison,three_contributions}.json`
- ⛔`tmp/S16_BASELINE_2026-08-14/` 는 실행 중 — `run_s16.py` 에 **grep 1회**만.

**안 연 것**(따라서 주장 안 함): S1–S6·S8·S9·S10 의 셸 표, `results/direction2/3` 전체, 논문 tex.

---

## 1. 판정표 — (배치 × 두 팔 × 분류 × 셸 기록 팔 × 근거)

| # | 배치 | 팔 A | 팔 B | **분류** | 셸/고리 기록 팔 | 근거 `파일:줄` |
|---|---|---|---|---|---|---|
| 1 | **Rx-6** 08-07 | `AUTO` | `AUTO-NOTRACK` | **`TREATMENT`** | 둘 다 WaveSplat. ⚠단 이 배치는 **셸 없음** — `avg_RMSE_unobserved` **단일 영역**만 덤프(관측영역은 미덤프, 그 배치 스스로 기록) | `run_ed.py:360-362`(RX6_CELLS) · `Rx6/audit_pro_mech.py:14,76,105` · `results.jsonl` 13 셀 × 96행 = **전부 WaveSplat** · `memory/project_rx6_result_2026-08-07.md:158,163` |
| 2 | **Rx-7** 08-07 | `AUTO` | `AUTO-NOTRACK` | **`TREATMENT`** | 둘 다(같은 채점기) | `analyze_rx7.py:120-122` (`avg_RMSE_observed`/`_unobserved`, 판정량 = `AUTO − AUTO-NOTRACK`) · `results.jsonl` 6 셀 전부 WaveSplat |
| 3 | **E-B** 08-07 | `AUTO` | `AUTO-NOTRACK` | **`TREATMENT`** | 둘 다 | `analyze_eb.py:171-181` (`rmse_by_radius`) · `results_lane{1,2}.jsonl` 6 셀 전부 WaveSplat |
| 4 | **E-C** 08-07 | **팔 안의 `oos−in` gap** | (대비 아님) | **재현으로 셀 수 없음** | `AUTO` 한 팔 | `results.jsonl` 6 셀 전부 WaveSplat · 선행 재계수 `R2_P1_metrology.md:212` 와 일치 |
| 5 | **E-D** 08-07 | `AUTO` | `AUTO-NOTRACK` | **`TREATMENT`** | 둘 다(9셸 + `shell_frac`) | `analyze_ed.py:143-152` · `run_ed.py:377`(RX7_CELLS 재사용) |
| 6 | **K6** 08-07 | `ORACLE`·`ORACLE-K`·`K6-NSCHED`·`K6-KLR*` | `AUTO` | **`ORACLE` + `TREATMENT`** — 그리고 ★**셸/근원 주장 자체가 없다** | 해당 없음 | `run_ed.py:196-197,300-301` · `K6/results.jsonl` 8 셀 전부 WaveSplat · `memory/project_k6_result_2026-08-07.md` 에 `SUPPORT-TRADEOFF`·셸 언급 **0건** ⟹ **후보 목록의 오등재** |
| 7 | **S7** 08-10 | `AUTO-NT-L1-1e4` (추적기 off **+** λ=1e-4, 노브 2개) | `AUTO` (출하 기본값) | **`TREATMENT`** | 둘 다(9셸) | `analyze_s7.py:39-40` (`PRIMARY` P2) · `results.jsonl` 6 셀 전부 WaveSplat |
| 8 | **S11** 08-12 | `AUTO-NT` | `AUTO-L1` | **`TREATMENT`** | 둘 다(9셸) | `run_s11.py:57-63`(NEW_CELLS 는 `R.CELLS['AUTO']` 복제) · `pro_rederive.py:155-156,203` · `memory/project_s11_null_corrected_2026-08-12.md:39-43` |
| 9 | **B-1** 08-12 | 궤적 정책 `CEM`/`greedy`/`learned` | `stationary`/`straight-cross` | **`TREATMENT`(궤적 노브) + `ORACLE`(양 팔 다 진짜 파수)** | 둘 다 — ⚠**다른 추정기**(`relax_step`+`tnse`, `wavesim` 임포트 0) · **다른 자**(경로적분 20 m) | `scripts/d74_closed_loop_trajectory.py:231`(*"use TRUE wavenumbers (oracle ID)"*)·`:245`(POLICIES)·`:332` · `B1/EVIDENCE.md:7-15` |
| 10 | **BLANKSLATE R2** 08-12 | `SHIP` | `NT` | **`TREATMENT`** | 둘 다 | `tmp/DEBATE_blankslate_2026-08-12/r2_p1_support_probe.py:1` · `VERDICT_R2.md:172` |

### 1.1 집계

- **`METHOD-VS-BASELINE` = 0 / 10.** 열 개 후보 어디에도 ETKF·LETKF·GPR·제로필드 팔이 **한 행도 없다**
  (각 `results*.jsonl` 의 `cell`/`arm` 카운트를 재집계했고, 전 셀이 `init_method='auto'` 계열).
- 실제 **`TREATMENT` 대비는 7건**(#1·2·3·5·7·8·10 + #9 는 다른 추정기라 별도),
  **셀 수 없는 것 2건**(#4 다른 양 · #6 셸 주장 부재), **`ORACLE` 개입 2건**(#6·#9).
- 요인으로 세면 **#2·3·5·7·8·10 이 전부 `tracking_lr==0`(추적기 off) 하나**다.
  S7 이 같은 배치 안에서 **요인을 분해해 놓았다** — 추적기를 **켠 채** λ 만 바꾼 `AUTO-L1-1e4` 는
  **혼잡·희소 9셸 전부 개선**(혼잡 `0_5` 28/32 −11.3%, 희소 `0_5` 25/32 −26.8%;
  `S7/posthoc_single_shells.txt:5-9,18-22`) ⟹ **서명을 만드는 것은 λ 가 아니라 추적기 off**.

---

## 2. 대칭 감사 — method-vs-baseline 에서 정말 한 번도 안 쟀는가

### 2.1 기준선이 셸을 **안 기록한** 배치 (재도출)

| 배치 | 팔 | 행 수 | `shell_rmse` 보유 행 |
|---|---|---|---|
| `S13_oos_2026-08-12` | `SHIP` / `SHIP_NOCONF` / `SHIP_NODISP` | 128 / 128 / 128 | **128 / 128 / 128** |
| | `LETKF` / `LETKF_PLUS` | 128 / 128 | **0 / 0** |
| `HOSGATE_2026-08-13` | `SHIP` | 64 | **64** |
| | `LETKF_PLUS` / `ZERO` | 64 / 64 | **0 / 0** |

⟹ S13·HOSGATE 에서 method-vs-baseline 셸 비교는 **원리적으로 불가**였다(`S-2a` 결함, 이미 기록됨).

### 2.2 ★그러나 기준선 셸/고리를 **실제로 기록한 자료가 3건 있다** (아는 1건 + **신규 2건**)

#### (a) 이미 아는 1건 — P4 의 HOSGATE `LETKF_PLUS` 재실행
`tmp/DEBATE_novelty_2026-08-14/p4_shell_rerun/{p4_letkf_shells.py, p4_shells.out}`.
`LIN` 혼잡 16ep, `SHIP` 이 **9셸 전부 승**, `obs:0_5` **15/16 −41.6% p=0.0005**, 가장 얇은 곳이 **먼바다**
(`150_inf` 11/16 **p=0.21 널**). `P4_skeptic.md:275-283`. ⟹ **부호가 내부 대조와 반대.**

#### (b) ★신규 — `tmp/S15_DIRECTION_2026-08-14/smoke_results.jsonl` (**이 감사에서 처음 집계**)
S15 러너는 `shell_rmse`(obs 9구간)와 `ship_ring_rmse`(ship 7구간)를 **전 팔 동일 헬퍼**로 낸다
(`run_s15.py:17-18`), 게이트 **G-8 필드 존재 PASS · G-8b 셸 정의 `worst|Δ|=0.000e+00` PASS**
(`SMOKE.md:26-28`). 원자료 재집계: `LETKF_PLUS` **4/4 행**, `ZERO` **4/4 행**이 셸·고리를 **보유**한다.

> ⚠**라벨: 탐색적.** 스모크 2짝(4 에피소드), `SHIP` 은 추정 시드 3개 중 1개와 짝, 사전등록 §10 이
> **판정에서 제외**한다고 명시. **확증 아님, 배포 문장 금지.**

`SHIP − LETKF_PLUS` 상대차(음수 = SHIP 승):

| 짝 | `obs:0_5` | `obs:5_10` | `obs:10_20` | `obs:30_50` | `obs:150_inf` | `ship:0_25` | `ship:25_50` |
|---|---|---|---|---|---|---|---|
| dir0 · 혼잡 15comp | −27.2% | −24.4% | −30.7% | −35.7% | −30.1% | −52.4% | −37.5% |
| dir0 · 희소 2comp | −96.9% | −97.8% | −98.1% | −97.9% | −96.6% | −97.0% | −98.2% |
| **dir180 · 혼잡 15comp** | **+7.7%** | **+3.3%** | −6.7% | −21.7% | −25.6% | −43.4% | **+12.9%** |
| dir180 · 희소 2comp | −96.2% | −97.1% | −97.5% | −97.6% | −95.4% | −94.5% | −97.0% |

⟹ **4짝 중 3짝에서 근거리 포함 전 구간 SHIP 승** = P4 의 HOSGATE 결과와 같은 방향, **독립 경로**.
★**단 dir180 혼잡 1건은 `obs:0_5`·`obs:5_10`·`ship:25_50` 에서 SHIP 이 진다** —
**method-vs-baseline 근거리 손실이 원자료에 존재하는 최초 사례**다. **n=1 에피소드, 탐색적.**

#### (c) ★신규 — `results/b1_full_comparison.json` (**저장소에서 가장 오래된 method-vs-baseline 공간분해**)
`WaveSplat` · `ETKF` · `GPR-1` **세 팔 전부**가 `avg_RMSE_far`(x ≥ 80 m, `metrics.py:95-122`) 와
`avg_RMSE_unobserved`(최근접 관측점 ≥ 10 m, `metrics.py:124-158`) 를 들고 있다. 100 에피소드 재집계:

| 지표 | WS 중앙 | ETKF 중앙 | WS<ETKF | 중앙 rel |
|---|---|---|---|---|
| `avg_nRMSE` | 0.2294 | 0.3104 | **84/100** | −27.7% |
| `avg_RMSE_far` | 0.1949 | 0.3700 | **84/100** | −28.0% |
| `avg_RMSE_unobserved` | 0.1932 | 0.3692 | **84/100** | −28.4% |

`ncomp≤5`(n=60): 세 지표 전부 **60/60**, −42.8 / −43.7 / −43.8%.
`ncomp≥15`(n=40): 세 지표 전부 **24/40**, −2.1 / −0.7 / −2.5%.
`results/etkf_comparison.json`(n=12) 도 같다 — `avg_nRMSE` 8/12 −23.9%, `avg_RMSE_far` 8/12 −21.3%.

⟹ **부호·승수가 전역과 구별되지 않는다 = 거래 서명 없음.** ★★★**그러나 그 이유가 결함이다** ↓

> ★**`RMSE_near` 는 계산됐는데 디스크에 안 쓰였다.**
> `wavesplat/evaluation/metrics.py:433` 이 `metrics['RMSE_near']` 를 채우는데,
> `scripts/run_b1_full_comparison.py:51-54` 의 `_metrics_keys()` 가
> `['nRMSE','RMSE_z','Hs_error','Variance_Ratio','Pearson_rho','Skill_Score','RMSE_far','RMSE_unobserved','Directional_R2']`
> — **`RMSE_near` 만 빠져 있다.** 그리고 남은 두 지표는 **거의 격자 전체**다
> (`unobserved` 는 d≥10 m; BLANKSLATE 가 이 코퍼스에서 ρ=0 면적을 **94.4%** 로 실측).
> ⟹ **이 파일에서 `SUPPORT-TRADEOFF` 는 설계상 검출 불가**였다. "거래 없음" 은 **부재증명이 아니다.**
>
> ⚠추가 결함 2건: ① **프레임 불일치** — WS 평균 **167** 프레임 vs ETKF·GPR **301**
> (이 감사에서 재도출, `P3_committee.md:27` 과 일치) ⟹ 이 파일 전체가 헤드라인 자격 없음.
> ② WS 팔이 **`v2_konly`**(`run_b1_full_comparison.py:6`) = **출하 `auto` 아님**.
> ⟹ **(c) 는 `ARGUED` 이하. 인용 금지, "근거리 미측정"이라는 사실만 인용 가능.**

`results/100ep_ws_letkf.json`(현 헤드라인 파일)에는 공간 분해가 **전혀 없다**
(키 = `nRMSE, Hs_err, VR, rho, SS, Dir_R2, ms, N`). `gpr_comparison`·`three_contributions`·
`p1_comparison`·`multiseed_comparison` 도 없다.

### 2.3 진행 중 배치
`tmp/S16_BASELINE_2026-08-14/run_s16.py:78,82,270,362` — `shell_rmse` + **`zero_shell`** 을 필수 필드로
검사한다 ⟹ 최소한 제로필드 셸은 팔별로 기록된다. **더 안 열었다**(실행 중).

---

## 3. P4 주장 A 판정 = **`CONFIRMED`** (그리고 P4 가 자기에게 과소평가했다)

**(A-1) "재현은 전부 내부 처치 대조였다" — `CONFIRMED`, 원자료.**
후보 10건 전부에서 두 팔이 WaveSplat 계열이다. `results*.jsonl` 의 `cell`/`arm` 열에 외부 기준선이
**단 한 행도 없다**(§1). 게다가 7건 중 6건이 **같은 노브 하나**(`tracking_lr=0`)이고, 나머지 하나(S7)는
그 노브를 **포함한 묶음**이다.

**(A-2) 정정 — "7회" 자체가 과대계수다.**
`K6`(#6)는 셸/근원 주장이 **아예 없다**(임무가 준 후보 목록의 오등재).
`E-C`(#4)는 팔 간 대비가 아니라 **한 팔 안의 gap**. `B-1`(#9)은 **다른 추정기·다른 자·다른 격자**.
⟹ 정직한 수는 **"단일 요인(추적기 off) × 단일 자(최근접 관측점 거리) × 6회, 독립 처치 0건"**
(2026-08-12 재계수와 일치, `R2_P1_metrology.md:229-233`). `CLAUDE.md:121-122` 의 서수 금지는 **유지**.

**(A-3) 대칭 감사 — P4 의 결론 방향은 오히려 강화된다.**
method-vs-baseline 자료 **3건**(HOSGATE 재실행 · S15 스모크 · b1/etkf `far`+`unobserved`)이
**전부 거래 서명 없음** 쪽이다. 단 셋 다 결정적이지 않다:
(a) 탐색적 n=16 · (b) 스모크 n=4, **1건은 반대 부호** · (c) **근거리를 아예 안 잼**.
⟹ **"method-vs-baseline 에서 거래가 없다" 도 아직 `ARGUED`.** 확정에는 S16/S15 전량이 필요하다.

**⛔이 감사가 말하지 않는 것**: "출하 시스템에 근거리 손상이 없다"가 아니다.
말하는 것은 **"근거리 손상은 지금까지 *우리 노브를 끄는 행위*에 붙어 있었고, 기준선 대비로는
겨우 세 자료에서만 재봤으며 그중 하나에서 부호가 반대로 나온다"** 이다.

---

## 4. 규율 처방 — `CLAUDE.md` 붙여넣기 문장

`CLAUDE.md:120-124`(`- **전역 승리는 근/원 셸로 분해할 것** …` 항목)을 아래로 **교체**:

```markdown
- **전역 승리는 근/원 셸로 분해할 것** (`SUPPORT-TRADEOFF`).
  ⛔**재현 횟수를 서수로 세지 마라** — "6번째/7번째"가 문서마다 되감겼고, 원본 판정은
  "단일 요인(`tracking_lr==0`) × 단일 자 × 6회, **독립 처치 0건**" 이다.
  ⛔**`SUPPORT-TRADEOFF` 를 "우리 방법의 이득은 먼바다에 있다" 로 읽지 마라**
  (2026-08-15 전수 감사, `tmp/DEBATE_novelty_2026-08-14/AUDIT_tradeoff_axis.md`):
  **후보 10건 전부가 WaveSplat 내부 처치 대조이고 외부 기준선 행이 0개다**
  (`Rx6·Rx7·E-B·E-D·S7·S11·BLANKSLATE` = 추적기 on/off · `K6`·`B-1` = 오라클 · `E-C` = 팔 안 gap).
  허용 = **"우리 노브(주로 추적기 off)를 돌리면 근거리를 팔아 먼바다를 산다"**.
  ⚠**기준선 대비 자료는 3건뿐이고 전부 미확정**: HOSGATE `LETKF_PLUS` 재실행(9셸 전부 승, n=16 탐색적) ·
  S15 스모크(`smoke_results.jsonl`, 4짝 중 **1짝은 `obs:0_5` +7.7% 로 SHIP 이 진다**) ·
  `results/b1_full_comparison.json`(`far`·`unobserved` 만 있고 **`RMSE_near` 는
  `scripts/run_b1_full_comparison.py:51-54` 가 버려서** 근거리가 **설계상 검출 불가**).
  ⚠**셸(`obs:`)과 배-거리 고리(`ship:`)는 다른 양이다** — 같은 표본에서 1.8–4.1배 갈린다.
  어느 쪽을 쟀는지 **접두어까지 적을 것**(S13 이 여기서 사전등록을 이탈했다).
  ★**팔별 필드 존재를 PREREG 게이트로 검사할 것** — S13·HOSGATE 가 기준선 셸을 0행 기록해
  `S-2a` 를 두 번 못 쟀다. 본보기 = `tmp/S15_DIRECTION_2026-08-14` 의 **G-8/G-8b**.
```

`reference_citation_bans.md` 에 **1줄 추가** 권고:

```markdown
- **(08-15 축 감사)** ⛔**`SUPPORT-TRADEOFF` 를 method-vs-baseline 명제로** → 후보 10건 전수에서
  외부 기준선 행 **0개**(`tmp/DEBATE_novelty_2026-08-14/AUDIT_tradeoff_axis.md` §1); 허용 = "내부 노브의 성질" ·
  ⛔**`K6` 를 `SUPPORT-TRADEOFF` 재현으로** → 그 배치에 셸/근원 주장이 **없다** ·
  ⛔**`results/b1_full_comparison.json` 의 `far`/`unobserved` 동부호를 "거래 없음"의 증거로** →
  `RMSE_near` 가 저장 안 됐고(`run_b1_full_comparison.py:51-54`) 두 지표가 격자 ~94% 를 덮는다
  ⟹ **검출 불가**; 게다가 프레임 불일치(WS 167 vs ETKF 301) + 팔이 `v2_konly`.
```

---

## 5. 예상 밖에 나온 것

1. ★★★**`RMSE_near` 는 46일 내내 계산되고 있었는데 한 번도 저장된 적이 없다.**
   `metrics.py:433` 이 채우고 `run_b1_full_comparison.py:51-54` 가 버린다. 저장소가 "근거리를
   기준선 대비로 못 쟀다" 고 믿어온 이유가 **모델이나 기준선의 한계가 아니라 키 목록 한 줄**이다.
   ⟹ 이 축의 최소 비용 수리는 **새 배치가 아니라 `_metrics_keys()` 에 `'RMSE_near'` 추가**다.
2. ★★**`S-2a` 결함의 수리본이 이미 존재하고 게이트까지 통과했다** — `S15` 의 G-8/G-8b.
   기준선 셸이 **원자료에 있다**(`smoke_results.jsonl`, `LETKF_PLUS`·`ZERO` 각 4행).
   08-14 정정문이 *"배치 자체는 여전히 기준선 셸을 기록하지 않는다"* 고 쓴 것은
   HOSGATE 에 대해선 참이나 **S15 에 대해선 이미 거짓**이었다.
3. ★**method-vs-baseline 근거리 패배가 원자료에 한 건 존재한다** — S15 스모크 dir180 혼잡,
   `obs:0_5` **+7.7%** · `obs:5_10` +3.3% · `ship:25_50` +12.9%, 더 먼 구간은 전부 SHIP 승.
   P4 의 F-0("HOSGATE 에는 거래가 없다")과 **모순이 아니라 조건부**다 — HOSGATE 는 `direction_deg=0`,
   이 패배는 **`direction_deg=180` 에서만** 나온다. ⟹ 08-12 토론이 최상위로 올린 `direction_deg` 축이
   **정확도 축과 거래 축을 동시에 움직인다**는 첫 신호. **n=1, 탐색적, 확정 인용 금지.**
4. ⚠**임무가 준 후보 목록에 `K6` 가 잘못 들어 있었다** — 그 배치는 셸을 아예 안 쟀다.
   목록의 출처는 `reference_citation_bans.md:74` 의 K6 관련 별개 금지항(오라클-k 3요인 묶음)으로 보인다.
5. ⚠**Rx-6 도 셸 배치가 아니다** — `avg_RMSE_unobserved` **한 영역만** 덤프했고,
   그 배치 자신이 *"다음 드라이버는 관측영역 RMSE 도 반드시 덤프할 것"* 이라 적어놨다
   (`memory/project_rx6_result_2026-08-07.md:163`). 즉 "2.12배 집중" 은 **두 영역 대비가 아니다.**
