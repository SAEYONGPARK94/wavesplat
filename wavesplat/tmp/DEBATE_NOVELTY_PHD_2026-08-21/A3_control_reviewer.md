# A3 — 선박운동·제어(seakeeping / motion prediction / control) 심사위원

**2026-08-21 · 추정 런 0회 · 파일 수정 0건(이 문서 제외).**
질문: ***이 연구는 "제어에 쓸 수 있는 조우 예보"인가?***

---

## 0-A. 판정 요약 (맨 위)

| # | 판정 | 근거 등급 |
|---|---|---|
| **A3-1** | ★★★**"제어-호환 기둥이 죽었다"는 현재 원장(ledger)이 틀렸다 — 단 우리에게 유리한 쪽으로.** 저장소에는 **사전등록·동결·집행된 폐루프 제어 게이트가 두 개**(`D-17` FAIL, `D-18` **PASS**) 있고, `D-18` 은 **24/24 전승**인데 **`MEMORY.md`·`STATUS_2026-08-21`·08-14·08-19·08-21 토론 어디에도 없다.** | `MEASURED` (원자료 재집계) |
| **A3-2** | ⛔**S2 판정("η 가 잘 나오는 것과 제어 입력이 나오는 것은 다르다")은 뒤집히지 않았다.** W4·W5·`B`·`C` **네 배치 전부 `alpha` 문자열 0회** ⟹ 반증 시도 자체가 없었다. **판정은 유지되고, 사유는 "재시험 후 생존"이 아니라 "미재시험"이다.** | `MEASURED` |
| **A3-3** | ★★★**S2 의 널이 약한 진짜 이유는 추정기가 아니라 계측 지점이다.** 제어 프리뷰 점 8개가 **전부(8/8) 정지 코퍼스의 최근접 LiDAR 반환(10.0 m)보다 배에 가깝고**, 최원단도 **0.10 λ_p** 다 ⟹ **α_wave 는 센서 사각지대 안에서 채점되고 있다.** | `MEASURED` |
| **A3-4** | ★★★**`τ = 0.5 s` 프리뷰는 완벽한 파고장을 줘도 값이 거의 없다.** `D-17`: GT 천장(참 파고·완전 프리뷰) − persistence(참 파고·프리뷰 0) = **히브 4.4 pp**, 가속도는 **−3.35 pp(GT 가 진다)**. ⟹ **저장소는 이길 것이 없는 리드 대역에서 제어 축을 재 왔다.** | `MEASURED` |
| **A3-5** | ★★★**`C` 의 `13/16 −17.6%` 는 제어 신호가 아니다.** ① 프레임 지연 **미측정**(`results.jsonl` 에 `elapsed_ms` 키 0개) ② PRIMARY 셀이 `following` = **ω_e ≈ 0 퇴화점** = 제어가 프리뷰를 가장 덜 필요로 하는 기하 ③ ★**제어상 옳은 head 셀이 더 강한데**(`15/16 −23.91% p=2.6e-04`) **부차로 밀려 있다.** | `MEASURED` |
| **A3-6** | ★★★**`T < 1.8·Tp` 는 Kim 2023 원문에 축자로 있으나, 그것은 그들의 WG21 기하가 Eq.(21)로 낸 값이지 보편 규칙이 아니다.** 같은 Eq.(21)을 **우리 개구**에 적용하면 예산은 **0.7–5.7 s** 다. 배포 초기화 **17.0 s(C, 작동점)** ⟹ **3.0–7.4배 초과**, head 셀은 **9.5–31배 초과**. ⟹ ⛔**`STATUS` 표 9행("63/64 통과")은 빌려온 예산에 대한 통과다.** | `VERIFIED` (원문 PDF) + `MEASURED` + `ARGUED` (재도출) |
| **A3-7** | ★★★**제어 쪽 문헌은 개구 기준을 청구항에 박아 놨고 우리는 그 안에 못 들어간다.** Naiad **US 10,259,544** 청구항 1: *"the far field being defined as **four or more wavelengths** from the vessel … the not yet existing predicted sea condition is **in the far field**"*, 명세서: *"preferably is approximately **2-4 wave lengths** … **or 30 seconds**"*. 우리 개구 = **0.20–0.65 λ_p** ⟹ **3.1–6.2배 미달**, 예보 지평 **5–43배 미달**. | `VERIFIED` (청구항 축자) + `MEASURED` |
| **A3-8** | ★★★**08-19 의 특허 정정이 절반만 전파됐다.** `US 12,054,225`(Candela)는 파도장 재구성 특허가 **아니지만**, 청구항 1이 *"obtaining information indicating **shape of water surface in front of the hydrofoil**, predicting wave acceleration … **using a neural network** … calculating an **AoA** … "* 이고 센서 목록에 **LIDAR-unit** 이 있다. ⟹ **추정 novelty 원장에서 뺀 것은 옳고, 제어 novelty 원장에 넣지 않은 것은 틀렸다.** 게다가 저장소의 기준 선체가 **바로 그 회사의 Candela C-8** 이다. | `VERIFIED` (청구항 축자) + `MEASURED` (코드) |
| **A3-9** | ★**살아있는 제어 서사가 하나 있다 — 단 지금 팔리는 서사가 아니다.** `D-18` 이 산 것은 *"예보의 가치"* 가 아니라 ***"지연되지 않은 선체 위 나우캐스트의 가치"*** 다(질의점이 `build_query_current` = 현재 포일 위치, τ-무관). 그리고 그 24/24 는 **`mmse_gpu(kp_hint)` = 오라클 kp 를 받은 비-출하 추정기**가 냈고 **learned 팔은 통째로 생략**됐다. | `MEASURED` (코드·config 축자) |
| **A3-10** | **D-1 권고 = (a) 좁힌다, 단 "고정 플랫폼"이 아니라 ★"단주기 선체-근접 파고 보상"으로.** 아래 §7. | `ARGUED` |

**한 줄**: ***조우는 이제 쟀지만 제어는 아직 안 쟀다. 그리고 제어를 쟀던 두 번은 08-14·08-19·08-21 원장에서 통째로 빠져 있었다 — 한 번은 실패로, 한 번은 통과로.***

---

## 0-B. 열람 범위 (= 주장 범위)

**연 것 (저장소)**
- `tmp/DEBATE_NOVELTY_PHD_2026-08-21/BRIEF.md` (전문)
- `~/.claude/.../memory/`: `MEMORY.md` · `reference_citation_bans.md`(전문 + 제어/α_wave/조우/지연 축 grep) · `direction1_crb.md` · `project_c_encounter_2026-08-20.md` · `project_s2_control_metric_2026-08-08.md`(전문) · `project_blank_slate_debate_2026-08-04.md`(전문) · `project_wp_e2e_2026-07-26.md`
- `tmp/STATUS_2026-08-21/STATUS.md` (전문)
- `tmp/C_ENCOUNTER_2026-08-20/`: `RESULT.md` · `AMENDMENT_1.md` · `gen_c.py` · `run_c.py` · **`results.jsonl` 448행 전건 재집계** · 에피소드 HDF5 **13개**(4셀 × 3–4개, 관측 기하·성분표 추출)
- `tmp/S2_control_metric_2026-08-08/`: `alpha_wave.py`(전문) · **`results.jsonl` 288행 전건 재집계**
- `tmp/E_REALTIME_2026-08-19/RESULT.md` (전문, 429줄)
- ★`results/direction3/d17_ctrl_eval.json` · `d18_ctrl_tune.json` · `d20_replay_eval.json` **전건 재집계**
- ★`docs/D17_CONTROL_prereg_2026-08-... (D17_CONTROL_prereg_2026-07-12.md)` **전문**
- ★`scripts/d3_d17_ctrl_closedloop.py` · `scripts/d3_d18_ctrl_tune.py` (관측·질의·팔 정의 구간) · `scripts/d3_gpu_mmse.py`(`kp_hint` 확인)
- ★`hydrofoil/controller.py` · `hydrofoil/config.py` · `hydrofoil/wavesplat_bridge.py` **전문**, `hydrofoil/disturbance.py`(시그니처)
- `tmp/DEBATE_novelty_2026-08-19/` 4문서 (제어·조우·α_wave·특허 축 grep, `P1`·`P4` 해당 절 정독)

**연 것 (문헌, 1차 출처 직접)**
- ★**Kim, Ducrozet, Bonnefoy, Leroy & Perignon (2023)**, *Ocean Eng.* **276**:114212 — **Zenodo 승인원고 PDF 17쪽 전문 텍스트 추출 후 전수 검색** (`zenodo.org/records/7891109`)
- ★**US 10,259,544 B2** (Naiad Maritime Group) — Google Patents, **청구항 1 전문 + 명세서 축자**
- ★**US 12,054,225 B2** (Candela Technology AB / Hasselskog) — Google Patents, **청구항 1 전문 + 초록 + 센서 목록 축자**
- WebSearch 4회 (DSWP 제어 요구조건 · FutureWaves · 하이드로포일 preview control · Kim 서지)

**안 연 것 (판정은 `[인용]` 또는 미판정)**
- Wang et al. 2025 *Coastal Eng.* 197:104690 (유료) · Qi et al. 2018 JFM 838 / Wave Motion 77 (유료) · Desmars JFM 975 A8 **원문**(E 배치의 2차 인용만 사용) · Ehlers *Phys. Fluids* 37:107119 **원문**(동상) · FutureWaves IEEE 원문 · 하이드로포일 preview servo 논문 원문
- W4·W5·`B` 배치 문서 본문(α 부재는 `grep` 로만 확인) · `hydrofoil/sim.py`·`vehicle.py`·`run_phase3_e2e.py` 본문 · `results/direction3` 의 나머지 파일
- **런 0회.** 어떤 추정도 실행하지 않았다.

---

## 1. 08-14 / 08-19 대비 **무엇이 바뀌었나** (BRIEF §0-7 필수 항목)

이 축에서 바뀐 것은 **새 자료가 아니라 원장(inventory)** 이다.

| | 08-14 · 08-19 가 말한 것 | 08-21 에 내가 확인한 것 |
|---|---|---|
| 제어 응용(기여 ⑦) | `P4_committee_chair.md:58` *"α_wave 3연속 널 + 제어 프리뷰점이 LiDAR 사각지대 안"* `[인용]` 표기 | **α_wave 3연속 널 = 유지**(재집계로 확증). **사각지대는 `[인용]` 이 아니라 지금 `MEASURED` 다: 8/8** |
| 제어 축의 상태 | `P1_ocean_reviewer.md:630` *"⛔현재 이 질문에 답이 없다. W4/W5 는 α_wave 를 재지 않았다"* / 처방 *"초록에서 제어를 빼라"* | **W4/W5 뿐 아니라 `B`·`C` 도 안 쟀다**(4/4 배치 `alpha` 0회). ★**그러나 "답이 없다"는 거짓** — `D-17`/`D-18` 이 2026-07-12 에 **사전등록 게이트로 답했다**. 두 문서 모두 그 존재를 모른다 |
| 조우 기하 | `P4:277` **Q1 최위험** *"조우를 한 번이라도 측정하셨습니까?"* → 08-20 `C` 가 최초로 답했다고 기록 | ⛔**"저장소 최초로 센서를 움직였다"는 배포 팔·`wavesim` 파이프라인 한정이다.** `d3_d18_ctrl_tune.py:314` `direction_deg=180.0` + `:collect_obs_window` **`sx = U * t`** ⟹ **2026-07-12 에 이미 이동 센서 · head seas · 선체추종 부채꼴로 폐루프를 돌렸다** |
| 특허 선점 | `P1:313-318` *"US 12,054,225 는 수중익선 승차감 제어 특허 ⟹ 선점하는 것은 응용 프레이밍이지 재구성 알고리즘이 아니다"* | **정정의 전반부는 옳다. 후반부가 전파되지 않았다** — "응용 프레이밍을 선점당했다"는 곧 **우리 제어 기여가 선점당했다**는 뜻인데, 어느 원장에도 그렇게 적히지 않았다 |
| 실시간 축 | `STATUS §1` 9행 *"Kim 2023 `T<1.8Tp`: 배포 0/64 → 새 구성 63/64 통과"* | **`1.8Tp` 는 원문 축자 확인(VERIFIED). 그러나 그것은 Eq.(21)이 그들 기하에서 낸 값이다.** 우리 기하로 재도출하면 **0.7–5.7 s** ⟹ **9행은 통과가 아니다** |

**바뀌지 않은 것**: 척추(위상 하위공간 최적화 실패)·기준선 축·기제 novelty. 나는 그 축들을 재판정하지 않는다.

---

## 2. 재판정 ① — **"제어-호환" 기둥은 살아 있는가** (과제 항목 1)

### 2.1 원 판정 두 건을 직접 읽었다

**(i) `project_blank_slate_debate_2026-08-04.md`** — 지연 헤드라인 사망:
> 같은 100ep·같은 스크립트 per-frame ms: **WS 7.02 / ETKF 18.46 / LETKF 29.51 / GPR-1 32.86 / sliding-GPR 98.73 / GPR-disp 102.11** ⟹ 격차 **2.6배**, **≤100 ms 예산을 넘는 팔이 사실상 0개** ⟹ *"제어-호환이 이 방법을 변별하지 못한다"*

★ **이 판정은 그대로 유효하고, 나는 그것을 강화한다**: 제어-호환은 **변별자가 아니라 충족된 요구사항**이며, 아래 §4 가 보이듯 **진짜 예산은 100 ms 가 아니라 예보 지평이고 거기서는 아무도 충족하지 못한다.**

**(ii) `project_s2_control_metric_2026-08-08.md`** — 288런. **재집계로 전건 재현했다**(`MEASURED`):

| 레짐 | 셀 | α_wave nRMSE @0.2 s (에피소드 median-of-3) | 무기술 널 대비 skill |
|---|---|---|---|
| **혼잡** | `AUTO` | **0.8283** | **17.2%** |
| **혼잡** | `AUTO-NOTRACK` | **0.7801** | **22.0%** |
| 희소 | `AUTO` | 0.0567 | 94.3% |
| 희소 | `AUTO-NOTRACK` | 0.0402 | 96.0% |
| (참고) 혼잡 전역 η | `AUTO` | 0.1863 | — |

★**내가 추가로 뽑은 것**(원 RESULT 에 없다): **혼잡에서 `α_wave ≥ 1.0`(= 제로예측기보다 나쁨) 에피소드가 `AUTO` 2/16 · `AUTO-NOTRACK` 3/16** 이다. 희소는 0/16.
`stencil_err` 중앙 **0.1301** 도 재현(= 스텐실 평활 13.0%).

### 2.2 W4 · W5 · `B` · `C` 가 이것을 뒤집었는가 — **아니다. 시도조차 없다.**

`MEASURED`, `grep` 전수:

| 배치 | `*.jsonl` 에 `alpha` | `*.py` 에 `alpha_wave` |
|---|---|---|
| `W4_SOLVER_2026-08-19` | **0** | **0** |
| `W5_WHY_2026-08-19` | **0** | **0** |
| `B_DEPLOY_2026-08-19` | **0** | **0** |
| `C_ENCOUNTER_2026-08-20` | **0** | **0** |

⟹ ★**정직한 재판정**: *"η 가 잘 나오는 것과 제어 입력이 나오는 것은 다르다"* 는 **뒤집히지 않았다.** 다만 그 이유는 **"재시험에서 살아남았다"가 아니라 "재시험이 없었다"** 다. 08-19 `P1:630` 의 *"현재 이 질문에 답이 없다"* 는 **W4/W5 에 대해서는 정확**하다.
그리고 ⛔ 인용금지 목록 **(08-11 S8)** 의 *"제어가 좋아진다"* 금지는 **여전히 발동 상태**다.

### 2.3 ★★★그러나 원장이 틀렸다 — **폐루프 제어 게이트 두 건이 이미 존재한다**

`docs/D17_CONTROL_prereg_2026-07-12.md`(사전등록, 동결) + `results/direction3/d17_ctrl_eval.json` + `d18_ctrl_tune.json`. **전건 재집계했다.**

#### `D-17` (동결 예보 FF, n=4 = BF3/BF4 × 시드 2) — **FAIL**, 재현

| 팔 | 히브 RMS 감소(무FF 대비) 중앙 | >0 인 셀 |
|---|---|---|
| persistence (참 파고, 프리뷰 0) | **+27.96%** | 4/4 |
| **hand-FF** (`mmse_gpu(kp_hint)`) | **−4.01%** | **0/4** |
| **learned-FF** | **−5.35%** | **0/4** |
| GT-FF (참 파고, 완전 프리뷰) | **+31.56%** | 4/4 |

★★★**그리고 이 표의 진짜 내용은 추정기가 아니다**:
> **GT − persistence = 히브 median +4.40 pp, 가속도 median −3.35 pp(GT 가 진다).**

⟹ ★★★**`preview_dt = 0.5 s` 에서는 완벽한 파고장을 공짜로 줘도 프리뷰 자체가 거의 아무것도 사지 못한다.** 사전등록 문서가 이미 그렇게 적었다(축자):
> *"**GT조차 persistence 대비 근소**(0.0967 vs 0.1010) = 이 작동점(τ=0.5 s 프리뷰, 11.3 m/s)의 **프리뷰 한계이득 자체가 얇음**"*

⟹ ★**S2 의 "혼잡 17–22%" 를 추정기의 실패로 읽는 것은 부분적으로 틀렸다.** 같은 리드 대역(0.1–0.5 s)에서 **참 파고장의 한계이득도 얇다**. **계측 지점이 문제의 일부다.**

#### `D-18` (스트리밍 재추정 FF, n=24 = BF3/BF4/BF5 × 시드 2 × τ∈{0.5,1,2,3}) — **PASS**, 재현

| 팔 | 히브 RMS 감소 중앙 | >0 인 셀 |
|---|---|---|
| persistence (지연된 참 파고) | **−22.15%** | **2/24** |
| hand_frozen (워밍업 1회 추정, 재추정 없음) | −1.43% | 8/24 |
| ★**hand_stream** (1 s 마다 재추정) | ★**+12.29%** | ★**24/24** |
| GT (완전 상쇄 천장) | +15.28% | 24/24 |

- **hand_stream > persistence: 24/24 · > noFF: 24/24 · GT 천장 회수 중앙 0.899**(min 0.593, max 1.076).
- ★**기제는 예보가 아니라 개구 추종이다**: 전방 skill corr 0–5→20–30 s 구간이 **frozen 0.699→0.149**, **streaming 0.713→0.696(평탄)**.
- ★**이득이 τ-평탄**(+12.54 / +12.54 / +12.25 / +12.28%): **프리뷰 길이를 늘려도 이득이 안 는다.** 늘어나는 것은 **persistence 의 피해**뿐이다.

#### ⚠ `D-18` 을 인용하기 전에 반드시 붙일 것 (내가 코드에서 직접 확인)

1. ⛔**출하 추정기가 아니다.** 팔은 `mmse_gpu(P,z,depth,Pq,sigma0,**kp_hint**)` 이고 `kp_hint = (2π/Tp)²/g` = **참 해상상태에서 온 오라클 kp**. 인용금지 목록 **(08-12 전체 감사)** 의 *"`WaveSplat` 을 단일 추정기로 쓰지 마라 · `results/direction3/*` 는 출하 추정기 산출이 아니다"* 가 **그대로 발동**한다.
2. ⛔**learned 팔은 통째로 생략**됐다 — `config.learned_arm` 축자: *"SKIPPED (CPU ~2.4s/call; 30 re-est x cells prohibitive) -- hand only"*.
3. ⛔**관측이 `wavesim` LiDAR 가 아니다.** 두 스크립트 모두 `wavesim` 에서 `jonswap_components` **하나만** 임포트하고, 관측은 `eta(x,y,t) + N(0, 0.02)` 를 부채꼴(r 20–100 m, ±60°, 12 pt/0.1 s)에서 뽑는다. **레이캐스트·raydrop·차폐·입사각 결손·FOV 클리핑 전부 없다.** 이는 인용금지 **(08-12 B-1)** 이 `d72`/`d74` 에 건 제한과 **같은 부류**다.
4. ⛔**n=24 는 독립 24 가 아니다** — τ 4수준이 같은 에피소드 안의 반복이므로 **독립 셀 6개**, 부호검정 바닥 **p=0.031**.
5. ⛔**예산 판정이 평균으로 내려졌다.** `stream_latency_ms` = **mean 34.76 / max 2932.65 (n=702)**. 사전등록 판정은 *"재추정 평균 35 ms — 100 ms PASS"* 인데 ★**최댓값이 예산을 29.3배 넘긴다.** 이것은 이 저장소가 08-19 `E` 에서 스스로 잡아낸 *"꼬리가 없다"* 결함의 **더 나쁜 판본이다 — 꼬리가 파일 안에 있는데 판정이 그걸 안 봤다.**
6. ⛔**τ 의 의미가 사전등록 중에 바뀌었다.** 축자: *"literal τ=preview_dt 는 스모크에서 퇴화 확인 … → τ를 **액추에이터 정보 지연**으로 구현(명령은 t−τ 발행)"*. 메인 검수가 수용했다고 적혀 있으나, **결과를 보기 전 변경**이라는 증거는 문서 안에 없다.
7. ★★**그래서 `D-18` 이 실제로 산 문장은 예보가 아니다.** `build_query_current` 축자: *"Query points at the foil's **CURRENT** position (x_cg + x_foil, t) … **tau-INDEPENDENT**"*. 즉 추정기가 공급하는 것은 **지금 포일 밑의 η와 그 미분**이고, τ 는 그것을 늦게 쓸 뿐이다.
   ⟹ ✅**살 수 있는 문장** = ***"액추에이터 정보 지연 0.5–3 s 아래에서, 선체를 추종하며 1 s 마다 재추정하는 파고장 나우캐스트는 무보상 대비 히브 RMS 를 12% 줄이고 완전상쇄 천장의 90% 를 회수한다. 같은 조건에서 지연된 직접 계측(persistence)은 무보상보다 나쁘다(2/24)."***
   ⛔**못 사는 문장** = *"예보(forecast)의 제어 가치를 입증했다"* — 질의점이 미래가 아니다.

#### `D-20` (실측 스테레오 리플레이, 5 레코드 × 12창 × τ 2수준 = 120행) — 재집계

| τ | persistence | hand | learned | learned < persistence |
|---|---|---|---|---|
| 0.5 s | 0.776 (55/60 도움) | 0.629 (51/60) | **0.523** (54/60) | 49/60 |
| **1.0 s** | **1.347** (**12/60** = 대부분 해악) | 0.621 (52/60) | **0.535** (53/60) | **55/60** |

(값 = 무보상 운동 대비 잔여 RMS 비. 낮을수록 좋다.)
⟹ ★**실측 위에서도 같은 모양**: τ 가 0.5→1.0 s 로 늘면 **persistence 가 무너지고(중앙 1.347) 추정기는 버틴다(0.535).**
⚠ 단 이것도 `net_gamma_mmse`/`mmse_gpu` 계열이고, **RAO≈1 히브 프록시**이며, 독립 레코드 5개 ⟹ **`DESCRIPTIVE`**.

### 2.4 § 2 결론

> **기둥은 죽지 않았다. 다만 지금 팔리는 형태로는 죽었다.**
> 죽은 것: *"우리 방법이 제어 지표를 개선한다"*(S2·S7·S8 3연속 널, 유지) · *"지연이 변별자다"*(08-04, 유지) · *"τ=0.5 s 프리뷰가 제어 가치를 만든다"*(`D-17` 이 **참 파고장으로도** 부정).
> 산 것: *"선체추종 스트리밍 재추정이 없으면 파고장 FF 는 무가치하고(8/24), 있으면 12%를 번다(24/24)"* — 그러나 **오라클 kp · 비-출하 추정기 · 비-`wavesim` 관측 · 독립 6셀 · 최악지연 2933 ms** 라는 네 겹 조건 안에서만.

---

## 3. 재판정 ② — **`C` 의 조우 신호는 제어에 쓸 만한가** (과제 항목 2)

### 3.1 먼저 `C` 를 전건 재집계했다 (`MEASURED`, 448행)

| 셀 | `SHIP<ZERO` | rel median | p | `V2K<ZERO` |
|---|---|---|---|---|
| A 정지 following | 16/16 | −57.98% | 1.5e-05 | 10/16 (p=0.227) |
| B 정지 head | 14/16 | −15.60% | 2.1e-03 | 7/16 (p=0.773) |
| ★**C 이동 following (PRIMARY)** | **13/16** | **−17.56%** | **1.06e-02** | **6/16 (p=0.895)** |
| ★**D 이동 head** | ★**15/16** | ★**−23.91%** | ★**2.59e-04** | 10/16 (p=0.227) |
| `S-2` 이동 vs 정지(following, SHIP) | **0/16** | **+87.83%** | — | — |

`RESULT.md` 의 모든 수치가 소수점까지 재현된다. **측정은 건전하다.**

### 3.2 제어 관점에서 이 수를 읽으면 — **세 가지가 걸린다**

**(a) ⛔`C` 는 지연을 한 번도 재지 않았다.**
`results.jsonl` 의 키 = `['arm','avg_RMSE_near','avg_nRMSE','cell','code_md5','effective','est_seed','init_time_s','key','n_frames','route','seed','shell_frac','shell_rmse','wall_s']`. **`elapsed_ms` 계열 0개.**
⟹ ★**저장소 유일의 조우 기하 배치가 프레임 갱신 비용을 안 쟀다.** 제어 심사에서 이것은 "정확도만 재고 실시간성은 다른 기하에서 잰" 것이고, 이 저장소가 `b1_full_comparison.json` 에서 이미 한 번 크게 물린 **"두 팔이 같은 조건에서 채점된 적이 없다"** 부류와 동형이다.
내가 대신 뽑을 수 있는 유일한 지연 프록시는 `wall_s` 다: 30.0 s 에피소드에 대해 **SHIP 중앙 26.1 / 29.8 / 31.7 / 32.3 s = 0.87 / 0.99 / 1.06 / 1.08 × 실시간**(A/C/B/D). ⚠**채점 비용 포함 · 8워커 경합** ⟹ `ARGUED`, 그러나 **여유가 1배 근처라는 방향은 견고하다.**

**(b) ★★★PRIMARY 셀이 제어상 가장 덜 흥미로운 기하다.**
`AMENDMENT_1` 이 스스로 적었다: 작동점의 `ω_e` 퇴화점은 `Tp = 7.24 s` 이고 BF4 가 7.0 s = **3% 안**, 최대 진폭 성분이 30 s 동안 **0.15 사이클**, 에너지의 **39%가 주기 63 s 이상**.
제어 관점의 번역: ***following 작동점에서 선체가 보는 교란은 거의 정적이다.*** 준정적 교란은 **PD 피드백이 원래 잡는 것**이고, 프리뷰가 필요한 상황이 아니다. 프리뷰가 필요한 것은 **ω_e 가 높은 head seas** 다.
그리고 ★**`D` 셀(이동 head)이 `C` 보다 강하다**: `15/16 −23.91% p=2.59e-04` vs `13/16 −17.56% p=1.06e-02`.
⟹ ★**제어 심사위원으로서의 정확한 문장**: ***"조우 기하에서 제로필드를 이긴다"는 주장은 제어상 옳은 heading(head)에서 더 강한데, 배치는 그것을 부차로 두었다.*** (⚠ `D` 는 사전등록 PRIMARY 가 아니므로 **확증이 아니다**. 요구하는 것은 승격이 아니라 **재사전등록**이다.)

**(c) ⛔`V2K` 가 작동점에서 제로필드에 진다(6/16)** — 이것이 제어에서는 정확도보다 무겁다.
제어기는 **오류의 평균이 아니라 꼬리**에 물린다. 학습 체크포인트가 분포 밖으로 나가는 순간 팔이 **무기술 이하**로 떨어진다는 것은, FF 경로에서 **액추에이터를 틀린 방향으로 때린다**는 뜻이다(`D-17` 에서 실제로 그렇게 나타났다: hand/learned 0/4, **noFF 보다 나쁨**). 정확도 라인에서는 "널"이지만 제어 라인에서는 **`active harm`** 이다.

### 3.3 예보 지평·갱신율·지연으로 채점하면

| 자 | 요구 | 우리 값 | 판정 |
|---|---|---|---|
| **예보 지평** (§4 에서 우리 기하로 도출) | 제어가 쓸 수 있으려면 최소 액추에이터 지연 τ 보다 커야 함 | **0.7–5.7 s** (head 0.7–2.3 · following 2.3–5.7) | 아슬아슬 — 존재는 한다 |
| **초기화** | < 예보 지평 | **17.0 s** (C 작동점 med3), **21.8 s** (D) | ⛔**3.0–31배 초과** |
| **갱신율** | ≤ 100 ms (10 Hz LiDAR 프레임) | ⛔**`C` 미측정**. 대리값 `wall_s/30 s ≈ 0.87–1.08×` | ⛔**미판정** |
| **재추정 주기** | `D-18` 이 요구: **1 s** (없으면 8/24 로 무가치) | 출하 합성 경로는 `initialize()` **1회**, 재초기화 트리거 없음(`E §3.5`) · 실측 경로 `bank_refit_every_s` 기본 `None` | ⛔**출하 기본값이 `D-18` 이 통과시킨 구성이 아니다** |

⟹ ★★★**§3 판정**: ***`13/16 −17.6%` 는 통계적 부호이지 제어 신호가 아니다.*** 제어 신호가 되려면 (i) 프레임 지연·최악 프레임을 같은 런에서 재고 (ii) heading 을 head 로 사전등록하고 (iii) 초기화를 예보 지평 아래로 내리고 (iv) **스트리밍 재추정을 켠 채로** 재야 한다. 지금은 넷 다 없다.
그리고 ⛔**`C · RESULT.md §6-3` 이 이미 *"seakeeping·제어 주장 금지(선체 응답 모형 없음)"* 라고 못박아 놓았다** — 나는 그 금지를 **지지**한다. ⚠단 사유는 정정한다: **선체 응답 모형은 `hydrofoil/` 에 있다**(Candela C-8, 2-DOF 히브·피치, 탠덤 포일, `run_simulation`). 없는 것은 모형이 아니라 ***`C` 코퍼스를 그 모형에 물린 런*** 이다.

---

## 4. 실시간 문턱 `T < 1.8·Tp` — **원문 검증과 우리 기하 재도출** (과제 항목 3)

### 4.1 원문 확인 `VERIFIED` — Zenodo 승인원고 PDF 17쪽 전문 추출

**Kim, I.-C., Ducrozet, G., Bonnefoy, F., Leroy, V. & Perignon, Y. (2023)**, *Real-time phase-resolved ocean wave prediction in directional wave fields: Enhanced algorithm and experimental validation*, **Ocean Engineering 276, 114212**. (Zenodo `records/7891109`, CC-BY)

축자 (§ 결과부):
> "The real-time system generally has a total numerical time 𝑇ₜₒₜₐₗ, composed of times for nowcast 𝑇ₙₒwcₐₛₜ and forecasts 𝑇fₒᵣₑcₐₛₜ, **which must be smaller than the length of the practical prediction zone in time at the location of the turbine** (i.e., 𝑇ₜₒₜₐₗ < 𝑡′ₘₐₓ = 𝑡ₘₐₓ − 𝑡ᵣ, **see Eq. (21)**). The length of the practical prediction zone from the latest time of nowcast 𝑡ᵣ until WG21 falls outside the prediction zone 𝑡ₘₐₓ **is about 1.8𝑇𝑝 = 21.6 s at full scale**. … **All the computations were run on an Intel(R) Core(TM) i7-8700 CPU 3.20 GHz.**"

Table 2 축자 (Case A @ WG21):

| | LWT | ICWM Previous | ICWM Simplified | LWT-CDR Previous | LWT-CDR Simplified |
|---|---|---|---|---|---|
| 𝑇ₙₒwcₐₛₜ (s) | 1.1 | **95.7** | **7.6** | **52.1** | **3.0** |
| 𝑇fₒᵣₑcₐₛₜ (s) | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 |
| 𝑇ₜₒₜₐₗ (s) | 1.7 | 95.7 | 7.6 | 52.1 | 3.0 |
| 반복수 | – | 22.7 | 5.8 | 15.9 | 3.0 |
| 수렴확률(Nₛ=400) | – | 83.5% | 98.5% | 88% | 100% |

⟹ ✅**E 배치(08-19)의 `1.8·Tp` 인용은 정확하다. 내가 1차 출처로 독립 확인했다.**

### 4.2 ★★★그러나 `1.8` 은 규칙이 아니라 **그들 기하의 산물**이다 `VERIFIED`

Eq. (20)–(21) 축자:
> "𝑡′ₘₐₓ = min[ (𝑥−𝑥_𝐷)/(𝑐_g,max cos(𝜃_B/2)), (𝑥−𝑥_𝐵+𝑦′_𝐵)/(𝑐_g,max cos45°) ] (21)"
> "the prediction zone **is governed by the group velocity**" · "𝑐_g,max = 𝑔/2𝜔ₘᵢₙ and 𝑐_g,min = 𝑔/2𝜔ₘₐₓ" (𝜔 컷오프는 피크 에너지밀도의 𝜇=0.05)
> "the left boundary propagates with 𝑐_g,max whereas the right boundary is determined by 𝑐_g,min. Accordingly, **the predictable domain begins to shrink and finally disappears** as the left boundary moves ahead of the right boundary."

⟹ **예산 = (관측대에서 관심점까지의 거리) ÷ (최대 군속도).** `Tp` 는 그 비를 정규화한 결과일 뿐이다.
그리고 ⚠ **관심점이 무엇인지도 다르다** — 축자: *"at the location of **the turbine**"*, *"the optimal control of issues such as **load mitigation on offshore floating wind turbines**"*, *"This network of wave measurements has been chosen to have a prediction horizon **suitable for the control strategies of the wind turbines**"*. ★**Kim 의 관심점은 고정 구조물이고 센서만 vessel-mounted 다. 우리는 관심점이 센서와 함께 움직인다** — 이 차이는 예산을 **줄인다**(관심점이 관측대 쪽으로 다가가므로).

### 4.3 ★★★같은 Eq.(21)을 **우리 기하**에 적용 — `MEASURED` 입력 + `ARGUED` 도출

입력(전부 `C` 코퍼스 HDF5·성분표에서 직접 측정):
- 성분 `ω ∈ [0.4879, 2.6723]` rad/s ⟹ **`c_g,max = 10.05 m/s`**, `c_g,peak = 4.76`, `c_g,min = 1.84`
- **센서 기준 LiDAR 발자국**(4셀 × 3–4 에피소드, 유효점 ~60,000/셀):
  `min 4.8–10.0 · p05 7.7–10.6 · **median 14.6–16.4** · p95 36.8–39.3 · max 215–247 m`
- `U = 11.3 m/s`, 관심점 = 포일(CG ±2.5 m)
- 폐쇄속도: head = `U + c_g`, following = `U − c_g`

| heading | `c_g` | 발자국 median 15 m | 발자국 p95 37 m |
|---|---|---|---|
| **head (dir 180)** | `c_g,max` | **0.70 s** | **1.73 s** |
| **head** | `c_g,peak` | **0.93 s** | **2.30 s** |
| following (dir 0) | `c_g,peak` | 2.29 s | 5.66 s |
| following | `c_g,max` | 12.03 s | 29.68 s ⚠**퇴화**(`U−c_g,max = 1.25 m/s`) |

⟹ ★★★**우리 실용 예보 지평 = 0.7–5.7 s** (following·`c_g,max` 코너는 `ω_e≈0` 퇴화라 제외).
**빌려온 `1.8·Tp = 12.6 s` 는 우리 기하가 주는 값의 2.2–18배로 관대하다.**

### 4.4 우리 수를 우리 자로 다시 채점 (`MEASURED` 초기화 × `ARGUED` 예산)

| 팔 · 셀 | 초기화(med3) | head 예산 0.70–2.30 s | following 예산 2.29–5.66 s |
|---|---|---|---|
| `SHIP` C 이동 following (**작동점**) | **16.99 s** | — | ⛔**3.0–7.4배 초과** |
| `SHIP` D 이동 head | **21.83 s** | ⛔**9.5–31배 초과** | — |
| `SHIP` A 정지 following | 15.93 s | — | ⛔2.8–7.0배 |
| `V2K` (전 셀) | **38.0–42.5 s** | ⛔17–61배 | ⛔6.7–19배 |
| `L-VP-ONLY` (08-19, **다른 코퍼스·정지**) | 8.47 s | ⛔**3.7–12.1배** | ⛔1.5–3.7배 |

**Kim 자를 `1.8·Tp = 12.6 s` 로 두고 `C` 배치를 채점하면**(내가 `C` 원자료로 직접 계산):
> **`SHIP` 이 4 셀 × 16 에피소드 = 64/64 전건 실패. `V2K` 도 64/64 실패. 행 단위로도 192/192 · 192/192 실패.**

⟹ ★★★**§4 판정 셋**
1. ✅**`1.8·Tp` 는 원문 축자다** — E 배치의 검증은 옳고 나는 그것을 1차 출처로 확인했다.
2. ⛔**`STATUS §1` 9행 *"배포 0/64 실패 → 새 구성 63/64 통과"* 를 "실시간 문턱을 넘었다"로 읽지 마라.** 그 통과는 **(i) 빌려온 기하의 예산**에 대한 것이고 **(ii) 정지 코퍼스**에서 잰 것이며 **(iii) 조우 기하에서는 같은 자로도 64/64 실패**한다.
3. ★**허용되는 문장** = ***"우리 초기화 비용은 Kim et al. 이 실시간 불충족이라 판정한 `Previous` 동화(52.1 / 95.7 s)와 그들이 통과시킨 `Simplified`(3.0 / 7.6 s) 사이에 있고, 우리 자신의 관측 개구가 주는 예보 지평(0.7–5.7 s)에는 아직 들어가지 못했다."***
   ⚠ 반드시 병기: **하드웨어(7800X3D 1스레드·8워커 경합) · 층(혼잡) · 합성(GT 가 모델식과 동일) · 경로(`learned` 고정)**.

---

## 5. 문헌 — **제어 쪽이 요구하는 성능 기준 (축자)** 과 우리 위치 (과제 항목 4)

### 5.1 ★★★Naiad Maritime Group, **US 10,259,544 B2** *"Predictive sea state mapping for ship motion control"* (우선일 2012-10-24, 등록 2019-04-16) `VERIFIED`

**청구항 1 축자** (개구를 청구항 안에 넣었다):
> "wherein the one or more sensors are configured to measure sea conditions in the seafaring vessel's **near and far fields when the vessel is underway** … **the far field being defined as four or more wavelengths from the vessel**; the near field being between the vessel and the far field; **wherein the not yet existing predicted sea condition is in the far field.**"

**명세서 축자** (센서·지평 요구):
> "Navigational radar, microwave radar, ultrasonic detectors, stereo-optic imaging, and **LiDAR (Light Detection and Ranging) laser systems**, among others, may be used."
> "This may use short range radar to provide data on the immediate waves approaching the vessel from any direction. This may be limited by the mast height of the sensor mounting, but **preferably is approximately 2-4 wave lengths, depending on the wave period, or 30 seconds, depending on the encounter frequency.**"

### 5.2 우리가 그 기준의 어디에 있는가 — `MEASURED`

| 제어측 기준 (축자) | 요구 | 우리 (C 코퍼스, 센서 기준 실측) | 격차 |
|---|---|---|---|
| DSWP 입력 개구 | **2–4 λ** (=153–306 m @ Tp 7 s) | median **0.20 λ** (15.5 m) · p95 **0.49 λ** (37.5 m) | ⛔**4.1–20배 미달** |
| 청구항 far-field | **≥ 4 λ** (306 m), 그리고 *예측 대상이 far field 에 있어야* | 우리 발자국은 **전부 near field**. 최대 tail 231 m 조차 3.0 λ | ⛔**청구 영역 밖** |
| 예보 지평 | **30 s** | **0.7–5.7 s** (§4.3) | ⛔**5.3–43배 미달** |
| 제어 프리뷰 점 | (암묵) far field | **8/8 이 최근접 LiDAR 반환보다 배에 가깝다**, 최원단 **0.10 λ** | ⛔**사각지대** |

⚠ **개구 수치의 이중 확인**: 저장소 자신의 기록(`blank_slate` R3)이 *"배포 0.63λ"* 라 적었고, 내가 원자료에서 독립 도출한 p95 값 **0.65 λ**(실현 최대진폭 성분 기준) / **0.49 λ**(공칭 Tp=7 기준)와 **일치**한다.

### 5.3 ★★★Candela Technology AB, **US 12,054,225 B2** (우선일 2021-05-06, 등록 2024-08-06, 발명자 G. Hasselskog) `VERIFIED`

**청구항 1 축자 (핵심부)**:
> "A method … for controlling motion of a watercraft (120) with a hydrofoil (130) … : **obtaining (A010) information indicating shape of water surface in front of the hydrofoil (130)**, **predicting (A040) wave acceleration of the watercraft (120) using a neural network**, wherein the neural network is trained to output the wave acceleration using training data … derived from information indicating shape of water surface in front of the hydrofoil, **speed of the watercraft**, **angle of attack of the hydrofoil, "AoA"**, and measured acceleration, determining (A050) a target route and corresponding total acceleration … under a set of constraints … **calculating (A060) an AoA for the target route** … and **sending (A070) a signal for adjusting the hydrofoil (130) according to the AoA**."

**센서 축자**:
> "The sensor device may comprise a 3d-shape detecting unit, an image sensor, a camera, a video camera, **a light detection and ranging sensor, 3D-laser scanner unit, laser imaging detection and ranging sensor, LIDAR-unit**, a radar sensor, or the like."

★★★**우리 제어 파이프라인과 대조** (코드에서 직접, `MEASURED`):

| 청구항 요소 | 우리 구현 |
|---|---|
| "shape of water surface **in front of the hydrofoil**" | `hydrofoil/controller.py` FF 는 `(x_foil + V·Δt, y)` 의 파고를 쓴다 — **같다** |
| **neural network** 로 파랑 기인량 예측 | 배포 라우터의 `learned` 경로 = 확산 프라이어 체크포인트 — **같은 부류** |
| **AoA** 산출·포일 조정 | `compute_flap_command` 이 `α_wave = −slope − w/V` 로 `δ_ff` 를 만든다 — **같다** |
| 선속 · 참조 선체 | `SimConfig.speed = 11.3`, `candela_c8()` = **Candela C-8** — ★**출원인 자사 선박** |

⟹ ★★★**08-19 정정의 미전파**: `P1_ocean_reviewer.md:313-318` 은 *"이건 승차감 제어 특허다 ⟹ 선점하는 것은 **응용 프레이밍**이지 재구성 알고리즘이 아니다"* 라 옳게 적었다. **그러나 이 저장소의 기여 ⑦은 정확히 그 "응용 프레이밍"이다.** 추정 원장에서 뺀 것이 제어 원장에 들어가지 않았다.
⚠ **정직 조항**: 특허 선행기술은 **청구항 단위로 판단**해야 하고, 우리 것이 그 청구항에 읽히는지는 **변리사 판단 영역**이다. 내가 주장하는 것은 침해가 아니라 ***"제어 응용을 novelty 로 팔면 이 청구항이 첫 반론으로 온다"*** 이다.

### 5.4 나머지 문헌 (검색 결과, 2차) `[인용]`

- **FutureWaves™** (Kusters et al., OCEANS 2016; APS): X-band 레이더 **5 km** 관측, **최대 5분 전** 개별 파랑·선체 임팩트 예보, 정온기(quiescent period) 탐지. ⟹ **상용 제품이 이미 존재**하고 지평이 우리보다 **50–400배** 길다. ⚠원문 미열람.
- **하이드로포일 preview control** (Ocean Eng. 2004, optimal preview servo; JMSE 10(10):1390 LQRY-SMC 등): 전수중익선 종운동에 **feedforward + preview** 를 붙이는 것은 **표준 설계**이며, 설계 항목이 *"necessary **prediction range** of future information"* 이다. ⟹ ★**"프리뷰를 쓴다"는 novelty 가 아니다. novelty 후보는 "프리뷰의 출처가 희소 LiDAR 파고장 재구성"이라는 것뿐인데, 그것이 §5.3 의 청구항이다.**

---

## 6. ★내 정정·철회 (BRIEF §6 필수)

1. ⛔**내가 처음 세운 프레임을 철회한다** — 나는 *"제어-호환 기둥은 08-04·08-08 에 죽었고 08-21 자료가 살릴 수 있는지만 보면 된다"* 는 전제로 시작했다. **틀렸다.** `D-17`·`D-18` 이 그 사이(2026-07-12)에 있었고, 하나는 **PASS** 다. **부재를 확인하기 전에 부재를 전제했다** — E 배치가 08-19 에 스스로 적은 교훈(*"부재 증거로 정정하지 마라"*)의 재발이고, 이번엔 내가 저질렀다.
2. ⛔**"C 가 저장소 최초로 센서를 움직였다" 를 무조건 인용하려 했다가 철회한다.** `scripts/d3_d18_ctrl_tune.py` 의 `sx = U * t` + `direction_deg=180.0` 이 **5주 먼저** 있었다. 정확한 형태 = ***"`wavesim` 레이캐스트 LiDAR 파이프라인과 배포 팔에서는 `C` 가 최초"*** 다.
3. ⛔**나는 `D-17` 의 `persistence` 를 처음에 "값싼 지연선 경쟁자"로 읽었다 — 틀렸다.** 코드 축자로 `persistence` 는 **`true wave, preview_dt=0`** = **참 파고장 오라클**이다. ⟹ *"추정기가 지연선에 진다"* 는 문장은 **쓸 수 없고**, 정확한 문장은 *"추정기가 프리뷰 0 짜리 참 파고장에 진다"* 다. 이 정정은 **우리에게 유리**하다(경쟁자가 오라클이므로).
4. ⛔**나는 `D-18` 의 `+12%` 를 "예보의 제어 가치"로 옮기려 했다 — 철회.** `build_query_current` 축자가 *"tau-INDEPENDENT"* 다 ⟹ 산 것은 **나우캐스트**의 가치이고, 사전등록 문서의 D-19 결과 문장(*"스트리밍 **예보** 피드포워드는 …"*)도 **같은 이유로 측정보다 넓다**. 이것은 저장소의 **미보고 서술 과잉 1건**이다.
5. ⛔**`STATUS §1` 9행을 "실시간을 통과했다"로 요약하려던 것을 철회한다** — §4.3 재도출로 예산이 우리 것이 아님이 드러났다. 다만 ★**E 배치의 `1.8Tp` 인용 자체는 옳았고**, 나는 그것을 **깎으려다 실패**했다(원문에 축자로 있다). 이것도 §6-1 과 같은 부류의 내 오류다.
6. ⚠**`C · RESULT.md §6-3` 의 "선체 응답 모형 없음" 을 사유째로 인용하려던 것을 정정한다** — 모형은 `hydrofoil/`(Candela C-8, `run_simulation`)에 **있다**. 없는 것은 **`C` 코퍼스를 그 모형에 물린 런**이다. **금지 자체는 옳으므로 유지**하고 사유만 교체한다.

### ⛔ 이 문서가 새로 제안하는 인용 금지

| # | 금지 | 사유 |
|---|---|---|
| C-1 | ⛔**`D-18` 의 `24/24`·`+12%`·`회수 90%` 를 "출하 시스템의 제어 가치"로** | 팔이 **오라클 `kp_hint`** 를 받은 **비-출하 `mmse_gpu`**, learned 팔 **생략**, 관측이 **비-`wavesim` 해석장+2 cm 잡음**, 독립 셀 **6개**(바닥 p=0.031) |
| C-2 | ⛔**`D-18` 의 "재추정 평균 35 ms — 100 ms PASS"** | 같은 파일에 **`max 2932.65 ms`(n=702)** 가 있다. 허용 = **"평균 34.8 ms · 최악 2933 ms(예산 29.3배 초과)"** |
| C-3 | ⛔**`D-17`/`D-18`/`D-20` 을 "예보(forecast)"로** | 질의점이 **현재 포일 위치**(`build_query_current`, τ-무관). 허용 = **"선체 위 나우캐스트"** |
| C-4 | ⛔**`C` 의 `wall_s` 를 지연으로** | 채점 비용 포함 · 8워커 경합. 허용 = **"30 s 에피소드에 대해 0.87–1.08× 실시간(경합 포함, 방향만)"** |
| C-5 | ⛔**`1.8·Tp` 를 "우리 실시간 예산"으로** | Eq.(21)이 **Kim 의 WG21 기하**에서 낸 값. 우리 기하 값은 **0.7–5.7 s** |
| C-6 | ⛔**`C` 의 `D` 셀(15/16 −23.9%)을 확증으로** | PRIMARY 가 아니다. 허용 = **"사전등록 밖 기술통계이며, 제어상 옳은 heading 이 더 강하다는 것은 재사전등록 사유"** |
| C-7 | ⛔**"제어 응용은 우리 기여다"** | **US 12,054,225 청구항 1**(LiDAR 수면형상 → NN 파랑가속도 → 포일 AoA)이 **정면 선행기술**이고, 우리 참조 선체가 **그 출원인의 배**다 |

---

## 7. ★D-1 권고 (필수 마지막 절)

### 7.1 권고 = **(a) 좁힌다. 단 BRIEF 가 적은 (a) 문장으로는 안 된다.**

BRIEF §3 의 (a) = *"고정/저속 플랫폼의 실시간 파고장 추정"*.
⛔**제어 심사위원으로서 이 문장에 반대한다.** 세 가지 이유:

1. **"고정 플랫폼"으로 좁히면 §5.3 의 특허 문제도 §5.1 의 개구 문제도 안 풀리고, 대신 Kim et al. 2023 과 정면 충돌한다** — 그들의 관심점이 **정확히 고정 부유식 풍력터빈**이고, 하드웨어·표·수조검증·통과판정을 이미 냈다.
2. **"고정"은 `D-18` 이 산 유일한 문장을 버린다.** `D-18` 의 이득은 **부채꼴이 선체를 추종**해서 나온 것이다(frozen 8/24 → streaming 24/24, skill 0.149 → 0.696). **이동이 비용이 아니라 이득의 조건이었다.**
3. **`C` 는 (b) 를 죽인 게 아니다.** `C` 가 죽인 것은 *"정지 자료로만 학습한 체크포인트가 작동점에서 버틴다"* 이지 *"이동 기하에서 제어 가치가 없다"* 가 아니다. 후자는 **`D-18` 이 반대 방향으로 측정했다.**

### 7.2 내가 제안하는 (a′)

> ### **(a′) *"선체-근접 단주기 파고 보상 — 자기 개구가 허용하는 0.5–5 s 지평 안에서, 이동하는 희소 LiDAR 로부터의 스트리밍 나우캐스트"***

**왜 이 문장인가 — 전부 이미 측정된 것으로만 지탱된다:**

| 문장의 조각 | 지탱하는 측정 |
|---|---|
| "선체-근접" | 우리 개구 **0.20–0.65 λ** ⟹ Naiad 의 far field(≥4 λ)를 **주장하지 않는다** = 청구항과 충돌하지 않는다 |
| "0.5–5 s 지평" | **Kim Eq.(21)을 우리 기하에 적용한 값 0.7–5.7 s**(§4.3). ★**남이 준 문턱이 아니라 우리 센서가 정한 문턱** |
| "이동하는" | `D-18` 24/24 (부채꼴 추종이 이득의 **조건**) + `C` `D` 셀 15/16 |
| "스트리밍 나우캐스트" | `D-18` frozen 8/24 vs streaming 24/24, skill 0.149 vs 0.696. ★**`bank_refit_every_s` 가 08-20 에 이미 노출됐다**(`frozen 0.9452 → refit@2s 0.7932`) — **같은 기제의 실측판** |
| "희소 LiDAR" | 혼잡 우위 3중 방어(S13·S13b·HOSGATE) |

★★★**이 문장의 결정적 장점**: **저장소가 이미 두 경로에서 같은 기제를 독립적으로 측정했다** — 합성 제어 폐루프(`D-18` streaming vs frozen)와 실측 나우캐스트(`bank_refit_every_s` refit@2s vs frozen). **서로 다른 추정기·서로 다른 자료·서로 다른 지표인데 같은 방향이고, 크기도 같은 자릿수다.** 이것이 현재 저장소에서 **기준선과 무관하게** 살아 있는 두 번째 축이다(첫 번째는 척추).

### 7.3 (a′) 를 사려면 필요한 것 — 우선순위와 비용

| # | 작업 | 왜 결정적인가 | 비용 |
|---|---|---|---|
| **K-1** | ★★★**`C` 코퍼스를 `hydrofoil/sim.py` 에 물린다** (`D-18` 프로토콜 = 스트리밍 재추정 + 액추에이터 지연 τ 스윕, **팔은 출하 `SequentialUpdater`**, heading **head 사전등록**) | 이것 하나가 **§2·§3 의 모든 구멍을 동시에 닫는다**: 출하 팔 · `wavesim` 관측 · 이동 기하 · 폐루프 · learned 팔. ⟹ Q1(*"조우를 쟀는가"*)과 Q-제어(*"25%가 조종에서 무엇을 바꾸는가"*)에 **한 배치로 답한다** | 에피소드 **재생성 0건**(C 데이터 재사용) + 추정 ≈ C 규모(33분) + sim. **~3–5 h** |
| **K-2** | ★**모든 런에 `elapsed_ms_max` · `p99` · `predict_field` 계측을 넣는다** | 08-19 `E` 의 M1·M2 가 **아직 미이행**이고, `D-18` 에서 이미 **최악 2933 ms** 로 물렸다. **제어에서 100 ms 는 마감이지 평균이 아니다** | 코드 **2–3줄** |
| **K-3** | ★**`α_wave` 의 널을 제로예측기에서 `persistence`(지연된 참 파고)로 교체**하고 프리뷰 점을 **관측 지지집합 안으로** 옮긴다(≥15 m ⟹ lead ≥ 1.2 s) | 현 널은 제어에서 **틀린 널**이고, 현 프리뷰 점은 **8/8 사각지대**다. `D-20` 이 **τ≥1 s 에서만 추정기가 persistence 를 55/60 으로 이긴다**고 이미 말했다 | `alpha_wave.py` **LEADS·FOILS 수정 + 널 팔 1개**. 런은 K-1 에 합승 |
| **K-4** | **초기화를 예보 지평 아래로** — VARPRO(`L-VP-ONLY`) 를 조우 기하에서 재측정 | 현재 **17.0 s vs 예산 0.7–5.7 s**. VARPRO 3.19배로도 **head 는 3.7–12배 부족** ⟹ 초기화가 아니라 **`bank_mmse` 급 닫힌형(실측 경로 0.7–1.4 s)** 을 합성 경로에도 여는 것이 진짜 후보 | 런 0회 설계 → K-1 에 팔 1개 추가 |
| **K-5** | **`D-17`·`D-18`·`D-20` 을 `MEMORY.md`·`STATUS` 원장에 등재** | 저장소가 **자기가 가진 유일한 폐루프 제어 게이트를 세 번의 토론 동안 몰랐다.** 이것은 08-19~20 에 다섯 번 재발한 *"같은 것을 두 곳에 따로 적어 어긋남"* 의 **가장 비싼 판본이다 — 이번엔 한 곳에만 적혀 있어서 아예 사라졌다** | **0** |

### 7.4 (b) 를 고른다면 — 내 조건

BRIEF 의 (b)(*"11.3 m/s 항주 선박의 조우 예보"*)를 지키겠다면, **혼합 커리큘럼 재학습보다 먼저 K-1 을 해야 한다.** 이유: `C` 는 **정확도**가 이동에서 1.9배 나빠진다고 말했지만, `D-18` 은 **제어 이득은 이동(부채꼴 추종)에서 나온다**고 말한다. **두 문장이 같은 배치에서 동시에 참일 수 있다** — 그렇다면 재학습의 목표함수가 nRMSE 가 아니라 **잔여 히브 RMS** 여야 한다. **K-1 없이 재학습하면 틀린 목적함수에 GPU 를 태운다.**

### 7.5 ★한 문단 요약 (사용자 결정용)

> **조우는 쟀고(`C`), 제어는 2026-07-12 에 이미 쟀는데 원장에서 사라졌다(`D-17` FAIL / `D-18` PASS 24/24).**
> `D-18` 이 산 문장은 *"예보"* 가 아니라 ***"선체를 추종하며 계속 다시 맞추는 나우캐스트"*** 이고, 그것은 08-20 실측 `refit@2s` 와 **같은 기제**다. 반면 지금 팔리는 문장 — *"제어-호환 실시간 조우 **예보**"* — 은 세 곳에서 막힌다: **개구가 제어 문헌 요구의 1/4–1/20**(Naiad 청구항 ≥4 λ vs 우리 0.2–0.65 λ), **초기화가 우리 자신의 예보 지평의 3–31배**(Kim Eq.21 재도출 0.7–5.7 s vs 17–22 s), 그리고 **응용 프레이밍이 Candela 특허 청구항 1 에 정면으로 놓여 있다**.
> ⟹ **범위를 (a′) 로 좁히고, `C` 데이터를 재생성 없이 `hydrofoil` 폐루프에 물리는 한 배치(K-1, 3–5 h)로 제어 기여를 한 번에 확정하거나 죽여라.** 그 배치가 실패하면 제어 축은 **정직한 음성**으로 종결하면 되고, 성공하면 **저장소에서 유일하게 기준선과 특허 양쪽을 피해가는 기여**가 된다.
