# P1 — 해양공학 저널 심사위원 판정 (2026-08-19)

**역할**: Ocean Engineering / Applied Ocean Research 급 리뷰어. 실해역 타당성 · 물리 일관성 · 재현성 1순위,
알고리즘 novelty 2순위. **런 0회** — 기존 기록 열람 + 저장된 행의 재집계 + 문헌 검증만.

**규율**: `MEASURED`(내가 원자료에서 직접 재도출) / `ARGUED`(내 판단) / `UNVERIFIED`(확인 실패) 분리.
⛔`reference_citation_bans.md` 의 금지 수치는 근거로 쓰지 않았다.

## 0. 열람 범위 = 주장 범위

**연 것**: `memory/{MEMORY.md, reference_citation_bans.md, project_s17_locladder_2026-08-19.md,
project_w4_remedy_2026-08-19.md, project_novelty_search_2026-08-15.md,
project_persona_debate_novelty_2026-08-14.md}` ·
`tmp/S17_WHITE25_2026-08-18/{RESULT.md, verdict_s17.json, results.jsonl}` ·
`tmp/W4_SOLVER_2026-08-19/{RESULT.md, PREREG.md, corpus_w4.json, verdict_w4.json, run_w4.py}` ·
`tmp/W5_WHY_2026-08-19/{RESULT.md, PREREG.md, AUDIT_PRO.md, analyze_w5.py, verdict_w5.json, results.jsonl}` ·
`tmp/S16_BASELINE_2026-08-14/results.jsonl` · `tmp/STATUS_2026-08-18/STATUS.md` ·
`tmp/DEBATE_novelty_2026-08-14/P2_ocean_reviewer.md` · `tmp/cdip_support_2026-08-03/{cdip_support.py, cdip_support_raw.json}` ·
`data_s8_crowded_*/`·`data_s9_crowded_*/`·`data_s8_sparse_*/`·`data_s9_sparse_*/` 의 HDF5 `config/wave_components`.

**안 연 것**: `paper/wavesplat.tex` · `thesis/thesis_main.tex`(08-14 P2 의 축자 인용에 의존) ·
S13/S13b/HOSGATE 원자료 · W4 의 `AUDIT_*` 3종 전문 · `results/` 대부분.
⟹ 원고 문면에 대한 판정은 **2차 서술 기반**이고, 아래 §1 은 **"이 자료로 쓸 수 있는 논문"** 에 대한 판정이다.

**문헌 검증**: 3개 독립 에이전트가 WebSearch/WebFetch 수행. 결과 §3.

---

# 1. 판정 — **된다. 단 "우리가 이겼다" 논문이 아니라 "무엇이 벽이고 어떤 매개화가 그것을 푸는가" 논문으로만.**

## 1.1 08-14 판정이 바뀌었는가 = **부분적으로 바뀌었다. 세 자료 중 두 개가 판정을 움직인다.**

08-14 에 나(P2 페르소나)는 **REJECT** 를 냈다. 그 판정의 두 기둥은 이랬다:

| 08-14 기둥 | 2026-08-19 상태 |
|---|---|
| **(A) 기여 ⑥ 사망** — 기준선이 백색 배경공분산이라 약했고, 그래서 마진이 기준선 결함의 산물일 수 있다 | ★**S17 이 닫았다**(§2.1). 부재가 아니라 **측정**으로. **판정 이동** |
| **(B) 우위가 바다가 만들지 않는 방향분산 대역에 있다** | ⚠**여전히 열려 있다**. 코퍼스가 CDIP **0.3 → 6.7 퍼센타일**로 올라왔을 뿐(§2.3). **판정 미이동** |
| (C) 알고리즘 축 선점 | ⛔**더 나빠졌다** — VARPRO×스파이크 디컨볼루션 조합이 **2026-07 에 선점**됐다(§3.2) |
| (D) 인용 가능한 정확도 **크기** 0개 | ★**바뀌었다** — W4/W5 가 **자기 대조(ablation) 크기**를 하나 만들었다(§2.2) |

⟹ **REJECT 를 유지할 근거가 두 개 남고, 두 개는 사라졌다.** 그런데 사라진 쪽이 *논문의 골격을 바꾸는* 쪽이다.

## 1.2 결정적 변화 — **논문이 기준선에서 독립했다**

08-14 까지 이 저장소가 팔려던 문장은 전부 `WaveSplat < LETKF` 형태였고, 그래서 기여 ⑥ 4중 사망이
논문 전체를 인질로 잡았다. **W4/W5 는 다르다.** 주 대조가 `L-VP-ONLY < L-SHIP` = **같은 추정기의 절제**다.
LETKF 가 강하든 약하든, 배경공분산이 백색이든 JONSWAP 이든, **이 문장의 진리값은 변하지 않는다.**

이것이 왜 큰가 `ARGUED`: OE 리뷰어가 방법 논문을 죽이는 가장 흔한 방법이 *"기준선이 약하다"* 인데,
**그 공격면이 구조적으로 존재하지 않는다.** 그리고 08-14 에 내가 유일하게 살렸던 자산
(*"사전등록 · 사망조건 · 적대적 감사"*)이 여기서 처음으로 **기여를 지키는 데** 쓰인다 —
같은 배치가 자기 처방을 사고 자기 진단을 죽였다(W4 `OPTIMIZER-EARNS`).

## 1.3 그럼에도 지금 그대로는 못 낸다 — **남은 결함 2개**

1. ⛔**세 배치 전부 합성 · 선형 GT · `direction_deg=0` · 혼잡 층 · 라우터 `learned` 고정**이다.
   OE 는 이 조합만으로는 방법 논문을 받지 않는다(§3.3 의 편집 정책 확인 결과).
2. ⛔**방향분산 지지 문제가 미해결**(§2.3). 08-14 가 *"최대 결함"* 이라 부른 그것이고,
   그 뒤 다섯 배치(S15·S16·S17·W3·W4·W5) 중 **어느 것도 이 축을 건드리지 않았다.**
   `S14-SPREAD` 는 08-14 에 설계까지 됐는데 **한 번도 실행되지 않았다**(`grep` 확인: 저장소 전체에서
   `S14-SPREAD` 문자열이 `P2_ocean_reviewer.md` 에만 등장).

⟹ **판정: 지금 투고하면 Major Revision ~ Reject 경계. §5 의 배치 1건을 추가하면 Ocean Engineering /
Applied Ocean Research 급에 실릴 수 있다.** ⛔**Coastal Engineering 은 스코프 불일치**(§3.3(h)).

★**문헌 검증이 이 판정을 08-14 대비 두 칸 올렸다** — 08-14 의 REJECT 를 떠받치던 전제 셋 중 둘이 거짓이었다:
- ⛔*"합성 전용은 OE 에 못 낸다"* → **거짓**. 같은 문제·같은 저널의 **Ehlers 2023 이 100% 합성**이고
  선점 논문이라던 **Desmars JFM 975 A8 도 합성 전용**이다(§3.3(f)).
- ⛔*"오래된 방법 적용은 결격"* → **거짓**. Coastal Eng 19/20 · JOMAE 19/20 이 차용 방법이고
  **1795년 방법(Prony)** 을 쓴 논문이 2021년 Ocean Engineering 에 있다(§3.3(b)(c)).
- ✅남은 하나(**"우위가 실측이 만들지 않는 레짐에 있다"**)는 **여전히 유효**하고, 그래서 §5 배치가 필요하다.

## 1.4 제목과 초록 (내가 직접 쓴다)

**Title**

> **Variable projection replaces joint optimisation in grid-free phase-resolved wave-field
> estimation from sparse shipborne LiDAR: a 3× faster initialisation with 25% lower error**

(대안, 기제를 앞세우는 판:
*"The wall is the solver, not the information: separable least squares for phase-resolved
wave-field estimation from sparse shipborne LiDAR"*)

**Abstract**

> Phase-resolved reconstruction of a sea surface from a single sparse remote sensor is usually framed
> as a geometrically limited problem — the predictable zone set by group velocity — and effort is spent
> on richer observation models or stronger assimilation. We show, for a grid-free spectral-Gaussian
> estimator driven by ship-mounted LiDAR, that inside the observed support the binding constraint in
> crowded seas is instead the parameterisation of the optimisation. Variable projection is standard in
> seismic full-waveform inversion but has, to our knowledge, not been applied to phase-resolved ocean
> wave field reconstruction; we transfer it and measure the mechanism.
> Starting the estimator from the true wave parameters, it still halts at an observation misfit 26.9×
> worse than its start (48/48 episodes), and 99% of the residual field error is phase decorrelation
> rather than amplitude error — both measured on the deployed path. Motivated by this, we replace the
> 500-step joint fit over amplitude, phase and wavenumber with a 400-step variable-projection scheme
> (Golub & Pereyra, 1973) in which amplitude and phase are eliminated analytically and descent acts on
> wavenumber alone. On 64 held-out crowded episodes with three estimator seeds, the replacement lowers
> median field nRMSE by 25.3% (58/64 episodes, all nine range shells, p = 4.5e-12), cuts initialisation
> from 26.8 s to 8.5 s, and leaves the per-frame update unchanged at ~9 ms; adding the joint fit back
> changes nothing (36/64, p = 0.38). The mechanism is identified: the decorrelation term falls by 52%
> and the entire gain is realised in initialisation, not in tracking. We also report two negatives
> preregistered before the runs — the diagnosis that the existing optimiser "earns nothing" is refuted
> by the same batch, and the sparse-sea advantage of this estimator family is an artefact of agreement
> between the generator and the estimator function space, measured by intervention. All batches are
> preregistered with git-anchored freezes and adversarial pre-run audits.

⚠**이 초록에 들어가면 안 되는 것**(내가 의도적으로 뺐다): 기준선 대비 크기 · 전역 nRMSE 헤드라인 ·
희소 우위 · 지연 배수 비교 · "실시간" · "novelty" · "초해상"(§3.4) · "CRB 프레이밍과 모순"(§3.1(f)).

⚠**투고 전 반드시 바꿔야 할 것 하나**: 위 초록은 `nRMSE` 로 쓰여 있다. §3.3(g) 대로
**`SSP`(Surface Similarity Parameter) 를 병기**하면 이 하위문헌의 리뷰어가 즉시 읽는다.
⟹ ★**§5 배치에서 SSP 를 기록하고 초록의 수치를 두 지표로 다시 쓸 것.**

---

# 2. 새 자료 3건 — 심사위원이 실제로 사는 것과 안 사는 것

## 2.1 S17 (768런) — ★**사는 것: 반박 하나가 부재에서 측정으로 바뀌었다**

`MEASURED`(`verdict_s17.json` 직독): `W` = `WHITER50` 61 · `WHITER25` 61 · `WHITER12` 62 · `WHITER6` 63 ·
`WHITERHS50` 60 · `SPEC50` 60 · **`SPEC25` 59** · `SPEC100` 59 (n=64). `best = SPEC25`, `W_env = 59`,
`iut = 63`, `thin_hits = 3/19`.

**심사위원 관점에서 이것이 값진 이유** `ARGUED`: *"당신의 기준선이 제대로 조정되지 않았다"* 는
리뷰어가 던지는 가장 흔하고 가장 답하기 어려운 질문이다. 보통 저자는 *"몇 개 스윕했다"* 로 답하고
리뷰어는 *"그 스윕에 내가 생각한 축이 없다"* 로 되받는다. S17 은 그 축(국소화 반경)을
**격자 해상도 바닥(지지폭 2.03 칸)까지** 밀었고 **아래 두 단에서 예외가 0/64** 다.
⟹ *"더 세게 국소화하면 좋아졌을 것"* 이 **논리적으로 닫힌다**. 이건 좋은 논문 재료다.

**⛔그런데 저자가 이걸 팔려고 하면 안 된다.** `S17/RESULT.md §6-1` 이 스스로 정확히 적었다:
*"기준선이 강해지지 않았다. 59/64 는 S16 의 수이지 새 수가 아니다."* ⟹ 논문에서 이것은
**결과 절이 아니라 robustness 절**이고, 한 단락 + 표 하나면 족하다.

⚠**내가 확인한 미해결 1건**: `S17` 의 `SHIP` 은 **배포 라우터(`auto`)** 이고 `W4/W5` 의 `L-SHIP` 은
**`learned` 고정**이다. 같은 64 에피소드인데 **다른 팔**이다(§2.4 에서 이 차이가 61 vs 59 로 나타난다).
논문에서 두 결과를 같은 표에 놓으면 리뷰어가 잡는다.

## 2.2 W4 + W5 — ★★★**이것이 논문이다**

### `MEASURED` (내가 `W5/results.jsonl` + 재사용 W4 열에서 직접 재집계, 중앙값)

| 팔 | n | wall_s | **init_time_s** | elapsed_ms | **decorr_mean** | init_nRMSE | 최종 nRMSE |
|---|---|---|---|---|---|---|---|
| `ZERO` | 64 | 0.2 | — | — | — | — | — |
| `L-ITER0` | 192 | 12.3 | 9.05 | 9.24 | 0.5292 | 0.1837 | 0.1826 |
| **`L-SHIP`(배포)** | 192 | 29.7 | **26.84** | 9.06 | **0.3851** | 0.1494 | **0.1578** |
| `L-VP`(배포+VP) | 192 | 30.8 | 25.70 | 8.91 | 0.2236 | 0.1222 | 0.1177 |
| ★**`L-VP-ONLY`** | 192 | **14.5** | ★**8.46** | 8.89 | ★**0.1848** | 0.1055 | ★**0.1058** |
| `L-VP-ONLY-NW` | 192 | 14.7 | 8.50 | 8.89 | 0.1908 | 0.1008 | — |
| `L-VP800` | 192 | 32.3 | 25.14 | 8.55 | 0.2109 | 0.1206 | — |
| `ORA-K-CVX`(상한) | 64 | 2.1 | — | — | — | — | 0.0013 |

### ★심사위원이 사는 문장 셋 (이 순서로)

**(1) 초기화 비용이 3.17배 줄면서 오차가 25% 준다.** `MEASURED`
**26.84 s → 8.46 s**, 프레임당 갱신은 **9.06 → 8.89 ms 로 사실상 불변**.
⟹ **이것이 이 저장소가 46일 만에 만든 첫 "공짜 점심"이다** — 정확도와 비용이 **같은 방향**으로 움직인다.
저장소가 지금까지 잰 것은 거의 전부 거래(`SUPPORT-TRADEOFF`)였다.

★**그리고 이것이 문헌이 명시적으로 비워 둔 유일한 칸에 정확히 떨어진다**: Ehlers et al.
(arXiv:2508.03315)이 Desmars 를 *"real-time capability is not demonstrated"* 라고 직접 깎는다(§3.1).
⚠**단 아직 못 산다**: 8.46 s 초기화는 Ehlers 자신의 0.014 s/sample 옆에 놓을 수 없다.
정직한 문장은 **"초기화를 3.17배 줄였고, 정상상태 갱신은 이미 ≤100 ms 제약 안(≈8.9 ms)"** 까지다.

**(2) 기제가 특정됐고, 그 기제는 11일 전에 다른 배치가 독립적으로 지목한 양이다.** `MEASURED`
S3(2026-08-08)이 *"남은 오차의 99%가 decorrelation, 진폭은 0.9%"* 라고 못 박았다.
그 `decorr_mean` 이 **0.3851 → 0.1848 (−52.0%)**.
⟹ ★**"이 노브를 돌렸더니 좋아졌다"가 아니라 "우리가 범인이라고 지목해 둔 양이 실제로 줄었다"** 다.
이것이 이 결과를 튜닝 보고서가 아니라 **기제 논문**으로 만든다.

`MEASURED`(정의 확인, `run_w4.py:380-384`): `decorr = 1 − ρ²`, `shrink = (√var_ratio − ρ)²`
= Murphy/Taylor 형 MSE 분해의 **decorrelation 항과 조건부 편향(진폭) 항**이고, S3 가 쓴 것과 같은 양이다.
⟹ ★**물리적으로 읽으면**: 추정장–진실장 상관이 **ρ ≈ 0.78 → 0.90** 으로 오른다.
논문에는 `decorr_mean` 보다 **이 ρ 로 적는 편이 해양공학 독자에게 훨씬 잘 읽힌다** `ARGUED`.
`MEASURED`(내 재집계, 중앙값):

| 팔 | `decorr`(1−ρ²) | 함의 ρ | `shrink` | decorr 지분 |
|---|---|---|---|---|
| `L-ITER0` | 0.5292 | 0.686 | 0.0621 | 89.5% |
| `L-SHIP`(배포) | 0.3851 | **0.784** | 0.0221 | 94.6% |
| `L-VP` | 0.2236 | 0.881 | 0.0103 | 95.6% |
| ★`L-VP-ONLY` | **0.1848** | ★**0.903** | 0.0082 | 95.7% |

⟹ **두 항이 함께 내려가지만 decorrelation 지분은 오히려 커진다**(94.6 → 95.7%)
⟹ ★**처방은 위상 항을 줄였고, 남은 오차의 성격은 바뀌지 않았다** — 즉 **벽의 정체가 같은 채로 벽이 낮아졌다.**
이것은 §4.2 Q3(상한 대비 ×81)과 정합적이고, 논문에서 정직하게 쓸 수 있는 한계 문장이다.

⚠**단 지위를 정확히 적어야 한다**: 이 분해는 **`AUDIT_PRO.md §4` 의 사후 관찰**이고 `D` 족에
사전등록되지 않았다(`W5/RESULT.md` 가 `D-5` 부재를 스스로 기록). ⟹ 논문에서는
**"exploratory mechanism analysis"** 로 라벨하고, §5 의 다음 배치에서 **사전등록**할 것.

**(3) 이득은 전부 초기화에서 나고, 추적은 배포 팔에서 오히려 악화시킨다.** `MEASURED`
`L-SHIP` init **0.1494** → 최종 **0.1578**(**+5.6% 악화**). `L-VP-ONLY` init 0.1055 → 최종 0.1058(중립).
⟹ W4 의 `OPTIMIZER-EARNS` 는 *"추적기가 번다"* 가 아니라 **"초기화가 번다"** 로 좁혀진다.

### ⛔심사위원이 **안** 사는 것

1. ⛔**`L-VP-ONLY` 를 "결합 적합만 뺀 팔"로 부르는 것.** `W5/RESULT.md §4` 가 실측했다 —
   `learned_init_iters=0` 은 **성장 분기 발화를 3배로 만든다**(47/192 → 141/192). 최소 4요인 묶음.
   ⟹ 논문은 **"we replace the initialisation stage"** 라고 써야지 **"we ablate the joint fit"** 이라 쓰면 안 된다.
2. ⛔**"매개화 덕분"**(W4 `S-2c` 가 가중을 요인으로 확정, `A-4` 귀속 조건 발동).
   ⚠단 W5 `S-3` 에서는 같은 가중 대조가 **널**(40/64, p=0.060)이다 ⟹ 요인 구조가 배치 간에 안정하지 않다.
3. ⛔**`ORA-K-CVX` 대비 ×81 을 "우리가 이만큼 남겨 뒀다"로 읽는 것** — 6요인 묶음.
4. ⛔**"실시간"** — 초기화 8.46 s.

### ⚠내가 찾은 무결성 결함 1건 (판정 불변, 그러나 기록 값어치)

`MEASURED`: **`W5/verdict_w5.json` 의 `iut` 필드가 `[55, 64]` 인데 실제 IUT 는 `63/64` 다.**
원인은 `analyze_w5.py` 의 **변수 섀도잉** — `:79` 에서 `w, n, rel = pair('L-SHIP','ZERO')` 로 잡은 뒤
`:124`(SECONDARY 루프)와 `:131`(S-4 셸 루프)이 같은 이름 `w, n` 을 재바인딩하고, `:167` 의
`iut=[w, n]` 이 **마지막 셸(`obs:150_inf`)의 값**을 기록한다.
- 내가 저장 행에서 직접 재계산: `L-SHIP < ZERO` = **63/64** ⟹ `RESULT.md` 의 인쇄값이 맞고 JSON 이 틀렸다.
- **판정 영향 없음**(둘 다 문턱 41 통과). 그러나 ★**W5 는 W4 와 달리 분석기 이중 검증(D1↔D2)이 없었다**
  — `W5/RESULT.md §6-7` 이 스스로 *"`AUDIT_CON` 은 MDE·검정력·셸 설계를 안 봤다"* 고 신고한 그 범위 축소와
  같은 방향의 대가다. ⟹ **처방: 동결 산출물(JSON)도 인쇄물과 대조하는 게이트를 넣을 것.**

## 2.3 ⚠**닫히지 않은 것 — 방향분산 지지** (내가 이번에 새로 측정)

`MEASURED` (런 0회. 저장된 HDF5 `config/wave_components` 에서 에피소드별 성분을 읽어
**저장소 자신의 정의**(`cdip_support.py:92 spread_from_moments`, σ₁ = √(2(1−r₁)), 에너지가중 a₁/b₁)로
재계산하고, `cdip_support_raw.json` 의 **실측 25,426 CDIP 레코드**에 대해 퍼센타일을 구했다):

| 코퍼스 | n | 방향분산 σ₁ 중앙 | **CDIP 퍼센타일** | Tp 중앙 |
|---|---|---|---|---|
| 레거시 100ep (08-14 가 문제 삼은 것) | 100 | 16.1° | **0.30%** | — |
| ★**혼잡 S8+S9 (S13·S16·S17·W3·W4·W5 전부가 쓴 코퍼스)** | 64 | **22.0°** | ★**6.7%** | 7.2–8.0 s |
| 희소 S8+S9 | 96 | 9.7° | **0.0%** | 6.7 s |
| **실측 CDIP** | 25,426 | **35.3°**(p5 21.1 · p25 28.9 · p75 43.0) | — | 9.88 s |

⟹ ★**08-14 의 최대 결함은 완화됐지 해소되지 않았다.** 코퍼스는 0.3 퍼센타일에서 **6.7 퍼센타일**로
올라왔지만, **살아남은 모든 주장이 실측 방향분산 분포의 좁은 쪽 7% 에서 측정됐다.**
(⚠Tp 는 오히려 괜찮다 — 중앙 7.2–8.0 s = CDIP 의 31–36 퍼센타일.)

**리뷰어의 질문은 정확히 이것이다**: *"방향분산이 2배가 되면 파수 공간에서 성분이 덜 분리된다.
당신의 처방은 바로 그 분리에 의존하는 것 아닌가?"*

### ★★★그런데 — 저장된 행이 이미 부분적으로 답한다 (내가 이번에 계산)

`MEASURED` **(EXPLORATORY · 사후 · 사전등록 없음)**. W5 의 64 혼잡 에피소드를 방향분산으로 층화:

| 층 | n | `L-VP-ONLY < L-SHIP` | med(rel) | p |
|---|---|---|---|---|
| 전체 | 64 | 58/64 | −25.28% | 4.5e-12 |
| σ₁ < 22.0° (중앙 미만) | 32 | 27/32 | −23.40% | 5.7e-05 |
| σ₁ ≥ 22.0° | 32 | **31/32** | **−30.19%** | 7.7e-09 |
| σ₁ ≥ CDIP p25 (28.9°) | 17 | **17/17** | **−33.54%** | 7.6e-06 |
| ★σ₁ ≥ **CDIP 중앙**(35.3°) | 10 | ★**10/10** | ★**−36.29%** | 9.8e-04 |

**교락 점검**(필수 — 성분수·BF 가 방향분산과 같이 움직일 수 있다): 고분산 부분집합의 셀 구성은
`σ₁≥28.9` 에서 (BF4-15/BF4-30/BF5-15/BF5-30) = **2/5/5/5**, `σ₁≥35.3` 에서 **1/4/3/2** 로
한 셀에 몰려 있지 않다. 그리고 **BF×성분수 네 셀 각각 안에서** 고분산 절반을 보면
**8/8 · 8/8 · 8/8 · 8/8 = 32/32**, 저분산 절반은 6/8 · 5/8 · 8/8 · 7/8 = 26/32.

⟹ ★**부호는 네 셀 전부에서 고분산 쪽이 더 강하고, 이는 BF 나 성분수로 설명되지 않는다.**
⚠**단 크기 순서는 일관되지 않는다** — `BF4-15comp` 에서는 저분산 −23.4% vs 고분산 −13.6% 로 **역전**된다.
⟹ 정직한 문장 = **"부호는 방향분산이 커질수록 강해지고, 크기는 그렇지 않다."**

⚠⛔**이것으로 §2.3 의 결함을 닫았다고 쓰면 안 된다**: ①사전등록 없음 ②n=10/17 ③방향분산이
에피소드 시드가 뽑는 다른 모든 것과 공변 ④여전히 실측 분포의 **상위 꼬리를 합성으로 흉내낸 것**이지
실측 스웰·이봉 스펙트럼이 아니다. **그러나 §5 배치의 사전 위험을 크게 낮춘다** — 처방이 실측 레짐에서
무너질 것이라는 예측에 대한 **첫 반대 증거**다.

## 2.4 ⚠**한 번도 측정되지 않은 것 — VARPRO 팔 vs 외부 기준선**

`MEASURED`: W4·W5 의 팔 집합에 **외부 기준선이 `ZERO` 하나뿐**이다(LETKF 계열 0개).
⟹ ★**"VARPRO 를 넣은 시스템이 자료동화 기준선을 이기는가" 는 아직 아무도 안 쟀다.**

**EXPLORATORY 교차배치 추정**(내가 계산. ⛔**결과 아님** — S16 과 W5 는 서로 다른 배치·`code_md5` 이고
W5 의 `G-0` 비트중립성 게이트에 해당하는 것이 **없다**. 게다가 S17 의 `SHIP` 은 라우터 `auto`,
W5 의 `L-SHIP` 은 `learned` 고정이라 **같은 팔이 아니다** — 그 차이가 아래 61 vs S17 의 59 로 나타난다):

| 팔 | vs `SPEC25`(최강 수리 기준선) | med(rel) | 팔 중앙 nRMSE |
|---|---|---|---|
| `L-SHIP` | 61/64 | −22.35% | 0.1578 |
| `L-VP` | 61/64 | −37.39% | 0.1177 |
| ★`L-VP-ONLY` | **60/64** | ★**−46.23%** | **0.1058** |
| (`SPEC25`) | — | — | 0.2063 |

⟹ `ARGUED`: **판돈이 크다**(마진이 대략 2배가 된다). 그리고 이 표는 **논문에 실을 수 없다** —
게이트가 없다. **§5 의 배치가 정확히 이 칸을 정식으로 채운다.**

### ★★그리고 같은 교차배치 계산의 **셸 분해가 더 흥미롭다** (동일 경고 적용)

| 셸 | `L-SHIP < SPEC25` | `L-VP-ONLY < SPEC25` |
|---|---|---|
| ★`obs:0_5` | **41/64  −8.7%** | ★**56/64  −32.8%** |
| `obs:5_10` | 47/64  −20.0% | **59/64  −38.6%** |
| `obs:10_20` | 54/64  −30.9% | 62/64  −45.3% |
| `obs:20_30` | 58/64  −34.3% | 62/64  −47.1% |
| `obs:30_50` | 62/64  −31.9% | 61/64  −49.0% |
| `obs:50_75` | 62/64  −26.1% | 60/64  −50.0% |
| `obs:75_100` | 58/64  −21.6% | 59/64  −51.7% |
| `obs:100_150` | 56/64  −22.6% | 61/64  −48.3% |
| `obs:150_inf` | 51/64  −19.7% | 59/64  −42.8% |

★**S16 이 발견한 "근거리는 기준선과 대등"(`obs:0_5` 46/64)이 VARPRO 팔에서 사라진다.**
가장 큰 상대 개선이 **가장 가까운 셸**에서 나고(−8.7% → −32.8%), 9셸이 전부 함께 좋아진다.
⟹ `ARGUED`: 해양공학 독자에게 이건 큰 의미다 — 배가 실제로 예보를 필요로 하는 구간이 근거리이고,
저장소가 여섯 번 물린 `SUPPORT-TRADEOFF` 의 **정반대 모양**이다.
⛔**단 반복한다: 게이트 없는 교차배치 추정이고 `L-SHIP`(learned 고정) ≠ S16 의 `SHIP`(auto) 이다.**
이 표의 용도는 **주장이 아니라 §5 배치의 검정력 설계**다.

---

# 3. 문헌 검증 결과 (3개 독립 에이전트, WebSearch/WebFetch)

## 3.1 해양공학 쪽 선점 상황 — ★**저장소가 믿고 있던 것 세 개가 틀렸다**

### (a) ⛔**`US 12,054,225` 는 파도장 재구성 선점이 아니다** `CONFIRMED`

**"Method and controller unit for controlling motion of a hydrofoil watercraft"**,
발명자 Gustav Hasselskog, 출원인 **Candela Technology AB**(전기 수중익선 회사).
우선일 2021-05-06, 등록 2024-08-06. 청구항 1 = 수중익 **앞쪽 수면 형상** 취득 → **신경망**으로 파랑 가속도 예측
→ 총가속도 최소 경로 → 받음각 작동. ★**"wave field" · "phase-resolved" · "spectrum" 이 문서에 등장하지 않는다.**
LiDAR 는 상용구 센서 목록에만 있고 독립항에 없다.
⟹ **이건 승차감 제어 특허다.** 선점하는 것은 **응용 프레이밍**이지 재구성 알고리즘이 아니다.
⛔**`MEMORY.md`·`STATUS.md` 의 *"알고리즘 축은 선점(Desmars JFM 975 A8 · US 12,054,225)"* 은 후자에서 거짓이다.**
⚠**진짜 특허 선행기술은 따로 있다**: **US 10,259,544 B2** + 계속출원 **US 10,538,299 B2**,
*"Predictive sea state mapping for ship motion control"*, **Naiad Maritime Group**, 우선일 2012-10-24 —
**선박 인근 수면을 LiDAR 로 스캔**하고 결정론적 파랑 예측을 승차제어에 먹인다. ⚠단 **독립항에는 없고 명세서 서술뿐**.

### (b) ⚠**Desmars JFM 975 A8 은 저장소가 믿는 것보다 훨씬 좁다** `CONFIRMED`

> Desmars, Hartmann, Behrendt, Hoffmann & Klein (2023), *"Nonlinear deterministic reconstruction and
> prediction of remotely measured ocean surface waves"*, **JFM 975, A8**, DOI 10.1017/jfm.2023.841.

- 방법 = **최적화/변분**(HOS–OCM + LGMRES · AMBM + Levenberg–Marquardt), **앙상블 DA 아님**
- 센서 = **해상 레이더 모형**, ⛔**LiDAR 아님**(LiDAR 쪽은 Desmars 의 **박사학위논문** tel-03185222)
- ★**응용이 전부 1차원 단방향**(*"the applications presented later are restricted to unidirectional waves"*)
- ★**모든 사례에서 센서가 정지**, 이동 플랫폼 사례 **0건**
- ★**CPU 시간·실시간 분석 없음**, 비볼록성/국소최소 **언급 0회**
- ★**그 자체가 합성 전용**(*"Three types of synthetic observations are used… numerically generated"*)

⟹ ★**"알고리즘 축 선점" 은 과잉 서술이다.** 정직한 문장 = **"문제 정의와 변분 정식화는 선점됐다.
2-D · 이동 센서 · 비용 분석은 그 논문에 없다."**
그리고 그 논문 서론이 저장소의 공백을 **이름으로 지목만 하고 채우지 않는다**:
*"in situ wave measurements… are not adaptable to predictions over a region in motion, such as around a
ship with forward speed"*.

### (c) ★★★**진짜 최근접 경쟁자를 저장소가 모르고 있다** `CONFIRMED`

> **Wang, F., Zhu, Q., Cai, C., Wang, X. & Qiao, R. (2025), *"From points to waves: Fast ocean wave
> spatial–temporal fields estimation using ensemble transform Kalman filter with optical measurement"*,
> Coastal Engineering 197, 104690**, DOI 10.1016/j.coastaleng.2024.104690

**희소 3-D 점군 → 파랑 시공간장 · 선형 분산관계 · ETKF(퍼지 이상치 처리) · CPU+GPU 실시간 ·
공개 자료 *및* 실측으로 검증 · "동일 패러다임 하에서 우월" 주장.**
⟹ ★**문제 진술도 기준선 계열도 같다. 그리고 실측이 있다.** 저장소의 어느 문서에도 이 논문이 없다.
⛔**논문 작성 전에 반드시 읽고 정면으로 위치지어야 한다.**

### (d) ★★**`LETKF` 는 위상분해 파랑장에 적용된 적이 없다** `CONFIRMED`

OpenAlex 체계 검색: LETKF+wave **53건 중 해양 중력파는 1건이고 그것은 위상평균 SWH 동화**
(Houghton et al., *Ocean Modelling* 183, 102200, 2023).
⟹ 저장소의 `LETKFGrid(loc50)` 는 **방어 가능하지만 비표준**이다. **인용해야 할 선례는 기상 LETKF 문헌이
아니라 위 (c) 의 Wang et al. 2025(ETKF)** 이고, `loc_radius` 는 **공동체 표준값이 아니므로 스스로 정당화해야 한다**.
★**그리고 이것이 S16 의 결과를 독립적으로 설명한다** — *"LETKF 를 움직인 축은 배경 스펙트럼이 아니라
`loc_radius`"* 는 애초에 그 값에 공동체 표준이 존재하지 않기 때문이다.

### (e) ★**남아 있는 공백 = 세 조건의 교집합 하나** `CONFIRMED`

**LiDAR + 이동 선박 + 위상분해** = 비어 있다. 그러나 **어느 하나씩은 전부 점유**돼 있다:
- 이동 선박 + 위상분해(레이더): ★**Yao, Zhang, Zhou, Song & Toffoli (2025), *Phys. Rev. Fluids* 10(9),
  094801** — 3차 HOS + 선박 레이더 영상 동화, 명시적으로 *"from stationary or moving ships"*. ⟹ **2026 이 아니라 2025 다.**
- 이동 스테레오: Alberello et al., *Nat. Commun.* 13 (2022); Bergamasco et al. (2016·2017·2021)
- LiDAR 위상분해: Nouguier, Grilli & Guérin, *IEEE TGRS* 52(3):1761–1771 (2014) — **시뮬레이션 전용, 오래됨**
- 최근 LiDAR 파랑 논문은 **스펙트럼(위상평균)** 이지 위상분해가 아니다

⚠**저장소의 *"이동선 재구성 3편이 2026 에 나온다"* 는 재서술이 필요하다**: 편수는 3편보다 많지만
**LiDAR 도 아니고 이동선 위상분해도 아니다**(Lowell & Irani *Measurement* 282:121990 = 스테레오·전진속도 없음;
Fan et al. *IEEE TGRS* 64 · Chen et al. *JPCS* 3178:012071 = 선박 레이더 **스펙트럼**).
⟹ **시간 압박의 근거는 2026 논문 3편이 아니라 Yao et al. 2025 하나다.**

### (f) ★★★**척추("문헌의 CRB 프레이밍과 모순")가 문헌 검증에서 성립하지 않는다** `CONFIRMED`

- OpenAlex `"phase-resolved" AND "Cramer-Rao"` = **0건**, arXiv `"Cramer-Rao" AND "ocean wave"` = **0건**.
  ⟹ ★**위상분해 재구성 문헌은 한계를 CRB 로 프레이밍한 적이 없다.** 그 분야의 한계 개념은
  **기하학적 predictable zone(군속도 부채꼴)** 이다: Wu 2004 · **Naaijen, Trulsen & Blondel-Couprie,
  *Int. Shipbuilding Progress* 61(3-4):203–223 (2014)** · **Qi et al., *Wave Motion* 77 (2018)** ·
  여전히 현행(Kim & Kim, *Ocean Eng.* 365, 127336, 2026).
- ⟹ ⛔**"우리는 문헌의 CRB 프레이밍과 모순된다" 는 쓸 수 없다.** 모순 대상을 **기하학적 예측가능영역**으로
  바꿔 써야 하고, 그러면 **이미 진행 중인 논쟁에 합류하는 것**이지 논쟁을 여는 것이 아니다:
  ★**Mérigaud, Zhu & Tona (2025), *JFM* 1007, A47**, DOI 10.1017/jfm.2025.84 — 확률적 예측가능영역을
  도입하고 **실측 스테레오 자료로 부채꼴 결정론적 예측가능영역을 경험적으로 반박한다**(상한 군속도가
  스펙트럼 피크 주파수로 더 잘 근사됨). ⛔**논문의 프레이밍 절을 쓰기 전에 반드시 읽을 것.**
  (부수: Fucile, Bulian & Lugni, *Ocean Eng.* 163:718–736, 2018 — GP 기반 예측오차 정량화.)
- ★**반대로 좋은 소식**: **모델 클래스가 정확히 같은 CRB 가 이미 존재하는데 이 분야가 안 쓰고 있다** —
  **Alnajjab & Blum (2017), *IEEE Trans. Sustainable Energy* 8(1):179–191**, DOI 10.1109/tste.2016.2590567:
  **서로 다른 주파수·방향의 평면파 합**으로 모델링된 해면을 **공간 분산 잡음 센서망**에서 추정할 때의
  **Fisher 정보행렬과 CRB**. 인용 6건 전부 WEC 하드웨어, **재구성 문헌에서 0건**.
  ⟹ ★**"정보는 충분한데 솔버가 못 쓴다"(M1 의 ρ=0.999964)를 정량화할 도구가 이미 있다. 만들 필요가 없다.**
- ⚠**그러나 대가가 있다**: *"한계는 옵티마이저 비볼록성이다"* 를 **주장한 사람도 없다**
  (문헌은 최적화를 **비용** 문제로만 다루고 **정확도** 문제로 다루지 않는다).
  ⟹ ★**반박이 아니라 확립을 해야 하고, 그쪽이 더 어렵다.**

### (g) ★**λ 결과의 독립 확증** `CONFIRMED` (보너스)

**Jiang, Kalisch, Benoit, Holand & Sprenger (2026), arXiv:2605.25074**, 축자:
*"conventional compressed sensing methods based on ℓ₁-shrinkage tend to **underestimate signal energy**
when sparsity is not strictly satisfied, leading to biased spectral estimation"*.
⟹ ★**S5/W3 가 측정한 기제(λ 벌점이 해를 민다)를 독립적인 해양파 문헌이 확인한다.**
⚠범위: 그들은 **Datawell 부이의 통계 스펙트럼**이지 위상분해 필드가 아니다. 그래도 **인용 가치가 크다** —
저장소의 가장 재현성 높은 음성 결과에 외부 앵커가 생긴다.

### (h) ⚠**표현(representation)의 최근접 친척** `CONFIRMED`

> **Chen, Z., Wu, J., Shen, L. & Li, P. Y. (2023), *"A Kalman Filter Approach to the Estimation and
> Reconstruction of Ocean Wave Fields"*, 2023 American Control Conference, 3696–3702**,
> DOI 10.23919/acc55779.2023.10155910

**진폭·위상이 미지인 Airy 성분의 합**으로 필드를 모델링하고 칼만 관측기로 추정.
⟹ ⛔**"성분 기반 순차 추적" 을 신규로 주장하기 전에 반드시 읽을 것.**

## 3.2 ★★★"1973년 방법" 의 진짜 문제는 1973년이 아니다

`CONFIRMED`(에이전트가 원문 초록 직독):

| 항목 | 확인 결과 |
|---|---|
| **Golub & Pereyra 1973** | *SIAM J. Numer. Anal.* **10(2):413–432**, DOI 10.1137/0710036. 확인 |
| **Golub & Pereyra 2003** (30년 리뷰) | *Inverse Problems* **19(2):R1–R26**. ★**초록이 이미 축자로 주장한다**: *"not only reduces the dimension of the parameter space **but also results in a better-conditioned problem**"* |
| **O'Leary & Rust 2013** | *Comput. Optim. Appl.* 54(3):579–593. *"better likelihood of finding a **global** minimizer rather than a local one"* |
| **arXiv:2511.01234** (2025-11) | `CONFIRMED`. Gan, Chen, Yi, Yang, *"A Saddle Point Remedy: Power of Variable Elimination in Non-convex Optimization"*. 안장점→축소지형 국소최대 정리(Hessian inertia + Schur 보수). ★**검증 영역은 행렬분해·2파라미터 NN·ResNet 뿐 — 물리 역문제 0건**. 정리 가정: `F∈C²`, 내부 문제의 해가 **유일·C¹·강볼록**, 내부 Hessian PD |

### ⛔그러나 에이전트가 **저장소가 모르던 선점 2건**을 찾았다 — 이쪽이 훨씬 무겁다

1. ★★★**arXiv:2607.09593** (Kalra, Ferreira Da Costa, Lee; v1 2026-07-10, v2 2026-07-24)
   *"Characterization of the Basin of Convexity for Multi-Snapshot Spike Deconvolution via Variable Projection"*.
   `CONFIRMED`(초록 직독): **"a variable-projection formulation in which the amplitudes are eliminated
   in closed form, thereby reducing the task to a nonconvex least-squares problem over the spike
   locations alone"**(그들은 `VarProSD` 라 명명) + **"an explicit characterization of the basin of
   convexity ... showing how sampling bandwidth and spike separation affect the local geometry"**.
   ⟹ ⛔**방법 축과 "지형" 축을 동시에 점유한다. 5주 전 논문이다.**
2. **"Variable Projection for Multiple Frequency Estimation"**, ICASSP 2020 (IEEE Xplore 9052907).
   다중 복소 정현파 주파수 추정을 **separable NLS + variable projection** 으로 정식화.
   ⟹ ⛔**"VARPRO 를 선스펙트럼 추정에 쓴다" 자체가 2020년에 이미 이름까지 붙어 출판됐다.**
3. 그리고 그 조작 자체는 **1990년대 표준**이다: Besson & Stoica 1999(*Digital Signal Processing* 9(1):45–56)
   — NLS 기준을 진폭·포락에 대해 **명시적으로 최소화하고 주기도형 함수만 남긴다**;
   RELAX(Li & Stoica, *IEEE TSP* 44(2):281–295, 1996); NOMP(Mamandipoor et al., *IEEE TSP* 64(19):5066–5081, 2016).
   ⚠RELAX·NOMP 의 **내부 진폭 소거 단계는 원문 미확인 = `UNVERIFIED`**(유료장벽).

⟹ ★★★**결론 정정: 저장소의 자기 금지문("VARPRO 는 1973년이라 novelty 0")은 이유를 잘못 짚었다.**
**진짜 사유는 "1973년이라서"가 아니라 "1996–2020년에 바로 이 문제 유형에 이미 적용됐고, 2026-07 에
지형 분석까지 나왔기 때문"** 이다. ⟹ **방법 novelty 는 확실히 0**이고, `arXiv:2607.09593` 은
**반드시 인용하고 정면으로 위치지어야 한다**(안 하면 리뷰어가 찾는다).

## 3.3 ★★★"1973년 방법 적용"이 해양공학에서 결격인가 = **아니다. 그게 이 분야의 정상 출판 양식이다.**

### (a) 편집 정책 — **다섯 저널 중 어느 곳도 새 방법을 요구하지 않는다** `CONFIRMED`(원문 인용)

| 저널 | 축자 | 함의 |
|---|---|---|
| **Ocean Engineering** | 범위 밖 목록에 *"…or **purely theoretical papers without any application to ocean engineering**"* | ★**벌하는 것은 응용 없는 이론이지 오래된 방법이 아니다** |
| **Applied Ocean Research** | *"all papers, particularly very theoretical ones, should stress the practical importance and applicability… **Purely theoretical papers with no application are discouraged**"* | ★같은 방향, 더 강함 |
| **Coastal Engineering** | *"Combining practical applications with modern technological and scientific approaches…"*, "fundamental studies **as well as** case studies" | 선언적 |
| **JOMAE (ASME)** | *"emphasizes the **fundamental mechanics** of topics, rather than the applications and case histories"* | ⚠**여기만 반대 방향** |
| **JMSE (MDPI)** | 건전성·재현성 심사, 방법 신규성 요구 없음 | 가장 느슨 |

### (b) 기저율 — **이식이 예외가 아니라 규칙이다** `CONFIRMED`(Crossref 질의)

- **Coastal Engineering 2015–2026: 반환된 20편 중 19편이 차용 방법**(ML · 자료동화 · 칼만).
- **JOMAE 2015–2026: 20편 중 19편이 이식** — *"fundamental mechanics"* 라는 범위 문구에도 **불구하고**.
- Ocean Engineering 에는 제목이 문자 그대로 *"Application of…"* 로 시작하는 논문이 일상적이다.

### (c) ★**차용 방법의 나이 — 1973년은 어리다** `CONFIRMED`

| 논문 | 차용 방법 | 나이 | 검증 |
|---|---|---|---|
| ★**Sun, Hu & Li (2021), *Ocean Eng.* 219, 108425** | **Prony-SS (Prony 1795)** | ★**~225년** | **수치 전용** |
| Wermbter, Tan & Abdel-Maksoud (2026), *JMSE* 14(9):819 | PSWF(Slepian 1961) + 통계적 선형화(Booton 1953) | 65 / 72년 | **수치 전용** |
| Komoriyama et al. (2023), *Ocean Eng.* 271, 113627 | 칼만 1960 | 63년 | 수조 |
| Guo et al. (2023), *Appl. Ocean Res.* 132, 103465 | POD(Lumley 1967) | 56년 | 실험 PIV |
| Wang et al. (2025), *Coastal Eng.* 197, 104690 | ETKF(Bishop 2001) | 24년 | 공개 + **실측** |
| Laface et al. (2018), *JWPCOE* 144(5):04018010 | 압축센싱(2006) | 12년 | **실측** |

⟹ ★★★**53년은 눈에 띄지 않는다. "1973년" 은 결격 사유가 아니라 비-사유다.**

### (d) ★**복사해야 할 프레이밍 문장** `CONFIRMED`(초록 축자)

> **Serani, Dragone, Stern & Diez (2023), *Ocean Engineering* 267, 113235**:
> *"…based on dynamic mode decomposition (DMD)… **Its use for ship operating in waves has been little
> discussed and a systematic analysis of its forecasting capabilities is still needed in this context.**"*

기여 = **이식 + 체계적 평가**. 새 방법을 주장하지 않는다. ⟹ **이 문장 구조를 그대로 쓸 것.**

### (e) ★★★**VARPRO 는 해양공학에 적용된 적이 없다 — 그런데 자매 분야에는 확립돼 있다** `CONFIRMED`(이중 확인 음성)

두 독립 검색자가 Crossref(`container-title:Ocean Engineering` 필터 포함) · OpenAlex · Semantic Scholar 에서
**해양파 재구성 또는 해양공학에 대한 variable projection / separable NLS 적용을 찾지 못했다.**
방증: Golub & Pereyra 2003 리뷰가 스스로 나열한 응용 분야 = *"electrical engineering, medical and
biological imaging, chemistry, robotics, vision, and environmental sciences"* — **해양/해사 부재**.
⚠Scopus/WoS 전문 접근 없음 ⟹ **범위 한정 부재 증거**.

★**그런데 지구물리 전파형 역산(FWI)에는 확립돼 있고, 그게 완벽한 계보 인용이 된다**:
- **Li, Rickett & Abubakar (2013), *"Application of the variable projection scheme for frequency-domain
  full-waveform inversion"*, *Geophysics* 78, R249–R257**, DOI 10.1190/geo2012-0351.1
  ⟹ ★**제목이 문자 그대로 "Application of…" 이고 그 분야 최상위 저널이다. 수사적 선례.**
- Rickett (2013), *Geophys. Prospect.* 61:874–881 · Aravkin & van Leeuwen (2012), *Inverse Problems* 28:115016
  · van Leeuwen & Mulder, *Geophysics* DOI 10.1190/geo2017-0700.1

⟹ ★★★**정직하고 강한 기여 문장이 나온다**:
> *"Variable projection is standard in seismic full-waveform inversion but has not been applied to
> phase-resolved ocean wave field reconstruction. We transfer it, and we measure why it works."*

### (f) ★★★**합성 전용 검증은 게재 가능하다 — 그리고 08-14 의 전제가 반증됐다** `CONFIRMED`

- ⛔**"Ocean Engineering 이 검증 없는 수치 논문을 데스크리젝트한다" 는 1차 출처가 없다.**
  Guide for Authors 전문(아카이브 45,315자)에 **검증 조항이 아예 없다**("validation" 은 CRediT 역할 목록에만).
  그 주장의 유일한 추적 가능 출처는 **투고 전 점검 서비스 업체의 무출처 블로그**다.
- 유일한 1차 출처 검증 조항은 **"Automatic Control of Marine Systems" 소분야에만** 붙고,
  축자로 *"should include the validation of results **preferably**, by full-scale measurements or
  model-scale tests **or, when this is not possible, by high fidelity simulations.** Methods used in
  papers should be validated in **conditions that represent realistic marine environments**."*
  ⟹ ★**실측 요구가 아니라 "현실적 해양 환경" 요구다.**
- ★**같은 문제·같은 저널의 합성 전용 선례가 있다**:
  **Ehlers, Klein, Heinlein, Wedler, Desmars, Hoffmann & Stender (2023), *Ocean Engineering* 288, 116059**
  — *"Machine learning for phase-resolved reconstruction of nonlinear ocean wave surface elevations
  from sparse remote sensing data"*. **100% 합성**(HOS + 기하 레이더 모형), 전문에서 `tank` 0회 ·
  `field measurement` 0회. **그리고 Desmars JFM 975 A8 도 합성 전용이다**(§3.1(b)).
- ⚠**정직한 균형추**(에이전트 자기 신고): 표면화된 위상분해 예보 논문의 **대략 절반은 수조/실측을 갖는다**.
  합성 전용은 **게재되지만 다수 양식은 아니다**.

### (g) ★★★**그렇다면 합성 전용에 **무엇이** 따라붙는가** — 이게 실행 항목이다 `CONFIRMED`(Ehlers 2023 전문 + 확인 4편)

| 요건 | 저장소 상태 |
|---|---|
| ★**공동체 표준 지표 + 기성 문턱** — **Surface Similarity Parameter**(Perlin & Bustamante 2014), **SSP ≤ 0.10** | ⛔**저장소는 `nRMSE` 만 쓴다. SSP 0회.** |
| 선언된 held-out 시험집합 | ✅S13 이 라우터 미노출 시드로 함 |
| 지배 물리 파라미터 평면 전반 스윕(*"to avoid any possible incidental observations"*) | ⚠BF4–BF5 · 15/30 성분뿐 |
| 비교 팔 최소 1개 | ✅`ZERO`(W4/W5), `SPEC25`(S16/S17) |
| 현실적 센서 전방 모형 | ✅`wavesim` LiDAR(광선·차폐·raydrop) — **이건 강점이다** |
| 관측가능성별 오차 분해(그림자 vs 가시) | ✅**9셸 분해가 구조적으로 같은 수** |
| 잡음 강건성 | ⚠미측정(Ehlers 에도 없었다 ⟹ 필수 아님) |

★★★**가장 값싸고 가장 큰 한 줄**: **SSP 를 `nRMSE` 와 병기하라.**
그리고 **SSP 가 존재하는 이유가 정확히 이 저장소의 발견과 같다** — 유클리드 지표가
*"두 표면의 **주파수 또는 위상** 차이를 **진폭** 오차로 취급한다"* 는 것.
⟹ ★**`decorr`/`shrink` 분해(§2.2)와 SSP 는 같은 문제의식이고, SSP 는 그 문제의식의 공동체 표준 이름이다.**
저장소가 자기 언어로 재발명한 것을 **분야의 언어로 번역하면 리뷰어가 즉시 알아본다.**

### (h) ⚠**스코프 — Coastal Engineering 은 피할 것**

`CONFIRMED`: CE 핵심 = *"waves, currents and sediment transport; coastal, estuarine and offshore
morphology; design of coastal and harbour structures"*. 원양 선박 탑재 센싱은 구조물·지형·퇴적 어디에도 안 닿는다.
⚠**단 반례가 하나 있다** — §3.1(c) 의 Wang et al. 2025 가 바로 CE 에 실렸다.
⟹ **안전한 표적 = Ocean Engineering · Applied Ocean Research.**

⚠**두 개의 진짜 위험**(방법 나이가 아니라):
1. ★**Ocean Engineering 이 "Ocean environmental predictions" 항목을 *"only for ocean structures and
   vessel operation and design"* 로 울타리 치고, 광의의 환경·센싱·자료집합은 해양학 저널로 보낸다.**
   ⟹ **"선박 운용" 논문으로 프레이밍해야지 "해양 센싱" 논문으로 쓰면 안 된다.**
2. ★**Applied Ocean Research 의 applicability 조항이 저장소에 가장 아프게 물린다** — 기여 ②c 미측정 +
   초기화 8.5–24 s. ⟹ **AOR 에 낼 거면 §2.2 의 초기화 3.17배 단축이 서두에 와야 한다.**

## 3.4 ⚠**용어 정정 — "초해상 영역"은 틀린 라벨일 가능성이 높다**

`W4/PREREG.md:36` 이 *"혼잡 성분 분리 **1.29 Rayleigh** vs 250 m 외삽 요구 **0.07 Rayleigh** ⟹ 초해상 영역"*
이라 적었고, 이 문장이 W4·W5 의 인용금지 목록(*"혼잡은 초해상 영역"*)에 그대로 실려 있다.

`ARGUED`(문헌 에이전트 확인에 기초): **이 라벨은 부호가 반대다.**
- Rayleigh 한계는 개구 `L` 에 대해 `Δk = 2π/L`. 분리가 **1 Rayleigh 미만**일 때 초해상이다.
- **1.29 Rayleigh 는 고전 한계 *위*** = 통상적 주기도형/빔포밍으로도 분해되는 영역이다.
  (다만 Candès–Fernandez-Granda 류 볼록완화 보증이 요구하는 `Δ ≥ 2` 에는 미달 ⟹ **제3의 영역**.)
- 그리고 **1.29 와 0.07 은 같은 물리량이 아니다** — 앞은 성분 간 **분리(resolution)**, 뒤는 250 m 외삽에
  필요한 `k` **정밀도(precision)**. 둘을 같은 Rayleigh 단위로 병기하면 리뷰어가 반드시 지적한다
  (Li & Tang 의 논문 제목 *"A Tale of Resolution and Precision"* 이 정확히 이 혼동 때문에 존재한다).
- `UNVERIFIED`: *"X Rayleigh vs Y required"* 형식이 초해상 영역을 서술하는 관행이라는 문헌 근거는 **찾지 못했다.**

⟹ ★**정직한 재서술**: *"성분은 고전 분해 한계 위에서 분리돼 있다(1.29 Rayleigh). 벽은 분해가 아니라
**외삽 정밀도**다 — 250 m 외삽은 Rayleigh 단위의 0.07 배에 해당하는 `k` 정밀도를 요구한다."*
★**그리고 이 재서술은 저장소에 유리하다**: 정밀도는 SNR 로 사는 양이라 분해 한계처럼 원리적 벽이 아니고,
**"정보는 충분한데 솔버가 못 쓴다"(M1 의 오라클-k `lstsq` ρ=0.999964)와 정합적**이다.
⛔반대로 "초해상"이라고 쓰면 **리뷰어에게 "그럼 원리적으로 불가능한 것 아니냐"는 무기를 쥐여 준다.**

---

# 4. 가장 강한 판매 문장 + 심사위원의 첫 질문 3개

## 4.1 내가 고르는 문장

> ★**"진폭·위상을 해석적으로 소거하고 파수에서만 하강하도록 초기화를 바꾸면, 혼잡한 바다에서
> 필드 오차가 중앙 25.3% 낮아지고(58/64, 9셸 전부) 초기화가 26.8 s 에서 8.5 s 로 줄며,
> 프레임당 갱신 비용은 9 ms 로 변하지 않는다 — 그리고 우리가 사전에 범인으로 지목해 둔 양
> (위상 decorrelation, 잔여 오차의 99%)이 실제로 52% 줄었다."**

**왜 이것인가** `ARGUED`: ①기준선과 무관 ②정확도와 비용이 **같은 방향** ③기제가 **선행 배치가 미리
지목한 양**으로 확인 ④전 구간(9셸) — 이 저장소를 여섯 번 물었던 `SUPPORT-TRADEOFF` 가 **없다**.
⛔대안으로 `−46.23%`(vs SPEC25) 를 고르고 싶겠지만 **게이트가 없다**(§2.4). 고르면 안 된다.

## 4.2 심사위원이 그 문장에 던질 첫 질문 3개

⚠**문헌 검증 후 순서를 바꿨다.** 원래 나는 방향분산을 1번에 놓았는데, §3.3(f)(g) 를 확인한 뒤
**"모델 동일성"이 1번**이 됐다 — 이 하위문헌의 합성 검증 표준이 **HOS 기반 비선형**이기 때문이다.

**★Q1 — 모델 동일성 (가장 위험한 질문)**
> *"당신의 GT 는 선형 성분의 합이고 당신의 추정기도 선형 성분의 합이다 — 당신 자신의 감사가
> `max|Δ| = 0.000e+00` 으로 **비트 동일**임을 측정했다. 그런데 변수투영은 **바로 그 모델 형태를 가정하고**
> 진폭과 위상을 해석적으로 소거한다. 그렇다면 −25.3% 중 얼마가 '더 나은 솔버'이고 얼마가
> '완벽한 모델 일치를 더 잘 이용하는 솔버'인가? HOS M=3 에서는 어떻게 되는가?"*
- ★**이 질문은 가설이 아니라 이미 부분적으로 답이 나와 있다**: HOSGATE 가 비선형화 비용이 **팔마다 다르다**고
  측정했다(`SHIP` 희소 절대오차 **3.6배** vs `LETKF+`·`ZERO` **±0.5% 무감**).
  ⟹ **VARPRO 는 구조적으로 가장 노출된 팔이다.**
- ⛔**현재 답이 없다.** W4·W5 는 **선형 GT 전용**이다.
- ⟹ ★**§5 배치의 PRIMARY 를 HOS 팔에 놓는 이유가 이것이다.**

**Q2 — 해상상태 지지**
> *"당신의 64 에피소드는 25,426 개 실측 부이 레코드 기준 방향분산 분포의 **6.7 퍼센타일**에 있다
> (σ₁ 중앙 22.0° vs 실측 35.3°). 방향분산이 실측 중앙값이 되면 성분들이 파수 공간에서 덜 분리되는데,
> 진폭·위상을 소거하고 파수에서만 하강하는 방식은 바로 그 분리에 가장 민감한 방식 아닌가?
> 그리고 당신 코퍼스에는 스웰이 없다."*
- **현재 답할 수 있는 것**: 사후 층화에서 **부호가 반대 방향으로 움직인다**(σ₁≥35.3° 에서 10/10, −36.3%;
  BF×성분수 네 셀 각각의 고분산 절반에서 32/32). ⚠탐색적, n=10.
- **완전한 답에는 §5 의 배치가 필요하다.** ⟹ **이것이 논문의 생사를 가르는 질문이다.**

**Q3 — 요인 귀속**
> *"`L-VP-ONLY` 는 결합 적합만 뺀 팔이 아니다. 당신 자신의 §4 가 성장 분기 발화가 **47/192 → 141/192
> 로 3배**가 된다고 적었고, `clamp_steepness` 투영도 사라진다. 25.3% 중 VARPRO 의 몫은 얼마인가?
> 그리고 W4 에서는 가중이 유의한 요인이었는데(`S-2c` p=0.008) W5 에서는 널이다(p=0.060) —
> 요인 구조가 배치 간에 안정하지 않다."*
- **현재 답**: `S-1`(`L-VP-ONLY < L-ITER0` 60/64, −42.2%)이 *"VARPRO 가 결합 적합을 잃은 팔을 구조한다"*
  를 보인다 ⟹ VARPRO 가 필요조건임은 산다. ⛔**그러나 지분은 미측정이고, 저장소가 스스로 귀속을 금지했다.**
- **처방**: 성장 분기를 **고정**한(발화율을 양 팔에서 강제로 맞춘) 팔 하나면 이 질문이 닫힌다.

**(Q4 — 셋을 넘어가지만 반드시 온다) 창에서 필드로의 전이, 그리고 실질**
> *"창 잔차는 0.0263 인데 필드 nRMSE 는 0.1200 — **4.6배**다. 당신의 닫힌형 상한 대비 여전히 **×81** 이고,
> 250 m 외삽은 Rayleigh 단위의 0.07 배 정밀도를 요구한다. 그렇다면 25% 는 조종·제어에서 **무엇을
> 바꾸는가**? 당신 저장소는 혼잡에서 제어 지표 α_wave 가 무기술 널 대비 17–22% 뿐이라고 이미 측정했다."*
- ⛔**현재 이 질문에 답이 없다.** W4/W5 는 α_wave 를 재지 않았다.
- `ARGUED`: OE 는 "정확도 개선"만으로도 받지만, **초록에 제어를 언급하면 반드시 이 질문이 온다.**
  ⟹ **처방: 초록에서 제어를 빼라**(내 §1.4 초록은 뺐다). 또는 §5 배치에 `α_wave` 를 기술통계로 붙여라.

---

# 5. 어떤 추가 측정 하나가 논문을 가장 크게 올리는가

## ★권고 — **`S18 · REAL-REGIME`: 현실적 해상상태 + 비선형 GT 위에서 VARPRO 팔을 기준선과 함께 재측정**

⚠**문헌 검증 결과가 이 권고의 우선순위를 바꿨다.** 원래 나는 방향분산 하나만 보려 했다.
그런데 §3.3(f) 가 밝힌 검증 조항의 실제 문언이 **"realistic marine environments"** 이고,
같은 하위문헌의 합성 전용 선례(Ehlers 2023 · Desmars 2023)가 **전부 HOS 기반 비선형 합성**이다.
⟹ ★**저장소의 최대 검증 취약점은 방향분산이 아니라 "GT 가 추정기 모델식과 비트 동일(max|Δ|=0.000)" 이고,
그리고 VARPRO 는 구조적으로 그 동일성에 가장 많이 기대는 팔이다**(진폭·위상을 **모델식 가정 하에**
해석적으로 소거한다). HOSGATE 가 `SHIP` 희소에서 비선형화 비용 **3.6배**를 이미 봤다.
⟹ **두 축을 한 배치에 넣되, 비선형이 주(主)다.**

**한 배치가 다섯 가지를 동시에 산다.**

| 사는 것 | 현재 상태 |
|---|---|
| ① ★**"현실적 해양 환경" 검증 조항 충족** — 분야의 합성 전용 표준(HOS) | 선형 GT = 모델식과 **동일**, 실측 불일치 0% |
| ② **해상상태 외적 타당성** — 방향분산 6.7 → ~50 퍼센타일 | `S14-SPREAD` 08-14 설계 후 **미실행** |
| ③ **VARPRO 팔 vs 외부 기준선** — 정식 게이트 하에서 **최초** | §2.4, 지금은 게이트 없는 교차배치 추정뿐 |
| ④ **신선 표본** | S8+S9 는 S13·S16·S17·W3·W4·W5 가 전부 사용 |
| ⑤ **기제의 사전등록** — `decorr`/`shrink` 를 `D-5` 로 | 현재 `AUDIT_PRO` 의 사후 관찰 |

### 설계 (S13/W5/HOSGATE 에서 그대로 가져온다 — 새 자유도 최소화)

| 항목 | 값 |
|---|---|
| **생성기 변경 2개** | ①방향분산 목표 **중앙 σ₁ ≈ 35°**(현재 22.0°) + `Tp` 상한 **9 → 14 s** ②**HOSGATE 의 짝 설계** — 같은 성분에서 **(선형 GT, HOS M=3 GT) 쌍**(`config.hos.enabled`). ⛔`wavesim/**` 무수정 |
| **코퍼스** | 혼잡 **64 에피소드** × **GT 2종 = 128**, **신선 에피소드 시드**(교집합 0 확인 필수) |
| **추정기 시드** | median-of-3, 잔여 원장 **[1598–1818]** |
| **팔 4개** | `L-VP-ONLY` · `L-SHIP`(배포, `learned` 고정) · **`SPEC25`**(S16/S17 최강 수리 기준선, 동일 구성) · `ZERO` |
| **런 수** | 64 × 2 × 3 × 4 = **1,536런** |
| **PRIMARY (2개, Holm-2) — 전부 `HOS` 팔에서** | **P-1** `L-VP-ONLY < L-SHIP` ≥41/64 · **P-2** `L-VP-ONLY < SPEC25` ≥41/64 (★**저장소 최초의 "처방 + 기준선" 동시 검정**) |
| **IUT** | `L-SHIP < ZERO` **및** `SPEC25 < ZERO` ≥41/64 — ⚠**양 팔 모두**(08-14 P4 가 기준선의 제로필드 4/64 패배를 잡았다) |
| **SECONDARY** | **S-1** 비선형화 비용 `HOS vs LIN` **팔별**(★HOSGATE 가 팔마다 다르다고 측정했다 — VARPRO 가 가장 노출된 팔인지가 핵심) · **S-2** 셸 9개(`obs:` 접두어 명시) · **S-3 `D-5` = `decorr`/`shrink` 분해**(★사전등록) · **S-4 `init_time_s`** |
| ★**지표** | **`nRMSE` 와 `SSP`(Perlin & Bustamante 2014) 병기, 문턱 `SSP ≤ 0.10` 을 사전등록**(§3.3(g)). ★**이 한 줄이 논문을 분야의 언어로 옮긴다.** |
| **필수 게이트** | ① ★**메시-Nyquist 사전검사** — `Tp` 14 s 는 07-23 메시 이슈를 재개방한다. `mesh_verdict` 를 **런 전** 층별 인쇄, `INADEQUATE` 셀은 **사전등록 시점에 배제**(HOSGATE 의 `DEGRADED` 교훈) ② ★**팔별 필드 존재 검사**(S13·HOSGATE 가 기준선 셸을 **두 번** 못 쟀다 — `S15` 의 `G-8/G-8b` 를 본보기로) ③ **분석기를 자료 보기 전에 동결 + 코드 미열람 독립 검증기**(W4 방식; W5 가 이걸 안 해서 §2.2 의 JSON 결함이 났다) ④ ★**동결 산출물 ↔ 인쇄물 대조 게이트**(내가 찾은 `iut` 섀도잉) ⑤ 임포트 경로 파일 **md5 동반 기록** ⑥ **개입 기대값 단일출처화**(W4 가 하루에 네 번 물린 그것) ⑦ **짝 무결성** — LIN/HOS 두 에피소드의 성분이 비트 동일(HOSGATE 프로브 재사용) |
| **사망조건** | `P-1` **HOS 에서** 실패 ⟹ ★**"VARPRO 처방은 표현 일치의 산물"** 확정 ⟹ 논문 골격을 §1.4 대안 제목(기제 논문)으로 후퇴하고 **정확도 주장을 뺀다**. `P-2` 실패 ⟹ 기준선 비교를 논문에서 **완전히 뺀다**(절제 논문 확정) |

### 비용 `MEASURED` 기반 추정

- **생성**: 선형 `data_s8_crowded/_manifest.json` 실측 **167 s/에피소드**;
  HOS 는 `project_hos_gate_and_training` 실측 **115 s/에피소드**.
  128 에피소드 ≈ **5–6 h**(직렬) ⟹ 병렬로 **1.5–3 h**. ⚠`Tp` 14 s 가 도메인·메시를 키우면 **상향**.
- **런**: 실측 중앙 wall = `L-VP-ONLY` 14.5 s · `L-SHIP` 29.7 s · `ZERO` 0.2 s · `SPEC25` **미측정**(≈30–60 s 가정).
  직렬 합 ≈ 64×2×3×(14.5+29.7+0.2+45) ≈ **34,400 s ≈ 9.6 h** ⟹ **8워커 ≈ 1.5–2.5 h**.
- **총계 ≈ 8–14 h**(생성 포함, 사전등록·감사 제외). 시드 소모 3개.

⚠**축소판이 필요하면**: HOS 팔만 돌리고 LIN 을 빼면 768런(≈5–8 h)이지만 **비선형화 비용의 팔별 분해(S-1)를
잃는다.** ⟹ ★**권하지 않는다** — 그 분해가 HOSGATE 에서 가장 값진 산출이었다.

### ⛔이것보다 낮은 우선순위인 것 (이유 병기)

| 후보 | 판단 |
|---|---|
| **실측 전이**(Bergamasco 5 레코드) | ⛔**유의성 배치로 설계 불가** — 독립 단위가 레코드 5개 ⟹ 클러스터 바닥 p = 2⁻⁵ = **0.031**. 2026-08-07 이 표준집합 4개를 죽인 바로 그 결함. ★**그리고 §3.3(f) 가 그 급함을 낮췄다** — 같은 하위문헌의 최상위 선례(Ehlers 2023 · Desmars 2023)가 **합성 전용**이다. **필요하지만 게재의 전제조건은 아니다** |
| `direction_deg=180`(조우 기하) | ★중요하고 **박사 골격에는 필수**지만 **OE 논문의 생사를 가르지 않는다**(정확도 논문은 nowcast 로 성립). ⚠단 §3.1(e) 의 **Yao et al. 2025 (이동선 · 레이더 · 위상분해)** 가 이미 나와 있으므로 **박사 쪽 시계는 빠르게 간다** |
| **성장 분기 고정 팔**(Q2 귀속) | ★**싸고 좋다**(≈192런). ⟹ **§5 배치에 5번째 팔로 끼워 넣는 것을 권함** |
| `L-VP-ONLY` 를 배포 라우터(`auto`)에서 재측정 | 필요(현재 `learned` 고정). **논문 투고 전에는 반드시**, 그러나 §5 다음 |
| ★**런 0회 항목 4개(오늘 당장)** | ⑴**SSP 지표 도입**(§3.3(g)) ⑵**§7 의 문헌 정정 5건을 `MEMORY.md`·`STATUS.md` 에 반영** ⑶**Wang et al. 2025 · Ehlers 2023 · Mérigaud 2025 · Chen et al. 2023 정독** ⑷**"초해상 영역" 용어 수정**(§3.4) |

---

# 6. ⛔저장소가 고쳐야 할 문헌 관련 서술 5건 (런 0회, 오늘 반영 가능)

| # | 현재 서술 | 판정 | 대체 |
|---|---|---|---|
| 1 | *"알고리즘 축은 선점(Desmars JFM 975 A8 · **US 12,054,225**)"* (`MEMORY.md`·`STATUS.md`) | ⛔**거짓** | 그 특허는 **Candela 수중익선 승차감 제어**다(§3.1(a)). 진짜 특허 선행기술은 **Naiad US 10,259,544 / 10,538,299** 이고 **독립항에는 없다** |
| 2 | *"Desmars 가 알고리즘 축을 선점"* | ⚠**과잉** | 그 논문은 **1-D 단방향 · 정지 센서 · 레이더 모형 · 합성 전용 · 비용 분석 0**(§3.1(b)) |
| 3 | *"척추: 문헌의 **CRB 프레이밍**과 모순"* | ⛔**성립 안 함** | 분야의 프레이밍은 **기하학적 predictable zone**(Wu 2004 · Naaijen 2014 · Qi 2018)이다. 그리고 **Mérigaud et al. 2025 JFM 1007 A47 이 실측으로 이미 그것을 깎기 시작했다** ⟹ **논쟁을 여는 게 아니라 합류하는 것**(§3.1(f)) |
| 4 | *"혼잡은 **초해상 영역**(1.29 vs 요구 0.07 Rayleigh)"* | ⛔**부호 반대** | **1.29 Rayleigh 는 고전 한계 *위*** = 초해상이 아니다. 그리고 1.29(분해)와 0.07(정밀도)은 **다른 물리량**이다(§3.4). 정직한 재서술은 **"벽은 분해가 아니라 외삽 정밀도"** 이고 **저장소에 유리하다** |
| 5 | *"VARPRO 는 1973년이라 novelty 0"* | ⚠**사유가 틀림** | 1973년이라서가 아니라 **1996–2020 에 이미 선스펙트럼 추정에 적용**됐고(Besson & Stoica 1999 · RELAX 1996 · ICASSP 2020) **2026-07 에 지형 분석까지 나왔기 때문**(`arXiv:2607.09593`). ★**반대로 해양공학에는 적용된 적이 없다**(§3.3(e)) |

★**그리고 새로 얻은 자산 3건**: **Alnajjab & Blum 2017**(우리 모델 클래스의 CRB 가 이미 있고 미인용) ·
**Jiang et al. 2026 arXiv:2605.25074**(해양파 문헌이 λ 기제를 독립 확인) ·
**Li, Rickett & Abubakar 2013 *Geophysics***(VARPRO 이식의 수사적 선례).

---

# 7. 요약 — 08-14 판정 대비 변경

| | 08-14 | 2026-08-19 |
|---|---|---|
| **게재 판정** | **REJECT** | ★**"배치 1건 추가 후 Ocean Engineering / Applied Ocean Research 투고 가능"** |
| 논문의 축 | `WaveSplat < LETKF`(기준선 인질) | ★**`VP < 배포`(기준선 무관 절제) + 기제** |
| 인용 가능한 크기 | **0개** | ★**1개**(자기 대조 −25.3%, 혼잡 층 · 합성 · 선형 GT · 라우터 고정 한정) |
| 최대 결함 | 방향분산 지지 | ⚠**바뀌었다** → ★**GT 모델 동일성**(합성 전용은 허용되나 **비선형이어야** 한다). 방향분산은 2순위로 내려가고 **탐색적 층화가 유리한 방향**(§2.3) |
| "합성 전용" | 치명적이라 봄 | ⛔**전제가 거짓**(§3.3(f)) |
| "1973년 방법" | — | ⛔**비-사유**(Prony 1795 선례, 기저율 19/20) |
| 방법 novelty | 선점 | ⛔**더 나빠짐**(`arXiv:2607.09593` 2026-07 이 방법+지형 동시 점유) ★**단 해양공학 이식은 미실시** |
| 최근접 경쟁자 | Desmars | ★**Wang et al. 2025 *Coastal Eng.* 197:104690**(저장소가 모름, **실측 보유**) |

★**한 문장 판정**:
> **알고리즘 novelty 는 여전히 0 이고 앞으로도 0 이다 — 그러나 이 분야는 novelty 를 요구하지 않는다.
> 저장소는 08-19 에 처음으로 "실릴 수 있는 논문"의 재료를 갖췄다: 기준선과 무관하고, 비용과 정확도가
> 같은 방향으로 움직이며, 기제가 11일 전 다른 배치가 지목한 양과 일치하는 결과 하나.
> 남은 하나는 그것이 *합성 선형 우주 밖에서도* 서는지이고, 그건 한 배치로 답한다.**

⚠**마지막으로, 심사위원으로서 가장 하고 싶은 말**: 이 저장소의 가장 큰 위험은 측정이 아니라
**자기 문헌 서술이 검증 없이 굳어 온 것**이다. §6 의 5건 중 2건(특허 · CRB 프레이밍)은
**연구 방향 결정에 직접 쓰인 문장**이었고 둘 다 틀렸다.
⟹ ★**상시지시 `feedback_litcheck_before_novelty` 를 "novelty 제안 전"이 아니라
"문헌 문장을 메모리에 쓸 때"로 앞당길 것.**
