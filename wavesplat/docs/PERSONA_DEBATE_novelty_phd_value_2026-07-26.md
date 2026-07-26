# PERSONA DEBATE — novelty and PhD value of the WavePrior/WaveSplat program
### Final synthesis by the JUDGE · 2026-07-26
### 3 rounds × 5 personas · sources: `tmp/persona_debate_2026-07-26/R1_*.md`, `R1_CRUXES.md`, `R2_*.md`, `R3_closings.md` · briefing packet: `docs/HANDOFF_OPUS_2026-07-26.md`

> Adjudication only. This document introduces no new empirical claim. Every number is
> traceable to the packet or to a persona's re-derivation from the session JSONs, and is
> tagged **MEASURED** (re-derivable from result files), **ARGUED** (interpretation), or
> **RULING** (the judge's binding decision on wording, scope, or procedure).
> The do-not-cite list (`HANDOFF §5`) binds this document's own wording.

---

## 1. 요약 (Korean executive summary)

1. **판정 요약: Q1(novelty) = ML 기여로는 만장일치 REJECT, OE/측정과학 기여로는 PARTIAL PASS~ACCEPT. Q2(박사 가치) = 5/5 조건부 PASS.** 무조건 PASS 0명, FAIL 0명.
2. **모두가 서명한 단 하나의 비협상 조건 = 신선 40 에피소드(희소16/중간8/혼잡16), 상수 동결, 단 한 번 채점.** P1·P3·P5는 이것을 각자의 non-negotiable로 명시했고, P2는 합격선을 직접 써 냈으며, P4는 이의 없음. P2는 여기에 "실측 30창 blind 라우터 실행"을, P4는 "Katende 인용 단락"을 각각 하나씩 더 걸었다 — 판정자는 셋 다 최소 조건에 포함한다.
3. **자산 순위 최종(라운드 2~3에서 역전됨): ①부정-기제(proposal-generation OOD + e2e null) ②경계의 위치와 양측 기제(kx/ky 배치 vs greedy basin) ③interlock(실측).** 라운드 1의 순위(①실측 역전 보험)는 P3의 포렌식과 P1의 무통제 지적으로 뒤집혔다.
4. **논지의 대상은 추정기가 아니라 경계/실패 지도다.** 5인 전원 동의. P1의 표현: "이 프로그램의 평균값이 곧 그 실패 양식이며, 그것이 항목별로 열거되어 있다."
5. **죽은 것:** "보험(insurance)"이라는 단어, "3/3", mean 기반 실측 승리, 라우터를 기여로 주장하는 것, Ch.4 아키텍처 신규성(Dekel&Elad·deep unfolding·set-diffusion cardinality로 선점), CRB 척추(부록으로), 16초 예보기, non-inferiority 의례, 그리고 사다리 그림의 0.199 출발점.
6. **0.0875는 held-out 실행 전까지 ARGUED다.** capstone 사슬 어디에도 held-out 집합이 없다. 상수는 "체리픽"이 아니라 **채점 데이터로부터 추정**된 것이고(P3: 고르기 아님·맞추기 맞음), 사전등록 기준 자체가 검정 없는 맨 점 비교였다.
7. **해석 팔의 이름을 붙여라: dispersion-constrained 2D-MNOMP.** "우리가 만든 것을 우리가 이겼다"는 P1의 반론 절반을 공짜로 무력화한다(P4).
8. **두 축 문제(P3, 미해결):** 위상도는 `n_components` 위에 그려졌는데 계기는 `n_selected`를 읽고, 이 둘은 **측정으로 분리된다**(1성분 바다에서 n_selected≈30). 관측가능 좌표 위상도가 나오기 전까지 "경계는 GT 없이 검출가능"을 축 A 위에서 주장할 수 없다.
9. **사전등록은 진짜로 작동하고 있음이 검증됐다(가장 강한 신뢰 증거):** 가드 그리드에서 **더 좋은 값(0.04→0.08725)이 아니라 사전등록 값(0.05→0.08747)을 보고했다.** 다만 고칠 수 있는 의례적 결함 2건 — 영-팽창 non-inferiority와 동어반복 기준, 그리고 **빈 구간 안에만 놓인 민감도 그리드**(설계상 무정보) → 전부 **마진 보고**로 교체.
10. **토론이 자기 브리핑 팩을 반증했다:** 3/3→2/3(동결 기준), mean 기반 TL;DR 줄(중앙값 불변), oracle 집계 불일치(배치 0.17982 < "상한" 0.18053). 앞의 둘은 핸드오프에 이미 정정됨. **셋째는 아직 §1에 남아 있다** — "oracle 간극 2/3 봉합"은 한 스크립트로 재계산하기 전까지 **무근거**.
11. **판정자가 추가로 확인한 팩 결함 1건:** 핸드오프 §2는 라우터를 `n_selected >= 6`이라고 적었지만 **실제 배포 기본값은 `auto_switch_thresh = 8`**(코드 확인). 수치 영향은 미미(0.09790 vs 0.09786)하나, 동결 실행 전에 상수를 한 문장으로 확정해야 한다.
12. **실측 아레나의 정직한 이름은 "진짜 바다, 합성 센서"다** — WASS 재구성면 위의 가상 fan, GT가 같은 재구성물, raydrop/dropout/자세오차 없음, 설계점(11.3 m/s) 밖(1.2–2.4 m/s). 초록에 넣어야 한다.
13. **실측 붕괴 성분의 정직한 이름은 interlock**(관측만으로 재구성 실패를 검출하고 오차 크기를 제한하는 유효성 플래그)이고, 채점 단위는 nRMSE 감소가 아니라 **동작특성(operating characteristic)**이다. 승격된 것이 아니라 **재분류**된 것이다.
14. **코드와 논문이 지금 서로 다르다**(코드는 플립됨, `wavesplat.tex`는 0.199/0.223과 버그 시대 서술 유지) = 살아 있는 무결성 노출. **집필 전에 봉합**, Paper 1 먼저.
15. **일정(P5, 판정자 승인): 15 작업일 스프린트 — 재학습 0, 새 데이터 0, FNO 0 — 이후 집필 ~4개월, 방어는 12개월 현실적·15개월 여유.**

---

## 2. Process & scorecard

### 2.1 Who argued what

| | Persona | Core lens | Single biggest contribution to the debate |
|---|---|---|---|
| P1 | Harsh ML reviewer | Is there a contribution an ML venue would accept? | The **two-arena contradiction** ("one estimator, one config" is true on synthetic and untested on real); the arm-swap crux |
| P2 | Senior ocean-engineering professor | Would this survive an OE defense and a sea trial? | Killed the word "insurance" and supplied **interlock**; the ledger reading; the **discreteness** hypothesis; the 13-record collapse harvest |
| P3 | Thesis-committee methodologist | Is the rigor load-bearing or costume? | **Falsified the packet three times**; verified prereg is real (reported-the-worse-value); the **two-axes** charge; machine-generated evidence maps |
| P4 | Literature hawk | What is already published? | Killed Ch.4's architecture novelty and the router-as-contribution; **named the analytic arm 2D-MNOMP**; ranked the assets |
| P5 | Pragmatic advisor | What ships, in what order, by when? | The 8-chapter spine; the 15-day sprint; **promoted the negative-mechanism chapter to co-headline** |

### 2.2 Verdict trajectory (Q1 novelty / Q2 PhD value)

| Persona | R1 | R2 | R3 (closing) | Net movement |
|---|---|---|---|---|
| **P1** | Q1 MAJOR REVISION → REJECT as framed; Q2 CONDITIONAL PASS (OE yes, ML no) | Q1 REJECT-as-ML *hardened*, MAJOR REVISION as OE; Q2 CONDITIONAL PASS, **confidence up** | Q1 ACCEPT as OE **reframed as a failure map**; Q2 CONDITIONAL PASS, "the conditional is one experiment wide" | Q1 label harder, Q2 **warmer** — inverted his own "average hides failure modes" belief |
| **P2** | Q1 PARTIAL PASS; Q2 PASS w/ 3 conditions | Q1 PARTIAL PASS *downgraded*; Q2 PASS w/ 4 conditions | Q1 PARTIAL PASS; Q2 PASS conditional; **weighting reversed** (real chapter demoted, synthetic promoted) | Stable verdict, **inverted weighting** |
| **P3** | Q1 split earned/not-earned; prereg "70% real / 30% ritual"; Q2 PASS only under runtime-detection framing | Circularity charge **softened** (tautology → unvalidated in-sample correlation); prereg raised to **75/25**; new decisive finding | Q2 PASS as measurement science; **never** with Ch.5 written before the fresh 40 | Softened the attack, **hardened the procedure** |
| **P4** | Niche degraded; assets ①accuracy cross ②rescue inversion ③proposal OOD | Ranking **inverted**: ①negative-mechanism ②boundary ③interlock; Ch.4 architecture declared fully preempted | Q1 survives on three negatives; **"the dissertation IS the negative-mechanism chapter"** | Largest single move in the debate |
| **P5** | Q1 spine "defensible **today**"; Q2 ~2 weeks of closes | **Withdrew "defensible today"** → after the sprint; Ch.5 headline must be reworded | No longer believes Ch.5 is the thesis; **Ch.6 promoted to co-headline**; defense realistic at 12 months | Most self-corrected |

**Convergence:** all five ended at *conditional pass on Q2* and *"the object is the boundary/failure map, not the estimator"* on Q1. They never agreed on a Q1 label because they were not using one rubric (see §5e).

### 2.3 The debate falsified parts of its own briefing packet

This is the most quotable process fact of the exercise: the packet written **to prevent** criterion drift contained criterion drift, and adversarial reading found it in one day.

| # | Debate-caught error | Found by | Status |
|---|---|---|---|
| E1 | **"repairs 3/3"** — the frozen criterion is `learned < 1.0`; the scorer's own output reads `AA14 w0: 2.309 -> 1.071 [not fixed]. Result: 2/3`. End-to-end (detector ∘ fallback) brings **1 of 3** below the line, and the one window with a genuinely useful repair (1.12→0.79) is the one the detector misses. **MEASURED** | P3 (R1) | **Corrected in `HANDOFF` §0 + §3** (🛑 CORRECTION block). MEMORY.md still carries "전부 수리" — sweep it. |
| E2 | **The mean-based TL;DR line** — "30-window mean 0.881→0.775" quoted as the deployed result, while the **median is unchanged (0.8091→0.8091)** and two windows drive the entire delta. This violated the packet's own do-not-cite rule §5 ("mean 기반 실측 승리 주장 금지"). **MEASURED** | P3 (R1) | **Corrected in `HANDOFF` §0 + §3.** Not quotable in any form except with the median stated. |
| E3 | **Oracle aggregation inconsistency** — the crowded slice of the deployed config (0.17982) is **below** its own "oracle upper bound" (0.18053), because the oracle is per-episode min-of-medians while the switch is per-lane-then-median. Therefore **"the guard closes ~2/3 of the oracle gap" is unsupported** until one script computes both the same way. **MEASURED** | P3 (R1), elevated by P5 (R2) | **NOT yet corrected** — `HANDOFF` §1 line 42 and MEMORY.md still assert it. Day-1 worklist item W1b. |
| E4 | **Router threshold, doc vs code** — `HANDOFF` §2 describes the router as `n_selected >= 6` and §1 sources the 0.0979 rung from `by_threshold["6"]`, but the shipped default is `auto_switch_thresh = 8` (`sequential_update.py:508`), which is also the constant the guard prereg used. Numerically immaterial (0.09790 vs 0.09786, the grid is flat) but the frozen run cannot start on an ambiguous constant. **MEASURED (judge-verified from source)** | Judge (not debated) | Open — W1b. |

P3's structural reading of E1–E3: **the summary layer carried 2 material errors in a 7-row evidence map (~29%), both propagated to MEMORY.md.** The ranking layer has a clean record (17 discards audited, 0 unjustified); the *narrative* layer does not. The catch rate is high under commissioned adversarial reading and low in the default loop — which is precisely why W1a (machine-generated evidence maps) is in the worklist as a build step and not as advice.

---

## 3. CONSENSUS FINDINGS — all five personas agree

**C1 — The real-data collapse component is renamed: "insurance" is retired; the component is an INTERLOCK.**
**RULING.** Honest name (P2, adopted verbatim by P3/P4/P5): *GT-free reconstruction-failure detection with bounded-error degradation* — functionally a validity flag with graceful degradation, in the same family as X-band quality masks and buoy QC flags. **The scoring rule changes with the name: score it as an operating characteristic on detecting unusable output, never as an nRMSE reduction** (P3: "delete, do not caveat, every accuracy-flavored number attached to it"). Supporting facts, **MEASURED**: the analytic arm's failures are separated from every normal window by an empty margin (fires at 2.40 / 2.53 vs normal max 1.20, τ=1.5); learned reduces field error on the three collapse windows (2.31→1.07, 1.12→0.79, 2.93→0.96); 0/27 false alarms at an in-sample τ; under the frozen criterion the count is 2/3 and end-to-end 1/3; repaired fields land at ≈0.96–1.07, i.e. at or near a no-skill predictor. **ARGUED:** the demonstrated benefit is *bounding the magnitude of a wrong answer*, not restoring skill — which is operationally real, because a confidently-wrong field commands the wrong actuator.

**C2 — The thesis object is the boundary / failure map, not the estimator.**
**RULING**, unanimous and the single largest reframing of the debate. P1's closing formulation, which P2/P3/P4/P5 all endorse in their own words: *"a measured, GT-free-decidable failure map with named failure mechanisms on both sides."* The estimator is the instrument that makes the boundary claim falsifiable; the router is not the claim. Corollary all five accept: the **transition width** (the 5-component cell, 6 losses / 4 wins) is a measurement to report, not an embarrassment to hide; and the oracle gap's **concentration** (46% of it in one episode, 91% in eight, the failing episode's signature nameable as bf_peak_ratio ≈ 0.19, above any zero-crowded-flip threshold) is a diagnostic capability rather than a defect. **MEASURED** for the concentration and cell counts; **ARGUED** for "diagnostic capability is what a doctorate certifies."

**C3 — The negative-mechanism asset is promoted to co-headline.**
**RULING** (P5 R3 proposed, P4 R3 went further — "the dissertation *is* the negative-mechanism chapter"; P1/P2/P3 concur it is the least-preempted item). Two **MEASURED** components: (i) **proposal-generation OOD** — on `BF2_1comp` the network does not emit atoms near the true swell at all, so no post-processing can repair it; two pre-registered parity rounds failed, and peak-anchor post-processing is crowded-harmful and must not be deployed; 35/40 non-catastrophic episodes were already at parity (0.0198 vs 0.0193), the entire gap is 5 episodes. (ii) **the e2e fine-tune null** (medΔ −0.0007, p=0.30), with the leak localized to the observation-distribution gap (random-footprint training vs LiDAR-sweep deployment). **ARGUED** (P4's literature basis, snippet-limited on two paywalled sources): the learned-greedy / deep-unfolding line reports only wins and contains no atom-level account of a generation-side failure that post-processing provably cannot repair; the e2e null is a directly citable negative against that line's central premise.

**C4 — The 0.0875 capstone is ARGUED until a held-out run exists.**
**RULING**, unanimous. Three independent reasons, all **MEASURED**: (i) there is **no held-out set anywhere in the capstone chain** — the router threshold, the guard ratio, the guard *signal choice* (bf_peak_ratio was the third candidate examined), and the real detector τ were all set by inspecting the same 100 episodes / 30 windows on which the results are reported; (ii) the pre-registered criterion was a **naked point comparison** (switch mean ≤ min(pure means)) with no test and no interval — the capstone claim has never been subjected to inference; (iii) P3's precise distinction, which both R1 readings got wrong: *selection among reported candidates did not happen* (they reported the worse pre-registered value), but *estimation of the constant from the scoring data did* (0.05 was placed inside the measured empty gap between the crowded floor 0.0553 and the misroute cluster 0.024–0.030 **before** the sensitivity grid was run). **Fitting is not shopping — and the program's seven disciplines ban the first and have no rule against the second.** Until W4 returns, the phrase "one estimator beats both pure arms" is not available in any document.

**C5 — Name the analytic arm: dispersion-constrained 2D-MNOMP.**
**RULING** (P4 R2; unopposed). The greedy beamform-plus-refine arm is not home-made — it is a dispersion-constrained instance of 2D Newtonized OMP (Mamandipoor et al. arXiv:1509.01942; MNOMP arXiv:1802.01266). Naming it converts "we beat our own prior" into "we beat a recognized continuum-greedy standard" at zero experimental cost, and it retires roughly half of P1's objection 6.

**C6 — The router is not claimable as a contribution; the preemption literature must be engaged.**
**RULING** (P4 R2 final, P1 concurring, P2 conceding the class is occupied). The pre-solve / post-solve distinction (input-side regime identification vs a-posteriori solution certification) is a difference of **degree, not kind**: a referee places this router inside Katende's "certification-and-selection layer" as a cheaper, *uncertified* instance. What survives is one sentence of engineering delta (compute allocation under a ≤100 ms budget: the router does not have to run the learned solver, 279 ms vs 504 ms) and, for OE venues only, an **application gap** (validity-flag culture gates *data*; nothing published routes between *estimators* in phase-resolved reconstruction). Mandatory citations: **Katende arXiv:2606.07153** (most dangerous), **Kachhadiya arXiv:2605.13068**, **Dekel & Elad arXiv:2010.07069 / CVIU 2021**, **arXiv:2505.15661**, **arXiv:2405.10106**. P4's verbatim Related-Work sentence — the only version P1, P3 and P4 all sign:

> "We do not claim the selection layer as a contribution. We claim the measured location of the boundary it exploits, and we report both its false-alarm rate and its misroutes in the regime where the proxy decouples from crowding."

**C7 — The two-axes problem: the phase diagram is drawn on one variable and the instrument reads another.**
**MEASURED** decoupling, **RULING** on what may be claimed. The regime map's x-axis is `n_components` (a generator knob); the deployed router reads `n_selected` (a statistic of the proposal network's behavior). These are measured to come apart: `n_selected ≈ 30` on a 1-component sea, and `n_eff ≈ 24` on essentially every real window. P3's refined charge — after withdrawing "algebraic tautology" — stands: *an in-sample correlation between a learned statistic and a simulator knob, with the single observed decoupling patched by a second constant fitted to that same decoupling.* Two consequences all five accept: (a) "the boundary is GT-free detectable" cannot be asserted on axis A while the detector reads axis B, until the observable-coordinate diagram (W9) exists; (b) P1's related objection that `n_components` cannot be varied in nature is **partially answered** — the axis is measurable outside the simulator (bf_peak_ratio and entropy are computable on every real window) but is only *varied* inside it, and the honest figure shows real windows clustering in one narrow band of the plane.

**C8 — Pre-registration is verified load-bearing, with two fixable ritual defects.**
**MEASURED** (P3's verification, the strongest trust evidence in the packet and the item that raised P1's Q2 confidence): on the guard sensitivity grid the program **reported the pre-registered value 0.05 (0.08747) rather than the better value 0.04 (0.08725)**; the router grid 6/8/10 is flat (0.09790 / 0.09786 / 0.09819) with no cherry-pick; and the `wp_real` four-cell interpretation table was frozen before the numbers, with the unfavourable cell landing and being published (accuracy LOSS, 6/27, p=0.005, rank-biserial −0.608). **RULING** on the two defects, both fixable by drafting alone:
- **(a) Ritual criteria.** Zero-inflated non-inferiority is structurally near-unfalsifiable (threshold-8 sparse: mean 0.0509 vs reference 0.0204 — 2.5× worse — yet median_delta = 0.0, p = 0.248, `noninf_pass = True`), and the guard's PRIMARY criterion #2 ("0 crowded flips") is a tautology by construction because the threshold was placed below the measured crowded floor. **Retire non-inferiority endpoints; report `n_discordant` per slice and test on the discordant set only; `n_discordant < 8` ⟹ declare UNDERPOWERED and claim no direction.**
- **(b) Uninformative sensitivity grids.** The grid {0.04, 0.05, 0.055} lies entirely inside the measured empty gap 0.030–0.0553; a grid that never crosses a decision boundary cannot detect sensitivity, so "threshold-insensitive" is uninformative by construction. The same holds for any real τ ∈ (1.196, 2.402). **Replace every sensitivity table in both arenas with the separation MARGIN, or widen the grids (e.g. 0.02–0.10).**

**C9 — The real arena is "real sea, synthetic sensor," and it goes in the abstract.**
**MEASURED** protocol fact (`scripts/d3_compare_gifs.py:144`): a virtual forward fan (12 pts/frame, ±60°, cap 600, 2 cm white noise) resampling a WASS stereo surface, with ground truth taken from the *same* reconstruction on a 4 m grid — so common-mode error cancels. No raydrop, no incidence dropout, no whitecap loss, no range bias, no pose error; and the encounter window (1.2–2.4 m/s) is off the design point (11.3 m/s). **RULING** (P2, adopted by P5): the phrase "real LiDAR validation" is not available anywhere in the thesis; "real sea surfaces observed through a synthetic sensor" appears in the abstract, not in a limitations paragraph.

**C10 — "One estimator, one config" is not established across the two arenas.**
**MEASURED**: the synthetic deployment is route-then-veto; the real deployment is analytic-everywhere plus an amplitude-ratio fallback — two different policies, and the router's own diagnostic (`n_eff ≈ 24` on real windows) predicts that the synthetic policy would route almost every real window to the arm that measurably loses there. **RULING** on what would make P5's three-clause unification legitimate rather than a post-hoc re-description (P1/P2/P3 converged on this form): **one callable, the same constants, no arena-conditional branch (checked at lint level and pre-registered), executed blind on both arenas, with the route distribution written down in advance.** Any constant changed after seeing the real routes disqualifies the unification. P2's pre-registered prediction is on the record: *the guard should veto most real windows back to the analytic arm* — if it does, the clause introduced to fix `BF2_1comp` also keeps real seas on physics for its own principled reason, and P1's "most damaging fact" becomes the best result in the thesis.

**C11 — Code and paper now disagree; this is a live integrity exposure that gates writing.**
**MEASURED**: the defaults were flipped on 2026-07-26 (`aa8e4c3`: `TRACK_PHASE_PREDICT=False`, `init_method='auto'`), while `wavesplat.tex` still carries the 0.199 / 0.223-era headline and a §Limitations narrative (L2183–2213) that explains a weakness now known to be a tracker bug. P3's sharpening: before the flip, code and paper *agreed* (both bug-era); flipping the code without propagating **increased** net exposure. **RULING**: containment first — Paper 1 (corrected WaveSplat) ships before the boundary paper, and not as a find-and-replace (see C14 and W12).

**C12 — There is no non-learned control on the collapse repairs.**
**RULING** (P1 objection 5; P2 conceded he should have demanded it; P4 endorsed it as a *literature* objection). The diagnosed cause is greedy basin capture. Until multi-restart greedy, second-peak initialization and relaxed `min_dk` are run on the same events, "the learned arm repairs collapses" is not distinguishable from "any different initialization escapes a bad basin" — which is preempted by random restart since the 1980s. This is the single condition that decides whether the interlock is a section or a paragraph.

**C13 — Evidence maps must be machine-generated, never hand-typed.**
**RULING** (P3, "the cheapest structural fix nobody proposed"). A ~30-line script that applies the frozen criteria in code to the result JSONs would have printed `[not fixed]` automatically and E1 would never have reached MEMORY.md. Pre-register that handoffs are *generated*, not written.

**C14 — The ladder figure starts at 0.1145, not 0.1990.**
**RULING** (P1 objection 3; P5 conceded without reservation). The 42.5% rung is the repair of a self-inflicted double phase advance in the program's own tracker; claiming it on the headline figure is indefensible. **But it is not wasted**: it becomes a METHODS NOTE in Paper 1 describing the double phase advance (warm start advances `phi += ω·dt` while the forward operator already carries `−ω·t`) as a general pitfall in sequential phase-resolved assimilation — turning the embarrassment into Paper 1's second contribution, conditional on a literature check (warm-start phase error; EnKF-HOS phase drift).

---

## 4. RESOLVED DISPUTES — what flipped, and who conceded

1. **"The deployed constants were tuned"** → narrowed to **"never tested out-of-sample."** *P1 conceded* after P3 verified the program reported the worse pre-registered value.
2. **"Auto-beats-both is an algebraic tautology"** → withdrawn. *P3 conceded*; P1 refused to co-sign it (if `n_selected` were a relabel of the stratification variable, the router would *be* the oracle — it is not, 0.0979 vs 0.0818). Replaced by the stronger and still-standing **two-axes** charge (C7).
3. **"The router loses on real data"** → withdrawn as a *finding*, retained as an *experiment*. *P1 conceded*: nobody has run it, and the guard clause may veto most real windows by design.
4. **"Pre-solve vs post-solve is a difference in kind"** → withdrawn. *P4 conceded* to P1: degree, not kind; survives only as a one-sentence compute-allocation delta.
5. **Asset ranking: interlock #1 → #3.** *P4 conceded*, citing P3's forensics plus P1's no-control objection — "un-preempted-but-underpowered is a worse position than preempted-but-solid, because the literature will not be what kills it."
6. **"Insurance" → "interlock."** *P2 proposed; P3, P4, P5 all adopted*, and P3 extended it from a wording fix to an **endpoint-class** change.
7. **The collapse ledger.** *P2 conceded* that `LJ18g_w3` does cross the line (2.93→0.961), so the correct statement is **1/3 end-to-end below the line, landing at ≈0.96 — "below the line" is not "usable."**
8. **"The spine is defensible today."** *P5 withdrew it* — and conceded he had reproduced his own Ch.2 failure mode by repeating "3/3" and the mean line in the same document that proposes Ch.2 as the instrument that catches such things. He ruled that this debate goes *into* Ch.2 as a documented instance.
9. **"The paper-propagation crux is dead because the code was flipped."** *P3 conceded the premise, then produced a worse fact*: the flip made code and paper disagree, so exposure went up, not down.
10. **Over-valuation of the real arena.** *P1 conceded* to P2's protocol reading (virtual fan over WASS, common-mode cancelling).
11. **Ch.4's architecture novelty.** *Killed by P4's R2 citations* (learned greedy cardinality and atom selection are 1–5 years old); **nobody defended it.** Ch.4 is retitled to "where the learned proposal's information actually lives" and rests on the kx/ky ablation (p=0.94) plus the e2e null.
12. **P5's three-clause policy.** *P1 accepted* that it is not post-hoc at the semantic level (the code comment states clause 2 as a general principle and predates the debate), *conditional* on blind execution — which P2 and P3 then made falsifiable (C10).
13. **"The guard closes 2/3 of the oracle gap."** *P5 flagged it unsupported* after P3's aggregation finding; nobody defended it. (E3 — still uncorrected in the handoff.)
14. **The FNO baseline: from "existential gate" to "not thesis-required."** The 07-24 ledger and P4's R1 called it existential; *P1 ruled in R2* that it answers "is your estimator good," not "is your boundary real," and that he would reject an ML submission without it but would not fail a thesis for lacking it — provided the abstract never claims neural-camp superiority. P5 costed it out (2–4 weeks minimum, with observation-distribution matching as a prerequisite and one geometry retrain already failed) and moved it to a **hard gate on Paper 2**. P2 and P4 did not dissent.

---

## 5. REMAINING DISAGREEMENTS — stated honestly, not papered over

**(a) The weight of the interlock: paragraph, section, or chapter.** P1: "a paragraph, not a pillar." P4: "a conference-note observation, not a thesis pillar." P2: "a section, not a chapter, not the keystone." P5: keeps a full Ch.7, but with the *keystone* moved off the interlock and onto the confirmed prediction of the program's own defeat. **Judge's note:** the disagreement is narrower than it looks — all four agree that Ch.7's keystone is the pre-registered prediction that survives any *n*, and all four agree the interlock's weight is empirically determined by W8 (the 13-record harvest plus the non-learned controls). **This dispute does not need to be settled by argument; it is scheduled.**

**(b) Whether the FNO baseline is thesis-required.** **Consensus: no for the degree, yes for Paper 2 / any neural-camp claim.** Residual disagreement: P4 doubts that abstention is stable — a PoF-2025 FNO referee may write the rejection sentence regardless of what the abstract claims — while P1 holds that a thesis committee will not fail a dissertation for the absence of a baseline it never invokes. Unresolved, and correctly deferred to the Paper-2 decision.

**(c) P2's discreteness-coordinate hypothesis — PROMISING, UNADJUDICATED.** Only P2 champions it; P1, P3, P4 and P5 were silent. The claim: the missing coordinate is not crowding but **spectral discreteness** — synthetic "crowded" is crowded-**and-discrete** (countable atoms), real seas are crowded-**and-continuous**; a set-proposal network trained on discrete atoms, shown a continuum, tiles it with ~24–30 atoms, which is exactly the diagnosed `n_eff ≈ 24`. One variable would then explain the synthetic win, the real loss, and the proposal OOD simultaneously. **Judge's ruling: mark ARGUED, run the one-day test (W10), and do not build a chapter on it before it returns.** It is the cheapest single hypothesis in the packet with the largest explanatory span, and it is currently one persona's conjecture with zero adjudication.

**(d) What the arm swap must prove.** P1 requires it and flips his Q1 on it, with a stated bar (≥ half the crowded effect, ≥ −22% at 15 components). P2 asked instead for a *third estimator* (ETKF/GPR/FNO). P5 scheduled the analytic-side swap only and ruled out ETKF/GPR as boundary arms (field-space assimilators with no initialization slot — not a well-posed swap). P3 did not weigh in; P4 considers the architecture question moot since it is preempted. **Unresolved:** P1's numeric bar carries no endorsement from anyone else. The judge adopts it in W11 for lack of a competing proposal, flagged as single-sourced.

**(e) The Q1 label.** P1 says REJECT-as-ML / ACCEPT-as-OE; P2 says PARTIAL PASS; P3 itemizes earned vs not-earned without a label; P4 says the niche survives on three negatives. **They agree almost entirely on the content and differ on the rubric** — there was never a shared definition of what "novelty PASS" means across an ML venue, an OE journal and a thesis committee. Do not treat the labels as commensurable.

**(f) Sequencing vs the HOS condition.** P5 ships Paper 1 first and makes the 12-episode HOS slice the *first item to drop* if the sprint slips; P2 lists "HOS re-run of the headline ladder" as a **condition** of his Q2 pass. These are not reconciled. **Judge's note:** they are compatible only if the HOS slice is understood as a *scope check* (does the linear-GT ladder survive nonlinearity qualitatively) rather than a re-derivation of the headline. If it is dropped, the thesis must state in the abstract that the ladder is linear-GT and the HOS arm is unrun.

---

## 6. THE UNIFIED WORKLIST

P5's revised 15-day sprint, merged with P1's, P2's and P3's required additions, into one ordered list. **All bars are pre-stated here, before the data exist.** Where personas' numeric bars differed, the reconciliation is explicit: the strictest defensible bar is adopted and the alternatives are recorded.

### Day 1 — freeze, correct, and spend the free hours (zero compute)

**W1 · Integrity + freeze day** — demanded by P3 (a), P5 (item 8, moved forward), P2, P1; judge adds (b4).
- **(a) Build the machine-generated evidence map.** A script that reads the result JSONs, applies the frozen criteria **in code**, and emits the evidence table. **Pass bar:** the script prints `[not fixed]` for `AA14_w0` with no human intervention, and the handoff's §3 table is regenerated from it. Pre-register that all future handoffs are generated, not typed.
- **(b) Packet corrections:** (b1) replace "3/3" everywhere with the two-count report (C1/R3 rule); (b2) delete the mean-only real line, or always state the unchanged median beside it; (b3) **recompute the oracle lane-consistently with one script and either restate or delete "closes 2/3 of the oracle gap"** (E3, still live); (b4) resolve the router-threshold discrepancy — doc says 6, code default is 8 (`sequential_update.py:508`) — and state the frozen triple in one sentence; (b5) relabel the real arena "real sea, synthetic sensor" including in MEMORY.md; (b6) fix the "bank-MMSE" scope mismatch in the niche sentence (the shipped artifact is coarse-to-fine greedy); (b7) delete "insurance" everywhere; (b8) regenerate `program_journey.png` starting at **0.1145**.
- **(c) Write every bar in §6 into a dated prereg file before any run starts.**
- **Hinges on it:** nothing directly — but P3 and P5 both ruled that Ch.5 may not be drafted until (a) exists, and it is the structural fix for the 29% summary-layer error rate.

**W2 · `bf_peak_ratio` on the real windows — one hour, zero new runs, prediction written FIRST** — demanded by P3 (decisive), prediction supplied by P2, endorsed by P1/P5.
- Verified by the judge: `bf_peak_ratio` occurs only in `sequential_update.py` and the `unified_guard_2026-07-26/` files — **it has never been computed on any real window.** P1 and P5 had therefore evaluated only clause 1 of the three-clause policy.
- **Protocol:** write the prediction down first (P2's is on the record: *most real windows fall below 0.05, so the guard vetoes them back to the analytic arm*), then run `explore_beamform`-style computation over the `wp_real` init windows.
- **Interpretation, pre-stated:** below 0.05 ⟹ the three-clause policy already degrades to analytic-everywhere on real data **for its own principled reason**, and the unification is legitimate. Above 0.05 ⟹ P1's objection 2 is confirmed and the unification is dead.
- **Hinges on it:** CRUX B, resolved before any compute is spent. This is the highest information-per-hour item in the entire list.

**W3 · Related-Work delta paragraph — one hour** — demanded by P4 as his non-negotiable ("binary, costs an hour, and its absence is the one thing in this packet that would let a single referee end the defense with a URL").
- Engage **Katende arXiv:2606.07153**, **Kachhadiya arXiv:2605.13068**, **Dekel & Elad arXiv:2010.07069**, **arXiv:2505.15661**, **arXiv:2405.10106**; refresh the Ehlers / PINO / msFNO (PoF 37(11):117116) / Fucile deltas.
- Rename the analytic arm **dispersion-constrained 2D-MNOMP** (Mamandipoor arXiv:1509.01942; MNOMP arXiv:1802.01266) in code comments, paper and thesis.
- **Pass bar:** the paragraph contains P4's verbatim disclaiming sentence (C6).
- **Hinges on it:** Q1 survivability under referee attack; nothing experimental.

### Days 2–5 — the one run everybody signed

**W4 · THE HELD-OUT FORTY** — demanded by **P1, P3, P5 as their non-negotiable**; bars also supplied by P2. **This is the item the degree hinges on.**
- **Protocol:** 40 fresh-seed episodes (16 sparse / 8 moderate / 16 crowded), constants **frozen** (`auto_switch_thresh=8`, `auto_guard_ratio=0.05`, real τ=1.5), three arms on identical lanes, median-of-5 per episode, **scored once, no peeking, no re-running.**
- **Bars (reconciled):**

| ID | Bar | Source & reconciliation |
|---|---|---|
| **S1 (PRIMARY)** | Episode-bootstrap 95% BCa CI for mean(auto+guard) − mean(pure_learned) entirely **< 0**, **and** the same versus pure_v2. **Both required.** | P3 — **adopted as strictest defensible**: it is draw-invariant, whereas absolute levels depend on the difficulty of the fresh draw. Supersedes P2's and P5's one-sided paired Wilcoxon α=0.05 (weaker) — report the Wilcoxon too, it is free. |
| **S1b (absolute, wording gate)** | mean ≤ **0.095** to retain headline-number wording; mean > **0.1011** ⟹ capstone dead outright. | P1 set ≤0.095, P2 and P5 set ≤0.1011. Both retained, at different jobs: 0.095 gates the *wording*, 0.1011 gates *survival*. Between them ⟹ "matches the better specialist per regime," no aggregate win. |
| **S2** | sparse mean ≤ pure_v2 + **0.005** | P3 and P5 (0.005) adopted over P1 (0.006); P2's absolute ≤0.030 recorded as a cross-check. |
| **S3** | crowded mean ≤ pure_learned + **0.005** | P3 adopted over P1 (0.006). |
| **S4** | auto − oracle ≤ **0.015**, oracle recomputed lane-consistently by the W1b script (in-sample 0.0057, so this allows 2.6× degradation) | P3, with P5's aggregation fix folded in. |
| **S5** | Guard fires on **0** crowded episodes **AND** min(bf_peak_ratio ∣ fresh crowded) > 0.05 | P3 — converts the tautological criterion #2 into a real test. |
| **S6** | ≤ **2** damaging misroutes (in-sample was 5/100; pro-rata on 40) | P3. Failure ⟹ the guard was a patch, not a mechanism. |
| **S7** | The moderate (5-component) cell must not lose more than it wins (in-sample 6 L / 4 W) | P1. |
| **Reporting rule** | Report `n_discordant` per slice; any non-inferiority statement is computed on the discordant set only; `n_discordant < 8` ⟹ **UNDERPOWERED**, no direction claimed. Non-inferiority endpoints otherwise retired. | P3, P5. |

- **Pre-committed failure branches:** S1 fails while S2/S3 pass ⟹ the claim becomes "matches the better specialist in each regime," and the aggregate win is deleted — **the regime map still passes.** S2 fails ⟹ the guard did not generalize and the Ch.5 headline is cut. Any failure ⟹ delete the unified headline and ship the failure map.
- **Hinges on it:** P1's Q2 conditional ("one experiment wide"), P3's refusal to allow Ch.5 to be drafted, P5's refusal to schedule a defense ("I will not schedule a defense on a number that has never met data it did not help choose").

### Days 6–8 — the real arena, honestly

**W5 · Blind shipped policy on the 30 real windows** (days 6–7) — demanded by **P2 as his non-negotiable**, by P1 as REQUIRED, formalized by P3, scheduled by P5.
- **Protocol:** one callable, the same constants, **no arena-conditional branch** (lint-level check, pre-registered), run against analytic-everywhere; route distribution predicted in writing beforehand (from W2's output).
- **Bar:** PASS = policy median ≤ analytic + **0.010** (P1 and P2 agree). Reportable-and-honest secondary: learned-routed loss on ≥20/27 with medΔ < −0.03 ⟹ the abstract says synthetic-scoped. **Pre-committed failure branch (P3):** if the policy loses on real by more than +0.02, the thesis says **"two policies,"** in that wording.
- **Hinges on it:** whether "one estimator, one config" may be written at all (C10); P2's Q2 condition — *"either result is publishable; not running it is not."*

**W6 · Record-blocked re-analysis of the real primary** (day 7, zero compute) — demanded by P2, scheduled by P5.
- Windows within a record are not independent; the effective *n* is nearer 5 than 30. Re-run the primary as a record-blocked permutation test or a mixed model.
- **Bar:** the pre-registered accuracy loss must survive at the record analysis unit; if it does not, report it at the correct unit and say so.
- **Hinges on it:** Ch.7's keystone — the confirmed prediction of the program's own defeat — must be stated at the right analysis unit or it is not a keystone.

**W7 · Out-of-sample detector specificity on the 8 unused Bergamasco records** (day 8) — demanded by P1, P2, P3, P5 (unanimous among those who addressed it).
- **Protocol:** τ = 1.5 frozen, ≥40 new normal windows, analysis unit = **record** (n = 8).
- **Bars:** **R1** 0 false fires = PASS; 1 = PARTIAL (report a cluster-robust exact-binomial CI); ≥2 = FAIL; **no single record may contribute more than 1**. *(P3's, adopted as strictest; P2 allowed ≤3 over ≥40 by rule-of-three ≈7%, P1 asked for 0/40 → ~7.5% bound — both recorded, both looser.)* **R2 (margin, replaces every sensitivity table):** min(amp_ratio ∣ fired) − max(amp_ratio ∣ not fired) must remain positive; **any** normal window above 1.5 breaks the separation regardless of the count.
- **Hinges on it:** whether the interlock's false-alarm claim is an in-sample fit or a specificity.

### Days 8–10 — the two experiments that decide what Ch.7 and Ch.5 are

**W8 · Collapse-event harvest + non-learned controls** (days 8–9, CPU only, data on disk) — **new demand by P2**, mandated by P1's objection 5, endorsed by P4 as a literature objection. P2: *"the cheapest high-value run in the packet."*
- **Protocol:** run the analytic arm over **all windows of all 13 records** (≈10% collapse rate ⟹ ≈25 events, making sensitivity estimable for the first time). Then, on the harvested events, run the cheap non-learned escapes: multi-restart greedy, second-peak initialization, relaxed `min_dk`.
- **Bars:** **R3** — report **two counts, never merged**: (a) large error reduction, Δ ≥ 0.5; (b) below the line, learned < 1.0. Fewer than 3 harvested collapse events ⟹ sensitivity **UNDERPOWERED**, claim specificity only. **Control bar (P2):** the learned arm must repair **≥50% more events than the best cheap analytic escape**; otherwise Ch.7's interlock becomes a paragraph about initialization robustness.
- **Hinges on it:** §5(a) — the paragraph-vs-section dispute is settled by this run and by nothing else; and P4's flag-plant on the reversed rescue direction, which he made **conditional** on exactly this control.

**W9 · Observable-coordinate phase diagram** (days 9–10) — demanded by P5 (item 6), P3 (crux A), P1 (crux F), P2 (partial rescue).
- **Protocol:** plot the per-episode (learned − analytic) delta against **bf_peak_ratio** (proposal-independent), coloured by `n_components`, with the 30 real windows overplotted. All inputs already exist (`explore_beamform.json` covers all 100; `_auto_bf_peak_ratio` is logged; computable on every real window).
- **Pre-stated interpretation:** if bf_peak_ratio predicts the delta **at least as well as** `n_components`, the router is not a relabel and P3's circularity charge is answered. If `n_components` dominates, **P3 is right and Ch.5 says so in those words.**
- **Hinges on it:** C7, the two-axes problem — and, secondarily, P1's objection 1: the figure will show honestly that real windows occupy one narrow band of the plane (P1's R2 point: the plane's emptiest cell).

**W10 · The discreteness test** (day 10) — demanded by **P2 only** (§5c, promising-unadjudicated).
- **Protocol:** hold `n_components` fixed and apply per-component directional spreading to produce a continuum-like synthetic sea; re-score both arms.
- **Pre-stated interpretation:** the learned advantage survives ⟹ discreteness is irrelevant and P1's objection 1 is answered on physical grounds. The advantage collapses ⟹ **the actual boundary variable has been found**, and the real-data loss is retro-actively a prediction rather than a defeat.
- **Hinges on it:** nothing currently claimed. It is the only proposed single explanation spanning the synthetic win, the real loss and the proposal OOD — high upside, no downside, one day.

### Days 11–15 — generality, containment, scope

**W11 · Arm swap, analytic side only** (days 11–13) — demanded by **P1 (CRUX 1, verdict-flipping)**, in weaker third-arm form by P2, scheduled by P5.
- **Protocol:** BLASSO / sliding Frank-Wolfe (`tmp/blasso_feasibility.py`) replaces greedy beamform on 20 crowded + 20 sparse episodes, median-of-3. Swap the arm the program *claims fails* — that is the sharp test.
- **Bar (P1, single-sourced — flagged in §5d):** the crossover must reproduce within one complexity step at **≥ half the crowded effect (≥ −22% at 15 components)**. Erased or reversed ⟹ Ch.4–5 are re-scoped to "greedy 2D-MNOMP initialization has a crowding failure mode that a learned proposal repairs," and the thesis has no general boundary object.
- **Explicitly out of scope:** the DETR learned-side swap (in the documented ~40-architecture FAIL bucket — a failure there is uninterpretable), and ETKF/GPR as boundary arms (field-space assimilators with no initialization slot; keep as a context row).
- **Hinges on it:** P1's Q1 upgrade to accept-as-diagnostic and his Q2 upgrade to clear pass.

**W12 · Second integrity sweep and the paper decision** (day 14) — demanded by P5 (crux H), P3 (containment), P2 (integrity), P1 (writing task).
- **Paper 1 (corrected WaveSplat) ships first**, to Ocean Engineering, drafted in parallel with the sprint. Two conditions: **(a)** it is not a find-and-replace — it carries a METHODS NOTE describing the double phase advance as a general pitfall in sequential phase-resolved assimilation (subject to the literature check on warm-start phase error / EnKF-HOS drift); **(b)** Paper 1 must **not** claim the boundary — one footnote at most. Decisive reason (P5): with no learned arm, there is no ML reviewer, so the FNO rejection sentence cannot be written against it.
- **Consequence to accept consciously:** the boundary chapter cannot borrow Paper 1's authority, so **Ch.5 stands entirely on W4**.

**W13 · 12-episode HOS scope check** (day 15 — **DROP FIRST if the sprint slips**) — P2's condition, P5's droppable item 9.
- 4 sparse / 4 moderate / 4 crowded, frozen, one shot. It is a **scope check** (does the linear-GT ladder survive nonlinearity qualitatively), not a re-derivation of the headline. If dropped, the abstract must state that the ladder is linear-GT with the HOS arm unrun.

### Explicitly OUT of the sprint, with reasons on the record

| Item | Ruling | Reason |
|---|---|---|
| **FNO / neural-operator baseline** | Hard **gate on Paper 2**, not a sprint item, not a degree condition | 2–4 weeks minimum; observation-distribution matching is a prerequisite; one geometry retrain already failed. Zero benefit to Paper 1, total to Paper 2. Consensus: required for any neural-camp superiority claim — **so the thesis must not make one.** (Residual dissent: §5b.) |
| **DETR learned-side arm swap** | OUT | Sits in the documented ~40-architecture FAIL bucket; a failure would be uninterpretable. |
| **ETKF / GPR as boundary arms** | OUT (keep as context row) | Field-space assimilators with no initialization slot — not a well-posed swap. |
| **Full 100-episode HOS ladder** | OUT | 1–2 weeks of compute; replaced by W13. |

---

## 7. FINAL THESIS STATEMENT

### 7.1 P5's revised statement (R1 + R2 amendment)

> In grid-free, dispersion-constrained spectral estimation of ocean wave fields from sparse moving elevation observations, the value of a learned prior is confined to spectral-component placement under crowding — a boundary that can be pre-registered, mechanistically attributed, and detected without ground truth — so that a single auto-switching estimator outperforms both specialists in simulation while degrading safely to the analytic specialist on real seas, where the learned arm's measured contribution is failure-mode [interlock] rather than accuracy; **and on real sea surfaces the same observation-only signal inverts the usual hybrid: the learned arm becomes the fault interlock that protects the physics arm, not the reverse.**

*(P5 himself ruled in R2 that the clause "outperforms both specialists in simulation" must be replaced by "the regime boundary is decidable from the observations alone, at a measured cost against an oracle," because the first does not survive a held-out miss and the second does.)*

### 7.2 P2's statement (every clause measured or explicitly negated)

> A shipborne phase-resolved wave estimator becomes deployable not by winning everywhere, but by inferring from the observations alone which regime it is in: this thesis measures the physics/learning regime boundary under pre-registration, builds a single self-routing estimator that matches the better specialist in every regime, and shows on five real sea records that the same observation-only signal acts as a fault interlock that turns the analytic path's catastrophic reconstruction failures into detected, safely degraded ones.

*P2's own assessment of the pair: "mine = every clause measured or explicitly negated (defensible); P5's = more sellable. Recommendation: P5's structure, my clauses, and write neither down until the blind run reports."*

### 7.3 The judge's merged recommendation

One sentence; every clause is either measured or explicitly scoped, and the capstone is held open:

> **In grid-free, dispersion-constrained spectral estimation of ocean wave fields from sparse moving elevation samples, this thesis maps under pre-registration — on a linear-ground-truth synthetic benchmark — where a learned set-proposal prior earns its keep and where it provably cannot: the gain is localized by ablation to spectral-component placement under crowding, its absence in the sparse regime is localized at atom level to a proposal-generation failure that no post-processing can repair and that three pre-registered retraining axes failed to remove, and the resulting boundary is decidable at runtime from the observations alone at a measured cost against an oracle (the single-configuration aggregate result is held for a frozen held-out run and is not claimed here); while on real sea surfaces observed through a synthetic sensor, off the design point, the same observation-only signals confirmed a pre-registered prediction of the learned arm's own defeat on accuracy and reduce its role to a fault interlock that converts detected analytic failures into bounded, safely degraded output without restoring usable skill.**

Clause audit: *"localized by ablation"* MEASURED · *"localized at atom level"* MEASURED · *"three pre-registered retraining axes"* MEASURED · *"decidable at runtime … at a measured cost against an oracle"* MEASURED in-sample, and the parenthetical scopes the aggregate as held open · *"synthetic sensor, off the design point"* MEASURED protocol · *"confirmed a pre-registered prediction of its own defeat"* MEASURED · *"without restoring usable skill"* MEASURED (repairs land at ≈0.96–1.07).

### 7.4 Chapter spine — P5's eight chapters with the R2/R3 amendments

| Ch. | Title (amended) | Claim | Artifact / status |
|---|---|---|---|
| 1 | Which estimator, and when can you trust it? | The deployable unit is a **policy**, not a method | — |
| 2 | The instrument: a benchmark that can decide | Building it overturned three of our own claims | Mesh regeneration 20/27→27/27; median-of-k after CV 53%; `min_dk` caveat; 17/0 discard ledger; 158 tests. **Amendment: this debate goes in as a documented instance (P5's concession), and W1a's machine-generated evidence map is part of the instrument.** |
| 3 | The analytic path and its true ceiling | Sparse is essentially solved; residual error concentrates in crowding | **Amendments: the arm is named dispersion-constrained 2D-MNOMP; the ladder figure starts at 0.1145; the tracker bug becomes a methods note, not a rung; the 16 s frozen-init forecaster is one paragraph, not a contribution.** |
| 4 | **Where the learned proposal's information actually lives** (P4's retitle) | The entire gain is learned kx/ky placement | Ablation `learned_konly ≈ learned`, p = 0.94, plus the e2e null. **Amendment: architecture novelty is explicitly disclaimed with citations (C6) — the chapter's content is the localization, not the architecture.** |
| 5 | The boundary is decidable from the observations alone | **Reworded (P5 R2): "decidable from the observations alone, at a measured cost against an oracle"** — never "one estimator beats both" | Stands **entirely** on W4. Must disclose the two-axes problem (C7) and carry W9's observable-coordinate diagram. **Do not draft before W4 returns.** |
| 6 | **What does not transfer — CO-HEADLINE** | Proposal-generation OOD is a generation-side failure post-processing provably cannot repair; e2e training buys nothing; fan-geometry retrain failed | **Amendment (P5 R3, P4 R3): promoted from "negatives chapter" to co-headline. This is the least-preempted, most mechanistic asset in the program.** |
| 7 | **Real seas, synthetic sensor: a confirmed prediction, a blind router, and a fault interlock** | Keystone = *a pre-registered prediction of our own defeat, which then happened* | Claim 1: the loss (medΔ −0.057, 6/27, p = 0.005, rank-biserial −0.608; pooled **and** record-blocked, W6). Claim 2: the blind router, whatever W5 returns, with the descriptor-plane figure. **Claim 3 — verbatim, as a section not a chapter (see below).** |
| 8 | Operating guidance and limits | — | Abstract-level, not buried: synthetic sensor on real surfaces (virtual fan over WASS, ground truth from the same reconstruction, no raydrop/dropout/pose); off design point; n = 3 events; τ in-sample. **Deleted everywhere: "3/3", "0.881→0.775", "insurance".** |

**Ch.7 Claim 3 — P5's verbatim wording, adopted as binding:**

> "Frozen monitor τ=1.5 detects 2/3 collapse windows, 0/27 false alarms design records, [X/N] on untouched BS; on detected windows learned reduces field error 54-67% (2.31→1.07, 2.93→0.96), converting confidently-wrong into detected, safely-degraded. **Does not restore usable accuracy: end-to-end 1/3 below the usability line; the one usable restoration (1.12→0.79) is missed. Evidence = the margin (2.40/2.53 vs normal max 1.20), not a p-value.**"

**Cut and confirmed cut** (P5 R1, unopposed across three rounds): the CRB-anchored operability-envelope spine → appendix plus one equation (D_eff = L + c_g·T_obs); the information-channel five-element accounting; the 16 s frozen-init forecaster as a contribution; the B-2 task-loss result; all UQ claims; Doppler (one line in the negatives ledger).

---

## 8. VERDICT — the judge's own

### 8.1 Q1 — novelty: what survives, ranked

**Ranked 1-2-3, after three rounds of attrition:**

**① The negative mechanism.** Proposal-generation OOD as a *measured, atom-level* account of why a learned prior cannot be post-processed into parity — the network does not emit atoms near the true swell at all, two pre-registered parity rounds failed, and the only post-processing that closes it is harmful in the other regime — together with the **e2e fine-tune null** (medΔ −0.0007, p = 0.30) with its leak localized to the observation-distribution gap. **MEASURED** at the core. **ARGUED**, on P4's snippet-limited search, that this is unoccupied: the learned-greedy and deep-unfolding literature reports wins and does not localize generation-side failures, and its central premise is end-to-end differentiability — against which this program's null is a direct, citable negative. *Why it ranks first: it is the only asset whose value does not depend on any run in §6.*

**② The boundary location with a two-sided mechanism.** Pre-registered, reproduced, and mechanistically attributed on **both** sides: the learned gain localized by ablation to kx/ky placement under crowding (crowded init-level −45%, 39/40, p = 1.8e-12; tracked 0.2525→0.1809, 36/40, p = 5e-11), the analytic failure localized to greedy basin capture, and the transition width reported rather than hidden. **MEASURED** on a linear-GT synthetic benchmark; **generality pending W11**, and the claim that the boundary is GT-free *decidable* is pending W4 and W9. *Why it ranks second and not first: it is currently a statement about one method pair on one generator, and it is drawn on an axis the instrument does not read (C7).*

**③ The reversed rescue direction, tightly scoped.** On sensor-derived real records the learned arm serves as the fault interlock for the physics arm rather than the reverse — unpreempted **in direction** (every hybrid paper P4 found runs the other way), and it is a real-sensor result that the entire neural-wave camp lacks. But it is **underpowered**: n = 3 events, no non-learned control, 2/3 under the frozen criterion and 1/3 end-to-end, repairs landing at ≈0.96–1.07. *Why it ranks third and dropped from first during the debate: as P4 put it, un-preempted-but-underpowered is a worse position than preempted-but-solid, because the literature is not what will kill it.* W8 is what moves it.

**Explicitly NOT novel, to be cited as occupied:** the selection/router layer (Katende); learned proposal plus classical polish, including its regime theorems (Kachhadiya); learned cardinality and atom selection inside greedy pursuit (Dekel & Elad; deep greedy unfolding); set-conditional diffusion with a cardinality head; grid-free dispersion-constrained greedy estimation itself (= 2D-MNOMP); and "learning's value grows with underdetermination," which is theory to cite and instantiate, never to claim.

**Label:** REJECT as a machine-learning contribution (P1, hardened, unopposed). **PARTIAL PASS to ACCEPT as an ocean-engineering / measurement-science contribution**, on assets ① and ②, conditional on §8.2. Note §5e: these labels are not commensurable, and the personas agreed on the content long before they disagreed on the word.

### 8.2 Q2 — PhD value: PASS, conditional — the minimal condition set

All five personas returned **conditional pass**; none returned an unconditional pass, and none returned a fail. Consolidating their conditions and removing everything that upgrades the thesis without gating the degree, the minimal set is **four items**:

1. **W4 — the held-out forty**, frozen, scored once, against the S1–S7 bars pre-stated in §6. *(P1, P3, P5's stated non-negotiable; P2 supplied bars.)* Any failure is survivable **provided the pre-committed branch is taken**: delete the unified headline and ship the failure map, which still passes.
2. **W5 + W2 — the blind one-policy run on the real windows**, with the route distribution and the `bf_peak_ratio` prediction written down first. *(P2's stated non-negotiable.)* Either outcome is publishable; only silence is not.
3. **W1 + W12 — integrity closure**: the machine-generated evidence map, the four packet corrections including the still-live oracle-aggregation item, and the reconciliation of code with paper via Paper 1. *(P3's structural demand, P2's and P5's integrity condition.)* This gates the degree because the alternative is defending a document whose numbers the candidate's own files contradict.
4. **W3 — the Related-Work paragraph** engaging Katende and Kachhadiya, with the disclaiming sentence, and the analytic arm renamed 2D-MNOMP. *(P4's stated non-negotiable; one hour.)*

Everything else in §6 — W6, W7, W8, W9, W10, W11, W13 — strengthens the thesis and determines the *weight* of individual chapters (most sharply, W8 decides whether Ch.7's interlock is a section or a paragraph), but does not gate the degree.

**No condition requires retraining, new data collection, or an FNO.** Timeline, adopted from P5 and endorsed by P3: ~3 weeks of frozen one-shot runs, ~4 months of writing, Paper 1 in parallel; a defense is realistic at 12 months and comfortable at 15.

**The framing under which this passes, and the framing under which it fails.** It passes as *"where a learned prior earns its keep in sparse phase-resolved wave estimation, why it cannot earn it elsewhere, and how a shipboard system decides which case it is in without ground truth"* — a doctorate in failure characterization and safe degradation. It fails, in every persona's judgement, under *"learning beats physics"* or under a headline built on "one estimator beats both pure arms," because a committee finds the in-sample re-assembly in ten minutes.

### 8.3 The single most important sentence of the debate

> **"Sealed negatives are only a virtue when the positive they surround is independently validated; otherwise they are a large denominator advertised as a result."**
> — **P3, thesis-committee methodologist, Round 1**

It is the load-bearing sentence because it reorganized everything downstream: it is why the held-out forty became the one item all five signed, why the honest-negatives ledger — the program's proudest asset — cannot carry the thesis by itself, and why the correct order of operations is *validate the positive, then bank the negatives*, rather than the reverse.

**Runner-up, and the sentence for the abstract** — P1, Round 3, reversing his own opening position: *"This program's average IS its failure mode, itemized. That is a diagnostic capability, and diagnostic capability is what a doctorate certifies."*

---

## 9. Monday-morning directives — verbatim, all five

> **P1 (harsh ML reviewer):** "Stop defending 0.0875 and start defending the thing that is actually rare: you can name, at atom level, why each arm fails — including on the one episode your guard cannot see — so run the held-out forty, report whatever it says, and let that sentence be the thesis."

> **P2 (senior ocean engineering professor):** "You did not build a method that wins — you built an instrument honest enough to find its own bug, predict its own defeat, and locate the one episode where it still fails; write that thesis, and stop apologizing for the loss on real water."

> **P3 (thesis-committee methodologist):** "Your protocols are better than your press releases — generate the evidence map from the JSONs in code and never type a number again; then go get the forty episodes, because the only thing standing between you and a defended thesis is one number you haven't looked at yet."

> **P4 (literature hawk):** "Stop trying to make the router the discovery — name your analytic arm 2D-MNOMP, cite the people who already learned greedy cardinality and placement, and spend the week proving the thing nobody else has published: exactly why your network cannot see a single swell."

> **P5 (pragmatic advisor):** "Your program's best work is the four times you proved yourself wrong — so this month you stop adding evidence, freeze the three constants, and let one held-out run tell you whether the capstone is real; I will back whichever answer comes back, and I will not back a defense without it."

---

*Judge's closing note on procedure: this document is an adjudication, not evidence. Its factual base is the persona transcripts and the packet; the two source-verified additions (E4, the router-threshold discrepancy; and the confirmation that `bf_peak_ratio` occurs in no real-data file) are marked as such and were checked against `wavesplat/optimizer/sequential_update.py` and a repository-wide grep. Nothing here may be cited as a result. The results are in §6, and they do not exist yet.*

---

## POST-DEBATE ADDENDUM (same day, 2026-07-26) — crux B pre-check RESULT

The one-hour pre-check (P3's decisive computation, run under
`tmp/figs_opus_real_2026-07-26/PREREG_bf_ratio_real.md` with predictions written first):

**Pre-registered cell C. P2's falsifiable prediction is REFUTED; P1's objection 2 is
CONFIRMED by measurement.** bf_peak_ratio on all 30 real windows: 0.27–1.00
(normal-window min 0.27, median ≈ 0.90) — every window sits far above the 0.05 veto
threshold. Real broadband seas produce many comparable beamform peaks even when
swell-dominated; the guard, calibrated on synthetic discrete spectra (crowded floor
0.055), carries no information on real seas. Combined with n_eff ≈ 24 ≥ 8, blind
`init_method='auto'` would route the measured losing arm on essentially every real
window. Consequences, per the frozen interpretation cell:
1. The thesis reports **"two policies"** in those words; the unified-estimator claim is
   scoped to synthetic in the abstract.
2. Both GT-free observables (n_selected, bf_peak_ratio) read "crowded" on real seas
   where the accuracy winner is analytic — **strengthening P2's discreteness hypothesis**
   (the missing coordinate is not crowding but spectral discreteness) and raising the
   priority of the discreteness experiment (worklist W9).
3. Sprint item W5 (blind auto-on-real) remains required for the record; its routing
   outcome is now predicted by measurement, and any deviation is itself reportable.
