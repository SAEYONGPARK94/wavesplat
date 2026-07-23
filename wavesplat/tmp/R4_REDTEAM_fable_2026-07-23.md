# Red-team audit — ALTMETHODS_GATE Round 4 + two integrity verdicts (2026-07-23)

Auditor: independent session, adversarial mandate. All numbers below are **re-derived**
from the h5 data with new probe scripts (`wavesplat/tmp/r4_nearfar_power_probe.py`,
`wavesplat/tmp/r4_fulldata_advantage_probe.py`), not read off the target documents. Full
logs: `wavesplat/tmp/r4_nearfar_power_probe.log`, `wavesplat/tmp/r4_fulldata_advantage_probe.log`.

---

## Target 1 — ALTMETHODS_GATE_RESULT_2026-07-23.md

### Verdict: **MATERIAL REVISION** (near-band "parity" claim REFUTED; far-band
"extrapolation artifact" claim UPHELD; top-line "no novelty lever" survives on a
narrower basis than claimed)

The headline — *"where both families are informed … they are equivalent (0.0716 vs
0.0743 …); the apparent 3–4× wide-grid gap is entirely an extrapolation artifact"* — is
**not supported once tested beyond the single frame the doc reports**. The
extrapolation-dominates-the-far-band mechanism is real and reproduces cleanly. But the
"near-band parity" reading was built on n=1 frame per episode, and it does not survive
replication to 9 frames per episode: BANK is measurably worse than BEAM in the near band
too, in both episodes, just by less than in the far band.

#### Attack #3 (statistical power) — decisive, run first because it subsumes #1/#2

I re-ran the exact same fitted BEAM (`greedy`+`joint_refine`, `n_true+1` atoms) and BANK
(`mmse_bank_estimate`, project's tuned estimator) — same subsample, same seed — and
queried them at **9 frames spread across each episode** (30, 60, …, 270) instead of only
frame 150. Frame 150 alone exactly reproduces the published numbers (0.0743/0.0716 for
BF3, 0.0603/0.0847 for BF4 — confirms my probe is faithful to the original), but it is
not representative:

**BF3 5comp s0** — near-band (0–20 m) nRMSE, n=9 frames:
| | mean | sd | 
|---|---|---|
| BEAM | 0.0742 | 0.0005 |
| BANK | 0.0859 | 0.0142 |
| Δ(BANK−BEAM) | **+0.0117** | SE=0.0048, **t=2.46**, n=9 |

Frame 150's Δ=−0.0027 ("bank edges beamforming") was the **most favorable of the 9
frames to BANK** — 3/9 frames show BANK at parity or narrowly ahead (frames 120, 150,
180: −0.0018, −0.0027, +0.0005), the other 6/9 show BANK behind by +0.006 to +0.036. The
9-frame mean is positive (BEAM wins) and 2.46 SE from zero. **The single reported data
point for this episode was the outlier that generated the "parity" narrative, not the
representative case.**

**BF4 15comp s0** — near-band, n=9 frames:
| | mean | sd |
|---|---|---|
| BEAM | 0.0608 | 0.0130 |
| BANK | 0.0835 | 0.0177 |
| Δ(BANK−BEAM) | **+0.0227** | SE=0.0022, **t=10.36**, n=9 |

Every single one of the 9 frames shows BANK behind BEAM in the near band (range +0.008
to +0.031). This episode's reported single-frame number (+0.0244) was **already
representative** — the doc's own text ("behind by 0.024 on 15comp") was honest about it,
but the TL;DR still folded it into "equivalent"/"parity," which the per-frame data does
not support.

**Labeling:** BF4 near-band gap is **MEASURED, n=9 (correlated, single episode/seed), well
powered (SE≪effect)** — not underpowered, directionally robust. BF3 near-band gap is
**MEASURED, n=9, borderline (t=2.46)** — powered enough to reject the *specific* "BANK
0.0716 < BEAM 0.0743" headline as representative, not powered enough for a strong
standalone directional claim on this episode alone. Cross-episode (n=2) generalization
of "near-band gap is real" is **ARGUED**, supported by two independently-replicated
within-episode signals pointing the same direction.

This directly triggers the *second* of the two outcomes the script's own docstring
pre-registered ("BANK loses even in the near band ⇒ the family really is worse at the
estimation task itself, and the verdict stands unqualified") for BF4 outright, and
partially for BF3. The doc reported the *first* pre-registered outcome ("parity") on the
strength of the one frame that happened to support it.

**Mechanism corroboration:** far-band results replicate cleanly and support the doc's
story. BF3 BEAM_far is essentially flat across all 9 frames (0.0763–0.0774, sd≈0.0004)
while BANK_far ranges 0.271–0.296 (~3.7× worse, stable) — the "BEAM's plane-wave model
extrapolates for free because the synthetic truth *is* a plane-wave sum" explanation
holds up robustly. BF4 BEAM_far is far noisier (0.108–0.178, consistent with the doc's
own caveat that this "gift" shrinks with more components) while BANK_far (0.205–0.251)
stays a smaller, ~1.3–2.3× multiple of BEAM_far — also consistent with the doc's point 4.
**This part of the argument is UPHELD, MEASURED, n=18 frame-level replicates.**

Corrected statement licensed by the data: *far-band gap is dominated by extrapolation
(robust); a smaller but real near-band estimation-efficiency gap also exists and favors
the matched parametric (BEAM) family over the shrinkage bank — the wide-grid gap is
**mostly**, not **entirely**, an extrapolation artifact.*

#### Attack #1 (oracle atom count `n_true+1`)

Swept atom count for BEAM, scored at the near band (frame 150):

- BF3 5comp: n_atoms ∈ {3,5,6,8,12,20} → near-band nRMSE flat at 0.0733–0.0743.
  **Oracle count has no measurable effect here.**
- BF4 15comp: n_atoms=8 (undercount) → 0.0759 (+26% relative vs the n=15/16 value of
  0.0603); n_atoms=20,25 (overcount) → 0.0570/0.0569 (**slightly better**, not worse,
  than the oracle n_true+1 value). **Only under-provisioning hurts; over-provisioning is
  free or beneficial.** Since the tested "+1" is a minimal, not generous, slack, this
  does not privilege BEAM the way a fixed strawman "generous oracle" story would suggest
  — a non-oracle deployment choosing a generously large atom budget would do at least as
  well.

**Verdict: attack fails to overturn anything. MEASURED, n=1 episode×5(6) atom-counts each.**
Also confirmed via the project's own code
(`wavesplat/wavesplat/optimizer/sequential_update.py:84,321`, `run.py:34`) that the
*deployed* pipeline defaults to `adaptive_n: True` (energy-pruning/residual-growth, no
oracle count) — so the gate's `n_true+1` convention is specific to this test harness and
not representative of the deployed estimator's real capability either way; worth noting
for precision but not a confound for the family-vs-family comparison itself.

A second, more concrete version of this attack: `blasso_feasibility.greedy()`'s
per-atom `refine_single`/`fit_amp` calls use the **full ~15,000-point episode** (all 301
frames), not the 2,600-point subsample the doc's header claims ("all arms use the same
2600-point observation subsample") — only the coarse peak-search step is
subsample-restricted. I built a "fair" variant (`greedy_fair` in
`r4_fulldata_advantage_probe.py`) restricting refine+amplitude-fit to the same subsample
throughout, and re-ran both episodes:

| episode | metric | BEAM as-published (full-data refine) | BEAM 'fair' (subsample-only) |
|---|---|---|---|
| BF3 5comp | near/far/full | 0.0743 / 0.0764 / 0.0756 | 0.0743 / 0.0763 / 0.0755 |
| BF4 15comp | near/far/full | 0.0603 / 0.1515 / 0.1156 | 0.0601 / 0.1478 / 0.1131 |

Differences are ≤0.4% relative and, if anything, slightly favor the "fair" version —
`joint_refine`'s final L-BFGS step (already correctly subsample-restricted for both
variants) washes out the initialization difference. **MEASURED, n=2 episodes: this is a
real methods-description inaccuracy (the "same inputs" claim is technically false for
one internal step of BEAM's fit) but empirically inert for the reported conclusion —
cosmetic-to-material, not fatal.** Individual atom |Δk| between the two variants can be
large for the crowded 15-comp case (up to 0.22) but these turn out to be
order-of-discovery swaps between near-degenerate correlation peaks (matching pairs
appear, just relabeled), not a systematic bias.

#### Attack #2 (near/far window definition — the `win` fallback)

Confirmed exactly as flagged: at frame 150, the primary threshold
(`dt < median(Δt)*1.5`) selects only 18 of 2600 subsampled points — under the script's
own `< 50` gate, so the **fallback branch silently fires** for every case checked
(`dt < quantile(dt, 0.02)`), which actually spans frame-offsets **0 through 3** (up to
0.3 s / up to ±3 frames), not a single frame. This reproduces the doc's own printed
"n_obs_in_frame=41" (BF3) / "51" (BF4) exactly — those counts are real, but the *label*
"in-frame" is wrong; it is a ~7-frame / 0.6 s swath along the ship track.

**Consequence, checked:** this does not corrupt which data trains which estimator (both
BEAM and BANK are always fit on the full 2600-point subsample regardless of `win`) — it
only affects which grid points get *classified* into the 0–20 m "informed" bin used for
the headline table. At 11.3 m/s design speed, 0.6 s ≈ 6.8 m of ship travel, non-trivial
relative to the 20 m band width. This is a **material labeling/precision bug** (the
physical claim "grid points within ~20 m of an observation" implicitly reads as
single-ping proximity, when it is actually proximity to a several-frame swath) and it
plausibly contributes to the frame-to-frame instability found in attack #3 (variable
window composition per frame → variable near-band point count, 298–835 across the 9
frames tested) — but it is not, by itself, what overturns the parity claim; the
multi-frame replication does that regardless of window definition, since the *same*
(buggy) window definition was used consistently across all 9 frames in my probe and the
gap still shows up systematically.

**Verdict: real bug, MEASURED (reproduced exactly), material for interpretation of "how
local is near", not fatal to the comparison's internal consistency.**

#### Attack #4 (does the conclusion follow)

The near/far test as designed *was* falsifiable — the script's own docstring
pre-registered both possible readings before running. That is good practice and is not
itself the problem. The problem is that the **published TL;DR selected the
first pre-registered reading using data that, on replication, more strongly supports the
second reading for one episode and marginally for the other.** The broader interpretive
leap — "swapping estimator family changes nothing … consistent with, and supporting, the
CRB/information spine" — is explicitly hedged as **ARGUED** in the doc's own "does not
license" section, which is appropriate; I am not overturning that hedge, only the
MEASURED claim underneath it (near-band parity) that the ARGUED claim leaned on for
its strongest evidence.

#### Other gaps the author missed
- Run 4 quietly dropped from 3 episodes (Runs 1–3) to 2 (BF3 5comp s0, BF4 15comp s0),
  losing BF3 5comp s1 as a third data point without comment — likely a runtime choice,
  not concealment, but it further thins an already-thin n. A third episode's near/far
  table would meaningfully firm up the cross-episode generalization; not run here for
  time (this audit's within-episode replication was judged more decisive per unit
  effort).
- No HOS/nonlinear replication — correctly and explicitly disclosed by the doc as
  synthetic-linear-only; not flagged as a new gap, just noting the standing requirement
  remains open.
- `BEAM_REF` numbers hardcoded in `altmethods_gate.py` (0.0748/0.0787/0.1088) do not
  exactly match the values later recomputed live in `altmethods_diag.py`/`realbank.py`
  (0.0755/0.0766/0.1068) — cosmetic (different script, minor lambda/seed-path
  differences), does not affect any conclusion since Run 4 (the operative one) always
  recomputes BEAM live.

---

## Target 2 — INTEGRITY_FIXES_2026-07-23.md

### Item 1 (arXiv 2511.21856 misattribution) — Verdict: **UPHELD, with one correction**

Independent repo-wide grep (`grep -rn "2511.21856" .`, all extensions, `.git` excluded)
finds the string in **four** files, not the two the doc claims:
`tmp/NOVELTY_REREVIEW_ROUND2_2026-07-23.md`, `tmp/INTEGRITY_FIXES_2026-07-23.md` (itself),
`tmp/NOVELTY_ROUND3_execution_2026-07-23.md`, **and**
`tmp/NOVELTY_AGENT_G_crb_calibrated_uq_2026-07-23.md` (missed by the audit's own count).
Read the fourth file directly: it also correctly flags the citation as misattributed and
explicitly instructs "do not cite for the UQ gap" — i.e. the 4th occurrence is *another*
correction record, not a live re-use of the bad citation. **The substantive disposition
("nothing to retract in any live document") is confirmed independently**: no hits in any
`.tex`, `.bib`, or `paper/` file; `wavesplat/paper/wavesplat.tex` and `cover_letter.tex`
mention "calibration" only in unrelated senses (sensor/depth calibration), never a UQ
open-gap argument.

**MEASURED, n=1 (repo-wide grep, exhaustive by construction).** Cosmetic correction: the
doc undercounted occurrences (2 vs actual 4); does not change the verdict.

### Item 2 (companion RLS≡batch vs Adam tracker) — Verdict: **UPHELD, with one caveat**

Independently confirmed via three routes:
1. `wavesplat/paper/encounter_forecast/supplementary/docs/DRAFT_main_paper_v0.md` §3.1
   verbatim: *"No component identification is attempted — the 200-mode identification
   programme of our earlier work consistently lost to a prior-weighted linear
   estimator."* §3.2 proves RLS≡batch specifically for that bank estimator. Confirmed by
   direct read, not just grep.
2. Code: `wavesplat/wavesplat/optimizer/sequential_update.py:123,164` —
   `SequentialUpdater` really does build `torch.optim.Adam` optimizers; this is a
   distinct module from `scripts/d3_stream_seqmmse.py`'s `StreamMMSE`/`batch_mmse`
   (numpy, no torch, no Adam anywhere in the file).
3. Test: `wavesplat/wavesplat/tests/test_stream_seqmmse.py` imports `StreamMMSE`,
   `batch_mmse`, `make_bank`, `design` from `d3_stream_seqmmse.py` — **not** from
   `sequential_update.py`. Its four tests (`test_rls_equals_batch`,
   `test_rebuild_equals_batch`, `test_P_stays_symmetric`, `test_nis_calibrated`) do
   verify exactly the claims made in the paper text (RLS≡batch to <1e-8, rebuild≡batch,
   P symmetry after 200 rank-m downdates, NIS∈[0.5,2.0]). **The test tests what the
   paper says it tests.**

**Caveat found and worth recording:** `paper/encounter_forecast/encounter_forecast.tex`
is a 121-line **skeleton** — section headers plus a comment ("markdown §3 완문…") pointing
at the markdown draft, with no actual prose yet. The audit's grep of this file for
"Adam"/"sequential_update" is **vacuously** zero-hit (there is nothing to hit) rather
than a meaningful confirmation. The real content lives entirely in
`DRAFT_main_paper_v0.md`, which *was* meaningfully checked and is correct. This doesn't
change the verdict (FALSE ALARM stands) but the audit's "grep both files" framing
overstates how much the `.tex` check contributed — worth a note for whoever later
transcribes the markdown into LaTeX, since the .tex currently offers zero protection
against the conflation happening at that time.

**MEASURED, n=1 (direct code/test/text read), decisive.**

---

## What may be cited, what may not

**May cite:**
- Far-band ("extrapolation gap") explanation for the bank vs beamforming wide-grid gap —
  robust across 18 frame-level replicates, 2 episodes.
- "Estimator family is not the primary novelty lever" as an ARGUED conclusion (was
  already correctly hedged).
- Both integrity-fix dispositions (arXiv 2511.21856 contained; RLS≡batch/Adam-tracker
  false alarm) — both independently confirmed.
- Oracle atom-count (`n_true+1`) does not materially confound the BEAM vs BANK
  comparison (checked by sweep).

**May NOT cite:**
- "0.0716 vs 0.0743 … equivalent" / "bank even edges beamforming" as evidence of
  near-band parity — refuted by 9-frame replication (BF3: BEAM ahead on average,
  t=2.46; BF4: BEAM ahead on every frame tested, t=10.36).
- "The apparent 3–4× wide-grid gap is **entirely** an extrapolation artifact" — revise
  to "predominantly"; a real, smaller near-band estimation-efficiency gap also
  contributes and should not be zeroed out.
- "All arms use the same 2600-point observation subsample" as an unqualified methods
  claim — false for one internal step of BEAM's fit (empirically inert, but state it
  accurately).
- "n_obs_in_frame" diagnostic numbers (41, 51) as literal single-frame return counts —
  they are counts within a silently-widened ~7-frame/0.6s window.
- 2511.21856 for any UQ-calibration-gap claim (already the doc's own rule; reconfirmed).
