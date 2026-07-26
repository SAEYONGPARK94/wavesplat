# RESULT — Held-out real-data detector evaluation (W6 + W7 + W8)

**2026-07-26. Bars frozen in `PREREG.md` before any held-out score existed. CPU only.
Learned arm never run (GPU reserved). Nothing committed. No source file modified.**

Artifacts: `PREREG.md`, `heldout_cfg.py`, `probe_geometry.py`, `probe_load.py`,
`harvest.py` → `harvest_heldout.json`, `analyze.py` → `results.json`, `analysis.txt`.

---

## 0. Headline

| ID | What it tests | Verdict |
|---|---|---|
| **R1** | held-out specificity of the frozen τ=1.5 interlock | **PASS** — 0 false fires / 33 normal windows / 8 unseen records |
| **R2** | separation margin | **not broken**, but the margin collapses **+1.206 → +0.130** (−89 %) |
| **R3** | sensitivity on harvested collapses | **powered for the first time, and LOW: 3/15 new events detected (20 %)** |
| **W6** | record-blocked re-analysis of the 27-window primary | **PASS-in-direction; p = 0.005 → 0.0625**, which is the attainable floor |

The interlock's *false-alarm* claim survives contact with unseen data. Its *coverage*
claim does not: on 15 newly harvested collapse events the frozen detector fires on 3.

---

## 1. Record inventory

13 `.nc` files on disk under `data/real_bergamasco/`.

- **DESIGN (in-sample, 5, 30 windows):** AA14, AA15a, AA15b, LJ18g, YS17.
- **HELD-OUT (8, 48 windows):** BS11a, BS11b, BS11c, BS11d (BS_2011);
  BS13a, BS13b, BS13c (BS_2013); LJ18n (LJ_2018 Grid1).

**Nothing was excluded.** All 8 held-out records built successfully through the
**unmodified** `d3_compare_gifs.scene_real` path (load-feasibility probe run *before* the
prereg was frozen; `probe_load.json`). The 8 entries were injected into the live
`d3_falcon_real_operator.RECORDS` dict at runtime by `heldout_cfg.py` — additive only, no
loader logic touched, no file under `scripts/` edited. `depth` and `stride` were taken
verbatim from the pre-existing committed config `d3_sa_multirecord.py::RECORDS`; the fan
geometry follows the frozen similarity rule `scale = Lmin/55` applied to the AA design
recipe (PREREG §1.1). **No new physical constant was invented by this analysis.**

Windowing is unchanged: `scene_real` hard-codes `linspace(...,6)[win]`, so 6 windows per
record everywhere — no record supports more without editing the loader. 13 × 6 = **78
windows**. The 30 design windows were **not** recomputed; they are read verbatim from
`tmp/unified_2026-07-26/hand_amp.json`, which had already passed a bit-identity control
against `wp_real.json` (re-asserted in `analyze.py`). Only the 48 held-out windows were run.

**Declared-in-advance validity threat (PREREG §1.2), confirmed by the run.** `scene_real`
fixes the query grid at an absolute 4.0 m spacing, so the small BS patches (13–19 m) yield
16–25 query points against 400–4000 for the design records. Changing that constant would be
a loader modification and was forbidden. The pre-registered grid-adequacy subset
(npx ≥ 25: BS13a/b/c, LJ18n) is reported alongside the primary throughout. Relatedly, the
peak wavelength returned on these patches is **patch-limited** (λ_p ≈ 13–19 m on 13–19 m
patches; λ_p ≈ 40 m on the LJ Grid1 patch whose true swell is λ ≈ 300 m) — the artifact
already documented for Grid1 in `d3_sa_multirecord.py`. Both effects make the held-out
arena *harder*, so the specificity test is conservative, not optimistic.

---

## 2. R1 — held-out specificity — **PASS**

Frozen τ = 1.5 on `amp_ratio_grid`; population = held-out windows with `hand_now < 1.0`.

| | value |
|---|---|
| held-out windows | 48 |
| normal windows (hand < 1.0) | **33** |
| **false fires** | **0** |
| records with ≥1 false fire | **0 / 8** |
| max false fires in any one record | 0 (bar: ≤1) |
| cluster-robust exact-binomial CI, **record unit** | **[0.000, 0.369]** |
| window-level exact CI (reported second, wrong unit) | [0.000, 0.106] |
| grid-adequacy subset (npx ≥ 25) | 0 / 12 normal → **PASS** |

**Verdict: PASS** under the strictest bar in the packet (P3's 0-false-fire rule at the
record unit). This also clears the looser bars that were recorded and not adopted (P2's
≤3 over ≥40; P1's 0/40).

What this licenses: *"0 false alarms in 33 normal windows across 8 records the threshold
never saw."* What it does **not** licence: a false-alarm *rate* better than ~37 % at the
record unit — with 8 records the exact CI simply cannot exclude more than that. The claim is
a passed specificity test, not a precise rate.

---

## 3. R2 — separation margin — **not broken, but 89 % of the margin is gone**

`margin = min(amp | collapse that fired) − max(amp | normal)`.

| arena | n collapse | fired | min amp fired | max amp normal | **margin** | broken? |
|---|---|---|---|---|---|---|
| design (in-sample) | 3 | 2 | 2.402 | 1.196 | **+1.206** | no |
| held-out | 15 | 3 | 1.579 | 1.450 | **+0.130** | no |
| held-out, adequacy subset | 12 | 3 | 1.579 | 1.082 | +0.497 | no |
| **pooled (design + held-out)** | 18 | 5 | 1.579 | 1.450 | **+0.130** | no |

No normal window exceeds 1.5, so the frozen "separation broken" condition is **not**
triggered. But the margin narrows **at both ends**: the weakest firing collapse falls
2.402 → 1.579 (LJ18n_w1) and the loudest normal window rises 1.196 → **1.450**
(BS11d_w1, a *good* window at hand nRMSE 0.555). τ = 1.5 now sits in a 0.13-wide gap
between a healthy window and a collapse.

**Reportable statement:** τ = 1.5 transferred without a false alarm, but the operating point
is no longer comfortable — it is a 0.13 margin, not a 1.21 margin. Per P3's ruling this
margin, not a threshold-sensitivity table, is the robustness statistic to publish.

---

## 4. R3 — collapse harvest — sensitivity estimable for the first time, and it is **low**

P2's demand ("run the analytic arm over all windows of all 13 records; ~10 % collapse rate
⟹ sensitivity estimable") was executed. The realised collapse rate is **three times his
estimate**.

| | value |
|---|---|
| **new collapse events** (`hand_now ≥ 1.0`, outside the 3 known) | **15 / 48 held-out windows = 31.2 %** |
| detected by the frozen detector | **3 / 15 = 20 %** (exact CI **[0.043, 0.481]**) |
| pooled over all collapse events (design + held-out) | **5 / 18 = 27.8 %** |
| UNDERPOWERED (<3 new events)? | **No** — the sensitivity bar is met on count |
| **(a) large error reduction Δ ≥ 0.5** | **PENDING-GPU** — requires the learned arm |
| **(b) below the line, learned < 1.0** | **PENDING-GPU** — requires the learned arm |
| non-learned escape controls (multi-restart, 2nd-peak init, relaxed `min_dk`) | **NOT RUN** (out of scope) |

Counts (a) and (b) are reported separately and were **not** merged, and neither was
estimated, imputed, or proxied — this is the structural fix for the 3/3-vs-2/3 error.

Per record — detection is entirely confined to one record:

| record | windows | collapses | detected | collapse windows |
|---|---|---|---|---|
| BS11a | 6 | 0 | – | – |
| BS11b | 6 | 1 | 0 | w4 |
| BS11c | 6 | 2 | 0 | w3, w4 |
| BS11d | 6 | 0 | – | – |
| BS13a | 6 | 5 | 0 | w0, w1, w2, w4, w5 |
| BS13b | 6 | 0 | – | – |
| BS13c | 6 | 2 | 0 | w2, w3 |
| **LJ18n** | 6 | 5 | **3** | w0, w1, w3, w4, w5 |

The three detections are LJ18n_w0 (hand 7.611, amp 5.346), LJ18n_w4 (4.252, 4.705) and
LJ18n_w1 (1.610, 1.579). Every collapse in the seven BS records is missed, and so are
LJ18n_w3 (1.368) and LJ18n_w5 (1.914).

**Mechanistic reading — EXPLORATORY, not pre-registered, must not be cited as a result.**
The 12 missed collapses all have `amp_ratio_grid` ≤ 1.022 (range 0.267–1.022, median 0.596),
i.e. the analytic field is *amplitude-suppressed*, collapsing toward the zero-predictor
(nRMSE → 1). The 5 detected collapses are *amplitude-runaway* failures. The frozen detector
is a **one-sided upper threshold**, so it is structurally blind to the suppressed mode. A
two-sided rule is the obvious hypothesis and is **not** claimed here — it would have to be
pre-registered and tested on data it did not help choose, exactly as τ = 1.5 now has been.

**What R3 changes about the interlock claim.** The honest scope is now: *the interlock
detects analytic amplitude-runaway failures — the ones that produce a confidently wrong,
large-amplitude field — and detects roughly one collapse in five overall.* "The detector
catches analytic collapse" is not available; "the detector catches the runaway mode" is.

---

## 5. W6 — record-blocked re-analysis of the 27-window primary — **PASS-in-direction**

Zero compute; input `tmp/wp_real_2026-07-25/wp_real.json`. Δ = hand − learned (negative ⟹
learned worse). Published primary: window-level Wilcoxon on 27 windows, mean Δ = −0.0611,
**p = 0.004613**.

Record-level means (normal-27):

| record | n windows | mean Δ |
|---|---|---|
| AA14 | 5 | −0.0261 |
| AA15a | 5 | −0.0488 |
| AA15b | 6 | −0.0694 |
| LJ18g | 5 | −0.0132 |
| YS17 | 6 | −0.1319 |

- **All 5 records point the same way** (learned worse): 5/5.
- Record-blocked exhaustive sign permutation (2⁵ = 32): **p = 0.0625**.
- Wilcoxon signed-rank on the 5 record means: p = 0.0625. Paired t on record means: p = 0.0500.

**As pre-stated in PREREG §3, 0.0625 is the attainable floor** at n = 5 records — no data
configuration could have preserved p = 0.005. The pre-registered verdict for "all records
agree in sign AND the permutation p attains its floor" is **PASS-in-direction**, which is
what occurred.

**Correct reporting unit and wording:** the windows within a record are not independent, so
the effective n is 5, not 27. The result must be reported as *"the learned arm was worse on
every one of the 5 records (record-level mean Δ −0.013 to −0.132; record-blocked
permutation p = 0.0625, the smallest value attainable with 5 records)"* — **not** as
p = 0.005. The pre-registered defeat is confirmed in direction and unanimous across
records; it is no longer significant at α = 0.05 at the correct unit, and the thesis must
say so in those words.

Secondary (all-30, i.e. including the 3 collapse windows): record means flip to mixed sign
(learned worse in 2/5), permutation p = 0.5625, Wilcoxon(record) p = 0.625. The all-30
aggregate remains what it always was — a null dominated by the 3 collapse windows.

---

## 6. Runtime, exclusions, and what was not done

- **Runtime:** geometry probe ≈ 20 s; load-feasibility probe ≈ 25 s; **held-out harvest 48
  windows in 32 s** on 6 CPU workers (`OMP/OPENBLAS/MKL=1`, `nice -n 10`); analysis < 2 s.
  Total well under 2 minutes of compute. The BS windows are cheap because the frozen 4 m
  query grid yields only 16–25 query points.
- **GPU:** untouched (verified busy at 98 % / 6.8 GB by another experiment). No diffusion
  model loaded, no learned-arm inference anywhere.
- **Excluded:** nothing. All 13 records ran; all 78 windows scored; 0 failures.
- **Not run (declared):** the learned arm (⟹ R3 counts (a) and (b) are PENDING-GPU); the
  cheap non-learned escape controls of W8's second half; W5's blind shipped-policy run;
  W2's `bf_peak_ratio`-on-real computation.
- **Data integrity:** `data/` untouched; the design-set bit-identity control
  (`hand_now_maxdiff < 1e-9`) is re-asserted at load time by `analyze.py`; the frozen
  collapse definition `hand_now ≥ 1.0` is asserted to reproduce the hard-coded design
  `COLLAPSE` set exactly, so it is not a second free parameter.

## 7. Pre-committed consequences that are now triggered

- R1 PASSED ⟹ the false-alarm claim may be stated as a **specificity**, with its
  record-clustered CI [0, 0.369], and no longer needs the "in-sample fit" caveat.
- R2 not broken but margin +0.130 ⟹ publish the **margin**, not a sensitivity table, and
  state that the operating point is narrow off the design set.
- R3 3/15 ⟹ the interlock's scope is narrowed **in writing** to the amplitude-runaway
  failure mode; the general phrase "detects analytic collapse" is not available. The
  paragraph-vs-section dispute (§5(a) of the debate) is not yet settled, because the
  learned-repair counts (a)/(b) and the non-learned controls are still outstanding.
- W6 PASS-in-direction ⟹ Ch.7's keystone survives **as a direction that is unanimous across
  all 5 records**, and every statement of it must carry p = 0.0625 at the record unit, never
  p = 0.005.
