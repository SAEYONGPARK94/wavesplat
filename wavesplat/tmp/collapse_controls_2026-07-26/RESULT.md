# RESULT — W8 second half: non-learned escape controls on the 18 collapse windows

**2026-07-26. All cells frozen in `PREREG.md` (sha256 `b31e7466375ce91e…`) before any escape
score existed. CPU only — no torch, no GPU, the diffusion model was never imported. Nothing
committed. No file under `scripts/`, `wavesplat/` or `data/` modified.**

Artifacts: `PREREG.md`; `run_escapes.py` → `escapes.json` (198 `mmse_bank_estimate` calls,
211 s, 6 workers); `analyze.py` → `results.json`, `analysis.txt`;
`run_normals_posthoc.py` → `normals_posthoc.json` (post-hoc, labelled).

**Bit-identity control: PASS.** All 18 recomputed baselines match the stored harvest values to
`0.000e+00`. The run is valid under the PREREG's own invalidation clause.

---

## 0. Headline — the answer to objection C12

**A cheap, frozen, GT-free reconfiguration of the analytic arm escapes the collapses at least as
well as the learned arm does, on every window where the two can currently be compared.**

Doubling the `kp_hint` (`E2-double`, one fixed config, no ground truth, no learning, ~1 s) repairs
**6/18** windows by Δ ≥ 0.5 and puts **11/18** below the line. On the **5 windows the frozen
τ = 1.5 interlock actually fires on** — the only windows where any fallback would ever run — it
repairs **5/5 on both criteria**, and on the two of those where the learned value is known it is
**better on both**: AA14_w0 **0.825** vs learned 1.071; LJ18g_w3 **0.726** vs learned 0.961.

It is not a disguised bug fix: as a blanket default `kp_hint × 2` **worsens 42/60 normal windows**
and creates **11 new collapses**, leaving the total collapse count over all 78 real windows at
**18 → 18**. It is a genuine escape — which is exactly the role the learned arm plays.

**P2's bar is now numerically specified and is steep:** the learned arm must reach
**≥ 9/18 (a)** and **≥ 17/18 (b)** to clear `1.5 × C_best`. Those counts are PENDING-GPU.

---

## 1. Scope correction, declared in the PREREG before running

The real-record hand arm is `d3_sa_nomp.mmse_bank_estimate` — a spectrum-weighted **linear MMSE
over a fixed continuous bank**, not the coarse-to-fine greedy initializer. Therefore:

- **There is no `min_dk` on this arm** and no greedy component-selection loop. The debate's third
  named escape does not exist here; E3 (`sigma0` scaling) replaced it, declared in advance.
- **"Multi-restart" can only mean val-split resampling** — the split that selects
  `pexp ∈ {0.5,1,2}`, `smult ∈ {1,3,10}` and gates the SBL-EM steps is the arm's *entire*
  run-to-run freedom. Measured: **7 of 90 E1 runs are numerically identical to the baseline**
  (a resampled split can select the same hyperparameters and reject the first EM step, in which
  case `gam` is split-independent); per-window sd of the 5 E1 readings is a median of 0.110.
- The words "random restart of a greedy optimizer" must not be used for E1 anywhere.

## 2. Frozen-cell counts — all 18 collapse windows

**(a) LARGE REDUCTION** = `base − escape ≥ 0.5`. **(b) BELOW-LINE** = `escape < 1.0`.
Two counts, never merged.

### Deployable (GT-free: a single fixed config, or a median)

| escape | (a) /18 | (b) /18 |
|---|---|---|
| E1-median (5 val-split restarts) | 2 | 2 |
| E2-peak2 (2nd beamform peak) | 5 | 5 |
| E2-half (`kp × 0.5`) | 5 | 4 |
| **E2-double (`kp × 2.0`)** | **6** | **11** |
| E3-half (`sigma0 × 0.5`) | 0 | 1 |
| E3-double (`sigma0 × 2.0`) | 4 | 6 |

**C_best(a) = 6/18** (E2-double). **C_best(b) = 11/18** (E2-double).
Even the *median* cheap escape scores ≈ 4.5/18 on both criteria — the result does not hinge on
E2-double alone.

### Oracle-flavored — winner picked using ground truth, **not eligible for any bar**

| reading | (a) /18 | (b) /18 |
|---|---|---|
| E1-best (best of 5 seeds) | 6 | 7 |
| E2-bestvar | 7 | 13 |
| E3-bestvar | 4 | 6 |
| **ANY-bestvar** (best of all 11 runs) | **7** | **15** |

The oracle envelope is the honest measure of how *reachable* these windows are without learning:
**15 of the 18 collapses have at least one cheap analytic configuration that lands below 1.0.**

### By arena

| escape | held-out 15 (a)/(b) | design 3 (a)/(b) |
|---|---|---|
| E1-median | 2 / 2 | 0 / 0 |
| E2-peak2 | 3 / 3 | 2 / 2 |
| E2-half | 3 / 3 | 2 / 1 |
| **E2-double** | **4 / 8** | **2 / 3** |
| E3-half | 0 / 1 | 0 / 0 |
| E3-double | 2 / 5 | 2 / 1 |

## 3. Design-3 spot check against the known learned repairs (n = 3, underpowered)

Learned (already on disk, `tmp/unified_2026-07-26/hand_amp.json`):
AA14_w0 2.309 → **1.071**, AA15a_w3 1.120 → **0.791**, LJ18g_w3 2.925 → **0.961**
⟹ under the frozen definitions, learned scores **(a) 2/3, (b) 2/3**.

*(Note on the debate's "end-to-end 1/3": that phrase uses a stricter notion of usability than the
frozen `< 1.0` line — only 0.791 is clearly usable, 0.961 and 1.071 sit at or above a no-skill
predictor. Both readings are given here; they are not merged.)*

| window | base | **learned** | E1-median | E2-peak2 | E2-half | **E2-double** | E3-half | E3-double |
|---|---|---|---|---|---|---|---|---|
| AA14_w0 | 2.309 | 1.071 | 2.032 | 0.782 | 1.069 | **0.825** | 1.873 | 1.533 |
| AA15a_w3 | 1.120 | 0.791 | 1.002 | 0.854 | 0.911 | **0.696** | 1.502 | 0.796 |
| LJ18g_w3 | 2.925 | 0.961 | 3.001 | 1.556 | 1.121 | **0.726** | 3.964 | 1.587 |

- **E2-double: (a) 2/3, (b) 3/3 — it matches the learned arm on (a) and beats it on (b), and it is
  numerically better than the learned arm on all three windows individually** (0.825 < 1.071,
  0.696 < 0.791, 0.726 < 0.961). Under the debate's stricter usability reading the gap widens:
  all three E2-double repairs land at 0.70–0.83, i.e. all three are "usable", against one of three
  for the learned arm.
- E2-peak2 also matches learned on both counts (2/3, 2/3).
- E1-median (the literal "multi-restart" control) scores **0/3** — the val-split resample alone
  does **not** escape these collapses. That part of P1's objection is *not* supported.

Three windows is three windows; this is a spot check, not the bar. But it is the exact evidence
the interlock's headline sentence currently rests on, and on that evidence the learned arm is not
the thing doing the work.

## 4. Post-hoc (NOT pre-registered — added after scoring, labelled throughout)

The frozen grid cannot distinguish "escape" from "better default". So `kp_hint × 2` was run on the
**60 normal windows**:

- improved **18/60**, worsened **42/60**, median Δ **−0.050**, mean **−0.096**;
- created **11 new collapses** (BS11a_w4, BS11b_w2, BS11b_w5, BS11d_w0, BS11d_w3, BS13b_w3,
  BS13c_w0, BS13c_w1, BS13c_w5, LJ18g_w0, YS17_w2);
- **net over all 78 windows: 18 collapses → 7 surviving + 11 new = 18.** Exactly a wash.

⟹ `kp_hint × 2` is **not** a better default and **not** a bug fix; it is a real regime trade. This
*strengthens* the control's relevance rather than weakening it: it is a fallback that must be
*triggered*, which is precisely the learned arm's architecture in Ch.7.

**Operational subset** (post-hoc, but defined by deployment, not by the scores): the 5 windows
where the frozen τ = 1.5 detector fires are the only ones where a fallback ever runs.

| window | base | amp | **E2-double** | learned |
|---|---|---|---|---|
| AA14_w0 | 2.309 | 2.402 | **0.825** | 1.071 |
| LJ18g_w3 | 2.925 | 2.534 | **0.726** | 0.961 |
| LJ18n_w0 | 7.611 | 5.346 | **0.954** | PENDING-GPU |
| LJ18n_w1 | 1.610 | 1.579 | **0.679** | PENDING-GPU |
| LJ18n_w4 | 4.252 | 4.705 | **0.548** | PENDING-GPU |

**E2-double on the fired subset: (a) 5/5, (b) 5/5.** Learned, where known: (a) 2/2, (b) 1/2.
An interlock whose fallback is `kp_hint × 2` would, on this evidence, be a complete Ch.7 with no
learned component in it at all.

## 5. PENDING-GPU comparison stub (frozen protocol, executable as soon as the learned arm runs)

> **RESOLVED same day** — the learned arm has since been run on all 18 windows under
> `PREREG_GPU_ADDENDUM.md`; see **`RESULT_GPU.md`**. Outcome: learned **(a) 6/18,
> (b) 6/18** ⟹ **both bars FAIL** (needed ≥ 9/18 and ≥ 17/18), and the pre-committed
> consequence below is in force on both criteria. The stub text is left unedited below as
> the frozen protocol it was.

The learned arm's readings on the 15 held-out collapse windows do not exist. When they do:

- **Deployable comparator** `C_best` = max count over the six GT-free readings
  {E1-median, E2-peak2, E2-half, E2-double, E3-half, E3-double}, per criterion.
  **Measured now: C_best(a) = 6/18, C_best(b) = 11/18.** This max-over-six is deliberately
  generous to the control and that asymmetry must be stated wherever the comparison appears.
- **P2's bar, applied per criterion, never merged:**

  | criterion | C_best | learned must reach | learned actual |
  |---|---|---|---|
  | (a) Δ ≥ 0.5 | 6/18 | **≥ 9/18** | **PENDING-GPU** |
  | (b) < 1.0 | 11/18 | **≥ 17/18** | **PENDING-GPU** |

- Bar met on a criterion ⟹ that criterion supports "the learned repair is not merely
  re-initialization". Bar missed ⟹ for that criterion **Ch.7's interlock is a paragraph about
  initialization robustness**, in those words.
- Oracle-flavored readings (E1-best, *-bestvar, ANY-bestvar) may not be used to satisfy or fail
  the bar in either direction.
- The learned arm must be scored on **the same 18 windows, the same lead-0 metric, the same
  pooled-GT normalization**; no substitution, proxy, or averaging with an escape reading.

## 6. EXPLORATORY (declared exploratory in advance; no claim, no deployable threshold)

Can *any* two-sided amplitude rule separate the suppressed-mode collapses the one-sided τ = 1.5 is
blind to? **All 78 real windows, `amp_ratio_grid` vs hand nRMSE.** Verdict: **no.**

| | n | amp range | median |
|---|---|---|---|
| collapses (hand ≥ 1.0) | 18 | 0.267 – 5.346 | 0.700 |
| normals (hand < 1.0) | 60 | 0.255 – 1.450 | 0.686 |
| suppressed collapses (amp < 1.5) | 13 | 0.267 – 1.022 | — |

- The suppressed-collapse amp range **[0.267, 1.022] is entirely contained inside** the normal
  range [0.255, 1.450]. **Complete overlap, no gap anywhere on the low side.**
- `P(amp_collapse < amp_normal) = 0.456` — below the 0.5 no-information point, i.e. the low side of
  `amp_ratio_grid` carries **no usable signal, and what little it has points the wrong way.**
- Best **low-side-only** rule, *fitted on these same 78 windows*: `amp < 0.587` →
  sensitivity 6/18 at **13/60 false fires**, Youden J = 0.117. Useless.
- Best **two-sided** rule, *fitted on these same 78 windows*: `amp < 0.587 or amp > 1.450` →
  sensitivity 11/18 at **13/60 = 22 % false fires**, J = 0.394. That is the **in-sample ceiling**;
  a held-out rule can only be worse. Compare the frozen one-sided τ = 1.5 on the same data:
  **5/18 sensitivity at 0/60 false fires.**
- **Reading:** the entire value of `amp_ratio_grid` is on the *high* side. Adding a low-side arm
  buys +6 detections at the cost of 13 false alarms — it destroys the one property (0 false fires)
  that made the interlock reportable. The suppressed collapse mode is **not detectable from this
  statistic**, and no threshold is proposed, fitted for deployment, or adopted. τ is unchanged.

## 7. Honest summary sentence

> On the 18 harvested real-record collapse windows, a single frozen ground-truth-free
> reconfiguration of the analytic arm — doubling `kp_hint` — reduces error by ≥ 0.5 on 6/18 and
> lands below nRMSE 1.0 on 11/18, and on the 5 windows the frozen τ = 1.5 detector actually fires
> on it repairs 5/5 on both criteria and is numerically better than the learned arm on both
> windows where the learned value exists (0.825 vs 1.071; 0.726 vs 0.961); the literal
> multi-restart control (val-split resampling) does nothing (2/18, 0/3), so the escape is not
> "any different initialization" but specifically a spectral-support correction; `kp_hint × 2` is
> not a better default (42/60 normal windows worse, 11 new collapses, net 18 → 18 collapses), so
> it is a genuine triggered fallback — the same architectural role Ch.7 assigns to the learned
> arm; and no two-sided rule on `amp_ratio_grid` can extend the detector to the suppressed
> collapse mode (complete range containment, in-sample ceiling 11/18 at 22 % false fires).

## 8. Pre-committed consequences now triggered

- **C12 is answered in the direction P1 predicted, but by a different mechanism than he named.**
  Multi-restart is *not* the escape (E1: 2/18, 0/3). `kp_hint` misspecification is. The correct
  literature framing is therefore not "random restart since the 1980s" but "the analytic arm's
  collapses are dominated by a mis-set spectral support parameter", which is a weaker novelty
  position, not a stronger one.
- **The interlock's fallback need not be learned.** Until the PENDING-GPU counts clear
  ≥ 9/18 and ≥ 17/18, the honest statement is that a 1-second analytic reconfiguration is the
  incumbent fallback and the learned arm has not been shown to beat it.
- **§5(a) of the debate (paragraph vs section) now has its data, minus one input.** Everything
  except the learned arm's 15 held-out readings is measured. The design-3 evidence points to
  *paragraph*.
- **The suppressed-mode detector idea is closed, not deferred.** §6 shows there is nothing to
  pre-register: the statistic has no low-side information (0.456).
- **A follow-up worth pre-registering separately** (not done here, not claimed): if `kp_hint`
  quality drives the collapses, the deployable question is a GT-free `kp_hint` self-check or a
  two-config val-selected `kp`, which would be an analytic fix competing with Ch.7 entirely.
