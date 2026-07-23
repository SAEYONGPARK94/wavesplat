# Forensic trace: where do seed 42 (good) and seed 1 (bad) actually diverge? — 2026-07-23

Follows `tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md`, `tmp/VARIANCE_VERIFY_fable_2026-07-23.md`,
`tmp/VARIANCE_MECHANISM_HUNT_2026-07-23.md` (three refuted knob-ablations: prune threshold,
padding amplitude, `v2_konly`'s random-w/φ re-draw). This document does **not** ablate anything.
It instruments the shipped default pipeline **unchanged** — same config as
`tmp/estimator_variance.py`, same episode (`data/episode_BF3_2comp_seed0_uniform.h5`), same
seeds 42 (nRMSE 0.0524) and 1 (nRMSE 0.1682) — and snapshots the model's full parameter state at
every 10 iterations, at every coarse-to-fine growth event, at the prune/grow decision, and at
every tracking frame, then diffs the two runs directly. A third seed (3, nRMSE 0.1066) and a
fourth (7, nRMSE 0.0590, a second "good" seed) were added for context/robustness — reasons below.

Script `tmp/variance_trace.py` (read-only instrumentation, does not modify or import
`tmp/grow_phase_fix*`, does not touch the `GROW_RANDOM_PHI` / `KONLY_RANDOM_WPHI` / `PAD_W`
module flags — all left at their shipped defaults True/True/0.01). Raw traces
`tmp/variance_trace_data.json` (seeds 42, 1, 3) and `tmp/variance_trace_seed7.json` (seed 7).

**Self-test (MEASURED):** the instrumented run reproduces `tmp/estimator_variance_results.json`
bit-for-bit for all three seeds (`match=True` to 1e-12) — the trace adds zero RNG consumption of
its own, so what follows describes the actual production code path, not an artifact of the
harness.

## Headline (read this first)

**The divergence is a single, nameable discrete event in Phase 1** — the coarse-to-fine
`N=8→15` residual-beamforming growth step (`coarse_to_fine.py:advance()`, called from
`sequential_update.py:initialize()`) — but its downstream *consequence* is not a jump, it is a
**periodic oscillation during tracking** whose period matches the true field's own missing
wave. And **a central piece of the prior narrative needs correcting**: on this episode, neither
the "good" nor the "bad" run actually recovers the field's second (weak) true wave component —
both structurally lose it. What differs is how many *redundant copies of the strong wave*
survive pruning (2 vs 1), and that redundancy is what lets the tracking phase absorb the
ever-present interference from the un-modeled weak wave. "n_gaussians_final=2 = both true
components recovered" (as stated in `VARIANCE_VERIFY`) is **not what is happening here** — see
§3.

---

## 0. The episode's true field (MEASURED, from the file itself)

`config/wave_components` in the h5 file, columns `[A, ω, kx, ky, φ, Q]`
(`wavesim/data_writer.py:53`):

| # | A [m] | ω [rad/s] | kx [rad/m] | ky [rad/m] | φ [rad] |
|---|---|---|---|---|---|
| 0 (strong) | 0.2088 | 1.2852 | 0.1590 | 0.0554 | 4.0021 |
| 1 (weak)   | 0.0372 | 2.7132 | 0.7493 | 0.0412 | 1.6951 |

Component 1 carries only **3.2% of component 0's energy** (A² ratio). This ratio matters a
great deal below.

## 1. Where do the component sets first differ, and in what variable?

**At N=3 init (iteration 0): not at all in kx/ky/w — only in φ.** Beamforming is deterministic
and seed-independent; `v2_konly`'s shipped default (`KONLY_RANDOM_WPHI=True`) sets
`init_w = 0.01` uniformly. Confirmed directly: seeds 42, 1, 3 all get identical
`kx=[0.16,-0.35,2.81]`, `ky=[0.05,0.01,-0.81]`, `w=[0.01,0.01,0.01]` — only `φ` (drawn by
`torch.rand`, seeded) differs. None of these 3 initial beamform peaks is near either true
wave except the first (dist 0.005 from true#0); the rest are sidelobe noise.

**At the N=3→8 growth (iteration 0 for every seed): a minor, inconsequential difference.** The
5 new components' (kx, ky) are computed by residual-beamforming on the model's state after
exactly one gradient step (this stage's loss threshold, 0.1, is already satisfied by the raw
MSE of the near-flat N=3 initial model, before any real fitting — see §5). Seeds 42 and 1 agree
on 4 of 5 new (kx, ky) locations to 4 decimal places; the 5th slot swaps between two
low-amplitude (w≈0.01) noise peaks. Inconsequential — none of the 5 is near true#1 either.

**At the N=8→15 growth: the fork that decides everything.** This is where the good and bad runs'
component *sets* first differ materially, and the variable is **kx/ky (component existence at
a given wavenumber)**, not w or φ directly:

| seed | global iter of N=8→15 advance | new comp nearest true#1 (weak wave) |
|---|---|---|
| 42 (good)  | 4 | (kx=0.667, ky=0.153), dist **0.139** — found |
| 7  (good)  | — (see §3, same story) | found, similarly close |
| 3  (medium)| 1 | (kx=0.820, ky=0.020), dist **0.046** — found, closest of all |
| 1  (bad)   | 1 | closest new comp dist **2.36** — not found at all |

Seed 1 never adds *anything* near the true weak wave's location at this growth step; every
candidate the greedy beamform-search locks onto is a sidelobe of the strong wave's still-messy
residual, ≥0.9 away in wavenumber space. **This — whether the greedy residual peak-search
happens to sample a point near (0.75, 0.04) before it runs out of the `n_new` new slots to
allocate — is the fork**, and it is entirely a function of what the residual field looks like at
the moment `advance()` is called, which is downstream of the random φ draws (below).

## 2. Discrete or gradual? — **Both, at different stages**

**Root cause: discrete, and it's a race decided by loss-threshold miscalibration, not
optimization quality.** Full `loss_history` (MEASURED, first 5 values of 900):

| seed | iter 0 (N=3, before advance) | iter 1 (N=8, first step) | 3→8 advance | 8→15 advance |
|---|---|---|---|---|
| 42 | 0.01706 | **0.06946** (> 0.05 threshold) | iter 0 | iter **4** |
| 1  | 0.01606 | **0.00964** (< 0.05 threshold) | iter 0 | iter **1** |
| 3  | 0.01829 | **0.00465** (< 0.05 threshold) | iter 0 | iter **1** |

The `N=3→8` threshold (0.1) is satisfied trivially at iteration 0 for *every* seed — the raw
MSE loss of this low-`Hs` (0.6 m) episode starts near/below 0.1 before any real fitting happens,
so this stage never actually optimizes at all. The `N=8→15` threshold (0.05) is the real fork:
whether the just-grown N=8 model's loss — determined almost entirely by whether the 5 new
components' *randomly drawn* phases (`GROW_RANDOM_PHI`, `coarse_to_fine.py:157`) happen to
align well with the residual by luck — already sits below 0.05 in one step (seed 1, seed 3) or
needs 3 more steps (seed 42). Whichever happens, the residual handed to the `N=8→15`
beamforming search reflects a model that has had **1 gradient step** (bad/medium) vs **4
gradient steps** (good) of real refinement — enough of a difference in how "clean" the residual
is that the greedy peak-search (which permanently suppresses a 0.15-wide window around each pick
and never revisits) does or doesn't land near the true weak wave.

**Downstream consequence: not a jump — a periodic oscillation tracking the missing wave's own
period.** `init_nrmse` (end of Phase 1) is essentially seed-invariant — 0.0474 (42), 0.0472 (1),
0.0488 (3) — confirming the earlier finding that Phase-1 fit quality *looks* the same across
seeds. The gap opens up entirely during the 190-frame (19 s) tracking phase, and it is
**oscillatory, not monotonic**: seed 1's per-frame nRMSE swings between ≈0.044–0.05 (matching
seed 42's level almost exactly) and periodic peaks of 0.23–0.28, at roughly the same locations
every cycle. Peak-to-peak spacing (n=6 gaps, MEASURED): mean **2.63 s**, close to the true weak
wave's own period **2π/ω₁ = 2.32 s** — consistent with, though not exactly matching (peak-picking
imprecision at n=6 noted), a beat driven by the wave the model never represents. **Null-checked
alternative:** this is not the 3-frame tracking-window period (0.3 s) or the 10 Hz frame rate
(0.1 s) — both far shorter than the observed 2.6 s cycle.

## 3. What does the bad run actually converge to? — **CORRECTION to the prior framing**

Post-prune final state (MEASURED, `sequential_update.py:_adaptive_prune_grow`'s actual output,
not a hand re-derivation):

| seed | final n | components | nearest true match |
|---|---|---|---|
| 1 | 1 | kx=0.1592, ky=0.0554, w=0.1951, φ=4.0027 | true#0, dist **0.0002** (essentially exact) |
| 42 | 2 | kx=0.1590/0.1589, ky=0.0549/0.0567, w=−0.1699/0.0710, φ=6.77/5.05 | **both** match true#0 (dist 0.0005, 0.0013) |
| 7 | 2 | kx=0.1590/0.1588, ky=0.0553/0.0552, w=0.1016/0.0928 | **both** match true#0 (dist 0.0001, 0.0003) |

**Neither run recovers true#1 (the weak wave).** Seed 42's second surviving component is not a
fit of the weak wave — it is a second, near-degenerate copy of the *same* strong wave (kx, ky
differ from each other by <0.002; both are within 0.0013 of true#0 and >0.5 from true#1).
Verified on a second independent "good" seed (7) to rule out this being a fluke of seed 42
specifically — same pattern exactly.

Recombining the two seed-42/seed-7 components as one effective sinusoid (since their kx, ky are
nearly identical, this is a fair approximation) and comparing all three seeds' recovery of
true#0 directly:

| seed | effective A (of true#0's 0.2088) | effective φ (of true#0's 4.0021) |
|---|---|---|
| 1 (1 comp)  | 0.1951 (93.4%) | 4.0027 |
| 42 (2 comp) | 0.1937 (92.8%) | 4.0006 |
| 7 (2 comp)  | 0.1944 (93.1%) | 4.0030 |

**All three recover true#0 to the same accuracy (~93%, phase within 0.003 rad).** So the
divergence in outcome is caused by **neither** (a) how well the strong wave is fit, **nor**
(b) whether the weak wave is captured — both are equalized across good/bad. It is caused
purely by **how many redundant degrees of freedom (1 vs 2) survive into the tracking phase**,
and what that redundancy does when the tracking-phase optimizer meets the un-modeled weak
wave's leftover signal (§4). This is not "merged," not "wrong wavenumber," not "phase-flipped"
— it's **a second copy of the right answer** vs **no spare capacity**.

**Why does H1 (prune-threshold sweep, 0.10→0.02→0.005) find no effect?** Because it isn't
close: even the *best-found* candidate match to true#1 across seeds 42/3 (0.08% and 0.09% of
total energy respectively) sits **three orders of magnitude below** even the strictest threshold
tested (0.5% keep-cut = 99.5% target fraction ≈ discard 0.5%). H1's null result is confirmed and
now has a concrete numeric explanation, not just an empirical one.

**One more observation, unresolved (ARGUED, not confirmed):** tracing seed 42's best true#1
candidate (its component index 10) iteration-by-iteration through the 500-iteration final stage
shows its weight oscillating near zero for most of that budget, then **turning a corner and
rising monotonically only in the last ~150 of 900 total iterations** (energy fraction 0.000%→
0.08% between iterations 730–890, still climbing when the budget ends). This suggests the
iteration budget itself, not only the random-phase bug, may be a second lever — untested here,
flagged for a future, deliberately separate experiment (do not conflate with the phase-fix
ablation currently running).

## 4. Does the failure show up in obs_loss at the time, or only later in nRMSE?

**For the pair the task named (42 vs 1): no inversion — obs_loss tracks nRMSE faithfully, both
within-run and on average.** Within-seed-1's own 190-frame trajectory, `corr(nRMSE, obs_loss) =
+0.949` (MEASURED, n=190) — the periodic beat shows up in the training loss in lockstep with the
held-out nRMSE (first 5 frames: nRMSE 0.051→0.062→0.076→0.089→0.103 alongside obs_loss
0.00082→0.00151→0.00233→0.00334→0.00391). Averaged over the whole tracking phase, seed 1's
avg_obs_loss (0.0074) is properly, substantially higher than seed 42's (0.0006) — the failure is
visible in obs_loss at essentially the same time it appears in nRMSE, with no hidden lag.

**⚠ Correction/clarification: the "bad basin fits observations better" inversion cited in
`RESTART_SELECTION_RESULT` is about a *different* seed — 3, not 1.** Re-checking that document
and `tmp/restart_selection_results.json` directly: the run selected by GT-free `argmin(obs_loss)`
across all 8 seeds was **seed 3** (nRMSE 0.1066, obs_loss 0.000534) — its average obs_loss is
*lower* than seed 42's (0.000593) despite *worse* nRMSE. Seed 1 (this trace's assigned "bad"
seed) shows the opposite, expected relationship (obs_loss correctly higher, no inversion). These
are genuinely different failure modes:

- **Seed 1 (under-prune, this trace's target):** prune succeeds, collapses 15→1. One rigid
  component, no spare capacity, tracking exposes the full periodic beat against the un-modeled
  weak wave directly in both nRMSE and obs_loss.
- **Seed 3 (prune *rollback*):** `[Adaptive] Pruning rolled back: residual would increase 0.0198
  → 0.0288` — the 30%-safety-check *fired*, and **all 15 components survive** with no reduction.
  30 free (w, φ) DOF then locally overfit each 3-frame tracking window (within-run
  `corr(nRMSE, obs_loss) = +0.847`, still positive — not simulated/fake data — but the *typical
  level* of obs_loss this regime reaches is lower than the 2-component regime's typical level,
  a classic training-loss/held-out-error generalization gap from over-parameterization, not an
  inverted signal).

So: **no inversion for 42-vs-1**, and the inversion that *does* exist elsewhere in this project's
evidence is a distinct, third mechanism (prune-rollback → overfitting), not a property of the
42-vs-1 comparison this task specified. Worth flagging since the task text described it as
established fact about "the bad basin" generically.

## 5. What this implies for a fix

1. **The coarse-to-fine schedule's stage-advance loss thresholds (0.1, 0.05) are miscalibrated
   for this class of episode** (low `Hs`), being satisfied within 0–4 gradient steps regardless
   of seed — this is what creates the *race* whose winner is decided by the random phase draws.
   A scale-invariant threshold (e.g., relative to a running loss baseline) or a minimum-iteration
   floor per stage, applied *before* any threshold check, would give the `N=8→15` residual a
   chance to stabilize before the crucial peak-search runs — addressing the timing root cause
   directly, independent of (and complementary to) the sibling phase-determinism fix.
2. **Fixing only the random phase draws (the currently-running ablation) is very likely to
   remove this specific 42-vs-1 divergence** — since with deterministic φ at both sites, there
   is no remaining randomness to make the N=8-stage iteration count (1 vs 4) differ by seed, and
   the fork traced above depends entirely on that count. This trace is consistent with, and
   supportive of, that ablation succeeding.
3. **But it does not obviously fix the seed-3-type failure (prune rollback / overfit at N=15)**,
   which is a capacity/generalization problem, not a phase-randomization problem — a
   deterministic run could still occasionally trip the rollback safety check and retain too many
   DOF. Any future GT-free restart-selection scheme needs to guard against *this* class of
   failure specifically (e.g., held-out validation frames rather than reusing the fitting
   window), separately from the under-prune class already shown to fail in
   `RESTART_SELECTION_RESULT`.
4. **The weak wave is not "sometimes recovered, sometimes not" on this episode — it is
   essentially never recovered** (best candidate found: 0.08–0.09% energy, 3 orders of
   magnitude under any tested prune threshold, though possibly still rising when the iteration
   budget ends — see §3). Any claim of the form "the estimator recovers N true components" for
   sparse, high-energy-contrast fields like this one (3% energy ratio) should be checked
   per-component before being asserted; it was not correct here.

## Labels and limits

- **MEASURED** (n=4 seeds, one episode, bit-exact self-tested against `estimator_variance_results.json`):
  the loss-threshold race (§2), the identical Phase-1 `init_nrmse` across seeds, the periodic
  nature and rough period of seed 1's tracking-phase nRMSE oscillation (n=6 peak gaps — small n,
  imprecise but directionally clear), the redundant-pair finding for seeds 42 and 7 (n=2, but
  mechanistically explained by the energetics, not coincidence), the within-run
  `obs_loss`/`nRMSE` correlations, and the seed-1-vs-seed-3 distinction (both directly read from
  existing `restart_selection_results.json`).
- **ARGUED, not measured**: that fixing both random-phase sites would eliminate this specific
  fork (plausible from the trace, not itself ablated here per instructions); that extending the
  iteration budget would let the true#1-matched component reach a materially higher energy
  (only one component's trajectory inspected, on one seed).
- **UNDERPOWERED / untested**: whether this exact mechanism (loss-threshold race → which peak
  the N=8→15 search finds → 1-vs-N redundant copies → tracking-phase resilience to the
  unmodeled residual) generalizes to the other 3 episodes in the original variance study (5comp,
  15comp) — not traced here, out of scope for this pass. The 15-comp crowded episode's much
  smaller CV (2.8%) is consistent with this same mechanism diluted across many components (per
  `VARIANCE_VERIFY`'s existing argument) but that dilution claim itself remains ARGUED, not
  retraced at this level of detail.

## Files

- `tmp/variance_trace.py` — instrumentation script (subclasses `SequentialUpdater`, adds
  read-only snapshots, calls the real `_adaptive_prune_grow` unmodified via stdout capture for
  the rollback/grow log — zero new RNG consumption, self-tests bit-exact reproduction).
- `tmp/variance_trace_data.json` — raw trace, seeds 42, 1, 3.
- `tmp/variance_trace_seed7.json` — supplementary seed-7 trace (robustness check for §3).
