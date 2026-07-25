# Tracking-stage amplitude decay — diagnosis, fix, evaluation (2026-07-25)

Explains the task13 §3.7 finding: on the fine mesh, initialization nRMSE improves −54% (up to
−94% on sparse seas) but tracked nRMSE only −12%, and on the two calmest episodes tracking turns
a near-exact init (~0.009) into ~0.08. Pre-registration: `tmp/TRACKER_AMPDECAY_PREREG_2026-07-25.md`.
Scripts: `tmp/tracker_ampdecay_2026-07-25/`.

## 1. The mechanism (MEASURED)

**Root cause: the tracking warm-start's `φ += ω·dt` phase advance
(`sequential_update.py:update`) is redundant with the forward model's own `−ω·t` term and
freezes the wave at the initialization instant.**

The forward model is `z = Σ wᵢ·cos(kxᵢ x + kyᵢ y − ωᵢ t + φᵢ)` — it already carries absolute
time `t`. Evaluating `forward(t)` with the init-time phase therefore already gives the
correctly-propagated field. Adding `ω·(t−t_init)` to `φ` on top cancels the `−ω·t` evolution,
leaving `cos(kx x + ky y + φ_init − ω·t_init)` — the frozen field at `t_init`. The 10 Adam
steps/frame then have to re-slew the phase back to the truth every single frame; that per-frame
tug-of-war, under Adam momentum, bleeds amplitude.

### Decisive instrumentation (BF3_1comp finemesh; same post-init state per variant)

| variant | what it isolates | avg nRMSE | sum\|w\| trajectory |
|---|---|---|---|
| init (fitted, seed42, N=2) | baseline reference | 0.0138 | — |
| **gt-inject init** (true k/A, N=1) | is there a physics floor? | **0.0000** | — |
| V4 gt + phase-only (no grad) | pure `φ+=ω·dt` on a perfect model | **0.3165** | 0.212→0.212 |
| V5 gt + baseline tracking | shipped tracking from perfect init | **0.1633** | 0.198→**0.138 (−30%)** |
| V0 baseline (seed42) | shipped | 0.0340 | 0.199→0.221 |
| V2 L1=0 (seed42) | is L1 the cause? | 0.0734 (sum\|w\|→0.55, explodes) | — |
| V0 baseline (seed1, N=1) | worst basin | **0.1633** | 0.204→**0.139 (−32%)** |
| V2 L1=0 (seed1) | is L1 the cause? | **0.1631** (≈ V0) | 0.205→0.142 |
| V3 freeze-w (phi_only grad) | phase-gradient alone | 0.30–0.32 | frozen |

Reading:
1. **No physics floor.** The true single component reproduces the scored GT field exactly
   (0.0000). Gerstner nonlinearity affects only the LiDAR raycast, not the elevation field nRMSE
   scores. So the degradation is not model mismatch.
2. **The phase advance freezes the wave.** A *parameter-exact* model, propagated only by
   `φ+=ω·dt`, decorrelates to 0.3165 — the antiphase noise level. This is the smoking gun.
3. **Amplitude decay reproduces from a perfect init** (V5: 0.16, sum|w| −30%): so it is not
   caused by init imperfection, contra the intuitive reading. It is the gradient fighting the
   frozen-wave kick each frame; a cos component with a forced phase error is best-fit by shrinking
   its amplitude, so Adam bleeds |w|.
4. **L1 is not the cause.** `lambda_l1=0` gives 0.1631 vs 0.1633 (seed1) — identical decay; with
   the looser N=2 basin, removing L1 makes it *worse* (amplitude explodes to sum|w|=0.55). L1 is
   beneficial regularization here.
5. **Decays even with perfect init and the scored field being noise-free-linear** (V5). It is not
   a raydrop-sparsity artifact — it is intrinsic to the phase-advance step.

**Relation to the paper (untouched, for the main session).** `wavesplat.tex` (~L2200) attributes
this to "Adam momentum accumulat[ing] oscillating gradient signals when the phase prediction
φ+=ωΔt is slightly inaccurate." The oscillation and amplitude-decay symptoms are real, but the
attribution is wrong: the phase prediction is not *slightly* inaccurate, it is *systematically
redundant* and freezes the wave; the amplitude decay is a downstream symptom, not a separate
Adam pathology. The limitation paragraph and the VR-deficit attribution (VR=0.83) would need
revision if the fix is adopted. **Paper not edited — flagged only.**

## 2. The fix

`TRACK_PHASE_PREDICT` (module flag, `sequential_update.py`). Default **True** = shipped (execute
`φ+=ω·dt`, verified bit-identical: `control_behaviour_preserved.py` 6/6). **False** skips the
redundant advance so `forward(t)` tracks true evolution directly. Single candidate — the
mechanism is unambiguous, so amplitude-anchor / L1-exemption would only treat symptoms.

## 3. Pre-registered adoption rule (frozen before eval)

Adopt iff ALL of: **R1** improve on ≥4/6 flag-sweep regimes; **R2** mean Δ over 12 episodes
< −0.031; **R3** both calm episodes' init→track gap shrinks >50%. (Full text in the prereg file.)

## 4. Evaluation (median-of-5, finemesh, paired) — ADOPT

Both arms produced on the same device (GPU) from an **identical** post-init snapshot per
(episode, seed): init once, snapshot, track twice (flag OFF = shipped, flag ON = fix). See the
DEVIATION note in the prereg (CPU baseline could not be reused — adaptive-N is FP-sensitive and
the GPU init diverges from the CPU init, breaking the per-seed pairing). Matrix verified complete:
60/60 (episode, seed) pairs, no missing, no errors, no duplicates, both arms each.
`ampfix_paired_runs.jsonl`; assembled by `analyse_paired.py` (`analysis_final.txt`).

| episode | OFF med (sd) | ON med (sd) | Δ | init | dir |
|---|---|---|---|---|---|
| BF2_1comp    | 0.1665 (0.023) | 0.0285 (0.006) | **−0.1380** | 0.036 | ↓ |
| BF3_1comp    | 0.1633 (0.058) | 0.0087 (0.002) | **−0.1546** | 0.009 | ↓ |
| BF3_2comp    | 0.1319 (0.087) | 0.0129 (0.002) | **−0.1190** | 0.010 | ↓ |
| BF3_5comp    | 0.2054 (0.036) | 0.0242 (0.009) | **−0.1812** | 0.014 | ↓ |
| BF4_1comp    | 0.1032 (0.060) | 0.0044 (0.004) | **−0.0988** | 0.004 | ↓ |
| BF4_5comp    | 0.2031 (0.008) | 0.0187 (0.006) | **−0.1843** | 0.007 | ↓ |
| BF4_15comp   | 0.3004 (0.007) | 0.2795 (0.005) | −0.0209 | 0.300 | ↓ |
| BF4_30comp   | 0.2974 (0.028) | 0.2817 (0.029) | −0.0157 | 0.293 | ↓ |
| BF5_15comp   | 0.2770 (0.020) | 0.2336 (0.014) | −0.0434 | 0.222 | ↓ |
| BF5_30comp   | 0.2910 (0.014) | 0.2536 (0.029) | −0.0373 | 0.231 | ↓ |
| BF3_2comp_slope | 0.0765 (0.095) | 0.0118 (0.003) | **−0.0647** | 0.009 | ↓ |
| BF3_5comp_slope | 0.1993 (0.033) | 0.0164 (0.002) | **−0.1830** | 0.015 | ↓ |

Aggregate (12 episodes): **improve 12/12**; mean Δ **−0.1034**; OFF mean-of-medians 0.2013 → ON
0.0978 (−51%); OFF median-of-medians 0.2012 → ON 0.0215 (−89%). Every ON-arm episode with a
resolvable init (≤5-component) now tracks essentially at its initialization value; the three
truly crowded 15–30c episodes with a bad init still improve, just modestly.

### Verdict against the frozen rule
- **R1** (≥4/6 flag-sweep regimes improve): **6/6** → PASS.
- **R2** (mean Δ over 12 < −0.031): mean Δ = −0.1034 → PASS.
- **R3** (both calm episodes' init→track gap shrinks >50%): BF3_1comp gap +0.1543 → −0.0003
  (**100%**); BF4_1comp gap +0.0990 → +0.0002 (**100%**) → PASS.
- **ADOPT = TRUE** (R1 ∧ R2 ∧ R3).

Per the standing constraint ("never change default behaviour"), the shipped default is left at
`TRACK_PHASE_PREDICT=True` (previous behaviour, bit-identical). The rule qualifies the flag for
adoption; **flipping the default to `False` and propagating to the paper is left to the main
session** because it changes headline tracking numbers.

### Secondary finding — the fix also collapses the calm/sparse run-to-run variance
The estimator-variance report (`tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md`) measured a
**bimodal CV up to 53%** on BF3_2comp; my OFF arm reproduces it (CV 56.1%, sd 0.087). The fix
collapses the **absolute** run-to-run spread on exactly the previously-bimodal sparse episodes:
BF3_2comp sd 0.087→0.002, BF3_2comp_slope 0.095→0.003, BF3_1comp 0.058→0.002, BF4_1comp
0.060→0.004 (≈15–50× tighter). Mechanism: the bimodality was the *broken tracker* amplifying a
small init-basin difference (N=1 vs N=2) into 0.16 vs 0.03; with `forward(t)` no longer frozen,
both basins land at ~0.008. (Relative CV is mixed because the mean collapses too; the absolute
spread is the honest statistic here.) So the random-phase-init variance and this tracker defect
were partially the same phenomenon at the delivered-nRMSE level.

## 5. Files
- Flag + gating: `wavesplat/optimizer/sequential_update.py` (`TRACK_PHASE_PREDICT`, default True).
- Diagnosis: `tmp/tracker_ampdecay_2026-07-25/diag_tracker.py`, `diag_gpu.log`.
- Pre-registration: `tmp/TRACKER_AMPDECAY_PREREG_2026-07-25.md`.
- Paired eval: `tmp/tracker_ampdecay_2026-07-25/ampfix_paired.py`, `ampfix_paired_runs.jsonl`,
  `shard_{0,1,2}.log`; analysis `analyse_paired.py`, `analysis_final.txt`.
- Bit-identity control: `tmp/rerun_2026-07-24/control_behaviour_preserved.py` → 6/6 bit-identical.

## 6. Paper implication (NOT edited — for the main session)
`wavesplat.tex` L2170–2213: (a) the "Tracking phase dependence on initialization" paragraph's
framing that tracking is *inherently* capped by init because `w_phi` freezes wavenumbers is only
half the story — the dominant cause of the observed cap was the redundant phase-advance defect,
not the frozen wavenumbers; with the fix, tracking recovers to ≈init even though wavenumbers stay
frozen. (b) The "Amplitude decay during tracking" paragraph (L2200) and its VR-deficit
attribution (VR=0.83) should be rewritten: the decay is a downstream symptom of the frozen-wave
tug-of-war, not a separate Adam-momentum-vs-inaccurate-phase pathology. (c) The task13 §3.7
narrative ("most of what a better sensor buys is realised at init and given back by tracking")
inverts under the fix: tracking no longer gives it back.
