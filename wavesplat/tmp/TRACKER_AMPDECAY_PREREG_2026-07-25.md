# Pre-registration — tracking-stage amplitude-decay fix (2026-07-25)

Written **before** the evaluation runs. Fixes the candidate flag, the evaluation set, and the
adoption rule so the read-out cannot be chosen after seeing the numbers. Companion diagnosis and
results: `tmp/TRACKER_AMPDECAY_2026-07-25.md`.

## 1. Measured mechanism (diagnosis, `tmp/tracker_ampdecay_2026-07-25/diag_tracker.py`)

On the two calmest finemesh episodes (BF3_1comp, BF4_1comp — 1 component), initialization
reaches field nRMSE ~0.005–0.014 but tracking degrades to 0.05–0.16. Instrumented variants from
the same post-init state (BF3_1comp, seeds 42 and 1) show:

- **The scored GT field is pure-linear.** Injecting the TRUE single component (dispersion ω)
  gives init nRMSE **0.0000** — the Gerstner nonlinearity lives only in the LiDAR raycast, not in
  the elevation field that nRMSE scores. So there is **no physics floor** to blame.
- **The `φ += ω·dt` phase-prediction step (sequential_update.py `update`) is redundant with the
  forward model's own `−ω·t` term and freezes the wave at the init instant.** With a
  parameter-exact model and NO gradient, pure `φ+=ω·dt` propagation decorrelates to
  **avg nRMSE 0.3165** (= the antiphase noise level for a single wave). Removing the gradient is
  not the issue; the phase advance itself is the defect.
- **From a perfect init, shipped tracking (gradient ON) still decays to 0.1633 with sum|w| −32%**
  (0.198→0.138). This reproduces the paper's "amplitude decay" — but from a *perfect* init, so it
  is not caused by init imperfection.
- **L1 is not the cause.** Setting `lambda_l1=0` leaves the decay essentially unchanged
  (0.1631 vs 0.1633 on seed 1); with the sparser/looser N=2 basin it makes things *worse*
  (amplitude explodes, sum|w| 0.20→0.55). So L1 is beneficial regularization, not the driver.

Root cause: the forward model already evolves the wave (`z = Σ w·cos(kx x + ky y − ω t + φ)`), so
evaluating `forward(t)` with the init-time phase already yields the correctly-propagated field.
The extra `φ += ω·dt` cancels the `−ω·t` evolution and pins the field at `t_init`; the 10 Adam
steps/frame must re-slew the phase every frame, and that per-frame tug-of-war (Adam momentum)
bleeds amplitude. **The paper's stated mechanism ("Adam momentum accumulates oscillating gradient
signals when the phase prediction is *slightly* inaccurate") is directionally right about the
oscillation but misattributes the cause: the phase prediction is not slightly inaccurate, it is
systematically redundant and freezes the wave.** (Reported to the main session; paper left
untouched.)

## 2. Candidate fix (single candidate)

`TRACK_PHASE_PREDICT` — module-level flag in `wavesplat/optimizer/sequential_update.py`.
- **Default `True`** = shipped behaviour: execute `φ += ω·dt` (bit-identical to pre-edit code).
- **`False`** = skip the redundant advance; `forward(t)` tracks the true time evolution directly,
  leaving only genuine slow drift from wavenumber error for the gradient to correct.

Only one candidate is registered because the mechanism is unambiguous; an amplitude-anchor or
L1-exemption would be treating symptoms of this one defect.

## 3. Evaluation set

- Episodes: the 12 representative Table-1 episodes (the task13 / `ws_runs` set):
  `BF2_1comp, BF3_1comp, BF3_2comp, BF3_5comp, BF4_1comp, BF4_5comp, BF4_15comp, BF4_30comp,
  BF5_15comp, BF5_30comp` (uniform) + `BF3_2comp_slope`, `BF3_5comp_slope`, all `seed0`.
- Data: `data_finemesh_2026-07-23/`. Device: CPU (matches the reused baseline).
- Seeds: [42, 1, 2, 3, 4]; report **median-of-5** with spread (per the median-of-k convention).
- Paired: the flag changes only the tracking phase, so init and adaptive-N are identical per
  seed; comparison is paired at the episode level.
- **Flag-OFF baseline** = the finemesh P1 runs already in `tmp/rerun_2026-07-24/task13_runs.jsonl`
  (verified bit-identical to `ws_runs.jsonl` P1, max diff 0.0 over 60 runs). Confirmed reusable
  once the bit-identity control passes for the edited default path.
- **Flag-ON** = re-run the 12×5 with `TRACK_PHASE_PREDICT=False`, P1 config → `ampfix_runs.jsonl`.

## 4. Adoption rule (FROZEN before measuring)

Per-episode statistic: median-of-5 `avg_nRMSE`. Δ = median(ON) − median(OFF); Δ<0 = improvement.

Adopt the flag as the new default **iff ALL THREE hold**:

- **R1 (direction, literal repo ≥4/6 rule):** on the 6 flag-sweep regimes
  {BF3_2comp, BF3_5comp, BF4_5comp, BF4_15comp, BF4_30comp, BF5_15comp}, the paired median
  improves (Δ<0) on **≥4 of 6**.
- **R2 (magnitude, seed-sd floor):** the mean Δ across the full 12-episode set is an improvement
  exceeding the floor, i.e. **mean Δ < −0.031**.
- **R3 (calm regression shrink):** for BF3_1comp and BF4_1comp, the init→track gap
  (gap = median_track − median_init) shrinks by **>50%** under the flag (both episodes).

If not all three hold → **NOT adopted**; the flag stays documented and default-off, exactly like
the four 2026-07-24 flags.

Reported regardless of verdict: 12-episode direction count, 12-ep aggregate median & mean OFF vs
ON, per-episode median(sd)/min/max scatter, the 6-regime subset mean Δ, and the two calm-episode
gap-shrink numbers.

### DEVIATION (recorded during execution, before reading the aggregate)

The planned reuse of the CPU `task13` baseline was **abandoned** for a stricter method: the flag
runs on the GPU (the CPU is saturated by another agent's ETKF jobs), and the adaptive-N path is
FP-sensitive, so a per-seed init on GPU can diverge from the CPU init (observed: BF2_1comp seed1
init 0.035 on GPU vs 0.063 on CPU) — which would break the per-seed pairing. Instead, both arms
are produced on the **same device (GPU)** by a single harness (`ampfix_paired.py`) that
initializes **once** per (episode, seed), snapshots the post-init model, and runs the tracking
stage **twice** (flag OFF then ON) from that identical snapshot. This makes the pairing exact
(identical init and adaptive-N per seed) rather than approximate. The bit-identity control
(6/6) still validates that the code edit does not change the default path. The adoption rule
(R1/R2/R3) is unchanged. The OFF arm reproduces the diagnostic's shipped-tracking numbers
(BF3_1comp seed42 OFF=0.0340, matching diag V0=0.0340).

## 5. Compute / discipline

CPU-capped (`OMP_NUM_THREADS`≤4), `nice`, ≤3 concurrent (box shared with a CPU-heavy ETKF job).
Default behaviour never changes. Repo tmp only. No commit. Paper not edited.
