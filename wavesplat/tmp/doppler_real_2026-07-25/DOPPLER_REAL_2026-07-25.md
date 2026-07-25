# STAGE-3a — Current-Doppler phase-rate term on REAL stereo (analytic estimator)

Date 2026-07-25. Branch `learn-boundwave-retrain`. Protocol frozen in `PREREG.md`
BEFORE this run. Analytic arm only (`mmse_bank_estimate`). No commits, no paper edits,
no training on real data. Raw results: `doppler_real.json`; run log: `run.log`.

--------------------------------------------------------------------------------
## PRE-REGISTERED VERDICT: **REFUTE** (blanket Doppler improvement)

Primary quantity: paired per-window `Delta = nRMSE(U=0) - nRMSE(U_hat)` (positive =
Doppler helps), 27 NORMAL windows (3 known collapse windows pre-excluded), one-sided
Wilcoxon signed-rank, H1 median>0, alpha 0.05.

| stat | value |
|---|---|
| n (normal) | 27 (26 nonzero) |
| mean Delta | **-0.0400** |
| median Delta | **-0.0138** |
| windows improved (Delta>0) | **9 / 27** |
| Wilcoxon (greater) | W=151, **p = 0.733** |

Frozen rule: `median(Delta) <= 0 -> REFUTE`. median = -0.0138 <= 0, and p=0.733 is nowhere
near significance. **A single shared per-window current, selected from the observations to
minimise the held-out observation residual, does NOT reduce real-data nowcast error across
the record population — it slightly hurts on the pooled median.**

Secondary (all confirm the same picture):
- ALL-30 paired: mean -0.0035, median -0.0095, 11/30 wins, Wilcoxon p=0.598.
- Lead-averaged nRMSE (normal): mean -0.0233, median -0.0039, 12/27 wins, p=0.605.

--------------------------------------------------------------------------------
## But the failure is NOT uniform — one record cleanly confirms the mechanism

Per-record paired `d_now` (6 windows each; * = record contains 1 collapse window):

| record | mean d | median d | wins | mean \|U_hat\| | U-vector stability (dir circ-std) |
|---|---|---|---|---|---|
| **YS17**  | **+0.135** | **+0.134** | **6/6** | 1.14 | **6 deg** (tight) |
| AA14 * | +0.134 | +0.015 | 3/6 | 0.73 | 95 deg (scattered) |
| AA15a *| -0.079 | -0.084 | 0/6 | 0.47 | 50 deg |
| AA15b  | -0.115 | -0.065 | 0/6 | 0.47 | 64 deg |
| LJ18g *| -0.093 | -0.026 | 2/6 | 1.34 | 50 deg |

- **YS17 is the whole positive signal.** 6/6 windows improve; nowcast map nRMSE drops
  0.835 -> 0.700 on average (~16% relative). The estimated current is **stable across all
  six independent windows**: `Ux = -0.90 +- 0.14`, `Uy = +0.68 +- 0.12`, `|U| = 1.14 +-
  0.15 m/s`, direction circular-std **6 deg**. Six observation-only estimates, no
  cross-window fitting, converge on one ~1.1 m/s current vector — the signature of a
  *real* ambient current, not a per-window artifact.
- AA14's positive MEAN is an artifact of its excluded collapse window (w0, +0.960); its 5
  normal windows are net negative (mean -0.031, median +0.015). Its per-window U jumps
  sign (Uy circ-std 95 deg) — no coherent current.
- AA15a, AA15b: U_hat is small and directionally scattered; every window is worse or flat.
- LJ18g: large but scattered U_hat, and the base fit barely works here (held-out
  residual `vr_rel ~ 1.0`, i.e. fan << swell wavelength). 4/6 worse.

### The decisive diagnostic — obs-residual gain does NOT track GT gain
Across all 30 windows the correlation between the reduction in held-out **observation**
residual (what U-selection optimises) and the reduction in **GT** nowcast nRMSE (what we
care about) is **r = -0.004** — essentially zero. Every U_hat lowers the observation
residual by construction (mean vr drop +0.019), yet that buys nothing for the GT except in
YS17. **The self-contained U criterion overfits the observations wherever there is no
stable current to lock onto.** Two windows (LJ18g w1, w4) pinned U at the |U|=1.5 boundary
(extrapolation flag); the most spectacular overfits — AA15b w2 (0.688->1.099) and LJ18g w0
(0.981->1.743) — are near-boundary U that shredded the GT while nudging the obs residual
down.

### Effect on the 3 collapse windows
Doppler reduces the collapse magnitude in 2/3 but **fixes none** (all remain nRMSE >= 1.0):
AA14 w0 2.309->1.350, LJ18g w3 2.925->2.721, AA15a w3 1.120->**1.309 (worse)**. Doppler is
not the cause of, nor the cure for, the collapse windows.

--------------------------------------------------------------------------------
## MEASURED vs ARGUED

**MEASURED (from the frozen protocol, GT untouched during U selection):**
1. Pooled pre-registered test REFUTES a blanket Doppler improvement (median -0.0138,
   p=0.733, 9/27 wins).
2. YS17 alone: 6/6 windows improve, mean nowcast nRMSE 0.835->0.700 (~16% rel.), with a
   stable observation-estimated current |U|=1.14+-0.15 m/s, direction circ-std 6 deg.
3. Observation-residual reduction is uncorrelated with GT-nowcast reduction (r=-0.004 over
   30 windows).
4. Baseline (Uc=(0,0)) reproduced the prior `wp_realdata_fixed` WaveSplat numbers
   bit-identically (e.g. YS17 w0 0.741, AA15b w0 0.667), confirming the additive term is
   default-off and the comparison is controlled.
5. Doppler reduces but does not repair collapse windows (2/3 improved, 0/3 fixed).

**ARGUED (interpretation):**
- The record-dependence is consistent with the July attribution being *real but
  conditional*: current-induced Doppler (`omega_obs = omega_intrinsic + k.U`) is an
  estimable, beneficial correction only where (a) a genuine stable ambient current exists
  and (b) the aperture/regime makes the bank fit well-conditioned. YS17 (220x240 m
  real-scale patch, depth 50 m, `vr_rel ~0.6-0.7`) meets both; the small shallow AA patches
  and the fan<<lambda LJ18g storm swell (`vr_rel ~1.0`) meet neither, so the free U knob
  overfits.
- Within-record U-vector stability (direction circ-std) is itself a GT-free discriminator
  of "real current present" vs "overfitting" — 6 deg (YS17) vs 50-95 deg (all others).

--------------------------------------------------------------------------------
## Implication for the sim-real gap attribution: **BOUND (do not over-claim)**

The July GP space-time analysis attributed the sim-real gap to current-Doppler. This test
**bounds** that claim rather than confirming it:
- It is **refuted as a general explanation** of real-data error: adding a self-estimated
  current does not improve the pooled real nowcast, and where the base fit is poor it makes
  things worse.
- It is **confirmed as a real, physical effect in the one record with a stable estimable
  current** (YS17: a coherent ~1.1 m/s current recovered blind from observations, worth
  ~16% nowcast error). So current-Doppler is genuinely present in real seas and matters —
  but it is one conditional contributor, not the dominant driver of the sim-real gap across
  sites/regimes.
- Net: the sim-real gap is only *partly* Doppler, and only where aperture + a steady
  current coincide. This is a publishable, quantified negative-with-caveat.

--------------------------------------------------------------------------------
## Should the Doppler term enter Stage-3b (learned-arm real test)?

**Yes, but only gated/regularised — never as a default-on free per-window U.** Rationale:
1. Handed as a free knob it is a net liability (pooled median negative; r(obs,GT)~0).
2. It pays off precisely where a stable current is detectable, and that condition is
   itself observable GT-free (within-record U-vector consistency; low `vr_rel`).

Recommended Stage-3b form:
- Estimate U per window as here, but **gate** its application on within-record U stability
  and/or `vr_rel` below a threshold (well-conditioned fit); otherwise fall back to U=(0,0).
- Or shrink U toward 0 / penalise the |U|=1.5 boundary so the learned arm cannot exploit
  the overfitting direction.
- Test whether the learned prior changes the obs-vs-GT decoupling (r=-0.004) — if the
  learned prior regularises the bank enough that observation gains transfer to GT, the
  Doppler term could become broadly useful; that is the specific Stage-3b hypothesis this
  run motivates.

--------------------------------------------------------------------------------
## Files written (all under `tmp/doppler_real_2026-07-25/`)
- `PREREG.md` — frozen pre-registration.
- `bench_bitid.py` — bit-identity (Uc=None == Uc=(0,0), max|diff|=0.0 synthetic + real) +
  timing benchmark.
- `run_doppler_real.py` — 30-window U-search + paired analytic-arm comparison (3 workers,
  nice -10, OMP=1).
- `run.log` — full run output.
- `doppler_real.json` — per-window raw results (U_hat, vr, nRMSE before/after, leads).
- `analyze.py` — pre-registered + secondary statistics.
- `DOPPLER_REAL_2026-07-25.md` — this report.

## Deviations from PREREG
- U grid: coarse 0.3 (81 pts) + local 0.1 refine (7x7), as the PREREG specified, because the
  full 0.1 grid (709 pts) projected to ~3 h (> 2.5 h budget). Final U resolution 0.1 m/s.
  No other deviations. Defaults untouched; additive `Uc` term default-off; bit-identity
  verified before the run.
