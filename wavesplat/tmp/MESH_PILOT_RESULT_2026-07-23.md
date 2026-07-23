# Mesh-refinement pilot (n=12) — does the paper's headline actually move? — 2026-07-23

Script `tmp/mesh_headline_subset.py`, log `tmp/mesh_headline_subset.log`, results
`tmp/mesh_headline_subset_results.json`. Pre-registered reads were stated in the script
header **before** any numbers (three outcomes A/B/C, see the docstring).

**Design.** The paper's own 12-episode main comparison set (`scripts/run_p1_comparison.py:EPISODES`),
each regenerated **paired** — same seed ⇒ identical wave components — at the shipped mesh
(40×30, dx 6.15 / dy 6.90 m) and a refined mesh (180×150), then WaveSplat-P1 and ETKF re-run
on both. Nothing in `data/` was touched; output went to `tmp/mesh_subset_ep/`.

## Result — the aggregate headline does **not** move

| | coarse | fine | paired Δ | SE | t |
|---|---|---|---|---|---|
| WaveSplat mean nRMSE | 0.2135 | 0.2055 | **−0.0080** | 0.0184 | −0.44 |
| ETKF mean nRMSE | 0.4361 | 0.4416 | **+0.0055** | 0.0169 | +0.33 |
| margin (ETKF−WS)/ETKF | 51.0 % | 53.5 % | **+2.3 pts** | 4.5 | +0.51 |

**All three are indistinguishable from zero (n = 12).** 6/12 episodes improved, 6/12 did not.

## But there is a real, large **interaction with component count**

| episode | WS coarse → fine | Δ% |
|---|---|---|
| BF2 1comp uniform | 0.3150 → 0.2251 | **−28.5 %** |
| BF3 1comp uniform | 0.2173 → 0.0946 | **−56.5 %** |
| BF3 2comp uniform | 0.0524 → 0.0418 | −20.2 % |
| BF3 2comp slope | 0.0607 → 0.0379 | −37.5 % |
| BF4 1comp uniform | 0.1706 → 0.1612 | −5.6 % |
| BF3 5comp uniform | 0.1675 → 0.3104 | **+85.4 %** |
| BF3 5comp slope | 0.1517 → 0.1488 | −1.9 % |
| BF4 5comp uniform | 0.1996 → 0.2202 | +10.3 % |
| BF4 15comp uniform | 0.2999 → 0.3002 | +0.1 % |
| BF4 30comp uniform | 0.3194 → 0.3315 | +3.8 % |
| BF5 15comp uniform | 0.3086 → 0.2858 | −7.4 % |
| BF5 30comp uniform | 0.2994 → 0.3085 | +3.0 % |

**n_comp ≤ 2 (n=5): WS −29.7 % · n_comp ≥ 5 (n=7): WS +13.3 %.** The two halves cancel.

## What this corrects

1. ⛔ **"The paper's absolute headline numbers (0.223 / 0.319 / 30.1 %) are substantially
   artifact-inflated" is NOT supported.** That claim came from an n=1 probe on
   **BF2_1comp_seed0 — the paper's quoted *worst case*, and the single most sub-Nyquist
   episode in the set** (1.18 points per wavelength at the peak). It is the extreme, not the
   typical. At n=12 the aggregate effect is +2.3 ± 4.5 margin points, i.e. nothing.
   *(Third self-correction of the day; this one in the alarmist direction. The severity
   rating "MATERIAL verging on FATAL for absolute numbers" was over-generalized from n=1,
   by the red-team reviewer and then amplified by me.)*
2. ✅ **What survives unchanged:** the mesh defect itself is real and undisputed — **0 of 102
   shipped BF2–BF5 episodes pass `wavesim/mesh_adequacy.py`** (50 INADEQUATE, 52 DEGRADED),
   independently re-derived by two routes. Regeneration is still warranted for
   **correctness and disclosure**, and it is a prerequisite for the short-λ re-checks.
   It is **not** going to rewrite the paper's headline.
3. ⚠ **The short-λ alarm is weakened too.** The worry was that a mesh unable to render short
   wavelengths made "weak short-λ components are unrecoverable by all methods" circular. If
   restoring short-λ content mattered much, the many-component episodes (15/30 comp, which
   carry the short-λ tail) should have shifted most. **They barely moved at all**
   (+0.1 %, +3.8 %, −7.4 %, +3.0 %). The direct re-run of that claim is still required, but
   the expected effect is small.

## Separate finding worth its own line: **estimator run-to-run instability**

`BF3_5comp_seed0` has *identical wave components* in its uniform and slope arms, yet the
uniform arm moved **+85.4 %** while the slope arm moved **−1.9 %**. A swing of that size
from a change that leaves the wave field untouched means **WaveSplat's episode-level
variance is comparable to every effect measured here.** The paper reports single numbers per
episode. This should be characterized (repeat seeds / restarts) before any small per-episode
delta is cited — including the deltas in this very table.

## Caveats

- ETKF here is a fixed configuration (40×30 analysis grid, Ne=50), **not** the paper's
  "hyperparameter-tuned" ETKF, so the absolute ETKF column is not comparable to the paper's
  table. The valid quantity is the **paired coarse→fine delta**, which is what is reported.
- A separate, unresolved issue stands: on refined-mesh data ETKF *degrades* in several
  episodes and a tuning sweep (grid/Ne/inflation ×6, then obs_noise) failed to close it —
  see `REGEN_SPEC_fable_2026-07-23.md` §4. **No ETKF number from fine-mesh data may be cited
  until that is understood.**
- Synthetic linear episodes, single seed per configuration, n=12. Nonlinear (HOS) not tested.
