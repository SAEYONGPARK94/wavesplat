# WaveSplat Novelty — ROUND 3: execution (ANM feasibility gate) + 2 new-direction agents — 2026-07-23

**Prompt (user):** "진행해보고 하면서 추가적 에이전트 투입해서 계속 새로운 방향 탐색." → proceed with the Round-1 lever (gated) AND keep exploring new directions.

## A. EXECUTION — gridless atomic-norm (BLASSO / sliding-Frank-Wolfe) feasibility gate → **FAILS**

Built a self-contained controlled probe (`wavesplat/tmp/blasso_feasibility.py`): shared dispersion-manifold matched-filter atom search, then **GREEDY** (peak→LS amp→subtract, never revisit) vs **GREEDY+JOINT** (greedy then ONE joint L-BFGS refine of all atoms = analog of WaveSplat's existing init→coarse_to_fine Adam polish) vs **SFW** (peak→add→joint re-optimize each round, with L1). Field nRMSE @frame150 (÷Hs); oracle-k = amplitudes fit at TRUE wavenumbers.

| episode | GREEDY | GREEDY+JOINT | SFW | oracle-k | comp recovery |
|---|---|---|---|---|---|
| BF3 5comp s0 | 0.123 | **0.0748** | **0.0747** | 0.070 | both 3/5 |
| BF3 5comp s1 | 0.082 | **0.0787** | **0.0785** | 0.069 | both 4/5 |
| BF4 15comp s0 | 0.239 | **0.1088** | **0.1104** | 0.147 | greedy 9/15, SFW 12/15 |
| BF3 2comp | 0.0505 | – | 0.0508 | 0.0476 | both 1/2 (weak short-λ unrecoverable) |

**Decisive reads:**
1. **SFW ≈ GREEDY+JOINT in every episode.** The large win over *naive* greedy (0.12→0.075, 0.24→0.11) is **entirely joint refinement**, which the current WaveSplat pipeline already does (Adam coarse-to-fine). ⟹ the atomic-norm reformulation gives **no field-error improvement over the existing method.**
2. **Weak short-λ components are unrecoverable by ALL methods at ALL apertures** (2comp |k|=0.75 A=0.037 missed at 3/10/30 s windows; 5comp/15comp miss the smallest-amplitude short comps). ⟹ ANM does **not** supply the missing "achievability" half — the crowded/weak regime is the information wall, exactly as the record says.
3. Only surviving SFW edge: **dense-scene component identification** (15comp: 12/15 vs 9/15) — but it did **not** lower field error (SFW 0.110 ≈ greedy+joint 0.109) and is a single episode. Marginal.
4. (15comp both beat oracle-k 0.147: free-k fits absorb linear-vs-Gerstner model mismatch that fixed-true-k cannot — a modeling nuance, not an ANM win.)

**Gate verdict: NEGATIVE.** Per the Round-1 memo's own decision rule ("if no → fall back to the converse-only CRB + operability-envelope spine, unchanged; no sunk cost"), we route back. Caveat: SFW here is the practical sliding-Frank-Wolfe BLASSO, not an SDP-ANM; and nRMSE is single-frame — but the SFW≈greedy+joint pattern is consistent across 4 episodes, robust enough to fail the gate.

## B. Agent F — method generalization → **DILUTES; theorem is a corollary**

The general "manifold-constrained gridless spectral estimation" theorem **already exists**: Chandrasekaran–Recht–Parrilo–Willsky (FoCM 2012, arbitrary atomic sets), and — decisively — **Yang & Xi, "Gridless Chirp Parameter Retrieval via Constrained 2-D ANM", arXiv:2503.15164 (2025)**: dual-certificate exact recovery of atoms on a **curved locus in 2-D frequency space**, structurally identical to the dispersion shell. Near-field Bessel–Vandermonde SR (arXiv 2604.10234, 2026) does the same for a spherical-wavefront manifold. ⟹ the ocean dispersion-ANM recovery theorem would likely be a **near-corollary**; theoretical novelty is thin. Surviving novelty = ocean-specific geometry + sensor CRB + LiDAR validation, stated **narrowly**. Journal: stay niche. Thesis: generalization OK as framing/lens, not as a claimed theorem.

## C. Agent E — CRB-driven active sensing → **real knob, off-the-shelf method, null-risk; corollary subsection**

The Falcon K genuinely has a programmable steerable dense ROI (~6×, 40°×4.8°) — the "controllable scan allocation" premise is physically real (and absent from `wavesim`, which uses a uniform raster). But the method (submodular FIM/log-det sensor selection: Joshi–Boyd TSP 2009; OED; adaptive/foveated LiDAR) is all off-the-shelf, and the project's own information-wall findings (aperture binds, not sampling density) make a **null likely** except in the under-determined/short-λ regime. Best case = one honest CRB-corollary experiment (a subsection), publishable even as a negative. Not a spine.

## Net after Round 3

The "exciting" Round-1 lever (gridless ANM) does **not** survive: empirically ≈ the current pipeline (A), theoretically a corollary (B). Active sensing (C) and generalization (B) are corollary/dilution. ⟹ **10 orthogonal lenses over 3 rounds all route back to the same modest-solid survivor: the CRB-anchored operability-envelope spine with WaveSplat's existing estimator as the vehicle.** The exploration is saturating: each new "headline" angle either dies or collapses to the spine.

**Recommendation:** shift from angle-hunting to (1) execution/positioning of the CRB-operability-envelope spine, and (2) the two integrity fixes from Round 2 (misattributed 2511.21856; companion RLS-theorem vs Adam-tracker). ONE genuinely-un-probed *synthesis* remains worth a look before stopping: **CRB-anchored / inefficiency-calibrated runtime UQ** — combining the surviving CRB spine with the UQ residue, motivated by the E-59 finding that σ_CRB strongly predicts real-error LEVEL (ρ 0.89–1.00). Launched as new-direction agent G.
