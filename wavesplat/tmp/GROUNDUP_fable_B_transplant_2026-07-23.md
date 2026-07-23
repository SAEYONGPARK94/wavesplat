# Ground-up review — Reviewer B (cross-domain transplant), 2026-07-23

> Lens: the WaveSplat measurement geometry — moving, one-sided, grazing-incidence, sparse,
> specular-gated sampler of a dispersive travelling field — recurs across disciplines. This
> document checks each candidate discipline against the actual code/data of this repo (not
> against the abstract idea of the discipline), and against a WebSearch literature check, and
> against the project's own 13-item exclusion list. Method: state the isomorphism precisely,
> then the disanalogy, then test the disanalogy against code/data, then (if it survives)
> design the cheapest kill experiment.
>
> **Bottom line up front: clean null.** Every candidate transplant examined either (a) is
> **already implemented inside this repo under different vocabulary** (convergent invention —
> not a new lever), (b) **collides with one of the 13 excluded items** the moment you trace it
> to its one non-trivial application here, or (c) is **foreclosed by a hard, code-verified fact**
> of this sensor/trajectory/dataset. No survivor clears all three checks. Two implementation-
> level micro-tweaks are flagged honestly as unresolved-but-low-value (§5).

---

## 0. Structural facts of this project, verified in code/data (not from memory)

These facts are the disanalogy-checking ground truth used throughout. All read directly from
`wavesplat/wavesim/*.py` and three actual `.h5` episodes on 2026-07-23.

| Fact | Value | Source |
|---|---|---|
| Points per frame | **~50, constant** across BF2/BF3/BF4, 1–30 components, uniform/slope raydrop (measured 49.75–49.83 on 3 episodes; `config.episode.n_pts=50` is a fixed *target*, not a function of sea state) | `wavesim/config.py:67`, `wavesim/pipeline.py:45,193,301`; measured on `episode_BF2_1comp_seed0_slope.h5`, `episode_BF4_30comp_seed0_uniform.h5`, `episode_BF3_5comp_seed0_uniform.h5` |
| Frame rate / episode length | 10 Hz, 301 frames = 30.0 s | `wavesim/config.py` `EpisodeCfg` |
| FOV | 120° az × 25° el, one-sided, forward-looking, centered on ship heading | `wavesim/config.py` `LidarCfg`, `lidar_simulator.py:278-307` |
| Range | 1.5–250 m | `LidarCfg` |
| Ship trajectory | **Deterministic straight line**, `ship_pos(t) = origin + v·t` (constant heading/speed), in every script that constructs it (`d3_hull_value_test.py`, `d3_badwin_forensics.py`, `d3_required_horizon.py`, `d3_dualband_test.py`, `d3_e10_pattern_real.py`, `d3_falcon_hos_probe.py`). No loop, no revisit, no curved course found anywhere in the repo. | grep across `wavesplat/scripts/*.py` |
| Platform pose noise | **None injected.** Ship position/heading feed ray geometry directly (`ship_x=0.0, ship_y=0.0` literal in `pipeline.py:127`); no navigation/attitude error model exists in `wavesim/`. | `wavesim/pipeline.py`, `wavesim/lidar_simulator.py` |
| Return-selection physics | Two raydrop models: `uniform` (Bernoulli w/ Fresnel·cosθ/d²) and `slope` (**specular**: return iff reflected ray realigns with sensor within `alpha_threshold_deg=5°` cone — i.e., normal ≈ −ray direction) | `wavesim/raydrop_uniform.py`, `wavesim/raydrop_slope.py` |
| Per-observation side info | `incidence_angle`, `confidence` recorded and loaded, but used **only as a scalar heteroscedastic weight** `cos θ/(d/d_ref)` on the height residual — never as an independent slope-direction equation | `wavesplat/data_loader.py:77-98`, `wavesplat/model/confidence.py` |
| Dispersion | Single branch, finite depth, hard-constrained: `ω = √(g·\|k\|·tanh(\|k\|·d))` | `wavesplat/model/dispersion.py`, CLAUDE.md |
| Receiver channels | **One** physical sensor (scanning ToF LiDAR), not an array of independently-erred receivers | `wavesim/config.py` `LidarCfg` (single unit) |
| Initializer | `_beamform_coarse` (periodogram power map over (kx,ky), dispersion-aware) → `_beamform_refine` → direction sweep + linear projection. Explicitly the **RELAX** family (Li & Stoica 1996): greedy coarse grid + Newton refine + **cyclic joint refinement** of all components after each addition. | `wavesplat/optimizer/initializer.py`, `wavesplat/optimizer/relax_estimator.py` (docstring cites Li & Stoica 1996 directly) |
| Tracker | Predict `φᵢ += ωᵢ·Δt`, then 5–10 Adam gradient steps against the window | `wavesplat/optimizer/sequential_update.py` |

Two facts do almost all of the disanalogy work below: **(1) ~50 sparse points/frame** (no dense
per-snapshot image exists to correlate, tomo-reconstruct, or grid with density compensation
worth the trouble), and **(2) the trajectory and platform pose are deterministic, given, and
noise-free** (nothing exists in this simulator for a self-calibration/autofocus technique to
correct).

---

## 1. Isomorphism / disanalogy table

| Source field & technique | Structural isomorphism (what maps to what) | Disanalogy, checked against this repo's code/data | Verdict |
|---|---|---|---|
| **Seismology — migration / f-k velocity analysis** | Recorded surface field at known (x,t) → back-project through a known dispersion/velocity model into (kx,ky) energy | None — this **is** what `_beamform_coarse` already does (dispersion-aware periodogram over (kx,ky), i.e. migration/f-k stack under a different name) | **Already implemented** (convergent invention, not new) |
| **Seismology — blended acquisition / deblending** | Many overlapping "sources" (wave components) superposed in one recording, separated via sparsity | Deblending is a sparse-recovery/L1 technique — same family as atomic-norm/BLASSO, already gated NEGATIVE (excluded item 3) | **Collides with #3** |
| **Seismology — ambient-noise interferometry / virtual-source Green's-function retrieval** | Cross-correlate the field at two along-track "virtual receivers" (positions the moving ship occupied at different times) to get the direct travel-time/Green's function between them, bypassing wavenumber decomposition | Real published technique, but requires (i) a **diffuse, quasi-isotropic** source field and (ii) **long stacking** (the seafloor-DAS ocean-wave analogue, Williams et al. 2022 JGR Oceans, uses dense fixed arrays and long observation windows). This project has (i) narrow directional-spread swell components (not diffuse) and (ii) 30 s episodes under a ≤100 ms control-latency budget. Both preconditions structurally violated. | **Killed by data** (short/non-diffuse) |
| **Radio astronomy — aperture synthesis** | Moving platform integrates (u,v) coverage over time like an interferometer array | Already tried and killed: the "moving platform buys aperture" premium evaporated on replication (excluded item 13) | **Excluded (#13)** |
| **Radio astronomy — CLEAN deconvolution** | Iteratively find peak in dirty map, subtract a scaled beam, repeat | This **is** the existing greedy-add step of `relax_estimator.py`, explicitly RELAX (Li & Stoica 1996) — the array-processing sibling of CLEAN | **Already implemented** |
| **Radio astronomy — "dirty beam"/PSF of the actual (kx,ky,ω) sampling pattern** | Deterministic sidelobe structure induced by *this* non-uniform sample set, independent of noise | Not identical to CRB (deterministic geometry vs. noise-weighted variance bound) but is the beamform-power-map already computed by `_beamform_coarse` — nothing new to run; at best a **figure/exposition** reuse, not an empirical lever, and sits adjacent to excluded item 2's territory | **Already implemented; expository only** |
| **Radio astronomy — closure phase / self-calibration** | Use redundancy among receivers to solve for unknown per-receiver phase/gain error jointly with the image | Requires ≥1 **unknown nuisance calibration parameter per receiver/sample**. Verified: no pose/attitude/timing noise is injected anywhere in `wavesim/`; ship position feeds ray geometry as ground truth. Nothing exists to self-calibrate. Adding such noise would be a new simulator capability → excluded item 9. | **Killed by data + collides with #9** |
| **MRI — non-Cartesian k-space / NUFFT gridding + density compensation** | Irregular (x,y) samples → grid for spectral analysis, as the initializer's `bin_observations_to_grid`/`fft_peak_extraction` already does (naively, no density compensation) | Plausible micro-improvement to *initialization only*; likely swallowed by the downstream Newton/L-BFGS joint refine (which already re-fits from any starting point). Borderline "better estimator" territory (excluded item 1 in spirit). | **Low-value, likely inert** — see §5 |
| **MRI — parallel imaging (SENSE/GRAPPA)** | Multiple receive coils with distinct spatial sensitivity fill in undersampled k-space | Requires **multiple independent receive channels**. This project has one physical LiDAR. No coil-sensitivity diversity to exploit. | **Killed by data** (single channel) |
| **MRI/CS — compressed-sensing trajectory design** | Choose sampling trajectory to satisfy incoherence for sparse recovery | Trajectory is exogenous by project scoping (excluded item 12); CS reconstruction itself is the sparse-recovery family (excluded item 3/4) | **Excluded (#12, #3/4)** |
| **SAR — phase-gradient autofocus (PGA)** | Estimate & correct unmodeled platform-motion phase error from data redundancy | Same object as closure-phase self-cal above: no unmodeled pose error exists in this sim to autofocus against; adding one is a new simulator feature (#9) | **Killed by data + collides with #9** |
| **SAR/SAR-interferometry — along-track interferometry (ATI) for current/velocity** | Two simultaneous looks at the same patch from slightly offset along-track positions → phase-difference gives radial (current) velocity directly, without wavenumber decomposition | (i) Requires **two simultaneous receive apertures**; single LiDAR has one. (ii) The classical marine-radar cousin (X-band radar dispersion-shell current retrieval, confirmed via WebSearch: 3D FFT of *dense, consecutive full-scan images* → subtract dispersion shell → least-squares Doppler fit) needs **dense per-snapshot spatial coverage**; this project has ~50 pts/frame, no per-snapshot image to 3D-FFT. (iii) The underlying unknown (single sparse track only observes combined ω_intrinsic + k·Uc) is exactly the already-tried, already-failed blind-current estimation (4 objective functions, non-identifiable from a single sparse track — prior project finding, memory: "해류 블라인드 추정 4개 목적함수 실패"). | **Killed by data (sparsity + single channel) + re-hits an already-failed identifiability wall** |
| **SAR — synthetic aperture sonar micronavigation** | Same as PGA, sonar flavor | Same kill as PGA/autofocus | **Killed by data + collides with #9** |
| **Computational imaging — light field / coded aperture** | Physically coded sensing element trades spatial for angular/spectral resolution | Requires a **different physical sensor** (a coding mask, or an event-based pixel array). This project's sensor (Seyond Falcon) is given/fixed by explicit project scoping — swapping sensor hardware is out of scope, and it also collides with excluded item 9 (simulator/systems contribution). | **Killed by scope** |
| **Computational imaging — event cameras** | Asynchronous per-pixel change detection instead of synchronous frames | Falcon LiDAR is a synchronous 10 Hz scanning ToF unit; no event-based analogue exists or is being proposed as new hardware. Same scope kill as above. | **Killed by scope** |
| **Robotics — SLAM loop closure** | Revisit a previously-seen landmark to correct accumulated drift | Verified in code: ship trajectory is a **single deterministic straight pass** in every script found; it never revisits a spatial location within an episode. Ocean waves also propagate (no static landmark to loop-close on even if it did revisit). | **Killed by data (no revisit) + no landmark exists** |
| **Robotics — factor-graph / bundle-adjustment reformulation of the tracker** | Recast the predict/gradient-correct loop (`φᵢ += ωᵢΔt` + Adam steps) as a sparse nonlinear least-squares factor graph (iSAM-style) | This is a reformulation of the sequential/streaming/recursive estimation problem — explicitly owned by a companion paper (excluded item 7); memory also flags an unresolved discrepancy between a companion "streaming≡batch(RLS)" proof and the actual Adam-based tracker, which is that paper's territory to resolve, not a fresh transplant here | **Collides with #7** |
| **Robotics — nonlinear observability analysis (Lie-derivative rank tests)** | Determine which state directions are locally distinguishable given the measurement map and trajectory | Mathematically the same family of object as local Fisher information/CRB (rank of an observability Gramian vs. eigenstructure of the FIM) — this project's information/identifiability characterization is explicitly the current thesis spine (excluded item 2) | **Collides with #2** |
| **Robotics/sensor-networks — graph rigidity theory** | Combinatorial (not statistical) criterion for whether a network of pairwise-range measurements uniquely localizes its nodes | Rigidity theory acts on a **graph of multiple nodes with pairwise distance/range edges** (classic multi-agent SNL/SLAM pose-graph setting). This project is a single agent sampling a continuous field — there is no multi-node pairwise-range graph object for rigidity theory to act on. Forcing the analogy ("is the time-sample graph rigid for recovering (kx,ky)?") just restates identifiability/CRB in graph words. | **No object exists; restates #2** |
| **Fluid experiment — PIV (cross-correlate image patches between frames)** | Track local displacement by correlating small dense image windows across time | Needs **locally dense regularly-sampled patches**. At ~50 scattered points/frame there is no dense window to correlate; would require gridding first, re-entering the already-explored sparse-recovery family. | **Killed by data (sparsity)** |
| **Fluid experiment — tomographic reconstruction (tomo-PIV, ART/SART)** | Combine multiple simultaneous viewpoints algebraically into a 3D reconstruction | Requires **multiple simultaneous viewpoints** of the same volume; single LiDAR has one viewpoint at a time. Using time-sequential views instead is the already-killed synthetic-aperture premise (#13). | **Killed by data + collides with #13** |
| **Fluid experiment — sub-pixel/parabolic peak interpolation** | Refine a discretized correlation/FFT peak to sub-bin precision before downstream use | Cheap, generic numerical technique. `fft_peak_extraction` in `initializer.py` currently picks discrete FFT bins with no sub-bin refinement. Same "likely swallowed by joint refine" argument as the MRI gridding item. | **Low-value, likely inert** — see §5 |
| **NDT/guided-wave ultrasonics — dispersion compensation / warped-frequency mode separation** | Time-frequency warping to separate multiple overlapping *dispersive modes* sharing a time/frequency region (e.g., Lamb S0/A0) | Confirmed via WebSearch as a real, mature technique (warped frequency transform, generalized dispersive mode decomposition). But it targets **multiplicity of physical dispersion branches**; ocean gravity waves here have a **single** branch (`ω=√(g\|k\|tanh(\|k\|d))`, hard-constrained). The one place multiple "branches" appear (free-wave vs. bound/nonlinear-harmonic energy near 2ω in the HOS regime) is squarely nonlinear-physics-as-contribution territory — excluded item 6. | **No object outside excluded scope; collides with #6** |
| **Optical/photogrammetric remote sensing — Cox & Munk sun-glitter slope statistics / deflectometry** | A specular return's existence implies the local surface normal points back toward the sensor to within the specular acceptance cone — i.e., every specular hit is *also* an (interval-censored) local-slope observation, not just a height sample | This is the **exact physical mechanism** already investigated and quantified as "Door 1" (informative-missingness) in this repo (12 days prior work, `docs/E5_DOOR1_IS_ONE_CHANNEL_2026-07-10.md` etc.): the mask/selection pattern's mutual information with local slope was measured directly (2.5–17.1% depending on detector-threshold regime), explicitly characterized as **"slope info, not phase info,"** and its exploitation (via a selection-aware learned posterior) returned a **null gain** (−0.0000±0.0020). Any non-learned/hard-constraint route to the same specular-alignment fact draws on the *identical* physical channel and is therefore bounded by the *same already-measured* capacity ceiling (≤1 bit/return ray, ceiling ≈ +0.002 to +0.05 nRMSE depending on sea state) — it cannot exceed a ceiling already measured for the underlying geometric fact regardless of which algorithm extracts it. | **Already investigated under different name; info-theoretically subsumed by prior result** |

---

## 2. Literature check (WebSearch, 2026-07-23)

- X-band marine radar current retrieval via dispersion-shell subtraction (3D FFT of *dense*
  consecutive scans, least-squares Doppler fit) — real, mature field. [An Algorithm for Surface Current Retrieval from X-band Marine Radar Images](https://www.researchgate.net/publication/278029235_An_Algorithm_for_Surface_Current_Retrieval_from_X-band_Marine_Radar_Images), [Marine X-Band Radar Currents and Bathymetry (Lund 2020, JGR Oceans)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JC015618), [Estimation of Sea Surface Current from X-Band Marine Radar Images by Cross-Spectrum Analysis](https://doi.org/10.3390/rs11091031), [A Coherent on Receive X-Band Marine Radar for Ocean Observations](https://pmc.ncbi.nlm.nih.gov/articles/PMC8659718/).
- Seismic/ocean-bottom DAS interferometry for ocean-wave and current retrieval — real, but on
  dense fixed fiber arrays over long windows, not sparse single-track. [Surface Gravity Wave Interferometry and Ocean Current Monitoring With Ocean-Bottom DAS (Williams 2022, JGR Oceans)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JC018375), [Reconstruction of Nearshore Surface Gravity Wave Heights From DAS (Meulé 2024)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024EA003589), [Empirical Green's Function Retrieval Using Ambient Noise Source-Receiver Interferometry](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019JB018261).
- Along-track interferometric SAR (ATI-SAR) for ocean current/wave measurement — requires two
  physical antennas. [Measurements of ocean surface waves and currents using L- and C-band ATI-SAR](https://www.researchgate.net/publication/3203348_Measurements_of_ocean_surface_waves_and_currents_using_L-_and_C-band_along-track_interferometric_SAR), [Remote sensing of ocean waves by ATI-SAR (Zhang 2009, JGR Oceans)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009JC005310), [Mapping Surface Currents and Waves with InSAR in Coastal Waters (2013)](https://journals.ametsoc.org/view/journals/phoc/43/3/jpo-d-12-0128.1.xml).
- SAR phase-gradient autofocus — mature technique, requires unmodeled platform phase error.
  [Expediting phase gradient autofocus algorithm for SAR imaging (IEEE)](https://ieeexplore.ieee.org/document/9323622/).
- Cox & Munk sun-glitter wave-slope statistics, incl. modern airborne-LiDAR-glint variants —
  real, mature, and (per §1) already the object of this repo's Door-1 investigation.
  [Cox-Munk Sea Surface Slope Statistics](https://www.oceanopticsbook.info/view/surfaces/cox-munk-sea-surface-slope-statistics), [Revisiting the Cox and Munk wave-slope statistics using IASI observations (2022)](https://arxiv.org/abs/2210.05456), [Airborne Measurements of Surface Wind and Slope Statistics over the Ocean (2019)](https://journals.ametsoc.org/jpo/article/49/11/2799/344384/Airborne-Measurements-of-Surface-Wind-and-Slope).
- Radio-interferometry closure phase / self-calibration — mature, requires ≥3 antennas / a
  redundant receiver array; no ocean-wave shipboard application found (consistent with the
  scope kill above). [An Introduction to Closure Phase and Self-Calibration](https://link.springer.com/chapter/10.1007/978-94-009-2428-4_6), [Closure phase (Wikipedia)](https://en.wikipedia.org/wiki/Closure_phase), [Advanced Gain Calibration Techniques in Radio Interferometry](https://arxiv.org/pdf/1805.05266).
- Guided-wave dispersion compensation / Lamb-mode separation (warped-frequency transform,
  generalized dispersive mode decomposition) — mature NDT field, targets multi-branch
  dispersion, no direct precedent found for single-branch ocean gravity waves outside the
  bound/free-harmonic (nonlinear) case. [Mode separation of Lamb waves based on dispersion compensation](https://pubmed.ncbi.nlm.nih.gov/22501050/), [Dispersion curve analysis method for Lamb wave mode separation (2020)](https://journals.sagepub.com/doi/abs/10.1177/1475921719890590), [Ultrasonic guided-wave characterization with warped frequency transforms](https://pubmed.ncbi.nlm.nih.gov/19942510/).
- Compressed-sensing MRI / non-Cartesian k-space — mature field, no ocean-wave transplant found
  in this search; conceptually overlaps the already-gated sparse-recovery family. [Sparse MRI (Lustig 2007)](https://onlinelibrary.wiley.com/doi/10.1002/mrm.21391), [Variable Density Compressed Sensing In MRI](https://arxiv.org/pdf/1307.6836).
- Graph rigidity theory for sensor-network localization — real, distinct-from-Fisher-information
  combinatorial tool, but built for multi-node pairwise-range networks; no natural object in a
  single-agent field-sampling problem. [Graph invariants for unique localizability in cooperative localization](https://arxiv.org/abs/1502.01699), [Localization in WSNs Using Rigid Graphs: A Review](https://link.springer.com/article/10.1007/s11277-017-4397-7).
- Event-camera / coded-aperture light-field imaging — active, cutting-edge computational-
  imaging field, but requires different physical sensor hardware than the given, fixed
  Falcon LiDAR. [Coded-E2LF: Coded Aperture Light Field Imaging from Events](https://arxiv.org/html/2602.22620), [High-speed Imaging through Turbulence with Event-based Light Fields](https://arxiv.org/html/2603.14023v1).
- Same-domain (not cross-domain, noted for awareness only, not pursued further here): two very
  recent (2025) papers reconstruct wave fields from **stereo cameras on moving/moored vessels**
  with 5–20% surface coverage — i.e., near-identical sparse-moving-platform problem, but same
  discipline (already part of the crowded field noted in the task brief). [Wave-field reconstruction using stereo cameras on a floating platform (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0029801825006717), [Wave-Field Reconstruction from Moored Floating Structure Using Deep Stereo](https://www.researchgate.net/publication/382850574_Wave-Field_Reconstruction_from_Moored_Floating_Structure_Using_Deep_Stereo).

---

## 3. Survivors

**None clear all three checks** (isomorphism real + disanalogy survives code/data + doesn't
collide with the 13 excluded items).

---

## 4. Convergent-invention note (not a contribution claim, but worth stating precisely for the paper's exposition)

The lens *did* find real structural kin — the project's existing pipeline independently
reinvented three named cross-domain techniques rather than missing them:

- `_beamform_coarse` periodogram-over-(kx,ky) with the dispersion relation imposed ≈
  **seismic f-k migration** / **radio-astronomy dirty-map formation**.
- Greedy coarse-grid detection + cyclic joint refinement (`relax_estimator.py`, correctly
  attributed to Li & Stoica 1996) ≈ **CLEAN deconvolution**'s array-signal-processing sibling.
- `joint_refine`'s Newton/L-BFGS full-window fit ≈ **full-waveform inversion (FWI)**.

None of this is a new lever — it just means the "go find the discipline that already solved
this" instruction returns "already inside the repo" for the three techniques that would
otherwise be the headline candidates. This is useful for related-work framing (per the
standing instruction to explain *how* the method works, `feedback_paper_explain_estimation`)
but is exposition, not a new empirical claim.

---

## 5. Two low-value residuals (flagged honestly, not promoted)

Neither of these is recommended as a research direction; both are noted because they were not
killed by a hard structural fact (unlike everything in §1), only by a mechanistic prediction
that was not executed in this review (effort/value tradeoff — both are estimator-internals
micro-tweaks, adjacent to the already-excluded "better estimator" item 1, and predicted null).

1. **NUFFT-style density-compensated gridding** for `bin_observations_to_grid`/
   `fft_peak_extraction` (MRI non-Cartesian reconstruction practice), replacing naive
   averaging-into-bins.
   - Cheapest kill experiment: re-run `initialize_from_frames_v2` with and without density
     compensation on 3 seeds × 2 sea states, compare (a) the coarse periodogram peak location
     error against ground-truth wave components, and (b) final post-`joint_refine` nRMSE.
   - Predicted kill result: peak-location error changes measurably pre-refine, but post-refine
     nRMSE difference ≤0.002 (within existing episode-to-episode noise) because Newton/L-BFGS
     already re-optimizes from any reasonable start — same mechanism that already flattened
     other initializer-only differences in this codebase.

2. **Sub-bin parabolic peak interpolation** (PIV/audio-pitch-detection practice) in
   `fft_peak_extraction`, which currently returns raw FFT-bin peaks with no interpolation.
   - Cheapest kill experiment: same protocol as above, isolate to the peak-refinement step.
   - Predicted kill result: same — swallowed by joint refine.

If someone wants empirical (not just structural) closure on these two, the test above is a
<30-minute run on existing episodes; this review did not spend budget executing it given the
near-certain predicted null and the low ceiling even in the best case.

---

## 6. Blunt bottom line

Eight-plus source fields were examined (seismology, radio astronomy, MRI, SAR/SONAR,
computational imaging, robotics/SLAM, fluid-experiment PIV/tomography, NDT guided-wave
ultrasonics) plus the two explicitly-flagged deeper cousins (Cox-Munk sun-glitter/deflectometry,
graph rigidity theory). **Nothing transplants as a genuine new empirical lever.** The pattern is
consistent across all of them: this project's own constraints — a **single** receiver channel,
a **deterministic, noise-free, single-pass straight trajectory**, **~50 sparse points per
10 Hz frame** with no dense per-snapshot image, a **single-branch** hard-constrained dispersion
relation, and a **given, fixed** sensor — are exactly the preconditions that every one of these
disciplines' distinctive tricks (multi-receiver redundancy, dense per-snapshot images, loop
revisits, multi-branch dispersion, coded/event hardware) depends on and that this problem does
not have. Where the isomorphism is real anyway, the technique turns out to already be inside
the repo under signal-processing names (RELAX ≈ CLEAN, beamform-coarse ≈ f-k migration,
joint_refine ≈ FWI) rather than missing. Where a genuinely interesting mechanism was found
(specular-return-implies-local-slope, the Cox-Munk/deflectometry analogy), it turned out to be
the *same physical channel* this repo already measured and closed ten days earlier under the
"informative missingness" investigation, with a quantified, near-zero ceiling that bounds any
algorithm drawing on that channel, learned or not. This is a clean null, not a search failure —
the project's own scoping choices (single given sensor, exogenous trajectory, sparse regime)
are precisely what forecloses the cross-domain playbook.
