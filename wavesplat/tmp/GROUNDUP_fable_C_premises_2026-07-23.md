# Reviewer C — Attack the Premises (ground-up re-examination)

**Date:** 2026-07-23. **Scope:** `wavesplat/` codebase, `paper/wavesplat.tex` (the
actual OE submission), `data/*.h5` (the shipped 195-episode set), and
`data/real_bergamasco/` (stereo stand-in). **Method:** every claim below is
either (a) a direct measurement against the current code/data — script and
numbers given — or (b) explicitly labelled UNTESTED CONJECTURE. Severity:
**FATAL** = invalidates a specific published/claimed number, **MATERIAL** =
forces re-scoping or a caveat, **COSMETIC** = worth a sentence, not a rewrite.
Where a premise turns out sound, that is stated as plainly as a break.

I did **not** touch: proposing a better estimator, the CRB/operability-envelope
spine, atomic-norm, estimator-family swaps, learned priors, nonlinear-physics
modelling, forecast/streaming/latency, calibrated UQ, benchmark/protocol
pivots, control-value demos, active sensing, trajectory design. Those are
out of lens by the task brief.

---

## Executive severity ranking

| # | Finding | Severity | Threatens |
|---|---|---|---|
| 1 | Shipped raycast mesh is sub-Nyquist/degraded for **100% of in-scope (BF2–BF5) episodes** — measured directly; direct A/B regeneration shows the paper's own quoted worst-case number (0.315) **drops 28.5% to 0.225 on mesh refinement alone** | **MATERIAL, verging on FATAL for absolute numbers** (ranking survives) | Table `tab:fivemetric` (0.223/0.319/0.261), Table `tab:raydrop`, Table `tab:bybf`, abstract "30.1%", line 1555 worst-case claim |
| 2 | "slope" raydrop model's specular gate **never fires** at Falcon's geometry (measured: 0/63,095) — it silently degenerates to unweighted uniform subsampling, but the paper's mechanistic narrative for it is the opposite of what the code's own later self-correction says | **MATERIAL** | §"Raydrop model sensitivity" (Table `tab:raydrop`, Fig. 5), highlights line "robust to specular raydrop" |
| 3 | The "uniform" raydrop model — the *primary* model, used in every headline episode — reuses the exact formula the codebase's own later audit calls a radiometric category error (specular Fresnel R used as a diffuse albedo at ~78–88° grazing) | **MATERIAL** (pattern, not budget) | Every uniform-raydrop episode's point *pattern*; undisclosed in paper |
| 4 | Evaluation arena: 100% of episodes share one JONSWAP shape (γ=3.3), one directional spread (30°, Gaussian), one mean direction (0°, unused 180° set); "100 episodes spanning BF2–5" is really a 1-D (Hs,Tp) × complexity sweep | **MATERIAL** | Generalization claim behind every table; not fatal to internal validity |
| 5 | Stationary-sensor assumption (`ship_x=ship_y=0` hardcoded, `ship_speed_kts` dead config, no per-frame pose in HDF5) at every one of 195 episodes, vs. the project's own agreed 11.3 m/s design speed | **MATERIAL, but disclosed** in paper §Limitations | Scope mismatch between title ("shipboard") and what was actually simulated |
| 6 | Stereo-camera "real data" substitution is geometrically/radiometrically incommensurate with the simulated LiDAR (dense vs sparse, small vs large aperture, nadir vs grazing, fixed platform, disparity vs ToF noise) | **MATERIAL for thesis "real anchor" claims** | Does **not** touch `paper/wavesplat.tex` — verified zero citations of Bergamasco/stereo in the OE draft |
| 7 | Deployment/integration story: real-sensor ingestion code (`real_data.py`) has never ingested a single real file; planned tank validation is unexecuted and, even if run, cannot validate the paper's 2-D directional claims | **MATERIAL** (premise: "is this deployable"), but paper text itself is appropriately hedged | Broader PhD narrative, not the OE draft's explicit claims |
| 8 | Metrics: nRMSE-only / zero-predictor blindness | **CLEARED — already fixed** | — |
| 9 | Scoring domain outside sensor FOV | **COSMETIC — measured, small (12.5%)** | — |

---

## 1. Simulator forward model — raycast mesh resolution

### 1a. Premise: "the 100-episode synthetic dataset resolves the sea states it claims to simulate"

**Check run:** loaded `wavesim.mesh_adequacy.check()` (the repo's own adequacy
criterion — ADEQUATE ≤5% unresolved variance & peak above Nyquist, DEGRADED
≤25%, else INADEQUATE) directly against the actual `wave_components` stored in
every shipped `data/episode_*_{uniform,slope}.h5` file (excluding the
`dir180` variants and the out-of-scope BF6/BF7 supplementary files), using
the mesh the config that generated them (`configs/default.yaml`: `mesh.nx=40,
mesh.ny=30` → dx=6.15 m, dy=6.90 m — unchanged since the files were written).

```
N episodes checked (BF2–BF5, uniform+slope): 110
Counter({'DEGRADED': 53, 'INADEQUATE': 50, 'ADEQUATE': 7})
```

The 7 "ADEQUATE" hits are all **BF6/BF7** episodes — explicitly **outside**
the paper's declared operating range ("100 episodes spanning Beaufort 2–5").
Restricting to the in-scope BF2–BF5 set actually used for every table:

| Sea state | Verdict | pts/λ (peak) | pts/λ (min sig. comp.) | % variance below Nyquist band |
|---|---|---|---|---|
| BF2 1comp | **INADEQUATE** | 1.2 | 1.2 | 100% |
| BF3 1comp | **INADEQUATE** | 2.2 | 2.2 | 100% |
| BF3 2comp | **INADEQUATE** | 5.4 | 1.2 | 100% |
| BF3 5comp | **INADEQUATE** | 4.4 | 0.9 | 77% |
| BF4 1comp | **INADEQUATE** | 3.6 | 3.6 | 100% |
| BF4 5comp | DEGRADED | 7.1 | 1.5 | 21% |
| BF4 15comp | DEGRADED | 9.5 | 2.6 | 22% |
| BF4 30comp | DEGRADED | 10.2 | 3.2 | 21% |
| BF5 15comp | DEGRADED | 15.5 | 4.2 | 9% |
| BF5 30comp | DEGRADED | 16.8 | 5.2 | 7% |

**0 of 95 in-scope, shipped episodes pass the codebase's own mesh-adequacy
bar.** This independently reproduces (on the *actual paper-cited files*,
not a separate research cache) the finding already made in-repo by
`docs/E2_MESH_AUDIT_prereg_2026-07-13.md` — but that audit was run against a
*different* 35-episode ablation set for a *different* (learning-direction)
comparison, was frozen **2026-07-13**, and the fix it prescribes (regenerate
at 180×150 mesh, and have every episode carry its own `mesh_dx`/
`mesh_adequate` attrs) was never applied to `configs/default.yaml` or to the
`data/*.h5` files the paper actually cites (`config` group in
`episode_BF4_5comp_seed0_uniform.h5` etc. has only
`['Hs','Tp','depth','gamma','n_components','sea_state']` — no mesh
metadata at all — confirmed by direct h5py inspection). `paper/wavesplat.tex`
(finalized 2026-05-03, i.e. **before** the mesh audit existed) contains no
mention of "Nyquist" or raycast-mesh resolution as a limitation anywhere.

**What this means physically:** at BF2–BF4/1–5-comp, the ray-mesh
triangulation cannot geometrically represent the wavelength being
simulated — the LiDAR "sees" facet/render error, not ocean signal, for a
large fraction of the field's variance. A directly comparable intervention
(E-3, same repo, different comparison) measured **operator error 0.048 m
(coarse) → 0.006 m (fine, 8× reduction)** — i.e. at the shipped resolution,
facet-rendering error is **~2.4× the sensor's own nominal noise floor
(2 cm)**. In that same intervention, refining the mesh moved the effect of
interest by only ~1/3 (sign and magnitude class preserved) — suggesting
*relative* rankings across estimators may be fairly robust to this artifact,
but *absolute* performance numbers are not cleanly separable from it.

**Direct test on the paper's own dataset (this review):** I regenerated the
paper's stated worst case for WaveSplat — `BF2_1comp_seed0` (nRMSE=0.315 is
explicitly quoted in wavesplat.tex line 1555 as the worst case) — at both the
shipped coarse mesh (40×30) and the fine mesh (180×150, same seed → bit-identical
wave components, GPU raycast validated hit-for-hit against CPU per
`wavesim/lidar_gpu.py`), then ran WaveSplat-P1 and ETKF-medium (same code
path as `scripts/run_p1_comparison.py`) on both, full 301-frame episodes.
Script: `wavesplat/tmp/premise_mesh_headline_test.py`. Result in §1b.

**Severity:** MATERIAL for the paper's absolute headline numbers
(nRMSE 0.223/0.319/0.261; "30.1% improvement"; the raydrop-sensitivity Δ%
column), because the underlying observations were generated on a mesh the
repo's own instrumented criterion calls DEGRADED-or-worse for every in-scope
episode, and the paper discloses none of this. Whether it is FATAL to the
*relative ranking* (WS beats ETKF/GPR) is the specific thing tested in §1b.

### 1b. Direct A/B result: does refining the mesh change the WS-vs-ETKF margin?

Regenerated `BF2_1comp_seed0` (same wave components bit-for-bit, verified by
construction: `jonswap_components` seeding is mesh-independent) at the
shipped coarse mesh (40×30) and at a fine mesh (180×150), ran WaveSplat-P1
(adaptive N, `scripts/run_p1_comparison.py`'s exact config) and ETKF-medium
(`n_ensemble=50`, same code path) on both, full 301-frame episodes.
Result (`wavesplat/tmp/premise_mesh_headline_test_results.json`):

| | coarse (shipped, 40×30) | fine (180×150) | Δ |
|---|---|---|---|
| WaveSplat nRMSE | **0.3150** | 0.2251 | **−28.5%** |
| ETKF nRMSE | 0.4831 | 0.3590 | −25.7% |
| Gap (ETKF−WS), absolute | 0.1681 | 0.1340 | −20% |
| Gap (ETKF−WS), relative to ETKF | 34.8% | 37.3% | +2.5 pt |

Two things are now measured, not conjectured:

1. **The coarse-mesh WaveSplat number, 0.3150, reproduces
   `paper/wavesplat.tex` line 1555's quoted worst case (nRMSE=0.315) to 3
   decimal places** — this is not a different episode or a different
   pipeline, it is the exact stored artifact the paper cites. Refining
   *only* the raycast mesh — no change to the estimator, the wave physics,
   the seed, or the sensor spec — drops that number by **28.5%**, to 0.225.
   The paper's own stated "worst case" is, on this evidence, **substantially
   a mesh-rendering artifact**, not a property of the BF2 sea state or of
   WaveSplat's estimator.
2. **The relative ranking is robust.** WaveSplat still beats ETKF by a wider
   margin (37.3%, up from 34.8%) at the fine mesh. This corroborates the E-3
   precedent (a different ablation, same repo) that mesh refinement mostly
   redistributes *absolute* error and does not flip *comparative* claims.

**Severity, refined:** the mesh-Nyquist violation is **MATERIAL, verging on
FATAL, for every absolute nRMSE number quoted in the paper** (abstract's
"0.223 vs 0.319," the by-BF-state Table `tab:bybf`, the per-episode
worst-case claim, the raydrop-sensitivity Δ% column) — these numbers are
demonstrably inflated by an unrelated, undisclosed simulator artifact, by an
amount (here, 28.5% on the one episode tested) large enough to change how
the results should be read even though it does not reverse who wins. It is
**not FATAL to the paper's comparative conclusion** (WaveSplat < ETKF <
GPR-1 ordering) on this one-episode test. Generalizing "28.5%" to the other
94 in-scope episodes was not attempted (would require ~95 paired
regenerations at several minutes each) and should not be — this is one
decisive data point, not a claimed population effect.

---

## 2. Simulator forward model — the "slope" raydrop model is not what the paper says it is

### Premise: "WaveSplat is robust to both diffuse and specular LiDAR raydrop physics" (abstract, highlights, §Raydrop model sensitivity, conclusion)

**Check run:** generated one representative frame (BF4, 5 components, seed 0,
t=5 s, the actual pipeline geometry: height 5 m, FOV 120°×[-25°,0°], 500×152
beams) and evaluated `wavesim.raydrop_slope.specular_alignment()` — the
function that decides whether a hit qualifies as a "specular" return —
against every geometric hit.

```
n_rays total: 76,000   n_hits: 63,095
incidence angle at hits [deg]: min 62.5, mean 77.6, max 90.0
surface slope (tan) at hits: mean 0.036, std 0.024, max 0.141
specular candidates (5° alignment threshold): 0 / 63,095  =  0.0%
```

**Zero** returns anywhere in the frame satisfy the specular condition. This
independently confirms — on the same file, with the same numbers — a
self-correction already recorded in `scripts/d3_s2b_true_f.py` (dated
2026-07-10): the required facet tilt for a specular return at this grazing
geometry is enormous relative to actual ocean wave slope (`facet slope
r/h≈3–40` needed vs. measured wave-slope σ≈0.02–0.04), so `raydrop_slope.py`
**always** takes its constant-probability fallback
(`P_uniform = target_n_pts / len(hit_idx)`, no incidence-angle, no range,
no Fresnel weighting whatsoever).

The two "raydrop models" the paper compares are therefore **not** two
different sensor-physics assumptions:

- `*_uniform.h5` → `raydrop_uniform.py`: `P ∝ R_fresnel(θ)·cos(θ)/d²` — a
  genuinely radiometric, incidence/range-weighted model.
- `*_slope.h5` → `raydrop_slope.py`, *specular branch dead* → falls straight
  to flat, unweighted random subsampling of whatever the raycast already hit
  — i.e. an **occlusion-only control**, no radiometry at all.

`paper/wavesplat.tex` (§"Raydrop model sensitivity", Table `tab:raydrop`,
Fig. 5, lines 1337–1391) describes (b) as "specular backscatter" and
explicitly explains WaveSplat's/ETKF's improvement under it as
**"specular backscatter preferentially preserves wave-crest points"**
(line ~1384–1386, repeated line 1570). This mechanism cannot be operating —
there are no specular selections in the data. The in-repo self-correction
(`d3_s2b_true_f.py`) independently reaches the same conclusion and explicitly
relabels the two file suffixes: *"this also inverts what our episode
filenames mean: `*_slope.h5` → the OCCLUSION-ONLY control, `*_uniform.h5` →
occlusion + radiometric selection."* That correction is dated **after**
`paper/wavesplat.tex`'s last edit (2026-05-03), so it never propagated back
into the manuscript.

**What actually likely explains the "slope helps" effect** (not tested here,
flagged as the natural next hypothesis): removing the Fresnel/range
weighting changes *which* geometrically-hit points are kept — e.g. it may
retain more far-range/grazing points that the radiometric model would have
preferentially dropped. That is a real, testable, but different effect than
"specular backscatter reveals wave crests," and it is currently mis-cited in
the manuscript as a physical mechanism.

**Severity: MATERIAL.** It doesn't invalidate that WaveSplat wins under both
episode families (that ordinal claim is untouched), but the *physical
interpretation* the paper gives for the raydrop-sensitivity table and the
"robust to specular models" claim in the abstract/highlights/conclusion is
not what the code does, and an in-repo audit already knew this before the
paper text was final in a form the paper writer evidently didn't reconcile.

---

## 3. Simulator forward model — the primary "uniform" raydrop formula is a documented category error

**Premise under test:** "the uniform (Fresnel-weighted) dropout model is a
physically motivated diffuse-scattering model of LiDAR return probability."

`raydrop_uniform.py`'s `return_probability()` — the formula behind every
`*_uniform.h5` episode, i.e. the *primary* raydrop model cited throughout
CLAUDE.md and the paper — computes
`P ∝ R_fresnel(θ)·cos(θ)/d²`, using the **specular** Fresnel power
reflectance coefficient as a **diffuse** albedo. `raydrop_threshold.py`
(a newer, unpublished sibling model, not used by the paper) documents this in
its own docstring: *"received_power uses the specular Fresnel coefficient as
if it were a diffuse albedo, which at the grazing incidence of this geometry
(theta → 88 deg) sends the reflected energy away from a monostatic receiver
… The sibling model in raydrop_slope.py, applied to the same geometry,
predicts essentially zero returns. The two disagree by orders of magnitude
and the absolute scale has never been checked against a real sensor over
water."* A companion audit (`docs/E2_MESH_AUDIT_prereg_2026-07-13.md` §E-6b)
independently derives the same conclusion in more detail and explicitly
retracts the phrase "measured return count," calling the model's implied 60k
returns/frame "not a measurement" and the category error "physically
ungrounded" (원문: "물리적으로 근거가 없다").

**What this does and doesn't threaten.** The *total* point budget per frame
is insulated from the error: `apply_dropout()` rescales `η` every frame so
`E[N_returns] = target_n_pts` (=50), so the paper's "~50 points/frame"
sparsity claim is unaffected regardless of the radiometric model's validity.
What is affected is the **spatial pattern** — which points (near/far,
grazing/less-grazing) are more likely to survive — since that pattern comes
directly from the same `R_fresnel(θ)·cos(θ)/d²` shape flagged as unsound.
I did not re-derive a corrected radiometric model or re-run the estimator
under one (out of scope: this would be "propose a better forward model"); I
verified only that (a) the formula in question is exactly the one the
codebase's own later work calls invalid, (b) it is the model behind the
paper's *primary* episode family, and (c) none of this is disclosed in
`paper/wavesplat.tex`.

**Severity: MATERIAL**, narrower than §1/§2 because the count-level effect is
calibrated away; the open risk is entirely in the *pattern* of which points
inform the estimator, which is exactly the channel WaveSplat's
geometry-aware confidence weighting is built to exploit (`c_j =
cos(θ_j)/(d_j/d_ref)`) — i.e. WaveSplat's confidence model and the raydrop
model that generates its training/eval observations share un-audited,
correlated assumptions about the same incidence-angle/range dependence.
That correlation is worth flagging even though I did not quantify its size.

---

## 4. Evaluation arena — diversity is narrower than "spanning Beaufort 2–5" implies

**Check run:** read `wavesim/wave_generator.py::jonswap_components()` defaults
and every call site that generates the shipped dataset
(`wavesim/pipeline.py::run_episode`, `configs/default.yaml`,
`configs/server.yaml`).

- `gamma` (JONSWAP peak enhancement): hardcoded to **3.3** at every call —
  never varied, never present in any config file's `dataset:` block.
- `direction_spread_deg`: hardcoded default **30°**, Gaussian — never varied.
- `direction_deg`: **0°** for every episode `paper/wavesplat.tex` actually
  uses (I grepped the manuscript for "dir180" / "direction" — zero
  references to the 180°-direction variant). 30 `*_dir180*.h5` files exist
  in `data/` but are not part of the paper's dataset.
- Every episode is **unimodal** — no swell+wind-sea two-peak mixtures, no
  crossing seas, ever, in the generator used for the paper.

So "100 episodes spanning Beaufort 2–5 sea states (1–30 wave components)"
is accurate as far as it goes, but the actual varying dimensions are: (Hs,Tp)
tied 1:1 to a 4-way BF class, N_components (a discretization/complexity
knob, 1/2/5/15/30), 5 seeds, and 2 raydrop models (see §2–3 for what those
really vary). Spectral shape, directional spread, and mean direction are
**constants across the entire dataset**. This is the same structural
narrowness a prior in-repo audit found for a *different* arena
(`project_novelty_rereview` / `project_e29_e39` family, memory:
"평가 아레나가 Tp7·γ3.3·spread30·단일모드 하드코딩") — this review confirms
the same constants are baked into the actual paper's data-generation path,
not just a downstream research arena.

**Severity: MATERIAL** — it bears on how far "spanning Beaufort 2–5" can be
read as "spanning realistic sea-state diversity" (it can't: it spans wave
*height/period* and *component count*, not spectral shape or directionality).
It does not invalidate any specific reported number; it bounds what those
numbers can be claimed to generalize to. Not tested: whether relaxing γ or
spread would change the WS-vs-baseline ranking (flagged as the natural
follow-up, out of scope here since it borders "evaluation-protocol pivot").

---

## 5. Deployment story — stationary sensor vs. the project's own design speed

**Check run:** traced `pipeline.py::run_episode` — rays are generated **once**,
outside the per-frame time loop, with `ship_x=0.0, ship_y=0.0` hardcoded
(comment: `# Pre-generate rays (same for all frames since ship is
stationary)`). `WaveSimConfig.dataset.scenarios[i].ship_speed_kts` is defined
in `config.py` but never read anywhere in `wavesim/` or `wavesplat/`
(`grep -rn ship_speed_kts` returns only its own definition). No per-frame
ship position/attitude is stored in the HDF5 schema
(`wavesim/data_writer.py`: only `timestamp`, `lidar_points`, `gt_wave_field`,
etc. per frame). This matches a prior in-repo, 3-way self-check
(`scripts/generate_labmeeting_2026_05_13.py`, code+dataset+empirical
verification: one episode's LiDAR x-centroid varies by only σ≈1.68 m over
30 s, vs. the ~339 m a vessel at the project's own agreed 11.3 m/s design
speed (memory: `reference_agreed_operating_point.md`) would have travelled).

**What's already handled correctly:** `paper/wavesplat.tex` §"Limitations"
(lines 1997–2004) discloses this explicitly and appropriately: *"the current
simulation assumes a stationary sensor platform. In practice, heave, pitch,
and roll of the vessel change the LiDAR beam pattern frame-to-frame; an
IMU/INS-based coordinate transform is a necessary preprocessing step for
sea-going deployment…"* — this is honest, specific, and correctly scoped.
I re-verify it and find no overclaim in the manuscript text itself.

**Severity: MATERIAL-but-disclosed.** The residual concern is a framing one,
not a hidden defect: the title is "…from Sparse **Shipboard** LiDAR" and the
deployment section (§"Practical deployment considerations") discusses vessel
control integration, both of which read as underway operation, while
literally 0 of 195 episodes simulate a moving, pitching, rolling ship. That
tension is disclosed but easy to miss on a first read of the abstract/title
alone. I clear this as *not* a silent premise violation, but flag the
scope mismatch between what the title/application section imply and what
was ever run.

---

## 6. The stereo-camera "real data" substitution

**Premise under test:** IFREMER/Bergamasco stereo-reconstructed sea-surface
records are a defensible stand-in for the ship-mounted Falcon LiDAR this
project targets.

**Check run:** does `paper/wavesplat.tex` rely on this substitution at all?
`grep -in "bergamasco|stereo|guimar|ifremer|field data|real-world"` on the
manuscript returns **zero** hits beyond generic phrases like "real-data
deployment" and "real data" as a future-work noun, and one caption about
approximating surface normals "for real data application." **The OE
submission makes no empirical claim that rests on the stereo dataset.** This
premise, however material it is, does not touch a single number in the
paper currently being submitted — worth stating plainly since it clears the
manuscript of that specific risk.

It **does** underwrite the project's broader "real anchor" claims recorded
in project memory (E-59, "실측 anchor 완료 = QUALIFIED PARTIAL") and the
thesis-level sim-to-real narrative. Fresh numbers on the mismatch, read
directly from the `.nc` files' own attributes (h5py, `Grid discretization
[m]`, `Cameras distance [m]`, `Z.shape`):

| Site | Frame grid (px) | Grid spacing | Effective patch | Rate | Camera baseline |
|---|---|---|---|---|---|
| BS_2011 | up to 291×381 | 0.05 m | ~14.5 m × 19.0 m | 12–15 Hz | 2.03 m |
| BS_2013 | 171–191² | 0.10 m | ~17–19 m × same | 10–12 Hz | 1.87 m |
| AA_2014/2015 | ~350–431² | 0.20 m | ~70–86 m × same | 12 Hz | 2.55 m |
| LJ_2018 Grid1 | 401×401 | 0.10 m | 40 m × 40 m | 10 Hz | 5.45 m |
| LJ_2018 Grid2 | 601×501 | 0.50 m | 300 m × 250 m | 10 Hz | 5.45 m |
| YS_2017 | 241×221 | 1.00 m | 240 m × 220 m | 10 Hz | 5.04 m |
| **Simulated Falcon LiDAR (this project)** | ~50 valid pts (sparse, raydrop-selected out of 76,000 rays) | irregular, range-dependent | 240 m × 200 m fan, forward-looking, up to 250 m range | 10 Hz | (ToF, not stereo) |

Reading across this table: most sites (BS, AA) are **two to three orders of
magnitude denser** and have a **6–17× smaller aperture** than the simulated
LiDAR footprint; only LJ_2018-Grid2 and YS_2017 approach the simulated
footprint's *size*, but at coarse (0.5–1.0 m) yet still *fully dense* (no
raydrop, no dropout, no incidence-angle-dependent noise) grid sampling from
what is a fixed coastal platform (Acqua Alta oceanographic tower, La Jument
lighthouse, etc.), viewing at near-nadir — not the simulated sensor's
78–90° oblique/grazing forward scan from a 5 m ship-bow height. This
corroborates, with fresh numbers, the project's own prior finding (memory:
`project_realdata_causes_2026-07-02`) that the aperture-to-wavelength ratio
achievable at several sites (as low as 0.11λ at La Jument) is the leading
driver of "real" underperformance relative to synthetic headline numbers —
i.e., the substitution's mismatch is not a hidden risk, it's already the
project's own leading internal explanation for a documented gap. I confirm
it is real and quantify the geometric mismatch directly; I did not find a
new failure mode beyond what the project had already identified.

**Severity: MATERIAL for thesis-level "real anchor" claims; does not touch
the OE paper submission** (verified: zero citations).

---

## 7. Deployment story — is there an executed integration path?

**Check run:** `wavesplat/wavesplat/real_data.py` implements a full
Falcon-K1 PCD/CSV → WaveSplat `FrameData` pipeline, with a
`LiDARMountConfig` dataclass explicitly commented "for the tank experiment."
Searched the entire repository for any `.pcd` or matching raw-sensor file:

```
find . -iname "*.pcd"  →  (no results)
```

Zero real sensor files exist anywhere in the project. `git log` shows
`real_data.py` was authored 2026-04-16 ("SNU 수조실험 사전 소프트웨어 준비").
`docs/snu_towing_tank_experiment_plan.md` (2026-06-27, i.e. *after* the
ingestion code) is explicit that the experiment is **not yet run**, and that
even once run, a towing tank produces **uni-directional** waves only —
*"The paper's hardest, most novel regime is multi-directional complex seas,
which a towing tank cannot produce… the multi-directional claims stay in
simulation."* So the one concrete step toward real-sensor validation that
exists in the repo (a) has never executed, and (b) is pre-scoped to be
unable to validate the paper's central 2-D directional claims even if it
does run.

`paper/wavesplat.tex`'s own "Practical deployment considerations" (line
1602) is appropriately hedged — it argues *suitability* from compute
footprint and interpretability, cites a companion hydrofoil application
paper, and does not claim a fielded system or executed hardware test. The
Avikus/HD Hyundai affiliation and the acknowledgment ("discussions on
shipboard LiDAR integration") indicate a plausible industry contact point,
but "discussions" is explicitly weaker than any executed integration.

**Severity: MATERIAL as a premise** (the deployment story is presently
100% simulation, 0% executed real-sensor validation, at any scale, and the
one planned physical test cannot validate the headline directional claims
even in principle) **but the manuscript itself does not overclaim** — this
clears the paper text of a specific violation while leaving the underlying
premise ("this is close to deployable") resting entirely on aspiration.

---

## 8. Metrics — nRMSE / zero-predictor blindness: CLEARED, already fixed

**Premise under test (from the task brief):** "field nRMSE (÷Hs) is the
scorer, and cannot distinguish a real estimate from a zero predictor."

**Check run:** read `wavesplat/evaluation/metrics.py` and
`wavesplat/tests/test_metrics_defense.py`, and the actual headline table in
`paper/wavesplat.tex` (`tab:fivemetric`).

This concern is **already found, already fixed, and already load-bearing in
the paper** — not a new premise gap. The team pre-registered and reports a
5-metric panel (nRMSE, Variance Ratio, Pearson ρ, Skill Score vs. a zero
predictor, RMSE at points ≥10 m from any observation), explicitly to catch
exactly this failure mode, and it *does* catch it: GPR's VR=0.065 (≈6.5% of
true variance reconstructed) and ρ=0.109/SS=−0.70 for ETKF are reported and
discussed as "near-zero predictor" and "shape-incoherent" pathologies
respectively — with WaveSplat's own Skill Score margin reported honestly as
thin (+0.034, "the 95% bootstrap CI includes zero on the aggregate
dataset"). `test_metrics_defense.py` unit-tests that Variance Ratio and
Skill Score correctly separate perfect / over-smoothed / zero / inverted
predictions.

I additionally tested the specific sub-claim "is the arena scored where the
estimator is actually informed": computed the fraction of the 80×60 GT grid
that is geometrically **outside** the sensor's azimuth/range footprint on a
calm sea (independent of any raydrop model) —

```
GT grid cells outside FOV azimuth cone or beyond range_max: 12.5%
```

— a real, non-zero effect (concentrated in near-range/high-bearing grid
corners the 120°-FOV sensor can never illuminate even in principle), but
small enough that it is not plausibly the dominant driver of the reported
margins; the existing `RMSE_unobserved` metric already partially controls
for this by constraining to points that also passed **inside** the FOV in
that frame.

**Severity: COSMETIC / CLEARED.** I looked for a load-bearing metrics gap and
did not find one that the project hasn't already found and mitigated itself.
The 12.5%-out-of-FOV figure is worth a footnote, not a rewrite.

---

## Summary of what was tested vs. conjectured

| Claim | Tested | Result |
|---|---|---|
| Shipped mesh is sub-Nyquist for in-scope episodes | ✅ direct `mesh_adequacy.check()` on paper-cited files | 0/95 ADEQUATE |
| "Slope" raydrop never triggers specular branch | ✅ direct `specular_alignment()` run on representative frame | 0/63,095 |
| Uniform raydrop reuses the flagged category-error formula | ✅ code inspection, cross-referenced to in-repo docstrings | confirmed, same formula |
| Arena spectral shape/spread/direction is constant | ✅ grep of generator defaults + all config/call sites | confirmed |
| Ship motion is absent from every episode | ✅ code + schema inspection (re-derives a prior in-repo check) | confirmed; **disclosed in paper** |
| Stereo substitution mismatches LiDAR geometry | ✅ direct `.nc` attribute readout, 8 files across 6 sites | confirmed; **not used by paper.tex** |
| Real-sensor ingestion is unexecuted | ✅ filesystem search + git log + plan doc read | confirmed |
| nRMSE hides zero predictors | ✅ read metrics code + paper table | already fixed by the team |
| WS-vs-ETKF margin is mesh-artifact-robust | ✅ direct regeneration + re-estimation A/B on the paper's own stated worst case | **absolute numbers inflated 28.5%(WS)/25.7%(ETKF); relative ranking robust (gap 34.8%→37.3%)** |
| Uniform-raydrop's pattern (not budget) biases estimator ranking | ❌ UNTESTED CONJECTURE — flagged, not run (would require a corrected forward model, out of lens) | — |
| γ/spread relaxation changes ranking | ❌ UNTESTED CONJECTURE — flagged as follow-up, borders "evaluation-protocol pivot" (out of lens) | — |
