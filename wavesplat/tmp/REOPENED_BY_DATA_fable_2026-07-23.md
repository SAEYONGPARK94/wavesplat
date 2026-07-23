# What the fine-mesh regen re-opens — fable review, 2026-07-23

Scope: (1) red-team `tmp/REVALIDATION_CENSUS_2026-07-23.md`'s scoping claim
("11/413 scripts touch the raycast mesh"); (2) search the record for methods
abandoned for a reason that turns out to be a *data* limitation the fine mesh
now removes. Rules followed: MEASURED vs ARGUED tagged throughout; no new
headline proposed (novelty is closed, `tmp/NOVELTY_REREVIEW_*`); estimator
CV up to 53% (`tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md`) priced into every
retry-cost estimate; strict discriminator — literature pre-emption, structural
facts, and mean-collapse failures excluded even where the mesh also happened
to be bad.

**Headline: something genuinely re-opens, and it is already MEASURED, not
just plausible.** The current paper draft's own explanation for its single
worst reported result ("Failure case: BF2 1-component", nRMSE = 0.315,
`paper/wavesplat.tex` L1553–1563) is a physical-SNR story that a direct,
today, production-estimator, multi-seed, properly-paired re-run falsifies:
nRMSE 0.31 → 0.16, field ρ 0.02 → 0.66, **8/8 fine-mesh runs beat 8/8
old-mesh runs** on two independent data seeds (§Part 2, item 1). This is not
a new experiment design — it is the same config the paper table already used,
pointed at the paired replacement directory. Total compute: ~3 minutes.

---

## Part 1 — red-teaming the census's scoping claim

**Verdict: the literal claim is correct and I could not break it, but the
scope it was drawn from has one blind spot that turns out to be exactly
where the biggest reopened result lives.**

### What checks out (MEASURED, independently reproduced)

- `grep -rln "run_episode\|LidarSimulator\|raycast" scripts/*.py` → the same
  11 files the census names, no more, no less (reran it myself, no
  overlooked hits, no case/alias tricks — checked separately for
  `build_mesh`/`generate_rays`/`LidarConfig(`/`wavesim.pipeline` imports,
  all subsets of the same 11).
- Spot-checked the named-safe arcs by reading the actual code, not the
  census's prose, for one representative script each:
  - `d1_crb.py` (CRB spine): `collect_obs()` extracts only `pts[:,0]`/`pts[:,1]`
    (x, y) from `lidar_points` and discards column 2 (z) — geometry-only,
    exactly as claimed.
  - `d3_h7_hos_arena.py` / `d3_f1_hos_door1.py` / `d3_door1b_field_bias.py`
    (E-12–16, Door-1/H1/H2): all define their own `MESH_DX = 2.0` domain
    (`d3_door1b_field_bias.py:158`, confirmed by line number) — a
    self-contained, already-adequate HOS-scale mesh, not `data/`.
  - `tmp/e28_gennet_A.py` / `_B.py` (E-28): import `jonswap_components` +
    `sample_obs`, no `h5py`, no raycast — analytic sampler, confirmed.
  - `tmp/traj_step1b_relax.py` (TRAJ FAIL): imports `make_sea` from
    `d74_closed_loop_trajectory`, an analytic Airy superposition — confirmed.
  - Better-net/gap-ladder/wavenumber-ID-CRB "inverse crime" claim
    (`tmp/BETTERNET_fable_2026-07-22.md`): independently corroborated —
    `_fixedT`/`_slope`/`_uniform` share byte-identical `wave_components` for
    a given seed, and no episode's `z` passes through anything but
    `predict(theta)+Gaussian`.

### The one blind spot

The census (and the `REGEN_SPEC` it rests on) scopes "the shipped data" as
the **100-file paired set** (`{BF2:1,BF3:1,BF3:2,BF3:5,BF4:1,BF4:5,BF4:15,
BF4:30,BF5:15,BF5:30} × seed{0-4} × {uniform,slope}`) plus 2 stray files =
102 — matching the "0/102 adequate" figure. `data/` actually holds **195**
`.h5` files: those 100/102, plus **85 `_fixedT.h5`** (dense, 8000 pts/frame)
**+ 30 `_fixedT_dir180.h5`** files, a second, older evaluation harness
consumed by 9 named scripts (`d3_a2_dir180_eval.py`, `d3_a7_finebank.py`
/`d3_a7v2_refine.py`, `d3_appendix_figs_v2.py`, `d3_b1v3_peaknet.py`
/`d3_b1deep_peaknet.py`, `d3_critic_labels.py`, `d3_e6_sampling_test.py`,
`d3_h6_v2_e2e.py`, `d3_m8_residual.py`, `d3_wave_gallery.py`). None of these
115 files have a fine-mesh companion in `data_finemesh_2026-07-23/`.

This exclusion is **not a silent gap** — `REGEN_SPEC_fable_2026-07-23.md` §0
explicitly and correctly scopes `fixedT`/`dir180`/BF6-7 out, with stated
reasons (a different raydrop research track, not the paper's raydrop
comparison; already outside the paper's declared primary scope; flagged as
"should get its own, later regen pass" for `fixedT`). I verified the two
`dir180`/BF6-7 claims that reasoning leans on: the paper's directional-deficit
section (`wavesplat.tex` L1516–1524, "$-0.25$ vs $+0.07$") is computed on the
**standard 100-episode set**, not the 30-file `dir180` probe, and its stated
mechanism is sensor-swath geometry (structural, correctly excludable); BF6/7
is explicitly hedged by the paper itself as a linear-simulator-only
confirmatory check, not a primary claim. So the REGEN_SPEC's scoping calls
are individually defensible and I did not find the census's TOP-3 list to be
wrong.

**Where the blind spot bites**: the census's unit of analysis is
*headline claims / arcs* (checked against tables and result JSONs). It never
audited **prose that makes an episode-specific mechanistic claim** about a
file that *is* in the 100-file in-scope set. `paper/wavesplat.tex`'s
"Failure case: BF2 1-component" is exactly that — sourced from
`results/12ep_full_analysis.json` and Table (L1056), fully in-scope,
already re-generated at fine mesh today — but its accompanying *explanation*
was never checked against the new data because no table diff would surface a
change in prose. Compounding this: the same underlying question (is BF2's
weakness sensor SNR or renderer artifact?) was **already asked and partly
answered ten days before this census**, on the separately-scoped `fixedT`
harness (`docs/B1B2_DEEP_prereg_2026-07-12.md` E-1/E-1b,
`docs/E2_MESH_AUDIT_prereg_2026-07-13.md` E-2c/E-3b) — that thread found the
mesh-artifact mechanism, flagged the paper's Table B4 BF2 row and caption as
"WRONG STATEMENT — replace with fine-mesh rescoring" (E2_MESH_AUDIT L418),
and was never carried into `wavesplat.tex` (confirmed: `git log` shows the
paper file's last content commit is 2026-05-03, i.e. it predates and has
never been touched by that finding or anything since — the paper is a
dormant artifact, not something anyone is actively failing to update).

**Net verdict**: not "badly mis-sized" — the top-3 re-validation list is
sound and the scope decisions are documented and reasonable. But "check every
headline/arc" missed a standing prose claim, and that claim turns out to be
the single largest reversal found in this whole exercise (see below). One
concrete process fix: before the mandatory 100-episode paper re-run
(census item 3), also grep `paper/wavesplat.tex` for other episode-specific
mechanistic prose (`\subsection{Failure case`, `worst case`, `floor}`) — I did
not find another instance beyond BF2 and the (harmless, structural)
directional-ambiguity section, but that grep should be someone's job, not
mine alone.

---

## Part 2 — ranked table: what's now worth trying / re-checking

| # | Method / claim | Documented in | Stated failure reason | Old mesh explains it? | Cost of multi-seed retry | What would justify reviving |
|---|---|---|---|---|---|---|
| **1** | **"Failure case: BF2 1-component"** — WaveSplat's reported worst episode | `paper/wavesplat.tex` L1553–1563, L1056, L1357; `results/12ep_full_analysis.json` | Physical: "signal-to-noise ratio is inherently low — wave amplitudes are on the order of LiDAR noise (~0.02 m)"; recommends an Hs-tuned SNR threshold | **MEASURED — yes, and it reverses.** This is exactly `mesh_adequacy.py`'s own flagship violation example (λ=8.2 m, BF2 1-comp, 1.2 pts/λ, below Nyquist). Ran `wavesplat.run` (`init_method=v2_konly, adaptive_n=True`, the paper's own config) on `data_finemesh_2026-07-23/episode_BF2_1comp_seed{0,1}_uniform.h5` vs `data/`: **old** nRMSE 0.271–0.320, ρ −0.055…+0.060 (8 runs, 2 data seeds × 3–5 estimator seeds, all near-zero field correlation) → **new** nRMSE 0.050–0.231, ρ 0.406–0.963 (8 runs). Every fine-mesh run beats every old-mesh run on both metrics. A prior (07-12/13), separately-scoped mini-audit on the `fixedT` harness independently found the same direction (fan 0.184→0.362, enc 0.348→0.504, `results/direction3/e3b_bf2_rescore.json`) and flagged the paper row as wrong (never corrected). | **Already done** (~3 min, 16 runs total, this session). A disclosure-quality version (5 data seeds × 5 estimator seeds × 2 mesh = 50 runs) is <30 min. | Rewrite §"Failure case: BF2 1-component" to state the actual mechanism (mesh under-resolution, not physical SNR) and cite the fine-mesh number instead of 0.315 once the mandatory 100-episode re-run (census item 3) lands. No new experiment needed — this table row is already produced by that re-run; only the prose is currently wrong and must be updated by hand. |
| **2** | A-7 / A-7v2 "quasi-monochromatic single-line super-resolution hybrid" (continuous-k refine + local fine wavenumber bank) | `docs/A7_D20_prereg_2026-07-12.md`; code `scripts/d3_a7_finebank.py`; `results/direction3/a7_finebank_eval.json`, `a7v2_refine_eval.json` | A-7 FAIL 0/5: "bottleneck is not the bank, it's peak super-resolution" — beamform max lands on a sidelobe in 3/5 seeds. A-7v2 FAIL 0/5: "discrimination-margin collapse at low SNR" — top-8 candidates' holdout residuals differ by <2%; true-k absent from top-8 in 2/5 seeds. | **MEASURED, partially, and self-contradicted in the record — genuinely unresolved.** Same-day follow-up (`docs/B1B2_DEEP_prereg_2026-07-12.md` E-1/E-1b) reran the identical BF2 episodes on an ad hoc fine mesh and got the A-7 hybrid to field corr 0.98–1.00 / encounter ρ 0.90–0.96, 5/5 — apparently overturning the FAIL. But `docs/E2_MESH_AUDIT_prereg_2026-07-13.md` E-2c found that test had an **uncontrolled second variable** (fine-mesh run used 50 obs/frame vs the FAILED benchmark's 8000) and, once matched, "the recovery margin roughly halved (enc 0.87→0.50)" — **for the plain hand estimator only** (`e3b_bf2_rescore.json` uses `mmse_gpu`, not the A-7 hybrid path). The matched-observation, mesh-only test was never rerun for the hybrid itself. Direction (mesh hurts, fine mesh helps) is solid; the specific "hybrid earns its complexity" magnitude (0.90–0.96) is unconfirmed on a fair test. | **Cheap but not free**: needs a small dedicated fine-mesh regen of the 5 BF2-1comp `fixedT` episodes at matched `n_pts=8000` (REGEN_SPEC already flags `fixedT` as deferred, ~165 s/episode per today's manifest → ~14 min for 5 seeds), then rerun the existing `d3_a7_finebank.py` (seconds). | Encounter ρ materially above the ~0.50 corrected-plain-hand floor **from the hybrid specifically**, isolating its own contribution over plain hand on the same matched fine-mesh data, in ≥3/5 seeds. A null (hybrid ≈ plain hand once matched) is also a fine, final answer — it would mean A-7's FAIL stands on its own terms and the mesh only explains the generic recovery. |
| **2b** | (no-cost footnote, not a separate item) `init_method='relax'` (continuous-k refinement, already implemented, same family as A-7v2's refiner) vs default `v2_konly` on fine-mesh BF2-1comp | checked directly this session, `wavesplat/run.py` | n/a — not a documented failure, a live option nobody compared here | n/a | **Already done**, ~1 min | `relax`: nRMSE 0.1663/0.1664/0.1663, ρ 0.618/0.618/0.618 across 3 seeds (N=1 every time) — essentially deterministic. `v2_konly` mean nRMSE 0.149 is slightly better but with real spread (0.050–0.231). Not a win, but a free, already-integrated low-variance alternative worth knowing about for weak/quasi-monochromatic regimes given the estimator's documented bimodality — does not need reviving, it already exists and works passably. |
| 3 | BF6/BF7 "scalability to higher sea states" (complexity, not sea state, drives degradation) | `paper/wavesplat.tex` L1734–1760 ("Scalability to higher sea states"); `scripts/run_bf67_experiment.py`; `results/bf67_experiment.json` | Complexity (component count), not Beaufort state, drives WS's ≥15-comp degradation (WS SS −0.28 vs LETKF +0.13) | **MEASURED — essentially no.** `run_bf67_experiment.py` is one of the 11 mesh-touching scripts, uses the plain default `WaveSimConfig()` (same violating mesh), and has **zero** fine-mesh companion (not in `data_finemesh_2026-07-23`; `REGEN_SPEC` §0 explicitly, defensibly scoped it out — paper already hedges this experiment as a linear-simulator-only confirmatory check). But BF6/BF7's *actual* wave components have shortest true wavelengths 17.8–26 m (higher Tp at higher Hs pushes the whole spectrum longer) — **not** the <15 m band the mesh destroys. Oracle-k check on the real files: transmission at those shortest components is 0.54–0.94 (moderate, not the 4–29% near-total loss seen elsewhere), each carrying <1.4% of field variance. | Cheap if ever prioritized (8 episodes, ~3 min/episode raycast per today's manifest timings, <30 min total) — but low expected payoff. | Only worth it if the eventual mandatory re-run's ≥15-comp SS moves by more than estimator noise; not expected to overturn "complexity, not sea state" given the transmission numbers above. **Clean null, correctly low priority as REGEN_SPEC already implies.** |
| — | Directional estimation deficit ($R^2$ −0.25 vs +0.07) | `paper/wavesplat.tex` L1516–1524 | Sensor-geometry azimuth-swath bias (structural) | **Excluded by the task's own discriminator** — structural sensor-FOV fact, not short-λ content; confirmed the metric is computed on the standard 100-episode set (will be swept up by the already-mandated re-run) and its stated mechanism does not depend on mesh-destroyed wavelengths. Listed only to show it was checked, not overlooked. | n/a | n/a — not reopened, no action beyond the standard re-run. |

### What did *not* reopen (checked and excluded)

- **`dir180` 30-file family**: paper's directional-deficit numbers come from
  the standard 100-episode set, not this probe; `d3_a2_dir180_eval.py`'s own
  claims are not cited in the paper with hard numbers. REGEN_SPEC's exclusion
  holds.
- **~40-architecture mean-collapse catalog** (T-NSE, TNP/TeTNP, KalmanNet,
  PINO, `wave_diffusion` v1-v8, `wave_np` v1-v4, `conv_cnp` v1-v3, DPPS, DDKL):
  excluded per the task's own rule — mean-collapse is architecture-level,
  independent of what fraction of variance a coarse mesh removes. Not
  re-examined further.
- **E-12~E-16 steepness/aperture arc, E-28, C1/C2, M-3/M-3G, B-2/E-60,
  TRAJ, Door-1/H1/H2, better-net/gap-ladder/wavenumber-ID-CRB, CRB spine**:
  independently spot-checked in Part 1, confirmed structurally immune, not
  reopened.
- **≤5 vs ≥15-component margin claim** (paper's model-order finding): already
  correctly flagged by the census (item 3) as needing the mandatory re-run;
  not treated as a new Part-2 item since it is not an *abandoned* method,
  it is a live claim already scheduled for revalidation.

---

## Bottom line

The un-asked question has a real, cheap, already-partly-answered "yes":
**the paper's own worst-case failure story (BF2 1-component, "inherently low
SNR") is not what it says it is.** It is the flagship example the project's
own mesh-adequacy checker was written to warn about, the direction was
already suspected and partially measured back on 2026-07-12/13 on a
different (deliberately out-of-scope) dataset, and a fresh, direct,
production-estimator, multi-seed test on the newly complete, properly paired
`data_finemesh_2026-07-23` set today confirms it outright (nRMSE roughly
halves, field correlation goes from noise-level to 0.66, 8/8 seeds). Fixing
it does not require new research: rewrite one paper subsection once the
already-mandated 100-episode re-run lands, and optionally finish the one
cheap, well-specified retest (A-7 hybrid, matched observations) that was
interrupted by a confound in July and never resumed. Everything else checked
in the candidate territory (BF6/BF7 nonlinearity, directional ambiguity, the
mean-collapse architecture catalog, the structurally-immune arcs) is a clean
null — genuinely not reopened, and not worth further effort.

## Scripts produced this session (all under `wavesplat/tmp/`, none touch `data/`)

- `reopen_bf2_estimator_check.py` — old vs fine mesh, BF2-1comp seed0, 5
  estimator seeds each.
- `reopen_bf2_estimator_check_seed1.py` — same, BF2-1comp seed1, 3 estimator
  seeds each (generalization check).
- `reopen_bf2_relax_check.py` — `v2_konly` vs `relax` init on fine-mesh
  BF2-1comp.
