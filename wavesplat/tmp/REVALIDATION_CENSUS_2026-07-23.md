# Re-validation census — which past negatives the fine-mesh dataset actually threatens

2026-07-23. Scope: every documented negative / failed-method result in `docs/`, `tmp/`,
`results/` weighed against the paired fine-mesh replacement `data_finemesh_2026-07-23/`
(100/100 files, `config/wave_components` bit-identical to `data/` for the same seed).

> **한 줄 요약**: 진짜 재검증이 필요한 건 **1건**(ANM 게이트의 short-λ 주장 — 전제가 이미
> 반증됨), 값싸서 같이 돌릴 만한 게 **1건**(min_dk 혼잡레짐), 그리고 어차피 예정된
> 100-에피소드 재실행에 **multi-seed를 붙이는 것**뿐이다. 나머지 수십 건은 재검증 불필요하며,
> 이유는 대부분 "그 실험은 애초에 레이캐스트 메시를 지나간 적이 없다"이다.

---

## 0. The one measurement this census turns on

Everything below is ranked against a single new, cheap, oracle-k measurement of the
**observation channel itself** — not of any estimator.
Script `tmp/revalidation_shortlambda_probe.py`, log `tmp/revalidation_shortlambda_probe.log`,
pre-registered read stated in the script docstring **before** the numbers. It fits cos/sin
coefficients at each episode's **true** `(kx, ky, ω)` to the actual LiDAR returns
(over-determined ~500:1) and reports `A_recovered / A_true` — an upper bound on what *any*
estimator could extract.

| episode | λ | A_true | **OLD mesh** | **NEW mesh** |
|---|---|---|---|---|
| BF3 2comp s1 slope | 8.37 m | 0.0372 | **0.05** | **0.90** |
| BF3 5comp s0 | 6.25 m | 0.0279 | **0.04** | **0.83** |
| BF3 5comp s0 | 9.33 m | 0.0455 | 0.09 | 0.93 |
| BF4 15comp s0 | 8.99–13.10 m | 0.022–0.035 | 0.16–0.29 | 0.95–0.96 |
| BF4 30comp s0 | 8.74–11.84 m | 0.015–0.021 | 0.07–0.42 | 0.92–0.97 |
| BF5 30comp s0 | 14.45–19.58 m | 0.025–0.036 | 0.37–0.75 | 0.97–0.99 |
| all episodes | λ > ~20 m | — | 0.93–1.00 | 0.99–1.00 |

Two facts follow, and they pull in opposite directions:

1. **Below λ ≈ 15 m the shipped data did not carry the signal at all** (4–29 % of true
   amplitude). Any claim of the form *"component X was not recovered"* about such a
   component is circular — it was not there to recover.
2. **That band carries 1.7–6.3 % of the field variance.** So field nRMSE, which is
   variance-weighted, barely moves. This is exactly why the 12-episode pilot found the
   headline flat (WS 0.2135 → 0.2055, t = −0.44) *and* why the many-component episodes
   "barely moved" (+0.1 %, +3.8 %, −7.4 %, +3.0 %) — those two observations are **not**
   evidence that short-λ content was already fine. They are evidence that short-λ content
   is energetically negligible. The two questions were being conflated.

⟹ The regeneration can only overturn **component-level claims about λ ≲ 15 m**. It cannot
overturn field-error claims, and it cannot overturn anything that never touched the mesh.

## 0.1 The triage criterion, applied — and one part of it corrected

- **(a) depended on short-wavelength content** — live, and it is the *only* live axis.
  Narrow it to: *claims about identifying or recovering individual components with
  λ ≲ 15 m*.
- **(b) effect size small enough to be moved by the mesh change** — live but almost always
  **misdiagnosed**. The estimator has run-to-run CV 2.8–53 % on a fixed episode
  (`ESTIMATOR_VARIANCE_RESULT_2026-07-23.md`), i.e. paired-difference noise sd 0.011–0.070
  nRMSE. Effects smaller than that were never resolvable **at any mesh**. Such items do not
  need new data; they need **seeds**. Re-running them single-seed on the fine mesh would
  produce a second uninterpretable number, not a correction.
- **(c) depended on the *pattern* of LiDAR returns** — ⛔ **empirically inert for this
  regeneration, and I recommend dropping it.** MEASURED: returns per frame 49.4–50.2 on
  both meshes; range distribution p5/p50/p95 agree within 0.3 m on BF3-5comp and BF4-15comp;
  total valid returns per episode within 0.7 %. The mesh changed **what elevation a return
  reports**, not **where returns land**. The correct replacement is
  **(c′) claims that depend on the high-wavenumber content of the reported elevation** —
  which is just (a) again. Consequence: `f6_c8_own_verify.py` (C8 estimator-independence)
  and `d1_crb.py` (CRB / predictable zone), both of which consume only the return
  **geometry** `(x, y, t)`, are **safe**, despite reading `frames/lidar_points`.
- **Safe by structure / pre-emption / understood-failure** — as given, and it absorbs the
  overwhelming majority of the record (§3).

## 0.2 Two prior results that bound how much anything can change

- **E-3 (2026-07-13) already ran this exact intervention.** 35 episodes, mesh only
  (40×30 → 180×150), paired, `wave_components` identical. Operator error fell
  **0.048 → 0.006 m (8×)**; the measured effect Δ(learned − hand) moved **−0.045 → −0.032**,
  i.e. ~1/3 of it was the renderer, **no sign flip**, all three pre-registered gates PASS
  (`docs/E2_MESH_AUDIT_prereg_2026-07-13.md` §E-3). A repeat of that magnitude is the
  prior for every field-skill quantity here.
- **E-4 (same round) cleared the HOS arena**, which is where most of the learning arc lives:
  `MESH_DX = 2.0 m`, λ_peak ≈ 99.6 m ⇒ 49.8 pts/λ, unresolved-band energy **0.17 %**, bound
  harmonics at λ/2, λ/3 ⇒ 25 / 17 pts/λ. The HOS stage is not affected by this regeneration
  and never was.

---

# TOP 3 — this is the whole list that is worth effort

Ranked by (threat to a live paper/thesis claim) × P(the regenerated data changes it) ÷ cost.
**Item 1 is in a different class from items 2 and 3**: its premise is already measured false.
Items 2 and 3 are prudent, cheap, and expected to *confirm*.

| # | Claim (as the project states it) | Documented in | Class | Why | Cost (multi-seed) | What would overturn it |
|---|---|---|---|---|---|---|
| **1** | *"**Weak short-λ components are unrecoverable by ALL methods at ALL apertures** (2comp \|k\|=0.75 A=0.037 missed at 3/10/30 s windows; 5comp/15comp miss the smallest-amplitude short comps). ⟹ ANM does **not** supply the missing 'achievability' half — the crowded/weak regime is the information wall, exactly as the record says."* | `tmp/NOVELTY_ROUND3_execution_2026-07-23.md` §A read 2; inherited verbatim as a caveat by `tmp/MIN_DK_RESULT_2026-07-23.md` (last §) and flagged independently as *"the single most important documented conclusion to re-derive"* by `tmp/REGEN_SPEC_fable_2026-07-23.md` §7.3 | **(a)** | The named component **is** `episode_BF3_2comp_seed1_slope.h5`, \|k\|=0.75 ⇒ **λ = 8.37 m**, A = 0.0372 — measured above at **5 % amplitude transmission on the shipped mesh, 90 % on the fine mesh**. At 1.2 pts/λ the raycast surface could not render it. The claim is therefore **circular as written**: "unrecoverable at all apertures" was measured on a channel that carried 5 % of the component. The aperture sweep (3/10/30 s) cannot rescue it — no window length recovers content the renderer removed. This is load-bearing: it is the empirical half of the spine's *"crowded/weak regime is the information wall"* and the reason the ANM/atomic-norm lever was declared to supply no achievability guarantee. | **Low.** `tmp/blasso_feasibility.py` against `data_finemesh_2026-07-23/`, same 4 episodes, 3 obs-subsample seeds. These are single-shot fits on a 2600-point subsample, not 301-frame estimator runs. **≈ 30–60 min.** The oracle-k half is already done (10 s, above). | It is already overturned **as stated**. The re-run decides which of two restatements is true: (i) *the components are now recovered* ⇒ delete the claim, and the "information wall" loses its short-λ leg (the wall then rests on crowding/`min_dk`/CRB alone); or (ii) *they are present at 90 % amplitude and still missed by greedy+joint and by SFW* ⇒ the claim is **re-established on sound data and becomes much stronger**, and the ANM gate's NEGATIVE verdict is genuinely earned. Either outcome is publishable; the current one is not citable. |
| **2** | *"Lowering `min_dk` is not the fix … there **is** an algorithmic component: ~0.016 nRMSE, about **5 %** of the crowded-regime error … but **the crowded-regime wall is not an artifact of this constant**."* | `tmp/MIN_DK_RESULT_2026-07-23.md` | **(a) + (b)** | Measured on BF4-15comp / BF3-5comp — precisely the episodes where 4–8 true components were transmitting at **7–42 %** amplitude. The experiment asks how much of the crowded-regime error is algorithmic (resolution floor) versus informational; it was run with part of the crowded spectrum physically absent. Restoring it changes the number of genuinely-present close pairs the greedy search must separate, so the 5 % / 95 % split is measured on the wrong spectrum. The verdict's *direction* (loosening helps crowded, hurts medium) is a within-mesh A/B and will probably survive; the **quantitative bound**, which is what the information-wall argument leans on, will not necessarily. Also note the crowded result was already borderline (t = −2.06, carried by 5/5 sign consistency). | **Low.** Re-run `tmp/min_dk_sweep.py` verbatim against the fine-mesh set: 3 episodes × 3 radii × 5 seeds ≈ 45 runs × ~30 s ≈ **25 min**. Add BF4-30comp + BF5-30comp for the genuinely crowded end: 75 runs ≈ **40 min**. | The algorithmic share rising materially above ~5 % of crowded-regime error — say Δ(0.05 − 0.15) worse than about −0.04 with 5/5 direction — would mean the crowded-regime "wall" is substantially an artifact of a hardcoded constant, which directly weakens the empirical leg of the information-wall claim. The control (BF3-2comp, no true pair < 0.15) must stay null; if it moves, the measurement is invalid. |
| **3** | *"these aggregates are dominated by high-complexity episodes (≥15 components) where beamforming cannot resolve individual modes. **The ≤5-component threshold follows from the spatial Nyquist limit of the beamforming initializer** … rather than a post-hoc search"* (and the paired negative: WS SS CI `[−0.57, −0.36]` for ≥15-comp) | `paper/wavesplat.tex` L986–1000, L1708–1720, L1950–1958 | **(b)**, plus a wording defect | The *ranking* is safe — the pilot moved the margin by +2.3 ± 4.5 points and the ≥15-comp episodes moved 0.1–3.8 %. What is not safe is the **attribution**: "spatial Nyquist limit" now has two possible referents (the beamformer's wavenumber resolution, and the *simulator mesh's* spatial Nyquist limit, which was violated on 100 % of these episodes) and the paper does not distinguish them. Separately, every per-episode number in the paper is a **single draw from a bimodal estimator** (CV 2.8–53 %); the ≥15-comp CI is the most defensible of them only because that regime happens to have the *lowest* CV (2.8 %). | **High**, but already mandated for correctness and disclosure independent of this census (`REGEN_SPEC` §7.1). 100 episodes × 5 seeds × ~30 s ≈ **4.2 h** for WaveSplat alone. ⛔ **Blocked for any margin claim**: ETKF *degrades* on fine-mesh data and a grid/Ne/inflation/obs_noise sweep failed to explain it (`REGEN_SPEC` §4); no fine-mesh ETKF number may be cited until that closes. | A sign flip in the ≤5 vs ≥15 complementarity (median-of-5, not seed 42) would rewrite Section "strong baselines". Much more likely outcome: same direction, wider and honest intervals, and the "spatial Nyquist" sentence has to be rewritten to name the beamformer explicitly. Note the ≥15-comp regime now carries ~1.8 % *more* genuinely-present short-λ variance that the model must fit, so the crowded penalty could get slightly **worse**, not better. |

**Prerequisite before items 2 and 3 (not itself a past negative):** re-measure run-to-run
variance on the fine-mesh episodes — 4 episodes × 8 seeds ≈ **16 min**
(`tmp/estimator_variance.py` pointed at the new directory). Every ranking above assumes the
bimodality carries over; if the finer surface removes the bad basin, the whole re-validation
gets cheaper, and if it does not, median-of-5 with spread is the minimum reportable unit.
`tmp/RESTART_SELECTION_RESULT_2026-07-23.md` already showed GT-free best-of-k does **not**
fix it, so this is error bars, not selection.

**Everything below this line is lower value. I do not recommend spending re-validation
effort on it, and several entries are actively impossible to change with new data.**

---

# 1. Explicitly NOT threatened — because the experiment never touched the raycast mesh

This is the single largest class and it is worth stating precisely, because "measured on the
old data" has been assumed far more broadly than the code supports. Only **11 of 413
scripts** in `scripts/` touch `run_episode` / `LidarSimulator` / `raycast_surface` at all
(`grep -rln "run_episode\|LidarSimulator\|raycast" scripts/*.py`), and 5 of those 11 are the
mesh-audit / regeneration tooling itself.

| Claim / arc | Documented in | Why it is safe |
|---|---|---|
| **E-12 ~ E-16 aperture / steepness arc** — "steepness is uncorrelated with the learning gain (ρ −0.21)"; "the axis is aperture ratio (ρ +0.41)"; E-16 "operator fixed, physics swapped ⇒ dd ≈ 0, long-λ aperture makes the gain, short-λ physics misspecification is the only driver" | `docs/E12_KP_FLOOR_AUDIT_prereg_2026-07-14.md` §15; `tmp/e16_physics_operator.py` | Runs on the **HOS arena** (`d3_h7_hos_arena` / `d3_f1_hos_door1`, `MESH_DX = 2.0 m`), not on `data/`. E-4 cleared that mesh: 49.8 pts/λ at the peak, 0.17 % unresolved energy. Its short-λ conclusions are about **its own** well-resolved arena. |
| **E-28 generalisation nets** — "two orthogonal designs, neither beats the current net ⇒ information wall confirmed on the design side" | `docs/E12_KP_FLOOR_AUDIT_prereg…` §17; `tmp/e28_gennet_{A,B}.py` | Data comes from `jonswap_components` + `sample_obs` — an **analytic sampler**. No mesh, no raycast, no LiDAR returns. Mesh regeneration is a no-op. |
| **E-29 ~ E-46 regime arc** — blind ≈ +0.0024 ± 0.0090; prize +0.072 ± 0.004; buoy-grade supply has no real input; the 12-record census refuting "real seas are cross-Tp" | `docs/E29_E39_CLOSURE_OVERTURNED_2026-07-17.md`, `docs/E42_E43_REGIME_ROUTE_2026-07-17.md` | Same analytic-sampler arenas plus real stereo records. The census leg is literature/records. Nothing raycast. |
| **Better-net FAIL / parity; gap-ladder "+0.45 headroom"; wavenumber-ID CRB** | `tmp/BETTERNET_fable_2026-07-22.md`, `tmp/GAP_LADDER_RESULT_2026-07-22.md`, `tmp/WAVENUMBER_ID_CRB_RESULT_2026-07-22.md` | These **do** open `data/*.h5`, but only to read `config/wave_components` — every observation is re-synthesised as `predict(θ) + N(0, σ²)` (fable verified: *"no episode's z observations pass through anything but predict(theta)+Gaussian… despite the episode files storing them"*). Components are **bit-identical** between the two datasets, so the fine mesh changes nothing. Their real defect is the **inverse crime** already logged — that is a design problem, not a data problem, and new data does not touch it. |
| **C1 common ceiling** (learned prior ≈ analytic prior; physics-constrained cluster above weak RELAX) | `tmp/C1_PILLAR_common_ceiling_2026-07-22.md`, `results/direction3/compare_metrics.json` | Sources are "Linear sim (200-comp JONSWAP)" analytic arena, HOS ladder, real stereo, and the C8 sandbox. None is a shipped raycast episode. |
| **C2 control demo NO-WIN** (Δ +0.038, CI [−0.028, +0.107]) | `tmp/C2_RESULT_2026-07-22.md` | 32 HOS seas, RELAX fits, no raycast. And it died of **power**, not of data quality: the effect collapsed monotonically +0.088 (n16) → +0.038 (n32) → +0.010 (n64). New data cannot revive an effect that shrinks as power grows. |
| **M-3 sensor-kp FLAG / M-3G retrain gate VOID** | `tmp/M3_RESULT_2026-07-22.md`, `tmp/M3G_RESULT_2026-07-22.md` | HOS arena. M-3G is a **no-op by code inspection** (`d3_prior_net_data.py:132-138` already beamforms kp), which no dataset can change. |
| **B-2 loss-swap arc (E-60/E-B2K)** | `tmp/E60K_RESULT_2026-07-20.md` + audits | HOS 32-sea arena. Its live threat is oracle-kp dependence (M-3), not the mesh. |
| **Trajectory-policy FAIL** ("moving-trajectory advantage is an oracle-wavenumber artifact") | `tmp/TRAJ_RESULT_2026-07-22.md` | Analytic sampler, linear seas, n=24. Mesh-independent, and the mechanism (wide sparse coverage makes wavenumber estimation harder) is structural. |
| **SBI/NPE INCONCLUSIVE** | `tmp/SBI_RESULT_summary_2026-07-22.md` | Synthetic regime-shape simulator. Died of NPE mis-calibration (PIT KS p 1e-5…1e-11). |
| **C8 estimator-independence (scope-limited SUPPORT; GP counter-example)** | `tmp/C8_RESULT_2026-07-22.md`, `tmp/F6_c8_verify_2026-07-22.md` | Reads `frames/lidar_points` but uses only the **geometry** `(x, y, t)`; `z` is synthesised. Geometry is statistically identical between meshes (§0.1c). Safe. |
| **CRB spine: predictable zone, information-wall time, θ-CRB (E-50/E-51)** | `docs/direction1_crb*`, `scripts/d1_crb.py`, `tmp/e50_crb_direction_check.md` | Fisher information is a function of true θ and observation geometry only. θ is bit-identical; geometry is unchanged. The CRB numbers do not move. (Their real open item remains GUARD2 achievability, unrelated to mesh.) |

# 2. Explicitly NOT threatened — structural, pre-empted, or a large understood failure

| Claim / arc | Documented in | Killed by |
|---|---|---|
| **Door-1 single-channel closure** — mask/occlusion channels all closed; ray-inequality ceiling +0.002; hard inequality −0.0420 ± 0.0152 once the leak is removed | `docs/E5_DOOR1_IS_ONE_CHANNEL_2026-07-10.md`, `docs/DOOR1_INFORMATIVE_MISSINGNESS_2026-07-10.md`, `docs/G0_…`, `docs/F2_DETECTION_PHYSICS_2026-07-10.md` | **Structural + already fine-meshed.** Runs on its own `MESH_DX = 2.0 m` surface (`d3_door1b_field_bias.py:158`), and the mechanism is a statement about physics, not resolution: *the inequality only carries information where the ray grazes, and that is exactly where the range is unknown*. A finer mesh does not give you the range. |
| **H1 "the mask has no destination" / H2 task-loss closure** | `docs/H1_MASK_DESTINATION_2026-07-11.md`, `docs/H2_TASKLOSS_CLOSURE_2026-07-11.md` | Same 2.0 m surface; and the closure is triply instrumented (hand EP, τ-sweep, task-loss ×2) with an exact MCAR identity control at 0. The residual +0.009 was traced to a ~7× overconfident posterior. Not a data-quality result. |
| **W7 self-supervised headline falsified** (LORO R² −0.603 = record fingerprinting) | `docs/SSL_FALSIFIED_and_LJ_GT_RANGE_2026-07-10.md` | Real stereo records; never passes through the raycast operator (`d3_falcon_real_operator.py` reads the stereo grid directly, per `E2_MESH_AUDIT` ripple list). |
| **Raw-coordinate / feed-forward regression mean-collapse family** — T-NSE input-invariant collapse, TNP/TeTNP, KalmanNet stages, PINO stages, `wave_diffusion` v1–v8, `wave_np` v1–v4, `conv_cnp` v1–v3, DPPS, DDKL | `results/*.json` (≈30 files), `docs/LEARNING_HISTORY_REVIEW_2026-07-08.md`, `project_tnse_status` | **Large, mechanistically-understood failure.** ~40-architecture map: every raw→point-estimate design mean-collapses; every success is physics-regularised prior + closed-form solve. A 2 % variance band at λ < 15 m does not resurrect a collapsed regressor. |
| **ANM/atomic-norm as a novelty lever — theory half** (Agent F: dispersion-ANM recovery would be a near-corollary of Chandrasekaran et al. FoCM 2012 and Yang & Xi arXiv:2503.15164) | `tmp/NOVELTY_ROUND3_execution_2026-07-23.md` §B | **Literature pre-emption.** Unaffected by data. Note this is *independent* of item 1 above: even if the ANM gate's empirical verdict is restated, the theoretical novelty stays dead. |
| **CRB-driven active sensing (Agent E)** | `tmp/NOVELTY_ROUND3_execution_2026-07-23.md` §C | Off-the-shelf method (Joshi–Boyd, OED) + null-risk. Pre-emption. |
| **Method-novelty pre-emption generally** (Ehlers OE2023, Desmars JFM2023, PINO2025, Fucile 2018, Qi 2018) | `project_novelty_phd_verdict_2026-07-15`, `docs/NOVELTY_survey_amortized_lidar_2026-07-01.md` | Pre-emption. No dataset changes it. |
| **E-10 return-pattern sensitivity FAIL** (underpowered, 5 clusters) | `docs/E10_PATTERN_prereg_2026-07-13.md` | Reweights **dense stereo pixels**, not simulated returns; and it failed on power. Also note it is the closest thing the project has to a genuine class-(c) test, and it is not affected by this regeneration. |
| **Alt-methods (bank family) gate NEGATIVE** — "the bank/covariance family never beats the beamforming component family anywhere tested"; wide-grid gap is predominantly extrapolation | `tmp/ALTMETHODS_GATE_RESULT_2026-07-23.md` | Runs on shipped returns, so not mesh-*independent* — but it is a **within-mesh A/B** where both arms see the identical 2600-point subsample, and the surviving margins are large (far-band 3–4×; near-band BF4 t = 10.36). The 2 % restored variance cannot reverse a 3× gap. Optional 20-min confirmation only if item 1's re-run is being set up anyway (same episodes, same loader). The gate's real caveats — the plane-wave model-class gift and no HOS replication — are untouched by new data. |
| **E-6 "BF2 residual weakness is a sampling limit" — hypothesis rejected** | `docs/E2_MESH_AUDIT_prereg_2026-07-13.md` §E-6 | Already re-judged **on fine-mesh data** in July, by design. Done. |

# 3. Open items that are *not* re-validation but block citing fine-mesh numbers

1. **ETKF degrades on fine-mesh data and the mechanism is unresolved** (`REGEN_SPEC` §4;
   plus G8: the pipeline uses `n_ensemble=50` while the repo's own sweep found 100 better,
   0.358 vs 0.416). ⛔ No fine-mesh ETKF number, and therefore no margin claim, until this
   closes. This is the actual critical path for the paper, ahead of anything in the census.
2. **The peak-locked `_pk` companion (30 files) has not been generated.** Until it is, the
   1-component episodes still confound *mesh* with *spectral-flank placement*
   (`n_components=1` sits at 1.75·ω_p ⇒ λ = λ_peak/3.06). The pilot's most spectacular
   improvements were all 1-comp/2-comp episodes (−28 % to −57 %), and those are exactly the
   episodes with this second defect. **Do not cite the 1-comp improvements as a mesh effect.**
3. **Scale-dependent constants are live and unfixed** — `grow_residual_threshold = 0.05 m`
   compared against a raw metre residual fires on every ≥15-comp episode at Hs ≥ 1.2, and
   `lambda_l1 = 0.001` is up to 62 % of the total loss (`tmp/SCALE_AUDIT_fable_2026-07-23.md`).
   These interact with item 3 of the top-3: re-running the paper table on new data while
   these remain uncalibrated re-measures the same defects on better data. Deciding whether
   to fix them **before** the 100-episode re-run is a bigger lever than any item in this
   census — but it is a code change, not a re-validation, so it is out of scope here.

# 4. Honest bottom line

- **One genuine re-validation: item 1.** Its premise is measured false, it is cheap, and
  both possible outcomes are usable.
- **One cheap companion: item 2**, worth doing in the same sitting because it shares the
  crowded episodes and the code already exists.
- **Item 3 is not really a census entry** — the 100-episode re-run is mandated for
  correctness and disclosure regardless. The census's contribution there is only:
  *do it median-of-5, and rewrite the "spatial Nyquist" sentence.*
- **Everything else: no.** The dominant reason is not that the effects are small; it is that
  **the experiments never went through the raycast mesh.** The learning arc, the regime arc,
  the CRB spine, Door-1, and the C-series all live on the analytic sampler, the 2.0 m HOS
  arena, real stereo, or `config/wave_components` — and components are bit-identical.
- **The mesh defect remains real and worth disclosing.** It is a correctness and
  reproducibility matter (0/102 shipped episodes passed the repo's own
  `wavesim/mesh_adequacy.py`), not a results matter. That framing is now supported by three
  independent measurements pointing the same way: the pilot (headline flat), E-3 (8× less
  operator error, no sign flip), and §0 above (short-λ absent but energetically negligible).

## Limits of this census

Desk review plus one 10-second probe on 5 episodes, 1 seed per (episode, mesh) — the probe
is an oracle-k LS fit with no estimator stochasticity, so it needs no seeds, but it does
assume the 5 episodes are representative of their families. Script data-source attributions
were established by reading imports and call sites, not by executing every script; a script
that reads `config/wave_components` and synthesises observations was classed mesh-independent
on that basis. No claim here has been re-derived by running its original experiment.
