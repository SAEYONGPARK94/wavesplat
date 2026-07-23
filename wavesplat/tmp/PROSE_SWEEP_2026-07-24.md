# Full-text prose sweep for "rendering artifact attributed to physics/algorithm" — 2026-07-24

Scope: every sentence in `paper/wavesplat.tex`, not just headline claims/tables, per the task
brief (the prior round missed both confirmed instances because it only looked at headlines).

## 0. IMPORTANT — the file was being edited concurrently while this sweep ran

`paper/wavesplat.tex` has **uncommitted local changes** on top of `HEAD` (`d99014b`,
2026-07-23T23:58 KST). At the start of this sweep the working tree already differed from `HEAD`
by 81 insertions/30 deletions; by the end it differed by **113 insertions / 30 deletions**
(re-checked twice, stable at 00:32 and 00:36 KST) — i.e. **someone or some other process was
actively revising this exact file during this audit**, in the direction of fixing exactly this
error class (see §2). This report is locked against the state as of **2026-07-24T00:36 KST**,
working tree vs. commit `d99014b` (`git diff --stat -- wavesplat/paper/wavesplat.tex` →
`143(+++++...---), 1 file changed, 113 insertions(+), 30 deletions(-)`). All line numbers below
are current-file line numbers as of that snapshot. If the file moves again before submission,
re-run `git diff -- wavesplat/paper/wavesplat.tex` against the numbers cited here before trusting
them literally — but the *substance* of each finding (which sentence, which evidence, which grade)
does not depend on exact line numbers and should still be checked against whatever text remains.

**Consequence for the report below**: several locations that an earlier round
(`tmp/METHODS_REREVIEW_fable_2026-07-23.md`) flagged as MATERIAL/COSMETIC have **already been
fixed** in the current working tree, apparently by whoever is mid-edit on this file right now.
I report those as "already fixed" (§3) rather than re-flagging them, and I found several **new**
live instances of the same error class that no prior document lists (marked NEW below).

---

## 1. Findings table — live (unfixed) instances

| Line(s) | Quoted sentence (short) | Claimed cause | What the evidence actually supports | Grade | MEASURED / ARGUED |
|---|---|---|---|---|---|
| 1632–1650 | `\subsection{Failure case: BF2 1-component}`: "the signal-to-noise ratio is inherently low... This suggests a minimum-SNR threshold or regularization schedule tuned to $H_s$" | Low physical SNR at BF2 | **Already confirmed FATAL by the launching task** (`% !!! CORRECTION REQUIRED` block already present, lines 1614–1631). I independently re-ran this exact episode this session (see below) and it reproduces the paper's number to 4 decimals, confirming the block's evidence is sound and the subsection prose is still unfixed. | **FATAL** (unfixed) | MEASURED (this session, see §1a) |
| 2011–2032 | `\emph{Tracking phase dependence on initialization}`: "$\|k\|\gtrsim 0.4$ rad/m fall below the beamforming detection threshold" | Beamformer's own detection/resolution property | **Already confirmed FATAL by the launching task** (`% !!! CORRECTION REQUIRED` block already present, lines 2011–2030). Still unfixed in the current prose. | **FATAL** (unfixed) | MEASURED (prior audit `METHODS_REREVIEW`, not re-run this session — budget) |
| **76–77 (Abstract)** | "At very low sea states the adaptive mechanism can overfit, a failure mode analyzed in the Discussion." | Low sea state → overfitting (points directly at the FATAL BF2 subsection above) | **NEW, this sweep.** MEASURED, this session: reran the exact cited case (`BF2_1comp_seed0`, `--seed 42`, `--adaptive_n true`) on both the shipped mesh and `data_finemesh_2026-07-23/`. Shipped mesh: `Pruned 15→8, Grew +3→N=11`, avg nRMSE = **0.3150** (matches the paper's quoted worst case, 0.315, to 3 decimals). Fine mesh, *same seed, same episode, same sea state*: `Pruned 15→1 (kept 99.9% energy), no growth needed` (residual RMS 0.0064 ≪ threshold 0.05), avg nRMSE = **0.1665**. The "overfit at very low sea states" behaviour the abstract reports as a property of the sea state essentially disappears when the observation channel is Nyquist-adequate, at the *same* sea state. | **MATERIAL** (borders FATAL; graded one notch below the BF2 subsection because "can overfit" is a hedged existential claim I have not disproven for *all* low-Hs scenes, only reproduced-and-flipped for the one the Discussion actually cites) | **MEASURED** (this session, n=1 seed, exact match to the paper's own number; SCALE_AUDIT independently found all 4 shipped-mesh BF2 seeds behave the same way, so this is not seed-cherry-picked) |
| **1030–1037** | "these aggregates are dominated by high-complexity episodes ($\geq$15 components) where our beamforming initializer, at its configured peak-separation radius (Sec.~\ref{sec:limitations}), cannot resolve individual modes. The $\leq$5-component threshold follows from the **spatial Nyquist limit** of the beamforming initializer (Section~\ref{sec:init})..." | Beamformer's "spatial Nyquist limit" | The first sentence is **already fixed** (correctly attributes to the *configured* peak-separation radius, cross-references the now-good Limitations disclosure — see §3). The **second sentence, unchanged**, still uses "spatial Nyquist limit," a term that (a) is not the beamforming initializer's actual limiting mechanism (that's the fixed $\Delta k_{\min}$ suppression radius, not a Nyquist/aliasing property) and (b) is easily confused with the *simulator mesh's* own, real, Nyquist violation, which the paper never discloses anywhere in its main text (see §5). | **COSMETIC** (downgraded from MATERIAL because the substantive claim right next to it is now correctly hedged; this is a terminology leftover) | ARGUED (wording precision) + MEASURED (mesh var_under numbers below back the disambiguation concern) |
| **1131–1133** | "For BF4–5 with 15–30 components, WaveSplat still outperforms ETKF... though the gap narrows as the spectral complexity approaches the **initialization resolution limit**." | Same "resolution limit" framing, general | **NEW, this sweep** — not in `METHODS_REREVIEW`'s list of 4. Same ambiguity as above: unqualified, no cross-reference to the now-good Limitations disclosure, reads as a fixed/method-independent wall. | **MATERIAL/COSMETIC** | ARGUED (same class as the already-audited locations; not separately re-measured) |
| **1541–1542** | "...narrows at higher complexity (BF5: 6.3\%...), reflecting the **initialization resolution limit** as beamforming peaks crowd for $N_{\text{true}}\geq 15$." | Same pattern | One of the original 4 locations `METHODS_REREVIEW` flagged (as old-line ≈1481); **still unfixed** in the current working tree even though 2 of its 4 sibling locations were fixed in the same edit pass. | **MATERIAL/COSMETIC** | ARGUED, cross-referencing `MIN_DK_RESULT_2026-07-23.md`'s measured ~5% algorithmic / ~95% survives-on-finemesh split (MEASURED by that prior audit) |
| **1779–1782** | "The 100-episode supplementary directional $R^2$ ($-0.253$...) is dominated by 30-component episodes (Dir. $R^2=-2.13$...), where **wavenumber-space crowding** makes the spectrum-space $R^2$ statistic catastrophically sensitive..." | Same pattern | **NEW, this sweep** — not in `METHODS_REREVIEW`'s list. | **MATERIAL/COSMETIC** | ARGUED |
| **1883–1886** | "For $\geq$15 components, detection drops to 50–38\% due to **wavenumber-space crowding in beamforming**, consistent with the initialization resolution limitation discussed above." | Same pattern | One of the original 4 (old-line ≈1807–1810); **still unfixed**, even though the passage it points to ("discussed above") is now, in one of its two remaining live instances, still just as unqualified. | **MATERIAL/COSMETIC** | MEASURED (var_under_resolved = 0.216 for `BF4_15comp`, 0.206 for `BF4_30comp`, independently re-derived this session, see §1b — confirms the mesh contributes a real but minority ~21–22% variance-fraction that this sentence never names) |
| **641–647** | "$\tau_R$... derived from the theoretical residual floor: combining LiDAR range noise ($\sigma_r\approx0.02$m)... yields a lower-bound RMS of $\sim$0.022m; $\tau_R=0.05$m provides a $\sim$2.3$\times$ safety margin" | Physical noise-floor budget only | **NEW, this sweep.** The derivation omits the mesh's own facet/rendering noise. `docs/E2_MESH_AUDIT_prereg_2026-07-13.md`'s E-3 gate measured **operator error (LiDAR z vs. the exact analytic surface) = 0.048 m on the standard coarse mesh** (same 40×30 config, BF3/BF4/BF5 5–15-comp episodes) vs. 0.006 m on a refined mesh — i.e. on shipped data, facet noise alone is **~2.2× the paper's own derived 0.022 m physical floor** and **~96% of the chosen 0.05 m threshold itself**. `SCALE_AUDIT_fable_2026-07-23.md` independently measured `residual_rms` at the grow-check point on 15-/30-comp episodes at 0.069–0.175 m — comparable in magnitude. The "2.3× safety margin" framing therefore overstates how well-understood the threshold's margin is on the data the paper actually uses. | **MATERIAL** | MEASURED (cross-referencing two existing prior audits' direct measurements; not independently re-run this session beyond confirming the source documents) |
| **1558–1562 (Discussion, headline 6× claim)** | "The most striking result is the $\sim$6$\times$ average advantage on 2-component seas... (WaveSplat 0.057 vs. ETKF 0.327...)" | (No explicit misattribution — flagged as adjacent/precision, not the core error class) | The 2-comp episode this number is built on (`BF3_2comp_seed0_uniform`) is **100% var\_under\_resolved** (independently re-measured this session, §1b), and the n=12 fine-vs-coarse pilot (`MESH_PILOT_RESULT_2026-07-23.md`) measured WaveSplat's nRMSE on this exact class moving **−20.2% to −37.5%** under mesh refinement, while the paired ETKF number is currently unusable (ETKF degrades on fine-mesh data for reasons still unresolved — `REGEN_SPEC` §4). The *direction* (WaveSplat ≫ ETKF) is not in question; the specific **"6×"** magnitude is not yet a number that survives the pending Nyquist-adequate re-run and should not be repeated as precise until it does. | **MATERIAL** (precision/citability caveat — this is the *adjacent* "symmetric audit" case: the paper is not necessarily over-claiming direction, but a headline ratio is being quoted to more precision than the underlying mesh-contaminated data supports) | MEASURED (var_under_resolved, this session) + MEASURED (pilot deltas, prior audit) |

### 1a. My own re-run of the abstract/BF2 case (MEASURED, this session)

```
python -m wavesplat.run --episode data/episode_BF2_1comp_seed0_uniform.h5 \
    --adaptive_n true --seed 42 --verbose
  -> [Adaptive] Pruned: 15 → 8 (kept 92.5% energy); Grew: +3 → N=11 (residual RMS=0.0578)
  -> Avg nRMSE = 0.3150   (paper line 1638: "nRMSE=0.315" — exact match)

python -m wavesplat.run --episode data_finemesh_2026-07-23/episode_BF2_1comp_seed0_uniform.h5 \
    --adaptive_n true --seed 42 --verbose
  -> [Adaptive] Pruned: 15 → 1 (kept 99.9% energy); no growth (residual RMS=0.0064 < 0.05)
  -> Avg nRMSE = 0.1665
```
Same episode, same seed, same sea state, same estimator, same code — only the raycast mesh
differs. On the shipped mesh the adaptive mechanism overparameterizes (N=1 true → N=11 fit,
exactly reproducing the paper's own worst-case number). On the Nyquist-adequate mesh it correctly
converges to N=1. This directly measures that the abstract's causal story ("very low sea states"
→ "overfit") is, for the one case the Discussion actually analyzes, substantially a rendering
artifact, not an SNR/sea-state property.

### 1b. My own independent re-derivation of `mesh_adequacy` numbers (MEASURED, this session)

Ran `wavesim.mesh_adequacy.check()` directly against the actual `config/wave_components` stored
in the shipped `data/*.h5` files (same procedure as prior audits, independently re-executed, not
copied):

```
mesh spacing: dx=6.154 m, dy=6.897 m  (config/default.yaml, mesh.nx=40, mesh.ny=30)

episode_BF2_1comp_seed0_uniform:  var_under_resolved=1.000  INADEQUATE
episode_BF3_2comp_seed0_uniform:  var_under_resolved=1.000  INADEQUATE
episode_BF3_5comp_seed0_uniform:  var_under_resolved=0.773  INADEQUATE
episode_BF4_15comp_seed0_uniform: var_under_resolved=0.216  DEGRADED
episode_BF4_30comp_seed0_uniform: var_under_resolved=0.206  DEGRADED
episode_BF5_15comp_seed0_uniform: var_under_resolved=0.093  DEGRADED
episode_BF5_30comp_seed0_uniform: var_under_resolved=0.072  DEGRADED
```
All values match the prior audits' cited numbers exactly (`METHODS_REREVIEW`'s 0.773 for BF3
5-comp; `GROUNDUP_fable_C_premises`'s table). Independent confirmation, not a re-citation.

---

## 2. Already fixed by the in-progress edit (credit, not double-counted)

Found via `git diff -- paper/wavesplat.tex` (working tree vs. `HEAD`). Listed so the sweep is
auditable as complete and so no one re-does this work:

| Was | Now (current text) | Verdict |
|---|---|---|
| Line ~1032–1035: "beamforming cannot resolve individual modes" (unqualified) | "our beamforming initializer, **at its configured peak-separation radius** (Sec.~\ref{sec:limitations}), cannot resolve individual modes" | **Fixed.** Correctly attributes to the tunable $\Delta k_{\min}$, not a fundamental wall. |
| Line ~1254 (fig:adaptive caption): "saturates... due to beamforming resolution limits." (unqualified, was the end of the caption) | Same sentence + appended: "The saturation reflects the configured beamforming peak-separation radius, not a method-independent limit (Sec.~\ref{sec:limitations})." | **Fixed.** |
| Limitations §, "Initialization resolution" bullet (2 sentences, unqualified: "beamforming cannot resolve all components due to wavenumber-space crowding") | Fully rewritten (now ~24 lines, 2013–2036 region) with the exact `min_dk` sweep numbers: $\Delta k_{\min}=0.15$, 126/195 episodes affected, 5/5-seed crowded-regime gain of 0.015 nRMSE (5.2%), 0/5-seed medium-regime harm, control-episode null (t=0.72), and the explicit caveat: *"statements in this paper of the form 'these components are not recovered' are properties of the estimator as configured... they should not be read as method-independent impossibility."* | **Fixed, and done well** — this is essentially the correction I would have proposed for the pattern, already in place, matching `tmp/MIN_DK_RESULT_2026-07-23.md`'s numbers to 3 significant figures. |
| Raydrop model description (§3.1) and the "specular backscatter" mechanism claims in both `\subsection{Raydrop model sensitivity}` (Results) and `\subsection{Effect of observation model}` (Discussion) | Rewritten to disclose that the "slope" model's specular gate is geometrically unreachable at this LiDAR's grazing angles ($\alpha/2=2.5°$ required vs. $78$–$90°$ beam fan) and always falls back to unweighted sampling; both mechanism sentences now explicitly decline to attribute the ETKF-vs-WaveSplat asymmetry to a mechanism ("we have not isolated this mechanism, and it should not be cited as established") | **Fixed**, and independently resolves `GROUNDUP_fable_C_premises_2026-07-23.md` finding #2 (a different, non-mesh MATERIAL finding from that audit — noted for completeness, out of this task's specific scope). |
| Related Work | Added a differentiation paragraph against Fucile 2018 (LEPrE) | Not part of this error class; matches the user's own standing note (`reference` memory: "Related Work에 Fucile OE2018 LEPrE 대비 delta 필수"). Mentioned only so the diff is fully accounted for. |

**Net effect on the "wavenumber-space crowding" pattern**: of 7 live locations found across this
sweep and the prior one, **2 are now fixed, 1 is partially fixed** (COSMETIC residue, line
1030–1037), **4 remain live** (lines 1133, 1541–1542, 1781, 1885 — see §1).

---

## 3. Checked and clean

Examined for the same attribution pattern (physics/algorithm invoked where mesh rendering could
be the real cause) or the adjacent sub-claims (nonlinearity, steepness, SNR, sparsity, resolution,
crowding) named in the task brief. No issue found; reasoning given so the sweep is auditable.

- **L184–190** (Intro, grid-based paradigm limitations): "smoothing near the Nyquist limit (5 m
  grid → ~10 m minimum wavelength...)". This is about the **ETKF baseline's own analysis grid**
  (a genuinely different grid, used for a genuinely different purpose — EnKF state discretization,
  not raycast rendering), and the argument (2 grid points per minimum wavelength) is standard and
  correct. Coincidentally the paper's tuned ETKF also happens to use a 40×30 grid
  (`tab:complexity`), which could read as the *same* grid as the raycast mesh to an attentive
  reader — worth a disambiguating footnote (see §5) but not a false attribution. ARGUED.
- **L584–590** (beamforming fine-refinement justification): "the 0.10 rad/m coarse grid would
  limit $\|\mathbf{k}\|$ accuracy to roughly 17%..." — checked the arithmetic: half-bin-width
  0.05/0.6 ≈ 8%, full-bin-width 0.10/0.6 ≈ 16.7% ≈ "roughly 17%." This is a real, verifiable
  property of the beamforming search grid's own $\Delta k$ spacing, unrelated to the raycast mesh.
  MEASURED (arithmetic check).
- **L683–689** (tracking-mode wavenumber freeze justification): "ill-conditioned due to spatial
  aliasing in sparse observations" — about **point-count sparsity** (how many LiDAR returns per
  frame), not elevation fidelity. `REVALIDATION_CENSUS_2026-07-23.md` §0.1(c) already measured
  that the raycast-mesh regeneration does **not** change return counts or spatial pattern (49.4–
  50.2 pts/frame on both meshes) — only what elevation each point reports. Different mechanism
  class from the target error. Clean by cross-reference (MEASURED, prior audit).
- **L792–795** (simulation environment description): "ground-truth elevation fields evaluated on
  a 80×60 spatial grid." Verified directly in `wavesim/pipeline.py`: the GT field (`Z_gt`) used
  for every metric in this paper is computed by `evaluate_field(components, X_gt, Y_gt, t)` —
  a **direct analytic evaluation at 80×60 grid nodes** (`gt_grid.nx=80, ny=60`, dx≈3.0 m,
  dy≈3.4 m) — and **never touches the raycast mesh** (`mesh.nx=40, ny=30`, dx=6.15 m, dy=6.90 m)
  used for LiDAR ray-triangle intersection. This sentence is accurate as far as it goes. The real
  problem — which this sentence does not create but also does not disclose — is that the paper
  never mentions the *separate, coarser* raycast mesh anywhere in its main text (see §5). MEASURED
  (code read, this session).
- **L1309–1322** (A1 ablation, off-shell drift "reflects over-fitting" not physical signal): a
  dimensional argument (5N free params vs. sparse observations) for why removing the dispersion
  constraint lets $\omega$ drift. Plausible and internally consistent; not contradicted by
  anything found. Not independently re-run on `data_finemesh_2026-07-23/` (would require rerunning
  the ablation; out of this task's budget). ARGUED, lower-priority open item, not a confirmed
  finding either way.
- **L1585–1596** (directional estimation deficit mechanism: narrow LiDAR angular sector,
  raydrop geometry preferentially keeping along-heading returns): a sensor-geometry argument,
  independent of elevation-rendering fidelity. Clean.
- **L1700–1840** (Comparison with stronger baselines: regime-dependent SS, LETKF variance
  deficit, localization-radius sweep, BF6–7 scalability/nonlinearity): all attribute effects to
  localization tapering, grid representational cost, or genuine second-order nonlinearity at
  BF6–7 — none invoke the raycast mesh, and the BF6/7 supplementary episodes are, per
  `GROUNDUP_fable_C_premises_2026-07-23.md`, the *only* 7 mesh-ADEQUATE episodes in the whole
  195-file set (their long wavelengths render fine on the coarse mesh), so this section's own
  premises are, if anything, the best-grounded in the paper. Clean.
- **L1889–1898 / L2041–2045** (amplitude decay / VR=0.66 "root cause": Adam momentum
  accumulating oscillating gradient signals from imperfect phase prediction): a generic optimizer
  dynamics story. I did not find contradicting evidence, but I also did not independently verify
  it against fine-mesh data (would require inspecting per-frame amplitude trajectories on both
  datasets — out of budget here). Flagged as a **lower-priority open question** for a future
  sweep, not a confirmed finding either way: it is plausible that some of the reported 0.17–8.2×
  amplitude relative error in the crowded regime is inflated by matching against components whose
  true amplitude was itself suppressed by the mesh before matching, but I have not measured this.
  ARGUED, unresolved.
- **L1475–1479** (sparsity fig caption: "BF3 2-comp episode shows a sharp improvement at 100% due
  to successful 2-component resolution"): about observation *density* (subsampling), which
  `REVALIDATION_CENSUS` already showed is empirically unaffected by the mesh (same point pattern,
  different elevation). This is a genuine beamforming-detection-count effect, not an elevation-
  rendering one, despite sharing an episode (`BF3_2comp`) with the headline-margin caveat in §1.
  Clean as written, though downstream of the same contaminated episode as the 6× claim.
- **L850–909** (ETKF/GPR baseline descriptions and the 24-configuration sweep): all about
  baseline hyperparameters and their own grids/kernels, independently re-confirmed sound by
  `METHODS_REREVIEW_fable_2026-07-23.md` findings #6–#7 (GPR length-scale self-calibrates,
  `mmse_bank_estimate`/annulus band is scale-invariant and never touches the raycast mesh). Not
  re-verified again here; cross-referenced.
- All other "due to / because / attributable to / traced to" sentences in the file (full list
  obtained via `grep -n "due to\\|because\\|attributable\\|traced to\\|root cause"`, 20 hits) were
  individually read; the ones not listed above (computational-complexity explanations, GPR
  $O(M^3)$ cost, statistical-variability footnotes, confidence-weighting interaction with the
  spectral basis, safety-rollback triggers under raydrop changes) are generic algorithm/statistics
  explanations unconnected to surface-elevation rendering fidelity. Clean.

---

## 4. Proposed replacement prose (LaTeX, ready to paste)

Only for items graded MATERIAL or FATAL in §1 that are **not already fixed**. The two already-
confirmed FATAL subsections (BF2 case, beamforming threshold) already carry a full corrected
paragraph in the `% !!! CORRECTION REQUIRED` comment blocks at their location — not repeated here.

**Abstract (L76–77), MATERIAL:**
```latex
% was: "At very low sea states the adaptive mechanism can overfit,
%       a failure mode analyzed in the Discussion."
At very low sea states, the worst-case episode in this dataset triggers
spurious adaptive-$N$ growth; Section~\ref{sec:limitations} traces most of
this to a simulator rendering limitation rather than an intrinsic low-SNR
failure of the estimator.
```

**$\tau_R$ derivation (L641–647), MATERIAL — append one caveat sentence:**
```latex
% after "...unambiguous detection of unmodeled signal."
On the raycast-mesh-rendered dataset used throughout this paper, however,
the mesh's own facet-rendering error contributes comparably to this budget
(mean $\approx$0.048~m, measured directly for BF3--BF5 multi-component
scenes on the shipped 40$\times$30 mesh, vs.\ the $\sim$0.022~m physical
floor derived above); part of what currently triggers residual-based growth
may therefore be unmodeled rendering error rather than unmodeled wave
content, and $\tau_R$ should be re-validated on a Nyquist-adequate mesh.
```

**Line 1036–1037 (COSMETIC, trailing phrase only):**
```latex
% was: "The $\leq$5-component threshold follows from the spatial Nyquist
%       limit of the beamforming initializer (Section~\ref{sec:init})
%       rather than a post-hoc search:"
The $\leq$5-component threshold follows from the beamforming initializer's
configured peak-separation radius (Sec.~\ref{sec:limitations}), not from
the sensor's spatial Nyquist limit, and not from a post-hoc search:
```

**Line 1131–1133, MATERIAL/COSMETIC:**
```latex
% was: "...though the gap narrows as the spectral complexity approaches
%       the initialization resolution limit."
...though the gap narrows as spectral complexity approaches the
beamforming initializer's configured peak-separation radius
(Sec.~\ref{sec:limitations}), not a method-independent resolution bound.
```

**Line 1541–1542, MATERIAL/COSMETIC:**
```latex
% was: "...reflecting the initialization resolution limit as beamforming
%       peaks crowd for $N_{\text{true}}\geq 15$."
...reflecting the beamforming initializer's configured peak-separation
radius as candidate peaks crowd for $N_{\text{true}}\geq 15$
(Sec.~\ref{sec:limitations} quantifies this as a regime-dependent trade,
not a hard wall).
```

**Line 1779–1782, MATERIAL/COSMETIC:**
```latex
% was: "...where wavenumber-space crowding makes the spectrum-space $R^2$
%       statistic catastrophically sensitive to single-bin peak misalignment."
...where the beamforming initializer's peak-separation radius
(Sec.~\ref{sec:limitations}) leaves candidate peaks closely spaced, making
the spectrum-space $R^2$ statistic catastrophically sensitive to single-bin
peak misalignment.
```

**Line 1883–1886, MATERIAL/COSMETIC:**
```latex
% was: "...detection drops to 50--38\% due to wavenumber-space crowding
%       in beamforming, consistent with the initialization resolution
%       limitation discussed above."
...detection drops to 50--38\%, consistent with the beamforming
peak-separation radius discussed in Sec.~\ref{sec:limitations}, which
bounds but does not fully explain the crowded-regime penalty.
```

**Headline 6× claim (L1558–1562), MATERIAL — add a footnote rather than rewrite the sentence:**
```latex
% after "...Table~\ref{tab:main})."
\footnote{Both 2-component episodes underlying this ratio render 100\% of
their true field variance below this simulator's raycast-mesh resolving
power (\texttt{wavesim/mesh\_adequacy.py}). A 12-episode paired
coarse/fine-mesh pilot found the \emph{aggregate} WaveSplat--ETKF margin
stable under mesh refinement ($+2.3\pm4.5$ points), but did not resolve
this specific pair individually. The precise ``$6\times$'' figure should be
treated as provisional pending the Nyquist-adequate 100-episode re-run.}
```

---

## 5. A structural observation, not separately graded

The raycast mesh — the actual mechanism behind both already-confirmed FATAL instances and most
of the MATERIAL items above — **is never named anywhere in the paper's main text**. `grep -n
"raycast\|mesh\b"` on the file returns only the three `% !!! CORRECTION REQUIRED` comment blocks
(LaTeX comments, not published text) and one unrelated "point-based rendering" analogy in a
footnote about the method's name. Section~\ref{sec:experiments} (`\subsection{Simulation
environment}`, L750–801) describes the 80×60 *ground-truth evaluation* grid (accurate, see §3)
but never mentions the separate, coarser 40×30 *raycast* mesh used to generate the LiDAR
observations that every table in the paper is actually built from, nor `wavesim/mesh_adequacy.py`'s
own finding that 0/102 shipped BF2–BF5 episodes pass its adequacy bar. This omission is the root
enabler of the whole pattern above: a reader has no way to independently notice the ambiguity in
"spatial Nyquist limit," "resolution limit," or "SNR," because the one additional Nyquist-relevant
object in the pipeline is undisclosed. Recommend disclosing the raycast mesh's resolution and the
`mesh_adequacy` verdict explicitly in Section~\ref{sec:experiments}, once, rather than relying on
per-sentence wording fixes throughout the rest of the paper.

---

## 6. Coverage note

Read the entire file, current version (2226 lines as of the snapshot in §0), sentence by sentence,
in overlapping passes covering line ranges 1–260, 250–500, 500–750, 750–1000, 1000–1250,
1249–1510, 1508–1552, 1540–1640, 1639–1940, 1935–2005, 2004–2145 (original ~2143-line numbering
before the concurrent edit grew the file), plus targeted re-reads of every touched region after
discovering the concurrent edits (via `git diff`), plus systematic `grep -n` sweeps across the
**current** file for: `SNR|signal-to-noise|steepness|nonlinear|crowd|resolution limit|Nyquist|
aliasing|sparsity|low sea state|overfit` and separately `due to|because|attributable|is caused by|
arises from|stems from|the reason|is explained by|results from|traced to|root cause` — every hit
from both sweeps was read in context and is accounted for in §1 or §3 above. Every
`\subsection`/`\section` heading in the document (full list obtained via
`grep -n "^\\\\subsection\|^\\\\section"`, 40 headings) was individually visited. Not read as
prose (tables/algorithm listings/figure files themselves, only their captions and surrounding
text): the raw numeric contents of Tables `tab:nomenclature`, `tab:sensor_comparison`,
`tab:complexity`, `tab:etkf_sweep`, `tab:fivemetric`, `tab:bybf`, `tab:main`, `tab:adaptive`,
`tab:ablation`, `tab:raydrop`, `tab:sparsity`, `tab:noise`, `tab:strong_baseline`,
`tab:spectral_accuracy`, `tab:sparse_comparison`, `tab:forecast` — their captions were read as
prose and are covered above; cell-by-cell arithmetic re-verification of every table was out of
scope (the task specified prose, and this round already independently re-derived the two most
load-bearing numeric claims — the BF2 worst case and the mesh `var_under_resolved` figures — by
direct re-execution, §1a–1b).

## 7. What was NOT independently re-run (budget), and why it doesn't weaken the grades above

- The $\geq$15/30-comp crowding-pattern sentences (§1) are graded MATERIAL/COSMETIC, not FATAL,
  because `tmp/MIN_DK_RESULT_2026-07-23.md` already measured (both on the original and the
  finemesh dataset, with a pre-registered control) that the crowded-regime wall is **~95% real**
  and only ~5% attributable to the fixed-radius artifact — i.e. the *substance* of "crowding hurts
  ≥15-comp episodes" is sound; only the *wording* ("Nyquist limit," "resolution limit" used
  without ever naming the tunable radius or the mesh) is imprecise. I did not re-run `min_dk_sweep`
  myself this session; I independently re-derived the `mesh_adequacy` `var_under_resolved` numbers
  it depends on (§1b) as a partial check.
- I did not re-run the ablation (A1/A2) or the amplitude-decay mechanism on
  `data_finemesh_2026-07-23/` — flagged in §3 as open, unresolved questions, not findings.
- I did not attempt to reproduce the 6× headline claim's ETKF side on fine-mesh data — this is
  already blocked project-wide (`REGEN_SPEC_fable_2026-07-23.md` §4: ETKF degrades on fine-mesh
  data for reasons unresolved; no fine-mesh ETKF number is currently citable).
