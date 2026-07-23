# 2026-07-24 session — paper re-run, prose corrections, code scale fixes

Scope: the 18-item backlog agreed at the start of the session, minus item 17 (supervisor
sign-off, excluded by the user). Work was run in parallel: one CPU batch for the table
re-run plus four agents (prose sweep, ETKF diagnosis, regeneration gap, scope/Fucile).

---

## 1. What was corrected in the paper, and why it mattered

### 1.1 The two `CORRECTION REQUIRED` blocks are now resolved into prose

Both had sat in the file as comments since 2026-07-23; one of them had been flagged as a
"WRONG STATEMENT" by an internal audit on **2026-07-12/13** and never carried into the text.

| site | claimed cause | what is true |
|---|---|---|
| `Failure case: BF2 1-component` | "signal-to-noise ratio is inherently low"; remedy = $H_s$-tuned regularizer | $\lambda=8.2$ m rendered at 1.2 pts/$\lambda$ — below Nyquist. Median over 5 seeds on the adequate re-render: **0.315 → 0.169 ± 0.066** |
| `Tracking phase dependence` | "$\|k\|\gtrsim0.4$ rad/m falls below the **beamforming detection threshold**" | The beamformer has no such threshold (recovers $\|k\|$ 0.15–1.00, amplitudes to 0.01 m). 0.4 rad/m ⟺ $\lambda\approx15.7$ m = the **old mesh's** transmission collapse |

**The BF2 rewrite is not a simple retraction.** A single-seed re-run (seed 42) shows the
fine mesh converging cleanly to $N$=1, which reads as "pure artifact". With $k=5$ the model
order ranges $N$ = 1, 3, 15, 15, 15 on the *same fixed episode*. So spurious adaptive-$N$
growth on calm seas is real; it is a convergence-basin phenomenon, not a sea-state SNR
limit. The paper now says that, and the prescription changes accordingly — multiple restarts
with median reporting, not an $H_s$-tuned regularizer.

### 1.2 New disclosure section: `Mesh adequacy of the reference field`

The mesh defect was **never disclosed anywhere in the paper's main text**. It is now a
subsection of the experimental setup: spacing, the adequacy criterion, the fact that no
released episode passes it, the paired regeneration, and the explicitly regime-dependent
effect (sparse episodes improve up to −77 %; crowded 15–30-component episodes barely move,
which is itself the evidence that the crowded-regime error is not a rendering artifact).

### 1.3 The raydrop "specular" model does not exist

`wavesim/raydrop_slope.py`'s specular branch is **geometrically unreachable** at this
sensor's incidence angles. Closed form: `alignment = -1 + 2(d·n)²`, so the 5° gate is
exactly "incidence within 2.5° of the surface normal", against a beam fan of 78–90° from
vertical and wave slopes ≤ 7°. Numerically: 0 firings in 200 000 rays per sea state, max
alignment −0.39 against a +0.996 threshold, even at an unphysical 21.8° slope. The code
always takes its unweighted-Bernoulli fallback (`raydrop_slope.py:115-120`), which
`scripts/d3_s2b_true_f.py:4-7` had already noted independently and which never reached the
paper.

So the paper's "diffuse vs specular" contrast is really **geometry-weighted vs unweighted
return sampling**. Corrected in 7 places (abstract, highlights, methods, results, figure
caption, discussion, conclusion). The *ranking* survives — the two arms are genuinely
different sampling patterns — but the mechanism ("specular backscatter preferentially
preserves wave-crest points") is deleted rather than replaced, because there is no crest
preference in an unweighted subsample. Where ETKF gains more than WaveSplat, the paper now
says so as an observation and explicitly declines to attach a mechanism.

Detail: `tmp/RAYDROP_SPECULAR_2026-07-24.md`.

### 1.4 `min_dk` caveat, and the five "resolution limit" claims

The `Initialization resolution` limitation entry is replaced with the full disclosure: the
constant, the census (126/195 episodes contain a closer true pair; 100 % of the 15/20/30-comp
ones), the pre-registered sweep with its control, and the honest read-out — relaxing it wins
5.2 % of the crowded regime's error on 5/5 seeds but *loses* on 0/5 in the mid-density
regime. So the wall survives; only "no method can" phrasing is optimistic by a few percent.
Five nearby claims that read as method-independent now point at that entry.

### 1.5 Related work: Fucile et al. (2018)

Added, with a delta on three verified axes and a deliberate refusal to assert four other
things that could not be checked without the full text. **Two errors in the original brief
were caught and corrected before they reached the paper**: LEPrE is "Linear Estimator of
Prediction *Error*", and Fucile/Bulian/Lugni are **not** part of the stereo/WASS line (his
companion sensing work is X-band radar; the paper's 38 references contain no WASS citation).
Independently confirmed via ScienceDirect. Also recorded: Fucile's *"Predictability Region"*
pre-empts the "predictable zone" terminology intended for the encounter-forecast paper.

Detail: `tmp/SCOPE_AND_FUCILE_2026-07-24.md`. Note that finding's by-product — **the paper
contains no CRB analysis at all** (0 hits for CRB/Cramér/Fisher), so the planned "CRB scope
statements in 5 places" does not apply to this paper; the drafted block belongs in
`paper/encounter_forecast/`.

### 1.6 WavePrior mechanism narrative

`docs/COMPARE_wavesplat_vs_waveprior_2026-07-09.md` §2 rewritten from banner to body: the
"learning compensates for bound waves" story is refuted three independent ways (both arms
ran `bound_wave=False`; `prior_net_v3` trained on linear JONSWAP only; bound waves proved
unlearnable for this observation class by Isserlis 8 hours after that document was
committed). Rankings kept, mechanism marked **unexplained**, and the surviving candidate
(free-wave spectral shape rather than bound waves) written down together with the experiment
that would kill it.

---

## 2. Code

All four changes ship with their previous behaviour as the default, following the repo's
existing `MIN_DK` / `GROW_RANDOM_PHI` pattern, so nothing moved silently.

| change | file | default |
|---|---|---|
| `max_steepness` was documented and set in YAML but **never read** — all 3 `clamp_steepness()` call sites used the function default | `sequential_update.py` | unchanged (YAML value equals the old default) |
| `SCALE_INVARIANT_GROW` — growth trigger compares absolute metres; normalise by observed signal RMS instead | `sequential_update.py` | `False` |
| `SCALE_INVARIANT_L1` — L1 penalty adds [m] to an [m²] loss; multiply by observation RMS | `sequential_update.py` | `False` |
| `PEAK_SELECT_MODE` — see below | `initializer.py` | `'static_suppress'` |

**On `min_dk`, the diagnosis moved.** The greedy peak search fits and subtracts each
component from the residual, but selects the next peak from a power map computed **once**
from the original observations, using a `MIN_DK`-radius box as a stand-in for recomputing it.
That is why the constant cannot simply be lowered: with a small radius the search re-picks a
component that is still present in the stale map — exactly the duplicate-peak failure the
sweep measured in the mid-density regime. So the principled repair is not an adaptive radius
(guessing a better separation scale) but `PEAK_SELECT_MODE='recompute_residual'`, which
rebuilds the map from the residual each step so the radius stops being load-bearing.
Implemented; measured in the flag sweep.

---

## 3. Measurements

All on `data_finemesh_2026-07-23/`, median over 5 estimator seeds, code confirmed
behaviour-preserving (`control_behaviour_preserved.py`: 6/6 bit-identical to the pre-edit
recorded runs).

### 3.0 100-episode headline (feeds the abstract)

WS-P1 over 100 episodes, median-of-5, finemesh: **nRMSE 0.199** (was 0.223), VR 0.833
(was 0.661), ρ 0.551 (was 0.446), SS **+0.171** (was +0.034), Dir-R² −0.025, 11.0 ms/frame.
Every panel metric improved; the "only method with positive Skill Score" claim strengthens
(SS +0.034 → +0.171). By complexity: 1c 0.126 / 2c 0.095 / 5c 0.190 / 15c 0.282 / 30c 0.288.
The matching 100-ep tuned ETKF and GPR are queued on the CPU chain (they became 100× slower
under GPU-chain contention, so they run after it); the abstract's ETKF headline and the
five-metric / by-BF baseline columns are finalized once those land. RMSE_unobs is not
recomputed (needs a separate obs-mask pass); the panel refresh is scoped to nRMSE/VR/ρ/SS.

### 3.1 Table 1 — 12 representative episodes

| | WS-P0 | WS-P1 | ETKF (as-published) |
|---|---|---|---|
| aggregate | 0.205 | **0.182** | 0.438 |

WS-P1 wins 12/12 vs as-published ETKF; P1 vs P0 +11.2 %. Mean per-episode seed sd 0.0365 ⟹
noise floor on the 12-episode mean 0.0105. The paper's 12-episode WS number was 0.211
(single seed 42); the median-of-5 value is 0.182, i.e. seed 42 sat on the unfavourable side
here. The ETKF column is the *as-published default*; the fair comparison uses the hold-out
retune (§3.4), still running.

### 3.2 Table 2 — ablation (the headline correction; §1 above)

Full 0.184 / A1 0.175 / A2 0.174 / v2-init 0.088. Paired over 30 runs: A1−Full $-0.007$
($p=0.77$), A2−Full $+0.001$ ($p=0.94$), v2-init−Full $-0.090$ ($p<10^{-4}$). The paper's
"A2 degrades 14.5 %, 2.3× on BF3 2comp slope" does not reproduce (that episode: 1.01×).

### 3.3 Table 4 — raydrop (WS arm)

Geometry-weighted 0.195 → unweighted 0.181, −7.4 % (unweighted better on 6/10). The paper's
"slope improves WS by 9.1 %" was measured with the mislabelled model on the old mesh; the
sign of the aggregate is now the other way, though within the regime scatter. ETKF raydrop
arm on the fine mesh not yet run.

### 3.4 ETKF diagnosis + retune (agent, `tmp/ETKF_FINEMESH_2026-07-24.md`)

The "ETKF degrades on the fine mesh" premise is **not supported**: aggregate 0.4155 → 0.4324,
$t(11)=+0.91$, n.s. The two resolvable per-episode swings (BF2 1comp −22 %, BF3 1comp +29 %)
are an inverse-crime artifact — the shipped raycast mesh (40×30) was numerically the ETKF's
own analysis grid, so renderer and baseline shared an operator (mean |β diff| 0.024). ETKF
carries ρ≈0 on **both** meshes and at every ceiling configuration tested (best ρ 0.07, best
nRMSE = the zero predictor), so it is structurally incapable of reconstruction here — which
is the paper's own thesis. **LETKF, the strong baseline, is mesh-invariant: Δ = −0.0006
(t = −1.56).** So the headline margin survives regeneration. Separately, the paper's ETKF
tuning was scored on the episodes it reports (hold-out defect); the retune fixes that with a
seed-1-tune / seed-0-report split.

### 3.5 Regeneration gap (agent, `tmp/REGEN_GAP_2026-07-24.md`)

56 missing `fixedT` episodes regenerated (paired, additive), plus the first mesh-adequacy
census of BF4_20comp / BF6 / BF7. **BF6/BF7 are 7/8 already ADEQUATE at the shipped mesh**
(longest-wave sea states), so the paper's BF6–7 paragraph was not compromised by the mesh
artifact — unlike the BF2–BF5 tables. BF4_20comp is DEGRADED (18.7 % var-under), no paper
dependency.

### 3.6 Flag sweep (tasks 9-11) — no adoption; `tmp/rerun_2026-07-24/FLAG_SWEEP_VERDICT.md`

Pre-registered rule (≥4/6 same direction AND mean move > seed sd 0.031): **no flag clears the
bar.** `recompute_residual` (the proposed principled `min_dk` fix) actively *hurts*
(+0.0153, worst on BF3_5comp 0.170→0.242) — the mid-density duplicate-peak problem is not
solved by recomputing the map either, so `min_dk` is a genuine regime-dependent hard trade
and the paper's hedged caveat is correct. `deterministic` gives CV→0 but a within-noise,
regime-split mean — a reproducibility/accuracy trade, not adopted. All four code changes stay
default-off.

### 3.7 task13 — the two-estimator discrepancy resolved (MEASURED, paired old↔fine)

**Initialization nRMSE improves −54% on the adequate mesh (up to −94% on sparse seas); tracked
nRMSE improves only −12%,** and on the two calmest episodes tracking turns a near-exact init
(0.009) into 0.08. So the mesh gain lands at initialization and the `w_phi` tracker (frozen
wavenumbers + Adam amplitude decay) gives most of it back. This resolves the pilot puzzle
("tracker didn't respond to the mesh change") and explains why the standalone greedy+joint
estimator moved −77% while the tracker aggregate barely moved. Quantified into the tracking
limitation. Suggests the amplitude-decay fix is higher-value than previously weighted.

### 3.8 Hyperparameter re-sweep (task 3) — reassuring null

lr / λ₁ / τ_prune / τ_grow swept ±one step on 4 regimes (sparse/tuned-on/crowded/worst),
5 seeds. **No knob uniformly improves;** all moves are small, regime-dependent, sign-mixed,
and the defaults avoid the bad corners (lr=0.03 costs sparse +0.071). So the production
hyperparameters — tuned on BF3_5comp — are not badly mis-set for other regimes, giving the
"low sensitivity across BF2–5" claim genuine multi-regime finemesh support (the methodology
audit had flagged it as verified only inside the tuning regime).

### 3.9 Production-init ablation — A2 null confirmed, A1 cleaner

Repeating A1/A2 with the production initializer (`v2_konly`): **A2 (confidence) null survives**
(Δ+0.005, p=0.61) — dropped the "not verified" hedge from the paper. A1 (dispersion) under
production init is neutral-to-slightly-worse-when-removed (Δ+0.044, p=0.072), a cleaner "keep
the constraint" story than the v1-init overfitting picture.

### 3.10 Final headline and baselines (all measured)

| metric (100 ep, finemesh) | WaveSplat | ETKF (tuned) | GPR-1 | old WS / ETKF |
|---|---|---|---|---|
| nRMSE | **0.199** | 0.332 | 0.263 | 0.223 / 0.319 |
| VR | 0.833 | 1.048 | 0.062 | 0.661 / 0.898 |
| ρ | 0.551 | 0.106 | 0.194 | 0.446 / 0.109 |
| SS | **+0.171** | −0.844 | −0.164 | +0.034 / −0.702 |

**Headline: WS 0.199 vs ETKF 0.332 = 40.1% improvement (was 30.1%); the margin widened.**
12-ep: WS 0.182 vs tuned ETKF 0.377 = 51.7%, 12/12 wins. WaveSplat now wins every sea-state
band including BF2 (0.301→0.145, the sub-Nyquist episodes were the worst-hit). ETKF and GPR
are near-mesh-invariant in nRMSE (0.319→0.332, 0.261→0.263); ETKF's VR rises toward unity
while ρ stays ≈0.1 (degenerate, as the agent found); GPR stays a near-zero predictor.

### 3.11 Paper tables updated vs. carried over

**Refreshed on the regenerated data (median-of-5):** abstract, highlights, conclusion,
five-metric panel (nRMSE/VR/ρ/SS), by-sea-state table, main 12-episode comparison
(tab:main + fig1 text), ablation (tab:ablation + fig4), raydrop (tab:raydrop). Headline
numbers consistent across all.

**Carried over with an explicit provenance note** (Section "Comparison with stronger
baselines"): the LETKF/GPR-disp stronger-baseline tables, the paired-run Wilcoxon numbers,
and the spectral-accuracy tables retain the original coarse-mesh single-seed evaluation. The
note states the boundary and gives the two facts that bound its impact (LETKF mesh-invariant
Δ=−0.0006; ETKF near-mesh-invariant in nRMSE). Full multi-method median-of-5 refresh of those
tables is the remaining deferred item.

**RMSE_unobs** (five-metric col 5) carried from the original evaluation (needs an obs-mask
pass); flagged in the caption. **N_a** column in tab:main reported as per-seed median with a
seed-bimodality footnote (the adaptive-N count is itself seed-dependent on calm seas).

---

## 4. Discipline notes

- Every per-episode number in this session is a median over $k=5$ estimator seeds reported
  with its spread, per `docs/REPORTING_CONVENTION_median_of_k_2026-07-23.md`.
- Two claims were downgraded during the session rather than after it: the ETKF "systematic
  finemesh regression" (aggregate is 0.4361 → 0.4381, i.e. +0.5 %, five episodes better and
  seven worse — the premise as stated was not supported), and the single-seed reading of the
  BF2 case (§1.1).
- The independent-verification rule was load-bearing twice: the Fucile brief contained two
  reviewer-fatal errors, and the initializer's `init_nrmse` is an **in-sample** number on the
  last initialization frame, so the large init-vs-tracked gap must not be read as a tracking
  failure without the paired old-vs-fine comparison that isolates it.
