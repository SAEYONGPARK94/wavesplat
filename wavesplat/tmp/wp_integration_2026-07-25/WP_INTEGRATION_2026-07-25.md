# WavePrior STAGE-1 — learned set-proposal as tracker initializer: integration + measurement (2026-07-25)

Pre-registered (`tmp/wp_integration_2026-07-25/PREREG.md`, frozen before any full run). Additive
implementation, defaults untouched. Raw runs: `arm_runs.jsonl` (780, 0 err), `atoms_cache.jsonl`
(260, 0 err). Analysis `analyze.py` → `results.json`.

Follows the head-to-head PASS (`tmp/wp_crowded_2026-07-25/`): the cond-diffusion+polish proposal
beat the analytic `ws_best` by median −0.105 nRMSE/Hs (−45%, 39/40, p=1.8e-12) as a **nowcast of
the raw proposal**. STAGE-1 asks whether that advantage survives when the proposal is wired as
`init_method='learned'` and the shipped joint-fit / adaptive-N / phase-fixed tracker run on top.

---

## 1. PRE-REGISTERED VERDICT — **PRIMARY PASS** (retention 58.6%, just under the 60% bar)

Primary: 40 crowded finemesh episodes, per-episode median-of-5, **full-sequence tracked
avg_nRMSE/Hs**, `learned` vs `v2_konly`, one-sided Wilcoxon (H1: learned < v2_konly), α=0.05.

| quantity | value |
|---|---|
| learned mean | **0.1809** |
| v2_konly mean | 0.2525 |
| median paired Δ (learned − v2_konly) | **−0.0616** (−24%) |
| mean paired Δ | −0.0715 |
| wins (learned better) | **36 / 40** (4 losses all ties, Δ +0.001…+0.011) |
| rank-biserial | **+0.973** |
| one-sided Wilcoxon p | **5.0 × 10⁻¹¹** |
| **P1 (p<0.05)** | ✅ |
| **P2 (median Δ<0)** | ✅ |
| **DECISION** | **PASS** |
| retention vs frozen bar | **58.6%** (bar = 60%) → **retention_pass = False** |

Both frozen decision conditions hold → **PRIMARY PASS**: wiring the learned proposal into the
production tracker gives a decisive, significant crowded-regime improvement over the analytic
init under the identical phase-fixed tracker.

**Retention.** The head-to-head init-level advantage was median Δ = −0.1052. The integrated
tracked median Δ is −0.0616 = **58.6%** of it (or −0.0543 = **51.6%** measured apples-to-apples
against `v2_konly ws_best`, p=6.1e-6, 28/40). Per the frozen rule this is **PASS + retention just
under 60% = "significant but attenuated"**; §5 gives the mechanism (roughly half the raw-proposal
advantage is lost, in two identifiable places, neither fatal and both addressable in Stage-2).

The 4 "losses" are statistical ties (BF5_15c_s0_u +0.002, BF5_15c_s2_u +0.001, BF5_30c_s1_u
+0.002, BF5_30c_s4_u +0.011) — learned effectively wins-or-ties **40/40**.

**Default-path preservation (checked BEFORE eval):** `default_bit_identical.py` replayed 3 recorded
`v2_konly` jobs through the edited code — avg_nRMSE AND init_nrmse match the head-to-head
`ws_runs.jsonl` to **1e-10, 3/3**. The additive `learned` branch does not perturb the shipped path.

---

## 2. Full 5-metric panel (crowded 40, median-of-5, means)

| metric | learned | v2_konly | Δ (learned − v2) | direction |
|---|---|---|---|---|
| nRMSE/Hs ↓ | **0.181** | 0.252 | **−0.071** | learned better |
| Variance Ratio →1 | 0.730 | 0.925 | −0.194 | **v2 closer to 1** |
| Pearson ρ ↑ | **0.648** | 0.418 | **+0.230** | learned better |
| Skill Score ↑ | **+0.372** | −0.136 | **+0.508** | learned better |
| Directional R² ↑ | **+0.329** | −0.522 | **+0.851** | learned better |

The learned init reconstructs the crowded **spatial/directional structure** far better (ρ, Skill,
Dir-R²), while the analytic v2 init has higher variance ratio but **negative Skill and Dir-R²** —
i.e. v2 puts roughly the right amount of energy into the wrong pattern (wrong-phase / mis-located
components in the crowded regime), whereas learned gets the pattern right and slightly
under-energizes (VR 0.73). This is the crowded "wall" being a **representation/placement** problem,
not an energy problem — consistent with the head-to-head's ARGUED framing, now measured through
the tracker.

## 3. Splits (median Δ, wins, p — all significant)

| split | n | learned | v2 | median Δ | wins | p |
|---|---|---|---|---|---|---|
| 15comp | 20 | 0.157 | 0.239 | −0.086 | 18/20 | 4.8e-06 |
| 30comp | 20 | 0.205 | 0.266 | −0.036 | 18/20 | 4.8e-06 |
| BF4 | 20 | 0.179 | 0.272 | −0.098 | 20/20 | 9.5e-07 |
| BF5 | 20 | 0.182 | 0.233 | −0.038 | 16/20 | 5.2e-05 |
| uniform | 20 | 0.216 | 0.263 | −0.025 | 16/20 | 5.2e-05 |
| slope | 20 | 0.146 | 0.242 | −0.106 | 20/20 | 9.5e-07 |

Largest at 15comp / slope / BF4; narrowest but still significant at 30comp and BF5-uniform
(the densest + the regime where v2_init is occasionally strong). Mirrors the head-to-head shape.

## 4. `learned_konly` ablation — the win is the kx/ky LOCATIONS, not the LS w/phi

To rule out the confound "learned uses deterministic LS w/phi, v2_konly uses random w/phi":

| comparison | median Δ | wins | p |
|---|---|---|---|
| learned_konly (kx/ky only, v2_konly-style w=0.01/random φ) vs v2_konly | −0.064 | 39/40 | 3.0e-11 |
| learned_konly vs learned (LS w/phi) | +0.005 | 15/40 | 0.94 (n.s.) |

**learned_konly ≈ learned** (n.s.). Replacing the LS-fit w/phi with v2_konly's random-phase w/phi
costs nothing: the entire integrated advantage comes from the learned **component locations
(kx, ky)**, exactly the quantity frozen through the `w_phi` tracker. This is the clean ablation the
PREREG called for and it removes the w/phi confound entirely.

## 5. Init decomposition — WHERE the advantage is (partly) lost

Crowded medians, per complexity:

| stage | 15comp | 30comp |
|---|---|---|
| learned raw-init (LS w/phi, NO joint fit, @init frame) | 0.156 | 0.200 |
| learned post-jointfit `init_nrmse` | **0.144** | **0.172** |
| learned tracked `avg_nRMSE` | 0.155 | 0.200 |
| v2_konly `init_nrmse` | 0.213 | 0.238 |
| v2_konly tracked `avg_nRMSE` | 0.245 | 0.270 |
| final N (learned / v2) | 14 / 18 | 22 / 13 |

Two mechanisms account for the ~50% attenuation vs the head-to-head:

1. **Conversion to a single-frame component set.** The head-to-head nowcast averaged 4 polished
   fields (BMA) at 3 mid-sequence frames (0.09–0.16). The initializer must instead emit ONE
   discrete component set, rendered at the (early) init frame — raw-init 0.156/0.200. So the arm
   *starts* from a weaker point than the head-to-head number; the BMA variance-reduction and the
   later-frame aperture are not available at init time. **(MEASURED.)**
2. **The phase-fixed `w_phi` tracker erodes the joint-fit gain.** The joint fit *sharpens* the
   learned init (0.156→0.144, 0.200→0.172), but per-frame `w_phi` tracking against sparse
   single-frame obs drifts it back up (→0.155, →0.200 ≈ raw). The same erosion hits v2
   (0.213→0.245). Net: for the learned arm joint-fit and tracker roughly cancel; the tracker is
   the component that gives back the joint-fit improvement. **(MEASURED.)**

The advantage that *does* survive is entirely carried by the learned kx/ky (see §4): even after
tracker erosion, learned tracked (0.155/0.200) beats v2 tracked (0.245/0.270) decisively because
the analytic init never places the crowded components as well to begin with.

## 6. Sparse control (12) — regime crossover holds, but softened by the pipeline

| quantity | value |
|---|---|
| learned mean | 0.060 |
| v2_konly mean | 0.020 |
| median Δ (learned − v2) | **+0.021** |
| learned wins | 4 / 12 |
| p (learned<v2) | 0.99 |

v2_konly wins the sparse median, as pre-registered (a control, not a failure). But unlike the
head-to-head's clean cond 0/12, the **pipeline rescues** several sparse cases (learned wins 4/12,
including BF2_1comp_s0_**uniform** 0.025 vs 0.047) — the joint fit + energy-prune collapse the
net's over-proposal back to the true few components. The exception is the one **catastrophic**
learned failure: **BF2_1comp_s0_slope 0.326** (≈ zero-predictor floor) — a single clean swell on
which the 32-comp-trained net proposes n_selected=30 atoms and the prune fails to recover on the
slope variant. This is the sharp edge of the OOD behaviour and the reason a regime gate is needed.

## 7. Deliverable — naive regime-switch (52 ep, GT-free signal)

Signal = `n_selected` (components the learned proposal returns after pool_and_select), FROZEN rule
**n_selected ≥ 8 → learned, else v2_konly** (threshold from the design regimes, not tuned on nRMSE).

| policy | 52-ep mean tracked nRMSE/Hs |
|---|---|
| pure v2_konly | 0.1988 |
| pure learned | 0.1530 |
| **naive switch (n_sel≥8)** | **0.1490** |
| oracle switch (per-episode best; NOT deployable) | 0.1429 |

The naive switch **beats both pure arms** and is within 0.006 of the oracle. Signal separation:
crowded n_selected median 15.5 (min 6); sparse ≤7 **except both BF2_1comp episodes at n_selected=30**
(the net over-proposes on a single clean swell). The switch therefore mis-routes the 2 BF2_1comp to
learned (fine for uniform, the 0.326 catastrophe for slope) and mis-routes a few n_sel<8 crowded
episodes to v2 — yet still wins aggregate. **A deployable regime-switch is viable**; a better
crowding signal (or a low-N guard that vetoes learned when the residual after prune stays high)
would fix the BF2 edge.

## 8. Full paired table (40 crowded, median-of-5, tracked avg_nRMSE/Hs)

`learn_raw` = LS-w/phi render, no joint fit, @init frame; `learn_postfit` = after joint fit.

<!-- table -->
| episode | nc | learned trk (sd) | v2 trk (sd) | Δtrk | learn_raw | learn_postfit | v2_init |
|---|--|--|--|--|--|--|--|
| BF4_15c_s0_u | 15 | 0.224 (0.027) | 0.274 (0.013) | -0.050 | 0.183 | 0.218 | 0.317 |
| BF4_15c_s0_sl | 15 | 0.099 (0.008) | 0.253 (0.034) | -0.154 | 0.097 | 0.059 | 0.201 |
| BF4_15c_s1_u | 15 | 0.161 (0.016) | 0.222 (0.064) | -0.061 | 0.168 | 0.178 | 0.233 |
| BF4_15c_s1_sl | 15 | 0.128 (0.006) | 0.242 (0.025) | -0.115 | 0.103 | 0.139 | 0.221 |
| BF4_15c_s2_u | 15 | 0.163 (0.033) | 0.284 (0.045) | -0.121 | 0.150 | 0.156 | 0.245 |
| BF4_15c_s2_sl | 15 | 0.086 (0.005) | 0.223 (0.033) | -0.137 | 0.077 | 0.082 | 0.173 |
| BF4_15c_s3_u | 15 | 0.201 (0.016) | 0.295 (0.021) | -0.094 | 0.154 | 0.172 | 0.272 |
| BF4_15c_s3_sl | 15 | 0.101 (0.039) | 0.221 (0.042) | -0.120 | 0.111 | 0.102 | 0.192 |
| BF4_15c_s4_u | 15 | 0.187 (0.016) | 0.282 (0.012) | -0.095 | 0.139 | 0.148 | 0.246 |
| BF4_15c_s4_sl | 15 | 0.108 (0.018) | 0.250 (0.037) | -0.142 | 0.071 | 0.085 | 0.198 |
| BF4_30c_s0_u | 30 | 0.286 (0.032) | 0.312 (0.030) | -0.026 | 0.271 | 0.227 | 0.304 |
| BF4_30c_s0_sl | 30 | 0.147 (0.030) | 0.289 (0.012) | -0.142 | 0.200 | 0.125 | 0.273 |
| BF4_30c_s1_u | 30 | 0.238 (0.007) | 0.291 (0.015) | -0.053 | 0.224 | 0.220 | 0.280 |
| BF4_30c_s1_sl | 30 | 0.182 (0.020) | 0.283 (0.021) | -0.101 | 0.171 | 0.127 | 0.269 |
| BF4_30c_s2_u | 30 | 0.276 (0.006) | 0.299 (0.006) | -0.023 | 0.233 | 0.209 | 0.239 |
| BF4_30c_s2_sl | 30 | 0.262 (0.052) | 0.295 (0.011) | -0.032 | 0.188 | 0.187 | 0.237 |
| BF4_30c_s3_u | 30 | 0.189 (0.006) | 0.318 (0.018) | -0.128 | 0.268 | 0.178 | 0.260 |
| BF4_30c_s3_sl | 30 | 0.120 (0.008) | 0.318 (0.033) | -0.198 | 0.179 | 0.110 | 0.237 |
| BF4_30c_s4_u | 30 | 0.242 (0.009) | 0.262 (0.007) | -0.020 | 0.211 | 0.228 | 0.238 |
| BF4_30c_s4_sl | 30 | 0.189 (0.017) | 0.226 (0.026) | -0.037 | 0.168 | 0.159 | 0.190 |
| BF5_15c_s0_u | 15 | 0.216 (0.027) | 0.214 (0.005) | +0.002 | 0.207 | 0.185 | 0.224 |
| BF5_15c_s0_sl | 15 | 0.136 (0.010) | 0.246 (0.011) | -0.110 | 0.159 | 0.093 | 0.204 |
| BF5_15c_s1_u | 15 | 0.158 (0.023) | 0.181 (0.062) | -0.023 | 0.254 | 0.155 | 0.206 |
| BF5_15c_s1_sl | 15 | 0.103 (0.012) | 0.171 (0.012) | -0.068 | 0.201 | 0.098 | 0.165 |
| BF5_15c_s2_u | 15 | 0.251 (0.012) | 0.250 (0.032) | +0.001 | 0.236 | 0.252 | 0.225 |
| BF5_15c_s2_sl | 15 | 0.119 (0.077) | 0.197 (0.041) | -0.078 | 0.169 | 0.105 | 0.194 |
| BF5_15c_s3_u | 15 | 0.232 (0.018) | 0.243 (0.022) | -0.011 | 0.189 | 0.206 | 0.198 |
| BF5_15c_s3_sl | 15 | 0.151 (0.042) | 0.213 (0.005) | -0.062 | 0.128 | 0.124 | 0.200 |
| BF5_15c_s4_u | 15 | 0.196 (0.025) | 0.263 (0.013) | -0.066 | 0.220 | 0.180 | 0.260 |
| BF5_15c_s4_sl | 15 | 0.109 (0.052) | 0.248 (0.020) | -0.139 | 0.133 | 0.075 | 0.239 |
| BF5_30c_s0_u | 30 | 0.258 (0.019) | 0.273 (0.035) | -0.016 | 0.320 | 0.208 | 0.251 |
| BF5_30c_s0_sl | 30 | 0.211 (0.023) | 0.224 (0.022) | -0.013 | 0.257 | 0.165 | 0.203 |
| BF5_30c_s1_u | 30 | 0.251 (0.023) | 0.250 (0.007) | +0.002 | 0.238 | 0.224 | 0.255 |
| BF5_30c_s1_sl | 30 | 0.197 (0.019) | 0.233 (0.004) | -0.035 | 0.199 | 0.181 | 0.242 |
| BF5_30c_s2_u | 30 | 0.203 (0.008) | 0.225 (0.037) | -0.022 | 0.178 | 0.156 | 0.165 |
| BF5_30c_s2_sl | 30 | 0.184 (0.043) | 0.245 (0.014) | -0.061 | 0.194 | 0.117 | 0.174 |
| BF5_30c_s3_u | 30 | 0.161 (0.033) | 0.307 (0.034) | -0.145 | 0.215 | 0.151 | 0.258 |
| BF5_30c_s3_sl | 30 | 0.129 (0.013) | 0.267 (0.009) | -0.138 | 0.200 | 0.115 | 0.231 |
| BF5_30c_s4_u | 30 | 0.218 (0.021) | 0.207 (0.009) | +0.011 | 0.172 | 0.214 | 0.200 |
| BF5_30c_s4_sl | 30 | 0.163 (0.008) | 0.203 (0.011) | -0.041 | 0.160 | 0.154 | 0.191 |

---

## 9. MEASURED vs ARGUED

**MEASURED (this test):**
- On crowded synthetic (in-distribution for the cond net), wiring the learned proposal as
  `init_method='learned'` under the identical phase-fixed tracker beats `v2_konly` on
  **36/40** episodes (4 ties, effectively 40/40), median −0.062 nRMSE/Hs, p=5.0e-11; and beats
  `v2_konly ws_best` 28/40, median −0.054, p=6.1e-6.
- The integrated advantage is carried **entirely by the learned kx/ky component locations**:
  `learned_konly` (kx/ky only, v2-style w/φ) ≈ `learned` (LS w/φ), Δ +0.005, p=0.94.
- Panel: learned wins ρ (+0.23), Skill (+0.51), Dir-R² (+0.85); v2 has higher VR (0.92 vs 0.73)
  but negative Skill/Dir-R² on crowded.
- The joint fit sharpens the learned init (−0.02…−0.03); the `w_phi` tracker then erodes that gain
  back to ≈ raw-init. Roughly half the head-to-head advantage does not reach the tracked product,
  from (i) single-frame component-set conversion vs BMA nowcast and (ii) tracker erosion.
- Sparse control: v2 wins the median (Δ +0.021, 8/12); the pipeline rescues some over-proposals
  (learned 4/12 wins) but one catastrophic OOD failure remains (BF2_1comp_slope 0.326).
- A naive GT-free regime-switch (n_selected≥8) beats both pure arms (0.149 vs 0.153 / 0.199),
  near oracle (0.143).

**ARGUED (supported, not proven here):**
- The surviving crowded wall is a **representation/placement** limit of the greedy analytic init,
  not an information/energy limit — strongly supported (a better component set on the *same* obs,
  under the *same* tracker, recovers ρ/Skill/Dir-R² that v2 cannot; VR shows v2 has the energy but
  not the pattern), but "representational" remains an inference.
- The tracker, not the init, is now the ceiling for the learned arm (post-fit 0.144/0.172 is the
  best point; tracking gives it back). ARGUED from the decomposition; a frozen-init / lighter
  tracker test would confirm (cf. MEMORY frozen-init forecaster note) — deferred, out of scope.

**Scope (unchanged from head-to-head §7):** in-distribution synthetic only; does NOT license
real-data claims (the synthetic-clean transfer caveat stands), does NOT reopen the refuted
γ/bank-MMSE axis, does NOT authorize any default flip or paper edit. No defaults changed, no
commits, no paper edits.

---

## 10. Stage-2 e2e fine-tuning — differentiability inspection (NO training run)

Goal for Stage-2: fine-tune the cond-net so the *tracked* (or at least init-level) field loss
flows back to its weights. Tracing the current STAGE-1 graph obs→…→nRMSE:

**Differentiable building blocks that already exist (torch, on-graph):**
- `beamform()` power map (`d3_beamform_net`) — differentiable.
- `CondEncoder` + `CondSetDenoiser` forward — differentiable.
- `denorm_atoms` (linear + exp) — differentiable.
- `loglik_atoms` — differentiable (autograd is already taken through it for DPS guidance).
- `render_wave_field_torch` (`tnse`) — differentiable field rendering at obs/grid.
- A torch LS amplitude/phase solve is expressible via `torch.linalg.solve` (the current
  `fit_amplitudes_phases` is the same normal-equations solve, just in numpy).

**Gradient BLOCKERS in today's path (each with the fix):**
1. `sample()` is `@torch.no_grad` ancestral DDPM (500 stochastic steps). → deterministic DDIM /
   few-step sampler run with grad, reparameterized noise, or supervise the x̂₀-prediction directly.
2. `polish_atoms` is an inner Adam loop returning `.detach()`. → unroll a fixed few
   Gauss-Newton/Adam steps WITH grad, or implicit-function-theorem differentiation through the
   polish fixed point.
3. `pool_and_select` — numpy hard threshold + greedy dedup + argsort + cap (discrete, off-graph).
   → emit fixed-cardinality M atoms (drop selection) or a soft top-k / differentiable matching.
4. `ls_fit_wphi` — numpy `np.linalg.solve` (off-graph). → the same solve in torch is a
   differentiable linear layer.
5. The **joint-fit and per-frame tracker Adam loops** mutate `nn.Parameter` in place under
   `no_grad`; they are unrolled optimizers, not meta-differentiable. → for Stage-2 do NOT backprop
   through them: fine-tune the PROPOSAL against a proposal-level differentiable loss
   (atoms → torch-LS w/φ → `render_wave_field_torch` → nRMSE at the init/nowcast frames), then run
   the frozen downstream pipeline. Full through-tracker meta-gradients (differentiable-optimizer /
   implicit diff over the tracker) are a larger, separate effort.

**Recommended minimal Stage-2 path (differentiable end-to-end, no tracker meta-grad):**
obs → `beamform` → cond-net (few-step/DDIM sampler, grad on) → `denorm_atoms` → torch LS w/φ →
`render_wave_field_torch` → nRMSE, trained on **synthetic** (labels held-out per the standing
learn=synthetic rule). This removes blockers 1–4; blocker 5 is intentionally excluded by
fine-tuning the proposal rather than the optimizer. Since §4 shows the win is the kx/ky locations,
Stage-2 could even fine-tune with w/φ fixed (LS-solved), concentrating gradient on component
placement — the exact quantity that survives the tracker.

---

## 11. Files (all under `tmp/wp_integration_2026-07-25/`) + code

Code (additive; defaults untouched, verified bit-identical):
- `wavesplat/optimizer/learned_init.py` — NEW. Proposal (GPU) + pooling/selection + LS w/φ + the
  installable init params + raw-init diagnostic. Imported only when `init_method=='learned'`.
- `wavesplat/optimizer/sequential_update.py` — `_initialize_learned()` method + one dispatch line
  in `initialize()`; `'learned'` added to the residual/grow membership checks (no effect on v2).
- `wavesplat/run.py` — `'learned'` added to the `--init_method` CLI choices.

Experiment artifacts:
- `PREREG.md` — frozen pre-registration.
- `default_bit_identical.py` — default-path preservation control (3/3 to 1e-10).
- `precompute_atoms.py`, `atoms_cache.jsonl` (260, 0 err), `precompute.log` — GPU proposal cache.
- `run_arms.py`, `arm_runs.jsonl` (780, 0 err), `run_arms.log` — CPU tracker arms (v2_konly,
  learned, learned_konly), full panel, phase-fix per worker.
- `analyze.py`, `results.json` — paired analysis.
- `WP_INTEGRATION_2026-07-25.md` — this report.

**Deviations from PREREG:** none material. Retention landed at 58.6% (1.4 pts under the frozen
60% bar) → reported as PASS-but-attenuated with mechanism, exactly as the PREREG pre-stated for
that case. Wall-clock: precompute ~11 min (GPU) + arms ~81 min (CPU, 4 workers nice-10) ≈ 1.5 h,
well under the 3 h budget; no arm was dropped. `data/` (LFS-restored mid-run by the main session)
was never touched — all runs read `data_finemesh_2026-07-23/`.
