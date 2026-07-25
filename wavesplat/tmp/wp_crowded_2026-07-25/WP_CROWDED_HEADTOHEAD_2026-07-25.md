# WavePrior vs phase-fixed WaveSplat — CROWDED-regime head-to-head (2026-07-25)

Pre-registered, at-scale, paired confirmation of the G1 A1 pilot. Frozen design:
`tmp/wp_crowded_2026-07-25/PREREG.md`. Raw runs: `ws_runs.jsonl` (260, 0 err),
`cond_runs.jsonl` (265, 5 err — all on the flagged extra). Analysis: `analyze.py` →
`results.json`. Both arms score nRMSE = RMSE/Hs on the paper's GT/obs at the same 3 nowcast
frames (parity verified §1 of PREREG). WS arm phase-fixed (`TRACK_PHASE_PREDICT=False`, set per
worker; shipped default untouched). Median-of-5 seeds per episode both arms.

---

## 1. PRE-REGISTERED VERDICT — **PRIMARY PASS**

Primary endpoint: paired per-episode nRMSE/Hs, cond-diffusion+polish nowcast vs
`ws_best = min(median WS_init, median WS_tracked)`, on all **40** crowded finemesh episodes
(BF4/BF5 × 15/30comp × seed0–4 × uniform/slope). One-sided Wilcoxon (H1: cond < ws_best), α=0.05.

| quantity | value |
|---|---|
| cond mean (nRMSE/Hs) | **0.1245** |
| ws_best mean | 0.2252 |
| median paired Δ (cond − ws_best) | **−0.1052** (−45%) |
| mean paired Δ | −0.1007 |
| wins (cond better) | **39 / 40** |
| rank-biserial effect size | **+0.998** |
| one-sided Wilcoxon p | **1.82 × 10⁻¹²** |
| **P1 (p<0.05)** | ✅ True |
| **P2 (median Δ<0)** | ✅ True |
| **DECISION** | **PASS** |

Both frozen conditions hold → **PRIMARY PASS**. The single crowded episode where cond does not
win is `BF5_30comp_seed2_uniform` (cond 0.179 vs ws_best 0.165, Δ +0.014) — and there ws_best is
an atypically good WS *init* (0.165); cond still beats WS *tracked* (0.225) on that episode.

**Secondary (context, not the decision variable):** cond vs WS **tracked** (the paper's shipped
product): cond 0.1245 vs 0.2525, median Δ **−0.1376** (−54%), **40/40 wins**, p = 9.1×10⁻¹³.

The pilot effect (2–3× on n=7) reproduces at scale: cond roughly **halves** the crowded-regime
error the analytic path leaves on the table, on the same observations and same GT.

---

## 2. By-regime splits (secondary, cond vs ws_best)

| split | n | cond | ws_best | median Δ | wins | rank-biserial | p (1-sided) |
|---|---|---|---|---|---|---|---|
| 15comp | 20 | 0.091 | 0.216 | −0.130 | 20/20 | +1.00 | 9.5e-07 |
| 30comp | 20 | 0.158 | 0.234 | −0.079 | 19/20 | +0.99 | 1.9e-06 |
| BF4 | 20 | 0.135 | 0.239 | −0.112 | 20/20 | +1.00 | 9.5e-07 |
| BF5 | 20 | 0.114 | 0.212 | −0.102 | 19/20 | +0.99 | 1.9e-06 |
| uniform | 20 | 0.153 | 0.239 | −0.087 | 19/20 | +0.99 | 1.9e-06 |
| slope | 20 | 0.096 | 0.211 | −0.119 | 20/20 | +1.00 | 9.5e-07 |

The cond advantage is largest at 15comp (medΔ −0.130) and narrows but survives at 30comp
(medΔ −0.079); it holds across BF, and across uniform/slope. The 30comp margin is thinner because
that is the densest regime AND because WS_init is occasionally strong there (`ws_best` picks it).

---

## 3. SPARSE CONTROL (1/2/5comp, n=12) — expected WS-wins, **confirmed**

| quantity | value |
|---|---|
| cond mean | 0.0837 |
| ws_best mean | 0.0137 |
| median Δ (cond − ws_best) | **+0.0320** |
| cond wins | **0 / 12** |
| one-sided (cond<ws_best) p | 1.00 |

WS wins **all 12** sparse episodes. Per PREREG this is a **control, not a failure**: the niche
claim is regime-specific. The learned discrete set-proposal is *worse* exactly where the analytic
path is near-exact. It is catastrophic on the two BF2_1comp episodes (cond 0.27–0.32 ≈ the
zero-predictor nRMSE floor, vs WS 0.026–0.038) — a single clean swell is the one thing the
32-comp-trained set-proposal reconstructs poorly, while the analytic init resolves it exactly.
This clean regime crossover (cond wins crowded, loses sparse) is the strongest single piece of
evidence that the effect is genuinely regime-specific and not a metric or protocol artifact.

---

## 4. Full paired table (40 crowded episodes)

nRMSE/Hs, median-of-5 (sd). ws_best = min(WSinit, WStrk). Δ = cond − ws_best (negative = cond wins).

| episode | nc | cond (sd) | WSinit (sd) | WStrk (sd) | ws_best | Δbest |
|---|---|---|---|---|---|---|
| BF4_15comp_s0_slope | 15 | 0.059 (.002) | 0.201 (.054) | 0.253 (.030) | 0.201 | −0.142 |
| BF4_15comp_s1_slope | 15 | 0.079 (.006) | 0.221 (.020) | 0.242 (.023) | 0.221 | −0.142 |
| BF4_15comp_s2_slope | 15 | 0.061 (.004) | 0.173 (.024) | 0.223 (.030) | 0.173 | −0.112 |
| BF4_15comp_s3_slope | 15 | 0.066 (.006) | 0.192 (.036) | 0.221 (.038) | 0.192 | −0.126 |
| BF4_15comp_s4_slope | 15 | 0.061 (.007) | 0.198 (.022) | 0.250 (.033) | 0.198 | −0.137 |
| BF4_15comp_s0_uniform | 15 | 0.090 (.014) | 0.317 (.028) | 0.274 (.011) | 0.274 | −0.184 |
| BF4_15comp_s1_uniform | 15 | 0.160 (.006) | 0.233 (.019) | 0.222 (.057) | 0.222 | −0.062 |
| BF4_15comp_s2_uniform | 15 | 0.133 (.007) | 0.245 (.029) | 0.284 (.040) | 0.245 | −0.112 |
| BF4_15comp_s3_uniform | 15 | 0.125 (.011) | 0.272 (.013) | 0.295 (.019) | 0.272 | −0.147 |
| BF4_15comp_s4_uniform | 15 | 0.124 (.010) | 0.246 (.011) | 0.282 (.010) | 0.246 | −0.121 |
| BF5_15comp_s0_slope | 15 | 0.069 (.008) | 0.204 (.009) | 0.246 (.009) | 0.204 | −0.134 |
| BF5_15comp_s1_slope | 15 | 0.054 (.003) | 0.165 (.005) | 0.171 (.011) | 0.165 | −0.111 |
| BF5_15comp_s2_slope | 15 | 0.059 (.003) | 0.194 (.028) | 0.197 (.037) | 0.194 | −0.135 |
| BF5_15comp_s3_slope | 15 | 0.056 (.004) | 0.200 (.004) | 0.213 (.005) | 0.200 | −0.144 |
| BF5_15comp_s4_slope | 15 | 0.063 (.006) | 0.239 (.023) | 0.248 (.018) | 0.239 | −0.176 |
| BF5_15comp_s0_uniform | 15 | 0.111 (.008) | 0.224 (.008) | 0.214 (.004) | 0.214 | −0.103 |
| BF5_15comp_s1_uniform | 15 | 0.103 (.014) | 0.206 (.027) | 0.181 (.056) | 0.181 | −0.078 |
| BF5_15comp_s2_uniform | 15 | 0.110 (.008) | 0.225 (.037) | 0.250 (.029) | 0.225 | −0.115 |
| BF5_15comp_s3_uniform | 15 | 0.105 (.008) | 0.198 (.010) | 0.243 (.019) | 0.198 | −0.094 |
| BF5_15comp_s4_uniform | 15 | 0.127 (.010) | 0.260 (.022) | 0.263 (.011) | 0.260 | −0.133 |
| BF4_30comp_s0_slope | 30 | 0.124 (.004) | 0.273 (.009) | 0.289 (.010) | 0.273 | −0.149 |
| BF4_30comp_s1_slope | 30 | 0.142 (.009) | 0.269 (.014) | 0.283 (.019) | 0.269 | −0.127 |
| BF4_30comp_s2_slope | 30 | 0.137 (.007) | 0.237 (.012) | 0.295 (.010) | 0.237 | −0.100 |
| BF4_30comp_s3_slope | 30 | 0.146 (.009) | 0.237 (.020) | 0.318 (.029) | 0.237 | −0.091 |
| BF4_30comp_s4_slope | 30 | 0.144 (.003) | 0.190 (.017) | 0.226 (.023) | 0.190 | −0.046 |
| BF4_30comp_s0_uniform | 30 | 0.230 (.007) | 0.304 (.015) | 0.312 (.027) | 0.304 | −0.074 |
| BF4_30comp_s1_uniform | 30 | 0.209 (.009) | 0.280 (.016) | 0.291 (.014) | 0.280 | −0.071 |
| BF4_30comp_s2_uniform | 30 | 0.224 (.006) | 0.239 (.012) | 0.299 (.006) | 0.239 | −0.015 |
| BF4_30comp_s3_uniform | 30 | 0.205 (.013) | 0.260 (.010) | 0.318 (.016) | 0.260 | −0.056 |
| BF4_30comp_s4_uniform | 30 | 0.189 (.015) | 0.238 (.011) | 0.262 (.007) | 0.238 | −0.048 |
| BF5_30comp_s0_slope | 30 | 0.115 (.011) | 0.203 (.015) | 0.224 (.019) | 0.203 | −0.088 |
| BF5_30comp_s1_slope | 30 | 0.125 (.003) | 0.242 (.002) | 0.233 (.003) | 0.233 | −0.108 |
| BF5_30comp_s2_slope | 30 | 0.123 (.004) | 0.174 (.006) | 0.245 (.012) | 0.174 | −0.051 |
| BF5_30comp_s3_slope | 30 | 0.123 (.004) | 0.231 (.029) | 0.267 (.008) | 0.231 | −0.107 |
| BF5_30comp_s4_slope | 30 | 0.121 (.007) | 0.191 (.009) | 0.203 (.010) | 0.191 | −0.070 |
| BF5_30comp_s0_uniform | 30 | 0.167 (.004) | 0.251 (.031) | 0.273 (.031) | 0.251 | −0.084 |
| BF5_30comp_s1_uniform | 30 | 0.160 (.007) | 0.255 (.017) | 0.250 (.006) | 0.250 | −0.090 |
| **BF5_30comp_s2_uniform** | 30 | 0.179 (.011) | 0.165 (.038) | 0.225 (.033) | 0.165 | **+0.014** |
| BF5_30comp_s3_uniform | 30 | 0.156 (.015) | 0.258 (.024) | 0.307 (.030) | 0.258 | −0.102 |
| BF5_30comp_s4_uniform | 30 | 0.147 (.004) | 0.200 (.007) | 0.207 (.008) | 0.200 | −0.053 |

Sparse control (12), same columns:

| episode | nc | cond (sd) | WSinit (sd) | WStrk (sd) | ws_best | Δbest |
|---|---|---|---|---|---|---|
| BF2_1comp_s0_slope | 1 | 0.320 (.005) | 0.030 (.005) | 0.026 (.005) | 0.026 | +0.294 |
| BF2_1comp_s0_uniform | 1 | 0.270 (.016) | 0.038 (.013) | 0.047 (.018) | 0.038 | +0.232 |
| BF3_1comp_s0_uniform | 1 | 0.020 (.003) | 0.010 (.003) | 0.009 (.002) | 0.009 | +0.011 |
| BF4_1comp_s0_uniform | 1 | 0.029 (.011) | 0.006 (.001) | 0.006 (.005) | 0.006 | +0.024 |
| BF3_2comp_s0_slope | 2 | 0.031 (.010) | 0.009 (.002) | 0.009 (.004) | 0.009 | +0.022 |
| BF3_2comp_s0_uniform | 2 | 0.040 (.012) | 0.009 (.003) | 0.011 (.005) | 0.009 | +0.031 |
| BF3_5comp_s0_slope | 5 | 0.051 (.006) | 0.012 (.002) | 0.019 (.003) | 0.012 | +0.039 |
| BF3_5comp_s0_uniform | 5 | 0.072 (.007) | 0.014 (.006) | 0.018 (.007) | 0.014 | +0.058 |
| BF3_5comp_s1_uniform | 5 | 0.046 (.007) | 0.013 (.001) | 0.020 (.005) | 0.013 | +0.033 |
| BF4_5comp_s0_slope | 5 | 0.039 (.005) | 0.009 (.072) | 0.015 (.091) | 0.009 | +0.030 |
| BF4_5comp_s0_uniform | 5 | 0.046 (.011) | 0.011 (.098) | 0.018 (.100) | 0.011 | +0.035 |
| BF4_5comp_s1_uniform | 5 | 0.040 (.010) | 0.008 (.046) | 0.039 (.047) | 0.008 | +0.031 |

**Flagged extra (BF4_20comp_seed7, DEGRADED set): NOT EVALUABLE.** All 5 cond runs raised
`IndexError('index 157 is out of bounds for axis 0 with size 26')` — a mesh/frame shape mismatch,
consistent with the DEGRADED flag on `data_finemesh_extras_2026-07-24`. Excluded (no WS run
either); it was never in the primary set or any test.

---

## 5. Stochasticity accounting

Both arms are stochastic and were run median-of-5. Mean per-episode sd over the 40 crowded:
- WS_init 0.0186, WS_track 0.0202, **cond 0.0075**.
The cond arm is the *tightest* — diffusion sampling variance (across seeds 0–4, each a 4-sample
BMA + polish) is smaller than WS's random-phase-init variance. The seed variance is far below the
paired Δ (median −0.105), so no single-seed anti-conservatism: the win is not a seed-selection
artifact (median-of-5 both arms, no oracle best-of-k, no GT used to select samples). The sparse
control exposes the known bimodal WS_init spread (BF4_5comp WS_init sd 0.05–0.10) — median-of-5
absorbs it and the medians are stable at ~0.01.

---

## 6. MEASURED vs ARGUED

**MEASURED (this test):**
- On crowded synthetic (in-distribution for the cond net), the learned set-proposal beats the
  phase-fixed analytic WaveSplat's *best of init-or-tracked* on 39/40 episodes, median −0.105
  nRMSE/Hs (−45%), p=1.8×10⁻¹²; and beats the analytic *tracked product* on 40/40 (−54%).
- On sparse synthetic (1/2/5comp) the analytic path wins 12/12; the learned proposal is worse,
  catastrophically so at 1comp. The regime crossover is clean.
- The crowded advantage holds across BF4/BF5, uniform/slope, and both 15comp and 30comp.

**ARGUED (supported by, but not proven by, this test):**
- That the surviving crowded "wall" is an *initialization/representation* limit of the greedy
  analytic path rather than an information-theoretic wall. The measurement is strongly consistent
  with this — a different estimator recovers ~half the residual on the *same* observations and GT,
  so the information is present in the data — but "the wall is representational" is an inference,
  not a directly measured quantity. (This does, however, empirically refute the earlier ARGUED
  framing that the crowded residual is "inherently informational"; that correction is already
  logged in `PHASEFIX_100EP_2026-07-25.md`.)
- The *mechanism* by which the learned proposal wins (better multi-peak disambiguation from the
  beamform-conditioned prior) is not dissected here; only the outcome is measured.

---

## 7. What this licenses / does NOT license

**Licenses:**
- A regime-specific, **in-distribution synthetic** claim: the learned set-proposal (WavePrior /
  cond-diffusion+polish) outperforms the analytic WaveSplat on crowded synthetic seas, decisively
  and at scale. Crowded synthetic being in-distribution for the cond net (trained on direction3
  32-comp JONSWAP) **is the niche claim**, not a confound — it is the intended operating regime.

**Does NOT license:**
- **Real-data claims.** The cond net is trained on synthetic-clean 32-comp obs; the standing
  caveat (`d3_beamform_cond_diffusion.py:12`) that synthetic-clean nets fail to transfer to real
  broadband+noisy data stands. Real-data transfer (e.g. Bergamasco stereo) is a separate future
  test and is NOT supported by this result.
- **Resurrection of the refuted γ / bank-MMSE-prior superiority claims.** hand≈v3≈retrain on that
  axis stands (`project_waveprior_promotion_2026-07-24`). This is a *different object* (discrete
  set-proposal that can represent crowded/sparse peaks, not a continuous bank prior), a *different
  endpoint* (crowded-regime field nowcast), and a *different comparison* (vs analytic WaveSplat,
  not vs a hand prior). It does not reopen that axis.
- **Any headline/paper edit or default change.** No defaults were changed; the phase-fix flag
  stays default-True (shipped); paper untouched. Flipping the default and paper propagation remain
  a user decision (see PHASEFIX_100EP report).

---

## 8. Integration recommendation (NEXT-STEP ENGINEERING — not a measured result)

The A1_WIRING_PLAN "Then" step is now well-motivated: wire the learned proposal as
`init_method='learned'` feeding its (kx,ky) atoms into `coarse_to_fine` / `initialize_from_frames`
so the phase-fixed sequential tracker runs on top. Rationale (engineering, **not measured here**):
on crowded episodes cond's nowcast (0.124) already beats WS *tracked* (0.253), so seeding the
tracker from the learned atoms should let the (now-fixed) tracker hold rather than degrade the
better init — potentially giving both a strong crowded init AND temporal coherence. This would
need its own pre-registered benchmark on the full 5-metric panel (nRMSE, VR, ρ, Skill, Hs-err) vs
WS/ETKF/GPR before any claim. Flagged as the recommended next experiment, not a finding.

---

## 9. Files written (all under repo `tmp/wp_crowded_2026-07-25/`)
- `PREREG.md` — frozen pre-registration.
- `run_ws_crowded.py`, `ws_runs.jsonl` (260 runs, 0 err), `ws_run.log` — WS arm (phase-fixed, cpu).
- `run_cond_crowded.py`, `cond_runs.jsonl` (265 runs, 5 err = flagged extra), `cond_run.log` — cond arm (gpu).
- `analyze.py`, `results.json` — paired analysis + machine-readable output.
- `WP_CROWDED_HEADTOHEAD_2026-07-25.md` — this report.

Deviations from PREREG: none material. The flagged BF4_20comp extra was not evaluable (DEGRADED
data IndexError) and is reported as such; it was never in the primary set. No sparse-control cut
was needed (total wall-clock ~44 min WS / ~31 min cond, well under 2.5 h).
