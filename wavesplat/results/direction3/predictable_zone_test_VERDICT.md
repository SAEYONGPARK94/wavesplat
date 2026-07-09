# VERDICT — FIM predictable zone vs geometric DPZ (make-or-break novelty test, 2026-06-29)

**Question (from novelty review):** is the predictable zone of phase-resolved sea estimation set
by Fisher-information IDENTIFIABILITY (our Dir-I claim), distinct from and better than the canonical
GEOMETRIC group-velocity DPZ (Naaijen; Lee-Nam-Liu-Yang 2024)? Run on NONLINEAR HOS data with the
deployable RELAX estimator + forward forecast. `scripts/d62_predictable_zone_test.py`.

## Result — the strong claim does NOT survive

**On realistic HOS seas the two zones are TIED and largely COINCIDE.** Pooled over 2 seas
(BF4-15c, BF5-30c) × 2 seeds × forecast horizons {2,4,6}s, 69 120 grid cells:

| predictor of where the forecast actually works | Spearman vs \|err\| | AUC (τ=0.25) | AUC (τ=0.15) |
|---|---|---|---|
| FIM σ_z/Hs (CRB, from estimated components) | +0.312 | 0.661 | 0.648 |
| GEO distance-to-advected-energy (geometric DPZ) | +0.309 | 0.660 | 0.651 |

- **IoU(FIM-zone, GEO-zone) at matched area = 0.71** → the two predictable zones are largely the
  same region. Per-horizon AUCs are tied within ±0.01 (GEO marginally better at Δt≥4s).
- Both are only **moderate** predictors (AUC ~0.66): on continuous/nonlinear seas, neither "zone"
  precisely predicts pointwise forecast error.
- **Expanded statistics (3 seas BF3-5c/BF4-15c/BF5-30c × 4 seeds, 207 360 cells) CONFIRM:** FIM AUC
  0.661 = GEO 0.661, Spearman +0.317 ≈ +0.316, **IoU 0.795** (zones coincide even more with more
  data). The tie is robust, not a small-sample effect.

## Why they coincide (principled)
The FIM's synthetic-aperture term is *itself* group-velocity transport: ∂ψ/∂k = r − c_g·k̂·t. So the
Fisher information inherits the same c_g structure as the geometric DPZ; on a continuous spectrum the
two are two views of one physics and converge.

## The discrete reference — confirms the test has power, and is even less kind to the claim
Same pipeline on a clean DISCRETE linear sea (10 components, Δt=6s). RELAX identified only **5/10**:
- actual e/Hs: **0.075 inside cone → 0.244 outside** (unmodeled energy intrudes downstream);
- GEO distance correctly tracks it (16 m → 93 m);
- **FIM σ of the estimated model stays ~0 everywhere (0.006 → 0.012) — OVER-CONFIDENT**, blind to
  the unidentified components.

→ The test *can* separate the two (so the HOS tie is real, not a power failure). And for any
imperfect/real estimator, the **FIM-of-the-estimated-model is over-optimistic**; the model-agnostic
geometric transport is the safer bound. The idealized "discrete → predictable everywhere (no cone)"
picture (old `d1_predictable_zone.py`) is exactly the discrete idealization the project keeps finding
misleading — it evaporates under (a) spectral continuity (HOS) and (b) imperfect identification.

## Implication for the paper (honest)
- **KILL** the headline "the predictable zone is Fisher-information, not geometry / CRB corrects the
  DPZ." It is false on realistic seas and backwards for deployable estimators. Do not write it.
- **SURVIVES:** (1) information-limit framing on *accuracy magnitude* (separate result, oracle 0.037);
  (2) the discrete-benchmark-bias critique — *gains* a clean new instance (the FIM-zone novelty is a
  discrete artifact); (3) the deployable amortized posterior + UQ.
- **CONSTRUCTIVE PIVOT (emerges from this negative):** a deployable predictable-zone / UQ must add a
  **model-mismatch term that grows along c_g** (unmodeled-energy transport) on top of the FIM
  parameter uncertainty — FIM alone is over-confident downstream. i.e. calibrated UQ =
  near-field FIM ⊕ far-field geometric residual-energy transport. This unifies both and is the honest,
  novel-enough UQ contribution that directly serves the amortized-posterior story.

## Caveats
- 2 seas × 2 seeds (small); horizons ≤6s; FIM uses estimated (deployable) components, not an oracle
  discrete decomposition (appropriate for the deployable question, but means "true CRB" is not tested
  on HOS — there is no discrete truth on a continuum). Conclusion is about the *deployable* zone.
