# 8-hour autonomous session — summary (2026-06-28)

All deliverables committed/pushed; all headline numbers cross-checked (consistent).

## What was produced
1. **Main result (paper centerpiece).** Deployable recursive estimator (beamform →
   iterative RELAX add/prune → off-grid refine → warm-start), observations only,
   period-scaled window, 30 episodes × 5 frames, full metric panel:
   **overall nRMSE 0.119 (Pearson 0.84) vs ST-DF-GP 0.205 (−42%), oracle 0.037.**
   Beats SOTA on every sea-state; real reconstruction (mean VR 0.79, not over-smoothed).
   `main_results.json` (1-pass 0.130), `main_results_iter.json` (3-pass 0.119),
   `fig_main_results.png`.
2. **Forecast.** Components propagated by dispersion → skill flat over 0–8 s
   (simple: P 0.95–0.98; complex BF4-30c 0.88; BF5-30c 0.58–0.64) while persistence
   collapses after ~1 s. `forecast.json`, `forecast_complex.json`, figs.
3. **Generalization + active-sensing evidence.** Slope-raydrop (22 valid episodes):
   0.061 vs uniform 0.124 — generalizes *and* improves; verified identical field, and
   CRB confirms slope sampling ~6–30× more informative → real-sensor evidence that
   sampling geometry sets identifiability. `slope_generalization.json`.
4. **Oracle-gap reduction.** Iterative (3-pass) RELAX recovers masked components
   (0.130 → 0.119; BF4-15c 0.131 → 0.102). `relax_iterative.json`.
5. **Non-stationarity correction.** Sea-state coherence ~20 min (literature) ≫ the
   12–20 s window → stationarity-safe; my earlier ~10–20 s assumption was wrong.
6. **Paper draft.** Scaffold + full prose: `docs/paper_draft_v2.md`,
   `docs/paper_draft_full.md`. Honest framing (limits → information → beat-by-design →
   deployable estimator + calibrated UQ). Towing-tank plan: `snu_towing_tank_*`.
7. **Visual proof.** Reconstruction GIFs (nowcast→forecast), iterative estimator on
   complex seas, `gifs/*_iterative.gif`.
8. **Fact-check + consistency.** `FACT_CHECK_2026-06-27.md` (incl. non-stationarity
   correction); final cross-check passed (GP 0.205 self-consistent; 3-pass beats GP on
   every sea-state; VR 0.79 = real amplitude).

## Honest status (carry to the paper)
- oracle (0.037) = information bound; deployable (0.119) has headroom = component-id.
- period-scaled window vs GP's 1 s: advantage = time accumulation (operationally free,
  stationarity-safe, GP cannot exploit it). State this.
- simulation only; multi-directional → ocean basin; real data = tank (future).
- **DATA ISSUE:** 8 benchmark slope `.h5` corrupt (BF5_15c s0–3, BF5_30c s0/s2/s3/s4)
  + many non-benchmark — invalid HDF5; needs regeneration (untouched by me).

## Open / next (need user or real data)
1. Give DPS/ST-DF-GP the same multi-frame/time budget for a fully symmetric table.
2. Multi-directional ocean-basin validation; SNU towing-tank campaign (1-D/temporal).
3. Close the remaining oracle gap (more component-id work; prior-regularized RELAX).
4. Regenerate corrupt slope episodes.

## Commits (this session)
5db06ab (non-stat correction) … f0bed28 (Phase1) … 24bc535 (Phase2+3) … f189bbe
(full draft) … 4afde27 (slope) … 7af4fe8 (iter RELAX) … b8a65dc (complex forecast) …
e01ce8b (final main 0.119) … 67f2eab (iter GIFs).
