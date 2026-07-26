# PREREG-mini — bf_peak_ratio on the 30 real windows (debate crux B pre-check)

Written 2026-07-26, BEFORE computing. Trigger: user question "deploy한 것은 학습 기반
아니냐" + debate crux B (P3: "one-hour computation that decides crux B; pre-register
with the prediction written first"). This is the routing-signal pre-check only; the full
blind end-to-end auto-on-real run remains sprint item ② and is NOT replaced by this.

## Frozen procedure
For each of the 30 wp_real windows (5 records × 6 windows), compute
`bf_peak_ratio` = power[2nd peak]/power[1st peak] via the SAME code path as the shipped
guard (`initializer._beamform_coarse` + MIN_DK box suppression, kx,ky ∈ (−4,4), n=81,
DC zeroed), on the SAME observation set the wp_real arms saw (`cg.scene_real` P, z).
Guard rule as shipped: routed-learned AND ratio < 0.05 → veto to analytic.
Router clause 1 on real is already measured: proposals are dense (n_eff ≈ 24 ≥ 8), so
clause 1 says "learned" on essentially every real window; the guard therefore decides.

## Predictions (written first)
- P2 (R2, falsifiable prediction): real records are swell-dominated → beamform map has
  ONE dominant peak → **most real windows read ratio < 0.05 → guard vetoes to analytic**.
- P1 (R2): does not predict; demands the distribution be written down before running.
- Moderator's numeric expectation following P2: ≥ 20 of 27 normal windows veto.

## Interpretation cells (frozen)
- **A: ≥20/27 normal windows < 0.05 (veto)** → the single shipped `auto` config lands on
  the analytic arm on real seas FOR ITS OWN PRINCIPled REASON (same clause that fixes
  BF2_1comp). "One policy, two arenas" survives this check; full blind run still required.
- **B: 7–19/27 veto** → routing on real is inconsistent → policy is arena-dependent in
  practice; the abstract must scope the unified claim to synthetic.
- **C: ≤6/27 veto** → P1 objection 2 CONFIRMED: auto would route learned (the measured
  losing arm) on real seas; report "two policies" in those words.
The 3 collapse windows are recorded separately (fallback clause is a separate mechanism).
