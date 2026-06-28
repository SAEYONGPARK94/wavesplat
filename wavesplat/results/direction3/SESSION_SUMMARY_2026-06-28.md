# Direction-III — learning vs the information limit: full session summary (2026-06-28)

A single question ("is learning the main result, and is the prior really mandatory?") opened
an exhaustive investigation (experiments d30–d61) into where, if anywhere, learning beats
classical methods for phase-resolved ship-LiDAR wave estimation. This is the consolidated map.

## Headline conclusion (honest, fully tested)
**The problem is information-limited everywhere (CRB).** More information (synthetic aperture =
time, observation density) is the universal lever, and an **adaptive classical estimator
exploits it best on accuracy**. **Learning has no durable accuracy advantage** — every apparent
edge is a regime/window artifact or is matched by a non-learned method:
- estimation: CRB-capped (5 learning negatives — d34/d35/d37/d38/d36);
- active sensing (free-aim): random-uniform is optimal (d42/d43);
- continuous spectra: a continuous *representation* wins, not learning — non-learned F1 ties
  the learned DPS (~0.233), and it is a SHORT-WINDOW effect that RELAX overtakes with more
  aperture (d44/d45/d47/d49/d51/d52/d54);
- nonlinearity: bound harmonics stay ~4% even at high steepness (d53).

**But deployment changes the question** (user insight): real-time control forbids long windows
and slow inference. In the deployable regime (short window + ms inference + UQ), **no classical
method gives the needed package**; a fast amortized neural posterior does (d55/d56/d57).

## The contribution (defensible, learning-central + deployment-grounded)
**A fast, physics-informed multi-task amortized neural posterior** (d56/d58/d61):
- nowcast field nRMSE **0.232** (≈ best non-learned F1 0.234; beats RELAX 0.274), **8–9 ms**
  (50–1180× faster than F1/DPS), **calibrated UQ** (±1σ 0.68, ±2σ 0.95);
- **multi-second forecast** (beats persistence at all leads, UQ grows with lead — d58);
- interpretable **dispersion-constrained spectral** head + physics-consistency (d61) to defend
  ML novelty.
The genuine novelty is the **system + analysis**: CRB information-limit theory → derive that
the deployable optimum is a fast amortized posterior with UQ → expose the discrete-benchmark
bias. Not "learning beats the limit"; "learning delivers the deployable package the limit +
real-time constraints demand."

## Paper structure (recommended; venue = Ocean Engineering / IEEE JOE)
1. CRB information-wall theory (D_eff=L+c_g·T_obs synthetic aperture; T_wall; predictable-zone).
2. Exhaustive analysis: information-limited, not algorithm-limited (the 5 negatives + levers).
3. Discrete-benchmark-bias critique: realistic continuous spectra need continuous representations.
4. Deployment reframing: real-time = short window + ms + UQ.
5. Method: fast physics-informed multi-task amortized posterior (nowcast + forecast + UQ).

## Index — verdicts (results/direction3/*_VERDICT.md)
symmetric_dps, symmetric_gp, scarcity_crossover, learned_offgrid, learned_detection,
learned_acquisition, fundamental_F1F4, hos_nonlinear, beam_steering, window_aperture,
deploy_speed, amortized_posterior.

## Index — figures (regenerate via the d4*/d46/d48 scripts; *.png is gitignored)
fig_headline_flip (realism flips the winner), fig_comparisons (4-panel),
fig_deployment (speed×accuracy), fig_forecast_uq (forecast + UQ), fig_scarcity_crossover.

## Index — key scripts
d30 symmetric DPS · d31 symmetric GP · d33 scarcity · d34 off-grid · d35 hybrid ·
d36 DPS-guidance diagnostic · d37/d38 learned detection · d39/d40/d41 active sensing ·
d42/d43 beam-steering · d44/d45/d47/d49 HOS continuous · d51 F1 · d52 F4 ·
d53 steepness · d54 window aperture · d55 deploy speed · d56/d57/d58 amortized posterior ·
d59/d60 physics-spectral · d61 multi-task posterior.

## Standing methodology (docs/VALIDATION_METHODOLOGY.md)
R1 — validation datasets must always include nonlinear (HOS) GT, not only linear discrete
synthetic (the discrete benchmark masks the limits).
