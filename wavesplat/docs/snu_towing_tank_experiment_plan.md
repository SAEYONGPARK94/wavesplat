# SNU towing-tank validation plan — LiDAR wave-field estimation

**Date:** 2026-06-27. Experiment design to validate the WaveSplat program
(CRB limits, grid-free prior, active sensing, sequential fusion) at the Seoul
National University towing tank (예인수조).

## 0. What is the "deployable sequential estimator"?
The step-3 result (sequential time fusion → complex seas 0.02–0.03) used the
**efficient estimator**, which *knows the true wavenumbers* and only fits
amplitudes/phases. That is an oracle — it proves the *information* is present once
T_obs > T_wall, but it is not deployable.

The **deployable sequential estimator** is the real algorithm that achieves those
numbers from observations alone, by *estimating the spectrum from the
time-accumulated track* [t*−T, t*]:
1. accumulate W frames of sparse LiDAR returns into one space-time set;
2. estimate the wave components (kx, ky, ω) by dispersion-constrained
   beamforming / MUSIC / atomic optimization on the fused set (now well-posed:
   with 10–15 s of data the CRB is 0.01–0.27, so the components are identifiable —
   far easier than single-window);
3. fit amplitudes/phases by LS and reconstruct/forecast the field;
4. optionally regularize with the grid-free diffusion prior + carry calibrated UQ.
Building it turns "the information is there" into a working, deployable method, and
is the natural realization of the sequential-fusion finding.

## 1. Critical constraint — a towing tank is (mostly) UNI-DIRECTIONAL
A towing tank generates waves travelling **down the tank (one direction)**. The
paper's hardest, most novel regime is **multi-directional** complex seas, which a
towing tank **cannot** produce (that needs a multi-directional/ocean basin with
segmented wavemakers, e.g. KRISO Ocean Engineering Basin). Therefore:
- The towing tank validates the **1-D / frequency-domain** aspects (dispersion,
  frequency-spectrum estimation, the information wall vs *bandwidth*, sequential
  fusion in time, active sensing along/across the tank, forecasting,
  sensor/forward-model fidelity, calibrated UQ) — the *methodology and the
  time-fusion & active-sensing levers*.
- The **multi-directional** claims stay in simulation + are flagged for a future
  ocean-basin campaign. Frame the tank work as "physical validation of the
  methodology and the temporal/aperture levers; directional generalization by
  simulation + future ocean basin."
- If SNU's wavemaker is **segmented** (can make oblique/short-crested waves) or an
  oblique-incidence setup is possible, add a limited directional-spread test (E10).

*Model scale:* tank waves are Froude-scaled (H ~ 0.05–0.3 m, T ~ 0.6–2 s vs
full-scale Hs 1.2 m / Tp 7 s). Report both model and full-scale-equivalent.

## 2. Common setup
- **LiDAR:** ship-class solid-state unit (e.g. Seyond Falcon, σ_range ≈ 2 cm) mounted
  above the tank on the carriage or a fixed gantry, looking forward/down at a grazing
  angle to mimic ship-mounted geometry; scan rate ≥ 10 Hz (match dt = 0.1 s).
- **Ground truth:** array of resistance/capacitance **wave probes** at known x (and a
  few across-tank y) positions — the reference elevation field/time-series; plus a
  probe co-located near the LiDAR footprint for direct comparison.
- **Synchronization:** common time base (trigger) between LiDAR, probes, wavemaker
  signal, and carriage encoder.
- **Repeats:** ≥ 5 repeats per condition (different wavemaker phase seeds) for UQ /
  statistics.

## 3. Experiment table
| # | Experiment | Validates (claim) | Wave condition | LiDAR / sensor setup | Ground truth | Key metric | Prio |
|---|---|---|---|---|---|---|---|
| E1 | Regular-wave sensor check | LiDAR→elevation fidelity; dispersion | Regular, 3–4 (H,T); low steepness | Static gantry, dense scan | 1 probe at footprint | RMSE_z, ω(k) match | ★★★ |
| E2 | Steepness sweep | linear vs nonlinear (Stokes) model mismatch | Regular, fixed T, steepness Hk = 0.05→0.25 | Static, dense | probe | bias vs steepness; need for Stokes2/HOS | ★★ |
| E3 | Irregular 1-D nowcast | core frequency-spectrum estimation | JONSWAP, uni-dir, 3 (Hs,Tp), γ=3.3 | Static, operational sparse scan | probe array (5–8 along x) | field nRMSE, spectral R², Hs/Tp err | ★★★ |
| E4 | **Window-length sweep (sequential fusion)** | information wall & **time-fusion lever**; T_wall(bandwidth) | JONSWAP, fixed Hs/Tp | sparse; vary fusion T_obs = 0.5→15 s | probe array | nRMSE vs T_obs; compare to measured-CRB | ★★★ |
| E5 | **Scan-pattern sweep (active sensing)** | **active-sensing lever**; D_eff = L + c_g·T_obs | JONSWAP, fixed | dense vs sparse; narrow vs wide footprint; along- vs cross-tank lines | probe array | nRMSE & CRB vs pattern; cross-range effect | ★★★ |
| E6 | Bandwidth / complexity sweep | information wall vs **spectral complexity** (1-D proxy) | narrow-band → broad-band irregular (vary γ, freq range) | sparse, fixed | probe array | nRMSE & cond(FIM) vs bandwidth; wall onset | ★★ |
| E7 | **Forecasting / predictable zone** | predictable-zone CRB; **forecast skill** (control-relevant) | JONSWAP | nowcast at carriage, predict downstream probe | downstream probes at +x | forecast skill vs horizon & distance; UQ coverage | ★★★ |
| E8 | Moving carriage (ship motion) | ego-motion / encounter-frequency realism | Regular + irregular | LiDAR on towed carriage, U = 2–3 speeds | probes (earth-fixed) + carriage encoder | nRMSE under motion; Doppler/encounter handling | ★★ |
| E9 | UQ calibration | **calibrated uncertainty** (the (C) contribution) | irregular, ≥ 5 repeats/condition | sparse | probe array | ±1σ/±2σ coverage, reliability diagram | ★★ |
| E10 | (if segmented WM) oblique / short-crested | limited **directional** validation | oblique regular / short-crested | sparse | probe array (2-D) | directional nRMSE; partial multi-dir check | ★ |

## 4. Mapping experiments → paper claims
- **CRB / information wall (Dir-I):** E4 (T_wall vs T_obs), E6 (vs bandwidth), E5 (aperture).
- **Active sensing lever:** E5 (scan pattern), with cross- vs along-tank contrast.
- **Sequential time fusion:** E4 — the headline physical confirmation that more time
  reconstructs harder spectra.
- **Predictable zone + forecast (application):** E7 — and the information-latency point.
- **Grid-free estimator + calibrated UQ:** E3, E9.
- **Forward-model fidelity / nonlinearity:** E1, E2 (justifies linear model + when
  Stokes2/HOS is needed).
- **Multi-directional (hard regime):** NOT in the tank — simulation + future ocean basin.

## 5. Deliverables from the campaign
1. Physical confirmation of the **time-fusion** and **active-sensing** levers (E4/E5)
   — the strongest, novel, advisor-positive results, now on real water.
2. Real-data **forecast skill + calibrated UQ** (E7/E9) for the control story.
3. A measured forward-model fidelity / nonlinearity budget (E1/E2) supporting the
   modeling choices.
4. Honest scoping: directional generalization remains simulation + future ocean basin.

*Confirm SNU towing-tank specs (length/width/depth, wavemaker type — flap/plunger,
segmented?, max H, T range) before finalizing conditions and Froude scale.*
