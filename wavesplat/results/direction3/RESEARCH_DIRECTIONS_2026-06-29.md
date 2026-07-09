# Careful review — what (if anything) is worth researching next (2026-06-29)

User: "더 연구해야 될 방향이 있는지 면밀히 검토." Honest strategic assessment after the d62–d68 hardening.

## 1. What is SOLID now (the assets)
1. **CRB information-wall theory** (Dir-I): D_eff = L + c_g·T_obs, T_wall, predictable zone. Rigorous.
2. **Discrete-benchmark-bias critique — 4× confirmed, mechanistic, significant** (d44 RELAX over-rating;
   d62 FIM-zone; d63 active-sensing; **d65 continuity = the mechanism**, p=2.6e-3). The strongest,
   most novel contribution.
3. **Exhaustive learning negatives** — estimation CRB-capped; active-sensing a discrete artifact;
   physics-feature redundant. Rigorous "why learning can't beat the limit."
4. **Deployable amortized posterior** — fast (≈8 ms) + accurate (≈ classical) + **lead-calibrated UQ**
   (forecast-trained) + nowcast/forecast. Robustness-confirmed (d66) across noise & density.
5. **FIM ≈ geometric DPZ on real seas** (honest correction; folds into the critique).
6. **Physics UQ for classical RELAX** (d64) + **forecast-trained learned UQ** for the posterior (d67).

→ The project is **theory-complete and method-complete**. It can already support a strong
"information limit + benchmark critique + deployable posterior" paper.

## 2. Honest weaknesses a committee/reviewer will attack
- **W1 — all synthetic + HOS, no real/experimental data.** The single biggest gap for Ocean Eng / IEEE JOE.
- **W2 — the headline leans NEGATIVE** (learning doesn't beat classical; the strong theory claim died).
  Needs a crisp POSITIVE anchor (the critique + the deployable posterior + the theory).
- **W3 — no downstream impact.** "We estimate waves + UQ" — but it is never shown to improve a decision
  or control task. Limits significance.
- **W4 — the posterior's accuracy ≈ classical** (info-limited); its win is speed + UQ. The UQ needs a job.

## 3. Candidate directions — honest scoring
| dir | what | value | novelty | risk | effort | verdict |
|---|---|---|---|---|---|---|
| **D2** | **downstream control/decision using the calibrated UQ** (risk-aware feedforward motion comp / operability go-no-go) | **HIGH** | **HIGH** | med | med | **top new direction** |
| **D1** | **real-data / towing-tank / public-dataset validation** | **HIGH (gating)** | low | data-dependent | med–high | **pursue / protocol** |
| D3 | continuous/HOS evaluation **benchmark + protocol** (constructive half of the critique) | med-high | med | low | med | fold into paper |
| D5 | **theory crispening**: prove FIM-zone → geometric DPZ on continuous spectra (c_g-transport bridge); diverge only for discrete-identifiable | med | med | low | low | do while writing |
| D4 | better **continuous representation** (neural spectral-density field) to chip at oracle gap | high *if* it worked | med-high | **HIGH (likely info-capped; d50 failed)** | high | low priority |
| D6 | multi-sensor fusion (LiDAR+radar/IMU) to enlarge D_eff | med | med | high | high | future work |
| D7 | non-stationarity / online calibration-drift detection via UQ | med | med | med | med | optional extension |

## 4. Recommendation
The analytical/method work is mature; the two pieces that most raise the ceiling:

- **D2 (STRONGEST new research):** give the calibrated UQ a JOB. Demonstrate, in simulation, that
  UQ-aware phase-resolved forecasting improves a downstream task vs UQ-blind — e.g. (a) feedforward
  ship-motion compensation (couple the forecast to a simple RAO/motion model; show reduced residual
  motion, with the UQ gating when to trust the feedforward), or (b) operability go/no-go for
  crane/gangway ops (calibrated predictive interval → safe operating windows; show fewer
  false-safe/false-abort decisions than a UQ-blind or persistence baseline). This converts the work
  from "estimation + analysis" into "estimation **enables better decisions**," makes the UQ the hero
  (its unique deployable feature), and supplies the positive anchor W2 needs. Novel for ship-LiDAR
  phase-resolved waves.

- **D1 (GATING for publication):** secure a real or public dataset (tank or field LiDAR/radar), or at
  minimum write a concrete validation protocol + run the deployable posterior on whatever is
  obtainable. Without any real data the paper is vulnerable regardless of how rigorous the synthetics.

- **While writing:** fold in **D3** (propose the continuous/HOS benchmark — be constructive, not just
  critical) and **D5** (a clean derivation that the FIM predictable boundary reduces to the geometric
  DPZ on continuous spectra via the synthetic-aperture term, diverging only for discretely-identifiable
  spectra — this *explains* today's findings and unifies the theory).

- **De-prioritize D4** — the information limit + d50/d34/d51 negatives predict another cap; only worth
  it out of curiosity or as one more documented negative.

## 5. The honest meta-question
There is also the option of **stopping new experiments and writing now** — the work is already a
coherent, defensible paper (CRB limit + 4×-confirmed benchmark critique + deployable posterior +
robustness). The real choice is: *consolidate & write* (solid paper now) vs *add D2 (+D1)* for a
higher-impact paper at the cost of weeks. D2 is the one addition that materially raises the ceiling
rather than adding another negative; recommend it if continuing, otherwise write.
