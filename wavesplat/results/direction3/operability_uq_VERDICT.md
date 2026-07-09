# VERDICT — D2: calibrated UQ for risk-aware operability — HONEST MIXED result (2026-06-29)

Goal (D2): give the calibrated UQ a downstream JOB — show UQ-aware phase-resolved forecasting
improves operational go/no-go decisions. Decision = (op point, forecast lead); safe = |η_GT| <
0.25·Hs; GO on an unsafe case = VIOLATION. Forecast (mean, σ) from the deployable amortized
posterior (σ temperature-calibrated on held-out seeds, s≈0.94–1.0). Regime: realistic short
feedforward leads {0.5–3 s}, 3 seas (BF3-5c/BF4-15c/BF5-30c) × 4 seeds, 1200 test decisions.
`scripts/d70_operability_uq.py`.

## Result — honest, mixed (a modest positive squeezed by the information limit and a strong baseline)
- base unsafe 30%; **oracle 70% uptime / 0% violation** (what perfect knowledge could do).
- **point/blind** (no UQ): 100% uptime, **30% violation**, NO risk control.
- **persistence** ("operate if calm now"): **70% uptime, 24% violation** — a STRONG baseline.
- **calibrated UQ frontier** (sweep risk z):
  - learned heteroscedastic σ **modestly beats a constant-σ margin** (+2.3% mean violation
    reduction at matched uptime — allocates the risk budget where the forecast is uncertain);
  - it provides **tunable, calibration-guaranteed risk control** — the operator can DIAL a target
    violation rate and (because σ is calibrated) actually get it; blind/persistence offer a single
    uncontrollable risk level.

### BUT — the honest negatives (the script's auto "UQ wins" line over-claimed; corrected here)
- **Persistence is NOT beaten.** ZERO posterior-frontier points dominate persistence (uptime ≥0.70
  AND violation ≤0.24); at ~70–80% uptime the forecast policy has ~30% violation > persistence 24%.
  The learned forecast+UQ does **not** enable clearly better operability than naive "is it calm now?"
- **The frontier is information-limited.** To reach a ≤10% violation target you must drop to ~0%
  uptime — you cannot operate at low risk and useful uptime, because the dangerous events (large
  future |η|) are exactly the *unpredictable* ones (forecast mean hedges to ~0; σ is large but
  fairly uniform). The oracle's 70%/0% is unreachable by any forecast policy.

## Honest conclusion
D2 is a **weak/mixed positive, not the hoped-for clear win.** The calibrated UQ's genuine,
defensible value is **quantified, tunable, calibration-guaranteed risk** (a safety dial that point
forecasts and persistence lack) plus a small frontier edge over a uniform margin. It does **not**
beat naive persistence on the safety–uptime frontier, and absolute operability is capped by the
information limit — fully consistent with the project's pervasive theme. Reporting this straight
(not the over-claimed auto-verdict) per the standing fact-check directive.

## Implication for the paper
- Do NOT claim "UQ-aware forecasting enables better operability than baselines" — persistence rivals
  it. The honest claim: *calibrated UQ provides controllable, guaranteed operational risk (a safety
  dial) that naive methods cannot, and modestly improves on a uniform margin; the achievable
  operating envelope is set by the information limit.*
- This means D2 (operability framing) does NOT strongly rescue the "deployable posterior has impact"
  story. Options: (i) try the feedforward MOTION-COMPENSATION framing (forecast pre-cancels
  predicted motion — value lives in the predictable component, may beat persistence more clearly),
  or (ii) accept the honest result and frame the UQ contribution as risk-controllability, not
  superior decisions.

## Caveats
- 3 seas × 4 seeds (cal {0,1}/test {2,3}); single posterior training run; binary threshold task with
  H_safe=0.25·Hs; discrete z-sweep (coarse frontier). The qualitative conclusions (persistence
  strong, frontier info-limited, σ calibrated, learned σ ≳ constant) are robust to these.
