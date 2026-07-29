# PREREG — record-conditional Doppler (Uc) for the operational bank-MMSE (2026-07-29)

FROZEN BEFORE THE CONFIRMATION RUN. Exploration that motivated this (already
consumed, may not be cited as confirmation): YS17 w0 — GT-free-selected
U ≈ 1.05 m/s @ 150°, nowcast 0.7180 → 0.5936 (−17%), lead-mean 0.8705 → 0.7565
(ys17_doppler.json, ys17_doppler_ext.json). July precedent: pooled blanket-U
REFUTED; YS17-only conditional CONFIRM with within-record U convergence — the
GT-free discriminator is U-stability, and selection must never touch GT.

## Hypothesis
On records with a real current (YS17), adding a current-Doppler term
om += kx·Ux + ky·Uy to the bank, with (Ux,Uy) selected PER WINDOW by the fit's
own held-out val misfit (vr), improves the deployed blind median-of-3 field.
On no-current records (AA15a/AA15b) the same selection does no harm.

## Frozen protocol
- Estimator: research mmse_bank_estimate, blind parametrization exactly as the
  deployed port (sigma0 = 0.2·std(z), kp_hint=None), protocol-lead scoring with
  dense-GT std normalization (identical to the port verification).
- U grid: speeds {0.15, 0.30, ..., 1.20} x 12 directions (30° step) + U=0.
- Selection: seed-42 fits with 2-row Pq (no render), Usel = argmin vr.
  GT never enters selection.
- Score: U0 vs Usel, each = pointwise-median field over seeds {42,1042,2042},
  nowcast + lead-mean protocol nRMSE.
- Test windows (all FRESH, never used in this port's exploration):
  YS17 w1..w5 (5 windows).
- Null-control windows: AA15a w3, AA15b w1 (fresh), AA15a w5, AA15b w3 (reused,
  harm-check only).

## Frozen bars
- P1 (win, YS17): median Δnowcast(Usel−U0) over the 5 fresh windows ≤ −0.02
  AND nowcast improves in ≥ 3/5 windows.
- P2 (mechanism, YS17): circular spread of selected U directions over the 5
  fresh windows ≤ 60° (a record current must be window-coherent). Windows whose
  selected speed is 0.15 (grid floor -> "no current found") are excluded from
  the spread; if ≥ 3 windows select the floor, P2 FAILS.
- P3 (lead no-harm, YS17): median Δlead-mean ≤ +0.01.
- P4 (null-control, AA): median Δnowcast over the 4 AA windows ≤ +0.01 and no
  single AA window worsens nowcast by > +0.05.

## Pre-committed interpretation
- ALL PASS -> propose operational flag `bank_doppler: auto` (per-window GT-free
  U selection behind the real gate; default OFF until user approves deployment).
- P1 FAIL -> honest negative: the w0 gain was window-specific; no deployment.
- P1 PASS but P2 FAIL -> selection is fitting noise, not a current; no
  deployment (July rule: stability IS the discriminator).
- P4 FAIL -> no deployment regardless of YS17 outcome (harm on no-current
  records is disqualifying).
