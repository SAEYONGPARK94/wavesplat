# PREREG 2 — record-coherence gate for bank-MMSE Doppler (2026-07-29)

FROZEN BEFORE THE RUN. Follow-up to PREREG_DOPPLER_CONFIRM.md, whose verdict
was: P1/P2/P3 PASS on 5/5 fresh YS17 windows (median Δnowcast −0.147, all
selected directions exactly 150°) but **P4 FAIL** (AA15a w3: blind per-window
argmin picked U=0.6 @ −30° and hurt nowcast +0.32) → per the pre-committed
interpretation, blind per-window selection is NOT deployable.

The failure mechanism matches the July rule ("within-record U-stability is the
GT-free discriminator"): YS17 selections are window-coherent (150° in all 6
windows incl. the consumed w0), the no-current records select incoherent
directions and/or floor speeds. This prereg freezes that rule as a GATE and
tests ONLY its false-fire rate on fresh no-current windows (the YS17 fire is a
derived statistic on already-consumed windows and is labeled as such).

## Frozen gate rule
Per window: Usel = argmin vr over the PREREG-1 grid (speeds 0.15..1.20 x 12
dirs + U0; seed-42 fits, 2-row Pq; GT never enters).
For a record with W >= 4 probed windows, the gate FIRES iff:
  (a) >= 3 windows select a non-floor speed (> 0.15), AND
  (b) circular spread of the non-floor directions <= 45 deg, AND
  (c) median non-floor speed >= 0.30 m/s.
If the gate fires, deployment applies the record-median non-floor U (not the
per-window U). If it does not fire, Uc = 0 (today's shipped behaviour).

## Frozen test set (all windows FRESH for this question)
No-current records: AA15a {w0,w1,w2,w4}, AA15b {w0,w2,w4,w5}, AA14 {w0..w5},
LJ18g {w0..w5} — 20 windows, 4 records.

## Frozen bars
- N1: the gate fires on 0 of the 4 no-current records.
- Y1 (derived, consumed data, non-confirmatory): the gate fires on YS17
  (w0..w5 selections: 150 deg unanimous, speeds 0.75-1.2) — recorded for
  completeness, carries no fresh evidential weight.

## Pre-committed interpretation
- N1 PASS -> propose operational flag `bank_doppler: 'record_gated'`
  (DEFAULT OFF; enabling it is a user deployment decision). The YS17 gain it
  would unlock is the PREREG-1 measured median −0.147 nowcast.
- N1 FAIL -> the coherence gate is insufficient as specced; honest negative,
  no deployment, no threshold re-tuning on this data (any retuned gate needs
  new fresh windows).
