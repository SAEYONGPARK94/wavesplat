# PROTOCOL — rolling-refit streaming validation for the deployed bank-MMSE (2026-07-29)

Written before the run. This measures an engineering knob (refit cadence), not a
selection rule: no GT-driven choice is made inside any arm, so consumed windows
may be reused; noted anyway.

## Question
The deployed real route fits once on the trailing 40 s window and then forecasts
(bank_refit_every_s default None). In operation observations keep arriving. How
much sustained accuracy does periodic refitting recover, at what compute cost?

## Streaming replay
Extend the frozen scene_real protocol into the lead span: the observation stream
continues over [t0, t0+lead_max] with the SAME fan trajectory, per-frame sample
count (m_scan=12), rng stream, and noise as the frozen window (the memory-window
part of the replicated stream must reproduce sc['P'] exactly — asserted per
window before any arm runs). At evaluation time tau in {0,1,...,lead_max} s the
CURRENT-TIME field estimate is scored against the stereo frame at tau
(4 m-subsampled grid, valid-pixel mask, normalized by the dense-GT std — the
ledger normalization).

## Arms (all = deployed stack: blind sigma0=0.2*std(z_window), kp_hint=None,
## seeds {42,1042,2042}, pointwise-median field, 600-obs cap via rng(0))
- frozen : one fit at t0 (today's shipped behaviour), pure forecast afterwards.
- refit-R, R in {2, 5, 10} s: at tau the active fit is the one at
  t_f = t0 + R*floor(tau/R), fitted on the trailing 40 s of the stream up to t_f.

## Windows
AA15a {w1,w3,w5}, AA15b {w1,w3,w5}, YS17 {w1,w3,w5} (9 windows; w5/w3 reuse from
the port verification is harmless here — no selection happens).

## Pre-stated criteria
- ADOPT-CANDIDATE R*: the largest R whose sustained mean nRMSE (over tau) is
  within 0.01 of the best cadence's, per record majority — prefer the cheapest
  cadence that saturates the gain.
- The refit arm must beat frozen sustained-mean on >= 2/3 records and make no
  window worse by > +0.02; otherwise record an honest negative.
- Compute feasibility is reported (fit wall-time vs cadence; members are
  parallelizable, so the binding number is the single-member fit time).
- Any change of the shipped default (bank_refit_every_s) is a USER decision on
  these numbers; this run changes no code.
