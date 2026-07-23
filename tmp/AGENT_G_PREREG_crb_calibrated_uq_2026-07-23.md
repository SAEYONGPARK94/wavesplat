# Agent G — PRE-REGISTRATION: CRB-anchored / inefficiency-calibrated runtime UQ
**Frozen 2026-07-23, BEFORE any calibration number was computed.**
Discipline: [[feedback_dual_verification_default]] + [[feedback_direction_flip_discipline]] (power gate,
symmetric audit, MEASURED/ARGUED split, goalpost freeze, deliberate under-claim on first reading).

## 0. The claim under test
Not (a) "σ_CRB ranks where skill is high" — that is the project's existing envelope spine (E-58/E-59/C8),
already established as PARTIAL/qualified and **not** what this gate is about.
This gate tests (b): **σ_CRB, times ONE globally-fitted inefficiency factor α, is a calibrated absolute
predictive error bar** — i.e. it can be shipped at runtime as `±1.96·α·σ_CRB(q)` and the stated coverage
holds, better than cheap baselines.

## 1. Substrate (frozen)
- **Arm H (headline, nonlinear)**: HOS seas via `tmp/hos_s8_envelope.py` machinery
  (`e15_hos_multidraw.build_sea/draw`, RELAX θ̂ fit, ridge-LS field prediction, σ_CRB **at fitted θ̂**,
  σ_emp = ridge posterior std, dist = nearest-obs distance). Real nonlinear HOS truth, non-oracle.
  Multiple observation draws per sea (raydrop + noise + range-stratified subsample rng) = error ensemble.
- **Arm S (regime axis, synthetic)**: `data/episode_BF*_{5,15,30}comp_*.h5` via `tmp/e58_envelope_validation.py`
  machinery (oracle-θ CRB, matched linear model class). Gives the clean under-det→well-det (kp) axis.
- Errors are normalised by Hs (dimensionless), pooled over query grid × leads {0,2,4,6}s.

## 2. Candidate bars (all get IDENTICAL calibration machinery — one global scalar)
For anchor `s(q)` the bar is `σ̂(q) = α·s(q)` with the Gaussian-MLE scale
`α = sqrt( mean_over_train[ e² / s² ] )` fit on TRAINING seas/episodes only.
| id | anchor s(q) | runtime-computable? | pre-empted by |
|---|---|---|---|
| CONST | 1 | yes | trivial |
| **CRB** | σ_CRB(q) | yes (θ̂ + geometry) | **the thing under test** |
| EMP | σ_emp(q) = ridge posterior std | yes | Fucile 2018 LEPrE, Rodgers OE |
| DIST | nearest-obs distance | yes | trivial geometric |
| PERSEA | 1, but α refit per sea | **NO — needs GT** | oracle reference (episode-level ceiling) |

## 3. Metrics (held-out, leave-one-sea-out / leave-one-episode-out)
Primary: **Gaussian NLL** per held-out unit, `mean_q[ 0.5·log(2πσ̂²) + e²/(2σ̂²) ]` (proper scoring rule).
Secondary: coverage@68/95, z-std, PIT KS statistic, spread–skill (Spearman σ̂ vs binned RMS(e), and
regression slope of log RMS(e) on log σ̂ — ideal 1).
Diagnostic: **spread of the per-unit α** (does ONE global factor transfer? report IQR and max/min).

## 4. PRE-REGISTERED DECISION RULE (frozen; no post-hoc goalposts)
Let ΔNLL(X−CRB) = NLL of bar X minus NLL of CRB bar, per held-out unit; report median + 95% bootstrap CI
(clustered on sea/episode = the effective n) and win-fraction.

**GO (as a supporting layer)** requires ALL of:
- **G1** median ΔNLL(CONST−CRB) > 0, 95% CI excludes 0, win-fraction ≥ 0.70. *(CRB beats a constant bar)*
- **G2** CRB bar coverage on held-out data within [0.62,0.74] @68% and [0.90,0.98] @95%.
- **G3** median ΔNLL(EMP−CRB) ≥ 0. If its CI includes 0 → "ties EMP" → downgrade to **CONDITIONAL**
  (because EMP ≡ an analytic estimator-specific prediction-error std ≡ Fucile 2018 LEPrE / Rodgers OE
  posterior covariance — already published; then the *CRB-specific* part is not load-bearing).
- **G4** sign of G1 positive in BOTH regimes (kp split at median in Arm S; Hs/steepness split in Arm H).
- **G5** the headline is reproduced in **Arm H (HOS nonlinear)** — a synthetic-only win is not a headline
  ([[feedback_validation_nonlinearity]]).

**NO-GO** if: median ΔNLL(CONST−CRB) ≤ 0, **or** CI includes 0 with |median| < 0.05 nats,
**or** the per-unit α spread exceeds 3× (max/min) — a single global factor then demonstrably does not transfer.

**UNDERPOWERED (no direction label)** if bootstrap SE > |median|/2 (discipline: power gate).

## 5. Dual verification (required before any direction label)
1. Route 1 = the gate script's LOSO aggregation.
2. Route 2 = an independent re-derivation from the raw per-(unit,q) dump: recompute α, NLL, coverage with
   NumPy-only code that does not import the gate module, plus a permutation control (shuffle σ_CRB across q
   within a unit — the CRB bar must lose its advantage).

## 6. Known threats, declared in advance
- **T1 σ_CRB needs θ̂**: Arm S uses ORACLE θ (inverse crime for calibration). Arm H uses fitted θ̂ (honest).
  Only Arm H can support a runtime claim.
- **T2 CRB is a LOWER bound**: α>1 is expected and is exactly the "inefficiency factor"; α<1 would signal
  a mis-specified/biased error ensemble, not efficiency.
- **T3 Error ensemble definition**: HOS truth is one realization/sea; draws vary observation sampling+noise.
  Coverage is therefore *observation-conditional*, not phase-conditional.
- **T4 Gaussianity** of the error assumed for coverage/NLL; PIT KS is the check.
- **T5 n is small** (seas / episodes are the cluster unit) — CIs are clustered, not per-query-point.
