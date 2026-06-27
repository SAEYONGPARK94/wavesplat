# Novelty & Contribution review — persona debate (Directions I/II/III)

**Date:** 2026-06-27. Adversarial review of the grid-free spectral-prior + CRB work,
grounded in the actual numbers (incl. the over-smoothing diagnostic
`results/direction3/diag_smoothing.json`).

## 0. Claimed contributions (as stated)
1. **(I) CRB backbone:** dispersion-constrained space-time Fisher information for
   sparse ship-LiDAR; effective aperture D_eff = L + c_g·T_obs; information wall
   T_wall(complexity); predictable-zone = identifiability (not geometric c_g-cone);
   estimator near-optimality.
2. **(II→III) Grid-free learned prior:** atomic set-diffusion prior + DPS with the
   dispersion-constrained GP-marginal likelihood; beats the grid representation;
   complexity-robust; calibrated UQ.
3. **Unifying claim:** the bottleneck is information, not algorithm — negative
   results (NN-estimators, prior scaling) explained by the CRB wall.

## 0.1 The number that forces honesty (over-smoothing diagnostic)
| sea | nRMSE | VR | Pearson | Skill | zero-pred nRMSE |
|---|---|---|---|---|---|
| BF4 5c  | 0.232 | 0.17 | 0.37 | 0.14 | 0.250 |
| BF4 30c | 0.263 | 0.22 | 0.31 | 0.07 | 0.273 |
| BF5 30c | 0.168 | 0.33 | 0.67 | 0.42 | 0.222 |

The DPS **BMA is over-smoothed** (VR 0.17–0.33 = ~20–30% of true amplitude) and on
the complex seas its nRMSE barely beats the zero-predictor (skill 0.07 on BF4-30c).
This must be confronted head-on.

---

## 1. Persona debate

### 🟥 Reviewer 2 (skeptical methods reviewer)
- "**Novelty is assembly.** Set/point-cloud diffusion (Luo&Hu 2021), DPS (Chung 2023),
  atomic-norm gridless estimation, GP-marginal likelihood — all pre-existing. You
  combined them for waves. That's an application paper, not a method paper."
- "**You don't beat SOTA.** Overall 0.220 vs ST-DF-GP 0.205. Your own prior method
  wins. A new method that loses overall is hard to accept."
- "**The complex-sea 'win' is an artifact.** VR=0.22 means the BMA predicts ~22% of
  the variance — it's an ensemble average smoothed toward zero. On a zero-mean field
  with std≈Hs/4 the zero-predictor gets nRMSE 0.25; your 0.23 is *barely* better
  (skill 0.07). You're not reconstructing complex seas — you're failing toward the
  mean. BMA over nRMSE is a known trap; your own metrics.py warns about it."
- "**Sim-only.** No real LiDAR. Linear vs Gerstner mismatch. Single-frame eval."

### 🟦 Ocean-engineering / DSWP domain expert
- "**The CRB framing has real value.** DSWP people use the geometric predictable zone
  (c_g-cone) operationally but rarely write the *dispersion-constrained space-time
  Fisher information*. Your D_eff = L + c_g·T_obs re-derives the predictable-zone
  timescale from estimation theory, and the **predictable-zone = identifiability,
  not geometry** result is a genuine reframing — for the parametric/spectral model
  the boundary is set by which components are observable, not by ray-tracing. That's
  publishable in an ocean-eng venue *as theory*."
- "**T_wall(complexity) is useful** — telling an operator 'a 30-component sea needs
  ~10 s ≈ 1.4 Tp before the field is identifiable' is actionable for phased control."
- "**But** the reconstruction itself at nRMSE 0.22 on complex seas is operationally
  weak — that's not a usable phase-resolved field for feed-forward control. Be honest
  that this regime is *unreconstructable from one window*, which your own CRB says."
- "**Calibrated UQ is welcome** — DSWP rarely ships honest uncertainty. A trustworthy
  predictable-zone mask + calibrated variance is more useful operationally than a
  slightly-better point RMSE."

### 🟩 ML / Bayesian-inference expert
- "**The set-diffusion + physics-DPS is a clean instantiation**, and the atomic
  representation keeping the GP-marginal likelihood exact/continuous is the right
  design — that's a tasteful contribution even if the parts are known."
- "**But the BMA is the wrong estimator to headline.** The posterior-mean field
  minimizes MSE and therefore over-smooths (VR↓). You're reporting the MMSE estimate
  and calling its low nRMSE a 'win', while the *per-sample* VR (0.51–0.74) shows the
  posterior actually carries amplitude. Report posterior *samples* (calibrated, full
  variance) as the product, not the variance-collapsed mean — otherwise Reviewer 2 is
  right that the win is smoothing."
- "**The calibration is real but thin:** one global temperature s=1.38 → ±1σ 0.68 is
  good, but it's marginal coverage; check spatial/conditional calibration, not just
  the pooled histogram."
- "**The strongest scientific result is the negative one done rigorously:** scaling
  the prior 10× doesn't move the field error, and the ε-loss floor is irreducible —
  that, tied to the CRB, is a real 'information-limited' proof. Lead with that."

### 🟨 Advocate (author/advisor)
- "Granted the BMA over-smooths — but the *honest* framing is strong: **on complex
  multidirectional seas the field is provably unreconstructable from one window
  (CRB), so the right goal is graceful degradation + calibrated UQ, not point
  accuracy.** Grid-free DPS degrades *toward a calibrated near-mean estimate (skill
  ≥ 0)*; ST-DF-GP degrades *catastrophically (skill < 0 on several complex seas —
  it injects wrong variance)*. 'Better-behaved failure with honest UQ' in the regime
  the theory says is hopeless is a contribution, not a bug."
- "The paper is not 'a new SOTA estimator'. It's **'where is the limit, why, and how
  to behave honestly against it'** — backbone = CRB (I), mechanism = grid-free prior
  (III), conclusion = calibrated UQ + graceful degradation. That arc is novel as a
  *whole* even if each brick is known."
- "Robustness is real: flat 0.19–0.23 across 5→30 components vs ST-DF-GP's 0.07→0.31.
  Predictable failure is valuable for safety-critical control."

### ⚖️ Chair (synthesis)
What survives adversarial review, ranked by defensibility:
1. **CRB / information-wall theory (I)** — *strongest, most novel, hardest to attack.*
   Dispersion-constrained space-time FIM, D_eff = L + c_g·T_obs, T_wall(complexity),
   and **predictable-zone = identifiability ≠ geometric cone**. This is theory, it is
   domain-relevant, and it reframes the problem. *Lead the paper with this.*
2. **Rigorous information-limited demonstration** — NN-as-estimator fails; NN-as-prior
   is mandatory but saturates; 10× training doesn't help; all explained by (1). A
   "negative results, done right, unified by theory" contribution.
3. **Grid-free atomic set-diffusion + physics-DPS (III)** — novel *assembly*, removes
   the grid bottleneck, complexity-robust, ships calibrated UQ. *Must* be presented
   honestly: it does **not** reconstruct complex seas (BMA over-smoothed, skill≈0);
   its value is **graceful degradation + calibrated UQ** in the regime theory deems
   unreconstructable — not a point-accuracy win.

**Verdict:** This is a credible PhD-level contribution **iff reframed** from
"new estimator that wins" to **"the limits of phase-resolved wave estimation from
sparse ship-LiDAR (CRB), and an honest, calibrated, grid-free estimator that
respects them."** As a "beat SOTA" paper it fails (0.220 > 0.205, over-smoothed BMA).
As a "limits + honest UQ + graceful degradation" paper it is novel and defensible.

---

## 2. Required fixes before claiming the contribution
1. **Report the full metric panel everywhere** (VR, Pearson, skill), never nRMSE
   alone — pre-empt the over-smoothing critique by owning it.
2. **Headline posterior samples / calibrated UQ, not the variance-collapsed BMA**;
   if reporting BMA, report its VR alongside and explain it as the MMSE estimate.
3. **Reframe the complex-sea result** as graceful-vs-catastrophic degradation
   (skill ≥ 0 vs skill < 0), tied to the CRB unreconstructability.
4. **Real-data validation** (even one real ship-LiDAR / wave-tank set) to answer
   "sim-only".
5. **Multi-frame evaluation** to remove single-frame noise from the head-to-head.
6. **Lead with the CRB theory**; position III as the mechanism, UQ as the product.

## 3. One-paragraph honest contribution statement
> We derive the Cramér–Rao bound for phase-resolved wave-field estimation from sparse
> ship-mounted LiDAR under the dispersion constraint, identifying an *information wall*
> T_wall(complexity), an effective synthetic aperture D_eff = L + c_g·T_obs, and a
> predictability boundary set by spectral *identifiability* rather than the geometric
> group-velocity cone. Guided by this, we show learned estimators fail as amortized
> regressors but are necessary as *priors*, and build a grid-free atomic set-diffusion
> prior with physics-constrained posterior sampling that removes the grid-quantization
> bottleneck and yields a *complexity-robust, calibrated* estimator. On complex
> multidirectional seas — which the CRB proves are unreconstructable from a single
> window — it degrades gracefully toward a calibrated near-mean estimate (skill ≥ 0)
> where a tuned space-time GP degrades catastrophically (skill < 0), while remaining
> competitive elsewhere. The contribution is not a new accuracy record but a
> theory of the limits and an honest, uncertainty-aware estimator that respects them.
