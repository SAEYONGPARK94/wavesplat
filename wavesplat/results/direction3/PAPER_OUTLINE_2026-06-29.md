# Paper consolidation — main content + methodology (2026-06-29)

After the full d30–d71 investigation. Honest, write-ready synthesis: what the paper CLAIMS and HOW
it was established.

## One-line thesis
**Phase-resolved ocean-wave estimation and short-term forecasting from sparse ship-mounted LiDAR is
fundamentally information-limited; the field's discrete-spectrum benchmarks have masked this. We
characterize the limit (Cramér–Rao), show every classical and learned method is capped by it, expose
how the discrete benchmark manufactures illusory advantages that vanish on realistic continuous/
nonlinear seas, and deliver a deployable real-time posterior with calibrated uncertainty while
honestly bounding its value.**

---

## Main content — the contributions (ranked by novelty)

### C1 — An information-wall theory (Cramér–Rao / Fisher information). [NOVEL, foundational]
- **Effective synthetic aperture** D_eff = L_footprint + c_g·T_obs: platform + wave motion extend the
  sampling aperture along the propagation direction; wavenumber CRB ∝ 1/D_eff.
- **Information wall** T_wall: observation time needed to beat the zero-predictor; diverges for
  complex seas → bias/prior provably required.
- **Predictable zone** set by Fisher-information identifiability. On continuous (realistic) spectra it
  COINCIDES with the classical geometric (group-velocity) DPZ — they diverge only for the (artificial)
  discretely-identifiable case [feeds C2]. No prior CRB treatment of this problem exists.

### C2 — The discrete-benchmark bias, with the continuity mechanism. [NOVEL, the provocative hook]
The field evaluates phase-resolved estimators on discrete-component synthetic seas. This structurally
manufactures "advantages" that EVAPORATE on realistic continuous/nonlinear seas. **4× confirmed; the
mechanism is spectral CONTINUITY, not nonlinearity; statistically significant.**
- discrete benchmark over-rates the discrete estimator (RELAX) vs continuous reality;
- a "novel FIM predictable zone ≠ geometric DPZ" is a discrete artifact (coincide on continuous);
- a CRB-driven active-sensing "learning lever" (+24% discrete) → +2% on continuous (n.s.);
- controlled ladder (discrete-linear → continuous-linear → continuous-nonlinear) isolates continuity
  as the killer (the drop is complete at the linear→continuous step; p = 2.6e-3).

### C3 — Exhaustive evidence that no method beats the limit. [RIGOROUS negative, the support]
Estimation (learned prior/off-grid/warm-start/detection), active sensing (free-aim is uniform-optimal;
constrained is a discrete artifact), continuous representation (non-learned ties learned), nonlinearity
(bound harmonics ~4%) — all capped by or at the information limit. Every apparent learning edge is a
regime/window artifact or matched by a non-learned method.

### C4 — A deployable real-time posterior + the honest scope (incl. the zero-predictor reframing).
[engineering + sober reframing]
- A fast **amortized neural posterior** (obs → per-query mean, σ; ~10 ms; calibrated, lead-stable UQ;
  nowcast + multi-second forecast) — the only method giving fast + UQ at the real-time operating point.
- **Honest bounds:** accuracy ≈ classical and its win is **conditional on the real-time compute/latency
  cycle** (longer-window RELAX is more accurate but non-deployable). It provides **no downstream
  control value** over trivial baselines (operability: persistence rivals it; motion compensation:
  forecast cancels ~0% of motion vs persistence's ~50% at short latency).
- **⭐ Zero-predictor reframing:** nRMSE 0.25 = predicting flat water (σ_surface = Hs/4). Deployable
  short-window methods sit at ~0.22–0.25 → near-zero usable skill for control; report SKILL vs the
  0.25 baseline, not raw nRMSE. (Real skill only at high-obs/long-window: 0.119 → ~52% skill.)
- Surviving value of the UQ = **risk awareness** (its σ correctly predicts where the forecast fails),
  not better control.

---

## Methodology (what was actually used)

**Wave physics & sensing**
- Linear synthesis (sum-of-cosines / JONSWAP components) AND nonlinear **High-Order Spectral (HOS, M=3)**
  simulation (bound harmonics) for realistic seas [standing directive: validate on HOS].
- Ship-mounted LiDAR sampling model: footprint pool, raydrop (uniform & slope), range noise σ≈0.02 m.
- Dispersion as a HARD constraint everywhere: ω = √(g|k|·tanh(|k|d)).

**Theory (C1)**
- Analytic Fisher Information Matrix, 4N params (w, kx, ky, φ); analytic Jacobian validated vs autograd
  (5e-15); field-reconstruction CRB = √(tr(F⁻¹G)); controlled aperture-law experiment (log-log slope
  −1.09 ≈ −1); GPU float64.

**Estimators**
- Classical: **RELAX** (off-grid matching pursuit) + iterative RELAX; beamforming init; ridge dense-grid
  LSQ (**F1**); **Space-Time Dispersion-Fused GP**; **ETKF** & **GPR** baselines.
- Learned: **set-diffusion grid-free prior + DPS**; **amortized neural posterior** (transformer
  encoder–decoder, Gaussian NLL, per-query mean/σ); **neural acquisition policy** (cross-attention,
  greedy-D-optimal imitation); detection **CNN**; neural operator (**F4**).

**Active sensing (C2/C3)**
- Greedy **D-optimal** Fisher selection (the CRB-driven teacher); learned acquisition; beam-steering.

**Controlled experiments / fairness**
- **Continuity ladder** (discrete-linear / continuous-linear M=1 / continuous-nonlinear M=3);
- **symmetric (equal-budget)** comparisons (same window, same #obs);
- **window-aperture** sweep; **noise σ × density** robustness sweep; **compute/latency Pareto**.

**Evaluation & UQ**
- nRMSE (+ the zero-predictor 0.25 reframing), Pearson, skill score, oracle, persistence;
- UQ: Gaussian NLL training, temperature recalibration, ±1σ/±2σ coverage, corr(σ, |err|), lead-calibration;
- multi-seed with significance (sign test, Wilcoxon);
- downstream tasks: **risk-aware operability** and **feedforward motion compensation**.

---

## Suggested paper structure
1. Intro: phase-resolved ship-LiDAR estimation for real-time use; the over-optimism problem.
2. CRB information-wall theory (C1).
3. The discrete-benchmark bias + continuity mechanism (C2) — the hook.
4. Exhaustive limit analysis: classical & learned, all capped (C3).
5. Deployable posterior + honest scope + zero-predictor reframing + downstream negatives (C4).
6. Discussion: report skill vs zero-predictor; propose a continuous/HOS evaluation protocol; real-data
   validation as the next step (D1, current gap).
Venue: Ocean Engineering / IEEE JOE (theory + critique + deployable method + honesty).

## Honest framing note (advisor tension)
The rigorous story is **negative-leaning** ("learning doesn't win; the deployable estimate is near the
information wall"). The POSITIVE anchors that carry the paper are **C1 (novel theory)** and **C2 (novel,
provocative, reproducible benchmark critique)** — these are genuine contributions a reviewer cannot
dismiss. C4's posterior is the deployable artifact; the negatives + zero-predictor reframing are the
honesty layer that de-hypes the field. The main *remaining* gap is **real-data validation (D1)** — the
work is otherwise theory- and method-complete.
