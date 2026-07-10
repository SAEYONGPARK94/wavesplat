# N5 — the derived network core

2026-07-10. Code: `scripts/d3_n5_core.py`, tests `scripts/d3_n5_tests.py`.
No training is done here. This document records an architecture whose every
component is forced by a measured result of this project, and whose two known
learned-component failure modes are made *architecturally* impossible rather
than merely discouraged. Nothing here is a result about accuracy; it is
scaffolding whose correctness is proven to machine precision.

The tests print numbers; those numbers are reproduced verbatim below. Re-run
`python3 scripts/d3_n5_tests.py` (exit 0 = all pass) before trusting any of them.

---

## Why a "derived" architecture at all

Two learned components have died in this project, both for the same structural
reason — their safe output was never the behaviour training rewarded:

- **Fields.** C1 theorem: Gaussian sea + Gaussian noise + linear observation +
  ignorable missingness ⇒ the posterior mean is linear and MSE-optimal. A
  network emitting a field can at best re-learn that linear map and otherwise
  loses. It lost.
- **Priors γ(k).** A network emitting the prior weights memorized the training
  seas' support: it placed energy on modes the *training* set populated but the
  *test* observation geometry could not constrain. Off-distribution this
  injected energy into unconstrained modes (OOD −45 %, reported z-std 68,710).

The lesson (#45) is that a learned correction must be capped, at test time, by
what the data at test time can actually know — not by what training happened to
contain. That cap is a *computed* quantity (S1 contraction), and the
architecture below wires it in as a hard multiplier with zero parameters.

---

## The three primitives, each traced to a result

### P1 — EP-neuron (activation = truncated-Gaussian moment match) → bug-fix #40

`ep_neuron(mc, vc, h, sgn=+1, slack=None) -> (tau, nu)`.

A LiDAR ray that returns at range `s_hit` asserts that for every `s < s_hit` the
ray passed *above* the surface: `phiᵀa < h`, a one-sided inequality on a linear
functional of the coefficients. Under a Gaussian prior this truncation is
handled by one expectation-propagation site — a Gaussian message on that scalar
row — whose parameters are the truncated-Gaussian moment match. This is not a
chosen nonlinearity; it is the moment match, derived:

```
alpha = sgn (h - mc) / st,   st = sqrt(vc + slack^2),   lam = phi(alpha)/Phi(alpha)
E[x | event]   = mc - sgn (vc/st) lam
Var[x | event] = vc - (vc^2/(vc+slack^2)) lam (lam + alpha)     (= vc(1 - lam^2 - alpha lam) at slack=0)
tau = 1/Var - 1/vc   (>= 0),   nu = E/Var - mc/vc
```

Bug-fix #40: get either sign wrong and `Var > vc`, so `tau < 0`; the
`clamp_min(0)` then zeros every site and the constraints silently do nothing —
the exact symptom the first shadow-constraint script showed (active fraction
0.00). The signs above are the fix; the clamp is only a guard. The formulas are
**copied verbatim** from the validated `scripts/d3_e5_shadow_constraints.py::ep_solve`
(the probit `slack` variant softens the height, which is built from a noisy
range). The unit's identity property — `alpha → +∞ ⇒ lam → 0 ⇒ tau=0, nu=0` —
is what makes an ignorable (MCAR) site change the solve by exactly zero.

### P2 — information gate (correction δ emitted as c⊙δ) → S1 contraction

`contraction_gate(Phi, gam, s2) -> c`  and  `GatedDelta`.

S1 (`scripts/d3_s1_posterior_contraction.py`) measured what the data can know:
the per-mode posterior contraction

```
c_i = 1 - Var_post(a_i)/gamma_i = gamma_i phi_i^T K^-1 phi_i,   K = Phi diag(gam) Phi^T + s2 I,
```

`c_i ∈ [0,1]`; `c_i = 0` means the observations say nothing about mode `i`. We
compute it in the **kernel form** (one `n×n` solve, `n≈600`) rather than the
precision form `1 − [(ΦᵀΦ/s²+diag(1/γ))⁻¹]_ii/γ_i` (a `2B×2B≈3800` solve that
also forms `1/γ`, which explodes for the near-zero beamform weights). The two
are equal by Woodbury; the kernel form is cheaper and more stable.

`GatedDelta` wraps any learned module `m(features)→δ`, bounds it
`δ = scale·tanh(m) ∈ [−scale, scale]`, and emits `out = a_physics + c ⊙ δ`. The
learned correction to mode `i` is therefore capped at `c_i·scale`: a mode the
data cannot see cannot be moved. This is theorem T1 and it makes the
support-memorization failure impossible by construction.

### P3 — wave-group equivariance (complex-linear, magnitude nonlinearity) → G3

`phase_rotate(a, om, dt)`  and  `PhaseEquivariantBlock`.

The only input/state representation that ever transferred to real data (G3
lesson) is the amplitude on the **normalized dispersion shell** `(k/k_p, ψ)`. On
that shell a time shift `Δ` is the exact physics evolution `a_k → a_k·e^{−iω_kΔ}`
— a per-mode phase rotation and a group action. A module that respects this
symmetry must commute with the diagonal unitary `diag(e^{−iω_kΔ})`. That forces:

- **complex-linear and diagonal** in the mode basis (a full complex matrix would
  commute only with a *global* phase, not the per-mode rotation dispersion
  dictates; a real bias would not rotate at all — both are forbidden by the
  symmetry, not by taste), and
- **magnitude-only nonlinearities** — `|a_k|` is phase-invariant, so any gain
  `g(|a_k|)` commutes with the rotation.

`PhaseEquivariantBlock` is `a_k → g(|a_k|)·w_k·a_k` with `w_k` a learned complex
weight and `g` a learned magnitude MLP; it commutes exactly (test EQ). Mode
mixing is deliberately absent — its expressiveness cost is an open empirical
question, not something this note settles.

---

## Verbatim test output (`python3 scripts/d3_n5_tests.py`, exit 0)

```
device cuda  dtype torch.float64

[R1] reduction: architecture(delta=0, no sites) == naive posterior mean
     primal vs dual posterior-mean field  max|diff| = 2.27e-14
     architecture vs naive field (1200 q) max|diff| = 2.27e-14   gate 1e-06
     PASS

[ID] MCAR / zero-site identity: tau = 0 sites change nothing
     121 constraint rows, tau = 0 : max|diff| = 0.0e+00 (must be exactly 0)
     ep_neuron on satisfied events (median alpha = 116.9 sigma): max site tau = 0.0e+00, solve max|diff| = 0.0e+00
     PASS

[EQ] phase equivariance: block commutes with per-mode rotation
     block: max|rotate.apply - apply.rotate| = 1.42e-14   gate 1e-06
     physics group law: max|R(s)R(t) - R(s+t)| = 1.78e-15
     PASS

[C3] support-memorization impossibility on a 2.5 s window
     T=20.0s  sub-floor(k<0.35kp) c mean 0.0867 max 0.2055   in-band(1.0-1.6kp) c 0.038
     T= 2.5s  sub-floor(k<0.35kp) c mean 0.0209 max 0.0328   in-band(1.0-1.6kp) c 0.025
     sub-floor contraction collapse 20s -> 2.5s: 4.15x
     worst-case injectable amplitude (gated)  = 9.2892e-03
     theoretical bound  sum c_i |delta| sqrt(gam_i) = 9.2892e-03
     sub-floor prior amplitude (ungated net)  = 4.4478e-01
     injectable / prior = 2.09%   (max sub-floor c 0.0328 < 0.06)
     PASS
```

- **R1** confirms the reduction: with every learned part disabled the
  architecture *is* the naive posterior mean, computed by an independent
  (kernel) route, agreeing to 2.27e-14 on 1,200 query points — far inside the
  1e-6 fp64 gate. The primal (normal-equations) and dual (kernel) posterior
  means also agree to 2.27e-14, so the two solve routes are the same map.
- **ID** confirms the MCAR/zero-site identity exactly (0.0), and that the
  EP-neuron self-deactivates (tau = 0) on constraints satisfied by ~117σ.
- **EQ** confirms per-mode phase equivariance (1.42e-14) and the exact group law
  of the physics evolution (1.78e-15).
- **C3** is theorem T1 in numbers; see below.

---

## Theorem T1 — support-memorization impossibility

**Setup.** For coefficient `i` the architecture emits `out_i = phys_i + c_i δ_i`
with `δ_i = scale·tanh(m_i(features))`, `Δmax := scale`, and
`c_i = γ_i φ_iᵀK⁻¹φ_i ∈ [0,1]` computed from the *current* `(Φ, γ, s²)`.

**Claim.** If `c_i ≤ ε` then `|out_i − phys_i|·√γ_i ≤ ε·Δmax·√γ_i`. Hence over a
band `B` with `c_i ≤ ε ∀ i∈B`, the amplitude the network can inject is
`≤ ε·Δmax·Σ_{i∈B}√γ_i` — a fraction `≤ ε·Δmax` of the band's prior amplitude.

**Proof (≤10 lines).**
1. `out_i − phys_i = c_i δ_i` (definition of `GatedDelta`).
2. `δ_i = scale·tanh(·)` ⇒ `|δ_i| ≤ scale = Δmax`.
3. `c_i ≥ 0` ⇒ `|out_i − phys_i| = c_i|δ_i| ≤ c_i Δmax`.
4. `√γ_i ≥ 0` ⇒ `|out_i − phys_i|·√γ_i ≤ c_i Δmax √γ_i`.
5. `c_i ≤ ε` ⇒ `≤ ε Δmax √γ_i` (per mode).
6. Sum over `B`: `Σ_B |out_i − phys_i|√γ_i ≤ ε Δmax Σ_B √γ_i`. ∎

**Assumptions.** (a) the output is exactly `phys + c⊙δ` with no other path
writing `out_i`; (b) `c_i` is *computed* from the true current `(Φ, γ, s²)`, not
predicted by the network; (c) `δ` is tanh-bounded. Under these, no training set
and no adversarial feature vector can move an uninformed mode.

**Why the killed failure cannot recur.** The learned prior injected energy into
modes the test geometry could not constrain (OOD −45 %). Here `c_i` is evaluated
at test time from the actual observation geometry; a test scene that does not
constrain mode `i` has `c_i ≈ 0` there, so the correction is silenced regardless
of what was learned. Support memorization has nowhere to land.

**Measured (test C3).** On a 2.5 s window the sub-floor band `k < 0.35 k_p` is
uninformed (S1): `max c = 0.0328 < 0.06`. A full-scale adversary (`Δmax = 1`,
`δ = ±1` everywhere, pushed through the actual `GatedDelta`) injects amplitude
`9.2892e-03`, exactly the bound `Σ_B c_i|δ|√γ_i = 9.2892e-03`, against a
sub-floor prior amplitude of `4.4478e-01` — **2.09 %**. The same band at a 20 s
window has `mean c = 0.0867`; the 2.5 s window collapses it **4.15×**,
reproducing S1's finding that the long waves are constrained by observation
*time*, not by the bank floor.

---

## Theorem T2 — UQ floor (provable part) and what remains a design goal

`variance_floor(c, gam) = (1 − c)·gam`.

**Provable.** For the linear-Gaussian model the posterior variance of coefficient
`i` is exactly `Var_post(a_i) = (1 − c_i)γ_i` (definition of contraction). If a
learned variance head reports `var_out_i = max(net_var_i, (1−c_i)γ_i)`, then
`var_out_i ≥ Var_post(a_i)` by construction. The reported uncertainty can never
claim to know a mode better than the data allow; on an uninformed mode
(`c_i ≈ 0`) the floor is the *full* prior `γ_i`, so overconfident collapse — the
direction that produced z-std 68,710 — is impossible.

**Design goal, marked honestly.** This bounds the per-mode *coefficient*
variance, not the end-to-end field z-std. Field calibration additionally
depends on how coefficient variances propagate through the reconstruction `Φ_q`,
on the cross-mode posterior covariance the diagonal floor ignores, on the
beamform prior's own miscalibration, and on non-Gaussian effects (HOS,
preferential sampling — G3 already showed z-std 2.4, miscalibrated). T2 removes
the collapse direction; it does **not** by itself deliver z-std ∈ [0.9, 1.1].
That requires the full-pipeline calibration test and is unmeasured here. The
variance *head* is specified, not wired end-to-end; `variance_floor` ships so the
floor is a real, testable function, not a promise.

---

## Pre-registered gates for any FUTURE training of this architecture

Frozen now, before any weights are trained. A run that fails any of these is not
reported as a positive.

- **G-R1 reduction (machine precision).** With all learned parts disabled the
  architecture must equal the physics posterior mean to `≤ 1e-6` fp64
  (currently 2.27e-14). The analogue of "gain = 1 ⇒ |diff| = 0". If it drifts,
  the trained and untrained paths are not the same map and there is no verdict.
- **G-EQUIV negative controls (equivalence, not significance).** On a stationary
  sea and under ignorable (MCAR) missingness the learned parts must not help
  beyond noise: `|gain| ≤ 0.002` (the E5/Door-1b equivalence framing — a
  significance test against an exactly-zero null is guaranteed to fail once `n`
  resolves any residual confound, so the gate is on magnitude).
- **G-BASELINE (must beat no-learning).** The trained net must beat both the
  physics MMSE and a warm-started no-learning streaming EP on the
  accuracy–latency frontier. If it does not, the learning is downgraded to an
  engineering note (E5 G5).
- **G-CEILING (lesson #45).** The energy any learned correction injects into a
  band must sit *below* that band's computed information ceiling at test time:
  `Σ_B |c_i δ_i|√γ_i ≤ Σ_B c_i √γ_i`. T1 guarantees this by construction; it is
  also enforced as a runtime monitor, and a run whose corrections approach or
  exceed the ceiling is rejected as support memorization.
- **G-HOS (standing instruction).** The headline must reproduce on HOS
  nonlinearity; a positive that appears only on linear synthetic seas is not
  reported as the result.

---

## NO NOVELTY CLAIM

Deep unfolding / algorithm unrolling, KalmanNet, learned EP messages (Heess,
Tarlow & Winn, *Learning to Pass Expectation Propagation Messages*, NeurIPS
2013), and steerable / complex-valued networks (deep complex networks; harmonic
/ steerable CNNs) are all prior art, and each of the three primitives here has
close relatives in that literature. A separate litcheck task is running; until
it reports, this document and the code make **no** novelty claim of any kind.
The only defence offered here is that the composition is *forced* by this
project's own measurements (#40, S1, G3) and is verifiable to machine precision.

---

## What is NOT here (limitations, cold)

- **No training, no accuracy result.** This is verified scaffolding. It does not
  show the architecture beats the physics MMSE; that requires the pre-registered
  training run and is untested.
- **The EP-neuron is the single-site moment match**, not the solver. The full
  multi-sweep EP loop (cavity, damping) lives in `d3_e5_shadow_constraints.py`;
  the core exposes the neuron and `physics_posterior` (the augmented solve), not
  a re-implemented sweeper.
- **The variance head (T2) is specified, not wired.** `variance_floor` exists;
  end-to-end z-std calibration is unmeasured.
- **C3's absolute contraction values depend on the prior/noise normalization**
  (flat prior, `NOISE = 0.05`). They differ from S1's inflated-noise numbers,
  but the *structure* (4.15× collapse at 2.5 s, sub-floor `< 0.06`) reproduces,
  and the impossibility bound `realized ≤ Σ c√γ` is independent of the
  normalization.
- **The gate uses the diagonal (marginal) contraction**, ignoring cross-mode
  posterior covariance. This is exactly the S1 quantity and is a conservative,
  honest per-mode cap, not the full joint story.
- **The equivariant block is diagonal (per-mode) by force of the symmetry.**
  Whether a per-mode-only learned map is expressive enough is an open empirical
  question.
