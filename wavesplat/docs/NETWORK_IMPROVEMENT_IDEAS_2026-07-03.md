# Network-improvement ideas (no RELAX hybrid) — 2026-07-03

## The bottleneck, stated precisely
On a structured, few-component ("simple") sea with sparse observations, the posterior
`p(components | obs)` is genuinely **multimodal**: many distinct component sets fit the
sparse data comparably (k-space aliasing / near-degeneracy). DPS draws samples from
different modes; their fields differ; the **posterior mean (BMA) blurs** → nRMSE ~0.21.
RELAX avoids this by *committing greedily to one consistent set* (a sharp point estimate)
→ nRMSE ~0.05–0.09.

**Ruled out (empirically):** more data (150k, no change), prior cardinality coverage
(mixed prior generates sparse, nowcast unchanged), guidance (0.05 optimal, higher hurts),
and deployable likelihood-selection (overfits, worse than BMA). The loss floor (~0.128)
is data/capacity-independent. ∴ the gap is the **estimator**, not the prior.

**Key realisation:** a *generative posterior mean* is the wrong point estimator on a
multimodal posterior. The fix is a network that yields a **sharp point estimate** (mode /
MAP / amortized regression) while the generative model supplies **calibrated UQ** around it.

## Options, ranked by fit to the mechanism

### ★1. Discriminative-proposal + generative-UQ cascade (all-neural, reuses assets)
Stage-1: an amortized **discriminative** component network regresses the sharp component
set directly (no posterior-mean blur — it outputs *one* answer). We already have
`component_distill_mixed.pt`, which **matches/beats RELAX on synthetic** incl. simple seas
(4-comp: net 0.946–0.984 vs RELAX 0.905–0.992). Stage-2: the set-diffusion model, **seeded
from / conditioned on** the proposal, provides posterior samples → calibrated UQ (and can
refine off-grid). Point estimate is sharp (discriminative); UQ is generative. **Not** a
physics hybrid — both stages are learned. Highest ROI: combines two existing assets and
directly removes the mean-blur.

### ★2. Conditional (amortized) set-diffusion with observation cross-attention
Replace "unconditional prior + external DPS gradient" with a denoiser whose atom tokens
**cross-attend to observation tokens (x,y,t,z)**. Learns `p(components|obs)` end-to-end, so
the posterior is *as sharp as the data allows* (DPS's manifold-gradient is only a crude
correction). Take the point estimate as a **mode-seeking readout** (below), not the mean.
Most principled; larger build. Note: even a conditional model's *mean* blurs on multimodal
posteriors — so pair with #3.

### 3. Mode-seeking readout instead of the mean
The posterior mean minimises MSE only for ~unimodal posteriors. For multimodal ones use the
**mode**: (a) low-temperature / annealed sampling to concentrate mass; (b) cluster the K
samples (in component space, permutation-invariant) and take the dominant-cluster
representative; (c) a short MAP ascent (maximise prior+likelihood) initialised from the
best-clustered sample. Keeps a generative UQ head. Directly attacks the blur; cheap-ish.

### 4. Better posterior sampler than vanilla DPS
ΠGDM / covariance-corrected guidance (Song 2023), or predictor–corrector with a few
Langevin steps per diffusion step, or annealed guidance. More *accurate* posterior
sampling. Cheapest to try, but it samples the *same* multimodal posterior more faithfully →
the mean still blurs. Useful complement, not a standalone fix.

### 5. Sharper prior: set-loss + native cardinality latent
Train with a permutation-invariant **set loss** (Sinkhorn/Chamfer) + **energy-weighting**
(dominant atoms weighted up), and factor an explicit **cardinality latent** instead of
32-atoms-with-null-padding (which wastes capacity and smears sparse spectra). Sharpens the
prior on sparse configs; complementary to #1/#2.

## Recommendation
Start with **#1 (discriminative proposal + generative UQ cascade)** — it reuses a
RELAX-matching asset for the sharp point estimate and the set-diffusion for calibrated UQ,
so it can close the simple-sea gap *and* keep the complex/nonlinear + UQ wins, entirely with
learned components. If a single end-to-end model is preferred, build **#2 + #3**
(conditional set-diffusion with a mode-seeking readout). #4/#5 are low-cost complements.

Caveat (sim2real, from prior work): `component_distill` beat RELAX on *synthetic* but failed
to transfer to real broadband LiDAR. Any discriminative stage must be trained on a
realistic distribution (broadband + noise + forward-cone + scale-normalised) before real use.

## Empirical outcome (2026-07-03) — the simple-sea gap did NOT close by any learned route
Tried exhaustively (multi-seed, train-enough, hyperparameters), all measured on the real
episodes; simple-sea nRMSE ≈0.19-0.25 vs RELAX 0.05-0.09 throughout:
- **#1 discriminative component net** (`d3_component_net.py`, Fourier-feature DETR, Chamfer+
  Hungarian, 6000 steps): **mean-collapse** — field loss flat 0.061, matching loss flat 1.71,
  eval 0.248. Same for d34's raw-coord net (0.248).
- **amortized field regressors** (all 3 existing, and a fresh **episode-matched** one with UQ
  head trained on the real footprint, `d3_amortized_field.py`, 5000 steps): **mean-collapse**
  to 0.24-0.25, even on the *training* distribution (NLL gamed by variance inflation).
- **#3 mode-seeking on the DPS** (`tmp/mode_seek.py`, medoid / correlation-cluster of 16
  samples): **worse than BMA** (0.229/0.219 vs 0.160). On simple seas the 16 samples are
  mutually uncorrelated (cluster 1/16) and the oracle-best sample (0.25) is *worse* than BMA
  (0.19) — the DPS posterior is diffuse and **contains no RELAX-level solution**, so no
  readout recovers one. BMA is already the best learned readout.
- data scale-up + mixed-cardinality prior (48k, 150k/d320): overall unchanged/slightly worse.

### ★★ BREAKTHROUGH — change the INPUT domain (beamform) + detection + iteration
The above all *regress from raw obs points* → mean-collapse. RELAX succeeds because it works
in the **beamform (k-space matched-filter) domain** where components are localized peaks.
Adopting that domain as the **input** and reframing as **detection** cracks it
(`scripts/d3_beamform_net.py`):
- Compute the dispersion-focused beamform power map `B(kx,ky)` from obs.
- A CNN outputs a component-location **heatmap** (detection, not regression → no collapse;
  loss converges to 0.003; the CNN learns to deconvolve the beamformer sidelobes = learned
  CLEAN). Peaks → off-grid refine → **subtract residual → re-detect** (iterative, 2 passes).

Result (2-pass): **BF3_5c 0.076 ≈ RELAX 0.075; BF4_5c 0.057 ≈ RELAX 0.051** — the FIRST
learned estimator to reach RELAX-level sharpness on simple seas (prior best DPS 0.19,
mean-collapse 0.25). On the GIF the beamform-detect field is visually indistinguishable from
RELAX. Complex seas remain weak (0.25-0.29 — the beamform map is diffuse with many
overlapping components) → **DPS's domain**. So a **fully-learned regime-complete** picture
emerges: simple = iterative beamform-detect (≈RELAX), complex/nonlinear = DPS + UQ — both
learned, no physics-estimator hybrid (beamforming is a fixed linear front-end). Caveat: the
off-grid refine + residual iteration are a RELAX-shaped scaffold around the learned detector.

### Superseded conclusion (kept for the record)
Before the beamform-domain result, we had concluded *no* learned point-regression,
generative-mean, or generative-mode-seeking reaches RELAX-level on simple seas — true, but
the fix was to change the **input representation**, not the estimator family. Learned point regressors give the posterior *mean* (blurs on
underdetermined fields → mean-collapse); the DPS posterior is over-dispersed there and holds
no sharp solution; only RELAX's greedy *commitment to one mode* is sharp. The learned
generative approach's genuine, defensible value is **complex/nonlinear seas (beats RELAX at
equal obs) + calibrated UQ (which RELAX cannot give)**. The honest system is regime-aware:
generative-DPS+UQ for complex/nonlinear, physics (RELAX) for simple seas. Best learned config
in place = DPS-BMA, `_long` prior, guid 0.05, Woodbury likelihood (overall 0.176, UQ +0.36).
