# Learned off-grid estimator — feasibility verdict (2026-06-28)

**Design:** `docs/learned_offgrid_design.md`. **Code:** `scripts/d34_learned_offgrid.py`.
Set-transformer encoder over the obs window → DETR-style 32 atom queries → off-grid
(kx,ky,A,φ) + dispersion hard constraint → render as a sparse cosine sum (sharp by
construction). Trained on synthetic JONSWAP with field-reconstruction + obs-consistency +
a wavenumber Chamfer matching loss (anti-collapse). 8000 steps, ~3 min on RTX 4080.

## Result — pure feedforward amortization FAILS (theory-consistent negative)
| | value |
|---|---|
| match loss (wavenumber placement) | 0.16 → **0.003** (converges — atoms ARE placed near truth, coarsely) |
| field loss (reconstruction, ~nRMSE²) | **0.06, flat from step 200 to 8000** (never drops) |
| benchmark eval nRMSE (6 episodes) | **0.248** (BF4_15c 0.254 … BF4_5c 0.250) |
| vs DPS (~0.206) / RELAX (0.119) / zero-predictor (~0.25) | **worse than DPS, ≈ zero-predictor** |

## Why (this is the CRB precision barrier, made concrete)
- The matching loss converges (0.003 → normalized k-distance ≈ 0.05 rad/m): the network
  learns the **coarse** spectral structure easily.
- But field reconstruction needs wavenumbers accurate to **~1/D_eff ≈ 0.005 rad/m**
  (10× finer): a kx error of 0.05 over the 180 m footprint is ~9 rad of phase — the field
  is then uncorrelated with truth. So coarse placement ⇒ zero-predictor field.
- The field/obs loss is **non-convex and oscillatory** in the fine wavenumber/phase, so a
  feedforward map cannot descend to the required precision — the training field loss is
  dead-flat while the matching loss converges. (Not a train/eval transfer issue: it fails
  on the training distribution itself.)
- This is exactly why non-learned **RELAX wins**: it reaches precision by *iterative*
  matching-pursuit + per-component local refinement (each component refined in a locally
  convex neighborhood), which a single feedforward pass cannot replicate.

## Takeaway for the thesis (strengthens it)
The hard, CRB-limited part of the inverse problem is **off-grid component identification at
the precision 1/D_eff** — and it is *not* cheaply amortizable by a feedforward network.
Coarse structure is learnable; precision is not (non-convex). This is a third, independent
confirmation that the lever is iterative off-grid identification, not learning capacity:
- d30: learned prior ties at equal budget (over-smooths).
- d33: more observations don't help the learned prior (over-smoothing, not cap).
- d34: a learned *off-grid* net plateaus at the zero-predictor (precision barrier).

## Hybrid tested (learned warm-start → RELAX) — neutral
`scripts/d35_hybrid.py`. Variants on complex episodes (same window):
L = learned→refine only; B = RELAX (beamform init); H = learned atoms warm-start
relax_step→refine; HU = union(RELAX, learned)→refine.

| set | L | B (RELAX) | H (learn-warm) | HU (union) | oracle |
|---|---|---|---|---|---|
| 30c × 10 (rough 8k model) | 0.209 | 0.179 | **0.170 (−5%)** | 0.170 | 0.029 |
| 15c+30c × 20 (v2 20k model) | 0.199 | 0.143 | **0.141 (−1.4%)** | 0.157 | 0.027 |

- Refining the learned atoms (L) recovers them from 0.25 (feedforward) to ~0.17–0.20, but
  not to RELAX level.
- The learned **warm-start (H)** is essentially **neutral overall** (−1 to −5%, within
  seed noise): large wins on some episodes (BF4_30c_s0 0.196→0.138), losses on others.
- The improved model (Hungarian one-to-one regression, NQ=48, 20k steps) did **not** beat
  the rough one — proposal wavenumber precision **plateaus at ~0.04 rad/m** (hung-loss
  0.0024 flat), still ~8× coarser than the 1/D_eff the field needs. Even *direct
  supervision* cannot amortize the precision: the obs→precise-wavenumber map is
  ill-conditioned (the CRB barrier), so only iterative per-component refinement reaches it.

**Conclusion:** learning does not robustly improve the deployable estimator on accuracy —
a 4th confirmation of the thesis. The genuine niche for learning is elsewhere (the
observation-scarce regime — see `scarcity_sweep.json`), not headline accuracy.
