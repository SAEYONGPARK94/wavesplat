# DCSpecNet Literature Survey (2026-05-14)

## Key Gap Identified
No papers combine deep unfolding + dispersion constraints + set networks
for ocean wave spectral estimation. Two threads exist separately:
(A) deep unfolding for sparse spectral estimation (DOA, CS)
(B) physics-constrained NN for ocean waves (PINNs with dispersion)

## 1. Deep Unfolding / Algorithm Unrolling

### LISTA (Gregor & LeCun, ICML 2010)
- Unfold ISTA into fixed-depth NN. Learnable W, thresholds.
- 10 layers matches 100+ ISTA iterations.
- Code: github.com/faisaljayousi/LISTA-SC

### ISTA-Net+ (Zhang & Ghanem, CVPR 2018)
- Unfold ISTA for image CS with learned conv proximal operators + skip connections.
- Code: github.com/jianzhangcs/ISTA-Net-PyTorch

### LISTA-CPSS (Chen et al., NeurIPS 2018 Spotlight)
- Linear convergence proof for unfolded ISTA. Led to ALISTA (ICLR 2019).
- Code: github.com/VITA-Group/LISTA-CPSS

### Shlezinger & Eldar Survey (F&T Signal Processing, 2023)
- Taxonomy: DNN-aided inference / deep unfolding / DNN-augmented optimization.
- arXiv:2306.04469

### Deep Unfolding for DOA
- LISTA for DOA on metasurface (Remote Sensing, 2025) — LISTA layers for angular spectrum
- ANM-ADMM-Net (Remote Sensing, 2023) — gridless DOA via unfolded atomic norm ADMM
- DeepFPC (Signal Processing, 2020) — unfolded FPC for 1-bit CS DOA
- ADMM-Net beamforming (Remote Sensing, 2024) — unfolded ADMM + atomic norm

## 2. Set-to-Parameter Networks

### DeepSets (Zaheer et al., NeurIPS 2017)
- rho(SUM(phi(x_i))) — universal approx for permutation-invariant functions.
- Code: github.com/manzilzaheer/DeepSets

### Set Transformer (Lee et al., ICML 2019)
- Self-attention (SAB), inducing points (ISAB), pooling (PMA).
- O(nm) via ISAB. Captures pairwise interactions.
- Code: github.com/juho-lee/set_transformer

### PointNet (Qi et al., CVPR 2017)
- Shared MLPs + max-pool for 3D point clouds.
- Code: github.com/charlesq34/pointnet

## 3. Physics-Constrained Output Layers

### Beucler et al. (PRL 2021) — Hard constraints via architecture
- Last layer constructed to enforce analytic constraints (energy conservation) to machine precision.
- Architecture-based >> loss-based for hard constraints.
- **Directly applicable**: dispersion w = sqrt(g|k|tanh(|k|d)) as deterministic output.

### ICNN (Amos, Xu, Kolter, ICML 2017)
- Output guaranteed convex w.r.t. inputs via non-negative weights + convex activations.
- Code: github.com/locuslab/icnn
- **Advisor connection**: 이신형 교수 ICNN for fuel (2026) paper.

### Valente et al. (Commun. Phys. 2025) — Post-hoc projection onto manifold
- Differentiable projection after NN forward pass. Model-agnostic.

### Yang et al. (NeurIPS 2020) — Learned projection
- Learn constraint manifold from data. Code available.

## 4. Wave NN (Ehlers group dominates)

### Ehlers et al. (Phys. Rev. Fluids, 2025) — PINN wave data assimilation
- NN(x,t) → (eta, phi). PDE residuals in loss. No labeled data needed.
- Handles nonlinear, dispersive waves. No public code.

### Ehlers et al. (Ocean Eng., 2024) — FNO/U-Net for radar-to-wave
- U-Net and FNO trained on HOS + radar sim. FNO has spectral bias advantage.

### Ehlers et al. (Phys. Fluids, 2025) — PINO real-time
- Neural Operator with physics loss. Generalizes across sea states.

### Desmars (PhD thesis, 2020, ECN)
- Inverse problem with HOS model + LiDAR. Not NN-based.
- Established the LiDAR simulation framework that Ehlers builds on.
