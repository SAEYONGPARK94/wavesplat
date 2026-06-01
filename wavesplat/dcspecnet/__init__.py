"""
DCSpecNet — Dispersion-Constrained Neural Spectral Estimator.

Novel architecture for ocean wave spectral estimation from sparse LiDAR.
Key innovations:
  1. Hard dispersion constraint: w = sqrt(g|k|tanh(|k|d))
  2. Set Transformer encoder for unordered point clouds
  3. Adaptive N via existence probability (slot-based)
  4. Phase propagation prior for temporal coherence
"""
from dcspecnet.model import DCSpecNet
