"""
spectral_gaussian.py
Spectral Gaussian wave field representation (PyTorch).

Forward model (linear sinusoidal, NOT Gerstner):
    z(x, y, t) = Σᵢ wᵢ · cos(kxᵢ·x + kyᵢ·y - ωᵢ·t + φᵢ)

Free parameters per component: wᵢ, kxᵢ, kyᵢ, φᵢ (4 each).
ωᵢ is computed from |kᵢ| via dispersion relation (hard constraint).

The intentional model mismatch with Gerstner GT is O(ε²) ≈ 0.25% at BF4.

Note on Q-constraint (Principle #4):
    The Q-parameter constraint (ΣQᵢωᵢAᵢ ≤ 1) prevents particle crossing in
    Gerstner waves, which have horizontal displacement. This linear model has
    NO horizontal displacement — only vertical elevation z(x,y,t). Therefore
    Q-constraint is structurally inapplicable here. Wave steepness is bounded
    by L1 regularization on weights and an optional amplitude clamp.
    See wavesim/wave_generator.py for Q-constraint in the GT simulator.
"""

import torch
import torch.nn as nn
import math

from wavesplat.model.dispersion import dispersion_omega, G
from wavesplat.model.stokes2 import Stokes2Transfer
from wavesplat.model.neural_corrector import (
    NeuralCorrector, NeuralFieldCorrector, NeuralFieldCorrectorV2
)


class SpectralGaussianModel(nn.Module):
    """
    Spectral Gaussian wave field model.

    Parameters:
        w:   (N,) amplitudes / weights [m]
        kx:  (N,) x-wavenumber components [rad/m]
        ky:  (N,) y-wavenumber components [rad/m]
        phi: (N,) phase offsets [rad]
    """

    def __init__(self, n_gaussians: int, depth: float = 50.0,
                 use_dispersion: bool = True, use_stokes2: bool = False,
                 use_neural_corrector: bool = False):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.depth = depth
        self.use_dispersion = use_dispersion
        self.use_stokes2 = use_stokes2
        self.use_neural_corrector = use_neural_corrector
        self._stokes2_transfer = None  # lazy init
        self._neural_corrector = None  # loaded from checkpoint

        # Free parameters — initialized small; proper init via initializer.py
        self.w = nn.Parameter(torch.randn(n_gaussians) * 0.01)
        self.kx = nn.Parameter(torch.randn(n_gaussians) * 0.05)
        self.ky = nn.Parameter(torch.randn(n_gaussians) * 0.05)
        self.phi = nn.Parameter(torch.randn(n_gaussians) * math.pi)

        # A1 ablation: ω as free parameter (no dispersion constraint)
        if not use_dispersion:
            self.omega_free = nn.Parameter(torch.randn(n_gaussians) * 0.5)

    def load_neural_corrector(self, checkpoint_path: str):
        """Load a trained neural field corrector from checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        arch = ckpt.get('arch', 'v1')
        if arch == 'v2':
            self._neural_corrector = NeuralFieldCorrectorV2(
                hidden_dim=ckpt['hidden_dim'],
                n_heads=ckpt.get('n_heads', 4),
                n_attn_layers=ckpt['n_layers'],
            )
        else:
            self._neural_corrector = NeuralFieldCorrector(
                hidden_dim=ckpt['hidden_dim'],
                n_layers=ckpt['n_layers'],
            )
        self._neural_corrector.load_state_dict(ckpt['model_state_dict'])
        self._neural_corrector.eval()
        for p in self._neural_corrector.parameters():
            p.requires_grad_(False)
        self.use_neural_corrector = True

    def update_stokes2_transfer(self):
        """Recompute Stokes 2nd-order transfer functions from current kx/ky.
        Call this after kx/ky change (e.g., after init or coarse-to-fine)."""
        if not self.use_stokes2:
            return
        with torch.no_grad():
            kx_np = self.kx.detach().cpu().numpy()
            ky_np = self.ky.detach().cpu().numpy()
        self._stokes2_transfer = Stokes2Transfer(kx_np, ky_np, self.depth)

    def _omega(self) -> torch.Tensor:
        """Compute ω — via dispersion or as free parameter (ablation A1)."""
        if self.use_dispersion:
            k_mag = torch.sqrt(self.kx ** 2 + self.ky ** 2 + 1e-12)
            return dispersion_omega(k_mag, self.depth)
        else:
            return torch.abs(self.omega_free)  # ω must be positive

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                t: float) -> torch.Tensor:
        """
        Evaluate wave field at observation points.

        Args:
            x: (M,) x-coordinates [m]
            y: (M,) y-coordinates [m]
            t: time [s]

        Returns:
            z: (M,) predicted wave elevation [m]
        """
        omega = self._omega()  # (N,)

        # phase = kx·x + ky·y - ω·t + φ  → shape (M, N)
        phase = (self.kx[None, :] * x[:, None]
                 + self.ky[None, :] * y[:, None]
                 - omega[None, :] * t
                 + self.phi[None, :])

        # z = Σᵢ wᵢ · cos(phaseᵢ)  → (M,)
        z = torch.sum(self.w[None, :] * torch.cos(phase), dim=1)

        # Stokes 2nd-order correction
        if self.use_stokes2 and self._stokes2_transfer is not None:
            z2 = self._stokes2_transfer.correction(
                x, y, t, self.w, self.kx, self.ky, self.phi, omega)
            z = z + z2

        # Neural field corrector: additive 3rd+ order correction
        if self.use_neural_corrector and self._neural_corrector is not None:
            z_nn = self._neural_corrector(
                self.kx, self.ky, self.w, self.phi, omega,
                self.depth, x, y, t)
            z = z + z_nn

        return z

    def predict_grid(self, grid_x: torch.Tensor, grid_y: torch.Tensor,
                     t: float) -> torch.Tensor:
        """
        Evaluate wave field on a regular 2D grid.

        Args:
            grid_x: (Nx,) x-coordinates
            grid_y: (Ny,) y-coordinates
            t: time [s]

        Returns:
            Z: (Ny, Nx) wave elevation
        """
        Xg, Yg = torch.meshgrid(grid_x, grid_y, indexing='xy')
        x_flat = Xg.reshape(-1)
        y_flat = Yg.reshape(-1)

        z_flat = self.forward(x_flat, y_flat, t)
        return z_flat.reshape(len(grid_y), len(grid_x))

    def get_spectrum_params(self):
        """Return current spectral parameters as numpy arrays."""
        with torch.no_grad():
            k_mag = torch.sqrt(self.kx ** 2 + self.ky ** 2).cpu().numpy()
            omega = self._omega().cpu().numpy()
            return {
                'w': self.w.cpu().numpy().copy(),
                'kx': self.kx.cpu().numpy().copy(),
                'ky': self.ky.cpu().numpy().copy(),
                'phi': self.phi.cpu().numpy().copy(),
                'k_mag': k_mag,
                'omega': omega,
            }

    def add_gaussians(self, n_new: int, init_kx=None, init_ky=None,
                      init_w=None, init_phi=None):
        """
        Add new Gaussian components (for coarse-to-fine refinement).
        Preserves existing parameters.
        """
        device = self.w.device

        new_w = init_w if init_w is not None else torch.randn(n_new, device=device) * 0.01
        new_kx = init_kx if init_kx is not None else torch.randn(n_new, device=device) * 0.05
        new_ky = init_ky if init_ky is not None else torch.randn(n_new, device=device) * 0.05
        new_phi = init_phi if init_phi is not None else torch.randn(n_new, device=device) * math.pi

        self.w = nn.Parameter(torch.cat([self.w.data, new_w]))
        self.kx = nn.Parameter(torch.cat([self.kx.data, new_kx]))
        self.ky = nn.Parameter(torch.cat([self.ky.data, new_ky]))
        self.phi = nn.Parameter(torch.cat([self.phi.data, new_phi]))
        if not self.use_dispersion:
            new_omega = torch.randn(n_new, device=device) * 0.5
            self.omega_free = nn.Parameter(
                torch.cat([self.omega_free.data, new_omega]))
        self.n_gaussians += n_new

    def remove_gaussians(self, indices_to_keep: torch.Tensor):
        """
        Remove Gaussian components by keeping only specified indices.
        Used for adaptive N pruning (remove weak/redundant components).

        Args:
            indices_to_keep: (K,) integer tensor of indices to retain
        """
        device = self.w.device
        self.w = nn.Parameter(self.w.data[indices_to_keep].clone())
        self.kx = nn.Parameter(self.kx.data[indices_to_keep].clone())
        self.ky = nn.Parameter(self.ky.data[indices_to_keep].clone())
        self.phi = nn.Parameter(self.phi.data[indices_to_keep].clone())
        if not self.use_dispersion:
            self.omega_free = nn.Parameter(
                self.omega_free.data[indices_to_keep].clone())
        self.n_gaussians = len(indices_to_keep)

    def clamp_steepness(self, max_steepness: float = 0.15):
        """
        Clamp amplitudes to prevent nonphysical wave steepness.
        Steepness ε = |k|·|w| for each component; typical ocean < 0.1.
        """
        with torch.no_grad():
            k_mag = torch.sqrt(self.kx ** 2 + self.ky ** 2 + 1e-12)
            steepness = k_mag * torch.abs(self.w)
            excess = steepness > max_steepness
            if torch.any(excess):
                scale = max_steepness / (steepness[excess] + 1e-12)
                self.w.data[excess] *= scale

    def compute_Hs(self) -> float:
        """Estimate significant wave height from current weights."""
        with torch.no_grad():
            return float(4.0 * torch.sqrt(torch.sum(self.w ** 2) / 2.0))
