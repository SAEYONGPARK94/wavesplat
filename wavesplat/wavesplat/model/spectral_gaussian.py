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


class SpectralGaussianModel(nn.Module):
    """
    Spectral Gaussian wave field model.

    Parameters:
        w:   (N,) amplitudes / weights [m]
        kx:  (N,) x-wavenumber components [rad/m]
        ky:  (N,) y-wavenumber components [rad/m]
        phi: (N,) phase offsets [rad]
    """

    def __init__(self, n_gaussians: int, depth: float = 50.0):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.depth = depth

        # Free parameters — initialized small; proper init via initializer.py
        self.w = nn.Parameter(torch.randn(n_gaussians) * 0.01)
        self.kx = nn.Parameter(torch.randn(n_gaussians) * 0.05)
        self.ky = nn.Parameter(torch.randn(n_gaussians) * 0.05)
        self.phi = nn.Parameter(torch.randn(n_gaussians) * math.pi)

    def _omega(self) -> torch.Tensor:
        """Compute ω from current (kx, ky) via dispersion relation."""
        k_mag = torch.sqrt(self.kx ** 2 + self.ky ** 2 + 1e-12)
        return dispersion_omega(k_mag, self.depth)

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
        self.n_gaussians += n_new

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
