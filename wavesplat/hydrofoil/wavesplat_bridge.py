"""Bridge module: WaveSplat estimation → hydrofoil feedforward controller.

Converts WaveSplat's spectral parameters (w, kx, ky, phi, omega) into
WaveComponent-compatible objects that the existing hydrofoil disturbance
and feedforward modules can consume directly.

Usage:
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    from hydrofoil.wavesplat_bridge import wavesplat_to_components

    updater = SequentialUpdater(config)
    updater.initialize(init_frames, ...)

    # After tracking update:
    components = wavesplat_to_components(updater.model)
    # These work directly with disturbance.py and sim.py
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class WaveSplatComponent:
    """WaveComponent-compatible representation from WaveSplat model.

    Matches the field interface of wavesim.wave_generator.WaveComponent:
      A, omega, kx, ky, phi, Q
    """
    A: float        # amplitude [m]
    omega: float    # angular frequency [rad/s]
    kx: float       # x-wavenumber [rad/m]
    ky: float       # y-wavenumber [rad/m]
    phi: float      # phase offset [rad]
    Q: float = 0.0  # Gerstner steepness (0 for linear model)

    @property
    def k(self) -> float:
        return np.sqrt(self.kx**2 + self.ky**2)

    @property
    def direction(self) -> float:
        return np.arctan2(self.ky, self.kx)


def wavesplat_to_components(model) -> list:
    """Convert WaveSplat SpectralGaussianModel to component list.

    Args:
        model: SpectralGaussianModel (trained or tracking)

    Returns:
        list of WaveSplatComponent — compatible with disturbance.py functions
    """
    params = model.get_spectrum_params()
    components = []
    for i in range(len(params['w'])):
        components.append(WaveSplatComponent(
            A=float(params['w'][i]),
            omega=float(params['omega'][i]),
            kx=float(params['kx'][i]),
            ky=float(params['ky'][i]),
            phi=float(params['phi'][i]),
            Q=0.0,  # Linear model — no horizontal displacement
        ))
    return components


def h5_gt_to_components(h5_wave_components: np.ndarray) -> list:
    """Convert HDF5 GT wave_components array to component list.

    The HDF5 stores wave_components as (N, 6) array with columns:
    [A, omega, kx, ky, phi, Q]

    Args:
        h5_wave_components: (N, 6) numpy array from HDF5 episode

    Returns:
        list of WaveSplatComponent
    """
    components = []
    for row in h5_wave_components:
        components.append(WaveSplatComponent(
            A=float(row[0]),
            omega=float(row[1]),
            kx=float(row[2]),
            ky=float(row[3]),
            phi=float(row[4]),
            Q=float(row[5]),
        ))
    return components
