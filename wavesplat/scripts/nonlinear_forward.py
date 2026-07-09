"""Analytic weakly-nonlinear forward for LABELED HOS-style training augmentation.

Full HOS surfaces have no discrete component labels, so label-based nets (A1/B1) can't train on
them directly. Instead we keep the free-wave components as labels and synthesize NONLINEAR
observations by adding the 2nd-order Stokes bound harmonic (self-interaction) to the linear
sum — the leading weakly-nonlinear effect (sharp crests / flat troughs). Finite-depth Stokes
coefficient (Dean & Dalrymple); deep-water limit k/2 used where kd>10 (within 1e-3, avoids
cosh/sinh overflow). Real full-HOS (cross-interactions + evolution) is kept for EVALUATION.

    η = Σ aᵢ cos ψᵢ + Σ cᵢ aᵢ² cos(2ψᵢ),   cᵢ = (kᵢ/4)·cosh(kd)(2+cosh 2kd)/sinh³(kd) → kᵢ/2
"""
import numpy as np


def stokes2_coeff(kmag, depth):
    """2nd-harmonic self-interaction coefficient cᵢ (finite depth, deep-water-safe)."""
    kd = kmag * depth
    deep = kd > 10.0
    c = np.empty_like(kmag)
    c[deep] = kmag[deep] / 2.0                                    # cosh/sinh ratio → 2 within 1e-3
    s = ~deep
    if np.any(s):
        kds = kd[s]
        c[s] = (kmag[s] / 4.0) * np.cosh(kds) * (2.0 + np.cosh(2.0 * kds)) / np.sinh(kds) ** 3
    return c


def elevation(kx, ky, A, om, phi, ox, oy, ot, depth, nonlinear=True):
    """Surface elevation at obs points (ox,oy,ot). kx,ky,A,om,phi: (M,); ox,oy,ot: (m,).
    nonlinear=True adds the 2nd-order Stokes self-harmonic bound wave (labeled by the same
    free components)."""
    psi = (kx[None] * ox[:, None] + ky[None] * oy[:, None]
           - om[None] * ot[:, None] + phi[None])                 # (m, M)
    z = (A[None] * np.cos(psi)).sum(1)
    if nonlinear:
        kmag = np.sqrt(kx ** 2 + ky ** 2) + 1e-12
        c = stokes2_coeff(kmag, depth)                           # (M,)
        z = z + (c[None] * (A[None] ** 2) * np.cos(2.0 * psi)).sum(1)
    return z
