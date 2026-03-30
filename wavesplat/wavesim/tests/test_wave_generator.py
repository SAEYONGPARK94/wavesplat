"""
Unit tests for wave_generator.py

Verification strategy (Trust Chain):
  1. single component: compare Z with analytic Gerstner solution
  2. full dispersion: ω² = g·k·tanh(k·d) satisfied
  3. Q constraint: ΣQᵢ·ωᵢ·Aᵢ ≤ 1 always
  4. JONSWAP: Hs from spectrum ≈ Hs input, Tp within tolerance
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest
from wavesim.wave_generator import (
    WaveComponent, full_dispersion, wavenumber_from_omega,
    single_gerstner, check_q_constraint, jonswap_components, compute_Hs
)


# ── 1. Dispersion relation ────────────────────────────────────────────────────

def test_full_dispersion_deep_water():
    """Deep water (kd >> 1): ω ≈ √(gk)"""
    k = 0.1  # rad/m, depth=1000m → kd=100 >> 1
    omega = full_dispersion(k, depth=1000.0)
    omega_deep = np.sqrt(9.81 * k)
    assert abs(omega - omega_deep) / omega_deep < 1e-4


def test_full_dispersion_shallow_water():
    """Shallow water (kd << 1): ω ≈ k·√(gd)"""
    k = 0.001   # rad/m
    depth = 5.0  # kd = 0.005 << 1
    omega = full_dispersion(k, depth)
    omega_shallow = k * np.sqrt(9.81 * depth)
    assert abs(omega - omega_shallow) / omega_shallow < 0.01


def test_inverse_dispersion_roundtrip():
    """wavenumber_from_omega is inverse of full_dispersion."""
    for omega in [0.5, 1.0, 1.5, 2.0]:
        k = wavenumber_from_omega(omega, depth=50.0)
        omega_back = full_dispersion(k, depth=50.0)
        assert abs(omega_back - omega) < 1e-6, f"ω={omega}: roundtrip error {abs(omega_back-omega):.2e}"


# ── 2. Single Gerstner wave analytic comparison ───────────────────────────────

def test_single_gerstner_analytic():
    """
    For a single component (Q=0 → pure sinusoidal):
      Z(x, y, t) = A·cos(kx·x + ky·y - ω·t + φ)
    Compare against analytic formula. RMSE < 1e-10 m.
    """
    A, omega, phi = 0.5, 1.0, 0.3
    depth = 50.0
    k = wavenumber_from_omega(omega, depth)
    kx = k * np.cos(np.deg2rad(30))
    ky = k * np.sin(np.deg2rad(30))

    comp = [WaveComponent(A=A, omega=omega, kx=kx, ky=ky, phi=phi, Q=0.0)]

    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    X, Y = np.meshgrid(x, y)
    t = 5.0

    _, _, Z, _, _, _ = single_gerstner(comp, X, Y, t)
    Z_analytic = A * np.cos(kx * X + ky * Y - omega * t + phi)

    rmse = np.sqrt(np.mean((Z - Z_analytic)**2))
    assert rmse < 1e-10, f"Single Gerstner RMSE={rmse:.2e} m (should be < 1e-10 m)"


def test_gerstner_normal_upward():
    """Surface normal z-component must always be positive (surface faces up)."""
    components = jonswap_components(Hs=1.2, Tp=7.0, n_components=10, seed=42)
    x = np.linspace(0, 200, 30)
    y = np.linspace(0, 200, 30)
    X, Y = np.meshgrid(x, y)
    _, _, _, _, _, nz = single_gerstner(components, X, Y, t=0.0)
    assert np.all(nz > 0), "Surface normal z-component must be positive everywhere"


# ── 3. Q constraint ──────────────────────────────────────────────────────────

def test_q_constraint_single():
    """Single component: Q·ω·A ≤ 1 must hold."""
    A, omega = 1.0, 1.5
    Q = 1.0 / (omega * A) - 0.01  # just below max
    comp = [WaveComponent(A=A, omega=omega, kx=0.1, ky=0.0, phi=0.0, Q=Q)]
    satisfied, val = check_q_constraint(comp)
    assert satisfied, f"Q constraint violated: ΣQωA={val:.3f}"


def test_q_constraint_jonswap():
    """JONSWAP output must always satisfy Q constraint."""
    for Hs, Tp in [(0.6, 5.0), (1.2, 7.0), (2.0, 9.0)]:
        comps = jonswap_components(Hs=Hs, Tp=Tp, n_components=30, seed=0)
        satisfied, val = check_q_constraint(comps)
        assert satisfied, f"Hs={Hs}, Tp={Tp}: Q constraint violated (ΣQωA={val:.3f})"


# ── 4. JONSWAP spectrum properties ───────────────────────────────────────────

def test_jonswap_Hs_accuracy():
    """
    Hs from JONSWAP components should be within 20% of target Hs.
    (Statistical estimate from finite components, 5 seeds)
    """
    target_Hs = 1.2
    x = np.linspace(0, 500, 200)
    y = np.linspace(0, 500, 200)
    X, Y = np.meshgrid(x, y)

    errors = []
    for seed in range(5):
        comps = jonswap_components(Hs=target_Hs, Tp=7.0, n_components=30, seed=seed)
        Z, _, _, _ = superpose_wrapper(comps, X, Y, t=0.0)
        Hs_est = compute_Hs(Z)
        errors.append(abs(Hs_est - target_Hs) / target_Hs)

    mean_err = np.mean(errors)
    assert mean_err < 0.20, f"JONSWAP Hs error = {mean_err*100:.1f}% (should be < 20%)"


def test_jonswap_component_count():
    """jonswap_components returns the requested number of components."""
    for n in [1, 5, 15, 30]:
        comps = jonswap_components(Hs=1.0, Tp=7.0, n_components=n, seed=0)
        assert len(comps) == n


def test_jonswap_positive_amplitudes():
    """All amplitudes must be positive."""
    comps = jonswap_components(Hs=1.2, Tp=7.0, n_components=30, seed=0)
    for c in comps:
        assert c.A > 0


def superpose_wrapper(components, X, Y, t):
    from wavesim.wave_generator import single_gerstner
    _, _, Z, nx, ny, nz = single_gerstner(components, X, Y, t)
    return Z, nx, ny, nz


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
