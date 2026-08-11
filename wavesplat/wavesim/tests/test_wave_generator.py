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


# ── 5. Tank depth d=3.5m tests ───────────────────────────────────────────────

def test_dispersion_tank_depth():
    """Dispersion at d=3.5m: finite-depth effects for long-period waves."""
    depth = 3.5
    # T=1.0s (kd >> 1): should match deep water
    k_short = wavenumber_from_omega(2 * np.pi, depth)
    k_deep = wavenumber_from_omega(2 * np.pi, 1000.0)
    assert abs(k_short - k_deep) / k_deep < 0.01, \
        "T=1.0s at d=3.5m should be deep-water"

    # T=3.0s (kd ≈ 1.6): finite-depth correction significant
    omega_long = 2 * np.pi / 3.0
    k_shallow = wavenumber_from_omega(omega_long, depth)
    k_deep_long = wavenumber_from_omega(omega_long, 1000.0)
    # Finite depth → k increases (wavelength shortens)
    assert k_shallow > k_deep_long * 1.01, \
        f"k_shallow={k_shallow:.4f} should exceed k_deep={k_deep_long:.4f} at T=3.0s"


def test_dispersion_crossvalidate_wavesplat():
    """wavesim and wavesplat dispersion must agree at d=3.5m."""
    import torch
    from wavesplat.model.dispersion import dispersion_omega as ws_dispersion

    depth = 3.5
    k_vals = [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    for k_val in k_vals:
        omega_sim = full_dispersion(k_val, depth)
        omega_splat = ws_dispersion(torch.tensor([k_val]), depth).item()
        assert abs(omega_sim - omega_splat) < 1e-5, \
            f"k={k_val}, d=3.5: wavesim={omega_sim:.6f}, wavesplat={omega_splat:.6f}"


def test_inverse_dispersion_tank_depth():
    """Inverse dispersion roundtrip at d=3.5m."""
    depth = 3.5
    for omega in [1.5, 2.0, 3.0, 5.0, 8.0]:
        k = wavenumber_from_omega(omega, depth)
        omega_back = full_dispersion(k, depth)
        assert abs(omega_back - omega) < 1e-5, \
            f"ω={omega}: roundtrip error {abs(omega_back-omega):.2e} at d=3.5m"


def test_jonswap_tank_depth():
    """JONSWAP spectrum generation at d=3.5m (tank-scale waves)."""
    # Tank-scale: Hs=0.05m, Tp=1.0s
    comps = jonswap_components(Hs=0.05, Tp=1.0, n_components=10,
                                seed=42, depth=3.5)
    assert len(comps) == 10
    for c in comps:
        assert c.A > 0
        # Verify dispersion relation is satisfied
        k_mag = np.sqrt(c.kx**2 + c.ky**2)
        omega_expected = full_dispersion(k_mag, depth=3.5)
        assert abs(c.omega - omega_expected) < 1e-4, \
            f"Component ω={c.omega:.4f} doesn't match dispersion ω={omega_expected:.4f}"


def test_jonswap_tank_Hs_accuracy():
    """JONSWAP at d=3.5m: Hs should be reasonable for tank-scale waves."""
    x = np.linspace(0, 50, 100)
    y = np.linspace(-4, 4, 20)
    X, Y = np.meshgrid(x, y)

    errors = []
    for Hs_target, Tp in [(0.05, 1.0), (0.10, 1.5), (0.15, 2.0)]:
        for seed in range(3):
            comps = jonswap_components(Hs=Hs_target, Tp=Tp,
                                        n_components=15, seed=seed, depth=3.5)
            Z, _, _, _ = superpose_wrapper(comps, X, Y, t=0.0)
            Hs_est = compute_Hs(Z)
            errors.append(abs(Hs_est - Hs_target) / Hs_target)

    mean_err = np.mean(errors)
    assert mean_err < 0.30, \
        f"Tank JONSWAP Hs error = {mean_err*100:.1f}% (should be < 30%)"


def test_gerstner_tank_depth():
    """Gerstner wave evaluation at d=3.5m with Q=0 (linear)."""
    depth = 3.5
    omega = 2 * np.pi / 1.5  # T=1.5s
    k = wavenumber_from_omega(omega, depth)
    comp = [WaveComponent(A=0.02, omega=omega, kx=k, ky=0.0, phi=0.0, Q=0.0)]

    x = np.linspace(0, 20, 40)
    y = np.zeros(1)
    X, Y = np.meshgrid(x, y)
    t = 1.0

    _, _, Z, _, _, nz = single_gerstner(comp, X, Y, t)
    Z_analytic = 0.02 * np.cos(k * X - omega * t)

    rmse = np.sqrt(np.mean((Z - Z_analytic)**2))
    assert rmse < 1e-10, f"Gerstner at d=3.5m RMSE={rmse:.2e}"
    assert np.all(nz > 0), "Surface normal must point upward"


# ---------------------------------------------------------------------------
# Independence knobs (2026-08-07).  These lock in the two properties that make
# the knobs safe to add: the default path is unchanged, and each knob moves
# exactly one factor.  See tmp/audit_independence_2026-08-07/ for the measured
# defect that motivated them.
# ---------------------------------------------------------------------------

def test_independence_knobs_default_is_unchanged():
    """Both knobs off must reproduce the legacy draw exactly, or every stored
    episode silently stops being reproducible."""
    base = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=0)
    same = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=0,
                              random_amplitudes=False, stream_tag=None)
    for a, b in zip(base, same):
        assert (a.A, a.omega, a.kx, a.ky, a.phi, a.Q) == \
               (b.A, b.omega, b.kx, b.ky, b.phi, b.Q)


def test_legacy_amplitudes_carry_no_randomness():
    """Documents the defect: without random_amplitudes, the seed cannot change
    the spectrum, so a dataset holds at most one spectrum per configuration."""
    a = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=0)
    b = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=1)
    assert np.allclose([c.A for c in a], [c.A for c in b])
    assert np.allclose([c.omega for c in a], [c.omega for c in b])


def test_random_amplitudes_varies_spectrum_and_preserves_Hs():
    a = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=0,
                           random_amplitudes=True)
    b = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=1,
                           random_amplitudes=True)
    assert not np.allclose([c.A for c in a], [c.A for c in b])
    for comps in (a, b):
        Hs = 4.0 * np.sqrt(sum(c.A ** 2 / 2.0 for c in comps))
        assert abs(Hs - 1.2) < 1e-9, f"Hs drifted to {Hs}"


def test_random_amplitudes_moves_only_amplitude():
    """Single-knob discipline: at one seed, toggling the flag must leave the
    phases and directions bit-identical."""
    det = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=0)
    rnd = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=0,
                             random_amplitudes=True)
    assert np.allclose([c.phi for c in det], [c.phi for c in rnd])
    assert np.allclose([np.arctan2(c.ky, c.kx) for c in det],
                       [np.arctan2(c.ky, c.kx) for c in rnd])
    assert not np.allclose([c.A for c in det], [c.A for c in rnd])


def test_legacy_stream_is_shared_across_sea_states_and_nested_in_n():
    """Documents the second defect: the legacy rng ignores sea_state and
    n_components entirely."""
    bf4 = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=0)
    bf5 = jonswap_components(2.0, 9.0, n_components=15, depth=50.0, seed=0)
    assert np.allclose([c.phi for c in bf4], [c.phi for c in bf5])
    n30 = jonswap_components(1.2, 7.0, n_components=30, depth=50.0, seed=0)
    assert np.allclose([c.phi for c in bf4], [c.phi for c in n30][:15])


def test_stream_tag_decorrelates():
    bf4 = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=0,
                             stream_tag='BF4_15')
    bf5 = jonswap_components(2.0, 9.0, n_components=15, depth=50.0, seed=0,
                             stream_tag='BF5_15')
    assert not np.allclose([c.phi for c in bf4], [c.phi for c in bf5])
    n30 = jonswap_components(1.2, 7.0, n_components=30, depth=50.0, seed=0,
                             stream_tag='BF4_30')
    assert not np.allclose([c.phi for c in bf4], [c.phi for c in n30][:15])
    # and it must still be deterministic
    again = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=0,
                               stream_tag='BF4_15')
    assert np.allclose([c.phi for c in bf4], [c.phi for c in again])


def test_legacy_frequencies_carry_no_randomness():
    """Documents the third defect: omega sits at the bin centre, a pure function
    of (Tp, n_components).  |k| follows from omega, so the radial wavenumber
    structure repeats across every seed of a configuration."""
    a = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=0)
    b = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=1)
    assert np.allclose([c.omega for c in a], [c.omega for c in b])


def test_random_frequencies_varies_omega_within_band():
    a = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=0,
                           random_frequencies=True)
    b = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=1,
                           random_frequencies=True)
    assert not np.allclose([c.omega for c in a], [c.omega for c in b])
    omega_p = 2 * np.pi / 7.0
    for comps in (a, b):
        for c in comps:
            assert 0.5 * omega_p <= c.omega <= 3.0 * omega_p
        Hs = 4.0 * np.sqrt(sum(x.A ** 2 / 2.0 for x in comps))
        assert abs(Hs - 1.2) < 1e-9


def test_random_frequencies_uses_its_own_stream():
    """Single-knob discipline: toggling it must not shift phases or directions."""
    det = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=0)
    rf = jonswap_components(1.2, 7.0, n_components=15, depth=50.0, seed=0,
                            random_frequencies=True)
    assert np.allclose([c.phi for c in det], [c.phi for c in rf])
    assert not np.allclose([c.omega for c in det], [c.omega for c in rf])


def test_all_three_knobs_off_is_still_bit_identical():
    base = jonswap_components(1.2, 7.0, n_components=30, depth=50.0, seed=2)
    same = jonswap_components(1.2, 7.0, n_components=30, depth=50.0, seed=2,
                              random_amplitudes=False, random_frequencies=False,
                              stream_tag=None)
    for a, b in zip(base, same):
        assert (a.A, a.omega, a.kx, a.ky, a.phi, a.Q) == \
               (b.A, b.omega, b.kx, b.ky, b.phi, b.Q)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
