"""
test_hos.py
Validation tests for HOS (Higher-Order Spectral) wave simulator.

Tests verify:
  1. FFT convention consistency
  2. M=1 convergence to linear theory (O(A^2) error)
  3. Energy conservation for M>=2
  4. JONSWAP initialization matches target Hs
  5. Nonlinear effects (M=2 vs M=1)
"""

import numpy as np
import pytest
from wavesim.hos_simulator import HOSSimulator, HOSConfig
from wavesim.wave_generator import WaveComponent, full_dispersion


def _make_grid_components(Lx=200.0, Ly=200.0, depth=50.0):
    """Create test components with wavenumbers exactly on the FFT grid."""
    dkx = 2 * np.pi / Lx
    kx1, ky1 = 3 * dkx, 0.0
    kx2, ky2 = 2 * dkx, 2 * dkx
    return [
        WaveComponent(A=0.3,
                      omega=full_dispersion(np.sqrt(kx1**2 + ky1**2), depth),
                      kx=kx1, ky=ky1, phi=0.5, Q=0.0),
        WaveComponent(A=0.2,
                      omega=full_dispersion(np.sqrt(kx2**2 + ky2**2), depth),
                      kx=kx2, ky=ky2, phi=1.2, Q=0.0),
    ]


class TestHOSInit:
    """Test initialization routines."""

    def test_from_components_exact(self):
        """Grid-exact wavenumbers should give zero init error."""
        Lx, Ly, Nx, Ny, depth = 200.0, 200.0, 64, 64, 50.0
        components = _make_grid_components(Lx, Ly, depth)

        cfg = HOSConfig(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, depth=depth, M=1)
        sim = HOSSimulator(cfg)
        sim.initialize_from_components(components)

        x = np.arange(Nx) * Lx / Nx
        y = np.arange(Ny) * Ly / Ny
        X, Y = np.meshgrid(x, y, indexing='ij')

        eta_analytical = np.zeros_like(X)
        for c in components:
            eta_analytical += c.A * np.cos(c.kx * X + c.ky * Y + c.phi)

        eta_hos = sim.get_eta()
        rmse = np.sqrt(np.mean((eta_hos - eta_analytical)**2))
        assert rmse < 1e-10, f"Init RMSE {rmse} too large"

    def test_jonswap_hs_match(self):
        """JONSWAP initialization should match target Hs."""
        cfg = HOSConfig(Lx=500, Ly=400, Nx=64, Ny=64, depth=50.0, M=2)
        sim = HOSSimulator(cfg)

        for Hs_target in [0.5, 1.0, 2.0]:
            sim.initialize_jonswap(Hs=Hs_target, Tp=7.0, seed=42)
            Hs_actual = sim.get_Hs()
            rel_err = abs(Hs_actual - Hs_target) / Hs_target
            assert rel_err < 0.05, \
                f"Hs mismatch: target={Hs_target}, actual={Hs_actual:.4f}"


class TestLinearConvergence:
    """M=1 HOS should converge to linear theory as amplitude -> 0."""

    def test_order_convergence(self):
        """Error should scale as O(A^2) — quadratic in amplitude."""
        Lx, Ly, Nx, Ny, depth = 200.0, 200.0, 64, 64, 50.0
        t = 2.0
        n_steps = 40
        dt = t / n_steps

        x = np.arange(Nx) * Lx / Nx
        y = np.arange(Ny) * Ly / Ny
        X, Y = np.meshgrid(x, y, indexing='ij')

        errors = []
        for amp_scale in [0.1, 0.01]:
            components = _make_grid_components(Lx, Ly, depth)
            for c in components:
                c.A *= amp_scale

            cfg = HOSConfig(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, depth=depth, M=1)
            sim = HOSSimulator(cfg)
            sim.initialize_from_components(components)

            for _ in range(n_steps):
                sim.step(dt)

            eta_hos = sim.get_eta()
            eta_linear = np.zeros_like(X)
            for c in components:
                eta_linear += c.A * np.cos(c.kx*X + c.ky*Y - c.omega*t + c.phi)

            rel_err = np.sqrt(np.mean((eta_hos - eta_linear)**2)) / np.std(eta_linear)
            errors.append(rel_err)

        # When amplitude drops by 10x, relative error should drop by ~10x (O(A^2)/O(A) = O(A))
        ratio = errors[0] / errors[1]
        assert ratio > 5, f"Expected ~10x error reduction, got {ratio:.2f}x"


class TestEnergyConservation:
    """HOS should conserve total energy (Hamiltonian system)."""

    def test_m2_energy_conservation(self):
        """M=2 should conserve energy within 0.1% over 10 seconds."""
        cfg = HOSConfig(Lx=500, Ly=400, Nx=64, Ny=64, depth=50.0, M=2)
        sim = HOSSimulator(cfg)
        sim.initialize_jonswap(Hs=1.0, Tp=7.0, seed=42)

        Ek0, Ep0 = sim.compute_energy()
        E0 = Ek0 + Ep0

        for _ in range(200):  # 10 seconds at dt=0.05
            sim.step(0.05)

        Ek1, Ep1 = sim.compute_energy()
        E1 = Ek1 + Ep1
        dE = abs(E1 - E0) / abs(E0) * 100

        assert dE < 0.1, f"Energy change {dE:.4f}% exceeds 0.1% threshold"

    def test_m2_hs_stability(self):
        """Hs should remain stable for M=2."""
        cfg = HOSConfig(Lx=500, Ly=400, Nx=64, Ny=64, depth=50.0, M=2)
        sim = HOSSimulator(cfg)
        sim.initialize_jonswap(Hs=2.0, Tp=8.0, seed=42)

        Hs0 = sim.get_Hs()
        for _ in range(100):
            sim.step(0.05)

        Hs1 = sim.get_Hs()
        rel_change = abs(Hs1 - Hs0) / Hs0 * 100
        assert rel_change < 2.0, \
            f"Hs changed {rel_change:.2f}%: {Hs0:.4f} -> {Hs1:.4f}"


class TestNonlinearEffects:
    """M>=2 should show nonlinear wave features absent in M=1."""

    def test_m1_energy_drift(self):
        """M=1 should show energy drift (inconsistent formulation)."""
        cfg = HOSConfig(Lx=500, Ly=400, Nx=64, Ny=64, depth=50.0, M=1)
        sim = HOSSimulator(cfg)
        sim.initialize_jonswap(Hs=2.0, Tp=8.0, seed=42)

        Ek0, Ep0 = sim.compute_energy()
        E0 = Ek0 + Ep0

        for _ in range(100):
            sim.step(0.05)

        Ek1, Ep1 = sim.compute_energy()
        E1 = Ek1 + Ep1
        dE = abs(E1 - E0) / abs(E0) * 100

        # M=1 has inconsistent formulation, expect > 0.5% drift
        assert dE > 0.5, f"M=1 energy drift {dE:.4f}% unexpectedly small"

    def test_m2_better_energy_than_m1(self):
        """M=2 should conserve energy better than M=1."""
        results = {}
        for M in [1, 2]:
            cfg = HOSConfig(Lx=500, Ly=400, Nx=64, Ny=64, depth=50.0, M=M)
            sim = HOSSimulator(cfg)
            sim.initialize_jonswap(Hs=2.0, Tp=8.0, seed=42)
            Ek0, Ep0 = sim.compute_energy()
            E0 = Ek0 + Ep0

            for _ in range(100):
                sim.step(0.05)

            Ek1, Ep1 = sim.compute_energy()
            E1 = Ek1 + Ep1
            results[M] = abs(E1 - E0) / abs(E0) * 100

        assert results[2] < results[1], \
            f"M=2 energy drift ({results[2]:.4f}%) should be less than M=1 ({results[1]:.4f}%)"


class TestRunTo:
    """Test the run_to convenience method."""

    def test_run_to_with_snapshots(self):
        """run_to should return snapshots at specified intervals."""
        cfg = HOSConfig(Lx=200, Ly=200, Nx=32, Ny=32, depth=50.0, M=2)
        sim = HOSSimulator(cfg)
        sim.initialize_jonswap(Hs=1.0, Tp=7.0, seed=42)

        results = sim.run_to(t_target=1.0, dt=0.05, store_interval=0.5)
        assert len(results['times']) >= 3, \
            f"Expected >=3 snapshots, got {len(results['times'])}"
        assert len(results['eta_fields']) == len(results['times'])
        assert results['eta_fields'][0].shape == (32, 32)
