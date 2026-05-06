"""Verification tests for hydrofoil 2DOF simulation.

Test hierarchy (Rev.2 recommended):
  1. Static trim: no waves → vessel should stay at equilibrium
  2. Foil physics: lift/drag at known conditions
  3. Single sinusoidal wave: response should be bounded and periodic
  4. Multi-directional waves: encounter frequency correctness
"""

import numpy as np
import pytest

from hydrofoil.config import candela_c8, FoilConfig, SimConfig, _set_trim_alpha
from hydrofoil.foil import (
    foil_forces, compute_cl, compute_cd, submergence_factor, planform_area,
)
from hydrofoil.vehicle import (
    HydrofoilState, WaveDisturbanceInput, compute_derivatives,
    effective_alpha, foil_submergence,
)
from hydrofoil.sim import find_trim, run_simulation


# ============================================================
# Test 1: Foil aerodynamics
# ============================================================

class TestFoilPhysics:
    """Verify foil lift/drag model."""

    def test_lift_increases_with_alpha(self):
        """CL should increase linearly with alpha."""
        foil = FoilConfig(
            span=1.0, chord=0.2, cl_alpha=2*np.pi,
            cl_max=1.5, cd0=0.01, alpha_zl=0.0,
            x_pos=0.0, z_pos=1.0,
        )
        cl_1 = compute_cl(0.05, foil)
        cl_2 = compute_cl(0.10, foil)
        assert cl_2 > cl_1
        assert abs(cl_2 - 2 * cl_1) < 1e-10  # linear

    def test_cl_clamps_at_max(self):
        """CL should not exceed cl_max."""
        foil = FoilConfig(
            span=1.0, chord=0.2, cl_alpha=2*np.pi,
            cl_max=1.0, cd0=0.01, alpha_zl=0.0,
            x_pos=0.0, z_pos=1.0,
        )
        cl = compute_cl(1.0, foil)  # huge alpha
        assert cl == 1.0

    def test_drag_increases_with_lift(self):
        """Induced drag should grow with CL²."""
        foil = FoilConfig(
            span=1.0, chord=0.2, cl_alpha=2*np.pi,
            cl_max=2.0, cd0=0.01, alpha_zl=0.0,
            x_pos=0.0, z_pos=1.0,
        )
        cd_0 = compute_cd(0.0, foil)
        cd_1 = compute_cd(0.5, foil)
        cd_2 = compute_cd(1.0, foil)
        assert cd_0 == pytest.approx(0.01, abs=1e-6)  # just cd0
        assert cd_2 > cd_1 > cd_0

    def test_submergence_deep(self):
        """Deep foil should have no correction."""
        assert submergence_factor(1.0, 0.2) == 1.0  # h/c = 5

    def test_submergence_surface(self):
        """Foil at surface should have zero lift."""
        assert submergence_factor(0.0, 0.2) == 0.0

    def test_submergence_breached(self):
        """Foil above water should have zero lift."""
        assert submergence_factor(-0.1, 0.2) == 0.0

    def test_submergence_shallow(self):
        """Shallow foil should have reduced but nonzero lift."""
        sf = submergence_factor(0.1, 0.2)  # h/c = 0.5
        assert 0.5 < sf < 1.0

    def test_foil_forces_symmetry(self):
        """Negative alpha should produce negative lift."""
        foil = FoilConfig(
            span=1.0, chord=0.2, cl_alpha=2*np.pi,
            cl_max=1.5, cd0=0.01, alpha_zl=0.0,
            x_pos=0.0, z_pos=1.0,
        )
        L_pos, _ = foil_forces(0.05, 10.0, 1.0, foil)
        L_neg, _ = foil_forces(-0.05, 10.0, 1.0, foil)
        assert L_pos > 0
        assert L_neg < 0
        assert abs(L_pos + L_neg) < 1e-6  # antisymmetric


# ============================================================
# Test 2: Static trim equilibrium
# ============================================================

class TestTrim:
    """Verify that trim state exists and is stable."""

    def test_trim_alpha_zl_nonzero(self):
        """After _set_trim_alpha, alpha_zl should be nonzero (negative)."""
        cfg = candela_c8()
        # alpha_zl should be negative → positive CL at alpha=0
        assert cfg['front_foil'].alpha_zl < 0
        assert cfg['rear_foil'].alpha_zl < 0

    def test_trim_force_balance(self):
        """At trim, total lift should equal weight."""
        cfg = candela_c8()
        hull = cfg['hull']
        sim = cfg['sim']
        ff = cfg['front_foil']
        rf = cfg['rear_foil']

        W = hull.mass * sim.g
        # At trim: alpha=0 for both foils, deep submergence
        L_f, _ = foil_forces(0.0, sim.speed, ff.z_pos, ff, sim.rho)
        L_r, _ = foil_forces(0.0, sim.speed, rf.z_pos, rf, sim.rho)

        assert L_f + L_r == pytest.approx(W, rel=0.01)

    def test_trim_moment_balance(self):
        """At trim, net moment about CG should be zero."""
        cfg = candela_c8()
        sim = cfg['sim']
        ff = cfg['front_foil']
        rf = cfg['rear_foil']

        L_f, _ = foil_forces(0.0, sim.speed, ff.z_pos, ff, sim.rho)
        L_r, _ = foil_forces(0.0, sim.speed, rf.z_pos, rf, sim.rho)

        M = L_f * ff.x_pos + L_r * rf.x_pos
        assert abs(M) < 1.0  # < 1 N·m

    def test_find_trim_converges(self):
        """find_trim should converge to near-zero velocities."""
        cfg = candela_c8()
        state = find_trim(cfg['front_foil'], cfg['rear_foil'],
                          cfg['hull'], cfg['sim'])
        assert abs(state.z_dot) < 1e-4
        assert abs(state.theta_dot) < 1e-4

    def test_trim_stays_stable(self):
        """Short calm-water sim from trim should remain near trim."""
        cfg = candela_c8()
        cfg['sim'].duration = 5.0  # 5 seconds
        result = run_simulation(cfg, components=[], vessel_heading=0.0)

        # Heave and pitch should stay small
        assert np.max(np.abs(result.z)) < 0.05  # < 5cm
        assert np.max(np.abs(result.theta)) < np.radians(1.0)  # < 1°


# ============================================================
# Test 3: Single sinusoidal wave response
# ============================================================

class TestSingleWaveResponse:
    """Verify response to a single regular wave."""

    @staticmethod
    def _make_single_wave(A=0.3, T=6.0, direction_deg=180.0, depth=50.0):
        """Create a single WaveComponent for testing.

        Uses wavesim's WaveComponent if available, else a simple dataclass.
        """
        try:
            from wavesim.wave_generator import (
                WaveComponent, full_dispersion,
            )
            omega = 2 * np.pi / T
            from wavesim.wave_generator import wavenumber_from_omega
            k_mag = wavenumber_from_omega(omega, depth)
            theta_w = np.radians(direction_deg)
            kx = k_mag * np.cos(theta_w)
            ky = k_mag * np.sin(theta_w)
            return [WaveComponent(A=A, omega=omega, kx=kx, ky=ky,
                                  phi=0.0, Q=0.0)]
        except ImportError:
            # Fallback: simple namespace
            from types import SimpleNamespace
            g = 9.81
            omega = 2 * np.pi / T
            k_mag = omega**2 / g  # deep water approx
            theta_w = np.radians(direction_deg)
            return [SimpleNamespace(
                A=A, omega=omega,
                kx=k_mag * np.cos(theta_w),
                ky=k_mag * np.sin(theta_w),
                phi=0.0, Q=0.0, k=k_mag,
                direction=theta_w,
            )]

    def test_bounded_response(self):
        """Heave response should be bounded (not diverging)."""
        cfg = candela_c8()
        cfg['sim'].duration = 20.0
        components = self._make_single_wave(A=0.3, T=6.0, direction_deg=180.0)
        result = run_simulation(cfg, components, vessel_heading=0.0)

        # Heave should be bounded: not more than ~10x wave amplitude
        assert np.max(np.abs(result.z)) < 3.0  # < 3m for 0.3m wave

    def test_periodic_steady_state(self):
        """After transient, response should become approximately periodic."""
        cfg = candela_c8()
        cfg['sim'].duration = 30.0
        components = self._make_single_wave(A=0.3, T=6.0, direction_deg=180.0)
        result = run_simulation(cfg, components, vessel_heading=0.0)

        # Check last 10 seconds for periodicity
        mask = result.time > 20.0
        z_late = result.z[mask]

        # Standard deviation should be roughly constant over last period
        T_enc = 6.0  # approximate encounter period
        n_per_period = int(T_enc / cfg['sim'].dt)
        if len(z_late) > 2 * n_per_period:
            std_first = np.std(z_late[:n_per_period])
            std_last = np.std(z_late[-n_per_period:])
            # Should be within 50% of each other (generous for transient)
            if std_first > 0.001:
                ratio = std_last / std_first
                assert 0.5 < ratio < 2.0

    def test_head_sea_vs_following_sea(self):
        """Head sea should have higher encounter frequency than following sea."""
        cfg = candela_c8()
        cfg['sim'].duration = 20.0

        # Head sea: wave comes from ahead (direction 180° = -x direction)
        comp_head = self._make_single_wave(A=0.3, T=6.0, direction_deg=180.0)
        result_head = run_simulation(cfg, comp_head, vessel_heading=0.0)

        # Following sea: wave comes from behind (direction 0° = +x direction)
        comp_follow = self._make_single_wave(A=0.3, T=6.0, direction_deg=0.0)
        result_follow = run_simulation(cfg, comp_follow, vessel_heading=0.0)

        # Count zero crossings of mean-subtracted signal as proxy for frequency
        mask = result_head.time > 10.0
        z_head = result_head.z[mask] - np.mean(result_head.z[mask])
        z_follow = result_follow.z[mask] - np.mean(result_follow.z[mask])
        zc_head = np.sum(np.diff(np.sign(z_head)) != 0)
        zc_follow = np.sum(np.diff(np.sign(z_follow)) != 0)

        # Head sea should have more zero crossings (higher encounter freq)
        assert zc_head > zc_follow


# ============================================================
# Test 4: Effective alpha components
# ============================================================

class TestEffectiveAlpha:
    """Verify angle of attack computation."""

    def test_alpha_from_pitch(self):
        """Pitch angle should directly contribute to alpha."""
        cfg = candela_c8()
        state = HydrofoilState(theta=0.05)  # 0.05 rad bow-up
        wave = WaveDisturbanceInput()
        alpha = effective_alpha(state, cfg['front_foil'], wave,
                                cfg['sim'], is_front=True)
        # alpha should include theta contribution
        assert alpha == pytest.approx(0.05, abs=0.001)

    def test_alpha_from_heave_rate(self):
        """Upward heave should reduce alpha."""
        cfg = candela_c8()
        state = HydrofoilState(z_dot=1.0)  # 1 m/s upward
        wave = WaveDisturbanceInput()
        alpha = effective_alpha(state, cfg['front_foil'], wave,
                                cfg['sim'], is_front=True)
        # alpha_heave = -z_dot / V = -1/11.3 ≈ -0.088
        assert alpha < 0

    def test_alpha_wave_slope(self):
        """Positive wave slope should reduce alpha."""
        cfg = candela_c8()
        state = HydrofoilState()
        wave = WaveDisturbanceInput(slope_front=0.1)  # tilted up ahead
        alpha = effective_alpha(state, cfg['front_foil'], wave,
                                cfg['sim'], is_front=True)
        assert alpha < 0  # wave slope reduces effective incidence


# ============================================================
# Test 5: Single-point feedforward (SP-1pt)
# ============================================================

class TestSinglePointFF:
    """Verify compute_single_point_ff_info behaviour."""

    @staticmethod
    def _single_wave(A=0.3, T=6.0, direction_deg=180.0, depth=50.0):
        try:
            from wavesim.wave_generator import WaveComponent, wavenumber_from_omega
            omega = 2 * np.pi / T
            k = wavenumber_from_omega(omega, depth)
            theta = np.radians(direction_deg)
            return [WaveComponent(A=A, omega=omega, kx=k*np.cos(theta),
                                  ky=k*np.sin(theta), phi=0.0, Q=0.0)]
        except ImportError:
            from types import SimpleNamespace
            omega = 2 * np.pi / T
            k = omega**2 / 9.81
            theta = np.radians(direction_deg)
            return [SimpleNamespace(A=A, omega=omega, kx=k*np.cos(theta),
                                    ky=k*np.sin(theta), phi=0.0, Q=0.0)]

    def test_same_signal_both_foils(self):
        """SP-1pt must feed identical slope/w to front and rear foils."""
        from hydrofoil.single_point_ff import compute_single_point_ff_info
        comps = self._single_wave()
        info = compute_single_point_ff_info(
            x_cg=0.0, components=comps, t=1.0,
            vessel_heading=0.0, speed=11.3, preview_dt=0.3)
        assert info.slope_front == info.slope_rear
        assert info.w_front == info.w_rear

    def test_head_sea_slope_and_w_nonzero(self):
        """Head sea wave should produce nonzero slope and w at preview point."""
        from hydrofoil.single_point_ff import compute_single_point_ff_info
        comps = self._single_wave(A=0.5, T=6.0, direction_deg=180.0)
        # Average over one wave period to avoid phase coincidences
        values = [compute_single_point_ff_info(
            x_cg=0.0, components=comps, t=t,
            vessel_heading=0.0, speed=11.3, preview_dt=0.3)
                  for t in np.linspace(0, 6.0, 60)]
        slopes = [v.slope_front for v in values]
        ws = [v.w_front for v in values]
        assert np.max(np.abs(slopes)) > 0.01
        assert np.max(np.abs(ws)) > 0.01

    def test_beam_sea_heading_projection_near_zero(self):
        """Pure beam sea slope projects to ~0 along vessel heading (x-axis)."""
        from hydrofoil.single_point_ff import compute_single_point_ff_info
        comps = self._single_wave(A=0.5, T=6.0, direction_deg=90.0)
        # Average over time to eliminate phase luck
        slopes = [compute_single_point_ff_info(
            x_cg=0.0, components=comps, t=t,
            vessel_heading=0.0, speed=11.3, preview_dt=0.3).slope_front
                  for t in np.linspace(0, 6.0, 120)]
        # For pure beam sea kx=0 → deta/dx=0 → slope=0 always
        assert np.max(np.abs(slopes)) < 1e-10

    def test_preview_position_uses_cg_not_foil(self):
        """Preview point should be CG + V*dt, not foil-specific position."""
        from hydrofoil.single_point_ff import compute_single_point_ff_info
        from hydrofoil.disturbance import wave_slope_x_at_point
        comps = self._single_wave(A=0.5, T=5.0, direction_deg=180.0)
        t, speed, preview_dt = 2.5, 11.3, 0.3
        x_cg = 50.0
        info = compute_single_point_ff_info(
            x_cg=x_cg, components=comps, t=t,
            vessel_heading=0.0, speed=speed, preview_dt=preview_dt)
        # Manually compute expected slope at CG+V*dt
        x_expected = x_cg + speed * preview_dt
        slope_expected = wave_slope_x_at_point(comps, x_expected, 0.0, t)
        assert info.slope_front == pytest.approx(slope_expected, abs=1e-12)


# ============================================================
# Test 6: TemporalBowSensor
# ============================================================

class TestTemporalBowSensor:
    """Verify TemporalBowSensor behaviour."""

    @staticmethod
    def _single_wave_pure(A=0.3, T=6.0, direction_deg=180.0):
        from types import SimpleNamespace
        omega = 2 * np.pi / T
        k = omega**2 / 9.81
        theta = np.radians(direction_deg)
        return [SimpleNamespace(A=A, omega=omega, kx=k*np.cos(theta),
                                ky=k*np.sin(theta), phi=0.0, Q=0.0)]

    def test_first_call_returns_zero(self):
        """Without history, sensor should return zero FF info."""
        from hydrofoil.single_point_ff import TemporalBowSensor
        sensor = TemporalBowSensor(dt=0.01, c_phase=8.0, noise_std=0.0)
        comps = self._single_wave_pure()
        info = sensor.compute_ff_info(
            x_cg=0.0, components=comps, t=0.0,
            vessel_heading=0.0, speed=11.3, preview_dt=0.3)
        assert info.slope_front == 0.0
        assert info.w_front == 0.0

    def test_temporal_derivative_tracks_wave(self):
        """After warm-up, w_est should match wave deta/dt at bow position."""
        from hydrofoil.single_point_ff import TemporalBowSensor
        from hydrofoil.disturbance import wave_vertical_velocity_at_point
        dt = 0.01
        sensor = TemporalBowSensor(dt=dt, c_phase=8.0, noise_std=0.0,
                                   bow_offset=4.0)
        comps = self._single_wave_pure(A=0.3, T=6.0)
        # Warm up 100 steps
        for i in range(100):
            info = sensor.compute_ff_info(
                x_cg=0.0, components=comps, t=i*dt,
                vessel_heading=0.0, speed=11.3, preview_dt=0.0)
        t_now = 100 * dt
        # w_est should approximate deta/dt at bow+preview point
        x_bow = 0.0 + (4.0 + 11.3 * 0.0)  # bow_offset + V*preview_dt
        w_true = wave_vertical_velocity_at_point(comps, x_bow, 0.0, t_now)
        # Finite difference is 1st-order, allow ~2% amplitude tolerance
        assert abs(info.w_front - w_true) < 0.05 * comps[0].A * comps[0].omega

    def test_sensor_state_advances_per_call(self):
        """Each call should update internal state (_eta_prev)."""
        from hydrofoil.single_point_ff import TemporalBowSensor
        sensor = TemporalBowSensor(dt=0.01, noise_std=0.0)
        comps = self._single_wave_pure()
        # First call initialises state
        sensor.compute_ff_info(x_cg=0.0, components=comps, t=0.0,
                               vessel_heading=0.0, speed=11.3, preview_dt=0.0)
        eta_after_1 = sensor._eta_prev
        # Second call at different time → state should change
        sensor.compute_ff_info(x_cg=0.0, components=comps, t=0.01,
                               vessel_heading=0.0, speed=11.3, preview_dt=0.0)
        eta_after_2 = sensor._eta_prev
        assert eta_after_1 != eta_after_2

    def test_head_sea_slope_sign(self):
        """In head sea, slope_est sign should be correct (positive w → negative slope)."""
        from hydrofoil.single_point_ff import TemporalBowSensor
        dt = 0.01
        c_phase = 9.81 * 6.0 / (2 * np.pi)  # deep-water c = g*T/(2π)
        sensor = TemporalBowSensor(dt=dt, c_phase=c_phase, noise_std=0.0)
        comps = self._single_wave_pure(A=0.5, T=6.0, direction_deg=180.0)
        # Warm up to find a time when w_est is clearly positive or negative
        prev = None
        for i in range(200):
            info = sensor.compute_ff_info(
                x_cg=0.0, components=comps, t=i*dt,
                vessel_heading=0.0, speed=11.3, preview_dt=0.0)
            if abs(info.w_front) > 0.3:
                # slope_est = -w/c → opposite sign to w
                assert np.sign(info.slope_front) == -np.sign(info.w_front)
                break


# ============================================================
# Test 7: ff_info_provider parameter in run_simulation
# ============================================================

class TestFfInfoProvider:
    """Verify that ff_info_provider correctly replaces the default FF computation."""

    @staticmethod
    def _make_jonswap(seed=42):
        try:
            from wavesim.wave_generator import jonswap_components
            return jonswap_components(
                Hs=0.88, Tp=5.0, n_components=8, depth=50.0,
                direction_deg=180.0, direction_spread_deg=20.0, seed=seed)
        except ImportError:
            return []

    def test_zero_provider_equals_pd_only(self):
        """A provider that always returns zero FF should give PD-only result."""
        from hydrofoil.controller import FeedforwardWaveInfo, FeedforwardConfig
        comps = self._make_jonswap()
        if not comps:
            pytest.skip("wavesim not available")

        cfg = candela_c8()
        cfg['sim'].duration = 10.0

        # Zero-output provider
        def zero_provider(state, ff, rf, sim, comps, t, heading, ff_cfg):
            return FeedforwardWaveInfo(slope_front=0.0, slope_rear=0.0,
                                      w_front=0.0, w_rear=0.0)

        np.random.seed(42)
        r_pd = run_simulation(cfg, comps, vessel_heading=0.0,
                              feedforward=None)
        np.random.seed(42)
        r_zero = run_simulation(cfg, comps, vessel_heading=0.0,
                                feedforward=FeedforwardConfig(
                                    preview_dt=0.3, K_ff=0.8),
                                ff_info_provider=zero_provider)

        # With zero provider, FF is all-zero → same as PD-only
        np.testing.assert_allclose(r_pd.heave_accel, r_zero.heave_accel,
                                   rtol=1e-6)

    def test_provider_called_during_simulation(self):
        """ff_info_provider should be called at every timestep."""
        from hydrofoil.controller import FeedforwardWaveInfo, FeedforwardConfig
        comps = self._make_jonswap()
        if not comps:
            pytest.skip("wavesim not available")

        cfg = candela_c8()
        cfg['sim'].duration = 5.0
        call_count = [0]

        def counting_provider(state, ff, rf, sim, comps, t, heading, ff_cfg):
            call_count[0] += 1
            return FeedforwardWaveInfo()

        run_simulation(cfg, comps, vessel_heading=0.0,
                       feedforward=FeedforwardConfig(preview_dt=0.3, K_ff=0.8),
                       ff_info_provider=counting_provider)

        expected = int(cfg['sim'].duration / cfg['sim'].dt) + 1
        assert call_count[0] == pytest.approx(expected, abs=1)

    def test_sp1pt_provider_factory(self):
        """make_sp1pt_provider should return a callable that runs without error."""
        from hydrofoil.controller import FeedforwardConfig
        from hydrofoil.run_single_point_comparison import make_sp1pt_provider
        comps = self._make_jonswap()
        if not comps:
            pytest.skip("wavesim not available")

        cfg = candela_c8()
        cfg['sim'].duration = 5.0
        np.random.seed(0)
        result = run_simulation(
            cfg, comps, vessel_heading=0.0,
            feedforward=FeedforwardConfig(preview_dt=0.3, K_ff=0.8),
            ff_info_provider=make_sp1pt_provider(comps))

        assert result.heave_accel is not None
        assert len(result.heave_accel) > 0
        assert np.isfinite(result.heave_accel).all()

    def test_sp_temporal_provider_factory(self):
        """make_sp_temporal_provider should return a callable that runs without error."""
        from hydrofoil.controller import FeedforwardConfig
        from hydrofoil.run_single_point_comparison import make_sp_temporal_provider
        comps = self._make_jonswap()
        if not comps:
            pytest.skip("wavesim not available")

        cfg = candela_c8()
        cfg['sim'].duration = 5.0
        dt = cfg['sim'].dt
        np.random.seed(0)
        result = run_simulation(
            cfg, comps, vessel_heading=0.0,
            feedforward=FeedforwardConfig(preview_dt=0.3, K_ff=0.8),
            ff_info_provider=make_sp_temporal_provider(comps, 5.0, dt))

        assert result.heave_accel is not None
        assert np.isfinite(result.heave_accel).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
