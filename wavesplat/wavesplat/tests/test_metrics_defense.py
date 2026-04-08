"""
Tests for reviewer-defense metrics: Variance Ratio, Pearson correlation,
Skill Score.

These tests verify that the metrics correctly distinguish between:
- Accurate prediction (perfect or near-perfect reconstruction)
- Over-smoothed prediction (under-fitting; e.g., GPR with too-large length scale)
- Constant/zero prediction (trivial baseline)
- Inverted/wrong-sign prediction (catastrophic failure)
"""

import numpy as np
import pytest

from wavesplat.evaluation.metrics import (
    variance_ratio,
    pearson_correlation,
    skill_score,
    nrmse,
)


@pytest.fixture
def wave_field():
    """A representative 2D wave field with Hs ≈ 1.0 m."""
    rng = np.random.default_rng(42)
    Nx, Ny = 80, 60
    x = np.linspace(0, 200, Nx)
    y = np.linspace(0, 150, Ny)
    X, Y = np.meshgrid(x, y)
    # Two-component sea: 1 swell + 1 wind sea
    z = (
        0.4 * np.cos(0.05 * X + 0.02 * Y + 1.0)
        + 0.3 * np.cos(0.08 * X - 0.03 * Y + 2.0)
        + 0.05 * rng.standard_normal((Ny, Nx))
    )
    return z


class TestVarianceRatio:
    def test_perfect_prediction(self, wave_field):
        """Perfect prediction → VR = 1.0."""
        assert variance_ratio(wave_field, wave_field) == pytest.approx(1.0)

    def test_over_smoothed_prediction(self, wave_field):
        """Over-smoothed (constant zero) prediction → VR ≈ 0."""
        zero_pred = np.zeros_like(wave_field)
        vr = variance_ratio(zero_pred, wave_field)
        assert vr < 0.01, f"Zero predictor VR should be ~0, got {vr}"

    def test_half_amplitude_prediction(self, wave_field):
        """Half-amplitude prediction → VR ≈ 0.25."""
        half = 0.5 * wave_field
        vr = variance_ratio(half, wave_field)
        assert vr == pytest.approx(0.25, rel=1e-6)

    def test_over_fitting_prediction(self, wave_field):
        """2x amplitude → VR ≈ 4."""
        double = 2.0 * wave_field
        vr = variance_ratio(double, wave_field)
        assert vr == pytest.approx(4.0, rel=1e-6)


class TestPearsonCorrelation:
    def test_perfect_prediction(self, wave_field):
        """Perfect prediction → ρ = 1.0."""
        assert pearson_correlation(wave_field, wave_field) == pytest.approx(1.0)

    def test_inverted_prediction(self, wave_field):
        """Sign-flipped prediction → ρ = -1.0."""
        assert pearson_correlation(-wave_field, wave_field) == pytest.approx(-1.0)

    def test_amplitude_invariance(self, wave_field):
        """ρ is invariant to amplitude scaling — half-amplitude still ρ = 1."""
        assert pearson_correlation(0.5 * wave_field, wave_field) == pytest.approx(1.0)

    def test_uncorrelated_prediction(self, wave_field):
        """Independent random prediction → ρ ≈ 0."""
        rng = np.random.default_rng(123)
        random_pred = rng.standard_normal(wave_field.shape)
        rho = pearson_correlation(random_pred, wave_field)
        assert abs(rho) < 0.1, f"Random predictor ρ should be ~0, got {rho}"


class TestSkillScore:
    def test_perfect_prediction(self, wave_field):
        """Perfect prediction → SS = 1.0."""
        assert skill_score(wave_field, wave_field) == pytest.approx(1.0)

    def test_zero_predictor_baseline(self, wave_field):
        """Zero predictor against zero reference → SS = 0."""
        zero_pred = np.zeros_like(wave_field)
        ss = skill_score(zero_pred, wave_field)
        assert abs(ss) < 1e-9, f"Zero predictor SS should be 0, got {ss}"

    def test_negative_skill(self, wave_field):
        """Worse-than-zero predictor → SS < 0."""
        # 2x predictor amplifies error
        bad_pred = 2.0 * wave_field
        gt_zero_mse = np.mean(wave_field ** 2)
        bad_mse = np.mean((bad_pred - wave_field) ** 2)
        # bad_mse = mse(z, 2z) = mse(z) which equals var if zero-mean
        # Actually mse(2z - z) = mse(z) — same as zero predictor!
        # So use a clearer bad case:
        bad_pred = -wave_field  # inverted
        ss = skill_score(bad_pred, wave_field)
        assert ss < 0, f"Inverted predictor SS should be negative, got {ss}"


class TestUnderFittingExposure:
    """The key motivating test: nRMSE looks similar but VR/ρ exposes the gap."""

    def test_under_fitting_vs_accurate(self, wave_field):
        """
        Two predictors with similar nRMSE but very different actual quality.

        Predictor A: scaled-down version of GT (under-fitting, "GPR-like")
        Predictor B: GT + small noise (accurate, "WaveSplat-like")

        Expectation:
        - Both have moderate nRMSE
        - A has VR << 1 and ρ = 1 (under-fitting exposed by VR)
        - B has VR ≈ 1 and ρ ≈ 1 (accurate)
        """
        Hs = 4.0 * np.std(wave_field)

        # Predictor A: 30% amplitude (under-fitting)
        pred_A = 0.3 * wave_field

        # Predictor B: GT + noise calibrated to match A's nRMSE
        rng = np.random.default_rng(7)
        noise_std = 0.7 * np.std(wave_field)
        pred_B = wave_field + noise_std * rng.standard_normal(wave_field.shape)

        nrmse_A = nrmse(pred_A, wave_field, Hs)
        nrmse_B = nrmse(pred_B, wave_field, Hs)

        vr_A = variance_ratio(pred_A, wave_field)
        vr_B = variance_ratio(pred_B, wave_field)

        # Both should have non-trivial nRMSE
        assert nrmse_A > 0.05
        assert nrmse_B > 0.05

        # Critical: VR distinguishes them clearly
        assert vr_A < 0.2, f"Under-fitter VR should be <<1, got {vr_A}"
        assert 0.7 < vr_B, f"Accurate predictor VR should be ≈1, got {vr_B}"
