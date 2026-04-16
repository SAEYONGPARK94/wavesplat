"""
Unit tests for wave_probe.py — wave probe comparison module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest
import tempfile

from wavesplat.evaluation.wave_probe import (
    load_wave_probe_csv,
    extract_prediction_at_probe,
    synchronize_timeseries,
    compute_probe_metrics,
    compare_spectra,
)


# ── CSV loading ──────────────────────────────────────────────────────────────

def test_load_wave_probe_csv():
    """Load a simple CSV file with time and elevation."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("time,elevation\n")
        for i in range(100):
            t = i * 0.01
            eta = 0.05 * np.sin(2 * np.pi * t / 1.5)
            f.write(f"{t:.4f},{eta:.6f}\n")
        path = f.name

    try:
        t, eta = load_wave_probe_csv(path)
        assert len(t) == 100
        assert len(eta) == 100
        assert t[0] == pytest.approx(0.0, abs=1e-4)
        assert t[-1] == pytest.approx(0.99, abs=1e-4)
    finally:
        os.unlink(path)


def test_load_wave_probe_csv_nan_removal():
    """NaN values should be removed."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("time,elevation\n")
        f.write("0.0,0.01\n")
        f.write("0.1,nan\n")
        f.write("0.2,0.02\n")
        path = f.name

    try:
        t, eta = load_wave_probe_csv(path)
        assert len(t) == 2  # NaN row removed
    finally:
        os.unlink(path)


# ── Bilinear interpolation ──────────────────────────────────────────────────

def test_extract_prediction_at_grid_point():
    """Extracting at an exact grid point should return that value."""
    grid_x = np.array([0.0, 10.0, 20.0])
    grid_y = np.array([0.0, 5.0])
    timestamps = np.array([0.0, 0.1])

    # pred_fields: (2, 2, 3) — known values
    pred_fields = np.zeros((2, 2, 3))
    pred_fields[0, 0, 1] = 0.42  # grid point (y=0, x=10)
    pred_fields[1, 0, 1] = 0.84

    t, eta = extract_prediction_at_probe(pred_fields, timestamps,
                                          grid_x, grid_y,
                                          probe_x=10.0, probe_y=0.0)

    assert eta[0] == pytest.approx(0.42, abs=1e-6)
    assert eta[1] == pytest.approx(0.84, abs=1e-6)


def test_extract_prediction_bilinear():
    """Bilinear interpolation between grid points."""
    grid_x = np.array([0.0, 10.0])
    grid_y = np.array([0.0, 10.0])
    timestamps = np.array([0.0])

    pred_fields = np.zeros((1, 2, 2))
    pred_fields[0, 0, 0] = 0.0  # (y=0, x=0)
    pred_fields[0, 0, 1] = 1.0  # (y=0, x=10)
    pred_fields[0, 1, 0] = 0.0  # (y=10, x=0)
    pred_fields[0, 1, 1] = 1.0  # (y=10, x=10)

    # Probe at center: should get 0.5
    _, eta = extract_prediction_at_probe(pred_fields, timestamps,
                                          grid_x, grid_y,
                                          probe_x=5.0, probe_y=5.0)
    assert eta[0] == pytest.approx(0.5, abs=1e-6)


# ── Time synchronization ────────────────────────────────────────────────────

def test_synchronize_overlap():
    """Synchronization should find overlapping time range."""
    t_pred = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    eta_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    t_probe = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    eta_probe = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    t_c, ep, eq = synchronize_timeseries(t_pred, eta_pred, t_probe, eta_probe)

    # Overlap: [0.1, 0.4]
    assert len(t_c) == 4
    assert t_c[0] == pytest.approx(0.1)
    assert t_c[-1] == pytest.approx(0.4)
    # Pred values should be unchanged (exact timestamps)
    np.testing.assert_allclose(ep, [2.0, 3.0, 4.0, 5.0])


def test_synchronize_interpolation():
    """Probe should be interpolated to prediction timestamps."""
    t_pred = np.array([0.0, 1.0, 2.0])
    eta_pred = np.array([0.0, 0.0, 0.0])

    # Probe at higher rate
    t_probe = np.linspace(0, 2, 201)
    eta_probe = np.sin(2 * np.pi * t_probe)

    t_c, _, eq = synchronize_timeseries(t_pred, eta_pred, t_probe, eta_probe)

    # At t=0: sin(0) = 0
    assert eq[0] == pytest.approx(0.0, abs=0.01)
    # At t=1: sin(2π) = 0
    assert eq[1] == pytest.approx(0.0, abs=0.01)


# ── Metrics computation ─────────────────────────────────────────────────────

def test_metrics_perfect_prediction():
    """Perfect prediction should give RMSE=0, Pearson=1, VR=1."""
    N = 200
    dt = 0.1
    eta = np.sin(2 * np.pi * np.arange(N) * dt / 1.5) * 0.05

    m = compute_probe_metrics(eta, eta, dt, Hs_nominal=0.1)

    assert m['RMSE'] == pytest.approx(0.0, abs=1e-10)
    assert m['nRMSE'] == pytest.approx(0.0, abs=1e-10)
    assert m['Pearson_rho'] == pytest.approx(1.0, abs=1e-6)
    assert m['Variance_Ratio'] == pytest.approx(1.0, abs=1e-6)
    assert m['Hs_rel_error'] == pytest.approx(0.0, abs=1e-6)


def test_metrics_inverted_prediction():
    """Inverted prediction: Pearson = -1."""
    N = 200
    dt = 0.1
    eta = np.sin(2 * np.pi * np.arange(N) * dt / 1.5) * 0.05

    m = compute_probe_metrics(-eta, eta, dt)
    assert m['Pearson_rho'] == pytest.approx(-1.0, abs=1e-5)


def test_metrics_phase_lag():
    """Known phase lag should be detected."""
    N = 500
    dt = 0.1
    t = np.arange(N) * dt
    eta_probe = np.sin(2 * np.pi * t / 2.0) * 0.05

    # Shift by 5 samples
    lag_samples = 5
    eta_pred = np.roll(eta_probe, lag_samples)

    m = compute_probe_metrics(eta_pred, eta_probe, dt)
    assert abs(m['phase_lag_samples']) == pytest.approx(lag_samples, abs=2)


def test_metrics_spectrum():
    """Spectrum comparison with identical signals should give R²≈1."""
    N = 300
    dt = 0.1
    eta = np.sin(2 * np.pi * np.arange(N) * dt / 1.5) * 0.05

    m = compute_probe_metrics(eta, eta, dt)
    assert 'Spectral_R2' in m
    assert m['Spectral_R2'] == pytest.approx(1.0, abs=1e-6)


# ── Spectrum comparison function ─────────────────────────────────────────────

def test_compare_spectra_shape():
    """compare_spectra should return equal-length arrays."""
    N = 200
    dt = 0.1
    eta = np.random.randn(N) * 0.05

    freq, psd_pred, psd_probe = compare_spectra(eta, eta, dt)
    assert len(freq) == len(psd_pred) == len(psd_probe)
    assert np.all(freq >= 0)
    assert np.all(psd_pred >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
