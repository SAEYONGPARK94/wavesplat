"""
Unit tests for gt_generator.py, noise_model.py, data_writer.py
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest
from wavesim.gt_generator import make_grid, evaluate_field, compute_spectrum_1d
from wavesim.noise_model import add_noise, compute_confidence
from wavesim.data_writer import save_episode, load_episode
from wavesim.wave_generator import jonswap_components, WaveComponent
from wavesim.lidar_simulator import LidarConfig


# ── GT generator ─────────────────────────────────────────────────────────────

def test_make_grid_shape():
    X, Y = make_grid((0, 200), (0, 200), nx=50, ny=40)
    assert X.shape == (40, 50)
    assert Y.shape == (40, 50)


def test_evaluate_field_shape():
    comps = jonswap_components(Hs=1.2, Tp=7.0, n_components=5, seed=0)
    X, Y = make_grid((0, 200), (0, 200), nx=30, ny=30)
    Z, NX, NY, NZ = evaluate_field(comps, X, Y, t=0.0)
    assert Z.shape == (30, 30)
    assert NZ.shape == (30, 30)


def test_gt_nz_positive():
    """Ground truth surface normals must point upward."""
    comps = jonswap_components(Hs=1.2, Tp=7.0, n_components=10, seed=1)
    X, Y = make_grid((0, 200), (0, 200), nx=40, ny=40)
    _, _, _, NZ = evaluate_field(comps, X, Y, t=0.0)
    assert np.all(NZ > 0)


def test_spectrum_1d_shape():
    comps = jonswap_components(Hs=1.2, Tp=7.0, n_components=30, seed=0)
    spec = compute_spectrum_1d(comps, n_freq=32)
    assert spec.shape == (32, 2)
    assert np.all(spec[:, 0] > 0)  # frequencies positive
    assert np.all(spec[:, 1] >= 0)  # spectrum non-negative


# ── Noise model ───────────────────────────────────────────────────────────────

def test_noise_range_small():
    """Range noise should be small relative to typical LiDAR range."""
    rng = np.random.default_rng(0)
    N = 1000
    pts = np.tile([50.0, 0.0, 0.0], (N, 1)).astype(float)
    dirs = np.tile([1.0, 0.0, 0.0], (N, 1)).astype(float)

    noisy = add_noise(pts, dirs, sigma_range=0.02, rng=rng)
    diffs = np.linalg.norm(noisy - pts, axis=1)
    assert np.mean(diffs) < 0.1, f"Mean noise displacement {np.mean(diffs):.3f}m should be < 0.1m"


def test_confidence_range():
    """Confidence values must be in (0, 1]."""
    incidence = np.deg2rad(np.linspace(0, 80, 50))
    distances = np.linspace(10, 200, 50)
    conf = compute_confidence(incidence, distances)
    assert np.all(conf >= 0) and np.all(conf <= 1)


def test_confidence_near_higher():
    """Closer points should have higher confidence (all else equal)."""
    incidence = np.full(10, np.deg2rad(30))
    d_near = np.full(10, 20.0)
    d_far = np.full(10, 150.0)
    conf_near = compute_confidence(incidence, d_near).mean()
    conf_far = compute_confidence(incidence, d_far).mean()
    assert conf_near > conf_far


# ── Data writer (round-trip) ─────────────────────────────────────────────────

def test_hdf5_roundtrip():
    """Save and load episode: verify arrays are preserved exactly."""
    comps = jonswap_components(Hs=1.2, Tp=7.0, n_components=5, seed=0)
    cfg = LidarConfig()

    X, Y = make_grid((0, 200), (0, 200), nx=20, ny=20)
    gt_x = X[0, :]
    gt_y = Y[:, 0]

    N_frames = 3
    frames = []
    rng = np.random.default_rng(42)
    for i in range(N_frames):
        t = i * 0.1
        Z, _, _, _ = evaluate_field(comps, X, Y, t)
        N_pts = 20
        pts = rng.uniform(0, 200, (N_pts, 3))
        pts[:, 2] = rng.normal(0, 0.3, N_pts)
        frames.append({
            'timestamp': t,
            'lidar_points': pts,
            'lidar_valid_mask': np.ones(N_pts, dtype=bool),
            'lidar_incidence_angles': np.deg2rad(rng.uniform(60, 85, N_pts)),
            'lidar_confidence': rng.uniform(0.1, 0.9, N_pts),
            'gt_wave_field': Z,
        })

    spec = compute_spectrum_1d(comps, n_freq=16)

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        path = tmp.name

    try:
        save_episode(path, comps, cfg, frames, gt_x, gt_y,
                     'uniform', {'eta': 1.0},
                     'BF4', 1.2, 7.0, 3.3, 50.0, spec)
        loaded = load_episode(path)

        assert loaded['sea_state'] == 'BF4'
        assert abs(loaded['Hs'] - 1.2) < 1e-9
        assert loaded['gt_wave_field'].shape == (N_frames, 20, 20)
        assert loaded['lidar_points'].shape[0] == N_frames

        # Check GT field preserves values
        Z_orig = frames[0]['gt_wave_field']
        Z_load = loaded['gt_wave_field'][0]
        assert np.allclose(Z_orig, Z_load), "GT wave field not preserved in HDF5 round-trip"
    finally:
        os.unlink(path)


# ── Tank depth d=3.5m tests ──────────────────────────────────────────────────

def test_evaluate_field_tank_depth():
    """GT field evaluation at d=3.5m produces valid output."""
    comps = jonswap_components(Hs=0.08, Tp=1.2, n_components=5,
                                seed=0, depth=3.5)
    X, Y = make_grid((0, 50), (-4, 4), nx=30, ny=10)
    Z, NX, NY, NZ = evaluate_field(comps, X, Y, t=0.0)
    assert Z.shape == (10, 30)
    assert np.all(NZ > 0), "Surface normals must point upward at d=3.5m"
    # Wave amplitude should be on tank scale
    assert np.max(np.abs(Z)) < 1.0, \
        f"Wave amplitude {np.max(np.abs(Z)):.3f}m too large for tank"


def test_hdf5_roundtrip_tank_depth():
    """HDF5 save/load roundtrip at d=3.5m preserves depth."""
    comps = jonswap_components(Hs=0.05, Tp=1.0, n_components=3,
                                seed=0, depth=3.5)
    cfg = LidarConfig()

    X, Y = make_grid((0, 50), (-4, 4), nx=10, ny=10)
    gt_x = X[0, :]
    gt_y = Y[:, 0]

    frames = []
    rng = np.random.default_rng(42)
    for i in range(2):
        Z, _, _, _ = evaluate_field(comps, X, Y, t=i * 0.1)
        pts = rng.uniform(0, 50, (10, 3))
        pts[:, 2] = rng.normal(0, 0.02, 10)
        frames.append({
            'timestamp': i * 0.1,
            'lidar_points': pts,
            'lidar_valid_mask': np.ones(10, dtype=bool),
            'lidar_incidence_angles': np.deg2rad(rng.uniform(60, 85, 10)),
            'lidar_confidence': rng.uniform(0.1, 0.9, 10),
            'gt_wave_field': Z,
        })

    spec = compute_spectrum_1d(comps, n_freq=16)

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        path = tmp.name

    try:
        save_episode(path, comps, cfg, frames, gt_x, gt_y,
                     'uniform', {'eta': 1.0},
                     'tank', 0.05, 1.0, 3.3, 3.5, spec)
        loaded = load_episode(path)

        assert loaded['sea_state'] == 'tank'
        assert abs(loaded['Hs'] - 0.05) < 1e-9
        assert abs(loaded['Tp'] - 1.0) < 1e-9
        assert loaded['gt_wave_field'].shape == (2, 10, 10)

        Z_orig = frames[0]['gt_wave_field']
        Z_load = loaded['gt_wave_field'][0]
        assert np.allclose(Z_orig, Z_load)
    finally:
        os.unlink(path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
