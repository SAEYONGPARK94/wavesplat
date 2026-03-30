"""
Integration test: full WaveSim pipeline end-to-end.
Uses assumed/synthetic data (BF4, 5 components, 10 pts, 3 frames).
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest
from wavesim.pipeline import run_episode
from wavesim.data_writer import load_episode


def test_pipeline_uniform_basic():
    """Pipeline runs without error and produces valid output (uniform model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = run_episode(
            sea_state='BF4',
            n_components=5,
            n_pts=10,
            seed=0,
            raydrop_model='uniform',
            duration=0.5,
            dt=0.1,
            output_dir=tmpdir,
            verbose=False
        )
        assert os.path.exists(fp), "Output file not created"
        data = load_episode(fp)

        # Check non-empty frames
        assert data['lidar_points'].shape[0] >= 1, "Should have at least 1 frame"

        # Points within expected range
        valid = data['lidar_valid_mask']
        pts = data['lidar_points']
        valid_pts = pts[valid]
        assert valid_pts.shape[1] == 3

        # GT wave field has sensible Hs
        gt = data['gt_wave_field']
        Hs_est = 4.0 * np.nanstd(gt)
        assert 0.1 < Hs_est < 5.0, f"GT Hs={Hs_est:.3f}m out of range"

        print(f"  Hs_est={Hs_est:.3f}m (target=1.2m), frames={gt.shape[0]}")


def test_pipeline_slope_model():
    """Pipeline runs with slope-based raydrop model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = run_episode(
            sea_state='BF3',
            n_components=5,
            n_pts=10,
            seed=1,
            raydrop_model='slope',
            duration=0.3,
            dt=0.1,
            output_dir=tmpdir,
            verbose=False
        )
        assert os.path.exists(fp)
        data = load_episode(fp)
        assert data['lidar_points'].shape[0] >= 1


def test_pipeline_hdf5_consistency():
    """Saved GT field values are physically consistent with wave parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = run_episode(
            sea_state='BF4',
            n_components=10,
            n_pts=20,
            seed=7,
            duration=0.2,
            dt=0.1,
            output_dir=tmpdir,
            verbose=False
        )
        data = load_episode(fp)

        # GT surface: mean should be near 0 (waves are zero-mean)
        for i in range(data['gt_wave_field'].shape[0]):
            Z = data['gt_wave_field'][i]
            assert abs(np.mean(Z)) < 0.5, f"Frame {i}: mean elevation {np.mean(Z):.3f}m != ~0"

        # Confidence values in [0, 1]
        conf = data['lidar_confidence']
        valid = data['lidar_valid_mask']
        assert np.all(conf[valid] >= 0) and np.all(conf[valid] <= 1)

        # Incidence angles in [0, π/2]
        theta = data['lidar_incidence_angles']
        assert np.all(theta[valid] >= 0) and np.all(theta[valid] <= np.pi / 2 + 0.01)


def test_pipeline_different_sea_states():
    """Pipeline works for all sea state presets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for ss in ['BF2', 'BF3', 'BF4', 'BF5']:
            fp = run_episode(
                sea_state=ss,
                n_components=3,
                n_pts=5,
                seed=0,
                duration=0.2,
                dt=0.1,
                output_dir=tmpdir,
                verbose=False
            )
            assert os.path.exists(fp), f"{ss} did not produce output file"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
