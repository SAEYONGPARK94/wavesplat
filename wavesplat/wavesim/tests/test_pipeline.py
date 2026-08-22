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


# ── Tank depth d=3.5m pipeline tests ─────────────────────────────────────────

def test_pipeline_tank_depth():
    """Full pipeline at d=3.5m with tank-scale wave parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = run_episode(
            sea_state='BF3',
            n_components=5,
            n_pts=10,
            seed=0,
            duration=0.5,
            dt=0.1,
            depth=3.5,
            output_dir=tmpdir,
            verbose=False
        )
        assert os.path.exists(fp)
        data = load_episode(fp)

        assert data['lidar_points'].shape[0] >= 1
        gt = data['gt_wave_field']
        # GT field should be physically valid
        for i in range(gt.shape[0]):
            assert abs(np.mean(gt[i])) < 1.0, \
                f"Frame {i}: mean elevation too large at d=3.5m"


def test_pipeline_tank_depth_dispersion():
    """At d=3.5m, saved wave components satisfy dispersion relation."""
    from wavesim.wave_generator import full_dispersion
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = run_episode(
            sea_state='BF4',
            n_components=10,
            n_pts=10,
            seed=42,
            duration=0.2,
            dt=0.1,
            depth=3.5,
            output_dir=tmpdir,
            verbose=False
        )
        data = load_episode(fp)
        comps = data['wave_components']  # (N, 6): A, omega, kx, ky, phi, Q

        for i in range(comps.shape[0]):
            omega = comps[i, 1]
            kx, ky = comps[i, 2], comps[i, 3]
            k_mag = np.sqrt(kx**2 + ky**2)
            omega_disp = full_dispersion(k_mag, depth=3.5)
            assert abs(omega - omega_disp) < 1e-3, \
                f"Component {i}: ω={omega:.4f} != dispersion ω={omega_disp:.4f} at d=3.5m"


# ── HOS pipeline test ────────────────────────────────────────────────────────

def test_pipeline_hos_basic():
    """Pipeline runs with HOS enabled and produces valid nonlinear output."""
    from wavesim.config import WaveSimConfig, HOSCfg

    cfg = WaveSimConfig()
    cfg.hos = HOSCfg(enabled=True, M=2, Nx=64, Ny=64, Lx=300.0, Ly=250.0, dt=0.02)

    with tempfile.TemporaryDirectory() as tmpdir:
        fp = run_episode(
            sea_state='BF3',
            n_components=5,
            n_pts=10,
            seed=0,
            duration=0.3,
            dt=0.1,
            output_dir=tmpdir,
            verbose=False,
            cfg=cfg,
        )
        assert os.path.exists(fp), "HOS pipeline did not produce output file"
        data = load_episode(fp)

        # Must have produced frames
        assert data['lidar_points'].shape[0] >= 1
        gt = data['gt_wave_field']
        assert gt.shape[0] >= 2, f"Expected >=2 frames, got {gt.shape[0]}"

        # GT field should be physically valid (zero-mean waves)
        for i in range(gt.shape[0]):
            assert abs(np.mean(gt[i])) < 0.5, \
                f"Frame {i}: mean elevation {np.mean(gt[i]):.3f}m != ~0"


def test_speed_tag_is_empty_only_when_stationary():
    """Regression for the C-batch filename collision (2026-08-20, AMENDMENT_1 A-3).

    `U=0, dir=0` and `U=11.3, dir=0` used to assemble the SAME filename, and the
    second overwrote the first -- an entire arm of a 2x2 design would have vanished
    silently.  The tag must be non-empty and DISTINCT for distinct speeds, and empty
    at 0 so no pre-2026-08-21 filename changes.
    """
    from wavesim.pipeline import speed_tag

    assert speed_tag(0.0) == ''
    assert speed_tag(-0.0) == ''
    assert speed_tag(11.3) == '_u11p3'
    assert speed_tag(5.0) == '_u5'
    assert speed_tag(-5.0) == '_um5'
    # the actual defect: distinct speeds must not collide with each other or with 0
    tags = {speed_tag(u) for u in (0.0, 2.5, 5.0, 11.3, 15.0)}
    assert len(tags) == 5, f"speed tags collide: {tags}"
    # tags stay single shell-safe tokens (no '.', no '-', no whitespace)
    for u in (2.5, 11.3, -5.0):
        assert not (set(speed_tag(u)) & set('.- ')), speed_tag(u)


def test_pipeline_filename_and_attr_carry_ship_speed():
    """A moving-platform episode must not be able to overwrite the stationary one.

    Also pins the backward-compatibility half: at the default speed the filename
    is exactly what it was before the parameter existed, and no attr is added.
    """
    from wavesim.pipeline import run_episode as _run
    import h5py

    common = dict(sea_state='BF4', n_components=5, n_pts=10, seed=0,
                  raydrop_model='uniform', duration=0.3, dt=0.1, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        fp_still = _run(output_dir=tmpdir, **common)
        fp_moving = _run(output_dir=tmpdir, ship_speed_mps=11.3, **common)

        assert fp_still != fp_moving, \
            "moving and stationary episodes collide on one filename (C-batch defect)"
        assert os.path.exists(fp_still) and os.path.exists(fp_moving)
        # backward compatible: the stationary name is the historical one
        assert os.path.basename(fp_still) == 'episode_BF4_5comp_seed0_uniform.h5'
        assert '_u11p3' in os.path.basename(fp_moving)

        with h5py.File(fp_moving, 'r') as f:
            assert f['config'].attrs['ship_speed_mps'] == pytest.approx(11.3)
        with h5py.File(fp_still, 'r') as f:
            assert 'ship_speed_mps' not in f['config'].attrs, \
                "stationary episodes must stay attr-identical to pre-existing files"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
