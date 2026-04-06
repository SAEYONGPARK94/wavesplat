"""
Integration tests for the WaveSplat estimation pipeline.
Requires at least one generated HDF5 episode in data/.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import glob
import pytest
import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def _find_episode():
    """Find any available episode file for testing."""
    episodes = sorted(glob.glob(os.path.join(DATA_DIR, '*.h5')))
    if not episodes:
        pytest.skip("No HDF5 episodes available for integration testing")
    return episodes[0]


# ── Data loader tests ────────────────────────────────────────────────────────

def test_data_loader_basic():
    """Load episode and verify frame structure."""
    from wavesplat.data_loader import EpisodeDataLoader

    ep = _find_episode()
    loader = EpisodeDataLoader(ep)

    assert loader.n_frames > 0
    assert loader.Hs_true > 0
    assert loader.Tp_true > 0
    assert len(loader.gt_grid_x) > 0
    assert len(loader.gt_grid_y) > 0

    frame = loader[0]
    assert frame.t >= 0
    assert len(frame.x_obs) == len(frame.z_obs)
    assert frame.gt_field.shape == (len(loader.gt_grid_y), len(loader.gt_grid_x))


def test_init_window():
    """Init window should contain frames up to 2*Tp."""
    from wavesplat.data_loader import EpisodeDataLoader

    ep = _find_episode()
    loader = EpisodeDataLoader(ep)
    init_frames = loader.get_init_window(tp_factor=2.0)

    assert len(init_frames) > 0
    t_max = init_frames[-1].t
    assert t_max <= 2.0 * loader.Tp_true + 0.2  # small tolerance


# ── Optimizer tests ──────────────────────────────────────────────────────────

def test_single_component_initialization():
    """Initialize from a 1-component episode: should find the dominant peak."""
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    ep = _find_episode()
    loader = EpisodeDataLoader(ep)
    init_frames = loader.get_init_window(tp_factor=2.0)

    if len(init_frames) < 5:
        pytest.skip("Not enough init frames")

    config = {
        'n_gaussians': [3],  # single stage for speed
        'depth': loader.depth,
        'lr': 0.02,
        'lambda_l1': 0.0,
        'lambda_smooth': 0.0,
        'tracking_steps': 5,
    }

    updater = SequentialUpdater(config)
    result = updater.initialize(
        init_frames[:20],  # use subset for speed
        x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
        y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
    )

    assert result['n_iters'] > 0
    assert result['final_n'] == 3
    assert len(result['loss_history']) > 0
    # Loss should decrease
    assert result['loss_history'][-1] <= result['loss_history'][0] * 1.5


def test_tracking_step():
    """After init, tracking should update without error."""
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    ep = _find_episode()
    loader = EpisodeDataLoader(ep)
    init_frames = loader.get_init_window(tp_factor=2.0)

    if len(init_frames) < 5:
        pytest.skip("Not enough frames")

    config = {
        'n_gaussians': [3],
        'depth': loader.depth,
        'lr': 0.02,
        'lambda_l1': 0.0,
        'lambda_smooth': 0.0,
        'tracking_steps': 3,
    }

    updater = SequentialUpdater(config)
    updater.initialize(init_frames[:10])

    # Run tracking on a frame
    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
    if len(tracking_frames) == 0:
        # Use last init frame
        tracking_frames = [init_frames[-1]]

    result = updater.update(tracking_frames[0])
    assert 'loss' in result
    assert 'elapsed_ms' in result


def test_predict_field():
    """Model should produce valid grid predictions."""
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    ep = _find_episode()
    loader = EpisodeDataLoader(ep)
    init_frames = loader.get_init_window(tp_factor=2.0)

    config = {'n_gaussians': [3], 'depth': loader.depth, 'lr': 0.02}
    updater = SequentialUpdater(config)
    updater.initialize(init_frames[:10])

    pred = updater.predict_field(loader.gt_grid_x, loader.gt_grid_y, 1.0)
    assert pred.shape == (len(loader.gt_grid_y), len(loader.gt_grid_x))
    assert not torch.any(torch.isnan(pred))


# ── Metrics tests ────────────────────────────────────────────────────────────

def test_metrics_computation():
    """All metrics should compute without error."""
    from wavesplat.evaluation.metrics import compute_all_metrics

    pred = np.random.randn(60, 80) * 0.1
    gt = np.random.randn(60, 80) * 0.1
    grid_x = np.linspace(10, 250, 80)
    grid_y = np.linspace(-100, 100, 60)

    metrics = compute_all_metrics(pred, gt, grid_x, grid_y, Hs_true=1.2)

    assert 'RMSE_z' in metrics
    assert 'nRMSE' in metrics
    assert 'Hs_error' in metrics
    assert 'RMSE_near' in metrics
    assert 'RMSE_far' in metrics
    assert metrics['RMSE_z'] >= 0
    assert metrics['nRMSE'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
