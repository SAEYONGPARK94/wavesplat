"""
data_writer.py
HDF5 episode data writer for WaveSim.

File structure per episode:
  episode_BF4_30comp_seed42.h5
  ├── /config/
  ├── /frames/
  ├── /raydrop_model/
  └── /gt_params/
"""

import numpy as np
import h5py
import os
from typing import List, Dict, Any, Optional
from wavesim.wave_generator import WaveComponent


def save_episode(filepath: str,
                 components: List[WaveComponent],
                 lidar_cfg,
                 frames_data: List[Dict[str, Any]],
                 gt_grid_x: np.ndarray,
                 gt_grid_y: np.ndarray,
                 raydrop_model_type: str,
                 raydrop_params: Dict[str, Any],
                 sea_state: str,
                 Hs: float, Tp: float,
                 gamma: float,
                 depth: float,
                 spectrum_1d: np.ndarray):
    """
    Save a full episode to HDF5.

    Args:
        filepath: output .h5 file path
        components: list of WaveComponent
        lidar_cfg: LidarConfig instance
        frames_data: list of per-frame dicts with keys:
            'timestamp', 'lidar_points', 'lidar_valid_mask',
            'lidar_incidence_angles', 'lidar_confidence', 'gt_wave_field'
        gt_grid_x, gt_grid_y: 1D coordinate arrays
        raydrop_model_type: "uniform" or "slope_dependent"
        raydrop_params: dict of model parameters
        sea_state: e.g. "BF4"
        Hs, Tp, gamma, depth: wave parameters
        spectrum_1d: (N_freq, 2) array [omega, S]
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    # Pack wave components into array: [A, omega, kx, ky, phi, Q]
    comp_array = np.array([[c.A, c.omega, c.kx, c.ky, c.phi, c.Q]
                            for c in components])

    # Determine max points across frames
    max_pts = max(f['lidar_points'].shape[0] for f in frames_data)
    N_frames = len(frames_data)
    Ny, Nx = frames_data[0]['gt_wave_field'].shape

    with h5py.File(filepath, 'w') as f:
        # /config
        cfg = f.create_group('config')
        cfg.attrs['sea_state'] = sea_state
        cfg.attrs['Hs'] = Hs
        cfg.attrs['Tp'] = Tp
        cfg.attrs['gamma'] = gamma
        cfg.attrs['n_components'] = len(components)
        cfg.attrs['depth'] = depth
        cfg.create_dataset('wave_components', data=comp_array)

        lidar_grp = cfg.create_group('lidar_config')
        lidar_grp.attrs['height'] = lidar_cfg.height
        lidar_grp.attrs['channels'] = lidar_cfg.channels
        lidar_grp.attrs['scan_rate'] = lidar_cfg.scan_rate

        # /frames
        frm = f.create_group('frames')
        timestamps = np.array([fd['timestamp'] for fd in frames_data])
        frm.create_dataset('timestamps', data=timestamps)

        # Pre-allocate padded arrays
        pts_arr = np.full((N_frames, max_pts, 3), np.nan)
        valid_arr = np.zeros((N_frames, max_pts), dtype=bool)
        incidence_arr = np.full((N_frames, max_pts), np.nan)
        conf_arr = np.full((N_frames, max_pts), np.nan)
        gt_arr = np.full((N_frames, Ny, Nx), np.nan)

        for i, fd in enumerate(frames_data):
            n = fd['lidar_points'].shape[0]
            pts_arr[i, :n] = fd['lidar_points']
            valid_arr[i, :n] = True
            incidence_arr[i, :n] = fd['lidar_incidence_angles']
            conf_arr[i, :n] = fd['lidar_confidence']
            gt_arr[i] = fd['gt_wave_field']

        frm.create_dataset('lidar_points', data=pts_arr, compression='gzip')
        frm.create_dataset('lidar_valid_mask', data=valid_arr, compression='gzip')
        frm.create_dataset('lidar_incidence_angles', data=incidence_arr, compression='gzip')
        frm.create_dataset('lidar_confidence', data=conf_arr, compression='gzip')
        frm.create_dataset('gt_wave_field', data=gt_arr, compression='gzip')
        frm.create_dataset('gt_grid_x', data=gt_grid_x)
        frm.create_dataset('gt_grid_y', data=gt_grid_y)

        # /raydrop_model
        rdm = f.create_group('raydrop_model')
        rdm.attrs['model_type'] = raydrop_model_type
        for k, v in raydrop_params.items():
            rdm.attrs[k] = v

        # /gt_params
        gtp = f.create_group('gt_params')
        gtp.attrs['Hs_true'] = Hs
        gtp.attrs['Tp_true'] = Tp
        gtp.create_dataset('spectrum_1d', data=spectrum_1d)


def load_episode(filepath: str) -> Dict[str, Any]:
    """Load an episode from HDF5. Returns dict with all data."""
    data = {}
    with h5py.File(filepath, 'r') as f:
        # Config
        data['sea_state'] = f['config'].attrs['sea_state']
        data['Hs'] = f['config'].attrs['Hs']
        data['Tp'] = f['config'].attrs['Tp']
        data['wave_components'] = f['config/wave_components'][:]

        # Frames
        data['timestamps'] = f['frames/timestamps'][:]
        data['lidar_points'] = f['frames/lidar_points'][:]
        data['lidar_valid_mask'] = f['frames/lidar_valid_mask'][:]
        data['lidar_incidence_angles'] = f['frames/lidar_incidence_angles'][:]
        data['lidar_confidence'] = f['frames/lidar_confidence'][:]
        data['gt_wave_field'] = f['frames/gt_wave_field'][:]
        data['gt_grid_x'] = f['frames/gt_grid_x'][:]
        data['gt_grid_y'] = f['frames/gt_grid_y'][:]

        # GT params
        data['Hs_true'] = f['gt_params'].attrs['Hs_true']
        data['Tp_true'] = f['gt_params'].attrs['Tp_true']
        data['spectrum_1d'] = f['gt_params/spectrum_1d'][:]

    return data
