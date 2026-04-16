"""
real_data.py
Pipeline for converting real Falcon K1 LiDAR data to WaveSplat FrameData.

Falcon K1 coordinate system:  X=up, Y=right, Z=forward
WaveSplat coordinate system:   x=forward, y=lateral, z=elevation

Transform:  ws_x = k1_z,  ws_y = k1_y,  ws_z = k1_x

Supported input formats:
  - PCD (Point Cloud Data) files from Seyond get_pcd tool
  - CSV files from Seyond recording
  - Directory of sequential PCD/CSV files (one per frame)
"""

import numpy as np
import os
import glob
import h5py
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from wavesplat.data_loader import FrameData
from wavesplat.model.confidence import observation_weights


@dataclass
class LiDARMountConfig:
    """LiDAR mounting configuration for the tank experiment."""
    height: float = 3.5        # sensor height above water surface [m]
    tilt_angle: float = 0.0    # downward tilt from horizontal [rad]
    x_offset: float = 0.0      # forward offset from reference point [m]
    y_offset: float = 0.0      # lateral offset [m]
    d_ref: float = 5.0         # reference distance for confidence [m]
                                # (tank scale, not ocean 125m)


@dataclass
class SurfaceFilterConfig:
    """Parameters for extracting water surface from raw point cloud."""
    z_min: float = -1.0        # min elevation relative to still water [m]
    z_max: float = 1.0         # max elevation relative to still water [m]
    x_min: float = 0.0         # min forward distance [m]
    x_max: float = 110.0       # max forward distance (tank length) [m]
    y_min: float = -4.0        # min lateral (half tank width) [m]
    y_max: float = 4.0         # max lateral [m]
    intensity_min: int = 0     # min reflectance for valid point
    max_range: float = 50.0    # max sensor range to accept [m]


def read_pcd(filepath: str) -> np.ndarray:
    """
    Read a PCD (Point Cloud Data) file.

    Supports ASCII and binary PCD formats from Seyond get_pcd tool.
    Returns (N, 4+) array: [x, y, z, intensity, ...] in Falcon K1 coords.
    """
    with open(filepath, 'r', errors='replace') as f:
        header_lines = []
        n_points = 0
        fields = []
        data_format = 'ascii'

        while True:
            line = f.readline().strip()
            header_lines.append(line)

            if line.startswith('FIELDS'):
                fields = line.split()[1:]
            elif line.startswith('POINTS'):
                n_points = int(line.split()[1])
            elif line.startswith('DATA'):
                data_format = line.split()[1]
                break

        if data_format != 'ascii':
            # For binary PCD, fall back to open3d if available
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(filepath)
                pts = np.asarray(pcd.points)
                if pcd.has_colors():
                    intensity = np.asarray(pcd.colors)[:, 0] * 255
                else:
                    intensity = np.ones(len(pts)) * 128
                return np.column_stack([pts, intensity])
            except ImportError:
                raise RuntimeError(
                    "Binary PCD requires open3d. Install: pip install open3d"
                )

        # ASCII PCD
        n_fields = len(fields)
        data = np.zeros((n_points, n_fields))
        for i in range(n_points):
            line = f.readline().strip()
            if not line:
                data = data[:i]
                break
            vals = line.split()
            data[i, :len(vals)] = [float(v) for v in vals[:n_fields]]

    return data


def read_csv_pointcloud(filepath: str) -> np.ndarray:
    """
    Read CSV point cloud from Seyond recording.

    Expected columns: x, y, z, intensity (or reflectance), ...
    Returns (N, 4+) array in Falcon K1 coords.
    """
    # Try to detect header
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()

    if any(c.isalpha() for c in first_line):
        # Has header
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    else:
        data = np.genfromtxt(filepath, delimiter=',')

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data


def falcon_to_wavesplat(points_k1: np.ndarray,
                         mount: LiDARMountConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform Falcon K1 coordinates to WaveSplat coordinates.

    Falcon K1: X=up, Y=right, Z=forward
    WaveSplat: x=forward, y=lateral, z=elevation (relative to still water)

    Also accounts for sensor mounting height and tilt.

    Args:
        points_k1: (N, 3+) array in Falcon K1 coords [X_up, Y_right, Z_forward, ...]
        mount: mounting configuration

    Returns:
        ws_x: (N,) forward coordinates [m]
        ws_y: (N,) lateral coordinates [m]
        ws_z: (N,) elevation relative to still water [m]
    """
    k1_x = points_k1[:, 0]  # up
    k1_y = points_k1[:, 1]  # right
    k1_z = points_k1[:, 2]  # forward

    # Apply tilt rotation (around Y-axis in K1 frame)
    # Positive tilt_angle = sensor tilted downward (forward axis points down)
    # This rotates K1 Z(forward) toward -X(down): standard Ry(-tilt)
    if abs(mount.tilt_angle) > 1e-6:
        cos_t = np.cos(mount.tilt_angle)
        sin_t = np.sin(mount.tilt_angle)
        k1_x_rot = cos_t * k1_x - sin_t * k1_z
        k1_z_rot = sin_t * k1_x + cos_t * k1_z
        k1_x = k1_x_rot
        k1_z = k1_z_rot

    # Coordinate transform + mounting offset
    ws_x = k1_z + mount.x_offset       # forward
    ws_y = k1_y + mount.y_offset       # lateral
    ws_z = k1_x - mount.height          # elevation relative to still water

    return ws_x, ws_y, ws_z


def compute_incidence_angles(ws_x: np.ndarray, ws_y: np.ndarray,
                              ws_z: np.ndarray,
                              mount: LiDARMountConfig) -> np.ndarray:
    """
    Compute incidence angles from ray geometry.

    The incidence angle is the angle between the LiDAR ray and the local
    surface normal (assumed vertical for water surface).

    For a ray from sensor at (x_offset, y_offset, height) to point (x, y, z),
    the incidence angle θ = arccos(|ray_z| / |ray|).

    Args:
        ws_x, ws_y, ws_z: WaveSplat coordinates of surface points
        mount: sensor mounting config

    Returns:
        angles: (N,) incidence angles [rad]
    """
    # Ray vectors from sensor to each point
    dx = ws_x - mount.x_offset
    dy = ws_y - mount.y_offset
    dz = ws_z - mount.height  # negative (pointing down)

    ray_length = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-12)

    # Incidence = angle from vertical
    # cos(θ) = |vertical component| / |ray|
    cos_theta = np.abs(dz) / ray_length
    angles = np.arccos(np.clip(cos_theta, 0.0, 1.0))

    return angles


def compute_distances(ws_x: np.ndarray, ws_y: np.ndarray,
                       ws_z: np.ndarray,
                       mount: LiDARMountConfig) -> np.ndarray:
    """Compute sensor-to-point distances."""
    dx = ws_x - mount.x_offset
    dy = ws_y - mount.y_offset
    dz = ws_z - mount.height
    return np.sqrt(dx**2 + dy**2 + dz**2)


def filter_surface_points(ws_x: np.ndarray, ws_y: np.ndarray,
                            ws_z: np.ndarray,
                            intensity: np.ndarray,
                            cfg: SurfaceFilterConfig) -> np.ndarray:
    """
    Filter raw point cloud to keep only water surface returns.

    Removes: walls, floor, above-surface objects, out-of-range points.

    Returns:
        mask: (N,) boolean mask of valid surface points
    """
    mask = (
        (ws_z >= cfg.z_min) & (ws_z <= cfg.z_max) &
        (ws_x >= cfg.x_min) & (ws_x <= cfg.x_max) &
        (ws_y >= cfg.y_min) & (ws_y <= cfg.y_max) &
        (intensity >= cfg.intensity_min)
    )

    # Range filter (from sensor origin, not coordinate origin)
    # Sensor is at approximately (0, 0, height) in WaveSplat coords
    # but we don't have mount info here, so use Euclidean from (0,0,0)
    # which is the still-water surface below sensor — close enough for filtering
    dist_horiz = np.sqrt(ws_x**2 + ws_y**2)
    mask &= (dist_horiz <= cfg.max_range)

    return mask


def process_single_frame(filepath: str,
                          timestamp: float,
                          mount: LiDARMountConfig,
                          surface_cfg: SurfaceFilterConfig,
                          device: str = 'cpu') -> Optional[FrameData]:
    """
    Process a single PCD/CSV file into a FrameData object.

    Args:
        filepath: path to PCD or CSV point cloud file
        timestamp: frame timestamp [s]
        mount: LiDAR mounting configuration
        surface_cfg: surface filtering parameters
        device: torch device

    Returns:
        FrameData or None if no valid surface points
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pcd':
        raw = read_pcd(filepath)
    elif ext == '.csv':
        raw = read_csv_pointcloud(filepath)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    if len(raw) == 0:
        return None

    # Extract intensity (column 3 if available)
    intensity = raw[:, 3] if raw.shape[1] > 3 else np.ones(len(raw)) * 128

    # Transform to WaveSplat coordinates
    ws_x, ws_y, ws_z = falcon_to_wavesplat(raw, mount)

    # Filter surface points
    surface_mask = filter_surface_points(ws_x, ws_y, ws_z, intensity,
                                          surface_cfg)

    if not np.any(surface_mask):
        return None

    ws_x = ws_x[surface_mask]
    ws_y = ws_y[surface_mask]
    ws_z = ws_z[surface_mask]

    # Compute incidence angles and distances
    inc_angles = compute_incidence_angles(ws_x, ws_y, ws_z, mount)
    distances = compute_distances(ws_x, ws_y, ws_z, mount)

    # Compute confidence weights
    dev = torch.device(device)
    conf = observation_weights(
        torch.tensor(inc_angles, dtype=torch.float32),
        torch.tensor(distances, dtype=torch.float32),
        d_ref=mount.d_ref
    )

    return FrameData(
        t=timestamp,
        x_obs=torch.tensor(ws_x, dtype=torch.float32, device=dev),
        y_obs=torch.tensor(ws_y, dtype=torch.float32, device=dev),
        z_obs=torch.tensor(ws_z, dtype=torch.float32, device=dev),
        confidence=conf.to(dev),
        incidence_angles=torch.tensor(inc_angles, dtype=torch.float32, device=dev),
        gt_field=None,
    )


class RealDataLoader:
    """
    Load real Falcon K1 LiDAR data and iterate over frames.

    Usage:
        loader = RealDataLoader(
            data_dir='experiment/case_01/',
            mount=LiDARMountConfig(height=3.5),
            frame_rate=10.0,
        )
        for frame in loader:
            # use frame.x_obs, frame.z_obs, etc.

    The data_dir should contain sequentially named PCD or CSV files.
    """

    def __init__(self,
                 data_dir: str,
                 mount: LiDARMountConfig = None,
                 surface_cfg: SurfaceFilterConfig = None,
                 frame_rate: float = 10.0,
                 file_pattern: str = '*.pcd',
                 device: str = 'cpu',
                 wave_probe_file: Optional[str] = None,
                 Hs_nominal: float = 0.0,
                 Tp_nominal: float = 0.0,
                 depth: float = 3.5):
        """
        Args:
            data_dir: directory containing PCD/CSV files
            mount: LiDAR mounting config (default: 3.5m height)
            surface_cfg: surface filter config
            frame_rate: LiDAR frame rate [Hz]
            file_pattern: glob pattern for point cloud files
            device: torch device
            wave_probe_file: optional CSV with wave probe measurements
            Hs_nominal: nominal significant wave height [m]
            Tp_nominal: nominal peak period [s]
            depth: water depth [m]
        """
        self.mount = mount or LiDARMountConfig()
        self.surface_cfg = surface_cfg or SurfaceFilterConfig()
        self.device = torch.device(device)
        self.frame_rate = frame_rate
        self.depth = depth
        self.Hs_true = Hs_nominal
        self.Tp_true = Tp_nominal

        # Find and sort point cloud files
        self.files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        if not self.files:
            # Try CSV if no PCD found
            self.files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
        self.n_frames = len(self.files)

        # Generate timestamps (try to extract from PCD headers, fallback to uniform)
        self.timestamps = self._extract_timestamps(frame_rate)

        # Load wave probe data if available
        self.wave_probe = None
        if wave_probe_file and os.path.exists(wave_probe_file):
            self.wave_probe = np.genfromtxt(wave_probe_file, delimiter=',',
                                             skip_header=1)

        # Placeholder GT grid (no GT for real data)
        self.gt_grid_x = torch.linspace(0, 50, 50, device=self.device)
        self.gt_grid_y = torch.linspace(-4, 4, 16, device=self.device)

    def _extract_timestamps(self, frame_rate: float) -> np.ndarray:
        """
        Extract timestamps from PCD headers or filenames.
        Falls back to uniform spacing at frame_rate.

        Falcon K1 PCD files contain ts_start_us in the header.
        Some recording tools encode timestamps in filenames (e.g., 000001_1234567890.pcd).
        """
        timestamps = np.arange(self.n_frames, dtype=np.float64) / frame_rate

        if not self.files:
            return timestamps

        # Try parsing from PCD header (ts_start_us field)
        if self.files[0].endswith('.pcd'):
            ts_list = []
            for filepath in self.files:
                ts = self._read_pcd_timestamp(filepath)
                if ts is None:
                    return timestamps  # fallback to uniform
                ts_list.append(ts)

            if ts_list:
                ts_arr = np.array(ts_list)
                # Convert to relative seconds from first frame
                ts_arr = (ts_arr - ts_arr[0]) / 1e6  # μs → s
                if np.all(np.diff(ts_arr) > 0):  # monotonically increasing
                    return ts_arr

        return timestamps

    @staticmethod
    def _read_pcd_timestamp(filepath: str) -> Optional[float]:
        """Extract timestamp from PCD header (ts_start_us field)."""
        try:
            with open(filepath, 'r', errors='replace') as f:
                for _ in range(30):  # header is typically <20 lines
                    line = f.readline()
                    if not line:
                        break
                    if line.startswith('DATA'):
                        break
                    # Seyond PCD may have COMMENT with timestamp info
                    if 'ts_start_us' in line or 'TIMESTAMP' in line.upper():
                        parts = line.split()
                        for p in parts:
                            try:
                                val = float(p)
                                if val > 1e9:  # looks like microsecond timestamp
                                    return val
                            except ValueError:
                                continue
        except (IOError, UnicodeDecodeError):
            pass
        return None

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx: int) -> Optional[FrameData]:
        return process_single_frame(
            self.files[idx],
            timestamp=self.timestamps[idx],
            mount=self.mount,
            surface_cfg=self.surface_cfg,
            device=str(self.device),
        )

    def __iter__(self):
        for i in range(self.n_frames):
            frame = self[i]
            if frame is not None:
                yield frame

    def get_init_window(self, tp_factor: float = 2.0) -> List[FrameData]:
        """Get frames for initialization window."""
        t_init = tp_factor * self.Tp_true if self.Tp_true > 0 else 5.0
        frames = []
        for i in range(self.n_frames):
            if self.timestamps[i] > t_init:
                break
            frame = self[i]
            if frame is not None:
                frames.append(frame)
        return frames

    def get_tracking_frames(self, tp_factor: float = 2.0) -> List[FrameData]:
        """Get frames after initialization window."""
        t_init = tp_factor * self.Tp_true if self.Tp_true > 0 else 5.0
        frames = []
        for i in range(self.n_frames):
            if self.timestamps[i] > t_init:
                frame = self[i]
                if frame is not None:
                    frames.append(frame)
        return frames


def convert_to_hdf5(data_dir: str,
                     output_path: str,
                     mount: LiDARMountConfig = None,
                     surface_cfg: SurfaceFilterConfig = None,
                     frame_rate: float = 10.0,
                     file_pattern: str = '*.pcd',
                     Hs_nominal: float = 0.0,
                     Tp_nominal: float = 0.0,
                     depth: float = 3.5,
                     gt_grid_x: np.ndarray = None,
                     gt_grid_y: np.ndarray = None):
    """
    Convert a directory of PCD/CSV files to WaveSplat HDF5 format.

    This creates an HDF5 file compatible with EpisodeDataLoader,
    allowing the existing run.py pipeline to process real data unchanged.

    Args:
        data_dir: directory containing point cloud files
        output_path: output .h5 file path
        mount: LiDAR mounting config
        surface_cfg: surface filtering config
        frame_rate: frame rate [Hz]
        file_pattern: glob pattern for files
        Hs_nominal: nominal Hs from wavemaker setting [m]
        Tp_nominal: nominal Tp from wavemaker setting [s]
        depth: water depth [m]
        gt_grid_x, gt_grid_y: GT grid coordinates (optional)
    """
    mount = mount or LiDARMountConfig()
    surface_cfg = surface_cfg or SurfaceFilterConfig()

    files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No point cloud files in {data_dir}")

    N_frames = len(files)
    timestamps = np.arange(N_frames) / frame_rate

    # Default GT grid
    if gt_grid_x is None:
        gt_grid_x = np.linspace(0, 50, 50)
    if gt_grid_y is None:
        gt_grid_y = np.linspace(-4, 4, 16)

    Ny = len(gt_grid_y)
    Nx = len(gt_grid_x)

    # First pass: determine max points
    max_pts = 0
    for filepath in files:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.pcd':
            raw = read_pcd(filepath)
        else:
            raw = read_csv_pointcloud(filepath)
        if len(raw) > 0:
            intensity = raw[:, 3] if raw.shape[1] > 3 else np.ones(len(raw)) * 128
            ws_x, ws_y, ws_z = falcon_to_wavesplat(raw, mount)
            mask = filter_surface_points(ws_x, ws_y, ws_z, intensity, surface_cfg)
            max_pts = max(max_pts, int(np.sum(mask)))

    if max_pts == 0:
        raise ValueError("No valid surface points found in any frame")

    # Add padding
    max_pts = int(max_pts * 1.1) + 10

    # Second pass: build arrays
    pts_arr = np.full((N_frames, max_pts, 3), np.nan)
    valid_arr = np.zeros((N_frames, max_pts), dtype=bool)
    incidence_arr = np.full((N_frames, max_pts), np.nan)
    conf_arr = np.full((N_frames, max_pts), np.nan)
    gt_arr = np.zeros((N_frames, Ny, Nx))  # no GT for real data

    for i, filepath in enumerate(files):
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.pcd':
            raw = read_pcd(filepath)
        else:
            raw = read_csv_pointcloud(filepath)

        if len(raw) == 0:
            continue

        intensity = raw[:, 3] if raw.shape[1] > 3 else np.ones(len(raw)) * 128
        ws_x, ws_y, ws_z = falcon_to_wavesplat(raw, mount)
        mask = filter_surface_points(ws_x, ws_y, ws_z, intensity, surface_cfg)

        if not np.any(mask):
            continue

        ws_x = ws_x[mask]
        ws_y = ws_y[mask]
        ws_z = ws_z[mask]
        n = len(ws_x)

        pts_arr[i, :n, 0] = ws_x
        pts_arr[i, :n, 1] = ws_y
        pts_arr[i, :n, 2] = ws_z
        valid_arr[i, :n] = True

        inc = compute_incidence_angles(ws_x, ws_y, ws_z, mount)
        dist = compute_distances(ws_x, ws_y, ws_z, mount)
        incidence_arr[i, :n] = inc

        conf_t = observation_weights(
            torch.tensor(inc, dtype=torch.float32),
            torch.tensor(dist, dtype=torch.float32),
            d_ref=mount.d_ref
        ).numpy()
        conf_arr[i, :n] = conf_t

    # Write HDF5
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['sea_state'] = 'tank_experiment'
        cfg.attrs['Hs'] = Hs_nominal
        cfg.attrs['Tp'] = Tp_nominal
        cfg.attrs['gamma'] = 0.0
        cfg.attrs['n_components'] = 0
        cfg.attrs['depth'] = depth
        cfg.create_dataset('wave_components', data=np.zeros((0, 6)))

        lidar_grp = cfg.create_group('lidar_config')
        lidar_grp.attrs['height'] = mount.height
        lidar_grp.attrs['channels'] = 152
        lidar_grp.attrs['scan_rate'] = frame_rate

        frm = f.create_group('frames')
        frm.create_dataset('timestamps', data=timestamps)
        frm.create_dataset('lidar_points', data=pts_arr, compression='gzip')
        frm.create_dataset('lidar_valid_mask', data=valid_arr, compression='gzip')
        frm.create_dataset('lidar_incidence_angles', data=incidence_arr,
                            compression='gzip')
        frm.create_dataset('lidar_confidence', data=conf_arr, compression='gzip')
        frm.create_dataset('gt_wave_field', data=gt_arr, compression='gzip')
        frm.create_dataset('gt_grid_x', data=gt_grid_x)
        frm.create_dataset('gt_grid_y', data=gt_grid_y)

        rdm = f.create_group('raydrop_model')
        rdm.attrs['model_type'] = 'real_data'

        gtp = f.create_group('gt_params')
        gtp.attrs['Hs_true'] = Hs_nominal
        gtp.attrs['Tp_true'] = Tp_nominal
        gtp.create_dataset('spectrum_1d', data=np.zeros((0, 2)))

    print(f"Saved {N_frames} frames ({max_pts} max pts) → {output_path}")
