"""
config.py
YAML-based configuration for WaveSim pipeline.

Usage:
    from wavesim.config import load_config
    cfg = load_config('configs/server.yaml')
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class LidarCfg:
    height: float = 5.0
    channels: int = 152
    scan_rate: float = 10.0
    fov_h_deg: float = 120.0
    fov_v_min_deg: float = -25.0
    fov_v_max_deg: float = 0.0
    range_min: float = 1.5
    range_max: float = 250.0
    n_h_beams: int = 500
    wavelength_nm: int = 1550


@dataclass
class RaycastCfg:
    batch_size: int = 4000
    spatial_grid_nx: int = 20
    spatial_grid_ny: int = 15


@dataclass
class MeshCfg:
    nx: int = 40
    ny: int = 30


@dataclass
class GtGridCfg:
    x_range: Tuple[float, float] = (10.0, 250.0)
    y_range: Tuple[float, float] = (-100.0, 100.0)
    nx: int = 80
    ny: int = 60


@dataclass
class HOSCfg:
    enabled: bool = False       # use HOS instead of linear Gerstner
    M: int = 2                  # nonlinear order (1=linear, 2=second, 3=third)
    Nx: int = 128               # spectral modes x
    Ny: int = 128               # spectral modes y
    dt: float = 0.02            # HOS internal time step [s] (must be smaller than episode dt)
    Lx: float = 600.0           # domain length x [m]
    Ly: float = 400.0           # domain length y [m]


@dataclass
class EpisodeCfg:
    duration: float = 30.0
    dt: float = 0.1
    depth: float = 50.0
    n_pts: int = 50
    n_freq: int = 64


@dataclass
class ScenarioCfg:
    name: str = ''
    sea_state: str = 'BF4'
    n_components: int = 30
    ship_speed_kts: float = 0.0
    # Directional spreading [deg, std dev of the per-component direction draw].
    # 30.0 reproduces every episode generated before 2026-08-05, when this
    # knob existed in wave_generator but was not reachable from any config
    # (see tmp/blank_slate_2026-08-04/). Exposing it makes spread an
    # independent factor from n_components, which the deconfounding analysis
    # showed is the only variable explaining the WS-vs-LETKF gap once
    # component count is controlled (partial rho +0.460, p=0.0028).
    direction_spread_deg: float = 30.0
    # Directions placed on each frequency bin.  1 = every episode generated
    # before 2026-08-05, where the within-frequency directional spread is
    # identically zero (an algebraic identity of one-direction-per-bin, not a
    # tuning artefact) while measured CDIP records put it above zero in
    # 6,263/6,263 cases, median 29.5 deg.  M>1 is an additive research arm,
    # never a replacement (tmp/s4_within_spread_2026-08-05/PREREG.md).
    dirs_per_freq: int = 1
    # --- independence knobs (2026-08-07) ------------------------------------
    # Measured defect they address (tmp/audit_independence_2026-08-07/):
    # with both False, amplitudes and omegas carry NO rng draw, so every
    # episode of one configuration shares one spectrum however many seeds are
    # run, and the phase/direction stream ignores sea_state and n_components,
    # so BF4/BF5 at one seed get identical draws and 15comp is an exact prefix
    # of 30comp.  The 16 "crowded" episodes of data_finemesh_2026-07-23 hold
    # 4 distinct amplitude vectors and 8 distinct phase vectors, which caps
    # the p any cluster-robust test on that set can reach.
    # Both default False = every stored episode reproduces bit-for-bit.
    random_amplitudes: bool = False
    decorrelate_stream: bool = False
    # omega sits at the bin centre by default -- a pure function of (Tp,
    # n_components) with no rng draw -- and |k| follows from omega, so the
    # radial wavenumber structure repeats across every seed of a config.
    random_frequencies: bool = False


@dataclass
class DatasetCfg:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    raydrop_models: List[str] = field(default_factory=lambda: ['uniform', 'slope'])
    output_dir: str = 'data'
    num_workers: int = 1
    scenarios: List[ScenarioCfg] = field(default_factory=list)


@dataclass
class WaveSimConfig:
    lidar: LidarCfg = field(default_factory=LidarCfg)
    raycast: RaycastCfg = field(default_factory=RaycastCfg)
    mesh: MeshCfg = field(default_factory=MeshCfg)
    gt_grid: GtGridCfg = field(default_factory=GtGridCfg)
    episode: EpisodeCfg = field(default_factory=EpisodeCfg)
    hos: HOSCfg = field(default_factory=HOSCfg)
    sea_states: Dict[str, dict] = field(default_factory=lambda: {
        'BF2': {'Hs': 0.3, 'Tp': 4.0},
        'BF3': {'Hs': 0.6, 'Tp': 5.5},
        'BF4': {'Hs': 1.2, 'Tp': 7.0},
        'BF5': {'Hs': 2.0, 'Tp': 9.0},
    })
    dataset: DatasetCfg = field(default_factory=DatasetCfg)


def load_config(path: str = None) -> WaveSimConfig:
    """Load config from YAML file. Falls back to defaults if no path given."""
    cfg = WaveSimConfig()
    if path is None:
        return cfg

    with open(path) as f:
        raw = yaml.safe_load(f)

    if 'lidar' in raw:
        cfg.lidar = LidarCfg(**raw['lidar'])
    if 'raycast' in raw:
        cfg.raycast = RaycastCfg(**raw['raycast'])
    if 'mesh' in raw:
        cfg.mesh = MeshCfg(**raw['mesh'])
    if 'gt_grid' in raw:
        gt = raw['gt_grid']
        cfg.gt_grid = GtGridCfg(
            x_range=tuple(gt['x_range']),
            y_range=tuple(gt['y_range']),
            nx=gt['nx'], ny=gt['ny']
        )
    if 'episode' in raw:
        cfg.episode = EpisodeCfg(**raw['episode'])
    if 'hos' in raw:
        cfg.hos = HOSCfg(**raw['hos'])
    if 'sea_states' in raw:
        cfg.sea_states = raw['sea_states']
    if 'dataset' in raw:
        ds = raw['dataset']
        scenarios = [ScenarioCfg(**s) for s in ds.get('scenarios', [])]
        cfg.dataset = DatasetCfg(
            seeds=ds.get('seeds', [0, 1, 2, 3, 4]),
            raydrop_models=ds.get('raydrop_models', ['uniform', 'slope']),
            output_dir=ds.get('output_dir', 'data'),
            num_workers=ds.get('num_workers', 1),
            scenarios=scenarios,
        )
    return cfg


def config_summary(cfg: WaveSimConfig) -> str:
    """Print config summary."""
    n_rays = cfg.lidar.channels * cfg.lidar.n_h_beams
    n_frames = int(cfg.episode.duration / cfg.episode.dt) + 1
    n_episodes = (len(cfg.dataset.scenarios) * len(cfg.dataset.seeds)
                  * len(cfg.dataset.raydrop_models))
    # With spatial pre-filtering, each ray tests ~3×3 cells worth of triangles
    n_tri_total = cfg.mesh.nx * cfg.mesh.ny * 2
    tri_per_cell = max(1, n_tri_total // (cfg.raycast.spatial_grid_nx
                                          * cfg.raycast.spatial_grid_ny))
    local_tri = tri_per_cell * 9  # 3×3 neighbor cells
    mem_per_batch_mb = cfg.raycast.batch_size * local_tri * 3 * 8 * 10 / 1e6
    return (
        f"=== WaveSim Config ===\n"
        f"LiDAR: {cfg.lidar.channels}ch × {cfg.lidar.n_h_beams}h = {n_rays:,} rays/frame\n"
        f"  FOV: {cfg.lidar.fov_h_deg}°×"
        f"[{cfg.lidar.fov_v_min_deg}°,{cfg.lidar.fov_v_max_deg}°], "
        f"range [{cfg.lidar.range_min}–{cfg.lidar.range_max}m]\n"
        f"Mesh: {cfg.mesh.nx}×{cfg.mesh.ny} ({n_tri_total} triangles)\n"
        f"GT grid: {cfg.gt_grid.nx}×{cfg.gt_grid.ny}\n"
        f"Raycast: batch={cfg.raycast.batch_size}, "
        f"spatial grid={cfg.raycast.spatial_grid_nx}×{cfg.raycast.spatial_grid_ny}\n"
        f"  Local triangles/ray: ~{local_tri} (spatial filtered), "
        f"~{mem_per_batch_mb:.0f} MB/batch\n"
        f"Episode: {cfg.episode.duration}s @ {1/cfg.episode.dt:.0f}Hz = {n_frames} frames\n"
        f"Dataset: {len(cfg.dataset.scenarios)} scenarios × "
        f"{len(cfg.dataset.seeds)} seeds × "
        f"{len(cfg.dataset.raydrop_models)} models = {n_episodes} episodes\n"
        f"  Workers: {cfg.dataset.num_workers}\n"
    )
