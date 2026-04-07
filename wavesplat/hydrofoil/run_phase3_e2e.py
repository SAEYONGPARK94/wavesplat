"""Phase 3: End-to-end WaveSplat → hydrofoil feedforward demo.

Pipeline:
  1. Load HDF5 episode (GT Gerstner wave field + LiDAR observations)
  2. Run WaveSplat estimation (init + tracking)
  3. At each tracking frame, extract estimated wave components
  4. Run hydrofoil simulation with three control modes:
     a. PD-only (no wave knowledge)
     b. PD + GT feedforward (oracle — uses true wave components)
     c. PD + WaveSplat feedforward (uses estimated components)
  5. Compare: heave, pitch, acceleration, comfort

This is the definitive "WaveSplat helps real control" demonstration.
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import torch
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.run import DEFAULT_CONFIG

from hydrofoil.config import candela_c8
from hydrofoil.controller import PDControllerConfig, FeedforwardConfig
from hydrofoil.sim import run_simulation
from hydrofoil.wavesplat_bridge import wavesplat_to_components, h5_gt_to_components


def _flip_to_head_sea(components):
    """Flip wave propagation direction to create head sea conditions.

    Episode waves propagate in +x (kx > 0). Vessel heading=0 moves in +x.
    → following sea (feedforward not effective).

    Negating kx makes waves propagate in -x → head sea for vessel_heading=0.
    This is physically equivalent to the vessel heading into the waves.
    """
    for c in components:
        c.kx = -c.kx
    return components


def run_e2e_comparison(h5_path: str, duration: float = 20.0,
                       verbose: bool = True) -> dict:
    """Run end-to-end WaveSplat → hydrofoil comparison.

    Args:
        h5_path: path to HDF5 episode
        duration: simulation duration [s] (within episode time range)
        verbose: print progress

    Returns:
        dict with results for each control strategy
    """
    # --- Step 1: Load episode ---
    loader = EpisodeDataLoader(h5_path, device='cpu')
    Hs = loader.Hs_true
    Tp = loader.Tp_true

    if verbose:
        print(f"  Episode: {os.path.basename(h5_path)}")
        print(f"  Sea state: {loader.sea_state}, Hs={Hs:.2f}m, Tp={Tp:.1f}s")
        print(f"  Frames: {loader.n_frames}, dt={loader.timestamps[1]-loader.timestamps[0]:.2f}s")

    # --- Step 2: Run WaveSplat estimation ---
    if verbose:
        print(f"\n  [WaveSplat] Initializing...")

    ws_config = DEFAULT_CONFIG.copy()
    ws_config['device'] = 'cpu'

    updater = SequentialUpdater(ws_config)
    init_frames = loader.get_init_window(tp_factor=2.0)
    init_result = updater.initialize(
        init_frames,
        x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
        y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
        verbose=False,
    )

    if verbose:
        print(f"  [WaveSplat] Init done: {init_result['n_iters']} iters, "
              f"{init_result['elapsed_s']:.1f}s")

    # Track through all frames to get best estimation
    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
    n_tracked = 0
    for frame in tracking_frames:
        updater.update(frame)
        n_tracked += 1

    if verbose:
        print(f"  [WaveSplat] Tracked {n_tracked} frames")

    # Extract estimated components
    ws_components = wavesplat_to_components(updater.model)
    if verbose:
        print(f"  [WaveSplat] Estimated {len(ws_components)} components")
        for i, c in enumerate(ws_components):
            print(f"    comp {i}: A={c.A:.4f}m, ω={c.omega:.3f}rad/s, "
                  f"k=({c.kx:.3f},{c.ky:.3f})")

    # --- Step 3: Get GT wave components ---
    import h5py
    with h5py.File(h5_path, 'r') as f:
        gt_wc_array = f['config/wave_components'][()]
    gt_components = h5_gt_to_components(gt_wc_array)

    # Flip to head sea: episode waves go +x, vessel goes +x → following sea
    # Negate kx to make waves come FROM ahead → head sea (where FF is effective)
    gt_components = _flip_to_head_sea(gt_components)
    ws_components = _flip_to_head_sea(ws_components)

    if verbose:
        print(f"\n  [GT] {len(gt_components)} true components (flipped to head sea)")
        for i, c in enumerate(gt_components):
            print(f"    comp {i}: A={c.A:.4f}m, ω={c.omega:.3f}rad/s, "
                  f"k=({c.kx:.3f},{c.ky:.3f})")

    # --- Step 4: Run hydrofoil simulations ---
    cfg = candela_c8()
    cfg['sim'].duration = duration
    ctrl = PDControllerConfig()
    ff_cfg = FeedforwardConfig(
        preview_dt=0.5, K_ff=0.8, lp_alpha=0.05,
        noise_std=0.0, slope_noise_std=0.0,
    )

    results = {}

    # 4a. PD-only (GT disturbance, no feedforward)
    if verbose:
        print(f"\n  [Sim] Running PD-only...", end=" ", flush=True)
    np.random.seed(42)
    results['PD-only'] = run_simulation(
        cfg, gt_components, vessel_heading=0.0,
        controller=ctrl, feedforward=None,
    )
    if verbose:
        print("done")

    # 4b. PD + GT FF (GT components for both disturbance AND feedforward)
    if verbose:
        print(f"  [Sim] Running PD + GT FF...", end=" ", flush=True)
    np.random.seed(42)
    results['PD+GT_FF'] = run_simulation(
        cfg, gt_components, vessel_heading=0.0,
        controller=ctrl, feedforward=ff_cfg,
    )
    if verbose:
        print("done")

    # 4c. PD + WaveSplat FF
    # Disturbance from GT (reality), feedforward from WaveSplat estimate
    # This requires a custom simulation where:
    #   - Dynamics use GT wave disturbance (what actually hits the foils)
    #   - Feedforward uses WaveSplat estimate (what the controller thinks)
    if verbose:
        print(f"  [Sim] Running PD + WaveSplat FF...", end=" ", flush=True)
    np.random.seed(42)
    results['PD+WS_FF'] = run_simulation_split(
        cfg, gt_components, ws_components,
        vessel_heading=0.0, controller=ctrl, ff_cfg=ff_cfg,
    )
    if verbose:
        print("done")

    return {
        'results': results,
        'Hs': Hs, 'Tp': Tp,
        'n_gt_components': len(gt_components),
        'n_ws_components': len(ws_components),
    }


def run_simulation_split(cfg, gt_components, ff_components,
                         vessel_heading=0.0, controller=None, ff_cfg=None):
    """Run simulation with split wave sources.

    Dynamics (disturbance) uses gt_components — the real wave field.
    Feedforward uses ff_components — the estimated wave field.

    This models the realistic scenario: the ocean is what it is (GT),
    but the controller only knows what WaveSplat estimated.
    """
    from hydrofoil.controller import FeedforwardFilter, FlapCommand
    from hydrofoil.vehicle import HydrofoilState, compute_derivatives
    from hydrofoil.disturbance import compute_disturbance
    from hydrofoil.sim import (
        SimResult, find_trim, _compute_feedforward_info, _get_flaps,
    )

    front_foil = cfg['front_foil']
    rear_foil = cfg['rear_foil']
    hull = cfg['hull']
    sim = cfg['sim']

    ctrl = controller
    if ctrl is None:
        ctrl = PDControllerConfig()

    ff_filter = FeedforwardFilter() if ff_cfg and ff_cfg.K_ff > 0 else None

    # Find trim in calm water
    state = find_trim(front_foil, rear_foil, hull, sim, ctrl)

    n_steps = int(sim.duration / sim.dt) + 1
    time_arr = np.linspace(0.0, sim.duration, n_steps)

    result = SimResult(
        time=time_arr,
        z=np.zeros(n_steps), theta=np.zeros(n_steps),
        z_dot=np.zeros(n_steps), theta_dot=np.zeros(n_steps),
        x_pos=np.zeros(n_steps),
        eta_front=np.zeros(n_steps), eta_rear=np.zeros(n_steps),
        flap_front=np.zeros(n_steps), flap_rear=np.zeros(n_steps),
    )

    for i, t in enumerate(time_arr):
        result.z[i] = state.z
        result.theta[i] = state.theta
        result.z_dot[i] = state.z_dot
        result.theta_dot[i] = state.theta_dot
        result.x_pos[i] = state.x

        # Disturbance from GT (reality)
        wave = compute_disturbance(state, front_foil, rear_foil, sim,
                                   gt_components, t, vessel_heading)
        result.eta_front[i] = wave.eta_front
        result.eta_rear[i] = wave.eta_rear

        # Feedforward from WaveSplat estimate (controller's knowledge)
        ff_delta_front = 0.0
        ff_delta_rear = 0.0
        if ff_cfg and ff_cfg.K_ff > 0:
            ff_info = _compute_feedforward_info(
                state, front_foil, rear_foil, sim, ff_components,
                t, vessel_heading, ff_cfg)
            K = ff_cfg.K_ff
            raw_front = -K * (-ff_info.slope_front - ff_info.w_front / sim.speed)
            raw_rear = -K * (-ff_info.slope_rear - ff_info.w_rear / sim.speed)
            if ff_filter is not None:
                ff_delta_front, ff_delta_rear = ff_filter.filter(
                    raw_front, raw_rear, ff_cfg.lp_alpha)
            else:
                ff_delta_front, ff_delta_rear = raw_front, raw_rear

        # Record flaps
        if ctrl:
            flaps = _get_flaps(state, ctrl, speed=sim.speed)
            result.flap_front[i] = np.clip(
                flaps.front + ff_delta_front,
                -ctrl.delta_max, ctrl.delta_max)
            result.flap_rear[i] = np.clip(
                flaps.rear + ff_delta_rear,
                -ctrl.delta_max, ctrl.delta_max)

        # RK4 integration (GT disturbance, PD + pre-filtered WS FF)
        if i < n_steps - 1:
            dt = time_arr[i + 1] - time_arr[i]
            y = state.as_array()
            x_pos = state.x

            def deriv(y_local, t_local, x_local):
                s = HydrofoilState.from_array(y_local, x=x_local)
                w = compute_disturbance(s, front_foil, rear_foil, sim,
                                        gt_components, t_local, vessel_heading)
                f = _get_flaps(s, ctrl, speed=sim.speed)
                return compute_derivatives(
                    s, front_foil, rear_foil, hull, sim, w,
                    flap_front=f.front + ff_delta_front,
                    flap_rear=f.rear + ff_delta_rear)

            dx = sim.speed * dt
            k1 = deriv(y, t, x_pos)
            k2 = deriv(y + 0.5*dt*k1, t + 0.5*dt, x_pos + 0.5*dx)
            k3 = deriv(y + 0.5*dt*k2, t + 0.5*dt, x_pos + 0.5*dx)
            k4 = deriv(y + dt*k3, t + dt, x_pos + dx)
            y_new = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            state = HydrofoilState.from_array(y_new, x=x_pos + dx)

    result.heave_accel = np.gradient(result.z_dot, result.time)
    result.pitch_accel = np.gradient(result.theta_dot, result.time)
    return result


def print_comparison(data: dict, scenario_name: str):
    """Print comparison table for one episode."""
    results = data['results']

    print(f"\n{'=' * 85}")
    print(f"  {scenario_name}")
    print(f"  Hs={data['Hs']:.2f}m, Tp={data['Tp']:.1f}s | "
          f"GT: {data['n_gt_components']} comp, WaveSplat: {data['n_ws_components']} comp")
    print(f"{'=' * 85}")
    print(f"  {'Strategy':<16} {'z_RMS':>8} {'θ_RMS':>8} {'a_RMS':>8} "
          f"{'z_max':>8} {'Comfort':>13} {'Δz':>8} {'Δa':>8}")
    print(f"  {'':16} {'[cm]':>8} {'[deg]':>8} {'[m/s²]':>8} "
          f"{'[cm]':>8} {'':>13} {'':>8} {'':>8}")
    print("  " + "-" * 81)

    baseline_z = baseline_a = None
    for name in ['PD-only', 'PD+GT_FF', 'PD+WS_FF']:
        r = results[name]
        mask = r.time > 5.0
        z = r.z[mask]
        theta = r.theta[mask]
        z_ddot = r.heave_accel[mask]

        z_rms = np.std(z) * 100
        th_rms = np.degrees(np.std(theta))
        a_rms = np.std(z_ddot)
        z_max = np.max(np.abs(z)) * 100

        # Comfort
        if a_rms < 0.315:
            comfort = "Comfortable"
        elif a_rms < 0.63:
            comfort = "Slight"
        elif a_rms < 1.0:
            comfort = "Fairly"
        elif a_rms < 1.6:
            comfort = "Uncomfortable"
        else:
            comfort = "Very Uncomf."

        if baseline_z is None:
            baseline_z, baseline_a = z_rms, a_rms

        dz = (1 - z_rms / baseline_z) * 100 if baseline_z > 0 else 0
        da = (1 - a_rms / baseline_a) * 100 if baseline_a > 0 else 0
        imp = "" if name == 'PD-only' else f"{dz:>+7.1f}% {da:>+7.1f}%"

        print(f"  {name:<16} {z_rms:>8.2f} {th_rms:>8.3f} {a_rms:>8.3f} "
              f"{z_max:>8.2f} {comfort:>13} {imp}")


def main():
    print("=" * 85)
    print("  Phase 3: End-to-End WaveSplat → Hydrofoil Feedforward Demo")
    print("  Pipeline: LiDAR → WaveSplat estimation → wave preview → feedforward control")
    print("=" * 85)

    # Select episodes for demo
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    episodes = [
        ("BF3 5-comp (moderate)", "episode_BF3_5comp_seed0_uniform.h5"),
        ("BF4 5-comp (rough)", "episode_BF4_5comp_seed0_uniform.h5"),
        ("BF3 2-comp (simple)", "episode_BF3_2comp_seed0_uniform.h5"),
    ]

    all_data = {}
    for name, filename in episodes:
        h5_path = os.path.join(data_dir, filename)
        if not os.path.exists(h5_path):
            print(f"\n  SKIP: {filename} not found")
            continue

        print(f"\n{'─' * 85}")
        print(f"  Episode: {name}")
        print(f"{'─' * 85}")

        t0 = time.time()
        data = run_e2e_comparison(h5_path, duration=20.0, verbose=True)
        elapsed = time.time() - t0

        all_data[name] = data
        print(f"\n  Total time: {elapsed:.1f}s")

    # Print comparison tables
    for name, data in all_data.items():
        print_comparison(data, name)

    # Summary
    print(f"\n{'=' * 85}")
    print("  SUMMARY: WaveSplat E2E Feedforward Performance")
    print(f"{'=' * 85}")
    print(f"  {'Episode':<30} {'GT FF Δa':>10} {'WS FF Δa':>10} {'WS/GT':>8}")
    print("  " + "-" * 58)

    for name, data in all_data.items():
        results = data['results']
        mask = results['PD-only'].time > 5.0
        a_base = np.std(results['PD-only'].heave_accel[mask])
        a_gt = np.std(results['PD+GT_FF'].heave_accel[mask])
        a_ws = np.std(results['PD+WS_FF'].heave_accel[mask])

        da_gt = (1 - a_gt / a_base) * 100
        da_ws = (1 - a_ws / a_base) * 100
        ratio = da_ws / da_gt * 100 if abs(da_gt) > 0.01 else float('nan')

        print(f"  {name:<30} {da_gt:>+9.1f}% {da_ws:>+9.1f}% {ratio:>7.0f}%")

    print(f"\n  WS/GT = what fraction of GT's improvement WaveSplat achieves")
    print(f"  100% = WaveSplat matches perfect wave knowledge")


if __name__ == '__main__':
    main()
