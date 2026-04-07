"""Phase 1: End-to-end hydrofoil simulation with wavesim GT waves.

Scenarios:
  1. BF3 head sea (Hs=0.88m, Tp=5.0s, direction=180°)
  2. BF4 head sea (Hs=1.88m, Tp=7.5s, direction=180°)
  3. BF3 beam sea (Hs=0.88m, Tp=5.0s, direction=90°)
  4. BF3 quartering sea (Hs=0.88m, Tp=5.0s, direction=135°)

For each scenario, run 30s simulation and report:
  - Max/RMS heave, pitch
  - Max heave acceleration (comfort metric)
  - Encounter frequency check
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from wavesim.wave_generator import jonswap_components
from hydrofoil.config import candela_c8
from hydrofoil.sim import run_simulation


def run_scenario(name, Hs, Tp, direction_deg, duration=30.0, seed=42):
    """Run a single scenario and print results."""
    cfg = candela_c8()
    cfg['sim'].duration = duration

    components = jonswap_components(
        Hs=Hs, Tp=Tp, n_components=15,
        depth=cfg['sim'].depth,
        direction_deg=direction_deg,
        direction_spread_deg=20.0,
        seed=seed,
    )

    result = run_simulation(cfg, components, vessel_heading=0.0)

    # Skip first 5s for transient
    mask = result.time > 5.0
    z = result.z[mask]
    theta = result.theta[mask]
    z_ddot = result.heave_accel[mask]
    theta_ddot = result.pitch_accel[mask]

    print(f"\n{'='*60}")
    print(f"Scenario: {name}")
    print(f"  Hs={Hs}m, Tp={Tp}s, Dir={direction_deg}°, Speed={cfg['sim'].speed:.1f}m/s")
    print(f"{'='*60}")
    print(f"  Heave [m]:       max={np.max(np.abs(z)):.4f},  "
          f"RMS={np.std(z):.4f}")
    print(f"  Pitch [deg]:     max={np.degrees(np.max(np.abs(theta))):.3f},  "
          f"RMS={np.degrees(np.std(theta)):.3f}")
    print(f"  Heave accel [m/s²]: max={np.max(np.abs(z_ddot)):.3f},  "
          f"RMS={np.std(z_ddot):.3f}")
    print(f"  Pitch accel [°/s²]: max={np.degrees(np.max(np.abs(theta_ddot))):.3f},  "
          f"RMS={np.degrees(np.std(theta_ddot)):.3f}")

    # Encounter frequency estimate via FFT
    dt = result.time[1] - result.time[0]
    z_centered = z - np.mean(z)
    if np.std(z_centered) > 1e-6:
        fft = np.fft.rfft(z_centered)
        freqs = np.fft.rfftfreq(len(z_centered), dt)
        peak_idx = np.argmax(np.abs(fft[1:])) + 1
        f_enc = freqs[peak_idx]
        T_enc = 1.0 / f_enc if f_enc > 0 else float('inf')
        print(f"  Encounter freq:  {f_enc:.3f} Hz (T_enc={T_enc:.2f}s)")
    else:
        print(f"  Encounter freq:  N/A (no oscillation)")

    # Comfort assessment (ISO 2631 simplified)
    rms_accel = np.std(z_ddot)
    if rms_accel < 0.315:
        comfort = "Comfortable"
    elif rms_accel < 0.63:
        comfort = "Slightly uncomfortable"
    elif rms_accel < 1.0:
        comfort = "Fairly uncomfortable"
    else:
        comfort = "Uncomfortable"
    print(f"  Comfort (ISO2631): {comfort} (RMS accel={rms_accel:.3f} m/s²)")

    return result


def main():
    print("=" * 60)
    print("Phase 1: Hydrofoil 2DOF Response to WaveSim GT Waves")
    print("Vessel: Candela C-8 class (~1700kg, 22 knots)")
    print("=" * 60)

    scenarios = [
        ("BF3 Head Sea",       0.88, 5.0, 180.0),
        ("BF4 Head Sea",       1.88, 7.5, 180.0),
        ("BF3 Beam Sea",       0.88, 5.0,  90.0),
        ("BF3 Quartering Sea", 0.88, 5.0, 135.0),
    ]

    results = {}
    for name, Hs, Tp, direction_deg in scenarios:
        results[name] = run_scenario(name, Hs, Tp, direction_deg)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Scenario':<25} {'Heave RMS':>10} {'Pitch RMS':>10} {'Accel RMS':>10}")
    print(f"{'':25} {'[m]':>10} {'[deg]':>10} {'[m/s²]':>10}")
    print("-" * 60)
    for name, Hs, Tp, direction_deg in scenarios:
        r = results[name]
        mask = r.time > 5.0
        hz = np.std(r.z[mask])
        hp = np.degrees(np.std(r.theta[mask]))
        ha = np.std(r.heave_accel[mask])
        print(f"{name:<25} {hz:>10.4f} {hp:>10.3f} {ha:>10.3f}")


if __name__ == '__main__':
    main()
