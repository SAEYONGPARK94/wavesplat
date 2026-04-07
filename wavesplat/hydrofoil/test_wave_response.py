"""Wave response analysis: detailed time-domain results with physics metrics.

Runs BF3 head sea (Hs=0.88m, Tp=5.0s) and reports:
  - Time history of heave, pitch, wave at foils
  - Controller effort (flap deflections)
  - Stall margin monitoring
  - RAO (Response Amplitude Operator) estimate
  - Encounter frequency verification
  - Energy balance
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from wavesim.wave_generator import jonswap_components, full_dispersion
from hydrofoil.config import candela_c8
from hydrofoil.foil import compute_cl, submergence_factor, planform_area
from hydrofoil.vehicle import (
    HydrofoilState, WaveDisturbanceInput, effective_alpha, foil_submergence,
)
from hydrofoil.disturbance import compute_disturbance
from hydrofoil.sim import run_simulation
from hydrofoil.controller import PDControllerConfig, compute_flap_command


def wave_response_analysis():
    """Detailed wave response for BF3 head sea."""
    print("=" * 70)
    print("WAVE RESPONSE ANALYSIS: BF3 Head Sea")
    print("=" * 70)

    cfg = candela_c8()
    sim = cfg['sim']
    ff = cfg['front_foil']
    rf = cfg['rear_foil']
    hull = cfg['hull']
    cfg['sim'].duration = 30.0

    Hs, Tp = 0.88, 5.0
    components = jonswap_components(
        Hs=Hs, Tp=Tp, n_components=15,
        depth=sim.depth, direction_deg=180.0,
        direction_spread_deg=20.0, seed=42,
    )

    print(f"\n--- Wave Conditions ---")
    print(f"  Hs = {Hs} m, Tp = {Tp} s")
    print(f"  Direction: 180° (head sea)")
    print(f"  Components: {len(components)}")
    print(f"  {'#':>4} {'A[m]':>8} {'T[s]':>8} {'λ[m]':>8} {'Dir[°]':>8} {'ε=kA':>8}")
    for i, c in enumerate(components):
        T = 2 * np.pi / c.omega
        k_mag = np.sqrt(c.kx**2 + c.ky**2)
        lam = 2 * np.pi / k_mag if k_mag > 0 else float('inf')
        eps = k_mag * c.A
        d = np.degrees(np.arctan2(c.ky, c.kx))
        print(f"  {i:>4} {c.A:>8.4f} {T:>8.2f} {lam:>8.1f} {d:>8.1f} {eps:>8.4f}")

    # Run simulation
    result = run_simulation(cfg, components, vessel_heading=0.0)

    # Skip transient (first 5s)
    mask = result.time > 5.0
    t = result.time[mask]
    z = result.z[mask]
    theta = result.theta[mask]
    z_dot = result.z_dot[mask]
    theta_dot = result.theta_dot[mask]
    z_ddot = result.heave_accel[mask]
    theta_ddot = result.pitch_accel[mask]
    flap_f = result.flap_front[mask]
    flap_r = result.flap_rear[mask]
    eta_f = result.eta_front[mask]
    eta_r = result.eta_rear[mask]

    # --- Response statistics ---
    print(f"\n--- Response Statistics (t > 5s) ---")
    print(f"  {'Quantity':<25} {'Max':>10} {'Min':>10} {'RMS':>10} {'Mean':>10}")
    print("  " + "-" * 65)
    print(f"  {'Heave z [m]':<25} {np.max(z):>10.4f} {np.min(z):>10.4f} "
          f"{np.std(z):>10.4f} {np.mean(z):>10.4f}")
    print(f"  {'Pitch θ [deg]':<25} {np.degrees(np.max(theta)):>10.3f} "
          f"{np.degrees(np.min(theta)):>10.3f} "
          f"{np.degrees(np.std(theta)):>10.3f} {np.degrees(np.mean(theta)):>10.3f}")
    print(f"  {'Heave vel [m/s]':<25} {np.max(z_dot):>10.4f} {np.min(z_dot):>10.4f} "
          f"{np.std(z_dot):>10.4f} {np.mean(z_dot):>10.4f}")
    print(f"  {'Heave accel [m/s²]':<25} {np.max(z_ddot):>10.3f} {np.min(z_ddot):>10.3f} "
          f"{np.std(z_ddot):>10.3f} {np.mean(z_ddot):>10.3f}")
    print(f"  {'Pitch accel [°/s²]':<25} {np.degrees(np.max(theta_ddot)):>10.2f} "
          f"{np.degrees(np.min(theta_ddot)):>10.2f} "
          f"{np.degrees(np.std(theta_ddot)):>10.2f} "
          f"{np.degrees(np.mean(theta_ddot)):>10.2f}")
    print(f"  {'Wave η_front [m]':<25} {np.max(eta_f):>10.4f} {np.min(eta_f):>10.4f} "
          f"{np.std(eta_f):>10.4f} {np.mean(eta_f):>10.4f}")
    print(f"  {'Wave η_rear [m]':<25} {np.max(eta_r):>10.4f} {np.min(eta_r):>10.4f} "
          f"{np.std(eta_r):>10.4f} {np.mean(eta_r):>10.4f}")
    print(f"  {'Flap front [deg]':<25} {np.degrees(np.max(flap_f)):>10.3f} "
          f"{np.degrees(np.min(flap_f)):>10.3f} "
          f"{np.degrees(np.std(flap_f)):>10.3f} {np.degrees(np.mean(flap_f)):>10.3f}")
    print(f"  {'Flap rear [deg]':<25} {np.degrees(np.max(flap_r)):>10.3f} "
          f"{np.degrees(np.min(flap_r)):>10.3f} "
          f"{np.degrees(np.std(flap_r)):>10.3f} {np.degrees(np.mean(flap_r)):>10.3f}")

    # --- RAO estimate ---
    print(f"\n--- RAO (Response Amplitude Operator) ---")
    z_amp = np.std(z) * np.sqrt(2)  # assuming sinusoidal
    eta_amp = Hs / 2  # significant amplitude
    rao_heave = z_amp / eta_amp
    theta_amp = np.std(theta) * np.sqrt(2)
    k_peak = (2 * np.pi / Tp) ** 2 / 9.81  # approx peak wavenumber
    rao_pitch = theta_amp / (eta_amp * k_peak)  # normalized by wave slope

    print(f"  Heave RAO = z_rms√2 / (Hs/2) = {z_amp:.4f} / {eta_amp:.3f} = {rao_heave:.3f}")
    print(f"  Pitch RAO = θ_rms√2 / (kA)    = {rao_pitch:.3f}")
    print(f"  (RAO < 1 means wave-attenuating, typical for controlled hydrofoil)")

    # --- Encounter frequency ---
    print(f"\n--- Encounter Frequency ---")
    dt = t[1] - t[0]
    z_centered = z - np.mean(z)
    fft_z = np.fft.rfft(z_centered)
    freqs = np.fft.rfftfreq(len(z_centered), dt)
    power = np.abs(fft_z) ** 2
    # Peak frequency
    peak_idx = np.argmax(power[1:]) + 1
    f_enc = freqs[peak_idx]
    T_enc = 1.0 / f_enc if f_enc > 0 else float('inf')

    # Theoretical encounter freq for dominant component
    w_peak = 2 * np.pi / Tp
    V = sim.speed
    # Head sea: ω_e = ω + k*V (wave propagates in -x, vessel in +x)
    f_enc_theory = (w_peak + k_peak * V) / (2 * np.pi)

    print(f"  Measured peak:     f_enc = {f_enc:.3f} Hz (T_enc = {T_enc:.2f} s)")
    print(f"  Theory (dominant): f_enc = {f_enc_theory:.3f} Hz (T_enc = {1/f_enc_theory:.2f} s)")
    print(f"  Match: {'GOOD' if abs(f_enc - f_enc_theory) / f_enc_theory < 0.2 else 'CHECK'}")

    # --- Stall margin monitoring ---
    print(f"\n--- Stall Margin Check ---")
    ctrl = PDControllerConfig()
    q = 0.5 * sim.rho * V ** 2

    # Recompute alpha at worst case
    worst_z_idx = np.argmax(np.abs(z))
    worst_theta_idx = np.argmax(np.abs(theta))

    for label, idx in [("Max |z|", worst_z_idx), ("Max |θ|", worst_theta_idx)]:
        i_full = np.where(mask)[0][idx]
        state = HydrofoilState(
            z=result.z[i_full], theta=result.theta[i_full],
            z_dot=result.z_dot[i_full], theta_dot=result.theta_dot[i_full],
            x=result.x_pos[i_full],
        )
        wave = compute_disturbance(state, ff, rf, sim, components,
                                   result.time[i_full], vessel_heading=0.0)
        flaps = compute_flap_command(state, ctrl)

        for foil_name, foil, is_front in [("Front", ff, True), ("Rear", rf, False)]:
            alpha = effective_alpha(state, foil, wave, sim, is_front=is_front)
            alpha += flaps.front if is_front else flaps.rear
            cl = compute_cl(alpha, foil)
            margin = foil.cl_max - abs(cl)
            h = foil_submergence(state, foil,
                                 wave.eta_front if is_front else wave.eta_rear)
            sf = submergence_factor(h, foil.chord)

            print(f"  [{label}] {foil_name}: α={np.degrees(alpha):+.2f}°, "
                  f"CL={cl:.3f}, margin={margin:.3f}, "
                  f"h={h:.3f}m (h/c={h/foil.chord:.1f}), SF={sf:.3f}")

    # --- Time history sample (5 seconds) ---
    print(f"\n--- Time History Sample (t=10-15s) ---")
    sample = (t >= 10.0) & (t <= 15.0)
    t_s = t[sample]
    # Subsample every 0.1s for readability
    step = int(0.1 / dt)
    print(f"  {'t[s]':>6} {'z[cm]':>8} {'θ[deg]':>8} {'η_f[cm]':>8} "
          f"{'η_r[cm]':>8} {'δ_f[°]':>8} {'δ_r[°]':>8}")
    print("  " + "-" * 56)
    for j in range(0, np.sum(sample), step):
        if j >= len(t_s):
            break
        i = np.where(sample)[0][j]
        print(f"  {t[i]:>6.1f} {z[i]*100:>8.2f} {np.degrees(theta[i]):>8.3f} "
              f"{eta_f[i]*100:>8.2f} {eta_r[i]*100:>8.2f} "
              f"{np.degrees(flap_f[i]):>8.2f} {np.degrees(flap_r[i]):>8.2f}")

    # --- Comfort (ISO 2631) ---
    print(f"\n--- Comfort Assessment (ISO 2631-1) ---")
    rms_a = np.std(z_ddot)
    print(f"  RMS vertical acceleration: {rms_a:.3f} m/s²")
    if rms_a < 0.315:
        print(f"  Rating: COMFORTABLE (< 0.315)")
    elif rms_a < 0.63:
        print(f"  Rating: SLIGHTLY UNCOMFORTABLE (0.315-0.63)")
    elif rms_a < 1.0:
        print(f"  Rating: FAIRLY UNCOMFORTABLE (0.63-1.0)")
    elif rms_a < 1.6:
        print(f"  Rating: UNCOMFORTABLE (1.0-1.6)")
    else:
        print(f"  Rating: VERY UNCOMFORTABLE (> 1.6)")
    print(f"  → WaveSplat feedforward 추가 시 개선 기대 (Phase 2)")


if __name__ == '__main__':
    wave_response_analysis()
