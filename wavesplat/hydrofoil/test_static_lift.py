"""Static lift verification: detailed physics analysis in calm water.

Tests:
  1. Trim condition: force/moment balance at design speed
  2. Speed sweep: lift vs speed curve (hullborne → foilborne transition)
  3. Alpha sweep: CL vs alpha for each foil
  4. Submergence sweep: lift degradation near surface
  5. Calm water time-domain: stability verification (30s)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from hydrofoil.config import candela_c8, FoilConfig, SimConfig
from hydrofoil.foil import (
    foil_forces, compute_cl, compute_cd, submergence_factor,
    planform_area, aspect_ratio,
)
from hydrofoil.vehicle import (
    HydrofoilState, WaveDisturbanceInput, effective_alpha,
    foil_submergence, compute_derivatives,
)
from hydrofoil.sim import find_trim, run_simulation
from hydrofoil.controller import PDControllerConfig


def test1_trim_balance():
    """Test 1: Force and moment balance at trim."""
    print("=" * 70)
    print("TEST 1: Static Trim Force/Moment Balance")
    print("=" * 70)

    cfg = candela_c8()
    hull = cfg['hull']
    sim = cfg['sim']
    ff = cfg['front_foil']
    rf = cfg['rear_foil']

    W = hull.mass * sim.g
    q = 0.5 * sim.rho * sim.speed ** 2
    V = sim.speed

    print(f"\n--- Vessel Parameters ---")
    print(f"  Mass:        {hull.mass} kg")
    print(f"  Weight:      {W:.1f} N")
    print(f"  Speed:       {sim.speed:.1f} m/s ({sim.speed * 1.944:.1f} knots)")
    print(f"  Dynamic pressure q: {q:.0f} Pa")
    print(f"  I_yy:        {hull.I_yy:.0f} kg·m²")

    print(f"\n--- Front Foil ---")
    print(f"  Span × Chord:  {ff.span:.2f} × {ff.chord:.3f} m")
    print(f"  Area S:         {planform_area(ff):.4f} m²")
    print(f"  AR:             {aspect_ratio(ff):.1f}")
    print(f"  CL_alpha:       {ff.cl_alpha:.3f} /rad ({np.degrees(1/ff.cl_alpha):.2f} deg/unit CL)")
    print(f"  CL_max:         {ff.cl_max}")
    print(f"  alpha_zl:       {ff.alpha_zl:.5f} rad ({np.degrees(ff.alpha_zl):.3f}°)")
    print(f"  x_pos (from CG): {ff.x_pos:+.2f} m")
    print(f"  z_pos (depth):  {ff.z_pos:.2f} m")

    print(f"\n--- Rear Foil ---")
    print(f"  Span × Chord:  {rf.span:.2f} × {rf.chord:.3f} m")
    print(f"  Area S:         {planform_area(rf):.4f} m²")
    print(f"  AR:             {aspect_ratio(rf):.1f}")
    print(f"  CL_alpha:       {rf.cl_alpha:.3f} /rad ({np.degrees(1/rf.cl_alpha):.2f} deg/unit CL)")
    print(f"  alpha_zl:       {rf.alpha_zl:.5f} rad ({np.degrees(rf.alpha_zl):.3f}°)")
    print(f"  x_pos (from CG): {rf.x_pos:+.2f} m")
    print(f"  z_pos (depth):  {rf.z_pos:.2f} m")

    # At trim (alpha=0), compute lift
    S_f = planform_area(ff)
    S_r = planform_area(rf)
    CL_f = compute_cl(0.0, ff)  # alpha=0 at trim
    CL_r = compute_cl(0.0, rf)
    sf_f = submergence_factor(ff.z_pos, ff.chord)
    sf_r = submergence_factor(rf.z_pos, rf.chord)

    L_f = q * S_f * CL_f * sf_f
    L_r = q * S_r * CL_r * sf_r
    M = L_f * ff.x_pos + L_r * rf.x_pos

    print(f"\n--- Trim Lift (alpha=0, calm water) ---")
    print(f"  Front CL:   {CL_f:.4f}  (submergence factor: {sf_f:.3f})")
    print(f"  Rear  CL:   {CL_r:.4f}  (submergence factor: {sf_r:.3f})")
    print(f"  Front Lift: {L_f:.1f} N  ({L_f/W*100:.1f}% of weight)")
    print(f"  Rear  Lift: {L_r:.1f} N  ({L_r/W*100:.1f}% of weight)")
    print(f"  Total Lift: {L_f + L_r:.1f} N")
    print(f"  Weight:     {W:.1f} N")
    print(f"  Balance:    ΔF = {L_f + L_r - W:.2f} N ({(L_f+L_r-W)/W*100:.4f}%)")
    print(f"  Moment:     M = {M:.2f} N·m")

    # Lift/Weight ratio at different attitudes
    print(f"\n--- Lift Sensitivity (per degree pitch) ---")
    dtheta = np.radians(1.0)
    CL_f_1 = compute_cl(dtheta, ff)
    CL_r_1 = compute_cl(dtheta, rf)
    dL_f = q * S_f * (CL_f_1 - CL_f)
    dL_r = q * S_r * (CL_r_1 - CL_r)
    dM = dL_f * ff.x_pos + dL_r * rf.x_pos
    print(f"  ΔL_front/deg: {dL_f:.1f} N")
    print(f"  ΔL_rear/deg:  {dL_r:.1f} N")
    print(f"  ΔL_total/deg: {dL_f + dL_r:.1f} N ({(dL_f+dL_r)/W*100:.1f}% of W)")
    print(f"  ΔM/deg:       {dM:.1f} N·m  ({'DESTABILIZING' if dM > 0 else 'STABILIZING'})")


def test2_speed_sweep():
    """Test 2: Lift vs speed curve."""
    print(f"\n{'=' * 70}")
    print("TEST 2: Speed Sweep — Lift vs Speed")
    print("=" * 70)

    cfg = candela_c8()
    hull = cfg['hull']
    sim = cfg['sim']
    ff = cfg['front_foil']
    rf = cfg['rear_foil']
    W = hull.mass * sim.g

    speeds_knots = np.arange(5, 35, 1)
    speeds_ms = speeds_knots / 1.944

    print(f"\n  {'Speed':>8} {'q':>10} {'L_front':>10} {'L_rear':>10} "
          f"{'L_total':>10} {'L/W':>8} {'Foilborne':>10}")
    print(f"  {'[knots]':>8} {'[Pa]':>10} {'[N]':>10} {'[N]':>10} "
          f"{'[N]':>10} {'[-]':>8} {'':>10}")
    print("  " + "-" * 68)

    for V_kn, V_ms in zip(speeds_knots, speeds_ms):
        q = 0.5 * sim.rho * V_ms ** 2
        S_f = planform_area(ff)
        S_r = planform_area(rf)
        CL_f = compute_cl(0.0, ff)
        CL_r = compute_cl(0.0, rf)
        sf_f = submergence_factor(ff.z_pos, ff.chord)
        sf_r = submergence_factor(rf.z_pos, rf.chord)
        L_f = q * S_f * CL_f * sf_f
        L_r = q * S_r * CL_r * sf_r
        L_total = L_f + L_r
        ratio = L_total / W
        foilborne = "YES" if ratio >= 1.0 else "no"
        if abs(ratio - 1.0) < 0.05:
            foilborne = "TRANSITION"

        print(f"  {V_kn:>8.0f} {q:>10.0f} {L_f:>10.0f} {L_r:>10.0f} "
              f"{L_total:>10.0f} {ratio:>8.2f} {foilborne:>10}")


def test3_alpha_sweep():
    """Test 3: CL and CD vs alpha for each foil."""
    print(f"\n{'=' * 70}")
    print("TEST 3: CL, CD vs Alpha (each foil)")
    print("=" * 70)

    cfg = candela_c8()
    ff = cfg['front_foil']
    rf = cfg['rear_foil']
    sim = cfg['sim']
    q = 0.5 * sim.rho * sim.speed ** 2

    alphas_deg = np.arange(-5, 16, 1)
    alphas_rad = np.radians(alphas_deg)

    for name, foil in [("Front", ff), ("Rear", rf)]:
        S = planform_area(foil)
        print(f"\n  --- {name} Foil (alpha_zl = {np.degrees(foil.alpha_zl):.2f}°) ---")
        print(f"  {'alpha':>8} {'CL':>8} {'CD':>8} {'L/D':>8} {'Lift':>10} {'Drag':>10}")
        print(f"  {'[deg]':>8} {'[-]':>8} {'[-]':>8} {'[-]':>8} {'[N]':>10} {'[N]':>10}")
        print("  " + "-" * 56)

        for a_deg, a_rad in zip(alphas_deg, alphas_rad):
            cl = compute_cl(a_rad, foil)
            cd = compute_cd(cl, foil)
            L = q * S * cl
            D = q * S * cd
            ld = cl / cd if abs(cd) > 1e-8 else float('inf')
            print(f"  {a_deg:>8.0f} {cl:>8.4f} {cd:>8.5f} {ld:>8.1f} "
                  f"{L:>10.0f} {D:>10.0f}")


def test4_submergence():
    """Test 4: Submergence factor vs depth."""
    print(f"\n{'=' * 70}")
    print("TEST 4: Submergence Effect — Lift vs Depth")
    print("=" * 70)

    cfg = candela_c8()
    ff = cfg['front_foil']
    rf = cfg['rear_foil']
    sim = cfg['sim']
    q = 0.5 * sim.rho * sim.speed ** 2

    print(f"\n  {'h [m]':>8} {'h/c_f':>8} {'SF_f':>8} {'L_f [N]':>10} "
          f"{'h/c_r':>8} {'SF_r':>8} {'L_r [N]':>10} {'L_total':>10}")
    print("  " + "-" * 74)

    depths = np.arange(-0.1, 2.01, 0.1)
    for h in depths:
        sf_f = submergence_factor(h, ff.chord)
        sf_r = submergence_factor(h, rf.chord)
        CL_f = compute_cl(0.0, ff) * sf_f
        CL_r = compute_cl(0.0, rf) * sf_r
        L_f = q * planform_area(ff) * CL_f
        L_r = q * planform_area(rf) * CL_r
        print(f"  {h:>8.2f} {h/ff.chord:>8.2f} {sf_f:>8.3f} {L_f:>10.0f} "
              f"{h/rf.chord:>8.2f} {sf_r:>8.3f} {L_r:>10.0f} {L_f+L_r:>10.0f}")


def test5_calm_water_stability():
    """Test 5: Calm water 30s time-domain stability."""
    print(f"\n{'=' * 70}")
    print("TEST 5: Calm Water Stability (30s simulation)")
    print("=" * 70)

    cfg = candela_c8()
    cfg['sim'].duration = 30.0

    # Find trim
    trim_state = find_trim(cfg['front_foil'], cfg['rear_foil'],
                           cfg['hull'], cfg['sim'], PDControllerConfig())
    print(f"\n  Trim state:")
    print(f"    z     = {trim_state.z:.6f} m")
    print(f"    theta = {trim_state.theta:.6f} rad ({np.degrees(trim_state.theta):.4f}°)")
    print(f"    z_dot = {trim_state.z_dot:.8f} m/s")
    print(f"    th_dot= {trim_state.theta_dot:.8f} rad/s")

    # Run calm water
    result = run_simulation(cfg, components=[], vessel_heading=0.0)

    # Report stability metrics
    print(f"\n  30s calm water simulation:")
    print(f"    z     max/min: [{np.min(result.z):.6f}, {np.max(result.z):.6f}] m")
    print(f"    theta max/min: [{np.degrees(np.min(result.theta)):.4f}, "
          f"{np.degrees(np.max(result.theta)):.4f}]°")
    print(f"    z RMS:         {np.std(result.z):.8f} m")
    print(f"    theta RMS:     {np.degrees(np.std(result.theta)):.6f}°")
    print(f"    Flap front:    [{np.min(result.flap_front):.6f}, {np.max(result.flap_front):.6f}] rad")
    print(f"    Flap rear:     [{np.min(result.flap_rear):.6f}, {np.max(result.flap_rear):.6f}] rad")

    stable = np.max(np.abs(result.z)) < 0.001 and np.max(np.abs(result.theta)) < np.radians(0.01)
    print(f"\n  VERDICT: {'STABLE ✓' if stable else 'CHECK ✗'}")

    # Check derivatives at t=0 (should be near zero at trim)
    state = HydrofoilState(z=trim_state.z, theta=trim_state.theta)
    no_wave = WaveDisturbanceInput()
    from hydrofoil.controller import compute_flap_command
    flaps = compute_flap_command(state, PDControllerConfig())
    deriv = compute_derivatives(state, cfg['front_foil'], cfg['rear_foil'],
                                cfg['hull'], cfg['sim'], no_wave,
                                flap_front=flaps.front, flap_rear=flaps.rear)
    print(f"\n  Derivatives at trim:")
    print(f"    z_dot      = {deriv[0]:.8f} m/s")
    print(f"    theta_dot  = {deriv[1]:.8f} rad/s")
    print(f"    z_ddot     = {deriv[2]:.8f} m/s²")
    print(f"    theta_ddot = {deriv[3]:.8f} rad/s²")


if __name__ == '__main__':
    test1_trim_balance()
    test2_speed_sweep()
    test3_alpha_sweep()
    test4_submergence()
    test5_calm_water_stability()
