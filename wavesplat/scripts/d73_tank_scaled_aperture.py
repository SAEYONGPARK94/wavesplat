"""
d73 — DESIGN: platform-motion synthetic aperture at TOWING-TANK scale (Froude-scaled), feasibility.

The directional benefit (d72) needs CROSS-track motion. In a standard towing tank the carriage moves
ALONG the tank — which is along-track (the BAD, surfing case) if waves are head-on. The fix: OBLIQUE
(directional) waves at mean angle θ to the tank axis, so the carriage's along-tank velocity v has a
CROSS-wave component v·sin(θ) → it synthesizes the cross-range aperture WITHOUT needing a wide basin.

Directional CRB (in degrees) is FROUDE-SCALE-INVARIANT: it depends only on the dimensionless geometry,
which Froude scaling preserves. So a CRB computed at tank scale equals the full-scale value. We:
  (1) sweep carriage speed × wave-obliqueness θ at 1:25 tank scale, confirm the directional CRB still
      collapses in the tank-feasible range;
  (2) print the Froude-scaling table mapping full-scale ocean ↔ 1:25 and 1:50 tank realizations.

Run: python3 scripts/d73_tank_scaled_aperture.py
"""
import os, sys, json, math, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from d1_crb import DEVICE, DT, fisher, direction_crb_deg, cg_of_k

SIGMA_FULL = 0.02            # full-scale LiDAR elevation noise [m]; scales as length (λ)


def sea_dir_crb(carriage_speed, theta_mean_deg, k0, depth, L_foot, T_obs, sigma, N=8, M=600,
                spread_deg=20, seed=1):
    """Multidirectional sea with mean propagation angle theta_mean (deg, rel to tank/carriage +x axis);
    sensor carriage moves along +x at carriage_speed. Returns median directional CRB [deg].
    All lengths in the chosen scale's units (CRB in deg is scale-invariant)."""
    rng = np.random.default_rng(seed)
    ang = np.radians(theta_mean_deg + rng.uniform(-spread_deg, spread_deg, N))
    kmag = k0 * rng.uniform(0.7, 1.4, N)
    A = 0.2 * np.exp(-rng.uniform(0, 1.2, N))
    theta = torch.tensor(np.stack([A, kmag * np.cos(ang), kmag * np.sin(ang), rng.uniform(0, 6.28, N)], 1),
                         dtype=DT, device=DEVICE)
    x0 = rng.uniform(0, L_foot, M); y0 = rng.uniform(-L_foot / 2, L_foot / 2, M)
    t = rng.uniform(0, T_obs, M)
    # carriage moves along +x (the tank axis); obliqueness lives in the wave directions
    x = torch.tensor(x0 + carriage_speed * t, dtype=DT, device=DEVICE)
    y = torch.tensor(y0, dtype=DT, device=DEVICE)
    tt = torch.tensor(t, dtype=DT, device=DEVICE)
    F = fisher(theta, x, y, tt, depth, sigma)
    n = F.shape[0]
    Finv = torch.linalg.inv(F + 1e-10 * torch.trace(F) / n * torch.eye(n, dtype=DT, device=F.device))
    return float(np.median(direction_crb_deg(theta, Finv, N)))


def froude_table():
    """Full-scale ocean ↔ tank realizations (λ = full/model). Velocity & c_g ~ √λ, length ~ λ, time ~ √λ."""
    Tp_f, v_head_f, L_f, Tobs_f, depth_f = 7.0, 5.0, 50.0, 20.0, 50.0
    cg_f = 0.78 * Tp_f                                    # deep-water group velocity ≈ 0.78·Tp
    rows = []
    for lam in [1, 25, 50]:
        s = math.sqrt(lam)
        rows.append(dict(scale=f"1:{lam}" if lam > 1 else "full",
                         Tp=Tp_f / s, cg=cg_f / s, v_headline=v_head_f / s,
                         v_range=[2.0 / s, 8.0 / s], L_foot=L_f / lam, T_obs=Tobs_f / s,
                         cross_aperture_headline=(v_head_f / s) * (Tobs_f / s),
                         depth=depth_f / lam, sigma=SIGMA_FULL / lam))
    return rows


def main():
    out = 'results/direction1'; t0 = time.time()
    # ---- tank-scale (1:25) parameters ----
    lam = 25; s = math.sqrt(lam)
    Tp = 7.0 / s; cg = 0.78 * Tp; depth = 50.0 / lam
    L_foot = 50.0 / lam; T_obs = 20.0 / s
    # SNR (A/σ) must be Froude-PRESERVED: amplitude in sea_dir_crb is 0.2, so keep σ matched to that
    # ratio (not scaled by λ) — directional CRB in deg is then scale-invariant.
    sigma = SIGMA_FULL                                      # A/σ = 0.2/0.02 = 10, same as full scale
    k0 = (2 * math.pi) / (1.56 * (7.0 ** 2)) * lam          # full k≈0.082 → tank k = k_full·λ
    speeds = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]                 # tank carriage speeds [m/s] (v_full = v·5)
    print(f"=== 1:25 tank scale ===  Tp={Tp:.2f}s  c_g={cg:.2f}m/s  footprint={L_foot:.1f}m  "
          f"T_obs={T_obs:.1f}s  k={k0:.2f}/m  depth={depth:.1f}m")
    print(f"  carriage speeds [m/s] (×5 = full-scale): {speeds}\n")
    print("  median directional CRB [deg] vs carriage speed, by wave obliqueness θ (deg to tank axis):")
    res = {}
    for th in [0, 30, 45, 60, 90]:
        crb = [sea_dir_crb(v, th, k0, depth, L_foot, T_obs, sigma) for v in speeds]
        res[str(th)] = crb
        note = "head (surf=worst)" if th == 0 else ("beam (pure cross)" if th == 90 else "oblique")
        print(f"   θ={th:3d}° [{note:18s}] " + " ".join(f"{c:6.1f}" for c in crb))

    # headline: oblique 45° collapse
    c0 = res['45'][0]; c_top = res['45'][speeds.index(1.6)]
    h0 = res['0'][0]; h_top = res['0'][speeds.index(1.6)]
    print(f"\n  ==> oblique θ=45°: stationary {c0:.1f}° → moving@1.6m/s {c_top:.1f}° (×{c0/max(c_top,1e-9):.1f})")
    print(f"  ==> head    θ=0° : stationary {h0:.1f}° → moving@1.6m/s {h_top:.1f}° (×{h0/max(h_top,1e-9):.1f}, ≈ no gain — surfing)")
    feasible = c_top < 0.6 * c0
    print(f"  ==> tank-feasible directional gain (oblique waves + along-carriage): "
          f"{'CONFIRMED' if feasible else 'WEAK'}")

    print("\n=== Froude-scaling table (full ocean ↔ tank) ===")
    rows = froude_table()
    print(f"  {'scale':6s} {'Tp[s]':>6s} {'c_g[m/s]':>8s} {'v_head[m/s]':>11s} {'v_range[m/s]':>14s} "
          f"{'L_foot[m]':>9s} {'T_obs[s]':>8s} {'cross_ap[m]':>11s} {'depth[m]':>8s}")
    for r in rows:
        print(f"  {r['scale']:6s} {r['Tp']:6.2f} {r['cg']:8.2f} {r['v_headline']:11.2f} "
              f"[{r['v_range'][0]:.1f},{r['v_range'][1]:.1f}]".rjust(14) +
              f" {r['L_foot']:9.1f} {r['T_obs']:8.1f} {r['cross_aperture_headline']:11.1f} {r['depth']:8.1f}")

    json.dump(dict(tank_scale=lam, speeds=speeds, dir_crb_by_theta=res, froude_table=rows,
                   feasible=bool(feasible)), open(f'{out}/tank_scaled_aperture.json', 'w'), indent=2)
    print(f"\nsaved {out}/tank_scaled_aperture.json   ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
