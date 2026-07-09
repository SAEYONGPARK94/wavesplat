"""E1 — fix the broken oracle floor (d3_r5_oracle O1) into a TRUE linear-representation floor.

Two confirmed flaws in the old O1 (r5_oracle.json, O1_32=0.249):
  (i)  factor-2 amplitude bug: init_modes used A = 2|eta_hat|, but the HOS convention is
       eta(x) = sum_k Re(eta_hat[k] e^{ik.x})  (one-sided spectrum, Re[] — no conjugate pair),
       so the per-mode cosine amplitude is |eta_hat|. Verified: scripts/e1_factor2_audit.py
       (single-mode render matches evaluate_at to 6e-16 with A=|a|; 2x overshoot with 2|a|).
       This is why oracle Hs_ratio was 1.11-1.68 (>1, impossible for a partial mode set).
  (ii) the old O1 renders ONE element of the linear class (init modes propagated to t), not the
       BEST linear fit at t — after nonlinear HOS evolution the init A,phi are no longer optimal.

Corrected floors computed here per window (same 15 windows as d3_r5_oracle / ceiling_metrics):
  O1_fix_N   : raw render of top-N init modes with corrected A=|eta_hat| (flaw (i) fixed only).
  scalar     : optimal per-window scalar rescale of the raw oracle field  = sqrt(1-rho^2)/4
               (cross-check: predicted ~0.169 pooled from ceiling_metrics corr column).
  O1p_N      : TRUE linear floor — LS-refit (ridge, light) of A,phi of the fixed top-N (kx,ky)
               geometry directly to the FULL GT grid at t*. Best the linear class can do.
               Monotonicity O1p_64 <= O1p_32 must hold.

Run: python3 scripts/d3_e1_oracle_fix.py [--smoke]
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
import d44_hos_nonlinear as H
from d3_r5_oracle import init_modes

SEAS = [('BF3', 0.6, 5.5), ('BF4', 1.2, 7.0), ('BF5', 2.0, 9.0)]
SEEDS = [0, 1, 2, 3, 4]
GX = np.linspace(10, 250, 80); GY = np.linspace(-100, 100, 60)
DEPTH = H.DEPTH
CACHE = 'tmp/e1_windows'                       # repo tmp/ per project convention
RIDGE = 1e-2                                   # light: Gram diag ~ n_grid/2 = 2400 >> 1e-2


def gt_window(tag, Hs, Tp, seed):
    """Cache only what E1 needs from the heavy build_window: GT field + time at t*."""
    os.makedirs(CACHE, exist_ok=True)
    fp = f'{CACHE}/{tag}_s{seed}.npz'
    if os.path.exists(fp):
        d = np.load(fp)
        return d['gt'], float(d['t'])
    n_back = int(round(2.2 * Tp / 0.1))
    fr = H.build_window(Hs, Tp, seed, n_back, GX, GY, M=3)
    i_star = len(fr) - 2
    gt = fr[i_star]['gt']; t = float(fr[i_star]['t'])
    np.savez_compressed(fp, gt=gt, t=t)
    return gt, t


def nrmse(f, gt, Hs):
    return float(np.sqrt(np.mean((f - gt) ** 2)) / max(Hs, 1e-6))


def ls_refit_floor(kx, ky, gt, t, Hs_t):
    """Best-linear-fit floor: ridge-LS of A,phi for fixed (kx,ky) on the full GT grid at t*."""
    Xg, Yg = np.meshgrid(GX, GY)
    obs = np.column_stack([Xg.ravel(), Yg.ravel(),
                           np.full(Xg.size, t), gt.ravel()]).astype(np.float64)
    A, phi = fit_amplitudes_phases(kx.astype(np.float64), ky.astype(np.float64),
                                   obs, DEPTH, lambda_reg=RIDGE)
    f = render_wave_field(kx, ky, A, phi, DEPTH, GX, GY, t)
    return nrmse(f, gt, Hs_t), f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    seeds = [0] if args.smoke else SEEDS

    # B2 + old-oracle per-window references (same windows: d3_ceiling_gif uses build_window too)
    ceil = json.load(open('results/direction3/ceiling_metrics.json'))

    rows = {}; t0 = time.time()
    hdr = (f"{'window':9s} {'oldO1_32':>8} {'O1fix32':>8} {'Hsr_fix':>7} {'rho':>6} "
           f"{'scalar':>7} {'O1p_32':>7} {'O1p_64':>7} {'B2':>6}")
    print("E1 oracle fix — corrected linear floors (15 HOS windows)"); print(hdr, flush=True)
    for tag, Hs, Tp in SEAS:
        for seed in seeds:
            gt, t = gt_window(tag, Hs, Tp, seed)
            Hs_t = float(gt.std() * 4)
            kx, ky, A2, phi = init_modes(Hs, Tp, seed)   # A2 carries the old 2x bug
            A = A2 / 2.0                                  # corrected amplitude = |eta_hat|

            # raw renders (old vs fixed) — old must reproduce ceiling_metrics oracle nrmse
            f_old = render_wave_field(kx[:32], ky[:32], A2[:32], phi[:32], DEPTH, GX, GY, t)
            f_fix32 = f_old / 2.0                         # render is linear in A
            f_fix64 = render_wave_field(kx[:64], ky[:64], A[:64], phi[:64], DEPTH, GX, GY, t)
            o_old = nrmse(f_old, gt, Hs_t)
            o_fix32 = nrmse(f_fix32, gt, Hs_t); o_fix64 = nrmse(f_fix64, gt, Hs_t)
            hsr_fix = float(f_fix32.std() / gt.std())

            # optimal scalar rescale of the raw oracle field (identity: sqrt(1-rho^2)/4)
            rho = float(np.corrcoef(f_fix32.ravel(), gt.ravel())[0, 1])
            fc = f_fix32 - f_fix32.mean(); gc = gt - gt.mean()
            beta = float((fc * gc).mean() / max((fc * fc).mean(), 1e-12))
            o_scalar = nrmse(beta * fc + gt.mean(), gt, Hs_t)

            # TRUE linear floor: LS refit A,phi on full GT grid at t*
            o1p32, _ = ls_refit_floor(kx[:32], ky[:32], gt, t, Hs_t)
            o1p64, _ = ls_refit_floor(kx[:64], ky[:64], gt, t, Hs_t)

            key = f'{tag}_s{seed}'
            b2 = ceil[key]['b2']['nrmse']; o_old_ref = ceil[key]['oracle']['nrmse']
            rows[key] = dict(old_O1_32=o_old, old_O1_32_ref=o_old_ref,
                             O1_fix_32=o_fix32, O1_fix_64=o_fix64, Hs_ratio_fix=hsr_fix,
                             rho=rho, scalar_floor=o_scalar,
                             scalar_identity=float(np.sqrt(max(1 - rho ** 2, 0)) / 4),
                             O1p_32=o1p32, O1p_64=o1p64, B2=b2)
            print(f"{key:9s} {o_old:8.3f} {o_fix32:8.3f} {hsr_fix:7.3f} {rho:6.3f} "
                  f"{o_scalar:7.3f} {o1p32:7.3f} {o1p64:7.3f} {b2:6.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- summary ----
    keys = ['old_O1_32', 'O1_fix_32', 'O1_fix_64', 'scalar_floor', 'O1p_32', 'O1p_64', 'B2']
    tags = np.array([k.split('_')[0] for k in rows])
    summary = {}
    print("\n=== E1 summary (mean; per-sea) ===", flush=True)
    for k in keys:
        arr = np.array([r[k] for r in rows.values()])
        ps = {s: float(arr[tags == s].mean()) for s in ('BF3', 'BF4', 'BF5') if (tags == s).any()}
        summary[k] = dict(mean=float(arr.mean()), per_sea=ps)
        print(f"  {k:12s} {arr.mean():.3f}  |  " +
              "  ".join(f"{s} {v:.3f}" for s, v in ps.items()), flush=True)

    # ---- sanity gates ----
    mono = [k for k, r in rows.items() if r['O1p_64'] > r['O1p_32'] + 1e-6]
    below = [k for k, r in rows.items() if r['O1p_32'] > r['B2'] + 1e-6]
    repro = float(np.max([abs(r['old_O1_32'] - r['old_O1_32_ref']) for r in rows.values()]))
    ident = float(np.max([abs(r['scalar_floor'] - r['scalar_identity']) for r in rows.values()]))
    print("\n=== sanity gates ===", flush=True)
    print(f"  old-O1 reproduction vs ceiling_metrics: max |diff| = {repro:.4f}"
          f"  ({'OK' if repro < 5e-3 else 'MISMATCH'})", flush=True)
    print(f"  scalar identity sqrt(1-rho^2)/4:        max |diff| = {ident:.4f}"
          f"  ({'OK' if ident < 5e-3 else 'MISMATCH'})", flush=True)
    print(f"  monotonicity O1p_64 <= O1p_32:          violations = {mono or 'none'}", flush=True)
    print(f"  floor below B2 (O1p_32 < B2):           violations = {below or 'none'}", flush=True)

    out = dict(summary=summary, per_window=rows, ridge=RIDGE, smoke=args.smoke,
               gates=dict(mono_violations=mono, floor_above_b2=below,
                          old_o1_repro_maxdiff=repro, scalar_identity_maxdiff=ident))
    if not args.smoke:
        json.dump(out, open('results/direction3/e1_oracle_fix.json', 'w'), indent=2)
        print("saved results/direction3/e1_oracle_fix.json", flush=True)


if __name__ == '__main__':
    main()
