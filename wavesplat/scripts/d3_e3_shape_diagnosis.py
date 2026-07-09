"""E3 — shape diagnosis: split B2's estimation error into mode-IDENTIFIABILITY vs
A,phi-FITTING.

E1: TRUE linear floor O1p_32 = 0.128 pooled (LS-refit of true top-32 (kx,ky) to full GT).
E2: even INSIDE the cg-cone informed zone B2 sits ~+0.11 above that floor -> the
bottleneck is the ESTIMATOR. E3 asks WHICH part of the estimator:

Ladder per window at the nowcast t* (nRMSE + spatial rho):
  B2_obs      : B2 as-is — DPS sample + polish, A,phi fit to the 400 SPARSE obs
                (d3_ceiling_gif.b2_components -> render_bma). Real pipeline. [~0.25]
  B2modes_GT  : "oracle-polish" — B2's ESTIMATED (kx,ky) kept, but A,phi re-fit to the
                FULL TRUE GT grid at t* (same ridge-LS as O1p, ridge 1e-2), per BMA
                sample, fields averaged exactly like B2_obs. Removes ALL amplitude/
                phase-fitting error, keeps B2's mode GEOMETRY.
  O1p_true    : E1's corrected linear floor — TRUE top-32 modes refit to GT (must
                reproduce e1_oracle_fix.json exactly; same code path).
Reading the ladder (gap = B2_obs - O1p_true):
  d_fit  = B2_obs - B2modes_GT   : error attributable to A,phi FITTING from sparse obs
  d_geom = B2modes_GT - O1p_true : residual even with PERFECT A,phi on GT
                                   = mode-geometry (IDENTIFIABILITY) error
  d_geom small -> geometry is fine, the lever is a better likelihood/obs-fit/more obs.
  d_geom large -> mode selection is wrong, the lever is better detection
                  (e.g. the beamform-detect line).
Auxiliary:
  b2modes_gt_ps : mean over BMA samples of the PER-SAMPLE GT-refit nRMSE
                  (capacity-matched, <=30 modes vs true 32; no averaging confound).
  chamfer       : one-sided mean NN distance in (kx,ky) from B2 modes (pooled over
                  samples) to the true top-32 set, raw [rad/m] and /k_p; plus an
                  obs-fit-amplitude-weighted version (heavy modes matter more).

Run:  python3 scripts/d3_e3_shape_diagnosis.py --smoke   (BF3_s0/BF4_s0/BF5_s0, GPU)
      python3 scripts/d3_e3_shape_diagnosis.py           (full 15 windows)
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import torch
import d44_hos_nonlinear as H
import d3_setdiff_posterior as DP                      # sets DEVICE (GPU if free)
from d3_setdiff_prior import SetDenoiser
from d3_r5_oracle import init_modes
from d3_e1_oracle_fix import ls_refit_floor, nrmse, GX, GY, CACHE
from d3_ceiling_gif import b2_components, render_bma, PRIOR

G = 9.81
DEPTH = H.DEPTH
SEAS = [('BF3', 0.6, 5.5), ('BF4', 1.2, 7.0), ('BF5', 2.0, 9.0)]
SEEDS = [0, 1, 2, 3, 4]
OUT_JSON = 'results/direction3/e3_shape_diagnosis.json'


def window_data(tag, seed):
    """E2-extended cache (gt, t, P, z) — all 15 windows are pre-cached."""
    fp = f'{CACHE}/{tag}_s{seed}.npz'
    if not os.path.exists(fp):
        raise FileNotFoundError(f'{fp}: run d3_e2_zone_split.py --cache_only {tag}_s{seed}')
    d = np.load(fp)
    assert 'P' in d.files, f'{fp}: old E1-only cache; re-extend via d3_e2_zone_split.py'
    return d['gt'], float(d['t']), d['P'], d['z']


def peak_wavenumber(Tp, d=DEPTH):
    om = 2 * np.pi / Tp
    k = om ** 2 / G
    for _ in range(60):
        k = om ** 2 / (G * np.tanh(k * d))
    return k


def rho(f, gt):
    return float(np.corrcoef(f.ravel(), gt.ravel())[0, 1])


def chamfer_to_true(comps, kx_true, ky_true):
    """One-sided NN distance in (kx,ky): B2 modes (pooled over samples) -> true set.
    Returns (mean, A_obs-weighted mean) in rad/m."""
    ds, ws = [], []
    for kx, ky, A, _ in comps:
        d = np.sqrt((kx[:, None] - kx_true[None, :]) ** 2 +
                    (ky[:, None] - ky_true[None, :]) ** 2).min(axis=1)
        ds.append(d); ws.append(A)
    d = np.concatenate(ds); w = np.concatenate(ws)
    return float(d.mean()), float((d * w).sum() / max(w.sum(), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='seeds [0] only (3 windows)')
    args = ap.parse_args()
    seeds = [0] if args.smoke else SEEDS

    ck = torch.load(PRIOR, map_location=DP.DEVICE)
    stats = ck['stats']
    prior = SetDenoiser(**ck.get('arch', {})).to(DP.DEVICE)
    prior.load_state_dict(ck['model']); prior.eval()
    abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)
    ceil = json.load(open('results/direction3/ceiling_metrics.json'))
    e1 = json.load(open('results/direction3/e1_oracle_fix.json'))['per_window']

    rows = {}; t0 = time.time()
    print(f'E3 shape diagnosis — B2_obs -> B2modes_GT -> O1p_true  [device={DP.DEVICE}]',
          flush=True)
    hdr = (f"{'window':8s} {'nM':>5} | {'B2obs':>5} {'B2mGT':>5} {'psGT':>5} {'O1p':>5} | "
           f"{'dFit':>5} {'dGeom':>5} {'geo%':>4} | {'rObs':>5} {'rmGT':>5} | "
           f"{'chm':>5} {'ch/kp':>5} {'chW':>5}")
    print(hdr, flush=True)

    for tag, Hs, Tp in SEAS:
        kp = peak_wavenumber(Tp)
        for seed in seeds:
            gt, t, P, z = window_data(tag, seed)
            Hs_t = float(gt.std() * 4)
            idx = np.random.default_rng(seed).choice(len(z), min(400, len(z)), replace=False)
            Ps, zs = P[idx].astype(np.float32), z[idx].astype(np.float32)

            # 1) B2_obs — real pipeline (DPS + polish + sparse-obs A,phi fit, BMA of 2)
            comps = b2_components(prior, stats, abar, betas, Ps, zs, DEPTH)
            if not comps:
                print(f'{tag}_s{seed}: EMPTY B2 component set — skipped'); continue
            f_obs = render_bma(comps, DEPTH, t)
            nr_obs, rho_obs = nrmse(f_obs, gt, Hs_t), rho(f_obs, gt)

            # 2) B2modes_GT — same modes, A,phi refit to FULL GT grid, same BMA average
            ps_nr, f_refits = [], []
            for kx, ky, _, _ in comps:
                nr_s, f_s = ls_refit_floor(kx.astype(np.float64), ky.astype(np.float64),
                                           gt, t, Hs_t)
                ps_nr.append(nr_s); f_refits.append(f_s)
            f_mgt = np.mean(f_refits, axis=0)
            nr_mgt, rho_mgt = nrmse(f_mgt, gt, Hs_t), rho(f_mgt, gt)
            nr_ps = float(np.mean(ps_nr))                 # capacity-matched auxiliary

            # 3) O1p_true — E1 floor recomputed (must match e1_oracle_fix exactly)
            kxT, kyT, _, _ = init_modes(Hs, Tp, seed)
            nr_o1p, f_o1p = ls_refit_floor(kxT[:32], kyT[:32], gt, t, Hs_t)
            rho_o1p = rho(f_o1p, gt)

            # ladder split + mode-geometry Chamfer
            gap = nr_obs - nr_o1p
            d_fit, d_geom = nr_obs - nr_mgt, nr_mgt - nr_o1p
            geo_share = d_geom / gap if abs(gap) > 1e-9 else None
            chm, chm_w = chamfer_to_true(comps, kxT[:32], kyT[:32])
            n_modes = [int(len(c[0])) for c in comps]

            key = f'{tag}_s{seed}'
            rows[key] = dict(
                n_b2_modes=n_modes,
                b2_obs=dict(nrmse=nr_obs, rho=rho_obs),
                b2modes_gt=dict(nrmse=nr_mgt, rho=rho_mgt),
                b2modes_gt_persample=dict(nrmse=nr_ps, per_sample=ps_nr),
                o1p_true=dict(nrmse=nr_o1p, rho=rho_o1p),
                gap=gap, d_fit=d_fit, d_geom=d_geom, geom_share=geo_share,
                chamfer_mean=chm, chamfer_over_kp=chm / kp, chamfer_wmean=chm_w,
                k_p=float(kp),
                b2_ref_ceiling=ceil[key]['b2']['nrmse'], o1p_ref_e1=e1[key]['O1p_32'])
            print(f"{key:8s} {'+'.join(map(str, n_modes)):>5} | "
                  f"{nr_obs:5.3f} {nr_mgt:5.3f} {nr_ps:5.3f} {nr_o1p:5.3f} | "
                  f"{d_fit:5.3f} {d_geom:5.3f} {100 * (geo_share or 0):4.0f} | "
                  f"{rho_obs:5.2f} {rho_mgt:5.2f} | "
                  f"{chm:5.3f} {chm / kp:5.2f} {chm_w:5.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ── summary (window-mean, pooled + per-sea) ──
    tags = np.array([k.split('_')[0] for k in rows])

    def agg(fn):
        arr = np.array([fn(r) for r in rows.values()])
        ps = {s: float(arr[tags == s].mean()) for s in ('BF3', 'BF4', 'BF5')
              if (tags == s).any()}
        return dict(mean=float(arr.mean()), per_sea=ps)

    summary = dict(
        b2_obs=agg(lambda r: r['b2_obs']['nrmse']),
        b2modes_gt=agg(lambda r: r['b2modes_gt']['nrmse']),
        b2modes_gt_persample=agg(lambda r: r['b2modes_gt_persample']['nrmse']),
        o1p_true=agg(lambda r: r['o1p_true']['nrmse']),
        d_fit=agg(lambda r: r['d_fit']), d_geom=agg(lambda r: r['d_geom']),
        rho_b2_obs=agg(lambda r: r['b2_obs']['rho']),
        rho_b2modes_gt=agg(lambda r: r['b2modes_gt']['rho']),
        chamfer_over_kp=agg(lambda r: r['chamfer_over_kp']))
    gap_m = summary['b2_obs']['mean'] - summary['o1p_true']['mean']
    geo_share_m = summary['d_geom']['mean'] / gap_m if abs(gap_m) > 1e-9 else None

    print('\n=== E3 summary (window-mean) ===', flush=True)
    for k in ('b2_obs', 'b2modes_gt', 'b2modes_gt_persample', 'o1p_true'):
        s = summary[k]
        print(f"  {k:22s} {s['mean']:.3f}  |  " +
              "  ".join(f"{t} {v:.3f}" for t, v in s['per_sea'].items()), flush=True)
    print(f"  ladder: gap {gap_m:.3f} = d_fit {summary['d_fit']['mean']:.3f} "
          f"+ d_geom {summary['d_geom']['mean']:.3f}   "
          f"(geometry share {100 * geo_share_m:.0f}%)", flush=True)
    print(f"  chamfer/k_p mean {summary['chamfer_over_kp']['mean']:.2f}", flush=True)

    # ── sanity gates ──
    o1p_dev = max(abs(r['o1p_true']['nrmse'] - r['o1p_ref_e1']) for r in rows.values())
    b2_dev = max(abs(r['b2_obs']['nrmse'] - r['b2_ref_ceiling']) for r in rows.values())
    lad_hi = [k for k, r in rows.items() if r['b2modes_gt']['nrmse'] > r['b2_obs']['nrmse'] + 1e-6]
    lad_lo = [k for k, r in rows.items() if r['b2modes_gt']['nrmse'] < r['o1p_true']['nrmse'] - 1e-6]
    print('\n=== sanity gates ===', flush=True)
    print(f"  O1p_true == E1 O1p_32 (deterministic): max|diff| = {o1p_dev:.2e}  "
          f"({'OK' if o1p_dev < 1e-6 else 'MISMATCH'})", flush=True)
    print(f"  B2_obs vs ceiling ref (RNG-loose):     max|diff| = {b2_dev:.3f}  "
          f"({'OK' if b2_dev < 0.08 else 'CHECK'})", flush=True)
    print(f"  ladder B2modes_GT <= B2_obs:  violations = {lad_hi or 'none'}", flush=True)
    print(f"  ladder B2modes_GT >= O1p:     violations = {lad_lo or 'none'}"
          f"{'' if not lad_lo else '  (B2 capacity can exceed true-32; not fatal)'}",
          flush=True)

    # ── verdict ──
    if geo_share_m is None:
        verdict = 'gap ~ 0: nothing to split'
    elif geo_share_m >= 0.65:
        verdict = (f'IDENTIFIABILITY-dominated: {100 * geo_share_m:.0f}% of the '
                   f'B2->floor gap survives perfect A,phi on GT -> mode (kx,ky) '
                   f'selection is wrong -> lever = better DETECTION (beamform-detect line)')
    elif geo_share_m <= 0.35:
        verdict = (f'FITTING-dominated: only {100 * geo_share_m:.0f}% of the gap is '
                   f'mode geometry; {100 * (1 - geo_share_m):.0f}% vanishes once A,phi '
                   f'are fit to GT -> lever = better likelihood/obs-fit, NOT detection')
    else:
        verdict = (f'MIXED: geometry share {100 * geo_share_m:.0f}% — both detection '
                   f'and A,phi fitting contribute materially')
    print(f'\nREAD: {verdict}', flush=True)

    out = dict(summary=summary, gap_mean=gap_m, geom_share_mean=geo_share_m,
               per_window=rows, smoke=args.smoke, verdict=verdict,
               gates=dict(o1p_vs_e1_maxdiff=o1p_dev, b2_vs_ceiling_maxdiff=b2_dev,
                          ladder_above_b2obs=lad_hi, ladder_below_o1p=lad_lo))
    path = OUT_JSON if not args.smoke else OUT_JSON.replace('.json', '_smoke.json')
    json.dump(out, open(path, 'w'), indent=2)
    print(f'saved {path}', flush=True)


if __name__ == '__main__':
    main()
