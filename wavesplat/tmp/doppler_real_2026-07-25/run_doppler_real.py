"""STAGE-3a — current-Doppler phase-rate term on REAL stereo, analytic arm.

Per window (5 records x 6 windows): estimate a single ambient current U from the
observations alone (grid search minimizing the estimator's held-out observation
residual, GT never touched), then compare map_nrmse_nowcast of the analytic bank
solve with Uc=U_hat vs Uc=(0,0). Frozen protocol: see PREREG.md.

Run:  OMP_NUM_THREADS=1 nice -n 10 python3 tmp/doppler_real_2026-07-25/run_doppler_real.py
"""
import os
import sys
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
import d3_compare_gifs as cg                      # chdir's to pkg dir on import
from d3_sa_nomp import mmse_bank_estimate
from d3_falcon_real_operator import RECORDS

RMAX = 1.5
COARSE_STEP = 0.3
REFINE_STEP = 0.1
REFINE_HALF = 3
OUT = 'tmp/doppler_real_2026-07-25/doppler_real.json'
COLLAPSE = {('AA14', 0), ('AA15a', 3), ('LJ18g', 3)}   # WS>=1.0 in fixed run


def map_nrmse(sc, fq):
    """Exact d3_compare_gifs.metrics normalization (WaveSplat arm)."""
    npx, nl = sc['npx'], len(sc['leads'])
    maps = fq[:npx * nl].reshape(nl, *sc['Xq'].shape)
    gt = sc['maps_gt']
    sg = float(np.std(gt[np.isfinite(gt)]))        # pooled over all leads
    m0 = np.isfinite(gt[0]) & np.isfinite(maps[0])
    now = float(np.sqrt(np.mean((maps[0][m0] - gt[0][m0]) ** 2)) / sg)
    per = []
    for li in range(nl):
        m = np.isfinite(gt[li]) & np.isfinite(maps[li])
        if m.sum() < 8:
            continue
        per.append(np.sqrt(np.mean((maps[li][m] - gt[li][m]) ** 2)) / sg)
    lead = float(np.mean(per)) if per else None
    return now, lead


def coarse_grid():
    ax = np.arange(-RMAX, RMAX + 1e-9, COARSE_STEP)
    return [(round(float(ux), 4), round(float(uy), 4))
            for ux in ax for uy in ax if ux * ux + uy * uy <= RMAX ** 2 + 1e-9]


def refine_grid(u0):
    ux0, uy0 = u0
    pts = []
    for i in range(-REFINE_HALF, REFINE_HALF + 1):
        for j in range(-REFINE_HALF, REFINE_HALF + 1):
            ux, uy = ux0 + i * REFINE_STEP, uy0 + j * REFINE_STEP
            if ux * ux + uy * uy <= RMAX ** 2 + 1e-9:
                pts.append((round(ux, 4), round(uy, 4)))
    return pts


def run_window(args):
    record, win = args
    t0 = time.time()
    sc = cg.scene_real(record, win=win)
    s0 = 0.05 * sc['Hs']
    cache = {}

    def score(U):
        key = (round(U[0], 4), round(U[1], 4))
        if key in cache:
            return cache[key]
        gate = {}
        mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['P'][:1],
                           sigma0=s0, kp_hint=sc['kp'], seed=sc['seed'],
                           diag_out=gate, Uc=key)
        cache[key] = float(gate['vr'])
        return cache[key]

    # coarse search
    coarse = coarse_grid()
    vr_c = [(score(U), U) for U in coarse]
    vr_c.sort(key=lambda x: x[0])
    u_coarse = vr_c[0][1]
    # local refine
    for U in refine_grid(u_coarse):
        score(U)
    U_hat = min(cache.items(), key=lambda kv: kv[1])[0]
    vr_hat = cache[U_hat]
    vr_u0 = score((0.0, 0.0))

    # final full-Pq solves (analytic arm only)
    fq0 = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'],
                             sigma0=s0, kp_hint=sc['kp'], seed=sc['seed'],
                             Uc=(0.0, 0.0))
    fqh = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'],
                             sigma0=s0, kp_hint=sc['kp'], seed=sc['seed'],
                             Uc=U_hat)
    now0, lead0 = map_nrmse(sc, fq0)
    nowh, leadh = map_nrmse(sc, fqh)

    res = dict(record=record, win=int(win), collapse=(record, win) in COLLAPSE,
               Hs=float(sc['Hs']), kp=float(sc['kp']), n_obs=int(len(sc['z'])),
               U_hat=[float(U_hat[0]), float(U_hat[1])],
               U_mag=float(np.hypot(*U_hat)),
               U_at_boundary=bool(np.hypot(*U_hat) >= RMAX - 1e-6),
               vr_u0=float(vr_u0), vr_uhat=float(vr_hat),
               nrmse_now_u0=now0, nrmse_now_uhat=nowh,
               delta_now=now0 - nowh,
               nrmse_lead_u0=lead0, nrmse_lead_uhat=leadh,
               delta_lead=(lead0 - leadh) if (lead0 and leadh) else None,
               n_ucand=len(cache), sec=time.time() - t0)
    return res


def main():
    jobs = [(r, w) for r in RECORDS.keys() for w in range(6)]
    print(f"# {len(jobs)} windows, coarse {COARSE_STEP} + refine {REFINE_STEP}, "
          f"|U|<={RMAX}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(run_window, j): j for j in jobs}
        for fut in as_completed(futs):
            r, w = futs[fut]
            try:
                res = fut.result()
                results.append(res)
                tag = ' [COLLAPSE]' if res['collapse'] else ''
                print(f"[{res['record']} w{res['win']}] U_hat=({res['U_hat'][0]:+.1f},"
                      f"{res['U_hat'][1]:+.1f}) |U|={res['U_mag']:.2f}  "
                      f"nRMSE {res['nrmse_now_u0']:.3f}->{res['nrmse_now_uhat']:.3f} "
                      f"(d {res['delta_now']:+.3f})  vr {res['vr_u0']:.4f}->"
                      f"{res['vr_uhat']:.4f}  {res['sec']:.0f}s{tag}", flush=True)
            except Exception as e:
                import traceback
                print(f"[{r} w{w}] FAILED {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
    results.sort(key=lambda x: (x['record'], x['win']))
    json.dump(dict(results=results, prereg='PREREG.md',
                   coarse_step=COARSE_STEP, refine_step=REFINE_STEP, rmax=RMAX),
              open(OUT, 'w'), indent=1)
    print(f"\nsaved {OUT}  ({len(results)} windows)", flush=True)


if __name__ == '__main__':
    main()
