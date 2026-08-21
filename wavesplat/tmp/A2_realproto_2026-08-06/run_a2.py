"""A2 — real-data comparison protocol hygiene. Frozen protocol: PREREG.md (2026-08-06).

Hand arm (mmse_bank_estimate) run under a 2x2 factorial of the two GT-derived inputs
(kp_hint, sigma0) plus a 5th obs-budget-symmetric cell, on the same 30 scene_real windows
the 2026-07-25 wp_real run used. Learned arm is UNCHANGED (it never touched GT) and is
re-used from wp_real.json; a 6-window GPU spot-check is done separately.

Cells:
  FF     kp_hint=sc['kp'](GT)  sigma0=0.05*Hs_GT   600 obs   (published research arm, BUNDLED)
  BF     kp_hint=None          sigma0=0.05*Hs_GT   600 obs
  FB     kp_hint=sc['kp']      sigma0=0.05*Hs_obs  600 obs
  BB     kp_hint=None          sigma0=0.05*Hs_obs  600 obs   (SHIPPING)
  BB400  kp_hint=None          sigma0=0.05*Hs_obs  400 obs   (learned arm's obs budget)
Each cell: hand seeds {42,1042,2042} (shipping median-of-3 convention), median nowcast.
Plus FF_exact: seed=sc['seed'] single solve == wp_real reproduction control.
Also per-window flat-sea nulls.

Writes tmp/A2_realproto_2026-08-06/cells.json.  Additive only; nothing else is touched.
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

OUT = 'tmp/A2_realproto_2026-08-06/cells.json'
RECORDS_ORDER = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']
WINS = list(range(6))
HAND_SEEDS = [42, 1042, 2042]          # shipping: base + {0,1000,2000}, base=42
M_OBS_LEARNED = 400                    # d3_beamform_net.M_OBS


def nowcast_nrmse(sc, maps):
    """Exactly run_wp_real.score_maps lead-0 term: sg = std of pooled finite GT."""
    gt = sc['maps_gt']
    sg = float(np.std(gt[np.isfinite(gt)]))
    m = np.isfinite(gt[0]) & np.isfinite(maps[0])
    if m.sum() < 8:
        return None
    return float(np.sqrt(np.mean((maps[0][m] - gt[0][m]) ** 2)) / sg)


def kp_two_stage(P, z, depth):
    """Replicates mmse_bank_estimate's internal kp_hint=None estimate (logging only)."""
    from d18_relax_estimator import beamform_top
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          depth, klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          depth, klim=3.5 * kp, n=120)
    return float(max(np.hypot(bx, by), 1e-3))


def window_job(args):
    rec, win = args
    import d3_compare_gifs as cg
    from d3_sa_nomp import mmse_bank_estimate
    t_job = time.time()
    sc = cg.scene_real(rec, win=win)
    npx, nl = sc['npx'], len(sc['leads'])
    shp = sc['Xq'].shape
    P, z = sc['P'], sc['z']
    Hs_gt = float(sc['Hs']); kp_gt = float(sc['kp'])
    Hs_obs = 4.0 * float(np.std(z))

    # obs-budget-symmetric subsample: identical rule to run_wp_real.py:125-127
    if len(z) > M_OBS_LEARNED:
        ii = np.random.default_rng(0).choice(len(z), M_OBS_LEARNED, replace=False)
        P4, z4 = P[ii], z[ii]
    else:
        P4, z4 = P, z
    Hs_obs400 = 4.0 * float(np.std(z4))     # shipping caps THEN computes std

    # ---- nulls (flat sea) ----
    gt = sc['maps_gt']
    fin = np.isfinite(gt)
    sg = float(np.std(gt[fin]))
    mu = float(np.mean(gt[fin]))
    m0 = np.isfinite(gt[0])
    null_zero = float(np.sqrt(np.mean(gt[0][m0] ** 2)) / sg)
    null_mean = float(np.sqrt(np.mean((gt[0][m0] - mu) ** 2)) / sg)

    CELLS = {
        'FF':    dict(kp=kp_gt, s0=0.05 * Hs_gt,     obs='600'),
        'BF':    dict(kp=None,  s0=0.05 * Hs_gt,     obs='600'),
        'FB':    dict(kp=kp_gt, s0=0.05 * Hs_obs,    obs='600'),
        'BB':    dict(kp=None,  s0=0.05 * Hs_obs,    obs='600'),
        'BB400': dict(kp=None,  s0=0.05 * Hs_obs400, obs='400'),
    }

    out = dict(record=rec, win=win, n_obs=int(len(z)), n_obs400=int(len(z4)),
               Hs_gt=Hs_gt, Hs_obs=Hs_obs, Hs_obs400=Hs_obs400, kp_gt=kp_gt,
               sg=sg, gt_mean=mu, null_zero=null_zero, null_mean=null_mean,
               kp_est_600=kp_two_stage(P, z, sc['depth']),
               kp_est_400=kp_two_stage(P4, z4, sc['depth']),
               seed_i0=int(sc['seed']), cells={})

    for name, c in CELLS.items():
        Pc, zc = (P4, z4) if c['obs'] == '400' else (P, z)
        vals, secs, vrs = [], [], []
        for s in HAND_SEEDS:
            t = time.time()
            diag = {}
            fq = mmse_bank_estimate(Pc, zc, sc['depth'], sc['Pq'], sigma0=c['s0'],
                                    kp_hint=c['kp'], seed=s, diag_out=diag)
            v = nowcast_nrmse(sc, fq[:npx * nl].reshape(nl, *shp))
            vals.append(v); secs.append(round(time.time() - t, 2))
            vrs.append(float(diag.get('vr_rel', float('nan'))))
        out['cells'][name] = dict(kp_hint=c['kp'], sigma0=c['s0'], obs=c['obs'],
                                  seeds=HAND_SEEDS, per_seed=vals,
                                  median=float(np.median(vals)),
                                  spread=[float(min(vals)), float(max(vals))],
                                  vr_rel=vrs, sec=secs)

    # reproduction control: exact wp_real hand call
    t = time.time()
    fq = mmse_bank_estimate(P, z, sc['depth'], sc['Pq'], sigma0=0.05 * Hs_gt,
                            kp_hint=kp_gt, seed=int(sc['seed']))
    out['FF_exact'] = dict(nowcast=nowcast_nrmse(sc, fq[:npx * nl].reshape(nl, *shp)),
                           seed=int(sc['seed']), sec=round(time.time() - t, 2))
    out['job_sec'] = round(time.time() - t_job, 1)
    return out


def main():
    jobs = [(r, w) for r in RECORDS_ORDER for w in WINS]
    print(f"# A2 hand-arm factorial: {len(jobs)} windows x (5 cells x 3 seeds + exact) "
          f"= {len(jobs)*16} solves, 8 workers", flush=True)
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(window_job, j): j for j in jobs}
        for fut in as_completed(futs):
            o = fut.result(); res.append(o)
            c = o['cells']
            print(f"  [{o['record']} w{o['win']}] FF {c['FF']['median']:.4f} "
                  f"BF {c['BF']['median']:.4f} FB {c['FB']['median']:.4f} "
                  f"BB {c['BB']['median']:.4f} BB400 {c['BB400']['median']:.4f} | "
                  f"exact {o['FF_exact']['nowcast']:.4f} | null0 {o['null_zero']:.4f} "
                  f"| kp_gt {o['kp_gt']:.4f} Hs {o['Hs_gt']:.3f}/{o['Hs_obs']:.3f} "
                  f"({o['job_sec']}s)", flush=True)
    res.sort(key=lambda r: (RECORDS_ORDER.index(r['record']), r['win']))
    json.dump(dict(prereg='PREREG.md', hand_seeds=HAND_SEEDS,
                   m_obs_learned=M_OBS_LEARNED, results=res,
                   wall_sec=round(time.time() - t0, 1)),
              open(OUT, 'w'), indent=1)
    print(f"\nsaved {OUT}  ({len(res)} windows, {time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
