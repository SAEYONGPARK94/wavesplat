"""W7/W8 HARVEST — hand (analytic) arm over all 6 windows of the 8 HELD-OUT
Bergamasco records. Protocol frozen in tmp/heldout_real_2026-07-26/PREREG.md.

CPU ONLY. Learned arm never touched. The 30 design windows are NOT recomputed
(read verbatim from tmp/unified_2026-07-26/hand_amp.json, which already passed a
bit-identity control against wp_real.json).

  nice -n 10 python3 tmp/heldout_real_2026-07-26/harvest.py
"""
import os, sys, json, time, traceback
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp/heldout_real_2026-07-26')

WINS = list(range(6))                       # frozen 6-window convention (scene_real)
OUT = 'tmp/heldout_real_2026-07-26/harvest_heldout.json'


def hand_window(args):
    """Exact hand arm + the two GT-free amplitude signals (rescore_hand_amp.py)."""
    rec, win = args
    import heldout_cfg
    heldout_cfg.install()
    import d3_compare_gifs as cg
    from d3_sa_nomp import mmse_bank_estimate
    t0 = time.time()
    sc = cg.scene_real(rec, win=win)
    s0 = 0.05 * sc['Hs']; npx = sc['npx']; nl = len(sc['leads'])
    fq = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'], sigma0=s0,
                            kp_hint=sc['kp'], seed=sc['seed'])
    maps = fq[:npx * nl].reshape(nl, *sc['Xq'].shape)
    gt = sc['maps_gt']; sg = float(np.std(gt[np.isfinite(gt)]))
    m = np.isfinite(gt[0]) & np.isfinite(maps[0])
    now = float(np.sqrt(np.mean((maps[0][m] - gt[0][m]) ** 2)) / sg) if m.sum() >= 8 else None
    # lead-averaged (context only; not an endpoint)
    per = []
    for li in range(nl):
        mm = np.isfinite(gt[li]) & np.isfinite(maps[li])
        per.append(float(np.sqrt(np.mean((maps[li][mm] - gt[li][mm]) ** 2)) / sg)
                   if mm.sum() >= 8 else None)
    ok = [v for v in per if v is not None]
    lead = float(np.mean(ok)) if ok else None
    # ---- GT-free amplitude signals (verbatim definitions) ----
    z = np.asarray(sc['z'], float)
    z_rms = float(np.sqrt(np.mean(z ** 2)))
    finite = maps[0][np.isfinite(maps[0])]
    pred_grid_rms = float(np.sqrt(np.mean(finite ** 2))) if finite.size else float('nan')
    amp_ratio_grid = pred_grid_rms / z_rms if z_rms > 0 else float('nan')
    fo = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['P'], sigma0=s0,
                            kp_hint=sc['kp'], seed=sc['seed'])
    pred_obs = np.asarray(fo[:len(z)], float)
    pred_obs_rms = float(np.sqrt(np.mean(pred_obs ** 2)))
    amp_ratio_obs = pred_obs_rms / z_rms if z_rms > 0 else float('nan')
    return dict(record=rec, win=int(win), arena='heldout', hand_now=now, hand_lead=lead,
                amp_ratio_grid=amp_ratio_grid, amp_ratio_obs=amp_ratio_obs,
                z_rms=z_rms, pred_grid_rms=pred_grid_rms, pred_obs_rms=pred_obs_rms,
                Hs=float(sc['Hs']), kp=float(sc['kp']), lam_p=float(2 * np.pi / sc['kp']),
                n_obs=int(len(z)), npx=int(npx), n_gt_finite=int(m.sum()),
                gt_sigma=sg, sec=time.time() - t0)


def main():
    import heldout_cfg
    names = sorted(heldout_cfg.heldout_records())
    jobs = [(r, w) for r in names for w in WINS]
    print(f"# HELD-OUT harvest: {len(names)} records x {len(WINS)} windows "
          f"= {len(jobs)} hand-arm runs, 6 workers", flush=True)
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(hand_window, j): j for j in jobs}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                r = fu.result()
            except Exception as e:
                print(f"  {j[0]} w{j[1]}: FAIL {e!r}", flush=True)
                traceback.print_exc()
                rows.append(dict(record=j[0], win=j[1], arena='heldout', error=repr(e)))
                continue
            rows.append(r)
            print(f"  {r['record']}_w{r['win']} now={r['hand_now']:.3f} "
                  f"amp_grid={r['amp_ratio_grid']:.3f} amp_obs={r['amp_ratio_obs']:.3f} "
                  f"Hs={r['Hs']:.2f} npx={r['npx']} ({r['sec']:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r['record'], r['win']))
    json.dump(dict(rows=rows, wins=WINS, records=names,
                   cfg={k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                            for kk, vv in v.items()}
                        for k, v in heldout_cfg.heldout_records().items()},
                   prereg='tmp/heldout_real_2026-07-26/PREREG.md',
                   elapsed_s=time.time() - t0), open(OUT, 'w'), indent=1)
    print(f"\nwrote {OUT}: {len(rows)} rows, {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
