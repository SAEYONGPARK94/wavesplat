"""UNIFIED Part-C detector DESIGN: re-score the analytic (hand) arm on all 30 real
windows to extract GT-free amplitude diagnostics for the collapse detector.

Small re-scoring only: re-runs mmse_bank_estimate (the exact hand arm) per window, then
computes GT-FREE signals:
  amp_ratio_grid = std(pred nowcast field over query grid) / std(observed z)
  amp_ratio_obs  = std(pred field rendered AT the obs points) / std(observed z)
Verifies hand_now reproduces wp_real.json bit-identically (controlled).

Writes tmp/unified_2026-07-26/hand_amp.json. Uses 3 CPU workers, OMP=1.
  nice -n 10 python3 tmp/unified_2026-07-26/rescore_hand_amp.py
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

sys.path.insert(0, 'scripts')
WP_REAL = 'tmp/wp_real_2026-07-25/wp_real.json'
DOP = 'tmp/doppler_real_2026-07-25/doppler_real.json'
OUT = 'tmp/unified_2026-07-26/hand_amp.json'
COLLAPSE = {('AA14', 0), ('AA15a', 3), ('LJ18g', 3)}


def score_maps(sc, maps):
    gt = sc['maps_gt']; sg = float(np.std(gt[np.isfinite(gt)]))
    m = np.isfinite(gt[0]) & np.isfinite(maps[0])
    if m.sum() < 8:
        return None
    return float(np.sqrt(np.mean((maps[0][m] - gt[0][m]) ** 2)) / sg)


def hand_window(args):
    rec, win = args
    import d3_compare_gifs as cg
    from d3_sa_nomp import mmse_bank_estimate
    sc = cg.scene_real(rec, win=win)
    s0 = 0.05 * sc['Hs']; npx = sc['npx']; nl = len(sc['leads'])
    # field at query grid (identical call to wp_real hand arm)
    fq = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'], sigma0=s0,
                            kp_hint=sc['kp'], seed=sc['seed'])
    maps = fq[:npx * nl].reshape(nl, *sc['Xq'].shape)
    now = score_maps(sc, maps)
    # GT-free amplitude signals
    z = np.asarray(sc['z'], float)
    z_rms = float(np.sqrt(np.mean(z ** 2)))
    nowmap = maps[0]
    finite = nowmap[np.isfinite(nowmap)]
    pred_grid_rms = float(np.sqrt(np.mean(finite ** 2))) if finite.size else float('nan')
    amp_ratio_grid = pred_grid_rms / z_rms if z_rms > 0 else float('nan')
    # field rendered AT obs points (second solve, same bank fit, query=obs)
    fo = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['P'], sigma0=s0,
                            kp_hint=sc['kp'], seed=sc['seed'])
    npo = len(sc['z'])
    pred_obs = np.asarray(fo[:npo], float)
    pred_obs_rms = float(np.sqrt(np.mean(pred_obs ** 2)))
    amp_ratio_obs = pred_obs_rms / z_rms if z_rms > 0 else float('nan')
    return dict(record=rec, win=int(win), collapse=(rec, win) in COLLAPSE,
                Hs=float(sc['Hs']), kp=float(sc['kp']), n_obs=int(npo),
                hand_now=now, z_rms=z_rms, pred_grid_rms=pred_grid_rms,
                amp_ratio_grid=amp_ratio_grid, pred_obs_rms=pred_obs_rms,
                amp_ratio_obs=amp_ratio_obs)


def main():
    wp = {(r['record'], r['win']): r for r in json.load(open(WP_REAL))['results']}
    dop = {(r['record'], r['win']): r for r in json.load(open(DOP))['results']}
    wins = sorted(wp.keys())
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(hand_window, k): k for k in wins}
        for fu in as_completed(futs):
            r = fu.result()
            k = (r['record'], r['win'])
            r['vr_u0'] = dop[k]['vr_u0']              # held-out obs residual, reused
            r['hand_now_ref'] = wp[k]['hand_now']     # for bit-identity check
            r['learned_now'] = wp[k]['learned_now']   # fallback target
            r['n_eff'] = wp[k]['n_eff']
            rows.append(r)
            print(f"  {r['record']}_w{r['win']:d} coll={int(r['collapse'])} "
                  f"now={r['hand_now']:.3f}(ref {r['hand_now_ref']:.3f}) "
                  f"amp_grid={r['amp_ratio_grid']:.3f} amp_obs={r['amp_ratio_obs']:.3f} "
                  f"vr={r['vr_u0']:.3f}", flush=True)
    rows.sort(key=lambda r: (r['record'], r['win']))
    # bit-identity check
    maxdiff = max(abs(r['hand_now'] - r['hand_now_ref']) for r in rows)
    print(f"\nhand_now max|diff vs wp_real| = {maxdiff:.2e}  (bit-identity control)")
    json.dump(dict(rows=rows, hand_now_maxdiff=maxdiff, collapse=sorted(list(COLLAPSE))),
              open(OUT, 'w'), indent=1)
    print(f"wrote {OUT}; {len(rows)} windows, {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
