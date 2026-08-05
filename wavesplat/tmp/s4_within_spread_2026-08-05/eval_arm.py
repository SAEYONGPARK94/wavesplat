"""S4 arm evaluation -- WS-P1 (median-of-3) vs stock LETKF, one session, one script.

Protocols are lifted verbatim from the runs the MDE anchors came from, so the
new numbers are comparable to the old ones:
  WS    <- tmp/rerun_2026-07-25/batchC_ws_phasefix.py  (WS-P1, init_method
           explicit, per-frame panel averaged, spectral accuracy on final state)
  LETKF <- tmp/rerun_2026-07-25/batchA_baselines.py    (stock LETKFGrid, 40x30
           analysis grid, RegularGridInterpolator to the GT grid, seed 42)

Also records the REALIZED within / across / total sigma_1 of each saved
episode's stored wave_components -- the manipulation check on the data that was
actually estimated, not on a re-invocation of the generator.

Run:  WORKERS=5 python3 tmp/s4_within_spread_2026-08-05/eval_arm.py
Resumable: appends JSONL keyed by job_id.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
HERE = os.path.join(ROOT, 'tmp/s4_within_spread_2026-08-05')
DATA = os.path.join(ROOT, 'data_dirspread_2026-08-05')
OUT = os.path.join(HERE, 'eval.jsonl')

WS_SEEDS = [42, 123, 456]
P1 = {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
      'adaptive_n': True, 'tracking_window': 3}
PANEL = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
         'Directional_R2', 'Hs_error']
FMIN, FMAX = 0.04, 0.50


# ---------- realized directional structure of a SAVED episode ----------------
def realized_sigma(path):
    import h5py
    with h5py.File(path, 'r') as f:
        c = f['config/wave_components'][()]      # [A, omega, kx, ky, phi, Q]
        attrs = {k: (v.item() if hasattr(v, 'item') else v)
                 for k, v in f['config'].attrs.items()
                 if k in ('dirs_per_freq', 'direction_spread_deg', 'mesh_verdict',
                          'n_components', 'lambda_min')}
    A, om, kx, ky = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    th = np.arctan2(ky, kx)
    fr = om / (2 * np.pi)
    E = 0.5 * A ** 2
    m = (fr >= FMIN) & (fr <= FMAX)
    om, th, E = om[m], th[m], E[m]
    w = E / E.sum()
    r1_tot = float(np.hypot((w * np.cos(th)).sum(), (w * np.sin(th)).sum()))
    uniq = np.unique(om)
    r1_w, Ew = 0.0, 0.0
    for o in uniq:
        b = om == o
        wb = E[b] / E[b].sum()
        r1_w += E[b].sum() * float(np.hypot((wb * np.cos(th[b])).sum(),
                                            (wb * np.sin(th[b])).sum()))
        Ew += E[b].sum()
    r1_w /= Ew

    def s1(r):
        r = min(max(float(r), 0.0), 1.0)
        return float(np.degrees(np.sqrt(max(2.0 * (1.0 - r), 0.0))))
    return dict(n_freq_bins=int(uniq.size), sigma1_within_deg=s1(r1_w),
                sigma1_across_deg=s1(r1_tot / r1_w if r1_w > 0 else np.nan),
                sigma1_total_deg=s1(r1_tot),
                Hs_from_components=float(4.0 * np.sqrt((c[:, 0] ** 2 / 2.0).sum())),
                **attrs)


# ---------- spectral matching, verbatim from batchC_ws_phasefix.py -----------
def match_components(est_kx, est_ky, est_w, gt_kx, gt_ky, gt_A):
    N_est, N_gt = len(est_kx), len(gt_kx)
    if N_est == 0 or N_gt == 0:
        return [], list(range(N_gt)), list(range(N_est))
    k_scale = max(np.max(np.sqrt(gt_kx ** 2 + gt_ky ** 2)), 0.01)
    cost = np.zeros((N_est, N_gt))
    for i in range(N_est):
        for j in range(N_gt):
            dk = np.sqrt((est_kx[i] - gt_kx[j]) ** 2 + (est_ky[i] - gt_ky[j]) ** 2)
            cost[i, j] = dk / k_scale
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = [(r, c, cost[r, c]) for r, c in zip(row_ind, col_ind) if cost[r, c] < 0.5]
    mg = {m[1] for m in matched}
    me = {m[0] for m in matched}
    return matched, [j for j in range(N_gt) if j not in mg], \
        [i for i in range(N_est) if i not in me]


def spectral_accuracy(est_params, gt_components):
    gt_A, gt_omega = gt_components[:, 0], gt_components[:, 1]
    gt_kx, gt_ky = gt_components[:, 2], gt_components[:, 3]
    est_kx = np.asarray(est_params['kx']); est_ky = np.asarray(est_params['ky'])
    est_w = np.abs(np.asarray(est_params['w'])); est_omega = np.asarray(est_params['omega'])
    matched, _, _ = match_components(est_kx, est_ky, est_w, gt_kx, gt_ky, gt_A)
    res = {'n_gt': int(len(gt_A)), 'n_est': int(len(est_kx)), 'n_matched': int(len(matched)),
           'detection_rate': len(matched) / max(len(gt_A), 1)}
    if matched:
        km, oe, ae, de = [], [], [], []
        for ei, gi, _ in matched:
            gk = np.hypot(gt_kx[gi], gt_ky[gi]); ek = np.hypot(est_kx[ei], est_ky[ei])
            km.append((ek - gk) / max(gk, 1e-6))
            oe.append((est_omega[ei] - gt_omega[gi]) / max(gt_omega[gi], 1e-6))
            ae.append((est_w[ei] - gt_A[gi]) / max(gt_A[gi], 1e-6))
            gd = np.degrees(np.arctan2(gt_ky[gi], gt_kx[gi]))
            ed = np.degrees(np.arctan2(est_ky[ei], est_kx[ei]))
            d = abs(ed - gd); de.append(360 - d if d > 180 else d)
        res.update(k_mag_rel_error_mean=float(np.mean(np.abs(km))),
                   omega_rel_error_mean=float(np.mean(np.abs(oe))),
                   amp_rel_error_mean=float(np.mean(np.abs(ae))),
                   direction_error_deg_mean=float(np.mean(de)),
                   energy_detection_rate=float(np.sum(gt_A[[m[1] for m in matched]] ** 2)
                                               / max(float(np.sum(gt_A ** 2)), 1e-12)))
    else:
        res.update(k_mag_rel_error_mean=float('nan'), omega_rel_error_mean=float('nan'),
                   amp_rel_error_mean=float('nan'), direction_error_deg_mean=float('nan'),
                   energy_detection_rate=0.0)
    return res


def run_ws(path, seed):
    from wavesplat.run import run_estimation
    from wavesplat.data_loader import EpisodeDataLoader
    cfg = dict(P1); cfg['seed'] = seed
    r = run_estimation(path, cfg)
    frames = r['frame_results']
    avg = {k: float(np.nanmean([f[k] for f in frames if k in f])) for k in PANEL}
    ru = [f['RMSE_unobserved'] for f in frames
          if np.isfinite(f.get('RMSE_unobserved', np.nan))]
    loader = EpisodeDataLoader(path)
    spec = spectral_accuracy(r['spectrum_params'], np.asarray(loader.wave_components))
    return {**avg, 'avg_RMSE_unobserved': float(np.mean(ru)) if ru else float('nan'),
            'elapsed_ms': float(np.mean([f['elapsed_ms'] for f in frames])) if frames else float('nan'),
            'N': r.get('n_gaussians', -1), 'spec': spec}


def run_letkf(path):
    from wavesplat.baselines.letkf_grid import LETKFGrid
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.evaluation.metrics import compute_all_metrics
    from scipy.interpolate import RegularGridInterpolator
    np.random.seed(42)
    loader = EpisodeDataLoader(path)
    gx = loader.gt_grid_x.cpu().numpy(); gy = loader.gt_grid_y.cpu().numpy()
    egx = np.linspace(gx[0], gx[-1], 40); egy = np.linspace(gy[0], gy[-1], 30)
    b = LETKFGrid(egx, egy, depth=loader.depth, n_ensemble=100, mult_inflation=1.0,
                  add_inflation=0.005, loc_radius=50.0)
    Xg, Yg = np.meshgrid(gx, gy)
    fm = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy(); y = frame.y_obs.cpu().numpy(); z = frame.z_obs.cpu().numpy()
        el = b.update(x, y, z)
        pred = b.predict_grid()
        interp = RegularGridInterpolator((egy, egx), pred, method='linear',
                                        bounds_error=False, fill_value=0.0)
        pf = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(len(gy), len(gx))
        m = compute_all_metrics(pf, frame.gt_field.cpu().numpy(), gx, gy, loader.Hs_true,
                                obs_x=x if len(x) else None, obs_y=y if len(y) else None)
        m['elapsed_ms'] = el; fm.append(m)
    out = {k: float(np.nanmean([m.get(k, np.nan) for m in fm]))
           for k in PANEL + ['RMSE_unobserved']}
    out['elapsed_ms'] = float(np.nanmean([m['elapsed_ms'] for m in fm]))
    return out


ARM_OF = {(30, 1): 'A_M1_N30', (30, 3): 'B_M3_N30', (10, 1): 'C_M1_N10'}


def build_jobs():
    jobs = []
    for ep in sorted(f for f in os.listdir(DATA) if f.endswith('.h5')):
        ncomp = int(ep.split('_')[2].replace('comp', ''))
        M = 3 if '_m3' in ep else 1
        seed = int(ep.split('_seed')[1].split('_')[0])
        base = dict(episode=ep, arm=ARM_OF[(ncomp, M)], n_components=ncomp,
                    dirs_per_freq=M, ep_seed=seed)
        for s in WS_SEEDS:
            jobs.append({**base, 'method': 'WS', 'est_seed': s,
                         'job_id': f'WS|{ep}|{s}'})
        jobs.append({**base, 'method': 'LETKF', 'est_seed': 42,
                     'job_id': f'LETKF|{ep}'})
    return jobs


def run_one(job):
    sys.path.insert(0, ROOT)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    path = os.path.join(DATA, job['episode'])
    t0 = time.time()
    try:
        r = run_ws(path, job['est_seed']) if job['method'] == 'WS' else run_letkf(path)
        return {**job, **r, 'realized': realized_sigma(path), 'wall_s': time.time() - t0}
    except Exception as e:
        import traceback
        return {**job, 'error': repr(e), 'tb': traceback.format_exc(),
                'wall_s': time.time() - t0}


def _init():
    sys.path.insert(0, ROOT)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')


def main():
    import multiprocessing as mp
    jobs = build_jobs()
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add(r['job_id'])
    todo = [j for j in jobs if j['job_id'] not in done]
    nw = int(os.environ.get('WORKERS', '5'))
    print(f'total {len(jobs)} | done {len(done)} | todo {len(todo)} | workers {nw}', flush=True)
    if not todo:
        print('nothing to do'); return
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(nw, initializer=_init) as pool, open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] {rec['job_id']:<52s} "
                  f"{rec.get('nRMSE', rec.get('error','?'))} "
                  f"ETA {el/i*(len(todo)-i)/60:.1f}min", flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
