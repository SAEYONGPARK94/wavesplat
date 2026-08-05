"""E2 -- separate coarse-grid STEP from suppression RADIUS, and try to stop the stacking.

Pre-registered in tmp/e2_grid_sep_2026-08-05/PREREG.md. Cell definitions are frozen by that
file. Protocol inherited verbatim from tmp/e1_grid_2026-08-05/run_e1.py so cell A0 pairs
bit-for-bit with E1 cell A.

PRODUCTION CODE IS NOT MODIFIED. Every factor is applied by monkey-patching inside the
worker process:

  * STEP + COVERAGE -> wrap initializer._beamform_coarse (and the symbol coarse_to_fine
        imported at module load) so range AND grid count are forced at all THREE placement
        sites (initializer:465, coarse_to_fine:180, sequential_update:1015).

  * RADIUS -> set initializer.MIN_DK. The shipped arithmetic is
        `suppress = int(MIN_DK/dk) + 1` cells, i.e. an absolute half-width of
        `(int(MIN_DK/dk)+1)*dk`. To pin the absolute radius to c*dk we set
        MIN_DK = (c - 0.5)*dk, which makes int(MIN_DK/dk)+1 == c exactly. No production
        arithmetic is altered. Both 61-grid sites re-import MIN_DK inside the function, so
        they follow.

  * MINSEP (arm M) -> a model-level invariant "no two atoms closer than M rad/m":
      - site 1: wrap initialize_from_frames_v2. Call the SHIPPED extractor with an enlarged
        budget B = min(4n, 60) -- the greedy loop is sequential, so its first n atoms are
        identical to a budget-n call -- then keep, in order, the first n atoms that are >= M
        from every already-kept one. Rejected candidates keep their residual subtraction
        (the shipped sequence is not altered). Short results are padded by the shipped
        padding law (random k, PAD_W).
      - sites 2 & 3: wrap SpectralGaussianModel.add_gaussians and drop incoming atoms that
        are within M of an existing model atom or of an earlier-kept incoming one. Dropped
        atoms are NOT replaced (final_n can fall) -- pre-disclosed in PREREG section 2.3.

TIMING IS RECORDED BUT NOT CITABLE (this box's clock is under separate re-measurement).
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'results.jsonl')

SEEDS = [42, 123, 456]

# scripts/run_p1_comparison.py::EPISODES, verbatim (same order as E1).
EPISODES = [
    'episode_BF2_1comp_seed0_uniform.h5',
    'episode_BF3_1comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_slope.h5',
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]
PANEL = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
         'Directional_R2', 'Hs_error']

WIDE = (-4.0, 4.0)
NARROW = (-0.8, 0.8)


def _min_dk_for(krange, n, abs_radius):
    """MIN_DK that makes the shipped `int(MIN_DK/dk)+1` cell count give exactly
    `abs_radius` of absolute half-width. Returns (min_dk, cells, realised_radius)."""
    dk = (krange[1] - krange[0]) / (n - 1)
    c = int(round(abs_radius / dk))
    if c < 1:
        raise ValueError(f'abs_radius {abs_radius} < grid step {dk}: not representable')
    return (c - 0.5) * dk, c, c * dk


# cell -> dict(range, n_coarse or None=shipped-per-site, min_dk or None=shipped 0.15, minsep)
CELLS = {
    # anchor: shipped exactly (81/61/61, MIN_DK=0.15) == E1 cell A
    'A0':   dict(krange=WIDE,   n=None, abs_r=None, minsep=None),
    # 2x2  STEP x RADIUS, coverage held WIDE, all three sites uniform
    'U0':   dict(krange=WIDE,   n=81,  abs_r=0.200, minsep=None),
    'U1':   dict(krange=WIDE,   n=401, abs_r=0.200, minsep=None),
    'U2':   dict(krange=WIDE,   n=401, abs_r=0.100, minsep=None),
    'U3':   dict(krange=WIDE,   n=81,  abs_r=0.100, minsep=None),
    # COVERAGE alone (step 0.020 and abs radius 0.200 held vs U1)
    'U4':   dict(krange=NARROW, n=81,  abs_r=0.200, minsep=None),
    # MINSEP arms
    'U1M2': dict(krange=WIDE,   n=401, abs_r=0.200, minsep=0.02),
    'U1M5': dict(krange=WIDE,   n=401, abs_r=0.200, minsep=0.05),
    'U0M5': dict(krange=WIDE,   n=81,  abs_r=0.200, minsep=0.05),   # negative control
    'U4M5': dict(krange=NARROW, n=81,  abs_r=0.200, minsep=0.05),
}

# WS-P1, identical to E1 / tmp/s1s2_2026-08-05 arm S1-A0. sum(max_iters) == 900.
BASE_CFG = dict(init_method='v2_konly', use_dispersion=True, use_confidence=True,
                adaptive_n=True, tracking_window=3, prune_threshold=0.10, max_n=30,
                n_gaussians=[3, 8, 15], loss_thresholds=[0.1, 0.05, None],
                max_iters_per_stage=[200, 200, 500])


def cell_spec(cell):
    """Human-readable, fully resolved description of what the cell installs."""
    c = CELLS[cell]
    out = {'range': list(c['krange']), 'n_coarse': c['n'], 'minsep': c['minsep']}
    if c['abs_r'] is None:
        out['min_dk'] = 0.15
        out['abs_radius'] = 'shipped per-site'
    else:
        n = c['n']
        md, cells, r = _min_dk_for(c['krange'], n, c['abs_r'])
        out.update(min_dk=md, suppress_cells=cells, abs_radius=r,
                   step=(c['krange'][1] - c['krange'][0]) / (n - 1))
    return out


def apply_cell(cell):
    """Install the cell's factors by monkey-patch. Returns an undo callable."""
    import wavesplat.optimizer.initializer as I
    import wavesplat.optimizer.coarse_to_fine as C
    from wavesplat.model.spectral_gaussian import SpectralGaussianModel as M
    import torch

    spec = CELLS[cell]
    krange, n_force, abs_r, minsep = (spec['krange'], spec['n'],
                                      spec['abs_r'], spec['minsep'])

    orig_bf = I._beamform_coarse
    orig_mindk = I.MIN_DK
    orig_mode = I.PEAK_SELECT_MODE
    orig_v2 = I.initialize_from_frames_v2
    orig_add = M.add_gaussians

    # ---- STEP + COVERAGE -------------------------------------------------
    def bf(x, y, z, t, depth, kx_range=WIDE, ky_range=WIDE, n_kx=81, n_ky=81):
        nk = n_kx if n_force is None else n_force
        return orig_bf(x, y, z, t, depth, krange, krange, nk, nk)

    I._beamform_coarse = bf
    C._beamform_coarse = bf          # coarse_to_fine imported the symbol at module load

    # ---- RADIUS ----------------------------------------------------------
    if abs_r is not None:
        I.MIN_DK, _, _ = _min_dk_for(krange, n_force, abs_r)
    I.PEAK_SELECT_MODE = 'static_suppress'   # held at shipped for every E2 cell

    # ---- MINSEP ----------------------------------------------------------
    if minsep is not None:
        def v2_minsep(frames, n_gaussians, depth, *a, **kw):
            budget = int(min(4 * n_gaussians, 60))
            kx, ky, w, phi = orig_v2(frames, budget, depth, *a, **kw)
            kx = kx.numpy().astype(float); ky = ky.numpy().astype(float)
            w = w.numpy().astype(float); phi = phi.numpy().astype(float)
            keep = []
            for i in range(len(kx)):
                if len(keep) >= n_gaussians:
                    break
                ok = all(np.hypot(kx[i] - kx[j], ky[i] - ky[j]) >= minsep for j in keep)
                if ok:
                    keep.append(i)
            kk = np.array(keep, dtype=int)
            okx, oky = list(kx[kk]), list(ky[kk])
            ow, ophi = list(w[kk]), list(phi[kk])
            if len(okx) < n_gaussians:   # shipped padding law
                import math
                rng = np.random.RandomState(kw.get('seed', None))
                extra = n_gaussians - len(okx)
                okx.extend(rng.randn(extra).tolist())
                oky.extend(rng.randn(extra).tolist())
                ow.extend([I.PAD_W] * extra)
                ophi.extend(rng.uniform(0, 2 * math.pi, extra).tolist())
            f32 = torch.float32
            return (torch.tensor(okx[:n_gaussians], dtype=f32),
                    torch.tensor(oky[:n_gaussians], dtype=f32),
                    torch.tensor(ow[:n_gaussians], dtype=f32),
                    torch.tensor(ophi[:n_gaussians], dtype=f32))

        I.initialize_from_frames_v2 = v2_minsep

        def add_minsep(self, n_new, init_kx=None, init_ky=None,
                       init_w=None, init_phi=None):
            if init_kx is None or init_ky is None:
                return orig_add(self, n_new, init_kx, init_ky, init_w, init_phi)
            ex_kx = self.kx.data.detach().cpu().numpy().astype(float)
            ex_ky = self.ky.data.detach().cpu().numpy().astype(float)
            nkx = init_kx.detach().cpu().numpy().astype(float)
            nky = init_ky.detach().cpu().numpy().astype(float)
            keep = []
            for i in range(min(n_new, len(nkx))):
                if len(ex_kx) and np.min(np.hypot(nkx[i] - ex_kx,
                                                  nky[i] - ex_ky)) < minsep:
                    continue
                if any(np.hypot(nkx[i] - nkx[j], nky[i] - nky[j]) < minsep for j in keep):
                    continue
                keep.append(i)
            if not keep:
                return
            idx = torch.tensor(keep, dtype=torch.long, device=init_kx.device)
            sel = lambda v: None if v is None else v.index_select(0, idx.to(v.device))
            return orig_add(self, len(keep), sel(init_kx), sel(init_ky),
                            sel(init_w), sel(init_phi))

        M.add_gaussians = add_minsep

    def undo():
        I._beamform_coarse = orig_bf
        C._beamform_coarse = orig_bf
        I.MIN_DK = orig_mindk
        I.PEAK_SELECT_MODE = orig_mode
        I.initialize_from_frames_v2 = orig_v2
        M.add_gaussians = orig_add
    return undo


# ---- spectral matching, verbatim from tmp/e1_grid_2026-08-05/run_e1.py ----
def match_components(est_kx, est_ky, est_w, gt_kx, gt_ky, gt_A):
    N_est = len(est_kx); N_gt = len(gt_kx)
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
    matched_gt = {m[1] for m in matched}; matched_est = {m[0] for m in matched}
    unmatched_gt = [j for j in range(N_gt) if j not in matched_gt]
    unmatched_est = [i for i in range(N_est) if i not in matched_est]
    return matched, unmatched_gt, unmatched_est


def spectral_accuracy(est_params, gt_components):
    gt_A = gt_components[:, 0]; gt_omega = gt_components[:, 1]
    gt_kx = gt_components[:, 2]; gt_ky = gt_components[:, 3]
    est_kx = np.asarray(est_params['kx']); est_ky = np.asarray(est_params['ky'])
    est_w = np.abs(np.asarray(est_params['w'])); est_omega = np.asarray(est_params['omega'])
    matched, unmatched_gt, unmatched_est = match_components(
        est_kx, est_ky, est_w, gt_kx, gt_ky, gt_A)
    result = {'n_gt': int(len(gt_A)), 'n_est': int(len(est_kx)),
              'n_matched': int(len(matched)),
              'detection_rate': len(matched) / max(len(gt_A), 1)}
    if matched:
        k_mag_errors = []; omega_errors = []; amp_errors = []; direction_errors = []
        for est_idx, gt_idx, cost in matched:
            gt_kmag = np.sqrt(gt_kx[gt_idx] ** 2 + gt_ky[gt_idx] ** 2)
            est_kmag = np.sqrt(est_kx[est_idx] ** 2 + est_ky[est_idx] ** 2)
            k_mag_errors.append((est_kmag - gt_kmag) / max(gt_kmag, 1e-6))
            omega_errors.append((est_omega[est_idx] - gt_omega[gt_idx]) / max(gt_omega[gt_idx], 1e-6))
            amp_errors.append((est_w[est_idx] - gt_A[gt_idx]) / max(gt_A[gt_idx], 1e-6))
            gt_dir = np.degrees(np.arctan2(gt_ky[gt_idx], gt_kx[gt_idx]))
            est_dir = np.degrees(np.arctan2(est_ky[est_idx], est_kx[est_idx]))
            d_dir = abs(est_dir - gt_dir); d_dir = 360 - d_dir if d_dir > 180 else d_dir
            direction_errors.append(d_dir)
        result['k_mag_rel_error_mean'] = float(np.mean(np.abs(k_mag_errors)))
        result['omega_rel_error_mean'] = float(np.mean(np.abs(omega_errors)))
        result['amp_rel_error_mean'] = float(np.mean(np.abs(amp_errors)))
        result['direction_error_deg_mean'] = float(np.mean(direction_errors))
        matched_gt_indices = [m[1] for m in matched]
        matched_energy = float(np.sum(gt_A[matched_gt_indices] ** 2))
        total_energy = float(np.sum(gt_A ** 2))
        result['energy_detection_rate'] = matched_energy / max(total_energy, 1e-12)
    else:
        for k in ('k_mag_rel_error_mean', 'omega_rel_error_mean',
                  'amp_rel_error_mean', 'direction_error_deg_mean'):
            result[k] = float('nan')
        result['energy_detection_rate'] = 0.0
    return result


def nn_stats(kx, ky, prefix=''):
    """Nearest-neighbour spacing distribution of a placed atom set -- the direct
    stacking indicator (PREREG section 5.1)."""
    kx = np.asarray(kx, dtype=float); ky = np.asarray(ky, dtype=float)
    out = {}
    if len(kx) < 2:
        for s in ('min', 'p10', 'p25', 'median'):
            out[prefix + 'nn_' + s] = float('nan')
        out[prefix + 'n_pairs_lt_002'] = 0
        out[prefix + 'n_pairs_lt_005'] = 0
        return out
    D = np.hypot(kx[:, None] - kx[None, :], ky[:, None] - ky[None, :])
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    out[prefix + 'nn_min'] = float(np.min(nn))
    out[prefix + 'nn_p10'] = float(np.percentile(nn, 10))
    out[prefix + 'nn_p25'] = float(np.percentile(nn, 25))
    out[prefix + 'nn_median'] = float(np.median(nn))
    out[prefix + 'n_pairs_lt_002'] = int(np.sum(nn < 0.02))
    out[prefix + 'n_pairs_lt_005'] = int(np.sum(nn < 0.05))
    return out


def build_jobs(cells=None, episodes=None, seeds=None):
    cells = cells or list(CELLS)
    episodes = episodes or EPISODES
    seeds = seeds or SEEDS
    return [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in cells for ep in episodes for s in seeds]


def run_one(job):
    sys.path.insert(0, REPO)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from wavesplat.run import run_estimation
    from wavesplat.data_loader import EpisodeDataLoader
    import wavesplat.optimizer.initializer as _init

    base = {k: job[k] for k in ('job_id', 'cell', 'episode', 'seed')}
    t0 = time.time()
    path = os.path.join(DATA, job['episode'])
    undo = None
    try:
        cfg = dict(BASE_CFG); cfg['seed'] = job['seed']
        gt_comp = np.asarray(EpisodeDataLoader(path).wave_components)
        _init.ORACLE_COMPONENTS = None
        undo = apply_cell(job['cell'])
        try:
            r = run_estimation(path, cfg)
        finally:
            undo(); undo = None

        frames = r['frame_results']
        avg = {k: float(np.nanmean([f[k] for f in frames if k in f])) for k in PANEL}
        ru = [f['RMSE_unobserved'] for f in frames
              if 'RMSE_unobserved' in f and np.isfinite(f.get('RMSE_unobserved', np.nan))]
        avg_ru = float(np.mean(ru)) if ru else float('nan')
        try:
            spec = spectral_accuracy(r['spectrum_params'], gt_comp)
        except Exception as se:
            spec = {'spec_error': repr(se)}
        try:
            fnn = nn_stats(r['spectrum_params']['kx'], r['spectrum_params']['ky'],
                           prefix='final_')
        except Exception as ne:
            fnn = {'final_nn_error': repr(ne)}
        return {**base, **avg, 'avg_RMSE_unobserved': avg_ru,
                'elapsed_ms': float(np.mean([f['elapsed_ms'] for f in frames]))
                if frames else float('nan'),
                'N': r.get('n_gaussians', -1),
                'init_nrmse': float(r.get('init_nrmse', float('nan'))),
                'spec': spec, **fnn,
                'cell_cfg': cell_spec(job['cell']),
                'wall_s': time.time() - t0}
    except Exception as e:
        import traceback
        if undo is not None:
            undo()
        return {**base, 'error': repr(e), 'tb': traceback.format_exc()[-1500:],
                'wall_s': time.time() - t0}


def _init_worker():
    sys.path.insert(0, REPO)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')


def main():
    import multiprocessing as mp
    cells = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    eps = None
    if len(sys.argv) > 2 and sys.argv[2] != '-':
        eps = [e for e in EPISODES if any(tag in e for tag in sys.argv[2].split(','))]
    n_workers = int(os.environ.get('WORKERS', '12'))
    jobs = build_jobs(cells, eps)
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
    # longest first: the n_coarse=401 cells dominate the makespan
    todo.sort(key=lambda j: -(CELLS[j['cell']]['n'] or 81))
    print(f'total {len(jobs)} | done {len(done)} | todo {len(todo)} | workers {n_workers}',
          flush=True)
    if not todo:
        print('nothing to do'); return
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(n_workers, initializer=_init_worker) as pool, open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            el = time.time() - t0
            tag = rec.get('nRMSE', rec.get('error', '?'))
            if isinstance(tag, float):
                tag = f'{tag:.4f}'
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<52s} {tag} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
