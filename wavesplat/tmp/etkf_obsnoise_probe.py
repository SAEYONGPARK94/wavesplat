"""Quick follow-up: is BF3_1c's fine-mesh ETKF regression fixable by obs_noise
(not swept by the repo's existing 24-config sweep, which only varies
grid/Ne/mult_inflation/add_inflation)?  Also try disabling spectral damping."""
import sys, time, json, os
sys.path.insert(0, '.')
import numpy as np
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.baselines.etkf_grid import ETKFGrid
from wavesplat.evaluation.metrics import compute_all_metrics
from scipy.interpolate import RegularGridInterpolator

def run_etkf_one(h5_path, grid_nx, grid_ny, n_ens, mult_inf, add_inf, obs_noise, max_obs):
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gx = loader.gt_grid_x.cpu().numpy(); gy = loader.gt_grid_y.cpu().numpy()
    egx = np.linspace(gx[0], gx[-1], grid_nx); egy = np.linspace(gy[0], gy[-1], grid_ny)
    etkf = ETKFGrid(egx, egy, depth=loader.depth, n_ensemble=n_ens,
                    mult_inflation=mult_inf, add_inflation=add_inf,
                    obs_noise=obs_noise, max_obs=max_obs)
    nrmses = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy(); y = frame.y_obs.cpu().numpy(); z = frame.z_obs.cpu().numpy()
        etkf.update(x, y, z)
        pred_grid = etkf.predict_grid()
        interp = RegularGridInterpolator((egy, egx), pred_grid, method='linear',
                                         bounds_error=False, fill_value=0.0)
        Xg, Yg = np.meshgrid(gx, gy)
        pred = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(len(gy), len(gx))
        gt = frame.gt_field.cpu().numpy()
        m = compute_all_metrics(pred, gt, gx, gy, loader.Hs_true,
                                obs_x=x if len(x) else None, obs_y=y if len(y) else None)
        nrmses.append(m['nRMSE'])
    return float(np.mean(nrmses))

coarse_fp = 'tmp/mesh_subset_ep/episode_BF3_1comp_seed0_uniform_BF3_1c_s0_uniform_coarse.h5'
fine_fp   = 'tmp/mesh_subset_ep/episode_BF3_1comp_seed0_uniform_BF3_1c_s0_uniform_fine.h5'

CONFIGS = [
    # (grid_nx, grid_ny, ne, mi, ai, obs_noise, max_obs, label)
    (40, 30, 50, 1.0, 0.005, 0.05, 40, 'shipped'),
    (40, 30, 50, 1.0, 0.005, 0.10, 40, 'obs_noise x2'),
    (40, 30, 50, 1.0, 0.005, 0.20, 40, 'obs_noise x4'),
    (40, 30, 50, 1.0, 0.005, 0.02, 40, 'obs_noise /2.5 (nominal sensor)'),
    (40, 30, 50, 1.0, 0.005, 0.05, 74, 'max_obs=74 (no thinning, matches avg pts/frame)'),
]

RESULT = 'tmp/etkf_obsnoise_probe_results.json'
results = json.load(open(RESULT)) if os.path.exists(RESULT) else {}
for (gnx, gny, ne, mi, ai, on, mo, label) in CONFIGS:
    key = label
    if key in results:
        print('skip', key); continue
    t0 = time.time()
    nc = run_etkf_one(coarse_fp, gnx, gny, ne, mi, ai, on, mo)
    nf = run_etkf_one(fine_fp, gnx, gny, ne, mi, ai, on, mo)
    results[key] = dict(coarse=nc, fine=nf, delta_pct=100*(nf-nc)/nc)
    json.dump(results, open(RESULT, 'w'), indent=2)
    print(f"{label:45s} coarse={nc:.4f} fine={nf:.4f}  delta={100*(nf-nc)/nc:+.1f}%  [{time.time()-t0:.1f}s]", flush=True)
