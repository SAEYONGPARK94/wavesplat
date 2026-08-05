"""Fast disambiguating probe: does ETKF's coarse-mesh 'winning' hyperparameter
config (grid=40x30, Ne=50/100, mi=1.0, ai=0.005) still win on regenerated
FINE-mesh data, or does the ranking change?  Reuses the coarse/fine file pairs
already produced by tmp/mesh_headline_subset.py (no new generation)."""
import sys, os, time, glob, json
sys.path.insert(0, '.')
import numpy as np
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.baselines.etkf_grid import ETKFGrid
from wavesplat.evaluation.metrics import compute_all_metrics
from scipy.interpolate import RegularGridInterpolator

def run_etkf_one(h5_path, grid_nx, grid_ny, n_ens, mult_inf, add_inf):
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gx = loader.gt_grid_x.cpu().numpy(); gy = loader.gt_grid_y.cpu().numpy()
    egx = np.linspace(gx[0], gx[-1], grid_nx); egy = np.linspace(gy[0], gy[-1], grid_ny)
    etkf = ETKFGrid(egx, egy, depth=loader.depth, n_ensemble=n_ens,
                    mult_inflation=mult_inf, add_inflation=add_inf)
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

pairs = {
    'BF2_1c_s0': ('tmp/mesh_subset_ep/episode_BF2_1comp_seed0_uniform_BF2_1c_s0_uniform_coarse.h5',
                  'tmp/mesh_subset_ep/episode_BF2_1comp_seed0_uniform_BF2_1c_s0_uniform_fine.h5'),
    'BF3_1c_s0': ('tmp/mesh_subset_ep/episode_BF3_1comp_seed0_uniform_BF3_1c_s0_uniform_coarse.h5',
                  'tmp/mesh_subset_ep/episode_BF3_1comp_seed0_uniform_BF3_1c_s0_uniform_fine.h5'),
}

CONFIGS = [
    (40, 30, 50, 1.0, 0.005),    # shipped run_p1_comparison.py's actual config
    (40, 30, 100, 1.0, 0.005),   # sweep's stated best (on OLD data)
    (80, 60, 50, 1.0, 0.005),
    (80, 60, 100, 1.0, 0.005),
    (40, 30, 50, 1.02, 0.005),
    (20, 15, 50, 1.0, 0.005),    # coarser than shipped -- does even-coarser help more on fine truth?
]

RESULT = 'tmp/etkf_retune_probe_results.json'
results = json.load(open(RESULT)) if os.path.exists(RESULT) else {}

for tag, (coarse_fp, fine_fp) in pairs.items():
    for (gnx, gny, ne, mi, ai) in CONFIGS:
        key = f"{tag}_g{gnx}x{gny}_ne{ne}_mi{mi}_ai{ai}"
        if key in results:
            print('skip', key); continue
        t0 = time.time()
        nc = run_etkf_one(coarse_fp, gnx, gny, ne, mi, ai)
        nf = run_etkf_one(fine_fp, gnx, gny, ne, mi, ai)
        results[key] = dict(coarse=nc, fine=nf, delta_pct=100*(nf-nc)/nc)
        json.dump(results, open(RESULT, 'w'), indent=2)
        print(f"{key:45s} coarse={nc:.4f} fine={nf:.4f}  delta={100*(nf-nc)/nc:+.1f}%  [{time.time()-t0:.1f}s]", flush=True)

print("\n=== per-tag: best fine-mesh config ===")
for tag in pairs:
    rows = [(k,v) for k,v in results.items() if k.startswith(tag+'_')]
    rows.sort(key=lambda kv: kv[1]['fine'])
    print(f"\n{tag}:")
    for k,v in rows:
        print(f"  {k:45s} coarse={v['coarse']:.4f} fine={v['fine']:.4f} delta={v['delta_pct']:+.1f}%")
