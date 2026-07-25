"""Timing probe: one finemesh episode per method, single process.
Also verifies WS frame_results carry RMSE_unobserved (task 4 feasibility)."""
import os, sys, time, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
EP = 'episode_BF3_5comp_seed0_uniform.h5'
h5 = os.path.join(DATA, EP)

from wavesplat.run import run_estimation
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.baselines.etkf_grid import ETKFGrid
from wavesplat.baselines.letkf_grid import LETKFGrid
from wavesplat.baselines.gpr_baseline import GPRBaseline
from wavesplat.baselines.gpr_dispersion import GPRDispersion
from wavesplat.evaluation.metrics import compute_all_metrics
from scipy.interpolate import RegularGridInterpolator


def grid_baseline(cls, kw):
    np.random.seed(42)
    loader = EpisodeDataLoader(h5)
    gx = loader.gt_grid_x.cpu().numpy(); gy = loader.gt_grid_y.cpu().numpy()
    egx = np.linspace(gx[0], gx[-1], 40); egy = np.linspace(gy[0], gy[-1], 30)
    b = cls(egx, egy, depth=loader.depth, **kw)
    fm = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy(); y = frame.y_obs.cpu().numpy(); z = frame.z_obs.cpu().numpy()
        el = b.update(x, y, z)
        pred = b.predict_grid()
        interp = RegularGridInterpolator((egy, egx), pred, method='linear', bounds_error=False, fill_value=0.0)
        Xg, Yg = np.meshgrid(gx, gy)
        pf = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(len(gy), len(gx))
        m = compute_all_metrics(pf, frame.gt_field.cpu().numpy(), gx, gy, loader.Hs_true,
                                obs_x=x if len(x) else None, obs_y=y if len(y) else None)
        m['elapsed_ms'] = el; fm.append(m)
    avg = {k: float(np.nanmean([m.get(k, np.nan) for m in fm])) for k in fm[0]}
    return avg


def gpr_baseline(cls, kw):
    np.random.seed(42)
    loader = EpisodeDataLoader(h5)
    gx = loader.gt_grid_x.cpu().numpy(); gy = loader.gt_grid_y.cpu().numpy()
    g = cls(**kw); fm = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy(); y = frame.y_obs.cpu().numpy(); z = frame.z_obs.cpu().numpy()
        el = g.update(x, y, z)
        pred = g.predict_grid(gx, gy)
        m = compute_all_metrics(pred, frame.gt_field.cpu().numpy(), gx, gy, loader.Hs_true,
                                obs_x=x if len(x) else None, obs_y=y if len(y) else None)
        m['elapsed_ms'] = el; fm.append(m)
    avg = {k: float(np.nanmean([m.get(k, np.nan) for m in fm])) for k in fm[0]}
    return avg


t = time.time()
r = run_estimation(h5, {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
                        'adaptive_n': True, 'tracking_window': 3, 'seed': 42})
fr = r['frame_results']
ru = float(np.nanmean([f.get('RMSE_unobserved', np.nan) for f in fr]))
print(f'WS         wall={time.time()-t:6.1f}s nRMSE={r["avg_nRMSE"]:.4f} '
      f'RMSE_unobs={ru:.4f} has_key={"RMSE_unobserved" in fr[0]}', flush=True)

for name, cls, kw in [
    ('ETKF', ETKFGrid, dict(n_ensemble=100, mult_inflation=1.0, add_inflation=0.005)),
    ('LETKF', LETKFGrid, dict(n_ensemble=100, mult_inflation=1.0, add_inflation=0.005, loc_radius=50.0)),
]:
    t = time.time(); a = grid_baseline(cls, kw)
    print(f'{name:10s} wall={time.time()-t:6.1f}s nRMSE={a["nRMSE"]:.4f} '
          f'RMSE_unobs={a.get("RMSE_unobserved", float("nan")):.4f}', flush=True)

for name, cls, kw in [
    ('GPR-1', GPRBaseline, dict(kernel_type='matern', length_scale=20.0, noise_sigma=0.02, window_frames=1)),
    ('GPR-disp', GPRDispersion, dict(length_scale=20.0, noise_sigma=0.02, window_frames=3, depth=50.0)),
]:
    t = time.time(); a = gpr_baseline(cls, kw)
    print(f'{name:10s} wall={time.time()-t:6.1f}s nRMSE={a["nRMSE"]:.4f} '
          f'RMSE_unobs={a.get("RMSE_unobserved", float("nan")):.4f}', flush=True)
