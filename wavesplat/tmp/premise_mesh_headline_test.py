"""
premise_mesh_headline_test.py — Reviewer C (premises audit), 2026-07-23

Question: the shipped data/*.h5 episodes (coarse raycast mesh, 40x30,
dx=6.15m/dy=6.90m) were shown by the E-2/E-3 audit (docs/E2_MESH_AUDIT_prereg_
2026-07-13.md) to be sub-Nyquist / degraded for the wavelengths in every
in-scope (BF2-BF5) sea state -- confirmed independently below via
wavesim.mesh_adequacy.check() run directly against the shipped files.

E-3 showed this inflates OPERATOR ERROR by ~8x (0.048 -> 0.006 m) for one
specific hand-vs-learned ablation, but the *direction* of that specific
delta barely moved (-0.045 -> -0.032, ~1/3 shift).

This script asks the question that matters for the actual OE-paper headline:
does the reported WaveSplat-vs-ETKF margin (the number in the abstract,
0.223 vs 0.319) survive mesh refinement, or is it partly measuring which
estimator handles raycast-facet noise better rather than which estimator
handles ocean wave physics better?

Method: regenerate two paired episodes (same seed -> identical wave
components) at coarse (40x30, shipped default) and fine (180x150, E-3's
refinement) mesh, run WaveSplat-P1 and ETKF-medium (same code path as
scripts/run_p1_comparison.py) on both, and compare the WS-ETKF gap.

Targets:
  - BF2_1comp seed0: INADEQUATE (100% var under Nyquist) -- the paper's own
    quoted WORST CASE for WaveSplat (nRMSE=0.315, wavesplat.tex line 1555).
  - BF4_5comp seed0: DEGRADED (21% var under) -- representative mid-complexity
    episode from the main 12-episode comparison table.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesim.config import WaveSimConfig
from wavesim.pipeline import run_episode

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.run import run_estimation
from wavesplat.baselines.etkf_grid import ETKFGrid
from wavesplat.evaluation.metrics import compute_all_metrics
import numpy as np
from scipy.interpolate import RegularGridInterpolator

OUT_DIR = 'tmp/premise_mesh_ep'
os.makedirs(OUT_DIR, exist_ok=True)


def gen_pair(sea_state, n_components, seed, tag):
    cfg_coarse = WaveSimConfig()  # nx=40, ny=30 -- the shipped default
    cfg_fine = WaveSimConfig()
    cfg_fine.mesh.nx = 180
    cfg_fine.mesh.ny = 150

    t0 = time.time()
    p_coarse = run_episode(sea_state=sea_state, n_components=n_components, n_pts=50,
                            seed=seed, raydrop_model='uniform', duration=30.0, dt=0.1,
                            depth=50.0, output_dir=OUT_DIR, verbose=False, cfg=cfg_coarse,
                            use_gpu_raycast=True)
    os.rename(p_coarse, p_coarse.replace('.h5', f'_{tag}coarse.h5'))
    p_coarse = p_coarse.replace('.h5', f'_{tag}coarse.h5')
    print(f'  coarse gen: {time.time()-t0:.1f}s -> {p_coarse}')

    t0 = time.time()
    p_fine = run_episode(sea_state=sea_state, n_components=n_components, n_pts=50,
                          seed=seed, raydrop_model='uniform', duration=30.0, dt=0.1,
                          depth=50.0, output_dir=OUT_DIR, verbose=False, cfg=cfg_fine,
                          use_gpu_raycast=True)
    os.rename(p_fine, p_fine.replace('.h5', f'_{tag}fine.h5'))
    p_fine = p_fine.replace('.h5', f'_{tag}fine.h5')
    print(f'  fine gen:   {time.time()-t0:.1f}s -> {p_fine}')
    return p_coarse, p_fine


def run_ws(h5_path):
    config = {
        'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
        'adaptive_n': True, 'tracking_window': 3, 'seed': 42,
    }
    result = run_estimation(h5_path, config=config)
    return float(result['avg_nRMSE'])


def run_etkf(h5_path, grid_resolution='medium'):
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gt_grid_x = loader.gt_grid_x.cpu().numpy()
    gt_grid_y = loader.gt_grid_y.cpu().numpy()
    res_map = {'medium': (40, 30), 'fine': (80, 60)}
    etkf_nx, etkf_ny = res_map[grid_resolution]
    etkf_gx = np.linspace(gt_grid_x[0], gt_grid_x[-1], etkf_nx)
    etkf_gy = np.linspace(gt_grid_y[0], gt_grid_y[-1], etkf_ny)
    etkf = ETKFGrid(etkf_gx, etkf_gy, depth=loader.depth, n_ensemble=50)
    frame_results = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy(); y = frame.y_obs.cpu().numpy(); z = frame.z_obs.cpu().numpy()
        etkf.update(x, y, z)
        etkf_pred = etkf.predict_grid()
        interp = RegularGridInterpolator((etkf_gy, etkf_gx), etkf_pred,
                                          method='linear', bounds_error=False, fill_value=0.0)
        Xg, Yg = np.meshgrid(gt_grid_x, gt_grid_y)
        pred = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(len(gt_grid_y), len(gt_grid_x))
        gt = frame.gt_field.cpu().numpy()
        metrics = compute_all_metrics(pred, gt, gt_grid_x, gt_grid_y, loader.Hs_true,
                                       obs_x=x if len(x) > 0 else None, obs_y=y if len(y) > 0 else None)
        frame_results.append(metrics)
    return float(np.mean([r['nRMSE'] for r in frame_results])) if frame_results else float('nan')


import sys as _sys
SCOPE = _sys.argv[1] if len(_sys.argv) > 1 else 'bf2'
TARGETS = {
    'bf2': [('BF2', 1, 0, 'bf2_1c_s0')],
    'bf4': [('BF4', 5, 0, 'bf4_5c_s0')],
}[SCOPE]

results = {}
for sea_state, ncomp, seed, tag in TARGETS:
    print(f'=== {tag} ===')
    p_coarse, p_fine = gen_pair(sea_state, ncomp, seed, tag)

    ws_coarse = run_ws(p_coarse)
    ws_fine = run_ws(p_fine)
    etkf_coarse = run_etkf(p_coarse)
    etkf_fine = run_etkf(p_fine)

    r = dict(ws_coarse=ws_coarse, ws_fine=ws_fine,
             etkf_coarse=etkf_coarse, etkf_fine=etkf_fine,
             gap_coarse=etkf_coarse - ws_coarse, gap_fine=etkf_fine - ws_fine,
             gap_pct_coarse=100*(etkf_coarse - ws_coarse)/etkf_coarse,
             gap_pct_fine=100*(etkf_fine - ws_fine)/etkf_fine)
    results[tag] = r
    print(f'  WS   coarse={ws_coarse:.4f}  fine={ws_fine:.4f}')
    print(f'  ETKF coarse={etkf_coarse:.4f}  fine={etkf_fine:.4f}')
    print(f'  gap(ETKF-WS) coarse={r["gap_coarse"]:.4f} ({r["gap_pct_coarse"]:.1f}%)  '
          f'fine={r["gap_fine"]:.4f} ({r["gap_pct_fine"]:.1f}%)')
    print()

with open('tmp/premise_mesh_headline_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved tmp/premise_mesh_headline_test_results.json')
