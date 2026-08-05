"""Cheap check: on the fine-mesh BF2-1comp episode, does the ALREADY-IMPLEMENTED
'relax' init_method (continuous-k single-line refinement, same family as the closed
A-7v2 hybrid) do better than the paper's default 'v2_konly'? 3 estimator seeds,
fine mesh only (old mesh already shown hopeless above)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
from wavesplat.run import run_estimation

EPISODE = 'data_finemesh_2026-07-23/episode_BF2_1comp_seed0_uniform.h5'
SEEDS = [42, 1, 2]

for method in ['v2_konly', 'relax']:
    print(f'\n=== init_method={method} ===')
    nrmses, rhos = [], []
    for s in SEEDS:
        cfg = dict(init_method=method, adaptive_n=True, seed=s)
        res = run_estimation(EPISODE, config=cfg)
        nrmse = res['avg_nRMSE']
        rho = float(np.mean([fr['Pearson_rho'] for fr in res['frame_results']]))
        nrmses.append(nrmse); rhos.append(rho)
        print(f'  seed={s:3d}  nRMSE={nrmse:.4f}  rho={rho:.4f}  N={res["n_gaussians"]}')
    print(f'  --- mean nRMSE={np.mean(nrmses):.4f}  mean rho={np.mean(rhos):.4f}')
