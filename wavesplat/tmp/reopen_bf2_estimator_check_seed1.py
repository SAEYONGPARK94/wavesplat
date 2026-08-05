"""Same check as reopen_bf2_estimator_check.py but on a DIFFERENT data seed
(episode_BF2_1comp_seed1) to confirm the seed0 result is not a fluke of one
physical realization. 3 estimator seeds x 2 meshes (kept modest)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.run import run_estimation

EPISODE = 'episode_BF2_1comp_seed1_uniform.h5'
SEEDS = [42, 1, 2]
BASE_CFG = dict(init_method='v2_konly', adaptive_n=True)

for tag, root in [('OLD', 'data'), ('NEW', 'data_finemesh_2026-07-23')]:
    path = os.path.join(root, EPISODE)
    print(f'\n=== {tag} : {path} ===')
    nrmses, rhos = [], []
    for s in SEEDS:
        cfg = dict(BASE_CFG); cfg['seed'] = s
        res = run_estimation(path, config=cfg)
        nrmse = res['avg_nRMSE']
        rho = float(np.mean([fr['Pearson_rho'] for fr in res['frame_results']]))
        nrmses.append(nrmse); rhos.append(rho)
        print(f'  seed={s:3d}  nRMSE={nrmse:.4f}  rho={rho:.4f}  N={res["n_gaussians"]}')
    print(f'  --- mean nRMSE={np.mean(nrmses):.4f}  mean rho={np.mean(rhos):.4f}')
