"""Direct check: does the ACTUAL WaveSplat estimator (not an oracle-k probe, not the
mmse_gpu bank diagnostic) improve on BF2 1-comp when run on the paired fine mesh, using
the same config as the paper's 12-episode table (init_method=v2_konly, adaptive_n=True)?

Multi-seed (estimator RNG seed, not data seed) because of documented run-to-run CV up to
53% (tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md). Modest: 5 seeds x 2 meshes x 1 episode.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.run import run_estimation

EPISODE = 'episode_BF2_1comp_seed0_uniform.h5'
SEEDS = [42, 1, 2, 3, 4]

BASE_CFG = dict(init_method='v2_konly', adaptive_n=True)

for tag, root in [('OLD (data/)', 'data'), ('NEW (data_finemesh_2026-07-23/)',
                                             'data_finemesh_2026-07-23')]:
    path = os.path.join(root, EPISODE)
    print(f'\n=== {tag} : {path} ===')
    nrmses, rhos, Ns = [], [], []
    for s in SEEDS:
        cfg = dict(BASE_CFG)
        cfg['seed'] = s
        res = run_estimation(path, config=cfg)
        nrmse = res['avg_nRMSE']
        rho = float(np.mean([fr['Pearson_rho'] for fr in res['frame_results']]))
        N = res['n_gaussians']
        nrmses.append(nrmse); rhos.append(rho); Ns.append(N)
        print(f'  seed={s:3d}  nRMSE={nrmse:.4f}  rho={rho:.4f}  N={N}')
    nrmses = np.array(nrmses, float)
    rhos = np.array(rhos, float)
    print(f'  --- mean nRMSE={nrmses.mean():.4f} sd={nrmses.std():.4f}   '
          f'mean rho={rhos.mean():.4f} sd={rhos.std():.4f}')
