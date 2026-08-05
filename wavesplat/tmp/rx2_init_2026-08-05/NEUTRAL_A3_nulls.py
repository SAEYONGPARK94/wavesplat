"""NEUTRAL A3 — what null does init_nrmse fail to beat?

init_nrmse = RMSE(pred_field, gt_field) / Hs_true, evaluated on the GT grid at the LAST
init frame.  The cheapest null is the ZERO field (predict flat sea): its nRMSE is
std(gt_field)/Hs_true, which for a linear Gaussian sea is ~1/4.  Measure it directly on
the four Rx-2 episodes at the exact frame run.py uses.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(REPO, '..')))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')

from wavesplat.data_loader import EpisodeDataLoader

EPISODES = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']

rows = []
for ep in EPISODES:
    loader = EpisodeDataLoader(os.path.join(DATA, ep), device='cpu')
    frames = loader.get_init_window(tp_factor=2.0)
    last = frames[-1]
    gt = last.gt_field.cpu().numpy()
    Hs = loader.Hs_true
    zero_nrmse = float(np.sqrt(np.mean(gt ** 2)) / max(Hs, 1e-6))
    # mean-field null (predict the frame mean) and a 'best constant' null
    const_nrmse = float(np.sqrt(np.mean((gt - gt.mean()) ** 2)) / max(Hs, 1e-6))
    # tracking-phase zero null (mean over all tracking frames), for comparison to nRMSE
    tf = loader.get_tracking_frames(tp_factor=2.0)
    zs = [float(np.sqrt(np.mean(f.gt_field.cpu().numpy() ** 2)) / max(Hs, 1e-6)) for f in tf]
    rows.append(dict(episode=ep, Hs_true=float(Hs), Tp_true=float(loader.Tp_true),
                     n_init_frames=len(frames), t_last_init=float(last.t),
                     zero_null_init_nrmse=zero_nrmse, const_null_init_nrmse=const_nrmse,
                     zero_null_tracking_nRMSE_mean=float(np.mean(zs)),
                     gt_std_over_Hs=float(gt.std() / Hs)))
    print(json.dumps(rows[-1]), flush=True)

json.dump(rows, open(os.path.join(HERE, 'NEUTRAL_A3_nulls.json'), 'w'), indent=1)
