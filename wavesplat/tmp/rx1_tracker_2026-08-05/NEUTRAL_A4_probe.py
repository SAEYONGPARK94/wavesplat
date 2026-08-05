"""A4 diagnostic probe (NOT the Rx-1 sweep): knob independence + loss decomposition.

Runs ONE episode x ONE seed for a handful of configs, with init_method EXPLICIT.
Purpose: measure (i) whether K1/K2 move Phase-1 (init_nrmse) as well as Phase-2,
(ii) the confidence-weight factor behind K3, (iii) the regularisation share of the
tracking loss.  Does not touch paper/ or any initializer file.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
EP = 'episode_BF4_15comp_seed0_uniform.h5'   # the episode debate G quoted (0.1425 -> 0.0167)
SEED = 42
PATH = os.path.join(DATA, EP)

from wavesplat.run import run_estimation
from wavesplat.data_loader import EpisodeDataLoader
import wavesplat.optimizer.initializer as _init
import wavesplat.optimizer.sequential_update as SU

gt_comp = np.asarray(EpisodeDataLoader(PATH).wave_components)
N_GT = 15

BASE = dict(use_dispersion=True, use_confidence=True, adaptive_n=True,
            tracking_window=3, prune_threshold=0.10, max_n=30,
            init_method='oracle', n_gaussians=[N_GT], loss_thresholds=[None],
            max_iters_per_stage=[900], init_k_lr_scale=0.0, seed=SEED)

CELLS = {
    'X-BASE            (lam1=1e-3, lamS=1e-4)': {},
    'K1  lambda_l1=0   (BOTH phases)          ': dict(lambda_l1=0.0),
    'K2  lambda_smooth=0 (BOTH phases)        ': dict(lambda_smooth=0.0),
    'K1+K2 both lambdas 0 (BOTH phases)       ': dict(lambda_l1=0.0, lambda_smooth=0.0),
    'X-OFF tracking_lr=0                      ': dict(tracking_lr=0.0),
}

out = {}
for name, over in CELLS.items():
    cfg = dict(BASE); cfg.update(over)
    _init.ORACLE_COMPONENTS = gt_comp
    t0 = time.time()
    try:
        r = run_estimation(PATH, cfg)
    finally:
        _init.ORACLE_COMPONENTS = None
    fr = r['frame_results']
    rec = dict(
        init_nrmse=float(r['init_nrmse']),
        nRMSE=float(np.nanmean([f['nRMSE'] for f in fr])),
        rho=float(np.nanmean([f['Pearson_rho'] for f in fr])),
        VR=float(np.nanmean([f['Variance_Ratio'] for f in fr])),
        Hs_error=float(np.nanmean([f['Hs_error'] for f in fr])),
        skill=float(np.nanmean([f['Skill_Score'] for f in fr])),
        N=r.get('n_gaussians'), wall=time.time() - t0,
        sum_abs_w=float(np.sum(np.abs(r['spectrum_params']['w']))),
    )
    out[name] = rec
    print(f'{name}  init {rec["init_nrmse"]:.4f}  nRMSE {rec["nRMSE"]:.4f}  '
          f'rho {rec["rho"]:.4f}  VR {rec["VR"]:.4f}  sum|w| {rec["sum_abs_w"]:.4f}  '
          f'({rec["wall"]:.0f}s)', flush=True)

print()
print('=' * 84)
print('A4.2  confidence-weight factor behind K3  (mean(conf) vs the /n_obs divisor)')
print('=' * 84)
loader = EpisodeDataLoader(PATH)
iw = loader.get_init_window(tp_factor=2.0)
tf = loader.get_tracking_frames(tp_factor=2.0)
ci = np.concatenate([f.confidence.cpu().numpy() for f in iw if len(f.confidence)])
ct = np.concatenate([f.confidence.cpu().numpy() for f in tf if len(f.confidence)])
print(f'  init window : n_obs {len(ci)}  mean(conf) {ci.mean():.4f}  '
      f'min {ci.min():.4f}  max {ci.max():.4f}')
print(f'  tracking    : n_obs {len(ct)}  mean(conf) {ct.mean():.4f}  '
      f'min {ct.min():.4f}  max {ct.max():.4f}')
print(f'  K3 (/n_obs -> /sum(weights)) multiplies the DATA term by 1/mean(conf) = '
      f'{1/ct.mean():.4f} in tracking.')
print(f'  Under Adam (scale-invariant to a global loss factor) that is ALGEBRAICALLY the')
print(f'  same as multiplying BOTH lambdas by mean(conf) = {ct.mean():.4f}.')
print(f'  => K3 is a {100*(1-ct.mean()):.1f}% partial K1+K2, not an orthogonal lever;')
print(f'     and K3 is EXACTLY inert whenever lambda_l1 = lambda_smooth = 0.')
print(f'  n_frames: init {len(iw)}  tracking {len(tf)}  (arm-independent: fixed by')
print(f'     loader.get_tracking_frames(tp_factor=2.0), so every arm is scored on the')
print(f'     SAME {len(tf)} timepoints -- no definitional difference between ON and OFF.)')

json.dump({'cells': out, 'mean_conf_track': float(ct.mean()),
           'mean_conf_init': float(ci.mean()),
           'n_frames_init': len(iw), 'n_frames_track': len(tf)},
          open(os.path.join(HERE, 'NEUTRAL_A4_probe.json'), 'w'), indent=1)
print('\nwrote NEUTRAL_A4_probe.json')
