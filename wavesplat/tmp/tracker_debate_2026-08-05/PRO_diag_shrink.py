"""PRO agent, proposition G -- LIGHT single-episode diagnostic: WHICH knob shrinks the field?

One episode (BF4_15comp), one seed (42), oracle frozen-k placement (= arm S2-T1f), and one
knob changed at a time.  Per-frame it logs the quantities the S1/S2 panel does not carry:
sum|w| (total spectral amplitude), how often clamp_steepness actually bites, the
observation-fit loss the tracker is minimising, and the field metrics it is NOT.

Reference values already on record for this cell (tmp/s1s2_2026-08-05/results.jsonl, seed 42):
    S2-T1f (D0)        nRMSE 0.1425  VR 0.438  rho 0.845
    X-T1f-notrack (D6) nRMSE 0.0262  VR 0.923  rho 0.995
D0 and D6 are reproductions and must land on those numbers.

Research-only. Nothing in wavesplat/ is modified; the module-level flags are read, not written.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
EPISODE = 'episode_BF4_15comp_seed0_uniform.h5'
SEED = 42
N_GT = 15
OUT = os.path.join(HERE, 'PRO_diag_shrink.jsonl')

# base = S2-T1f exactly (oracle placement, k frozen, N = N_gt, 900 init steps)
BASE = dict(use_dispersion=True, use_confidence=True, adaptive_n=True,
            tracking_window=3, prune_threshold=0.10, max_n=30,
            init_method='oracle', n_gaussians=[N_GT], loss_thresholds=[None],
            max_iters_per_stage=[900], init_k_lr_scale=0.0, seed=SEED)

ARMS = {
    'D0-default':      {},                                # repro of S2-T1f
    'D1-l1_0':         dict(lambda_l1=0.0),               # L1 sparsity off
    'D2-l1_smooth_0':  dict(lambda_l1=0.0, lambda_smooth=0.0),
    'D3-noconf':       dict(use_confidence=False),        # confidence weighting off
    'D4-steps1':       dict(tracking_steps=1),            # 10 -> 1 gradient step/frame
    'D5-window12':     dict(tracking_window=12),          # 3 -> 12 frame memory
    'D7-noclamp':      dict(max_steepness=1e9),           # steepness projection off
    'D8-lr10x_less':   dict(tracking_lr=0.0005),          # 0.005 -> 0.0005
    'D6-notrack':      dict(tracking_lr=0.0),             # repro of X-T1f-notrack
}

# ---- E arms: the SAME knob under the DEPLOYABLE (beamform) placement = S1-A0 -----------
# This is the combination tmp/s1s2_2026-08-05/RESULT.md section 6 item 2 lists as not done.
A0 = dict(use_dispersion=True, use_confidence=True, adaptive_n=True,
          tracking_window=3, prune_threshold=0.10, max_n=30,
          init_method='v2_konly', n_gaussians=[3, 8, 15],
          loss_thresholds=[0.1, 0.05, None], max_iters_per_stage=[200, 200, 500])
E_ARMS = {
    'E0-A0-default':    {},                                    # repro of S1-A0
    'E1-A0-l1_0':       dict(lambda_l1=0.0),
    'E2-A0-l1smooth_0': dict(lambda_l1=0.0, lambda_smooth=0.0),
    'E6-A0-notrack':    dict(tracking_lr=0.0),                 # repro of X-A0-notrack
}

# lambda_l1 / lambda_smooth are read from self.* at every loss evaluation and are shared by
# BOTH phases, so 'D2'/'E2' change the initialisation as well -- a confound. Arms whose name
# ends in '-p2' zero the penalties ONLY AFTER initialize() returns, which isolates the tracker
# with a bit-identical Phase 1. Implemented in run_arm, not in the config.
PHASE2_ONLY = {'D2p2-track_only': ('oracle', dict(lambda_l1=0.0, lambda_smooth=0.0)),
               'E2p2-track_only': ('a0', dict(lambda_l1=0.0, lambda_smooth=0.0)),
               'D1p2-track_only_l1': ('oracle', dict(lambda_l1=0.0)),
               'E1p2-track_only_l1': ('a0', dict(lambda_l1=0.0))}


def run_arm(name, seed=SEED):
    from wavesplat.run import set_seed
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    from wavesplat.evaluation.metrics import compute_all_metrics, nrmse
    import wavesplat.optimizer.initializer as _init

    p2 = PHASE2_ONLY.get(name)
    if p2 is not None:
        oracle = p2[0] == 'oracle'
        cfg = dict(BASE) if oracle else dict(A0)
        over = {}                       # nothing changes before/inside initialize()
    else:
        oracle = name in ARMS
        cfg = dict(BASE) if oracle else dict(A0)
        over = ARMS[name] if oracle else E_ARMS[name]
    cfg.update(over)
    cfg['device'] = 'cpu'
    cfg['seed'] = seed
    t0 = time.time()
    set_seed(seed)
    path = os.path.join(DATA, EPISODE)
    loader = EpisodeDataLoader(path, device='cpu')
    cfg['depth'] = loader.depth
    _init.ORACLE_COMPONENTS = np.asarray(loader.wave_components) if oracle else None
    try:
        up = SequentialUpdater(cfg)
        init_frames = loader.get_init_window(tp_factor=2.0)
        gx, gy = loader.gt_grid_x, loader.gt_grid_y
        up.initialize(init_frames,
                      x_range=(float(gx[0]), float(gx[-1])),
                      y_range=(float(gy[0]), float(gy[-1])), verbose=False)

        # count how often the one-sided steepness projection actually bites
        clip_events = {'n': 0, 'frames_with_clip': 0, 'amp_removed': 0.0}
        model = up.model
        raw_clamp = model.clamp_steepness

        def counting_clamp(max_steepness=0.15):
            with torch.no_grad():
                kmag = torch.sqrt(model.kx ** 2 + model.ky ** 2 + 1e-12)
                st = kmag * torch.abs(model.w)
                n_ex = int((st > max_steepness).sum())
                before = float(torch.abs(model.w).sum())
            raw_clamp(max_steepness)
            with torch.no_grad():
                after = float(torch.abs(model.w).sum())
            if n_ex:
                clip_events['n'] += n_ex
                clip_events['frames_with_clip'] += 1
                clip_events['amp_removed'] += before - after
            return None
        model.clamp_steepness = counting_clamp

        if p2 is not None:                     # PHASE-2-ONLY: mutate after init, before tracking
            for k, v in p2[1].items():
                setattr(up, k, v)

        last = init_frames[-1]
        p0 = up.predict_field(gx, gy, last.t).cpu().numpy()
        init_n = float(nrmse(p0, last.gt_field.cpu().numpy(), loader.Hs_true))
        sumw0 = float(torch.abs(up.model.w).sum())

        frames = loader.get_tracking_frames(tp_factor=2.0)
        per = []
        for f in frames:
            u = up.update(f)
            pred = up.predict_field(gx, gy, f.t).cpu().numpy()
            m = compute_all_metrics(pred, f.gt_field.cpu().numpy(),
                                    gx.cpu().numpy(), gy.cpu().numpy(), loader.Hs_true,
                                    obs_x=f.x_obs.cpu().numpy() if len(f.x_obs) else None,
                                    obs_y=f.y_obs.cpu().numpy() if len(f.y_obs) else None)
            per.append(dict(t=float(f.t), nRMSE=float(m['nRMSE']),
                            VR=float(m['Variance_Ratio']),
                            rho=float(m['Pearson_rho']),
                            Hs_error=float(m['Hs_error']),
                            RMSE_unobs=float(m.get('RMSE_unobserved', np.nan)),
                            obs_loss=float(u['loss']),
                            sum_w=float(torch.abs(up.model.w).sum()),
                            max_steep=float((torch.sqrt(up.model.kx ** 2 + up.model.ky ** 2)
                                             * torch.abs(up.model.w)).max())))
        g = lambda k: float(np.nanmean([p[k] for p in per]))
        rec = dict(arm=name, cfg=dict(p2[1]) if p2 else dict(over),
                   episode=EPISODE, seed=seed,
                   init_nrmse=init_n, sum_w_init=sumw0,
                   sum_w_final=per[-1]['sum_w'],
                   sum_w_ratio=per[-1]['sum_w'] / sumw0,
                   nRMSE=g('nRMSE'), VR=g('VR'), rho=g('rho'),
                   Hs_error=g('Hs_error'), RMSE_unobs=g('RMSE_unobs'),
                   obs_loss=g('obs_loss'),
                   obs_loss_first=per[0]['obs_loss'], obs_loss_last=per[-1]['obs_loss'],
                   n_frames=len(per), clip=clip_events,
                   traj=per, wall_s=time.time() - t0)
    finally:
        _init.ORACLE_COMPONENTS = None
    return rec


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    todo = sys.argv[1:] or list(ARMS)
    with open(OUT, 'a') as fh:
        for spec in todo:
            a, _, s = spec.partition(':')
            r = run_arm(a, int(s) if s else SEED)
            fh.write(json.dumps(r) + '\n'); fh.flush()
            print(f"{a:<16} nRMSE {r['nRMSE']:.4f}  VR {r['VR']:.3f}  rho {r['rho']:.3f}  "
                  f"sum|w| {r['sum_w_ratio']:.3f}x  obs_loss {r['obs_loss']:.5f}  "
                  f"clip_frames {r['clip']['frames_with_clip']}/{r['n_frames']}  "
                  f"({r['wall_s']:.0f}s)", flush=True)
