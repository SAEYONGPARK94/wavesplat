"""Decisive tracker experiment (fork): over the FULL tracking-frame set, compare
  (A) frozen init, pure dispersion propagation (predict_field, NO update)
  (B) the shipped w_phi tracker (update() per frame)
  (C) phi_only tracker
on the sparse episodes where init is genuinely good (held-out 0.015).
If (A) ≪ (B) over all frames ⟹ the gradient tracking updates DEGRADE a good propagated
init (the tracker is the culprit, and 'propagate, don't gradient-track' is a cheap win).
If (A) ≈ (B) high ⟹ field-evolution/k-drift, not the tracker.
"""
import json
import numpy as np
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.evaluation.metrics import nrmse
from wavesplat.run import DEFAULT_CONFIG, set_seed

FINE = 'data_finemesh_2026-07-23/'
EPS = ['episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed1_uniform.h5',
       'episode_BF4_5comp_seed0_uniform.h5']
SEED = 42


def init_updater(L, mode):
    set_seed(SEED)
    cfg = dict(DEFAULT_CONFIG); cfg['seed'] = SEED; cfg['device'] = 'cpu'
    cfg['depth'] = L.depth; cfg['tracking_mode'] = mode
    upd = SequentialUpdater(cfg)
    upd.initialize(L.get_init_window(tp_factor=2.0),
                   x_range=(float(L.gt_grid_x[0]), float(L.gt_grid_x[-1])),
                   y_range=(float(L.gt_grid_y[0]), float(L.gt_grid_y[-1])))
    return upd


def main():
    out = {}
    for ep in EPS:
        L = EpisodeDataLoader(FINE + ep, device='cpu')
        gx, gy, Hs = L.gt_grid_x, L.gt_grid_y, L.Hs_true
        tf = L.get_tracking_frames(tp_factor=2.0)
        # (A) frozen init, pure propagation over ALL tracking frames
        upd = init_updater(L, 'w_phi')
        frozenA = [nrmse(upd.predict_field(gx, gy, f.t).cpu().numpy(),
                         f.gt_field.cpu().numpy(), Hs) for f in tf]
        # (B) shipped w_phi tracker
        upd = init_updater(L, 'w_phi')
        trackB = []
        for f in tf:
            upd.update(f)
            trackB.append(nrmse(upd.predict_field(gx, gy, f.t).cpu().numpy(),
                                f.gt_field.cpu().numpy(), Hs))
        # (C) phi_only tracker
        upd = init_updater(L, 'phi_only')
        trackC = []
        for f in tf:
            upd.update(f)
            trackC.append(nrmse(upd.predict_field(gx, gy, f.t).cpu().numpy(),
                                f.gt_field.cpu().numpy(), Hs))
        out[ep] = {'frozen_propagate': float(np.mean(frozenA)),
                   'w_phi_tracker': float(np.mean(trackB)),
                   'phi_only_tracker': float(np.mean(trackC)),
                   'n_frames': len(tf),
                   'frozen_first5': float(np.mean(frozenA[:5])),
                   'frozen_last5': float(np.mean(frozenA[-5:]))}
        print(f"{ep:38s} frozen {np.mean(frozenA):.3f} (first5 {np.mean(frozenA[:5]):.3f} "
              f"last5 {np.mean(frozenA[-5:]):.3f}) | w_phi {np.mean(trackB):.3f} | "
              f"phi_only {np.mean(trackC):.3f} | {len(tf)}f", flush=True)

    json.dump(out, open('tmp/G1_waveprior_power/fork_tracker_decisive.json', 'w'), indent=1)
    fr = np.mean([v['frozen_propagate'] for v in out.values()])
    wp = np.mean([v['w_phi_tracker'] for v in out.values()])
    po = np.mean([v['phi_only_tracker'] for v in out.values()])
    print(f"\n=== MEAN (sparse, all frames): frozen-propagate {fr:.3f} | w_phi {wp:.3f} | phi_only {po:.3f} ===")


if __name__ == '__main__':
    main()
