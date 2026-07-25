"""Held-out init test (both forks' convergent demand: in-sample confound).
The paper's init_nrmse (~0.011 on sparse) is measured on the LAST init-window frame the
initializer OPTIMIZED on = in-sample. If the init model, with NO tracking updates, is
evaluated on HELD-OUT frames (the first tracking frames), does it stay ~0.01 or jump to
~0.20? If it jumps, the "tracker is the weakness" story is an in-sample artifact — the init
was overfit and the tracker is (partly) exonerated.
"""
import json
import numpy as np
import torch

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.evaluation.metrics import nrmse
from wavesplat.run import DEFAULT_CONFIG, set_seed

FINE = 'data_finemesh_2026-07-23/'
EPISODES = ['episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed1_uniform.h5',
            'episode_BF4_5comp_seed0_uniform.h5']
SEEDS = (42, 1)


def run_one(ep, s):
    set_seed(s)
    cfg = dict(DEFAULT_CONFIG); cfg['seed'] = s; cfg['device'] = 'cpu'
    L = EpisodeDataLoader(FINE + ep, device='cpu'); cfg['depth'] = L.depth
    init_frames = L.get_init_window(tp_factor=2.0)
    upd = SequentialUpdater(cfg)
    upd.initialize(init_frames,
                   x_range=(float(L.gt_grid_x[0]), float(L.gt_grid_x[-1])),
                   y_range=(float(L.gt_grid_y[0]), float(L.gt_grid_y[-1])))
    gx, gy = L.gt_grid_x, L.gt_grid_y
    Hs = L.Hs_true
    # in-sample: last init frame (what run_estimation reports as init_nrmse)
    li = init_frames[-1]
    pin = upd.predict_field(gx, gy, li.t).cpu().numpy()
    nr_in = nrmse(pin, li.gt_field.cpu().numpy(), Hs)
    # held-out: first 5 tracking frames, NO update (frozen init model)
    tf = L.get_tracking_frames(tp_factor=2.0)[:5]
    nr_out = []
    for f in tf:
        p = upd.predict_field(gx, gy, f.t).cpu().numpy()
        nr_out.append(nrmse(p, f.gt_field.cpu().numpy(), Hs))
    return nr_in, float(np.mean(nr_out))


def main():
    out = {}
    print(f"{'episode':40s} {'nc':>3s} {'init_in':>8s} {'init_heldout':>12s}")
    for ep in EPISODES:
        nc = int(ep.split('comp')[0].split('_')[-1])
        ins, outs = [], []
        for s in SEEDS:
            a, b = run_one(ep, s)
            ins.append(a); outs.append(b)
        out[ep] = {'ncomp': nc, 'init_insample': float(np.median(ins)),
                   'init_heldout': float(np.median(outs))}
        print(f"{ep:40s} {nc:3d} {np.median(ins):8.3f} {np.median(outs):12.3f}", flush=True)

    json.dump(out, open('tmp/G1_waveprior_power/heldout_init.json', 'w'), indent=1)
    sp = [v for v in out.values() if v['ncomp'] <= 5]
    ii = np.mean([v['init_insample'] for v in sp]); ho = np.mean([v['init_heldout'] for v in sp])
    print(f"\n=== HELD-OUT INIT (sparse 5c): in-sample {ii:.3f} → held-out {ho:.3f} ===")
    print("  ⟹ " + ("held-out ≈ in-sample: init genuinely good, tracker IS the weakness (real lead)"
                     if ho < ii + 0.05 else
                     f"held-out JUMPS to {ho:.3f}: init 0.01 was overfit, tracker (partly) EXONERATED — thread weakens"))


if __name__ == '__main__':
    main()
