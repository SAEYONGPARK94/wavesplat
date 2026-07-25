"""Multi-seed / multi-sea-state confirmation of the session's strongest lead:
On SPARSE seas, WaveSplat's frozen init (forecast forward from the init window, NO new obs)
beats the w_phi tracker (which ingests fresh obs every frame). Characterize the regime map
(sparse win vs crowded neutral) and confirm the frozen forecast stays FLAT across the span
(⟹ a genuine 16-19 s forecaster, not just good near t0).
Metric = nRMSE/Hs (the paper's own), averaged over ALL tracking frames.
"""
import json
import numpy as np
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.evaluation.metrics import nrmse
from wavesplat.run import DEFAULT_CONFIG, set_seed

FINE = 'data_finemesh_2026-07-23/'
EPISODES = ['episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed1_uniform.h5',
            'episode_BF4_5comp_seed0_uniform.h5', 'episode_BF4_15comp_seed0_uniform.h5',
            'episode_BF4_30comp_seed0_uniform.h5', 'episode_BF5_15comp_seed0_uniform.h5',
            'episode_BF5_30comp_seed1_uniform.h5']
SEEDS = (42, 1)


def one(ep, s):
    set_seed(s)
    cfg = dict(DEFAULT_CONFIG); cfg['seed'] = s; cfg['device'] = 'cpu'
    L = EpisodeDataLoader(FINE + ep, device='cpu'); cfg['depth'] = L.depth
    Hs = L.Hs_true; gx, gy = L.gt_grid_x, L.gt_grid_y
    inf = L.get_init_window(tp_factor=2.0)
    tf = L.get_tracking_frames(tp_factor=2.0)
    xr = (float(gx[0]), float(gx[-1])); yr = (float(gy[0]), float(gy[-1]))
    # FROZEN: init once, forecast every frame with NO updates (only old init-window obs)
    u1 = SequentialUpdater(cfg); u1.initialize(inf, x_range=xr, y_range=yr)
    fro = np.array([nrmse(u1.predict_field(gx, gy, f.t).cpu().numpy(),
                          f.gt_field.cpu().numpy(), Hs) for f in tf])
    # TRACKED: w_phi tracker, fresh obs each frame
    u2 = SequentialUpdater(cfg); u2.initialize(inf, x_range=xr, y_range=yr)
    trk = []
    for f in tf:
        u2.update(f)
        trk.append(nrmse(u2.predict_field(gx, gy, f.t).cpu().numpy(), f.gt_field.cpu().numpy(), Hs))
    trk = np.array(trk)
    n = len(tf)
    return dict(frozen=float(fro.mean()), tracked=float(trk.mean()),
                frozen_first5=float(fro[:5].mean()), frozen_last5=float(fro[-5:].mean()),
                span_s=float(tf[-1].t - inf[-1].t), nframes=n)


def main():
    out = {}
    print(f"{'episode':40s} {'nc':>3s} {'frozen':>7s} {'tracked':>8s} {'fro_f5':>7s} {'fro_l5':>7s} {'span':>5s}")
    for ep in EPISODES:
        nc = int(ep.split('comp')[0].split('_')[-1])
        rs = [one(ep, s) for s in SEEDS]
        med = {k: float(np.median([r[k] for r in rs])) for k in rs[0]}
        out[ep] = {'ncomp': nc, **med}
        print(f"{ep:40s} {nc:3d} {med['frozen']:7.3f} {med['tracked']:8.3f} "
              f"{med['frozen_first5']:7.3f} {med['frozen_last5']:7.3f} {med['span_s']:5.0f}", flush=True)

    json.dump(out, open('tmp/G1_waveprior_power/frozen_forecast.json', 'w'), indent=1)
    for lab, sel in (('SPARSE 5c', lambda c: c <= 5), ('CROWDED 15/30c', lambda c: c >= 15)):
        eps = [e for e in EPISODES if sel(out[e]['ncomp'])]
        fr = np.mean([out[e]['frozen'] for e in eps]); tr = np.mean([out[e]['tracked'] for e in eps])
        print(f"\n{lab}: frozen-forecast {fr:.3f} vs w_phi tracked {tr:.3f}  "
              f"(tracker {'HARMS' if tr > fr + 0.03 else 'neutral/helps'}, {tr/fr:.1f}× )")
    sp = [e for e in EPISODES if out[e]['ncomp'] <= 5]
    flat = all(abs(out[e]['frozen_last5'] - out[e]['frozen_first5']) < 0.02 for e in sp)
    print(f"\nfrozen forecast FLAT across span on all sparse eps: {flat}  ⟹ genuine multi-second forecaster")


if __name__ == '__main__':
    main()
