"""ERROR-HUNT: is 'init 0.015 → tracker degrades to 0.20' a real tracker failure, or a
window/time confound? Compare, per tracking frame:
  frozen  = init model, NO updates (must forecast forward from t=14s obs)
  tracked = w_phi tracker (gets fresh obs each frame = nowcast)
If tracked << frozen at late frames, the tracker HELPS (uses new obs) and the 0.20 is
forecast-horizon/window, not tracker degradation → 'tracker is the weakness' is mis-framed.
"""
import numpy as np
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.evaluation.metrics import nrmse
from wavesplat.run import DEFAULT_CONFIG, set_seed, run_estimation

EP = 'data_finemesh_2026-07-23/episode_BF4_5comp_seed0_uniform.h5'

set_seed(42)
cfg = dict(DEFAULT_CONFIG); cfg['seed'] = 42; cfg['device'] = 'cpu'
L = EpisodeDataLoader(EP, device='cpu'); cfg['depth'] = L.depth
Hs = L.Hs_true; gx, gy = L.gt_grid_x, L.gt_grid_y
init_frames = L.get_init_window(tp_factor=2.0)
tf = L.get_tracking_frames(tp_factor=2.0)

upd = SequentialUpdater(cfg)
upd.initialize(init_frames, x_range=(float(gx[0]), float(gx[-1])), y_range=(float(gy[0]), float(gy[-1])))
t_init = init_frames[-1].t
# FROZEN init: predict every tracking frame, NO updates
frozen = []
for f in tf:
    p = upd.predict_field(gx, gy, f.t).cpu().numpy()
    frozen.append(nrmse(p, f.gt_field.cpu().numpy(), Hs))
frozen = np.array(frozen)

# TRACKED: full run, per-frame
r = run_estimation(EP, config={'adaptive_n': True, 'seed': 42, 'device': 'cpu'})
# re-run to grab per-frame? run_estimation returns summary only; recompute tracked per-frame:
upd2 = SequentialUpdater(cfg)
upd2.initialize(init_frames, x_range=(float(gx[0]), float(gx[-1])), y_range=(float(gy[0]), float(gy[-1])))
tracked = []
for f in tf:
    upd2.update(f)
    p = upd2.predict_field(gx, gy, f.t).cpu().numpy()
    tracked.append(nrmse(p, f.gt_field.cpu().numpy(), Hs))
tracked = np.array(tracked)

leads = np.array([f.t - t_init for f in tf])
print(f"init(last) nRMSE/Hs = {nrmse(upd.predict_field(gx,gy,t_init).cpu().numpy(), init_frames[-1].gt_field.cpu().numpy(), Hs):.3f}")
print(f"\n{'lead_s':>7s} {'frozen':>7s} {'tracked':>8s}")
for i in range(0, len(tf), max(1, len(tf)//12)):
    print(f"{leads[i]:7.1f} {frozen[i]:7.3f} {tracked[i]:8.3f}")
print(f"\nMEANS over all {len(tf)} tracking frames: frozen {frozen.mean():.3f}  tracked {tracked.mean():.3f}  (run_estimation avg_nRMSE {r['avg_nRMSE']:.3f})")
print(f"first-5 frames: frozen {frozen[:5].mean():.3f}  tracked {tracked[:5].mean():.3f}")
if tracked.mean() < frozen.mean() - 0.03:
    print("⟹ tracker BEATS frozen-forecast ⟹ it HELPS; the 0.20 is forecast-horizon/window, NOT tracker degradation ⟹ 'tracker is the weakness' MIS-FRAMED")
elif tracked.mean() > frozen.mean() + 0.03:
    print("⟹ tracker WORSE than frozen ⟹ tracker actively hurts ⟹ 'tracker is the weakness' STANDS")
else:
    print("⟹ tracker ≈ frozen ⟹ neither helps; degradation is field-evolution/window, not the tracker per se")
