"""Probe: run the DEPLOYED path (make_updater, init_method='auto' -> real-domain gate)
on real stereo windows and score the predicted fields against stereo GT."""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS','4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT,'scripts'))
import d3_compare_gifs as cg
from wavesplat.data_loader import FrameData
from wavesplat.optimizer.sequential_update import make_updater
from wavesplat.run import DEFAULT_CONFIG, set_seed


def frames_from_scene(sc, t_shift):
    """Format adapter only (verbatim from tmp/blind_auto_real_2026-07-26/run_blind_auto.py,
    plus a constant time offset so absolute record time does not enter the optimiser)."""
    P = np.asarray(sc['P'], float); z = np.asarray(sc['z'], float)
    frames = []
    for tt in np.unique(P[:, 2]):
        m = P[:, 2] == tt
        n = int(m.sum())
        frames.append(FrameData(t=float(tt - t_shift),
            x_obs=torch.tensor(P[m,0], dtype=torch.float32),
            y_obs=torch.tensor(P[m,1], dtype=torch.float32),
            z_obs=torch.tensor(z[m], dtype=torch.float32),
            confidence=torch.ones(n, dtype=torch.float32),
            incidence_angles=torch.zeros(n, dtype=torch.float32)))
    return frames


def run(rec, win, seed=42):
    sc = cg.scene_real(rec, win=win)
    t_shift = float(np.asarray(sc['P'])[:,2].min())
    t0 = float(sc['Pq'][0,2]) - t_shift
    frames = frames_from_scene(sc, t_shift)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), seed=seed, device='cpu')
    set_seed(seed)
    upd = make_updater(cfg)
    tt = time.time()
    res = upd.initialize(frames,
                         x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                         y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                         verbose=False)
    el = time.time()-tt
    gx = torch.tensor(sc['gxq'], dtype=torch.float32)
    gy = torch.tensor(sc['gyq'], dtype=torch.float32)
    gt = sc['maps_gt']; sg = float(np.std(gt[np.isfinite(gt)]))
    preds, nr, cc = [], [], []
    for li, L in enumerate(sc['leads']):
        p = upd.predict_field(gx, gy, t0 + float(L)).cpu().numpy()
        preds.append(p)
        m = np.isfinite(gt[li])
        nr.append(float(np.sqrt(np.mean((p[m]-gt[li][m])**2))/sg))
        cc.append(float(np.corrcoef(p[m], gt[li][m])[0,1]))
    print(f'{rec} w{win} seed{seed}: route={upd._primary.init_method} '
          f'real_gate={upd._primary._auto_real_gate} k={upd.ensemble_k} '
          f'n_gauss={upd.model.n_gaussians} init {el:.0f}s | '
          f'nowcast nRMSE {nr[0]:.3f} corr {cc[0]:.3f} | '
          f'lead-avg nRMSE {np.mean(nr):.3f} corr {np.mean(cc):.3f}', flush=True)
    return dict(sc=sc, preds=np.array(preds), nr=nr, cc=cc, sg=sg, t0=t0,
                n_gauss=int(upd.model.n_gaussians), route=upd._primary.init_method,
                real_gate=bool(upd._primary._auto_real_gate), k=int(upd.ensemble_k),
                init_s=el)

if __name__ == '__main__':
    for rec, win in [('AA15a',5),('AA15b',1),('AA14',2)]:
        run(rec, win)
