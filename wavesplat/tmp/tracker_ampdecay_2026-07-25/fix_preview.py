"""FIX PREVIEW: does removing the redundant phase-prediction (phi += omega*dt) fix the
init->track gap, and does it hurt crowded episodes where init is already bad?

Mechanism (established in diag_tracker): the forward model already evolves the wave via its
own -omega*t term, so the update()'s `phi += omega*dt` DOUBLE-counts time and freezes the wave
at the init time; the 10 Adam steps/frame must re-slew it every frame, imperfectly, bleeding
amplitude. Removing the phi kick should let forward(t) track the true evolution directly.

Variants (both full w_phi gradient, lambda_l1=0.001, from the SAME post-init state):
  A_baseline        phi += omega*dt  (shipped)
  B_no_phase_pred   skip the phi kick (candidate fix)

Also a no-gradient control for each to see the pure-propagation behaviour:
  A_prop_only       phi kick, no gradient  (should decorrelate ~0.31)
  B_prop_only       no kick, no gradient   (should stay ~ init: forward already evolves)
"""
import os, sys, json
import numpy as np
import torch
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)

from wavesplat.run import DEFAULT_CONFIG, set_seed
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.evaluation.metrics import nrmse

DEV = os.environ.get('DIAG_DEV', 'cuda')
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')


def snapshot(m):
    return dict(n=m.n_gaussians, w=m.w.detach().clone(), kx=m.kx.detach().clone(),
                ky=m.ky.detach().clone(), phi=m.phi.detach().clone())


def restore(m, s):
    m.n_gaussians = s['n']
    m.w = torch.nn.Parameter(s['w'].clone()); m.kx = torch.nn.Parameter(s['kx'].clone())
    m.ky = torch.nn.Parameter(s['ky'].clone()); m.phi = torch.nn.Parameter(s['phi'].clone())


def track(updater, frames, gx, gy, Hs, last_t, do_phase_predict=True, do_gradient=True):
    updater.tracking_mode = 'w_phi'; updater.lambda_l1 = 0.001
    updater._frame_buffer = []; updater._last_t = last_t
    updater._build_tracking_optimizer()
    nrs, sws = [], []
    for frame in frames:
        dt = frame.t - updater._last_t
        if do_phase_predict and abs(dt) > 1e-6:
            with torch.no_grad():
                updater.model.phi.add_(updater.model._omega() * dt)
        updater._last_t = frame.t
        if len(frame.x_obs) == 0:
            continue
        updater._frame_buffer.append(frame)
        if len(updater._frame_buffer) > updater.tracking_window:
            updater._frame_buffer.pop(0)
        if do_gradient:
            opt = updater.tracking_optimizer
            for _ in range(updater.tracking_steps):
                opt.zero_grad(); loss = updater._compute_loss(updater._frame_buffer)
                loss.backward(); opt.step()
            updater.model.clamp_steepness(updater.max_steepness)
        with torch.no_grad():
            pred = updater.predict_field(gx, gy, frame.t).cpu().numpy()
        nrs.append(nrmse(pred, frame.gt_field.cpu().numpy(), Hs))
        sws.append(float(updater.model.w.detach().abs().sum()))
    return np.array(nrs), np.array(sws)


def one(ep, seed):
    set_seed(seed)
    loader = EpisodeDataLoader(os.path.join(DATA, ep), device=DEV)
    Hs = loader.Hs_true; gx, gy = loader.gt_grid_x, loader.gt_grid_y
    cfg = DEFAULT_CONFIG.copy(); cfg['seed'] = seed; cfg['device'] = DEV
    inf = loader.get_init_window(2.0); trf = loader.get_tracking_frames(2.0)
    up = SequentialUpdater(cfg)
    up.initialize(inf, x_range=(float(gx[0]), float(gx[-1])),
                  y_range=(float(gy[0]), float(gy[-1])))
    last_t = inf[-1].t
    with torch.no_grad():
        p = up.predict_field(gx, gy, last_t).cpu().numpy()
    init_nr = nrmse(p, inf[-1].gt_field.cpu().numpy(), Hs)
    snap = snapshot(up.model); N = up.model.n_gaussians

    restore(up.model, snap); A_nr, A_sw = track(up, trf, gx, gy, Hs, last_t, True, True)
    restore(up.model, snap); B_nr, B_sw = track(up, trf, gx, gy, Hs, last_t, False, True)
    restore(up.model, snap); Ap_nr, _ = track(up, trf, gx, gy, Hs, last_t, True, False)
    restore(up.model, snap); Bp_nr, _ = track(up, trf, gx, gy, Hs, last_t, False, False)

    print(f"{ep:<40} seed={seed} N={N} init={init_nr:.4f} | "
          f"A_base={A_nr.mean():.4f} B_nophasepred={B_nr.mean():.4f} "
          f"(delta {B_nr.mean()-A_nr.mean():+.4f}) | "
          f"propOnly A={Ap_nr.mean():.4f} B={Bp_nr.mean():.4f} | "
          f"sum|w| A:{A_sw[0]:.3f}->{A_sw[-1]:.3f} B:{B_sw[0]:.3f}->{B_sw[-1]:.3f}",
          flush=True)
    return dict(ep=ep, seed=seed, N=int(N), init=float(init_nr),
                A_base=float(A_nr.mean()), B_nophasepred=float(B_nr.mean()),
                A_proponly=float(Ap_nr.mean()), B_proponly=float(Bp_nr.mean()),
                A_sw=[float(A_sw[0]), float(A_sw[-1])],
                B_sw=[float(B_sw[0]), float(B_sw[-1])])


if __name__ == '__main__':
    EPS = [('episode_BF3_1comp_seed0_uniform.h5', 42),  # calm
           ('episode_BF4_1comp_seed0_uniform.h5', 42),  # calm
           ('episode_BF3_2comp_seed0_uniform.h5', 42),  # sparse
           ('episode_BF3_5comp_seed0_uniform.h5', 42),  # mid (big gap)
           ('episode_BF4_5comp_seed0_uniform.h5', 42),  # mid
           ('episode_BF4_15comp_seed0_uniform.h5', 42), # crowded (no gap)
           ('episode_BF4_30comp_seed0_uniform.h5', 42), # crowded
           ('episode_BF3_1comp_seed0_uniform.h5', 1),   # calm, worst-basin seed
           ]
    res = [one(ep, sd) for ep, sd in EPS]
    with open(os.path.join(HERE, 'fix_preview_results.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print('\nwrote fix_preview_results.json')
