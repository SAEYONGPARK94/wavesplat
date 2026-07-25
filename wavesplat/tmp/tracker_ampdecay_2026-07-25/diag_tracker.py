"""DIAGNOSE: why does the w_phi tracker turn a near-exact init (nRMSE~0.005-0.01) into
0.05-0.16 on the two calmest (1-component) finemesh episodes?

We instrument the tracking stage and run controlled variants that each isolate one candidate
mechanism, all starting from the SAME post-init model state:

  V0 baseline     : shipped w_phi, lambda_l1=0.001, 10 Adam steps/frame   (reproduce)
  V1 phase_only   : NO gradient at all -- just phi += omega*dt each frame  (pure propagation)
  V2 l1_zero      : w_phi, lambda_l1=0                                     (L1 shrinkage test)
  V3 freeze_w     : phi_only mode (w frozen), lambda_l1 irrelevant to w    (is it the w update?)
  V4 gt_inject_lin: 1-comp model set to the TRUE fundamental, phase_only   (physics floor)
  V5 gt_inject_trk: 1-comp model set to the TRUE fundamental, baseline trk (does trk decay a
                    perfect linear solution?)

Per-frame we log full-field nRMSE (identical metric to run.py), sum|w|, and dt. For the first
few frames we log the within-frame per-step sum|w| trajectory (Adam+L1 shrinkage inside the 10
steps).
"""
import os, sys, json, copy
import numpy as np
import torch
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)

from wavesplat.run import DEFAULT_CONFIG, set_seed
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.model.dispersion import dispersion_omega
from wavesplat.evaluation.metrics import nrmse

DEV = os.environ.get('DIAG_DEV', 'cpu')
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')


def snapshot(model):
    return dict(n=model.n_gaussians,
                w=model.w.detach().clone(), kx=model.kx.detach().clone(),
                ky=model.ky.detach().clone(), phi=model.phi.detach().clone())


def restore(model, snap):
    model.n_gaussians = snap['n']
    model.w = torch.nn.Parameter(snap['w'].clone())
    model.kx = torch.nn.Parameter(snap['kx'].clone())
    model.ky = torch.nn.Parameter(snap['ky'].clone())
    model.phi = torch.nn.Parameter(snap['phi'].clone())


def run_variant(updater, tracking_frames, grid_x, grid_y, Hs, last_init_t,
                mode='w_phi', lambda_l1=0.001, do_gradient=True, log_steps_first=3):
    updater.tracking_mode = mode
    updater.lambda_l1 = lambda_l1
    updater._frame_buffer = []
    updater._last_t = last_init_t
    updater._build_tracking_optimizer()
    logs = []
    step_traces = []
    for i, frame in enumerate(tracking_frames):
        dt = frame.t - updater._last_t
        if abs(dt) > 1e-6:
            with torch.no_grad():
                omega = updater.model._omega()
                updater.model.phi.add_(omega * dt)
        updater._last_t = frame.t
        if len(frame.x_obs) == 0:
            continue
        updater._frame_buffer.append(frame)
        if len(updater._frame_buffer) > updater.tracking_window:
            updater._frame_buffer.pop(0)
        strace = []
        if do_gradient:
            opt = updater.tracking_optimizer
            for step in range(updater.tracking_steps):
                opt.zero_grad()
                loss = updater._compute_loss(updater._frame_buffer)
                loss.backward()
                opt.step()
                if i < log_steps_first:
                    strace.append(float(updater.model.w.detach().abs().sum()))
            updater.model.clamp_steepness(updater.max_steepness)
        with torch.no_grad():
            pred = updater.predict_field(grid_x, grid_y, frame.t).cpu().numpy()
        gt = frame.gt_field.cpu().numpy()
        logs.append(dict(i=i, t=float(frame.t), nrmse=float(nrmse(pred, gt, Hs)),
                         sumw=float(updater.model.w.detach().abs().sum()),
                         maxw=float(updater.model.w.detach().abs().max()),
                         n_pts=int(len(frame.x_obs)), dt=float(dt)))
        if strace:
            step_traces.append({'frame': i, 'sumw_per_step': strace})
    return logs, step_traces


def eval_field(updater, grid_x, grid_y, t, gt, Hs):
    with torch.no_grad():
        pred = updater.predict_field(grid_x, grid_y, t).cpu().numpy()
    return float(nrmse(pred, gt, Hs))


def diag_episode(ep_name, seed):
    path = os.path.join(DATA, ep_name)
    set_seed(seed)
    loader = EpisodeDataLoader(path, device=DEV)
    Hs = loader.Hs_true
    grid_x, grid_y = loader.gt_grid_x, loader.gt_grid_y
    cfg = DEFAULT_CONFIG.copy()
    cfg['seed'] = seed
    cfg['device'] = DEV

    init_frames = loader.get_init_window(tp_factor=2.0)
    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
    updater = SequentialUpdater(cfg)
    updater.initialize(init_frames,
                       x_range=(float(grid_x[0]), float(grid_x[-1])),
                       y_range=(float(grid_y[0]), float(grid_y[-1])))
    last_init = init_frames[-1]
    last_init_t = last_init.t
    init_nrmse = eval_field(updater, grid_x, grid_y, last_init.t,
                            last_init.gt_field.cpu().numpy(), Hs)
    post_init = snapshot(updater.model)
    N_init = updater.model.n_gaussians

    out = {'episode': ep_name, 'seed': seed, 'Hs': Hs, 'Tp': loader.Tp_true,
           'N_init': int(N_init), 'init_nrmse': init_nrmse,
           'n_track_frames': len(tracking_frames)}

    variants = {}

    # V0 baseline
    restore(updater.model, post_init)
    logs, straces = run_variant(updater, tracking_frames, grid_x, grid_y, Hs, last_init_t,
                                mode='w_phi', lambda_l1=0.001, do_gradient=True)
    variants['V0_baseline'] = {'logs': logs, 'step_traces': straces}

    # V1 phase_only (no gradient)
    restore(updater.model, post_init)
    logs, _ = run_variant(updater, tracking_frames, grid_x, grid_y, Hs, last_init_t,
                          mode='w_phi', lambda_l1=0.0, do_gradient=False)
    variants['V1_phase_only'] = {'logs': logs}

    # V2 l1_zero
    restore(updater.model, post_init)
    logs, straces = run_variant(updater, tracking_frames, grid_x, grid_y, Hs, last_init_t,
                                mode='w_phi', lambda_l1=0.0, do_gradient=True)
    variants['V2_l1_zero'] = {'logs': logs, 'step_traces': straces}

    # V3 freeze_w (phi_only)
    restore(updater.model, post_init)
    logs, _ = run_variant(updater, tracking_frames, grid_x, grid_y, Hs, last_init_t,
                          mode='phi_only', lambda_l1=0.001, do_gradient=True)
    variants['V3_freeze_w'] = {'logs': logs}

    # --- GT injection: perfect linear fundamental(s) ---
    wc = loader.wave_components
    if wc is not None:
        wc = np.array(wc)
        n_true = wc.shape[0]
        A = wc[:, 0].astype(np.float32)
        kx = wc[:, 2].astype(np.float32)
        ky = wc[:, 3].astype(np.float32)
        phi = wc[:, 4].astype(np.float32)
        gt_model = updater._build_model(n_true)
        with torch.no_grad():
            gt_model.w.copy_(torch.tensor(A, device=DEV))
            gt_model.kx.copy_(torch.tensor(kx, device=DEV))
            gt_model.ky.copy_(torch.tensor(ky, device=DEV))
            gt_model.phi.copy_(torch.tensor(phi, device=DEV))
        updater.model = gt_model
        gt_snap = snapshot(updater.model)
        gt_init_nrmse = eval_field(updater, grid_x, grid_y, last_init.t,
                                   last_init.gt_field.cpu().numpy(), Hs)
        out['gt_inject_init_nrmse'] = gt_init_nrmse
        out['n_true'] = int(n_true)

        # V4 gt phase_only (pure physics floor)
        restore(updater.model, gt_snap)
        logs, _ = run_variant(updater, tracking_frames, grid_x, grid_y, Hs, last_init_t,
                              mode='w_phi', lambda_l1=0.0, do_gradient=False)
        variants['V4_gt_phase_only'] = {'logs': logs}

        # V5 gt baseline tracking
        restore(updater.model, gt_snap)
        logs, straces = run_variant(updater, tracking_frames, grid_x, grid_y, Hs, last_init_t,
                                    mode='w_phi', lambda_l1=0.001, do_gradient=True)
        variants['V5_gt_baseline_trk'] = {'logs': straces and logs or logs,
                                          'step_traces': straces}

    out['variants'] = variants
    return out


def summarize(out):
    print(f"\n=== {out['episode']} seed={out['seed']} | Hs={out['Hs']} Tp={out['Tp']} "
          f"N_init={out['N_init']} n_true={out.get('n_true')} ===")
    print(f"  init_nrmse (fitted linear)      = {out['init_nrmse']:.4f}")
    if 'gt_inject_init_nrmse' in out:
        print(f"  gt_inject_init_nrmse (true k/A) = {out['gt_inject_init_nrmse']:.4f}")
    for name, v in out['variants'].items():
        logs = v['logs']
        nr = np.array([l['nrmse'] for l in logs])
        sw = np.array([l['sumw'] for l in logs])
        print(f"  {name:20s} avg_nRMSE={nr.mean():.4f}  "
              f"nRMSE[0]={nr[0]:.4f} nRMSE[-1]={nr[-1]:.4f}  "
              f"sum|w|: {sw[0]:.4f} -> {sw[-1]:.4f}")
    # within-frame step trace for baseline frame 0
    st = out['variants']['V0_baseline'].get('step_traces')
    if st:
        print(f"  V0 within-frame sum|w| per step (frame {st[0]['frame']}): "
              + ' '.join(f'{x:.4f}' for x in st[0]['sumw_per_step']))
    st2 = out['variants']['V2_l1_zero'].get('step_traces')
    if st2:
        print(f"  V2 (l1=0) within-frame sum|w| per step (frame {st2[0]['frame']}): "
              + ' '.join(f'{x:.4f}' for x in st2[0]['sumw_per_step']))


if __name__ == '__main__':
    EPS = [('episode_BF3_1comp_seed0_uniform.h5', 42),
           ('episode_BF3_1comp_seed0_uniform.h5', 1),
           ('episode_BF4_1comp_seed0_uniform.h5', 42),
           ('episode_BF2_1comp_seed0_uniform.h5', 42)]
    allout = []
    for ep, sd in EPS:
        o = diag_episode(ep, sd)
        summarize(o)
        allout.append(o)
    with open(os.path.join(HERE, 'diag_tracker_results.json'), 'w') as f:
        json.dump(allout, f, indent=1, default=float)
    print('\nwrote diag_tracker_results.json')
