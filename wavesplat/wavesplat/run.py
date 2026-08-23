"""
run.py
Main inference script for WaveSplat wave field estimation.

Usage:
    python -m wavesplat.run --episode data/episode_BF4_30comp_seed42_uniform.h5
    python -m wavesplat.run --episode data/episode_BF4_30comp_seed42_uniform.h5 --verbose
"""

import argparse
import json
import os
import time
import numpy as np
import torch

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import (SequentialUpdater, make_updater,
                                                   LAMBDA_L1_DEFAULT,
                                                   INIT_METHOD_DEFAULT,
                                                   VARPRO_STEPS_DEFAULT,
                                                   LEARNED_INIT_ITERS_DEFAULT,
                                                   AUTO_SWITCH_THRESH_DEFAULT,
                                                   BANK_REFIT_EVERY_S_DEFAULT)
from wavesplat.evaluation.metrics import rmse_observed, rmse_unobserved, compute_all_metrics, rmse, nrmse


DEFAULT_CONFIG = {
    'n_gaussians': [3, 8, 15],
    'depth': 50.0,
    'lr': 0.01,
    # 1e-3 -> 1e-4 on 2026-08-11 (S9 prereg confirmation, verdict BOTH).
    # Imported from the optimizer so the value has ONE definition (see
    # sequential_update.LAMBDA_L1_DEFAULT).  configs/estimation.yaml mirrors it.
    'lambda_l1': LAMBDA_L1_DEFAULT,
    'lambda_smooth': 0.0001,
    # VARPRO 배포 2026-08-22. 유일 정의는 sequential_update 이고 여기와
    # configs/estimation.yaml 이 그것을 미러한다 (lambda_l1 과 같은 규약).
    # ⛔`varpro_steps=0` 하나로는 배포 전 거동이 되지 않는다 (2026-08-22 정정).
    # 그 조합은 W4 의 `L-ITER0` 팔이고 배포 전보다 나쁘다 (7/64, +23.34%).
    # 되돌릴 때는 sequential_update.pre_varpro_config() 를 쓸 것 - 스위치는 둘이다.
    'varpro_steps': VARPRO_STEPS_DEFAULT,
    'learned_init_iters': LEARNED_INIT_ITERS_DEFAULT,
    'tracking_steps': 10,
    'tracking_mode': 'w_phi',
    'tracking_lr': 0.005,
    # 'auto' since 2026-07-26 (user decision): GT-free regime router (learned proposal
    # n_selected >= auto_switch_thresh) -> 'learned' or 'v2_konly'.
    # ⚠임계는 2026-08-22 에 8 -> 0 (합성 경로 통일) ⟹ 합성에서는 항상 'learned'.
    #   ★'auto' 를 유지하는 이유는 **실측 게이트가 이 분기 안에 있기 때문**이다.
    # Falls back to 'v2_konly' if the diffusion checkpoint is unavailable, and is FORCED
    # to single-run 'v2_konly' on real (GT-free) input by the real-domain gate.
    # Deployed configuration `auto_v1_m3`, authorised on the fresh reserve-seed 40
    # (tmp/confirm2_2026-07-27/RESULT_CONFIRM2.md §4): mixed_v1 checkpoint + router th=8
    # + guard RETIRED + median-of-3 analytic fallback. Slice means on that set:
    # all 0.0877 / sparse 0.0175 / 5comp 0.0315 / crowded 0.1860 / 2comp 0.0160.
    # Imported from the optimizer so the value has ONE definition (see
    # sequential_update.INIT_METHOD_DEFAULT).  configs/estimation.yaml mirrors it.
    'init_method': INIT_METHOD_DEFAULT,
    # Median-of-k field ensemble on the ANALYTIC fallback of the router only (see
    # sequential_update.MedianEnsembleUpdater). 3 = the pre-registered, confirmed value;
    # 1 disables it. Explicit init_method='v2_konly' is unaffected either way.
    'analytic_ensemble_k': 3,
    # 경로 통일 2026-08-22 — 0 = 라우터가 항상 학습을 고른다(합성 한정).
    # 유일 정의는 sequential_update.AUTO_SWITCH_THRESH_DEFAULT; 여기와 yaml 은 미러다.
    # ⛔실측은 이 값과 무관하게 실측 게이트가 bank_mmse 로 보낸다(W12).
    'auto_switch_thresh': AUTO_SWITCH_THRESH_DEFAULT,
    # What the REAL-DOMAIN gate routes to (2026-07-29 port): 'bank_mmse' = the research
    # estimator behind every published real-sea number, under observation-only
    # parameters, measured BETTER than the old single-v2_konly fallback on all 3
    # frozen real windows (sequential_update.SequentialUpdater._initialize_bank);
    # 'v2_konly' = the pre-port behaviour.
    #   CORRECTION 2026-08-11 (P1 audit): this said "~2x better".  Re-derived ratios
    #   are 1.723/1.392/1.460 -- median 1.46x, geometric mean 1.52x, and NO window
    #   reaches 2x.  The routing decision stands; only the magnitude was inflated.
    #   Do not cite "2x".  (Same correction applied at sequential_update.py:459.)
    'real_estimator': 'bank_mmse',
    # Rolling re-fit period for the bank_mmse route, in seconds.  None (the shipping
    # default, unchanged here) fits ONCE and freezes; a float re-fits whenever that much
    # episode time has passed (sequential_update.py:887).
    #   Exposed 2026-08-20 (D batch).  The implementation has existed since the
    # 2026-07-29 port but appeared in NO config surface -- not configs/, not run.py, not
    # the CLI -- so the only reachable behaviour was the frozen one.  That made it the
    # second instance of the learned_merge_frac pattern: a measured winner that no
    # deployed path could reach.  On the 9 frozen real windows of
    # tmp/bankmmse_port_2026-07-29/rolling_refit.json the frozen arm means 0.9452 and
    # refit@2s means 0.7932 (8/9 windows, 3/3 records lower), and the pre-run adoption
    # protocol (PROTOCOL_ROLLING_REFIT.md) selects R*=2s alone.
    #   That evidence is DESCRIPTIVE, not confirmatory: the real corpus has 5
    # independent records, so the exact one-sided binomial floor is 0.03125 and needs a
    # clean sweep.  Hence exposure only -- the default is NOT changed.
    # ★단일출처: sequential_update.BANK_REFIT_EVERY_S_DEFAULT (2026-08-23 배포, None -> 5.0)
    'bank_refit_every_s': BANK_REFIT_EVERY_S_DEFAULT,
    # 2026-08-23: 비동기 재적합. 기본 False = 종전 거동 비트 동일.
    # True 면 재적합이 배경 스레드로 나가 `update()` 를 붙잡지 않는다
    # (= tmp/B6_BUDGET_2026-08-23/REVERT_2026-08-23.md 가 지목한 선행 조건).
    'bank_refit_async': False,
    # Members of the real-domain bank-MMSE pointwise-median field ensemble (seeds
    # {s, s+1000, s+2000}); 1 disables it.
    'real_bank_ensemble_k': 3,
    'use_dispersion': True,
    'use_confidence': True,
    'adaptive_n': True,
    'prune_threshold': 0.10,
    'grow_residual_threshold': 0.05,
    'max_n': 30,
    'tracking_window': 3,
    'device': 'cpu',
    'seed': 42,
}


def set_seed(seed: int):
    """Set global random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(yaml_path: str) -> dict:
    """Load estimation config from YAML and flatten into optimizer-compatible dict."""
    import yaml
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    cfg = {}
    if 'model' in raw:
        cfg['n_gaussians'] = raw['model'].get('n_gaussians', [3, 8, 15])
        cfg['depth'] = raw['model'].get('depth', 50.0)
    if 'optimizer' in raw:
        cfg['lr'] = raw['optimizer'].get('lr', 0.01)
        # fallback must track DEFAULT_CONFIG, not a third hand-written copy (S9 lesson)
        cfg['lambda_l1'] = raw['optimizer'].get('lambda_l1', DEFAULT_CONFIG['lambda_l1'])
        cfg['lambda_smooth'] = raw['optimizer'].get('lambda_smooth', 0.0001)
        cfg['varpro_steps'] = raw['optimizer'].get(
            'varpro_steps', DEFAULT_CONFIG['varpro_steps'])
        cfg['learned_init_iters'] = raw['optimizer'].get(
            'learned_init_iters', DEFAULT_CONFIG['learned_init_iters'])
        cfg['auto_switch_thresh'] = raw['optimizer'].get(
            'auto_switch_thresh', DEFAULT_CONFIG['auto_switch_thresh'])
        cfg['tracking_steps'] = raw['optimizer'].get('tracking_steps', 10)
        cfg['tracking_mode'] = raw['optimizer'].get('tracking_mode', 'w_phi')
        cfg['tracking_lr'] = raw['optimizer'].get('tracking_lr', 0.005)
        # fallback must track DEFAULT_CONFIG, not a hand-written third value (S9 lesson).
        # ⚠It used to be a bare 'v2' -- which is neither the shipping default ('auto')
        # nor the class fallback ('v2_konly') but a THIRD value, so a yaml carrying an
        # `optimizer:` block without `init_method` would silently downgrade the shipped
        # router.  MEASURED 2026-08-21: unreachable with every config in this repo --
        # configs/estimation.yaml is the only file with an `optimizer:` block and it does
        # set `init_method`, so this change is a no-op on every existing run.
        cfg['init_method'] = raw['optimizer'].get('init_method',
                                                  DEFAULT_CONFIG['init_method'])
        if 'analytic_ensemble_k' in raw['optimizer']:
            cfg['analytic_ensemble_k'] = int(raw['optimizer']['analytic_ensemble_k'])
        if 'real_estimator' in raw['optimizer']:
            cfg['real_estimator'] = str(raw['optimizer']['real_estimator'])
        if 'real_bank_ensemble_k' in raw['optimizer']:
            cfg['real_bank_ensemble_k'] = int(raw['optimizer']['real_bank_ensemble_k'])
        # ⛔2026-08-23 수리 (RV-6 ③): 이 키는 yaml 에 **쓰여 있었지만 로더가 한 번도 읽지 않았다**
        # = 죽은 미러.  그 상태에서는 yaml 로 배포도 되돌림도 불가능하고, 그럼에도 회귀 시험이
        # "yaml 이 미러한다" 를 통과하고 있었다(존재만 봤다).  `None` 을 명시할 수 있어야
        # 되돌림이 yaml 경로로도 가능하므로 `in` 검사로 읽는다.
        if 'bank_refit_every_s' in raw['optimizer']:
            _v = raw['optimizer']['bank_refit_every_s']
            cfg['bank_refit_every_s'] = None if _v is None else float(_v)
    if 'physics' in raw:
        cfg['use_dispersion'] = raw['physics'].get('use_dispersion', True)
        cfg['use_confidence'] = raw['physics'].get('use_confidence', True)
    if 'adaptive' in raw:
        cfg['adaptive_n'] = raw['adaptive'].get('adaptive_n', True)
        cfg['prune_threshold'] = raw['adaptive'].get('prune_threshold', 0.10)
        cfg['grow_residual_threshold'] = raw['adaptive'].get('grow_residual_threshold', 0.05)
        cfg['max_n'] = raw['adaptive'].get('max_n', 30)
    if 'tracking' in raw:
        cfg['tracking_window'] = raw['tracking'].get('tracking_window', 3)
    if 'schedule' in raw:
        cfg['loss_thresholds'] = raw['schedule'].get('loss_thresholds')
        cfg['max_iters_per_stage'] = raw['schedule'].get('max_iters_per_stage')
    if 'device' in raw:
        cfg['device'] = raw['device']
    if 'seed' in raw:
        cfg['seed'] = raw['seed']
    return cfg


def run_estimation(h5_path: str, config: dict = None,
                   verbose: bool = False) -> dict:
    """
    Run full WaveSplat estimation on an episode.

    Args:
        h5_path: path to HDF5 episode file
        config: algorithm configuration dict
        verbose: print progress

    Returns:
        dict with per-frame metrics, timing, and parameters
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    cfg['depth'] = cfg.get('depth', 50.0)

    # Set random seed for reproducibility
    seed = cfg.get('seed', 42)
    if seed is not None:
        set_seed(seed)

    # Load episode
    loader = EpisodeDataLoader(h5_path, device=cfg['device'])
    cfg['depth'] = loader.depth

    if verbose:
        print(f"Episode: {h5_path}")
        print(f"  Sea state: {loader.sea_state}, Hs={loader.Hs_true:.2f}m, "
              f"Tp={loader.Tp_true:.1f}s")
        print(f"  Frames: {loader.n_frames}, "
              f"GT grid: {len(loader.gt_grid_y)}×{len(loader.gt_grid_x)}")

    # Phase 1: Initialization
    init_frames = loader.get_init_window(tp_factor=2.0)
    if verbose:
        print(f"\n[Phase 1] Initialization: {len(init_frames)} frames "
              f"({init_frames[-1].t:.1f}s)")

    # `auto` (shipped default) -> median-of-k analytic-fallback ensemble wrapper; every
    # explicit init_method -> the plain SequentialUpdater, bit-identical to before.
    updater = make_updater(cfg)
    init_result = updater.initialize(
        init_frames,
        x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
        y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
        verbose=verbose,
    )

    # Evaluate initialization on last init frame
    grid_x = loader.gt_grid_x
    grid_y = loader.gt_grid_y
    last_init = init_frames[-1]

    # Detect real data (no GT available): gt_field is None or all zeros
    has_gt = (last_init.gt_field is not None and
              float(torch.abs(last_init.gt_field).max()) > 1e-10)

    pred_init = updater.predict_field(grid_x, grid_y, last_init.t)
    if has_gt:
        init_rmse = rmse(pred_init.cpu().numpy(), last_init.gt_field.cpu().numpy())
        init_nrmse = nrmse(pred_init.cpu().numpy(), last_init.gt_field.cpu().numpy(),
                           loader.Hs_true)
    else:
        init_rmse = float('nan')
        init_nrmse = float('nan')

    if verbose:
        print(f"  Init RMSE={init_rmse:.4f}m, nRMSE={init_nrmse:.4f}")

    # Phase 2: Tracking
    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
    if verbose:
        print(f"\n[Phase 2] Tracking: {len(tracking_frames)} frames")

    frame_results = []
    t_converge = None
    converge_threshold = 0.05  # nRMSE threshold

    def _members_of(u):
        # MedianEnsembleUpdater.model returns MEMBER 0 only; the scored field is the
        # pointwise median over members.  Evaluating at observation points must use the
        # same object or the residual and the field metrics measure different models
        # (CON found this in E-B: 12/288 rows affected).
        return getattr(u, 'members', None) or [u]

    def _predict_at(u, x, y, t):
        with torch.no_grad():
            vals = torch.stack([m.model(x, y, t) for m in _members_of(u)])
            return vals.median(dim=0).values if vals.shape[0] > 1 else vals[0]

    # (S5 2026-08-08) TRAINING-WINDOW residual.  Opt-in via S5_WINDOW=1; with the gate off
    # this block does not execute and the run is bit-identical.
    #
    # S4 measured that seeding the TRUE (kx, ky, w, phi) and then running Phase 1 ends at an
    # observation loss 14.6x worse than the truth's, on 16/16 crowded episodes.  That is
    # consistent with two very different failures and S4 could not separate them, because the
    # loss it reported is averaged over the TRACKING frames while Phase 1 only ever fits
    # `init_frames`:
    #   (1) the optimizer fails to minimise at all -- it would be worse on init_frames too;
    #   (2) it minimises init_frames fine and the solution does not generalise
    #       -- better on init_frames, worse everywhere else (overfitting the window).
    # Scoring the SAME window Phase 1 trained on is the one number that tells them apart.
    # The model is untouched between initialize() and the first update(), so evaluating here
    # measures exactly the Phase-1 endpoint.
    window_resid = float('nan')
    window_n_obs = 0
    if os.environ.get('S5_WINDOW') == '1':
        _se, _cnt = 0.0, 0
        for _f in init_frames:
            if len(_f.x_obs) == 0:
                continue
            _zp = _predict_at(updater, _f.x_obs, _f.y_obs, _f.t)
            _se += float(torch.sum((_zp - _f.z_obs) ** 2).item())
            _cnt += int(_f.z_obs.numel())
        if _cnt:
            window_resid = float(np.sqrt(_se / _cnt))
            window_n_obs = _cnt

    for i, frame in enumerate(tracking_frames):
        # (E-C 2026-08-07) OUT-OF-SAMPLE residual: the model here has seen frames up to
        # i-1 only, so this is the genuine next-frame prediction error -- the quantity a
        # deployment actually has.  E-B measured the residual AFTER update() on the same
        # frame, which is a training residual, and that ambiguity blocked the one bit
        # separating "the objective prefers a worse field" from "training loss prefers a
        # more overfit model".
        _oos = float('nan')
        if len(frame.x_obs) > 0:
            _zp_oos = _predict_at(updater, frame.x_obs, frame.y_obs, frame.t)
            _oos = float(torch.sqrt(torch.mean((_zp_oos - frame.z_obs) ** 2)).item())
        # (E-D 2026-08-07) LEAD-TIME SWEEP.  E-C's out-of-sample residual used a 0.1 s
        # lead = 1.1-1.4% of a wave period, which only detects TEMPORALLY WHITE error:
        # a misplaced (w, phi) advects with its own omega and survives at cos(0.09)=0.996,
        # so it is invisible there.  Evaluating the SAME (un-updated) model further ahead
        # lets that component appear.  If the tracker's residual advantage decays with
        # lead while the floor's does not, the advantage was white noise-fitting; if it
        # persists, the tracker really did capture propagating structure.
        _lead = {}
        for _L in (1, 5, 10, 20):
            _j = i + _L - 1
            if _j < len(tracking_frames):
                _fj = tracking_frames[_j]
                if len(_fj.x_obs) > 0:
                    _zl = _predict_at(updater, _fj.x_obs, _fj.y_obs, _fj.t)
                    _lead[str(_L)] = float(
                        torch.sqrt(torch.mean((_zl - _fj.z_obs) ** 2)).item())

        update_result = updater.update(frame)

        # Evaluate on GT grid
        pred = updater.predict_field(grid_x, grid_y, frame.t)
        pred_np = pred.cpu().numpy()

        if has_gt:
            gt_np = frame.gt_field.cpu().numpy()
            frame_metrics = compute_all_metrics(
                pred_np, gt_np,
                grid_x.cpu().numpy(), grid_y.cpu().numpy(),
                loader.Hs_true,
                obs_x=frame.x_obs.cpu().numpy() if len(frame.x_obs) > 0 else None,
                obs_y=frame.y_obs.cpu().numpy() if len(frame.y_obs) > 0 else None,
            )
        else:
            # Real data mode: no GT metrics, report prediction statistics
            Hs_pred = float(4.0 * np.std(pred_np))
            frame_metrics = {
                'Hs_pred': Hs_pred,
                'pred_std': float(np.std(pred_np)),
                'pred_max': float(np.max(pred_np)),
                'pred_min': float(np.min(pred_np)),
            }

        frame_metrics['elapsed_ms'] = update_result['elapsed_ms']
        frame_metrics['n_pts'] = update_result['n_pts']
        # observation-fit loss: the only quality signal available WITHOUT ground truth,
        # so it is what a deployed multi-restart policy would have to select on.
        frame_metrics['obs_loss'] = update_result['loss']
        # (E-B 2026-08-07) GT-free deployment-measurable residual: the model evaluated
        # at the LiDAR return coordinates versus the returns themselves.  Unweighted and
        # unregularised, unlike `obs_loss`, and scored against the SENSOR rather than the
        # GT surface -- so unlike RMSE_observed it exists in real-data mode too.
        frame_metrics['obs_resid_rms_oos'] = _oos          # (E-C) out-of-sample
        # (S2 2026-08-08) CONTROL-RELEVANT PREVIEW POINTS.  Opt-in via S2_ALPHA=1; with the
        # gate off this block does not execute and the run is bit-identical.  The controller
        # (hydrofoil/controller.py:135) never consumes eta -- it consumes
        #     alpha_wave = -slope - w / U
        # at the foil preview point x_foil + U*L (controller.py:12), over the required
        # horizon 0.1-0.5 s (docs/STUDY_GUIDE_2026-07-06.md:135).  Both quantities are
        # derivatives, so they are evaluated here through an EXPLICIT stencil that the
        # scorer applies identically to the GT components -- the 2026-07-09 caveat was that
        # comparing an analytic GT derivative against a smoothed estimate is not a fair test.
        # Only the four stencil legs are stored; the scorer forms the differences.
        if os.environ.get('S2_ALPHA') == '1':
            _U = float(os.environ.get('S2_SPEED', '11.3'))
            _dx, _dt = 2.0, 0.05
            _pts = {}
            for _xf in (2.0, -2.5):                     # front / rear foil, config.py:67,79
                for _L in (0.1, 0.2, 0.3, 0.5):
                    _xp = _xf + _U * _L
                    _legs = [(_xp - _dx, frame.t), (_xp + _dx, frame.t),
                             (_xp, frame.t - _dt), (_xp, frame.t + _dt)]
                    _xs = torch.tensor([p[0] for p in _legs], dtype=torch.float32,
                                       device=grid_x.device)
                    _ys = torch.zeros_like(_xs)
                    _vals = []
                    for _k in range(4):
                        _vals.append(float(_predict_at(
                            updater, _xs[_k:_k + 1], _ys[_k:_k + 1],
                            _legs[_k][1]).reshape(-1)[0].item()))
                    _pts[f'{_xf}|{_L}'] = _vals
            frame_metrics['alpha_legs'] = _pts
            frame_metrics['alpha_stencil'] = {'dx': _dx, 'dt': _dt, 'U': _U}
        if len(frame.x_obs) > 0:
            _zp = _predict_at(updater, frame.x_obs, frame.y_obs, frame.t)
            frame_metrics['obs_resid_rms'] = float(
                torch.sqrt(torch.mean((_zp - frame.z_obs) ** 2)).item())
        # Radius profile for the observed/unobserved split.  Rx-7 fixed min_dist at the
        # inherited 10 m default and could not test it; at full grid the sign of the
        # tracker effect flips, so a crossover radius must exist between them.
        frame_metrics['lead_resid'] = _lead
        # Shells: additive and local, unlike the disk metrics.  One distance pass, then
        # slice -- the E-C audit showed a disk crossing is an area-weighted balance point
        # (r=75 m masked 44.5% of the grid) and cannot locate the physical sign flip.
        if has_gt and len(frame.x_obs) > 0:
            _gx = grid_x.cpu().numpy(); _gy = grid_y.cpu().numpy()
            _Xg, _Yg = np.meshgrid(_gx, _gy)
            _oxn = frame.x_obs.cpu().numpy(); _oyn = frame.y_obs.cpu().numpy()
            _md = np.full(_Xg.shape, np.inf)
            for _j2 in range(len(_oxn)):
                _md = np.minimum(_md, np.sqrt((_Xg - _oxn[_j2]) ** 2
                                              + (_Yg - _oyn[_j2]) ** 2))
            _edges = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, np.inf]
            _sh, _fr = {}, {}
            for _a, _b in zip(_edges[:-1], _edges[1:]):
                _m = (_md >= _a) & (_md < _b)
                _key = f'{int(_a)}_{"inf" if _b == np.inf else int(_b)}'
                _fr[_key] = float(np.mean(_m))
                _sh[_key] = (float(np.sqrt(np.mean((pred_np[_m] - gt_np[_m]) ** 2)))
                             if np.any(_m) else float('nan'))
            frame_metrics['shell_rmse'] = _sh
            frame_metrics['shell_frac'] = _fr

        if has_gt and len(frame.x_obs) > 0:
            _ox = frame.x_obs.cpu().numpy()
            _oy = frame.y_obs.cpu().numpy()
            for _r in (5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0):
                frame_metrics[f'RMSE_obs_r{int(_r)}'] = rmse_observed(
                    pred_np, gt_np, grid_x.cpu().numpy(), grid_y.cpu().numpy(),
                    _ox, _oy, min_dist=_r)
                frame_metrics[f'RMSE_unobs_r{int(_r)}'] = rmse_unobserved(
                    pred_np, gt_np, grid_x.cpu().numpy(), grid_y.cpu().numpy(),
                    _ox, _oy, min_dist=_r)
        frame_metrics['t'] = frame.t
        frame_results.append(frame_metrics)

        # Check convergence (only with GT)
        if has_gt and t_converge is None and frame_metrics.get('nRMSE', 1.0) < converge_threshold:
            t_converge = frame.t

        if verbose and i % 20 == 0:
            if has_gt:
                print(f"  [t={frame.t:.1f}s] RMSE={frame_metrics['RMSE_z']:.4f}m, "
                      f"nRMSE={frame_metrics['nRMSE']:.4f}, "
                      f"{update_result['elapsed_ms']:.1f}ms")
            else:
                print(f"  [t={frame.t:.1f}s] Hs_pred={frame_metrics['Hs_pred']:.4f}m, "
                      f"n_pts={frame_metrics['n_pts']}, "
                      f"{update_result['elapsed_ms']:.1f}ms")

    # Phase 3 (optional): Amplitude calibration (LS refit)
    refit_result = {}
    if cfg.get('amplitude_refit', False):
        refit_frames = tracking_frames[-5:] if len(tracking_frames) >= 5 \
            else tracking_frames
        if refit_frames:
            refit_result = updater.amplitude_refit(
                refit_frames, verbose=verbose)

            # Re-evaluate last frames with refitted amplitudes
            for fr_result in frame_results[-len(refit_frames):]:
                t_val = fr_result['t']
                matching = [f for f in refit_frames
                            if abs(f.t - t_val) < 1e-6]
                if matching and has_gt:
                    pred = updater.predict_field(grid_x, grid_y, t_val)
                    pred_np = pred.cpu().numpy()
                    gt_np = matching[0].gt_field.cpu().numpy()
                    new_metrics = compute_all_metrics(
                        pred_np, gt_np,
                        grid_x.cpu().numpy(), grid_y.cpu().numpy(),
                        loader.Hs_true,
                        obs_x=matching[0].x_obs.cpu().numpy() if len(matching[0].x_obs) > 0 else None,
                        obs_y=matching[0].y_obs.cpu().numpy() if len(matching[0].y_obs) > 0 else None,
                    )
                    fr_result.update(new_metrics)

    # Aggregate results
    if frame_results:
        avg_ms = np.mean([r['elapsed_ms'] for r in frame_results])
        if has_gt:
            avg_rmse = np.mean([r['RMSE_z'] for r in frame_results])
            avg_nrmse = np.mean([r['nRMSE'] for r in frame_results])
            avg_hs_err = np.mean([r['Hs_error'] for r in frame_results])
        else:
            avg_rmse = avg_nrmse = avg_hs_err = float('nan')
    else:
        avg_rmse = avg_nrmse = avg_ms = avg_hs_err = float('nan')

    summary = {
        'episode': h5_path,
        'sea_state': loader.sea_state,
        'Hs_true': loader.Hs_true,
        'Tp_true': loader.Tp_true,
        # None on the real-domain bank-MMSE route, which has no Gaussian components
        # (see 'bank_*' keys in init_result / 'bank_members' below).
        'n_gaussians': (updater.model.n_gaussians
                        if updater.model is not None else None),
        # WavePrior UNIFIED: which arm the init_method='auto' router chose, and the
        # GT-free signal it used. None for every other init_method (no behaviour change).
        'auto_selected': getattr(updater, '_auto_selected', None),
        'auto_n_selected': getattr(updater, '_auto_n_selected', None),
        # True when the real-domain gate bypassed the router (GT-free input); None on
        # every non-'auto' path. See sequential_update.SequentialUpdater.initialize.
        'auto_real_gate': getattr(updater, '_auto_real_gate', None),
        # Members actually run by the median-of-k analytic ensemble (1 = no ensemble).
        # NOTE: 'n_gaussians' and 'spectrum_params' above describe MEMBER 0 only -- the
        # ensemble has no single component set, its output is defined pointwise in the
        # field. Per-member counts are reported here instead of a meaningless aggregate.
        'ensemble_k': getattr(updater, 'ensemble_k', 1),
        'ensemble_member_n': init_result.get('member_n'),
        # Real-domain bank-MMSE route only (None otherwise): member count of the
        # internal pointwise-median field ensemble and per-member GT-free val misfit.
        'bank_members': init_result.get('bank_members'),
        'bank_val_misfit': init_result.get('bank_val_misfit'),
        'init_iters': init_result['n_iters'],
        # ★VARPRO 배포 2026-08-22 — **처치가 실제로 발화했는지를 결과 행에 남긴다.**
        # 저장소가 반복해서 물린 결함이 '안 돌아가는 코드를 연구한 것'이었다
        # (누적 26축). 추론이 아니라 카운터로 확인할 수 있어야 한다.
        'varpro_steps': init_result.get('varpro_steps'),
        'varpro_fired': bool(init_result.get('varpro')),
        'varpro_obj_before': (init_result.get('varpro') or {}).get('vp_obj_before'),
        'varpro_obj_after': (init_result.get('varpro') or {}).get('vp_obj_after'),
        'varpro_k_shift': (init_result.get('varpro') or {}).get('vp_k_shift'),
        'varpro_wall_s': (init_result.get('varpro') or {}).get('vp_wall_s'),
        'init_time_s': init_result['elapsed_s'],
        # ★2026-08-23 (A2 심사): `initialize()` 가 `loss_history` 를 **반환하는데 아무도
        # 읽지 않았다** — `RMSE_near`(08-19) · `avg_shell_rmse`(08-22) 에 이은 **세 번째**
        # "계산되는데 저장되지 않는" 사례다. 전체 이력은 크므로 **세 요약값**만 싣는다:
        # 시작·끝·개선비. ⟹ *"옵티마이저가 실제로 내려갔는가"* 가 추론이 아니라 **행의 수**가 된다.
        # ⚠이 키들은 **초기화 목적함수**이지 장 오차가 아니다(W12 가 그 둘이 갈릴 수 있음을 쟀다).
        'init_loss_first': (lambda h: float(h[0]) if h else None)(
            init_result.get('loss_history') or []),
        'init_loss_last': (lambda h: float(h[-1]) if h else None)(
            init_result.get('loss_history') or []),
        'init_loss_drop_frac': (lambda h: (float(h[0]) - float(h[-1])) / float(h[0])
                                if h and float(h[0]) != 0.0 else None)(
            init_result.get('loss_history') or []),
        'init_loss_n': len(init_result.get('loss_history') or []),
        # ⛔**실측(2026-08-23): 출하 경로에서 이 넷은 전부 `None`/`0` 이다.**  08-22 배포가
        # `learned_init_iters` 를 500 -> **0** 으로 바꿔 결합 경사하강이 **한 스텝도 돌지 않기**
        # 때문이다 ⟹ 손실 이력이 애초에 비어 있다.  A2 심사의 *"키 세 개면 기여 ④가 추론에서
        # 측정이 된다"* 는 **출하 경로에는 적용되지 않는다**(pre-2026-08-22 구성에서만 찬다).
        # ★출하 경로에서 같은 질문("목적함수가 실제로 내려갔는가")에 답하는 것은
        #   위의 `varpro_obj_before`/`varpro_obj_after` 이고, 그것은 이미 저장된다.
        'init_rmse': init_rmse,
        'init_nrmse': init_nrmse,
        # (S5) Phase-1 training-window residual; NaN unless S5_WINDOW=1.
        'window_resid': window_resid,
        'window_n_obs': window_n_obs,
        'tracking_frames': len(frame_results),
        # ★2026-08-23 (B-13): **채점 창을 행에 싣는다.**  옛 헤드라인을 죽인 1순위 사유가
        # *"두 팔이 같은 프레임에서 채점된 적이 없다"* 인데, 새 헤드라인(`B-1`)은 8일 떨어진
        # 두 배치 비교이면서 **양쪽 다 프레임 수를 기록하지 않아** 재발 여부를 기록만으로는
        # 판정할 수 없었다(확인하는 데 새 런이 필요했다 — 그 자체가 계측 결함이다).
        # ⟹ 다음 배치는 이 검증을 **런 없이** 할 수 있어야 한다.
        'scored_t_first': (float(frame_results[0]['t'])
                           if frame_results and 't' in frame_results[0] else None),
        'scored_t_last': (float(frame_results[-1]['t'])
                          if frame_results and 't' in frame_results[-1] else None),
        'episode_total_frames': int(getattr(loader, 'n_frames', 0)) or None,
        'avg_RMSE_z': avg_rmse,
        'avg_nRMSE': avg_nrmse,
        'avg_Hs_error': avg_hs_err,
        # (E1 2026-08-19) Near-field aggregate.  evaluation/metrics.py:433 has
        # been filling RMSE_near on every GT frame all along, and run.py:380
        # keeps it inside frame_results -- but nothing was ever aggregated out
        # of it, and main() strips 'frame_results' before json.dump (:626-627).
        # Every saved result file was therefore structurally blind to the near
        # field: near-range claims were not weak, they were undetectable.
        'avg_RMSE_near': (float(np.mean([r['RMSE_near'] for r in frame_results]))
                          if (frame_results and has_gt) else float('nan')),
        # ★2026-08-22 — `shell_rmse`/`shell_frac` 도 **같은 결함**을 앓고 있었다:
        # :412-413 이 프레임마다 채우는데 요약이 집계하지 않고 main() 이 frame_results 를
        # 지운 뒤 덤프한다 ⟹ 저장된 결과 파일이 **거리 분해에 구조적으로 눈이 멀어 있었다**
        # (RMSE_near 가 46일간 당한 것과 정확히 같은 부류; 2026-08-19 에 그것만 고쳤다).
        # 2026-08-22 검수(V4)가 W11/W13/W14 네 배치에서 shell 키 **0회**를 확인해 재발을 잡았다.
        # ⚠접두어 주의: 이 셸은 **최근접 관측점까지의 거리**(`obs:`)이지 배-거리가 아니다.
        #   같은 표본에서 두 양이 1.8-4.1배 갈린다.
        'avg_shell_rmse': ({k: float(np.nanmean([r['shell_rmse'][k]
                                                 for r in frame_results
                                                 if 'shell_rmse' in r]))
                            for k in frame_results[0]['shell_rmse']}
                           if (frame_results and has_gt
                               and 'shell_rmse' in frame_results[0]) else None),
        'shell_frac': (frame_results[0]['shell_frac']
                       if (frame_results and has_gt
                           and 'shell_frac' in frame_results[0]) else None),
        'avg_inference_ms': avg_ms,
        # GT-free run-quality signal, for multi-restart selection (see
        # tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md): mean observation-fit loss.
        'avg_obs_loss': float(np.mean([r['obs_loss'] for r in frame_results
                                       if np.isfinite(r.get('obs_loss', np.nan))]))
        if frame_results else float('nan'),
        'T_converge': t_converge,
        'frame_results': frame_results,
        'spectrum_params': (updater.model.get_spectrum_params()
                            if updater.model is not None else None),
        'refit': refit_result,
    }

    if verbose:
        print(f"\n[Summary]")
        print(f"  Avg RMSE_z = {avg_rmse:.4f}m")
        print(f"  Avg nRMSE  = {avg_nrmse:.4f}")
        print(f"  Avg Hs_err = {avg_hs_err:.4f}")
        print(f"  Avg speed  = {avg_ms:.1f}ms/frame")
        if t_converge:
            print(f"  T_converge = {t_converge:.1f}s")

    return summary


def run_estimation_multi(episode_path, config=None, seeds=(42, 1, 2, 3, 4), verbose=False):
    """Run `run_estimation` once per seed and report median + spread.

    WHY THIS EXISTS. A fixed episode with a fixed config still gives a **bimodal** spread of
    outcomes, with coefficient of variation up to 53% (measured; see
    tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md), because component phases are initialised at
    random in two places. The spread is *larger* on the Nyquist-adequate regenerated dataset,
    not smaller (tmp/MIN_DK_RESULT_2026-07-23.md). A single-seed number is therefore one draw
    from a wide, two-cluster distribution -- not an estimate of the method's performance.

    Selecting the *best* run is not an option: a pre-registered test of GT-free selection on
    the observation-fit loss FAILED (tmp/RESTART_SELECTION_RESULT_2026-07-23.md), and picking
    on ground truth is not deployable. So the honest summary is the **median with spread**.

    Returns the per-seed results plus median/mean/sd/min/max of avg_nRMSE. Report the median
    and the spread together; a median quoted without its spread is not citable.
    """
    import numpy as _np
    runs = []
    for s in seeds:
        cfg = dict(config or {})
        cfg['seed'] = s
        r = run_estimation(episode_path, cfg, verbose=verbose)
        runs.append({'seed': s, 'avg_nRMSE': float(r['avg_nRMSE']),
                     'avg_obs_loss': float(r.get('avg_obs_loss', float('nan')))})
    a = _np.array([r['avg_nRMSE'] for r in runs], float)
    return {
        'runs': runs,
        'n_seeds': len(runs),
        'nRMSE_median': float(_np.median(a)),
        'nRMSE_mean': float(a.mean()),
        'nRMSE_sd': float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        'nRMSE_min': float(a.min()),
        'nRMSE_max': float(a.max()),
        'nRMSE_cv_pct': float(100 * a.std(ddof=1) / a.mean()) if len(a) > 1 and a.mean() else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description='WaveSplat inference')
    parser.add_argument('--episode', type=str, required=True,
                        help='Path to HDF5 episode file')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to estimation YAML config')
    parser.add_argument('--n_gaussians', type=int, nargs='+', default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lambda_l1', type=float, default=None)
    parser.add_argument('--lambda_smooth', type=float, default=None)
    parser.add_argument('--varpro_steps', type=int, default=None)
    parser.add_argument('--auto_switch_thresh', type=int, default=None,
                        help='0 = router always picks learned (2026-08-22 unify); 8 = pre-unify')
    parser.add_argument('--learned_init_iters', type=int, default=None,
                        help='0 = VARPRO deploy default; pre-deploy = 500 with --varpro_steps 0')
    parser.add_argument('--tracking_steps', type=int, default=None)
    parser.add_argument('--tracking_mode', type=str, default=None,
                        choices=['phi_only', 'w_phi', 'all'],
                        help='Which params to update during tracking')
    parser.add_argument('--tracking_lr', type=float, default=None,
                        help='Learning rate for tracking phase')
    parser.add_argument('--init_method', type=str, default=None,
                        choices=['v1', 'v2', 'v2_konly', 'relax', 'relax_konly',
                                 'learned', 'auto'],
                        help='Init method: v1/v2/v2_konly/relax/relax_konly/learned/auto '
                             '(learned = cond-diffusion set-proposal, WavePrior Stage-1; '
                             'auto = GT-free regime router that picks learned vs v2_konly '
                             'from the proposal n_selected, WavePrior UNIFIED)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--n_seeds', type=int, default=None,
                        help='Run this many seeds and report median+spread instead of a '
                             'single run. Required for citable per-episode numbers: '
                             'run-to-run CV reaches 53%% (see '
                             'tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md).')
    parser.add_argument('--adaptive_n', type=str, default=None,
                        choices=['true', 'false'],
                        help='Enable adaptive N (prune weak, grow if needed)')
    parser.add_argument('--max_n', type=int, default=None,
                        help='Maximum number of Gaussians for adaptive N')
    parser.add_argument('--tracking_window', type=int, default=None,
                        help='Number of recent frames for multi-frame tracking')
    parser.add_argument('--neural_corrector', type=str, default=None,
                        help='Path to neural corrector checkpoint (.pt)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path for results')
    parser.add_argument('--bank_refit_every_s', type=str, default=None,
                        help="rolling re-fit period (s) for the bank_mmse real-domain "
                             "route; 'none' forces the frozen fit, omit to keep the "
                             "shipping default (frozen as of 2026-08-23).  Measured "
                             "recommendation is 5.0 -- see "
                             "sequential_update.recommended_real_refit_config()")
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Build config: YAML base → CLI overrides
    config = {}
    if args.config:
        config = load_yaml_config(args.config)
    # CLI flags override YAML
    if args.n_gaussians is not None:
        config['n_gaussians'] = args.n_gaussians
    if args.lr is not None:
        config['lr'] = args.lr
    if args.lambda_l1 is not None:
        config['lambda_l1'] = args.lambda_l1
    # Without this merge the flag would parse and then be silently dropped -- a dead CLI
    # flag, the same defect class as the unexposed key it is here to fix.
    if args.bank_refit_every_s is not None:
        # ⛔2026-08-23 (RV-6 ③): 종전에는 `type=float` 이라 **CLI 로 None 을 줄 방법이 없었다**
        # = 배포된 값을 명령줄에서 끌 수 없었다.  'none'/'null' 을 받아 그 구멍을 막는다.
        _raw = str(args.bank_refit_every_s).strip().lower()
        config['bank_refit_every_s'] = (None if _raw in ('none', 'null', '')
                                        else float(args.bank_refit_every_s))
    if args.lambda_smooth is not None:
        config['lambda_smooth'] = args.lambda_smooth
    if args.varpro_steps is not None:
        config['varpro_steps'] = args.varpro_steps
    if args.auto_switch_thresh is not None:
        config['auto_switch_thresh'] = args.auto_switch_thresh
    if args.learned_init_iters is not None:
        config['learned_init_iters'] = args.learned_init_iters
    if args.tracking_steps is not None:
        config['tracking_steps'] = args.tracking_steps
    if args.tracking_mode is not None:
        config['tracking_mode'] = args.tracking_mode
    if args.tracking_lr is not None:
        config['tracking_lr'] = args.tracking_lr
    if args.init_method is not None:
        config['init_method'] = args.init_method
    if args.seed is not None:
        config['seed'] = args.seed
    if args.adaptive_n is not None:
        config['adaptive_n'] = args.adaptive_n == 'true'
    if args.max_n is not None:
        config['max_n'] = args.max_n
    if args.tracking_window is not None:
        config['tracking_window'] = args.tracking_window
    if args.neural_corrector is not None:
        config['neural_corrector_path'] = args.neural_corrector

    if args.n_seeds:
        seeds = [42, 1, 2, 3, 4, 5, 6, 7][:args.n_seeds] if args.n_seeds <= 8 \
            else [42] + list(range(1, args.n_seeds))
        result = run_estimation_multi(args.episode, config, seeds=seeds,
                                      verbose=args.verbose)
        print(f"\n=== median-of-{result['n_seeds']} ===")
        print(f"  nRMSE median {result['nRMSE_median']:.4f}  "
              f"(sd {result['nRMSE_sd']:.4f}, CV {result['nRMSE_cv_pct']:.1f}%, "
              f"range {result['nRMSE_min']:.4f}-{result['nRMSE_max']:.4f})")
        print("  report the median WITH the spread; a median alone is not citable.")
    else:
        result = run_estimation(args.episode, config, verbose=args.verbose)

    if args.output:
        # Convert numpy arrays for JSON serialization
        save_result = {k: v for k, v in result.items()
                       if k not in ('frame_results', 'spectrum_params')}
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(save_result, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
