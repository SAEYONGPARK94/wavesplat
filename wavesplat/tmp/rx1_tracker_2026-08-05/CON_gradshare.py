"""CON probe A — VALUE share vs GRADIENT share vs STEP share of the regularisers
inside the TRACKING loop.  Attacks: "loss 의 77-92% 가 정규화 항" is a composition
share; a term can dominate the objective's value and still steer the optimum barely
at all -- and under Adam the step is scale-free in the gradient, so neither value
nor raw gradient share translates into displacement share.

We measure all three on the same steps:
  value share      = L_reg / L_total
  grad share       = ||dL_reg/dtheta|| / (||dL_data/dtheta|| + ||dL_reg/dtheta||)
  cosine alignment = cos(g_reg, g_total)   -- does the penalty push where Adam goes?
  step share       = ||Adam step under full L - Adam step under L_data only|| / ||full step||
"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, torch

REPO = os.path.abspath('.')
sys.path.insert(0, REPO)
from wavesplat.run import DEFAULT_CONFIG, set_seed
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
import wavesplat.optimizer.initializer as _init

DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
N_GT = {'episode_BF4_15comp_seed0_uniform.h5': 15,
        'episode_BF4_30comp_seed0_uniform.h5': 30,
        'episode_BF5_15comp_seed0_uniform.h5': 15,
        'episode_BF5_30comp_seed0_uniform.h5': 30}


def data_term(upd, frames):
    tot = torch.tensor(0.0); n = 0
    for f in frames:
        if len(f.x_obs) == 0: continue
        r = upd.model(f.x_obs, f.y_obs, f.t) - f.z_obs
        wgt = f.confidence if upd.use_confidence else torch.ones_like(f.confidence)
        tot = tot + (wgt * r ** 2).sum(); n += len(f.x_obs)
    return tot / n if n else tot


def l1_term(upd):
    return upd.lambda_l1 * torch.sum(torch.abs(upd.model.w))


def smooth_term(upd):
    m = upd.model
    if upd.lambda_smooth <= 0 or m.n_gaussians <= 1:
        return torch.tensor(0.0)
    with torch.no_grad():
        idx = torch.argsort(torch.sqrt(m.kx ** 2 + m.ky ** 2))
    ws = m.w[idx]
    return upd.lambda_smooth * torch.sum((ws[1:] - ws[:-1]) ** 2)


def gnorm(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(gs, params)])
    return flat


def run(ep, seed, arm):
    set_seed(seed)
    path = os.path.join(DATA, ep)
    loader = EpisodeDataLoader(path, device='cpu')
    cfg = dict(DEFAULT_CONFIG); cfg.update(
        depth=loader.depth, seed=seed, use_dispersion=True, use_confidence=True,
        adaptive_n=True, tracking_window=3, prune_threshold=0.10, max_n=30)
    if arm == 'oracle_T1f':
        cfg.update(init_method='oracle', n_gaussians=[N_GT[ep]],
                   loss_thresholds=[None], max_iters_per_stage=[900],
                   init_k_lr_scale=0.0)
        _init.ORACLE_COMPONENTS = np.asarray(loader.wave_components)
    else:   # deployed A0
        cfg.update(init_method='v2_konly', n_gaussians=[3, 8, 15],
                   loss_thresholds=[0.1, 0.05, None],
                   max_iters_per_stage=[200, 200, 500])
        _init.ORACLE_COMPONENTS = None
    upd = SequentialUpdater(cfg)
    init_frames = loader.get_init_window(tp_factor=2.0)
    upd.initialize(init_frames,
                   x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
                   y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])))
    _init.ORACLE_COMPONENTS = None

    tr = loader.get_tracking_frames(tp_factor=2.0)
    params = [upd.model.w, upd.model.phi]
    rec = []
    sumw0 = float(torch.sum(torch.abs(upd.model.w)).detach())
    for fi, frame in enumerate(tr):
        upd._last_t = frame.t
        if len(frame.x_obs) == 0: continue
        upd._frame_buffer.append(frame)
        if len(upd._frame_buffer) > upd.tracking_window:
            upd._frame_buffer.pop(0)
        opt = upd.tracking_optimizer
        for step in range(upd.tracking_steps):
            fb = upd._frame_buffer

            def losses():
                return data_term(upd, fb), l1_term(upd), smooth_term(upd)

            Ld, L1, Ls = losses()
            Ltot = Ld + L1 + Ls
            gd = gnorm(Ld, params)
            reg = L1 + Ls
            gr = (gnorm(reg, params) if float(reg) > 0 else torch.zeros_like(gd))
            gt_ = gd + gr
            nd, nr = float(gd.norm()), float(gr.norm())
            cos = float(torch.dot(gr, gt_) / (gr.norm() * gt_.norm() + 1e-30)) if nr > 0 else 0.0
            vals = dict(L_data=float(Ld.detach()), L_l1=float(L1.detach()),
                        L_smooth=float(Ls.detach()),
                        value_share_reg=float((reg / (Ltot + 1e-30)).detach()))

            def snap_opt():
                return {id(p_): {kk: (vv.clone() if torch.is_tensor(vv) else vv)
                                 for kk, vv in opt.state[p_].items()}
                        for p_ in params if p_ in opt.state}

            def restore_opt(st):
                for p_ in params:
                    if id(p_) in st:
                        for kk, vv in st[id(p_)].items():
                            opt.state[p_][kk] = vv.clone() if torch.is_tensor(vv) else vv

            saved = [p_.detach().clone() for p_ in params]
            st0 = snap_opt()

            def trial(use_reg):
                opt.zero_grad()
                a, b, c = losses()
                (a + b + c if use_reg else a).backward()
                opt.step()
                d = torch.cat([(p_.detach() - s_).reshape(-1)
                               for p_, s_ in zip(params, saved)])
                with torch.no_grad():
                    for p_, s_ in zip(params, saved):
                        p_.copy_(s_)
                restore_opt(st0)
                return d

            step_full = trial(True)
            step_data = trial(False)
            opt.zero_grad()
            a, b, c = losses()
            (a + b + c).backward()
            opt.step()

            rec.append(dict(frame=fi, step=step, **vals,
                            gnorm_data=nd, gnorm_reg=nr,
                            grad_share_reg=nr / (nd + nr + 1e-30),
                            cos_reg_total=cos,
                            step_full=float(step_full.norm()),
                            step_delta=float((step_full - step_data).norm()),
                            step_share_reg=float((step_full - step_data).norm()
                                                 / (step_full.norm() + 1e-30))))
        upd.model.clamp_steepness(upd.max_steepness)
    sumw1 = float(torch.sum(torch.abs(upd.model.w)).detach())
    return rec, sumw0, sumw1


if __name__ == '__main__':
    out = {}
    for arm in ['oracle_T1f', 'deployed_A0']:
        for ep in ['episode_BF4_15comp_seed0_uniform.h5',
                   'episode_BF5_30comp_seed0_uniform.h5']:
            rec, s0, s1 = run(ep, 42, arm)
            a = {k: np.array([r[k] for r in rec], float) for k in rec[0] if k not in ('frame','step')}
            key = f'{arm}|{ep}'
            out[key] = {k: [float(np.median(v)), float(np.percentile(v,10)), float(np.percentile(v,90))]
                        for k, v in a.items()}
            out[key]['sum_w_init'] = s0; out[key]['sum_w_final'] = s1
            out[key]['n_steps'] = len(rec)
            print(f'\n=== {key}  ({len(rec)} tracking steps) ===')
            print(f'  sum|w| {s0:.4f} -> {s1:.4f}  ({(s1/s0-1)*100:+.1f}%)')
            for k in ['value_share_reg','grad_share_reg','step_share_reg','cos_reg_total',
                      'L_data','L_l1','L_smooth','gnorm_data','gnorm_reg','step_full','step_delta']:
                m,lo,hi = out[key][k]
                print(f'  {k:20s} median={m:12.6f}  p10={lo:12.6f}  p90={hi:12.6f}')
    json.dump(out, open('tmp/rx1_tracker_2026-08-05/CON_gradshare.json','w'), indent=1)
