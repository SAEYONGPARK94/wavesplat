"""PROBE-B (EXPLORATORY, no PREREG, ZERO estimator runs).

Question: starting from the atom set (kx,ky) the estimator ACTUALLY ENDS WITH, does
variable projection -- eliminate (a,b) in closed form, descend on k only -- move the
FIELD, or does it only move the window residual?

  f(K) = || Phi(K) c(K) - z ||^2 / M ,   c(K) = (Phi'Phi + eps I)^{-1} Phi' z
  Phi(K) = [cos Theta | sin Theta],  Theta_ij = kx_j x_i + ky_j y_i - omega_j(k) t_i
  omega_j = sqrt(g |k_j| tanh(|k_j| d))                     (dispersion, differentiable)

Gradient by torch autograd through the normal-equation solve (Golub-Pereyra Jacobian is
available analytically but is not needed -- see W2_upper_bound.md section 4.2).

Inputs = stored final spectra from tmp/S4_phase_oracle_2026-08-08/results.jsonl.
No estimator is executed.
"""
import json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from wavesplat.data_loader import EpisodeDataLoader           # noqa: E402
from wavesplat.evaluation.metrics import compute_all_metrics  # noqa: E402

S4 = os.path.join(ROOT, 'tmp', 'S4_phase_oracle_2026-08-08', 'results.jsonl')
G = 9.81
STEPS = int(os.environ.get('VP_STEPS', '400'))
LR = float(os.environ.get('VP_LR', '2e-3'))
EPSR = 1e-8


def _win(loader):
    w = loader.get_init_window(tp_factor=2.0)
    X = np.concatenate([f.x_obs.cpu().numpy() for f in w])
    Y = np.concatenate([f.y_obs.cpu().numpy() for f in w])
    Z = np.concatenate([f.z_obs.cpu().numpy() for f in w])
    T = np.concatenate([np.full(len(f.x_obs), f.t) for f in w])
    return (torch.tensor(X, dtype=torch.float64), torch.tensor(Y, dtype=torch.float64),
            torch.tensor(Z, dtype=torch.float64), torch.tensor(T, dtype=torch.float64))


def _fc(kx, ky, X, Y, T, Z, depth):
    km = torch.sqrt(kx ** 2 + ky ** 2 + 1e-12)
    om = torch.sqrt(G * km * torch.tanh(km * depth))
    th = kx[None, :] * X[:, None] + ky[None, :] * Y[:, None] - om[None, :] * T[:, None]
    Phi = torch.cat([torch.cos(th), torch.sin(th)], 1)
    A = Phi.T @ Phi + EPSR * torch.eye(Phi.shape[1], dtype=torch.float64)
    c = torch.linalg.solve(A, Phi.T @ Z)
    r = Phi @ c - Z
    return (r ** 2).mean(), c, om


def _score(kx, ky, a, b, om, loader):
    gx = loader.gt_grid_x.cpu().numpy(); gy = loader.gt_grid_y.cpu().numpy()
    Xg, Yg = np.meshgrid(gx, gy)
    nr = []
    for f in loader.get_tracking_frames(tp_factor=2.0):
        th = (kx[None, :] * Xg.ravel()[:, None] + ky[None, :] * Yg.ravel()[:, None]
              - om[None, :] * f.t)
        pf = ((a[None, :] * np.cos(th) + b[None, :] * np.sin(th))
              .sum(1).reshape(Xg.shape))
        gt = f.gt_field.cpu().numpy()
        x = f.x_obs.cpu().numpy(); y = f.y_obs.cpu().numpy()
        nr.append(float(compute_all_metrics(pf, gt, gx, gy, loader.Hs_true,
                                            obs_x=x if len(x) else None,
                                            obs_y=y if len(y) else None)['nRMSE']))
    return float(np.mean(nr))


def main():
    rows = [json.loads(l) for l in open(S4)]
    rows = [r for r in rows if 'error' not in r and r['cell'] in ('AUTO-NT', 'ORA-K-NT')]
    seed = min(r['seed'] for r in rows)
    rows = [r for r in rows if r['seed'] == seed]
    out = open(os.path.join(HERE, 'probe_varpro.jsonl'), 'w')
    cache = {}
    for i, r in enumerate(sorted(rows, key=lambda z: (z['cell'], z['episode'])), 1):
        ep = r['episode']
        if ep not in cache:
            cache[ep] = EpisodeDataLoader(os.path.join(ROOT, ep))
        ld = cache[ep]
        X, Y, Z, T = _win(ld)
        gtc = np.asarray(ld.wave_components, float)
        k0 = np.stack([np.asarray(r['kx'], float), np.asarray(r['ky'], float)], 1)
        if len(k0) == 0:
            continue
        kx = torch.tensor(k0[:, 0], dtype=torch.float64, requires_grad=True)
        ky = torch.tensor(k0[:, 1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([kx, ky], lr=LR)
        with torch.no_grad():
            f0, c0, om0 = _fc(kx, ky, X, Y, T, Z, ld.depth)
        for _ in range(STEPS):
            opt.zero_grad()
            f, _c, _o = _fc(kx, ky, X, Y, T, Z, ld.depth)
            f.backward(); opt.step()
        with torch.no_grad():
            f1, c1, om1 = _fc(kx, ky, X, Y, T, Z, ld.depth)
        n = len(k0)
        kxn = kx.detach().numpy(); kyn = ky.detach().numpy()
        nr0 = _score(k0[:, 0], k0[:, 1], c0[:n].numpy(), c0[n:].numpy(), om0.numpy(), ld)
        nr1 = _score(kxn, kyn, c1[:n].numpy(), c1[n:].numpy(), om1.numpy(), ld)
        gk = gtc[:, 2:4]; A = gtc[:, 0]
        dd = lambda kk: float((np.sqrt(((gk[:, None, :] - kk[None, :, :]) ** 2)
                                       .sum(-1)).min(1) * A).sum() / A.sum())
        rec = dict(cell=r['cell'], episode=ep, regime=r['regime'], n_atoms=n,
                   stored_nRMSE=r['nRMSE'], refit_nRMSE=nr0, varpro_nRMSE=nr1,
                   win_rms_before=float(np.sqrt(f0)), win_rms_after=float(np.sqrt(f1)),
                   dk_before=dd(k0), dk_after=dd(np.stack([kxn, kyn], 1)),
                   kmove=float(np.median(np.hypot(kxn - k0[:, 0], kyn - k0[:, 1]))))
        out.write(json.dumps(rec) + '\n'); out.flush()
        print('[%3d/%d] %-10s %-52s refit %.5f -> varpro %.5f | win %.5f->%.5f | dk %.4f->%.4f'
              % (i, len(rows), r['cell'], ep.split('/')[-1][:50], nr0, nr1,
                 rec['win_rms_before'], rec['win_rms_after'], rec['dk_before'],
                 rec['dk_after']), flush=True)
    out.close()


if __name__ == '__main__':
    main()
