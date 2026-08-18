"""AUDIT_PRO independent re-derivation of W2 PROBE-B (VARPRO). ZERO estimator runs.

Deliberately independent of recon/probe_varpro.py:
  * (a,b) solved with torch.linalg.pinv (NOT the ridge normal equation + solve)
  * nRMSE computed from first principles (rmse/Hs) instead of compute_all_metrics
  * paired stats recomputed from scratch
Same optimiser (Adam lr 2e-3, 400 steps, float64) because the claim IS about that optimiser.
"""
import json, os, sys
import numpy as np
import torch

ROOT = '/home/vexit/wavesplat/wavesplat'
sys.path.insert(0, ROOT); os.chdir(ROOT)
from wavesplat.data_loader import EpisodeDataLoader           # noqa
from wavesplat.evaluation.metrics import compute_all_metrics  # noqa

S4 = os.path.join(ROOT, 'tmp', 'S4_phase_oracle_2026-08-08', 'results.jsonl')
G = 9.81
STEPS = int(os.environ.get('VP_STEPS', '400'))
LR = 2e-3
CELLS = os.environ.get('CELLS', 'AUTO-NT').split(',')
REG = os.environ.get('REG', 'crowded')
OUT = os.environ.get('OUT', '/tmp/claude-1000/-home-vexit-wavesplat/757850ed-2cf4-4f55-89d7-c86f428f0bbd/scratchpad/audit_vp.jsonl')


def win(ld):
    w = ld.get_init_window(tp_factor=2.0)
    X = np.concatenate([f.x_obs.cpu().numpy() for f in w])
    Y = np.concatenate([f.y_obs.cpu().numpy() for f in w])
    Z = np.concatenate([f.z_obs.cpu().numpy() for f in w])
    T = np.concatenate([np.full(len(f.x_obs), f.t) for f in w])
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    return t(X), t(Y), t(Z), t(T)


def loss_pinv(kx, ky, X, Y, T, Z, d):
    km = torch.sqrt(kx ** 2 + ky ** 2 + 1e-12)
    om = torch.sqrt(G * km * torch.tanh(km * d))
    th = kx[None, :] * X[:, None] + ky[None, :] * Y[:, None] - om[None, :] * T[:, None]
    Phi = torch.cat([torch.cos(th), torch.sin(th)], 1)
    Q, R = torch.linalg.qr(Phi, mode='reduced')   # <-- QR, independent of ridge normal-eq
    c = torch.linalg.solve_triangular(R, (Q.transpose(-2, -1) @ Z).unsqueeze(-1),
                                      upper=True).squeeze(-1)
    r = Phi @ c - Z
    return (r ** 2).mean(), c, om


def field_scores(kx, ky, a, b, om, ld):
    """returns (mean nRMSE mine, mean nRMSE via compute_all_metrics, mean RMSE_near)"""
    gx = ld.gt_grid_x.cpu().numpy(); gy = ld.gt_grid_y.cpu().numpy()
    Xg, Yg = np.meshgrid(gx, gy)
    mine, theirs, near = [], [], []
    for f in ld.get_tracking_frames(tp_factor=2.0):
        th = (kx[None, :] * Xg.ravel()[:, None] + ky[None, :] * Yg.ravel()[:, None]
              - om[None, :] * f.t)
        pf = (a[None, :] * np.cos(th) + b[None, :] * np.sin(th)).sum(1).reshape(Xg.shape)
        gt = f.gt_field.cpu().numpy()
        mine.append(float(np.sqrt(np.mean((pf - gt) ** 2)) / max(ld.Hs_true, 1e-6)))
        x = f.x_obs.cpu().numpy(); y = f.y_obs.cpu().numpy()
        m = compute_all_metrics(pf, gt, gx, gy, ld.Hs_true,
                                obs_x=x if len(x) else None, obs_y=y if len(y) else None)
        theirs.append(float(m['nRMSE'])); near.append(float(m['RMSE_near']))
    return float(np.mean(mine)), float(np.mean(theirs)), float(np.mean(near))


def main():
    rows = [json.loads(l) for l in open(S4)]
    rows = [r for r in rows if 'error' not in r and r['cell'] in CELLS]
    seed = min(r['seed'] for r in rows)
    rows = [r for r in rows if r['seed'] == seed and r.get('regime') == REG]
    print('seed', seed, 'n rows', len(rows), flush=True)
    out = open(OUT, 'w')
    for i, r in enumerate(sorted(rows, key=lambda z: (z['cell'], z['episode'])), 1):
        ld = EpisodeDataLoader(os.path.join(ROOT, r['episode']))
        X, Y, Z, T = win(ld)
        k0 = np.stack([np.asarray(r['kx'], float), np.asarray(r['ky'], float)], 1)
        if len(k0) == 0:
            continue
        kx = torch.tensor(k0[:, 0], dtype=torch.float64, requires_grad=True)
        ky = torch.tensor(k0[:, 1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([kx, ky], lr=LR)
        with torch.no_grad():
            f0, c0, om0 = loss_pinv(kx, ky, X, Y, T, Z, ld.depth)
        for _ in range(STEPS):
            opt.zero_grad(); f, _, _ = loss_pinv(kx, ky, X, Y, T, Z, ld.depth)
            f.backward(); opt.step()
        with torch.no_grad():
            f1, c1, om1 = loss_pinv(kx, ky, X, Y, T, Z, ld.depth)
        n = len(k0)
        kxn, kyn = kx.detach().numpy(), ky.detach().numpy()
        m0, t0_, n0 = field_scores(k0[:, 0], k0[:, 1], c0[:n].numpy(), c0[n:].numpy(), om0.numpy(), ld)
        m1, t1_, n1 = field_scores(kxn, kyn, c1[:n].numpy(), c1[n:].numpy(), om1.numpy(), ld)
        rec = dict(cell=r['cell'], episode=r['episode'], regime=r['regime'], n_atoms=n,
                   stored_nRMSE=r['nRMSE'], refit_mine=m0, refit_cam=t0_,
                   vp_mine=m1, vp_cam=t1_, refit_near=n0, vp_near=n1,
                   win_before=float(np.sqrt(f0)), win_after=float(np.sqrt(f1)))
        out.write(json.dumps(rec) + '\n'); out.flush()
        print('[%2d] %-46s refit %.5f -> vp %.5f | near %.4f -> %.4f | win %.5f->%.5f'
              % (i, r['episode'].split('/')[-1][:44], t0_, t1_, n0, n1,
                 rec['win_before'], rec['win_after']), flush=True)
    out.close()


main()
