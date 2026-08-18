"""PROBE-A (EXPLORATORY, no PREREG, ZERO estimator runs).

Question: on the atom set (kx,ky) the estimator ACTUALLY ENDS WITH, what does an
unpenalised closed-form (a,b) refit -- i.e. `sequential_update.amplitude_refit`, which
solves the SAME linear system as `W3/run_w3.py:run_cvx` -- buy?

Inputs are STORED final spectra from tmp/S4_phase_oracle_2026-08-08/results.jsonl
(fields 'kx','ky' = SpectralGaussianModel.get_spectrum_params()).  No estimator is run;
this script only loads episodes, does one numpy.linalg.lstsq per (cell, episode), and
scores with the repository scorer.

Scoring is byte-identical in form to run_w3.run_cvx: fit on loader.get_init_window(2.0),
score mean nRMSE over loader.get_tracking_frames(2.0) via compute_all_metrics.
"""
import json, os, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))   # .../wavesplat/wavesplat
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from wavesplat.data_loader import EpisodeDataLoader          # noqa: E402
from wavesplat.evaluation.metrics import compute_all_metrics  # noqa: E402

S4 = os.path.join(ROOT, 'tmp', 'S4_phase_oracle_2026-08-08', 'results.jsonl')
CELLS = ('AUTO-NT', 'ORA-K-NT', 'ORA-WPHI-NT', 'ORA-WPHI-0STEP')
G = 9.81


def refit(loader, kx, ky):
    depth = loader.depth
    kmag = np.sqrt(kx ** 2 + ky ** 2 + 1e-12)
    om = np.sqrt(G * kmag * np.tanh(kmag * depth))
    win = loader.get_init_window(tp_factor=2.0)
    X = np.concatenate([f.x_obs.cpu().numpy() for f in win])
    Y = np.concatenate([f.y_obs.cpu().numpy() for f in win])
    Z = np.concatenate([f.z_obs.cpu().numpy() for f in win]).astype(np.float64)
    T = np.concatenate([np.full(len(f.x_obs), f.t) for f in win])
    th = kx[None, :] * X[:, None] + ky[None, :] * Y[:, None] - om[None, :] * T[:, None]
    Phi = np.concatenate([np.cos(th), np.sin(th)], axis=1)
    c, _r, rank, sv = np.linalg.lstsq(Phi, Z, rcond=None)
    n = len(kx)
    a, b = c[:n], c[n:]
    resid = float(np.sqrt(np.mean((Phi @ c - Z) ** 2)))
    cond = float(sv[0] / sv[-1]) if sv.size and sv[-1] > 0 else float('inf')

    gx = loader.gt_grid_x.cpu().numpy(); gy = loader.gt_grid_y.cpu().numpy()
    Xg, Yg = np.meshgrid(gx, gy)
    frames = loader.get_tracking_frames(tp_factor=2.0)
    nr = []
    for f in frames:
        thg = (kx[None, :] * Xg.ravel()[:, None] + ky[None, :] * Yg.ravel()[:, None]
               - om[None, :] * f.t)
        pf = ((a[None, :] * np.cos(thg) + b[None, :] * np.sin(thg))
              .sum(axis=1).reshape(Xg.shape))
        gt = f.gt_field.cpu().numpy()
        x = f.x_obs.cpu().numpy(); y = f.y_obs.cpu().numpy()
        m = compute_all_metrics(pf, gt, gx, gy, loader.Hs_true,
                                obs_x=x if len(x) else None,
                                obs_y=y if len(y) else None)
        nr.append(float(m['nRMSE']))
    return dict(refit_nRMSE=float(np.mean(nr)), refit_win_resid=resid,
                refit_cond=cond, refit_rank=int(rank), n_atoms=n)


def main():
    rows = [json.loads(l) for l in open(S4)]
    rows = [r for r in rows if 'error' not in r and r['cell'] in CELLS]
    seed = min(r['seed'] for r in rows)
    rows = [r for r in rows if r['seed'] == seed]
    print('S4 seed used: %d   rows: %d' % (seed, len(rows)), flush=True)
    out = open(os.path.join(HERE, 'probe_postlasso.jsonl'), 'w')
    cache = {}
    for i, r in enumerate(sorted(rows, key=lambda z: (z['cell'], z['episode'])), 1):
        ep = r['episode']
        if ep not in cache:
            cache[ep] = EpisodeDataLoader(os.path.join(ROOT, ep))
        kx = np.asarray(r['kx'], float); ky = np.asarray(r['ky'], float)
        if kx.size == 0:
            continue
        res = refit(cache[ep], kx, ky)
        rec = dict(cell=r['cell'], episode=ep, regime=r['regime'], seed=seed,
                   stored_nRMSE=r['nRMSE'], stored_N=r['N'], **res)
        out.write(json.dumps(rec) + '\n'); out.flush()
        print('[%3d/%d] %-15s %-70s stored %.5f -> refit %.5f' %
              (i, len(rows), r['cell'], ep.split('/')[-1][:60], r['nRMSE'],
               res['refit_nRMSE']), flush=True)
    out.close()


if __name__ == '__main__':
    main()
