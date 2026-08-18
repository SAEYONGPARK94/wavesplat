#!/usr/bin/env python3
"""
B1 — independent reproduction of W2's PROBE-B (VARPRO upper-bound probe).

Written from scratch from (a) the narrative in recon/W2_upper_bound.md, (b) the
package source (wavesplat/model/spectral_gaussian.py, wavesplat/data_loader.py,
wavesplat/evaluation/metrics.py) and (c) the reference closed-form arm
tmp/W3_SOLVER_2026-08-15/run_w3.py::run_cvx.

*** W2's probe scripts (recon/probe_*.py) were NOT read. ***

Estimator runs: 0.  Inputs are the stored final spectra in
tmp/S4_phase_oracle_2026-08-08/results.jsonl plus the episode HDF5 observations.

Usage:
    python3 B1_varpro_independent.py selftest
    python3 B1_varpro_independent.py gate          # harness gate vs W3 ORA-K-CVX
    python3 B1_varpro_independent.py main          # PROBE-A/B reproduction
    python3 B1_varpro_independent.py steps         # step-count sensitivity
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = '/home/vexit/wavesplat/wavesplat'
OUT = os.path.join(ROOT, 'tmp/W4_SOLVER_2026-08-19/recon')
S4 = os.path.join(ROOT, 'tmp/S4_phase_oracle_2026-08-08/results.jsonl')
W3 = os.path.join(ROOT, 'tmp/W3_SOLVER_2026-08-15/results.jsonl')
G = 9.81
SEED_USED = 1557          # W2 says "S4 seed 1557"

torch.set_num_threads(8)
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------- forward model
def omega_np(kx, ky, depth):
    """spectral_gaussian.py:97-100 -> dispersion_omega(|k|, d)."""
    km = np.sqrt(kx ** 2 + ky ** 2 + 1e-12)
    return np.sqrt(G * km * np.tanh(km * depth))


def omega_t(kx, ky, depth):
    km = torch.sqrt(kx ** 2 + ky ** 2 + 1e-12)
    return torch.sqrt(G * km * torch.tanh(km * depth))


def design(kx, ky, om, X, Y, T):
    """Phi = [cos(theta) | sin(theta)], theta = kx*x + ky*y - om*t  (no phi)."""
    th = (kx[None, :] * X[:, None] + ky[None, :] * Y[:, None]
          - om[None, :] * T[:, None])
    return np.concatenate([np.cos(th), np.sin(th)], axis=1)


# ------------------------------------------------------- (1) phase-convention test
def selftest():
    """The model is z = sum w*cos(theta + phi).  Expanding:
           w*cos(theta+phi) = (w cos phi)*cos(theta) + (-w sin phi)*sin(theta)
       so with basis [cos|sin]:  a = w cos phi,  b = -w sin phi
       => w = hypot(a,b),  phi = atan2(-b, a).
    sequential_update.py:1335 installs atan2(b, a); model/tnse.py:216 uses atan2(-b, a).
    Decide numerically."""
    rng = np.random.default_rng(0)
    N, M, depth = 4, 500, 50.0
    kx = rng.uniform(-0.3, 0.3, N)
    ky = rng.uniform(-0.3, 0.3, N)
    w_t = rng.uniform(0.1, 1.0, N)
    phi_t = rng.uniform(-math.pi, math.pi, N)
    X = rng.uniform(10, 250, M)
    Y = rng.uniform(-100, 100, M)
    T = rng.uniform(0, 20, M)
    om = omega_np(kx, ky, depth)

    th = (kx[None, :] * X[:, None] + ky[None, :] * Y[:, None] - om[None, :] * T[:, None])
    z = (w_t[None, :] * np.cos(th + phi_t[None, :])).sum(axis=1)

    Phi = design(kx, ky, om, X, Y, T)
    c, *_ = np.linalg.lstsq(Phi, z, rcond=None)
    a, b = c[:N], c[N:]
    ls_resid = float(np.linalg.norm(Phi @ c - z))

    def recon(phi):
        return (np.hypot(a, b)[None, :] * np.cos(th + phi[None, :])).sum(axis=1)

    r_plus = float(np.sqrt(np.mean((recon(np.arctan2(b, a)) - z) ** 2)))
    r_minus = float(np.sqrt(np.mean((recon(np.arctan2(-b, a)) - z) ** 2)))
    print('[selftest] LS residual ||Ac-z||        = %.3e' % ls_resid)
    print('[selftest] recon RMSE, phi=atan2( b,a) = %.3e   (sequential_update.py:1335)' % r_plus)
    print('[selftest] recon RMSE, phi=atan2(-b,a) = %.3e   (model/tnse.py:216)' % r_minus)
    print('[selftest] max|w_hat - w_true|         = %.3e' % np.abs(np.hypot(a, b) - w_t).max())
    ok = r_minus < 1e-12 < r_plus
    print('[selftest] VERDICT: atan2(-b, a) is correct -> %s' % ok)
    # my own field predictor uses (a,b) directly, so it is phase-convention free;
    # this test only fixes the label used when reporting (w, phi).
    return ok


# ------------------------------------------------------------------ episode cache
_EP = {}


def episode(rel):
    if rel in _EP:
        return _EP[rel]
    from wavesplat.data_loader import EpisodeDataLoader
    L = EpisodeDataLoader(os.path.join(ROOT, rel))
    win = L.get_init_window(tp_factor=2.0)
    trk = L.get_tracking_frames(tp_factor=2.0)
    X = np.concatenate([f.x_obs.numpy() for f in win]).astype(np.float64)
    Y = np.concatenate([f.y_obs.numpy() for f in win]).astype(np.float64)
    Z = np.concatenate([f.z_obs.numpy() for f in win]).astype(np.float64)
    T = np.concatenate([np.full(len(f.x_obs), f.t) for f in win]).astype(np.float64)
    gx = L.gt_grid_x.numpy().astype(np.float64)
    gy = L.gt_grid_y.numpy().astype(np.float64)
    Xg, Yg = np.meshgrid(gx, gy)
    ts = np.array([f.t for f in trk], dtype=np.float64)
    gts = np.stack([f.gt_field.numpy().astype(np.float64) for f in trk])
    d = dict(X=X, Y=Y, Z=Z, T=T, Hs=float(L.Hs_true), depth=float(L.depth),
             xg=Xg.ravel(), yg=Yg.ravel(), shape=Xg.shape, ts=ts, gts=gts,
             gtc=np.asarray(L.wave_components, dtype=np.float64), n_track=len(trk))
    _EP[rel] = d
    return d


# ---------------------------------------------------- closed-form (a,b) + scoring
def fit_ab(ep, kx, ky):
    om = omega_np(kx, ky, ep['depth'])
    Phi = design(kx, ky, om, ep['X'], ep['Y'], ep['T'])
    c, _, rank, sv = np.linalg.lstsq(Phi, ep['Z'], rcond=None)
    n = len(kx)
    resid = float(np.sqrt(np.mean((Phi @ c - ep['Z']) ** 2)))
    return c[:n], c[n:], om, resid, int(rank)


def score_field(ep, kx, ky, a, b, om):
    """avg over tracking frames of RMSE(pred, gt)/Hs  == metrics.nrmse, run_w3:454."""
    thx = kx[None, :] * ep['xg'][:, None] + ky[None, :] * ep['yg'][:, None]   # (P,N)
    nr = []
    for i, t in enumerate(ep['ts']):
        th = thx - om[None, :] * t
        pf = (a[None, :] * np.cos(th) + b[None, :] * np.sin(th)).sum(axis=1)
        pf = pf.reshape(ep['shape'])
        nr.append(float(np.sqrt(np.mean((pf - ep['gts'][i]) ** 2)) / ep['Hs']))
    return float(np.mean(nr))


# --------------------------------------------------------------------- (2) VARPRO
def varpro(ep, kx0, ky0, steps=400, lr=2e-3, ridge=1e-8, trace=()):
    """Eliminate (a,b) in closed form, descend on k only.
       f(K) = ||z - Phi(K) c(K)||^2 / M,  c = (Phi'Phi + eps I)^-1 Phi' z
       autograd through the solve gives the Golub-Pereyra Jacobian."""
    dev = 'cpu'
    X = torch.tensor(ep['X'], dtype=torch.float64, device=dev)
    Y = torch.tensor(ep['Y'], dtype=torch.float64, device=dev)
    Z = torch.tensor(ep['Z'], dtype=torch.float64, device=dev)
    T = torch.tensor(ep['T'], dtype=torch.float64, device=dev)
    depth = ep['depth']
    kx = torch.tensor(np.asarray(kx0, dtype=np.float64), requires_grad=True)
    ky = torch.tensor(np.asarray(ky0, dtype=np.float64), requires_grad=True)
    n = kx.shape[0]
    I = torch.eye(2 * n, dtype=torch.float64) * ridge
    opt = torch.optim.Adam([kx, ky], lr=lr)
    snaps = {}

    def loss_fn():
        om = omega_t(kx, ky, depth)
        th = (kx[None, :] * X[:, None] + ky[None, :] * Y[:, None]
              - om[None, :] * T[:, None])
        Phi = torch.cat([torch.cos(th), torch.sin(th)], dim=1)
        A = Phi.T @ Phi + I
        c = torch.linalg.solve(A, Phi.T @ Z)
        r = Z - Phi @ c
        return (r ** 2).mean(), c

    for s in range(1, steps + 1):
        opt.zero_grad()
        f, _ = loss_fn()
        f.backward()
        opt.step()
        if s in trace:
            snaps[s] = (kx.detach().numpy().copy(), ky.detach().numpy().copy())
    return kx.detach().numpy(), ky.detach().numpy(), snaps


# ------------------------------------------------------------------ orchestration
def s4_rows(seed=SEED_USED):
    rows = [json.loads(l) for l in open(S4)]
    return [r for r in rows if r['seed'] == seed]


def regime(ep_rel):
    return 'crowded' if 'crowded' in ep_rel else 'sparse'


def sign_p_one_sided(wins, n):
    """exact one-sided binomial P(X >= wins | p=0.5)."""
    return float(sum(math.comb(n, i) for i in range(wins, n + 1)) / 2 ** n)


def med_rel(pre, post):
    return float(np.median([(b - a) / a for a, b in zip(pre, post)]))


def gate():
    """Harness gate: my refit on the stored TRUTH k (ORA-WPHI-0STEP rows)
       must reproduce W3's ORA-K-CVX avg_nRMSE on the same episodes."""
    w3 = [json.loads(l) for l in open(W3)]
    cvx = {r['episode']: r['avg_nRMSE'] for r in w3 if r['arm'] == 'ORA-K-CVX'}
    cvxw = {r['episode']: r['cvx_window_resid_rms'] for r in w3 if r['arm'] == 'ORA-K-CVX'}
    out = []
    for r in s4_rows():
        if r['cell'] != 'ORA-WPHI-0STEP':
            continue
        rel = r['episode']
        short = os.path.basename(rel).replace('episode_', '').replace('.h5', '')
        if short not in cvx:
            continue
        ep = episode(rel)
        kx = np.asarray(r['kx'], dtype=np.float64)
        ky = np.asarray(r['ky'], dtype=np.float64)
        a, b, om, resid, _ = fit_ab(ep, kx, ky)
        nr = score_field(ep, kx, ky, a, b, om)
        out.append(dict(ep=short, regime=regime(rel), mine=nr, w3=cvx[short],
                        rel=abs(nr - cvx[short]) / cvx[short],
                        mine_win=resid, w3_win=cvxw[short],
                        n_stored=len(kx), n_gt=len(ep['gtc'])))
        print('  %-34s mine %.6f  W3 %.6f  reldiff %.2e' %
              (short, nr, cvx[short], out[-1]['rel']))
    with open(os.path.join(OUT, 'B1_gate.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    for rg in ('crowded', 'sparse'):
        s = [o for o in out if o['regime'] == rg]
        if s:
            print('[gate] %-8s n=%2d  mine med %.5f  W3 med %.5f  max reldiff %.2e' %
                  (rg, len(s), np.median([o['mine'] for o in s]),
                   np.median([o['w3'] for o in s]), max(o['rel'] for o in s)))
    return out


def main(steps=400, cells=('AUTO-NT', 'ORA-K-NT'), tag='main'):
    res = []
    t0 = time.time()
    for r in s4_rows():
        if r['cell'] not in cells:
            continue
        rel = r['episode']
        ep = episode(rel)
        kx0 = np.asarray(r['kx'], dtype=np.float64)
        ky0 = np.asarray(r['ky'], dtype=np.float64)
        a0, b0, om0, w0, rank0 = fit_ab(ep, kx0, ky0)
        nr_refit = score_field(ep, kx0, ky0, a0, b0, om0)
        kx1, ky1, _ = varpro(ep, kx0, ky0, steps=steps)
        a1, b1, om1, w1, _ = fit_ab(ep, kx1, ky1)
        nr_vp = score_field(ep, kx1, ky1, a1, b1, om1)
        row = dict(cell=r['cell'], episode=os.path.basename(rel), regime=regime(rel),
                   seed=r['seed'], n_atoms=len(kx0), n_gt=len(ep['gtc']),
                   stored_nRMSE=r['nRMSE'], refit_nRMSE=nr_refit, varpro_nRMSE=nr_vp,
                   refit_window_resid=w0, varpro_window_resid=w1,
                   truth_window_resid=None, steps=steps, rank=rank0)
        res.append(row)
        print('%-9s %-8s %-30s stored %.5f refit %.5f varpro %.5f  win %.5f->%.5f' %
              (r['cell'], row['regime'], row['episode'][:30], row['stored_nRMSE'],
               nr_refit, nr_vp, w0, w1), flush=True)
    path = os.path.join(OUT, 'B1_%s.jsonl' % tag)
    with open(path, 'w') as fh:
        for row in res:
            fh.write(json.dumps(row) + '\n')
    print('[main] %d cells, %.1f s -> %s' % (len(res), time.time() - t0, path))
    report(res)
    return res


def report(res):
    print('\n=== B1 independent aggregation (per-episode paired, seed %d) ===' % SEED_USED)
    hdr = '%-9s %-8s %-8s %9s %9s %9s %8s %6s %9s'
    print(hdr % ('cell', 'regime', 'contrast', 'pre_med', 'post_med', 'med_rel',
                 'wins', 'n', 'p_1sided'))
    for cell in sorted(set(r['cell'] for r in res)):
        for rg in ('crowded', 'sparse'):
            s = [r for r in res if r['cell'] == cell and r['regime'] == rg]
            if not s:
                continue
            for name, pre_k, post_k in (
                    ('A:stor>refit', 'stored_nRMSE', 'refit_nRMSE'),
                    ('B:refit>vp', 'refit_nRMSE', 'varpro_nRMSE'),
                    ('B*:stor>vp', 'stored_nRMSE', 'varpro_nRMSE'),
                    ('win:refit>vp', 'refit_window_resid', 'varpro_window_resid')):
                pre = [r[pre_k] for r in s]
                post = [r[post_k] for r in s]
                wins = sum(1 for a, b in zip(pre, post) if b < a)
                print(hdr % (cell, rg, name, '%.5f' % np.median(pre),
                             '%.5f' % np.median(post), '%+.1f%%' % (100 * med_rel(pre, post)),
                             '%d' % wins, '%d' % len(s),
                             '%.2g' % sign_p_one_sided(wins, len(s))))


def steps_sensitivity(cell='AUTO-NT', rg='crowded', ladder=(400, 1000, 2000, 4000)):
    """Does the FIELD metric move between 400 and 4000 steps?  (W2 claims the
       *window residual* is flat; the field metric is the open question.)"""
    res = []
    for r in s4_rows():
        if r['cell'] != cell or regime(r['episode']) != rg:
            continue
        ep = episode(r['episode'])
        kx0 = np.asarray(r['kx'], dtype=np.float64)
        ky0 = np.asarray(r['ky'], dtype=np.float64)
        _, _, snaps = varpro(ep, kx0, ky0, steps=max(ladder), trace=set(ladder))
        row = dict(cell=cell, regime=rg, episode=os.path.basename(r['episode']))
        for s in ladder:
            kx1, ky1 = snaps[s]
            a1, b1, om1, w1, _ = fit_ab(ep, kx1, ky1)
            row['nRMSE_%d' % s] = score_field(ep, kx1, ky1, a1, b1, om1)
            row['win_%d' % s] = w1
        res.append(row)
        print(row, flush=True)
    with open(os.path.join(OUT, 'B1_steps.jsonl'), 'w') as fh:
        for row in res:
            fh.write(json.dumps(row) + '\n')
    print('\n=== step ladder (%s %s, n=%d) ===' % (cell, rg, len(res)))
    for s in ladder:
        print('  %5d steps: median nRMSE %.5f   median window %.5f' %
              (s, np.median([r['nRMSE_%d' % s] for r in res]),
               np.median([r['win_%d' % s] for r in res])))
    base = [r['nRMSE_400'] for r in res]
    for s in ladder[1:]:
        cur = [r['nRMSE_%d' % s] for r in res]
        wins = sum(1 for a, b in zip(base, cur) if b < a)
        print('  400 -> %5d : med rel %+.1f%%  better %d/%d  p1=%.3g' %
              (s, 100 * med_rel(base, cur), wins, len(res),
               sign_p_one_sided(wins, len(res))))
    return res


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if cmd in ('selftest', 'all'):
        selftest()
    if cmd in ('gate', 'all'):
        gate()
    if cmd in ('main', 'all'):
        main()
    if cmd in ('steps', 'all'):
        steps_sensitivity()
