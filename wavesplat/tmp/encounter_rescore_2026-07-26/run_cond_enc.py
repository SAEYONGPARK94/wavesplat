"""Learned arm (cond set-diffusion + polish) — per-window encounter forecast. PREREG.md sec 4.

Reuses d3_beamform_cond_diffusion verbatim; the ONLY change vs run_cond_crowded.eval_episode is
that the render target is the 6 encounter points x 4 leads (plus a full-grid anchor at lead 0)
instead of the nowcast GT grid. Checkpoint read-only.
"""
import os, sys, json, time, argparse
import warnings; warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp/encounter_rescore_2026-07-26')

import numpy as np
import torch
import d3_beamform_cond_diffusion as cd
from d3_setdiff_posterior import denorm_atoms
from wavesplat.model.tnse import render_wave_field
from common import EpisodeGeom, NB_COND, LEADS, DISTS, path_of, all_jobs, CELLS

SEEDS = (0, 1, 2, 3, 4)


def _fit_and_render(x0, stats, P, z, depth, xs, y_c, times,
                    rel_thresh=0.03, max_comp=30):
    """atoms_to_field's body, rendered at many (grid, t) targets from ONE amplitude fit."""
    from wavesplat.model.tnse import fit_amplitudes_phases
    kx, ky, a = denorm_atoms(x0, stats)
    kx, ky, a = kx.cpu().numpy(), ky.cpu().numpy(), a.cpu().numpy()
    if a.max() <= 0:
        return None, None
    keep = a > rel_thresh * a.max()
    kxi, kyi, ai = kx[keep], ky[keep], a[keep]
    if len(kxi) == 0:
        return None, None
    if len(kxi) > max_comp:
        order = np.argsort(-ai)[:max_comp]
        kxi, kyi = kxi[order], kyi[order]
    obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z]).astype(np.float32)
    A, phi = fit_amplitudes_phases(kxi, kyi, obs, depth, lambda_reg=0.5)
    out = [render_wave_field(kxi, kyi, A, phi, depth, xs, np.array([y_c]),
                             float(t)).reshape(-1) for t in times]
    return np.array(out), (kxi, kyi, A, phi)


def eval_one(model, stats, abar, betas, cell, ep, i_ref, sample_seed, nsamp=4):
    from wavesplat.data_loader import EpisodeDataLoader
    G = EpisodeGeom(cell, ep)
    tg = G.targets(i_ref)
    L = EpisodeDataLoader(path_of(cell, ep))
    depth = L.depth
    P, z = [], []
    for i in range(i_ref - NB_COND, i_ref + 1):
        f = L[i]
        P.append(np.column_stack([f.x_obs.numpy(), f.y_obs.numpy(),
                                  np.full(len(f.x_obs), float(f.t))]))
        z.append(f.z_obs.numpy())
    P, z = np.vstack(P), np.concatenate(z)
    if len(z) > cd.M_OBS:
        idx = np.random.default_rng(0).choice(len(z), cd.M_OBS, replace=False)
        P, z = P[idx], z[idx]
    t0 = P[:, 2].min()
    Ps = P.copy(); Ps[:, 2] -= t0
    Bmap = cd._bmap_from_obs(Ps, z)
    X = cd.sample(model, Bmap, abar, betas, n=nsamp, seed=int(sample_seed))
    Pt = torch.tensor(Ps, device=cd.DEVICE, dtype=torch.float32)
    zt = torch.tensor(z, device=cd.DEVICE, dtype=torch.float32)
    times = [tg['t_ref'] + float(Ld) - t0 for Ld in LEADS]
    fields, maps = [], []
    for b in range(nsamp):
        xb = cd.polish_atoms(X[b], stats, Pt, zt, depth, iters=80)
        f2, comp = _fit_and_render(xb, stats, Ps, z, depth, tg['xs'], tg['y_c'], times)
        if f2 is not None:
            fields.append(f2)
            kxi, kyi, A, phi = comp
            maps.append(render_wave_field(kxi, kyi, A, phi, depth, G.gx, G.gy,
                                          tg['t_ref'] - t0))
    if not fields:
        return None
    bma = np.mean(fields, axis=0)                       # (nlead, ndist)
    bmap = np.mean(maps, axis=0)
    gtg = G.eta_grid(G.gx, G.gy, tg['t_ref'])
    return dict(arm='cond', cell=cell, episode=ep, i_ref=i_ref, seed=int(sample_seed),
                t_ref=tg['t_ref'], x_edge=tg['x_edge'], y_c=tg['y_c'],
                pred=bma.tolist(), gt=tg['gt'].tolist(),
                map_rmse=float(np.sqrt(np.mean((bmap - gtg) ** 2))),
                map_rms0=float(np.sqrt(np.mean(gtg ** 2))),
                n_win_obs=tg['n_win_obs'], n_samples_ok=len(fields))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='tmp/encounter_rescore_2026-07-26/cond_enc.jsonl')
    ap.add_argument('--ckpt', default='results/direction3/beamform_cond_diffusion.pt')
    ap.add_argument('--cells', nargs='+', default=CELLS)
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    a = ap.parse_args()

    stats = torch.load(cd.PRIOR_CKPT, map_location=cd.DEVICE)['stats']
    betas = cd.cosine_betas(cd.T).to(cd.DEVICE)
    abar = torch.cumprod(1 - betas, 0)
    model = cd.CondSetDenoiser().to(cd.DEVICE)
    model.load_state_dict(torch.load(a.ckpt, map_location=cd.DEVICE))
    model.eval()

    jobs = [(c, ep, i, s) for (c, ep, i) in all_jobs(a.cells) for s in a.seeds]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                d = json.loads(line)
                if d.get('err') is None:
                    done.add((d['cell'], d['episode'], d['i_ref'], d['seed']))
            except Exception:
                pass
    jobs = [j for j in jobs if j not in done]
    print(f"cond encounter arm on {cd.DEVICE}: {len(jobs)} jobs ({len(done)} done)", flush=True)
    t0 = time.time(); n = 0
    with open(a.out, 'a', buffering=1) as f:
        for (c, ep, i, s) in jobs:
            tt = time.time()
            try:
                row = eval_one(model, stats, abar, betas, c, ep, i, s)
                err = None if row is not None else 'no_valid_sample'
            except Exception as e:
                row, err = None, repr(e)
            if row is None:
                row = dict(arm='cond', cell=c, episode=ep, i_ref=i, seed=s,
                           pred=None, gt=None)
            row['err'] = err
            row['sec'] = time.time() - tt
            f.write(json.dumps(row) + '\n'); n += 1
            print(f"[{n:4d}/{len(jobs)}] {c:5s} {ep[14:]:26s} i{i} s{s} "
                  f"{'ERR' if err else 'ok'} ({row['sec']:.1f}s)", flush=True)
    print(f"cond encounter arm done: {n} runs, {time.time()-t0:.0f}s wall", flush=True)


if __name__ == '__main__':
    main()
