"""cond arm — beamform-conditioned set-diffusion + polish nowcast on the crowded (+sparse
control +flagged extra) finemesh episodes. Per PREREG.md: median-of-5 diffusion sampling seeds
(0,1,2,3,4); best-of-N=4-sample BMA + 80-iter polish; /Hs at the same 3 nowcast frames as WS.
Reuses d3_beamform_cond_diffusion verbatim (checkpoint read-only). Writes JSONL per
(episode, seed). Data path is data_finemesh_2026-07-23/ (never data/).
"""
import os, sys, json, time, argparse
import warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
import numpy as np
import torch
import d3_beamform_cond_diffusion as cd   # this os.chdir's to repo root on import

FINE = 'data_finemesh_2026-07-23/'
EXTRA = 'data_finemesh_extras_2026-07-24/'
SEEDS = (0, 1, 2, 3, 4)

CROWDED = [f'episode_BF{bf}_{nc}comp_seed{s}_{arm}.h5'
           for bf in (4, 5) for nc in (15, 30)
           for s in range(5) for arm in ('uniform', 'slope')]
SPARSE = [
    'episode_BF2_1comp_seed0_uniform.h5', 'episode_BF2_1comp_seed0_slope.h5',
    'episode_BF3_1comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_slope.h5',  'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed1_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_slope.h5',  'episode_BF4_5comp_seed1_uniform.h5',
]
EXTRA_EPS = ['episode_BF4_20comp_seed7_uniform.h5']  # flagged DEGRADED, secondary only


def ncomp_of(ep):
    return int(ep.split('comp')[0].split('_')[-1])


def eval_episode(model, stats, abar, betas, path, sample_seed, nsamp=4):
    from wavesplat.data_loader import EpisodeDataLoader
    L = EpisodeDataLoader(path)
    depth, Hs = L.depth, L.Hs_true
    gx, gy = L.gt_grid_x.numpy(), L.gt_grid_y.numpy()
    nb = int(round(cd.K_TP * L.Tp_true / 0.1))
    i0, i1 = nb + 3, len(L) - 2
    sel = np.linspace(i0, i1, 3).astype(int)
    nr_pol = []
    for i_star in sel:
        fstar = L[int(i_star)]
        P, z = cd.gather_frames(L, int(i_star), nb)
        if len(z) > cd.M_OBS:
            idx = np.random.default_rng(0).choice(len(z), cd.M_OBS, replace=False)
            P, z = P[idx], z[idx]
        t0 = P[:, 2].min()
        Ps = P.copy(); Ps[:, 2] -= t0
        Bmap = cd._bmap_from_obs(Ps, z)
        X = cd.sample(model, Bmap, abar, betas, n=nsamp, seed=int(sample_seed))
        target = {'gx': gx, 'gy': gy, 't': float(fstar.t) - t0,
                  'gt': fstar.gt_field.numpy(), 'Hs': Hs}
        Pt = torch.tensor(Ps, device=cd.DEVICE, dtype=torch.float32)
        zt = torch.tensor(z, device=cd.DEVICE, dtype=torch.float32)
        fp = []
        for b in range(nsamp):
            xb = cd.polish_atoms(X[b], stats, Pt, zt, depth, iters=80)
            f2, _ = cd.atoms_to_field(xb, stats, Ps, z, depth, target)
            fp.append(f2)
        bma = np.mean([f for f in fp if f is not None], axis=0) \
            if any(f is not None for f in fp) else None
        nr_pol.append(float(np.sqrt(np.mean((bma - target['gt']) ** 2)) / max(Hs, 1e-6))
                      if bma is not None else 1.0)
    return float(np.mean(nr_pol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=['crowded', 'sparse', 'extra', 'all'], default='all')
    ap.add_argument('--out', default='tmp/wp_crowded_2026-07-25/cond_runs.jsonl')
    ap.add_argument('--ckpt', default='results/direction3/beamform_cond_diffusion.pt')
    a = ap.parse_args()

    stats = torch.load(cd.PRIOR_CKPT, map_location=cd.DEVICE)['stats']
    betas = cd.cosine_betas(cd.T).to(cd.DEVICE)
    abar = torch.cumprod(1 - betas, 0)
    model = cd.CondSetDenoiser().to(cd.DEVICE)
    model.load_state_dict(torch.load(a.ckpt, map_location=cd.DEVICE))
    model.eval()

    eps = []
    if a.set in ('crowded', 'all'):
        eps += [(FINE, e) for e in CROWDED]
    if a.set in ('sparse', 'all'):
        eps += [(FINE, e) for e in SPARSE]
    if a.set in ('extra', 'all'):
        eps += [(EXTRA, e) for e in EXTRA_EPS]
    jobs = [(d, e, s) for (d, e) in eps for s in SEEDS]

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                dd = json.loads(line)
                done.add((dd['dir'], dd['episode'], dd['seed']))
            except Exception:
                pass
    jobs = [j for j in jobs if j not in done]
    print(f"cond arm on {cd.DEVICE}: {len(eps)} episodes, {len(jobs)} (episode,seed) jobs "
          f"({len(done)} done)", flush=True)

    t0 = time.time()
    n = 0
    with open(a.out, 'a', buffering=1) as f:
        for (d, ep, s) in jobs:
            tt = time.time()
            try:
                v = eval_episode(model, stats, abar, betas, d + ep, s)
                err = None
            except Exception as e:
                v = None; err = repr(e)
            row = dict(dir=d, episode=ep, ncomp=ncomp_of(ep), seed=s,
                       cond_polish_nrmse_Hs=v, sec=time.time() - tt, err=err)
            f.write(json.dumps(row) + '\n')
            n += 1
            tag = 'ERR' if err else f"{v:.3f}"
            print(f"[{n:3d}/{len(jobs)}] {ep:40s} s{s} cond {tag} ({row['sec']:.1f}s)",
                  flush=True)
    print(f"cond arm done: {n} runs, {time.time()-t0:.0f}s wall", flush=True)


if __name__ == '__main__':
    main()
