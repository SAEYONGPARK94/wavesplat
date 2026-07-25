"""Config selection on a HELD-OUT SYNTHETIC validation set (frozen rule, PREREG §3).

16 fixed synthetic scenes (seed 12345, n_components in {15,30} to match the test regime, otherwise
the training distribution). Each config AND the current ckpt is run through the FULL FROZEN eval
pipeline (cd.sample n=4 -> cd.polish_atoms 80 -> cd.atoms_to_field BMA field nRMSE). Chosen config =
lower MEDIAN validation field nRMSE. NEVER touches the eval episodes or data/ episode HDF5s.
"""
import os, sys, json
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import torch

sys.path.insert(0, 'scripts')
import d3_beamform_cond_diffusion as cd
torch.set_num_threads(4)

DEVICE = cd.DEVICE
M_OBS = cd.M_OBS; DEPTH = cd.DEPTH; K_TP = cd.K_TP; T = cd.T
NGRID = 24
N_VAL = 16
VAL_SEED = 12345


def make_val_scenes():
    rng = np.random.default_rng(VAL_SEED)
    pool = cd.pool_xy().astype(np.float32)
    gxmin, gxmax = float(pool[:, 0].min()), float(pool[:, 0].max())
    gymin, gymax = float(pool[:, 1].min()), float(pool[:, 1].max())
    gx = np.linspace(gxmin, gxmax, NGRID); gy = np.linspace(gymin, gymax, NGRID)
    from wavesplat.model.tnse import render_wave_field
    scenes = []
    for i in range(N_VAL):
        nc = int(rng.choice([15, 30]))
        Hs = rng.uniform(0.8, 3.2); Tp = rng.uniform(5.0, 11.0)
        dird = rng.uniform(-50, 50); spread = rng.uniform(12, 40); gamma = rng.uniform(1.5, 4.0)
        c = cd.jonswap_components(Hs, Tp, n_components=nc, depth=DEPTH, gamma=gamma,
                                  direction_deg=dird, direction_spread_deg=spread,
                                  seed=int(rng.integers(1e9)))
        kx = np.array([x.kx for x in c], np.float32); ky = np.array([x.ky for x in c], np.float32)
        amp = np.array([x.A for x in c], np.float32); om = np.array([x.omega for x in c])
        ph = np.array([x.phi for x in c], np.float32)
        Tobs = K_TP * Tp
        idx = rng.integers(0, len(pool), M_OBS)
        ox, oy = pool[idx, 0], pool[idx, 1]; ot = rng.uniform(0, Tobs, M_OBS)
        z = cd.elevation(kx, ky, amp, om, ph, ox, oy, ot, DEPTH, nonlinear=False) \
            + rng.normal(0, 0.02, M_OBS)
        P = np.column_stack([ox, oy, ot]).astype(np.float32); z = z.astype(np.float32)
        t_tar = 0.5 * Tobs
        gt = render_wave_field(kx, ky, amp, ph, DEPTH, gx, gy, t_tar)   # (NGRID,NGRID)
        scenes.append(dict(P=P, z=z, gx=gx, gy=gy, t=t_tar, gt=gt, Hs=Hs, nc=nc))
    return scenes


def eval_model(model, stats, abar, betas, scenes, nsamp=4, seed=0):
    model.eval(); nrs = []
    for sc in scenes:
        P = sc['P']; z = sc['z']
        t0 = P[:, 2].min(); Ps = P.copy(); Ps[:, 2] -= t0
        Bmap = cd._bmap_from_obs(Ps, z)
        X = cd.sample(model, Bmap, abar, betas, n=nsamp, seed=seed)
        target = {'gx': sc['gx'], 'gy': sc['gy'], 't': float(sc['t'] - t0),
                  'gt': sc['gt'], 'Hs': sc['Hs']}
        Pt = torch.tensor(Ps, device=DEVICE, dtype=torch.float32)
        zt = torch.tensor(z, device=DEVICE, dtype=torch.float32)
        fp = []
        for b in range(nsamp):
            xb = cd.polish_atoms(X[b], stats, Pt, zt, DEPTH, iters=80)
            f2, _ = cd.atoms_to_field(xb, stats, Ps, z, DEPTH, target)
            fp.append(f2)
        bma = np.mean([f for f in fp if f is not None], axis=0) \
            if any(f is not None for f in fp) else None
        nr = float(np.sqrt(np.mean((bma - target['gt']) ** 2)) / max(sc['Hs'], 1e-6)) \
            if bma is not None else 1.0
        nrs.append(nr)
    return nrs


def main():
    stats = torch.load(cd.PRIOR_CKPT, map_location=DEVICE)['stats']
    betas = cd.cosine_betas(T).to(DEVICE); abar = torch.cumprod(1 - betas, 0)
    scenes = make_val_scenes()
    print(f"val: {N_VAL} synthetic scenes, nc counts {np.bincount([s['nc'] for s in scenes])}",
          flush=True)

    ckpts = {'current': 'results/direction3/beamform_cond_diffusion.pt',
             'A': 'results/direction3/prior_net_e2eft_A.pt',
             'B': 'results/direction3/prior_net_e2eft_B.pt'}
    out = {}
    for name, ck in ckpts.items():
        if not os.path.exists(ck):
            print(f"  {name}: MISSING {ck}, skip", flush=True); continue
        model = cd.CondSetDenoiser().to(DEVICE)
        model.load_state_dict(torch.load(ck, map_location=DEVICE))
        nrs = eval_model(model, stats, abar, betas, scenes)
        out[name] = dict(nrs=nrs, median=float(np.median(nrs)), mean=float(np.mean(nrs)))
        print(f"  {name:8s} val field nRMSE  median {out[name]['median']:.4f}  "
              f"mean {out[name]['mean']:.4f}", flush=True)

    cand = {k: v for k, v in out.items() if k in ('A', 'B')}
    chosen = min(cand, key=lambda k: cand[k]['median']) if cand else None
    out['chosen'] = chosen
    out['note'] = ('current median for reference: '
                   f"{out.get('current', {}).get('median')}")
    print(f"\nCHOSEN CONFIG (lower median val field nRMSE): {chosen}", flush=True)
    json.dump(out, open('tmp/wp_e2e_2026-07-25/val_select.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
