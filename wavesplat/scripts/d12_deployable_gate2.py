"""
Harden the hybrid (step 1) — search for a DEPLOYABLE identifiability gate.

Re-run DPS over the 30 episodes and capture candidate gate signals computed purely
from observations/estimates (no true components):
  - sig_post : mean posterior std / Hs over the grid (calibrated UQ from pilot b)
  - cond_est : cond(FIM) built from the DPS max-lik sample's significant atoms
  - n_sig    : effective model order = # significant atoms (energy > 10% max)
We then check which signal separates 5c from 15c/30c and reproduces the oracle
hybrid (0.174 with gate cond_true; gate sends identifiable→GP, else→DPS).

Run: python3 scripts/d12_deployable_gate2.py
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser, denorm_atoms
from d6_integrated_eval import gather_W, W, TI, tag_to_file
import d1_crb as C1

DEVICE = DP.DEVICE
CKPT = 'results/direction3/setdiff_prior_v2.pt'
NSAMP = 6


def cond_from_atoms(kx, ky, a, P, depth):
    """cond(FIM) from estimated significant atoms (deployable identifiability)."""
    keep = a > 0.1 * a.max()
    kxs, kys, ws = kx[keep], ky[keep], np.sqrt(np.clip(a[keep], 0, None))
    n = len(kxs)
    if n < 2:
        return 1.0, n
    theta = torch.tensor(np.stack([ws, kxs, kys, np.zeros(n)], 1), dtype=C1.DT, device=C1.DEVICE)
    x = torch.tensor(P[:, 0], dtype=C1.DT, device=C1.DEVICE)
    y = torch.tensor(P[:, 1], dtype=C1.DT, device=C1.DEVICE)
    t = torch.tensor(P[:, 2], dtype=C1.DT, device=C1.DEVICE)
    F = C1.fisher(theta, x, y, t, depth, 0.02)
    return float(torch.linalg.cond(F)), n


def main():
    ck = torch.load(CKPT, map_location=DEVICE)
    stats = ck['stats']
    model = SetDenoiser(**ck.get('arch', {})).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    abar = ck['abar'].to(DEVICE); betas = ck['betas'].to(DEVICE)

    integ = json.load(open('results/direction3/integrated_eval.json'))['rows']
    dps_bma = {r['tag']: r['dps_bma'] for r in integ}
    gp = {r['tag']: r['stdfgp_fused'] for r in integ}

    rows = []
    print(f"{'tag':12s} {'nc':>3} {'sig_post':>8} {'cond_est':>10} {'n_sig':>5}")
    for tag in dps_bma:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        frames = L.get_tracking_frames(2.0)
        P, z, _ = gather_W(L, TI, W)
        tgt = frames[TI + 1]
        target = {'gx': gx, 'gy': gy, 't': float(tgt.t), 'gt': tgt.gt_field.numpy(), 'Hs': Hs}
        Xd = DP.dps_sample(model, stats, abar, betas, P, z, depth, guidance=0.1, n=NSAMP, seed=0)
        fields, lls = [], []
        Pt = torch.tensor(P, dtype=torch.float32, device=DEVICE)
        zt = torch.tensor(z, dtype=torch.float32, device=DEVICE)
        eye = torch.eye(len(z), device=DEVICE)
        atoms = []
        for b in range(NSAMP):
            kx, ky, a = denorm_atoms(Xd[b], stats)
            f, _ = DP.atoms_to_field(Xd[b], stats, P, z, depth, target)
            ll = float(DP.loglik_atoms(kx, ky, torch.clamp(a, min=0), Pt, zt, depth, eye))
            fields.append(f if f is not None else np.zeros_like(target['gt'])); lls.append(ll)
            atoms.append((kx.cpu().numpy(), ky.cpu().numpy(), a.cpu().numpy()))
        Fst = np.stack(fields)
        sig_post = float(np.mean(Fst.std(0)) / Hs)
        kx, ky, a = atoms[int(np.argmax(lls))]
        cond_est, n_sig = cond_from_atoms(kx, ky, a, P, depth)
        nc = int(tag.split('_')[1][:-1])
        rows.append(dict(tag=tag, ncomp=nc, sig_post=sig_post, cond_est=cond_est,
                         n_sig=n_sig, dps=dps_bma[tag], gp=gp[tag]))
        print(f"{tag:12s} {nc:3d} {sig_post:8.4f} {cond_est:10.2e} {n_sig:5d}")

    nc = np.array([r['ncomp'] for r in rows]); d = np.array([r['dps'] for r in rows])
    g = np.array([r['gp'] for r in rows])
    print("\nsignal medians by complexity (5c / 15c / 30c):")
    for key in ['sig_post', 'cond_est', 'n_sig']:
        v = np.array([r[key] for r in rows])
        print(f"  {key:9s}: {np.median(v[nc==5]):.3g} / {np.median(v[nc==15]):.3g} / {np.median(v[nc==30]):.3g}")

    # best gate per signal (identifiable -> GP). cond_est/n_sig: low->GP; sig_post: low->GP
    print("\nbest deployable gate per signal (sweep threshold):")
    results = {}
    for key, lowGP in [('sig_post', True), ('cond_est', True), ('n_sig', True)]:
        v = np.array([r[key] for r in rows])
        cands = np.unique(v)
        best = (None, 9.9)
        for thr in cands:
            use_gp = (v <= thr) if lowGP else (v >= thr)
            hyb = np.where(use_gp, g, d).mean()
            if hyb < best[1]:
                best = (thr, hyb, use_gp.sum(), (nc[use_gp] <= 5).sum())
        results[key] = best
        print(f"  {key:9s}: hybrid {best[1]:.4f}  (thr={best[0]:.3g}, #GP={best[2]}, 5c-correct={best[3]}/{best[2]})")
    print(f"\n  ST-DF-GP {g.mean():.4f} | DPS {d.mean():.4f} | oracle-cond hybrid 0.1735 | oracle-min {np.minimum(d,g).mean():.4f}")
    json.dump(dict(rows=rows, gates={k: dict(thr=float(b[0]), hybrid=float(b[1]),
              n_gp=int(b[2]), n5_correct=int(b[3])) for k, b in results.items()}),
              open('results/direction3/deployable_gate2.json', 'w'), indent=2)
    print("saved results/direction3/deployable_gate2.json")


if __name__ == '__main__':
    main()
