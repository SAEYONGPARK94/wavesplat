"""
Harden the hybrid (step 1, attempt 2) — cross-validation residual gate.

DPS-internal signals failed (DPS over-spreads to ~12 atoms regardless of true
order). A signal that *concentrates* on simple seas: fit a LOW-ORDER parametric
model to a training subset of the window and predict held-out observations. On an
identifiable sea a low-order model predicts held-out points well (low CV residual);
on a complex multidirectional sea it cannot (high CV residual). Fully deployable.

Gate: CV residual < thr -> identifiable -> ST-DF-GP, else -> grid-free DPS.

Run: python3 scripts/d13_cv_gate.py
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from d6_integrated_eval import gather_W, W, TI, tag_to_file

G = 9.81
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def beamform_peaks(P, z, depth, K, klim=0.6, n=64):
    """Top-K dispersion-constrained beamform peaks (kx,ky) with non-max suppression."""
    x = torch.tensor(P[:, 0], dtype=torch.float32, device=DEV)
    y = torch.tensor(P[:, 1], dtype=torch.float32, device=DEV)
    t = torch.tensor(P[:, 2], dtype=torch.float32, device=DEV)
    zt = torch.tensor(z, dtype=torch.float32, device=DEV)
    ax = torch.linspace(-klim, klim, n, device=DEV)
    KX, KY = torch.meshgrid(ax, ax, indexing='xy')
    kx = KX.reshape(-1); ky = KY.reshape(-1)
    k = torch.sqrt(kx**2 + ky**2 + 1e-9)
    om = torch.sqrt(G * k * torch.tanh(k * depth) + 1e-9)
    psi = kx[:, None]*x[None] + ky[:, None]*y[None] - om[:, None]*t[None]
    re = (zt[None]*torch.cos(psi)).sum(1); im = (zt[None]*torch.sin(psi)).sum(1)
    p = (re**2 + im**2).reshape(n, n).cpu().numpy()
    # greedy peak pick with NMS radius
    pk = []
    pp = p.copy(); axn = ax.cpu().numpy(); rad = 4
    for _ in range(K):
        i = np.unravel_index(np.argmax(pp), pp.shape)
        pk.append((axn[i[1]], axn[i[0]]))
        pp[max(0,i[0]-rad):i[0]+rad+1, max(0,i[1]-rad):i[1]+rad+1] = 0
    return np.array(pk)


def cv_residual(P, z, depth, K, seed=0):
    rng = np.random.default_rng(seed)
    m = len(z); idx = rng.permutation(m); ntr = int(0.8*m)
    tr, te = idx[:ntr], idx[ntr:]
    pk = beamform_peaks(P[tr], z[tr], depth, K)
    obs_tr = np.column_stack([P[tr,0], P[tr,1], P[tr,2], z[tr]]).astype(np.float32)
    A, phi = fit_amplitudes_phases(pk[:,0], pk[:,1], obs_tr, depth, lambda_reg=0.5)
    # predict held-out
    kmag = np.sqrt(pk[:,0]**2 + pk[:,1]**2 + 1e-12)
    om = np.sqrt(G*kmag*np.tanh(kmag*depth))
    psi = (pk[:,0][None]*P[te,0][:,None] + pk[:,1][None]*P[te,1][:,None]
           - om[None]*P[te,2][:,None] + phi[None])
    pred = (A[None]*np.cos(psi)).sum(1)
    return float(np.sqrt(np.mean((pred - z[te])**2)) / (np.std(z[te]) + 1e-9))


def main():
    integ = json.load(open('results/direction3/integrated_eval.json'))['rows']
    dps = {r['tag']: r['dps_bma'] for r in integ}; gp = {r['tag']: r['stdfgp_fused'] for r in integ}
    rows = []
    print(f"{'tag':12s} {'nc':>3} {'cv@6':>7} {'cv@10':>7}")
    for tag in dps:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth
        P, z, _ = gather_W(L, TI, W)
        cv6 = np.mean([cv_residual(P, z, depth, 6, s) for s in range(3)])
        cv10 = np.mean([cv_residual(P, z, depth, 10, s) for s in range(3)])
        nc = int(tag.split('_')[1][:-1])
        rows.append(dict(tag=tag, ncomp=nc, cv6=float(cv6), cv10=float(cv10),
                         dps=dps[tag], gp=gp[tag]))
        print(f"{tag:12s} {nc:3d} {cv6:7.3f} {cv10:7.3f}")
    nc = np.array([r['ncomp'] for r in rows]); d = np.array([r['dps'] for r in rows])
    g = np.array([r['gp'] for r in rows])
    print("\nmedians 5c/15c/30c:")
    for key in ['cv6','cv10']:
        v = np.array([r[key] for r in rows])
        print(f"  {key}: {np.median(v[nc==5]):.3f} / {np.median(v[nc==15]):.3f} / {np.median(v[nc==30]):.3f}")
    print("\ndeployable CV gate (cv < thr -> GP, else DPS):")
    best_overall = None
    for key in ['cv6','cv10']:
        v = np.array([r[key] for r in rows]); best = (None, 9.9)
        for thr in np.unique(v):
            use_gp = v <= thr; hyb = np.where(use_gp, g, d).mean()
            if hyb < best[1]: best = (thr, hyb, use_gp.sum(), (nc[use_gp]<=5).sum())
        print(f"  {key}: hybrid {best[1]:.4f}  (thr={best[0]:.3f}, #GP={best[2]}, 5c-correct={best[3]}/{best[2]})")
        if best_overall is None or best[1] < best_overall[1]: best_overall = (key,)+best
    print(f"\n  ST-DF-GP {g.mean():.4f} | DPS {d.mean():.4f} | oracle-cond 0.1735 | oracle-min {np.minimum(d,g).mean():.4f}")
    json.dump(dict(rows=rows, best=dict(signal=best_overall[0], thr=float(best_overall[1]),
              hybrid=float(best_overall[2]))), open('results/direction3/cv_gate.json','w'), indent=2)
    print("saved results/direction3/cv_gate.json")


if __name__ == '__main__':
    main()
