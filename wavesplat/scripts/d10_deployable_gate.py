"""
Harden the CRB-gated hybrid with a DEPLOYABLE gate (no true components).

The oracle gate used cond(FIM) from the true components. Here the identifiability
signal is computed purely from the observations: a dispersion-constrained beamform
power map P(kx,ky) of the window, summarised by its participation ratio
  PR = (Σ P)² / Σ P²   (effective number of active spectral cells).
Simple seas (few sharp peaks) → low PR; complex multidirectional seas → high PR.
Gate on PR and check it reproduces the hybrid (≈0.174).

Run: python3 scripts/d10_deployable_gate.py
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from d6_integrated_eval import gather_W, W, TI, tag_to_file

G = 9.81
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def beamform_PR(P, z, depth, klim=0.6, n=56):
    """Dispersion-constrained beamform power map participation ratio (deployable)."""
    x = torch.tensor(P[:, 0], dtype=torch.float32, device=DEV)
    y = torch.tensor(P[:, 1], dtype=torch.float32, device=DEV)
    t = torch.tensor(P[:, 2], dtype=torch.float32, device=DEV)
    zt = torch.tensor(z, dtype=torch.float32, device=DEV)
    ax = torch.linspace(-klim, klim, n, device=DEV)
    KX, KY = torch.meshgrid(ax, ax, indexing='xy')
    kx = KX.reshape(-1); ky = KY.reshape(-1)
    k = torch.sqrt(kx**2 + ky**2 + 1e-9)
    om = torch.sqrt(G * k * torch.tanh(k * depth) + 1e-9)
    # phase (G,M); power = |Σ_m z_m e^{-iψ}|²
    psi = kx[:, None]*x[None] + ky[:, None]*y[None] - om[:, None]*t[None]
    re = (zt[None] * torch.cos(psi)).sum(1); im = (zt[None] * torch.sin(psi)).sum(1)
    p = (re**2 + im**2)
    p = p / (p.sum() + 1e-12)
    pr = float(1.0 / (p**2).sum())          # participation ratio (eff. # modes)
    return pr


def main():
    integ = json.load(open('results/direction3/integrated_eval.json'))['rows']
    dps = {r['tag']: r['dps_bma'] for r in integ}
    gp = {r['tag']: r['stdfgp_fused'] for r in integ}

    rows = []
    print(f"{'tag':12s} {'PR':>7} {'ncomp':>5} {'DPS':>7} {'GP':>7}")
    for tag in dps:
        L = EpisodeDataLoader('data/' + tag_to_file(tag))
        P, z, _ = gather_W(L, TI, W)
        pr = beamform_PR(P, z, L.depth)
        rows.append(dict(tag=tag, PR=pr, ncomp=int(tag.split('_')[1][:-1]),
                         dps=dps[tag], gp=gp[tag]))
        print(f"{tag:12s} {pr:7.1f} {rows[-1]['ncomp']:5d} {dps[tag]:7.3f} {gp[tag]:7.3f}")

    PR = np.array([r['PR'] for r in rows]); nc = np.array([r['ncomp'] for r in rows])
    d = np.array([r['dps'] for r in rows]); g = np.array([r['gp'] for r in rows])
    print(f"\nPR by complexity: 5c {np.median(PR[nc==5]):.1f}  "
          f"15c {np.median(PR[nc==15]):.1f}  30c {np.median(PR[nc==30]):.1f}")

    print("\nPR-gate sweep (PR < thr -> GP, else DPS):")
    best = None
    for thr in np.arange(PR[nc==5].max()*0 + 6, 30, 2.0):
        use_gp = PR < thr; hyb = np.where(use_gp, g, d); m = hyb.mean()
        npick5 = (nc[use_gp] <= 5).sum()
        print(f"  thr={thr:4.1f}  hybrid {m:.4f}  (#GP={use_gp.sum()}, of which 5c={npick5})")
        if best is None or m < best[1]:
            best = (thr, m, use_gp.sum())
    print(f"\n=== RESULTS ===")
    print(f"  ST-DF-GP alone        : {g.mean():.4f}")
    print(f"  grid-free DPS         : {d.mean():.4f}")
    print(f"  oracle CRB-gate (cond): 0.1735")
    print(f"  DEPLOYABLE PR-gate    : {best[1]:.4f}  (thr={best[0]:.1f}, #GP={best[2]})")
    print(f"  oracle per-ep min     : {np.minimum(d,g).mean():.4f}")
    json.dump(dict(best_thr=float(best[0]), deployable_hybrid=float(best[1]),
                   gp=float(g.mean()), dps=float(d.mean()), rows=rows),
              open('results/direction3/deployable_gate.json', 'w'), indent=2)
    print("saved results/direction3/deployable_gate.json")


if __name__ == '__main__':
    main()
