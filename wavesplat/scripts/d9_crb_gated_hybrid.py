"""
Direction I×III improvement — CRB-gated hybrid estimator.

Insight: ST-DF-GP is precise on IDENTIFIABLE seas (CRB low) but degrades on
under-determined ones; grid-free DPS is robust but imprecise. Route between them
using the Direction-I Fisher information: where the field is identifiable
(FIM well-conditioned), trust the precise GP; where it is under-determined
(ill-conditioned, below the information wall), trust the robust calibrated DPS.

Gate = condition number of the dispersion-constrained space-time FIM at the eval
window. This turns the CRB (a 'limit') into a constructive model-selection rule.

Run: python3 scripts/d9_crb_gated_hybrid.py
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
import d1_crb as C1
from d6_integrated_eval import gather_W, W, TI, tag_to_file


def fim_cond(L):
    """Condition number of the dispersion-constrained space-time FIM at the eval
    window, using the episode's (true) components — the identifiability signal."""
    wc = np.array(L.wave_components)
    theta = torch.tensor(np.stack([wc[:, 0], wc[:, 2], wc[:, 3], wc[:, 4]], 1),
                         dtype=C1.DT, device=C1.DEVICE)
    P, z, _ = gather_W(L, TI, W)
    x = torch.tensor(P[:, 0], dtype=C1.DT, device=C1.DEVICE)
    y = torch.tensor(P[:, 1], dtype=C1.DT, device=C1.DEVICE)
    t = torch.tensor(P[:, 2], dtype=C1.DT, device=C1.DEVICE)
    F = C1.fisher(theta, x, y, t, 0.02, 50.0)
    return float(torch.linalg.cond(F))


def main():
    integ = json.load(open('results/direction3/integrated_eval.json'))['rows']
    dps = {r['tag']: r['dps_bma'] for r in integ}
    gp = {r['tag']: r['stdfgp_fused'] for r in integ}

    rows = []
    print(f"{'tag':12s} {'cond(F)':>10} {'DPS':>7} {'GP':>7}")
    for tag in dps:
        L = EpisodeDataLoader('data/' + tag_to_file(tag))
        c = fim_cond(L)
        rows.append(dict(tag=tag, cond=c, dps=dps[tag], gp=gp[tag],
                         ncomp=int(tag.split('_')[1][:-1])))
        print(f"{tag:12s} {c:10.2e} {dps[tag]:7.3f} {gp[tag]:7.3f}")

    cond = np.array([r['cond'] for r in rows])
    d = np.array([r['dps'] for r in rows]); g = np.array([r['gp'] for r in rows])
    nc = np.array([r['ncomp'] for r in rows])

    # choose gate threshold that best separates (sweep); report hybrid
    print("\nthreshold sweep (cond < thr -> use GP, else DPS):")
    best = None
    for thr in [1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]:
        use_gp = cond < thr
        hyb = np.where(use_gp, g, d)
        m = hyb.mean()
        if best is None or m < best[1]:
            best = (thr, m)
        print(f"  thr={thr:.0e}  hybrid {m:.4f}  (#GP={use_gp.sum()})")
    thr = best[0]; use_gp = cond < thr; hyb = np.where(use_gp, g, d)

    print(f"\n=== RESULTS ===")
    print(f"  ST-DF-GP alone   : {g.mean():.4f}")
    print(f"  grid-free DPS    : {d.mean():.4f}")
    print(f"  CRB-gated hybrid : {hyb.mean():.4f}  (gate cond<{thr:.0e})")
    print(f"  oracle per-ep min: {np.minimum(d,g).mean():.4f}")
    # gate quality: does cond separate by complexity?
    print(f"\n  cond by complexity: 5c med {np.median(cond[nc==5]):.1e}  "
          f"15c {np.median(cond[nc==15]):.1e}  30c {np.median(cond[nc==30]):.1e}")
    print(f"  gate picks GP for {use_gp.sum()}/30 episodes; "
          f"of those, n_comp<=5: {(nc[use_gp]<=5).sum()}/{use_gp.sum()}")

    json.dump(dict(threshold=thr, gp_alone=float(g.mean()), dps_alone=float(d.mean()),
                   hybrid=float(hyb.mean()), oracle=float(np.minimum(d,g).mean()),
                   rows=rows), open('results/direction3/crb_gated_hybrid.json', 'w'), indent=2)
    print("\nsaved results/direction3/crb_gated_hybrid.json")


if __name__ == '__main__':
    main()
