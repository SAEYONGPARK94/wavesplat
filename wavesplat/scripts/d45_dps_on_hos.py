"""
#2b — does the learned CONTINUOUS-representation prior (DPS) beat discrete RELAX on
realistic CONTINUOUS-spectrum (HOS) waves?

d44 showed RELAX degrades to ~0.25-0.30 on HOS continuous spectra (vs 0.10-0.16 on the
discrete benchmark) — a representation gap, not nonlinearity. The set-diffusion DPS uses a
continuous atom prior; the over-smoothing that LOST on discrete components may be RIGHT for a
continuous spectrum. Head-to-head on HOS windows: RELAX vs DPS (both linear-physics
estimators, different representations).

Run: python3 scripts/d45_dps_on_hos.py
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from d23_main_results import panel
from d26_relax_iterative import iterative_relax
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser
import d44_hos_nonlinear as H

CKPT = 'results/direction3/setdiff_prior_v2.pt'
# distinct (Hs,Tp) conditions (HOS uses a CONTINUOUS spectrum; n_components is irrelevant)
SEAS = [('BF3', 0.6, 5.5), ('BF4', 1.2, 7.0), ('BF5', 2.0, 9.0)]
SEEDS = [0, 1, 2]


def main():
    ck = torch.load(CKPT, map_location=DP.DEVICE); stats = ck['stats']
    model = SetDenoiser(**ck.get('arch', {})).to(DP.DEVICE); model.load_state_dict(ck['model']); model.eval()
    abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    out = {}; t0 = time.time(); allr, alld = [], []
    for tag, Hs, Tp in SEAS:
        n_back = int(round(2.2 * Tp / 0.1)); rr, dd = [], []
        for seed in SEEDS:
            fr = H.build_window(Hs, Tp, seed, n_back, gx, gy, M=3)      # nonlinear HOS window
            i_star = len(fr) - 2
            target = {'gx': gx, 'gy': gy, 't': fr[i_star]['t'], 'gt': fr[i_star]['gt'],
                      'Hs': float(fr[i_star]['gt'].std() * 4)}
            P = np.vstack([f['P'] for f in fr[:i_star + 1]]); z = np.concatenate([f['z'] for f in fr[:i_star + 1]])
            fld_r, _ = iterative_relax(P, z, H.DEPTH, target, 3)
            nr_relax = panel(fld_r, target['gt'], target['Hs'])['nrmse']
            idx = np.random.default_rng(seed).choice(len(z), min(200, len(z)), replace=False)
            Ps, zs = P[idx], z[idx]
            Xd = DP.dps_sample(model, stats, abar, betas, Ps, zs, H.DEPTH, guidance=0.1, n=8, seed=0)
            flds = [DP.atoms_to_field(Xd[b], stats, Ps, zs, H.DEPTH, target)[0] for b in range(8)]
            flds = [f for f in flds if f is not None]
            bma = np.mean(flds, axis=0) if flds else np.zeros_like(target['gt'])
            nr_dps = panel(bma, target['gt'], target['Hs'])['nrmse']
            rr.append(nr_relax); dd.append(nr_dps); allr.append(nr_relax); alld.append(nr_dps)
            print(f"  [{tag} seed{seed}] RELAX {nr_relax:.3f} | DPS {nr_dps:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        out[tag] = dict(relax_mean=float(np.mean(rr)), relax_std=float(np.std(rr)),
                        dps_mean=float(np.mean(dd)), dps_std=float(np.std(dd)), n=len(SEEDS))
        print(f"[{tag}] RELAX {np.mean(rr):.3f}±{np.std(rr):.3f} | DPS {np.mean(dd):.3f}±{np.std(dd):.3f} "
              f"| {'DPS' if np.mean(dd)<np.mean(rr) else 'RELAX'} wins", flush=True)
    json.dump(out, open('results/direction3/dps_on_hos.json', 'w'), indent=2)
    allr, alld = np.array(allr), np.array(alld)
    wins = int((alld < allr).sum())
    print(f"\nOVERALL (HOS continuous, {len(allr)} windows)  RELAX {allr.mean():.3f}±{allr.std():.3f} | "
          f"DPS {alld.mean():.3f}±{alld.std():.3f}  | DPS wins {wins}/{len(allr)}  "
          f"(−{round(100*(allr.mean()-alld.mean())/allr.mean())}%)")
    # paired difference
    diff = allr - alld
    print(f"paired Δ(RELAX−DPS) mean {diff.mean():+.3f} ± {diff.std()/np.sqrt(len(diff)):.3f} (SE)")
    print("saved results/direction3/dps_on_hos.json")


if __name__ == '__main__':
    main()
