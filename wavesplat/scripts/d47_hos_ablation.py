"""
#4 ablation — is the DPS win on HOS continuous spectra due to LEARNING, or just atom
capacity / any continuous representation?

Compare on the SAME HOS windows (cached for speed):
  - RELAX, 32 atoms (the discrete baseline)
  - RELAX, 64 atoms (capacity ablation — does more atoms close the gap?)
  - DPS (learned continuous prior)
If RELAX-64 ≈ DPS, the gap is representation capacity (not learning). If RELAX-64 still loses
to DPS, the learned prior structure genuinely helps on continuous spectra.

Run: python3 scripts/d47_hos_ablation.py
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from d23_main_results import panel
from d18_relax_estimator import relax_step
from d17_recursive_estimator import refine_params
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser
import d44_hos_nonlinear as H

CKPT = 'results/direction3/setdiff_prior_v2.pt'
SEAS = [('BF3', 0.6, 5.5), ('BF4', 1.2, 7.0), ('BF5', 2.0, 9.0)]
SEEDS = [0, 1]
CACHE = 'results/direction3/_hos_windows.npz'


def get_windows(gx, gy):
    """Generate (or load cached) HOS windows: list of (tag, P, z, gt, t, Hs)."""
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return list(d['windows'])
    wins = []
    for tag, Hs, Tp in SEAS:
        n_back = int(round(2.2 * Tp / 0.1))
        for seed in SEEDS:
            fr = H.build_window(Hs, Tp, seed, n_back, gx, gy, M=3)
            i_star = len(fr) - 2
            P = np.vstack([f['P'] for f in fr[:i_star + 1]]); z = np.concatenate([f['z'] for f in fr[:i_star + 1]])
            wins.append(dict(tag=f'{tag}_s{seed}', P=P, z=z, gt=fr[i_star]['gt'],
                             t=fr[i_star]['t'], Hs=float(fr[i_star]['gt'].std() * 4)))
            print(f"  generated HOS window {tag}_s{seed}", flush=True)
    np.savez_compressed(CACHE, windows=np.array(wins, dtype=object))
    return wins


def relax_field(P, z, depth, target, max_comp, npass=3):
    kx, ky = np.zeros(0), np.zeros(0); fld = None
    for _ in range(npass):
        kx, ky = relax_step(kx, ky, P, z, depth, max_comp=max_comp)
        kx, ky, fld = refine_params(kx, ky, P, z, depth, target)
    return panel(fld, target['gt'], target['Hs'])['nrmse']


def main():
    ck = torch.load(CKPT, map_location=DP.DEVICE); stats = ck['stats']
    model = SetDenoiser(**ck.get('arch', {})).to(DP.DEVICE); model.load_state_dict(ck['model']); model.eval()
    abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    wins = get_windows(gx, gy)
    rows = []; t0 = time.time()
    for w in wins:
        target = {'gx': gx, 'gy': gy, 't': w['t'], 'gt': w['gt'], 'Hs': w['Hs']}
        P, z = w['P'], w['z']
        r32 = relax_field(P, z, H.DEPTH, target, 32)
        r64 = relax_field(P, z, H.DEPTH, target, 64)
        idx = np.random.default_rng(0).choice(len(z), min(200, len(z)), replace=False)
        Xd = DP.dps_sample(model, stats, abar, betas, P[idx], z[idx], H.DEPTH, guidance=0.1, n=8, seed=0)
        flds = [DP.atoms_to_field(Xd[b], stats, P[idx], z[idx], H.DEPTH, target)[0] for b in range(8)]
        flds = [f for f in flds if f is not None]
        dps = panel(np.mean(flds, axis=0) if flds else np.zeros_like(w['gt']), w['gt'], w['Hs'])['nrmse']
        rows.append(dict(tag=w['tag'], relax32=r32, relax64=r64, dps=dps))
        print(f"[{w['tag']}] RELAX-32 {r32:.3f} | RELAX-64 {r64:.3f} | DPS {dps:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(rows, open('results/direction3/hos_ablation.json', 'w'), indent=2)
    m = lambda k: float(np.mean([r[k] for r in rows]))
    print(f"\nOVERALL  RELAX-32 {m('relax32'):.3f} | RELAX-64 {m('relax64'):.3f} | DPS {m('dps'):.3f}")
    print(f"  → more atoms helps RELAX by {m('relax32')-m('relax64'):+.3f}; DPS still beats RELAX-64 by "
          f"{m('relax64')-m('dps'):+.3f} ({'LEARNING matters' if m('dps')<m('relax64')-0.005 else 'mostly capacity'})")
    print("saved results/direction3/hos_ablation.json")


if __name__ == '__main__':
    main()
