"""
D1 premise test — does DPS improve with MORE observations on CONTINUOUS (HOS) spectra?

On the DISCRETE benchmark DPS was flat in m (d36 — the prior can't represent precise discrete
components). But on a CONTINUOUS spectrum the prior CAN represent the field, so more
observations may help (less noise, tighter posterior) → a scalable likelihood (inducing
points) would widen the win. Test on the CACHED HOS windows (no regeneration).

Run: python3 scripts/d49_dps_obs_continuous.py
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

CKPT = 'results/direction3/setdiff_prior_v2.pt'
CACHE = 'results/direction3/_hos_windows.npz'
CAPS = [100, 200, 400, 800]
DEPTH = 50.0


def main():
    if not os.path.exists(CACHE):
        print("no cached HOS windows — run d47 first"); return
    wins = list(np.load(CACHE, allow_pickle=True)['windows'])
    ck = torch.load(CKPT, map_location=DP.DEVICE); stats = ck['stats']
    model = SetDenoiser(**ck.get('arch', {})).to(DP.DEVICE); model.load_state_dict(ck['model']); model.eval()
    abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    by_cap = {m: [] for m in CAPS}; relax_full = []
    t0 = time.time()
    for w in wins:
        target = {'gx': gx, 'gy': gy, 't': float(w['t']), 'gt': w['gt'], 'Hs': float(w['Hs'])}
        P, z = w['P'], w['z']
        fr, _ = iterative_relax(P, z, DEPTH, target, 3)
        relax_full.append(panel(fr, target['gt'], target['Hs'])['nrmse'])
        for m in CAPS:
            idx = np.random.default_rng(0).choice(len(z), min(m, len(z)), replace=False)
            Xd = DP.dps_sample(model, stats, abar, betas, P[idx], z[idx], DEPTH, guidance=0.1, n=8, seed=0)
            flds = [DP.atoms_to_field(Xd[b], stats, P[idx], z[idx], DEPTH, target)[0] for b in range(8)]
            flds = [f for f in flds if f is not None]
            by_cap[m].append(panel(np.mean(flds, axis=0) if flds else np.zeros_like(w['gt']), w['gt'], w['Hs'])['nrmse'])
        print(f"[{w['tag']}] DPS " + " ".join(f"m{m}:{by_cap[m][-1]:.3f}" for m in CAPS) +
              f" | RELAX-full {relax_full[-1]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print("\n=== DPS nRMSE vs observation budget on CONTINUOUS HOS ===")
    for m in CAPS:
        print(f"  m={m:4d}  {np.mean(by_cap[m]):.3f}")
    print(f"  RELAX(full obs) {np.mean(relax_full):.3f}")
    trend = np.mean(by_cap[CAPS[0]]) - np.mean(by_cap[CAPS[-1]])
    print(f"\n  DPS improves by {trend:+.3f} from m={CAPS[0]} to m={CAPS[-1]}  "
          f"→ {'MORE OBS HELPS on continuous (scalable likelihood = real lever, D1)' if trend > 0.01 else 'flat (prior-limited, D2 representation is the lever)'}")
    json.dump(dict(by_cap={str(m): float(np.mean(v)) for m, v in by_cap.items()},
                   relax_full=float(np.mean(relax_full))),
              open('results/direction3/dps_obs_continuous.json', 'w'), indent=2)
    print("saved results/direction3/dps_obs_continuous.json")


if __name__ == '__main__':
    main()
