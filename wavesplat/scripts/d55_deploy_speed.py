"""
Deployment tradeoff — (accuracy, inference time) at the DEPLOYABLE short window.

Real-time feed-forward control needs short windows (low latency, stationarity-safe) AND fast
inference. d54 showed RELAX only wins with LONG windows (non-deployable). Here, at the short
2.2·Tp window (the operationally-relevant regime), we measure accuracy AND wall-clock
inference time for each method on the cached HOS windows. This makes the deployment tradeoff
concrete: which method gives accuracy + speed + (UQ) for real-time control.

Run: python3 scripts/d55_deploy_speed.py
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from d23_main_results import panel
from d26_relax_iterative import iterative_relax
from d51_F1_continuous_density import f1_reconstruct
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser

CACHE = 'results/direction3/_hos_windows.npz'
CKPT = 'results/direction3/setdiff_prior_v2.pt'
DEPTH = 50.0


def main():
    wins = list(np.load(CACHE, allow_pickle=True)['windows'])
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    ck = torch.load(CKPT, map_location=DP.DEVICE); stats = ck['stats']
    model = SetDenoiser(**ck.get('arch', {})).to(DP.DEVICE); model.load_state_dict(ck['model']); model.eval()
    abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)
    res = {m: {'nr': [], 't': []} for m in ['F1', 'RELAX', 'DPS']}
    # GPU warmup
    _ = DP.dps_sample(model, stats, abar, betas, wins[0]['P'][:50], wins[0]['z'][:50], DEPTH, guidance=0.1, n=2, seed=0)
    for w in wins:
        target = {'gx': gx, 'gy': gy, 't': float(w['t']), 'gt': w['gt'], 'Hs': float(w['Hs'])}
        P, z = w['P'], w['z']
        # F1
        t0 = time.time(); nr = f1_reconstruct(P, z, target, ng=40, lam=5.0); res['F1']['t'].append(time.time()-t0); res['F1']['nr'].append(nr)
        # RELAX
        t0 = time.time(); fr, _ = iterative_relax(P, z, DEPTH, target, 3); res['RELAX']['t'].append(time.time()-t0)
        res['RELAX']['nr'].append(panel(fr, w['gt'], w['Hs'])['nrmse'])
        # DPS (200 obs cap)
        idx = np.random.default_rng(0).choice(len(z), min(200, len(z)), replace=False)
        torch.cuda.synchronize() if DP.DEVICE.type == 'cuda' else None
        t0 = time.time()
        Xd = DP.dps_sample(model, stats, abar, betas, P[idx], z[idx], DEPTH, guidance=0.1, n=8, seed=0)
        flds = [DP.atoms_to_field(Xd[b], stats, P[idx], z[idx], DEPTH, target)[0] for b in range(8)]
        torch.cuda.synchronize() if DP.DEVICE.type == 'cuda' else None
        res['DPS']['t'].append(time.time()-t0)
        flds = [f for f in flds if f is not None]
        res['DPS']['nr'].append(panel(np.mean(flds, axis=0) if flds else np.zeros_like(w['gt']), w['gt'], w['Hs'])['nrmse'])
    print(f"{'method':8s} {'nRMSE':>7} {'time/window':>12} {'UQ?':>5}  (short 2.2Tp window, HOS continuous)")
    uq = {'F1': 'no', 'RELAX': 'no', 'DPS': 'YES'}
    out = {}
    for m in ['F1', 'RELAX', 'DPS']:
        mn, mt = np.mean(res[m]['nr']), np.mean(res[m]['t'])
        out[m] = dict(nrmse=float(mn), time_ms=float(mt*1e3), uq=uq[m])
        print(f"{m:8s} {mn:7.3f} {mt*1e3:10.0f}ms {uq[m]:>5}")
    print("\n  F4 (neural operator, 1-forward): ~5 ms but 0.250 (inaccurate) — fast, no accuracy/UQ")
    print("  → deployable ideal = fast + short-window-accurate + UQ; no single method has all three.")
    json.dump(out, open('results/direction3/deploy_speed.json', 'w'), indent=2)
    print("saved results/direction3/deploy_speed.json")


if __name__ == '__main__':
    main()
