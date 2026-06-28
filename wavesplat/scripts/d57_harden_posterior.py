"""
Harden the amortized posterior — confirm accuracy + UQ calibration on MORE HOS windows
(the d56 result was 6 windows). Generate extra HOS windows, combine with the cache, train the
posterior, and evaluate with error bars + per-condition + F1/RELAX comparison + UQ coverage.

Run: python3 scripts/d57_harden_posterior.py [--steps 8000]
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import d44_hos_nonlinear as H
from d23_main_results import panel
from d26_relax_iterative import iterative_relax
from d51_F1_continuous_density import f1_reconstruct
import d56_amortized_posterior as AP

DEPTH = 50.0
CACHE = 'results/direction3/_hos_windows.npz'
BIG = 'results/direction3/_hos_windows_big.npz'
EXTRA = [('BF3', 0.6, 5.5, 2), ('BF4', 1.2, 7.0, 2), ('BF5', 2.0, 9.0, 2),
         ('BF3', 0.6, 5.5, 3), ('BF4', 1.2, 7.0, 3), ('BF5', 2.0, 9.0, 3)]


def build_big(gx, gy):
    if os.path.exists(BIG):
        return list(np.load(BIG, allow_pickle=True)['windows'])
    wins = list(np.load(CACHE, allow_pickle=True)['windows'])     # 6 cached (seeds 0,1)
    for tag, Hs, Tp, seed in EXTRA:
        n_back = int(round(2.2 * Tp / 0.1))
        fr = H.build_window(Hs, Tp, seed, n_back, gx, gy, M=3)
        i_star = len(fr) - 2
        P = np.vstack([f['P'] for f in fr[:i_star+1]]); z = np.concatenate([f['z'] for f in fr[:i_star+1]])
        wins.append(dict(tag=f'{tag}_s{seed}', P=P, z=z, gt=fr[i_star]['gt'],
                         t=fr[i_star]['t'], Hs=float(fr[i_star]['gt'].std()*4)))
        print(f"  generated {tag}_s{seed}", flush=True)
    np.savez_compressed(BIG, windows=np.array(wins, dtype=object))
    return wins


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=8000); a = ap.parse_args()
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    wins = build_big(gx, gy)
    print(f"\nHOS test windows: {len(wins)}", flush=True)
    print(f"training posterior {a.steps} steps...", flush=True)
    model = AP.train(a.steps); model.eval()
    Xg, Yg = np.meshgrid(gx, gy); grid = np.column_stack([Xg.ravel(), Yg.ravel()])
    rows = {}; errs, sigs = [], []
    for w in wins:
        cond = w['tag'].split('_')[0]
        target = {'gx': gx, 'gy': gy, 't': float(w['t']), 'gt': w['gt'], 'Hs': float(w['Hs'])}
        P, z = w['P'], w['z']
        Pi, zi = (P, z)
        if len(z) > 200:
            i = np.random.default_rng(0).choice(len(z), 200, replace=False); Pi, zi = P[i], z[i]
        ob = torch.tensor(np.column_stack([Pi[:,0], Pi[:,1], Pi[:,2]-Pi[:,2].min(), zi])[None], dtype=torch.float32, device=AP.DEVICE)
        q = torch.tensor(np.column_stack([grid, np.full(len(grid), float(w['t'])-Pi[:,2].min())])[None], dtype=torch.float32, device=AP.DEVICE)
        with torch.no_grad():
            mean, logv = model(ob, q)
        m = mean[0].cpu().numpy().reshape(len(gy), len(gx)); sig = (0.5*logv[0]).exp().cpu().numpy().reshape(len(gy), len(gx))
        nr_p = panel(m, w['gt'], w['Hs'])['nrmse']
        nr_f1 = f1_reconstruct(P, z, target, ng=40, lam=5.0)
        nr_rx = panel(iterative_relax(P, z, DEPTH, target, 3)[0], w['gt'], w['Hs'])['nrmse']
        rows.setdefault(cond, []).append((nr_p, nr_f1, nr_rx))
        errs.append(np.abs(m - w['gt']).ravel()); sigs.append(sig.ravel())
    print("\n=== accuracy by condition (posterior | F1 | RELAX) ===")
    allp, allf, allr = [], [], []
    for cond in sorted(rows):
        v = np.array(rows[cond])
        print(f"  {cond}: posterior {v[:,0].mean():.3f}±{v[:,0].std():.3f} | F1 {v[:,1].mean():.3f} | RELAX {v[:,2].mean():.3f}")
        allp += list(v[:,0]); allf += list(v[:,1]); allr += list(v[:,2])
    allp, allf, allr = np.array(allp), np.array(allf), np.array(allr)
    er = np.concatenate(errs); sg = np.concatenate(sigs)
    s = float(np.quantile(er/(sg+1e-9), 0.68)); cov1 = float((er <= s*sg).mean()); cov2 = float((er <= 2*s*sg).mean())
    print(f"\nOVERALL ({len(allp)} windows)  posterior {allp.mean():.3f}±{allp.std():.3f} | "
          f"F1 {allf.mean():.3f} | RELAX {allr.mean():.3f}")
    print(f"  posterior beats F1 on {int((allp<allf).sum())}/{len(allp)}; paired Δ(F1−post) {np.mean(allf-allp):+.3f}±{np.std(allf-allp)/np.sqrt(len(allp)):.3f}(SE)")
    print(f"  UQ: ±1σ {cov1:.2f} (.68) ±2σ {cov2:.2f} (.95) after s={s:.2f}")
    json.dump(dict(posterior=float(allp.mean()), posterior_std=float(allp.std()), F1=float(allf.mean()),
                   RELAX=float(allr.mean()), n=len(allp), cov1=cov1, cov2=cov2),
              open('results/direction3/posterior_hardened.json', 'w'), indent=2)
    print("saved results/direction3/posterior_hardened.json")


if __name__ == '__main__':
    main()
