"""
Diagnose why the learned DPS prior does NOT improve with more observations (user flagged
this as suspicious — a normal estimator should improve with data).

Hypothesis: the DPS guidance is NORMALISED per step (x += guidance * grad/||grad||), so the
likelihood's *strength* (which grows with m) is discarded — only its direction is kept,
scaled by a fixed small step. Then more observations cannot pull harder → flat in m.
This is an implementation artifact, not a property of learning.

Test: sweep guidance × m on complex episodes; report BMA and per-sample-BEST nRMSE.
- If higher guidance lets DPS improve with m → the flat-with-data was a guidance artifact.
- If DPS is flat even at high guidance → over-smoothing / prior-dominance is fundamental.
Also reports per-sample-best (isolates BMA over-smoothing).

Run: python3 scripts/d36_dps_diagnose.py
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, panel
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser

CKPT = 'results/direction3/setdiff_prior_v2.pt'
EPISODES = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_15c_s0']
MS = [100, 200, 400, 700]
GUID = [0.1, 0.3, 0.6]
NSAMP = 6


def sub(P, z, m, seed=0):
    if len(z) <= m:
        return P, z
    i = np.random.default_rng(seed).choice(len(z), m, replace=False)
    return P[i], z[i]


def main():
    ck = torch.load(CKPT, map_location=DP.DEVICE)
    stats = ck['stats']
    model = SetDenoiser(**ck.get('arch', {})).to(DP.DEVICE); model.load_state_dict(ck['model']); model.eval()
    abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)

    out = {}
    t0 = time.time()
    for tag in EPISODES:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        Tobs = K_TP * L.Tp_true; n_back = int(round(Tobs / 0.1)); nfr = len(L)
        i_star = min(nfr - 2, n_back + 50)
        fstar = L[int(i_star)]
        target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
        P, z = gather_frames(L, int(i_star), n_back)
        out[tag] = {}
        for g in GUID:
            row = {}
            for m in MS:
                Ps, zs = sub(P, z, m)
                Xd = DP.dps_sample(model, stats, abar, betas, Ps, zs, depth, guidance=g, n=NSAMP, seed=0)
                fields, rs = [], []
                for b in range(NSAMP):
                    f, r = DP.atoms_to_field(Xd[b], stats, Ps, zs, depth, target)
                    rs.append(r)
                    if f is not None:
                        fields.append(f)
                bma = np.mean(fields, axis=0) if fields else np.zeros_like(target['gt'])
                row[m] = dict(bma=panel(bma, target['gt'], Hs)['nrmse'], best=float(min(rs)))
            out[tag][g] = row
            bs = "  ".join(f"m{m}:{row[m]['bma']:.3f}/{row[m]['best']:.3f}" for m in MS)
            print(f"[{tag:11s} g={g:.1f}]  (BMA/best)  {bs}   ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open('results/direction3/dps_diagnose.json', 'w'), indent=2)

    print("\n=== mean over episodes: BMA nRMSE vs m, per guidance ===")
    print(f"{'guid':>5} " + " ".join(f"m{m:>4}" for m in MS) + "   trend(m:100->700)")
    for g in GUID:
        vals = [np.mean([out[t][g][m]['bma'] for t in out]) for m in MS]
        trend = "IMPROVES" if vals[-1] < vals[0] - 0.01 else ("flat" if abs(vals[-1]-vals[0]) <= 0.01 else "WORSENS")
        print(f"{g:>5.1f} " + " ".join(f"{v:.3f}" for v in vals) + f"   {trend}")
    print("\n=== mean: per-sample BEST nRMSE vs m (isolates BMA over-smoothing) ===")
    print(f"{'guid':>5} " + " ".join(f"m{m:>4}" for m in MS))
    for g in GUID:
        vals = [np.mean([out[t][g][m]['best'] for t in out]) for m in MS]
        print(f"{g:>5.1f} " + " ".join(f"{v:.3f}" for v in vals))
    print("saved results/direction3/dps_diagnose.json")


if __name__ == '__main__':
    main()
