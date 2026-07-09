"""Guidance sweep for the strong _long set-diffusion prior (pick operating guid).
DPS BMA nRMSE + generative-UQ (per-pixel corr(|err|,std)) across guid, 3 seas x 2 frames.
Run: python3 tmp/g_sweep.py
"""
import os, sys, time, json, numpy as np, torch
sys.path.insert(0, 'scripts')
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, panel
from wavesplat.data_loader import EpisodeDataLoader

CKPT = 'results/direction3/setdiff_prior_long.pt'
TAGS = ['BF4_5c_s0', 'BF4_15c_s0', 'BF4_30c_s0']
GUIDS = [0.0, 0.05, 0.1, 0.2, 0.35]
NSAMP = 6
MCAP = 200

ck = torch.load(CKPT, map_location=DP.DEVICE)
stats = ck['stats']; model = SetDenoiser(**ck['arch']).to(DP.DEVICE)
model.load_state_dict(ck['model']); model.eval()
abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)
print(f"ckpt={CKPT} arch={ck['arch']} nsamp={NSAMP} mcap={MCAP}")

# pre-gather obs/target per (tag, frame)
cases = []
for tag in TAGS:
    L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
    gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
    n_back = int(round(K_TP * L.Tp_true / 0.1)); nfr = len(L)
    for i_star in [n_back + 3, (n_back + 3 + nfr - 2) // 2]:
        fstar = L[int(i_star)]
        target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
        P, z = gather_frames(L, int(i_star), n_back)
        idx = np.random.default_rng(0).choice(len(z), min(MCAP, len(z)), replace=False)
        cases.append((tag, depth, target, P[idx], z[idx], Hs))

rows = []
for guid in GUIDS:
    nr, pc, spr = [], [], []
    t0 = time.time()
    for tag, depth, target, Ps, zs, Hs in cases:
        Xd = DP.dps_sample(model, stats, abar, betas, Ps, zs, depth, guidance=guid, n=NSAMP, seed=0)
        flds = [f for b in range(NSAMP)
                for f, _ in [DP.atoms_to_field(Xd[b], stats, Ps, zs, depth, target)] if f is not None]
        bma = np.mean(flds, axis=0) if flds else np.zeros_like(target['gt'])
        nr.append(panel(bma, target['gt'], Hs)['nrmse'])
        if len(flds) >= 2:
            std = np.stack(flds).std(0); err = np.abs(bma - target['gt'])
            s, e = std.ravel(), err.ravel()
            pc.append(float(np.corrcoef(s, e)[0, 1]) if s.std() > 1e-9 else 0.0)
            spr.append(float(std.mean()))
    rows.append(dict(guid=guid, nrmse=float(np.mean(nr)), pix_corr=float(np.mean(pc)) if pc else 0.0,
                     spread=float(np.mean(spr)) if spr else 0.0, per_case=[float(x) for x in nr]))
    print(f"guid={guid:.2f}  BMA nRMSE={np.mean(nr):.3f}  UQ pix_corr={np.mean(pc) if pc else 0:.3f}  "
          f"spread={np.mean(spr) if spr else 0:.3f}  ({time.time()-t0:.0f}s)", flush=True)

best = min(rows, key=lambda r: r['nrmse'])
print(f"\nBEST guid={best['guid']} nRMSE={best['nrmse']:.3f}")
json.dump(dict(ckpt=CKPT, rows=rows, best_guid=best['guid']),
          open('tmp/g_sweep_long.json', 'w'), indent=2)
print("saved tmp/g_sweep_long.json")
