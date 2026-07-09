"""Mode-seeking readout on the DPS posterior (Direction #3) — the only learned route to a
SHARP estimate once every point-regressor mean-collapses. Instead of BMA (posterior mean,
which blurs), extract the dominant MODE:
  BMA        = mean of K sample fields (baseline)
  medoid     = the single sample closest to all others (sharp, an actual mode representative)
  clustermean= mean of the largest correlation-cluster of samples (mode-averaged)
Compare on simple (blur-prone) + complex seas. Run: python3 tmp/mode_seek.py
"""
import os, sys, numpy as np, torch
sys.path.insert(0, 'scripts')
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser
from d23_main_results import K_TP, panel
from d15_sequential_fusion import gather_frames
from d6_integrated_eval import tag_to_file
from wavesplat.data_loader import EpisodeDataLoader

CKPT = 'results/direction3/setdiff_prior_long.pt'; GUID = 0.05; NSAMP = 16; MCAP = 3000
ck = torch.load(CKPT, map_location=DP.DEVICE); stats = ck['stats']
model = SetDenoiser(**ck['arch']).to(DP.DEVICE); model.load_state_dict(ck['model']); model.eval()
abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)
print(f"ckpt={CKPT} guid={GUID} nsamp={NSAMP}\n{'tag':11s} | {'BMA':>6} {'medoid':>6} {'cluster':>7} {'best':>6}")

TAGS = ['BF3_5c_s0', 'BF4_5c_s0', 'BF4_15c_s0', 'BF4_30c_s0', 'BF5_15c_s0']
rows = []
for tag in TAGS:
    L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
    gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy(); nb = int(round(K_TP * L.Tp_true / 0.1))
    f = L[nb + 3]; tgt = {'gx': gx, 'gy': gy, 't': float(f.t), 'gt': f.gt_field.numpy(), 'Hs': Hs}
    P, z = gather_frames(L, nb + 3, nb)
    if len(z) > MCAP:
        idx = np.random.default_rng(0).choice(len(z), MCAP, replace=False); P, z = P[idx], z[idx]
    Xd = DP.dps_sample(model, stats, abar, betas, P, z, depth, guidance=GUID, n=NSAMP, seed=0)
    flds = np.stack([f_ for b in range(NSAMP)
                     for f_, _ in [DP.atoms_to_field(Xd[b], stats, P, z, depth, tgt)] if f_ is not None])
    gt = tgt['gt']; nr = lambda F: panel(F, gt, Hs)['nrmse']
    bma = nr(flds.mean(0))
    # pairwise L2 distances between sample fields
    K = len(flds); V = flds.reshape(K, -1)
    D = np.sqrt(((V[:, None] - V[None]) ** 2).sum(-1))
    medoid = nr(flds[D.sum(1).argmin()])                          # most central sample
    # correlation clustering: dominant cluster = samples highly correlated with the medoid
    med_i = D.sum(1).argmin()
    C = np.corrcoef(V)
    clust = C[med_i] > 0.6
    clustermean = nr(flds[clust].mean(0)) if clust.sum() >= 1 else bma
    best = min(nr(F) for F in flds)                               # oracle (reference)
    rows.append((tag, bma, medoid, clustermean, best, int(clust.sum())))
    print(f"{tag:11s} | {bma:6.3f} {medoid:6.3f} {clustermean:7.3f} {best:6.3f}   (cluster {clust.sum()}/{K})")

import numpy as np
m = lambda i: np.mean([r[i] for r in rows])
print(f"\nMEAN        | {m(1):6.3f} {m(2):6.3f} {m(3):7.3f} {m(4):6.3f}")
print("RELAX_full simple ~0.05-0.09, complex ~0.10-0.13")
