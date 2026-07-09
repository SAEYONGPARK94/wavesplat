"""Is the DPS posterior-MEAN (BMA) the wrong summary? The Woodbury likelihood is now
cheap, so rank the K posterior samples by their GP-marginal OBSERVATION-likelihood
(deployable, no GT) and compare point-estimate rules:
  BMA (mean) | oracle-best (min nRMSE, cheats) | lik-best (max loglik, deployable)
  | lik-weighted (softmax over loglik).
If lik-best ≈ oracle-best, a deployable selection recovers the good mode on simple seas.
Run: python3 tmp/mapselect_test.py
"""
import os, sys, time, json, numpy as np, torch
sys.path.insert(0, 'scripts')
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser, denorm_atoms
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, panel
from wavesplat.data_loader import EpisodeDataLoader

CKPT = 'results/direction3/setdiff_prior_long.pt'
TAGS = ['BF4_5c_s0', 'BF3_5c_s0', 'BF4_15c_s0', 'BF5_15c_s0', 'BF4_30c_s0']
GUIDS = [0.05, 0.1]
NSAMP = 12
MCAP = 3000

ck = torch.load(CKPT, map_location=DP.DEVICE); stats = ck['stats']
model = SetDenoiser(**ck['arch']).to(DP.DEVICE); model.load_state_dict(ck['model']); model.eval()
abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)
print(f"nsamp={NSAMP} mcap={MCAP}\n{'tag':11s} {'guid':>4} | {'BMA':>6} {'ora':>6} {'likBEST':>7} {'likWavg':>7}")

rows = []
for tag in TAGS:
    L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
    gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy(); nb = int(round(K_TP * L.Tp_true / 0.1))
    i_star = nb + 3; f = L[i_star]
    tgt = {'gx': gx, 'gy': gy, 't': float(f.t), 'gt': f.gt_field.numpy(), 'Hs': Hs}
    P, z = gather_frames(L, i_star, nb)
    if len(z) > MCAP:
        idx = np.random.default_rng(0).choice(len(z), MCAP, replace=False); P, z = P[idx], z[idx]
    Pt = torch.tensor(P, device=DP.DEVICE, dtype=torch.float32)
    zt = torch.tensor(z, device=DP.DEVICE, dtype=torch.float32)
    for guid in GUIDS:
        Xd = DP.dps_sample(model, stats, abar, betas, P, z, depth, guidance=guid, n=NSAMP, seed=0)
        flds, lls, nrs = [], [], []
        for b in range(NSAMP):
            f_b, nr_b = DP.atoms_to_field(Xd[b], stats, P, z, depth, tgt)
            if f_b is None:
                continue
            kx, ky, a = denorm_atoms(Xd[b], stats); a = torch.clamp(a, min=0.0)
            ll = DP.loglik_atoms(kx, ky, a, Pt, zt, depth).item()   # deployable obs-likelihood
            flds.append(f_b); lls.append(ll); nrs.append(nr_b)
        flds = np.stack(flds); lls = np.array(lls); nrs = np.array(nrs)
        bma = panel(flds.mean(0), tgt['gt'], Hs)['nrmse']
        ora = nrs.min()
        likbest = nrs[int(lls.argmax())]                            # deployable pick
        w = np.exp(lls - lls.max()); w /= w.sum()
        likwavg = panel((w[:, None, None] * flds).sum(0), tgt['gt'], Hs)['nrmse']
        rows.append(dict(tag=tag, guid=guid, bma=float(bma), ora=float(ora),
                         likbest=float(likbest), likwavg=float(likwavg),
                         ll_nr_corr=float(np.corrcoef(-lls, nrs)[0, 1]) if nrs.std() > 1e-9 else 0.0))
        print(f"{tag:11s} {guid:>4} | {bma:6.3f} {ora:6.3f} {likbest:7.3f} {likwavg:7.3f}  "
              f"(corr(-ll,nRMSE)={rows[-1]['ll_nr_corr']:+.2f})", flush=True)
json.dump(rows, open('tmp/mapselect_test.json', 'w'), indent=2)
print("saved tmp/mapselect_test.json")
