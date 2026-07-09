"""At the lifted m-cap (m=3000), does stronger guidance collapse the posterior and
reduce the BMA blur? Reports BMA nRMSE (posterior mean) AND best-sample nRMSE (min over
samples) — separating 'posterior-mean too wide' from 'no good sample exists'.
2 simple + 2 complex seas. Run: python3 tmp/g_sweep_highm.py
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
TAGS = ['BF4_5c_s0', 'BF3_5c_s0', 'BF4_15c_s0', 'BF5_15c_s0']   # 2 simple, 2 complex
GUIDS = [0.05, 0.1, 0.2, 0.3]
NSAMP = 8
MCAP = 3000

ck = torch.load(CKPT, map_location=DP.DEVICE); stats = ck['stats']
model = SetDenoiser(**ck['arch']).to(DP.DEVICE); model.load_state_dict(ck['model']); model.eval()
abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)
print(f"ckpt={CKPT} mcap={MCAP} nsamp={NSAMP}")

cases = []
for tag in TAGS:
    L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
    gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy(); nb = int(round(K_TP * L.Tp_true / 0.1))
    i_star = nb + 3; f = L[i_star]
    tgt = {'gx': gx, 'gy': gy, 't': float(f.t), 'gt': f.gt_field.numpy(), 'Hs': Hs}
    P, z = gather_frames(L, i_star, nb)
    if len(z) > MCAP:
        idx = np.random.default_rng(0).choice(len(z), MCAP, replace=False); P, z = P[idx], z[idx]
    cases.append((tag, depth, tgt, P, z, Hs))

print(f"{'guid':>5} | " + " | ".join(f"{t.split('_')[0]+t.split('_')[1]:>10}" for t in TAGS) + "   (BMA / best)")
rows = []
for guid in GUIDS:
    line, per = [], {}
    t0 = time.time()
    for tag, depth, tgt, P, z, Hs in cases:
        Xd = DP.dps_sample(model, stats, abar, betas, P, z, depth, guidance=guid, n=NSAMP, seed=0)
        rs = [panel(f, tgt['gt'], Hs)['nrmse'] for b in range(NSAMP)
              for f, _ in [DP.atoms_to_field(Xd[b], stats, P, z, depth, tgt)] if f is not None]
        flds = [f for b in range(NSAMP)
                for f, _ in [DP.atoms_to_field(Xd[b], stats, P, z, depth, tgt)] if f is not None]
        bma = np.mean(flds, axis=0); bma_nr = panel(bma, tgt['gt'], Hs)['nrmse']; best = min(rs)
        per[tag] = dict(bma=bma_nr, best=best)
        line.append(f"{bma_nr:.3f}/{best:.3f}")
    rows.append(dict(guid=guid, per=per))
    print(f"{guid:>5} | " + " | ".join(f"{x:>10}" for x in line) + f"   ({time.time()-t0:.0f}s)", flush=True)
json.dump(rows, open('tmp/g_sweep_highm.json', 'w'), indent=2)
print("saved tmp/g_sweep_highm.json")
