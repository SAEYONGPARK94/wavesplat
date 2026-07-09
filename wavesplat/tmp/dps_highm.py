"""DPS with the lifted m-cap (exact Woodbury likelihood): does more observations close
the gap to RELAX-full (0.119)? DPS-only at configurable mcap, mirrors d30 frame protocol;
RELAX-full / GP / oracle pulled from the precomputed jsons (RELAX-full = 3p @ m=3000).
Run: python3 tmp/dps_highm.py --mcap 3000 --guid 0.05 [--smoke N]
"""
import os, sys, json, time, argparse, numpy as np, torch
sys.path.insert(0, 'scripts')
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser
from d23_main_results import K_TP, N_FRAMES, panel
from d15_sequential_fusion import gather_frames
from d6_integrated_eval import tag_to_file
from wavesplat.data_loader import EpisodeDataLoader

ap = argparse.ArgumentParser()
ap.add_argument('--mcap', type=int, default=3000)
ap.add_argument('--guid', type=float, default=0.05)
ap.add_argument('--nsamp', type=int, default=8)
ap.add_argument('--frames', type=int, default=N_FRAMES)
ap.add_argument('--smoke', type=int, default=0)
ap.add_argument('--ckpt', default='results/direction3/setdiff_prior_long.pt')
ap.add_argument('--out', default='results/direction3/dps_highm.json')
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location=DP.DEVICE); stats = ck['stats']
model = SetDenoiser(**ck['arch']).to(DP.DEVICE); model.load_state_dict(ck['model']); model.eval()
abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)
print(f"ckpt={a.ckpt} mcap={a.mcap} guid={a.guid} nsamp={a.nsamp}")

gp = json.load(open('results/spacetime_gp_30ep.json'))['per_episode']
relax_full = json.load(open('results/direction3/main_results.json'))
relax_iter = json.load(open('results/direction3/main_results_iter.json'))
tags = list(gp.keys())
if a.smoke:
    tags = tags[:a.smoke]

out = {}; t0 = time.time()
for ti, tag in enumerate(tags):
    L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
    gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
    Tobs = K_TP * L.Tp_true; n_back = int(round(Tobs / 0.1)); nfr = len(L)
    i0 = n_back + 3; i1 = nfr - 2
    if i1 - i0 < 4:
        i0 = max(n_back + 1, nfr - 60)
    frames = np.linspace(i0, i1, a.frames).astype(int)
    nr, pc, spN, mused = [], [], [], []
    for i_star in frames:
        fstar = L[int(i_star)]
        target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
        P, z = gather_frames(L, int(i_star), n_back)
        if len(z) > a.mcap:
            idx = np.random.default_rng(0).choice(len(z), a.mcap, replace=False); P, z = P[idx], z[idx]
        mused.append(len(z))
        Xd = DP.dps_sample(model, stats, abar, betas, P, z, depth, guidance=a.guid, n=a.nsamp, seed=0)
        flds = [f for b in range(a.nsamp)
                for f, _ in [DP.atoms_to_field(Xd[b], stats, P, z, depth, target)] if f is not None]
        bma = np.mean(flds, axis=0) if flds else np.zeros_like(target['gt'])
        nr.append(panel(bma, target['gt'], Hs)['nrmse'])
        if len(flds) >= 2:
            std = np.stack(flds).std(0); err = np.abs(bma - target['gt'])
            s, e = std.ravel(), err.ravel()
            pc.append(float(np.corrcoef(s, e)[0, 1]) if s.std() > 1e-9 else 0.0)
            spN.append(float(std.mean()) / max(Hs, 1e-6))
    out[tag] = dict(dps=float(np.mean(nr)), dps_std=float(np.std(nr)), m=int(np.mean(mused)),
                    uq_pix=float(np.mean(pc)) if pc else 0.0, spread_hs=float(np.mean(spN)) if spN else 0.0,
                    relax3_full=relax_iter[tag]['deployable']['nrmse'][0],
                    stdfgp=gp[tag]['fused']['nRMSE'], oracle=relax_full[tag]['oracle']['nrmse'][0])
    d = out[tag]
    print(f"[{ti+1:2d}/{len(tags)}] {tag:12s} m={d['m']:4d}  DPS {d['dps']:.3f}(±{d['dps_std']:.3f}) "
          f"| RELAX3_full {d['relax3_full']:.3f} | GP {d['stdfgp']:.3f} | ora {d['oracle']:.3f} "
          f"| UQ {d['uq_pix']:+.2f}  ({time.time()-t0:.0f}s)", flush=True)

seas = {}
for tag, r in out.items():
    s = '_'.join(tag.split('_')[:2]); seas.setdefault(s, []).append(r)
print(f"\n=== per sea-state (DPS @ m≤{a.mcap}, guid {a.guid}) ===")
print(f"{'sea':9s} {'DPS':>7} {'RELAX3full':>11} {'ST-DF-GP':>9} {'oracle':>7} {'UQ':>6}")
A = {k: [] for k in ['dps', 'rf', 'gp', 'o', 'uq', 'sh']}
for s in sorted(seas):
    rs = seas[s]; m = lambda f: float(np.mean([f(r) for r in rs]))
    row = dict(dps=m(lambda r: r['dps']), rf=m(lambda r: r['relax3_full']), gp=m(lambda r: r['stdfgp']),
               o=m(lambda r: r['oracle']), uq=m(lambda r: r['uq_pix']), sh=m(lambda r: r['spread_hs']))
    print(f"{s:9s} {row['dps']:7.3f} {row['rf']:11.3f} {row['gp']:9.3f} {row['o']:7.3f} {row['uq']:+6.2f}")
    for k in A: A[k].append(row[k])
mm = lambda k: float(np.mean(A[k]))
ep_corr = float(np.corrcoef([r['spread_hs'] for r in out.values()], [r['dps'] for r in out.values()])[0, 1])
print(f"\nOVERALL  DPS {mm('dps'):.3f} | RELAX3_full {mm('rf'):.3f} | ST-DF-GP {mm('gp'):.3f} | oracle {mm('o'):.3f}")
print(f"UQ: per-pixel {mm('uq'):+.3f} | across-ep corr(Hs-norm spread, nRMSE) {ep_corr:+.3f}")
json.dump(dict(mcap=a.mcap, guid=a.guid, nsamp=a.nsamp, per_episode=out,
               overall=dict(dps=mm('dps'), relax3_full=mm('rf'), stdfgp=mm('gp'), oracle=mm('o'),
                            uq_pix=mm('uq'), uq_ep_corr=ep_corr)), open(a.out, 'w'), indent=2)
print(f"saved {a.out}")
