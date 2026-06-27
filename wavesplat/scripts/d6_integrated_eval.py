"""
Direction III — main result: integrated grid-free DPS estimator on the 30-episode
benchmark, head-to-head with ST-DF-GP.

Estimator = (v2 set-diffusion prior) + (time-fused window) + (DPS posterior) +
(BMA point estimate) + (globally-calibrated posterior UQ).

For each of the 30 benchmark episodes: time-fused window (W frames) at a fixed
target frame → DPS posterior (nsamp) → BMA field + per-pixel σ_post. We report
BMA nRMSE per sea-state vs ST-DF-GP (results/spacetime_gp_30ep.json), and the
globally-recalibrated ±1σ/±2σ coverage of the posterior UQ.

Run: python3 scripts/d6_integrated_eval.py
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser

DEVICE = DP.DEVICE
CKPT = 'results/direction3/setdiff_prior_v2.pt'
W = 30                # time-fused window (frames @ 0.1s = 3 s)
TI = 40               # fixed target tracking frame
NSAMP = 10
GUID = 0.1


def tag_to_file(tag):
    # 'BF3_5c_s0' -> 'episode_BF3_5comp_seed0_uniform.h5'
    bf, comp, s = tag.split('_')
    return f"episode_{bf}_{comp.replace('c','comp')}_seed{s[1:]}_uniform.h5"


def gather_W(L, ti, w, m_cap=200):
    frames = L.get_tracking_frames(2.0)
    P, z = [], []
    for f in frames[ti-w+1:ti+1]:
        x = f.x_obs.numpy(); y = f.y_obs.numpy(); zz = f.z_obs.numpy(); t = float(f.t)
        P.append(np.column_stack([x, y, np.full_like(x, t)])); z.append(zz)
    P = np.vstack(P); z = np.concatenate(z)
    if len(z) > m_cap:
        idx = np.random.default_rng(0).choice(len(z), m_cap, replace=False)
        P, z = P[idx], z[idx]
    return P, z, frames


def main():
    ck = torch.load(CKPT, map_location=DEVICE)
    stats = ck['stats']
    model = SetDenoiser(**ck.get('arch', {})).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    abar = ck['abar'].to(DEVICE); betas = ck['betas'].to(DEVICE)

    gp = json.load(open('results/spacetime_gp_30ep.json'))['per_episode']
    tags = list(gp.keys())

    rows = []; cal_err = []; cal_sig = []
    t0 = time.time()
    print(f"{'tag':12s} {'N':>3} {'DPS_BMA':>8} {'STDFGP':>8} {'M':>5} {'Tspan':>6}")
    for tag in tags:
        fn = tag_to_file(tag)
        try:
            L = EpisodeDataLoader('data/' + fn)
        except Exception as e:
            print(f"  skip {tag}: {e}"); continue
        depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        frames = L.get_tracking_frames(2.0)
        if len(frames) < TI + 2:
            print(f"  skip {tag}: only {len(frames)} tracking frames"); continue
        P, z, _ = gather_W(L, TI, W)
        tgt = frames[TI + 1]
        target = {'gx': gx, 'gy': gy, 't': float(tgt.t), 'gt': tgt.gt_field.numpy(), 'Hs': Hs}
        Xd = DP.dps_sample(model, stats, abar, betas, P, z, depth, guidance=GUID, n=NSAMP, seed=0)
        fields = []
        for b in range(NSAMP):
            f, _ = DP.atoms_to_field(Xd[b], stats, P, z, depth, target)
            if f is not None:
                fields.append(f)
        F = np.stack(fields); bma = F.mean(0); sig = F.std(0)
        nr = float(np.sqrt(np.mean((bma - target['gt'])**2)) / Hs)
        gp_fused = gp[tag]['fused']['nRMSE']
        ncomp = int(tag.split('_')[1].replace('c', ''))
        rows.append(dict(tag=tag, sea=f"{tag.split('_')[0]}_{tag.split('_')[1]}",
                         dps_bma=nr, stdfgp_fused=gp_fused, ncomp=ncomp))
        cal_err.append(np.abs(bma - target['gt']).ravel()); cal_sig.append(sig.ravel())
        print(f"{tag:12s} {ncomp:3d} {nr:8.4f} {gp_fused:8.4f} {len(z):5d} "
              f"{P[:,2].max()-P[:,2].min():6.2f}  ({time.time()-t0:.0f}s)")

    # global UQ recalibration
    er = np.concatenate(cal_err); sg = np.concatenate(cal_sig)
    s_cal = float(np.quantile(er / (sg + 1e-9), 0.68))
    cov1 = float((er <= s_cal * sg).mean()); cov2 = float((er <= 2 * s_cal * sg).mean())

    # aggregate per sea-state
    seas = {}
    for r in rows:
        seas.setdefault(r['sea'], []).append(r)
    per_sea = {s: dict(dps=float(np.mean([r['dps_bma'] for r in rs])),
                       stdfgp=float(np.mean([r['stdfgp_fused'] for r in rs])),
                       n=len(rs)) for s, rs in seas.items()}
    overall_dps = float(np.mean([r['dps_bma'] for r in rows]))
    overall_gp = float(np.mean([r['stdfgp_fused'] for r in rows]))

    out = dict(W=W, TI=TI, nsamp=NSAMP, guidance=GUID, ckpt=CKPT,
               overall_dps_bma=overall_dps, overall_stdfgp=overall_gp,
               s_cal=s_cal, cov1=cov1, cov2=cov2, per_sea=per_sea, rows=rows)
    json.dump(out, open('results/direction3/integrated_eval.json', 'w'), indent=2)

    print("\n=== per sea-state (DPS_BMA vs ST-DF-GP) ===")
    for s, v in sorted(per_sea.items()):
        win = '✓DPS' if v['dps'] < v['stdfgp'] else ''
        print(f"  {s:10s} n={v['n']}  DPS {v['dps']:.3f}  ST-DF-GP {v['stdfgp']:.3f}  {win}")
    print(f"\nOVERALL  DPS_BMA {overall_dps:.4f}   ST-DF-GP fused {overall_gp:.4f}")
    print(f"UQ (global s={s_cal:.2f}): ±1σ cov {cov1:.2f} (ideal 0.68)  ±2σ {cov2:.2f} (ideal 0.95)")

    # figure
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    sea_order = sorted(per_sea.keys(), key=lambda s: (int(s.split('_')[0][2]), int(s.split('_')[1][:-1])))
    x = np.arange(len(sea_order)); wd = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - wd/2, [per_sea[s]['dps'] for s in sea_order], wd, label='grid-free DPS (BMA, time-fused)', color='tab:green')
    ax.bar(x + wd/2, [per_sea[s]['stdfgp'] for s in sea_order], wd, label='ST-DF-GP (fused)', color='tab:gray')
    ax.axhline(0.06, ls=':', c='k', label='exact (info-wall floor)')
    ax.set_xticks(x); ax.set_xticklabels(sea_order, rotation=30, ha='right')
    ax.set_ylabel('field nRMSE')
    ax.set_title(f'Direction III main result — grid-free DPS vs ST-DF-GP (30 episodes)\n'
                 f'overall DPS {overall_dps:.3f} vs ST-DF-GP {overall_gp:.3f}; '
                 f'DPS UQ calibrated (±1σ {cov1:.2f} after s={s_cal:.2f})')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig('results/direction3/fig_integrated_eval.png', dpi=140)
    print('saved results/direction3/integrated_eval.json + fig_integrated_eval.png')


if __name__ == '__main__':
    main()
