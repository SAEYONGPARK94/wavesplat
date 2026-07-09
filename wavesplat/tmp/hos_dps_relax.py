"""HOS nonlinear validation: DPS set-diffusion component prior vs RELAX on 3rd-order
HOS seas. Same realistic LiDAR footprint pool + window as d63; nowcast nRMSE vs HOS GT.
Shared subsampled obs (m_cap) to both estimators (fair); RELAX also at full pool (ref).
Run: python3 tmp/hos_dps_relax.py [--guid 0.05] [--nsamp 8] [--mcap 200] [--M 3]
"""
import os, sys, json, time, argparse, numpy as np, torch
sys.path.insert(0, 'scripts')
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser
from d23_main_results import panel
from d26_relax_iterative import iterative_relax
from d40_learned_acquisition import pool_xy
from d63_constrained_active_sensing_hos import build_feasible_window, DEPTH

ap = argparse.ArgumentParser()
ap.add_argument('--guid', type=float, default=0.05)
ap.add_argument('--nsamp', type=int, default=8)
ap.add_argument('--mcap', type=int, default=200)
ap.add_argument('--M', type=int, default=3, help='HOS nonlinear order (1=linear,3=3rd)')
ap.add_argument('--ckpt', default='results/direction3/setdiff_prior_long.pt')
ap.add_argument('--out', default='results/direction3/hos_dps_relax.json')
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location=DP.DEVICE); stats = ck['stats']
model = SetDenoiser(**ck['arch']).to(DP.DEVICE); model.load_state_dict(ck['model']); model.eval()
abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)
print(f"ckpt={a.ckpt} guid={a.guid} nsamp={a.nsamp} mcap={a.mcap} HOS order M={a.M}")

pool = pool_xy()
gx = np.linspace(pool[:, 0].min(), pool[:, 0].max(), 60)
gy = np.linspace(pool[:, 1].min(), pool[:, 1].max(), 40)
SETTINGS = [('BF3', 1.0, 7.0), ('BF4', 2.0, 8.5), ('BF5', 3.0, 10.0)]
SEEDS = [0, 1]

rows = []
t0 = time.time()
for label, Hs, Tp in SETTINGS:
    for seed in SEEDS:
        P, z, t_now, GT, Hs_n = build_feasible_window(Hs, Tp, seed, gx, gy, comps=None, M=a.M)
        idx = np.random.default_rng(0).choice(len(z), min(a.mcap, len(z)), replace=False)
        Ps, zs = P[idx].astype(np.float64), z[idx].astype(np.float64)
        target = {'gx': gx, 'gy': gy, 't': t_now, 'gt': GT, 'Hs': Hs_n}
        # DPS posterior-mean field + UQ
        Xd = DP.dps_sample(model, stats, abar, betas, Ps, zs, DEPTH, guidance=a.guid, n=a.nsamp, seed=0)
        flds = [f for b in range(a.nsamp)
                for f, _ in [DP.atoms_to_field(Xd[b], stats, Ps, zs, DEPTH, target)] if f is not None]
        bma = np.mean(flds, axis=0) if flds else np.zeros_like(GT)
        dps_nr = panel(bma, GT, Hs_n)['nrmse']
        pc, spread = 0.0, 0.0
        if len(flds) >= 2:
            std = np.stack(flds).std(0); err = np.abs(bma - GT)
            s, e = std.ravel(), err.ravel()
            pc = float(np.corrcoef(s, e)[0, 1]) if s.std() > 1e-9 else 0.0
            spread = float(std.mean())
        # RELAX (3-pass) on same shared obs, and on full pool (reference)
        f3, _ = iterative_relax(Ps, zs, DEPTH, target, 3); r3 = panel(f3, GT, Hs_n)['nrmse']
        ff, _ = iterative_relax(P.astype(np.float64), z.astype(np.float64), DEPTH, target, 3)
        r3f = panel(ff, GT, Hs_n)['nrmse']
        rows.append(dict(label=label, seed=seed, Hs=float(Hs_n), nobs_full=int(len(z)),
                         dps=dps_nr, relax3_shared=r3, relax3_full=r3f,
                         uq_pix_corr=pc, spread=spread))
        print(f"{label} s{seed} Hs={Hs_n:.2f} m={len(zs)}/{len(z)}  "
              f"DPS {dps_nr:.3f}(UQ{pc:+.2f}) | RELAX3sh {r3:.3f} | RELAX3full {r3f:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

agg = lambda k: float(np.mean([r[k] for r in rows]))
print(f"\nHOS(M={a.M}) OVERALL  DPS {agg('dps'):.3f} | RELAX3sh {agg('relax3_shared'):.3f} | "
      f"RELAX3full {agg('relax3_full'):.3f} | UQ pix_corr {agg('uq_pix_corr'):+.2f}")
json.dump(dict(guid=a.guid, M=a.M, mcap=a.mcap, ckpt=a.ckpt,
               overall=dict(dps=agg('dps'), relax3_shared=agg('relax3_shared'),
                            relax3_full=agg('relax3_full'), uq_pix_corr=agg('uq_pix_corr')),
               rows=rows), open(a.out, 'w'), indent=2)
print(f"saved {a.out}")
