"""
d66 — HARDENING: robustness of the two central findings to noise σ and observation density.

(I)  predictable-zone tie (FIM≈GEO) vs LiDAR noise σ.
(II) active-sensing null (HOS) vs (noise σ × budget/density K), with the discrete-linear win as a
     control to confirm the SEPARATION (discrete wins / continuous doesn't) is σ,K-invariant.

If (I) the tie margin stays small across σ, and (II) HOS gain stays ~0 while discrete stays large
across the σ×K grid, the day's conclusions are robust to the two main nuisance axes.

Run: python3 scripts/d66_robustness.py
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d62_predictable_zone_test as Z
import d63_constrained_active_sensing_hos as AS
from wavesplat.model.tnse import render_wave_field
from d63_constrained_active_sensing_hos import (build_feasible_window, discrete_components,
                                                coarse_theta, recon_nrmse)
from d11_active_sensing import greedy_dopt
from d40_learned_acquisition import pool_xy
from sklearn.metrics import roc_auc_score

SIGMAS = [0.01, 0.02, 0.05]
KS = [80, 150, 250]
SEAS = [('BF4_15c', 1.2, 7.0), ('BF5_30c', 2.0, 9.0)]
SEEDS = [0, 1]


def predictable_zone_tie(sigma):
    Z.SIGMA = sigma; Z.HORIZONS = [0.0, 2.0, 4.0, 6.0]
    gx = np.linspace(0, 320, 90); gy = np.linspace(-110, 110, 64)
    E, F, G = [], [], []
    for tag, Hs, Tp in SEAS:
        for seed in SEEDS:
            w = Z.build_window(Hs, Tp, seed, gx, gy, M=3)
            kx, ky, A, phi = Z.estimate(w['P'], w['z'], w['t_now'])
            th = np.column_stack([A, kx, ky, phi]); Hs_n = float(w['gt_at'][0][1].std()*4)
            for h in [2.0, 4.0, 6.0]:
                st = int(round(h/0.1))
                if st not in w['gt_at']:
                    continue
                t_star, GT = w['gt_at'][st]
                fc = render_wave_field(kx, ky, A, phi, Z.DEPTH, gx, gy, t_star)
                err = np.abs(fc-GT)/Hs_n
                sig = Z.fim_sigma_map(th, w['P'][:,0], w['P'][:,1], w['P'][:,2], gx, gy, t_star)/Hs_n
                dg = Z.geo_distance_map(kx, ky, w['P'][:,0], w['P'][:,1], w['P'][:,2], gx, gy, t_star)
                E.append(err.ravel()); F.append(sig.ravel()); G.append(dg.ravel())
    e = np.concatenate(E); f = np.concatenate(F); g = np.concatenate(G); good = e < 0.25
    return dict(auc_fim=float(roc_auc_score(~good, f)), auc_geo=float(roc_auc_score(~good, g)))


def active_gain(sigma, K, kind):
    """mean (random-coarseD)/random over SEAS×SEEDS at noise sigma, budget K. kind disc|cnl."""
    AS.SIGMA = sigma
    pool = pool_xy()
    gx = np.linspace(pool[:,0].min(), pool[:,0].max(), 60)
    gy = np.linspace(pool[:,1].min(), pool[:,1].max(), 40)
    gains = []
    for tag, Hs, Tp in SEAS:
        for seed in SEEDS:
            comps = discrete_components(seed) if kind == 'disc' else None
            P, z, t_now, GT, Hs_n = build_feasible_window(Hs, Tp, seed, gx, gy, comps=comps, M=3)
            rng = np.random.default_rng(seed)
            si = rng.choice(len(z), min(AS.SEEDK, len(z)), replace=False)
            th_c = coarse_theta(P, z, si)
            ridx = rng.choice(len(z), K, replace=False)
            r_rand = recon_nrmse(P[ridx], z[ridx], gx, gy, t_now, GT, Hs_n)
            cidx = greedy_dopt(th_c, P, K, Z.DEPTH, sigma=sigma)
            r_cd = recon_nrmse(P[cidx], z[cidx], gx, gy, t_now, GT, Hs_n)
            gains.append((r_rand - r_cd) / r_rand)
    return float(np.mean(gains))


def main():
    out = 'results/direction3'; t0 = time.time()
    pz = {}
    for s in SIGMAS:
        pz[str(s)] = predictable_zone_tie(s)
        print(f"[PZ σ={s}] AUC FIM {pz[str(s)]['auc_fim']:.3f}  GEO {pz[str(s)]['auc_geo']:.3f}  "
              f"margin {abs(pz[str(s)]['auc_fim']-pz[str(s)]['auc_geo']):.3f}   ({time.time()-t0:.0f}s)", flush=True)
    asg = {}
    for s in SIGMAS:
        asg[str(s)] = {}
        for K in KS:
            asg[str(s)][str(K)] = dict(disc=active_gain(s, K, 'disc'), cnl=active_gain(s, K, 'cnl'))
            r = asg[str(s)][str(K)]
            print(f"[AS σ={s} K={K}] discrete {r['disc']:+.0%}  HOS {r['cnl']:+.0%}   ({time.time()-t0:.0f}s)", flush=True)

    json.dump(dict(sigmas=SIGMAS, ks=KS, predictable_zone=pz, active_sensing=asg),
              open(f'{out}/robustness.json', 'w'), indent=2)
    print("\n================ ROBUSTNESS SUMMARY ================")
    print("  (I) predictable-zone tie vs noise σ  (margin |AUC_FIM−AUC_GEO|):")
    for s in SIGMAS:
        p = pz[str(s)]
        print(f"     σ={s}:  FIM {p['auc_fim']:.3f}  GEO {p['auc_geo']:.3f}  margin {abs(p['auc_fim']-p['auc_geo']):.3f}")
    print("  (II) active-sensing gain vs (σ × budget K) — discrete-linear / HOS:")
    for s in SIGMAS:
        line = f"     σ={s}: " + "  ".join(f"K{K} {asg[str(s)][str(K)]['disc']:+.0%}/{asg[str(s)][str(K)]['cnl']:+.0%}" for K in KS)
        print(line)
    dm = np.mean([abs(pz[str(s)]['auc_fim']-pz[str(s)]['auc_geo']) for s in SIGMAS])
    hos_g = np.mean([asg[str(s)][str(K)]['cnl'] for s in SIGMAS for K in KS])
    disc_g = np.mean([asg[str(s)][str(K)]['disc'] for s in SIGMAS for K in KS])
    print(f"\n  mean tie margin {dm:.3f} (small ⇒ robust);  mean HOS gain {hos_g:+.0%} vs discrete {disc_g:+.0%}")
    print(f"  => findings {'ROBUST' if dm < 0.05 and hos_g < 0.06 and disc_g > 0.15 else 'CHECK'} to σ,density")
    print(f"saved {out}/robustness.json   ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
