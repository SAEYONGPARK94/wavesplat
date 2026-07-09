"""
d65 — HARDENING: is it NONLINEARITY or spectral CONTINUITY that kills the active-sensing lever?

d63 showed CRB-driven selection wins on a DISCRETE-LINEAR benchmark (+40%) but not on continuous
nonlinear HOS (+1%). Two confounded differences: (linear→nonlinear) AND (discrete→continuous).
This isolates them with a controlled three-rung ladder, same harness, multiple seeds:

  A. discrete-linear     : 12 cosines (the classic benchmark)         — expect BIG win
  B. continuous-linear   : HOS order M=1 (linear, continuous JONSWAP) — diagnostic rung
  C. continuous-nonlinear: HOS order M=3 (bound harmonics)            — expect NO win

If B already kills the lever (B ≈ C ≈ 0 << A), the killer is **spectral continuity** (the
representation gap), NOT nonlinearity — i.e. the *discreteness* of the benchmark manufactured the
lever. That sharpens the discrete-benchmark-bias critique to a precise, mechanistic claim.

Reports per-rung mean gain ± SE across windows×budgets and a sign test (coarseD < random).
Run: python3 scripts/d65_continuity_ladder.py
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from scipy.stats import binomtest
from d63_constrained_active_sensing_hos import (build_feasible_window, discrete_components,
                                                coarse_theta, recon_nrmse, BUDGETS, SEEDK, DEPTH)
from d11_active_sensing import greedy_dopt
from d40_learned_acquisition import pool_xy
import d1_crb as C1

SEAS = [('BF4_15c', 1.2, 7.0), ('BF5_30c', 2.0, 9.0)]
SEEDS = [0, 1, 2, 3, 4]
RUNGS = [('discrete-linear', 'disc'), ('continuous-linear', 'clin'), ('continuous-nonlinear', 'cnl')]


def one_window(rung_key, Hs, Tp, seed):
    pool = pool_xy()
    gx = np.linspace(pool[:, 0].min(), pool[:, 0].max(), 60)
    gy = np.linspace(pool[:, 1].min(), pool[:, 1].max(), 40)
    if rung_key == 'disc':
        P, z, t_now, GT, Hs_n = build_feasible_window(Hs, Tp, seed, gx, gy, comps=discrete_components(seed))
    elif rung_key == 'clin':
        P, z, t_now, GT, Hs_n = build_feasible_window(Hs, Tp, seed, gx, gy, comps=None, M=1)
    else:
        P, z, t_now, GT, Hs_n = build_feasible_window(Hs, Tp, seed, gx, gy, comps=None, M=3)
    rng = np.random.default_rng(seed)
    seed_idx = rng.choice(len(z), min(SEEDK, len(z)), replace=False)
    th_c = coarse_theta(P, z, seed_idx)
    gains = []
    for K in BUDGETS:
        ridx = rng.choice(len(z), K, replace=False)
        r_rand = recon_nrmse(P[ridx], z[ridx], gx, gy, t_now, GT, Hs_n)
        cidx = greedy_dopt(th_c, P, K, DEPTH)
        r_cd = recon_nrmse(P[cidx], z[cidx], gx, gy, t_now, GT, Hs_n)
        gains.append(dict(K=K, random=r_rand, coarseD=r_cd, gain=(r_rand - r_cd) / r_rand))
    return gains


def main():
    out = 'results/direction3'; t0 = time.time()
    data = {key: [] for _, key in RUNGS}
    for name, key in RUNGS:
        for tag, Hs, Tp in SEAS:
            for seed in SEEDS:
                g = one_window(key, Hs, Tp, seed)
                for row in g:
                    row.update(sea=tag, seed=seed)
                data[key].extend(g)
                mg = np.mean([r['gain'] for r in g])
                print(f"  [{name:20s} {tag} s{seed}] mean-gain {mg:+.0%}  ({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    for name, key in RUNGS:
        gains = np.array([r['gain'] for r in data[key]])
        wins = int(np.sum(gains > 0)); n = len(gains)
        bt = binomtest(wins, n, 0.5, alternative='greater')
        summary[key] = dict(name=name, n=n, mean_gain=float(gains.mean()),
                            se=float(gains.std(ddof=1) / np.sqrt(n)),
                            median_gain=float(np.median(gains)), wins=wins,
                            sign_test_p=float(bt.pvalue))
    json.dump(dict(config=dict(seas=SEAS, seeds=SEEDS, budgets=BUDGETS), rows=data,
                   summary=summary), open(f'{out}/continuity_ladder.json', 'w'), indent=2)

    print("\n================ CONTINUITY LADDER — CRB-selection gain over uniform ================")
    print(f"  {'rung':24s} {'mean gain ± SE':>18s}  {'median':>8s}  {'wins/n':>9s}  {'sign-test p':>11s}")
    for _, key in RUNGS:
        s = summary[key]
        print(f"  {s['name']:24s} {s['mean_gain']:+6.1%} ± {s['se']:.1%}     "
              f"{s['median_gain']:+6.1%}   {s['wins']:3d}/{s['n']:<3d}   {s['sign_test_p']:.1e}")
    a, b, c = summary['disc']['mean_gain'], summary['clin']['mean_gain'], summary['cnl']['mean_gain']
    print(f"\n  ladder: discrete-linear {a:+.0%}  →  continuous-linear {b:+.0%}  →  "
          f"continuous-nonlinear {c:+.0%}")
    if b < 0.5 * a and abs(b - c) < max(0.05, 0.3 * a):
        print("  => CONTINUITY is the killer: the lever collapses once the spectrum is continuous,")
        print("     even WITHOUT nonlinearity. The discreteness of the benchmark manufactured it.")
    elif b > 0.5 * a and c < 0.5 * a:
        print("  => NONLINEARITY is the killer: continuous-linear keeps the lever; HOS removes it.")
    else:
        print("  => mixed; see per-rung numbers.")
    print(f"saved {out}/continuity_ladder.json   ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
