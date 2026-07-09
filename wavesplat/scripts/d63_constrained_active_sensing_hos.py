"""
d63 — Constrained active sensing on NONLINEAR HOS seas (revival of the learning-central thread).

Settled prior findings:
  - FREE-AIM sensing: random-uniform coverage is optimal; no lever (d42/d43 beam_steering).
  - LEARNED selection from the REALISTIC (raydrop-constrained) return set won −16% (d40) / coarse
    D-optimal −25% (d39) — BUT ONLY on the LINEAR benchmark, exactly where RELAX is near-optimal.

The open, decisive gap (flagged by beam_steering_VERDICT + the standing HOS directive): does
informative, CRB-driven selection from the constrained feasible return set still beat uniform on
NONLINEAR (HOS) seas? If even the greedy D-optimal *teacher* cannot beat uniform-over-feasible on
HOS, the constrained active-sensing lever is not real and the learned policy is moot. If it can,
the lever survives realism and the fast d40 policy (2.3 ms) is its deployable realization.

Arms (per window, fixed processing budget K from the feasible return pool):
  - random  : uniform-random K over the feasible (raydrop) returns,
  - coarseD : deployable CRB selection (seed→coarse RELAX→greedy D-optimal K),
  - fullpool: RELAX on the whole feasible pool (≈ unlimited-budget reference).
All reconstructed with the SAME deployable iterative RELAX; nowcast nRMSE vs the HOS GT.
A LINEAR control (discrete sum-of-cosines, same harness) confirms the known linear win reproduces.

Run: python3 scripts/d63_constrained_active_sensing_hos.py
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import torch
from wavesim.hos_simulator import HOSSimulator, HOSConfig
from wavesplat.model.tnse import fit_amplitudes_phases
from d18_relax_estimator import relax_step
from d26_relax_iterative import iterative_relax
from d23_main_results import panel
from d11_active_sensing import greedy_dopt
from d40_learned_acquisition import pool_xy
import d1_crb as C1

DEPTH = 50.0
G = 9.81
DT_FRAME = 0.1
K_TP = 2.2
SIGMA = 0.02
BUDGETS = [80, 150, 250]
POOL_CAP = 1000
PER_FRAME = 120
SEEDK = 80
SEAS = [('BF3_5c', 0.8, 5.5), ('BF4_15c', 1.2, 7.0), ('BF5_30c', 2.0, 9.0)]
SEEDS = [0, 1, 2, 3]


def omega_np(k):
    return np.sqrt(G * k * np.tanh(k * DEPTH) + 1e-12)


def discrete_components(seed, N=12):
    rng = np.random.default_rng(seed + 77)
    ang = np.deg2rad(rng.uniform(-30, 30, N))
    kmag = rng.uniform(0.03, 0.12, N)
    return dict(kx=kmag * np.cos(ang), ky=kmag * np.sin(ang),
                A=0.3 * np.exp(-rng.uniform(0, 1.5, N)), phi=rng.uniform(0, 2 * np.pi, N),
                om=omega_np(kmag))


def build_feasible_window(Hs, Tp, seed, gx, gy, comps=None, M=3):
    """Return feasible return pool (P,z) over the window + nowcast GT at t_now.
    comps=None → HOS sea (order M; M=1 is linear-continuous); else discrete linear control."""
    pool = pool_xy()
    n_back = int(round(K_TP * Tp / DT_FRAME))
    Xg, Yg = np.meshgrid(gx, gy)
    obs_P, obs_z = [], []
    if comps is None:
        cfg = HOSConfig(Lx=600.0, Ly=400.0, Nx=64, Ny=64, depth=DEPTH, M=M)
        sim = HOSSimulator(cfg)
        sim.initialize_jonswap(Hs=Hs, Tp=Tp, seed=seed, direction_deg=0.0, direction_spread_deg=30.0)
    t = 0.0
    GT = None
    for i in range(n_back + 1):
        if i > 0 and comps is None:
            sim.step(DT_FRAME)
        t = i * DT_FRAME
        idx = np.random.default_rng(2000 + i).choice(len(pool), min(PER_FRAME, len(pool)), replace=False)
        ox, oy = pool[idx, 0], pool[idx, 1]
        if comps is None:
            Zo, *_ = sim.evaluate_at(ox.reshape(1, -1), oy.reshape(1, -1)); zo = Zo.ravel()
        else:
            psi = comps['kx'][None] * ox[:, None] + comps['ky'][None] * oy[:, None] - comps['om'][None] * t + comps['phi'][None]
            zo = (comps['A'][None] * np.cos(psi)).sum(1)
        zo = zo + np.random.default_rng(3000 + i).normal(0, SIGMA, len(ox))
        obs_P.append(np.column_stack([ox, oy, np.full_like(ox, t)])); obs_z.append(zo)
    t_now = n_back * DT_FRAME
    if comps is None:
        Zg, *_ = sim.evaluate_at(Xg, Yg); GT = Zg.astype(np.float64)
    else:
        psi = comps['kx'][None, None] * Xg[..., None] + comps['ky'][None, None] * Yg[..., None] - comps['om'][None, None] * t_now + comps['phi'][None, None]
        GT = (comps['A'][None, None] * np.cos(psi)).sum(-1)
    P = np.vstack(obs_P).astype(np.float64); z = np.concatenate(obs_z).astype(np.float64)
    if len(z) > POOL_CAP:
        ii = np.random.default_rng(0).choice(len(z), POOL_CAP, replace=False); P, z = P[ii], z[ii]
    Hs_n = float(GT.std() * 4)
    return P, z, t_now, GT, Hs_n


def coarse_theta(P, z, seed_idx):
    Ps, zs = P[seed_idx], z[seed_idx]
    kx, ky = relax_step(np.zeros(0), np.zeros(0), Ps, zs, DEPTH)
    obs = np.column_stack([Ps[:, 0], Ps[:, 1], Ps[:, 2], zs]).astype(np.float64)
    A, phi = fit_amplitudes_phases(kx, ky, obs, DEPTH, lambda_reg=0.5)
    return torch.tensor(np.stack([A, kx, ky, phi], 1), dtype=C1.DT, device=C1.DEVICE)


def recon_nrmse(P, z, gx, gy, t_now, GT, Hs):
    target = {'gx': gx, 'gy': gy, 't': t_now, 'gt': GT, 'Hs': Hs}
    fld, _ = iterative_relax(P, z, DEPTH, target, 3)
    return panel(fld, GT, Hs)['nrmse']


def run_setting(label, Hs, Tp, seed, comps):
    pool = pool_xy()
    gx = np.linspace(pool[:, 0].min(), pool[:, 0].max(), 60)
    gy = np.linspace(pool[:, 1].min(), pool[:, 1].max(), 40)
    P, z, t_now, GT, Hs_n = build_feasible_window(Hs, Tp, seed, gx, gy, comps)
    rng = np.random.default_rng(seed)
    full = recon_nrmse(P, z, gx, gy, t_now, GT, Hs_n)
    seed_idx = rng.choice(len(z), min(SEEDK, len(z)), replace=False)
    th_c = coarse_theta(P, z, seed_idx)
    res = {}
    for K in BUDGETS:
        ridx = rng.choice(len(z), K, replace=False)
        r_rand = recon_nrmse(P[ridx], z[ridx], gx, gy, t_now, GT, Hs_n)
        cidx = greedy_dopt(th_c, P, K, DEPTH)
        r_cd = recon_nrmse(P[cidx], z[cidx], gx, gy, t_now, GT, Hs_n)
        res[K] = dict(random=float(r_rand), coarseD=float(r_cd))
    return dict(label=label, kind=('HOS' if comps is None else 'linear'),
                Hs=Hs_n, N_pool=int(len(z)), fullpool=float(full), budgets=res)


def main():
    out = 'results/direction3'; t0 = time.time()
    rows = []
    # HOS seas
    for tag, Hs, Tp in SEAS:
        for seed in SEEDS:
            r = run_setting(f'{tag}_s{seed}', Hs, Tp, seed, comps=None)
            rows.append(r)
            line = f"[HOS  {r['label']:10s}] Npool={r['N_pool']} full={r['fullpool']:.3f} | " + \
                   " ".join(f"K{K}: rnd {r['budgets'][K]['random']:.3f}/cD {r['budgets'][K]['coarseD']:.3f}" for K in BUDGETS)
            print(line + f"   ({time.time()-t0:.0f}s)", flush=True)
    # linear control (same harness) — confirm the known linear win reproduces
    for seed in SEEDS:
        r = run_setting(f'lin_s{seed}', 1.5, 8.0, seed, comps=discrete_components(seed))
        rows.append(r)
        line = f"[LIN  {r['label']:10s}] Npool={r['N_pool']} full={r['fullpool']:.3f} | " + \
               " ".join(f"K{K}: rnd {r['budgets'][K]['random']:.3f}/cD {r['budgets'][K]['coarseD']:.3f}" for K in BUDGETS)
        print(line + f"   ({time.time()-t0:.0f}s)", flush=True)

    # aggregate gains by kind & budget
    def agg(kind, K, arm):
        v = [r['budgets'][K][arm] for r in rows if r['kind'] == kind]
        return float(np.mean(v))
    summary = {}
    for kind in ['HOS', 'linear']:
        summary[kind] = {}
        for K in BUDGETS:
            rnd, cd = agg(kind, K, 'random'), agg(kind, K, 'coarseD')
            summary[kind][str(K)] = dict(random=rnd, coarseD=cd, gain_pct=100 * (rnd - cd) / rnd)
    json.dump(dict(config=dict(seas=SEAS, seeds=SEEDS, budgets=BUDGETS, sigma=SIGMA),
                   rows=rows, summary=summary), open(f'{out}/active_sensing_hos.json', 'w'), indent=2)

    print("\n================ SUMMARY: CRB-selection gain over uniform (mean nRMSE) ================")
    for kind in ['HOS', 'linear']:
        print(f"  [{kind}]")
        for K in BUDGETS:
            s = summary[kind][str(K)]
            print(f"    K={K:3d}  random {s['random']:.3f}  coarse-Dopt {s['coarseD']:.3f}  "
                  f"gain {s['gain_pct']:+.0f}%")
    g_hos = np.mean([summary['HOS'][str(K)]['gain_pct'] for K in BUDGETS])
    g_lin = np.mean([summary['linear'][str(K)]['gain_pct'] for K in BUDGETS])
    print(f"\n  mean CRB-selection gain:  HOS {g_hos:+.0f}%   linear {g_lin:+.0f}%")
    print(f"  => constrained active-sensing lever on realistic (HOS) seas is "
          f"{'REAL' if g_hos > 5 else 'WEAK/ABSENT'} (>5% = real).")
    print(f"saved {out}/active_sensing_hos.json   ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
