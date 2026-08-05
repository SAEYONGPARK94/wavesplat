"""Step 1 — trajectory policy achievability under the REALISTIC (non-oracle) net-MMSE
estimator. Prereg = tmp/TRAJ_gate_prereg_2026-07-22.md. Reuses d74 policy planning
(CRB-based headings + learned-CEM imitation), swaps achievability from RELAX+oracle
wavenumbers to net_gamma_mmse (beamform kp = kp_hint=None). Linear seas (Step 1).
Q: does the trajectory-control advantage survive a non-oracle estimator, or is it an
oracle-wavenumber artifact (like B-2)?
Usage: python3 tmp/traj_step1_netmmse.py [--episodes N]
Out: results/direction1/traj_step1_netmmse.json
"""
import os, sys, json, math, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')

import d74_closed_loop_trajectory as D74
from d74_closed_loop_trajectory import (make_sea, dom_dir, fly_path, T_OBS, DEPTH,
                                        SPEEDS_FULL, headings_straight, headings_greedy,
                                        headings_cem, headings_learned, train_policy, K)
from d3_prior_net_eval import get_net, net_gamma_mmse
from d3_l1_s2head_train import SmultHead

SIGMA = 0.02
SMULT_FIXED = 3.25     # discovered task-optimal constant (E-B2K); fixed for all policies
GXN = np.linspace(-30, 120, 32)
GYN = np.linspace(-80, 80, 32)
POLICIES = ['stationary', 'straight-cross', 'greedy', 'CEM', 'learned']

_net = None
def net():
    global _net
    if _net is None:
        _net = get_net()          # production prior_net_v3
    return _net


def gt_field_linear(sea):
    Xg, Yg = np.meshgrid(GXN, GYN)
    psi = (sea['kx'][None, None] * Xg[..., None] + sea['ky'][None, None] * Yg[..., None]
           - sea['om'][None, None] * T_OBS + sea['phi'][None, None])
    return (sea['A'][None, None] * np.cos(psi)).sum(-1)


def achieved_netmmse(sea, speed, headings, seed=0):
    """Reconstruct with net-MMSE (beamform kp = NON-oracle) along the flown path."""
    x, y, t, z = fly_path(sea, speed, headings, gen_z=True, seed=seed)
    P = np.column_stack([x, y, t]).astype(np.float64)
    if len(z) > 1500:
        idx = np.random.default_rng(0).choice(len(z), 1500, replace=False)
        P, z = P[idx], z[idx]
    Xg, Yg = np.meshgrid(GXN, GYN)
    Q = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, T_OBS)])
    pred = np.asarray(net_gamma_mmse(net(), P, z, DEPTH, Q, SIGMA, seed=0,
                                     kp_hint=None, use_em=False,
                                     smult_fixed=SMULT_FIXED)).reshape(Xg.shape)
    gt = gt_field_linear(sea)
    Hs = 4 * math.sqrt((sea['A'] ** 2 / 2).sum())
    return float(np.sqrt(np.mean((pred - gt) ** 2)) / Hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=24)
    ap.add_argument('--train_eps', type=int, default=60)
    a = ap.parse_args()
    speed = SPEEDS_FULL[-1]       # headline speed
    print(f"training learned policy (imitate CEM) @ {speed} m/s, {a.train_eps} eps...", flush=True)
    pol = train_policy(speed, a.train_eps)

    def headings_for(p, sea):
        if p == 'stationary':
            return np.full(K, dom_dir(sea))          # fixed heading, but speed=0 below
        if p == 'straight-cross':
            return headings_straight(sea, 90)
        if p == 'greedy':
            return headings_greedy(sea, speed)
        if p == 'CEM':
            return headings_cem(sea, speed, seed=0)
        if p == 'learned':
            return headings_learned(sea, speed, pol)

    rows = []
    t0 = time.time()
    for e in range(a.episodes):
        sea = make_sea(seed=1000 + e)
        row = {'ep': e}
        for p in POLICIES:
            spd = 0.0 if p == 'stationary' else speed
            h = headings_for(p, sea)
            row[p] = achieved_netmmse(sea, spd, h, seed=e)
        rows.append(row)
        if e % 4 == 0:
            print(f"  ep {e}: " + " ".join(f"{p}={row[p]:.3f}" for p in POLICIES)
                  + f"  [{time.time()-t0:.0f}s]", flush=True)
        json.dump(dict(rows=rows, speed=speed, smult=SMULT_FIXED,
                       note='net-MMSE (beamform kp, non-oracle) achievability, linear seas'),
                  open('results/direction1/traj_step1_netmmse.json', 'w'), indent=1)
    analyze(rows)


def analyze(rows):
    from scipy import stats
    arr = {p: np.array([r[p] for r in rows]) for p in POLICIES}
    stat = arr['stationary']
    print(f"\n=== Step 1: net-MMSE achieved nRMSE (n={len(rows)} eps, lower=better) ===")
    print(f"  {'policy':16s} {'nRMSE':>8s} {'Δ vs stat':>10s} {'CI95':>20s} {'sign<stat':>10s}")
    out = {}
    for p in POLICIES:
        v = arr[p]
        d = stat - v                       # positive = policy better than stationary
        rng = np.random.default_rng(0)
        bs = [np.mean(rng.choice(d, len(d))) for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[p] = dict(nrmse=float(v.mean()), delta_vs_stat=float(d.mean()),
                      ci=[float(lo), float(hi)], nbetter=int((d > 0).sum()))
        print(f"  {p:16s} {v.mean():8.3f} {d.mean():+10.3f} [{lo:+.3f},{hi:+.3f}]  {int((d>0).sum()):>3d}/{len(d)}")
    # learned vs CEM
    dlc = arr['CEM'] - arr['learned']
    print(f"  learned vs CEM: Δ(CEM−learned) {dlc.mean():+.3f} (learned matches CEM if ~0/neg)")
    # verdict
    best_ctrl = min(['straight-cross', 'CEM', 'learned'], key=lambda p: out[p]['nrmse'])
    d = out[best_ctrl]
    pass_ = d['delta_vs_stat'] >= 0.02 and d['ci'][0] > 0 and d['nbetter'] > len(rows) * 0.6
    fail = d['delta_vs_stat'] < 0.01 or d['ci'][0] <= 0
    verdict = ('PASS: trajectory advantage survives non-oracle estimator -> Step 2 (HOS)' if pass_
               else 'FAIL: advantage washed out under net-MMSE = oracle-wavenumber artifact' if fail
               else 'PARTIAL: report CIs, no forced label')
    print(f"\n=== VERDICT (best control={best_ctrl}) === {verdict}")
    out['verdict'] = verdict
    j = json.load(open('results/direction1/traj_step1_netmmse.json'))
    j['summary'] = out
    json.dump(j, open('results/direction1/traj_step1_netmmse.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
