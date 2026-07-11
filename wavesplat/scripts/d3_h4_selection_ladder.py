"""H4-SEL -- the observation-selection ladder at the REAL operating point.

H3-OD moved the operating point: under the sensor's fixed-T law the episode
yields ~88k returns, but the control budget (<=100 ms) cannot solve on all of
them -- H3-OD used a RANDOM 5k subsample and still got enc rho 0.61 (6/8 seas
0.82-0.99). So the learning question that is actually OPEN at deployment is:

    WHICH m of the 88k points buy the most task skill?

This script builds the learning-free ladder first (design law 4). Policies,
all deployment-observable (no GT):

  rand      uniform random (H3-OD's floor)
  strat_r   equal counts per range bin (12 bins, R_MIN..max return range):
            counters the near-range crowding that survives even fixed-T
  strat_rt  equal counts per (range bin x scan): adds time coverage
  lev       prior-leverage sampling: h_i = sum_b gam0_b phi_ib^2 with the
            BANK prior gam0 (closed form, no solve); sample p ~ h_i renorm.
            The principled D-optimal-ish competitor without learning
  rand4x    random at 4m budget: the budget-headroom control

Budgets m = 2000 and 5000 (rand4x = 8000/20000). Scored exactly as H3-OD
(naive MMSE, B1 caps, rho on fan/annulus/in-band slope/encounter), paired on
identical scenes+speckle, n = 8 head seas, T = T_m50.

READING (fixed before running): the ladder spread strat/lev - rand is the
value of selection. If <= 0.02 rho everywhere, selection is a dead knob and
a learned selector has no room (cell closes by measurement). If the spread
is real, its gap to rand4x bounds what selection can buy vs more budget, and
the learned-selector cell (A1's amortized active-set) opens with a measured
headroom target. No learning in this script; no verdict about learning
beyond the headroom number.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

import d3_h3_fixedT_operating_domain as h3
from d3_h3_fixedT_operating_domain import gen_hits, ENC_T, A_M
from d3_h1_mask_destination import sea_dir, eta_xt
from d3_gpu_mmse import mmse_gpu, build_bank, design
from d3_s2_mask_information import grad_design
from d3_door1b_field_bias import (eta_dr, R_MIN, R_MAX, SCAN_T, NOISE,
                                  DEPTH, FOV_H)
from d3_g0b_selection_aware import DEV, DT, BAND

OUT = 'results/direction3/h4_selection_ladder.json'
T_SEL = 0.0173                                  # T_m50
N_SEAS = 8
BUDGETS = [2000, 5000]


def select(policy, P, rr, m, rng, gam0=None, phi_fn=None):
    n = len(P)
    if m >= n:
        return np.arange(n)
    if policy == 'rand':
        return rng.choice(n, m, replace=False)
    if policy == 'strat_r':
        edges = np.linspace(R_MIN, rr.max() + 1e-6, 13)
        b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
        take = []
        for j in range(12):
            idx = np.where(b == j)[0]
            k = min(len(idx), int(np.ceil(m / 12)))
            if k:
                take.append(rng.choice(idx, k, replace=False))
        t = np.concatenate(take)
        return rng.choice(t, m, replace=False) if len(t) > m else t
    if policy == 'strat_rt':
        edges = np.linspace(R_MIN, rr.max() + 1e-6, 13)
        b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
        cell = b * 100 + np.searchsorted(SCAN_T, P[:, 2])
        cells = np.unique(cell)
        per = int(np.ceil(m / len(cells)))
        take = []
        for c in cells:
            idx = np.where(cell == c)[0]
            k = min(len(idx), per)
            if k:
                take.append(rng.choice(idx, k, replace=False))
        t = np.concatenate(take)
        return rng.choice(t, m, replace=False) if len(t) > m else t
    if policy == 'lev':
        h = phi_fn(P)                            # prior leverage, closed form
        p = h / h.sum()
        return rng.choice(n, m, replace=False, p=p)
    raise ValueError(policy)


def run_sea(seed, rng):
    S, kp, scans = gen_hits(seed, 3.33, rng)
    P, zv, rr = [], [], []
    for t, hp, r, pw in scans:
        m = pw > T_SEL
        P.append(np.column_stack([hp[m, :2], np.full(int(m.sum()), t)]))
        zv.append(hp[m, 2])
        rr.append(r[m])
    P, zv, rr = np.vstack(P), np.concatenate(zv), np.concatenate(rr)
    zv = zv + rng.normal(0, NOISE, len(zv))
    n_all = len(P)

    nq = 1200
    rq = np.sqrt(rng.uniform((R_MIN / R_MAX) ** 2, 1.0, nq)) * R_MAX
    aq = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), nq)
    Qf = np.column_stack([rq * np.cos(aq), rq * np.sin(aq), np.zeros(nq)])
    gt_f = eta_xt(Qf[:, 0], Qf[:, 1], Qf[:, 2], S)
    sig_f = float(np.std(gt_f))
    kt = np.hypot(S[0], S[1])
    ib = (kt >= BAND[0] * kp) & (kt <= BAND[1] * kp)
    S_ib = tuple(c[ib] for c in S)
    r90a = float(np.quantile(rr, 0.90))
    na = 400
    ra = np.sqrt(rng.uniform((R_MIN / r90a) ** 2, 1.0, na)) * r90a
    aa = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), na)
    Qa = np.column_stack([ra * np.cos(aa), ra * np.sin(aa), np.zeros(na)])
    gt_a = eta_xt(Qa[:, 0], Qa[:, 1], Qa[:, 2], S)
    gs_a = eta_dr(Qa[:, 0], Qa[:, 1], 0.0, S_ib)
    Qe = np.column_stack([np.zeros(len(ENC_T)), np.zeros(len(ENC_T)), ENC_T])
    gt_e = eta_xt(Qe[:, 0], Qe[:, 1], Qe[:, 2], S)

    kxb, kyb, om = build_bank(kp, BAND, kp / 10.0, DEPTH, DEV, torch.float32)
    kxb, kyb, om = kxb.to(DT), kyb.to(DT), om.to(DT)
    Cf = design(torch.tensor(Qf, device=DEV, dtype=DT), kxb, kyb, om)
    Ca = design(torch.tensor(Qa, device=DEV, dtype=DT), kxb, kyb, om)
    Cs = grad_design(torch.tensor(Qa, device=DEV, dtype=DT), kxb, kyb, om)
    Ce = design(torch.tensor(Qe, device=DEV, dtype=DT), kxb, kyb, om)

    # closed-form prior leverage on a pilot prior: gam0 from a pilot rand-2k
    pilot = rng.choice(n_all, 2000, replace=False)
    tP32 = torch.tensor(P[pilot], device=DEV, dtype=torch.float32)
    tz32 = torch.tensor(zv[pilot], device=DEV, dtype=torch.float32)
    _, mdl0 = mmse_gpu(tP32, tz32, DEPTH, tP32[:64], NOISE, kp, seed=seed,
                       return_model=True)
    gam0 = mdl0['gam'].to(DT)

    def phi_fn(Pq):
        h = np.empty(len(Pq))
        for i in range(0, len(Pq), 20000):
            Phi = design(torch.tensor(Pq[i:i + 20000], device=DEV, dtype=DT),
                         kxb, kyb, om)
            h[i:i + 20000] = ((Phi ** 2) * gam0[None]).sum(1).cpu().numpy()
        return h

    def score(idx):
        Ps, zs = P[idx], zv[idx]
        t32 = torch.tensor(Ps, device=DEV, dtype=torch.float32)
        z32 = torch.tensor(zs, device=DEV, dtype=torch.float32)
        _, mdl = mmse_gpu(t32, z32, DEPTH, t32[:64], NOISE, kp, seed=seed,
                          return_model=True)
        gam, s2 = mdl['gam'].to(DT), float(mdl['s2'])
        Phi = design(torch.tensor(Ps, device=DEV, dtype=DT), kxb, kyb, om)
        A0 = Phi.T @ Phi / s2 + torch.diag(1.0 / gam)
        a = torch.linalg.solve(A0, Phi.T @ torch.tensor(
            zs, device=DEV, dtype=DT) / s2)
        zsd = float(np.std(zs))

        def rho_of(C, gt):
            f = (C @ a).cpu().numpy()
            return float(np.corrcoef(f, gt)[0, 1]) if np.std(f) > 1e-12 else 0.0

        return {'fan': rho_of(Cf, gt_f), 'ann': rho_of(Ca, gt_a),
                'slp': rho_of(Cs, gs_a), 'enc': rho_of(Ce, gt_e)}

    out = {'seed': seed, 'n_all': int(n_all), 'r90': r90a, 'cells': {}}
    for m in BUDGETS:
        for pol in ['rand', 'strat_r', 'strat_rt', 'lev']:
            idx = select(pol, P, rr, m, rng, phi_fn=phi_fn)
            out['cells'][f'{pol}@{m}'] = score(idx)
        idx = select('rand', P, rr, 4 * m, rng)
        out['cells'][f'rand@{4*m}'] = score(idx)
    return out


def main():
    global gen_hits
    h3.sea = sea_dir(180.0)                       # head seas (module global)
    print(f"=== H4-SEL selection ladder  T={T_SEL} seas={N_SEAS} "
          f"budgets={BUDGETS} ===")
    rng = np.random.default_rng(47)
    seas = []
    for s in range(N_SEAS):
        t0 = time.time()
        r = run_sea(s, rng)
        seas.append(r)
        cs = r['cells']
        print(f"sea {s} n_all {r['n_all']} [{time.time()-t0:5.1f}s]")
        for k in cs:
            print(f"   {k:12s} fan {cs[k]['fan']:+.2f} ann {cs[k]['ann']:+.2f}"
                  f" slp {cs[k]['slp']:+.2f} enc {cs[k]['enc']:+.2f}",
                  flush=True)

    print("\n" + "=" * 72)
    agg = {}
    keys = list(seas[0]['cells'].keys())
    for k in keys:
        agg[k] = {f: float(np.mean([s['cells'][k][f] for s in seas]))
                  for f in ['fan', 'ann', 'slp', 'enc']}
        d_enc = np.array([s['cells'][k]['enc'] - s['cells'][
            f"rand@{k.split('@')[1]}"]['enc'] for s in seas]) \
            if not k.startswith('rand') else np.zeros(len(seas))
        se = d_enc.std(ddof=1) / np.sqrt(len(d_enc)) if len(d_enc) > 1 else 0
        print(f"{k:12s} fan {agg[k]['fan']:+.3f} ann {agg[k]['ann']:+.3f} "
              f"slp {agg[k]['slp']:+.3f} enc {agg[k]['enc']:+.3f}"
              + (f"   d_enc vs rand {d_enc.mean():+.3f}+-{se:.3f}"
                 if not k.startswith('rand') else ''))
    json.dump({'seas': seas, 'agg': agg, 'T': T_SEL}, open(OUT, 'w'), indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
