"""H7 -- the HOS x fixed-T arena: how big is the room learning can win?

LEARNING_DESIGN_v2 (2026-07-11) moved learning to where misspecification is
real. Before any training (accounting first, design law 1), this arena
measures the PRIZE: the same recipe estimator (fixed-T T_m50, range-stratified
2000 obs, misspecification noise floor) on HOS M=3 seas vs M=1 (linear
evolution, same machinery, same seed init -- perfectly paired), at Hs 1.2 and
3.0. The M3-M1 skill gap at each steepness IS the nonlinearity room:

  REGISTERED PREDICTIONS (design doc section 2): the gap grows with steepness
  (mechanism signature); at Hs 1.2 it is small (bound energy ~ (ka)^2 ~ 1-2%),
  at Hs 3.0 it is 0.03-0.08 in nRMSE-equivalent terms. L1/L2 training runs
  ONLY where this room exists; L0b (analytic Stokes-2 on the estimate) is the
  learning-free competitor that must be beaten from that room.

Machinery: A4's HOS plumbing (HOSSimulator M=3, spectral-mode evaluation at
arbitrary points, mesh raycast) + H3-OD's fixed-T selection + H5's recipe.
Head seas (WDIR 180): the encounter axis is causal. Scans: 12 over 20 s after
12 s spinup; forecast leads 0.25-10 s after the last scan.

Scored per (seed, Hs, M): ann rho (r 15-75 m), fan rho (15-200 m), enc rho
(sensor point, 40 leads), each against the HOS truth of ITS OWN sim.
This is an arena-characterisation run, not a gated experiment; the numbers
feed the G-HOS gate of the design doc.
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

from d3_f1_hos_door1 import (hos_run, hos_modes, hos_eval, sea_components,
                             CX, CY, SPINUP, DT, MESH_DX, DOM_X4, DOM_Y4)
from d3_g0b_selection_aware import A_of, DEV, DT as T64, BAND
from d3_gpu_mmse import mmse_gpu, build_bank, design
from d3_door1b_field_bias import ray_grid, R_MIN, R_MAX, FOV_H
from wavesim.lidar_simulator import build_mesh, raycast_surface
from wavesim.raydrop_threshold import T_MANUAL_50

OUT = 'results/direction3/h7_hos_arena.json'
NOISE = 0.05
SCAN_REL = np.linspace(-20.0, 0.0, 12)
ENC_T = np.arange(0.25, 10.001, 0.25)
SEEDS = [0, 1, 2, 3]
HS_LIST = [1.2, 3.0]
M_LIST = [1, 3]

_O, _D, _ = ray_grid()
_gx = np.arange(DOM_X4[0], DOM_X4[1] + MESH_DX, MESH_DX)
_gy = np.arange(DOM_Y4[0], DOM_Y4[1] + MESH_DX, MESH_DX)
_GX, _GY = np.meshgrid(_gx, _gy)


def surface_snap(sim):
    kx, ky, a = hos_modes(sim)
    return (kx, ky, a)


def z_at(snap, xs, ys):
    return hos_eval(*snap, CX + np.asarray(xs, float),
                    CY + np.asarray(ys, float), grad=False)


def slope_at(snap, xs, ys):
    ex, ey = hos_eval(*snap, CX + np.asarray(xs, float),
                      CY + np.asarray(ys, float), grad=True)
    r = np.clip(np.hypot(xs, ys), 1e-6, None)
    return (xs / r) * ex + (ys / r) * ey


def scan_hits(snap, rng):
    Z = z_at(snap, _GX.ravel(), _GY.ravel()).reshape(_GX.shape)
    V, T = build_mesh(_GX, _GY, Z)
    hm, hp, _ = raycast_surface(_O, _D, V, T, range_min=R_MIN,
                                range_max=R_MAX, batch_size=8000)
    idx = np.where(hm)[0]
    r = np.hypot(hp[idx, 0], hp[idx, 1])
    ok = (r > R_MIN) & (r < R_MAX)
    idx, r = idx[ok], r[ok]
    s = slope_at(snap, hp[idx, 0], hp[idx, 1])
    pw = A_of(s, r) * rng.exponential(1.0, len(idx))
    keep = pw > T_MANUAL_50
    return hp[idx[keep], :2], hp[idx[keep], 2], r[keep]


def run_cell(seed, Hs, M, rng):
    sim = hos_run(seed, M, Hs=Hs)
    while sim.t < SPINUP - 1e-9:
        sim.step(min(DT, SPINUP - sim.t))

    P, zv, rr = [], [], []
    t_end = SPINUP + 20.0
    for trel in SCAN_REL:
        tgt = t_end + trel
        while sim.t < tgt - 1e-9:
            sim.step(min(DT, tgt - sim.t))
        snap = surface_snap(sim)
        xy, z, r = scan_hits(snap, rng)
        P.append(np.column_stack([xy, np.full(len(z), trel)]))
        zv.append(z)
        rr.append(r)
        if trel == 0.0:
            snap_end = snap
    P, zv, rr = np.vstack(P), np.concatenate(zv), np.concatenate(rr)
    zv = zv + rng.normal(0, NOISE, len(zv))

    enc_gt = []
    for tl in ENC_T:
        tgt = t_end + tl
        while sim.t < tgt - 1e-9:
            sim.step(min(DT, tgt - sim.t))
        enc_gt.append(float(z_at(surface_snap(sim), np.zeros(1),
                                 np.zeros(1))[0]))
    enc_gt = np.array(enc_gt)

    # recipe: stratified 2000 + noise floor
    edges = np.linspace(R_MIN, rr.max() + 1e-6, 13)
    b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
    take = []
    for j in range(12):
        idx = np.where(b == j)[0]
        k = min(len(idx), 167)
        if k:
            take.append(rng.choice(idx, k, replace=False))
    sel = np.concatenate(take)
    Ps, zs = P[sel], zv[sel]

    S = sea_components(seed, Hs=Hs)
    kp = float(np.hypot(S['kx'], S['ky'])[np.argmax(S['A'])])
    t32 = torch.tensor(Ps, device=DEV, dtype=torch.float32)
    z32 = torch.tensor(zs, device=DEV, dtype=torch.float32)
    _, mdl = mmse_gpu(t32, z32, 50.0, t32[:64], NOISE, kp, seed=seed,
                      return_model=True, no_em=True, pexp=1.0, smult=10.0)
    gam, s2 = mdl['gam'].to(T64), float(mdl['s2'])
    kxb, kyb, om = build_bank(kp, BAND, kp / 10.0, 50.0, DEV, torch.float32)
    kxb, kyb, om = kxb.to(T64), kyb.to(T64), om.to(T64)
    Phi = design(torch.tensor(Ps, device=DEV, dtype=T64), kxb, kyb, om)
    A0 = Phi.T @ Phi / s2 + torch.diag(1.0 / gam)
    a = torch.linalg.solve(A0, Phi.T @ torch.tensor(
        zs, device=DEV, dtype=T64) / s2)

    def rho_pts(Q, gt):
        C = design(torch.tensor(Q, device=DEV, dtype=T64), kxb, kyb, om)
        f = (C @ a).cpu().numpy()
        return float(np.corrcoef(f, gt)[0, 1]) if np.std(f) > 1e-12 else 0.0

    na = 400
    ra = np.sqrt(rng.uniform((15.0 / 75.0) ** 2, 1.0, na)) * 75.0
    aa = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), na)
    Qa = np.column_stack([ra * np.cos(aa), ra * np.sin(aa), np.zeros(na)])
    gt_a = z_at(snap_end, Qa[:, 0], Qa[:, 1])
    rf = np.sqrt(rng.uniform((15.0 / 200.0) ** 2, 1.0, 800)) * 200.0
    af = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), 800)
    Qf = np.column_stack([rf * np.cos(af), rf * np.sin(af), np.zeros(800)])
    gt_f = z_at(snap_end, Qf[:, 0], Qf[:, 1])
    Qe = np.column_stack([np.zeros(len(ENC_T)), np.zeros(len(ENC_T)), ENC_T])

    return {'seed': seed, 'Hs': Hs, 'M': M, 'n_ret': int(len(P)),
            'sig': float(np.std(gt_f)),
            'ann': rho_pts(Qa, gt_a), 'fan': rho_pts(Qf, gt_f),
            'enc': rho_pts(Qe, enc_gt)}


def main():
    print(f"=== H7 HOS arena  seeds={SEEDS} Hs={HS_LIST} M={M_LIST} ===")
    rng = np.random.default_rng(7)
    rows = []
    for Hs in HS_LIST:
        for seed in SEEDS:
            for M in M_LIST:
                t0 = time.time()
                r = run_cell(seed, Hs, M, rng)
                rows.append(r)
                print(f"Hs {Hs} seed {seed} M{M} [{time.time()-t0:5.0f}s] "
                      f"n {r['n_ret']:6d} sig {r['sig']:.2f} | "
                      f"ann {r['ann']:+.2f} fan {r['fan']:+.2f} "
                      f"enc {r['enc']:+.2f}", flush=True)

    print("\n" + "=" * 64)
    for Hs in HS_LIST:
        for f in ('ann', 'fan', 'enc'):
            g = [next(r[f] for r in rows
                      if r['seed'] == s and r['Hs'] == Hs and r['M'] == 1)
                 - next(r[f] for r in rows
                        if r['seed'] == s and r['Hs'] == Hs and r['M'] == 3)
                 for s in SEEDS]
            g = np.array(g)
            se = g.std(ddof=1) / np.sqrt(len(g))
            print(f"Hs {Hs} {f:4s}: room (M1-M3 rho) = {g.mean():+.4f} "
                  f"+- {se:.4f}")
    json.dump(rows, open(OUT, 'w'), indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
