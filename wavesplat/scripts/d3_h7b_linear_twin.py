"""H7b -- the LINEAR TWIN reference for the steep block.

H7's M=1 arm at Hs 3.0 blew up (RK4 NaN -> hos_modes' |a| filter sees all-NaN
-> empty modes -> flat sea; HOS "M=1" keeps partial BC nonlinearity and is
NOT unconditionally stable). The correct paired reference is analytic: take
the SAME initial modes a0 the M=3 sim starts from (initialize_jonswap(seed)),
and evolve them by pure phase rotation a(t) = a0 exp(-/+ i omega(k) t),
omega = sqrt(g k tanh(k d)). Unconditionally stable, exactly phase-paired at
t = 0, no stepping cost.

Sign convention is VALIDATED, not assumed: at Hs 1.2 seed 0 the twin surface
at t = 32 s is correlated against the (healthy) M=1 HOS surface for both
signs; the winning sign must give rho > 0.95, else abort. Then the steep
block (Hs 3.0, seeds 0-3) is scored with the H7 machinery and the room is
read as twin - M3 per functional.
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

from d3_f1_hos_door1 import hos_run, hos_modes, hos_eval, sea_components, CX, CY
from d3_h7_hos_arena import (scan_hits, z_at, SCAN_REL, ENC_T, NOISE,
                             SEEDS, OUT as H7_OUT)
import d3_h7_hos_arena as h7
from d3_g0b_selection_aware import DEV, DT as T64, BAND
from d3_gpu_mmse import mmse_gpu, build_bank, design
from d3_door1b_field_bias import R_MIN, FOV_H

OUT = 'results/direction3/h7b_linear_twin.json'
G = 9.81
DEPTH = 50.0


class Twin:
    def __init__(self, seed, Hs, sign):
        sim = hos_run(seed, 3, Hs=Hs)          # init only; M irrelevant at t=0
        self.kx, self.ky, self.a0 = hos_modes(sim)
        k = np.hypot(self.kx, self.ky)
        self.om = np.sqrt(G * k * np.tanh(np.clip(k * DEPTH, 0, 20)))
        self.sign = sign

    def snap(self, t):
        return (self.kx, self.ky,
                self.a0 * np.exp(self.sign * 1j * self.om * t))


def validate_sign():
    """twin vs healthy M=1 HOS at Hs 1.2, t = 32 s, both signs."""
    sim = hos_run(0, 1, Hs=1.2)
    while sim.t < 32.0 - 1e-9:
        sim.step(min(0.1, 32.0 - sim.t))
    snap_m1 = hos_modes(sim)
    rng = np.random.default_rng(3)
    xs = rng.uniform(0, 260, 2000)
    ys = rng.uniform(-200, 200, 2000)
    z_ref = hos_eval(*snap_m1, CX + xs, CY + ys, grad=False)
    out = {}
    for sign in (-1.0, +1.0):
        tw = Twin(0, 1.2, sign)
        z_tw = hos_eval(*tw.snap(32.0), CX + xs, CY + ys, grad=False)
        out[sign] = float(np.corrcoef(z_tw, z_ref)[0, 1])
    print(f"sign validation vs M1@Hs1.2 t=32s: -1 -> {out[-1.0]:+.3f}, "
          f"+1 -> {out[+1.0]:+.3f}")
    best = max(out, key=out.get)
    assert out[best] > 0.95, f"no sign wins ({out}); convention audit needed"
    return best


def run_cell(seed, Hs, sign, rng):
    tw = Twin(seed, Hs, sign)
    t_end = 32.0
    P, zv, rr = [], [], []
    for trel in SCAN_REL:
        snap = tw.snap(t_end + trel)
        xy, z, r = scan_hits(snap, rng)
        P.append(np.column_stack([xy, np.full(len(z), trel)]))
        zv.append(z)
        rr.append(r)
        if trel == 0.0:
            snap_end = snap
    P, zv, rr = np.vstack(P), np.concatenate(zv), np.concatenate(rr)
    zv = zv + rng.normal(0, NOISE, len(zv))
    enc_gt = np.array([float(z_at(tw.snap(t_end + tl), np.zeros(1),
                                  np.zeros(1))[0]) for tl in ENC_T])

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
    _, mdl = mmse_gpu(t32, z32, DEPTH, t32[:64], NOISE, kp, seed=seed,
                      return_model=True, no_em=True, pexp=1.0, smult=10.0)
    gam, s2 = mdl['gam'].to(T64), float(mdl['s2'])
    kxb, kyb, om = build_bank(kp, BAND, kp / 10.0, DEPTH, DEV, torch.float32)
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
    return {'seed': seed, 'Hs': Hs, 'ann': rho_pts(Qa, gt_a),
            'fan': rho_pts(Qf, gt_f), 'enc': rho_pts(Qe, enc_gt),
            'sig': float(np.std(gt_f)), 'n_ret': int(len(P))}


def main():
    sign = validate_sign()
    rng = np.random.default_rng(7)
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        r = run_cell(seed, 3.0, sign, rng)
        rows.append(r)
        print(f"twin Hs 3.0 seed {seed} [{time.time()-t0:4.0f}s] "
              f"sig {r['sig']:.2f} | ann {r['ann']:+.2f} "
              f"fan {r['fan']:+.2f} enc {r['enc']:+.2f}", flush=True)

    h7rows = json.load(open(H7_OUT))
    print("\nROOM (twin - M3) @ Hs 3.0:")
    for f in ('ann', 'fan', 'enc'):
        d = [next(t[f] for t in rows if t['seed'] == s)
             - next(r[f] for r in h7rows
                    if r['seed'] == s and r['Hs'] == 3.0 and r['M'] == 3)
             for s in SEEDS]
        d = np.array(d)
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {f:4s}: {d.mean():+.4f} +- {se:.4f}")
    json.dump({'sign': sign, 'rows': rows}, open(OUT, 'w'), indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
