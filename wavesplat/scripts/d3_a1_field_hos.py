"""A-1: HOS M=3 reproduction of the whole-field learned-vs-hand verdict.

Registration (frozen before run): PERSONA_REVIEW_fullfield_worklist \
2026-07-12 §A-1 — direction-only judgment: pooled d(learned-hand) >= 0 on
8 cells (seeds {0,1,2,3} x Hs {1.2, 3.0}, M=3) with per-Hs sign consistency
-> the whole-field headline drops its "real-data-only" qualifier.

Machinery: H7 arena verbatim (fixedT scans, stratified 2000, noise 0.05 m);
arms use the frozen 60-window definitions: hand = mmse_gpu(kp_hint, default
EM path), learned = net_gamma_mmse + smult_head (bw_s0, use_em=False).
Whole-field queries: fan 800 pts (r 15-200 m) x leads {0,2,5,8} s vs the
HOS sim's own truth.
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

from d3_h7_hos_arena import (surface_snap, z_at, scan_hits, SCAN_REL,
                             SPINUP, NOISE)
from d3_f1_hos_door1 import hos_run, sea_components, DT, FOV_H
from d3_door1b_field_bias import R_MIN
from d3_gpu_mmse import mmse_gpu
from d3_prior_net_eval import get_net, net_gamma_mmse, DEVICE
from d3_l1_s2head_train import SmultHead

OUT = 'results/direction3/a1_field_hos.json'
SEEDS = [0, 1, 2, 3]
HS_LIST = [1.2, 3.0]
LEADS = [0.0, 2.0, 5.0, 8.0]


def run_cell(seed, Hs, net, head, rng):
    sim = hos_run(seed, 3, Hs=Hs)
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
    P, zv, rr = np.vstack(P), np.concatenate(zv), np.concatenate(rr)
    zv = zv + rng.normal(0, NOISE, len(zv))

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

    # whole-field fan queries at each lead, truth from the sim itself
    rf = np.sqrt(rng.uniform((15.0 / 200.0) ** 2, 1.0, 800)) * 200.0
    af = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), 800)
    qx, qy = rf * np.cos(af), rf * np.sin(af)
    Q, gt = [], []
    for l_ in LEADS:
        tgt = t_end + l_
        while sim.t < tgt - 1e-9:
            sim.step(min(DT, tgt - sim.t))
        gt.append(z_at(surface_snap(sim), qx, qy))
        Q.append(np.column_stack([qx, qy, np.full(800, l_)]))
    Q, gt = np.vstack(Q), np.concatenate(gt)

    S = sea_components(seed, Hs=Hs)
    kp = float(np.hypot(S['kx'], S['ky'])[np.argmax(S['A'])])
    with torch.no_grad():
        ph = mmse_gpu(torch.tensor(Ps, device=DEVICE, dtype=torch.float32),
                      torch.tensor(zs, device=DEVICE, dtype=torch.float32),
                      50.0,
                      torch.tensor(Q, device=DEVICE, dtype=torch.float32),
                      NOISE, kp_hint=kp).cpu().numpy()
    pn = np.asarray(net_gamma_mmse(net, Ps, zs, 50.0, Q, NOISE, seed=seed,
                                   kp_hint=kp, use_em=False,
                                   smult_head=head))
    row = {'seed': seed, 'Hs': Hs, 'n_obs': int(len(zs)),
           'hand': float(np.corrcoef(ph, gt)[0, 1]),
           'learned': float(np.corrcoef(pn, gt)[0, 1])}
    for li, l_ in enumerate(LEADS):
        s_ = slice(800 * li, 800 * (li + 1))
        row[f'hand_l{int(l_)}'] = float(np.corrcoef(ph[s_], gt[s_])[0, 1])
        row[f'learned_l{int(l_)}'] = float(
            np.corrcoef(pn[s_], gt[s_])[0, 1])
    return row


def main():
    net = get_net('results/direction3/prior_net_l1s2bw.pt')
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(
        'results/direction3/prior_net_l1s2bw.pt', map_location=DEVICE)['head'])
    head.eval()
    rng = np.random.default_rng(7)
    rows = []
    for Hs in HS_LIST:
        for seed in SEEDS:
            t0 = time.time()
            r = run_cell(seed, Hs, net, head, rng)
            rows.append(r)
            print(f"Hs {Hs} seed {seed} [{time.time()-t0:5.0f}s] "
                  f"hand {r['hand']:+.3f} learned {r['learned']:+.3f} "
                  f"d {r['learned']-r['hand']:+.3f}", flush=True)
    json.dump(rows, open(OUT, 'w'), indent=1)
    d = np.array([r['learned'] - r['hand'] for r in rows])
    print(f"\npooled d(learned-hand) {d.mean():+.4f}  cells>0 "
          f"{(d>0).sum()}/{len(d)}")
    for Hs in HS_LIST:
        dh = np.array([r['learned'] - r['hand'] for r in rows
                       if r['Hs'] == Hs])
        print(f"Hs {Hs}: d {dh.mean():+.4f}  sign {(dh>0).sum()}/{len(dh)}")
    ok = d.mean() >= 0 and all(
        np.mean([r['learned'] - r['hand'] for r in rows if r['Hs'] == h])
        >= 0 for h in HS_LIST)
    print(f"A-1 HOS reproduction: {'PASS' if ok else 'FAIL'} "
          f"(frozen: pooled>=0 AND per-Hs sign consistent)")
    print(f"wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
