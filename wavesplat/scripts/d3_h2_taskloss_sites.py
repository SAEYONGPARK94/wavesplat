"""H2 -- TASK-LOSS site head: learn to CONVERT the mask's accounting surplus.

Context. F2b: the mask carries 17% of the amplitude information (MI, gate-
PASS). G0b: the hand EP converts NONE of it into far-field accuracy (gain
-0.0000 +- 0.0020). H1 (d3_h1_mask_destination.py) measures the remaining
functionals; its HA accounting predicts a small annulus-elevation surplus
(pred_gain_frac ~ 0.024 of naive 0.34 => ~ +0.008 nRMSE) that the EP again
may or may not realize. The accounting-estimation gap has three named,
LEARNABLE loss sites (G0b autopsy, "secondary losses"): the miss-footprint
approximation (the true hit of a miss is unobservable -> proxy site
location), EP's Gaussian moment-matching, and the subsample precision
scaling. None of these is information -- they are INFERENCE deficiencies.

So the learning target is NOT to amortize the EP (③'s dead end: teacher gain
zero => nothing to amortize) but to EXCEED it: train the N5 site head
THROUGH the solve on the field error itself. This is the v3 lesson
(2026-07-05: label-regression failed, task-loss training passed G1) applied
at the likelihood site -- the one place the map allows learning to live.

Machinery: N5's SiteHead (7 deployment-observable features, L3 zero-start
init), N5's differentiable solve, H1's head-sea scene family and annulus
instrument. The teacher is a computation only for BASELINES (naive, EP);
training uses synthetic GT (design law 5: the teacher is a computation).

================================ PRE-REGISTRATION ============================
Frozen 2026-07-11 BEFORE H1's verdict and before the first training step.
Train seeds 100-111 (12 MNAR) + 300-303 (4 MCAR mixed per L3), val seeds
200-207 (8 seas, never trained). All head-sea (dir 180), AMP 3.33.

  HL-R1 ZERO-START. At step 0 (L3 init, tau ~ 0):
        |nrmse(net) - nrmse(naive)| <= 1e-4 on every val sea.
  HL1   VALUE. Val paired gain naive - net on annulus elevation (H1
        instrument, B1 cap): mean >= 0.005 AND t >= 3 (n = 8, clustered).
  HL2   CONVERSION. net beats the hand EP: (gain_net - gain_EP) > 0, t >= 2.
        This is the claim that the accounting-estimation gap was inference
        loss, not phantom information.
  HL3   MCAR-ZERO. mean tau(net) on MCAR val scenes <= 1% of MNAR mean
        (the learned do-no-harm, ③'s L3 verification, now under task loss).
  HL4   CEILING. gain_net <= 2 x HA-pred(eta_inf) from H1 main (~0.016).
        Above => accounting leak audit BEFORE any positive verdict.
  HL5   HOS. Any positive headline must reproduce on HOS seas (standing
        instruction). Separate run.
DECISION: HL1 & HL2 & HL3 & HL4 -> learning converts information the hand
inference provably leaves on the table: the accuracy chapter's positive.
HL1 fails with HL3 healthy -> the gap is NOT inference loss at this
operating point; the last accuracy cell closes at the measured ceiling, and
the number itself goes in the map. Both endings publishable; frozen now.
=============================================================================
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

import d3_g0b_selection_aware as g0b
from d3_g0b_selection_aware import DEV, DT, BAND
from d3_s2_mask_information import grad_design
from d3_door1b_field_bias import eta_dr, R_MIN, FOV_H
from d3_h1_mask_destination import sea_dir, eta_xt
from d3_n5_site_head import scene_tensors, teacher_sites, solve_with, SiteHead

OUT = 'results/direction3/h2_taskloss_sites'
TRAIN_SEEDS = list(range(100, 112))
MCAR_SEEDS = list(range(300, 304))
VAL_SEEDS = list(range(200, 208))
N_QINF = 400


def add_annulus(sc, rng):
    """H1 instrument: dense annulus queries r in [R_MIN, r90], + GT."""
    r90 = float(np.quantile(sc['ret_r'], 0.90))
    rq = np.sqrt(rng.uniform((R_MIN / r90) ** 2, 1.0, N_QINF)) * r90
    aq = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), N_QINF)
    Qi = np.column_stack([rq * np.cos(aq), rq * np.sin(aq), np.zeros(N_QINF)])
    gt = eta_xt(Qi[:, 0], Qi[:, 1], Qi[:, 2], sc['S'])
    from d3_gpu_mmse import design
    sc['Ca'] = design(torch.tensor(Qi, device=DEV, dtype=DT),
                      sc['kxb'], sc['kyb'], sc['om'])
    sc['gt_a'] = torch.tensor(gt, device=DEV, dtype=DT)
    sc['sig_a'] = float(np.std(gt))
    sc['r90'] = r90


def ann_nrmse(sc, a):
    """B1-capped annulus nRMSE -- the H1 instrument, torch end-to-end."""
    f = sc['Ca'] @ a
    beta = (sc['z'].std() / f.std().clamp_min(1e-12)).clamp(max=1.0)
    e = beta * f - sc['gt_a']
    return torch.sqrt((e ** 2).mean()) / max(sc['sig_a'], 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default=None)
    a_ = ap.parse_args()
    tag = a_.tag or f's{a_.seed}'
    torch.manual_seed(a_.seed)

    g0b.sea = sea_dir(180.0)                     # H1 main scene family
    rng = np.random.default_rng(77 + a_.seed)

    print(f"=== H2 task-loss site head  steps={a_.steps} seed={a_.seed} ===")
    print("PRE-REGISTERED: HL-R1 zero-start | HL1 val gain>=0.005 t>=3 |")
    print("  HL2 net>EP t>=2 | HL3 mcar tau<=1% | HL4 <=2x HA ceiling | HL5 HOS\n")

    scenes, val = [], []
    t0 = time.time()
    for sd in TRAIN_SEEDS:
        sc = scene_tensors(sd, rng); add_annulus(sc, rng); scenes.append(sc)
    for sd in MCAR_SEEDS:
        sc = scene_tensors(sd, rng, mcar=True); add_annulus(sc, rng)
        scenes.append(sc)
    for sd in VAL_SEEDS:
        sc = scene_tensors(sd, rng); add_annulus(sc, rng); val.append(sc)
    print(f"scenes built in {time.time()-t0:.0f}s "
          f"({len(scenes)} train incl {len(MCAR_SEEDS)} mcar, {len(val)} val)")

    # #50 checklist: feature/target dynamic-range and NaN audit BEFORE training
    for sc in scenes + val:
        assert torch.isfinite(sc['feats']).all(), 'NaN feature leaked'
        sc['G'] = grad_design(sc['sitesP'], sc['kxb'], sc['kyb'], sc['om'])
        assert torch.isfinite(sc['G']).all()

    # baselines on val: naive and hand-EP, once
    base = []
    for sc in val:
        a_n = solve_with(sc, None, None, None)
        G, tau, nu = teacher_sites(sc)
        a_e = solve_with(sc, G, tau, nu)
        base.append((float(ann_nrmse(sc, a_n)), float(ann_nrmse(sc, a_e))))
        del G, tau, nu
    b = np.array(base)
    print(f"val baselines: naive {b[:,0].mean():.4f}  EP {b[:,1].mean():.4f}  "
          f"EP gain {(b[:,0]-b[:,1]).mean():+.4f}+-"
          f"{(b[:,0]-b[:,1]).std(ddof=1)/np.sqrt(len(b)):.4f}")

    head = SiteHead().to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=a_.lr)

    # HL-R1: zero-start reduction
    with torch.no_grad():
        r1 = []
        for i, sc in enumerate(val):
            tau, nu = head.sites(sc['feats'])
            a0 = solve_with(sc, sc['G'], tau, nu)
            r1.append(abs(float(ann_nrmse(sc, a0)) - b[i, 0]))
        print(f"HL-R1 zero-start max|d| = {max(r1):.2e} "
              f"({'PASS' if max(r1) <= 1e-4 else 'FAIL'})")

    hist = []
    t0 = time.time()
    for step in range(1, a_.steps + 1):
        sc = scenes[int(torch.randint(len(scenes), (1,)))]
        tau, nu = head.sites(sc['feats'])
        a = solve_with(sc, sc['G'], tau, nu)
        loss = ann_nrmse(sc, a) ** 2
        if sc['mcar']:
            tt, _ = head(sc['feats'])
            loss = loss + 1e-3 * tt.double().mean()   # L3: explicit do-no-harm
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == a_.steps:
            with torch.no_grad():
                g_net, tt_mnar, tt_mcar = [], [], []
                for i, sc_v in enumerate(val):
                    tau, nu = head.sites(sc_v['feats'])
                    a_v = solve_with(sc_v, sc_v['G'], tau, nu)
                    g_net.append(b[i, 0] - float(ann_nrmse(sc_v, a_v)))
                for sc_t in scenes:
                    tt, _ = head(sc_t['feats'])
                    (tt_mcar if sc_t['mcar'] else tt_mnar).append(
                        float(tt.mean()))
                g = np.array(g_net)
                se = g.std(ddof=1) / np.sqrt(len(g))
                mc_ratio = (np.mean(tt_mcar) /
                            max(np.mean(tt_mnar), 1e-12))
                print(f"step {step:5d} [{time.time()-t0:6.0f}s] "
                      f"loss {float(loss):.4f}  val gain {g.mean():+.4f}"
                      f"+-{se:.4f} t {g.mean()/max(se,1e-12):+.2f}  "
                      f"mcar/mnar tt {mc_ratio:.4f}", flush=True)
                hist.append({'step': step, 'gain': float(g.mean()),
                             'se': float(se), 'mcar_ratio': float(mc_ratio),
                             'gains': g.tolist()})

    # final verdict rows
    g = np.array(hist[-1]['gains'])
    se = g.std(ddof=1) / np.sqrt(len(g))
    ep_g = b[:, 0] - b[:, 1]
    d = g - ep_g
    dse = d.std(ddof=1) / np.sqrt(len(d))
    print("\n" + "=" * 72)
    print(f"HL1 net gain   {g.mean():+.4f}+-{se:.4f}  t {g.mean()/se:+.2f}  "
          f"({'PASS' if g.mean() >= 0.005 and g.mean()/se >= 3 else 'FAIL'})")
    print(f"HL2 net - EP   {d.mean():+.4f}+-{dse:.4f}  t {d.mean()/dse:+.2f}  "
          f"({'PASS' if d.mean() > 0 and d.mean()/dse >= 2 else 'FAIL'})")
    print(f"HL3 mcar/mnar  {hist[-1]['mcar_ratio']:.4f}  "
          f"({'PASS' if hist[-1]['mcar_ratio'] <= 0.01 else 'FAIL'})")

    os.makedirs('results/direction3', exist_ok=True)
    torch.save(head.state_dict(), f'{OUT}_{tag}.pt')
    with open(f'{OUT}_{tag}.json', 'w') as f:
        json.dump({'steps': a_.steps, 'seed': a_.seed,
                   'val_naive': b[:, 0].tolist(), 'val_ep': b[:, 1].tolist(),
                   'hist': hist}, f, indent=2)
    print(f"wrote {OUT}_{tag}.json / .pt")
    return 0


if __name__ == '__main__':
    sys.exit(main())
