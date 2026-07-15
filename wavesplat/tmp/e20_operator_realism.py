"""E-20 (closing experiment #3) — operator-realism stress test.

The examiner's cheap de-risk for the "stereo-not-LiDAR" gap: you cannot get a
real Falcon in time, but you CAN inject the known real-LiDAR failure modes into
the simulated operator and re-run the decomposition. If the aperture-rho and the
short-lam sign-flip SURVIVE operator degradation, the geometry-vs-radiometry
worry is bounded and the aperture claim is robust; if they collapse, you found
the real boundary before an examiner did.

Failure modes injected in the DRAW (the clean raycast geometry is reused from
E-18's build_cell; degradation is applied to the returns):
  * range-dependent dropout  : keep prob decays as exp(-(r-R_MIN)/R_DECAY)
  * incidence-angle noise     : elevation noise grows with range, sigma(r) =
                                NOISE*(1 + INC_GROW*(r-R_MIN)/(R_MAX-R_MIN))
  * motion smear              : xy jitter N(0, SMEAR) from platform motion
Same fixed fan operator, same decorrelated grid as E-18; compare aperture-rho
and the sign-flip to the clean E-18 run.

  python3 tmp/e20_operator_realism.py
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp')
os.environ.setdefault('A1_N_OBS', '600')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from e18_decorrelated_2factor import build_cell, _lam, TPS, STEEPS, R_APERTURE  # noqa

OUT = 'results/direction3/e20_operator_realism.json'
R_DECAY = 120.0        # dropout e-fold range (m): returns thin out with distance
INC_GROW = 2.0         # noise triples by max range (incidence-angle proxy)
SMEAR = 0.5            # xy jitter std (m): platform motion during scan
NOISE = 0.05


def degraded_draw(sea, rng, net, head):
    """one_draw with real-LiDAR failure modes injected into the returns."""
    import torch
    from e15_hos_multidraw import N_OBS, LEADS
    from d3_h7_hos_arena import z_at
    from d3_door1b_field_bias import R_MIN, R_MAX, FOV_H
    from wavesim.raydrop_threshold import T_MANUAL_50
    from d3_gpu_mmse import mmse_gpu
    from d3_prior_net_eval import net_gamma_mmse, DEVICE
    DEPTH = 50.0
    P, zv, rr = [], [], []
    for sc in sea['scans']:
        pw = sc['amp'] * rng.exponential(1.0, len(sc['amp']))
        keep = pw > T_MANUAL_50
        # (1) range-dependent dropout
        pdrop = np.exp(-(sc['r'] - R_MIN) / R_DECAY)
        keep &= rng.random(len(keep)) < pdrop
        xy = sc['xy'][keep].copy()
        # (3) motion smear
        xy += rng.normal(0.0, SMEAR, xy.shape)
        P.append(np.column_stack([xy, np.full(int(keep.sum()), sc['trel'])]))
        zv.append(sc['z'][keep]); rr.append(sc['r'][keep])
    P, zv, rr = np.vstack(P), np.concatenate(zv), np.concatenate(rr)
    # (2) incidence-angle (range-growing) noise
    sig = NOISE * (1.0 + INC_GROW * (rr - R_MIN) / (R_MAX - R_MIN))
    zv = zv + rng.normal(0.0, 1.0, len(zv)) * sig
    edges = np.linspace(R_MIN, rr.max() + 1e-6, 13)
    b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
    take = []
    for j in range(12):
        idx = np.where(b == j)[0]
        k = min(len(idx), max(-(-N_OBS // 12), 1))
        if k:
            take.append(rng.choice(idx, k, replace=False))
    sel = np.concatenate(take)
    Ps, zs = P[sel], zv[sel]
    rf = np.sqrt(rng.uniform((R_MIN / R_MAX) ** 2, 1.0, 800)) * R_MAX
    af = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), 800)
    qx, qy = rf * np.cos(af), rf * np.sin(af)
    Q, gt = [], []
    for li, l_ in enumerate(LEADS):
        gt.append(z_at(sea['leads'][li], qx, qy))
        Q.append(np.column_stack([qx, qy, np.full(800, l_)]))
    Q, gt = np.vstack(Q), np.concatenate(gt)
    kp = sea['kp']
    with torch.no_grad():
        ph = mmse_gpu(torch.tensor(Ps, device=DEVICE, dtype=torch.float32),
                      torch.tensor(zs, device=DEVICE, dtype=torch.float32), DEPTH,
                      torch.tensor(Q, device=DEVICE, dtype=torch.float32),
                      NOISE, kp_hint=kp).cpu().numpy()
    pn = np.asarray(net_gamma_mmse(net, Ps, zs, DEPTH, Q, NOISE, seed=0,
                                   kp_hint=kp, use_em=False, smult_head=head))
    return float(np.corrcoef(ph, gt)[0, 1]), float(np.corrcoef(pn, gt)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--draws', type=int, default=4)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    seas = [(Tp, round(st * _lam(Tp), 3)) for Tp in TPS for st in STEEPS]
    tasks = [(Tp, Hs, sd, order) for (Tp, Hs) in seas
             for sd in a.seeds for order in (1, 3)]
    print(f"E-20 (degraded op): {len(seas)} seas x {len(a.seeds)} phases x2 = "
          f"{len(tasks)} cells; dropout Rd={R_DECAY} incGrow={INC_GROW} "
          f"smear={SMEAR}", flush=True)
    t0 = time.time()
    built = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for task, sea, note, dt in ex.map(build_cell, tasks):
            built[task] = sea
    print(f"  built {sum(v is not None for v in built.values())}/{len(tasks)} in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)

    import torch
    from e15_hos_multidraw import CKPT
    from d3_prior_net_eval import get_net, DEVICE
    from d3_l1_s2head_train import SmultHead
    net = get_net(CKPT)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head'])
    head.eval()

    rows = []
    for (Tp, Hs) in seas:
        lam = _lam(Tp); ar = R_APERTURE / lam
        for sd in a.seeds:
            sl, s3 = built[(Tp, Hs, sd, 1)], built[(Tp, Hs, sd, 3)]
            if sl is None or s3 is None:
                continue
            dl, d3 = [], []
            for rr in range(a.draws):
                rng = np.random.default_rng(700000 + 31 * rr)
                h, n_ = degraded_draw(sl, rng, net, head); dl.append(n_ - h)
                rng = np.random.default_rng(700000 + 31 * rr)
                h, n_ = degraded_draw(s3, rng, net, head); d3.append(n_ - h)
            dlin, dm3 = float(np.mean(dl)), float(np.mean(d3))
            rows.append(dict(sea=f'Tp{Tp:g}_Hs{Hs:g}', Tp=Tp, Hs=Hs, lam=lam,
                             steepness=Hs / lam, aperture_ratio=ar, seed=sd,
                             d_lin=dlin, d_m3=dm3, dd=dm3 - dlin,
                             flip=bool(dlin < 0 and dm3 > 0)))
            print(f"[DEG Tp{Tp:g} Hs{Hs:g} ar{ar:.2f} steep{Hs/lam:.3f} ph{sd}] "
                  f"Dlin {dlin:+.3f} Dm3 {dm3:+.3f} dd {dm3-dlin:+.3f}", flush=True)
            json.dump(rows, open(a.out, 'w'), indent=1)

    rows = [r for r in rows if np.isfinite(r['dd']) and np.isfinite(r['d_lin'])]
    dd = np.array([r['dd'] for r in rows]); dlin = np.array([r['d_lin'] for r in rows])
    arv = np.array([r['aperture_ratio'] for r in rows])
    stv = np.array([r['steepness'] for r in rows])
    r_ap, p_ap = stats.spearmanr(arv, dlin)
    r_st, p_st = stats.spearmanr(stv, dd)
    hi = arv >= np.median(arv)
    print(f"\n{'='*66}\nE-20 DEGRADED operator ({len(rows)} cells)\n{'='*66}")
    print(f"  Spearman(aperture_ratio, Delta_lin) {r_ap:+.3f} p {p_ap:.4f} "
          f"(clean E-18 was strongly negative)")
    print(f"  Spearman(steepness, dd)             {r_st:+.3f} p {p_st:.4f}")
    print(f"  high-aperture: dd {dd[hi].mean():+.4f} Dlin {dlin[hi].mean():+.4f} "
          f"(neg {(dlin[hi]<0).sum()}/{hi.sum()}) flip "
          f"{sum(r['flip'] for r,h in zip(rows,hi) if h)}/{hi.sum()}")
    print(f"  low-aperture : dd {dd[~hi].mean():+.4f} Dlin {dlin[~hi].mean():+.4f}")
    json.dump({'rows': rows, 'summary': dict(
        rho_ap_dlin=float(r_ap), p_ap=float(p_ap), rho_st_dd=float(r_st),
        p_st=float(p_st), dd_hi=float(dd[hi].mean()), dd_lo=float(dd[~hi].mean()),
        dlin_hi=float(dlin[hi].mean()), n_flip_hi=int(sum(
            r['flip'] for r, h in zip(rows, hi) if h)),
        degrade=dict(R_DECAY=R_DECAY, INC_GROW=INC_GROW, SMEAR=SMEAR))},
        open(a.out, 'w'), indent=2)
    print(f"\n-> {a.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
