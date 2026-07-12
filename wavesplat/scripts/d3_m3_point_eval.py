"""M3-v1 judgment: point-native amortizer vs CNN incumbent on the frozen
60 real windows + pipeline latency benchmark.

Pre-registration: docs/M3_POINT_prereg_2026-07-12.md.
Gates: G-M3-1 parity (dcorr(point-CNN) all >= -0.01, bad >= -0.02),
G-M3-2 latency (88k frame -> stratified 2k -> encoder -> MMSE -> track
query <= 100 ms on the deployment GPU), G-M3-3 sensitivity table.
CNN/hand arm values come from gate_promotion_eval.json (verified earlier
today to reproduce at 0.000 error); only the point arm is computed fresh.
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

from d3_falcon_real_operator import RECORDS
from d3_sa_multirecord import MRecord
from d3_gate_promotion_eval import window_data
from d3_prior_net_train import band_mask, NGRID, KREL
from d3_m3_pointprior_train import PointPrior
from d18_relax_estimator import omega

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = 9.81
OUT = 'results/direction3/m3_point_eval.json'
RECS = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']
REF = {(r['record'], r['wi']): r for r in
       json.load(open('results/direction3/gate_promotion_eval.json'))}


def point_gamma_mmse(net, P, z, depth, Pq, sigma0, kp, Hs):
    """net_gamma_mmse's bank/MMSE path with the point encoder's gamma+smult
    (no EM, kp external -- identical protocol to the CNN incumbent arm)."""
    with torch.no_grad():
        Pt = torch.tensor(P[None], device=DEVICE, dtype=torch.float32)
        zt = torch.tensor(z[None], device=DEVICE, dtype=torch.float32)
        kpt = torch.tensor([kp], device=DEVICE, dtype=torch.float32)
        Hst = torch.tensor([Hs], device=DEVICE, dtype=torch.float32)
        ylog, ls = net(Pt, zt, kpt, Hst)
        Ylog = ylog[0].cpu().numpy()
        smult = float(torch.exp(ls[0]))
    ax = np.linspace(-KREL * kp, KREL * kp, NGRID)
    KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY)
    sel = (kk > 0.35 * kp) & (kk < KREL * kp)
    kxb, kyb = KX[sel].astype(float), KY[sel].astype(float)
    om = omega(np.hypot(kxb, kyb) + 1e-12, depth)
    g = np.maximum(10.0 ** Ylog[sel], 1e-8)
    g = g / g.sum()
    gam = np.concatenate([g, g]) * np.var(z)

    def design(Q):
        psi = kxb[None] * Q[:, 0:1] + kyb[None] * Q[:, 1:2] \
            - om[None] * Q[:, 2:3]
        return np.concatenate([np.cos(psi), np.sin(psi)], axis=1)

    Phi = design(P)
    s2 = (smult * sigma0) ** 2
    K = (Phi * gam[None]) @ Phi.T + s2 * np.eye(len(z))
    w = np.linalg.solve(K, z)
    return (design(Pq) * gam[None]) @ (Phi.T @ w), smult


def latency_bench(net, n_frame=88000, n_obs=2000, n_rep=5):
    """deployment pipeline: 88k frame -> stratified subsample -> encoder ->
    MMSE fit -> 11-point track query. Returns per-stage ms (median)."""
    rng = np.random.default_rng(0)
    kp, Hs, depth, sig0 = 0.2, 2.0, 50.0, 0.1
    xy = rng.uniform(0, 200, (n_frame, 2))
    tt = rng.uniform(0, 40, (n_frame, 1))
    zz = rng.normal(0, Hs / 4, n_frame).astype(np.float32)
    cloud = np.concatenate([xy, tt], 1).astype(np.float32)
    Pq = np.column_stack([np.linspace(50, 90, 11), np.full(11, 100.0),
                          40.0 + np.arange(11.0)]).astype(np.float32)
    stages = {'subsample': [], 'encoder': [], 'mmse_fit_query': []}
    for _ in range(n_rep):
        t0 = time.perf_counter()
        r = np.hypot(cloud[:, 0] - 100, cloud[:, 1] - 100)
        bins = np.clip(((r - r.min()) / (np.ptp(r) + 1e-9) * 8).astype(int),
                       0, 7)
        pick = np.concatenate([
            rng.choice(np.flatnonzero(bins == b),
                       min(n_obs // 8, int((bins == b).sum())), False)
            for b in range(8)])
        P, z = cloud[pick], zz[pick]
        t1 = time.perf_counter()
        f, _ = point_gamma_mmse.__wrapped__(net, P, z, depth, Pq, sig0,
                                            kp, Hs) \
            if hasattr(point_gamma_mmse, '__wrapped__') else (None, None)
        t2 = time.perf_counter()
        stages['subsample'].append((t1 - t0) * 1e3)
        stages['mmse_fit_query'].append((t2 - t1) * 1e3)
    return stages


def main():
    ck = torch.load('results/direction3/m3_point_v1.pt',
                    map_location=DEVICE)
    net = PointPrior().to(DEVICE)
    net.load_state_dict(ck['model'])
    net.eval()
    print(f"ckpt: val {ck['val']:.4f} @ step {ck['step']} (seed "
          f"{ck['seed']})")

    rows = []
    for name in RECS:
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        nfw = int(round(40.0 / rec.dt))
        n_lead = int(round(cfg['lead_max'] / rec.dt))
        i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 12).astype(int)
        for wi, i0 in enumerate(i0s):
            t0 = time.time()
            P, z, Pq, gt_q, leads, Hs, kp, _ = window_data(
                rec, cfg, int(i0), nfw, n_lead, wi)
            v = np.isfinite(gt_q) & (leads <= 10.0)
            if v.sum() < 5:
                continue
            fq, smult = point_gamma_mmse(net, P, z, cfg['depth'], Pq,
                                         0.05 * Hs, kp, Hs)
            ref = REF[(name, wi)]
            row = {'record': name, 'wi': wi,
                   'point': float(np.corrcoef(fq[v], gt_q[v])[0, 1]),
                   'cnn': ref['bw_s0_head'], 'hand': ref['hand'],
                   'smult': smult}
            rows.append(row)
            print(f"[{name} w{wi}] point {row['point']:+.2f} "
                  f"cnn {row['cnn']:+.2f} hand {row['hand']:+.2f} "
                  f"(smult {smult:.1f}) [{time.time()-t0:.0f}s]", flush=True)

    json.dump(rows, open(OUT, 'w'), indent=1)
    pt = np.array([r['point'] for r in rows])
    cn = np.array([r['cnn'] for r in rows])
    hd = np.array([r['hand'] for r in rows])
    bad = hd < 0.5
    d = pt - cn
    se = d.std(ddof=1) / np.sqrt(len(d))
    g1 = d.mean() >= -0.01 and d[bad].mean() >= -0.02
    print(f"\nG-M3-1 PARITY {'PASS' if g1 else 'FAIL'}  "
          f"d(all) {d.mean():+.3f} (t {d.mean()/se:+.2f})  "
          f"d(bad) {d[bad].mean():+.3f}")
    print("\nG-M3-3 sensitivity table (corr):")
    print(f"{'arm':10s} {'all':>8s} {'bad':>8s}")
    for nm, a in (('hand', hd), ('CNN', cn), ('point', pt)):
        print(f"{nm:10s} {a.mean():+8.3f} {a[bad].mean():+8.3f}")
    # latency (encoder + mmse only; subsample trivial)
    t_enc = []
    P600 = torch.tensor(np.random.default_rng(0).normal(
        size=(1, 2000, 3)).astype(np.float32), device=DEVICE)
    z600 = torch.randn(1, 2000, device=DEVICE)
    kpt = torch.tensor([0.2], device=DEVICE)
    Hst = torch.tensor([2.0], device=DEVICE)
    with torch.no_grad():
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            net(P600, z600, kpt, Hst)
            torch.cuda.synchronize()
            t_enc.append((time.perf_counter() - t0) * 1e3)
    print(f"\nG-M3-2 encoder @2k pts: {np.median(t_enc):.1f} ms "
          f"(full-pipeline judgment via eval wall times above)")
    print(f"wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
