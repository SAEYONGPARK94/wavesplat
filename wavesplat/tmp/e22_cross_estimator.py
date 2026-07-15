"""E-22 (closing #B) — cross-estimator generalization the examiner demanded.

Every number in E-16..E-21 is Δ = net − hand-MMSE for ONE learned checkpoint vs
ONE classical baseline. The load-bearing claim is "NOT the algorithm — it's the
aperture." A referee attacks exactly there: you showed THIS net beats THIS hand
more when under-determined, not that learned priors as a class do. So swap the
classical baseline to RELAX (a genuinely different algorithm: greedy physics
component search, not a spectral-bank MMSE) and ask: does the learned gain over
RELAX also track aperture, and does the short-λ sign-flip hold?

Grid: the E-18 aperture axis (Tp 6-13, λ 56-231 m) at a fixed steepness 0.025,
× 2 phases. Three arms at the arena query points: hand-MMSE (mmse_gpu), net
(net_gamma_mmse), RELAX (relax_cv components evaluated via _design at Q — it
forecasts through the dispersion in psi). Report on the NONLINEAR (M3) field:
  gain_net_hand = corr(net) − corr(hand)
  gain_net_relax = corr(net) − corr(relax)     <- the cross-estimator gain
and on the LINEAR field the sign of gain_net_relax (the flip). PASS = the
learned gain over RELAX tracks aperture (Spearman < 0 on Δlin, same sign as the
net−hand result), i.e. the aperture law is not specific to the hand-MMSE arm.

  python3 tmp/e22_cross_estimator.py
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

from e18_decorrelated_2factor import build_cell, _lam, R_APERTURE  # noqa: E402

OUT = 'results/direction3/e22_cross_estimator.json'
TPS = [6.0, 7.0, 8.0, 9.0, 11.0, 13.0]
STEEP = 0.025


def relax_at_Q(P, z, Q, depth, seed):
    """RELAX reconstruction at arbitrary query points Q (forecasts via psi)."""
    from estimator_improved import estimate_klim, relax_cv, _design
    from d18_relax_estimator import ls_fit, beamform_top
    klim = estimate_klim(P, z, depth)
    kx, ky = relax_cv(P, z, depth, klim, seed=seed)
    if len(kx) == 0:
        bx, by = beamform_top(P, z, depth, klim=klim)
        kx, ky = np.array([bx]), np.array([by])
    coef, _, _ = ls_fit(np.asarray(kx), np.asarray(ky), P, z, depth)
    return _design(Q, np.asarray(kx), np.asarray(ky), depth) @ coef


def draw3(sea, rng, net, head):
    """one draw, three arms (hand, net, RELAX) at the arena queries."""
    import torch
    from e15_hos_multidraw import N_OBS, LEADS
    from d3_h7_hos_arena import z_at
    from d3_door1b_field_bias import R_MIN, R_MAX, FOV_H
    from wavesim.raydrop_threshold import T_MANUAL_50
    from d3_gpu_mmse import mmse_gpu
    from d3_prior_net_eval import net_gamma_mmse, DEVICE
    NOISE, DEPTH = 0.05, 50.0
    P, zv, rr = [], [], []
    for sc in sea['scans']:
        pw = sc['amp'] * rng.exponential(1.0, len(sc['amp']))
        keep = pw > T_MANUAL_50
        P.append(np.column_stack([sc['xy'][keep], np.full(int(keep.sum()), sc['trel'])]))
        zv.append(sc['z'][keep]); rr.append(sc['r'][keep])
    P, zv, rr = np.vstack(P), np.concatenate(zv), np.concatenate(rr)
    zv = zv + rng.normal(0, NOISE, len(zv))
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
    try:
        pr = relax_at_Q(Ps, zs, Q, DEPTH, seed=0)
    except Exception:                                       # noqa: BLE001
        pr = np.full(len(gt), np.nan)

    def c(p):
        return float(np.corrcoef(p, gt)[0, 1]) if np.std(p) > 1e-9 else np.nan
    return c(ph), c(pn), c(pr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--draws', type=int, default=3)
    ap.add_argument('--workers', type=int, default=11)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    tps = TPS[:2] if a.smoke else TPS
    seeds = a.seeds[:1] if a.smoke else a.seeds
    seas = [(Tp, round(STEEP * _lam(Tp), 3)) for Tp in tps]
    tasks = [(Tp, Hs, sd, order) for (Tp, Hs) in seas
             for sd in seeds for order in (1, 3)]
    print(f"E-22 cross-estimator: {len(seas)} seas × {len(seeds)} phases ×2 = "
          f"{len(tasks)} cells (hand/net/RELAX)", flush=True)
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
    net = get_net(CKPT); head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head']); head.eval()

    rows = []
    for (Tp, Hs) in seas:
        lam = _lam(Tp); ar = R_APERTURE / lam
        for sd in seeds:
            sl, s3 = built[(Tp, Hs, sd, 1)], built[(Tp, Hs, sd, 3)]
            if sl is None or s3 is None:
                continue
            H1, N1, R1, H3, N3, R3 = ([] for _ in range(6))
            for rr in range(a.draws):
                rng = np.random.default_rng(810000 + 31 * rr)
                h, n_, r = draw3(sl, rng, net, head); H1.append(h); N1.append(n_); R1.append(r)
                rng = np.random.default_rng(810000 + 31 * rr)
                h, n_, r = draw3(s3, rng, net, head); H3.append(h); N3.append(n_); R3.append(r)
            m = lambda v: float(np.nanmean(v))
            rows.append(dict(sea=f'Tp{Tp:g}_Hs{Hs:g}', Tp=Tp, lam=lam,
                             aperture_ratio=ar, seed=sd,
                             hand_lin=m(H1), net_lin=m(N1), relax_lin=m(R1),
                             hand_m3=m(H3), net_m3=m(N3), relax_m3=m(R3),
                             gnh_lin=m(N1) - m(H1), gnr_lin=m(N1) - m(R1),
                             gnh_m3=m(N3) - m(H3), gnr_m3=m(N3) - m(R3),
                             ghr_m3=m(H3) - m(R3)))
            print(f"[Tp{Tp:g} λ{lam:.0f} ar{ar:.2f} ph{sd}] M3 corr hand {m(H3):+.2f} "
                  f"net {m(N3):+.2f} relax {m(R3):+.2f} | net-hand {m(N3)-m(H3):+.3f} "
                  f"net-relax {m(N3)-m(R3):+.3f} | LIN net-relax {m(N1)-m(R1):+.3f}",
                  flush=True)
            json.dump(rows, open(a.out, 'w'), indent=1)

    fr = [r for r in rows if np.isfinite(r['gnr_lin']) and np.isfinite(r['gnr_m3'])]
    seas_u = sorted({r['sea'] for r in fr})
    sm = lambda k, s: np.nanmean([r[k] for r in fr if r['sea'] == s])
    ar = np.array([[r['aperture_ratio'] for r in fr if r['sea'] == s][0] for s in seas_u])
    gnr_lin = np.array([sm('gnr_lin', s) for s in seas_u])
    gnh_lin = np.array([sm('gnh_lin', s) for s in seas_u])
    gnr_m3 = np.array([sm('gnr_m3', s) for s in seas_u])
    print(f"\n{'='*70}\nE-22 cross-estimator ({len(seas_u)} seas, sea-level)\n{'='*70}")
    r_nr, p_nr = stats.spearmanr(ar, gnr_lin)
    r_nh, p_nh = stats.spearmanr(ar, gnh_lin)
    print(f"  aperture law with RELAX baseline:")
    print(f"    Spearman(aperture, net−RELAX on LINEAR) {r_nr:+.3f} p {p_nr:.4f}")
    print(f"    Spearman(aperture, net−hand  on LINEAR) {r_nh:+.3f} p {p_nh:.4f} (ref)")
    print(f"  net−RELAX linear sign: short-λ (hi ap) {gnr_lin[ar>=np.median(ar)].mean():+.3f} "
          f"vs long-λ {gnr_lin[ar<np.median(ar)].mean():+.3f}")
    print(f"  net−RELAX on M3: {gnr_m3.mean():+.3f} (learned still beats RELAX on nonlinear)")
    g = bool(r_nr < 0 and p_nr < 0.10)
    print(f"  *** aperture law generalizes to RELAX baseline: {g}")
    json.dump({'rows': rows, 'summary': dict(
        n_seas=len(seas_u),
        spearman_ap_net_relax_lin=[float(r_nr), float(p_nr)],
        spearman_ap_net_hand_lin=[float(r_nh), float(p_nh)],
        gnr_m3_mean=float(gnr_m3.mean()), generalizes=g)},
        open(a.out, 'w'), indent=2)
    print(f"\n-> {a.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
