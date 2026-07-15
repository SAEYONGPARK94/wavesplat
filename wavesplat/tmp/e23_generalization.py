"""E-23 (the decisive lever) — does the aperture law GENERALIZE across estimators?

The examiner's single decisive factor: is "learning gain is governed by
under-determination, with a sign-flip on correct physics" a property of the
PROBLEM (information) or of the one net-vs-hand-MMSE pair? E-22 showed it does
NOT hold vs RELAX — but RELAX has NO prior (it beats nothing at short λ). The
fair test needs WELL-SPECIFIED classical baselines and a SECOND learned prior:

  learned priors : net1 = prior_net_l1s2bw (used all session)
                   net2 = b2deep_L_s1      (different objective + seed)
  classical      : hand = mmse_gpu (linear bank + kp + spectrum gamma prior)
                   ridge = same linear dispersion bank + L2 (Tikhonov) prior
Both classical arms are correctly specified on the linear field (correct bank),
so a well-behaved learned prior should NOT beat them where the field is
well-determined. Test, on the E-18 aperture grid, all four learned×classical
pairs: does gain ~ aperture (Spearman<0) with a short-λ sign-flip? If it holds
for 3-4 of 4 pairs, the aperture law is a property of the problem, not of one
baseline; if only net1-vs-hand, it is a baseline artifact.

  python3 tmp/e23_generalization.py
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

OUT = 'results/direction3/e23_generalization.json'
TPS = [6.0, 7.0, 8.0, 9.0, 11.0, 13.0]
STEEP = 0.025
CKPT1 = 'results/direction3/prior_net_l1s2bw.pt'
CKPT2 = 'results/direction3/b2deep_L_s1.pt'


def ridge_at_Q(Ps, zs, Q, kp, depth):
    """Well-specified classical: correct dispersion bank + L2 (Tikhonov) prior."""
    import torch
    from d3_gpu_mmse import build_bank, design
    from d3_g0b_selection_aware import BAND, DEV
    T64 = torch.float64
    kxb, kyb, om = build_bank(kp, BAND, kp / 10.0, depth, DEV, torch.float32)
    kxb, kyb, om = kxb.to(T64), kyb.to(T64), om.to(T64)
    Phi = design(torch.tensor(Ps, device=DEV, dtype=T64), kxb, kyb, om)
    G = Phi.T @ Phi
    lam = 1e-2 * torch.trace(G) / G.shape[0]
    A0 = G + lam * torch.eye(G.shape[0], dtype=T64, device=DEV)
    coef = torch.linalg.solve(A0, Phi.T @ torch.tensor(zs, device=DEV, dtype=T64))
    C = design(torch.tensor(Q, device=DEV, dtype=T64), kxb, kyb, om)
    return (C @ coef).cpu().numpy()


def draw4(sea, rng, net1, head1, net2, head2):
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
        hand = mmse_gpu(torch.tensor(Ps, device=DEVICE, dtype=torch.float32),
                        torch.tensor(zs, device=DEVICE, dtype=torch.float32), DEPTH,
                        torch.tensor(Q, device=DEVICE, dtype=torch.float32),
                        NOISE, kp_hint=kp).cpu().numpy()
    rdg = ridge_at_Q(Ps, zs, Q, kp, DEPTH)
    n1 = np.asarray(net_gamma_mmse(net1, Ps, zs, DEPTH, Q, NOISE, seed=0,
                                   kp_hint=kp, use_em=False, smult_head=head1))
    n2 = np.asarray(net_gamma_mmse(net2, Ps, zs, DEPTH, Q, NOISE, seed=0,
                                   kp_hint=kp, use_em=False, smult_head=head2))

    def c(p):
        return float(np.corrcoef(p, gt)[0, 1]) if np.std(p) > 1e-9 else np.nan
    return c(hand), c(rdg), c(n1), c(n2)


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
    print(f"E-23 generalization: {len(seas)} seas × {len(seeds)} phases ×2 = "
          f"{len(tasks)} cells (hand/ridge/net1/net2)", flush=True)
    t0 = time.time()
    built = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for task, sea, note, dt in ex.map(build_cell, tasks):
            built[task] = sea
    print(f"  built {sum(v is not None for v in built.values())}/{len(tasks)} in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)

    import torch
    from d3_prior_net_eval import get_net, DEVICE
    from d3_l1_s2head_train import SmultHead
    net1 = get_net(CKPT1); head1 = SmultHead().to(DEVICE)
    head1.load_state_dict(torch.load(CKPT1, map_location=DEVICE)['head']); head1.eval()
    net2 = get_net(CKPT2); head2 = SmultHead().to(DEVICE)
    head2.load_state_dict(torch.load(CKPT2, map_location=DEVICE)['head']); head2.eval()

    rows = []
    for (Tp, Hs) in seas:
        lam = _lam(Tp); ar = R_APERTURE / lam
        for sd in seeds:
            sl, s3 = built[(Tp, Hs, sd, 1)], built[(Tp, Hs, sd, 3)]
            if sl is None or s3 is None:
                continue
            L = {k: [] for k in ('hand', 'ridge', 'n1', 'n2')}
            M = {k: [] for k in ('hand', 'ridge', 'n1', 'n2')}
            for rr in range(a.draws):
                rng = np.random.default_rng(820000 + 31 * rr)
                h, r, a1, a2 = draw4(sl, rng, net1, head1, net2, head2)
                L['hand'].append(h); L['ridge'].append(r); L['n1'].append(a1); L['n2'].append(a2)
                rng = np.random.default_rng(820000 + 31 * rr)
                h, r, a1, a2 = draw4(s3, rng, net1, head1, net2, head2)
                M['hand'].append(h); M['ridge'].append(r); M['n1'].append(a1); M['n2'].append(a2)
            m = lambda v: float(np.nanmean(v))
            row = dict(sea=f'Tp{Tp:g}_Hs{Hs:g}', Tp=Tp, lam=lam, aperture_ratio=ar,
                       seed=sd)
            for arm in ('hand', 'ridge', 'n1', 'n2'):
                row[f'{arm}_lin'] = m(L[arm]); row[f'{arm}_m3'] = m(M[arm])
            rows.append(row)
            print(f"[Tp{Tp:g} λ{lam:.0f} ar{ar:.2f} ph{sd}] LIN hand {m(L['hand']):+.2f} "
                  f"ridge {m(L['ridge']):+.2f} n1 {m(L['n1']):+.2f} n2 {m(L['n2']):+.2f}",
                  flush=True)
            json.dump(rows, open(a.out, 'w'), indent=1)

    # ---- aperture law across 4 learned×classical pairs (sea-level) ----
    fr = [r for r in rows if all(np.isfinite(r[f'{k}_lin']) for k in
                                 ('hand', 'ridge', 'n1', 'n2'))]
    seas_u = sorted({r['sea'] for r in fr})
    sm = lambda k, s: np.nanmean([r[k] for r in fr if r['sea'] == s])
    ar = np.array([[r['aperture_ratio'] for r in fr if r['sea'] == s][0] for s in seas_u])
    print(f"\n{'='*74}\nE-23 aperture-law generalization ({len(seas_u)} seas)\n{'='*74}")
    print(f"  {'pair':<14}{'ρ(ap,gain_lin)':>16}{'p':>8}{'short-λ gain':>14}"
          f"{'long-λ gain':>13}{'flip?':>7}")
    summ = {}
    for learned in ('n1', 'n2'):
        for classical in ('hand', 'ridge'):
            g = np.array([sm(f'{learned}_lin', s) - sm(f'{classical}_lin', s)
                          for s in seas_u])
            r_, p_ = stats.spearmanr(ar, g)
            hi = ar >= np.median(ar)
            flip = g[hi].mean() < 0 < g[~hi].mean()
            nm = f'{learned}-{classical}'
            summ[nm] = dict(rho=float(r_), p=float(p_), short=float(g[hi].mean()),
                            long=float(g[~hi].mean()), flip=bool(flip))
            print(f"  {nm:<14}{r_:>+16.3f}{p_:>8.3f}{g[hi].mean():>+14.3f}"
                  f"{g[~hi].mean():>+13.3f}{('YES' if flip else 'no'):>7}")
    n_law = sum(1 for v in summ.values() if v['rho'] < 0 and v['p'] < 0.20)
    print(f"\n  aperture law (ρ<0, p<0.2) holds in {n_law}/4 pairs; "
          f"sign-flip in {sum(v['flip'] for v in summ.values())}/4")
    print(f"  *** GENERALIZES (>=3/4 pairs): {n_law >= 3}")
    json.dump({'rows': rows, 'pairs': summ, 'n_law_pairs': n_law,
               'generalizes': bool(n_law >= 3)}, open(a.out, 'w'), indent=2)
    print(f"\n-> {a.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
