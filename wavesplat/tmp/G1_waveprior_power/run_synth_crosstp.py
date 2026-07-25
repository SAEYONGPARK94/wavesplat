"""Test (a) — held-out SYNTHETIC cross-Tp (crossing-sea) reconstruction.
Isolates learning-gap vs structural-failure: on cross-Tp bimodal scenes (the exact
regime the analytic single-peak beamform prior is structurally wrong AND v3 was NOT
trained for), does the retrained (bimodal-trained) net beat BOTH the analytic prior
AND v3?  Scenes mirror the training bimodal augmentation (two JONSWAP peaks, cross-Tp,
direction offset 40-140 deg) but on HELD-OUT seeds (train used seeds 9000+; here 50000+).
Arms scored at nowcast, /gt_std.  GO signal: retrained < v3 AND retrained < hand.
"""
import argparse
import json
import numpy as np
from scipy import stats as ss

import d3_compare_gifs as cg
from d3_sa_nomp import mmse_bank_estimate
from d3_prior_net_eval import get_net, net_gamma_mmse

G, DEPTH, SCAN_DT, M_PER_SCAN, NOISE = cg.G, cg.DEPTH, cg.SCAN_DT, cg.M_PER_SCAN, cg.NOISE


def bimodal_scene(Hs, Tp, seed, U=3.0, W=20.0, lead=0.0):
    """Cross-Tp crossing-sea scene (held-out): peak1 @ Tp dir 0, peak2 @ cross-Tp dir offset."""
    rng = np.random.default_rng(50000 + seed)
    r2 = rng.uniform(0.2, 0.5)
    tp2 = float(np.clip(Tp * rng.uniform(0.6, 0.85), 4.0, 12.0))  # cross-Tp (shorter)
    doff = rng.uniform(40, 140)
    kp = (2 * np.pi / Tp) ** 2 / G
    c1 = cg.jonswap_components(Hs=Hs * np.sqrt(1 - r2), Tp=Tp, n_components=120, depth=DEPTH,
                              direction_deg=0.0, direction_spread_deg=25.0, seed=50000 + seed)
    c2 = cg.jonswap_components(Hs=Hs * np.sqrt(r2), Tp=tp2, n_components=60, depth=DEPTH,
                              direction_deg=doff, direction_spread_deg=25.0, seed=57000 + seed)
    comps = list(c1) + list(c2)
    kx = np.array([c.kx for c in comps]); ky = np.array([c.ky for c in comps])
    A = np.array([c.A for c in comps]); ph = np.array([c.phi for c in comps])
    om = cg.omega_of(kx, ky, DEPTH)

    def eta_at(x, y, t):
        ps = (kx[None] * np.atleast_1d(x)[:, None] + ky[None] * np.atleast_1d(y)[:, None]
              - om[None] * np.atleast_1d(t)[:, None] + ph[None])
        return (A[None] * np.cos(ps)).sum(1)

    t0, cx, cy = W, 0.0, 0.0
    def ship(t): return np.array([cx - U * (t - t0), cy])
    P_all, z_all = [], []
    for t_s in np.arange(t0 - W, t0 + 1e-9, SCAN_DT):
        sp = ship(t_s)
        rr = rng.uniform(20.0, 100.0, M_PER_SCAN); aa = rng.uniform(-np.pi / 3, np.pi / 3, M_PER_SCAN)
        xs = sp[0] - rr * np.cos(aa); ys = sp[1] + rr * np.sin(aa)
        P_all.append(np.column_stack([xs, ys, np.full(M_PER_SCAN, t_s)]))
        z_all.append(eta_at(xs, ys, t_s) + rng.normal(0, NOISE, M_PER_SCAN))
    P, z = np.vstack(P_all), np.concatenate(z_all)
    gxq = np.arange(ship(t0)[0] - 30, ship(t0)[0] + 110, 4.0)
    gyq = np.arange(cy - 90, cy + 90, 4.0)
    Xq, Yq = np.meshgrid(gxq, gyq); npx = Xq.size
    Pq = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(npx, t0 + lead)])
    gt = eta_at(Xq.ravel(), Yq.ravel(), t0 + lead).reshape(Xq.shape)
    Ctrue = np.array([[c.kx, c.ky, c.A] for c in comps])
    return dict(P=P, z=z, Pq=Pq, Xq=Xq, depth=DEPTH, Hs=Hs, kp=kp, seed=seed, gt=gt, Ctrue=Ctrue)


def nrmse_now(fq, sc):
    m = fq.reshape(sc['Xq'].shape)
    gt = sc['gt']; sg = float(np.std(gt))
    return float(np.sqrt(np.mean((m - gt) ** 2)) / sg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=12)
    ap.add_argument('--Hs', type=float, default=2.0); ap.add_argument('--Tp', type=float, default=8.0)
    ap.add_argument('--retrained', default='results/direction3/prior_net_multimodal.pt')
    a = ap.parse_args()
    net_v3 = get_net('results/direction3/prior_net_v3.pt')
    net_rt = get_net(a.retrained)

    rows = []
    for r in range(a.reps):
        sc = bimodal_scene(a.Hs, a.Tp, r)
        s0 = 0.05 * sc['Hs']; P, z, Pq = sc['P'], sc['z'], sc['Pq']
        h = nrmse_now(mmse_bank_estimate(P, z, sc['depth'], Pq, sigma0=s0, kp_hint=sc['kp'], seed=r), sc)
        v = nrmse_now(net_gamma_mmse(net_v3, P, z, sc['depth'], Pq, sigma0=s0, seed=r, kp_hint=sc['kp']), sc)
        t = nrmse_now(net_gamma_mmse(net_rt, P, z, sc['depth'], Pq, sigma0=s0, seed=r, kp_hint=sc['kp']), sc)
        o = nrmse_now(net_gamma_mmse(net_v3, P, z, sc['depth'], Pq, sigma0=s0, seed=r,
                                     kp_hint=sc['kp'], oracle_components=sc['Ctrue']), sc)
        rows.append((h, v, t, o))
        print(f"[{r:2d}] hand {h:.3f}  v3 {v:.3f}  retrained-TF {t:.3f}  oracle {o:.3f}", flush=True)

    H = np.array([x[0] for x in rows]); V = np.array([x[1] for x in rows])
    T = np.array([x[2] for x in rows]); O = np.array([x[3] for x in rows])
    out = {'hand': float(H.mean()), 'v3': float(V.mean()), 'retrained': float(T.mean()), 'oracle': float(O.mean()),
           'rt_vs_hand': float((H - T).mean()), 'rt_vs_oracle_gap': float((T - O).mean()),
           'rt_beats_hand': int((T < H).sum()), 'n': len(rows)}
    json.dump(out, open('tmp/G1_waveprior_power/synth_crosstp_tf.json', 'w'), indent=1)
    print("\n=== SYNTHETIC CROSS-Tp (nowcast /gt_std) ===")
    print(f"  hand {H.mean():.3f}  v3 {V.mean():.3f}  retrained-TF {T.mean():.3f}  ORACLE(ceiling) {O.mean():.3f}")
    print(f"  retrained-TF vs hand: {(H-T).mean():+.3f} (TF better on {int((T<H).sum())}/{len(rows)}, "
          f"wilcoxon p={ss.wilcoxon(H-T).pvalue:.4f})")
    frac = (H.mean()-T.mean())/(H.mean()-O.mean()) if H.mean()>O.mean() else float('nan')
    print(f"  gap closed toward oracle: {frac*100:.0f}%  (hand {H.mean():.3f} → oracle {O.mean():.3f}; TF at {T.mean():.3f})")
    print("  ⟹ " + ("retrained-TF significantly beats hand on cross-Tp = task-finetune captured the structure ⟹ REVIVE"
                     if (H-T).mean() > 0.02 and ss.wilcoxon(H-T).pvalue < 0.05 else
                     "retrained-TF does NOT beat hand ⟹ even task-finetune fails to capture the 2nd lobe"))


if __name__ == '__main__':
    main()
