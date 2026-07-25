"""G2 (part 1) — OOD characterization of prior_net_v3 across BF2-BF7.

prior_net_v3 was trained on Hs~U[1,3] m, Tp~U[5,11] s (linear JONSWAP). The OE
paper spans BF2-BF7. This sweeps sea states and asks, per state:

  - does the learned prior (WavePrior) still BEAT/TIE the analytic prior
    (WaveSplat), or does it COLLAPSE out-of-distribution?
  - absolute WP nRMSE: if it blows toward 1.0 (zero-predictor) at Hs=0.3 or
    Hs=5, the net is unusable there and retraining is mandatory before the
    paper can cite WP across the Beaufort range.

Faithfully reuses d3_compare_gifs's LINEAR scene construction (lines 60-141),
parametrised by (Hs, Tp); solve()/metrics() are the repo's own. Ship speed U
held constant to isolate the prior's Hs/Tp generalisation (not encounter dynamics).
Headline metric = lead-averaged map nRMSE (the forecast metric G1 found decisive).
Positive paired delta = WavePrior better.
"""
import argparse
import json
import time

import numpy as np
import d3_compare_gifs as cg  # side effects: chdir to repo root, fix sys.path

# training range of prior_net_v3 (docs/COMPARE §, d3_prior_net_data.py)
TRAIN_HS = (1.0, 3.0)
TRAIN_TP = (5.0, 11.0)

# BF2-BF5 from configs/default.yaml; BF6/BF7 from labmeeting extension (per Fable)
SEA_STATES = {
    'BF2': (0.3, 4.0),
    'BF3': (0.6, 5.5),
    'BF4': (1.2, 7.0),
    'BF5': (2.0, 9.0),
    'BF6': (3.0, 10.0),  # corrected: real dataset BF6 = Tp 10.0 (was 11.0 bug)
    'BF7': (5.0, 12.0),
}


def linear_scene(Hs, Tp, seed, U=3.0, W=20.0, lead_max=20.0, lead_step=1.0, nc=200):
    """Exact copy of d3_compare_gifs.scene_sim's `linear` branch, parametrised.

    `nc` = number of JONSWAP components. Training used nc=200 (dense); the paper's
    episodes are sparse discrete fields (5/15/30), so nc<200 probes Fable's
    sparse-transfer concern (#2).
    """
    G, DEPTH, SCAN_DT = cg.G, cg.DEPTH, cg.SCAN_DT
    M_PER_SCAN, NOISE = cg.M_PER_SCAN, cg.NOISE
    leads = np.arange(0.0, lead_max + 1e-9, lead_step)
    kp = (2 * np.pi / Tp) ** 2 / G
    rng = np.random.default_rng(400 + seed)
    comps = cg.jonswap_components(Hs=Hs, Tp=Tp, n_components=nc, depth=DEPTH,
                                  direction_spread_deg=25.0, seed=9000 + seed)
    kx = np.array([c.kx for c in comps]); ky = np.array([c.ky for c in comps])
    A = np.array([c.A for c in comps]); ph = np.array([c.phi for c in comps])
    om = cg.omega_of(kx, ky, DEPTH)

    def eta_at(x, y, t):
        ps = (kx[None] * np.atleast_1d(x)[:, None]
              + ky[None] * np.atleast_1d(y)[:, None]
              - om[None] * np.atleast_1d(t)[:, None] + ph[None])
        return (A[None] * np.cos(ps)).sum(1)

    t0, cx, cy = W, 0.0, 0.0

    def ship_pos(t):
        return np.array([cx - U * (t - t0), cy])

    scan_ts = np.arange(t0 - W, t0 + 1e-9, SCAN_DT)
    P_all, z_all = [], []
    for t_s in scan_ts:
        sp = ship_pos(t_s)
        rr = rng.uniform(20.0, 100.0, M_PER_SCAN)
        aa = rng.uniform(-np.pi / 3, np.pi / 3, M_PER_SCAN)
        xs = sp[0] - rr * np.cos(aa)
        ys = sp[1] + rr * np.sin(aa)
        zz = eta_at(xs, ys, t_s)
        P_all.append(np.column_stack([xs, ys, np.full(M_PER_SCAN, t_s)]))
        z_all.append(zz + rng.normal(0, NOISE, M_PER_SCAN))
    P, z = np.vstack(P_all), np.concatenate(z_all)

    x_lo = ship_pos(t0 + lead_max)[0] - 30.0
    x_hi = ship_pos(t0)[0] + 110.0
    gxq = np.arange(x_lo, x_hi, 4.0)
    gyq = np.arange(cy - 90.0, cy + 90.0, 4.0)
    Xq, Yq = np.meshgrid(gxq, gyq)
    npx = Xq.size
    Pq = [np.column_stack([Xq.ravel(), Yq.ravel(), np.full(npx, t0 + l_)])
          for l_ in leads]
    Pq_enc = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in leads])
    Pq = np.vstack(Pq + [Pq_enc])

    maps_gt, enc_gt = [], []
    for l_ in leads:
        t_ = t0 + l_
        maps_gt.append(eta_at(Xq.ravel(), Yq.ravel(), t_).reshape(Xq.shape))
        sp = ship_pos(t_)
        enc_gt.append(float(eta_at(sp[:1], sp[1:2], t_)[0]))
    return dict(P=P, z=z, Pq=Pq, npx=npx, leads=leads, Xq=Xq, gxq=gxq, gyq=gyq,
                depth=DEPTH, Hs=Hs, kp=kp, seed=seed,
                maps_gt=np.array(maps_gt), enc_gt=np.array(enc_gt),
                label=f'Linear Hs{Hs} Tp{Tp}')


def in_dist(Hs, Tp):
    return (TRAIN_HS[0] <= Hs <= TRAIN_HS[1]) and (TRAIN_TP[0] <= Tp <= TRAIN_TP[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=8)
    ap.add_argument('--states', default='BF2,BF3,BF4,BF5,BF6,BF7')
    a = ap.parse_args()
    states = a.states.split(',')

    allm = {}
    for st in states:
        Hs, Tp = SEA_STATES[st]
        ws_lead, wp_lead, ws_now, wp_now = [], [], [], []
        d_lead = []
        t0 = time.time()
        for r in range(a.reps):
            sc = linear_scene(Hs, Tp, seed=r)
            sc['n_obs'] = len(sc['z'])
            met = cg.metrics(sc, cg.solve(sc))
            ws_lead.append(met['WaveSplat']['map_nrmse_mean_over_leads'])
            wp_lead.append(met['WavePrior']['map_nrmse_mean_over_leads'])
            ws_now.append(met['WaveSplat']['map_nrmse_nowcast'])
            wp_now.append(met['WavePrior']['map_nrmse_nowcast'])
            d_lead.append(ws_lead[-1] - wp_lead[-1])
        dl = np.array(d_lead)
        allm[st] = {
            'Hs': Hs, 'Tp': Tp, 'in_distribution': in_dist(Hs, Tp), 'reps': a.reps,
            'WS_lead_mean': float(np.mean(ws_lead)), 'WP_lead_mean': float(np.mean(wp_lead)),
            'WS_now_mean': float(np.mean(ws_now)), 'WP_now_mean': float(np.mean(wp_now)),
            'paired_d_lead_mean': float(np.mean(dl)), 'paired_d_lead_std': float(np.std(dl, ddof=1)),
            'wp_wins_lead': int(np.sum(dl > 0)),
            'per_rep_d_lead': [float(x) for x in dl],
        }
        tag = 'IN ' if in_dist(Hs, Tp) else 'OOD'
        print(f"[{st} {tag} Hs{Hs} Tp{Tp}] WS_lead {np.mean(ws_lead):.3f} "
              f"WP_lead {np.mean(wp_lead):.3f}  dLead {np.mean(dl):+.4f} "
              f"(WP wins {int(np.sum(dl>0))}/{a.reps})  {time.time()-t0:.0f}s", flush=True)

    with open('tmp/G1_waveprior_power/ood.json', 'w') as f:
        json.dump(allm, f, indent=1)
    print("\n-> tmp/G1_waveprior_power/ood.json")

    print("\n=== OOD VERDICT (lead-avg nRMSE, lower=better; + delta favours WP) ===")
    print(f"{'state':6s} {'dist':4s} {'WS':>6s} {'WP':>6s} {'dLead':>8s} {'WPwin':>6s}")
    for st in states:
        r = allm[st]
        print(f"{st:6s} {'IN' if r['in_distribution'] else 'OOD':4s} "
              f"{r['WS_lead_mean']:6.3f} {r['WP_lead_mean']:6.3f} "
              f"{r['paired_d_lead_mean']:+8.4f} {r['wp_wins_lead']:3d}/{r['reps']}")


if __name__ == '__main__':
    main()
