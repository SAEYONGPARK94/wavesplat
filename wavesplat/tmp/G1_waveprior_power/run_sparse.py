"""G2 (part 2) — sparse-component transfer of prior_net_v3.

Fable's MEDIUM concern #2: prior_net_v3 was trained on dense nc=200 JONSWAP
(near-continuous spectrum), but the OE paper's episodes are sparse discrete
fields (5/15/30 components). Does the net still help, or collapse, on sparse
spectra?

Held at BF4 (Hs=1.2, Tp=7.0, IN the training Hs/Tp range) so the ONLY variable
is component count — isolates sparse-transfer from the Hs/Tp-scale effect the
OOD sweep already characterised. Same reused d3_compare_gifs solver/metrics.
Headline = lead-averaged map nRMSE; positive paired delta = WavePrior better.
"""
import argparse
import json
import time

import numpy as np
import d3_compare_gifs as cg  # side effects: chdir, sys.path
from run_ood import linear_scene

HS, TP = 1.2, 7.0  # BF4, in-distribution — isolate component count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=8)
    ap.add_argument('--ncs', default='200,30,15,5')
    ap.add_argument('--ckpt', default=cg.NET_CKPT,
                    help='prior-net checkpoint the WavePrior arm uses')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    ncs = [int(x) for x in a.ncs.split(',')]
    cg.NET_CKPT = a.ckpt  # cg.solve() reads this module global at call time
    print(f"# WavePrior checkpoint: {cg.NET_CKPT}", flush=True)

    allm = {}
    for nc in ncs:
        ws_l, wp_l, ws_n, wp_n, dl = [], [], [], [], []
        t0 = time.time()
        for r in range(a.reps):
            sc = linear_scene(HS, TP, seed=r, nc=nc)
            sc['n_obs'] = len(sc['z'])
            met = cg.metrics(sc, cg.solve(sc))
            ws_l.append(met['WaveSplat']['map_nrmse_mean_over_leads'])
            wp_l.append(met['WavePrior']['map_nrmse_mean_over_leads'])
            ws_n.append(met['WaveSplat']['map_nrmse_nowcast'])
            wp_n.append(met['WavePrior']['map_nrmse_nowcast'])
            dl.append(ws_l[-1] - wp_l[-1])
        dl = np.array(dl)
        allm[str(nc)] = {
            'nc': nc, 'reps': a.reps,
            'WS_lead_mean': float(np.mean(ws_l)), 'WP_lead_mean': float(np.mean(wp_l)),
            'WS_now_mean': float(np.mean(ws_n)), 'WP_now_mean': float(np.mean(wp_n)),
            'paired_d_lead_mean': float(np.mean(dl)),
            'paired_d_lead_std': float(np.std(dl, ddof=1)),
            'wp_wins_lead': int(np.sum(dl > 0)),
            'per_rep_d_lead': [float(x) for x in dl],
        }
        print(f"[nc={nc:3d}] WS_lead {np.mean(ws_l):.3f} WP_lead {np.mean(wp_l):.3f}  "
              f"dLead {np.mean(dl):+.4f} (WP wins {int(np.sum(dl>0))}/{a.reps})  "
              f"{time.time()-t0:.0f}s", flush=True)

    suffix = ('_' + a.tag) if a.tag else ''
    outp = f'tmp/G1_waveprior_power/sparse{suffix}.json'
    with open(outp, 'w') as f:
        json.dump(allm, f, indent=1)
    print(f"\n-> {outp}")
    print("\n=== SPARSE-TRANSFER VERDICT (BF4, lead-avg nRMSE; + favours WP) ===")
    print(f"{'nc':>4s} {'WS':>6s} {'WP':>6s} {'dLead':>8s} {'WPwin':>6s}")
    for nc in ncs:
        r = allm[str(nc)]
        print(f"{nc:4d} {r['WS_lead_mean']:6.3f} {r['WP_lead_mean']:6.3f} "
              f"{r['paired_d_lead_mean']:+8.4f} {r['wp_wins_lead']:3d}/{r['reps']}")


if __name__ == '__main__':
    main()
