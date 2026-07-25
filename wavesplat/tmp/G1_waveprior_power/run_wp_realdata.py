"""WavePrior (the thesis topic) — REAL-DATA multi-record head-to-head, the one untested
place WP might genuinely win. The analytic prior (beamform power) assumes a single spectral
peak; real bimodal/storm seas violate that structurally, so a learned prior could win THERE.

net (WavePrior) vs analytic-prior mmse_bank ("WaveSplat" arm of d3_compare_gifs) on ALL
Bergamasco stereo records × all windows, paired stats. Direction3 only ever tested YS17 (6
windows, 4/6 n.s.). Metric = map nRMSE nowcast AND lead-averaged, /gt_std (direction3 self-
contained). Positive paired delta = WavePrior better.
"""
import argparse
import json
import numpy as np
from scipy import stats as ss

import d3_compare_gifs as cg
from d3_falcon_real_operator import RECORDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', default=','.join(RECORDS.keys()))
    ap.add_argument('--wins', type=int, default=6)
    ap.add_argument('--ckpt', default=cg.NET_CKPT,
                    help='WavePrior checkpoint (default v3; pass retrained for held-out test)')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    cg.NET_CKPT = a.ckpt  # cg.solve() reads this at call time
    print(f"# WavePrior ckpt: {cg.NET_CKPT}", flush=True)
    recs = a.records.split(',')

    out = {}
    all_now, all_lead = [], []
    for rec in recs:
        d_now, d_lead, ws_now, wp_now = [], [], [], []
        for w in range(a.wins):
            try:
                sc = cg.scene_real(rec, win=w)
                sc['n_obs'] = len(sc['z'])
                m = cg.metrics(sc, cg.solve(sc))
                a_now = m['WaveSplat']['map_nrmse_nowcast']; b_now = m['WavePrior']['map_nrmse_nowcast']
                a_ld = m['WaveSplat']['map_nrmse_mean_over_leads']; b_ld = m['WavePrior']['map_nrmse_mean_over_leads']
                if None in (a_now, b_now):
                    continue
                d_now.append(a_now - b_now); ws_now.append(a_now); wp_now.append(b_now)
                if a_ld is not None and b_ld is not None:
                    d_lead.append(a_ld - b_ld)
                print(f"  [{rec} w{w}] WS {a_now:.3f} / WP {b_now:.3f} | d_now {a_now-b_now:+.3f}", flush=True)
            except Exception as e:
                print(f"  [{rec} w{w}] FAILED {type(e).__name__}: {e}", flush=True)
        if not d_now:
            continue
        dn = np.array(d_now)
        note = RECORDS[rec].get('note', '')
        out[rec] = {'n': len(dn), 'd_now_mean': float(dn.mean()), 'd_now_std': float(dn.std(ddof=1) if len(dn) > 1 else 0),
                    'wp_wins': int((dn > 0).sum()), 'WS_now': float(np.mean(ws_now)), 'WP_now': float(np.mean(wp_now)),
                    'd_lead_mean': float(np.mean(d_lead)) if d_lead else None, 'note': note[:40]}
        all_now.extend(d_now); all_lead.extend(d_lead)
        print(f"  ==> {rec}: d_now {dn.mean():+.3f} (WP wins {int((dn>0).sum())}/{len(dn)})  {note[:40]}\n", flush=True)

    an = np.array(all_now)
    tn, pn = ss.ttest_1samp(an, 0) if len(an) > 1 else (0, 1)
    pooled = {'n': len(an), 'd_now_mean': float(an.mean()), 'd_now_t_p': float(pn),
              'wp_wins': int((an > 0).sum()),
              'd_lead_mean': float(np.mean(all_lead)) if all_lead else None}
    suf = ('_' + a.tag) if a.tag else ''
    json.dump({'per_record': out, 'pooled': pooled}, open(f'tmp/G1_waveprior_power/wp_realdata{suf}.json', 'w'), indent=1)

    print("\n=== WAVEPRIOR REAL-DATA (nowcast map nRMSE/gt_std; + = WP better) ===")
    print(f"{'record':7s} {'n':>2s} {'WS':>6s} {'WP':>6s} {'d_now':>7s} {'wins':>5s}  note")
    for rec in recs:
        if rec in out:
            o = out[rec]
            print(f"{rec:7s} {o['n']:2d} {o['WS_now']:6.3f} {o['WP_now']:6.3f} {o['d_now_mean']:+7.3f} "
                  f"{o['wp_wins']:2d}/{o['n']}  {o['note']}")
    print(f"\nPOOLED (n={pooled['n']}): d_now {pooled['d_now_mean']:+.3f} (t p={pooled['d_now_t_p']:.3f}), "
          f"WP wins {pooled['wp_wins']}/{pooled['n']}, d_lead {pooled['d_lead_mean']}")


if __name__ == '__main__':
    main()
