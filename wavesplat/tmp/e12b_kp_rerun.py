"""E-12b — does the real-data learning gain survive a corrected kp hint?

Pre-registered in docs/E12_KP_FLOOR_AUDIT_prereg_2026-07-14.md.

Frozen headline: d3_field_wide_eval.py, 5 records x 12 windows, whole-field corr,
  delta = net - hand = +0.0334, cluster-robust CI [+0.016, +0.051], t 3.70.
Both arms receive the SAME kp_hint, taken from the spatial spectrum of one frame.
The net was trained on a k/kp-normalised bank, so a wrong hint puts it OOD --
the measured gain could be "the net is robust to a bad hint" rather than "the net
is a better wave prior".

Design: single-variable ablation. The observation set (P, z), the windows, the
seeds, the fan geometry and the ship track are ALL held at the frozen protocol.
The ONLY thing that changes is the value of kp_hint handed to the two estimators:

    spatial : kp from directional_spread(Zmem[-1])     [frozen protocol]
    causal  : kp from the buffer's time axis           [E-12a, PASSED]

Both arms are re-run in the same loop, so every comparison is paired per window.
The spatial arm doubles as a reproduction check against the frozen JSON.

GATES (fixed before running):
  PASS      delta_causal >= +0.020 AND cluster-robust 95% CI excludes 0
  DEMOTED   significant but point estimate < +0.017 (less than half)
  FAIL      CI includes 0  -> withdraw the real-data learning-gain claim
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp')

from d3_falcon_real_operator import RECORDS      # noqa: E402
from d3_sa_multirecord import MRecord            # noqa: E402
from d3_prior_net_eval import get_net, DEVICE    # noqa: E402
from d3_l1_s2head_train import SmultHead         # noqa: E402
from d3_external_figs import Window              # noqa: E402
from d3_cluster_se import cluster_se             # noqa: E402
from kp_causal_lib import kp_causal             # noqa: E402

OUT = 'results/direction3/e12b_kp_rerun.json'
RECS = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']
LEADS = [0.0, 2.0, 5.0, 8.0]
CKPT = 'results/direction3/prior_net_l1s2bw.pt'


def score(w, leads):
    gx, gy, tr, fh, fn = w.fields(leads, stride=2)
    trs, fhs, fns = np.array(tr), np.array(fh), np.array(fn)
    m = np.isfinite(trs)
    if m.sum() < 500:
        return None
    return (float(np.corrcoef(fhs[m], trs[m])[0, 1]),
            float(np.corrcoef(fns[m], trs[m])[0, 1]), int(m.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_win', type=int, default=12)
    ap.add_argument('--records', nargs='+', default=RECS)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    net = get_net(CKPT)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head'])
    head.eval()

    rows = []
    for name in a.records:
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        nfw = int(round(40.0 / rec.dt))
        for wi in range(a.n_win):
            t0 = time.time()
            w = Window(name, wi, rec=rec, net=net, head=head)
            leads = [l_ for l_ in LEADS if l_ <= min(10.0, w.leads.max())]

            kp_sp = float(w.kp)                       # frozen protocol value
            Zmem = rec.window(w.i0 - nfw + 1, nfw)
            kp_cs, _ = kp_causal(Zmem, rec.dt, cfg['depth'])
            if not np.isfinite(kp_cs):
                print(f'[{name} w{wi}] causal kp failed — skipped')
                continue

            w.kp = kp_sp                              # arm 1: frozen
            s_sp = score(w, leads)
            w.kp = float(kp_cs)                       # arm 2: corrected hint
            s_cs = score(w, leads)
            if s_sp is None or s_cs is None:
                print(f'[{name} w{wi}] too few finite px — skipped')
                continue

            row = dict(record=name, wi=wi, n_px=s_sp[2],
                       kp_spatial=kp_sp, kp_causal=float(kp_cs),
                       hand_spatial=s_sp[0], learned_spatial=s_sp[1],
                       hand_causal=s_cs[0], learned_causal=s_cs[1])
            row['d_spatial'] = row['learned_spatial'] - row['hand_spatial']
            row['d_causal'] = row['learned_causal'] - row['hand_causal']
            rows.append(row)
            print(f"[{name} w{wi}] kp {kp_sp:.3f}->{kp_cs:.3f} | "
                  f"SPATIAL h {s_sp[0]:+.3f} n {s_sp[1]:+.3f} "
                  f"(d {row['d_spatial']:+.3f}) | "
                  f"CAUSAL h {s_cs[0]:+.3f} n {s_cs[1]:+.3f} "
                  f"(d {row['d_causal']:+.3f}) [{time.time()-t0:.0f}s]",
                  flush=True)
        rec.f.close()

    json.dump(rows, open(a.out, 'w'), indent=1)

    g = [r['record'] for r in rows]
    print(f"\n{'='*74}\nE-12b  (n={len(rows)} windows, {len(set(g))} record clusters)\n{'='*74}")
    print(f"{'arm':<10}{'hand':>9}{'net':>9}{'delta':>9}{'CR-SE':>9}"
          f"{'t_cl':>7}{'95% CI':>20}{'net wins':>10}")
    res = {}
    for mode in ('spatial', 'causal'):
        h = np.array([r[f'hand_{mode}'] for r in rows])
        n_ = np.array([r[f'learned_{mode}'] for r in rows])
        d = n_ - h
        se, G = cluster_se(d, g)
        lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
        res[mode] = dict(hand=float(h.mean()), net=float(n_.mean()),
                         delta=float(d.mean()), se_cluster=float(se),
                         n_clusters=G, ci=[float(lo), float(hi)],
                         t_cluster=float(d.mean() / se),
                         wins=int((d > 0).sum()), n=len(d))
        print(f"{mode:<10}{h.mean():+9.3f}{n_.mean():+9.3f}{d.mean():+9.4f}"
              f"{se:9.4f}{d.mean()/se:7.2f}"
              f"   [{lo:+.4f}, {hi:+.4f}]{f'{(d>0).sum()}/{len(d)}':>10}")

    # frozen reference
    try:
        fw = json.load(open('results/direction3/field_wide_eval.json'))
        d0 = np.array([r['field_learned'] - r['field_hand'] for r in fw])
        print(f"\nreproduction check: frozen field_wide_eval delta "
              f"{d0.mean():+.4f} (n={len(d0)})  vs  this run's SPATIAL arm "
              f"{res['spatial']['delta']:+.4f}")
    except FileNotFoundError:
        pass

    dc = res['causal']
    verdict = ('FAIL — CI includes 0; withdraw the real-data learning-gain claim'
               if dc['ci'][0] <= 0 <= dc['ci'][1] else
               'PASS — headline survives the corrected hint'
               if dc['delta'] >= 0.020 else
               'DEMOTED — significant but < half the frozen point estimate')
    print(f"\n*** E-12b VERDICT: {verdict}")
    print(f"    causal delta {dc['delta']:+.4f}, CR-95% CI "
          f"[{dc['ci'][0]:+.4f}, {dc['ci'][1]:+.4f}], t {dc['t_cluster']:+.2f}")
    res['verdict'] = verdict
    json.dump({'rows': rows, 'summary': res},
              open(a.out, 'w'), indent=1)
    print(f"\n-> {a.out}")


if __name__ == '__main__':
    sys.exit(main())
