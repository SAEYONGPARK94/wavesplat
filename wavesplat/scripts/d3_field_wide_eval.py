"""Whole-field audit: hand vs learned on the FULL stereo-valid grid,
60 frozen real windows, leads {0, 2, 5, 8} s.

User goal statement (2026-07-12): the target is wave-field estimation
itself -- everywhere, not just where the ship goes, invariant to
direction/speed. The 8-record gamma-prior win is already a whole-field
(common_unobs) result; this audit adds the direct 60-window whole-field
verdict for the deployed arms (hand vs net+s2-head), per lead and pooled.
Descriptive audit (frozen questions, no promotion gate):
  Q1 pooled whole-field corr, paired: does learned hold parity/win?
  Q2 per-lead crossing structure (does the AA14-w10 pattern generalize?)
  Q3 amplitude ratio per arm (texture/shrinkage account).
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
from d3_prior_net_eval import get_net, DEVICE
from d3_l1_s2head_train import SmultHead
from d3_external_figs import Window

OUT = 'results/direction3/field_wide_eval.json'
RECS = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']
LEADS = [0.0, 2.0, 5.0, 8.0]


def main():
    net = get_net('results/direction3/prior_net_l1s2bw.pt')
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(
        'results/direction3/prior_net_l1s2bw.pt', map_location=DEVICE)['head'])
    head.eval()

    rows = []
    for name in RECS:
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        for wi in range(12):
            t0 = time.time()
            w = Window(name, wi, rec=rec, net=net, head=head)
            leads = [l_ for l_ in LEADS
                     if l_ <= min(10.0, w.leads.max())]
            gx, gy, tr, fh, fn = w.fields(leads, stride=2)
            trs = np.array(tr)
            m = np.isfinite(trs)
            if m.sum() < 500:
                print(f"[{name} w{wi}] skipped (finite px {int(m.sum())})")
                continue
            fhs, fns = np.array(fh), np.array(fn)
            row = {'record': name, 'wi': wi, 'leads': leads,
                   'n_px': int(m.sum())}
            row['field_hand'] = float(np.corrcoef(fhs[m], trs[m])[0, 1])
            row['field_learned'] = float(np.corrcoef(fns[m], trs[m])[0, 1])
            per_h, per_n = [], []
            for i in range(len(leads)):
                mi = np.isfinite(trs[i])
                per_h.append(float(np.corrcoef(fhs[i][mi], trs[i][mi])[0, 1]))
                per_n.append(float(np.corrcoef(fns[i][mi], trs[i][mi])[0, 1]))
            row['per_lead_hand'] = per_h
            row['per_lead_learned'] = per_n
            sd_t = float(np.nanstd(trs[m]))
            row['amp_hand'] = float(np.std(fhs[m]) / sd_t)
            row['amp_learned'] = float(np.std(fns[m]) / sd_t)
            rows.append(row)
            print(f"[{name} w{wi}] field h {row['field_hand']:+.2f} "
                  f"l {row['field_learned']:+.2f} | lead0 h {per_h[0]:+.2f} "
                  f"l {per_n[0]:+.2f} [{time.time()-t0:.0f}s]", flush=True)

    json.dump(rows, open(OUT, 'w'), indent=1)
    fh_ = np.array([r['field_hand'] for r in rows])
    fn_ = np.array([r['field_learned'] for r in rows])
    d = fn_ - fh_
    se = d.std(ddof=1) / np.sqrt(len(d))
    recs_arr = np.array([r['record'] for r in rows])
    rec_means = [d[recs_arr == nm].mean() for nm in RECS]
    print(f"\n--- whole-field audit (n={len(rows)}) ---")
    print(f"Q1 pooled: hand {fh_.mean():+.3f}  learned {fn_.mean():+.3f}  "
          f"paired d {d.mean():+.3f} (t {d.mean()/se:+.2f}), "
          f"learned wins {(d>0).sum()}/{len(d)}, record sign "
          f"{sum(1 for x in rec_means if x>0)}/{len(RECS)}")
    for li, l_ in enumerate(LEADS):
        ph = [r['per_lead_hand'][li] for r in rows
              if len(r['per_lead_hand']) > li]
        pn = [r['per_lead_learned'][li] for r in rows
              if len(r['per_lead_learned']) > li]
        print(f"Q2 lead {l_:3.0f}s: hand {np.mean(ph):+.3f}  "
              f"learned {np.mean(pn):+.3f}  (n={len(ph)})")
    ah = np.mean([r['amp_hand'] for r in rows])
    al = np.mean([r['amp_learned'] for r in rows])
    print(f"Q3 amplitude: hand {ah:.2f}x  learned {al:.2f}x of truth")
    print(f"wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
