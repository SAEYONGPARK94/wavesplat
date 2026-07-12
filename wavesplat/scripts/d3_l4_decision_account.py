"""L4 step A: decision-space accounting on the frozen 60 real windows.

Pre-registration: docs/L4_DECISION_prereg_2026-07-12.md (frozen before this
run). Measures the no-learning arms' decision performance -- crest-arrival
error (T1) and sign accuracy (T2) at the ship encounter track -- and judges
gate G-L4-A: training starts ONLY if the best non-learned arm leaves
T1 MAE >= 0.8 s or T2 <= 85 %.
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

G = 9.81
OUT = 'results/direction3/l4_decision_account.json'
RECS = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']


def crest_time(leads, y):
    m = leads >= 1.0
    return float(leads[m][int(np.argmax(y[m]))])


def sign_acc(leads, pred, truth, hs):
    m = (leads >= 1.0) & (np.abs(truth) >= 0.05 * hs)
    if m.sum() == 0:
        return None
    return float(np.mean(np.sign(pred[m]) == np.sign(truth[m])))


def main():
    net = get_net('results/direction3/prior_net_l1s2bw.pt')
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(
        'results/direction3/prior_net_l1s2bw.pt', map_location=DEVICE)['head'])
    head.eval()

    rows, skipped = [], 0
    for name in RECS:
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        for wi in range(12):
            t_st = time.time()
            w = Window(name, wi, rec=rec, net=net, head=head)
            leads, gt, ph, pn = w.track_series(lead_max=10.0)
            fin = np.isfinite(gt)
            span = leads[fin].max() if fin.any() else 0.0
            if span < 6.0:
                skipped += 1
                print(f"[{name} w{wi}] skipped (finite span {span:.1f}s)")
                continue
            keep = fin & (leads <= span)
            L, T = leads[keep], gt[keep]
            H, N = ph[keep], pn[keep]
            om_p = np.sqrt(G * w.kp * np.tanh(w.kp * cfg['depth']))
            Tp = 2 * np.pi / om_p
            tri_sign = np.full_like(T, np.sign(T[0]) if T[0] != 0 else 1.0)
            tri_tc = float(np.clip(Tp / 2, 1.0, 10.0))
            tc_t = crest_time(L, T)
            row = {'record': name, 'wi': wi, 'span': float(span),
                   'Tp': float(Tp), 'Hs': float(w.Hs), 'tc_truth': tc_t}
            for arm, p in (('hand', H), ('learned', N)):
                row[f'tc_err_{arm}'] = abs(crest_time(L, p) - tc_t)
                row[f'sacc_{arm}'] = sign_acc(L, p, T, w.Hs)
            row['tc_err_trivial'] = abs(tri_tc - tc_t)
            row['sacc_trivial'] = sign_acc(L, tri_sign, T, w.Hs)
            rows.append(row)
            print(f"[{name} w{wi}] tc_err h {row['tc_err_hand']:.1f}s "
                  f"l {row['tc_err_learned']:.1f}s tri "
                  f"{row['tc_err_trivial']:.1f}s | sacc h "
                  f"{row['sacc_hand']:.2f} l {row['sacc_learned']:.2f} "
                  f"tri {row['sacc_trivial']:.2f} [{time.time()-t_st:.0f}s]",
                  flush=True)

    json.dump(rows, open(OUT, 'w'), indent=1)
    print(f"\nwrote {OUT} (n={len(rows)}, skipped {skipped})")

    print(f"\n--- decision accounting (n={len(rows)} windows) ---")
    best = {}
    for arm in ('hand', 'learned', 'trivial'):
        tc = np.array([r[f'tc_err_{arm}'] for r in rows])
        sa = np.array([r[f'sacc_{arm}'] for r in rows
                       if r[f'sacc_{arm}'] is not None])
        best[arm] = (tc.mean(), sa.mean())
        print(f"{arm:8s} T1 crest MAE {tc.mean():.2f}s (med "
              f"{np.median(tc):.2f})  T2 sign acc {sa.mean():.1%}")
    # gate G-L4-A: judged on the best NON-LEARNED arm (hand; trivial is floor)
    t1, t2 = best['hand']
    room = t1 >= 0.8 or t2 <= 0.85
    print(f"\nG-L4-A (train only if hand T1>=0.8s or T2<=85%): "
          f"{'OPEN - training justified' if room else 'CLOSED - no room'}"
          f"  (hand T1 {t1:.2f}s, T2 {t2:.1%})")
    print(f"reference incumbent(learned): T1 {best['learned'][0]:.2f}s, "
          f"T2 {best['learned'][1]:.1%}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
