"""Head-seed confirmation — pre-registered BEFORE running.

The 60-window promotion run produced a post-hoc observation: the learned
scene-adaptive noise floor (SmultHead, seed-0) clears the value thresholds
the gamma-net alone failed (bad-window d vs hand +0.227, t 4.75; vs its own
base +0.156, t 3.19; good windows unharmed). Post-hoc on ONE head. Gates for
promotion, frozen now:

  G-H1  for EVERY seed i in {0,1,2}: bad-window mean d(head_i - hand) >= +0.10,
        and pooled across seeds t >= 2.5.
  G-H2  for EVERY seed: bad-window mean d(head_i - base_i) > 0.
FAIL on any -> the head result is reported as seed-fragile, no claim.

This script evaluates the s1/s2 heads on the SAME 60 windows (same
window_data, same seeds) and reads the verdict with s0 from
gate_promotion_eval.json.
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
from d3_prior_net_eval import get_net, net_gamma_mmse, DEVICE
from d3_l1_s2head_train import SmultHead
from d3_gate_promotion_eval import window_data, RECS

OUT = 'results/direction3/gate_head_seeds.json'


def main():
    arms = {}
    for s in (1, 2):
        ck = torch.load(f'results/direction3/prior_net_l1s2bw_s{s}.pt',
                        map_location=DEVICE)
        head = SmultHead().to(DEVICE)
        head.load_state_dict(ck['head'])
        head.eval()
        arms[f's{s}'] = (get_net(f'results/direction3/prior_net_l1s2bw_s{s}.pt'),
                         head)

    base = json.load(open('results/direction3/gate_promotion_eval.json'))
    bmap = {(r['record'], r['wi']): r for r in base}
    rows = []
    for name in RECS:
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        nfw = int(round(40.0 / rec.dt))
        n_lead = int(round(cfg['lead_max'] / rec.dt))
        i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 12).astype(int)
        for wi, i0 in enumerate(i0s):
            if (name, wi) not in bmap:
                continue
            P, z, Pq, gt_q, leads, Hs, kp, _ = window_data(
                rec, cfg, int(i0), nfw, n_lead, wi)
            v = np.isfinite(gt_q) & (leads <= 10.0)
            out = dict(bmap[(name, wi)])
            for s, (net, head) in arms.items():
                fq = net_gamma_mmse(net, P, z, cfg['depth'], Pq, 0.05 * Hs,
                                    seed=int(i0), kp_hint=kp, use_em=False,
                                    smult_head=head)
                out[f'head_{s}'] = float(np.corrcoef(fq[v], gt_q[v])[0, 1])
            rows.append(out)
            print(f"[{name} w{wi}] head_s1 {out['head_s1']:+.2f} "
                  f"head_s2 {out['head_s2']:+.2f} "
                  f"(s0 {out['bw_s0_head']:+.2f} hand {out['hand']:+.2f})",
                  flush=True)

    json.dump(rows, open(OUT, 'w'), indent=1)
    h = np.array([r['hand'] for r in rows])
    bad = h < 0.5
    print(f"\nn {len(rows)} bad {bad.sum()}")
    heads = {'s0': np.array([r['bw_s0_head'] for r in rows]),
             's1': np.array([r['head_s1'] for r in rows]),
             's2': np.array([r['head_s2'] for r in rows])}
    bases = {'s0': np.array([r['bw_s0'] for r in rows]),
             's1': np.array([r['bw_s1'] for r in rows]),
             's2': np.array([r['bw_s2'] for r in rows])}
    dbads, gh1_each, gh2_each = [], [], []
    for s in ('s0', 's1', 's2'):
        db = (heads[s] - h)[bad]
        dbase = (heads[s] - bases[s])[bad]
        dbads.append(db)
        gh1_each.append(db.mean() >= 0.10)
        gh2_each.append(dbase.mean() > 0)
        print(f"{s}: d(head-hand|bad) {db.mean():+.3f}  "
              f"d(head-base|bad) {dbase.mean():+.3f}")
    pooled = np.concatenate(dbads)
    # cluster by window (3 seeds share windows): use per-window seed-mean
    wmean = np.mean(np.stack([d for d in dbads]), axis=0)
    se = wmean.std(ddof=1) / np.sqrt(len(wmean))
    t_ = wmean.mean() / se
    print(f"pooled (window-clustered): {wmean.mean():+.3f}+-{se:.3f} t {t_:+.2f}")
    gh1 = all(gh1_each) and t_ >= 2.5
    gh2 = all(gh2_each)
    print(f"G-H1 {'PASS' if gh1 else 'FAIL'}   G-H2 {'PASS' if gh2 else 'FAIL'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
