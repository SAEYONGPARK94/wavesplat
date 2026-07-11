"""Head anatomy -- WHY does the learned noise floor win?

Three questions on the same 60 windows (paper-grade explanation duty):

  Q1  ADAPTIVE OR CONSTANT? If a FIXED large floor (smult 10, H5's hand
      value) reproduces the head's bad-window gains, the head merely learned
      "always be humble" -- still useful, but a constant. Arms: bw_s0 gamma
      with smult in {3, 10} fixed vs the head. The head must beat fixed-10
      on good windows (where a big floor overdamps) to be called adaptive.
  Q2  WHAT DOES IT CHOOSE? per-window head smult vs the window's hand skill,
      record, Hs -- does it pick bigger floors exactly where hand fails?
  Q3  WHAT DOES IT READ? gradient saliency of log-smult wrt the 6 input
      channels (beamform power maps p1/p2/pmin + logs), averaged per window.
"""
import json
import os
import sys

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

OUT = 'results/direction3/head_anatomy.json'


def main():
    net = get_net('results/direction3/prior_net_l1s2bw.pt')
    ck = torch.load('results/direction3/prior_net_l1s2bw.pt',
                    map_location=DEVICE)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(ck['head'])
    head.eval()
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
            out['Hs'] = float(Hs)
            diag = {}
            for sm in (3.0, 10.0):
                fq = net_gamma_mmse(net, P, z, cfg['depth'], Pq, 0.05 * Hs,
                                    seed=int(i0), kp_hint=kp, use_em=False,
                                    smult_fixed=sm, diag_out=diag)
                out[f'fix{sm:g}'] = float(np.corrcoef(fq[v], gt_q[v])[0, 1])
            # head smult + saliency on this window's X
            Xt = torch.tensor(diag['X'][None], device=DEVICE,
                              requires_grad=True)
            ls = head(Xt)
            out['smult_head'] = float(torch.exp(ls[0]))
            ls.sum().backward()
            sal = Xt.grad[0].abs().mean(dim=(1, 2)).cpu().numpy()
            out['saliency'] = [float(s) for s in sal]
            rows.append(out)
            print(f"[{name} w{wi}] hand {out['hand']:+.2f} "
                  f"fix3 {out['fix3']:+.2f} fix10 {out['fix10']:+.2f} "
                  f"head {out['bw_s0_head']:+.2f} smult {out['smult_head']:.2f}",
                  flush=True)

    json.dump(rows, open(OUT, 'w'), indent=1)
    h = np.array([r['hand'] for r in rows])
    bad = h < 0.5

    print("\n=== Q1 adaptive vs constant ===")
    for k in ('fix3', 'fix10', 'bw_s0_head'):
        x = np.array([r[k] for r in rows])
        d = x - h
        print(f"{k:10s} d_bad {d[bad].mean():+.3f}  d_good {d[~bad].mean():+.3f}"
              f"  d_all {d.mean():+.3f}")
    dh = np.array([r['bw_s0_head'] - r['fix10'] for r in rows])
    seg = dh[~bad].std(ddof=1) / np.sqrt((~bad).sum())
    print(f"head - fix10: all {dh.mean():+.4f}  good {dh[~bad].mean():+.4f}"
          f"±{seg:.4f} t {dh[~bad].mean()/seg:+.2f}")

    print("\n=== Q2 what does it choose ===")
    sm = np.array([r['smult_head'] for r in rows])
    print(f"smult: bad {sm[bad].mean():.2f}±{sm[bad].std():.2f}  "
          f"good {sm[~bad].mean():.2f}±{sm[~bad].std():.2f}  "
          f"corr(smult, hand skill) {np.corrcoef(sm, h)[0,1]:+.3f}")
    hs = np.array([r['Hs'] for r in rows])
    print(f"corr(smult, Hs) {np.corrcoef(sm, hs)[0,1]:+.3f}")
    for name in RECS:
        s = [r['smult_head'] for r in rows if r['record'] == name]
        print(f"  {name:6s} smult {np.mean(s):.2f}±{np.std(s):.2f}")

    print("\n=== Q3 what does it read (channel saliency) ===")
    sal = np.array([r['saliency'] for r in rows]).mean(0)
    chn = ['p1', 'p2', 'pmin', 'log p1', 'log p2', 'log pmin']
    for c, s in sorted(zip(chn, sal), key=lambda t: -t[1]):
        print(f"  {c:8s} {s:.4f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
