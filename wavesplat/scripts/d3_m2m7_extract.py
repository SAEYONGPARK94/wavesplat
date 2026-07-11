"""M2xM7 dataset extraction -- dense sliding windows from the REAL records.

Layer-2 lesson (M8): clean synthetic is a G-ZERO arena; train and judge on
real data. Each stereo record has ~18k frames at 10 Hz; the 60-window
judgment set used only 12 window starts per record. This extractor pulls a
DENSE sliding set (stride ~5 s) of [forward-fan corridor observations ->
future encounter track] pairs with labels computed from the record's own
surfaces -- the M2xM7 training base, LORO-split by record.

Per window (verbatim d3_netfan_eval geometry): P (<=600, 3) obs, z, the
recipe-arm forecast track (net+s2-head, the deployable incumbent), the true
track gt(lead), leads <= 10 s, plus scalars (Hs, kp, record, t0).
Output: one npz per record in tmp/m2m7/ (obs kept for future end-to-end
arms; tracks are the primary training signal).

JUDGMENT windows (the 12 canonical i0s per record) are EXCLUDED from the
training extraction (exact i0 match) so the frozen 60-window set stays
untouched for gating.
"""
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

OUTDIR = 'tmp/m2m7'
STRIDE_S = 5.0


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    net = get_net('results/direction3/prior_net_l1s2bw.pt')
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load('results/direction3/prior_net_l1s2bw.pt',
                                    map_location=DEVICE)['head'])
    head.eval()
    for name in RECS:
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        nfw = int(round(40.0 / rec.dt))
        n_lead = int(round(cfg['lead_max'] / rec.dt))
        judge_i0s = set(np.linspace(nfw + 1, rec.T_all - n_lead - 2,
                                    12).astype(int))
        step = max(1, int(round(STRIDE_S / rec.dt)))
        i0s = [i for i in range(nfw + 1, rec.T_all - n_lead - 2, step)
               if i not in judge_i0s]
        rows = {'pred': [], 'gt': [], 'leads': [], 'Hs': [], 'kp': [],
                't0': [], 'i0': []}
        t0c = time.time()
        for k, i0 in enumerate(i0s):
            try:
                P, z, Pq, gt_q, leads, Hs, kp, _ = window_data(
                    rec, cfg, int(i0), nfw, n_lead, k + 100)
            except Exception:
                continue
            v = np.isfinite(gt_q) & (leads <= 10.0)
            if v.sum() < 8:
                continue
            fq = np.asarray(net_gamma_mmse(net, P, z, cfg['depth'], Pq,
                                           0.05 * Hs, kp_hint=kp,
                                           use_em=False, smult_head=head))
            rows['pred'].append(fq[v][:11])
            rows['gt'].append(gt_q[v][:11])
            rows['leads'].append(leads[v][:11])
            rows['Hs'].append(Hs)
            rows['kp'].append(kp)
            rows['t0'].append(float(rec.t[int(i0)]))
            rows['i0'].append(int(i0))
            if (k + 1) % 50 == 0:
                print(f"[{name}] {k+1}/{len(i0s)} windows "
                      f"[{time.time()-t0c:.0f}s]", flush=True)
        # pad ragged to 11
        for key in ('pred', 'gt', 'leads'):
            rows[key] = np.array([np.pad(x, (0, 11 - len(x)),
                                         constant_values=np.nan)
                                  for x in rows[key]], np.float32)
        np.savez_compressed(f"{OUTDIR}/{name}.npz",
                            **{k: np.asarray(v) for k, v in rows.items()})
        print(f"[{name}] saved {len(rows['Hs'])} windows -> "
              f"{OUTDIR}/{name}.npz [{time.time()-t0c:.0f}s]", flush=True)
    print("M2M7_EXTRACT_DONE")
    return 0


if __name__ == '__main__':
    sys.exit(main())
