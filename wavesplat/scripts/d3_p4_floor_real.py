"""P4-1 -- does the misspecification noise floor transfer to the OCEAN?

H5 found the n-poison's cause on sim: the held-out-residual grid picks a
noise floor that fits the window and kills extrapolation; fixing s2 high
(no_em, pexp 1, smult 10) rescued every sea. Real records are the arena where
misspecification is largest (broadband, currents, unknown spectra) -- if the
floor transfers, it becomes the L1 training baseline AND its s2 head target.

Arms, paired on identical draws (A2 machinery, 5 records x 5 seeds):
  default   deployed mmse_gpu (grid-select + EM)
  floor2.5  no_em, pexp 1, smult 2.5   } sig0 = 0.05*Hs here, so smult must
  floor5    no_em, pexp 1, smult 5.0   } be bracketed: smult 2.5 ~ the sim
  floor10   no_em, pexp 1, smult 10.0  } recipe's floor-to-sigma_eta ratio

Scored in-fan at t0 against the record's own surface, zero-predictor = 1.000
(A2 protocol). Reading: floor beating default on the majority of records =
the H5 mechanism is real-data-live; the best smult per record feeds L1
(amortized s2). This is a baseline run, no gates -- L1's G-REAL gate will be
judged against ITS output.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu
import d3_static_field as SF
from d3_a2_real_revalidation import draw, RECS, SEEDS

OUT = 'results/direction3/p4_floor_real.json'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ARMS = [('default', {}),
        ('floor2.5', dict(no_em=True, pexp=1.0, smult=2.5)),
        ('floor5', dict(no_em=True, pexp=1.0, smult=5.0)),
        ('floor10', dict(no_em=True, pexp=1.0, smult=10.0))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recs', nargs='*', default=RECS)
    a_ = ap.parse_args()
    t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    res = {}
    for rec in a_.recs:
        try:
            sc = SF.scene_real(rec)
        except Exception as e:
            print(f"{rec:<8} skip: {e}")
            continue
        Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
        eta = sc['make_eta']()
        gt_all = eta(Xq.ravel(), Yq.ravel(), sc['t0'])
        cx, cy = sc['center']
        inside, _ = SF.fan_mask(Xq, Yq, cx, cy, sc['hvec'])
        m = inside.ravel() & np.isfinite(gt_all)
        Pq = np.column_stack([Xq.ravel()[m], Yq.ravel()[m],
                              np.full(m.sum(), sc['t0'])])
        gtq = t(gt_all[m])
        gtq = gtq - gtq.mean()
        zero = float(gtq.std())
        sig0 = 0.05 * sc['Hs']

        cells = {n: [] for n, _ in ARMS}
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            P, Z = draw(sc, rng)
            Pt, zt, Qt = t(P), t(Z), t(Pq)
            for name, kw in ARMS:
                with torch.no_grad():
                    pr = mmse_gpu(Pt, zt, sc['depth'], Qt, sigma0=sig0,
                                  kp_hint=sc['kp'], **kw)
                    pr = pr - pr.mean()
                    beta = min(1.0, float((zt - zt.mean()).std()
                                          / pr.std().clamp_min(1e-9)))
                    r = float(torch.sqrt(((beta * pr - gtq) ** 2).mean())
                              / zero)
                cells[name].append(r)
        res[rec] = {n: [float(np.mean(v)), float(np.std(v, ddof=1))]
                    for n, v in cells.items()}
        line = '  '.join(f"{n} {res[rec][n][0]:.3f}±{res[rec][n][1]:.3f}"
                         for n, _ in ARMS)
        print(f"{rec:<8} {line}", flush=True)

    print()
    for n, _ in ARMS[1:]:
        d = [res[r]['default'][0] - res[r][n][0] for r in res]
        wins = sum(x > 0 for x in d)
        print(f"{n:9s} vs default: mean d {np.mean(d):+.4f}  wins {wins}/{len(d)}")
    json.dump(res, open(OUT, 'w'), indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
