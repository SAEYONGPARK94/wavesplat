"""E-14 — how much of the HOS "+0.091" is just the observation draw?

Found while running E-13: d3_a1_field_hos.py:117 builds ONE rng
(`np.random.default_rng(7)`) and hands it to all 8 cells in sequence, so each
cell's 600-point observation draw depends on how many draws the earlier cells
consumed. E-13 gives every cell a fresh rng instead.

Consequence, measured by accident:

    same sea (Tp 8, Hs 1.2, seed 0)  ->  IDENTICAL wave field (hos_run is
    deterministic in `seed`), only the 600-point sample and its noise differ:

        frozen run   delta = +0.068
        E-13 run     delta = +0.012

    A swing of 0.056 from re-drawing the observations -- comparable to the whole
    claimed HOS effect (+0.091). That variance has never been quantified: every
    frozen cell drew its 600 points exactly ONCE, and the "8/8 sign" was counted
    on that single draw.

This holds the sea FIXED and re-draws only the observations, R times per sea, on
the two frozen seas plus two E-13 seas. It answers:

  Q1  sd of delta across observation draws, within a sea.
  Q2  how does that compare to the between-sea sd (E-13) and to the claimed
      effect (+0.091)?
  Q3  does the 8/8 sign consistency survive re-drawing? (i.e. is the sign of the
      per-cell delta stable, or is it a coin flip on most cells?)

If the within-sea observation-draw sd is comparable to the effect, then the frozen
8-cell HOS estimate has a large unmodelled noise term and its CI -- computed as if
each cell were one clean measurement -- is far too narrow, on top of the k=2 sea
problem already found (E-12e section 10.4).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

os.environ.setdefault('A1_N_OBS', '600')

import d3_f1_hos_door1 as F1                     # noqa: E402
import d3_a1_field_hos as A1                     # noqa: E402
from d3_prior_net_eval import get_net, DEVICE    # noqa: E402
from d3_l1_s2head_train import SmultHead         # noqa: E402
import torch                                     # noqa: E402

OUT = 'results/direction3/e14_obs_draw_variance.json'
CKPT = 'results/direction3/prior_net_l1s2bw.pt'

# (Tp, Hs, seed): the two FROZEN seas + two E-13 seas that bracket the range
CASES = [
    (8.0, 1.2, 0),      # frozen cell, frozen delta +0.0677
    (8.0, 3.0, 0),      # frozen cell, frozen delta +0.0589
    (7.0, 3.0, 0),      # E-13's biggest gain (+0.147)
    (9.0, 4.0, 0),      # E-13's negative cell (-0.015)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=6,
                    help='independent 600-point observation draws per sea')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    net = get_net(CKPT)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head'])
    head.eval()

    rows = []
    for (Tp, Hs, seed) in CASES:
        F1.TP = float(Tp)
        ds = []
        for r in range(a.draws):
            t0 = time.time()
            # the sea (hos_run/sea_components) depends ONLY on `seed`, so the
            # wave field is byte-identical across draws; `rng` drives only the
            # scan sampling and the measurement noise.
            rng = np.random.default_rng(50_000 + 137 * r)
            row = A1.run_cell(seed, float(Hs), net, head, rng)
            d = row['learned'] - row['hand']
            ds.append(d)
            rows.append(dict(Tp=Tp, Hs=Hs, seed=seed, draw=r,
                             hand=row['hand'], learned=row['learned'],
                             delta=d, n_obs=row['n_obs']))
            print(f"[Tp {Tp:4.1f} Hs {Hs:3.1f} draw {r}] hand {row['hand']:+.3f}"
                  f"  net {row['learned']:+.3f}  d {d:+.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            json.dump(rows, open(a.out, 'w'), indent=1)
        ds = np.array(ds)
        print(f"  --> sea Tp{Tp:g}_Hs{Hs:g}: delta mean {ds.mean():+.4f}  "
              f"sd {ds.std(ddof=1):.4f}  range [{ds.min():+.4f}, {ds.max():+.4f}]"
              f"  sign {(ds>0).sum()}/{len(ds)}\n", flush=True)

    print('=' * 78)
    print('E-14 — observation-draw variance, sea held FIXED')
    print('=' * 78)
    print(f"{'sea':<14}{'mean d':>9}{'sd(draw)':>10}{'min':>9}{'max':>9}"
          f"{'sign':>7}")
    summ = {}
    for (Tp, Hs, seed) in CASES:
        ds = np.array([r['delta'] for r in rows
                       if r['Tp'] == Tp and r['Hs'] == Hs])
        sd = float(ds.std(ddof=1))
        summ[f'Tp{Tp:g}_Hs{Hs:g}'] = dict(
            mean=float(ds.mean()), sd_draw=sd, min=float(ds.min()),
            max=float(ds.max()), sign=f"{int((ds > 0).sum())}/{len(ds)}",
            n_draws=len(ds))
        print(f"Tp{Tp:g}_Hs{Hs:g}".ljust(14)
              + f"{ds.mean():+9.4f}{sd:10.4f}{ds.min():+9.4f}{ds.max():+9.4f}"
              + f"{f'{(ds>0).sum()}/{len(ds)}':>7}")

    sds = np.array([v['sd_draw'] for v in summ.values()])
    print(f"\n  pooled within-sea sd from re-drawing the 600 points: "
          f"{sds.mean():.4f}")
    try:
        e13 = json.load(open('results/direction3/e13_hos_sea_grid.json'))
        sm = e13['summary']['sea_means']
        between = float(np.std(list(sm.values()), ddof=1))
        print(f"  between-sea sd (E-13, {len(sm)} seas):                    "
              f"{between:.4f}")
        print(f"  the claimed HOS effect (frozen, 2 seas):             +0.0908")
        print(f"\n  --> a single observation draw carries +-{sds.mean():.3f} of "
              f"noise on a quantity claimed to be +0.091.")
        summ['_between_sea_sd_e13'] = between
    except (FileNotFoundError, KeyError):
        pass
    summ['_within_sea_sd_pooled'] = float(sds.mean())

    json.dump({'rows': rows, 'summary': summ}, open(a.out, 'w'), indent=2)
    print(f"\n-> {a.out}")


if __name__ == '__main__':
    sys.exit(main())
