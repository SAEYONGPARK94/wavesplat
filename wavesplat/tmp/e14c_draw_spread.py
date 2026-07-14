"""E-14c — resolve the Tp8/Hs1.2 anomaly.

E-14b proved the frozen JSON reproduces BIT-FOR-BIT under its own protocol
(shared rng(7), same cell order). So the frozen numbers are legitimate.

That sharpens a puzzle. For the cell (Tp 8, Hs 1.2, seed 0):

    frozen (rng 7, fresh)          delta = +0.0677
    E-14, 6 draws (rng 50000+137r) delta = -0.006 +- 0.011, range [-0.016, +0.015]

+0.0677 sits ~6 sd outside my six draws. Either
  (a) the true draw-to-draw distribution is wide and my six seeds happened to
      cluster (making E-14's sd=0.011 for this sea a fluke), or
  (b) something about the 50000-series seeds is not an independent sample.

Note the other three seas gave draw sd 0.038 / 0.061 / 0.085 -- so a sd of 0.011
for this sea is itself the outlier, 4-8x smaller than its neighbours.

This re-runs the SAME cell across a spread of unrelated rng seeds (small ints,
the 50000-series, and some primes) and reports the empirical distribution. It
decides whether E-14's headline (observation-draw noise sd ~ 0.049) is sound and
whether the frozen cell was simply a lucky draw.
"""
import os
import sys
import time
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

os.environ.setdefault('A1_N_OBS', '600')

import d3_f1_hos_door1 as F1                     # noqa: E402
import d3_a1_field_hos as A1                     # noqa: E402
from d3_prior_net_eval import get_net, DEVICE    # noqa: E402
from d3_l1_s2head_train import SmultHead         # noqa: E402
import torch                                     # noqa: E402

OUT = 'results/direction3/e14c_draw_spread.json'
CKPT = 'results/direction3/prior_net_l1s2bw.pt'
TP, HS, SEED = 8.0, 1.2, 0

# deliberately heterogeneous seeds: the frozen one, small ints, primes, and the
# 50000-series E-14 used (to test whether that series is somehow special)
SEEDS = [7, 1, 2, 3, 101, 997, 50_000, 50_137, 12_345, 424_242]

net = get_net(CKPT)
head = SmultHead().to(DEVICE)
head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head'])
head.eval()
F1.TP = TP

rows = []
print(f"cell: Tp {TP} Hs {HS} seed {SEED}  (frozen delta = +0.0677)")
print(f"\n{'rng seed':>10}{'hand':>10}{'net':>10}{'delta':>10}")
print('-' * 42)
for s in SEEDS:
    t0 = time.time()
    r = A1.run_cell(SEED, HS, net, head, np.random.default_rng(s))
    d = r['learned'] - r['hand']
    rows.append(dict(rng_seed=s, hand=r['hand'], learned=r['learned'],
                     delta=d, n_obs=r['n_obs']))
    print(f"{s:>10}{r['hand']:+10.4f}{r['learned']:+10.4f}{d:+10.4f}"
          f"   [{time.time()-t0:.0f}s]", flush=True)
    json.dump(rows, open(OUT, 'w'), indent=1)

d = np.array([r['delta'] for r in rows])
print(f"\nn={len(d)}  mean {d.mean():+.4f}  sd {d.std(ddof=1):.4f}  "
      f"range [{d.min():+.4f}, {d.max():+.4f}]  sign {(d>0).sum()}/{len(d)}")
old = np.array([r['delta'] for r in rows if r['rng_seed'] in (50_000, 50_137)])
print(f"  the two E-14 seeds in this set: {np.round(old, 4)}")
print(f"  the frozen seed (7): {d[0]:+.4f}  "
      f"-> z = {(d[0]-d.mean())/d.std(ddof=1):+.2f} within this spread")
json.dump({'rows': rows,
           'summary': dict(n=len(d), mean=float(d.mean()),
                           sd=float(d.std(ddof=1)), min=float(d.min()),
                           max=float(d.max()),
                           sign=f"{int((d>0).sum())}/{len(d)}")},
          open(OUT, 'w'), indent=2)
print(f"\n-> {OUT}")
