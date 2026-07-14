"""E-14b — why does the frozen HOS cell not reproduce?

E-14 held the sea FIXED (Tp 8, Hs 1.2, seed 0 -- the wave field is deterministic
in `seed`) and re-drew only the 600-point observation sample, 6 times:

    mean delta -0.0060, sd 0.0114, range [-0.0158, +0.0149], sign 1/6

The FROZEN value for that exact cell is +0.0677 (a1_field_hos_sparse600.json,
n_obs = 600 -- the same budget). That is ~6 sd outside the re-draw distribution.
Observation noise cannot do that. Something differs SYSTEMATICALLY between the
frozen run and mine.

The one thing I changed: the frozen main (d3_a1_field_hos.py:117) builds ONE
`np.random.default_rng(7)` and passes it to all 8 cells IN SEQUENCE. I give each
cell a fresh rng.

This reproduces the frozen protocol EXACTLY -- one shared rng(7), same cell order
(Hs outer, seed inner, HS_LIST=[1.2, 3.0], SEEDS=[0,1,2,3]) -- and prints the two
Tp=8 cells that E-14 could not match. Only the first 2 cells are needed to decide.

  If it reproduces +0.0677  -> the frozen number is real; my fresh-rng draws are a
                              different population, and I must find out why.
  If it does NOT reproduce  -> the frozen JSON was not produced by this code path
                              at this budget, and the HOS leg's provenance is
                              broken (a far more serious finding).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

os.environ.setdefault('A1_N_OBS', '600')

import d3_a1_field_hos as A1                     # noqa: E402
from d3_prior_net_eval import get_net, DEVICE    # noqa: E402
from d3_l1_s2head_train import SmultHead         # noqa: E402
import torch                                     # noqa: E402
import json                                      # noqa: E402

CKPT = 'results/direction3/prior_net_l1s2bw.pt'

print(f"A1.N_OBS = {A1.N_OBS}   (frozen file says n_obs=600)")
print(f"A1.SEEDS = {A1.SEEDS}   A1.HS_LIST = {A1.HS_LIST}")

net = get_net(CKPT)
head = SmultHead().to(DEVICE)
head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head'])
head.eval()

frozen = json.load(open('results/direction3/a1_field_hos_sparse600.json'))

# EXACTLY the frozen protocol: one shared rng, same loop order
rng = np.random.default_rng(7)
print(f"\n{'cell':<18}{'hand':>9}{'net':>9}{'delta':>9}   "
      f"{'frozen hand':>12}{'frozen net':>11}{'frozen d':>10}  match?")
print('-' * 88)
for Hs in A1.HS_LIST:
    for seed in A1.SEEDS:
        t0 = time.time()
        r = A1.run_cell(seed, Hs, net, head, rng)     # SAME rng object, in order
        d = r['learned'] - r['hand']
        fz = [x for x in frozen
              if x['seed'] == seed and abs(x['Hs'] - Hs) < 1e-9][0]
        fd = fz['learned'] - fz['hand']
        ok = abs(d - fd) < 1e-6
        print(f"Hs{Hs:g}_seed{seed}".ljust(18)
              + f"{r['hand']:+9.4f}{r['learned']:+9.4f}{d:+9.4f}   "
              + f"{fz['hand']:+12.4f}{fz['learned']:+11.4f}{fd:+10.4f}"
              + f"  {'EXACT' if ok else 'DIFFERS'}  [{time.time()-t0:.0f}s]",
              flush=True)
        if Hs == A1.HS_LIST[0] and seed == 1:
            print('\n  (first two cells decide it — stopping unless they match)')
            if not ok:
                print('  --> frozen protocol does NOT reproduce the frozen JSON.')
                sys.exit(0)
