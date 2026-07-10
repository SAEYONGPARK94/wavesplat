"""Which starved: the encoder, or the probe?

The sea-count sweep moved two things at once. With 13 training seas the encoder
saw 104 distinct frames AND the ridge probe fit 257 coefficients to 13 distinct
label values. Both are starved, so the collapse (SSL R^2 = -0.708 at 13 seas,
almost exactly W7's -0.603) does not yet identify a mechanism.

The tell is already in the sweep: the RANDOM-INIT encoder never touches data,
and it collapses harder (-2.134 at 13 seas, +0.339 at 1600). A collapse that
happens to an encoder which cannot possibly have been starved is the probe's.

Separate the axes on a 2x2, using the frozen 1600-sea checkpoint:

  encoder seas   probe seas   isolates
  ------------   ----------   --------
      1600           13       probe starvation alone
      1600         1600       neither
        13           13       both  (the sweep's cell)
        13         1600       encoder starvation alone

Held-out 400 seas throughout. If (1600, 13) is as bad as (13, 13), the encoder
was never the problem and the honest claim is narrower: "you cannot FIT or
VALIDATE this mapping from 13 seas", not "the encoder needs thousands".
"""
import os
import sys

import numpy as np
import torch

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')
sys.path.insert(0, '..')

from d3_ssl_pretrain import MAE
from d3_g1_synth_ssl import (build, standardize, features, probe, N_IN, PATCH,
                             DEV)

HELD = 400
# MATCHED STEPS. The first version paired the 100k-step main run (enc=1600)
# against a 20k-step sweep run (enc=13) and "found" that 13 seas gives better
# features. It does not: G1 showed the probe R2 DECAYS with pretraining steps,
# so that 2x2 was measuring steps, not seas. Both checkpoints below are 20k.
CKPTS = {1600: 'results/direction3/g1_synth_ssl_ns1600.pt',
         100: 'results/direction3/g1_synth_ssl_ns100.pt',
         13: 'results/direction3/g1_synth_ssl_ns13.pt'}
PROBE_SEAS = [13, 25, 50, 100, 400, 1600]


def main():
    X, K, KT, M = build(2000, 0)
    n_pool = len(K) - HELD
    Zte = torch.tensor(standardize(X[n_pool:]).reshape(-1, PATCH, PATCH))
    yte = np.repeat(K[n_pool:], N_IN)

    enc = {}
    for ns, p in CKPTS.items():
        if not os.path.exists(p):
            print(f"missing {p}; run the sweep first")
            return 1
        c = torch.load(p, map_location=DEV)
        m = MAE(dim=c['dim']).to(DEV)
        m.load_state_dict(c['model'])
        enc[ns] = m
    torch.manual_seed(0)
    enc['random'] = MAE(dim=256).to(DEV)

    Fte = {k: features(v, Zte) for k, v in enc.items()}

    print("held-out R^2 on 400 unseen seas\n")
    print(f"{'probe seas':<12}" + ''.join(f"{f'enc={k}':>14}" for k in enc))
    for ps in PROBE_SEAS:
        Ztr = torch.tensor(standardize(X[:ps]).reshape(-1, PATCH, PATCH))
        ytr = np.repeat(K[:ps], N_IN)
        row = f"{ps:<12}"
        for k, m in enc.items():
            row += f"{probe(features(m, Ztr), ytr, Fte[k], yte):>+14.3f}"
        print(row)

    print("\nread the two corners:")
    print("  enc=1600, probe=13    the probe alone, starved")
    print("  enc=13,   probe=1600  the encoder alone, starved")


if __name__ == '__main__':
    sys.exit(main())
