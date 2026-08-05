"""Audit check: is the Hs axis a pure scale symmetry of B's eval pipeline?
Same rng seed, hs_range forced to a point value (2.0 vs 4.5): every other
draw identical => if the pipeline is scale-equivariant, corr must match to
float precision, proving 'EX_Hs OOD' is not a functional extrapolation axis.
CPU only (avoid shared GPU)."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp')

from e28_gennet_B import (gen_widened, inform_features, mmse_field, corr,
                          FiLMPriorNet, _MASK)
from d3_prior_net_train import PriorNet

DEV = torch.device('cpu')
cur = torch.load('results/direction3/prior_net_v3.pt', map_location=DEV)
cur_net = PriorNet()
cur_net.load_state_dict(cur['model'])
cur_net.eval()
cur_ls = torch.tensor(float(cur['log_smult']))

print(f"{'seed':>4s} {'Hs_lo':>6s} {'Hs_hi':>6s} {'corr(cur)':>10s}")
for seed in (11, 12, 13, 14):
    row = []
    for hs in (2.0, 4.5):
        s = gen_widened(np.random.default_rng(seed), hs_range=(hs, hs),
                        side_range=(1.2, 1.8), bw_max=0.0)
        X = torch.tensor(s['X'][None])
        P = torch.tensor(s['P']); z = torch.tensor(s['z'])
        Q = torch.tensor(s['Q'])
        with torch.no_grad():
            oc = cur_net(X)[0]
            pc = mmse_field(oc, P, z, Q, s['kp_e'], s['nfrac'] * s['Hs'],
                            cur_ls)
        c = corr(pc.numpy(), s['gt_q'])
        row.append(c)
        print(f"{seed:4d} {hs:6.1f} {hs:6.1f} {c:+10.6f}")
    print(f"     |diff| = {abs(row[0]-row[1]):.2e}  "
          f"{'SCALE-SYMMETRIC (Hs is a no-op axis)' if abs(row[0]-row[1])<1e-3 else 'Hs MATTERS'}")
