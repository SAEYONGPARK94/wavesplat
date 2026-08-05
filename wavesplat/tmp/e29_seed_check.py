"""e29 diagnostic: is the ens gain real averaging or one lucky seed?
Arms: cur(=s0), s1, s2 individually, ens(mean log-gamma), on spread seas."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')
import numpy as np
import torch
from e29_gennet_C import (build_sea_c, corr, DEVICE, SEED_CKPTS, MeanNet,
                          load_incumbent)
from d3_prior_net_train import PriorNet
from d3_prior_net_eval import net_gamma_mmse

DEPTH = 50.0
nets = []
for p in SEED_CKPTS:
    ck = torch.load(p, map_location=DEVICE)
    n = PriorNet().to(DEVICE)
    n.load_state_dict(ck['model'])
    n.eval()
    nets.append(n)
ens = MeanNet(nets).eval()
_, head = load_incumbent()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
rows = []
for si in range(N):
    S = build_sea_c('spread', si)
    P, z, Q, gt = S['P'], S['z'], S['Pq'], S['gt_u']
    kp, s0 = S['kp'], S['sigma0']
    r = {}
    for name, net in [('s0', nets[0]), ('s1', nets[1]), ('s2', nets[2]),
                      ('ens', ens)]:
        f = net_gamma_mmse(net, P, z, DEPTH, Q, s0, seed=0, kp_hint=kp,
                           smult_head=head, bound_wave=True)
        r[name] = corr(f, gt)
    rows.append(r)
    print(f"sea {si:2d}: s0 {r['s0']:+.3f} s1 {r['s1']:+.3f} "
          f"s2 {r['s2']:+.3f} ens {r['ens']:+.3f}", flush=True)
for k in ('s0', 's1', 's2', 'ens'):
    m = np.mean([r[k] for r in rows])
    w = np.mean([r[k] >= r['s0'] for r in rows])
    print(f"{k}: mean {m:+.4f}  d(vs s0) {m - np.mean([r['s0'] for r in rows]):+.4f}  win {w:.0%}")
