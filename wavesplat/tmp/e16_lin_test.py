import os, sys, time, numpy as np
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')
os.environ.setdefault('A1_N_OBS', '600')
import torch
from e16_physics_operator import build_sea_linear, one_draw
from e15_hos_multidraw import CKPT, lam_p
from d3_prior_net_eval import get_net, DEVICE
from d3_l1_s2head_train import SmultHead

net = get_net(CKPT)
head = SmultHead().to(DEVICE)
head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head'])
head.eval()

for Tp, Hs in [(6.0, 1.2), (6.0, 2.0), (6.0, 4.0)]:
    t0 = time.time()
    sea = build_sea_linear(Tp, Hs, 0)
    ds = []
    for r in range(3):
        h, n, nob = one_draw(sea, np.random.default_rng(900000 + 31 * r), net, head)
        ds.append(n - h)
    ds = np.array(ds)
    finite = bool(np.isfinite(ds).all())
    print(f"LINEAR Tp{Tp} Hs{Hs} steep {Hs/lam_p(Tp):.4f}: "
          f"delta {np.nanmean(ds):+.4f} sd {np.nanstd(ds, ddof=1):.4f} "
          f"sign {(ds > 0).sum()}/3 finite={finite} n_obs~{nob} "
          f"[{time.time()-t0:.0f}s]", flush=True)
