"""e29 diagnostic: does the stage-2 residual MMSE direction f2 carry ANY
signal about the true residual (gt - f1) at held-out pixels? If corr ~ 0,
the boost mechanism is dead regardless of training budget."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')
import numpy as np
import torch
from e29_gennet_C import (build_sea_c, incumbent_field, make_stage2_inputs,
                          boost_f2, NGRID, DEVICE, corr, N_OBS)

print("sea | std(e)/sig0 | corr(f2, gt-f1)@query | best-beta gain(corr)")
for si in range(6):
    S = build_sea_c('spread', si)
    P, z, Q, gt = S['P'], S['z'], S['Pq'], S['gt_u']
    kp, s0 = S['kp'], S['sigma0']
    f1 = incumbent_field(P, z, np.vstack([P, Q]), kp, s0)
    e = (z - f1[:N_OBS]).astype(np.float32)
    _, pw_res, _ = make_stage2_inputs(P, z, e, kp, s0)
    r2 = torch.zeros(NGRID, NGRID, device=DEVICE)
    ls2 = torch.tensor(float(np.log(3.0)), device=DEVICE)
    with torch.no_grad():
        f2 = boost_f2(r2, ls2,
                      torch.tensor(P, dtype=torch.float32, device=DEVICE),
                      torch.tensor(e, device=DEVICE),
                      torch.tensor(Q, dtype=torch.float32, device=DEVICE),
                      kp, s0, torch.tensor(pw_res, device=DEVICE)).cpu().numpy()
    resid_true = gt - f1[N_OBS:]
    c_dir = corr(f2, resid_true)
    # best achievable corr gain from f = f1 + beta*f2 over beta grid
    base = corr(f1[N_OBS:], gt)
    best = base, 0.0
    for beta in np.linspace(-1.5, 1.5, 61):
        c = corr(f1[N_OBS:] + beta * f2, gt)
        if c > best[0]:
            best = (c, beta)
    print(f"  {si} | {np.std(e)/s0:5.2f} | {c_dir:+.3f} | "
          f"{best[0]-base:+.4f} @ beta {best[1]:+.2f}")
