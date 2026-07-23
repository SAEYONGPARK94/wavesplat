"""theory_audit_component_crb_check.py -- cheap, read-only sanity check for the
THEORY_AUDIT (2026-07-23). Question: on the exact episode/seed pair that produced
the bimodal-variance finding (BF3_2comp_seed0_uniform, true weak component A=0.037,
0.08% of energy, never recovered by any of 8 seeds per VARIANCE_TRACE_fable), does
the ORACLE (true theta, true N=2) component-level CRB *already* flag the weak
component as effectively unrecoverable at the observation budget the estimator
actually had at the moment of the N=8->15 growth decision (t_obs ~ 0.4-0.8s, a few
hundred points; and also at full 19s tracking-window end, ~15000 points)?

If sigma_CRB(w_weak) is itself large relative to the true amplitude 0.037 even
with N known and phase known-up-to-optimization (oracle setting), that means the
CRB *correctly* signals non-identifiability at the parameter level -- the failure
mode is not "CRB is wrong", it is "CRB's information is thrown away by a model-order
selection layer the CRB does not model" (this script only checks the parameter-level
claim; the model-order claim is argued from code inspection + the existing bimodal-
variance results, not from this script).

Uses scripts/d1_crb.py unmodified (import only, no edits). Read-only. Cheap: an
FIM inversion for N=2, M<=15000 -- no training, no GPU load of consequence.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'scripts')
import torch
from scripts.d1_crb import (DEVICE, DT, load_ep, collect_obs, fisher,
                             per_component_crb)

SIGMA = 0.02
PATH = 'data/episode_BF3_2comp_seed0_uniform.h5'

def main():
    ep = load_ep(PATH)
    theta = ep['theta']  # true (w, kx, ky, phi) x 2, oracle
    N = theta.shape[0]
    w_true = theta[:, 0].cpu().numpy()
    print(f"True amplitudes: {w_true} (component energy fractions: "
          f"{(w_true**2 / (w_true**2).sum())})")

    for t_obs in [0.1, 0.4, 0.8, 1.0, 3.0, 19.0]:
        x, y, t = collect_obs(ep, t_obs)
        if x.numel() < 4 * N:
            print(f"t_obs={t_obs:5.1f}s  M={x.numel():5d}  under-determined (M<4N), skip")
            continue
        F = fisher(theta, x, y, t, ep['depth'], SIGMA)
        lam = 1e-10 * torch.trace(F) / F.shape[0]
        Finv = torch.linalg.inv(F + lam * torch.eye(F.shape[0], dtype=DT, device=F.device))
        sw, skx, sky, sphi = per_component_crb(Finv, N)
        sw = sw.cpu().numpy(); skx = skx.cpu().numpy(); sky = sky.cpu().numpy()
        for i in range(N):
            snr_i = w_true[i] / sw[i] if sw[i] > 0 else float('inf')
            print(f"  t_obs={t_obs:5.1f}s M={x.numel():6d}  comp{i} (w_true={w_true[i]:.4f})  "
                  f"sigma_CRB(w)={sw[i]:.4f}  sigma_CRB(kx)={skx[i]:.4f}  sigma_CRB(ky)={sky[i]:.4f}  "
                  f"'CRB-SNR' w_true/sigma_w={snr_i:.2f}")

if __name__ == '__main__':
    main()
