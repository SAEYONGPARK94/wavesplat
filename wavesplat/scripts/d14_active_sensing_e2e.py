"""
Active sensing end-to-end (step 2) — does the CRB reduction translate to ACHIEVED
reconstruction error?

The d11 pilot showed D-optimal return selection cuts the field-CRB by up to 69%.
Here we run the *efficient* estimator (component-LS: fit amplitudes/phases given the
wavenumbers, then render) on the SAME subsets, and check the achieved field nRMSE
tracks the CRB. We compare:
  random-K | cross-range-spread-K (deployable heuristic) | D-optimal-K (oracle UB).

Run: python3 scripts/d14_active_sensing_e2e.py
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from d6_integrated_eval import gather_W, W, TI, tag_to_file
from d11_active_sensing import theta_of, field_crb_for, greedy_dopt
import d1_crb as C1


def achieved_nrmse(kx, ky, P, idx, target, depth):
    """Efficient estimator: fit A,phi (given true kx,ky) on the subset, render."""
    obs = np.column_stack([P[idx, 0], P[idx, 1], P[idx, 2], target['z'][idx]]).astype(np.float32)
    A, phi = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
    pred = render_wave_field(kx, ky, A, phi, depth, target['gx'], target['gy'], target['t'])
    return float(np.sqrt(np.mean((pred - target['gt'])**2)) / max(target['Hs'], 1e-6))


def main():
    tags = ['BF3_5c_s0', 'BF4_5c_s0', 'BF4_15c_s0']
    K = 60; rng = np.random.default_rng(0)
    print(f"K={K}.  CRB (and ACHIEVED nRMSE) by sampling strategy:")
    print(f"{'episode':12s} | {'random':>14} | {'xrange':>14} | {'D-opt':>14}")
    print(f"{'':12s} | {'CRB / nRMSE':>14} | {'CRB / nRMSE':>14} | {'CRB / nRMSE':>14}")
    rows = []
    for tag in tags:
        L = EpisodeDataLoader('data/' + tag_to_file(tag))
        theta = theta_of(L); depth = L.depth; Hs = L.Hs_true
        wc = np.array(L.wave_components); kx, ky = wc[:, 2], wc[:, 3]
        gxt = torch.tensor(L.gt_grid_x.numpy(), dtype=C1.DT, device=C1.DEVICE)
        gyt = torch.tensor(L.gt_grid_y.numpy(), dtype=C1.DT, device=C1.DEVICE)
        P, z, frames = gather_W(L, TI, W)
        tgt = frames[TI + 1]
        target = {'gx': L.gt_grid_x.numpy(), 'gy': L.gt_grid_y.numpy(),
                  't': float(tgt.t), 'gt': tgt.gt_field.numpy(), 'Hs': Hs, 'z': z}
        tstar = float(tgt.t); Mpool = len(P)
        # strategies
        ridx = [rng.choice(Mpool, K, replace=False) for _ in range(5)]
        ang = np.arctan2((theta[:,0]**2*theta[:,2]).sum().item(),
                         (theta[:,0]**2*theta[:,1]).sum().item())
        perp = -np.sin(ang)*P[:,0] + np.cos(ang)*P[:,1]
        xidx = np.argsort(np.abs(perp - perp.mean()))[-K:]
        didx = greedy_dopt(theta, P, K, depth)
        out = {}
        for name, idxs in [('random', ridx), ('xrange', [xidx]), ('dopt', [didx])]:
            crb = np.mean([field_crb_for(theta, P, ix, gxt, gyt, tstar, depth, Hs) for ix in idxs])
            ach = np.mean([achieved_nrmse(kx, ky, P, ix, target, depth) for ix in idxs])
            out[name] = (float(crb), float(ach))
        print(f"{tag:12s} | {out['random'][0]:6.3f}/{out['random'][1]:6.3f} | "
              f"{out['xrange'][0]:6.3f}/{out['xrange'][1]:6.3f} | "
              f"{out['dopt'][0]:6.3f}/{out['dopt'][1]:6.3f}")
        rows.append(dict(tag=tag, **{k: dict(crb=v[0], nrmse=v[1]) for k, v in out.items()}))
    json.dump(rows, open('results/direction3/active_sensing_e2e.json', 'w'), indent=2)
    print("\nIf D-opt nRMSE < random nRMSE (tracking CRB), the active-sensing lever is")
    print("confirmed end-to-end: better sampling -> better reconstruction, not just lower CRB.")


if __name__ == '__main__':
    main()
