"""
Alternative direction pilot — CRB-optimal active sensing (turn the limit into design).

Direction I gives the FIM; the field-CRB depends on WHERE we sample. If a smart
(D-optimal) subset of LiDAR returns gives a much lower CRB than a random subset of
the same size, then *scan/return selection* is a real lever — "where should I look
to learn the wave field fastest?". We greedily select K points to maximise the
Fisher information (max leverage Jᵀ F⁻¹ J per step) and compare the field-nRMSE CRB
to a random-K and a cross-range-spread heuristic.

Run: python3 scripts/d11_active_sensing.py
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import d1_crb as C1
from d6_integrated_eval import gather_W, W, TI, tag_to_file
from wavesplat.data_loader import EpisodeDataLoader

DT = C1.DT; DEV = C1.DEVICE


def theta_of(L):
    wc = np.array(L.wave_components)
    return torch.tensor(np.stack([wc[:, 0], wc[:, 2], wc[:, 3], wc[:, 4]], 1),
                        dtype=DT, device=DEV)


def field_crb_for(theta, P, idx, gx, gy, t_star, depth, Hs, sigma=0.02):
    x = torch.tensor(P[idx, 0], dtype=DT, device=DEV)
    y = torch.tensor(P[idx, 1], dtype=DT, device=DEV)
    t = torch.tensor(P[idx, 2], dtype=DT, device=DEV)
    F = C1.fisher(theta, x, y, t, depth, sigma)
    Gmat = C1.field_design(theta, gx, gy, t_star, depth)
    nr, _ = C1.field_nrmse_crb(F, Gmat, Hs)
    return nr


def greedy_dopt(theta, P, K, depth, sigma=0.02, ridge=1e-6):
    """Greedily pick K points maximising Fisher information (max Jᵀ F⁻¹ J each step)."""
    x = torch.tensor(P[:, 0], dtype=DT, device=DEV)
    y = torch.tensor(P[:, 1], dtype=DT, device=DEV)
    t = torch.tensor(P[:, 2], dtype=DT, device=DEV)
    J = C1.jacobian(theta, x, y, t, depth)          # (M, 4N)
    M, d = J.shape
    F = ridge * torch.eye(d, dtype=DT, device=DEV)
    chosen = []
    avail = list(range(M))
    for _ in range(K):
        Finv = torch.linalg.inv(F)
        # leverage of each available point
        Ja = J[avail]
        lev = torch.einsum('md,de,me->m', Ja, Finv, Ja)
        bi = int(torch.argmax(lev))
        m = avail.pop(bi)
        chosen.append(m)
        F = F + torch.outer(J[m], J[m]) / sigma**2
    return np.array(chosen)


def main():
    tags = ['BF4_5c_s0', 'BF4_15c_s0', 'BF4_30c_s0']
    K = 60; rng = np.random.default_rng(0)
    print(f"K={K} selected from the window pool. field-nRMSE CRB:")
    print(f"{'episode':12s} {'pool':>5} {'random':>8} {'xspread':>8} {'D-opt':>8} {'gain%':>6}")
    rows = []
    for tag in tags:
        L = EpisodeDataLoader('data/' + tag_to_file(tag))
        theta = theta_of(L); depth = L.depth; Hs = L.Hs_true
        gx = torch.tensor(L.gt_grid_x.numpy(), dtype=DT, device=DEV)
        gy = torch.tensor(L.gt_grid_y.numpy(), dtype=DT, device=DEV)
        P, z, frames = gather_W(L, TI, W)
        tstar = float(frames[TI + 1].t)
        Mpool = len(P)
        # random average over a few draws
        rand = np.mean([field_crb_for(theta, P, rng.choice(Mpool, K, replace=False),
                                       gx, gy, tstar, depth, Hs) for _ in range(5)])
        # cross-range spread heuristic: pick K points with extreme perpendicular coord
        ang = np.arctan2((theta[:,0]**2*theta[:,2]).sum().item(),
                         (theta[:,0]**2*theta[:,1]).sum().item())
        perp = -np.sin(ang)*P[:,0] + np.cos(ang)*P[:,1]
        xspread = field_crb_for(theta, P, np.argsort(np.abs(perp - perp.mean()))[-K:],
                                gx, gy, tstar, depth, Hs)
        dopt_idx = greedy_dopt(theta, P, K, depth)
        dopt = field_crb_for(theta, P, dopt_idx, gx, gy, tstar, depth, Hs)
        gain = 100 * (rand - dopt) / rand
        print(f"{tag:12s} {Mpool:5d} {rand:8.4f} {xspread:8.4f} {dopt:8.4f} {gain:6.1f}")
        rows.append(dict(tag=tag, pool=Mpool, random=float(rand), xspread=float(xspread),
                         dopt=float(dopt), gain_pct=float(gain)))
    json.dump(rows, open('results/direction3/active_sensing.json', 'w'), indent=2)
    print("\nD-opt << random ⇒ scan/return selection is a real lever (active sensing).")


if __name__ == '__main__':
    main()
