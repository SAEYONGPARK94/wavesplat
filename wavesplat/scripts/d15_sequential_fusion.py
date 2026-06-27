"""
Sequential/temporal fusion (step 3) — does accumulating time break the wall on
complex seas?

Single-window estimation fails on complex multidirectional seas (Dir-I: T_wall ≈
4 s for 15c, ≈10 s for 30c). Here we grow the observation window over a track
[t*−T_obs, t*] and reconstruct the field at t*, measuring both the field-CRB and
the achieved error of the efficient (component-LS) estimator vs T_obs. If both fall
sharply, sequential fusion is the lever that reconstructs complex seas — quantifying
the value of assimilating a track rather than one window.

Run: python3 scripts/d15_sequential_fusion.py
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from d11_active_sensing import theta_of, field_crb_for
from d14_active_sensing_e2e import achieved_nrmse
from d6_integrated_eval import tag_to_file
import d1_crb as C1

DT_FRAME = 0.1


def gather_frames(L, i_star, n_back):
    P, z = [], []
    for i in range(max(0, i_star - n_back), i_star + 1):
        f = L[i]
        P.append(np.column_stack([f.x_obs.numpy(), f.y_obs.numpy(),
                                   np.full(len(f.x_obs), float(f.t))]))
        z.append(f.z_obs.numpy())
    return np.vstack(P), np.concatenate(z)


def main():
    tags = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_30c_s1']
    i_star = 250                       # target frame (25 s)
    T_list = [1.0, 3.0, 6.0, 10.0, 15.0]
    print(f"target frame t*={i_star*DT_FRAME:.0f}s.  field-CRB / achieved nRMSE vs window:")
    header = "  ".join(f"T={t:.0f}s" for t in T_list)
    print(f"{'episode':12s} {'metric':>8}  {header}")
    rows = []
    for tag in tags:
        L = EpisodeDataLoader('data/' + tag_to_file(tag))
        theta = theta_of(L); depth = L.depth; Hs = L.Hs_true
        wc = np.array(L.wave_components); kx, ky = wc[:, 2], wc[:, 3]
        gxt = torch.tensor(L.gt_grid_x.numpy(), dtype=C1.DT, device=C1.DEVICE)
        gyt = torch.tensor(L.gt_grid_y.numpy(), dtype=C1.DT, device=C1.DEVICE)
        fstar = L[i_star]
        target = {'gx': L.gt_grid_x.numpy(), 'gy': L.gt_grid_y.numpy(),
                  't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs, 'z': None}
        crbs, achs, Ms = [], [], []
        for T in T_list:
            P, z = gather_frames(L, i_star, int(round(T / DT_FRAME)))
            target['z'] = z
            idx = np.arange(len(z))
            crbs.append(field_crb_for(theta, P, idx, gxt, gyt, float(fstar.t), depth, Hs))
            achs.append(achieved_nrmse(kx, ky, P, idx, target, depth))
            Ms.append(len(z))
        print(f"{tag:12s} {'CRB':>8}  " + "  ".join(f"{c:5.3f}" for c in crbs))
        print(f"{tag:12s} {'nRMSE':>8}  " + "  ".join(f"{a:5.3f}" for a in achs))
        print(f"{tag:12s} {'Mpts':>8}  " + "  ".join(f"{m:5d}" for m in Ms))
        rows.append(dict(tag=tag, T=T_list, crb=crbs, nrmse=achs, M=Ms))
    json.dump(rows, open('results/direction3/sequential_fusion.json', 'w'), indent=2)

    # figure
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cols = {'BF4_15c_s0': 'tab:orange', 'BF4_30c_s0': 'tab:red', 'BF5_30c_s1': 'tab:purple'}
    for r in rows:
        ax.plot(r['T'], r['nrmse'], 'o-', color=cols[r['tag']], label=r['tag'].replace('_s0','').replace('_s1',''))
    ax.axhline(0.205, ls='--', c='gray', label='ST-DF-GP overall (0.205)')
    ax.axhline(0.06, ls=':', c='k', label='exact')
    ax.set_xlabel('observation window / fusion length $T_{obs}$ [s]')
    ax.set_ylabel('achieved field nRMSE (efficient estimator)')
    ax.set_title('Step 3 — sequential time fusion breaks the wall on complex seas\n'
                 '(accumulating a track, not a single window)')
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig('results/direction3/fig_sequential_fusion.png', dpi=140)
    print('\nsaved sequential_fusion.json + fig_sequential_fusion.png')


if __name__ == '__main__':
    main()
