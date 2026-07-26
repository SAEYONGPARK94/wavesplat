"""MECHANISM diagnostic (not a gate).  Does the new checkpoint put atoms in the RIGHT PLACE
on the sea that broke the old one, or merely emit fewer of them?

Reproduces the statistic tmp/learned_only_2026-07-26/design_diag.py measured on the old net:
for every 1-comp held-out episode, the distance (in kp units) from the TRUE component to the
nearest SELECTED proposal atom, next to the same distance for the beamform dominant peak.
Old net on the catastrophic BF2_1comp slope episodes: nearest atom 0.742 kp, beamform peak
0.058 kp -- i.e. the conditioning input knew where the swell was and the proposal did not.

  python3 tmp/mixed_retrain_2026-07-26/mech_diag.py --ckpt <new> --out <json>
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import h5py
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

FINE = 'data_heldout40_2026-07-26/'
DSEEDS = (0, 1, 2, 3, 4)


def true_components(path):
    with h5py.File(path, 'r') as f:
        wc = f['config/wave_components'][:]      # (n, 6) = A, omega, kx, ky, phi, Q
    return wc[:, 2], wc[:, 3], wc[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/direction3/beamform_cond_diffusion_mixed_v1.pt')
    ap.add_argument('--out', default='tmp/mixed_retrain_2026-07-26/mech_diag.json')
    a = ap.parse_args()

    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.optimizer import learned_init as li

    eps = sorted(f for f in os.listdir(FINE) if '_1comp_' in f and f.endswith('.h5'))
    print(f"# ckpt {a.ckpt}\n# {len(eps)} one-comp episodes x {len(DSEEDS)} diffusion seeds",
          flush=True)
    rows = []
    for ep in eps:
        L = EpisodeDataLoader(FINE + ep)
        fr = L.get_init_window(2.0)
        kx_t, ky_t, A_t = true_components(FINE + ep)
        i = int(np.argmax(A_t))
        tx, ty = float(kx_t[i]), float(ky_t[i])
        kp = float(np.hypot(tx, ty))
        locs, kp_bf = li._beamform_peak_locations(fr, L.depth)
        d_bf = (float(np.min(np.hypot(locs[:, 0] - tx, locs[:, 1] - ty))) / kp
                if len(locs) else float('nan'))
        d_at, nsel = [], []
        for ds in DSEEDS:
            rkx, rky, ren = li.propose_atoms(fr, L.depth, ckpt=a.ckpt, nsamp=4,
                                             diffusion_seed=ds)
            sk, sy, se = li.pool_and_select(rkx, rky, ren, max_n=30)
            nsel.append(len(sk))
            d_at.append(float(np.min(np.hypot(sk - tx, sy - ty))) / kp if len(sk)
                        else float('nan'))
        r = dict(episode=ep, kp_true=kp, n_selected=nsel, n_sel_med=float(np.median(nsel)),
                 d_atom_kp=d_at, d_atom_kp_med=float(np.median(d_at)),
                 d_beamform_kp=d_bf)
        rows.append(r)
        print(f"  {ep:<44s} n_sel {r['n_sel_med']:4.1f}  nearest-atom {r['d_atom_kp_med']:.3f} kp"
              f"  beamform-peak {d_bf:.3f} kp", flush=True)

    bf2 = [r for r in rows if 'BF2_1comp' in r['episode']]
    summ = dict(all_median_d_atom=float(np.median([r['d_atom_kp_med'] for r in rows])),
                bf2_median_d_atom=float(np.median([r['d_atom_kp_med'] for r in bf2])),
                bf2_median_d_beamform=float(np.median([r['d_beamform_kp'] for r in bf2])),
                bf2_median_nsel=float(np.median([r['n_sel_med'] for r in bf2])),
                old_net_bf2_slope_d_atom=0.742, old_net_bf2_slope_d_beamform=0.058)
    print("\n=== summary ===")
    for k, v in summ.items():
        print(f"  {k:32s} {v:.4f}")
    json.dump(dict(ckpt=a.ckpt, rows=rows, summary=summ), open(a.out, 'w'), indent=1)
    print(f"wrote {a.out}")


if __name__ == '__main__':
    main()
