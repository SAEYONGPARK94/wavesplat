"""Pre-registration input: true |k| distribution + nearest-neighbour spacing of the
P1 12-episode set, on data_finemesh_2026-07-23. No estimation is run."""
import os, sys, json
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
sys.path.insert(0, REPO)

EPISODES = [
    'episode_BF2_1comp_seed0_uniform.h5',
    'episode_BF3_1comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_slope.h5',
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]

G = 9.81
rows = []
for ep in EPISODES:
    p = os.path.join(DATA, ep)
    with h5py.File(p, 'r') as f:
        comp = np.asarray(f['config/wave_components'])
        depth = float(np.asarray(f['config/depth'])) if 'config/depth' in f else float('nan')
    A, om, kx, ky = comp[:, 0], comp[:, 1], comp[:, 2], comp[:, 3]
    kmag = np.hypot(kx, ky)
    # dispersion cross-check to confirm column layout [A, om, kx, ky, phi, Q]
    om_pred = np.sqrt(G * kmag * np.tanh(kmag * depth))
    disp_err = float(np.max(np.abs(om_pred - om) / np.maximum(om, 1e-9)))
    N = len(A)
    if N > 1:
        D = np.hypot(kx[:, None] - kx[None, :], ky[:, None] - ky[None, :])
        np.fill_diagonal(D, np.inf)
        nn = D.min(axis=1)
    else:
        nn = np.array([np.inf])
    out_08 = int(np.sum(kmag > 0.8))
    e_out = float(np.sum(A[kmag > 0.8] ** 2) / max(np.sum(A ** 2), 1e-12))
    rows.append(dict(
        episode=ep, N=int(N), depth=depth, disp_rel_err_max=disp_err,
        kmag_min=float(kmag.min()), kmag_max=float(kmag.max()),
        kx_absmax=float(np.abs(kx).max()), ky_absmax=float(np.abs(ky).max()),
        nn_median=float(np.median(nn)) if N > 1 else None,
        nn_min=float(nn.min()) if N > 1 else None,
        n_lt_015=int(np.sum(nn < 0.15)), n_lt_010=int(np.sum(nn < 0.100)),
        n_lt_002=int(np.sum(nn < 0.020)),
        n_outside_08=out_08, energy_outside_08=e_out,
    ))

tot = {k: sum(r[k] for r in rows) for k in ('N', 'n_lt_015', 'n_lt_010', 'n_lt_002', 'n_outside_08')}
print(json.dumps({'rows': rows, 'totals': tot}, indent=1))
with open(os.path.join(HERE, 'survey_k.json'), 'w') as fh:
    json.dump({'rows': rows, 'totals': tot}, fh, indent=1)
