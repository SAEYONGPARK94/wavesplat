"""E-2: audit every stored episode for raycast-mesh adequacy.

Prereg: docs/E2_MESH_AUDIT_prereg_2026-07-13.md.

E-1 showed the BF2 1-comp "information limit" was the mesh failing to represent
an 8.2 m wave.  The same mesh rendered all 195 stored episodes, so the question
is how far the disease spreads.  This reads only the component tables in the
episode files -- no simulation -- and reports points-per-wavelength per sea.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.config import load_config                      # noqa: E402
from wavesim.mesh_adequacy import (PTS_PER_LAMBDA_NYQUIST,  # noqa: E402
                                   PTS_PER_LAMBDA_OK, mesh_spacing)

OUT = 'results/direction3/e2_mesh_audit.json'


def verdict(pts):
    if pts >= PTS_PER_LAMBDA_OK:
        return 'OK'
    if pts >= PTS_PER_LAMBDA_NYQUIST:
        return 'BORDERLINE'
    return 'UNDER-RENDERED'


def main():
    cfg = load_config('configs/default.yaml')
    dx, dy = mesh_spacing(cfg)
    d = max(dx, dy)
    print(f'raycast mesh: dx={dx:.2f} m, dy={dy:.2f} m '
          f'(mesh {cfg.mesh.nx}x{cfg.mesh.ny}); '
          f'adequacy needs lambda >= {PTS_PER_LAMBDA_OK * d:.0f} m\n')

    groups = defaultdict(list)
    for path in sorted(glob.glob('data/episode_*.h5')):
        with h5py.File(path, 'r') as h:
            c = h['config/wave_components'][:]        # [A, om, kx, ky, phi, Q]
            sea = h['config'].attrs['sea_state']
            ncomp = int(h['config'].attrs['n_components'])
        A = c[:, 0]
        lam = 2 * np.pi / np.maximum(np.hypot(c[:, 2], c[:, 3]), 1e-12)
        var = A ** 2
        groups[(str(sea), ncomp)].append(dict(
            lam_peak=float(lam[int(np.argmax(A))]),
            lam_min=float(lam[var >= 0.01 * var.sum()].min()),
            var_under=float(var[lam < PTS_PER_LAMBDA_OK * d].sum() / var.sum()),
        ))

    rows = []
    print(f'{"sea":<14} {"n":>3} {"lam_peak":>9} {"pts/lam":>8} '
          f'{"var<6dx":>8}  verdict')
    for (sea, ncomp), v in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        lp = float(np.mean([r['lam_peak'] for r in v]))
        vu = float(np.mean([r['var_under'] for r in v]))
        pts = lp / d
        row = dict(sea=sea, n_components=ncomp, n_episodes=len(v),
                   lambda_peak=lp, pts_per_lambda_peak=pts,
                   var_under_resolved=vu, verdict=verdict(pts))
        rows.append(row)
        print(f'{sea + "_" + str(ncomp) + "comp":<14} {len(v):3d} {lp:8.1f}m '
              f'{pts:8.1f} {100 * vu:7.0f}%  {row["verdict"]}')

    res = dict(date='2026-07-13',
               prereg='docs/E2_MESH_AUDIT_prereg_2026-07-13.md',
               mesh=dict(nx=cfg.mesh.nx, ny=cfg.mesh.ny, dx=dx, dy=dy),
               rule=dict(adequate=PTS_PER_LAMBDA_OK,
                         nyquist=PTS_PER_LAMBDA_NYQUIST),
               n_episodes=sum(r['n_episodes'] for r in rows),
               groups=rows)
    json.dump(res, open(OUT, 'w'), indent=1)

    bad = [r for r in rows if r['verdict'] != 'OK']
    print(f'\n{len(bad)}/{len(rows)} sea-groups are not adequately rendered '
          f'by the shipped mesh')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
