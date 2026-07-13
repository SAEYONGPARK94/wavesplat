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

from wavesim import mesh_adequacy                      # noqa: E402
from wavesim.config import load_config                 # noqa: E402
from wavesim.mesh_adequacy import (PTS_PER_LAMBDA_OK,  # noqa: E402
                                   mesh_spacing)
from wavesim.wave_generator import WaveComponent       # noqa: E402

OUT = 'results/direction3/e2_mesh_audit.json'


def main():
    cfg = load_config('configs/default.yaml')
    dx, dy = mesh_spacing(cfg)
    d = max(dx, dy)
    print(f'raycast mesh: dx={dx:.2f} m, dy={dy:.2f} m '
          f'(mesh {cfg.mesh.nx}x{cfg.mesh.ny}); '
          f'adequacy needs lambda >= {PTS_PER_LAMBDA_OK * d:.0f} m\n')

    # One criterion, shared with the generation guard: judge on the fraction of
    # variance the mesh cannot resolve, not on the peak alone.  (Before the
    # 2026-07-13 reconciliation this script judged on the peak while
    # mesh_adequacy judged on the shortest significant component, so the same
    # episode could be called both adequate and inadequate.)
    groups = defaultdict(list)
    for path in sorted(glob.glob('data/episode_*.h5')):
        with h5py.File(path, 'r') as h:
            c = h['config/wave_components'][:]        # [A, om, kx, ky, phi, Q]
            sea = h['config'].attrs['sea_state']
            ncomp = int(h['config'].attrs['n_components'])
        comps = [WaveComponent(A=r[0], omega=r[1], kx=r[2], ky=r[3], phi=r[4],
                               Q=r[5]) for r in c]
        groups[(str(sea), ncomp)].append(mesh_adequacy.check(comps, cfg))

    rows = []
    print(f'{"sea":<14} {"n":>3} {"lam_peak":>9} {"pk pts/lam":>10} '
          f'{"min pts/lam":>11} {"var<6dx":>8}  verdict')
    for (sea, ncomp), v in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        lp = float(np.mean([r['lambda_peak'] for r in v]))
        vu = float(np.mean([r['var_under_resolved'] for r in v]))
        pk = float(np.mean([r['pts_per_lambda_peak'] for r in v]))
        pm = float(np.mean([r['pts_per_lambda'] for r in v]))
        # Take the WORST episode's verdict, not the first one's.  Wavelengths are
        # seed-invariant today so every group is homogeneous, but reading v[0]
        # would silently hide a bad episode the day that stops being true.
        order = {mesh_adequacy.ADEQUATE: 0, mesh_adequacy.DEGRADED: 1,
                 mesh_adequacy.INADEQUATE: 2}
        vd = max((r['mesh_verdict'] for r in v), key=lambda x: order[x])
        mixed = len({r['mesh_verdict'] for r in v}) > 1
        rows.append(dict(sea=sea, n_components=ncomp, n_episodes=len(v),
                         lambda_peak=lp, pts_per_lambda_peak=pk,
                         pts_per_lambda_min=pm, var_under_resolved=vu,
                         verdict=vd, mixed_within_group=mixed))
        print(f'{sea + "_" + str(ncomp) + "comp":<14} {len(v):3d} {lp:8.1f}m '
              f'{pk:10.1f} {pm:11.1f} {100 * vu:7.0f}%  {vd}'
              f'{"  [MIXED]" if mixed else ""}')

    res = dict(date='2026-07-13',
               prereg='docs/E2_MESH_AUDIT_prereg_2026-07-13.md',
               mesh=dict(nx=cfg.mesh.nx, ny=cfg.mesh.ny, dx=dx, dy=dy),
               rule=dict(resolved_at=PTS_PER_LAMBDA_OK,
                         adequate_var_under=mesh_adequacy.VAR_UNDER_OK,
                         degraded_var_under=mesh_adequacy.VAR_UNDER_DEGRADED),
               n_episodes=sum(r['n_episodes'] for r in rows),
               groups=rows)
    json.dump(res, open(OUT, 'w'), indent=1)

    bad = [r for r in rows if r['verdict'] != mesh_adequacy.ADEQUATE]
    print(f'\n{len(bad)}/{len(rows)} sea-groups are not adequately rendered '
          f'by the shipped mesh (unified criterion: variance below 6*dx)')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
