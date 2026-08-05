"""Mesh-adequacy census for the families the 2026-07-23 census never covered:
BF4_20comp (all raydrop models), BF6_*, BF7_*.

Reads `config/wave_components` straight out of the shipped `.h5` files and runs
the repo's own `wavesim.mesh_adequacy.check()` against them -- same procedure as
`tmp/regen_mesh_calc.py` / `regen_mesh_search.py`, no hand re-derivation.

Reports the verdict at the shipped mesh (40x30) and at the regeneration mesh
(200x167).
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from wavesim import mesh_adequacy  # noqa: E402
from wavesim.config import WaveSimConfig  # noqa: E402


class _C:
    """Duck-typed component with the .A/.kx/.ky attributes check() wants."""

    __slots__ = ('A', 'omega', 'kx', 'ky', 'phi', 'Q')

    def __init__(self, row):
        self.A, self.omega, self.kx, self.ky, self.phi, self.Q = [float(v) for v in row]


def components_of(path):
    with h5py.File(path, 'r') as h:
        return [_C(r) for r in h['config/wave_components'][:]]


def main():
    pats = ['data/episode_BF4_20comp_*.h5', 'data/episode_BF6_*.h5', 'data/episode_BF7_*.h5']
    files = sorted({f for p in pats for f in glob.glob(p)})

    cfg_ship = WaveSimConfig()
    cfg_fine = WaveSimConfig()
    cfg_fine.mesh.nx, cfg_fine.mesh.ny = 200, 167

    rows = []
    for f in files:
        rec = {'file': os.path.basename(f)}
        try:
            comps = components_of(f)
        except Exception as e:                                  # noqa: BLE001
            rec['error'] = f'{type(e).__name__}: {e}'
            rows.append(rec)
            print(f'{rec["file"]:<45} CORRUPT/UNREADABLE: {e}', flush=True)
            continue
        s = mesh_adequacy.check(comps, cfg_ship)
        fine = mesh_adequacy.check(comps, cfg_fine)
        rec.update(n_components=len(comps),
                   lambda_peak=s['lambda_peak'], lambda_min=s['lambda_min'],
                   shipped=dict(dx=s['mesh_dx'], dy=s['mesh_dy'],
                                pts_per_lambda_peak=s['pts_per_lambda_peak'],
                                pts_per_lambda_min=s['pts_per_lambda'],
                                var_under=s['var_under_resolved'],
                                verdict=s['mesh_verdict']),
                   fine200x167=dict(dx=fine['mesh_dx'], dy=fine['mesh_dy'],
                                    pts_per_lambda_peak=fine['pts_per_lambda_peak'],
                                    pts_per_lambda_min=fine['pts_per_lambda'],
                                    var_under=fine['var_under_resolved'],
                                    verdict=fine['mesh_verdict']))
        rows.append(rec)
        print(f'{rec["file"]:<45} lam_peak={s["lambda_peak"]:7.2f} lam_min={s["lambda_min"]:7.2f} '
              f'| 40x30: {s["pts_per_lambda_peak"]:6.2f}/{s["pts_per_lambda"]:5.2f} pts/lam '
              f'var_under={100*s["var_under_resolved"]:5.1f}% {s["mesh_verdict"]:<11} '
              f'| 200x167: {fine["pts_per_lambda_peak"]:7.2f}/{fine["pts_per_lambda"]:6.2f} '
              f'var_under={100*fine["var_under_resolved"]:5.1f}% {fine["mesh_verdict"]}',
              flush=True)

    out = 'tmp/regen_gap_census.json'
    with open(out, 'w') as fh:
        json.dump(rows, fh, indent=2)
    print(f'\nwrote {out} ({len(rows)} files)')

    ok = [r for r in rows if 'error' not in r]
    for tier in ('ADEQUATE', 'DEGRADED', 'INADEQUATE'):
        n_s = sum(r['shipped']['verdict'] == tier for r in ok)
        n_f = sum(r['fine200x167']['verdict'] == tier for r in ok)
        print(f'  {tier:<11} shipped {n_s:>3} / 200x167 {n_f:>3}')


if __name__ == '__main__':
    main()
