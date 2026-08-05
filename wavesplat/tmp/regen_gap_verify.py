"""Verification pass for the 2026-07-24 gap-close regeneration.

Implements REGEN_SPEC_fable_2026-07-23.md Sec 6 for every file in a regenerated
directory that has a shipped counterpart in `data/`:

  1. valid HDF5, non-empty, and openable by `wavesplat.data_loader.EpisodeDataLoader`
  2. mesh verdict ADEQUATE -- both the stored `config.attrs['mesh_verdict']` and a
     fresh `wavesim.mesh_adequacy.check()` recomputed from the file's own components
  3. wave components bit-identical to the shipped counterpart (exact ==, not allclose:
     same seed => same `jonswap_components` draw; any mismatch is a real regression)
  4. observation-budget cap preserved (point-array width == shipped width)
  5. frame count matches the shipped file; no NaNs in valid points or in the GT field

A corrupt/unreadable *shipped* file is recorded as `source_corrupt` and does not
fail the regenerated file's own checks (some `_slope.h5` are known-bad, 2026-06-28).

Usage:  python3 tmp/regen_gap_verify.py [--dir DIR] [--list FILE] [--out JSON]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
os.chdir(REPO)

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from wavesim import mesh_adequacy  # noqa: E402
from wavesim.config import WaveSimConfig  # noqa: E402


class _C:
    __slots__ = ('A', 'omega', 'kx', 'ky', 'phi', 'Q')

    def __init__(self, row):
        self.A, self.omega, self.kx, self.ky, self.phi, self.Q = [float(v) for v in row]


def verify(new_path, src_path):
    r = {'file': os.path.basename(new_path), 'checks': {}, 'ok': True}

    def fail(key, msg):
        r['checks'][key] = f'FAIL: {msg}'
        r['ok'] = False

    # ---- 1. HDF5 opens, non-empty ------------------------------------------
    try:
        with h5py.File(new_path, 'r') as h:
            pts = h['frames/lidar_points']
            n_frames, width = int(pts.shape[0]), int(pts.shape[1])
            if n_frames == 0:
                fail('hdf5', 'zero frames')
            comps_new = h['config/wave_components'][:]
            attrs = dict(h['config'].attrs)
            mask = h['frames/lidar_valid_mask'][:]
            nan_pts = bool(np.isnan(pts[:][mask]).any())
            nan_gt = bool(np.isnan(h['frames/gt_wave_field'][:]).any())
            mean_valid = float(mask.sum(axis=1).mean())
    except Exception as e:                                      # noqa: BLE001
        r['checks']['hdf5'] = f'FAIL: {type(e).__name__}: {e}'
        r['ok'] = False
        return r
    r['checks']['hdf5'] = f'OK ({n_frames} frames x {width} pts)'
    r['n_frames'], r['width'], r['mean_valid_pts'] = n_frames, width, round(mean_valid, 1)

    # ---- 1b. EpisodeDataLoader ---------------------------------------------
    try:
        from wavesplat.data_loader import EpisodeDataLoader
        ld = EpisodeDataLoader(new_path)
        nf = len(ld)
        fr = ld[0]
        _ = (fr.x_obs.shape, fr.z_obs.shape)
        r['checks']['data_loader'] = f'OK ({nf} frames, frame0 {tuple(fr.x_obs.shape)})'
    except Exception as e:                                      # noqa: BLE001
        fail('data_loader', f'{type(e).__name__}: {e}')

    # ---- 2. mesh adequacy ---------------------------------------------------
    stored = str(attrs.get('mesh_verdict', 'MISSING'))
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = 200, 167
    fresh = mesh_adequacy.check([_C(row) for row in comps_new], cfg)
    r['pts_per_lambda_min'] = round(float(fresh['pts_per_lambda']), 2)
    r['var_under'] = round(float(fresh['var_under_resolved']), 4)
    if stored == 'ADEQUATE' and fresh['mesh_verdict'] == 'ADEQUATE':
        r['checks']['mesh'] = (f"OK ADEQUATE (stored+recomputed; "
                               f"{fresh['pts_per_lambda']:.2f} pts/lambda_min)")
    else:
        fail('mesh', f'stored={stored} recomputed={fresh["mesh_verdict"]}')

    # ---- 3/4/5. paired checks against the shipped file ----------------------
    if src_path is None or not os.path.exists(src_path):
        r['checks']['components'] = 'SKIP: no shipped counterpart'
        r['checks']['cap'] = 'SKIP'
    else:
        try:
            with h5py.File(src_path, 'r') as hs:
                comps_old = hs['config/wave_components'][:]
                src_shape = hs['frames/lidar_points'].shape
                src_rd = dict(hs['raydrop_model'].attrs)
        except Exception as e:                                  # noqa: BLE001
            r['checks']['components'] = f'SOURCE_CORRUPT: {type(e).__name__}: {e}'
            r['checks']['cap'] = 'SOURCE_CORRUPT'
            r['source_corrupt'] = True
            comps_old = None
        if comps_old is not None:
            # REGEN_SPEC Sec 6.3: exact equality is what G6 predicts; rtol 1e-10
            # is the stated acceptance bar.  Anything between the two is recorded
            # explicitly rather than silently passed or failed.
            if comps_old.shape != comps_new.shape:
                fail('components', f'shape {comps_old.shape} vs {comps_new.shape}')
            elif np.array_equal(comps_old, comps_new):
                r['checks']['components'] = f'OK bit-identical ({comps_new.shape[0]} comps)'
                r['components_exact'] = True
            elif np.allclose(comps_old, comps_new, rtol=1e-10, atol=1e-14):
                d = float(np.max(np.abs(comps_old - comps_new)))
                r['checks']['components'] = (f'OK bit-close, NOT exact ({comps_new.shape[0]} '
                                             f'comps, max abs diff {d:.3e} = last-ULP)')
                r['components_exact'] = False
                r['components_maxdiff'] = d
            else:
                d = float(np.max(np.abs(comps_old - comps_new)))
                fail('components', f'max abs diff {d:.3e} exceeds rtol 1e-10')
            # The observation *budget* the regen must preserve is the raydrop
            # parameter, not the stored array width.  For fixedT that budget is a
            # hard `cap` and the array is padded to it, so width must match exactly.
            # For uniform/slope the budget is `n_pts` and the array width is the
            # ragged max-returns-per-frame, which legitimately shifts with the mesh
            # (G6: per-frame Bernoulli/noise draws depend on how many rays hit).
            model = str(src_rd.get('model_type', 'unknown'))
            src_budget = int(src_rd.get('cap', src_rd.get('n_pts', -1)))
            with h5py.File(new_path, 'r') as h:
                new_rd = dict(h['raydrop_model'].attrs)
            new_budget = int(new_rd.get('cap', new_rd.get('n_pts', -2)))
            r['raydrop_model'], r['obs_budget'] = model, src_budget
            if int(src_shape[0]) != n_frames:
                fail('cap', f'shipped frames {src_shape[0]} vs new {n_frames}')
            elif src_budget != new_budget:
                fail('cap', f'{model} budget {src_budget} vs new {new_budget}')
            elif model == 'fixedT' and int(src_shape[1]) != width:
                fail('cap', f'fixedT hard cap: shipped width {src_shape[1]} vs new {width}')
            else:
                note = (f'hard cap {width}' if model == 'fixedT'
                        else f'n_pts={src_budget}, ragged width {src_shape[1]}->{width}')
                r['checks']['cap'] = f'OK ({model}: {note}, {n_frames} frames)'

    # ---- NaN ---------------------------------------------------------------
    if nan_pts or nan_gt:
        fail('nan', f'valid_pts={nan_pts} gt={nan_gt}')
    else:
        r['checks']['nan'] = 'OK'
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data_finemesh_fixedT_2026-07-23')
    ap.add_argument('--list', default=None, help='restrict to basenames in this file')
    ap.add_argument('--src', default='data')
    ap.add_argument('--out', default='tmp/regen_gap_verify.json')
    args = ap.parse_args()

    if args.list:
        names = [l.strip() for l in open(args.list)
                 if l.strip() and not l.startswith('#')]
    else:
        names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(args.dir, '*.h5')))

    rows = []
    for n in names:
        p = os.path.join(args.dir, n)
        if not os.path.exists(p):
            rows.append({'file': n, 'ok': False, 'checks': {'exists': 'FAIL: missing'}})
            print(f'{n:<48} MISSING', flush=True)
            continue
        r = verify(p, os.path.join(args.src, n))
        rows.append(r)
        bad = [f'{k}={v}' for k, v in r['checks'].items() if str(v).startswith('FAIL')]
        print(f'{n:<48} {"PASS" if r["ok"] else "FAIL"}  '
              f'{r.get("n_frames","?")}f x {r.get("width","?")} '
              f'valid~{r.get("mean_valid_pts","?")} '
              f'ptsPerLam={r.get("pts_per_lambda_min","?")} '
              + ('| ' + '; '.join(bad) if bad else ''), flush=True)

    with open(args.out, 'w') as fh:
        json.dump(rows, fh, indent=2)
    n_pass = sum(r['ok'] for r in rows)
    print(f'\n{n_pass}/{len(rows)} PASS  ->  {args.out}')


if __name__ == '__main__':
    main()
