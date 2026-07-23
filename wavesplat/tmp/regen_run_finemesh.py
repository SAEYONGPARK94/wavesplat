"""
regen_run_finemesh.py -- Primary mesh-regeneration driver (REGEN_SPEC Step 1/2).

Regenerates the 100 paired BF2-5 x {uniform,slope} episodes (10 scenarios x 5
seeds -- the exact grid in configs/default.yaml's `dataset.scenarios`) at a
Nyquist-adequate raycast mesh, using the GPU raycast path (wavesim/lidar_gpu.py)
-- required for tractable runtime; see REGEN_SPEC_fable_2026-07-23.md Sec 4.

Why not `python generate_dataset.py`: that driver's worker (_run_one in
generate_dataset.py) calls run_episode() WITHOUT use_gpu_raycast, so it
silently falls back to the ~14x slower CPU path (13.9 s/frame per
wavesim/lidar_gpu.py's own docstring at 180x150). This script is the fix.

Resumable: writes a JSON manifest (<output_dir>/_regen_manifest.json) after
every episode; re-invoking the same command skips anything already marked
'ok'. Safe to interrupt (Ctrl-C, session loss, etc.) at any point.

Two modes:
  1. Primary paired-replacement set (default): same (sea_state, n_components,
     seed, raydrop) grid as the shipped data/, at the new mesh resolution.
     Wave components will match the shipped files bit-for-bit for the same
     seed (see REGEN_SPEC Sec 1, finding G6) -- verify this, don't assume it.
  2. --peak_locked companion set: the three 1-component scenarios (BF2/BF3/
     BF4, n_components=1) re-generated with the component placed AT the
     spectral peak (Tp) instead of the default flank (1.75*omega_p). This is
     a SEPARATE, additional set (filename tag `_pk`, per pipeline.py) -- NOT
     a replacement for the primary set. Isolates defect (d) from defect (a).
     See REGEN_SPEC Sec 3, confound ruling on (d).

Usage:
  # Full 100-episode primary regen at the recommended margin-safe mesh
  python3 tmp/regen_run_finemesh.py --output_dir data_finemesh_2026-07-23

  # Minimal-just-clears-ADEQUATE mesh instead (thinner margin, ~20% cheaper)
  python3 tmp/regen_run_finemesh.py --nx 180 --ny 150 --output_dir data_finemesh_180x150_2026-07-23

  # Dry run: show the job list and exit
  python3 tmp/regen_run_finemesh.py --output_dir data_finemesh_2026-07-23 --dry-run

  # Subset for smoke-testing this script itself
  python3 tmp/regen_run_finemesh.py --only BF2 --output_dir tmp/regen_smoketest

  # Peak-locked 1-comp companion set (run AFTER the primary set; separate dir)
  python3 tmp/regen_run_finemesh.py --peak_locked --output_dir data_finemesh_2026-07-23_pk
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np

from wavesim.config import WaveSimConfig
from wavesim.pipeline import run_episode

# Mirrors configs/default.yaml `dataset.scenarios` (S01-S10). Do not add/remove
# entries without updating REGEN_SPEC's scope section -- this list IS the scope.
SCENARIOS = [
    ('BF2', 1), ('BF3', 1), ('BF3', 2), ('BF3', 5),
    ('BF4', 1), ('BF4', 5), ('BF4', 15), ('BF4', 30),
    ('BF5', 15), ('BF5', 30),
]
SEEDS = [0, 1, 2, 3, 4]
RAYDROP_MODELS = ['uniform', 'slope']  # fixedT/dir180/BF6-7 are out of scope -- see REGEN_SPEC Sec 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=200,
                     help='raycast mesh nx (default 200 = margin-safe; 180 is the bare-minimum ADEQUATE floor)')
    ap.add_argument('--ny', type=int, default=167,
                     help='raycast mesh ny (default 167; pair with --nx 180 --ny 150 for the thinner-margin option)')
    ap.add_argument('--output_dir', required=True,
                     help='MUST NOT be data/ or any path under it -- nothing may overwrite the shipped set')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default=None, help='comma-separated BF filter, e.g. BF2,BF3')
    ap.add_argument('--n_pts', type=int, default=50)
    ap.add_argument('--duration', type=float, default=30.0)
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--depth', type=float, default=50.0)
    ap.add_argument('--peak_locked', action='store_true',
                     help='companion set: 1-comp scenarios only, component at the spectral peak')
    args = ap.parse_args()

    out_abs = os.path.abspath(args.output_dir)
    data_abs = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
    if out_abs == data_abs or out_abs.startswith(data_abs + os.sep):
        raise SystemExit(f"REFUSING to write into {out_abs} -- this would overwrite the shipped dataset.")

    only = set(args.only.split(',')) if args.only else None
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, '_regen_manifest.json')
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = args.nx, args.ny

    scenarios = SCENARIOS
    if args.peak_locked:
        scenarios = [(ss, nc) for (ss, nc) in SCENARIOS if nc == 1]
        if not scenarios:
            raise SystemExit("no 1-component scenarios found -- nothing to peak-lock")

    jobs = []
    for sea_state, ncomp in scenarios:
        if only and sea_state not in only:
            continue
        for seed in SEEDS:
            for raydrop in RAYDROP_MODELS:
                jobs.append((sea_state, ncomp, seed, raydrop))

    dx = 240.0 / (args.nx - 1)
    dy = 200.0 / (args.ny - 1)
    print(f"Mesh: {args.nx}x{args.ny} (dx={dx:.3f}m dy={dy:.3f}m, max={max(dx,dy):.3f}m; "
          f"ADEQUATE requires <=1.3595m for the worst in-scope case, BF2 1-comp)")
    print(f"Mode: {'PEAK-LOCKED COMPANION' if args.peak_locked else 'PRIMARY paired-replacement'}")
    print(f"Jobs: {len(jobs)}  output_dir={args.output_dir}")
    if args.dry_run:
        for j in jobs:
            print(' ', j)
        return

    t_start = time.time()
    for i, (sea_state, ncomp, seed, raydrop) in enumerate(jobs):
        ptag = '_pk' if args.peak_locked else ''
        fname = f"episode_{sea_state}_{ncomp}comp_seed{seed}_{raydrop}{ptag}.h5"
        key = fname
        out_fp = os.path.join(args.output_dir, fname)

        if manifest.get(key, {}).get('status') == 'ok' and os.path.exists(out_fp):
            print(f"[{i+1}/{len(jobs)}] SKIP (done) {fname}")
            continue

        print(f"[{i+1}/{len(jobs)}] {fname} ...", flush=True)
        t0 = time.time()
        try:
            fp = run_episode(
                sea_state=sea_state, n_components=ncomp, n_pts=args.n_pts,
                seed=seed, raydrop_model=raydrop, duration=args.duration,
                dt=args.dt, depth=args.depth, output_dir=args.output_dir,
                verbose=False, cfg=cfg, use_gpu_raycast=True,
                peak_locked=args.peak_locked,
            )
            elapsed = time.time() - t0
            # Cheap post-write sanity check before marking the job done --
            # the full verification pass (REGEN_SPEC Sec 5) is separate and
            # more thorough; this just prevents a manifest lie.
            with h5py.File(fp, 'r') as f:
                n_frames = f['frames/lidar_points'].shape[0]
                verdict = f['config'].attrs.get('mesh_verdict', 'UNKNOWN')
                gt = f['frames/gt_wave_field'][:]
                has_nan_gt = bool(np.isnan(gt).any())
                pts = f['frames/lidar_points'][:]
                valid = f['frames/lidar_valid_mask'][:]
                has_nan_valid_pts = bool(np.isnan(pts[valid]).any())
            ok = (not has_nan_gt) and (not has_nan_valid_pts) and n_frames > 0
            manifest[key] = dict(
                status='ok' if ok else 'SUSPECT',
                elapsed_s=elapsed, n_frames=int(n_frames), mesh_verdict=str(verdict),
                nan_in_gt=has_nan_gt, nan_in_valid_pts=has_nan_valid_pts, ts=time.time())
            print(f"    {'OK' if ok else 'SUSPECT (NaNs present)'}  frames={n_frames} "
                  f"verdict={verdict}  [{elapsed:.0f}s]")
        except Exception as e:
            manifest[key] = dict(status='FAILED', error=str(e),
                                  traceback=traceback.format_exc(), ts=time.time())
            print(f"    FAILED: {e}")
        json.dump(manifest, open(manifest_path, 'w'), indent=2)

        n_ok = sum(1 for v in manifest.values() if v.get('status') == 'ok')
        elapsed_total = time.time() - t_start
        remaining = len(jobs) - (i + 1)
        rate = elapsed_total / (i + 1)
        print(f"    progress: {n_ok}/{len(jobs)} ok, {elapsed_total/60:.1f} min elapsed, "
              f"~{remaining*rate/60:.0f} min remaining at current rate")

    n_ok = sum(1 for v in manifest.values() if v.get('status') == 'ok')
    n_bad = sum(1 for v in manifest.values() if v.get('status') in ('FAILED', 'SUSPECT'))
    print(f"\nDone. {n_ok} ok, {n_bad} failed/suspect. Manifest: {manifest_path}")
    if n_bad:
        print("Re-run the exact same command to retry FAILED/SUSPECT jobs (they are not skipped).")


if __name__ == '__main__':
    main()
