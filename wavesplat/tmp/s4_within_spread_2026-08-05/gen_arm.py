"""S4 arm generation -- design frozen in PREREG_ARM.md (written before this ran).

Arm A (M=1, N=30) is INHERITED byte-for-byte from data_finemesh_2026-07-23, not
regenerated: the current code reproduces those files exactly (verified: 11/11
datasets bit-identical on a re-run, and wave_components + the first 11 frames of
the stored file match today's pipeline). Copying instead of regenerating removes
any chance of an arm-identity difference and saves ~15 min.

The copies are then stamped with the two provenance attrs (dirs_per_freq,
direction_spread_deg) that no episode carried before 2026-08-05. The pre-stamp
sha256 is checked against the source and both hashes are recorded, so the
inheritance stays auditable.

Run:  python3 tmp/s4_within_spread_2026-08-05/gen_arm.py
Resumable: skips anything the manifest already marks ok.
"""
import hashlib
import json
import os
import shutil
import sys
import time
import traceback

import h5py
import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
HERE = os.path.join(ROOT, 'tmp/s4_within_spread_2026-08-05')
SRC = os.path.join(ROOT, 'data_finemesh_2026-07-23')
OUT = os.path.join(ROOT, 'data_dirspread_2026-08-05')
MANIFEST = os.path.join(HERE, 'gen_manifest.json')

sys.path.insert(0, ROOT)
from wavesim.config import WaveSimConfig      # noqa: E402
from wavesim.pipeline import run_episode      # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
SPREAD = 30.0
COMMON = dict(sea_state='BF4', raydrop_model='uniform', n_pts=50,
              duration=30.0, dt=0.1, depth=50.0, direction_spread_deg=SPREAD)
ARMS = [('A_M1_N30', 30, 1), ('B_M3_N30', 30, 3), ('C_M1_N10', 10, 1)]


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dsets(f):
    out = []
    f.visititems(lambda n, o: out.append(n) if isinstance(o, h5py.Dataset) else None)
    return out


def gate(fp):
    """G2/G3/G4 -- refuse to mark a file ok unless it passes all of them."""
    with h5py.File(fp, 'r') as f:
        a = f['config'].attrs
        verdict = str(a.get('mesh_verdict', 'MISSING'))
        has_attrs = ('dirs_per_freq' in a) and ('direction_spread_deg' in a)
        n_frames = int(f['frames/lidar_points'].shape[0])
        gt = f['frames/gt_wave_field'][:]
        pts = f['frames/lidar_points'][:]
        valid = f['frames/lidar_valid_mask'][:]
        nan_gt = bool(np.isnan(gt).any())
        nan_pts = bool(np.isnan(pts[valid]).any())
        info = dict(mesh_verdict=verdict, has_prov_attrs=has_attrs, n_frames=n_frames,
                    nan_in_gt=nan_gt, nan_in_valid_pts=nan_pts,
                    lambda_min=float(a.get('lambda_min', np.nan)),
                    attr_dirs_per_freq=int(a.get('dirs_per_freq', -1)),
                    attr_direction_spread_deg=float(a.get('direction_spread_deg', np.nan)))
    info['gates_ok'] = bool(verdict == 'ADEQUATE' and has_attrs and n_frames == 301
                            and not nan_gt and not nan_pts)
    return info


def main():
    os.makedirs(OUT, exist_ok=True)
    man = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = 200, 167

    jobs = [(label, ncomp, M, sd) for (label, ncomp, M) in ARMS for sd in SEEDS]
    t_start = time.time()
    for i, (label, ncomp, M, sd) in enumerate(jobs, 1):
        mtag = '' if M == 1 else f'_m{M}'
        fname = f'episode_BF4_{ncomp}comp_seed{sd}_uniform{mtag}.h5'
        fp = os.path.join(OUT, fname)
        if man.get(fname, {}).get('status') == 'ok' and os.path.exists(fp):
            print(f'[{i}/{len(jobs)}] SKIP {fname}')
            continue
        print(f'[{i}/{len(jobs)}] {label} {fname} ...', flush=True)
        t0 = time.time()
        try:
            if label == 'A_M1_N30':
                # --- inherit, verify byte-identity (G1), then stamp attrs ---
                src = os.path.join(SRC, fname)
                h_src = sha256(src)
                shutil.copy2(src, fp)
                h_copy = sha256(fp)
                if h_src != h_copy:
                    raise RuntimeError('copy is not byte-identical to source')
                with h5py.File(src, 'r') as a, h5py.File(fp, 'a') as b:
                    pre = {d: a[d][()] for d in dsets(a)}
                    b['config'].attrs['dirs_per_freq'] = 1
                    b['config'].attrs['direction_spread_deg'] = SPREAD
                    bad = [d for d in dsets(b)
                           if not np.array_equal(pre[d], b[d][()], equal_nan=True)]
                if bad:
                    raise RuntimeError(f'stamping perturbed datasets: {bad}')
                prov = dict(mode='inherited', source=os.path.relpath(src, ROOT),
                            sha256_source=h_src, sha256_prestamp=h_copy,
                            sha256_poststamp=sha256(fp),
                            byte_identical_to_source_prestamp=True,
                            datasets_unchanged_by_stamp=True)
            else:
                out_fp = run_episode(n_components=ncomp, seed=sd, output_dir=OUT,
                                     verbose=False, cfg=cfg, use_gpu_raycast=True,
                                     dirs_per_freq=M, **COMMON)
                assert os.path.abspath(out_fp) == os.path.abspath(fp), (out_fp, fp)
                prov = dict(mode='generated', sha256=sha256(fp))
            g = gate(fp)
            man[fname] = dict(status='ok' if g['gates_ok'] else 'GATE_FAIL',
                              arm=label, n_components=ncomp, dirs_per_freq=M, seed=sd,
                              elapsed_s=time.time() - t0, **prov, **g)
            print(f"    {man[fname]['status']}  verdict={g['mesh_verdict']} "
                  f"frames={g['n_frames']} lam_min={g['lambda_min']:.2f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        except Exception as e:
            man[fname] = dict(status='FAILED', arm=label, error=str(e),
                              traceback=traceback.format_exc())
            print(f'    FAILED: {e}', flush=True)
        json.dump(man, open(MANIFEST, 'w'), indent=2)
        n_ok = sum(1 for v in man.values() if v.get('status') == 'ok')
        el = time.time() - t_start
        print(f'    progress {n_ok}/{len(jobs)} ok, {el/60:.1f} min elapsed, '
              f'~{(len(jobs)-i)*el/i/60:.0f} min left', flush=True)

    bad = {k: v['status'] for k, v in man.items() if v.get('status') != 'ok'}
    print(f"\nDone. ok={sum(1 for v in man.values() if v.get('status')=='ok')}/{len(jobs)}"
          + (f'  PROBLEMS: {bad}' if bad else '  all gates passed'))


if __name__ == '__main__':
    main()
