"""Close the fixedT regeneration coverage gap (2026-07-24).

`data/` holds 85 `*_fixedT*.h5` episodes; the 2026-07-23 pass regenerated only the
29 the A-7 v2 retry needed (`tmp/regen_fixedT_a7.py` / `.sh`).  The other 56 are
still on the sub-Nyquist 40x30 raycast mesh.

Procedure is deliberately identical to `tmp/regen_fixedT_a7.py`, which is itself
`tmp/regen_run_finemesh.py`'s procedure (REGEN_SPEC_fable_2026-07-23.md Sec 5):

  * mesh 200x167 (dx 1.206 m / dy 1.205 m) -- the *only* intervention
  * same seed  => `jonswap_components` reproduces bit-identical wave components
  * `n_pts` taken per-file from the shipped file's own `raydrop_model/cap`
    (8000 for every fixedT episode) so the observation budget is unchanged
  * duration/dt/depth read back from the shipped file (30.0 s / 0.1 s / 50.0 m)
  * `direction_deg=180` for the `_dir180` variants -- `run_episode` appends the
    `_dir180` filename tag itself
  * GPU raycast (REGEN_SPEC G3: the CPU path is ~14x slower), single process,
    sequential -- another GPU job is sharing the card

Resumable: an existing output file is skipped, and the manifest is rewritten
after every episode.  A corrupt/unreadable shipped source is recorded and
skipped rather than killing the batch (see `data_corrupt_slope_files.md`).

`data/` is never opened for writing (refused by path-prefix check).
"""
import argparse
import json
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
os.chdir(REPO)

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from wavesim.config import WaveSimConfig  # noqa: E402
from wavesim.pipeline import run_episode  # noqa: E402

SRC = 'data'
OUT = 'data_finemesh_fixedT_2026-07-23'
MANIFEST = os.path.join(OUT, '_regen_gap_manifest.json')
NX, NY = 200, 167                      # matches the 2026-07-23 regeneration exactly

PAT = re.compile(r'^episode_(BF\d)_(\d+)comp_seed(\d+)_([A-Za-z]+?)(_dir(\d+))?\.h5$')


def log(msg, fh=None):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    if fh:
        fh.write(line + '\n')
        fh.flush()


def parse_name(name):
    m = PAT.match(name)
    if not m:
        raise ValueError(f'unparsed episode name: {name}')
    sea, ncomp, seed, raydrop, _, deg = m.groups()
    return dict(sea_state=sea, n_components=int(ncomp), seed=int(seed),
                raydrop_model=raydrop, direction_deg=float(deg) if deg else 0.0)


def probe_source(path):
    """Read back the shipped file's own episode parameters (raises if corrupt)."""
    with h5py.File(path, 'r') as h:
        ts = h['frames/timestamps'][:]
        rd = dict(h['raydrop_model'].attrs)
        cap = int(rd.get('cap', rd.get('n_pts', 50)))
        n_frames = int(h['frames/lidar_points'].shape[0])
        width = int(h['frames/lidar_points'].shape[1])
        dt = float(np.round(ts[1] - ts[0], 6)) if len(ts) > 1 else 0.1
        depth = float(h['config'].attrs.get('depth', 50.0))
    return dict(cap=cap, n_frames=n_frames, width=width, dt=dt,
                duration=float(np.round((n_frames - 1) * dt, 6)), depth=depth)


def is_complete(path):
    """True only if `path` is a fully-written, openable episode.

    A run killed mid-`save_episode` would otherwise leave a truncated file that
    the resume logic happily skips forever.
    """
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, 'r') as h:
            return (h['frames/lidar_points'].shape[0] > 0
                    and 'gt_wave_field' in h['frames']
                    and 'wave_components' in h['config'])
    except Exception:                                           # noqa: BLE001
        return False


def load_manifest(path=None):
    MANIFEST = path or globals()['MANIFEST']
    if os.path.exists(MANIFEST):
        try:
            with open(MANIFEST) as fh:
                return json.load(fh)
        except Exception:                                       # noqa: BLE001
            pass
    return {}


def save_manifest(man, path=None):
    MANIFEST = path or globals()['MANIFEST']
    tmp = MANIFEST + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(man, fh, indent=2, sort_keys=True)
    os.replace(tmp, MANIFEST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', default='tmp/regen_gap_fixedT_todo.txt',
                    help='file of shipped basenames to regenerate')
    ap.add_argument('--output_dir', default=OUT)
    ap.add_argument('--log', default='tmp/regen_gap_fixedT.log')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    # Sharding lets a second worker share the load without racing the first:
    # disjoint episode slices, private staging dir, private manifest.
    ap.add_argument('--tag', default='', help='suffix for staging dir + manifest')
    ap.add_argument('--shard', default='', help='i/n -- take every n-th episode from offset i')
    args = ap.parse_args()

    out_dir = args.output_dir
    if os.path.abspath(out_dir) == os.path.abspath(SRC) or \
            os.path.abspath(out_dir).startswith(os.path.abspath(SRC) + os.sep):
        raise SystemExit(f'REFUSING to write into {SRC}/')
    os.makedirs(out_dir, exist_ok=True)

    names = [l.strip() for l in open(args.list) if l.strip() and not l.startswith('#')]
    if args.shard:
        i, n = (int(v) for v in args.shard.split('/'))
        names = names[i::n]
    if args.limit:
        names = names[:args.limit]

    suf = f'_{args.tag}' if args.tag else ''
    man_path = os.path.join(out_dir, f'_regen_gap_manifest{suf}.json')
    man = load_manifest(man_path)
    fh = open(args.log, 'a')
    log(f'=== regen_gap_fixedT: {len(names)} targets, mesh {NX}x{NY}, out={out_dir} ===', fh)

    stage = os.path.join(out_dir, f'_staging{suf}')
    os.makedirs(stage, exist_ok=True)
    for junk in os.listdir(stage):                # clear anything a kill left behind
        os.remove(os.path.join(stage, junk))

    todo = []
    for n in names:
        p = os.path.join(out_dir, n)
        if is_complete(p):
            continue
        if os.path.exists(p):
            log(f'truncated output found, will redo: {n}', fh)
            os.remove(p)
        todo.append(n)
    log(f'{len(names) - len(todo)} already complete, {len(todo)} to generate', fh)

    if args.dry_run:
        for n in todo:
            src = os.path.join(SRC, n)
            try:
                p = probe_source(src)
                spec = parse_name(n)
                log(f'PLAN {n}: {spec} n_pts={p["cap"]} dur={p["duration"]} '
                    f'dt={p["dt"]} depth={p["depth"]} (shipped width={p["width"]})', fh)
            except Exception as e:                              # noqa: BLE001
                log(f'PLAN-SKIP {n}: unreadable source: {e}', fh)
        return

    for i, name in enumerate(todo, 1):
        src = os.path.join(SRC, name)
        target = os.path.join(out_dir, name)
        rec = {'ts': time.time()}
        t0 = time.time()
        try:
            probe = probe_source(src)
        except Exception as e:                                  # noqa: BLE001
            rec.update(status='source_corrupt', error=f'{type(e).__name__}: {e}')
            man[name] = rec
            save_manifest(man, man_path)
            log(f'[{i}/{len(todo)}] SKIP {name}: shipped source unreadable ({e})', fh)
            continue

        spec = parse_name(name)
        cfg = WaveSimConfig()
        cfg.mesh.nx, cfg.mesh.ny = NX, NY
        # BF6/BF7 are not in configs/default.yaml; their Hs/Tp live only in
        # scripts/run_bf67_experiment.py.  Values re-confirmed against the
        # shipped files' own config attrs (BF6 3.0/10.0, BF7 5.0/12.0).
        cfg.sea_states.setdefault('BF6', {'Hs': 3.0, 'Tp': 10.0})
        cfg.sea_states.setdefault('BF7', {'Hs': 5.0, 'Tp': 12.0})
        try:
            # write into a staging dir first: only a complete file ever appears
            # under `out_dir`, so a kill mid-write cannot poison the resume logic
            produced = run_episode(n_pts=probe['cap'],
                                   duration=probe['duration'], dt=probe['dt'],
                                   depth=probe['depth'], output_dir=stage,
                                   verbose=False, cfg=cfg, use_gpu_raycast=True,
                                   **spec)
            if not is_complete(produced):
                raise RuntimeError(f'staged file incomplete: {produced}')
            os.replace(produced, target)
            with h5py.File(target, 'r') as h:
                verdict = str(h['config'].attrs.get('mesh_verdict', '?'))
                nfr = int(h['frames/lidar_points'].shape[0])
                w = int(h['frames/lidar_points'].shape[1])
                pts = h['frames/lidar_points'][:]
                mask = h['frames/lidar_valid_mask'][:]
                nan_pts = bool(np.isnan(pts[mask]).any())
                nan_gt = bool(np.isnan(h['frames/gt_wave_field'][:]).any())
            rec.update(status='ok', elapsed_s=time.time() - t0, n_frames=nfr,
                       width=w, src_width=probe['width'], n_pts=probe['cap'],
                       mesh_verdict=verdict, nan_in_valid_pts=nan_pts,
                       nan_in_gt=nan_gt, direction_deg=spec['direction_deg'])
            log(f'[{i}/{len(todo)}] done {name} [{time.time()-t0:.0f}s] '
                f'{verdict} frames={nfr} width={w}', fh)
        except Exception as e:                                  # noqa: BLE001
            rec.update(status='fail', error=f'{type(e).__name__}: {e}',
                       traceback=traceback.format_exc()[-1500:],
                       elapsed_s=time.time() - t0)
            for junk in (os.listdir(stage) if os.path.isdir(stage) else []):
                os.remove(os.path.join(stage, junk))  # never leave a partial behind
            if os.path.exists(target) and not is_complete(target):
                os.remove(target)
            log(f'[{i}/{len(todo)}] FAIL {name}: {e}', fh)
        man[name] = rec
        save_manifest(man, man_path)

    if os.path.isdir(stage) and not os.listdir(stage):
        os.rmdir(stage)
    have = sum(is_complete(os.path.join(out_dir, n)) for n in names)
    log(f'=== finished: {have}/{len(names)} present in {out_dir} ===', fh)
    fh.close()


if __name__ == '__main__':
    main()
