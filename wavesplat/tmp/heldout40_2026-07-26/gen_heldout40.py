"""W4 held-out-40 generation (CPU only).

Protocol frozen in tmp/heldout40_2026-07-26/PREREG.md.  Generates the 40 fresh-seed
episodes of the held-out set into data_heldout40_2026-07-26/ at the same
Nyquist-adequate mesh (200x167) and the same episode constants as
tmp/regen_run_finemesh.py produced for data_finemesh_2026-07-23 --- but on the CPU
raycast path (the GPU is reserved by a concurrent experiment).

Two CPU-path differences, both certified no-ops before freezing (PREREG Sec 3.2):
  * raycast_surface (numpy) instead of raycast_surface_gpu -- hit-for-hit identical
    per wavesim/lidar_gpu.py's docstring and wavesim/tests/test_lidar_gpu.py;
  * cfg.raycast.batch_size 4000 -> 64 -- pure memory chunking, verified bit-identical
    on a probe frame (61108 identical hits, 4.4x faster).
Nothing that selects candidate triangles is touched.

Mesh-adequacy gate: the verdict is computed from the components + mesh BEFORE any
episode is rendered; a cell whose verdict is not ADEQUATE walks the frozen reserve
seed ladder.  Every walk is logged.  The verdict written into the .h5 is re-checked
after the render.

  nice -n 10 python3 tmp/heldout40_2026-07-26/gen_heldout40.py --workers 6
"""
import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')      # no CUDA context from this phase

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, REPO)

import h5py                                            # noqa: E402
import numpy as np                                     # noqa: E402

OUT_DIR = os.path.join(REPO, 'data_heldout40_2026-07-26')
MANIFEST = os.path.join(OUT_DIR, '_heldout40_manifest.json')
SEEDLOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seed_ladder_log.json')

# --- FROZEN (PREREG Sec 2.1 / 2.3) -------------------------------------------------
SCENARIOS = [                      # = tmp/regen_run_finemesh.py SCENARIOS (S01-S10)
    ('BF2', 1), ('BF3', 1), ('BF3', 2), ('BF3', 5),
    ('BF4', 1), ('BF4', 5), ('BF4', 15), ('BF4', 30),
    ('BF5', 15), ('BF5', 30),
]
SEEDS = [777, 888]                 # FROZEN fresh seeds, disjoint from every used seed
RESERVE = [999, 1001, 1111, 1212, 1313, 1414]   # FROZEN ladder, consumed in order
RAYDROP_MODELS = ['uniform', 'slope']
MESH_NX, MESH_NY = 200, 167
N_PTS, DURATION, DT, DEPTH = 50, 30.0, 0.1, 50.0
BATCH_SIZE = 64                    # memory chunking only; verified bit-identical
# -----------------------------------------------------------------------------------


def make_cfg():
    from wavesim.config import WaveSimConfig
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = MESH_NX, MESH_NY
    cfg.raycast.batch_size = BATCH_SIZE
    return cfg


def adequacy(sea_state, ncomp, seed):
    """Verdict for a (scenario, seed) cell -- cheap, no rendering, no estimator."""
    from wavesim import mesh_adequacy
    from wavesim.wave_generator import jonswap_components
    from wavesim.pipeline import SEA_STATE
    cfg = make_cfg()
    params = cfg.sea_states.get(sea_state, SEA_STATE.get(sea_state))
    comps = jonswap_components(Hs=params['Hs'], Tp=params['Tp'], n_components=ncomp,
                               depth=DEPTH, seed=seed, direction_deg=0.0,
                               peak_locked=False)
    return mesh_adequacy.check(comps, cfg, hos_order=None)


def resolve_seeds():
    """Walk the frozen ladder per (scenario, slot) until ADEQUATE. Returns (plan, log)."""
    plan, log = [], []
    for sea_state, ncomp in SCENARIOS:
        used = []
        for slot, base_seed in enumerate(SEEDS):
            ladder = [base_seed] + [s for s in RESERVE if s not in used]
            chosen = None
            for seed in ladder:
                if seed in used:
                    continue
                info = adequacy(sea_state, ncomp, seed)
                rec = dict(scenario=f'{sea_state}_{ncomp}comp', slot=slot, seed=seed,
                           verdict=info['mesh_verdict'],
                           var_under=info['var_under_resolved'],
                           pts_per_lambda=info['pts_per_lambda'],
                           pts_per_lambda_peak=info['pts_per_lambda_peak'])
                if info['mesh_verdict'] == 'ADEQUATE':
                    rec['action'] = 'ACCEPT'
                    log.append(rec)
                    chosen = seed
                    break
                rec['action'] = 'REJECT->next reserve seed'
                log.append(rec)
            if chosen is None:
                raise SystemExit(f'reserve ladder exhausted for {sea_state}_{ncomp}comp '
                                 f'slot {slot} -- STOP (PREREG Sec 3.3)')
            used.append(chosen)
            plan.append((sea_state, ncomp, chosen))
    return plan, log


def one(job):
    sea_state, ncomp, seed, raydrop = job
    from wavesim.pipeline import run_episode
    t0 = time.time()
    fname = f'episode_{sea_state}_{ncomp}comp_seed{seed}_{raydrop}.h5'
    try:
        fp = run_episode(sea_state=sea_state, n_components=ncomp, n_pts=N_PTS,
                         seed=seed, raydrop_model=raydrop, duration=DURATION,
                         dt=DT, depth=DEPTH, output_dir=OUT_DIR, verbose=False,
                         cfg=make_cfg(), use_gpu_raycast=False, peak_locked=False)
        with h5py.File(fp, 'r') as f:
            n_frames = int(f['frames/lidar_points'].shape[0])
            verdict = str(f['config'].attrs.get('mesh_verdict', 'UNKNOWN'))
            var_under = float(f['config'].attrs.get('var_under_resolved', np.nan))
            gt = f['frames/gt_wave_field'][:]
            pts = f['frames/lidar_points'][:]
            valid = f['frames/lidar_valid_mask'][:]
            nan_gt = bool(np.isnan(gt).any())
            nan_pts = bool(np.isnan(pts[valid]).any())
        ok = (verdict == 'ADEQUATE') and (not nan_gt) and (not nan_pts) and n_frames > 0
        return fname, dict(status='ok' if ok else 'SUSPECT', seed=seed,
                           scenario=f'{sea_state}_{ncomp}comp', raydrop=raydrop,
                           mesh_verdict=verdict, var_under_resolved=var_under,
                           n_frames=n_frames, nan_in_gt=nan_gt,
                           nan_in_valid_pts=nan_pts, elapsed_s=time.time() - t0,
                           ts=time.time())
    except Exception as e:                                        # noqa: BLE001
        return fname, dict(status='FAILED', seed=seed, raydrop=raydrop,
                           scenario=f'{sea_state}_{ncomp}comp', error=repr(e),
                           traceback=traceback.format_exc(),
                           elapsed_s=time.time() - t0, ts=time.time())


def _init():
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.workers > 6:
        raise SystemExit('workers > 6 would starve the concurrent SPREAD agent')

    os.makedirs(OUT_DIR, exist_ok=True)
    plan, log = resolve_seeds()
    json.dump(log, open(SEEDLOG, 'w'), indent=1)
    n_walk = sum(1 for r in log if r['action'].startswith('REJECT'))
    print(f'seed ladder: {len(plan)} cells resolved, {n_walk} reserve walk(s)', flush=True)
    for r in log:
        if r['action'].startswith('REJECT'):
            print(f"  REGEN {r['scenario']} slot{r['slot']} seed{r['seed']} "
                  f"{r['verdict']} var_under={r['var_under']:.3f}", flush=True)

    jobs = [(ss, nc, sd, rd) for (ss, nc, sd) in plan for rd in RAYDROP_MODELS]
    assert len(jobs) == 40, len(jobs)

    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    todo = [j for j in jobs
            if manifest.get(f'episode_{j[0]}_{j[1]}comp_seed{j[2]}_{j[3]}.h5',
                            {}).get('status') != 'ok']
    print(f'{len(jobs)} episodes, {len(jobs)-len(todo)} done, {len(todo)} to go, '
          f'{a.workers} workers', flush=True)
    if a.dry_run:
        for j in jobs:
            print(' ', j)
        return

    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(a.workers, initializer=_init) as pool:
        for i, (fname, rec) in enumerate(pool.imap_unordered(one, todo, chunksize=1), 1):
            manifest[fname] = rec
            json.dump(manifest, open(MANIFEST, 'w'), indent=1)
            el = time.time() - t0
            print(f'[{i}/{len(todo)}] {fname:<46s} {rec["status"]:<8s} '
                  f'{rec.get("mesh_verdict","-"):<10s} {rec["elapsed_s"]/60:.1f}min  '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)

    n_ok = sum(1 for v in manifest.values() if v.get('status') == 'ok')
    print(f'\nDONE {n_ok}/40 ok in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
