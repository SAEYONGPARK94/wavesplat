"""Confirmation-40 generation (reserve seeds 999/1001), PREREG_CONFIRM.md Sec 1.

Adapted from the frozen tmp/heldout40_2026-07-26/gen_heldout40.py with exactly three
changes: (a) SEEDS = [999, 1001] with the remaining reserve ladder [1111, 1212, 1313, 1414];
(b) output to data_confirm40_2026-07-27/; (c) GPU raycast path (the original certified path
of the 07-24 finemesh regeneration — the CPU path was itself certified bit-identical against
it, so either path is licensed; the GPU is idle today and ~4x faster).
Scenario list, mesh, adequacy ladder, episode constants: unchanged.

  python3 tmp/confirm40_2026-07-27/gen_confirm40.py --workers 3
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

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, REPO)

import h5py                                            # noqa: E402
import numpy as np                                     # noqa: E402

OUT_DIR = os.path.join(REPO, 'data_confirm40_2026-07-27')
MANIFEST = os.path.join(OUT_DIR, '_confirm40_manifest.json')
SEEDLOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seed_ladder_log.json')

# --- FROZEN (PREREG_CONFIRM Sec 1; scenarios/mesh identical to heldout40) ----------
SCENARIOS = [
    ('BF2', 1), ('BF3', 1), ('BF3', 2), ('BF3', 5),
    ('BF4', 1), ('BF4', 5), ('BF4', 15), ('BF4', 30),
    ('BF5', 15), ('BF5', 30),
]
SEEDS = [999, 1001]                # first two of the frozen reserve ladder, consumed here
RESERVE = [1111, 1212, 1313, 1414]  # remaining ladder for adequacy walks
RAYDROP_MODELS = ['uniform', 'slope']
MESH_NX, MESH_NY = 200, 167
N_PTS, DURATION, DT, DEPTH = 50, 30.0, 0.1, 50.0
BATCH_SIZE = 4000                  # GPU raycast native batch
# -----------------------------------------------------------------------------------


def make_cfg():
    from wavesim.config import WaveSimConfig
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = MESH_NX, MESH_NY
    cfg.raycast.batch_size = BATCH_SIZE
    return cfg


def adequacy(sea_state, ncomp, seed):
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
                                 f'slot {slot} -- STOP')
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
                         cfg=make_cfg(), use_gpu_raycast=True, peak_locked=False)
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
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

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
