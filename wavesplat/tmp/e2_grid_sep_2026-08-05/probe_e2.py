"""E2 placement probe -- the placement axis, measured WITHOUT the optimizer.

Pre-registered in tmp/e2_grid_sep_2026-08-05/PREREG.md section 3/5. Reuses E1's probe design
(tmp/e1_grid_2026-08-05/probe_placement.py) verbatim in structure so cell A0 pairs
bit-for-bit with E1 cell A, and adds the nearest-neighbour spacing distribution that PREREG
section 5.1 requires (the direct stacking indicator).

Runs `initialize_from_frames_v2` ALONE on the init window (no optimizer, no adaptive-N, no
tracker). Deterministic: the only RNG is the pad branch and seed=42 is fixed.
Budgets {3, 15}: 3 is what the deployed ladder actually asks this function for, 15 is E1's
headline budget. (E1's n_g=30 dropped -- flagged there as counterfactual, and costly at
n_coarse=401.)
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from run_e2 import (CELLS, EPISODES, DATA, apply_cell, match_components,  # noqa: E402
                    nn_stats, cell_spec)

OUT = os.path.join(HERE, 'probe.jsonl')
BUDGETS = [3, 15]
SEED = 42


def probe_one(job):
    sys.path.insert(0, REPO)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import h5py
    from wavesplat.data_loader import EpisodeDataLoader
    import wavesplat.optimizer.initializer as I

    cell, ep, n_g = job['cell'], job['episode'], job['n_g']
    path = os.path.join(DATA, ep)
    t0 = time.time()
    undo = None
    try:
        with h5py.File(path, 'r') as f:
            gt = np.asarray(f['config/wave_components'])
            depth = float(f['config'].attrs['depth'])
        frames = EpisodeDataLoader(path).get_init_window(tp_factor=2.0)

        undo = apply_cell(cell)
        try:
            kx, ky, w, phi = I.initialize_from_frames_v2(frames, n_g, depth, seed=SEED)
            pad_w = I.PAD_W
        finally:
            undo(); undo = None

        kx = kx.numpy().astype(float); ky = ky.numpy().astype(float)
        w = np.abs(w.numpy().astype(float))
        gA, gkx, gky = gt[:, 0], gt[:, 2], gt[:, 3]
        matched, un_gt, un_est = match_components(kx, ky, w, gkx, gky, gA)
        rec = {'job_id': job['job_id'], 'cell': cell, 'episode': ep, 'n_g': n_g,
               'n_gt': int(len(gA)), 'n_matched': len(matched),
               'detection_rate': len(matched) / max(len(gA), 1)}
        if matched:
            kerr, derr = [], []
            for ei, gi, c in matched:
                gm = np.hypot(gkx[gi], gky[gi]); em = np.hypot(kx[ei], ky[ei])
                kerr.append(abs(em - gm) / max(gm, 1e-6))
                gd = np.degrees(np.arctan2(gky[gi], gkx[gi]))
                ed = np.degrees(np.arctan2(ky[ei], kx[ei]))
                dd = abs(ed - gd); dd = 360 - dd if dd > 180 else dd
                derr.append(dd)
            idx = [m[1] for m in matched]
            rec['k_mag_rel_error_mean'] = float(np.mean(kerr))
            rec['direction_error_deg_mean'] = float(np.mean(derr))
            rec['energy_detection_rate'] = float(
                np.sum(gA[idx] ** 2) / max(np.sum(gA ** 2), 1e-12))
            rec['match_cost_mean'] = float(np.mean([m[2] for m in matched]))
        else:
            rec.update(k_mag_rel_error_mean=float('nan'),
                       direction_error_deg_mean=float('nan'),
                       energy_detection_rate=0.0, match_cost_mean=float('nan'))
        # placement diagnostics
        rec['n_outside_narrow_box'] = int(np.sum((np.abs(kx) > 0.8) | (np.abs(ky) > 0.8)))
        rec['kmag_max_placed'] = float(np.max(np.hypot(kx, ky)))
        # atoms whose amplitude is exactly PAD_W are (almost surely) padding, not peaks
        rec['n_found_est'] = int(np.sum(w != pad_w))
        rec.update(nn_stats(kx, ky, prefix='placed_'))
        # kept for continuity with E1's field name
        rec['n_placed_pairs_lt_002'] = rec['placed_n_pairs_lt_002']
        rec['placed_nn_median'] = rec['placed_nn_median']
        rec['wall_s'] = time.time() - t0
        return rec
    except Exception as e:
        import traceback
        if undo is not None:
            undo()
        return {'job_id': job['job_id'], 'cell': cell, 'episode': ep, 'n_g': n_g,
                'error': repr(e), 'tb': traceback.format_exc()[-1200:]}


def _init_worker():
    sys.path.insert(0, REPO); sys.path.insert(0, HERE)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')


def main():
    import multiprocessing as mp
    cells = sys.argv[1].split(',') if len(sys.argv) > 1 else list(CELLS)
    jobs = [{'job_id': f'{c}|{ep}|{n}', 'cell': c, 'episode': ep, 'n_g': n}
            for c in cells for ep in EPISODES for n in BUDGETS]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add(r['job_id'])
    todo = [j for j in jobs if j['job_id'] not in done]
    todo.sort(key=lambda j: -(CELLS[j['cell']]['n'] or 81))
    print(f'total {len(jobs)} | todo {len(todo)}', flush=True)
    if not todo:
        print('nothing to do'); return
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(int(os.environ.get('WORKERS', '12')),
                  initializer=_init_worker) as pool, open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(probe_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<50s} '
                  f'det={rec.get("detection_rate", rec.get("error"))}', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
