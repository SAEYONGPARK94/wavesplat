#!/usr/bin/env python3
"""C runner -- PREREG §2 arms over the 2x2 corpus.  Frozen before any verdict row.

Arms: SHIP (init_method='auto' = the deployed router, ITT), V2K ('v2_konly', no learning
-- the pre-declared OOD decomposer of PREREG §1.2), ZERO (zero-field baseline).

ZERO is computed directly from the GT rather than through an estimator: the synthetic
harness normalises by Hs (metrics.py:36,65) and metrics.py:13 records that the zero field
lands at nRMSE ~ 0.25 there.  Computing it closed-form removes a whole class of plumbing
bug -- the B batch spent 128 rows on an arm-contract mismatch in exactly this place.

Shells are aggregated from frame_results here.  run.py:366 fills them PER FRAME and the
caller must aggregate; reading only the summary yields None on every row, which is what
happened to the B batch (its third recurrence that day of a key list dropping a
measurement).
"""
import glob
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

EST_SEEDS = [1614, 1615, 1616]                       # PREREG §3, median-of-3
ARMS = {
    'SHIP': {'init_method': 'auto'},                 # deployed router, ITT
    'V2K': {'init_method': 'v2_konly'},              # OOD decomposer
    'ZERO': None,
}
CELLS = ['A_static_following', 'B_static_head',
         'C_moving_following', 'D_moving_head']
_IMPORT_PATH_FILES = ['wavesplat/run.py', 'wavesplat/optimizer/sequential_update.py',
                      'wavesplat/evaluation/metrics.py']


def code_md5():
    """PREREG G-7: git_head alone does not catch a dirty tree (S13/S13b were both bitten
    mid-batch by another session editing an import-path file)."""
    h = hashlib.md5()
    for rel in _IMPORT_PATH_FILES + ['tmp/C_ENCOUNTER_2026-08-20/run_c.py',
                                     'tmp/C_ENCOUNTER_2026-08-20/moving_sensor.py']:
        with open(os.path.join(ROOT, rel), 'rb') as fh:
            h.update(fh.read())
    return h.hexdigest()


def episodes():
    out = []
    for cell in CELLS:
        for p in sorted(glob.glob(os.path.join(HERE, 'data', cell, '*.h5'))):
            seed = int(os.path.basename(p).split('seed')[1].split('_')[0])
            out.append(dict(cell=cell, seed=seed, path=p,
                            key='%s|%d' % (cell, seed)))
    return out


def _shells(frame_results, key):
    acc = {}
    for f in frame_results or []:
        for k, v in (f.get(key) or {}).items():
            if v is not None and v == v:
                acc.setdefault('obs:' + k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()} or None


def run_zero(ep):
    """nRMSE of predicting zero, in the synthetic (Hs-normalised) convention."""
    import h5py
    with h5py.File(ep['path'], 'r') as h:
        Z = h['frames/gt_wave_field'][:]
    per = [float(np.sqrt(np.mean(z ** 2)) / (4.0 * np.std(z))) for z in Z]
    return dict(avg_nRMSE=float(np.mean(per)), n_frames=len(per),
                shell_rmse=None, route=None)


def run_est(ep, cfg):
    from wavesplat.run import run_estimation
    s = run_estimation(h5_path=ep['path'], config=cfg)
    return dict(avg_nRMSE=s.get('avg_nRMSE'),
                avg_RMSE_near=s.get('avg_RMSE_near'),
                n_frames=s.get('n_frames'),
                route=s.get('auto_selected'),
                init_time_s=s.get('init_time_s'),
                shell_rmse=_shells(s.get('frame_results'), 'shell_rmse'),
                shell_frac=_shells(s.get('frame_results'), 'shell_frac'),
                # object read-back, NOT a copy of the input dict: S13b showed an
                # intent-table hides which knobs actually engaged (its ensemble ran on
                # 14% of runs while the intent table said 100%)
                effective=None)


def run_one(job):
    ep, arm, seed = job
    t0 = time.time()
    row = dict(cell=ep['cell'], seed=ep['seed'], arm=arm, est_seed=seed,
               key=ep['key'], code_md5=code_md5())
    try:
        if arm == 'ZERO':
            row.update(run_zero(ep))
        else:
            from wavesplat.run import DEFAULT_CONFIG
            cfg = dict(DEFAULT_CONFIG)
            cfg.update(ARMS[arm]); cfg['seed'] = seed
            row.update(run_est(ep, cfg))
    except Exception as exc:               # noqa: BLE001 -- recorded, never swallowed
        import traceback
        row.update(error=str(exc), tb=traceback.format_exc()[-1200:])
    row['wall_s'] = round(time.time() - t0, 2)
    return row


def main():
    eps = episodes()
    jobs = [(e, a, s) for e in eps for a in ARMS
            for s in (EST_SEEDS if a != 'ZERO' else EST_SEEDS[:1])]
    print('episodes=%d  jobs=%d  code_md5=%s' % (len(eps), len(jobs), code_md5()),
          flush=True)
    out = os.path.join(HERE, 'results.jsonl')
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            if 'error' not in r:
                done.add((r['key'], r['arm'], r['est_seed']))
    jobs = [j for j in jobs if (j[0]['key'], j[1], j[2]) not in done]
    print('remaining=%d' % len(jobs), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex, open(out, 'a') as fh:
        for i, row in enumerate(ex.map(run_one, jobs), 1):
            fh.write(json.dumps(row) + '\n'); fh.flush()
            if i % 25 == 0 or i == len(jobs):
                print('[%5.1f min] %d/%d' % ((time.time() - t0) / 60, i, len(jobs)),
                      flush=True)
    print('DONE %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
