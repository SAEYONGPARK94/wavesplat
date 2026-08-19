#!/usr/bin/env python3
"""S13 runner — see PREREG.md (frozen before any run).

Usage:
    python3 tmp/S13_oos_2026-08-12/run_s13.py gates
    python3 tmp/S13_oos_2026-08-12/run_s13.py run SHIP LETKF_PLUS      # Phase 1 (PRIMARY)
    python3 tmp/S13_oos_2026-08-12/run_s13.py run LETKF SHIP_L1E3      # Phase 2
    python3 tmp/S13_oos_2026-08-12/run_s13.py run SHIP_NOCONF SHIP_NODISP   # Phase 3

Results append to results.jsonl, one row per (episode, arm), resumable.
Every row carries the effective config + git HEAD + router route (PREREG G-1).
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # .../wavesplat/wavesplat
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = os.path.join(HERE, 'results.jsonl')

# ── sample (PREREG §3) ──────────────────────────────────────────────────────
SPARSE_DIR = ['data_s9_sparse_2026-08-11']
CROWDED_DIR = ['data_s8_crowded_2026-08-10', 'data_s9_crowded_2026-08-11']
# BS-3 pre-exposure (PREREG §3): SHIP absolute nRMSE was seen for these seeds.
PRE_EXPOSED = {'sparse': {56, 57, 58}, 'crowded': {52, 53, 54}}


def episodes():
    out = []
    for stratum, dirs in (('sparse', SPARSE_DIR), ('crowded', CROWDED_DIR)):
        for d in dirs:
            for p in sorted(glob.glob(os.path.join(d, '*.h5'))):
                b = os.path.basename(p)
                m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(.+)\.h5', b)
                out.append(dict(path=p, key=b[len('episode_'):-3], stratum=stratum,
                                bf=m.group(1), ncomp=int(m.group(2)),
                                seed=int(m.group(3)), suffix=m.group(4)))
    return out


# ── arms (PREREG §2) ────────────────────────────────────────────────────────
# SHIP is DEFAULT_CONFIG written out explicitly -- no fallback (PREREG §2 warning).
def ship_config():
    from wavesplat.run import DEFAULT_CONFIG
    return dict(DEFAULT_CONFIG)


WS_ARMS = {
    'SHIP': {},
    'SHIP_L1E3': {'lambda_l1': 1e-3},
    'SHIP_NOCONF': {'use_confidence': False},
    'SHIP_NODISP': {'use_dispersion': False},
}
LETKF_ARMS = {'LETKF': 100, 'LETKF_PLUS': 600}
CROWDED_ONLY = {'SHIP_L1E3'}

HEAD = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                      text=True).stdout.strip()


# ── gates (PREREG §7) ───────────────────────────────────────────────────────
def gates():
    ok = True
    eps = episodes()

    # G-3: sample integrity
    n_sp = sum(e['stratum'] == 'sparse' for e in eps)
    n_cr = sum(e['stratum'] == 'crowded' for e in eps)
    suff = {e['suffix'] for e in eps}
    dup = len({(e['stratum'], e['bf'], e['ncomp'], e['seed']) for e in eps}) != len(eps)
    seeds = {'sparse': {e['seed'] for e in eps if e['stratum'] == 'sparse'},
             'crowded': {e['seed'] for e in eps if e['stratum'] == 'crowded'}}
    g3 = (n_sp == 64 and n_cr == 64 and suff == {'uniform_ra_rf_ds'} and not dup)
    print(f'G-3 sample: sparse={n_sp} crowded={n_cr} suffixes={suff} dup={dup}')
    print(f'    seeds sparse={min(seeds["sparse"])}-{max(seeds["sparse"])} '
          f'crowded={sorted(seeds["crowded"])[:3]}...{sorted(seeds["crowded"])[-1]}')
    print(f'    G-3 {"PASS" if g3 else "FAIL"}')
    ok &= g3

    # G-2: checkpoint donors must not contain any evaluation seed
    sys.path.insert(0, os.path.join(ROOT, 'tmp', 'mixed_retrain_2026-07-26'))
    from gen_mixed import DONOR_FILES
    donor_seeds, donor_cells = set(), set()
    for _d, fn in DONOR_FILES:
        m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_', fn)
        donor_seeds.add(int(m.group(3)))
        donor_cells.add((m.group(1), int(m.group(2))))
    eval_seeds = seeds['sparse'] | seeds['crowded']
    eval_cells = {(e['bf'], e['ncomp']) for e in eps}
    overlap_seed = donor_seeds & eval_seeds
    overlap_cell = donor_cells & eval_cells
    g2 = not overlap_seed
    print(f'G-2 leakage: donor seeds={sorted(donor_seeds)} eval seeds={min(eval_seeds)}-{max(eval_seeds)}')
    print(f'    seed overlap={sorted(overlap_seed)}  cell overlap={sorted(overlap_cell)}')
    print(f'    G-2 {"PASS" if g2 else "FAIL"}  (donors also drop z: geometry only)')
    ok &= g2

    # G-1: effective-config recording is wired
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    cfg = ship_config()
    g1 = ('lambda_l1' in cfg and 'init_method' in cfg
          and SequentialUpdater(cfg).lambda_l1 == cfg['lambda_l1'])
    print(f'G-1 effective-config: HEAD={HEAD[:8]} lambda_l1={cfg["lambda_l1"]} '
          f'init_method={cfg["init_method"]} explicit_keys={len(cfg)}')
    print(f'    G-1 {"PASS" if g1 else "FAIL"}')
    ok &= g1

    print(f'\nGATES {"PASS -- cleared to run" if ok else "FAIL -- batch halted"}')
    return ok


# ── scoring ─────────────────────────────────────────────────────────────────
def run_letkf(path, n_ensemble):
    """Frame-matched LETKF + zero-field floor. Mirrors bs1_letkf_perframe.run_one,
    which reproduced the published LETKF column to max|d| = 1.5e-08 (BLANKSLATE G-0)."""
    from scipy.interpolate import RegularGridInterpolator
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.baselines.letkf_grid import LETKFGrid
    from wavesplat.evaluation.metrics import compute_all_metrics
    np.random.seed(42)
    loader = EpisodeDataLoader(path)
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    egx = np.linspace(gx[0], gx[-1], 40)
    egy = np.linspace(gy[0], gy[-1], 30)
    letkf = LETKFGrid(egx, egy, depth=loader.depth, n_ensemble=n_ensemble,
                      mult_inflation=1.0, add_inflation=0.005, loc_radius=50.0)
    Xg, Yg = np.meshgrid(gx, gy)
    q = np.column_stack([Yg.ravel(), Xg.ravel()])
    nr, zr, ms, t = [], [], [], []
    for i, frame in enumerate(loader):
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        ms.append(float(letkf.update(x, y, z)))
        pred = letkf.predict_grid()
        interp = RegularGridInterpolator((egy, egx), pred, method='linear',
                                         bounds_error=False, fill_value=0.0)
        pf = interp(q).reshape(len(gy), len(gx))
        gt = frame.gt_field.cpu().numpy()
        m = compute_all_metrics(pf, gt, gx, gy, loader.Hs_true,
                                obs_x=x if len(x) else None,
                                obs_y=y if len(y) else None)
        nr.append(float(m['nRMSE']))
        zr.append(float(np.sqrt(np.mean(gt ** 2)) / loader.Hs_true))
        t.append(float(loader.timestamps[i]))
    sel = np.array(t) > 2.0 * float(loader.Tp_true)
    return dict(avg_nRMSE=float(np.mean(np.array(nr)[sel])),
                zero_nRMSE=float(np.mean(np.array(zr)[sel])),
                n_match=int(sel.sum()), n_all=len(nr),
                ms_median=float(np.median(ms)),
                Tp=float(loader.Tp_true), Hs=float(loader.Hs_true))


def run_ws(path, overrides):
    from wavesplat.run import run_estimation
    cfg = ship_config()
    cfg.update(overrides)
    r = run_estimation(path, config=cfg, verbose=False)
    fr = r.get('frame_results', [])
    shells = {}
    for f in fr:
        for k, v in (f.get('shell_rmse') or {}).items():
            if v is not None and np.isfinite(v):
                shells.setdefault(k, []).append(float(v))
    return dict(avg_nRMSE=float(r['avg_nRMSE']),
                init_nrmse=float(r.get('init_nrmse', np.nan)),
                n_gaussians=int(r.get('n_gaussians', -1)),
                n_match=len(fr),
                shell_rmse={k: float(np.mean(v)) for k, v in shells.items()},
                effective_config={k: (v if isinstance(v, (int, float, str, bool, type(None)))
                                      else str(v)) for k, v in cfg.items()},
                route=r.get('auto_selected'),          # run.py:446 -- what the router picked
                n_selected=r.get('auto_n_selected'))


def main_run(arms):
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r['episode'], r['arm']))
    eps = episodes()
    jobs = [(e, a) for a in arms for e in eps
            if not (a in CROWDED_ONLY and e['stratum'] != 'crowded')]
    todo = [(e, a) for e, a in jobs if (e['key'], a) not in done]
    print(f'# arms={arms}  total={len(jobs)}  remaining={len(todo)}', flush=True)
    with open(OUT, 'a') as fh:
        for i, (e, arm) in enumerate(todo, 1):
            t0 = time.time()
            try:
                if arm in LETKF_ARMS:
                    res = run_letkf(e['path'], LETKF_ARMS[arm])
                else:
                    res = run_ws(e['path'], WS_ARMS[arm])
                row = dict(episode=e['key'], arm=arm, stratum=e['stratum'],
                           bf=e['bf'], ncomp=e['ncomp'], seed=e['seed'],
                           pre_exposed=e['seed'] in PRE_EXPOSED[e['stratum']],
                           wall_s=time.time() - t0, git_head=HEAD, **res)
            except Exception as ex:
                row = dict(episode=e['key'], arm=arm, stratum=e['stratum'],
                           error=f'{type(ex).__name__}: {ex}', git_head=HEAD)
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            print(f'[{i}/{len(todo)}] {arm:12s} {e["key"]:44s} '
                  f'{row.get("avg_nRMSE", row.get("error"))} ({time.time()-t0:.0f}s)',
                  flush=True)


if __name__ == '__main__':
    cmd = (sys.argv[1:] or ['gates'])[0]
    if cmd == 'gates':
        sys.exit(0 if gates() else 1)
    elif cmd == 'run':
        main_run(sys.argv[2:] or ['SHIP'])
    else:
        print(__doc__)
