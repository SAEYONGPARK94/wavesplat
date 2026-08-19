#!/usr/bin/env python3
"""S13b runner -- see PREREG.md (frozen before any run; md5 in FROZEN_md5.txt).

Re-runs ONLY the shipping arm of S13 with the estimator seed changed (43, 44).
Everything else -- episode list, config, scorer -- is bit-identical to S13's SHIP arm,
and that identity is PROVED by gates G-1/G-2 before any run happens.

Usage:
    python3 tmp/S13b_seed_robust_2026-08-13/run_s13b.py gates
    python3 tmp/S13b_seed_robust_2026-08-13/run_s13b.py smoke
    python3 tmp/S13b_seed_robust_2026-08-13/run_s13b.py run 43 44

Results append to results_s13b.jsonl, one row per (episode, arm), resumable.
PREREG G-5: effective_config is READ BACK FROM THE ESTIMATOR OBJECT, not copied from
the input dict (that was S13's G-1 defect: it recorded the intent table).

tmp/S13_oos_2026-08-12/ is READ-ONLY here -- it is opened for reading only and never
imported (importing would create __pycache__ inside it).
"""
import glob
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

OUT = os.path.join(HERE, 'results_s13b.jsonl')
S13_JSONL = os.path.join(ROOT, 'tmp', 'S13_oos_2026-08-12', 'results.jsonl')

# ── sample: VERBATIM COPY of run_s13.py:32-48 (PREREG §3; identity proved by G-1) ──
SPARSE_DIR = ['data_s9_sparse_2026-08-11']
CROWDED_DIR = ['data_s8_crowded_2026-08-10', 'data_s9_crowded_2026-08-11']
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


# ── config: VERBATIM COPY of run_s13.py:53-55 + the one changed key ──────────
def ship_config():
    from wavesplat.run import DEFAULT_CONFIG
    return dict(DEFAULT_CONFIG)


def arm_config(est_seed):
    cfg = ship_config()
    cfg['seed'] = est_seed          # THE ONLY DIFFERENCE FROM S13's SHIP ARM
    return cfg


def coerce(cfg):
    """S13's serialisation rule (run_s13.py:181), so G-2 compares like with like."""
    return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in cfg.items()}


HEAD = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                      text=True).stdout.strip()

EST_SEEDS = (43, 44)
ARM = 'SHIP@{}'.format


def s13_ship_rows():
    rows = {}
    with open(S13_JSONL) as fh:                       # read-only, never written
        for line in fh:
            r = json.loads(line)
            if r.get('arm') == 'SHIP' and 'error' not in r:
                rows[r['episode']] = r
    return rows


# ── effective-config readback (PREREG §7 G-5) ───────────────────────────────
READBACK_KEYS = ('seed', 'lambda_l1', 'lr', 'init_method', 'use_confidence',
                 'use_dispersion', 'adaptive_n', 'max_n', 'tracking_mode',
                 'prune_threshold', 'grow_residual_threshold', 'lambda_smooth',
                 'tracking_steps', 'tracking_lr', 'depth', 'device',
                 'member_seeds', 'ensemble_k', 'k_requested')


def readback(u):
    out = {'updater_class': type(u).__name__}
    for k in READBACK_KEYS:
        v = getattr(u, k, '<absent>')
        out[k] = (v if isinstance(v, (int, float, str, bool, type(None)))
                  else (list(v) if isinstance(v, (list, tuple)) else str(v)))
    return out


def run_ws_seeded(path, est_seed):
    """Identical to run_s13.run_ws for the SHIP arm, plus the G-5 readback."""
    import torch
    import wavesplat.run as WR
    cfg = arm_config(est_seed)

    captured = {}
    orig = WR.make_updater

    def _capture(config):
        u = orig(config)
        captured['u'] = u
        # set_seed() already ran inside run_estimation -> prove the seed reached the RNG
        captured['torch_initial_seed'] = int(torch.initial_seed())
        captured['np_state0'] = int(np.random.get_state()[1][0])
        return u

    WR.make_updater = _capture
    try:
        r = WR.run_estimation(path, config=cfg, verbose=False)
    finally:
        WR.make_updater = orig

    fr = r.get('frame_results', [])
    shells = {}
    for f in fr:
        for k, v in (f.get('shell_rmse') or {}).items():
            if v is not None and np.isfinite(v):
                shells.setdefault(k, []).append(float(v))
    u = captured.get('u')
    return dict(avg_nRMSE=float(r['avg_nRMSE']),
                init_nrmse=float(r.get('init_nrmse', np.nan)),
                n_gaussians=int(r.get('n_gaussians', -1)),
                n_match=len(fr),
                shell_rmse={k: float(np.mean(v)) for k, v in shells.items()},
                effective_config=readback(u) if u is not None else None,
                intent_config=coerce(cfg),
                torch_initial_seed=captured.get('torch_initial_seed'),
                np_state0=captured.get('np_state0'),
                route=r.get('auto_selected'),
                n_selected=r.get('auto_n_selected'))


# ── gates ───────────────────────────────────────────────────────────────────
def gates():
    ok = True
    eps = episodes()
    s13 = s13_ship_rows()

    mine = {e['key']: e for e in eps}
    g1a = set(mine) == set(s13)
    g1b = all(mine[k]['stratum'] == s13[k]['stratum'] and mine[k]['bf'] == s13[k]['bf']
              and mine[k]['ncomp'] == s13[k]['ncomp'] and mine[k]['seed'] == s13[k]['seed']
              for k in mine) if g1a else False
    n_sp = sum(e['stratum'] == 'sparse' for e in eps)
    n_cr = sum(e['stratum'] == 'crowded' for e in eps)
    g1 = g1a and g1b and n_sp == 64 and n_cr == 64
    print(f'G-1 sample identity: mine={len(mine)} s13_SHIP={len(s13)} '
          f'sparse={n_sp} crowded={n_cr}')
    if not g1a:
        print(f'    only-mine={sorted(set(mine)-set(s13))[:5]} only-s13={sorted(set(s13)-set(mine))[:5]}')
    print(f'    keyset_equal={g1a} meta_equal={g1b}  G-1 {"PASS" if g1 else "FAIL"}')
    ok &= g1

    ref = arm_config(42)
    ref_c = coerce(ref)
    s13_cfgs = {json.dumps(r['effective_config'], sort_keys=True) for r in s13.values()}
    g2a = len(s13_cfgs) == 1
    s13_cfg = json.loads(next(iter(s13_cfgs))) if g2a else {}
    diff = {k: (ref_c.get(k, '<absent>'), s13_cfg.get(k, '<absent>'))
            for k in set(ref_c) | set(s13_cfg)
            if ref_c.get(k, '<absent>') != s13_cfg.get(k, '<absent>')}
    g2 = g2a and not diff
    print(f'G-2 config identity (seed reverted to 42): s13 distinct configs={len(s13_cfgs)} '
          f'keys={len(ref_c)} diff={diff}')
    print(f'    G-2 {"PASS" if g2 else "FAIL"}')
    ok &= g2

    from wavesplat.optimizer.sequential_update import SequentialUpdater
    g5 = (SequentialUpdater(ref).lambda_l1 == ref['lambda_l1']
          and SequentialUpdater(ref).seed == 42)
    print(f'G-5 readback wiring: HEAD={HEAD[:8]} lambda_l1={ref["lambda_l1"]} '
          f'init_method={ref["init_method"]} seed_default={ref["seed"]}')
    print(f'    G-5 {"PASS" if g5 else "FAIL"}')
    ok &= g5

    print(f'\nGATES(G-1,G-2,G-5) {"PASS -- cleared to smoke" if ok else "FAIL -- batch halted"}')
    return ok


def smoke():
    """G-3 determinism vs S13, G-4 seed sensitivity. 2 episodes (1 sparse, 1 crowded)."""
    eps = episodes()
    s13 = s13_ship_rows()
    pick = [next(e for e in eps if e['stratum'] == 'sparse'),
            next(e for e in eps if e['stratum'] == 'crowded')]
    ok, sens = True, []
    for e in pick:
        ref = s13[e['key']]['avg_nRMSE']
        t0 = time.time()
        r42 = run_ws_seeded(e['path'], 42)
        r43 = run_ws_seeded(e['path'], 43)
        d = abs(r42['avg_nRMSE'] - ref)
        same = (r43['avg_nRMSE'] == r42['avg_nRMSE'])
        sens.append(not same)
        print(f'  {e["stratum"]:8s} {e["key"]}')
        print(f'    G-3 seed42 rerun={r42["avg_nRMSE"]:.12f} S13={ref:.12f} |d|={d:.3e} '
              f'{"PASS" if d <= 1e-9 else "FAIL"}   route {r42["route"]} vs {s13[e["key"]]["route"]}')
        print(f'    G-4 seed43      ={r43["avg_nRMSE"]:.12f}  differs={not same}  '
              f'rel={100*(r43["avg_nRMSE"]-r42["avg_nRMSE"])/r42["avg_nRMSE"]:+.3f}%')
        print(f'    readback42 seed={r42["effective_config"]["seed"]} '
              f'torch_initial_seed={r42["torch_initial_seed"]} '
              f'members={r42["effective_config"]["member_seeds"]} '
              f'class={r42["effective_config"]["updater_class"]}')
        print(f'    readback43 seed={r43["effective_config"]["seed"]} '
              f'torch_initial_seed={r43["torch_initial_seed"]} '
              f'members={r43["effective_config"]["member_seeds"]}  ({time.time()-t0:.0f}s)')
        ok &= (d <= 1e-9)
    print(f'\nG-3 determinism {"PASS" if ok else "FAIL -- HALT (env/code drifted since S13)"}')
    print(f'G-4 seed sensitivity: {sum(sens)}/2 episodes moved '
          f'{"-> normal branch" if any(sens) else "-> PREREG section-5 case 4 branch (seed-deterministic)"}')
    return ok


# ── main run ────────────────────────────────────────────────────────────────
def main_run(seeds):
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['episode'], r['arm']))
    eps = episodes()
    jobs = [(e, s) for s in seeds for e in eps]
    todo = [(e, s) for e, s in jobs if (e['key'], ARM(s)) not in done]
    print(f'# seeds={seeds} total={len(jobs)} remaining={len(todo)} HEAD={HEAD[:8]}', flush=True)
    t_start = time.time()
    with open(OUT, 'a') as fh:
        for i, (e, s) in enumerate(todo, 1):
            t0 = time.time()
            row = None
            for attempt in (1, 2):                    # PREREG section-4: one auto retry
                try:
                    res = run_ws_seeded(e['path'], s)
                    row = dict(episode=e['key'], arm=ARM(s), est_seed=s,
                               stratum=e['stratum'], bf=e['bf'], ncomp=e['ncomp'],
                               seed=e['seed'],
                               pre_exposed=e['seed'] in PRE_EXPOSED[e['stratum']],
                               attempt=attempt, wall_s=time.time() - t0,
                               git_head=HEAD, **res)
                    break
                except Exception as ex:
                    err = f'{type(ex).__name__}: {ex}'
                    print(f'    ! attempt {attempt} failed: {err}', flush=True)
                    row = dict(episode=e['key'], arm=ARM(s), est_seed=s,
                               stratum=e['stratum'], attempt=attempt,
                               error=err, wall_s=time.time() - t0, git_head=HEAD)
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            el = time.time() - t_start
            eta = el / i * (len(todo) - i) / 60
            print(f'[{i}/{len(todo)}] {ARM(s):8s} {e["key"]:44s} '
                  f'{row.get("avg_nRMSE", row.get("error"))} '
                  f'({time.time()-t0:.0f}s, ETA {eta:.0f}m)', flush=True)


if __name__ == '__main__':
    cmd = (sys.argv[1:] or ['gates'])[0]
    if cmd == 'gates':
        sys.exit(0 if gates() else 1)
    elif cmd == 'smoke':
        sys.exit(0 if smoke() else 1)
    elif cmd == 'run':
        main_run([int(x) for x in sys.argv[2:]] or list(EST_SEEDS))
    else:
        print(__doc__)
