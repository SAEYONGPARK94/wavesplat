#!/usr/bin/env python3
"""B · DEPLOY runner -- the W5 treatment, but in the SHIPPING configuration.

Two things change from W5 and nothing else:
  1. the router is NOT pinned (init_method stays 'auto', the shipped default), so the
     treatment only takes effect on the branch the router chooses -- an intent-to-treat
     design, which is the correct shape for a deployment question;
  2. both strata run (128 episodes), not crowded only.

Everything that defines the treatment is IMPORTED from the frozen W4/W5 modules -- varpro,
the counters, the expectation formula.  Nothing is copied: W4 paid four times in one day
for the same number living in four files.

    python3 run_b.py gates | smoke | run
"""
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
W4DIR = os.path.join(ROOT, 'tmp', 'W4_SOLVER_2026-08-19')
W5DIR = os.path.join(ROOT, 'tmp', 'W5_WHY_2026-08-19')
for p in (ROOT, W4DIR, W5DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

import run_w4 as W4                                           # noqa: E402
import arms_config as AC                                      # noqa: E402
import arms_config_w5 as A5                                   # noqa: E402
import varpro                                                 # noqa: E402

BATCH = 'B_DEPLOY_2026-08-19'
OUT = os.path.join(HERE, 'results.jsonl')
SMOKE_OUT = os.path.join(HERE, 'smoke_results.jsonl')
SEEDS = A5.__dict__.get('SEEDS', (1595, 1596, 1597))
if not isinstance(SEEDS, tuple):
    SEEDS = W4.SEEDS

SPARSE_DIR = ['data_s9_sparse_2026-08-11']
CROWDED_DIR = ['data_s8_crowded_2026-08-10', 'data_s9_crowded_2026-08-11']

# ── arms.  `auto` everywhere: the router decides, we only change a default. ──────
ARMS = {
    'A-SHIP':     dict(kind='est', delta={},                          vp=0,
                       seeds=SEEDS, factor=()),
    'A-VP':       dict(kind='est', delta={'learned_init_iters': 0},   vp=A5.VP,
                       seeds=SEEDS, factor=('learned_init_iters', '__varpro__')),
    'A-SHIP-DUP': dict(kind='est', delta={},                          vp=0,
                       seeds=(SEEDS[0],), factor=(), dup_of='A-SHIP'),
    'ZERO':       dict(kind='zero', delta=None,                       vp=0, seeds=(None,),
                       factor=()),
}
ARM_ORDER = ('ZERO', 'A-SHIP', 'A-SHIP-DUP', 'A-VP')


def episodes():
    out = []
    for stratum, dirs in (('crowded', CROWDED_DIR), ('sparse', SPARSE_DIR)):
        for d in dirs:
            for p in sorted(glob.glob(os.path.join(d, '*.h5'))):
                b = os.path.basename(p)
                m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(.+)\.h5', b)
                # W4.run_zero reads ep['rel'] (path relative to ROOT); ours must match
                # its contract exactly, since the arm is imported, not reimplemented.
                out.append(dict(path=p, rel=p, key=b[len('episode_'):-3], stratum=stratum,
                                bf=m.group(1), ncomp=int(m.group(2)),
                                ep_seed=int(m.group(3)), corpus=d))
    return out


EPISODES = episodes()


def arm_cfg(arm, est_seed):
    """Shipping base config, then the arm's delta.  ⛔The router is NOT pinned."""
    cfg = dict(AC.base_config()) if hasattr(AC, 'base_config') else {}
    cfg = dict(AC.arm_config('L-SHIP', est_seed=est_seed))
    cfg.pop('init_method', None)          # <- the whole point: let the router decide
    cfg['seed'] = est_seed
    d = ARMS[arm]['delta'] or {}
    cfg.update(d)
    return cfg


def run_one(job):
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')
    arm, ep, seed = job['arm'], job['ep'], job['seed']
    spec = ARMS[arm]
    t0 = time.time()
    base = dict(batch=BATCH, job_id=job['job_id'], arm=arm, episode=ep['key'],
                corpus=ep['corpus'], stratum=ep['stratum'], bf=ep['bf'],
                ncomp=ep['ncomp'], seed=ep['ep_seed'], est_seed=seed,
                dup_of=spec.get('dup_of'), git_head=W4.HEAD, git_dirty_n=W4.DIRTY)
    try:
        W4.preload()
        dg, dgf = W4.code_fingerprint()
        if spec['kind'] == 'zero':
            body = W4.run_zero(ep)
        else:
            A5.register()
            cfg = arm_cfg(arm, seed)
            counters, undo_c = AC.install_counters()
            undo_t = AC.install_tracking_phase()
            undo_u = AC.install_updater_capture()
            if spec['vp']:
                varpro.install(steps=spec['vp'], weighting='confidence')
            if hasattr(varpro, 'reset_stats'):
                varpro.reset_stats()          # counters are per-row, not per-worker
            try:
                body = W4.run_est_with_config(ep, cfg, counters) \
                    if hasattr(W4, 'run_est_with_config') else _run_est(ep, cfg, counters)
            finally:
                if spec['vp']:
                    varpro.uninstall()
                for u in (undo_u, undo_t, undo_c):
                    try:
                        u()
                    except Exception:
                        pass
        row = dict(base, **body, code_md5=dg, code_files_n=len(dgf),
                   wall_s=time.time() - t0)
        return row
    except Exception as e:
        import traceback
        return dict(base, error=str(e), tb=traceback.format_exc()[-1500:],
                    wall_s=time.time() - t0)


def _run_est(ep, cfg, counters):
    """Minimal estimator call + scoring, mirroring W4.run_est but with a free router."""
    from wavesplat.run import run_estimation
    s = run_estimation(ep['path'], config=cfg, verbose=False)
    ec = s.get('effective_config') or {}
    route = (s.get('auto_selected') or ec.get('init_method') or s.get('init_method'))
    # Shells live per-FRAME (run.py:366) and are aggregated by the caller; reading only the
    # summary silently yields None.  This is the third time today the repository lost a
    # measurement to a key list that did not carry it (RMSE_near in run.py, again in
    # run_b1_full_comparison.py, and now here), so aggregate them explicitly.
    fr = s.get('frame_results') or []
    def _mean_shell(key):
        acc = {}
        for f in fr:
            for k, v in (f.get(key) or {}).items():
                if v is not None and v == v:
                    acc.setdefault('obs:' + k, []).append(v)
        return {k: float(np.mean(v)) for k, v in acc.items()} or None
    c = dict(counters) if counters is not None else {}
    return dict(avg_nRMSE=s.get('avg_nRMSE'), zero_nRMSE=s.get('zero_nRMSE'),
                n_match=s.get('n_frames_scored'), n_all=s.get('n_frames'),
                shell_rmse=_mean_shell('shell_rmse'),
                zero_shell=_mean_shell('zero_shell'),
                shell_frac=_mean_shell('shell_frac'),
                avg_RMSE_near=s.get('avg_RMSE_near'),
                route=route, effective_config=ec,
                init_time_s=s.get('init_seconds', s.get('init_time_s')),
                elapsed_ms_mean=s.get('avg_inference_ms'),
                n_atoms_final=s.get('final_n'),
                adam_steps_phase1=c.get('adam_steps_phase1'),
                adam_steps_grow=c.get('adam_steps_grow'),
                auto_real_gate=s.get('auto_real_gate'),
                vp_steps=varpro.stats().get('vp_steps'),
                vp_entries=varpro.stats().get('vp_entries'))


def build_jobs(arms=None, keys=None):
    jobs = []
    for arm in (arms or ARM_ORDER):
        for seed in ARMS[arm]['seeds']:
            for ep in EPISODES:
                if keys and ep['key'] not in keys:
                    continue
                jobs.append(dict(arm=arm, ep=ep, seed=seed,
                                 job_id='%s|%s|%s' % (arm, ep['key'], seed)))
    return jobs


def _load(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _pump(todo, sink, tag):
    n = len(todo)
    w = int(os.environ.get('WORKERS', '8'))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=w) as ex:
        for i, row in enumerate(ex.map(run_one, todo), 1):
            sink.write(json.dumps(row) + '\n')
            sink.flush()
            el = time.time() - t0
            print('[%s %d/%d] %-10s|%-38s|%s %s route=%s vp=%s (%.0fs, ETA %.0fmin)'
                  % (tag, i, n, row['arm'], row['episode'][:36], row.get('est_seed'),
                     ('%.6f' % row['avg_nRMSE']) if row.get('avg_nRMSE') is not None
                     else 'ERR', row.get('route'), row.get('vp_steps'),
                     row.get('wall_s', 0), (el / i) * (n - i) / 60), flush=True)


def gates():
    ok = True
    p = print
    p('%s GATES (run-free)' % BATCH)
    n_cr = sum(e['stratum'] == 'crowded' for e in EPISODES)
    n_sp = sum(e['stratum'] == 'sparse' for e in EPISODES)
    g1 = (n_cr == 64 and n_sp == 64)
    p('G-1 sample        : crowded=%d sparse=%d (want 64/64) -> %s'
      % (n_cr, n_sp, 'PASS' if g1 else 'FAIL'))
    ok &= g1
    A5.register()
    cfg = arm_cfg('A-VP', SEEDS[0])
    g2 = ('init_method' not in cfg) and cfg.get('learned_init_iters') == 0
    p('G-2 router free   : init_method absent=%s | learned_init_iters=%s -> %s'
      % ('init_method' not in cfg, cfg.get('learned_init_iters'), 'PASS' if g2 else 'FAIL'))
    ok &= g2
    base = arm_cfg('A-SHIP', SEEDS[0])
    diff = sorted(k for k in set(base) | set(cfg)
                  if base.get(k, '<a>') != cfg.get(k, '<a>'))
    g3 = (diff == ['learned_init_iters'])
    p('G-3 single delta  : A-VP vs A-SHIP cfg-delta=%s -> %s' % (diff, 'PASS' if g3 else 'FAIL'))
    ok &= g3
    p('G-4 expectations  : imported from arms_config_w5 (POLISH=%d, VP=%d) -- learned rows '
      'A-SHIP %d / A-VP %d ; v2_konly rows: vp must be 0'
      % (A5.POLISH_STEPS, A5.VP, A5.POLISH_STEPS + A5.SHIP_ITERS,
         A5.POLISH_STEPS + A5.VP))
    g5 = hasattr(varpro, 'install') and hasattr(AC, 'install_counters')
    p('G-5 modules       : varpro=%s arms_config=%s -> %s'
      % (hasattr(varpro, 'install'), hasattr(AC, 'install_counters'),
         'PASS' if g5 else 'FAIL'))
    ok &= g5
    jobs = build_jobs()
    p('\nplan: %d arms -> %d jobs over %d episodes' % (len(ARM_ORDER), len(jobs), len(EPISODES)))
    for a in ARM_ORDER:
        p('   %-11s seeds=%s  %d jobs' % (a, ARMS[a]['seeds'],
                                          len(ARMS[a]['seeds']) * len(EPISODES)))
    p('\nGATES %s' % ('PASS' if ok else 'FAIL -- halted'))
    return ok


def main():
    cmd = (sys.argv[1:] or ['gates'])[0]
    if cmd == 'gates':
        sys.exit(0 if gates() else 1)
    keys = None
    out = OUT
    if cmd == 'smoke':
        keys = {EPISODES[0]['key'], [e for e in EPISODES if e['stratum'] == 'sparse'][0]['key']}
        out = SMOKE_OUT
    W4.preload()
    A5.register()
    jobs = build_jobs(keys=keys)
    done = {r['job_id'] for r in _load(out) if 'error' not in r}
    todo = [j for j in jobs if j['job_id'] not in done]
    print('%s | %s | jobs %d | done %d | todo %d | workers %s'
          % (BATCH, cmd, len(jobs), len(done), len(todo), os.environ.get('WORKERS', '8')),
          flush=True)
    if not todo:
        print('nothing to do')
        return
    t0 = time.time()
    with open(out, 'a') as fh:
        _pump(todo, fh, cmd)
    print('DONE %.1f min' % ((time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
