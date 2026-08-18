#!/usr/bin/env python3
"""W4_SOLVER runner -- implements PREREG.md (self-attested freeze; md5 in FROZEN_md5.txt).

    python3 tmp/W4_SOLVER_2026-08-19/run_w4.py gates          # ZERO estimator runs, ZERO episode loads
    python3 tmp/W4_SOLVER_2026-08-19/run_w4.py smoke          # 2 episodes x 7 arms + the row-dependent gates
    WORKERS=8 python3 tmp/W4_SOLVER_2026-08-19/run_w4.py run  # 960 rows
    WORKERS=8 python3 tmp/W4_SOLVER_2026-08-19/run_w4.py run L-SHIP,L-SHIP-DUP   # arm subset

One row per (arm, episode, est_seed) appended to results.jsonl; **resumable** -- a job whose
non-error row is already present is skipped.  Smoke rows go to smoke_results.jsonl and never
mix with the verdict file.

WHAT THIS FILE OWNS
  - the sample (PREREG section 1: crowded only, n=64, ncomp in {15,30})
  - the arm x seed plan and the SINGLE-FACTOR audit of every treatment (PREREG section 2)
  - the per-arm-kind field contract (DCC, PREREG section 5) and its enforcement
  - the gates G-1..G-8 (G-9 is watchdog.py) and the parallel job plumbing

WHAT THIS FILE DELIBERATELY DOES NOT OWN -- and why the boundary is where it is
  - `ZERO` and `ORA-K-CVX`: **imported unmodified from tmp/W3_SOLVER_2026-08-15/run_w3.py**
    (`run_zero`, `run_cvx`, plus the shared `_episode_static` / `_zero_and_shape` /
    `_mean_shells` scoring helpers).  Nothing is copied, so the two reference arms are
    byte-identical to W3 by construction.  Gate G-8b pins run_w3.py's md5 to the value W3's
    own FROZEN_md5.txt recorded, and G-8c re-derives the shell edges from the imported
    module instead of trusting this file's copy of them.
  - the A-family configs and the intervention counters: `arms_config.py`.  Its counters hook
    the statements that do the work and -- crucially -- attribute the 50-step fine-tune
    inside `_adaptive_prune_grow` to `adam_steps_grow`, NOT to `adam_steps_phase1`.  A
    naive "count Adam steps until the tracking optimiser is built" would have reported
    `adam_steps_phase1 == 50` for L-ITER0 on every episode where growth fires, and G-2
    ("exactly 0") would have failed for the wrong reason.
  - the `L-VP` treatment: `varpro.py`.  This runner installs it, orders the install against
    the counters, and records what it reports.  A missing/broken varpro.py costs the L-VP
    arm error rows and nothing else.
  - G-9 progress supervision: `watchdog.py`, already written.  `gates` only checks that the
    watchdog's plan and this runner's plan are the same numbers.

DISCIPLINE NOTES
  - Every shell key carries the **`obs:` prefix**, for the SHIP-family arms as well as for
    the reference arms.  The repository is inconsistent here (S13's SHIP rows are bare, the
    baseline rows are prefixed) and `obs:` vs `ship:` differ by 1.8-4.1x on the same sample,
    so this runner writes ONE convention everywhere and the analysis must never strip it.
  - `effective_config` is READ BACK FROM THE LIVE UPDATER OBJECT (S13b lesson) via
    `arms_config.effective_config`, which also records `config_is_input_dict` as proof it is
    not the intent table.  The intent dict is stored separately as `intent_config`.
  - `code_md5` is ONE digest over a FIXED file set (import-order independent, S17 A9).
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
W3DIR = os.path.join(ROOT, 'tmp', 'W3_SOLVER_2026-08-15')
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, W3DIR)
os.chdir(ROOT)

BATCH = 'W4_SOLVER_2026-08-19'
OUT = os.path.join(HERE, 'results.jsonl')
SMOKE_OUT = os.path.join(HERE, 'smoke_results.jsonl')
FP_PATH = os.path.join(HERE, 'code_fingerprint.json')

# G-8b anchor: the md5 W3's own FROZEN_md5.txt records for run_w3.py.  If this fails the
# "unmodified reuse" claim for ZERO / ORA-K-CVX is void and the batch must halt.
W3_RUN_MD5 = 'b221ec29198ea096bb3e81041cbca18a'

# ── sample (PREREG section 1): crowded only, n=64, ncomp in {15,30} ───────────
CROWDED_DIRS = ('data_s8_crowded_2026-08-10', 'data_s9_crowded_2026-08-11')
NCOMP_KEEP = (15, 30)
N_EP_EXPECTED = 64

# ── estimator seeds.  Ledger [1592-1818]; W3 consumed 1592/1593/1594 ─────────
SEEDS = (1595, 1596, 1597)

# ── VARPRO step budget, FROZEN before any result (PREREG section 2) ──────────
VP_STEPS = 400

# ── shells (edges identical to wavesplat/run.py:358) ─────────────────────────
EDGES = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, np.inf]
BARE_KEYS = ['%d_%s' % (int(a), 'inf' if b == np.inf else int(b))
             for a, b in zip(EDGES[:-1], EDGES[1:])]
SHELL_KEYS = ['obs:' + k for k in BARE_KEYS]

HEAD = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                      text=True).stdout.strip()
DIRTY = len(subprocess.run(['git', 'status', '--porcelain'], capture_output=True,
                           text=True).stdout.strip().splitlines())


# ══════════════════════════════════════════════════════════════════════════════
#  ARM TABLE (PREREG section 2)
#
#  Static metadata only.  The A-family CONFIGS come from `arms_config.arm_config()` so
#  there is exactly one definition of each arm in the batch; `cfg_from` says which
#  arms_config arm supplies it.  `L-VP` has no arms_config entry on purpose: its config IS
#  L-SHIP's, and its single factor is the varpro install, not a config key.
# ══════════════════════════════════════════════════════════════════════════════
ARMS = {
    'L-SHIP':     dict(kind='est', cfg_from='L-SHIP',     vp=0,        seeds=SEEDS,
                       factor=()),
    # G-5 determinism replicate: same arms_config delta ({}), same seed as L-SHIP's first.
    'L-SHIP-DUP': dict(kind='est', cfg_from='L-SHIP-DUP', vp=0,        seeds=(SEEDS[0],),
                       factor=(), dup_of='L-SHIP'),
    'L-ITER0':    dict(kind='est', cfg_from='L-ITER0',    vp=0,        seeds=SEEDS,
                       factor=('learned_init_iters',)),
    'L-VP':       dict(kind='est', cfg_from='L-SHIP',     vp=VP_STEPS, seeds=SEEDS,
                       factor=('__varpro_install__',)),
    'L-MERGE':    dict(kind='est', cfg_from='L-MERGE',    vp=0,        seeds=SEEDS,
                       factor=('learned_merge_frac',)),
    # ---- AMENDMENT_1 A-4: the two arms that de-confound L-VP -------------------
    # L-STEP400 : same optimiser, 400 MORE steps  -> isolates "budget"
    # L-VP-NW   : VARPRO, UNWEIGHTED              -> isolates "weighting", and is the
    #             setting that reproduces the pre-exposure probe exactly.
    'L-STEP400':  dict(kind='est', cfg_from='L-STEP400',  vp=0,        seeds=SEEDS,
                       factor=('learned_init_iters',)),
    # AMENDMENT_1 A-4: against L-SHIP this arm moves TWO things (varpro install +
    # weighting), so its single-factor comparator is L-VP, not the control.  `factor_vs`
    # makes that explicit instead of letting the gate silently compare the wrong pair.
    'L-VP-NW':    dict(kind='est', cfg_from='L-SHIP',     vp=VP_STEPS, seeds=SEEDS,
                       factor=('vp_weighting',), factor_vs='L-VP',
                       vp_weighting='none'),
    'ZERO':       dict(kind='zero', cfg_from=None,        vp=0,        seeds=(None,),
                       factor=()),
    'ORA-K-CVX':  dict(kind='cvx',  cfg_from=None,        vp=0,        seeds=(None,),
                       factor=()),
}

# The IUT control and the two PRIMARY arms first, then the determinism probe, so a
# truncated run still answers P-1/P-2 and still knows whether it is allowed to.
ARM_ORDER = ('ZERO', 'L-SHIP', 'L-SHIP-DUP', 'L-VP', 'L-ITER0', 'L-VP-NW',
             'L-STEP400', 'L-MERGE', 'ORA-K-CVX')

SMOKE_KEYS = ('BF4_15comp_seed32_uniform_ra_rf_ds',   # s8 corpus, 15 comp
              'BF5_30comp_seed51_uniform_ra_rf_ds')   # s9 corpus, 30 comp; ALSO in W3's
                                                      # sample -> live ZERO/CVX bit anchor


# ══════════════════════════════════════════════════════════════════════════════
#  FIELD CONTRACT (DCC) -- PREREG section 5, SPLIT BY ARM KIND
#
#  S17 died on exactly this: `background_diag` was made mandatory for every arm, but the
#  stock filter class structurally cannot produce it, so the halting gate fired on an arm
#  that was not even a verdict arm.  Rule adopted here: a field is REQUIRED of an arm only
#  if that arm CAN produce it.  Every exemption below is a structural impossibility that
#  was checked against the code, not a convenience.  See recon/C3_runner.md for the
#  producibility table and for the two PREREG section 5 names that had to be re-mapped.
# ══════════════════════════════════════════════════════════════════════════════
REQUIRED_COMMON = (
    'batch', 'arm', 'episode', 'corpus', 'stratum', 'bf', 'ncomp', 'seed',
    'wall_s', 'avg_nRMSE', 'zero_nRMSE', 'n_match', 'n_all',
    'shell_rmse', 'zero_shell', 'shell_frac',
    'window_resid_rms', 'vp_steps', 'n_gt', 'effective_config',
    'git_head', 'git_dirty_n', 'code_md5', 'code_files_n', 'deterministic')

# Estimator arms only.  ZERO has no optimiser, no router and no atom pool; ORA-K-CVX is a
# closed-form lstsq with zero gradient steps and no learned proposal.  Demanding any of
# these of those two arms is the S17 self-kill.
REQUIRED_EST = ('est_seed', 'route', 'route_source', 'adam_steps_phase1',
                'adam_steps_grow', 'adam_steps_track', 'adam_steps_other',
                'merge_removed', 'merge_calls', 'n_atoms_final', 'counters',
                'intent_config', 'lambda_l1_intent', 'lambda_l1_effective',
                'init_method_intent', 'window_n_obs', 'gates_pass')
REQUIRED_CVX = ('n_atoms_final', 'cvx_n_obs', 'cvx_cond', 'cvx_rank',
                'cvx_window_resid_rms', 'cvx_truth_window_resid_rms',
                'cvx_gtomega_nrmse', 'cvx_omega_source')
REQUIRED_ZERO = ()

# Recorded for the deterministic arms but allowed to be null there, because "this arm has
# no estimator seed" is the truth about the arm, not a missing measurement.
NULLABLE_FOR_DET = ('est_seed',)


def missing_fields(row):
    """DCC violations of one row.  Used by `run_one` (stamped into the row), by the smoke,
    by `gates` on whatever rows exist, and by the analysis before any verdict is printed."""
    kind = (ARMS.get(row.get('arm')) or {}).get('kind')
    req = list(REQUIRED_COMMON)
    req += {'est': list(REQUIRED_EST), 'cvx': list(REQUIRED_CVX),
            'zero': list(REQUIRED_ZERO)}.get(kind, [])
    miss = []
    for k in req:
        if k not in row:
            miss.append(k)
        elif row[k] is None and not (kind in ('zero', 'cvx') and k in NULLABLE_FOR_DET):
            miss.append(k + '=null')
    for fld in ('shell_rmse', 'zero_shell', 'shell_frac'):
        d = row.get(fld)
        if not isinstance(d, dict):
            miss.append(fld + '{}')
        else:
            miss += ['%s[%s]' % (fld, k) for k in SHELL_KEYS if k not in d]
    return miss


# ── sample construction ───────────────────────────────────────────────────────
def episodes():
    out = []
    for d in CROWDED_DIRS:
        for p in sorted(glob.glob(os.path.join(ROOT, d, '*.h5'))):
            b = os.path.basename(p)
            m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(.+)\.h5', b)
            if not m or int(m.group(2)) not in NCOMP_KEEP:
                continue
            out.append(dict(rel=os.path.join(d, b), corpus=d,
                            key=b[len('episode_'):-3], stratum='crowded',
                            bf=m.group(1), ncomp=int(m.group(2)),
                            ep_seed=int(m.group(3)), suffix=m.group(4)))
    return out


EPISODES = episodes()


# ── code fingerprint (G-8); fixed file set, import-order independent ─────────
def _md5(path):
    with open(path, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


def code_fingerprint():
    files = {}
    for pat in ('wavesplat/**/*.py', 'wavesim/**/*.py'):
        for f in sorted(glob.glob(os.path.join(ROOT, pat), recursive=True)):
            files[os.path.relpath(f, ROOT)] = _md5(f)
    for extra in ('configs/estimation.yaml', 'configs/default.yaml',
                  'configs/server.yaml'):
        p = os.path.join(ROOT, extra)
        if os.path.exists(p):
            files[extra] = _md5(p)
    # the treatment definition: this runner, the frozen prereg, the two modules the other
    # agents own, the watchdog, and the reused W3 runner
    for p in ([os.path.join(HERE, n) for n in
               ('run_w4.py', 'PREREG.md', 'arms_config.py', 'varpro.py', 'watchdog.py')]
              + [os.path.join(W3DIR, 'run_w3.py')]):
        if os.path.exists(p):
            files[os.path.relpath(p, ROOT)] = _md5(p)
    return hashlib.md5(json.dumps(files, sort_keys=True).encode()).hexdigest(), files


def preload():
    """Import every dependency BEFORE any job so `code_md5` is ONE digest (G-8)."""
    import wavesplat.run                            # noqa: F401
    import wavesplat.data_loader                    # noqa: F401
    import wavesplat.evaluation.metrics             # noqa: F401
    import wavesplat.optimizer.sequential_update    # noqa: F401
    import wavesplat.optimizer.coarse_to_fine       # noqa: F401
    import wavesplat.optimizer.initializer          # noqa: F401
    import wavesplat.optimizer.learned_init         # noqa: F401
    import wavesplat.model.spectral_gaussian        # noqa: F401
    import wavesplat.model.tnse                     # noqa: F401
    import torch                                    # noqa: F401
    import run_w3                                   # noqa: F401  (ZERO / ORA-K-CVX)
    for opt in ('arms_config', 'varpro'):
        try:
            __import__(opt)
        except Exception:
            pass


def w3mod():
    import run_w3
    return run_w3


def _opt_mod(name):
    """Import a sibling module written by another agent.  Absent or broken must cost the
    arms that need it, never the batch."""
    try:
        return __import__(name)
    except Exception:
        return None


def arm_cfg(arm, seed):
    """The config `run_estimation` will actually see for (arm, seed), MERGED over the
    shipped DEFAULT_CONFIG exactly as `run.py:139-141` merges it."""
    ac = _opt_mod('arms_config')
    if ac is None:
        raise RuntimeError('arms_config.py is not importable -- estimator arms cannot run')
    from wavesplat.run import DEFAULT_CONFIG
    merged = dict(DEFAULT_CONFIG)
    merged.update(ac.arm_config(ARMS[arm]['cfg_from'], est_seed=seed))
    return merged


# ── window residual for the deterministic arms ───────────────────────────────
def _zero_window_resid(loader):
    """The ZERO field's Phase-1 window residual is just the rms of the observations.
    Computed here (not inside the imported W3 `run_zero`, which is reused unmodified) so
    `window_resid_rms` exists on every arm and D-1 has its reference scale."""
    z = [f.z_obs.cpu().numpy() for f in loader.get_init_window(tp_factor=2.0)
         if len(f.z_obs)]
    z = np.concatenate(z) if z else np.zeros(1)
    return float(np.sqrt(np.mean(z ** 2))), int(z.size)


def _n_all(rel):
    """Total frames in the episode -- the denominator `results/b1_full_comparison.json`
    lacked, which is why 100/100 of its episodes scored the two arms on different frames
    and the whole file became uncitable.  Read straight from the file, no loader."""
    import h5py
    with h5py.File(os.path.join(ROOT, rel), 'r') as fh:
        return int(fh['frames/timestamps'].shape[0])


# ══════════════════════════════════════════════════════════════════════════════
#  arm: estimator (L-SHIP / L-SHIP-DUP / L-ITER0 / L-VP / L-MERGE)
#
#  PATCH ORDER IS LOAD-BEARING.  varpro is installed FIRST, arms_config's counters SECOND,
#  so that arms_config's `_initialize_learned` wrapper encloses varpro's -- which puts the
#  400 VARPRO Adam steps inside the `p1` phase.  Installed the other way round they would
#  land in `adam_steps_other`, which `arms_config.check_gates` reports as "unattributed
#  Adam steps" and the L-VP arm would fail a gate it does not violate.  Undo is strict
#  LIFO in the `finally`.
# ══════════════════════════════════════════════════════════════════════════════
def run_est(ep, arm, seed, spec):
    from wavesplat.run import run_estimation
    ac = _opt_mod('arms_config')
    if ac is None:
        raise RuntimeError('arms_config.py is not importable -- estimator arms cannot run')
    w3 = w3mod()

    cfg = arm_cfg(arm, seed)
    assert cfg.get('init_method') == 'learned', cfg.get('init_method')
    assert 'lambda_l1' in cfg, 'lambda_l1 must never be implicit (W3 discipline)'
    assert cfg.get('seed') == seed, (cfg.get('seed'), seed)

    loader, gtc = w3._episode_static(ep['rel'])
    zero_n, zero_sh, frac, n_fr, *_ = w3._zero_and_shape(loader)

    # env: fully rewritten so a previous job in this worker cannot leak a gate
    for k in ('RX2_GROW_SUBTRACT', 'RX2_GROW_RECOMPUTE', 'RX2_GROW_TIGHT_SUPPRESS',
              'RX2_GROW_FITPHI', 'S4_ORACLE_WPHI', 'S2_ALPHA',
              'S8_L1_TRACK_ONLY', 'S8_L1_PHASE1_ONLY'):
        os.environ[k] = '0'
    # S5_WINDOW=1 turns on the Phase-1 TRAINING-WINDOW residual in run.py.  It is a
    # read-only diagnostic under torch.no_grad() that consumes no RNG, and it is on for
    # EVERY estimator arm -- uniform, with G-5 (|delta|=0 vs L-SHIP-DUP) as the check that
    # it perturbed nothing.  Without it `window_resid_rms` is NaN and D-1 ("does a solver
    # gain reach the field?") cannot be measured at all.
    os.environ['S5_WINDOW'] = '1'

    vp = None
    vp_desc = None
    if spec['vp']:
        vp = _opt_mod('varpro')
        if vp is None:
            raise RuntimeError('L-VP requires varpro.py (not importable)')
        vp.reset_stats()
        vp_desc = vp.install(steps=spec['vp'],
                             weighting=spec.get('vp_weighting', 'confidence'))
    counters, undo_counters = ac.install_counters()     # 2nd: encloses varpro's wrapper
    undo_track = ac.install_tracking_phase()
    undo_cap = ac.install_updater_capture()
    try:
        r = run_estimation(os.path.join(ROOT, ep['rel']), cfg, verbose=False)
        eff = ac.effective_config(input_cfg=cfg)
        snap = ac.counters_snapshot()
        vstat = vp.stats() if vp is not None else {}
    finally:                                            # strict LIFO
        undo_cap()
        undo_track()
        undo_counters()
        if vp is not None:
            vp.uninstall()
        os.environ['S5_WINDOW'] = '0'

    fr_ = r['frame_results']
    nr = np.array([f['nRMSE'] for f in fr_ if 'nRMSE' in f], float)
    rho = np.array([f['Pearson_rho'] for f in fr_ if 'Pearson_rho' in f], float)
    vr = np.array([f['Variance_Ratio'] for f in fr_ if 'Variance_Ratio' in f], float)
    with np.errstate(invalid='ignore'):
        decorr = 1.0 - rho ** 2
        shrink = (np.sqrt(np.clip(vr, 0, None)) - rho) ** 2

    sh = {k: [] for k in SHELL_KEYS}
    for f in fr_:
        d = f.get('shell_rmse') or {}
        for bare, k in zip(BARE_KEYS, SHELL_KEYS):
            sh[k].append(d.get(bare, np.nan))

    def _m(key):
        v = [f[key] for f in fr_ if key in f and np.isfinite(f.get(key, np.nan))]
        return float(np.mean(v)) if v else float('nan')

    vp_calls = (vstat.get('vp_calls') or [])
    return dict(
        avg_nRMSE=float(np.nanmean(nr)), zero_nRMSE=zero_n,
        n_match=len(fr_), n_all=_n_all(ep['rel']), n_frames_scored=len(fr_),
        shell_rmse=w3._mean_shells(sh), zero_shell=zero_sh, shell_frac=frac,
        n_gt=int(len(gtc)),
        # ---- intervention counters (G-1..G-5), from arms_config -------------
        route=eff.get('init_method'), route_source='object-readback',
        learned_init_calls=int(snap.get('learned_init_calls', 0)),
        adam_steps_phase1=int(snap.get('adam_steps_phase1', 0)),
        adam_steps_grow=int(snap.get('adam_steps_grow', 0)),
        adam_steps_track=int(snap.get('adam_steps_track', 0)),
        adam_steps_other=int(snap.get('adam_steps_other', 0)),
        merge_removed=int(snap.get('merge_removed', 0)),
        merge_calls=int(snap.get('merge_calls', 0)),
        counters=snap,
        **ac.check_gates(ARMS[arm]['cfg_from'], snap),
        # ---- L-VP treatment counters (G-3) ----------------------------------
        vp_steps=int(vstat.get('vp_steps', 0) or 0),
        vp_entries=int(vstat.get('vp_entries', 0) or 0),
        vp_installed=bool(spec['vp']), vp_describe=vp_desc,
        vp_diag=(vp_calls[0] if vp_calls else None),
        vp_stats={k: v for k, v in vstat.items() if k != 'vp_calls'},
        # ---- atoms and the Phase-1 window residual (D-1, D-3) ---------------
        n_atoms_final=(int(r['n_gaussians']) if r.get('n_gaussians') is not None else None),
        window_resid_rms=float(r.get('window_resid', float('nan'))),
        window_n_obs=int(r.get('window_n_obs', 0)),
        # ---- config: intent AND object read-back, never conflated -----------
        intent_config={k: (list(v) if isinstance(v, (list, tuple)) else v)
                       for k, v in cfg.items() if k != 'learned_atoms'},
        effective_config=eff,
        init_method_intent=cfg['init_method'],
        lambda_l1_intent=float(cfg['lambda_l1']),
        lambda_l1_effective=eff.get('lambda_l1'),
        auto_selected=r.get('auto_selected'), init_iters_reported=r.get('init_iters'),
        # ---- descriptives (D family) ----------------------------------------
        decorr_mean=float(np.nanmean(decorr)) if rho.size else float('nan'),
        shrink_mean=float(np.nanmean(shrink)) if rho.size else float('nan'),
        obs_resid_rms_mean=_m('obs_resid_rms'), elapsed_ms_mean=_m('elapsed_ms'),
        init_nrmse=float(r.get('init_nrmse', float('nan'))),
        init_time_s=float(r.get('init_time_s', float('nan'))),
        Hs=float(loader.Hs_true), Tp=float(loader.Tp_true), depth=float(loader.depth))


# ── arms: ZERO / ORA-K-CVX -- W3 definitions, imported unmodified ────────────
def run_zero(ep):
    w3 = w3mod()
    body = w3.run_zero(ep)
    loader, _ = w3._episode_static(ep['rel'])
    wr, wn = _zero_window_resid(loader)
    body.update(n_match=body['n_frames_scored'], n_all=_n_all(ep['rel']),
                window_resid_rms=wr, window_n_obs=wn,
                vp_steps=0, vp_installed=False, n_atoms_final=None)
    return body


def run_cvx(ep):
    w3 = w3mod()
    body = w3.run_cvx(ep)
    body.update(n_match=body['n_frames_scored'], n_all=_n_all(ep['rel']),
                window_resid_rms=float(body['cvx_window_resid_rms']),
                window_n_obs=int(body['cvx_n_obs']),
                n_atoms_final=int(body['n_gt']),
                vp_steps=0, vp_installed=False)
    return body


# ── job plumbing ──────────────────────────────────────────────────────────────
def build_jobs(arms=None, keys=None):
    jobs = []
    for arm in (arms or ARM_ORDER):
        spec = ARMS[arm]
        for ep in EPISODES:
            if keys and ep['key'] not in keys:
                continue
            for s in spec['seeds']:
                jobs.append(dict(job_id='%s|%s|%s' % (arm, ep['key'], s),
                                 arm=arm, ep=ep, seed=s))
    return jobs


def run_one(job):
    sys.path[:0] = [ROOT, HERE, W3DIR]
    # one BLAS thread per worker: parallelism must move the clock, never a number
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')
    arm, ep, seed = job['arm'], job['ep'], job['seed']
    spec = ARMS[arm]
    kind = spec['kind']
    t0 = time.time()
    base = dict(batch=BATCH, job_id=job['job_id'], arm=arm, episode=ep['key'],
                corpus=ep['corpus'], stratum=ep['stratum'], bf=ep['bf'],
                ncomp=ep['ncomp'], seed=ep['ep_seed'], est_seed=seed,
                cfg_from=spec['cfg_from'], dup_of=spec.get('dup_of'),
                deterministic=(kind != 'est'), git_head=HEAD, git_dirty_n=DIRTY)
    try:
        preload()
        dg, dgf = code_fingerprint()
        body = (run_est(ep, arm, seed, spec) if kind == 'est'
                else run_cvx(ep) if kind == 'cvx' else run_zero(ep))
        row = dict(base, **body, code_md5=dg, code_files_n=len(dgf),
                   wall_s=time.time() - t0)
        row['dcc_missing'] = missing_fields(row)
        return row
    except Exception as e:
        import traceback
        dg, dgf = code_fingerprint()
        return dict(base, error=repr(e)[:300], tb=traceback.format_exc()[-2000:],
                    code_md5=dg, code_files_n=len(dgf), wall_s=time.time() - t0)


def _load_rows(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  GATES (PREREG section 4).  Everything decidable WITHOUT loading an episode is
#  decided here; the row-dependent halves are re-checked by `posthoc()` on whatever
#  rows exist -- reported as PENDING when there are none, never as PASS.
# ══════════════════════════════════════════════════════════════════════════════
def _g7_phase_identity():
    """G-7: the phase convention, as an identity, with zero runs.

    Forward model `w * cos(kx x + ky y - w t + phi)` (spectral_gaussian.py:6), so a
    closed-form (a, b) pair converts as w = hypot(a, b), phi = **atan2(-b, a)**
    (model/tnse.py:216).  `sequential_update.py:1335` uses `arctan2(b, a)` -- the sign flip
    W1 and W2 independently found -- so any L-VP reusing that helper is wrong before it
    starts.  Checked numerically here on random draws, and on `varpro.ab_to_wphi` itself
    when that module is importable (numpy first, torch fallback: varpro's is a torch fn).
    """
    rng = np.random.RandomState(0)
    a, b = rng.randn(4096), rng.randn(4096)
    th = rng.uniform(-20, 20, 4096)
    target = a * np.cos(th) + b * np.sin(th)

    def err(fn):
        try:
            w, phi = fn(a, b)
            w, phi = np.asarray(w, float), np.asarray(phi, float)
        except Exception:
            import torch
            w, phi = fn(torch.tensor(a), torch.tensor(b))
            w, phi = w.detach().numpy(), phi.detach().numpy()
        return float(np.max(np.abs(w * np.cos(th + phi) - target)))

    ref = err(lambda a_, b_: (np.hypot(a_, b_), np.arctan2(-b_, a_)))
    bad = err(lambda a_, b_: (np.hypot(a_, b_), np.arctan2(b_, a_)))
    vp = _opt_mod('varpro')
    got = err(vp.ab_to_wphi) if (vp is not None and hasattr(vp, 'ab_to_wphi')) else None
    return ref, bad, got


def gates(verbose=True):
    ok = True
    p = print if verbose else (lambda *a, **k: None)

    # ---- G-1: sample + router pinned to 'learned' for the whole A family -----
    missing = [e['rel'] for e in EPISODES
               if not os.path.exists(os.path.join(ROOT, e['rel']))]
    dup = len({e['key'] for e in EPISODES}) != len(EPISODES)
    ncomp = sorted({e['ncomp'] for e in EPISODES})
    strat = sorted({e['stratum'] for e in EPISODES})
    g1a = (not missing and not dup and len(EPISODES) == N_EP_EXPECTED
           and ncomp == [15, 30] and strat == ['crowded'])
    p('G-1 sample       : n=%d (want %d) ncomp=%s strata=%s dup=%s missing=%d -> %s'
      % (len(EPISODES), N_EP_EXPECTED, ncomp, strat, dup, len(missing),
         'PASS' if g1a else 'FAIL'))
    if missing:
        p('    missing:', missing[:4])
    ok &= g1a

    a_arms = [a for a in ARM_ORDER if ARMS[a]['kind'] == 'est']
    try:
        cfgs = {a: arm_cfg(a, SEEDS[0]) for a in a_arms}
    except Exception as e:
        p('G-1b configs     : arms_config unusable (%s) -> FAIL' % repr(e)[:100])
        cfgs = {}
        ok = False
    if cfgs:
        routes = sorted({c['init_method'] for c in cfgs.values()})
        g1b = (routes == ['learned'])
        p('G-1b route pinned: A-family init_method=%s on %d arms -> %s   '
          '(constructive: make_updater gives the plain SequentialUpdater for every explicit '
          'init_method, so the router is REMOVED, not sampled -- scope limit, PREREG 7.8)'
          % (routes, len(cfgs), 'PASS' if g1b else 'FAIL'))
        ok &= g1b

        # shipped-default guard -- the lambda_l1 failure mode, as a gate
        want = {'lambda_l1': 1e-4, 'tracking_mode': 'w_phi', 'tracking_lr': 0.005,
                'max_n': 30, 'adaptive_n': True, 'use_dispersion': True,
                'learned_konly': False, 'init_k_lr_scale': 1.0}
        ship = cfgs['L-SHIP']
        off = {k: (ship.get(k), v) for k, v in want.items() if ship.get(k) != v}
        p('G-1c shipped defs: %s -> %s'
          % ('all %d match' % len(want) if not off else 'DRIFT %s' % off,
             'PASS' if not off else 'FAIL'))
        ok &= not off

        # ---- G-2/G-3/G-4: SINGLE-FACTOR property, decided before any run ----
        p('G-2/3/4 single-factor (merged config vs the arm\'s comparator, seed %d):'
          % SEEDS[0])
        sf = True
        for arm in ('L-ITER0', 'L-VP', 'L-VP-NW', 'L-STEP400', 'L-MERGE',
                    'L-SHIP-DUP'):
            spec = ARMS[arm]
            base_arm = spec.get('factor_vs', 'L-SHIP')
            base = cfgs[base_arm]
            c = cfgs[arm]
            diff = sorted(k for k in set(c) | set(base)
                          if c.get(k, '<absent>') != base.get(k, '<absent>'))
            # varpro-level knobs live outside the estimator config, so fold them in
            for knob, dflt in (('vp', 0), ('vp_weighting', 'confidence')):
                if spec.get(knob, dflt) != ARMS[base_arm].get(knob, dflt):
                    diff.append('__varpro_install__' if knob == 'vp' else knob)
            diff = sorted(diff)
            want_f = sorted(spec['factor'])
            good = (diff == want_f)
            sf &= good
            p('    %-11s vs %-9s delta=%-30s declared=%-26s -> %s'
              % (arm, base_arm, str(diff or '(none)'), str(want_f or '(none)'),
                 'PASS' if good else 'FAIL'))
        p('    run-time counters that decide it: adam_steps_phase1 (L-ITER0 == 0, '
          'L-SHIP > 0; the prune/grow fine-tune is attributed to adam_steps_grow so a '
          '0-iteration arm is not credited with 50 steps) | vp_steps (L-VP == %d, else 0) '
          '| merge_removed (L-MERGE median > 0)' % VP_STEPS)
        ok &= sf

        # ---- G-5: the determinism probe is the SAME configuration -----------
        same = all(cfgs['L-SHIP-DUP'] == arm_cfg('L-SHIP', s) for s in (SEEDS[0],))
        g5 = same and ARMS['L-SHIP-DUP']['seeds'][0] == SEEDS[0]
        p('G-5 dup probe    : L-SHIP-DUP config == L-SHIP config = %s, seed=%s '
          '(= L-SHIP seed #1) -> %s  [numeric |delta|=0 check runs on rows]'
          % (same, ARMS['L-SHIP-DUP']['seeds'][0], 'PASS' if g5 else 'FAIL'))
        ok &= g5

    # ---- G-6: field contract, per arm kind ----------------------------------
    p('G-6 field contract (DCC), by ARM KIND -- S17 died making one contract universal:')
    p('    common (%d): %s' % (len(REQUIRED_COMMON), ', '.join(REQUIRED_COMMON)))
    p('    est   +(%d): %s' % (len(REQUIRED_EST), ', '.join(REQUIRED_EST)))
    p('    cvx   +(%d): %s' % (len(REQUIRED_CVX), ', '.join(REQUIRED_CVX)))
    p('    zero  +(%d): %s' % (len(REQUIRED_ZERO), ', '.join(REQUIRED_ZERO) or '(none)'))
    p('    EXEMPT for ZERO and ORA-K-CVX (structurally unproducible, not omitted): '
      'route, route_source, adam_steps_*, merge_*, counters, intent_config, '
      'lambda_l1_*, init_method_intent, window_n_obs, gates_pass.')
    p('    EXEMPT for ZERO additionally: n_atoms_final (the arm has no atoms at all).')
    p('    est_seed: required non-null on est arms, present-but-null on the two '
      'deterministic arms.')
    p('    shells required on every arm (obs: prefix, both families): %s'
      % ' '.join(SHELL_KEYS))
    g6 = (not set(REQUIRED_COMMON) & set(REQUIRED_EST)
          and not set(REQUIRED_COMMON) & set(REQUIRED_CVX))
    p('    contract sets disjoint from common -> %s' % ('PASS' if g6 else 'FAIL'))
    ok &= g6

    # ---- G-7: phase convention identity (numeric, run-free) ----------------
    ref, bad, got = _g7_phase_identity()
    g7 = ref <= 1e-12 and (got is None or got <= 1e-12)
    p('G-7 phase identity: atan2(-b,a) max|d|=%.3e (tol 1e-12) | the sequential_update.py'
      ':1335 convention atan2(+b,a) gives %.3e | varpro.ab_to_wphi max|d|=%s -> %s'
      % (ref, bad, ('%.3e' % got) if got is not None else 'module absent',
         'PASS' if g7 else 'FAIL'))
    ok &= g7

    # ---- G-8: code fingerprint, one digest ---------------------------------
    dg1, files = code_fingerprint()
    dg2, _ = code_fingerprint()
    g8 = dg1 == dg2
    p('G-8 fingerprint  : digest=%s files=%d stable=%s (git %s dirty %d) -> %s'
      % (dg1, len(files), g8, HEAD[:8], DIRTY, 'PASS' if g8 else 'FAIL'))
    with open(FP_PATH, 'w') as fh:
        json.dump({'digest': dg1, 'git_head': HEAD, 'git_dirty_n': DIRTY,
                   'files': files}, fh, indent=1)
    ok &= g8

    # ---- G-8b: the reused W3 arms are byte-identical to the frozen W3 ------
    w3p = os.path.join(W3DIR, 'run_w3.py')
    got_md5 = _md5(w3p) if os.path.exists(w3p) else None
    g8b = (got_md5 == W3_RUN_MD5)
    p('G-8b W3 reuse    : run_w3.py md5=%s vs W3 FROZEN_md5.txt %s -> %s  '
      '(ZERO / ORA-K-CVX are IMPORTED from it, never copied)'
      % (got_md5, W3_RUN_MD5, 'PASS' if g8b else 'FAIL'))
    ok &= g8b

    # ---- G-8c: the shells this file declares == the shells W3 scores with --
    try:
        w3 = w3mod()
        g8c = (list(w3.SHELL_KEYS) == SHELL_KEYS and list(w3.BARE_KEYS) == BARE_KEYS
               and all(k.startswith('obs:') for k in SHELL_KEYS))
        p('G-8c shell parity: run_w4 SHELL_KEYS == run_w3 SHELL_KEYS and all `obs:` '
          'prefixed = %s -> %s' % (g8c, 'PASS' if g8c else 'FAIL'))
        ok &= g8c
    except Exception as e:
        p('G-8c shell parity: run_w3 not importable (%s) -> FAIL' % repr(e)[:80])
        ok = False

    # ---- G-9: the watchdog exists; check only that the plans agree ---------
    jobs = build_jobs()
    try:
        import watchdog
        wplan = {a: n for a, n in watchdog.PLAN}
        mine = {a: len(ARMS[a]['seeds']) for a in ARM_ORDER}
        g9 = (wplan == mine and watchdog.N_EP == len(EPISODES)
              and watchdog.TOTAL == len(jobs))
        p('G-9 watchdog     : plan match=%s  N_EP %d/%d  TOTAL %d/%d -> %s   '
          '(progress supervision is watchdog.py -- file state only, never the process '
          'table; NOT reimplemented here)'
          % (wplan == mine, watchdog.N_EP, len(EPISODES), watchdog.TOTAL, len(jobs),
             'PASS' if g9 else 'FAIL'))
        if not g9:
            p('    watchdog=%s\n    runner  =%s' % (wplan, mine))
        ok &= g9
    except Exception as e:
        p('G-9 watchdog     : not importable (%s) -> FAIL' % repr(e)[:80])
        ok = False

    # ---- seed ledger (not a PREREG gate; kept so the ledger stays honest) --
    used = set()
    for path in glob.glob(os.path.join(ROOT, 'tmp', '**', '*.jsonl'), recursive=True):
        if path.startswith(HERE):
            continue
        try:
            for line in open(path, errors='ignore'):
                for v in re.findall(r'"(?:seed|est_seed|filter_seed)"\s*:\s*(\d+)', line):
                    if 1500 <= int(v) <= 1900:
                        used.add(int(v))
        except OSError:
            pass
    clash = sorted(set(SEEDS) & used)
    p('SEED ledger      : consume=%s max_used=%s clash=%s -> %s'
      % (list(SEEDS), max(used) if used else None, clash,
         'PASS' if not clash else 'FAIL'))
    ok &= not clash

    # ---- sibling-module status --------------------------------------------
    ac = _opt_mod('arms_config')
    if ac is None:
        p('MOD arms_config  : ABSENT -> every estimator arm will error out')
        ok = False
    else:
        mine_a = {a for a in ARMS if ARMS[a]['kind'] == 'est'}
        theirs = set(getattr(ac, 'ARM_DELTAS', {}))
        # AMENDMENT_1 A-4: an arm whose factor is the varpro install (or a varpro knob)
        # carries NO estimator-config delta, so it borrows another arm's entry via
        # `cfg_from`.  That is legitimate coverage -- what must never happen is an arm
        # with no entry AND no borrow.  Previously only `L-VP` was whitelisted by name,
        # which made every future varpro-side arm fail a gate it does not violate.
        borrowers = {a for a in mine_a
                     if a not in theirs and ARMS[a].get('cfg_from') in theirs}
        uncovered = mine_a - theirs - borrowers
        p('MOD arms_config  : present; ARM_DELTAS=%s | varpro-side arms borrowing a '
          'config: %s | est arms not covered: %s'
          % (sorted(theirs),
             sorted('%s->%s' % (a, ARMS[a]['cfg_from']) for a in borrowers),
             sorted(uncovered)))
        ok &= not uncovered
    vpm = _opt_mod('varpro')
    if vpm is None:
        p('MOD varpro       : ABSENT -> L-VP jobs become error rows; every other arm is '
          'unaffected')
        ok = False
    else:
        iface = [n for n in ('install', 'uninstall', 'stats', 'reset_stats',
                             'ab_to_wphi') if hasattr(vpm, n)]
        need = {'install', 'uninstall', 'stats', 'reset_stats'}
        g_vp = need <= set(iface)
        p('MOD varpro       : present; interface=%s -> %s  (runner installs varpro FIRST '
          'and arms_config counters SECOND so the %d VARPRO Adam steps land in the p1 '
          'phase, not in adam_steps_other)'
          % (iface, 'PASS' if g_vp else 'FAIL (missing %s)' % sorted(need - set(iface)),
             VP_STEPS))
        ok &= g_vp

    # ---- plan ---------------------------------------------------------------
    p('\nplan: %d arms -> %d jobs over %d episodes' % (len(ARM_ORDER), len(jobs),
                                                       len(EPISODES)))
    for a in ARM_ORDER:
        p('   %-11s %-5s cfg_from=%-11s seeds=%-20s %3d jobs'
          % (a, ARMS[a]['kind'], str(ARMS[a]['cfg_from']), str(ARMS[a]['seeds']),
             len(build_jobs(arms=[a]))))

    ok &= posthoc(_load_rows(OUT), 'results.jsonl', verbose=verbose)
    p('\nGATES(run-free) %s' % ('PASS' if ok else 'FAIL -- batch halted'))
    return ok


def posthoc(rows, label, verbose=True):
    """Row-dependent halves of G-2..G-8, on rows that exist.  0 rows -> PENDING, never
    PASS: the discipline is that an unexecuted check is not a passed check."""
    p = print if verbose else (lambda *a, **k: None)
    good = [r for r in rows if 'error' not in r]
    p('\n--- post-hoc gates on %s: %d rows (%d errors) ---'
      % (label, len(rows), len(rows) - len(good)))
    if not good:
        p('    0 usable rows -> G-2/G-3/G-4/G-5/G-6 PENDING (not PASS)')
        return True
    ok = True
    by = {}
    for r in good:
        by.setdefault(r['arm'], []).append(r)

    def med(arm, key):
        v = [r[key] for r in by.get(arm, []) if isinstance(r.get(key), (int, float))]
        return float(np.median(v)) if v else None

    # G-1 post-hoc: every estimator row actually took the learned branch
    est = [r for r in good if ARMS.get(r['arm'], {}).get('kind') == 'est']
    if est:
        bad = [(r['arm'], r['episode'], r.get('route'), r.get('learned_init_calls'))
               for r in est if r.get('route') != 'learned'
               or r.get('learned_init_calls') != 1]
        p('G-1 route/entry     : %d/%d est rows route==learned AND '
          'learned_init_calls==1 -> %s'
          % (len(est) - len(bad), len(est), 'PASS' if not bad else 'FAIL'))
        for b in bad[:3]:
            p('    ', b)
        ok &= not bad

    # G-2 adam_steps_phase1
    a0 = [r['adam_steps_phase1'] for r in by.get('L-ITER0', [])]
    a1 = [r['adam_steps_phase1'] for r in by.get('L-SHIP', [])]
    if a0 or a1:
        g2 = ((all(v == 0 for v in a0) if a0 else True)
              and (all(v > 0 for v in a1) if a1 else True))
        p('G-2 adam_steps_phase1: L-ITER0 distinct=%s (n=%d, all must be 0) | L-SHIP '
          'median=%s (n=%d, all must be >0) | L-ITER0 adam_steps_grow median=%s '
          '(attributed away on purpose) -> %s'
          % (sorted(set(a0)), len(a0), med('L-SHIP', 'adam_steps_phase1'), len(a1),
             med('L-ITER0', 'adam_steps_grow'), 'PASS' if g2 else 'FAIL'))
        ok &= g2

    # G-3 vp_steps
    vps = {a: sorted({r.get('vp_steps') for r in v}) for a, v in by.items()}
    g3 = all((s == [VP_STEPS]) if a == 'L-VP' else (s == [0]) for a, s in vps.items())
    p('G-3 vp_steps        : %s -> %s' % (vps, 'PASS' if g3 else 'FAIL'))
    ok &= g3

    # G-4 merge_removed -- soft: failure downgrades S-1 only (PREREG section 4)
    if 'L-MERGE' in by:
        mm, mc = med('L-MERGE', 'merge_removed'), med('L-MERGE', 'merge_calls')
        others = {a: sorted({r.get('merge_removed') for r in v})
                  for a, v in by.items() if a not in ('L-MERGE', 'ZERO', 'ORA-K-CVX')}
        p('G-4 merge_removed   : L-MERGE median=%s (merge_calls median=%s, must be >0) | '
          'other est arms %s -> %s   [failure downgrades S-1 only, never the batch]'
          % (mm, mc, others,
             'PASS' if (mm or 0) > 0 else 'S-1 UNDECIDED-NO-INTERVENTION'))

    # unattributed Adam steps (arms_config alarm)
    ua = [(r['arm'], r['episode'], r['adam_steps_other']) for r in est
          if r.get('adam_steps_other')]
    p('G-2b unattributed   : %d est rows with adam_steps_other>0 -> %s'
      % (len(ua), 'PASS' if not ua else 'INVESTIGATE'))
    for u in ua[:3]:
        p('    ', u)

    # G-5 duplicate probe
    ship = {r['episode']: r['avg_nRMSE'] for r in by.get('L-SHIP', [])
            if r.get('est_seed') == SEEDS[0]}
    dups = [(r['episode'], abs(r['avg_nRMSE'] - ship[r['episode']]))
            for r in by.get('L-SHIP-DUP', []) if r['episode'] in ship]
    if dups:
        nz = [e for e, d in dups if d != 0.0]
        g5 = not nz
        p('G-5 dup probe       : %d pairs, max|delta|=%.3e, nonzero=%d -> %s'
          % (len(dups), max(d for _, d in dups), len(nz),
             'PASS' if g5 else 'FAIL (NONDETERMINISM -- PREREG D-3, halt)'))
        if nz:
            p('    nonzero:', nz[:5])
        ok &= g5

    # G-6 DCC
    bad = [(r['arm'], r['episode'], missing_fields(r)) for r in good if missing_fields(r)]
    p('G-6 DCC             : %d/%d rows complete -> %s'
      % (len(good) - len(bad), len(good), 'PASS' if not bad else 'FAIL (no verdict)'))
    for b in bad[:5]:
        p('    ', b)
    ok &= not bad

    # G-8 one code digest
    digs = sorted({r.get('code_md5') for r in good})
    p('G-8 one digest      : %d distinct code_md5 -> %s'
      % (len(digs), 'PASS' if len(digs) == 1 else 'FAIL %s' % digs))
    ok &= (len(digs) == 1)

    # frame parity + zero-field identity: the b1_full_comparison BLOCKER, as a gate
    bad_fr, bad_z = [], []
    for ep in {r['episode'] for r in good}:
        rs = [r for r in good if r['episode'] == ep]
        if len({r['n_match'] for r in rs}) != 1:
            bad_fr.append((ep, sorted({(r['arm'], r['n_match']) for r in rs})))
        zs = [r['zero_nRMSE'] for r in rs]
        if max(zs) - min(zs) > 1e-12:
            bad_z.append((ep, max(zs) - min(zs)))
    p('G-6b frame parity   : %d episodes where an arm was scored on a different frame '
      'count -> %s' % (len(bad_fr), 'PASS' if not bad_fr else 'FAIL'))
    for b in bad_fr[:3]:
        p('    ', b)
    p('G-6c zero identity  : %d episodes where zero_nRMSE differs across arms -> %s'
      % (len(bad_z), 'PASS' if not bad_z else 'FAIL'))
    ok &= (not bad_fr and not bad_z)
    return ok


# ── smoke ─────────────────────────────────────────────────────────────────────
def smoke(arms=None):
    preload()
    jobs = build_jobs(arms=arms, keys=SMOKE_KEYS)
    done = {r['job_id'] for r in _load_rows(SMOKE_OUT) if 'error' not in r}
    todo = [j for j in jobs if j['job_id'] not in done]
    print('%s SMOKE | %d jobs | %d done | %d todo | episodes=%s'
          % (BATCH, len(jobs), len(done), len(todo), list(SMOKE_KEYS)), flush=True)
    with open(SMOKE_OUT, 'a') as fh:
        for i, job in enumerate(todo, 1):
            rec = run_one(job)
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            print('[%d/%d] %-56s %s  adam1=%s grow=%s vp=%s merged=%s N=%s  %.0fs'
                  % (i, len(todo), rec['job_id'],
                     ('%.6f' % rec['avg_nRMSE']) if 'avg_nRMSE' in rec
                     else str(rec.get('error'))[:70],
                     rec.get('adam_steps_phase1', '-'), rec.get('adam_steps_grow', '-'),
                     rec.get('vp_steps', '-'), rec.get('merge_removed', '-'),
                     rec.get('n_atoms_final', '-'), rec.get('wall_s', float('nan'))),
                  flush=True)

    rows = _load_rows(SMOKE_OUT)
    good = posthoc(rows, 'smoke_results.jsonl')

    # bit anchor for the reused W3 arms, on the smoke episode W3 also scored
    w3rows = _load_rows(os.path.join(W3DIR, 'results.jsonl'))
    print('\n--- W3 bit anchor (ZERO / ORA-K-CVX imported unmodified) ---')
    for arm in ('ZERO', 'ORA-K-CVX'):
        for r in rows:
            if r.get('arm') != arm or 'error' in r:
                continue
            ref = [x for x in w3rows if x.get('arm') == arm
                   and x.get('episode') == r['episode'] and 'error' not in x]
            if not ref:
                print('  %-10s %-44s no W3 row (expected: the s8 corpus is not in W3)'
                      % (arm, r['episode']))
                continue
            d = abs(r['avg_nRMSE'] - ref[0]['avg_nRMSE'])
            print('  %-10s %-44s W4=%.12f W3=%.12f |d|=%.3e %s'
                  % (arm, r['episode'], r['avg_nRMSE'], ref[0]['avg_nRMSE'], d,
                     'OK' if d == 0 else 'MISMATCH'))

    t = {}
    for r in rows:
        if 'error' not in r:
            t.setdefault(r['arm'], []).append(r['wall_s'])
    print('\n--- timing (uncontended min-based extrapolation) ---')
    tot = 0.0
    for a in ARM_ORDER:
        v = t.get(a)
        if not v:
            continue
        n = len(EPISODES) * len(ARMS[a]['seeds'])
        tot += min(v) * n / 3600
        print('  %-11s n=%d min=%.0fs median=%.0fs -> %d jobs = %.2f h'
              % (a, len(v), min(v), np.median(v), n, min(v) * n / 3600))
    print('  FULL BATCH (sequential, uncontended) = %.2f h  |  at 8 workers ~ %.2f h'
          % (tot, tot / 8))
    return good


# ── run ───────────────────────────────────────────────────────────────────────
def main():
    import multiprocessing as mp
    mode = sys.argv[1] if len(sys.argv) > 1 else 'gates'
    arms = (sys.argv[2].split(',') if len(sys.argv) > 2 else None)
    if mode == 'gates':
        sys.exit(0 if gates() else 1)
    if mode == 'smoke':
        sys.exit(0 if smoke(arms=arms) else 1)
    if mode != 'run':
        print(__doc__)
        return

    if not gates(verbose=False):
        print('GATES FAIL -- refusing to run (PREREG section 5 verdict ordering).  '
              'Run `run_w4.py gates` to see which.', flush=True)
        sys.exit(1)
    preload()
    jobs = build_jobs(arms=arms)
    done = {r['job_id'] for r in _load_rows(OUT) if 'error' not in r}
    todo = [j for j in jobs if j['job_id'] not in done]
    n_workers = int(os.environ.get('WORKERS', '8'))
    dg, files = code_fingerprint()
    print('%s | total %d | done %d | todo %d | workers %d | seeds %s'
          % (BATCH, len(jobs), len(done), len(todo), n_workers, list(SEEDS)), flush=True)
    print('code_md5=%s files=%d HEAD=%s dirty=%d' % (dg, len(files), HEAD[:8], DIRTY),
          flush=True)
    if not todo:
        print('nothing to do')
        return

    # Parallelism is safe because every job is an independent (arm, episode, seed) triple:
    # the estimator seed is passed explicitly, `code_md5` is a pure function of file
    # contents, BLAS is pinned to one thread per worker, and `spawn` gives each worker a
    # fresh interpreter so no monkey-patch can survive between jobs of different arms
    # (and every patch is undone in a `finally` anyway).  Only the ORDER of lines in
    # results.jsonl depends on the worker count -- G-5 is the check that nothing else does.
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(n_workers) as pool, open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            el = time.time() - t0
            tag = rec.get('avg_nRMSE', rec.get('error', '?'))
            tag = ('%.4f' % tag) if isinstance(tag, float) else str(tag)[:70]
            print('[%d/%d] %-56s %s  adam1=%s vp=%s  %.0fs  ETA %.0fmin'
                  % (i, len(todo), rec['job_id'], tag,
                     rec.get('adam_steps_phase1', '-'), rec.get('vp_steps', '-'),
                     rec.get('wall_s', float('nan')), el / i * (len(todo) - i) / 60),
                  flush=True)
    print('DONE %.1f min' % ((time.time() - t0) / 60), flush=True)
    posthoc(_load_rows(OUT), 'results.jsonl')


if __name__ == '__main__':
    main()
