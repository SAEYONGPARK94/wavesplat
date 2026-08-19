#!/usr/bin/env python3
"""W5_WHY runner -- implements PREREG.md (md5 f5dd2e386cea1e537c14150462ed9268).

    python3 tmp/W5_WHY_2026-08-19/run_w5.py gates    # ZERO estimator runs, ZERO episode loads
    python3 tmp/W5_WHY_2026-08-19/run_w5.py g0       # the G-0 bit-neutrality probe (4 cells)
    python3 tmp/W5_WHY_2026-08-19/run_w5.py smoke    # 2 episodes x the 4 NEW arms
    WORKERS=8 python3 tmp/W5_WHY_2026-08-19/run_w5.py run          # 640 rows
    WORKERS=8 python3 tmp/W5_WHY_2026-08-19/run_w5.py run L-VP-ONLY  # arm subset

W5 IS W4'S RAIL WITH DIFFERENT ARMS ON IT
-----------------------------------------
Nothing about the measurement apparatus is re-implemented.  This file imports
`tmp/W4_SOLVER_2026-08-19/run_w4.py` and uses its episode sample, its estimator driver
(`run_est`), its ZERO / ORA-K-CVX wrappers, its field contract, its shell keys and its
phase-convention identity.  `arms_config.py` and `varpro.py` are imported through it.  The
three W4 files are frozen artefacts and are not edited; G-0b pins their md5s.

WHAT IS NEW HERE, AND ONLY THIS
  1. four arms (`arms_config_w5.py`), and the single-source expectation table they carry
  2. **W4 column reuse**: `L-SHIP` / `L-VP` / `L-VP-NW` / `L-ITER0` / `ZERO` / `ORA-K-CVX`
     are READ from W4's results.jsonl.  Zero re-runs (PREREG section 0.1 / section 2).
  3. G-0 (bit neutrality of patch b1467bd) and the provenance-split G-8.

  ⚠ THE G-8 TRAP.  A reused row was written under git 348d0f2 with
  `code_md5=a29340ce...`; a W5 row is written today under `aeff4d8` with a different
  digest, because `b1467bd` touched two package files and because the W5 treatment files
  are themselves in the fingerprint.  W4's G-8 demanded "exactly 1 distinct code_md5" over
  all rows.  Applied unchanged that gate would FAIL A LEGITIMATE BATCH -- the digests are
  *supposed* to differ.  So G-8 here is split BY ROW PROVENANCE: 1 digest among the new
  rows, 1 digest among the reused rows, and the reused digest must equal the value W4
  actually recorded.  The claim that the two digests are interchangeable is not argued, it
  is measured -- that is what G-0 is, and G-4 (`L-SHIP-DUP2` vs reused `L-SHIP`, |delta|=0
  on all 64 episodes) re-measures it at full sample size.

DISCIPLINE
  - expectations (720 / 1620 / 820, vp 400 / 800 / 0) live in `arms_config_w5.py` ONLY.
    This file, `watchdog_w5.py` and any analyser call `expected_adam_p1()` /
    `expected_vp_steps()`.  W4 wrote those numbers in four places and two of them are
    still wrong today (see the `arms_config_w5` docstring).
  - shells keep the `obs:` prefix everywhere, inherited from run_w4.
  - `effective_config` is an object read-back, inherited from run_w4/arms_config.
"""
import hashlib
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))              # .../wavesplat/wavesplat
W4DIR = os.path.join(ROOT, 'tmp', 'W4_SOLVER_2026-08-19')
W3DIR = os.path.join(ROOT, 'tmp', 'W3_SOLVER_2026-08-15')
for _p in (ROOT, HERE, W4DIR, W3DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import arms_config_w5 as A5                                # noqa: E402
import run_w4 as W4                                        # noqa: E402

BATCH = 'W5_WHY_2026-08-19'
OUT = os.path.join(HERE, 'results.jsonl')
SMOKE_OUT = os.path.join(HERE, 'smoke_results.jsonl')
FP_PATH = os.path.join(HERE, 'code_fingerprint.json')
G0_PATH = os.path.join(HERE, 'g0_bitidentity.json')
PROBE = os.path.join(HERE, 'probe_bitidentity.py')
W4_RESULTS = os.path.join(W4DIR, 'results.jsonl')

PREREG_MD5 = 'f5dd2e386cea1e537c14150462ed9268'

# ── G-0b anchors: the reuse claim is void if any of these moved ──────────────
# values from tmp/W4_SOLVER_2026-08-19/FROZEN_md5.txt (post-run freeze)
W4_ANCHOR = {
    'run_w4.py':     '818eb7d9874f33c7af2e2ddf4f0b6af1',
    'arms_config.py': '5c63a1a294517ba4227e44150d391b45',
    'varpro.py':     '5e920c91481af582475547021aa4a747',
    'PREREG.md':     '24727f5464e909c0e932594db2fb34e9',
    'results.jsonl': '80a7bd5d43db04cd8745bd12d0d76663',
}
#: the `code_md5` every W4 result row carries (1 distinct value over 1,344 rows).
W4_ROW_CODE_MD5 = 'a29340cee6c1de6b3bc08c089ceb8e61'

# ── sample / seeds: IDENTICAL objects to W4, not a re-derivation ─────────────
EPISODES = W4.EPISODES
SEEDS = W4.SEEDS                                           # (1595, 1596, 1597)
N_EP_EXPECTED = W4.N_EP_EXPECTED                           # 64
SHELL_KEYS = W4.SHELL_KEYS
SMOKE_KEYS = W4.SMOKE_KEYS

HEAD = W4.HEAD
DIRTY = W4.DIRTY


# ══════════════════════════════════════════════════════════════════════════════
#  ARM TABLE.  Static plumbing only -- every number comes from arms_config_w5.
# ══════════════════════════════════════════════════════════════════════════════
def _arms():
    A5.register()
    arms = {}
    for arm, spec in A5.NEW_SPECS.items():
        arms[arm] = dict(kind='est', cfg_from=arm, vp=spec['vp'],
                         vp_weighting=spec['vp_weighting'],
                         seeds=SEEDS[:spec['n_seeds']],
                         factor=spec['factor'], factor_vs=spec['factor_vs'],
                         dup_of=spec.get('dup_of'), source='W5')
    for arm in A5.REUSE_ARMS:
        w4spec = W4.ARMS[arm]
        arms[arm] = dict(w4spec, source='W4-reuse')
    return arms


ARMS = _arms()

#: execution order.  P-1's treatment first, then the gate that decides whether the batch
#: is allowed to answer anything, so a truncated run still answers P-1 and still knows it.
ARM_ORDER = ('L-VP-ONLY', 'L-SHIP-DUP2', 'L-VP-ONLY-NW', 'L-VP800')
NEW_ARMS = A5.NEW_ARMS
REUSE_ARMS = A5.REUSE_ARMS

# W4's run_est / arm_cfg / missing_fields dispatch off `run_w4.ARMS`.  Inject the W5 arms
# into that table IN MEMORY (the file on disk is untouched) so one driver serves both
# batches -- copying run_est would have forked the estimator path, which is exactly the
# class of duplication this batch exists to stop.
W4.ARMS.update({a: s for a, s in ARMS.items() if s.get('source') == 'W5'})


# ── fingerprints ─────────────────────────────────────────────────────────────
def _md5(path):
    with open(path, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _pkg_files():
    """Everything that can change what the ESTIMATOR does."""
    files = {}
    for pat in ('wavesplat/**/*.py', 'wavesim/**/*.py'):
        for f in sorted(glob.glob(os.path.join(ROOT, pat), recursive=True)):
            files[os.path.relpath(f, ROOT)] = _md5(f)
    for extra in ('configs/estimation.yaml', 'configs/default.yaml',
                  'configs/server.yaml'):
        p = os.path.join(ROOT, extra)
        if os.path.exists(p):
            files[extra] = _md5(p)
    return files


def pkg_fingerprint():
    """Package-only digest.  G-0 evidence is keyed on THIS, not on the full digest, so
    editing run_w5.py does not invalidate a bit-neutrality measurement that only ever
    depended on the package."""
    files = _pkg_files()
    return hashlib.md5(json.dumps(files, sort_keys=True).encode()).hexdigest(), files


def code_fingerprint():
    """One digest over a fixed file set (import-order independent).  Includes the W5
    treatment definition AND the W4 modules it drives through."""
    files = _pkg_files()
    for p in ([os.path.join(HERE, n) for n in
               ('run_w5.py', 'PREREG.md', 'arms_config_w5.py', 'watchdog_w5.py',
                'probe_bitidentity.py')]
              + [os.path.join(W4DIR, n) for n in
                 ('run_w4.py', 'arms_config.py', 'varpro.py')]
              + [os.path.join(W3DIR, 'run_w3.py')]):
        if os.path.exists(p):
            files[os.path.relpath(p, ROOT)] = _md5(p)
    return hashlib.md5(json.dumps(files, sort_keys=True).encode()).hexdigest(), files


# ── rows ─────────────────────────────────────────────────────────────────────
def _load_rows(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def load_reused():
    """The six W4 columns, read verbatim.  Tagged `_src='W4'` -- never rewritten, never
    copied back into W5's results.jsonl, so `tmp/W4_SOLVER_2026-08-19/` stays read-only."""
    keep = set(REUSE_ARMS)
    out = []
    for r in _load_rows(W4_RESULTS):
        if r.get('arm') in keep and 'error' not in r:
            r['_src'] = 'W4'
            out.append(r)
    return out


def all_rows(path=OUT):
    """Verdict corpus = W5's new rows + W4's reused columns."""
    new = [dict(r, _src='W5') for r in _load_rows(path)]
    return new + load_reused()


# ── jobs ─────────────────────────────────────────────────────────────────────
def build_jobs(arms=None, keys=None):
    jobs = []
    for arm in (arms or ARM_ORDER):
        if ARMS[arm].get('source') != 'W5':
            continue                    # reused arms are read, never scheduled
        for ep in EPISODES:
            if keys and ep['key'] not in keys:
                continue
            for s in ARMS[arm]['seeds']:
                jobs.append(dict(job_id='%s|%s|%s' % (arm, ep['key'], s),
                                 arm=arm, ep=ep, seed=s))
    return jobs


def run_one(job):
    sys.path[:0] = [ROOT, HERE, W4DIR, W3DIR]
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')
    arm, ep, seed = job['arm'], job['ep'], job['seed']
    spec = ARMS[arm]
    t0 = time.time()
    base = dict(batch=BATCH, job_id=job['job_id'], arm=arm, episode=ep['key'],
                corpus=ep['corpus'], stratum=ep['stratum'], bf=ep['bf'],
                ncomp=ep['ncomp'], seed=ep['ep_seed'], est_seed=seed,
                cfg_from=spec['cfg_from'], dup_of=spec.get('dup_of'),
                vp_weighting_intent=spec.get('vp_weighting'),
                deterministic=False, git_head=HEAD, git_dirty_n=DIRTY)
    try:
        W4.preload()
        A5.register()
        dg, dgf = code_fingerprint()
        body = W4.run_est(ep, arm, seed, spec)      # the W4 driver, unmodified
        row = dict(base, **body, code_md5=dg, code_files_n=len(dgf),
                   wall_s=time.time() - t0)
        row.update(A5.row_gate_verdict(row))        # single-source verdict, on top of W4's
        row['dcc_missing'] = W4.missing_fields(row)
        return row
    except Exception as e:
        import traceback
        dg, dgf = code_fingerprint()
        return dict(base, error=repr(e)[:300], tb=traceback.format_exc()[-2000:],
                    code_md5=dg, code_files_n=len(dgf), wall_s=time.time() - t0)


# ══════════════════════════════════════════════════════════════════════════════
#  G-0: bit neutrality of the post-W4 package patch
# ══════════════════════════════════════════════════════════════════════════════
def g0_measure(n=4):
    """CALL `probe_bitidentity.py` (PREREG section 0.1) and record its verdict.

    The probe re-runs `L-SHIP@1595` on `n` W4 cells under today's tree and requires
    max|delta| == 0 against the stored W4 values.  Its exit code is the authority; the
    evidence is keyed on the PACKAGE digest so a later edit to a batch file does not
    silently invalidate it, and a later edit to `wavesplat/**` does.
    """
    pkg, _ = pkg_fingerprint()
    t0 = time.time()
    r = subprocess.run([sys.executable, PROBE, str(n)], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stdout.write(r.stderr[-2000:])
    worst = None
    for tok in r.stdout.split():
        pass
    for ln in r.stdout.splitlines():
        if 'max|delta|' in ln:
            try:
                worst = float(ln.split('=')[1].split()[0])
            except (IndexError, ValueError):
                pass
    rec = dict(gate='G-0', probe='probe_bitidentity.py',
               probe_md5=_md5(PROBE) if os.path.exists(PROBE) else None,
               n_cells=n, returncode=r.returncode, bit_identical=(r.returncode == 0),
               max_abs_delta=worst, pkg_md5=pkg, git_head=HEAD, git_dirty_n=DIRTY,
               ts=time.strftime('%Y-%m-%dT%H:%M:%S'), wall_s=time.time() - t0,
               stdout=r.stdout[-4000:])
    with open(G0_PATH, 'w') as fh:
        json.dump(rec, fh, indent=1)
    return rec


def g0_status():
    """(verdict, detail) for the gate.  Evidence must exist, must say bit-identical, and
    must have been taken against the package as it stands right now."""
    if not os.path.exists(G0_PATH):
        return 'MISSING', 'no g0_bitidentity.json -- run `run_w5.py g0` (4 cells, ~2 min)'
    try:
        rec = json.load(open(G0_PATH))
    except ValueError as e:
        return 'FAIL', 'unreadable evidence: %s' % e
    pkg, _ = pkg_fingerprint()
    if rec.get('pkg_md5') != pkg:
        return 'STALE', ('evidence taken against package %s, now %s -- re-run `g0`'
                         % (str(rec.get('pkg_md5'))[:8], pkg[:8]))
    if not rec.get('bit_identical'):
        return 'FAIL', ('max|delta|=%s over %s cells -> REUSE FORBIDDEN, every arm must '
                        'be re-run (PREREG section 0.1)'
                        % (rec.get('max_abs_delta'), rec.get('n_cells')))
    return 'PASS', ('%d cells L-SHIP@%d, max|delta|=%s, pkg %s'
                    % (rec.get('n_cells', 0), SEEDS[0], rec.get('max_abs_delta'),
                       pkg[:8]))


# ══════════════════════════════════════════════════════════════════════════════
#  GATES (PREREG section 4).  Everything decidable without loading an episode.
# ══════════════════════════════════════════════════════════════════════════════
def gates(verbose=True):
    ok = True
    p = print if verbose else (lambda *a, **k: None)
    halting = {}

    def rec(name, good):
        halting[name] = bool(good)
        return bool(good)

    # ---- G-0 / G-0b: the reuse is legitimate --------------------------------
    st, detail = g0_status()
    p('G-0  bit neutrality : %-6s %s' % (st, detail))
    p('     (patch b1467bd touched amplitude_refit sign + avg_RMSE_near; both argued inert '
      'for the shipping path, so PREREG section 0.1 MEASURES it instead of arguing it)')
    ok &= rec('G-0', st == 'PASS')

    anchor_bad = []
    for name, want in W4_ANCHOR.items():
        pth = os.path.join(W4DIR, name)
        got = _md5(pth) if os.path.exists(pth) else None
        if got != want:
            anchor_bad.append((name, want[:8], (got or 'ABSENT')[:8]))
    p('G-0b W4 read-only   : %d/%d frozen md5 match -> %s%s'
      % (len(W4_ANCHOR) - len(anchor_bad), len(W4_ANCHOR),
         'PASS' if not anchor_bad else 'FAIL', '' if not anchor_bad else ' %s' % anchor_bad))
    ok &= rec('G-0b', not anchor_bad)

    n_reused = len(load_reused())
    by_src = {}
    for r in load_reused():
        by_src[r['arm']] = by_src.get(r['arm'], 0) + 1
    want_reuse = {'L-SHIP': 192, 'L-VP': 192, 'L-VP-NW': 192, 'L-ITER0': 192,
                  'ZERO': 64, 'ORA-K-CVX': 64}
    g0c = (by_src == want_reuse)
    p('G-0c reuse census   : %s (want %s) total=%d -> %s'
      % (by_src, want_reuse, n_reused, 'PASS' if g0c else 'FAIL'))
    ok &= rec('G-0c', g0c)

    # ---- PREREG freeze -------------------------------------------------------
    pm = _md5(os.path.join(HERE, 'PREREG.md'))
    g_pre = (pm == PREREG_MD5)
    p('PREREG freeze       : md5=%s (frozen %s) -> %s'
      % (pm, PREREG_MD5, 'PASS' if g_pre else 'FAIL'))
    ok &= rec('PREREG', g_pre)

    # ---- G-1: sample + router ------------------------------------------------
    missing = [e['rel'] for e in EPISODES
               if not os.path.exists(os.path.join(ROOT, e['rel']))]
    ncomp = sorted({e['ncomp'] for e in EPISODES})
    strat = sorted({e['stratum'] for e in EPISODES})
    dup = len({e['key'] for e in EPISODES}) != len(EPISODES)
    g1 = (not missing and not dup and len(EPISODES) == N_EP_EXPECTED
          and ncomp == [15, 30] and strat == ['crowded'])
    p('G-1  sample         : n=%d (want %d) ncomp=%s strata=%s dup=%s missing=%d -> %s'
      % (len(EPISODES), N_EP_EXPECTED, ncomp, strat, dup, len(missing),
         'PASS' if g1 else 'FAIL'))
    ok &= rec('G-1', g1)

    # the reused columns must be scored on THE SAME 64 episodes
    ep_keys = {e['key'] for e in EPISODES}
    reuse_keys = {r['episode'] for r in load_reused()}
    g1s = (reuse_keys == ep_keys)
    p('G-1b sample parity  : reused-column episodes == W5 sample (%d vs %d, sym.diff %d)'
      ' -> %s' % (len(reuse_keys), len(ep_keys), len(reuse_keys ^ ep_keys),
                  'PASS' if g1s else 'FAIL'))
    ok &= rec('G-1b', g1s)

    A5.register()
    cfgs = {a: W4.arm_cfg(a, SEEDS[0]) for a in NEW_ARMS}
    routes = sorted({c['init_method'] for c in cfgs.values()})
    g1c = (routes == ['learned'])
    p('G-1c route pinned   : new arms init_method=%s on %d arms -> %s   (the router is '
      'REMOVED, not sampled -- scope limit, PREREG section 5.7)'
      % (routes, len(cfgs), 'PASS' if g1c else 'FAIL'))
    ok &= rec('G-1c', g1c)

    want = {'lambda_l1': 1e-4, 'tracking_mode': 'w_phi', 'tracking_lr': 0.005,
            'max_n': 30, 'adaptive_n': True, 'use_dispersion': True,
            'learned_konly': False, 'init_k_lr_scale': 1.0}
    off = {k: (cfgs['L-SHIP-DUP2'].get(k), v) for k, v in want.items()
           if cfgs['L-SHIP-DUP2'].get(k) != v}
    p('G-1d shipped defs   : %s -> %s'
      % ('all %d match' % len(want) if not off else 'DRIFT %s' % off,
         'PASS' if not off else 'FAIL'))
    ok &= rec('G-1d', not off)

    # ---- G-2 / G-3: the SINGLE-SOURCE expectation table ----------------------
    p('\nG-2/G-3 expectations -- defined ONCE in arms_config_w5.py and read from there by '
      'this runner, watchdog_w5.py and the analyser:')
    p('     formula: adam_steps_phase1 == POLISH_STEPS(%d) + joint_fit + vp | '
      'vp_steps == vp' % A5.POLISH_STEPS)
    for a in ARM_ORDER + tuple(x for x in REUSE_ARMS if x in A5.ALL_SPECS):
        e = A5.expectation_table()[a]
        p('     %-13s adam_p1=%-5s vp=%-4s weighting=%-11s %s'
          % (a, e['adam_steps_phase1'], e['vp_steps'], e['vp_weighting'],
             'REUSED (0 runs)' if e['reused'] else 'NEW'))
    try:
        A5._verify_single_source()
        p('     cross-check: formula == PREREG section 4 literals (%d arms) == W4 measured '
          'counters (%d arms) -> PASS' % (len(A5.PREREG_QUOTED), len(A5.W4_MEASURED)))
        g23 = True
    except A5.SingleSourceError as e:
        p('     cross-check: -> FAIL %s' % e)
        g23 = False
    # the registered row-level gate must BE the formula, not a second copy of it
    reg_bad = [a for a in NEW_ARMS
               if A5.AC.ARM_GATES.get(a, {}).get('adam_steps_phase1')
               != ('==', A5.expected_adam_p1(a))]
    p('     ARM_GATES registered from the formula for %d/%d new arms -> %s'
      % (len(NEW_ARMS) - len(reg_bad), len(NEW_ARMS), 'PASS' if not reg_bad else 'FAIL'))
    ok &= rec('G-2/G-3', g23 and not reg_bad)

    # ---- single-factor audit -------------------------------------------------
    p('\n     single-factor audit (merged config vs the arm\'s declared comparator, '
      'seed %d):' % SEEDS[0])
    sf = True
    all_cfg = dict(cfgs)
    for a in ('L-SHIP', 'L-VP', 'L-VP-NW', 'L-ITER0'):
        all_cfg[a] = W4.arm_cfg(a, SEEDS[0])
    for arm in NEW_ARMS:
        spec = ARMS[arm]
        base_arm = spec['factor_vs']
        base, c = all_cfg[base_arm], all_cfg[arm]
        diff = sorted(k for k in set(c) | set(base)
                      if c.get(k, '<absent>') != base.get(k, '<absent>'))
        if ARMS[arm]['vp'] != ARMS[base_arm].get('vp', 0):
            diff.append('vp_steps' if (ARMS[arm]['vp'] and ARMS[base_arm].get('vp'))
                        else '__varpro_install__')
        if ARMS[arm].get('vp_weighting', 'confidence') != \
                ARMS[base_arm].get('vp_weighting', 'confidence'):
            diff.append('vp_weighting')
        diff = sorted(diff)
        good = (diff == sorted(spec['factor']))
        sf &= good
        p('     %-13s vs %-11s delta=%-28s declared=%-24s -> %s'
          % (arm, base_arm, str(diff or '(none)'), str(sorted(spec['factor']) or '(none)'),
             'PASS' if good else 'FAIL'))
    ok &= rec('single-factor', sf)

    # ---- G-4: the determinism / patch-neutrality replicate -------------------
    same = (W4.arm_cfg('L-SHIP-DUP2', SEEDS[0]) == W4.arm_cfg('L-SHIP', SEEDS[0]))
    g4 = same and ARMS['L-SHIP-DUP2']['seeds'] == (SEEDS[0],)
    p('\nG-4  dup2 probe     : L-SHIP-DUP2 config == L-SHIP config = %s, seed=%s -> %s'
      % (same, ARMS['L-SHIP-DUP2']['seeds'], 'PASS' if g4 else 'FAIL'))
    p('     [numeric |delta|=0 vs the REUSED L-SHIP@%d column runs on rows; because the '
      'two rows carry different code_md5 this doubles as G-0 at n=64]' % SEEDS[0])
    ok &= rec('G-4', g4)

    # ---- G-5: field contract (DCC), inherited from run_w4 --------------------
    p('G-5  field contract : common(%d) + est(%d) / cvx(%d) / zero(%d), per ARM KIND '
      '(S17 died making one contract universal)'
      % (len(W4.REQUIRED_COMMON), len(W4.REQUIRED_EST), len(W4.REQUIRED_CVX),
         len(W4.REQUIRED_ZERO)))
    g5 = (not set(W4.REQUIRED_COMMON) & set(W4.REQUIRED_EST)
          and not set(W4.REQUIRED_COMMON) & set(W4.REQUIRED_CVX)
          and all(ARMS[a]['kind'] in ('est', 'zero', 'cvx') for a in ARMS))
    p('     shells (obs: prefix, every arm): %s' % ' '.join(SHELL_KEYS))
    p('     contract sets disjoint + every W5 arm has a known kind -> %s'
      % ('PASS' if g5 else 'FAIL'))
    ok &= rec('G-5', g5)

    # ---- G-7: phase convention identity (numeric, run-free) ------------------
    ref, bad, got = W4._g7_phase_identity()
    g7 = ref <= 1e-12 and (got is None or got <= 1e-12)
    p('G-7  phase identity : atan2(-b,a) max|d|=%.3e (tol 1e-12) | sequential_update.py'
      ':1335 atan2(+b,a) gives %.3e | varpro.ab_to_wphi max|d|=%s -> %s'
      % (ref, bad, ('%.3e' % got) if got is not None else 'absent',
         'PASS' if g7 else 'FAIL'))
    ok &= rec('G-7', g7)

    # ---- G-8: code fingerprint, SPLIT BY PROVENANCE --------------------------
    dg1, files = code_fingerprint()
    dg2, _ = code_fingerprint()
    reuse_digests = sorted({r.get('code_md5') for r in load_reused()})
    g8_stable = (dg1 == dg2)
    g8_reuse = (reuse_digests == [W4_ROW_CODE_MD5])
    p('G-8  fingerprint    : W5 digest=%s files=%d stable=%s (git %s dirty %d)'
      % (dg1, len(files), g8_stable, HEAD[:8], DIRTY))
    p('     reused rows carry %s -> %s   (W4 ran under 348d0f2; patch b1467bd + the W5 '
      'treatment files make today\'s digest differ BY DESIGN)'
      % (reuse_digests, 'PASS' if g8_reuse else 'FAIL'))
    p('     ⚠ G-8 is per-provenance: 1 digest among NEW rows AND 1 among REUSED rows. '
      'Demanding "1 digest" over the pooled corpus would fail a legitimate batch; the '
      'digests are reconciled by MEASUREMENT (G-0, and G-4 at n=64), not by assertion.')
    with open(FP_PATH, 'w') as fh:
        json.dump({'digest': dg1, 'pkg_digest': pkg_fingerprint()[0],
                   'w4_row_code_md5': W4_ROW_CODE_MD5, 'git_head': HEAD,
                   'git_dirty_n': DIRTY, 'files': files}, fh, indent=1)
    ok &= rec('G-8', g8_stable and g8_reuse)

    # ---- G-9: watchdog plans agree ------------------------------------------
    jobs = build_jobs()
    try:
        import watchdog_w5
        wplan = {a: n for a, n in watchdog_w5.PLAN}
        mine = {a: len(ARMS[a]['seeds']) for a in ARM_ORDER}
        g9 = (wplan == mine and watchdog_w5.N_EP == len(EPISODES)
              and watchdog_w5.TOTAL == len(jobs)
              and watchdog_w5.EXPECT_SOURCE == 'arms_config_w5')
        p('G-9  watchdog       : plan match=%s  N_EP %d/%d  TOTAL %d/%d  expectations '
          'imported from %s -> %s   (file state only -- never the process table)'
          % (wplan == mine, watchdog_w5.N_EP, len(EPISODES), watchdog_w5.TOTAL, len(jobs),
             watchdog_w5.EXPECT_SOURCE, 'PASS' if g9 else 'FAIL'))
        if not g9:
            p('     watchdog=%s\n     runner  =%s' % (wplan, mine))
        ok &= rec('G-9', g9)
    except Exception as e:
        p('G-9  watchdog       : not importable (%s) -> FAIL' % repr(e)[:90])
        ok &= rec('G-9', False)

    # ---- sibling modules -----------------------------------------------------
    vpm = W4._opt_mod('varpro')
    iface = [n for n in ('install', 'uninstall', 'stats', 'reset_stats', 'ab_to_wphi')
             if vpm is not None and hasattr(vpm, n)]
    g_vp = vpm is not None and {'install', 'uninstall', 'stats', 'reset_stats'} <= set(iface)
    p('MOD  varpro         : %s -> %s   (installed FIRST, arms_config counters SECOND, so '
      'VARPRO Adam steps land in adam_steps_phase1)'
      % (iface or 'ABSENT', 'PASS' if g_vp else 'FAIL'))
    ok &= rec('MOD varpro', g_vp)
    covered = all(ARMS[a]['cfg_from'] in A5.AC.ARM_DELTAS for a in NEW_ARMS)
    p('MOD  arms_config    : W4 ARM_DELTAS extended in memory with %s -> %s   '
      '(imported, not copied: one counter dict `arms_config._C` for the whole batch)'
      % (sorted(NEW_ARMS), 'PASS' if covered else 'FAIL'))
    ok &= rec('MOD arms_config', covered)

    # ---- seed ledger ---------------------------------------------------------
    p('SEED ledger         : W5 re-uses W4 seeds %s ON PURPOSE (paired reuse of W4 '
      'columns requires identical estimator seeds -- PREREG section 1) -> PASS' % list(SEEDS))

    # ---- plan ----------------------------------------------------------------
    p('\nplan: %d NEW arms -> %d jobs over %d episodes  (+%d reused rows, 0 runs)'
      % (len(ARM_ORDER), len(jobs), len(EPISODES), n_reused))
    for a in ARM_ORDER:
        p('   %-13s %-4s vp=%-4s w=%-11s seeds=%-20s %3d jobs'
          % (a, ARMS[a]['kind'], ARMS[a]['vp'], ARMS[a]['vp_weighting'],
             str(ARMS[a]['seeds']), len(build_jobs(arms=[a]))))
    for a in REUSE_ARMS:
        p('   %-13s %-4s REUSED from W4 results.jsonl                       %3d rows'
          % (a, ARMS[a]['kind'], by_src.get(a, 0)))

    ok &= posthoc(all_rows(), 'results.jsonl + W4 reuse', verbose=verbose)
    p('\nGATES(run-free) %s' % ('PASS' if ok else 'FAIL -- batch halted'))
    if not ok:
        p('   failing: %s' % sorted(k for k, v in halting.items() if not v))
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  post-hoc: the row-dependent halves, on whatever rows exist
# ══════════════════════════════════════════════════════════════════════════════
def posthoc(rows, label, verbose=True):
    p = print if verbose else (lambda *a, **k: None)
    good = [r for r in rows if 'error' not in r]
    new = [r for r in good if r.get('_src') != 'W4']
    p('\n--- post-hoc gates on %s: %d rows (%d new, %d reused, %d errors) ---'
      % (label, len(rows), len(new), len(good) - len(new), len(rows) - len(good)))
    if not new:
        p('    0 new rows -> G-2/G-3/G-4/G-5/G-6 PENDING (not PASS)')
        return True
    ok = True
    by = {}
    for r in good:
        by.setdefault(r['arm'], []).append(r)

    # G-1 route / entry
    est = [r for r in good if ARMS.get(r['arm'], {}).get('kind') == 'est']
    bad = [(r['arm'], r['episode'], r.get('route'), r.get('learned_init_calls'))
           for r in est if r.get('route') != 'learned' or r.get('learned_init_calls') != 1]
    p('G-1 route/entry     : %d/%d est rows route==learned AND learned_init_calls==1 -> %s'
      % (len(est) - len(bad), len(est), 'PASS' if not bad else 'FAIL'))
    for b in bad[:3]:
        p('    ', b)
    ok &= not bad

    # G-2 / G-3 straight off the single source
    p('G-2/G-3 counters vs the single-source expectation table:')
    g23 = True
    for arm in sorted(by):
        if ARMS.get(arm, {}).get('kind') != 'est':
            continue
        wa, wv = A5.expected_adam_p1(arm), A5.expected_vp_steps(arm)
        ga = sorted({r.get('adam_steps_phase1') for r in by[arm]})
        gv = sorted({r.get('vp_steps') for r in by[arm]})
        good_a, good_v = (ga == [wa]), (gv == [wv])
        g23 &= good_a and good_v
        p('    %-13s n=%-4d adam_p1 want=%-5s got=%-12s %s | vp want=%-4s got=%-8s %s  '
          '[%s]' % (arm, len(by[arm]), wa, str(ga), 'OK' if good_a else 'MISMATCH',
                    wv, str(gv), 'OK' if good_v else 'MISMATCH',
                    'reused' if by[arm][0].get('_src') == 'W4' else 'new'))
    p('    -> %s' % ('PASS' if g23 else 'FAIL'))
    ok &= g23

    ua = [(r['arm'], r['episode'], r['adam_steps_other']) for r in est
          if r.get('adam_steps_other')]
    p('G-2b unattributed   : %d est rows with adam_steps_other>0 -> %s'
      % (len(ua), 'PASS' if not ua else 'INVESTIGATE'))

    # G-4 the dup2 replicate, cross-provenance
    ship = {r['episode']: r['avg_nRMSE'] for r in by.get('L-SHIP', [])
            if r.get('est_seed') == SEEDS[0]}
    dups = [(r['episode'], abs(r['avg_nRMSE'] - ship[r['episode']]))
            for r in by.get('L-SHIP-DUP2', []) if r['episode'] in ship]
    if dups:
        nz = [e for e, d in dups if d != 0.0]
        p('G-4 dup2 |delta|    : %d pairs vs REUSED L-SHIP@%d, max=%.3e, nonzero=%d -> %s'
          % (len(dups), SEEDS[0], max(d for _, d in dups), len(nz),
             'PASS' if not nz else 'FAIL (nondeterminism OR patch is not bit-neutral)'))
        for e in nz[:5]:
            p('    ', e)
        ok &= not nz
    else:
        p('G-4 dup2 |delta|    : 0 comparable pairs -> PENDING')

    # G-5 DCC
    bad = [(r['arm'], r['episode'], W4.missing_fields(r)) for r in new
           if W4.missing_fields(r)]
    p('G-5 DCC             : %d/%d new rows complete -> %s'
      % (len(new) - len(bad), len(new), 'PASS' if not bad else 'FAIL (no verdict)'))
    for b in bad[:5]:
        p('    ', b)
    ok &= not bad

    # G-6 frame parity + zero-field identity (non-halting, recorded)
    bad_fr, bad_z = [], []
    for ep in {r['episode'] for r in good}:
        rs = [r for r in good if r['episode'] == ep]
        if len({r['n_match'] for r in rs}) != 1:
            bad_fr.append((ep, sorted({(r['arm'], r['n_match']) for r in rs})))
        zs = [r['zero_nRMSE'] for r in rs if 'zero_nRMSE' in r]
        if zs and max(zs) - min(zs) > 1e-12:
            bad_z.append((ep, max(zs) - min(zs)))
    p('G-6 frame parity    : %d episodes scored on differing frame counts -> %s '
      '(non-halting, recorded)' % (len(bad_fr), 'PASS' if not bad_fr else 'RECORDED'))
    for b in bad_fr[:3]:
        p('    ', b)
    p('G-6b zero identity  : %d episodes where zero_nRMSE differs across arms -> %s'
      % (len(bad_z), 'PASS' if not bad_z else 'RECORDED'))

    # G-8 provenance split
    dnew = sorted({r.get('code_md5') for r in new})
    dold = sorted({r.get('code_md5') for r in good if r.get('_src') == 'W4'})
    g8 = (len(dnew) == 1 and (not dold or dold == [W4_ROW_CODE_MD5]))
    p('G-8 digests         : NEW %s | REUSED %s -> %s   (split by provenance ON PURPOSE)'
      % (dnew, dold, 'PASS' if g8 else 'FAIL'))
    ok &= g8

    # single-source row verdict (recomputed, never the stored flag)
    vbad = [(r['arm'], r['episode'], A5.row_gate_verdict(r)['w5_gate_fails'])
            for r in good if not A5.row_gate_verdict(r)['w5_gates_pass']]
    stale = [r for r in good if r.get('gates_pass') is False
             and A5.row_gate_verdict(r)['w5_gates_pass']]
    p('W5  row verdict     : %d/%d rows pass the single-source gate -> %s'
      % (len(good) - len(vbad), len(good), 'PASS' if not vbad else 'FAIL'))
    if stale:
        p('    NOTE %d reused rows carry gates_pass=False from a STALE W4 expectation '
          '(arms_config.ARM_GATES["L-ITER0"] still says adam==0; measured 320). Re-judged '
          'from the formula, they pass. arms: %s'
          % (len(stale), sorted({r['arm'] for r in stale})))
    for b in vbad[:5]:
        p('    ', b)
    ok &= not vbad
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  smoke
# ══════════════════════════════════════════════════════════════════════════════
def _pump(todo, sink, tag):
    import multiprocessing as mp
    n_workers = max(1, int(os.environ.get('WORKERS', '8')))
    t0 = time.time()
    if n_workers == 1:
        it = (run_one(j) for j in todo)
    else:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(n_workers)
        it = pool.imap_unordered(run_one, todo, chunksize=1)
    try:
        for i, r in enumerate(it, 1):
            sink.write(json.dumps(r) + '\n')
            sink.flush()
            el = time.time() - t0
            v = r.get('avg_nRMSE', r.get('error', '?'))
            v = ('%.6f' % v) if isinstance(v, float) else str(v)[:70]
            print('[%s %d/%d] %-46s %s  adam1=%s vp=%s N=%s  %.0fs  ETA %.0fmin'
                  % (tag, i, len(todo), r['job_id'], v,
                     r.get('adam_steps_phase1', '-'), r.get('vp_steps', '-'),
                     r.get('n_atoms_final', '-'), r.get('wall_s', float('nan')),
                     el / i * (len(todo) - i) / 60), flush=True)
    finally:
        if n_workers > 1:
            pool.close()
            pool.join()


def smoke(arms=None):
    W4.preload()
    A5.register()
    jobs = build_jobs(arms=arms, keys=SMOKE_KEYS)
    done = {r['job_id'] for r in _load_rows(SMOKE_OUT) if 'error' not in r}
    todo = [j for j in jobs if j['job_id'] not in done]
    print('%s SMOKE | %d jobs | %d done | %d todo | episodes=%s | workers=%s'
          % (BATCH, len(jobs), len(done), len(todo), list(SMOKE_KEYS),
             os.environ.get('WORKERS', '8')), flush=True)
    if todo:
        with open(SMOKE_OUT, 'a') as fh:
            _pump(todo, fh, 'smoke')

    rows = [dict(r, _src='W5') for r in _load_rows(SMOKE_OUT)]
    print('\n' + '=' * 78)
    print('SMOKE COUNTER CHECK -- observed vs the arms_config_w5 single source')
    print('  (PREREG section 4: L-VP-ONLY/-NW 720, L-VP800 1620 | vp 400 / 800 / 0)')
    print('=' * 78)
    by = {}
    for r in rows:
        if 'error' not in r:
            by.setdefault(r['arm'], []).append(r)
    allok = True
    for arm in ARM_ORDER:
        rs = by.get(arm, [])
        if not rs:
            print('  %-13s NO ROWS' % arm)
            allok = False
            continue
        wa, wv = A5.expected_adam_p1(arm), A5.expected_vp_steps(arm)
        ga = sorted({r.get('adam_steps_phase1') for r in rs})
        gv = sorted({r.get('vp_steps') for r in rs})
        gg = sorted({r.get('adam_steps_grow') for r in rs})
        gw = sorted({(r.get('vp_stats') or {}).get('vp_weighting_effective',
                      r.get('vp_weighting_intent')) for r in rs})
        good = (ga == [wa] and gv == [wv])
        allok &= good
        print('  %-13s n=%-2d  adam_p1 want %-5s got %-10s | vp want %-4s got %-8s | '
              'grow %-10s | w=%-16s %s'
              % (arm, len(rs), wa, str(ga), wv, str(gv), str(gg), str(gw),
                 'MATCH' if good else '*** MISMATCH ***'))
    print('  %s' % ('ALL COUNTERS MATCH THE PREREG EXPECTATION'
                    if allok else '*** COUNTER MISMATCH -- this is the top finding, the '
                                  'expectation may be wrong (W4: learned_init_iters=0 was '
                                  '320, not 0) ***'))

    good = posthoc(rows, 'smoke_results.jsonl')

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
        print('  %-13s n=%d min=%.0fs median=%.0fs -> %d jobs = %.2f h'
              % (a, len(v), min(v), np.median(v), n, min(v) * n / 3600))
    print('  FULL BATCH (sequential) = %.2f h  |  at 8 workers ~ %.2f h  '
          '[reused columns cost 0]' % (tot, tot / 8))
    return good and allok


# ══════════════════════════════════════════════════════════════════════════════
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'gates'
    arms = (sys.argv[2].split(',') if len(sys.argv) > 2 else None)
    if mode == 'gates':
        sys.exit(0 if gates() else 1)
    if mode == 'g0':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        rec = g0_measure(n)
        print('\nG-0 recorded -> %s  (bit_identical=%s)'
              % (os.path.relpath(G0_PATH, ROOT), rec['bit_identical']))
        sys.exit(0 if rec['bit_identical'] else 1)
    if mode == 'smoke':
        sys.exit(0 if smoke(arms=arms) else 1)
    if mode != 'run':
        print(__doc__)
        return

    # PREREG section 0.1: re-measure G-0 immediately before the main run.
    st, detail = g0_status()
    if st in ('MISSING', 'STALE'):
        print('G-0 %s (%s) -- re-measuring before the run' % (st, detail), flush=True)
        g0_measure(4)
    if not gates(verbose=False):
        print('GATES FAIL -- refusing to run (PREREG section 4 verdict ordering).  '
              'Run `run_w5.py gates` to see which.', flush=True)
        sys.exit(1)
    W4.preload()
    A5.register()
    jobs = build_jobs(arms=arms)
    done = {r['job_id'] for r in _load_rows(OUT) if 'error' not in r}
    todo = [j for j in jobs if j['job_id'] not in done]
    dg, files = code_fingerprint()
    print('%s | new-arm jobs %d | done %d | todo %d | workers %s | seeds %s | '
          'reused rows %d (0 runs)'
          % (BATCH, len(jobs), len(done), len(todo), os.environ.get('WORKERS', '8'),
             list(SEEDS), len(load_reused())), flush=True)
    print('code_md5=%s files=%d HEAD=%s dirty=%d' % (dg, len(files), HEAD[:8], DIRTY),
          flush=True)
    if not todo:
        print('nothing to do')
        return
    t0 = time.time()
    with open(OUT, 'a') as fh:
        _pump(todo, fh, 'run')
    print('DONE %.1f min' % ((time.time() - t0) / 60), flush=True)
    posthoc(all_rows(), 'results.jsonl + W4 reuse')


if __name__ == '__main__':
    main()
