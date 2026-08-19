#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gatecheck_w4.py -- INDEPENDENT post-hoc integrity gates for `W4 · SOLVER-REMEDY`.

WHY THIS FILE EXISTS
--------------------
`run_w4.py::posthoc()` already scores W4's gates.  That is the runner grading its own
homework.  The repository has been bitten by exactly that shape before (the
`b1_full_comparison.json` frame-count BLOCKER survived because the only checker was the
producer).  So the discipline is: **the gates are re-derived once more, by code that shares
no import with the runner.**

Consequently this module imports NOTHING from `run_w4.py`, `arms_config.py` or `varpro.py`.
Every constant below is transcribed from the frozen prose documents, with the section that
authorises it named on the same line.  Where this file and the runner agree, the agreement
is evidence.  Where they disagree, the disagreement is the finding.

WHAT IT IS NOT
--------------
It never computes a verdict, an effect size, a win count or a p-value.  It answers exactly
one question: **may a verdict be printed at all, and if so, with what scope?**
`analyze_w4.py` (verdict) and `verify_w4.py` (independent re-derivation of the verdict) are
different agents' files; this one gates them.

USAGE
-----
    python3 gatecheck_w4.py smoke_results.jsonl            # development / smoke
    python3 gatecheck_w4.py results.jsonl                  # the real batch
    python3 gatecheck_w4.py results.jsonl --descriptives   # + AMENDMENT_1 A-11 free stats
    python3 gatecheck_w4.py results.jsonl --json gate.json

The rows path is a REQUIRED positional argument on purpose: there is no default, so this
script can never open `results.jsonl` by accident while the batch is still running
(pre-exposure hygiene).  `--descriptives` is opt-in for the same reason -- the D-family
numbers are outcome-adjacent and must not leak into a routine integrity check.

EXIT CODES
----------
    0  no halting gate failed and nothing is outstanding      -> verdict printable
    1  something is PENDING (batch incomplete / arm missing)  -> verdict NOT yet printable
    2  a halting gate FAILED                                  -> verdict must NOT be printed
    3  usage / load error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
W3DIR = os.path.join(os.path.dirname(HERE), 'W3_SOLVER_2026-08-15')

# ══════════════════════════════════════════════════════════════════════════════
#  FROZEN CONSTANTS -- transcribed from the prose, not imported from the runner
# ══════════════════════════════════════════════════════════════════════════════

N_EP_EXPECTED = 64                       # AMENDMENT_1 A-3 (corpus freeze)
EST_SEEDS = (1595, 1596, 1597)           # AMENDMENT_1 A-3 (ledger seeds, 0 collisions)
VP_STEPS = 400                           # PREREG section 2 ("400 steps frozen before results")
POLISH_STEPS = 320                       # AMENDMENT_2 A2-1: nsamp 4 x polish_iters 80
SHIP_JOINT_STEPS = 500                   # AMENDMENT_2 A2-1: the shipped joint fit
N_MIN_PAIRS = 56                         # PREREG section 5 D-4: n < 56 -> UNDECIDED (n)

#: arm -> kind.  PREREG section 2 + AMENDMENT_1 A-4 (the two de-confounding arms).
ARM_KIND = {
    'L-SHIP': 'est', 'L-SHIP-DUP': 'est', 'L-ITER0': 'est', 'L-VP': 'est',
    'L-VP-NW': 'est', 'L-STEP400': 'est', 'L-MERGE': 'est',
    'ZERO': 'zero', 'ORA-K-CVX': 'cvx',
}

#: arm -> expected row count in a COMPLETE batch.  AMENDMENT_1 A-13:
#: 6 estimator arms x 64 episodes x 3 seeds = 1152, + 64 + 64 + 64 = 1344.
EXPECT_ROWS = {
    'L-SHIP': 192, 'L-ITER0': 192, 'L-VP': 192, 'L-VP-NW': 192,
    'L-STEP400': 192, 'L-MERGE': 192,
    'L-SHIP-DUP': 64, 'ZERO': 64, 'ORA-K-CVX': 64,
}
EXPECT_TOTAL = 1344                      # AMENDMENT_1 A-13

#: arm -> exact `adam_steps_phase1`.  AMENDMENT_2 A2-1 table.
#: MANDATED are the two the amendment wrote into gate G-2 (A2-2); the rest are the same
#: table's rows and are checked at ADVISORY severity, because A2-2 only promoted two.
EXPECT_PHASE1 = {
    'L-SHIP': POLISH_STEPS + SHIP_JOINT_STEPS,        # 820
    'L-SHIP-DUP': POLISH_STEPS + SHIP_JOINT_STEPS,    # 820 (same delta as L-SHIP)
    'L-ITER0': POLISH_STEPS,                          # 320
    'L-VP': POLISH_STEPS + SHIP_JOINT_STEPS + VP_STEPS,      # 1220
    'L-VP-NW': POLISH_STEPS + SHIP_JOINT_STEPS + VP_STEPS,   # 1220
    'L-STEP400': POLISH_STEPS + 900,                  # 1220
    'L-MERGE': POLISH_STEPS + SHIP_JOINT_STEPS,       # 820
}
G2_MANDATED = ('L-ITER0', 'L-SHIP')      # AMENDMENT_2 A2-2

#: arm -> expected `vp_steps`.  PREREG G-3 as generalised by AMENDMENT_2 A2-3
#: (declaration-driven, NOT "the arm literally named L-VP").
EXPECT_VP = {a: (VP_STEPS if a in ('L-VP', 'L-VP-NW') else 0) for a in ARM_KIND}

#: arm -> expected object-read-back of the config knob the arm delta moves.
#: `arms_config.ARM_DELTAS` + AMENDMENT_1 A-4.  Checked against `effective_config`, i.e.
#: the OBJECT, never the intent dict (CLAUDE.md: S13b found a 14%-effective ensemble that
#: the intent table showed as 100%).
EXPECT_INIT_ITERS = {
    'L-SHIP': 500, 'L-SHIP-DUP': 500, 'L-VP': 500, 'L-VP-NW': 500, 'L-MERGE': 500,
    'L-ITER0': 0, 'L-STEP400': 900,
}
EXPECT_MERGE_FRAC = {a: (0.5 if a == 'L-MERGE' else None) for a in EXPECT_INIT_ITERS}
EXPECT_LAMBDA_L1 = 1e-4                  # shipping default, `LAMBDA_L1_DEFAULT`

#: PREREG section 6: shells carry the `obs:` prefix.  `ship:` is a DIFFERENT quantity
#: (1.8-4.1x apart on the same sample) and S13 lost a claim by conflating them.
SHELL_EDGES = [0, 5, 10, 20, 30, 50, 75, 100, 150, 'inf']
SHELL_KEYS = ['obs:%s_%s' % (a, b) for a, b in zip(SHELL_EDGES[:-1], SHELL_EDGES[1:])]

# ── DCC (PREREG section 5 field list, split by arm kind per AMENDMENT_1 A-6) ──
#  A-6 is the S17 lesson: a flat list makes `ZERO` / `ORA-K-CVX` fail a contract they
#  cannot physically satisfy (no router, no optimiser, no atom pool), which halts the
#  batch on arms that are not even verdict arms.  Every exemption below is a structural
#  impossibility, not a convenience.
DCC_COMMON = (
    'batch', 'arm', 'episode', 'corpus', 'stratum', 'bf', 'ncomp', 'seed',
    'wall_s', 'avg_nRMSE', 'zero_nRMSE', 'n_match', 'n_all',
    'shell_rmse', 'zero_shell', 'shell_frac', 'window_resid_rms', 'vp_steps',
    'n_gt', 'effective_config', 'git_head', 'code_md5', 'deterministic',
)
DCC_EST = (
    'est_seed', 'route', 'route_source', 'adam_steps_phase1', 'adam_steps_grow',
    'adam_steps_track', 'adam_steps_other', 'merge_removed', 'merge_calls',
    'n_atoms_final', 'counters', 'intent_config', 'lambda_l1_intent',
    'lambda_l1_effective', 'init_method_intent', 'window_n_obs',
)
DCC_CVX = (
    'n_atoms_final', 'cvx_n_obs', 'cvx_cond', 'cvx_rank', 'cvx_window_resid_rms',
    'cvx_truth_window_resid_rms', 'cvx_gtomega_nrmse', 'cvx_omega_source',
)
DCC_ZERO = ()
DCC_NULLABLE_FOR_DET = ('est_seed',)     # A-6: "this arm has no estimator seed" is the
                                         # truth about the arm, not a missing measurement.

# ══════════════════════════════════════════════════════════════════════════════
#  severities -- what a failure COSTS.  This is the whole point of the file.
# ══════════════════════════════════════════════════════════════════════════════
HALT = 'HALT'          # PREREG section 5 + AMENDMENT_1 A-8: no verdict may be printed
SCOPED = 'SCOPED'      # a named hypothesis becomes unprintable; the rest survives
INFO = 'INFO'          # recorded, never blocks anything

PASS, FAIL, PENDING, INVESTIGATE, NOTE = 'PASS', 'FAIL', 'PENDING', 'INVESTIGATE', 'NOTE'


class Gate(object):
    """One gate result.  `blocks` names what may not be printed when it FAILs."""

    __slots__ = ('name', 'status', 'severity', 'blocks', 'detail', 'offenders')

    def __init__(self, name, status, severity, detail, offenders=None, blocks=None):
        self.name = name
        self.status = status
        self.severity = severity
        self.detail = detail
        self.offenders = list(offenders or [])
        self.blocks = blocks or ('the whole verdict' if severity == HALT else '-')

    def as_dict(self):
        return dict(name=self.name, status=self.status, severity=self.severity,
                    blocks=self.blocks, detail=self.detail,
                    n_offenders=len(self.offenders), offenders=self.offenders[:200])


# ══════════════════════════════════════════════════════════════════════════════
#  loading
# ══════════════════════════════════════════════════════════════════════════════
def load_rows(path):
    """Rows + error rows, kept apart.  A malformed line is itself a finding, never a crash."""
    good, errs, malformed = [], [], []
    with open(path, 'r') as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception as exc:                       # noqa: BLE001
                malformed.append('line %d: %s' % (i, exc))
                continue
            (errs if ('error' in r) else good).append(r)
    return good, errs, malformed


def _by_arm(rows):
    out = defaultdict(list)
    for r in rows:
        out[r.get('arm')].append(r)
    return out


def _ssorted(vals):
    """sorted() that survives a mixed/None-bearing set.  A missing field must produce a
    FINDING, never a TypeError: a checker that crashes on the very corruption it exists to
    catch is worse than no checker."""
    return sorted(vals, key=lambda v: (v is None, type(v).__name__, repr(v)))


def _median(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return statistics.median(vals) if vals else None


def _est_rows(rows):
    return [r for r in rows if ARM_KIND.get(r.get('arm')) == 'est']


# ══════════════════════════════════════════════════════════════════════════════
#  G-0  structural sanity (not a PREREG gate; the precondition for the others)
# ══════════════════════════════════════════════════════════════════════════════
def gate_0_structure(good, errs, malformed):
    out = []
    unknown = _ssorted({r.get('arm') for r in good} - set(ARM_KIND))
    out.append(Gate(
        'G-0 arm vocabulary', FAIL if unknown else PASS, HALT,
        '%d distinct arms, %d unknown' % (len({r.get('arm') for r in good}), len(unknown)),
        unknown, blocks='everything (an unknown arm means the row set is not this batch)'))

    out.append(Gate(
        'G-0b malformed lines', FAIL if malformed else PASS, HALT,
        '%d unparseable jsonl lines' % len(malformed), malformed,
        blocks='everything (a truncated write means rows may be partial)'))

    # duplicate job cells: a resumable runner that double-appends silently doubles a cell's
    # weight in any median-of-3.  Fewer rows than expected is PENDING; MORE is a defect.
    key = Counter((r.get('arm'), r.get('episode'), r.get('est_seed')) for r in good)
    dups = sorted([('%s|%s|%s' % k, n) for k, n in key.items() if n > 1],
                  key=lambda t: -t[1])
    out.append(Gate(
        'G-0c duplicate cells', FAIL if dups else PASS, HALT,
        '%d (arm, episode, est_seed) cells appear more than once' % len(dups), dups,
        blocks='every aggregate (a doubled cell re-weights med3 and the paired test)'))

    out.append(Gate(
        'G-0d error rows', INVESTIGATE if errs else PASS, INFO,
        '%d error rows (they reduce n; see G-8d for the n<%d check)'
        % (len(errs), N_MIN_PAIRS),
        [str(e.get('job_id') or e.get('error'))[:160] for e in errs[:20]],
        blocks='nothing by itself'))
    return out


def gate_0e_stale_row_gates(good):
    """The runner stamps a per-row `gates_pass` from `arms_config.ARM_GATES`.  That table
    was NOT updated by AMENDMENT_2 (which only changed `run_w4.posthoc`), so it still
    demands `L-ITER0.adam_steps_phase1 == 0` -- impossible per A2-1 -- and it still treats
    `L-MERGE.merge_removed == 0` as a failure, which A-7 explicitly downgraded to a
    descriptive.  Any consumer that filters rows on `gates_pass` therefore deletes P-2's
    entire treatment arm.  This gate exists to make that trap loud."""
    pat = Counter()
    for r in good:
        if 'gates_pass' in r:
            pat[(r.get('arm'), r.get('gates_pass'), tuple(r.get('gate_fails') or ()))] += 1
    stale = sorted('%s gates_pass=%s %s x%d' % (a, p, list(f), n)
                   for (a, p, f), n in pat.items() if not p)
    return Gate(
        'G-0e stale row-gate', NOTE if stale else PASS, INFO,
        'rows carrying gates_pass=False for a reason AMENDMENT_2/A-7 legalised: %d pattern(s)'
        % len(stale), stale,
        blocks='nothing -- BUT `gates_pass` must never be used as a row filter')


# ══════════════════════════════════════════════════════════════════════════════
#  G-1  every estimator row actually took the learned branch (PREREG section 4)
# ══════════════════════════════════════════════════════════════════════════════
def gate_1_route(good):
    est = _est_rows(good)
    if not est:
        return [Gate('G-1 route==learned', PENDING, HALT, 'no estimator rows yet')]
    bad = ['%s|%s route=%r learned_init_calls=%r'
           % (r['arm'], r.get('episode'), r.get('route'), r.get('learned_init_calls'))
           for r in est
           if r.get('route') != 'learned' or r.get('learned_init_calls') != 1]
    return [Gate(
        'G-1 route==learned', FAIL if bad else PASS, HALT,
        '%d/%d est rows have route==learned AND learned_init_calls==1'
        % (len(est) - len(bad), len(est)), bad)]


# ══════════════════════════════════════════════════════════════════════════════
#  G-2 / G-2b / G-2c / G-2d  the intervention budget
# ══════════════════════════════════════════════════════════════════════════════
def gate_2_phase1(good):
    """G-2 (AMENDMENT_2 A2-2): exact step counts, not `>0 vs ==0`.

    A2-1 measured what `learned_init_iters` actually governs: the learned proposal polishes
    each of nsamp=4 posterior samples for polish_iters=80 steps (320) BEFORE the knob is
    consulted, so `L-ITER0` removes the 500-step JOINT FIT, not "the descent".  Requiring 0
    would have halted the batch on an arm that violates nothing.
    """
    out, by = [], _by_arm(good)
    offenders, checked, missing_arms = [], 0, []
    for arm in G2_MANDATED:
        rows = by.get(arm, [])
        if not rows:
            missing_arms.append(arm)
            continue
        want = EXPECT_PHASE1[arm]
        for r in rows:
            checked += 1
            got = r.get('adam_steps_phase1')
            if got != want:
                offenders.append('%s|%s|seed=%s adam_steps_phase1=%r want=%d'
                                 % (arm, r.get('episode'), r.get('est_seed'), got, want))
    if missing_arms and not offenders:
        st = PENDING
    elif offenders:
        st = FAIL
    else:
        st = PASS
    dist = {a: _ssorted({r.get('adam_steps_phase1') for r in by.get(a, [])})
            for a in G2_MANDATED if by.get(a)}
    out.append(Gate(
        'G-2 phase1 budget (mandated)', st, HALT,
        'L-ITER0 must be exactly %d (proposal polish), L-SHIP exactly %d (polish + %d joint '
        'fit); observed %s; %d rows checked; missing arms: %s'
        % (EXPECT_PHASE1['L-ITER0'], EXPECT_PHASE1['L-SHIP'], SHIP_JOINT_STEPS,
           dist, checked, missing_arms or 'none'), offenders))

    # the remaining rows of the same A2-1 table, advisory only (A2-2 promoted two).
    adv, adv_missing = [], []
    for arm, want in EXPECT_PHASE1.items():
        if arm in G2_MANDATED:
            continue
        rows = by.get(arm, [])
        if not rows:
            adv_missing.append(arm)
            continue
        for r in rows:
            if r.get('adam_steps_phase1') != want:
                adv.append('%s|%s|seed=%s adam_steps_phase1=%r want=%d'
                           % (arm, r.get('episode'), r.get('est_seed'),
                              r.get('adam_steps_phase1'), want))
    out.append(Gate(
        'G-2* phase1 budget (advisory)', INVESTIGATE if adv else (PENDING if adv_missing
                                                                  else PASS), INFO,
        'the other rows of the A2-1 table: %s; missing arms: %s'
        % ({a: v for a, v in EXPECT_PHASE1.items() if a not in G2_MANDATED},
           adv_missing or 'none'), adv,
        blocks='nothing by itself, but a deviation here means the A2-1 table is wrong'))
    return out


def gate_2b_unattributed(good):
    """AMENDMENT_1 A-6.1: the grow branch runs Adam INSIDE phase 1; `arms_config` splits it
    onto a phase stack so `L-ITER0` is not credited with 50 steps it did not choose.  Any
    step that lands in neither bucket means the stack lost a frame -> the counters are no
    longer trustworthy, which is an alarm, not a halt (A-6 wording: "경보")."""
    bad = ['%s|%s|seed=%s adam_steps_other=%r'
           % (r['arm'], r.get('episode'), r.get('est_seed'), r.get('adam_steps_other'))
           for r in _est_rows(good) if r.get('adam_steps_other')]
    return Gate(
        'G-2b unattributed steps', INVESTIGATE if bad else PASS, INFO,
        '%d est rows with adam_steps_other > 0 (must be 0)' % len(bad), bad,
        blocks='nothing directly -- but it falsifies the G-2 counter, so investigate '
               'BEFORE trusting G-2')


def gate_2c_budget_parity(good):
    """AMENDMENT_2 A2-4.  `S-2b` (`L-VP < L-STEP400`) is a parameterisation-only contrast
    ONLY while the two arms spend the identical Phase-1 budget.  If they diverge, S-2b
    silently becomes a budget contrast and its interpretation flips."""
    by = _by_arm(good)
    bv = _ssorted({r.get('adam_steps_phase1') for r in by.get('L-VP', [])})
    bs = _ssorted({r.get('adam_steps_phase1') for r in by.get('L-STEP400', [])})
    if not bv or not bs:
        st = PENDING
    else:
        st = PASS if bv == bs else FAIL
    return Gate(
        'G-2c budget parity', st, SCOPED,
        'L-VP phase1=%s vs L-STEP400 phase1=%s (both expected [%d])'
        % (bv, bs, EXPECT_PHASE1['L-VP']),
        [] if st != FAIL else ['L-VP=%s' % bv, 'L-STEP400=%s' % bs],
        blocks='S-2b only -- it stops being a "parameterisation alone" contrast. '
               'NOTE: run_w4.posthoc() escalates this to a batch halt; AMENDMENT_1 A-8 '
               'lists only G-1/G-2/G-3/G-5/G-7/G-8 as halting, so the two differ. '
               'Resolve the discrepancy before printing anything if this FAILs.')


def gate_2d_delta_readback(good):
    """Not in the PREREG's numbered list -- it is the CLAUDE.md discipline turned into a
    gate: `effective_config` must be an OBJECT read-back, never a copy of the intent dict
    (S13b found an ensemble that the intent table showed at 100% and the object showed at
    14%).  Here it verifies that each arm's delta actually REACHED the estimator object,
    and that lambda_l1 is the shipping 1e-4 on every arm (the contamination class that
    invalidated BLANKSLATE)."""
    bad = []
    for r in _est_rows(good):
        arm, eff = r['arm'], r.get('effective_config') or {}
        if not isinstance(eff, dict):
            bad.append('%s|%s effective_config is not a dict' % (arm, r.get('episode')))
            continue
        if eff.get('config_is_input_dict') is not False:
            bad.append('%s|%s effective_config is an INPUT-DICT COPY (intent table)'
                       % (arm, r.get('episode')))
        if eff.get('config_id_matches_wrapper') is not True:
            bad.append('%s|%s wrapper/core config identity mismatch' % (arm, r.get('episode')))
        want_it = EXPECT_INIT_ITERS.get(arm)
        got_it = eff.get('cfg:learned_init_iters')
        if want_it is not None and got_it != want_it:
            bad.append('%s|%s cfg:learned_init_iters=%r want=%r'
                       % (arm, r.get('episode'), got_it, want_it))
        want_mf = EXPECT_MERGE_FRAC.get(arm, None)
        got_mf = eff.get('cfg:learned_merge_frac')
        if got_mf != want_mf:
            bad.append('%s|%s cfg:learned_merge_frac=%r want=%r'
                       % (arm, r.get('episode'), got_mf, want_mf))
        if eff.get('lambda_l1') != EXPECT_LAMBDA_L1:
            bad.append('%s|%s effective lambda_l1=%r want=%r'
                       % (arm, r.get('episode'), eff.get('lambda_l1'), EXPECT_LAMBDA_L1))
        if eff.get('init_method') != 'learned':
            bad.append('%s|%s effective init_method=%r want=learned'
                       % (arm, r.get('episode'), eff.get('init_method')))
    est = _est_rows(good)
    return Gate(
        'G-2d arm-delta read-back', (PENDING if not est else (FAIL if bad else PASS)), HALT,
        '%d est rows: object read-back of learned_init_iters / learned_merge_frac / '
        'lambda_l1 / init_method against the frozen arm table' % len(est), bad,
        blocks='the whole verdict (an arm whose delta never reached the object is not '
               'the arm that was pre-registered)')


# ══════════════════════════════════════════════════════════════════════════════
#  G-3  vp_steps  (PREREG section 4, generalised by AMENDMENT_2 A2-3)
# ══════════════════════════════════════════════════════════════════════════════
def gate_3_vp(good):
    bad = []
    for r in good:
        arm = r.get('arm')
        want = EXPECT_VP.get(arm)
        if want is None:
            continue
        if r.get('vp_steps') != want:
            bad.append('%s|%s|seed=%s vp_steps=%r want=%d'
                       % (arm, r.get('episode'), r.get('est_seed'), r.get('vp_steps'), want))
    dist = {a: sorted({r.get('vp_steps') for r in v}) for a, v in sorted(_by_arm(good).items())}
    # A2-3's own failure mode: naming L-VP by hand made L-VP-NW fail a gate it did not
    # violate.  So the expectation is declaration-driven here too.
    return Gate(
        'G-3 vp_steps', FAIL if bad else (PENDING if not good else PASS), HALT,
        'L-VP and L-VP-NW must be %d, every other arm 0; observed %s' % (VP_STEPS, dist),
        bad)


def gate_3b_vp_provenance(good):
    """Surrogate for G-7 (phase-convention identity).  The identity test itself needs an
    execution and this file runs nothing, so the row-level evidence used instead is:
    every VARPRO arm must report the FROZEN varpro.py digest and must have actually
    entered the patched routine.  PREREG section 2 forbids reusing `amplitude_refit`
    because its `arctan2(b, a)` has the sign flipped for the `w*cos(theta+phi)` forward
    model (W1 and W2 independently found the same line)."""
    frozen = _frozen_md5('varpro.py')
    bad, n = [], 0
    for r in good:
        if not EXPECT_VP.get(r.get('arm')):
            continue
        n += 1
        d = r.get('vp_describe') or {}
        if not d.get('installed'):
            bad.append('%s|%s vp_describe.installed is falsy' % (r['arm'], r.get('episode')))
        if frozen and d.get('code_md5') != frozen:
            bad.append('%s|%s varpro code_md5=%r != FROZEN %r'
                       % (r['arm'], r.get('episode'), d.get('code_md5'), frozen))
        if (r.get('vp_entries') or 0) < 1:
            bad.append('%s|%s vp_entries=%r (<1: the patch never ran)'
                       % (r['arm'], r.get('episode'), r.get('vp_entries')))
        want_w = 'none' if r['arm'] == 'L-VP-NW' else 'confidence'
        if d.get('vp_cfg_weighting') != want_w:
            bad.append('%s|%s vp weighting=%r want=%r'
                       % (r['arm'], r.get('episode'), d.get('vp_cfg_weighting'), want_w))
    return Gate(
        'G-3b/G-7 vp provenance', (PENDING if not n else (FAIL if bad else PASS)), HALT,
        '%d VARPRO rows checked against FROZEN varpro.py md5 %s (G-7 itself is the run-free '
        'identity test inside run_w4.gates(); this is its row-level surrogate)'
        % (n, frozen or 'UNKNOWN'), bad,
        blocks='P-1 and every VARPRO claim (a different varpro.py is a different treatment)')


# ══════════════════════════════════════════════════════════════════════════════
#  G-4  merge  (PREREG section 4, DOWNGRADED to a record by AMENDMENT_1 A-7)
# ══════════════════════════════════════════════════════════════════════════════
def gate_4_merge(good):
    """A-7 killed `S-1` before the run: the repository already measured merge_frac=0.5 on
    2026-07-26 (20 crowded cells, 0 wins, no_harm FAIL p=0.03125), the 0.5 radius is
    1.10-1.18 Rayleigh (over-merging), and dedup_dk=0.03 makes ~25/64 mathematically
    bit-identical no-ops.  So `L-MERGE` still runs, but only as descriptive D-4, and this
    gate can never halt the batch."""
    by = _by_arm(good)
    lm = by.get('L-MERGE', [])
    others = {a: _ssorted({r.get('merge_removed') for r in v})
              for a, v in by.items() if ARM_KIND.get(a) == 'est' and a != 'L-MERGE'}
    leak = ['%s merge_removed distinct=%s' % (a, s) for a, s in others.items() if s != [0]]
    if not lm:
        st, detail = PENDING, 'no L-MERGE rows yet'
    else:
        med = _median([r.get('merge_removed') for r in lm])
        noop = sum(1 for r in lm if not r.get('merge_removed'))
        st = PASS if (med or 0) > 0 and not leak else (
            FAIL if leak else 'S-1/D-4 UNDECIDED-NO-INTERVENTION')
        detail = ('L-MERGE merge_removed median=%s, merge_calls median=%s, %d/%d rows are '
                  'no-ops (A-7 predicted ~25/64 bit-identical); other est arms %s'
                  % (med, _median([r.get('merge_calls') for r in lm]), noop, len(lm), others))
    return Gate(
        'G-4 merge_removed', st, SCOPED, detail, leak,
        blocks='the L-MERGE descriptive (D-4) only -- NEVER the batch, per AMENDMENT_1 A-7 '
               'and A-8 (the verdict-ordering rule excludes G-4)')


# ══════════════════════════════════════════════════════════════════════════════
#  G-5  bit determinism  (PREREG section 4 / D-3)
# ══════════════════════════════════════════════════════════════════════════════
def gate_5_determinism(good):
    """W3 taught the batch that a `git_head` stamp does not pin reproduction: a dirty tree
    or a concurrent session editing an import-path file slips through.  `L-SHIP-DUP` is the
    live probe -- same delta, same seed, run again.  |delta| must be EXACTLY 0 on all 64."""
    by = _by_arm(good)
    ship = {}
    for r in by.get('L-SHIP', []):
        if r.get('est_seed') == EST_SEEDS[0]:
            ship[r.get('episode')] = r.get('avg_nRMSE')
    pairs, nonzero, orphan = [], [], []
    for r in by.get('L-SHIP-DUP', []):
        ep = r.get('episode')
        if ep not in ship:
            orphan.append('%s (no L-SHIP@%d partner)' % (ep, EST_SEEDS[0]))
            continue
        d = abs(float(r.get('avg_nRMSE')) - float(ship[ep]))
        pairs.append(d)
        if d != 0.0:
            nonzero.append('%s |delta|=%.3e' % (ep, d))
        if r.get('est_seed') != EST_SEEDS[0]:
            nonzero.append('%s DUP est_seed=%r (must be %d)'
                           % (ep, r.get('est_seed'), EST_SEEDS[0]))
    if not pairs:
        st = PENDING
    elif nonzero:
        st = FAIL
    elif len(pairs) < N_EP_EXPECTED:
        st = PENDING
    else:
        st = PASS
    return Gate(
        'G-5 dup determinism', st, HALT,
        '%d/%d pairs, max|delta|=%s, nonzero=%d, orphans=%d'
        % (len(pairs), N_EP_EXPECTED, ('%.3e' % max(pairs)) if pairs else 'n/a',
           len(nonzero), len(orphan)), nonzero + orphan,
        blocks='the whole verdict (PREREG D-3: nondeterminism halts)')


# ══════════════════════════════════════════════════════════════════════════════
#  G-6 / G-6b / G-6c  the field contract and the frame-parity BLOCKER class
# ══════════════════════════════════════════════════════════════════════════════
def dcc_missing(row):
    kind = ARM_KIND.get(row.get('arm'))
    req = list(DCC_COMMON) + list({'est': DCC_EST, 'cvx': DCC_CVX,
                                   'zero': DCC_ZERO}.get(kind, ()))
    miss = []
    for k in req:
        if k not in row:
            miss.append(k)
        elif row[k] is None and not (kind in ('zero', 'cvx') and k in DCC_NULLABLE_FOR_DET):
            miss.append(k + '=null')
    for fld in ('shell_rmse', 'zero_shell', 'shell_frac'):
        d = row.get(fld)
        if not isinstance(d, dict):
            miss.append(fld + '{} not-a-dict')
        else:
            miss += ['%s[%s]' % (fld, k) for k in SHELL_KEYS if k not in d]
            # PREREG section 6: the `obs:` prefix is load-bearing.  A `ship:` key here
            # means the batch drifted onto the other quantity (1.8-4.1x apart).
            miss += ['%s[%s] WRONG-PREFIX' % (fld, k) for k in d
                     if isinstance(k, str) and not k.startswith('obs:')]
    return miss


def gate_6_dcc(good):
    bad = []
    for r in good:
        m = dcc_missing(r)
        if m:
            bad.append('%s|%s|seed=%s -> %s'
                       % (r.get('arm'), r.get('episode'), r.get('est_seed'), m[:8]))
    return Gate(
        'G-6 DCC (by arm kind)', (PENDING if not good else (FAIL if bad else PASS)), HALT,
        '%d/%d rows satisfy the contract for their own arm kind (est/cvx/zero -- flat '
        'contracts are the S17 self-kill, AMENDMENT_1 A-6.2)' % (len(good) - len(bad),
                                                                 len(good)), bad,
        blocks='the whole verdict (PREREG section 4 G-6: "no verdict printed")')


def gate_6b_frame_parity(good):
    """The `b1_full_comparison.json` defect, promoted to a gate (AMENDMENT_1 A-12): that
    file scored WaveSplat on 120-220 frames and the baselines on 301, in 100/100 episodes,
    and nobody noticed for weeks because `n_all` was never recorded.  Both counts are
    checked: `n_match` (frames actually scored) and `n_all` (frames the episode has)."""
    bad_m, bad_a = [], []
    for ep in _ssorted({r.get('episode') for r in good}):
        rs = [r for r in good if r.get('episode') == ep]
        nm = {(str(r.get('arm')), r.get('n_match')) for r in rs}
        if len({m for _, m in nm}) > 1:
            bad_m.append('%s n_match=%s' % (ep, _ssorted(nm)))
        na = {(str(r.get('arm')), r.get('n_all')) for r in rs}
        if len({a for _, a in na}) > 1:
            bad_a.append('%s n_all=%s' % (ep, _ssorted(na)))
    bad = bad_m + bad_a
    return Gate(
        'G-6b frame parity', (PENDING if not good else (FAIL if bad else PASS)), HALT,
        '%d episodes with an n_match split, %d with an n_all split (arms must be scored on '
        'the SAME frames)' % (len(bad_m), len(bad_a)), bad,
        blocks='every cross-arm comparison, i.e. the whole verdict')


def gate_6c_zero_identity(good, tol=1e-12):
    bad = []
    for ep in _ssorted({r.get('episode') for r in good}):
        zs = [r.get('zero_nRMSE') for r in good
              if r.get('episode') == ep and isinstance(r.get('zero_nRMSE'), (int, float))]
        if zs and (max(zs) - min(zs)) > tol:
            bad.append('%s spread=%.3e' % (ep, max(zs) - min(zs)))
    return Gate(
        'G-6c zero-field identity', (PENDING if not good else (FAIL if bad else PASS)), HALT,
        '%d episodes where zero_nRMSE differs across arms (tol %g)' % (len(bad), tol), bad,
        blocks='the IUT (`L-SHIP < ZERO`) and therefore the whole verdict')


# ══════════════════════════════════════════════════════════════════════════════
#  G-8  provenance and census
# ══════════════════════════════════════════════════════════════════════════════
def _frozen_md5(name):
    """md5 recorded in FROZEN_md5.txt for `name` (None if the file/entry is absent)."""
    path = os.path.join(HERE, 'FROZEN_md5.txt')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        for line in fh:
            m = re.match(r'^([0-9a-f]{32})\s+(\S+)\s*$', line.strip())
            if m and os.path.basename(m.group(2)) == os.path.basename(name):
                return m.group(1)
    return None


def gate_8_provenance(good):
    out = []
    digs = _ssorted({r.get('code_md5') for r in good})
    out.append(Gate(
        'G-8 one code digest', (PENDING if not good else (PASS if len(digs) == 1 else FAIL)),
        HALT, '%d distinct code_md5: %s' % (len(digs), digs),
        [] if len(digs) <= 1 else digs,
        blocks='the whole verdict (two digests = two different systems in one table)'))

    heads = _ssorted({r.get('git_head') for r in good})
    out.append(Gate(
        'G-8b one git_head', (PENDING if not good else (PASS if len(heads) == 1 else FAIL)),
        HALT, '%d distinct git_head: %s' % (len(heads), heads),
        [] if len(heads) <= 1 else heads,
        blocks='the whole verdict'))

    # the code fingerprint the runner stamped, cross-checked against the file it wrote.
    # The digest covers `wavesplat/**`, `wavesim/**`, `configs/*.yaml` AND the batch's own
    # run_w4.py / PREREG.md, so it legitimately moves whenever an amendment edits the
    # runner.  Smoke rows written BEFORE such an edit therefore carry an older digest --
    # that is expected, not a defect, which is why this is INVESTIGATE and never halting.
    fp = os.path.join(HERE, 'code_fingerprint.json')
    detail, off, st = 'code_fingerprint.json absent', [], INVESTIGATE
    if os.path.exists(fp):
        try:
            j = json.load(open(fp))
            same = (not digs) or (digs == [j.get('digest')])
            st = PASS if same else INVESTIGATE
            detail = ('row code_md5 %s vs code_fingerprint.json digest %s (%d files, '
                      'git_dirty_n=%s)%s'
                      % (digs, j.get('digest'), len(j.get('files') or {}),
                         j.get('git_dirty_n'),
                         '' if same else '  -- EXPECTED for smoke rows predating the last '
                                         'runner edit; a defect for main-run rows'))
            if not same:
                off = ['rows=%s fingerprint=%s' % (digs, j.get('digest'))]
        except Exception as exc:                          # noqa: BLE001
            st, detail = INVESTIGATE, 'code_fingerprint.json unreadable: %s' % exc
    out.append(Gate(
        'G-8c fingerprint match', st, INFO, detail, off,
        blocks='nothing directly; a mismatch on MAIN-RUN rows means the digest was '
               're-stamped mid-batch'))

    # Independent freeze verification.  `git_head` alone does not pin reproduction (S13 and
    # S13b both lost it to a dirty tree / a concurrent session editing an import-path file),
    # which is why the batch records file md5s.  This re-checks them against disk NOW, with
    # no help from the runner.
    frozen_path = os.path.join(HERE, 'FROZEN_md5.txt')
    bad_f, n_f = [], 0
    if not os.path.exists(frozen_path):
        st_f, det_f = INVESTIGATE, 'FROZEN_md5.txt absent -- the freeze is unverifiable'
    else:
        with open(frozen_path) as fh:
            for line in fh:
                m = re.match(r'^([0-9a-f]{32})\s+(\S+)\s*$', line.strip())
                if not m:
                    continue
                want, rel = m.group(1), m.group(2)
                path = os.path.normpath(os.path.join(HERE, rel))
                if not os.path.exists(path):
                    bad_f.append('%s MISSING from disk' % rel)
                    continue
                n_f += 1
                got = hashlib.md5(open(path, 'rb').read()).hexdigest()
                if got != want:
                    bad_f.append('%s md5=%s FROZEN=%s' % (rel, got, want))
        st_f = FAIL if bad_f else (PASS if n_f else INVESTIGATE)
        det_f = '%d frozen files re-hashed from disk, %d deviations' % (n_f, len(bad_f))
    out.append(Gate(
        'G-8g freeze verification', st_f, SCOPED, det_f, bad_f,
        blocks='the "self-attested freeze + git-anchored" claim, and any statement that '
               'the run used the pre-registered code. It does NOT by itself invalidate '
               'rows already written by the frozen build (check G-8 / G-8c for that).'))

    dirty = _ssorted({r.get('git_dirty_n') for r in good if r.get('git_dirty_n') is not None})
    out.append(Gate(
        'G-8e dirty tree', NOTE, INFO,
        'git_dirty_n observed %s -- a git_head stamp does NOT pin a dirty tree (S13/S13b '
        'lost reproduction exactly here); the md5s in FROZEN_md5.txt are what pins it'
        % dirty, [], blocks='nothing -- it is a caveat that must appear in the write-up'))
    return out


def gate_8d_census(good):
    """Cell census.  FEWER rows than expected is PENDING (the batch is not finished);
    an arm with MORE than expected, or a wrong seed set, is a defect."""
    by = _by_arm(good)
    lines, short, over = [], [], []
    for arm in sorted(EXPECT_ROWS):
        rows = by.get(arm, [])
        want = EXPECT_ROWS[arm]
        eps = len({r.get('episode') for r in rows})
        seeds = _ssorted({r.get('est_seed') for r in rows})
        lines.append('%-11s %4d/%-4d rows  %2d episodes  seeds=%s'
                     % (arm, len(rows), want, eps, seeds))
        if len(rows) > want:
            over.append('%s has %d rows > %d expected' % (arm, len(rows), want))
        elif len(rows) < want:
            short.append('%s %d/%d' % (arm, len(rows), want))
        want_seeds = ([None] if ARM_KIND[arm] != 'est' else
                      ([EST_SEEDS[0]] if arm == 'L-SHIP-DUP' else list(EST_SEEDS)))
        if rows and seeds != _ssorted(want_seeds):
            if not set(seeds) <= set(want_seeds):
                over.append('%s seed set %s not within %s' % (arm, seeds, want_seeds))
    total = len(good)
    st = FAIL if over else (PASS if total == EXPECT_TOTAL and not short else PENDING)
    return Gate(
        'G-8d cell census', st, HALT,
        'total %d/%d rows; short: %s' % (total, EXPECT_TOTAL, short or 'none'),
        over + lines,
        blocks='the whole verdict when a cell is OVER-filled; an UNDER-filled table is '
               'PENDING, not FAIL -- an unfinished batch has simply not been measured yet')


def gate_8f_pair_availability(good):
    """PREREG section 5 D-4: fewer than 56 usable pairs -> `UNDECIDED (n)`.  Counting
    available pairs is not a verdict; it is the eligibility of one."""
    by = _by_arm(good)

    def eps_with(arm, full=False):
        c = Counter(r.get('episode') for r in by.get(arm, []))
        n_need = 3 if (full and ARM_KIND.get(arm) == 'est' and arm != 'L-SHIP-DUP') else 1
        return {e for e, n in c.items() if n >= n_need}

    rows = []
    for label, a, b in (('P-1  L-VP vs L-SHIP', 'L-VP', 'L-SHIP'),
                        ('P-2  L-ITER0 vs L-SHIP', 'L-ITER0', 'L-SHIP'),
                        ('IUT  L-SHIP vs ZERO', 'L-SHIP', 'ZERO'),
                        ('S-2  L-VP vs L-ITER0', 'L-VP', 'L-ITER0'),
                        ('S-2b L-VP vs L-STEP400', 'L-VP', 'L-STEP400'),
                        ('S-2c L-VP vs L-VP-NW', 'L-VP', 'L-VP-NW')):
        any_ = len(eps_with(a) & eps_with(b))
        med3 = len(eps_with(a, True) & eps_with(b, True))
        rows.append('%-24s pairs(any seed)=%3d  pairs(med3 complete)=%3d' % (label, any_, med3))
    prim = [r for r in rows if r.startswith(('P-1', 'P-2'))]
    ok = all(int(r.split('med3 complete)=')[1]) >= N_MIN_PAIRS for r in prim) if prim else False
    return Gate(
        'G-8f pair availability', PASS if ok else PENDING, SCOPED,
        'PREREG D-4 needs >= %d complete median-of-3 pairs per PRIMARY contrast' % N_MIN_PAIRS,
        rows, blocks='the affected contrast becomes `UNDECIDED (n)` -- not a FAIL')


# ══════════════════════════════════════════════════════════════════════════════
#  W3 bit anchor  (PREREG section 2: ZERO / ORA-K-CVX reused UNMODIFIED)
# ══════════════════════════════════════════════════════════════════════════════
def gate_w3_anchor(good, w3_path=None):
    """`ZERO` and `ORA-K-CVX` are W3's arms imported without modification, so on any episode
    both batches scored, |delta| must be EXACTLY 0.  Only the overlap is checkable: the s8
    corpus is not in W3's sample, so a missing partner is 'not applicable', never a failure.
    A break here is not cosmetic -- the IUT (`L-SHIP < ZERO`) and the ORA-K-CVX upper bound
    both stand on these two arms."""
    w3_path = w3_path or os.path.join(W3DIR, 'results.jsonl')
    if not os.path.exists(w3_path):
        return Gate('W3 bit anchor', PENDING, SCOPED,
                    'W3 results.jsonl not found at %s' % w3_path, [],
                    blocks='the "unmodified reuse" claim for ZERO / ORA-K-CVX')
    ref = {}
    with open(w3_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:                             # noqa: BLE001
                continue
            if 'error' in r or r.get('arm') not in ('ZERO', 'ORA-K-CVX'):
                continue
            ref[(r['arm'], r.get('episode'))] = r.get('avg_nRMSE')
    checked, bad, na = 0, [], 0
    for r in good:
        arm = r.get('arm')
        if arm not in ('ZERO', 'ORA-K-CVX'):
            continue
        key = (arm, r.get('episode'))
        if key not in ref:
            na += 1
            continue
        checked += 1
        d = abs(float(r.get('avg_nRMSE')) - float(ref[key]))
        if d != 0.0:
            bad.append('%s|%s W4=%.12f W3=%.12f |delta|=%.3e'
                       % (arm, r.get('episode'), r['avg_nRMSE'], ref[key], d))
    return Gate(
        'W3 bit anchor', (PENDING if checked == 0 else (FAIL if bad else PASS)), HALT,
        '%d overlapping rows checked, %d not-applicable (s8 corpus is not in W3), '
        '%d mismatches' % (checked, na, len(bad)), bad,
        blocks='the IUT and every ORA-K-CVX bound claim -- i.e. the verdict, since the '
               'IUT gates PRIMARY (PREREG section 1 / D-1)')


# ══════════════════════════════════════════════════════════════════════════════
#  reading rule (AMENDMENT_1 A-5) -- not a gate, it fixes the WORDING of P-2
# ══════════════════════════════════════════════════════════════════════════════
def reading_rule_a5(good):
    ship = [r for r in good if r.get('arm') == 'L-SHIP']
    hits = [(r.get('counters') or {}).get('clamp_hits_p1') for r in ship]
    hits = [h for h in hits if isinstance(h, int)]
    if not hits:
        return Gate('A-5 P-2 wording', PENDING, INFO,
                    'clamp_hits_p1 not available on L-SHIP rows yet', [],
                    blocks='the wording of the P-2 label')
    nz = sum(1 for h in hits if h > 0)
    if nz == 0:
        verdict = ('all %d L-SHIP rows have clamp_hits_p1 == 0 -> at effect level the arm '
                   'is single-factor: P-2 must be read as "the 500-step JOINT FIT" '
                   '(AMENDMENT_2 A2-1 wording), NOT "the descent".' % len(hits))
    else:
        verdict = ('%d/%d L-SHIP rows have clamp_hits_p1 > 0 -> the steepness clamp fired, '
                   'so L-ITER0 removes TWO things: P-2 must be read as "the whole Phase-1 '
                   'loop (joint fit + clamp projection)".' % (nz, len(hits)))
    return Gate('A-5 P-2 wording', NOTE, INFO, verdict, [],
                blocks='nothing -- it DICTATES the wording of the P-2 label')


# ══════════════════════════════════════════════════════════════════════════════
#  AMENDMENT_1 A-11 free descriptives.  ⛔ NEVER promote any of these to a hypothesis.
# ══════════════════════════════════════════════════════════════════════════════
def descriptives(good):
    """Zero extra runs; strictly the D family (PREREG section 3).  These are NOT
    pre-registered contrasts: A-11 says so in as many words.  Printed only on request so a
    routine integrity check does not leak outcome-adjacent numbers."""
    by = _by_arm(good)
    out = {'_warning': 'D-family descriptives. AMENDMENT_1 A-11: never promoted to a '
                       'hypothesis. No threshold, no direction, no p-value.'}
    for arm in sorted(by):
        rows = by[arm]

        def med(key):
            return _median([r.get(key) for r in rows])

        rec = {
            'n_rows': len(rows),
            'wall_s_median': med('wall_s'),          # D-2
            'wall_s_total_h': (sum(r.get('wall_s') or 0 for r in rows) / 3600.0),
            'window_resid_rms_median': med('window_resid_rms'),   # D-1
            'n_atoms_final_median': med('n_atoms_final'),
            'merge_removed_median': med('merge_removed'),         # D-3
            # rho / VR: run_w4 keeps them as the two derived aggregates, not the raw pair.
            'decorr_mean_median (1 - rho^2)': med('decorr_mean'),
            'shrink_mean_median ((sqrt(VR)-rho)^2)': med('shrink_mean'),
            'obs_resid_rms_mean_median': med('obs_resid_rms_mean'),
            'elapsed_ms_mean_median': med('elapsed_ms_mean'),
            'init_time_s_median': med('init_time_s'),
        }
        # RMSE_near: A-11 asked for it; the runner does not persist it (run_w4.py's return
        # dict has no such key), so report ABSENT rather than silently omitting -- absence
        # of the field is not absence of the effect.
        near = [r.get('RMSE_near') for r in rows if r.get('RMSE_near') is not None]
        rec['RMSE_near_median'] = _median(near) if near else 'ABSENT (not persisted by run_w4)'
        # VARPRO-only: the before/after window residual the patch itself reports.
        vd = [r.get('vp_diag') for r in rows if isinstance(r.get('vp_diag'), dict)]
        if vd:
            rec['vp_win_rms_before_median'] = _median([d.get('vp_win_rms_before') for d in vd])
            rec['vp_win_rms_after_median'] = _median([d.get('vp_win_rms_after') for d in vd])
            rec['vp_k_shift_median'] = _median([d.get('vp_k_shift') for d in vd])
            rec['vp_clamp_hits_median'] = _median([d.get('vp_clamp_hits') for d in vd])
        out[arm] = rec
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  driver
# ══════════════════════════════════════════════════════════════════════════════
def run_all(rows_path, w3_path=None):
    good, errs, malformed = load_rows(rows_path)
    gates = []
    gates += gate_0_structure(good, errs, malformed)
    gates.append(gate_0e_stale_row_gates(good))
    gates += gate_1_route(good)
    gates += gate_2_phase1(good)
    gates.append(gate_2b_unattributed(good))
    gates.append(gate_2c_budget_parity(good))
    gates.append(gate_2d_delta_readback(good))
    gates.append(gate_3_vp(good))
    gates.append(gate_3b_vp_provenance(good))
    gates.append(gate_4_merge(good))
    gates.append(gate_5_determinism(good))
    gates.append(gate_6_dcc(good))
    gates.append(gate_6b_frame_parity(good))
    gates.append(gate_6c_zero_identity(good))
    gates += gate_8_provenance(good)
    gates.append(gate_8d_census(good))
    gates.append(gate_8f_pair_availability(good))
    gates.append(gate_w3_anchor(good, w3_path))
    gates.append(reading_rule_a5(good))
    return good, errs, gates


def verdict_permission(gates):
    """The single output that matters: may `analyze_w4.py` print a verdict?"""
    halt_fail = [g for g in gates if g.severity == HALT and g.status == FAIL]
    halt_pend = [g for g in gates if g.severity == HALT and g.status == PENDING]
    scoped = [g for g in gates if g.severity == SCOPED and g.status not in (PASS, NOTE)]
    if halt_fail:
        return 'FORBIDDEN', halt_fail, scoped, 2
    if halt_pend:
        return 'NOT-YET (PENDING)', halt_pend, scoped, 1
    return 'PERMITTED', [], scoped, 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Independent post-hoc gate check for W4 (no run_w4 import).')
    ap.add_argument('rows', help='path to a jsonl of W4 rows (REQUIRED -- there is no '
                                 'default, so results.jsonl is never opened by accident)')
    ap.add_argument('--w3', default=None, help='override W3 results.jsonl for the bit anchor')
    ap.add_argument('--json', default=None, help='write the full gate report here')
    ap.add_argument('--descriptives', action='store_true',
                    help='also print the AMENDMENT_1 A-11 free descriptives (D family)')
    ap.add_argument('--offenders', type=int, default=12,
                    help='max offending rows printed per gate (0 = all)')
    a = ap.parse_args(argv)

    if not os.path.exists(a.rows):
        print('gatecheck: no such file: %s' % a.rows, file=sys.stderr)
        return 3

    good, errs, gates = run_all(a.rows, a.w3)
    perm, blockers, scoped, code = verdict_permission(gates)

    w = 30
    print('=' * 100)
    print('W4 GATECHECK -- independent re-derivation (imports nothing from run_w4.py)')
    print('rows      : %s' % os.path.abspath(a.rows))
    print('usable    : %d   error rows: %d' % (len(good), len(errs)))
    print('=' * 100)
    for g in gates:
        print('%-*s %-12s [%s] %s' % (w, g.name, g.status, g.severity, g.detail))
        lim = len(g.offenders) if a.offenders == 0 else a.offenders
        for o in g.offenders[:lim]:
            print(' ' * (w + 1) + '  - %s' % (o if not isinstance(o, tuple) else str(o)))
        if len(g.offenders) > lim:
            print(' ' * (w + 1) + '  ... %d more' % (len(g.offenders) - lim))

    print('-' * 100)
    print('VERDICT PRINTING: %s' % perm)
    for g in blockers:
        print('   BLOCKED BY %-26s (%s) -> %s' % (g.name, g.status, g.blocks))
    for g in scoped:
        print('   SCOPED     %-26s (%s) -> %s' % (g.name, g.status, g.blocks))
    if not blockers and not scoped:
        print('   no halting or scoped gate is outstanding')
    print('-' * 100)

    if a.descriptives:
        print('\n--- AMENDMENT_1 A-11 free descriptives (D family; NEVER a hypothesis) ---')
        print(json.dumps(descriptives(good), indent=2, default=str))

    if a.json:
        rep = dict(rows_path=os.path.abspath(a.rows), n_usable=len(good), n_error=len(errs),
                   verdict_printing=perm, exit_code=code,
                   gates=[g.as_dict() for g in gates])
        if a.descriptives:
            rep['descriptives'] = descriptives(good)
        with open(a.json, 'w') as fh:
            json.dump(rep, fh, indent=2, default=str)
        print('wrote %s' % a.json)
    return code


if __name__ == '__main__':
    sys.exit(main())
