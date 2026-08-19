#!/usr/bin/env python3
"""W5 arm definitions -- the SINGLE SOURCE for every W5 intervention expectation.

PREREG section 4 makes this a design requirement, not a style preference:

    "G-2/G-3 의 기대값은 한 곳에서만 정의하고 러너·워치독·분석기가 그것을 읽는다"

W4 duplicated the expectation (320/820) across the runner's G-2, the runner's G-3,
arms_config.ARM_GATES and watchdog.py, and paid for it FOUR times in one day -- twice as a
false GATE-VIOLATION on correct data.  So here the numbers exist once, as a FORMULA, and
`_verify_single_source()` cross-checks that formula against two independent witnesses:
the literals quoted from the PREREG, and the counters W4 actually measured.

    *** W4's arms_config is EXTENDED IN PLACE, never copied. ***

A copy would leave `W4.arm_config()` looking at the original dict, so every new arm would
KeyError at run time -- "defined in one place that nobody reads" is not single-sourcing.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
W4DIR = os.path.join(ROOT, 'tmp', 'W4_SOLVER_2026-08-19')
for _p in (ROOT, W4DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms_config as AC                                      # noqa: E402  (W4, unmodified)


class SingleSourceError(RuntimeError):
    """The formula disagreed with a witness -- do not run."""


# ── the numbers.  Each appears exactly once. ─────────────────────────────────────
POLISH_STEPS = 320        # learned proposal: nsamp=4 x polish_iters=80 (W4 AMENDMENT_2 A2-1)
SHIP_ITERS = 500          # shipped `learned_init_iters` (sequential_update.py:643)
VP = 400                  # VARPRO steps, as frozen by W4
VP_LONG = 800             # W5 saturation probe

NEW_SPECS = {
    # arm            joint_fit      vp        weighting      seeds factor            vs
    'L-VP-ONLY':    dict(joint_fit=0,          vp=VP,      vp_weighting='confidence',
                         n_seeds=3, factor=('learned_init_iters',), factor_vs='L-VP'),
    'L-VP-ONLY-NW': dict(joint_fit=0,          vp=VP,      vp_weighting='none',
                         n_seeds=3, factor=('vp_weighting',), factor_vs='L-VP-ONLY'),
    'L-VP800':      dict(joint_fit=SHIP_ITERS, vp=VP_LONG, vp_weighting='confidence',
                         n_seeds=3, factor=('vp_steps',), factor_vs='L-VP'),
    'L-SHIP-DUP2':  dict(joint_fit=SHIP_ITERS, vp=0,       vp_weighting='confidence',
                         n_seeds=1, factor=(), factor_vs='L-SHIP', dup_of='L-SHIP'),
}
REUSE_SPECS = {
    'L-SHIP':  dict(joint_fit=SHIP_ITERS, vp=0,  vp_weighting='confidence'),
    'L-VP':    dict(joint_fit=SHIP_ITERS, vp=VP, vp_weighting='confidence'),
    'L-VP-NW': dict(joint_fit=SHIP_ITERS, vp=VP, vp_weighting='none'),
    'L-ITER0': dict(joint_fit=0,          vp=0,  vp_weighting='confidence'),
    'ZERO':    None,                      # no estimator: no counters at all
    'ORA-K-CVX': None,
}
NEW_ARMS = tuple(NEW_SPECS)
REUSE_ARMS = tuple(REUSE_SPECS)
ALL_SPECS = dict(REUSE_SPECS, **NEW_SPECS)

# ── the formula ──────────────────────────────────────────────────────────────────
def expected_adam_p1(arm):
    """Phase-1 Adam steps.  The proposal polish ALWAYS runs (the config knob does not
    reach it); the joint fit is `learned_init_iters`; VARPRO lands in the same phase
    because the runner installs varpro FIRST and the counters SECOND."""
    s = ALL_SPECS.get(arm)
    return None if s is None else POLISH_STEPS + s['joint_fit'] + s['vp']


def expected_vp_steps(arm):
    s = ALL_SPECS.get(arm)
    return None if s is None else s['vp']


def expectation_table():
    return {a: dict(adam_steps_phase1=expected_adam_p1(a), vp_steps=expected_vp_steps(a),
                    vp_weighting=(ALL_SPECS[a] or {}).get('vp_weighting'),
                    reused=(a in REUSE_SPECS))
            for a in ALL_SPECS}


# ── witnesses: the formula must agree with BOTH, or the batch does not run ───────
PREREG_QUOTED = {          # literals copied from PREREG section 4 (transcription, not logic)
    'L-VP-ONLY': 720, 'L-VP-ONLY-NW': 720, 'L-VP800': 1620, 'L-SHIP': 820,
}
W4_MEASURED = {            # counters W4 actually recorded (tmp/W4.../results.jsonl)
    'L-SHIP': 820, 'L-ITER0': 320, 'L-VP': 1220, 'L-VP-NW': 1220,
}


def _verify_single_source():
    bad = []
    for src, table in (('PREREG section 4', PREREG_QUOTED), ('W4 measured', W4_MEASURED)):
        for arm, want in table.items():
            got = expected_adam_p1(arm)
            if got != want:
                bad.append('%s: %s formula=%s witness=%s' % (src, arm, got, want))
    if bad:
        raise SingleSourceError('; '.join(bad))
    return True


# ── in-place registration ────────────────────────────────────────────────────────
_REGISTERED = False


def register():
    """Extend W4's tables IN PLACE so `AC.arm_config()` resolves the new arms, and derive
    every row-level gate FROM THE FORMULA (never a second literal)."""
    global _REGISTERED
    if _REGISTERED:
        return
    _verify_single_source()
    for arm, spec in NEW_SPECS.items():
        delta = {} if spec['joint_fit'] == SHIP_ITERS else {'learned_init_iters': spec['joint_fit']}
        AC.ARM_DELTAS[arm] = delta
        AC.ARM_GATES[arm] = {'adam_steps_phase1': ('==', expected_adam_p1(arm))}
    _REGISTERED = True


def arm_config(arm, **kw):
    register()
    return AC.arm_config(arm, **kw)


def row_gate_verdict(row):
    """Row-level W5 gate, computed from the formula.  Never trust a stored `gates_pass`:
    W4 shipped 192 rows whose stored flag was False for a superseded reason (AMENDMENT_3
    A3-2), and filtering on it would have deleted an entire treatment arm."""
    arm = row.get('arm')
    fails = []
    exp_a, exp_v = expected_adam_p1(arm), expected_vp_steps(arm)
    if exp_a is not None:
        if row.get('adam_steps_phase1') != exp_a:
            fails.append('adam_steps_phase1 %s != %s' % (row.get('adam_steps_phase1'), exp_a))
        if row.get('vp_steps') != exp_v:
            fails.append('vp_steps %s != %s' % (row.get('vp_steps'), exp_v))
    return dict(w5_gates_pass=not fails, w5_gate_fails=fails)


if __name__ == '__main__':
    register()
    print('single-source cross-check: PASS (%d PREREG literals, %d W4 counters)'
          % (len(PREREG_QUOTED), len(W4_MEASURED)))
    print('%-14s %-26s %8s %6s %-11s %s'
          % ('arm', 'delta', 'adam_p1', 'vp', 'weighting', 'source'))
    for a in NEW_ARMS + tuple(x for x in REUSE_ARMS if ALL_SPECS[x]):
        e = expectation_table()[a]
        print('%-14s %-26s %8s %6s %-11s %s'
              % (a, str(AC.ARM_DELTAS.get(a, {})), e['adam_steps_phase1'], e['vp_steps'],
                 e['vp_weighting'], 'W4-reuse' if e['reused'] else 'W5-new'))
