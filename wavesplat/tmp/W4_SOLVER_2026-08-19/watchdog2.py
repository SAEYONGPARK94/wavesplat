#!/usr/bin/env python3
"""W4 watchdog v2 -- progress monitor that CANNOT self-match.

W3 lesson (project_w3_solver_2026-08-16 section 6.1): the W3 monitor used
`pgrep -f run_w3.py`, matched its OWN command line, and reported the batch alive for 8
hours after Phase A had finished.

    *** THIS WATCHDOG NEVER LOOKS AT THE PROCESS TABLE. ***

Liveness comes from FILE STATE ONLY -- row count of results.jsonl plus its mtime -- which
is observable, monotone, and cannot be faked by the watchdog's own existence.

It also carries a W4-specific job the S17 watchdog did not have: PREREG section 4 makes
the INTERVENTION COUNTERS a halting gate (W3 burned 96 runs on an arm that turned out to
intervene 17/96 times).  So the watchdog reports, live, whether each arm's counter is
actually moving -- a stalled counter is a reason to kill the batch early, not to wait.

    python3 watchdog.py             # one-shot status
    python3 watchdog.py --loop 120  # a line every 120 s until DONE or STALLED
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.jsonl')

# (arm, n_seeds) -- ZERO and ORA-K-CVX are deterministic, one seed each (PREREG section 2)
PLAN = (('L-SHIP', 3), ('L-ITER0', 3), ('L-VP', 3), ('L-VP-NW', 3),
        ('L-STEP400', 3), ('L-MERGE', 3),
        ('L-SHIP-DUP', 1), ('ZERO', 1), ('ORA-K-CVX', 1))
N_EP = 64
TOTAL = sum(n for _, n in PLAN) * N_EP
STALL_S = 1800

# PREREG section 4: what each arm's intervention counter MUST look like.
# (counter, predicate, human description)
EXPECT = {
    'L-SHIP':  ('adam_steps_phase1', lambda v: v == 820,      'adam==820'),
    # AMENDMENT_2 A2-1: `learned_init_iters` does NOT govern the whole Phase-1 descent.
    # The learned proposal polishes nsamp=4 posterior samples x polish_iters=80 = 320 steps
    # that the knob never reaches, so the arm's correct value is 320, not 0.  watchdog.py
    # still carries the pre-amendment `== 0` and therefore raised a FALSE GATE-VIOLATION on
    # correct rows; it is inside the code fingerprint and must not be edited during the run.
    'L-ITER0': ('adam_steps_phase1', lambda v: v == 320,      'adam==320'),
    'L-VP':    ('vp_steps',          lambda v: v == 400,      'vp==400'),
    'L-VP-NW': ('vp_steps',          lambda v: v == 400,      'vp==400'),
    'L-STEP400': ('adam_steps_phase1', lambda v: v and v > 0,  'adam>0'),
    # AMENDMENT_1 A-7 demoted L-MERGE/S-1 to a DESCRIPTIVE item and explicitly made G-4 a
    # record, not a halting gate: AUDIT_CON measured that dedup_dk=0.03 makes the merge a
    # mathematical no-op on a large share of episodes.  A zero here is the pre-registered
    # expectation, not a fault, so it must not raise GATE-VIOLATION.
    'L-MERGE': ('merge_removed',     lambda v: v is not None,           'recorded'),
}


def read_state():
    if not os.path.exists(OUT):
        return dict(rows=0, err=0, per_arm={}, ctr={}, bad={}, age=None)
    per_arm, ctr, bad, rows, err = {}, {}, {}, 0, 0
    with open(OUT) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows += 1
            a = r.get('arm', '?')
            per_arm[a] = per_arm.get(a, 0) + 1
            if 'error' in r:
                err += 1
                continue
            if a in EXPECT:
                key, pred, _ = EXPECT[a]
                v = r.get(key)
                ctr[a] = ctr.get(a, 0) + (1 if pred(v) else 0)
                if not pred(v):
                    bad[a] = bad.get(a, 0) + 1
    return dict(rows=rows, err=err, per_arm=per_arm, ctr=ctr, bad=bad,
                age=time.time() - os.path.getmtime(OUT))


def line():
    s = read_state()
    if s['rows'] == 0:
        status = 'NOT-STARTED'
    elif s['rows'] >= TOTAL:
        status = 'DONE'
    elif s['bad']:
        status = 'GATE-VIOLATION'          # PREREG section 4: stop, do not wait it out
    elif s['age'] is not None and s['age'] > STALL_S:
        status = 'STALLED'
    else:
        status = 'RUNNING'
    print('[%s] %-14s %4d/%d (%5.1f%%)  err=%d  idle=%s'
          % (time.strftime('%H:%M:%S'), status, s['rows'], TOTAL,
             100.0 * s['rows'] / TOTAL, s['err'],
             ('%.0fs' % s['age']) if s['age'] is not None else '-'), flush=True)
    print('    ' + '  '.join('%s=%d/%d' % (a, s['per_arm'].get(a, 0), n * N_EP)
                             for a, n in PLAN), flush=True)
    if s['per_arm']:
        parts = []
        for a, (_, _, desc) in EXPECT.items():
            got, tot = s['ctr'].get(a, 0), s['per_arm'].get(a, 0)
            if tot:
                parts.append('%s %s %d/%d%s' % (a, desc, got, tot,
                                                '  <-- VIOLATION' if s['bad'].get(a) else ''))
        if parts:
            print('    counters: ' + ' | '.join(parts), flush=True)
    return status


if __name__ == '__main__':
    if '--loop' in sys.argv:
        every = int(sys.argv[sys.argv.index('--loop') + 1])
        while True:
            st = line()
            if st in ('DONE', 'STALLED', 'GATE-VIOLATION'):
                sys.exit({'DONE': 0}.get(st, 2))
            time.sleep(every)
    else:
        sys.exit({'DONE': 0}.get(line(), 3))
