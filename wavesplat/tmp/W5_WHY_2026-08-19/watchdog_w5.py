#!/usr/bin/env python3
"""W5 watchdog -- file state only, and every expectation IMPORTED, never retyped.

Two W4 lessons are wired in:
  * process tables lie about your own monitor (W3 lost 8 h to `pgrep` self-matching), so
    liveness comes from results.jsonl row count + mtime and nothing else;
  * a watchdog carrying its OWN copy of the intervention expectation raised two false
    GATE-VIOLATIONs on correct data, because AMENDMENT_2 updated the runner and not this
    file.  Here the numbers come from arms_config_w5 -- the same object the runner and the
    analyser read.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arms_config_w5 as A5                                   # noqa: E402

# The runner's G-9 asserts this: the watchdog must NAME where its expectations come from,
# so a future amendment cannot update the runner and silently leave a stale copy here.
# W4 shipped exactly that bug and it fired two false GATE-VIOLATIONs on correct rows.
EXPECT_SOURCE = 'arms_config_w5'

OUT = os.path.join(HERE, 'results.jsonl')
N_EP = 64
PLAN = tuple((a, A5.NEW_SPECS[a]['n_seeds']) for a in A5.NEW_ARMS)
TOTAL = sum(n for _, n in PLAN) * N_EP
STALL_S = 1800


def read_state():
    if not os.path.exists(OUT):
        return dict(rows=0, err=0, per_arm={}, bad={}, ok={}, age=None)
    per_arm, bad, okc, rows, err = {}, {}, {}, 0, 0
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
            v = A5.row_gate_verdict(r)                        # the formula, not a copy
            okc[a] = okc.get(a, 0) + bool(v['w5_gates_pass'])
            if not v['w5_gates_pass']:
                bad.setdefault(a, []).append(v['w5_gate_fails'][0])
    return dict(rows=rows, err=err, per_arm=per_arm, bad=bad, ok=okc,
                age=time.time() - os.path.getmtime(OUT))


def line():
    s = read_state()
    if s['rows'] == 0:
        st = 'NOT-STARTED'
    elif s['bad']:
        st = 'GATE-VIOLATION'
    elif s['rows'] >= TOTAL:
        st = 'DONE'
    elif s['age'] is not None and s['age'] > STALL_S:
        st = 'STALLED'
    else:
        st = 'RUNNING'
    print('[%s] %-14s %4d/%d (%5.1f%%)  err=%d  idle=%s'
          % (time.strftime('%H:%M:%S'), st, s['rows'], TOTAL,
             100.0 * s['rows'] / TOTAL, s['err'],
             ('%.0fs' % s['age']) if s['age'] is not None else '-'), flush=True)
    print('    ' + '  '.join('%s=%d/%d' % (a, s['per_arm'].get(a, 0), n * N_EP)
                             for a, n in PLAN), flush=True)
    if s['per_arm']:
        print('    counters: ' + ' | '.join(
            '%s adam==%s vp==%s %d/%d%s'
            % (a, A5.expected_adam_p1(a), A5.expected_vp_steps(a),
               s['ok'].get(a, 0), s['per_arm'].get(a, 0),
               '  <-- ' + s['bad'][a][0] if a in s['bad'] else '')
            for a, _ in PLAN if s['per_arm'].get(a)), flush=True)
    return st


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
