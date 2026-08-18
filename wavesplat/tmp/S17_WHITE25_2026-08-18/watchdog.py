#!/usr/bin/env python3
"""S17 watchdog -- progress monitor that CANNOT self-match.

W3 lesson (project_w3_solver_2026-08-16 section 6.1): the W3 monitor used
`pgrep -f run_w3.py`, matched its OWN command line, and therefore reported the batch
as alive for 8 hours after Phase A had already finished.

    *** THIS WATCHDOG NEVER LOOKS AT THE PROCESS TABLE. ***

It judges liveness from FILE STATE ONLY: the row count of results.jsonl and its mtime.
That is observable, monotone, and cannot be faked by the watchdog's own existence.

    python3 watchdog.py            # one-shot status line
    python3 watchdog.py --loop 300 # print a line every 300 s until done or stalled
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.jsonl')

ARMS = ('WHITER25', 'WHITER12', 'WHITER6', 'WHITERHS50')
SEEDS = (42, 43, 44)
N_EP = 64                       # crowded only (AMENDMENT_1 A-10)
TOTAL = len(ARMS) * len(SEEDS) * N_EP
STALL_S = 1800                  # 30 min with no new row => STALLED (median run is ~40 s)


def read_state():
    if not os.path.exists(OUT):
        return dict(rows=0, err=0, per_arm={}, mtime=None, age=None)
    per_arm, rows, err = {}, 0, 0
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
            if 'error' in r:
                err += 1
            per_arm[r.get('arm', '?')] = per_arm.get(r.get('arm', '?'), 0) + 1
    mt = os.path.getmtime(OUT)
    return dict(rows=rows, err=err, per_arm=per_arm, mtime=mt, age=time.time() - mt)


def line(prev=None):
    s = read_state()
    pct = 100.0 * s['rows'] / TOTAL if TOTAL else 0.0
    if s['rows'] == 0:
        status = 'NOT-STARTED'
    elif s['rows'] >= TOTAL:
        status = 'DONE'
    elif s['age'] is not None and s['age'] > STALL_S:
        status = 'STALLED'
    else:
        status = 'RUNNING'
    rate = ''
    if prev and s['rows'] > prev['rows'] and prev.get('t'):
        dr = s['rows'] - prev['rows']
        dt = time.time() - prev['t']
        per = dt / dr
        left = (TOTAL - s['rows']) * per / 3600.0
        rate = '  %.1f s/run  ETA %.1f h' % (per, left)
    arms = ' '.join('%s=%d' % (a, s['per_arm'].get(a, 0)) for a in ARMS)
    print('[%s] %-11s %4d/%d (%5.1f%%)  err=%d  idle=%s%s\n    %s'
          % (time.strftime('%H:%M:%S'), status, s['rows'], TOTAL, pct, s['err'],
             ('%.0fs' % s['age']) if s['age'] is not None else '-', rate, arms),
          flush=True)
    s['t'] = time.time()
    return status, s


if __name__ == '__main__':
    if '--loop' in sys.argv:
        every = int(sys.argv[sys.argv.index('--loop') + 1])
        prev = None
        while True:
            status, prev = line(prev)
            if status in ('DONE', 'STALLED'):
                sys.exit(0 if status == 'DONE' else 2)
            time.sleep(every)
    else:
        status, _ = line()
        sys.exit({'DONE': 0, 'STALLED': 2}.get(status, 3))
