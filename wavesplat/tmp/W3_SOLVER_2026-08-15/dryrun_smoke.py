"""Exercise every code path in analyze_w3.py against the 2-episode SMOKE file.

The real analyzer reads `results.jsonl` ONLY.  This harness re-points it at
`smoke_results.jsonl` so the gate/verdict code is proven to run BEFORE the full batch --
S16's lesson (`dryrun_analyzer.py`).  It is NOT a verdict: G-1 will FAIL by construction
(2 episodes, not 48) and the sign tests below are printed only to prove they execute.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_w3 as A

A.RES = os.path.join(HERE, 'smoke_results.jsonl')
A.VERDICT = os.path.join(HERE, 'dryrun_smoke_verdict.json')
# AMENDMENT_2 U-3: this harness must never be able to print a verdict.  It may only
# re-point the analyser at the SMOKE file, and it may NOT touch the frozen thresholds
# (the old `THRESH = {(1,1)}` override made a 1/1 "pass" one edited line away).
assert A.RES.endswith('smoke_results.jsonl'), 'dryrun harness may only read the smoke file'
assert A.THRESH == {'crowded': (32, 23), 'sparse': (16, 13)}, 'thresholds are frozen'
rows, errs, dup = A.load()
ok, fit, dead = A.gates(rows, errs, dup)
print('\n--- smoke sign tests (NOT a verdict; n=1 per stratum) ---')
for sid, t, c in [A.PRIMARY] + A.SECONDARY:
    for st in ('crowded', 'sparse'):
        r = A.sign_test(rows, t, c, st)
        if r:
            print('%-4s %-14s < %-14s [%-7s] %d/%d  A %+.5f  B %+.2f%%'
                  % (sid, t, c, st, r['wins'], r['n'], r['sizeA_median_abs'],
                     r['sizeB_median_rel_pct']))
for st in ('crowded', 'sparse'):
    print(st, 'ZERO condition:', A.zero_condition(rows, st))
