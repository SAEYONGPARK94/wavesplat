#!/usr/bin/env python3
"""W5 gate G-0: is the 2026-08-19 package patch behaviour-neutral for the estimator?

W4 ran under git 348d0f2.  Afterwards `b1467bd` changed two things in the package:
  * sequential_update.py:1335  atan2(b,a) -> atan2(-b,a)   [inside amplitude_refit,
    whose gate run.py:398 defaults False and which has zero callers]
  * run.py                     adds 'avg_RMSE_near' to the summary dict

Both are *argued* to be inert for the shipping path.  W5 wants to REUSE W4's L-SHIP /
L-VP / L-ITER0 / L-VP-NW columns instead of re-running them, and that reuse is only
legitimate if the patch is bit-neutral.  So measure it: re-run a sample of W4 cells under
the patched tree and require max|delta| == 0 against the stored values.

    python3 probe_bitidentity.py [n_episodes] [ARM ...]

Exit 0 = bit identical (reuse permitted).  Exit 1 = differs (W5 must re-run every arm).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
W4 = os.path.join(ROOT, 'tmp', 'W4_SOLVER_2026-08-19')
sys.path.insert(0, ROOT)
sys.path.insert(0, W4)
os.chdir(ROOT)

import run_w4 as R                                            # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    # AUDIT_CON: the original probe touched L-SHIP only -- 1 of the 6 reused arms, and one
    # that runs ZERO VARPRO steps.  But P-2's control is the reused `L-VP` column, so the
    # arm that carries the treatment code must be probed too, or the reuse argument covers
    # the wrong column.
    arms = sys.argv[2:] or ['L-SHIP', 'L-VP']
    allrows = [json.loads(l) for l in open(os.path.join(W4, 'results.jsonl'))]
    print('W5 G-0 bit-identity probe: %d cells x %d arms (%s), seed=%d'
          % (n, len(arms), ', '.join(arms), R.SEEDS[0]))
    print('  W4 git_head 348d0f2 -> today %s' % R.HEAD[:8])
    worst, bad, cells = 0.0, 0, 0
    for arm in arms:
        stored = {r['episode']: r for r in allrows
                  if r.get('arm') == arm and r.get('est_seed') == R.SEEDS[0]}
        eps = [e for e in R.episodes() if e['key'] in stored][:n]
        for e in eps:
            row = R.run_one(dict(arm=arm, ep=e, seed=R.SEEDS[0], job_id='w5probe'))
            ref = stored[e['key']]
            d = max(abs(row[k] - ref[k]) for k in ('avg_nRMSE', 'zero_nRMSE'))
            worst = max(worst, d)
            bad += (d != 0.0)
            cells += 1
            print('  %-10s %-38s |d|=%.3e %s'
                  % (arm, e['key'][:36], d, 'OK' if d == 0.0 else 'DIFFERS'))
    ok = (bad == 0)
    print('\n  max|delta| = %.3e over %d cells -> %s'
          % (worst, cells, 'BIT-IDENTICAL: W4 columns may be reused'
             if ok else 'DIFFERS: W5 must re-run every arm'))
    json.dump(dict(bit_identical=ok, arms=arms, cells=cells, max_delta=worst,
                   note='AUDIT_CON fix: L-VP added because it is P-2 control and carries VARPRO'),
              open(os.path.join(HERE, 'g0_bitidentity_arms.json'), 'w'), indent=1)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
