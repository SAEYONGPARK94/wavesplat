"""GPR baseline on the Nyquist-adequate regeneration (2026-07-24 table re-run).

CPU-only, so it runs alongside the GPU WaveSplat batch. Note the GT grid grew from
40x30 = 1,200 points to 200x167 = 33,400 points, so per-frame prediction cost rises ~28x;
that is a property of the evaluation grid, not of the baseline, and must be reported as such
(the paper's ms/frame numbers for GPR are not comparable across the two datasets).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'gpr_runs.jsonl')

PAIRS = [('BF2', '1comp'), ('BF3', '1comp'), ('BF3', '2comp'), ('BF3', '5comp'),
         ('BF4', '1comp'), ('BF4', '5comp'), ('BF4', '15comp'), ('BF4', '30comp'),
         ('BF5', '15comp'), ('BF5', '30comp')]
EPISODES = [f'episode_{bf}_{c}_seed0_uniform.h5' for bf, c in PAIRS]


def main():
    from run_gpr_comparison import run_gpr

    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['episode'])
                except (json.JSONDecodeError, KeyError):
                    pass

    with open(OUT, 'a') as fh:
        for ep in EPISODES:
            if ep in done:
                continue
            t0 = time.time()
            try:
                r = run_gpr(os.path.join(DATA, ep))
                rec = {'episode': ep, **{k: float(v) for k, v in r.items()
                                         if isinstance(v, (int, float, np.floating))},
                       'wall_s': time.time() - t0}
            except Exception as e:
                rec = {'episode': ep, 'error': repr(e), 'wall_s': time.time() - t0}
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            print(f'{ep:<45s} {rec.get("avg_nRMSE", rec.get("error"))} '
                  f'({rec["wall_s"]:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
