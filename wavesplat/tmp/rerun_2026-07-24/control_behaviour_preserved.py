"""Control: do the 2026-07-24 code edits preserve shipped behaviour bit-for-bit?

The edits (max_steepness wiring, SCALE_INVARIANT_GROW, SCALE_INVARIANT_L1,
PEAK_SELECT_MODE) are all claimed to be no-ops at their default settings. That claim has to
be measured, not asserted: the whole table re-run was executed on the pre-edit code, so if a
"default-preserving" flag actually moved anything, every number in the re-run would be stale.

Method: replay a sample of already-recorded jobs through the edited code and require the
avg_nRMSE to match the recorded value to the last digit.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)

DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
RUNS = os.path.join(HERE, 'ws_runs.jsonl')

CFGS = {
    'P0': {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
           'adaptive_n': False, 'tracking_window': 1},
    'P1': {'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
           'adaptive_n': True, 'tracking_window': 3},
    'Full': {'use_dispersion': True, 'use_confidence': True, 'init_method': 'v1'},
    'A1_noDisp': {'use_dispersion': False, 'use_confidence': True, 'init_method': 'v1'},
    'A2_noConf': {'use_dispersion': True, 'use_confidence': False, 'init_method': 'v1'},
    'v2+disp': {'use_dispersion': True, 'use_confidence': True, 'init_method': 'v2'},
}


def main():
    from wavesplat.run import run_estimation
    from wavesplat.optimizer import initializer as I
    from wavesplat.optimizer import sequential_update as S

    print(f'flags: PEAK_SELECT_MODE={I.PEAK_SELECT_MODE} MIN_DK={I.MIN_DK} '
          f'SCALE_INVARIANT_GROW={S.SCALE_INVARIANT_GROW} '
          f'SCALE_INVARIANT_L1={S.SCALE_INVARIANT_L1}')

    rows = [json.loads(l) for l in open(RUNS)]
    rows = [r for r in rows if 'error' not in r]
    # one sample per config so every touched code path is exercised
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r['cfg_name'], []).append(r)
    random.seed(0)
    sample = [random.choice(v) for v in by_cfg.values()]

    n_ok = 0
    for r in sample:
        cfg = dict(CFGS[r['cfg_name']])
        cfg['seed'] = r['seed']
        out = run_estimation(os.path.join(DATA, r['episode']), cfg)
        got, want = float(out['avg_nRMSE']), r['avg_nRMSE']
        ok = f'{got:.10f}' == f'{want:.10f}'
        n_ok += ok
        print(f'{"OK  " if ok else "DIFF"} {r["cfg_name"]:<10} {r["episode"]:<42} '
              f'seed={r["seed"]:<3} recorded={want:.10f} replay={got:.10f}')
    print(f'\n{n_ok}/{len(sample)} bit-identical')
    sys.exit(0 if n_ok == len(sample) else 1)


if __name__ == '__main__':
    main()
