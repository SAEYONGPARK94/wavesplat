#!/usr/bin/env python3
"""AMENDMENT_1 pre-run validation of analyze_s16.py on SYNTHETIC baseline numbers.

    python3 tmp/S16_BASELINE_2026-08-14/dryrun_analyzer.py

*** THIS TOUCHES NO REAL DATA AND PRODUCES NO EVIDENCE. ***  It fabricates the five
baseline columns (the SHIP column is the real import, read-only) so that every branch of
the analyzer -- the AMENDMENT_1 R-2 fixed arm, the 8-4/8-5 priority block, the G-8
uniqueness check, the reduced-n alpha -- is executed once before 7.4 h of runs are spent.
Output goes to tmp/S16_BASELINE_2026-08-14/dryrun/ and is never read by analyze_s16.py.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_s16 as R                                     # noqa: E402
import analyze_s16 as A                                 # noqa: E402

DRY = os.path.join(HERE, 'dryrun')
os.makedirs(DRY, exist_ok=True)


def synth(scenario):
    """scenario: 'pass_s1a_fail' (PRIMARY passes, S-1a fails -> 8-4 vs 8-5 conflict)
                 'r2_fail'       (spectral background no better than white)"""
    rng = np.random.default_rng(7)
    ship = R.import_ship()
    rows = []
    for e in R.episodes():
        s = R.ship_med3(ship[e['key']])
        base = s['avg_nRMSE'] * (2.0 if scenario == 'pass_s1a_fail' else 2.0)
        for arm, mult in (('SPEC25', 1.02), ('SPEC50', 1.00), ('SPEC100', 1.20),
                          ('WHITER50', 1.08 if scenario == 'pass_s1a_fail' else 0.96),
                          ('WHITE50', 1.09)):
            v = base * mult * float(np.exp(rng.normal(0, 0.05)))
            zero = base * 1.35
            sh = {k: v * 1.5 for k in R.SHELL_KEYS}
            zsh = {k: zero * 1.05 for k in R.SHELL_KEYS}
            # far shell: baseline WORSE than the zero field -> S-1a must FAIL
            sh['obs:150_inf'] = zsh['obs:150_inf'] * 1.02
            rows.append(dict(
                episode=e['key'], arm=arm, stratum=e['stratum'], bf=e['bf'],
                ncomp=e['ncomp'], seed=e['seed'], wall_s=42.0, avg_nRMSE=v,
                zero_nRMSE=zero, n_match=160, n_all=301, ms_median=1.0,
                Hs=1.2, Tp=7.0, depth=50.0, shell_rmse=sh, zero_shell=zsh,
                shell_frac={k: 1.0 / 9 for k in R.SHELL_KEYS},
                effective_config={'x': 1}, loc_radius=float(arm[-2:].strip('C')) if 0 else
                {'SPEC25': 25.0, 'SPEC50': 50.0, 'SPEC100': 100.0,
                 'WHITER50': 50.0, 'WHITE50': 50.0}[arm],
                n_ensemble=600, background='jonswap', filter_class='X',
                filter_seed=42, hs_obs4std=1.4, git_head='dry', git_dirty_n=0,
                code_md5='dry', code_files_n=87))
    p = os.path.join(DRY, 'results.jsonl')
    with open(p, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    return p


for sc in ('pass_s1a_fail', 'r2_fail'):
    print('\n' + '#' * 78 + f'\n# SYNTHETIC SCENARIO: {sc}\n' + '#' * 78)
    synth(sc)
    with open(os.path.join(DRY, 'FROZEN_md5.txt'), 'w') as fh:
        fh.write(open(os.path.join(HERE, 'FROZEN_md5.txt')).read())
    A.HERE = DRY
    try:
        A.main()
    except SystemExit as ex:
        print('exit', ex)
print('\n*** dry-run complete -- synthetic, no evidence value ***')
