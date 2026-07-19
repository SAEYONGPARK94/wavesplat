"""E-B2K -- fable ruling R3 (arena-identity gate) + R5 (mandatory disclosures).

Ruling: tmp/E60K_fable_ruling_failwindow_k_2026-07-19.md (binding, issued
2026-07-19 BEFORE any HOS T/L/N outcome existed).

Kept SEPARATE from e60k_judge.py on purpose: R4 requires the judge to be run
unmodified, so the gate and the disclosures live here and touch nothing the
verdict depends on.

R3  every completed sea's 5 `hand` values must equal the E-15 cache bit-for-bit.
    The failure set is thereby pinned to 11 seas / 18 cells, fixed before the
    experiment. Any mismatch => HALT-PIPELINE (not FAIL, not VOID).
R5  disclosures that must accompany ANY verdict.
"""
import json, os, sys
import numpy as np
from collections import defaultdict

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
E15 = 'results/direction3/e15_hos_multidraw.json'
NEW = 'results/direction3/e60k_hos_arena.json'
DRAWS = 5
SEEDS = (0, 1, 2)


def e15_draws():
    rows = [r for r in json.load(open(E15))['rows'] if r['draw'] < DRAWS]
    by = defaultdict(dict)
    for r in rows:
        by[r['sea']][r['draw']] = r['hand']
    return by


def main():
    ref = e15_draws()
    fail_ref = {s: [d for d, h in v.items() if h < 0.5] for s, v in ref.items()}
    fail_ref = {s: d for s, d in fail_ref.items() if d}
    print(f'PINNED failure set (from E-15 cache, arm-blind, pre-experiment): '
          f'{len(fail_ref)} seas / {sum(len(v) for v in fail_ref.values())} cells')

    if not os.path.exists(NEW):
        print('arena not started')
        return 0
    rows = json.load(open(NEW))['rows']
    seas = sorted({r['sea'] for r in rows})
    print(f'arena progress: {len(seas)}/32 seas, {len(rows)} rows\n')

    # ---- R3 gate ----------------------------------------------------------
    bad = []
    for s in seas:
        got = {r['draw']: r['hand'] for r in rows if r['sea'] == s}
        if len(got) < DRAWS:
            continue
        d = max(abs(got[i] - ref[s][i]) for i in range(DRAWS))
        if d > 0.0:
            bad.append((s, d))
    print(f'[R3 arena-identity gate] seas checked {len(seas)}  '
          f'mismatches {len(bad)}')
    if bad:
        print('  *** HALT-PIPELINE -- arena is not the published E-15 arena:')
        for s, d in bad:
            print(f'     {s}  max|d| {d:.3e}')
        return 2
    print('  PASS -- every completed sea reproduces the E-15 hand values exactly\n')

    # ---- R5 disclosures ---------------------------------------------------
    fail_rows = [r for r in rows if r['hand'] < 0.5]
    by = defaultdict(list)
    for r in fail_rows:
        by[r['sea']].append(r)
    print('[R5 mandatory disclosures]')
    print(f'  failure-window clusters realized : k = {len(by)} '
          f'(of 32 arena seas)')
    print(f'  failure cells                    : n = {len(fail_rows)} '
          f'(of {len(rows)} draws)')
    if by:
        sizes = sorted(((s, len(v)) for s, v in by.items()), key=lambda x: -x[1])
        singles = sum(1 for _, n in sizes if n == 1)
        top2 = sum(n for _, n in sizes[:2])
        print(f'  concentration                    : top-2 seas '
              f'{[s for s, _ in sizes[:2]]} = {top2}/{len(fail_rows)} cells')
        print(f'  single-draw clusters             : {singles}/{len(by)}')
        m = np.array([np.mean([np.mean([r[f'T_s{q}'] - r[f'L_s{q}']
                                        for q in SEEDS]) for r in v])
                      for _, v in sorted(by.items())])
        from scipy import stats
        k = len(m)
        se = m.std(ddof=1) / np.sqrt(k)
        mde = stats.t.ppf(0.95, k - 1) * se
        print(f'  realized MDE (one-sided, alpha .05): {mde:+.4f}  '
              f'(observed mean {m.mean():+.4f})')
        # leave-one-sea-out, REPORT ONLY -- explicitly not a gate (R5)
        keys = sorted(by)
        loo = [float(np.delete(m, i).mean()) for i in range(k)]
        print(f'  leave-one-sea-out mean range     : '
              f'[{min(loo):+.4f}, {max(loo):+.4f}]  (report-only, not a gate)')
        worst = keys[int(np.argmax(m))]
        print(f'  most influential sea (dropping it gives the min): {worst}')
    print('\n  [R5] headline scope sentence, as amended by the ruling:')
    print('    "training-level loss-choice pathology established in the 11 '
          'failure-regime seas\n     of the 32-sea nonlinear HOS arena; '
          'direction-consistent in 5 real site-days."')
    print('  [R5] a FAIL must state whether it is a frozen-trigger FAIL '
          '(underpowered at k=11)\n       or a refutation. They are not the '
          'same claim.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
