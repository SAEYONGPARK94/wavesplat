#!/usr/bin/env python3
"""C analyser -- written and frozen BEFORE any verdict row exists (W4/W5 discipline).

Implements PREREG.md §1 (PRIMARY), §5 (secondaries), and AMENDMENT_1 §A-2 (the revised
G-2), and nothing else.  Every prohibition of §6 is printed with the result so a reader
cannot pick up a number without the constraint attached.
"""
import collections
import json
import os
import statistics as st
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
THR = 13                     # PREREG §1: crowded n=16, one-sided alpha=0.025
PRIMARY_CELL = 'C_moving_following'
SHELLS = ['obs:0_5', 'obs:5_10', 'obs:10_20', 'obs:20_30', 'obs:30_50',
          'obs:50_75', 'obs:75_100', 'obs:100_150', 'obs:150_inf']


def p_one(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2.0 ** n if n else 1.0


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))
            if l.strip()]
    good = [r for r in rows if 'error' not in r]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in good:
        by[(r['cell'], r['arm'])][r['seed']].append(r)

    def med(cell, arm, seed, key='avg_nRMSE'):
        v = [x.get(key) for x in by[(cell, arm)][seed] if x.get(key) is not None]
        return st.median(v) if v else float('nan')       # median-of-3 estimator seeds

    def seeds(cell):
        return sorted(by[(cell, 'SHIP')])

    def pair(cell, a, b, getter=None):
        g = getter or (lambda c, arm, s: med(c, arm, s))
        w = n = 0
        rel = []
        for s in seeds(cell):
            x, y = g(cell, a, s), g(cell, b, s)
            if x != x or y != y or x == y:
                continue
            n += 1
            w += (x < y)
            rel.append((x - y) / y)
        return w, n, (st.median(rel) if rel else float('nan'))

    print('=' * 92)
    print('C_ENCOUNTER_2026-08-20 -- %d rows (%d usable, %d error)'
          % (len(rows), len(good), len(rows) - len(good)))
    print('rules: PREREG.md + AMENDMENT_1.md.  PRIMARY is one test; nothing replaces it.')
    print('=' * 92)

    print('\n-- gates ' + '-' * 82)
    md5s = {r.get('code_md5') for r in good}
    print('  G-7 single code_md5: %s (%d distinct)'
          % ('PASS' if len(md5s) == 1 else 'FAIL', len(md5s)))
    print('  G-6 zero error rows: %s (%d)'
          % ('PASS' if len(rows) == len(good) else 'FAIL', len(rows) - len(good)))
    zs = [med(c, 'ZERO', s) for c in CELLS_PRESENT(by) for s in seeds(c)]
    zs = [z for z in zs if z == z]
    zok = zs and max(abs(z - 0.25) for z in zs) < 0.03
    print('  G-5 ZERO ~ 0.25: %s (range %.4f-%.4f)'
          % ('PASS' if zok else 'FAIL', min(zs), max(zs)) if zs else '  G-5 no rows')
    gm = os.path.join(HERE, 'gen_manifest.json')
    if os.path.exists(gm):
        man = json.load(open(gm))
        g1 = all(r['gate_G1'] for r in man)
        print('  G-1 sensor moved by U*t on every episode: %s (%d episodes)'
              % ('PASS' if g1 else 'FAIL', len(man)))
        # AMENDMENT_1 A-2: G-2a and G-2c, from the component tables
        mv = [r for r in man if r['U'] > 0]
        stat = [r for r in man if r['U'] == 0]
        same = {(r['seed'], r['direction_deg']): r['we_peak'] for r in stat}
        diff = [abs(r['we_peak'] - same[(r['seed'], r['direction_deg'])])
                for r in mv if (r['seed'], r['direction_deg']) in same]
        print('  G-2a we differs between U>0 and U=0: %s (min |delta| %.4f)'
              % ('PASS' if diff and min(diff) > 1e-6 else 'FAIL',
                 min(diff) if diff else float('nan')))
        print('  G-2c U=0 negative control (we == omega): %s'
              % ('PASS' if stat else 'no static rows'))
        print('  A-3 per-cell episode counts: %s'
              % {c: sum(1 for r in man if r['cell'] == c) for c in CELLS_PRESENT(by)})

    print('\n-- arm medians by cell ' + '-' * 68)
    for c in CELLS_PRESENT(by):
        line = '  %-20s ' % c
        for a in ('ZERO', 'V2K', 'SHIP'):
            v = [med(c, a, s) for s in seeds(c)]
            v = [x for x in v if x == x]
            line += '%s=%.5f  ' % (a, st.median(v) if v else float('nan'))
        print(line)

    w, n, rel = pair(PRIMARY_CELL, 'SHIP', 'ZERO')
    passed = w >= THR
    print('\n' + '=' * 92)
    print('PRIMARY -- SHIP < ZERO at the declared operating point '
          '(U=11.3, following), crowded')
    print('  %d/%d  rel median %+.2f%%  p=%.4e  threshold %d/16  -> %s'
          % (w, n, 100 * rel, p_one(w, n), THR, 'PASS' if passed else 'FAIL'))
    print('=' * 92)

    w2, n2, r2 = pair(PRIMARY_CELL, 'V2K', 'ZERO')
    print('\n-- S-1 / PREREG §1.2 decomposition (declared BEFORE results) ' + '-' * 31)
    print('  V2K < ZERO, same cell: %d/%d  rel %+.2f%%  p=%.4e' % (w2, n2, 100 * r2,
                                                                   p_one(w2, n2)))
    verdict = ('GEOMETRY-SURVIVES' if passed else
               ('OOD-CHECKPOINT' if w2 >= THR else 'GEOMETRY-BREAKS'))
    print('  -> %s' % {
        'GEOMETRY-SURVIVES': 'SHIP passes: the method holds at the operating point',
        'OOD-CHECKPOINT': 'SHIP fails but V2K passes ==> checkpoint OOD, not geometry',
        'GEOMETRY-BREAKS': 'both fail ==> geometry, not the checkpoint'}[verdict])

    print('\n-- S-2 / S-3 (DESCRIPTIVE ONLY -- aperture is confounded across U) ' + '-' * 25)
    for lab, ca, cb in (('S-2 moving vs static, following (SHIP)',
                         'C_moving_following', 'A_static_following'),
                        ('S-3 following vs head at U=11.3 (SHIP)',
                         'C_moving_following', 'D_moving_head')):
        wa = na = 0
        rels = []
        for s in seeds(ca):
            x, y = med(ca, 'SHIP', s), med(cb, 'SHIP', s)
            if x != x or y != y or x == y:
                continue
            na += 1; wa += (x < y); rels.append((x - y) / y)
        print('  %-40s %2d/%-2d  rel %+.2f%%' % (lab, wa, na,
                                                 100 * st.median(rels) if rels
                                                 else float('nan')))

    print('\n-- S-4 shells, PRIMARY cell (obs: prefix, SHIP vs ZERO) ' + '-' * 36)
    for k in SHELLS:
        def sh(c, arm, s, kk=k):
            v = [(x.get('shell_rmse') or {}).get(kk) for x in by[(c, arm)][s]]
            v = [x for x in v if x is not None and x == x]
            return st.median(v) if v else float('nan')
        w3, n3, r3 = pair(PRIMARY_CELL, 'SHIP', 'ZERO', sh)
        print('  %-13s %2d/%-2d  rel %+.2f%%  p=%.3e' % (k, w3, n3, 100 * r3,
                                                         p_one(w3, n3)))

    print('\n' + '-' * 92)
    print('PROHIBITIONS (PREREG §6, printed with the result on purpose):')
    for s in ('이전 배치와 절대값 비교 금지 (격자가 바뀌었다)',
              'seakeeping/제어 주장 금지 (선체 응답 모형 없음)',
              '"조우 예보를 검증했다" 금지 (합성·선형 GT·단일 체크포인트·혼잡 층)',
              '실측 전이 주장 금지 (R1 이 구조적 불가를 확정)',
              '희소 주장 · 전역 nRMSE 헤드라인 금지',
              '중앙값의 비 금지 -- 짝 통계만',
              'AMENDMENT_1: "0.15 사이클/58배" 는 단일 성분 이상화이지 이 코퍼스가 아니다'):
        print('  - ' + s)
    json.dump(dict(primary=[w, n, rel, p_one(w, n), passed], s1=[w2, n2, r2],
                   verdict=verdict),
              open(os.path.join(HERE, 'verdict_c.json'), 'w'), indent=1, default=str)


def CELLS_PRESENT(by):
    return sorted({c for c, _ in by})


if __name__ == '__main__':
    main()
