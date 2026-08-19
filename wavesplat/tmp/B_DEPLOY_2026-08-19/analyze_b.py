#!/usr/bin/env python3
"""B analyser -- PREREG.md exactly.  ITT is the headline; PP is reported but never leads."""
import collections
import json
import os
import statistics as st
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
THR = 41
SHELLS = ['obs:0_5', 'obs:5_10', 'obs:10_20', 'obs:20_30', 'obs:30_50',
          'obs:50_75', 'obs:75_100', 'obs:100_150', 'obs:150_inf']


def p_one(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2.0 ** n if n else 1.0


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl')) if l.strip()]
    good = [r for r in rows if 'error' not in r]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    strat, route = {}, collections.defaultdict(dict)
    for r in good:
        by[r['arm']][r['episode']].append(r)
        strat[r['episode']] = r.get('stratum')
        if r.get('route'):
            route[r['arm']].setdefault(r['episode'], []).append(r['route'])

    def med(a, e, k='avg_nRMSE'):
        v = [x.get(k) for x in by[a][e] if x.get(k) is not None]
        return st.median(v) if v else float('nan')

    def eps(s):
        return sorted(e for e in by['A-SHIP'] if strat.get(e) == s)

    def pair(a, b, E, getter=None):
        g = getter or (lambda arm, e: med(arm, e))
        w = n = 0
        rel = []
        for e in E:
            x, y = g(a, e), g(b, e)
            if x != x or y != y or x == y:
                continue
            n += 1
            w += (x < y)
            rel.append((x - y) / y)
        return w, n, (st.median(rel) if rel else float('nan'))

    print('=' * 92)
    print('B_DEPLOY_2026-08-19 -- %d rows (%d usable, %d error)'
          % (len(rows), len(good), len(rows) - len(good)))
    print('rules: PREREG.md | ITT headline, PP secondary and never leading')
    print('=' * 92)

    print('\n-- routing (the reason this is an ITT design) ' + '-' * 46)
    for s in ('crowded', 'sparse'):
        E = eps(s)
        for a in ('A-SHIP', 'A-VP'):
            c = collections.Counter(
                st.mode(route[a][e]) if e in route[a] else '?' for e in E)
            print('  %-8s %-7s %s' % (s, a, dict(c)))
        agree = sum(1 for e in E
                    if e in route['A-SHIP'] and e in route['A-VP']
                    and st.mode(route['A-SHIP'][e]) == st.mode(route['A-VP'][e]))
        print('  %-8s S-3 routing agreement: %d/%d' % (s, agree, len(E)))

    print('\n-- arm medians ' + '-' * 76)
    for s in ('crowded', 'sparse'):
        E = eps(s)
        line = '  %-8s ' % s
        for a in ('ZERO', 'A-SHIP', 'A-VP'):
            v = [med(a, e) for e in E if med(a, e) == med(a, e)]
            line += '%s=%.6f  ' % (a, st.median(v) if v else float('nan'))
        print(line)

    E = eps('crowded')
    w, n, rel = pair('A-SHIP', 'ZERO', E)
    iut = w >= THR
    print('\n-- IUT (against us): A-SHIP < ZERO, crowded ' + '-' * 48)
    print('  %d/%d  rel %+.2f%%  p=%.3e  -> %s' % (w, n, 100 * rel, p_one(w, n),
                                                   'PASS' if iut else 'FAIL'))

    w1, n1, r1 = pair('A-VP', 'A-SHIP', E)
    print('\n-- PRIMARY (ITT): A-VP < A-SHIP, crowded, one-sided >=41/64 ' + '-' * 32)
    print('  %d/%d  rel median %+.2f%%  p=%.3e  -> %s'
          % (w1, n1, 100 * r1, p_one(w1, n1), 'REJECT' if w1 >= THR else 'retain'))
    print('    n=%d (ties dropped: %d)' % (n1, len(E) - n1))

    pp = [e for e in E if e in route['A-SHIP'] and e in route['A-VP']
          and st.mode(route['A-SHIP'][e]) == 'learned' == st.mode(route['A-VP'][e])]
    w2, n2, r2 = pair('A-VP', 'A-SHIP', pp)
    print('\n  [PP, NEVER the headline -- PREREG 1.1] both arms routed learned: '
          '%d/%d  rel %+.2f%%  p=%.3e' % (w2, n2, 100 * r2, p_one(w2, n2)))

    Es = eps('sparse')
    w3, n3, r3 = pair('A-VP', 'A-SHIP', Es)
    print('\n-- S-1 sparse (ITT) ' + '-' * 71)
    print('  %d/%d  rel %+.2f%%  p=%.3e  -> %s'
          % (w3, n3, 100 * r3, p_one(w3, n3), 'PASS' if w3 >= THR else 'null'))

    print('\n-- S-2 shells, crowded (obs: prefix) ' + '-' * 55)
    for k in SHELLS:
        def sh(a, e, kk=k):
            v = [(x.get('shell_rmse') or {}).get(kk) for x in by[a][e]]
            v = [x for x in v if x is not None and x == x]
            return st.median(v) if v else float('nan')
        w, n, rl = pair('A-VP', 'A-SHIP', E, sh)
        print('  %-13s %3d/%-3d  rel %+.2f%%  p=%.3e  %s'
              % (k, w, n, 100 * rl, p_one(w, n), 'PASS' if w >= THR else 'ns'))

    print('\n-- D descriptives ' + '-' * 73)
    for a in ('A-SHIP', 'A-VP'):
        for k, lab in (('wall_s', 'wall'), ('init_time_s', 'init'),
                       ('elapsed_ms_mean', 'frame_ms')):
            v = [r.get(k) for r in good if r['arm'] == a and r.get(k) is not None]
            if v:
                print('  %-7s %-9s median %8.2f  n=%d' % (a, lab, st.median(v), len(v)))
    iv = [r for r in good if r['arm'] == 'A-VP' and r.get('route')]
    print('  A-VP intervention: vp=400 on %d/%d rows (learned rows: %d)'
          % (sum(1 for r in iv if r.get('vp_steps') == 400), len(iv),
             sum(1 for r in iv if r.get('route') == 'learned')))

    verdict = ('FAIL (IUT)' if not iut else
               'DEPLOY-CONFIRMED' if w1 >= THR else 'NO-DEPLOY-EFFECT')
    print('\n' + '=' * 92)
    print('VERDICT: %s' % verdict)
    print('=' * 92)
    for s in ('novelty 금지(W6 수학 종결) · 실측 주장 금지(R1 구조적 불가) · 희소 우위 금지'
              '(표현 아티팩트) · PP 헤드라인 금지 · W5 수치 인용 금지 · 중앙값의 비 금지',):
        print('  PROHIBITIONS: ' + s)
    json.dump(dict(verdict=verdict, itt=[w1, n1, r1], pp=[w2, n2, r2],
                   sparse=[w3, n3, r3], iut=[w, n]),
              open(os.path.join(HERE, 'verdict_b.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
