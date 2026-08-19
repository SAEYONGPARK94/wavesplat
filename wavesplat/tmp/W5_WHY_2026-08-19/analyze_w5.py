#!/usr/bin/env python3
"""W5 analyser -- implements PREREG.md + AMENDMENT_1 + AMENDMENT_2 exactly.

No rule is chosen here.  Everything was frozen and committed before the run (1d355cd);
AMENDMENT_2's label refinement was written while the batch was still running but WITHOUT
any analyser having read a W5 row, and it only removes claims (see that file).
"""
import collections
import json
import os
import statistics as st
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, os.path.join(ROOT, 'tmp', 'W4_SOLVER_2026-08-19'), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import run_w5 as R                                            # noqa: E402
import arms_config_w5 as A5                                   # noqa: E402

THR1, THR2 = 41, 42          # one-sided @.025 / two-sided @.025 (PREREG section 1)
SHELLS = ['obs:0_5', 'obs:5_10', 'obs:10_20', 'obs:20_30', 'obs:30_50',
          'obs:50_75', 'obs:75_100', 'obs:100_150', 'obs:150_inf']


def p_one(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2.0 ** n


def p_two(w, n):
    return min(1.0, 2.0 * min(p_one(w, n), p_one(n - w, n)))


def main():
    rows = [r for r in R.all_rows() if 'error' not in r]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r['arm']][r['episode']].append(r)
    eps = sorted(by['L-SHIP'])

    def med(arm, e, key='avg_nRMSE'):
        v = [x.get(key) for x in by[arm][e] if x.get(key) is not None]
        return st.median(v) if v else float('nan')

    def shell(arm, e, k):
        v = [(x.get('shell_rmse') or {}).get(k) for x in by[arm][e]]
        v = [x for x in v if x is not None and x == x]
        return st.median(v) if v else float('nan')

    def pair(a, b, getter=med):
        w = n = 0
        rel = []
        for e in eps:
            x, y = getter(a, e), getter(b, e)
            if x != x or y != y or x == y:
                continue
            n += 1
            w += (x < y)
            rel.append((x - y) / y)
        return w, n, (st.median(rel) if rel else float('nan'))

    print('=' * 96)
    print('W5_WHY_2026-08-19  --  %d rows (%d new, %d reused), %d episodes'
          % (len(rows), sum(r.get('source') != 'W4-reuse' for r in rows),
             sum(r.get('source') == 'W4-reuse' for r in rows), len(eps)))
    print('rules: PREREG + AMENDMENT_1 + AMENDMENT_2 | Holm-2 | med3 episode-level')
    print('=' * 96)
    print('\n-- arm medians (crowded, median-of-3) ' + '-' * 55)
    order = ['ZERO', 'L-ITER0', 'L-SHIP', 'L-SHIP-DUP2', 'L-STEP400', 'L-VP800',
             'L-VP', 'L-VP-ONLY', 'L-VP-NW', 'L-VP-ONLY-NW', 'ORA-K-CVX']
    for a in [x for x in order if x in by]:
        vals = [med(a, e) for e in eps if med(a, e) == med(a, e)]
        print('  %-13s n=%-3d median=%.6f   adam_p1=%-5s vp=%-4s'
              % (a, len(vals), st.median(vals), A5.expected_adam_p1(a), A5.expected_vp_steps(a)))

    w_iut, n_iut, rel_iut = pair('L-SHIP', 'ZERO')
    w, n, rel = w_iut, n_iut, rel_iut
    iut = w_iut >= THR1
    print('\n-- IUT (against us): L-SHIP < ZERO ' + '-' * 58)
    print('  %d/%d  rel median %+.2f%%  p=%.3e  -> %s' % (w, n, 100 * rel, p_one(w, n),
                                                          'PASS' if iut else 'FAIL'))

    w1, n1, rel1 = pair('L-VP-ONLY', 'L-SHIP')
    w2, n2, rel2 = pair('L-VP-ONLY', 'L-VP')
    pA, pB = p_one(w1, n1), p_two(w2, n2)
    print('\n-- PRIMARY family, Holm-2 (alpha=0.05) ' + '-' * 54)
    print('  P-1 (replacement, one-sided) L-VP-ONLY < L-SHIP : %d/%d  rel %+.2f%%  p=%.3e'
          % (w1, n1, 100 * rel1, pA))
    print('  P-2 (necessity, two-sided)   L-VP-ONLY vs L-VP  : %d/%d  rel %+.2f%%  p=%.3e'
          % (w2, n2, 100 * rel2, pB))
    order2 = sorted([('P-1', pA), ('P-2', pB)], key=lambda t: t[1])
    rej, alive = {}, True
    for i, (name, p) in enumerate(order2):
        a = 0.025 if i == 0 else 0.05
        rej[name] = alive and (p <= a)
        print('    Holm rank %d: %s p=%.3e vs alpha=%.4f -> %s'
              % (i + 1, name, p, a, 'REJECT' if rej[name] else 'retain'))
        alive = alive and rej[name]

    # AMENDMENT_2 A2-1: a label only when Holm actually TESTED P-2
    if not rej.get('P-2') and not rej.get('P-1') and order2[0][0] == 'P-1':
        label = 'JOINTFIT-UNTESTED (Holm)'
    elif rej['P-2']:
        label = 'JOINTFIT-HARMS' if w2 >= THR2 else 'JOINTFIT-ADDS'
    elif 24 <= w2 <= 40:
        label = 'JOINTFIT-REDUNDANT'
    else:
        label = 'JOINTFIT-UNTESTED (Holm)'
    print('\n  P-2 label : %s' % label)
    print('    A2-1 rule: REDUNDANT only when the RAW two-sided count is inside the '
          'alpha=0.05 acceptance region 24-40 (here %d); an untested P-2 never promotes.' % w2)

    verdict = ('FAIL (IUT)' if not iut else
               ('REPLACEMENT-CONFIRMED / ' + label) if rej['P-1'] else
               ('NO-REPLACEMENT / ' + label))
    print('\n' + '=' * 96)
    print('VERDICT: %s' % verdict)
    print('=' * 96)

    print('\n-- SECONDARY ' + '-' * 79)
    for lab, a, b, two in (('S-1  L-VP-ONLY < L-ITER0   ', 'L-VP-ONLY', 'L-ITER0', False),
                           ('S-2  L-VP800   vs L-VP     ', 'L-VP800', 'L-VP', True),
                           ('S-3  L-VP-ONLY-NW vs L-VP-ONLY', 'L-VP-ONLY-NW', 'L-VP-ONLY', True)):
        w, n, rel = pair(a, b)
        p = p_two(w, n) if two else p_one(w, n)
        ok = (w >= THR2 or w <= n - THR2) if two else (w >= THR1)
        print('  %s %d/%d  rel %+.2f%%  p=%.3e  -> %s'
              % (lab, w, n, 100 * rel, p, ('DIFFER' if two else 'PASS') if ok else 'null'))

    print('\n-- S-4 shells: L-VP-ONLY < L-SHIP (obs: prefix) ' + '-' * 45)
    for k in SHELLS:
        w, n, rel = pair('L-VP-ONLY', 'L-SHIP', lambda a, e, kk=k: shell(a, e, kk))
        print('  %-13s %3d/%-3d  rel %+.2f%%  p=%.3e  %s'
              % (k, w, n, 100 * rel, p_one(w, n), 'PASS' if w >= THR1 else 'ns'))

    print('\n-- D descriptives (not hypotheses) ' + '-' * 58)
    for a in [x for x in order if x in by]:
        ws = [r['wall_s'] for r in rows if r['arm'] == a and r.get('wall_s')]
        if ws:
            print('  %-13s wall median %6.1f s   n=%d' % (a, st.median(ws), len(ws)))
    # AMENDMENT_1 A-3 requires this: `learned_init_iters=0` does not only remove the joint
    # fit, it also changes how often the growth branch fires.  The field is
    # `adam_steps_grow` (the growth branch's own 50-step fine-tune), not `add_calls`.
    print('  growth firing (adam_steps_grow>0) -- AMENDMENT_1 A-3 co-factor:')
    for a in ('L-SHIP', 'L-VP', 'L-VP800', 'L-VP-ONLY', 'L-VP-ONLY-NW', 'L-ITER0'):
        rs = [r for r in rows if r['arm'] == a]
        if rs:
            print('    %-13s %3d/%d' % (a, sum((r.get('adam_steps_grow') or 0) > 0
                                               for r in rs), len(rs)))

    print('\n' + '=' * 96)
    print('PROHIBITIONS (PREREG section 5 + AMENDMENT_1) -- always printed under a verdict')
    print('=' * 96)
    for s in ('novelty 금지 (VARPRO = Golub-Pereyra 1973; arXiv:2511.01234 이 기제 정리 선점)',
              '"벽을 닫았다" 금지 (상한 대비 여전히 x90 이상, 혼잡은 초해상 영역)',
              'ORA-K-CVX 비를 솔버 이득으로 인용 금지 (6요인 묶음)',
              '전역 nRMSE 헤드라인 금지 / 희소 주장 금지',
              'A-3: L-VP-ONLY 는 묶음이다 -- "결합 적합만 뺐다" 로 쓰지 마라 '
              '(clamp 투영 소멸 + 성장 발화 변화 + VARPRO 입력 변화가 함께 온다)',
              'A-4: avg_RMSE_near 는 이 배치에도 없다 (D-4 철회). 부재는 음성 증거가 아니다',
              'A-5: S-2 널은 "400 포화" 와 "800 도 부족" 을 구분하지 못한다',
              'A2-1: JOINTFIT-UNTESTED 에서는 배포 승격을 발동하지 않는다',
              'A-6: AUDIT_PRO/구현 감사가 세션 한도로 빠졌다 -- 감사 범위가 좁다'):
        print('  - ' + s)
    json.dump(dict(verdict=verdict, label=label, P1=[w1, n1, rel1, pA], P2=[w2, n2, rel2, pB],
                   iut=[w_iut, n_iut]), open(os.path.join(HERE, 'verdict_w5.json'), 'w'),
              indent=1, default=str)
    print('\n-> verdict_w5.json')


if __name__ == '__main__':
    main()
