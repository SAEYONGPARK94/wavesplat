#!/usr/bin/env python3
"""`T6-BUDGET` 차단 게이트 — PREREG §4 를 코드로 옮긴 것.

★규율: 게이트는 **통과 가능**해야 하고 **위조는 실패**해야 한다(`selftest()`).
2026-08-30 `T2-STAT` 의 `G-7b` 가 문자열 검색이라 금지문 자신에 걸려 **구조적으로 항상
실패**했다 --- 어떤 결과로도 라벨이 불가능했다. 그 재발을 막는다.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

N_EXPECTED = 64
SHIPPED = {  # PREREG G-7 --- 배포 상수 되읽기 (⛔손복사 아님: 아래에서 소스와 대조한다)
    'lambda_l1': 1e-4, 'varpro_steps': 400, 'tracking_steps': 10,
    'tracking_window': 3, 'tracking_mode': 'w_phi', 'learned_init_iters': 0,
}
ROUTE_EXPECTED = {'LEARN': 'learned', 'ANA': 'v2_konly'}
BLOCKING = ('G-1', 'G-2', 'G-3', 'G-4', 'G-5', 'G-6', 'G-7', 'G-8', 'G-9', 'G-10')


def gate_id(name):
    return name.split()[0]


def evaluate(rows, calib):
    """[(이름, 통과?, 상세)] 반환. ⛔여기서 라벨을 정하지 않는다."""
    g = []
    ok = lambda c: all(c(r) for r in rows) if rows else False          # noqa: E731

    g.append(('G-1 스레드=1', ok(lambda r: r['threads'] == 1),
              f"{sum(1 for r in rows if r['threads'] == 1)}/{len(rows)}"))
    g.append(('G-2 장치 일치',
              ok(lambda r: r['cuda_available'] == (r['device'] == 'GPU')),
              f"{sum(1 for r in rows if r['cuda_available'] == (r['device'] == 'GPU'))}/{len(rows)}"))
    g.append(('G-3 경로 되읽기',
              ok(lambda r: r['route_effective'] == ROUTE_EXPECTED[r['arm']]),
              f"{sum(1 for r in rows if r['route_effective'] == ROUTE_EXPECTED[r['arm']])}/{len(rows)}"))
    pids = {(r['device'], r['pid']) for r in rows}
    per_dev = {}
    for d, p in pids:
        per_dev.setdefault(d, set()).add(p)
    g.append(('G-4 샤드당 1 PID', all(len(v) == 1 for v in per_dev.values()),
              str({k: len(v) for k, v in per_dev.items()})))
    g.append(('G-5 시계 교정',
              all(0.050 <= c <= 0.055 for c in calib.values()) and len(calib) > 0,
              str({k: round(v, 4) for k, v in calib.items()})))

    pair = {}
    for r in rows:
        pair.setdefault((r['device'], r['episode']), {})[r['arm']] = r['n_frames']
    bad6 = [k for k, v in pair.items() if len(v) == 2 and len(set(v.values())) != 1]
    g.append(('G-6 프레임 수 일치', not bad6 and len(pair) > 0,
              f'{len(pair) - len(bad6)}/{len(pair)} 짝'))

    # ⚠AMENDMENT_0: 여섯 중 넷만 객체 속성이다. **출처도 함께 검사**한다 ---
    #   출처가 조용히 `config` 로 바뀌면 그 키의 증거 등급이 내려간다.
    EFF_SOURCE = {'lambda_l1': 'attr', 'tracking_steps': 'attr',
                  'tracking_window': 'attr', 'tracking_mode': 'attr',
                  'varpro_steps': 'config', 'learned_init_iters': 'config'}

    def shipped_ok(r):
        e = r['effective']
        vals = all(abs(e[k] - v) < 1e-12 if isinstance(v, float) else e[k] == v
                   for k, v in SHIPPED.items())
        return vals and r.get('eff_source') == EFF_SOURCE
    g.append(('G-7 배포 상수', ok(shipped_ok),
              f"{sum(1 for r in rows if shipped_ok(r))}/{len(rows)}"))
    g.append(('G-8 행 수', len(rows) == N_EXPECTED, f'{len(rows)}/{N_EXPECTED}'))
    g.append(('G-9 값 유효',
              ok(lambda r: r['init_s'] > 0 and r['frame_ms'] and
                 all(0 < x < 1e6 for x in r['frame_ms'])),
              f"{sum(1 for r in rows if r['init_s'] > 0 and r['frame_ms'] and all(0 < x < 1e6 for x in r['frame_ms']))}/{len(rows)}"))

    def ratio(r):
        # ⚠p95 를 `int(0.95*(n-1))` 로 잡으면 **작은 n 에서 꼬리를 놓친다**
        #   (자가진단이 n=10 위조에서 잡았다: 9개 5 ms + 1개 500 ms 가 비 1.00).
        #   ⟹ 상위 5% 를 **올림**으로 잡는다.
        import math
        v = sorted(r['frame_ms'])
        i95 = min(len(v) - 1, max(0, math.ceil(0.95 * len(v)) - 1))
        return v[i95] / v[len(v) // 2]
    g.append(('G-10 부하 상한', ok(lambda r: ratio(r) <= 3.0),
              'max p95/med = %.2f' % (max(ratio(r) for r in rows) if rows else -1)))
    return g


def blocked_by(gates):
    """⛔**짧은 id 를 돌려준다.**

    ★2026-09-01 `T5-ENS` 에서 똑같은 버그를 냈다: 짧은 id 로 거르면서 **전체 이름**을
    돌려줘서, 호출자가 `'G-1' in blocked_by(...)` 로 물으면 항상 False 였다
    --- 차단 게이트 전부가 **장식**이 된다. 자가진단이 오늘 그것을 다시 잡았다.
    """
    return [gate_id(n) for n, p, _ in gates if not p and gate_id(n) in BLOCKING]


def label(rows):
    """PREREG §5 --- 가장 보수적인 칸(혼잡 · CPU · LEARN)의 프레임 중앙."""
    sel = [r for r in rows if r['arm'] == 'LEARN' and r['device'] == 'CPU'
           and r['stratum'] == 'crowded']
    if not sel:
        return 'NO-CELL', None
    med = sorted(sorted(r['frame_ms'])[len(r['frame_ms']) // 2] for r in sel)
    m = med[len(med) // 2]
    for thr, lab in ((10.0, 'FRAME-BUDGET-AMPLE'), (50.0, 'FRAME-BUDGET-FITS'),
                     (100.0, 'FRAME-BUDGET-TIGHT')):
        if m <= thr:
            return lab, m
    return 'FRAME-BUDGET-EXCEEDED', m


def selftest():
    """★항진 검사 --- 위조는 반드시 실패해야 하고, 정상은 통과해야 한다."""
    base = dict(threads=1, cuda_available=False, device='CPU', arm='LEARN',
                route_effective='learned', pid=1, episode='e1', stratum='crowded',
                n_frames=10, init_s=1.0, frame_ms=[5.0] * 10,
                effective=dict(SHIPPED),
                eff_source={'lambda_l1': 'attr', 'tracking_steps': 'attr',
                            'tracking_window': 'attr', 'tracking_mode': 'attr',
                            'varpro_steps': 'config', 'learned_init_iters': 'config'})
    good = []
    for i in range(N_EXPECTED):
        r = dict(base)
        r['episode'] = f'e{i // 4}'
        r['arm'] = 'LEARN' if i % 2 == 0 else 'ANA'
        r['route_effective'] = ROUTE_EXPECTED[r['arm']]
        r['device'] = 'CPU' if i % 4 < 2 else 'GPU'
        r['cuda_available'] = (r['device'] == 'GPU')
        r['pid'] = 1 if r['device'] == 'CPU' else 2
        r['effective'] = dict(SHIPPED)
        good.append(r)
    calib = {'CPU': 0.0505, 'GPU': 0.0508}
    bad = []
    if blocked_by(evaluate(good, calib)):
        bad.append(('정상', '통과 가능해야 하는데 막혔다: '
                    + str(blocked_by(evaluate(good, calib)))))

    # ⚠[AMENDMENT_0] 위조 이름과 **발화해야 할 게이트 이름**은 다를 수 있다
    #   (`eff_source` 위조는 `G-7` 이 잡아야 한다). 자가진단이 그 혼동을 잡았다.
    #   ⟹ 키를 `위조이름 -> (기대 게이트, 함수)` 로 바꾼다.
    FORGERIES = {
        'G-1': ('G-1', lambda rs: [dict(r, threads=8) for r in rs]),
        'G-2': ('G-2', lambda rs: [dict(r, cuda_available=True) for r in rs]),
        'G-3': ('G-3', lambda rs: [dict(r, route_effective='bank_mmse') for r in rs]),
        'G-4': ('G-4', lambda rs: [dict(r, pid=i) for i, r in enumerate(rs)]),
        'G-6': ('G-6', lambda rs: [dict(r, n_frames=(9 if r['arm'] == 'ANA' else 10)) for r in rs]),
        'G-7': ('G-7', lambda rs: [dict(r, effective=dict(SHIPPED, varpro_steps=0)) for r in rs]),
        'G-7b': ('G-7', lambda rs: [dict(r, eff_source=dict(r['eff_source'],
                                    varpro_steps='intent')) for r in rs]),
        'G-8': ('G-8', lambda rs: rs[:-1]),
        'G-9': ('G-9', lambda rs: [dict(r, init_s=0.0) for r in rs]),
        'G-10': ('G-10', lambda rs: [dict(r, frame_ms=[5.0] * 9 + [500.0]) for r in rs]),
    }
    for fname, (expect, forge) in FORGERIES.items():
        if expect not in blocked_by(evaluate(forge([dict(r) for r in good]), calib)):
            bad.append((fname, f'위조가 통과했다 (기대 게이트 {expect})'))
    if 'G-5' not in blocked_by(evaluate(good, {'CPU': 0.20})):
        bad.append(('G-5', '위조가 통과했다'))
    # 라벨 공역
    for ms, want in ((5.0, 'FRAME-BUDGET-AMPLE'), (30.0, 'FRAME-BUDGET-FITS'),
                     (80.0, 'FRAME-BUDGET-TIGHT'), (500.0, 'FRAME-BUDGET-EXCEEDED')):
        got, _ = label([dict(r, frame_ms=[ms] * 10) for r in good])
        if got != want:
            bad.append(('label', f'{ms} ms -> {got}, 기대 {want}'))
    # 모든 차단 게이트가 위조 표에 있는가 (사각지대 검사)
    uncovered = set(BLOCKING) - {e for e, _ in FORGERIES.values()} - {'G-5'}
    if uncovered:
        bad.append(('coverage', f'위조 시험 없는 차단 게이트: {sorted(uncovered)}'))
    for n, why in bad:
        print(f'  [BAD ] {n}: {why}')
    if bad:
        print(f'⛔자가진단 실패 {len(bad)}건')
        return 1
    print(f'✅자가진단 통과 — 정상 통과 · 위조 {len(FORGERIES) + 1}종 전건 차단 · '
          '라벨 공역 4/4 · 차단 게이트 전건 위조 시험 보유')
    return 0


if __name__ == '__main__':
    sys.exit(selftest() if '--selftest' in sys.argv else 0)
