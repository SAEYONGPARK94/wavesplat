#!/usr/bin/env python3
"""`T8` 자가진단 — **동결 전에** 돌린다(PREREG §6.3). ⛔런 0회(가짜 행으로만).

검사 넷:
  A **위조 탐침** — 게이트마다 그 게이트만 깨는 행을 만들어 **반드시 FAIL** 하는가
  B **`blocked_by` 가 id 를 돌려주는가** — ⛔`T5`·`T7` 에서 두 번 난 결함:
    이름 전체를 `BLOCKING` 과 비교해 **차단 목록이 항상 비었다**
  C **라벨 공역 전건** — 결정표의 네 칸이 전부 도달 가능한가
  D **문턱 재도출** — 사전등록의 수와 코드의 수가 일치하는가
"""
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_t8 as G  # noqa: E402

OK = [True]


def chk(name, cond, detail=''):
    print(('  ✅' if cond else '  ⛔') + f' {name}' + (f'   {detail}' if detail else ''))
    OK[0] &= bool(cond)


def fixture():
    """게이트를 전부 통과하는 가짜 행 384개 + 앵커 4개."""
    rows = []
    for e in G.EPISODES:
        for s in G.EST_SEEDS:
            for arm in G.ARM_ORDER:
                rows.append(dict(batch=G.BATCH, arm=arm, episode=e, seed=s,
                                 avg_nRMSE=0.05 if arm == 'M1' else 0.049,
                                 auto_selected='learned',
                                 ensemble_k=G.ARMS[arm]['K'],
                                 init_time_s=6.0 if arm == 'M1' else 16.0,
                                 deploy_pin_ok=True, threads=1, pid=1, wall_s=1.0))
    anc = [dict(arm='M1', episode=e, avg_nRMSE=v)
           for e, v in G._anchor_sens().items() if e in G.ANCHOR_EPISODES]
    return rows, anc


def main():
    rows, anc = fixture()
    print('── D 문턱 재도출 ' + '─' * 42)
    t = G.thresholds()
    chk('문턱 K_MIN == 56 (n=64, p0=0.75, α=0.025)', t['K_MIN'] == 56, str(t['K_MIN']))
    chk('검정력(p=0.90) >= 0.80', t['power_at_090'] >= 0.80, str(t['power_at_090']))
    chk('MDE(0.8) < 0.90', t['mde_p80'] < 0.90, str(t['mde_p80']))
    chk('비용 문턱이 동결 T6 에서 온다', abs(G.cost_threshold_s() - 17.08) < 0.01,
        f'{G.cost_threshold_s():.3f}s')
    chk('기대 행 == 384', G.expected_rows() == 384, str(G.expected_rows()))
    chk('앵커가 PRIMARY 코퍼스와 서로소',
        not (set(G.ANCHOR_EPISODES) & set(G.EPISODES)))

    print('\n── 픽스처 전건 통과 ' + '─' * 39)
    g0 = G.gates(rows, anchor_rows=anc)
    bad0 = [n for n, o, _ in g0 if not o]
    chk('픽스처에서 게이트 전건 통과', not bad0, str(bad0))
    chk('blocked_by == []', G.blocked_by(g0) == [], str(G.blocked_by(g0)))

    print('\n── A 위조 탐침 ' + '─' * 44)
    FORGE = {
        'G-1': lambda r: [dict(x, threads=4) for x in r],
        'G-2': lambda r: [dict(x, ensemble_k=99) if x['arm'] == 'M3' else x for x in r],
        'G-3': lambda r: [dict(x, auto_selected='v2_konly') for x in r],
        'G-4': lambda r: [dict(x, deploy_pin_ok=False) for x in r],
        'G-5': lambda r: r[:-3],
        'G-8': lambda r: [dict(x, avg_nRMSE=float('nan')) for x in r],
    }
    for gid, f in FORGE.items():
        gg = G.gates(f(copy.deepcopy(rows)), anchor_rows=anc)
        chk(f'{gid} 위조가 잡힌다', gid in G.blocked_by(gg), str(G.blocked_by(gg)))
    # G-6 : 앵커가 PRIMARY 표본에 섞이면
    leak = copy.deepcopy(rows)
    leak.append(dict(leak[0], episode=G.ANCHOR_EPISODES[0]))
    chk('G-6 앵커 누출이 잡힌다', 'G-6' in G.blocked_by(G.gates(leak, anchor_rows=anc)))
    # G-ANCHOR : 앵커 값이 어긋나면
    bad_anc = [dict(a, avg_nRMSE=(a['avg_nRMSE'] or 0) + 1e-9) for a in anc]
    chk('G-ANCHOR 표류가 잡힌다',
        'G-ANCHOR' in G.blocked_by(G.gates(rows, anchor_rows=bad_anc)))
    chk('G-ANCHOR 앵커 0행도 잡힌다',
        'G-ANCHOR' in G.blocked_by(G.gates(rows, anchor_rows=[])))

    print('\n── B `blocked_by` 가 id 를 돌려준다 ' + '─' * 24)
    chk('gate_id 가 접두 id 를 뽑는다', G.gate_id('G-ANCHOR M1 == 동결') == 'G-ANCHOR')
    chk('BLOCKING 의 모든 id 가 게이트 이름에 있다',
        set(G.BLOCKING) <= {G.gate_id(n) for n, _, _ in g0},
        str(sorted(set(G.BLOCKING) - {G.gate_id(n) for n, _, _ in g0})))

    print('\n── C 라벨 공역 ' + '─' * 44)
    seen = set()
    for K in (64, 60, 56, 40, 10):
        for cpass in (True, False):
            ni = dict(n=64, K=K, rel_median_pct=0.0, worst_pct=0.0, better=32,
                      p_one_sided=0.0, delta_pct=G.DELTA_PCT)
            lab, gr = G.label_for(ni, dict(pass_=cpass, **{'pass': cpass}))
            seen.add((lab or '').split(':: ')[-1])
    want = {'ENS-DEFAULT-READY', 'ENS-DEFAULT-COSTLY', 'ENS-DEFAULT-HARMS-SPARSE',
            'ENS-DEFAULT-NO', 'ENS-DEFAULT-UNDECIDED'}
    chk('결정표 다섯 칸이 전부 도달 가능', want <= seen, str(sorted(want - seen)))
    chk('NOROWS 가 라벨을 내지 않는다',
        G.label_for(dict(n=0, K=None), {'pass': True})[0] is None)

    print('\n' + ('✅자가진단 통과' if OK[0] else '⛔자가진단 실패'))
    return 0 if OK[0] else 1


if __name__ == '__main__':
    sys.exit(main())
