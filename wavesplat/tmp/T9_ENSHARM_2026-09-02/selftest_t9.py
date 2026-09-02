#!/usr/bin/env python3
"""`T9` 자가진단 — **동결 전에** 돌린다(PREREG §6.3). ⛔런 0회(가짜 행으로만).

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
T8 = os.path.abspath(os.path.join(HERE, '..', 'T8_ENSDEFAULT_2026-09-02'))
for _p in (HERE, T8, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_t8 as T8G  # noqa: E402
import gates_t9 as G  # noqa: E402

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
                                 avg_nRMSE=0.05 if arm == 'M1' else 0.0505,
                                 auto_selected='learned',
                                 ensemble_k=G.ARMS[arm]['K'],
                                 init_time_s=6.0 if arm == 'M1' else 16.0,
                                 deploy_pin_ok=True, threads=1, pid=1, wall_s=1.0))
    anc = [dict(arm='M1', episode=e, avg_nRMSE=v)
           for e, v in T8G._anchor_sens().items() if e in G.ANCHOR_EPISODES]
    tim = [dict(arm=a, episode=G.EPISODES[i], seed=42, threads=1,
                route_effective='learned', frame_ms=[6.0, 6.1, 6.2])
           for i in range(4) for a in G.ARM_ORDER]
    return rows, anc, tim


def main():
    rows, anc, tim = fixture()
    print('── D 문턱 재도출 ' + '─' * 42)
    t = G.thresholds()
    chk('문턱 K_MIN == 56 (n=64, p0=0.75, α=0.025)', t['K_MIN'] == 56, str(t['K_MIN']))
    chk('여유가 **격자**다(자유 매개변수 아님)',
        t['delta_grid'] == [0.001, 0.002, 0.005, 0.01], str(t['delta_grid']))
    chk('센서 눈금은 판정과 분리돼 있다', t['delta_sensor'] == 0.005)
    chk('★예측(p=0.906) 아래 검정력 >= 0.8', t['power_at_0906'] >= 0.80,
        str(t['power_at_0906']))
    chk('T8 코퍼스와 서로소', not (set(G.EPISODES) & set(T8G.EPISODES)))
    chk('검정력(p=0.90) >= 0.80', t['power_at_090'] >= 0.80, str(t['power_at_090']))
    chk('MDE(0.8) < 0.90', t['mde_p80'] < 0.90, str(t['mde_p80']))
    chk('기대 행 == 384', G.expected_rows() == 384, str(G.expected_rows()))
    chk('앵커가 PRIMARY 코퍼스와 서로소',
        not (set(G.ANCHOR_EPISODES) & set(G.EPISODES)))

    print('\n── 픽스처 전건 통과 ' + '─' * 39)
    g0 = G.gates(rows, anchor_rows=anc, timing_rows=tim)
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
        gg = G.gates(f(copy.deepcopy(rows)), anchor_rows=anc, timing_rows=tim)
        chk(f'{gid} 위조가 잡힌다', gid in G.blocked_by(gg), str(G.blocked_by(gg)))
    # G-6 : 앵커가 PRIMARY 표본에 섞이면
    leak = copy.deepcopy(rows)
    leak.append(dict(leak[0], episode=G.ANCHOR_EPISODES[0]))
    chk('G-6 앵커 누출이 잡힌다', 'G-6' in G.blocked_by(G.gates(leak, anchor_rows=anc, timing_rows=tim)))
    # ★G-9 : T8 에피소드가 섞이면
    mix = copy.deepcopy(rows)
    mix.append(dict(mix[0], episode=T8G.EPISODES[0]))
    chk('G-6 이 T8 에피소드를 stray 로 잡는다',
        'G-6' in G.blocked_by(G.gates(mix, anchor_rows=anc, timing_rows=tim)))
    # G-ANCHOR : 앵커 값이 어긋나면
    bad_anc = [dict(a, avg_nRMSE=(a['avg_nRMSE'] or 0) + 1e-9) for a in anc]
    chk('G-ANCHOR 표류가 잡힌다',
        'G-ANCHOR' in G.blocked_by(G.gates(rows, anchor_rows=bad_anc, timing_rows=tim)))
    chk('G-10 계측 0행이 잡힌다',
        'G-10' in G.blocked_by(G.gates(rows, anchor_rows=anc, timing_rows=[])))
    chk('G-10 스레드 위조가 잡힌다',
        'G-10' in G.blocked_by(G.gates(rows, anchor_rows=anc,
                                       timing_rows=[dict(t, threads=8) for t in tim])))
    chk('G-10 팔 불균형이 잡힌다',
        'G-10' in G.blocked_by(G.gates(rows, anchor_rows=anc, timing_rows=tim[:-1])))
    chk('G-ANCHOR 앵커 0행도 잡힌다',
        'G-ANCHOR' in G.blocked_by(G.gates(rows, anchor_rows=[], timing_rows=tim)))

    print('\n── B `blocked_by` 가 id 를 돌려준다 ' + '─' * 24)
    chk('gate_id 가 접두 id 를 뽑는다', G.gate_id('G-ANCHOR M1 == 동결') == 'G-ANCHOR')
    chk('BLOCKING 의 모든 id 가 게이트 이름에 있다',
        set(G.BLOCKING) <= {G.gate_id(n) for n, _, _ in g0},
        str(sorted(set(G.BLOCKING) - {G.gate_id(n) for n, _, _ in g0})))

    print('\n── C 라벨 공역 ' + '─' * 44)
    def cur(v):
        return {f'{d:g}': v for d in G.DELTA_GRID}
    seen = set()
    for c, star in ((cur(64), 0.001), (cur(57), 0.001), (cur(30), None), (cur(4), None)):
        lab, gr = G.label_for(dict(n=64, curve=c, delta_star=star))
        seen.add((lab or '').split(':: ')[-1].split(' (')[0])
    want = {'ENS-NO-HARM-SPARSE', 'ENS-HARMS-SPARSE', 'ENS-HARM-UNDECIDED'}
    chk('결정표 세 칸이 전부 도달 가능', want <= seen, str(sorted(want - seen)))
    chk('NOROWS 가 라벨을 내지 않는다',
        G.label_for(dict(n=0, curve=None))[0] is None)
    chk('★라벨이 delta* 를 담는다',
        'delta*=0.002' in (G.label_for(dict(n=64, curve=cur(60), delta_star=0.002))[0] or ''))

    print('\n' + ('✅자가진단 통과' if OK[0] else '⛔자가진단 실패'))
    return 0 if OK[0] else 1


if __name__ == '__main__':
    sys.exit(main())
