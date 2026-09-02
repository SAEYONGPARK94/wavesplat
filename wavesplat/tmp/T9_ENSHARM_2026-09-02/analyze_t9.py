#!/usr/bin/env python3
"""`T9-ENSHARM` 분석 — ⛔규칙은 `gates_t9` 단일출처. 읽고 찍기만 한다. 런 0회."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
T8 = os.path.abspath(os.path.join(HERE, '..', 'T8_ENSDEFAULT_2026-09-02'))
for _p in (HERE, T8, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_t9 as G  # noqa: E402


def _load(name):
    p = os.path.join(HERE, name)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def main():
    rows, anchor, timing = _load('results_t9.jsonl'), _load('anchor_t9.jsonl'), _load('timing_t9.jsonl')
    if not rows:
        print('⛔결과 행이 없다')
        return 2
    gates = G.gates(rows, anchor_rows=anchor, timing_rows=timing)
    blocked = G.blocked_by(gates)
    ni = G.noninferiority(rows)
    cost = G.cost(rows, timing)
    t = G.thresholds()

    print('── 게이트 ' + '─' * 50)
    for n, ok, d in gates:
        print(f'  {"✅" if ok else "⛔"} {n}' + (f'   {d}' if d else ''))
    npass = sum(1 for _, o, _ in gates if o)
    print(f'  게이트 {npass}/{len(gates)} · 차단 {blocked or "없음"}')

    print('\n── PRIMARY (여유 격자 곡선) ' + '─' * 32)
    print(f"  n={ni['n']} · 문턱 K >= {t['K_MIN']}")
    for d in t['delta_grid']:
        k = ni['curve'][f'{d:g}']
        mark = '✅' if k >= t['K_MIN'] else '  '
        star = ' ★δ*' if ni['delta_star'] == d else ''
        print(f"    {mark} δ={d:<6g}  K={k:2d}/{ni['n']}{star}")
    print(f"  ★δ* = {ni['delta_star']}  (센서 눈금 {t['delta_sensor']} 와 비교하라)")
    print(f"  Δ 중앙 {ni['delta_median']:+.5f} · 최악 {ni['worst_abs']:+.5f} · "
          f"더 좋음 {ni['better']}/{ni['n']} · 상대 중앙 {ni['rel_median_pct']:+.2f}%")
    print(f"  예측(결과 전) = {t['predicted']}")

    print('\n── SEC 비용 (⛔라벨 밖 · 문턱 없음) ' + '─' * 24)
    for a in G.ARM_ORDER:
        c = cost[a]
        print(f"  {a}: 초기화 {c['init_s_median']:.2f}s · "
              f"프레임 {c['frame_ms_median']:.2f}ms" if c['frame_ms_median'] else
              f"  {a}: 초기화 {c['init_s_median']:.2f}s")

    label, grade = G.label_for(ni)
    if blocked:
        label, grade = None, 'BLOCKED'
        print(f'\n⛔차단 {blocked} ⟹ **라벨 없음**.')
    else:
        print(f'\n★라벨: {label}   [{grade}]')
        print(f"  읽기: {G.READING.get((label or '').split(':: ')[-1].split(' (')[0], '')}")

    json.dump(dict(batch=G.BATCH, n_rows=len(rows), n_anchor=len(anchor),
                   n_timing=len(timing),
                   gates=[[n, o, d] for n, o, d in gates],
                   gates_pass=f'{npass}/{len(gates)}', blocked_by=blocked,
                   label=label, grade=grade, thresholds=t, primary=ni, cost=cost,
                   prereg='PREREG.md', disclosure=G.DISCLOSURE),
              open(os.path.join(HERE, 'verdict_t9.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f"\n→ tmp/T9_ENSHARM_2026-09-02/verdict_t9.json")
    return 0


if __name__ == '__main__':
    sys.exit(main())
