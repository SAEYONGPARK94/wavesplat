#!/usr/bin/env python3
"""`T8-ENSDEFAULT` 분석 — ⛔규칙은 `gates_t8` 단일출처. 이 파일은 **읽고 찍기만** 한다.

⛔런 0회. 동결 행만 읽는다.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_t8 as G  # noqa: E402


def _load(name):
    p = os.path.join(HERE, name)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def main():
    rows = _load('results_t8.jsonl')
    anchor = _load('anchor_t8.jsonl')
    timing = _load('timing_t8.jsonl')
    if not rows:
        print('⛔결과 행이 없다 — run_t8.py 를 먼저 돌려라')
        return 2

    gates = G.gates(rows, anchor_rows=anchor, timing_rows=timing)
    blocked = G.blocked_by(gates)
    ni = G.noninferiority(rows)
    cost = G.cost(rows)
    fr = G.frames(timing) if timing else None
    t = G.thresholds()

    print('── 게이트 ' + '─' * 50)
    for n, ok, d in gates:
        print(f'  {"✅" if ok else "⛔"} {n}' + (f'   {d}' if d else ''))
    npass = sum(1 for _, o, _ in gates if o)
    print(f'  게이트 {npass}/{len(gates)} · 차단 {blocked or "없음"}')

    print('\n── PRIMARY (희소 비열등성) ' + '─' * 33)
    print(f"  n={ni['n']} · δ={t['delta_pct']}% · K={ni['K']} (문턱 {t['K_MIN']})")
    print(f"  상대 중앙 {ni['rel_median_pct']:+.2f}% · 최악 {ni['worst_pct']:+.2f}% · "
          f"더 좋음 {ni['better']}/{ni['n']}")
    print(f"  p(H0: p<={t['p0']}) = {ni['p_one_sided']:.4g}")
    print(f"  검정력: p=0.90 → {t['power_at_090']} · p=0.95 → {t['power_at_095']} · "
          f"MDE(0.8) = {t['mde_p80']}")

    print('\n── SEC-A (비용) ' + '─' * 44)
    print(f"  M1 초기화 중앙 {cost['M1']['init_s_median']:.2f}s · "
          f"M3 {cost['M3']['init_s_median']:.2f}s · 문턱 {cost['threshold_s']:.2f}s "
          f"(T6 CPU|ANA|sparse)")
    print(f"  {'✅PASS' if cost['pass'] else '⛔FAIL'}")

    if fr:
        print('\n── SEC-B (프레임 예산) ' + '─' * 38)
        print(f"  M1 {fr['M1']['frame_ms_median']:.2f}ms · M3 {fr['M3']['frame_ms_median']:.2f}ms "
              f"· 10Hz 예산 {fr['budget_ms']:.0f}ms · {'✅PASS' if fr['pass'] else '⛔FAIL'}")

    label, grade = G.label_for(ni, cost)
    if blocked:
        label, grade = None, 'BLOCKED'
        print(f'\n⛔차단 {blocked} ⟹ **라벨 없음**. 게이트를 고쳐 다시 판정하지 않는다.')
    else:
        print(f'\n★라벨: {label}   [{grade}]')
        key = (label or '').split(':: ')[-1]
        print(f'  읽기: {G.READING.get(key, "")}')

    verdict = dict(
        batch=G.BATCH, n_rows=len(rows), n_anchor=len(anchor), n_timing=len(timing),
        gates=[[n, o, d] for n, o, d in gates],
        gates_pass=f'{npass}/{len(gates)}', blocked_by=blocked,
        label=label, grade=grade,
        thresholds=t, primary=ni, cost=cost, frames=fr,
        prereg='PREREG.md', disclosure=G.DISCLOSURE,
    )
    out = os.path.join(HERE, 'verdict_t8.json')
    json.dump(verdict, open(out, 'w'), ensure_ascii=False, indent=1)
    print(f'\n→ {os.path.relpath(out, ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
