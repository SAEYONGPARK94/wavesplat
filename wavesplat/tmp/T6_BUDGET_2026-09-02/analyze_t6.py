#!/usr/bin/env python3
"""`T6-BUDGET` 분석 — 게이트 → 라벨 → `verdict_t6.json`. ⛔런 0회."""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_t6 as G                                              # noqa: E402


def med(v):
    return sorted(v)[len(v) // 2]


def main():
    rows, calib = [], {}
    for dev in ('cpu', 'gpu'):
        p = os.path.join(HERE, f'results_t6_{dev}.jsonl')
        if os.path.exists(p):
            rows += [json.loads(x) for x in open(p)]
        c = os.path.join(HERE, f'calib_{dev}.json')
        if os.path.exists(c):
            calib.update(json.load(open(c)))
    gates = G.evaluate(rows, calib)
    blocked = G.blocked_by(gates)
    print('── 게이트 ──────────────────────────────────────────')
    for n, p, d in gates:
        print(f'  [{"OK  " if p else "FAIL"}] {n:22s} {d}')
    lab, m = G.label(rows)
    cells = {}
    for r in rows:
        k = (r['device'], r['arm'], r['stratum'])
        cells.setdefault(k, {'frame_med': [], 'init_s': [], 'final_n': []})
        cells[k]['frame_med'].append(med(r['frame_ms']))
        cells[k]['init_s'].append(r['init_s'])
        cells[k]['final_n'].append(r['final_n'])
    table = {'|'.join(k): {
        'n_episodes': len(v['init_s']),
        'frame_ms_median': round(med(v['frame_med']), 3),
        'frame_ms_iqr': [round(st.quantiles(v['frame_med'], n=4)[0], 3),
                         round(st.quantiles(v['frame_med'], n=4)[2], 3)]
        if len(v['frame_med']) >= 4 else None,
        'init_s_median': round(med(v['init_s']), 3),
        'init_s_min_max': [round(min(v['init_s']), 3), round(max(v['init_s']), 3)],
        'final_n_median': med(v['final_n']),
        'budget_frac_at_10hz': round(med(v['frame_med']) / 100.0, 4),
    } for k, v in sorted(cells.items())}
    print('\n── 셀 ─────────────────────────────────────────────')
    for k, v in table.items():
        print(f"  {k:24s} frame {v['frame_ms_median']:7.2f} ms "
              f"({v['budget_frac_at_10hz']*100:5.2f}% of 10 Hz)  "
              f"init {v['init_s_median']:7.2f} s  N={v['final_n_median']}")
    verdict = {
        'batch': 'T6-BUDGET', 'n_rows': len(rows),
        'gates': [[n, bool(p), d] for n, p, d in gates],
        'gates_pass': f'{sum(1 for _, p, _ in gates if p)}/{len(gates)}',
        'blocked_by': blocked,
        'label': (lab if not blocked else 'NO-LABEL (blocked)'),
        'label_metric_ms': m,
        'cells': table,
        'prereg': 'PREREG.md',
        'disclosure': [
            'blind 아님 — 스모크 2셀을 동결 전에 보았고 세 방향을 알고 있었다(PREREG §3).',
            '⛔정확도 주장 0건. 이 배치의 어떤 수도 정확도 비교에 쓰지 않는다.',
            '⛔철회된 프로세스간 수치(13.4/132.6 ms)와 비교 금지 — 다른 양이다.',
            '⛔기준선 지연과 비교 금지 — 이 배치는 기준선을 돌리지 않는다.',
            '⛔렌더 단가(B-6b 137 us/점)와 합산 금지 — 질의 쪽이고 팔이 다르다.',
            '⛔실측 전이 금지 — 실측은 다른 경로(bank_mmse)로 간다.',
            '⚠라벨은 추적만 서술한다. 초기화는 어느 라벨에도 들어가지 않는다.',
        ],
    }
    with open(os.path.join(HERE, 'verdict_t6.json'), 'w') as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    print(f"\n라벨: {verdict['label']}"
          + (f'  (혼잡·CPU·LEARN 프레임 중앙 {m:.2f} ms)' if m else ''))
    print(f"게이트 {verdict['gates_pass']}" + (f'  ⛔차단: {blocked}' if blocked else ''))
    return 1 if blocked else 0


if __name__ == '__main__':
    sys.exit(main())
