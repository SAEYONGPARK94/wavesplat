#!/usr/bin/env python3
"""`T7b-SPECTRAL` 분석 — 게이트 → 대조 → 라벨 → `verdict_t7b.json`. ⛔런 0회."""
import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_t7b as G                                              # noqa: E402


def sign_p(k, n):
    """단측 부호검정 p = P(X >= k), X~Bin(n, 0.5)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main():
    rows = [json.loads(x) for x in
            open(os.path.join(HERE, 'results_t7b.jsonl'))]
    gates = G.evaluate(rows)
    blocked = G.blocked_by(gates)
    print('── 게이트 ──────────────────────────────────────────')
    for n, p, d in gates:
        print(f'  [{"OK  " if p else "FAIL"}] {n:22s} {d}')
    c = G.contrasts(rows)
    lab = G.label(c)
    c['A_p_one_sided'] = sign_p(c['A_learn_better'], c['n'])
    c['B_p_one_sided'] = sign_p(c['B_learn_better'], c['n'])
    v = {
        'batch': 'T7b-SPECTRAL', 'n_rows': len(rows),
        'gates': [[n, bool(p), d] for n, p, d in gates],
        'gates_pass': f'{sum(1 for _, p, _ in gates if p)}/{len(gates)}',
        'blocked_by': blocked,
        'label': lab if not blocked else 'NO-LABEL (blocked)',
        'contrasts': c,
        'thresholds': {'NOMINAL': G.NOMINAL, 'CONFIRM': G.CONFIRM,
                       'NOT_EARN': G.NOT_EARN, 'n': G.N_PAIRS, 'q80': 0.7109},
        'prereg': 'PREREG.md',
        'disclosure': [
            '⛔⛔blind 아님(최대치) — `T7` 의 **모든 수치를 보았다**(두 대조의 방향·크기, '
            '해석적 팔이 제로장 아래로 떨어지는 양상까지) ⟹ **확증 주장 금지**, 등급 `EXPLORATORY` 고정.',
            '⛔`T7` 의 수치와 합산·비교 금지 — 코퍼스도 참값 생성기도 다르다.',
            '★`G-5c`(비교자가 제로장을 이기는 비율)는 **차단이 아니라 공시**다.',
            '⛔외부 기준선과 비교하지 않았다 — LETKF/ETKF 를 돌리지 않는다.',
            '⛔nRMSE 결론을 갱신하지 않는다 — 다른 지표다.',
            '⛔`directional_r2` 를 "스펙트럼 정확도" 로 일반화 금지 — 2D 파워 스펙트럼의 '
            'R² 이고 **위상을 보지 않는다**.',
            '⚠혼잡 층 한정(15·30 성분) · HOS 비선형 참값 · 단일 시드(42).',
            '⚠대조는 **우리 시스템 안의 두 초기화 경로**다(절제 축).',
        ],
    }
    json.dump(v, open(os.path.join(HERE, 'verdict_t7b.json'), 'w'),
              indent=2, ensure_ascii=False)
    print(f"\n대조 A (LEARN 방향 R² 높음): {c['A_learn_better']}/{c['n']} "
          f"(Δ중앙 {c['A_median_delta']:+.4f}, p={c['A_p_one_sided']:.3g})")
    print(f"대조 B (LEARN Hs 오차 낮음): {c['B_learn_better']}/{c['n']} "
          f"(Δ중앙 {c['B_median_delta']:+.4f}, p={c['B_p_one_sided']:.3g})")
    print(f"문턱 NOMINAL {G.NOMINAL} · CONFIRM {G.CONFIRM} · NOT_EARN {G.NOT_EARN}")
    print(f"\n라벨: {v['label']}   게이트 {v['gates_pass']}"
          + (f'  ⛔차단 {blocked}' if blocked else ''))
    return 1 if blocked else 0


if __name__ == '__main__':
    sys.exit(main())
