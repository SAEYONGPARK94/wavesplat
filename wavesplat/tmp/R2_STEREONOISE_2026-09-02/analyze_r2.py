"""`R2-STEREONOISE` 분석기 — ⛔규칙은 `gates_r2.py` 단일출처. 런 0회."""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_r2 as G                                                # noqa: E402

ROWS = os.path.join(HERE, 'results_r2.jsonl')
OUT = os.path.join(HERE, 'verdict_r2.json')


def main():
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    gts = G.gates(rows)
    blocked = G.blocked_by(gts)
    ok = [r for r in rows if 'error' not in r]

    w = sum(1 for r in ok if r['stereo_gt_gauge'])
    p = G.sign_test_one_sided(w, G.N) if len(ok) == G.N else None
    lab, msg = (G.label(w) if not blocked else (None, 'BLOCKED'))

    exc = [r['hf_excess_rms_mm'] for r in ok if r['stereo_gt_gauge']]
    tp_rel = [abs(r['Tp_stereo'] - r['Tp_gauge']) / r['Tp_gauge'] for r in ok]
    hm0 = [r['Hm0_stereo'] / r['Hm0_gauge'] for r in ok if r['Hm0_gauge'] > 0]

    v = dict(
        batch='R2-STEREONOISE', n_rows=len(rows),
        gates=[[a, bool(b), c] for a, b, c in gts],
        gates_pass=f'{sum(1 for _, b, _ in gts if b)}/{len(gts)}',
        blocked_by=blocked, grade=('BLOCKED' if blocked else 'EXPLORATORY'),
        label=(f'R2-STEREONOISE :: {lab}' if lab else None), label_msg=msg,
        primary=dict(better=w, n=G.N, p_one_sided=p, threshold=G.THRESH),
        secondary=dict(
            hf_excess_rms_mm=dict(
                min=(min(exc) if exc else None), max=(max(exc) if exc else None),
                median=(st.median(exc) if exc else None)),
            Tp_rel_median=(st.median(tp_rel) if tp_rel else None),
            Hm0_ratio=dict(min=(min(hm0) if hm0 else None),
                           max=(max(hm0) if hm0 else None))),
        disclosure=[
            '⛔blind 아님 — 동결 전 탐색으로 방향과 대략적 크기를 이미 보았다(PREREG §6).',
            '⛔등급을 CONFIRMATORY 로 청구하지 않는다 — 기능은 발견이 아니라 **기록**이다.',
            '⚠초과분산은 **상한**이다(게이지 자신의 잡음·공간 비균질이 섞인다).',
            '⛔Hm0 산포를 스테레오 오차로 귀속 금지 — 공위치가 아니다.',
            '⛔격자점별 오차장·면 참값 주장 금지 — 6점(반경 0.25 m)은 면이 아니다.',
            '⛔BS 7 레코드 · 4 사이트 중 1 — 집계 실측 판정의 재채점이 아니다.',
            '★G-3 은 **같은 다중성의 순환이동 귀무분포**와 비교한다(스모크에서 설계 정정).',
        ])
    json.dump(v, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f"게이트 {v['gates_pass']} · 차단 {blocked}")
    for a, b, c in gts:
        if not b:
            print(f'  FAIL {a}  {c}')
    print(f"PRIMARY {w}/{G.N}" + (f' p={p:.4g}' if p else ''))
    print(f"라벨: {v['label']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
