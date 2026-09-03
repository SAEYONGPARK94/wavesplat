"""`R4d-DISPERSION` 분석기 — ⛔런 0회. 규칙은 `gates_r4d.py` 단일출처."""
import json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_r4d as G                                               # noqa: E402
ROWS = os.path.join(HERE, 'results_r4d.jsonl'); OUT = os.path.join(HERE, 'verdict_r4d.json')


def summ(rs):
    return dict(n=len(rs), n_pass=sum(1 for r in rs if r['passes']),
                passed=[r['record'] for r in rs if r['passes']],
                obs_median=(st.median([r['sheetfrac_mean'] for r in rs]) if rs else None),
                tanh_median=(st.median([r['tanh_med'] for r in rs]) if rs else None),
                p95_delta_median=(st.median([r['null_p95'] - r['null_p95_r4c'] for r in rs])
                                  if rs else None))


def main():
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    ok = [r for r in rows if 'error' not in r and not r.get('void')]
    ctl = [r for r in ok if r['group'] == 'control']
    trt = [r for r in ok if r['group'] == 'agulhas']
    r4c = {}
    for l in open(G.R4C_ROWS):
        if l.strip():
            x = json.loads(l)
            if 'sheetfrac' in x:
                r4c[(x['group'], x['record'])] = x
    gts = G.gates(rows, r4c); blocked = G.blocked_by(gts)
    cp = sum(1 for r in ctl if r['passes']); tp = sum(1 for r in trt if r['passes'])
    lab, msg = (G.label(cp, tp) if not blocked else (None, 'BLOCKED'))

    v = dict(batch='R4d-DISPERSION', n_rows=len(rows),
             gates=[[a, bool(b), c] for a, b, c in gts],
             gates_pass=f'{sum(1 for _, b, _ in gts if b)}/{len(gts)}',
             blocked_by=blocked, grade=('BLOCKED' if blocked else 'EXPLORATORY'),
             label=(f'R4d-DISPERSION :: {lab}' if lab else None), label_msg=msg,
             primary=dict(control_pass=f'{cp}/{len(ctl)}', treat_pass=f'{tp}/{len(trt)}',
                          matches_r4c=bool(cp == G.N_CTRL and tp == 0),
                          biteq=sum(1 for r in ok if r.get('biteq_r4c'))),
             groups=dict(control=summ(ctl), agulhas=summ(trt)),
             intervention=dict(kind='분산면을 심해 근사 √(gk) → 유한 수심 √(gk·tanh(kd))',
                               depths={r['record']: r['depth'] for r in ok},
                               tanh_range=[min(r['tanh_med'] for r in ok),
                                           max(r['tanh_med'] for r in ok)]),
             prediction=dict(written_before_results='DEEPWATER-EQUIVALENT',
                             flip_probability_stated=0.05,
                             outcome=('예측 적중' if lab == 'DEEPWATER-EQUIVALENT'
                                      else '예측 빗나감')),
             disclosure=[
                 '⛔발견이 아니다 — 기능은 **도구 확장의 무해성 증명**이다.',
                 '⛔blind 아님 — R4b·R4c 의 모든 값과 tanh 측정을 알고 동결했다.',
                 '⛔R4b·R4c 의 계수를 갱신하지 않는다.',
                 '⚠**얕은 물 자료에 대한 검증은 이 배치에 없다** — 확장이 그쪽에서 옳게 '
                 '작동하는지는 그런 자료를 실제로 넣어 보는 후속이 필요하다.',
                 '⛔"남극 자료가 나쁘다" 아님 · 기제 지목 금지 · 사이트 일반화 금지.',
             ])
    json.dump(v, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f"게이트 {v['gates_pass']} · 차단 {blocked}")
    for a, b, c in gts:
        if not b: print(f'  FAIL {a}  {c}')
    print(f"대조 {cp}/{len(ctl)} · 처치 {tp}/{len(trt)} · R4c 와 동일={v['primary']['matches_r4c']}")
    print(f"비트 동일 {v['primary']['biteq']}/{len(ok)} · tanh 범위 {v['intervention']['tanh_range']}")
    print(f"예측: {v['prediction']['outcome']}")
    print(f"라벨: {v['label']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
