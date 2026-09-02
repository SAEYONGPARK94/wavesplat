"""`R4c-DISPERSION` 분석기 — ⛔런 0회. 규칙은 `gates_r4c.py` 단일출처."""
import json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_r4c as G                                               # noqa: E402
ROWS = os.path.join(HERE, 'results_r4c.jsonl'); OUT = os.path.join(HERE, 'verdict_r4c.json')


def summ(rs):
    return dict(n=len(rs), n_pass=sum(1 for r in rs if r['passes']),
                passed=[r['record'] for r in rs if r['passes']],
                obs_median=(st.median([r['sheetfrac_mean'] for r in rs]) if rs else None),
                p95_r4b_median=(st.median([r['null_p95_r4b'] for r in rs]) if rs else None),
                p95_median=(st.median([r['null_p95'] for r in rs]) if rs else None),
                p95_drop_pct_median=(st.median([(r['null_p95'] - r['null_p95_r4b'])
                                                / r['null_p95_r4b'] * 100 for r in rs])
                                     if rs else None),
                overlap_median=(st.median([r['overlap_frac'] for r in rs]) if rs else None),
                margin_min=(min(r['null_p95'] - r['sheetfrac_mean'] for r in rs) if rs else None))


def main():
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    ok = [r for r in rows if 'error' not in r and not r.get('void')]
    ctl = [r for r in ok if r['group'] == 'control']
    trt = [r for r in ok if r['group'] == 'agulhas']
    gts = G.gates(rows); blocked = G.blocked_by(gts)
    cp = sum(1 for r in ctl if r['passes']); tp = sum(1 for r in trt if r['passes'])
    lab, msg = (G.label(cp, tp) if not blocked else (None, 'BLOCKED'))

    v = dict(batch='R4c-DISPERSION', n_rows=len(rows),
             gates=[[a, bool(b), c] for a, b, c in gts],
             gates_pass=f'{sum(1 for _, b, _ in gts if b)}/{len(gts)}',
             blocked_by=blocked, grade=('BLOCKED' if blocked else 'EXPLORATORY'),
             label=(f'R4c-DISPERSION :: {lab}' if lab else None), label_msg=msg,
             primary=dict(control_pass=f'{cp}/{len(ctl)}', treat_pass=f'{tp}/{len(trt)}'),
             prediction=dict(written_before_results='대조 12/12 · 처치 0/6 유지',
                             flip_probability_stated=0.30,
                             most_likely_flip='1150 (비 1.06)',
                             outcome=('예측 적중' if cp == len(ctl) and tp == 0 else '예측 빗나감')),
             groups=dict(control=summ(ctl), agulhas=summ(trt)),
             fix=dict(kind='이중 계수 제거 (R4b 산술 보존 + 중복분 차감)',
                      why='두 ±3 밴드가 2(jl+s) ≡ 0 (mod nf) 부근에서 같은 빈을 공유',
                      confound='jl ∝ dt 이고 dt 가 군과 완전 교락 (처치 0.5 s 전건)'),
             disclosure=[
                 '⛔blind 아님 — R4b 의 관측·귀무 요약·라벨과 감사의 중첩률을 전부 알고 동결했다.',
                 '⛔CONFIRMATORY 아님 — EXPLORATORY.',
                 '⛔R4b 의 계수를 갱신하지 않는다 — 답한 질문은 "라벨이 인공물의 산물인가" 하나다.',
                 '⛔0/6 이 유지돼도 **부재의 증거가 아니다** — 관측은 자기 귀무 중앙보다 6/6 위다.',
                 '⛔dt 교락은 이것으로 완전히 닫히지 않는다 — 중복 계수는 그 한 통로일 뿐이다.',
                 '⛔정확도 주장 아님 · 기제 지목 금지 · 사이트 일반화 금지.',
             ])
    json.dump(v, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f"게이트 {v['gates_pass']} · 차단 {blocked}")
    for a, b, c in gts:
        if not b: print(f'  FAIL {a}  {c}')
    print(f"대조 {cp}/{len(ctl)} · 처치 {tp}/{len(trt)}")
    print(f"p95 변화 중앙: 대조 {v['groups']['control']['p95_drop_pct_median']:+.3f}% · "
          f"처치 {v['groups']['agulhas']['p95_drop_pct_median']:+.3f}%")
    print(f"자기중첩 중앙: 대조 {v['groups']['control']['overlap_median']:.3f} · "
          f"처치 {v['groups']['agulhas']['overlap_median']:.3f}")
    print(f"예측: {v['prediction']['outcome']}")
    print(f"라벨: {v['label']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
