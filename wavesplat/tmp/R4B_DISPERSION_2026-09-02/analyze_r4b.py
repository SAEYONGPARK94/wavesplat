"""`R4b-DISPERSION` 분석기 — ⛔런 0회. 규칙은 `gates_r4b.py` 단일출처."""
import json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_r4b as G                                                # noqa: E402
ROWS = os.path.join(HERE, 'results_r4b.jsonl'); OUT = os.path.join(HERE, 'verdict_r4b.json')


def summ(rs):
    v = [r['sheetfrac_mean'] for r in rs]
    return dict(n=len(rs), n_pass=sum(1 for r in rs if r['passes']),
                obs_min=(min(v) if v else None), obs_median=(st.median(v) if v else None),
                obs_max=(max(v) if v else None),
                ratio_min=(min(r['ratio_obs_null'] for r in rs) if rs else None),
                ratio_max=(max(r['ratio_obs_null'] for r in rs) if rs else None),
                passed=[r['record'] for r in rs if r['passes']],
                failed=[r['record'] for r in rs if not r['passes']])


def main():
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    ok = [r for r in rows if 'error' not in r and not r.get('void')]
    ctl = [r for r in ok if r['group'] == 'control']
    trt = [r for r in ok if r['group'] == 'agulhas']
    gts = G.gates(rows); blocked = G.blocked_by(gts)
    cp = sum(1 for r in ctl if r['passes']); tp = sum(1 for r in trt if r['passes'])
    lab, msg = (G.label(cp, tp) if not blocked else (None, 'BLOCKED'))
    phat = (cp / len(ctl)) if ctl else float('nan')

    v = dict(batch='R4b-DISPERSION', n_rows=len(rows),
             gates=[[a, bool(b), c] for a, b, c in gts],
             gates_pass=f'{sum(1 for _, b, _ in gts if b)}/{len(gts)}',
             blocked_by=blocked, grade=('BLOCKED' if blocked else 'EXPLORATORY'),
             label=(f'R4b-DISPERSION :: {lab}' if lab else None), label_msg=msg,
             primary=dict(control_pass=f'{cp}/{len(ctl)}', treat_pass=f'{tp}/{len(trt)}',
                          p_hat_control=phat,
                          prob_zero_of_6_if_like_control=((1 - phat) ** len(trt) if trt else None),
                          note='⛔사후 p̂ 이므로 유의성 주장 아님 — 크기 감각용(PREREG §3)'),
             groups=dict(control=summ(ctl), agulhas=summ(trt)),
             null=dict(kind='shifted-sheet (분산곡선을 ω 축에서 |s|≥8 빈 옮김)',
                       B=G.N_NULL, seed=G.NULL_SEED, pctl=G.PCTL,
                       why_not_iaaft='sheetfrac 은 |FFT|² 만의 함수 ⟹ 위상무작위화는 검정력 0'),
             disclosure=[
                 '⛔blind 아님 — R4 에서 관측 sheetfrac 18 개를 전부 보았다(귀무는 미노출).',
                 '⛔CONFIRMATORY 아님 — 기능은 **동결·재현 가능한 기록**이다.',
                 '⛔정확도·성능 주장 아님 — 추정기 런 0회.',
                 '⛔"남극 자료가 나쁘다" 아님 — **우리 모델의 대상이 아니다**가 전부다.',
                 '⛔기제 지목 금지 — 플랫폼 운동인지 긴 스웰인지 표면 자료로 분리 불가.',
                 '⛔사이트 일반화 금지 — 남극 1척·1일·1폭풍 · 실측 4 사이트.',
                 '⛔LJ18a 가 최저인 것과 그것이 배포 팔의 유일 반례인 것을 연결 금지(n=1·교락).',
                 '⚠창 1개. 시간 비정상성 미측정.',
             ])
    json.dump(v, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f"게이트 {v['gates_pass']} · 차단 {blocked}")
    for a, b, c in gts:
        if not b: print(f'  FAIL {a}  {c}')
    print(f"대조 통과 {cp}/{len(ctl)} · 처치 통과 {tp}/{len(trt)}")
    print(f"라벨: {v['label']}  ({msg})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
