"""`R6-CONF` 분석기 — ⛔런 0회. 규칙은 `gates_r6.py` 단일출처."""
import json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_r6 as G                                                # noqa: E402
ROWS = os.path.join(HERE, 'results_r6.jsonl'); OUT = os.path.join(HERE, 'verdict_r6.json')


def main():
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    ok = [r for r in rows if 'error' not in r]
    by = {(r['cell'], r['record']): r for r in ok}
    _, sha = G.measured_geometry()
    gts = G.gates(rows, sha); blocked = G.blocked_by(gts)

    sat = {r['record']: r['conf_stats']['frac_saturated']
           for r in ok if r.get('conf_stats')}
    n_sat = sum(1 for v in sat.values() if v >= 1.0)
    lab, msg = (G.label(n_sat) if not blocked else (None, 'BLOCKED'))

    def pair(base):
        ks = [k for k in G.RECORDS if (f'{base}-C1', k) in by and (f'{base}-CVAR', k) in by]
        chg = [k for k in ks if by[(f'{base}-C1', k)]['curve'] != by[(f'{base}-CVAR', k)]['curve']]
        rel = [(by[(f'{base}-CVAR', k)]['nowcast'] - by[(f'{base}-C1', k)]['nowcast'])
               / by[(f'{base}-C1', k)]['nowcast'] * 100 for k in ks]
        better = sum(1 for k in ks
                     if by[(f'{base}-CVAR', k)]['nowcast'] < by[(f'{base}-C1', k)]['nowcast'])
        return dict(n=len(ks), n_changed=len(chg), changed=chg, better=better,
                    rel_median_pct=(st.median(rel) if rel else None),
                    reads_confidence=(len(chg) == len(ks) and len(ks) > 0))

    v = dict(batch='R6-CONF', n_rows=len(rows),
             gates=[[a, bool(b), c] for a, b, c in gts],
             gates_pass=f'{sum(1 for _, b, _ in gts if b)}/{len(gts)}',
             blocked_by=blocked, grade=('BLOCKED' if blocked else 'EXPLORATORY'),
             label=(f'R6-CONF :: {lab}' if lab else None), label_msg=msg,
             primary=dict(n_saturated=n_sat, n=G.N, frac_by_record=sat,
                          d_ref=G.d_ref_default(),
                          raw_min=min((r['conf_stats']['raw_min'] for r in ok
                                       if r.get('conf_stats')), default=None),
                          raw_med_range=[min(r['conf_stats']['raw_med'] for r in ok if r.get('conf_stats')),
                                         max(r['conf_stats']['raw_med'] for r in ok if r.get('conf_stats'))],
                          note='⚠부호검정이 아니다 — 포화 비율은 자료의 성질이다'),
             secondary=dict(SHIP=pair('SHIP'), PROP=pair('PROP'),
                            geometry_range=dict(
                                r_med=[min(r['conf_stats']['r_med'] for r in ok if r.get('conf_stats')),
                                       max(r['conf_stats']['r_med'] for r in ok if r.get('conf_stats'))],
                                cos_med=[min(r['conf_stats']['cos_med'] for r in ok if r.get('conf_stats')),
                                         max(r['conf_stats']['cos_med'] for r in ok if r.get('conf_stats'))])),
             disclosure=[
                 '⛔blind 아님 — 동결 전 스모크가 포화표를 드러냈고 그것이 재설계의 근거다(PREREG §0c).',
                 '⛔CONFIRMATORY 아님 — EXPLORATORY.',
                 '⛔`CVAR` 는 **계측기**이지 배포 후보가 아니다 — 임의 재척도다.',
                 '⛔`CVAR` 로 정확도 주장 금지. 개선이 보여도 그것은 이 배치의 결론이 아니다.',
                 '⛔배포 기본값 변경 아님 — 하네스의 입력만 바꿨다.',
                 '⛔BS 7 · 1 사이트 — 사이트 일반화 금지, 실측 12 판정의 재채점 아님.',
                 '⛔"신뢰가중이 쓸모없다" 로 넓히지 마라 — 잰 것은 (a) 이 사거리에서 공식이 상수라는 것과 '
                 '(b) 어느 경로가 그 필드를 읽는가 둘뿐이다.',
                 '⚠평면 근사 법선을 썼다(GT 누출 회피). 참 경사를 쓰면 값이 달라진다.',
             ])
    json.dump(v, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f"게이트 {v['gates_pass']} · 차단 {blocked}")
    for a, b, c in gts:
        if not b: print(f'  FAIL {a}  {c}')
    print(f"PRIMARY 포화 {n_sat}/{G.N} · 미클램프 최소 {v['primary']['raw_min']:.3f} "
          f"· 중앙 {v['primary']['raw_med_range'][0]:.2f}–{v['primary']['raw_med_range'][1]:.2f}")
    for b_ in ('SHIP', 'PROP'):
        d = v['secondary'][b_]
        print(f"  SEC {b_:5s} 비트변화 {d['n_changed']}/{d['n']} · 읽는가={d['reads_confidence']} "
              f"· 개선 {d['better']}/{d['n']} · rel {d['rel_median_pct']}")
    print(f"라벨: {v['label']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
