"""`R4-DISPERSION` 분석기 — ⛔런 0회. 규칙은 `gates_r4.py` 단일출처."""
import json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_r4 as G                                                # noqa: E402
ROWS = os.path.join(HERE, 'results_r4.jsonl'); OUT = os.path.join(HERE, 'verdict_r4.json')


def main():
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    ok = [r for r in rows if 'error' not in r and not r.get('void')]
    ctl = [r for r in ok if r['group'] == 'control']
    trt = [r for r in ok if r['group'] == 'agulhas']
    gts = G.gates(rows); blocked = G.blocked_by(gts)

    ctl_min = min((r['sheetfrac_mean'] for r in ctl), default=None)
    w = sum(1 for r in trt if ctl_min is not None and r['sheetfrac_mean'] < ctl_min)
    p = G.sign_test_one_sided(w, G.N_TREAT) if len(trt) == G.N_TREAT else None
    lab, msg = (G.label(w) if not blocked else (None, 'BLOCKED'))

    v = dict(batch='R4-DISPERSION', n_rows=len(rows),
             gates=[[a, bool(b), c] for a, b, c in gts],
             gates_pass=f'{sum(1 for _, b, _ in gts if b)}/{len(gts)}',
             blocked_by=blocked, grade=('BLOCKED' if blocked else 'EXPLORATORY'),
             label=(f'R4-DISPERSION :: {lab}' if lab else None), label_msg=msg,
             primary=dict(better=w, n=G.N_TREAT, p_one_sided=p, control_min=ctl_min),
             groups=dict(
                 control=dict(n=len(ctl),
                              sheetfrac_min=ctl_min,
                              sheetfrac_median=(st.median([r['sheetfrac_mean'] for r in ctl])
                                                if ctl else None),
                              sheetfrac_max=max((r['sheetfrac_mean'] for r in ctl), default=None)),
                 agulhas=dict(n=len(trt),
                              sheetfrac_min=min((r['sheetfrac_mean'] for r in trt), default=None),
                              sheetfrac_median=(st.median([r['sheetfrac_mean'] for r in trt])
                                                if trt else None),
                              sheetfrac_max=max((r['sheetfrac_mean'] for r in trt), default=None))),
             context=dict(designsafe_basin='8/8 파일 sheetfrac 0.05–0.14 (별도 측정, ⛔이 판정에 쓰지 않음)'),
             disclosure=[
                 '⛔blind 아님 — 감사가 이미 방향과 크기를 쟀고 나도 보았다(PREREG §5).',
                 '⛔CONFIRMATORY 청구 안 함 — 기능은 **동결·재현 가능한 기록**이다.',
                 '⛔정확도·성능 주장 아님 — 추정기 런 0회.',
                 '⛔"남극 자료가 나쁘다" 아님 — **우리 모델의 대상이 아니다**가 전부다.',
                 '⛔기제 지목 금지 — 플랫폼 운동인지 긴 스웰인지 표면 자료로 분리 불가.',
                 '⛔사이트 일반화 금지 — 남극은 1척·1일·1폭풍.',
                 '⚠sheetfrac 은 상자 크기·결측률에 의존한다 — 두 군의 그 값을 병기하라.',
                 '⚠창 1개. 시간 비정상성은 재지 않았다.',
             ])
    json.dump(v, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f"게이트 {v['gates_pass']} · 차단 {blocked}")
    for a, b, c in gts:
        if not b: print(f'  FAIL {a}  {c}')
    print(f"대조 {len(ctl)}: 중앙 {v['groups']['control']['sheetfrac_median']}")
    print(f"처치 {len(trt)}: 중앙 {v['groups']['agulhas']['sheetfrac_median']}")
    print(f"PRIMARY {w}/{G.N_TREAT}" + (f' p={p:.4g}' if p else ''))
    print(f"라벨: {v['label']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
