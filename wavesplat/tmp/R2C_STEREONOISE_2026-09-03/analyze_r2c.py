"""`R2c-STEREONOISE` 분석기 — ⛔런 0회. 규칙은 `gates_r2c.py` 단일출처."""
import json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_r2c as G                                               # noqa: E402
ROWS = os.path.join(HERE, 'results_r2c.jsonl'); OUT = os.path.join(HERE, 'verdict_r2c.json')


def rng(v):
    return dict(min=min(v), median=st.median(v), max=max(v)) if v else None


def main():
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    ok = [r for r in rows if 'error' not in r]
    gts = G.gates(rows); blocked = G.blocked_by(gts)
    n_exc = sum(1 for r in ok if r['obs_exceeds_null'])
    lab, msg = (G.label(n_exc) if not blocked else (None, 'BLOCKED'))
    v = dict(batch='R2c-STEREONOISE', n_rows=len(rows),
             gates=[[a, bool(b), c] for a, b, c in gts],
             gates_pass=f'{sum(1 for _, b, _ in gts if b)}/{len(gts)}',
             blocked_by=blocked, grade=('BLOCKED' if blocked else 'EXPLORATORY'),
             label=(f'R2c-STEREONOISE :: {lab}' if lab else None), label_msg=msg,
             primary=dict(n_exceed=n_exc, n=G.N,
                          exceeded=[r['record'] for r in ok if r['obs_exceeds_null']],
                          obs_rmax=rng([r['obs_rmax'] for r in ok]),
                          obs_null_p95=rng([r['obs_null_p95'] for r in ok]),
                          margin=rng([r['obs_null_p95'] - r['obs_rmax'] for r in ok])),
             alignment=dict(clock_offset_s=rng([r['clock_offset_s'] for r in ok]),
                            win_start_s=rng([r['win_start_s'] for r in ok]),
                            lag_step_s=rng([r['lag_step_s'] for r in ok]),
                            n_lags=rng([r['n_lags'] for r in ok]),
                            obs_lag_minus_align=rng([r['obs_lag_minus_align'] for r in ok])),
             detector=dict(ctrl_rmax=rng([r['ctrl_rmax'] for r in ok]),
                           ctrl_pass=sum(1 for r in ok if r['ctrl_exceeds_null']),
                           lag_recovered=rng([r['ctrl_lag_recovered_s'] for r in ok]),
                           lag_injected=G.CTRL_LAG_S, timebase='absolute'),
             peak_coverage=rng([r['obs_peak_cov'] for r in ok]),
             supersedes_reason='R2b 의 lag 탐색이 참 정렬을 방문한 적이 없다 (R2b/AMENDMENT_1)',
             prediction=dict(written_before_results='POSITION-NULL-CONFIRMED (7/7 비초과)',
                             flip_probability_stated=0.20,
                             outcome=('예측 적중' if lab == 'POSITION-NULL-CONFIRMED'
                                      else '예측 빗나감')),
             disclosure=[
                 '⛔blind 아님 — 감사가 참 오프셋에서 두 레코드를 재본 것을 알고 동결했다.',
                 '⛔CONFIRMATORY 아님 — EXPLORATORY.',
                 '⛔정확도 주장 아님 · 추정기 런 0회 · R2b 계수 갱신 아님.',
                 '⛔BS 7 · 1 사이트 — 사이트 일반화 금지.',
                 '⚠다중성이 R2b 보다 낮다(탐색이 좁다) ⟹ 귀무 최대가 내려가 '
                 '"널이다" 가 더 어려워진다. 방향이 우리에게 불리한 쪽이라 그대로 썼다.',
                 '⚠널이 확인돼도 "이 자료·이 자·이 정렬에서" 의 진술이다.',
             ])
    json.dump(v, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f"게이트 {v['gates_pass']} · 차단 {blocked}")
    for a, b, c in gts:
        if not b: print(f'  FAIL {a}  {c}')
    print(f"PRIMARY 초과 {n_exc}/{G.N} · 여유 {v['primary']['margin']}")
    print(f"탐지기 대조 {v['detector']['ctrl_pass']}/{len(ok)} · lag 회수 {v['detector']['lag_recovered']}")
    print(f"예측: {v['prediction']['outcome']}")
    print(f"라벨: {v['label']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
