"""`C1b-BESTFIT` 분석기 — ⛔**결과 전에 작성·동결**."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_c1b as G                                                # noqa: E402
RES = os.path.join(HERE, 'results_c1b.jsonl'); META = os.path.join(HERE, 'meta_c1b.json')
OUT = os.path.join(HERE, 'verdict_c1b.json')


def q(a, p):
    a = sorted(a)
    if not a:
        return float('nan')
    return a[min(len(a) - 1, max(0, int(round(p * (len(a) - 1)))))]


def main():
    rows = [json.loads(l) for l in open(RES, encoding='utf-8') if l.strip()]
    meta = json.load(open(META, encoding='utf-8'))
    ok = [r for r in rows if 'error' not in r]
    crow = [r for r in ok if r['group'] == 'hos' and r['crowded']]
    spar = [r for r in ok if r['group'] == 'hos' and not r['crowded']]
    ctrl = [r for r in ok if r['group'] == 'linear']
    w = sum(1 for r in crow if r['rel_ls'] >= G.GAP_MIN)
    gate_rows = G.gates(rows, meta); blocked = G.blocked_by(gate_rows)
    lab, msg = (('BLOCKED', '차단 게이트 실패: ' + ','.join(blocked)) if blocked
                else G.label(w))
    lab = 'C1b-BESTFIT :: ' + lab

    def summ(rs):
        a = [r['rel_ls'] for r in rs]; b = [r['rel_gap_c1'] for r in rs]
        rr = [x / y for x, y in zip(a, b) if y > 0]
        return dict(n=len(rs), ls_min=q(a, 0), ls_med=q(a, .5), ls_max=q(a, 1),
                    c1_med=q(b, .5), ratio_med=q(rr, .5), ratio_min=q(rr, 0),
                    n_above=sum(1 for x in a if x >= G.GAP_MIN))
    v = dict(batch='C1b-BESTFIT', kind='LEAST-SQUARES-REFIT', new_runs=0,
             grade='CONFIRMATORY', n_rows=len(rows),
             gates=[dict(name=n, pass_=bool(o), detail=str(d)) for n, o, d in gate_rows],
             gates_pass=f'{sum(1 for _, o, _ in gate_rows if o)}/{len(gate_rows)}',
             blocked_by=blocked, threshold=G.GAP_MIN,
             primary=dict(stat='rel_ls >= 0.10 (HOS crowded)', w=w, n=len(crow)),
             strata=dict(hos_crowded=summ(crow), hos_sparse=summ(spar),
                         linear_control=summ(ctrl)),
             per_episode={r['episode']: dict(rel_ls=r['rel_ls'], rel_gap_c1=r['rel_gap_c1'],
                                             group=r['group'], crowded=r['crowded'])
                          for r in ok},
             label=lab, label_msg=msg,
             disclosures=[
                 '⛔C1 판정 갱신 아님 — 이 배치는 그 수치의 **해석**을 좁힌다.',
                 '⛔파수 자유도는 주지 않았다 — 이 잔차는 **상한**이다("모델류가 이만큼 못 한다" 로 읽지 마라).',
                 '⛔정확도 주장 아님 · 추정기 런 0회 · 실측 적용 불가.',
                 '⚠C1 과 같은 stride=25 프레임 표본.',
             ])
    json.dump(v, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{lab}\n  {msg}\n  gates {v["gates_pass"]}')
    for n, o, d in gate_rows:
        print(f"   [{'OK ' if o else 'FAIL'}] {n:46s} {d}")
    for k, s in v['strata'].items():
        print(f"  {k:16s} n={s['n']:3d}  rel_ls {s['ls_min']:.4f}/{s['ls_med']:.4f}/{s['ls_max']:.4f}"
              f"  (C1 중앙 {s['c1_med']:.4f}, 비 중앙 {s['ratio_med']:.3f})  ≥{G.GAP_MIN}: {s['n_above']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
