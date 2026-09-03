"""`C1-MODELGAP` 분석기 — ⛔**결과 전에 작성·동결**. 판정은 여기서만 나온다."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_c1 as G                                                 # noqa: E402

RES = os.path.join(HERE, 'results_c1.jsonl')
META = os.path.join(HERE, 'meta_c1.json')
OUT = os.path.join(HERE, 'verdict_c1.json')


def q(a, p):
    a = sorted(a)
    if not a:
        return float('nan')
    i = min(len(a) - 1, max(0, int(round(p * (len(a) - 1)))))
    return a[i]


def main():
    rows = [json.loads(l) for l in open(RES, encoding='utf-8') if l.strip()]
    meta = json.load(open(META, encoding='utf-8'))
    ok = [r for r in rows if 'error' not in r]
    crow = [r for r in ok if r['group'] == 'hos' and r['crowded']]
    spar = [r for r in ok if r['group'] == 'hos' and not r['crowded']]
    ctrl = [r for r in ok if r['group'] == 'linear']
    w = sum(1 for r in crow if r['rel_gap'] >= G.GAP_MIN)

    gate_rows = G.gates(rows, meta)
    blocked = G.blocked_by(gate_rows)
    if blocked:
        lab, msg = 'BLOCKED', '차단 게이트 실패: ' + ','.join(blocked)
    else:
        lab, msg = G.label(w)
    lab = 'C1-MODELGAP :: ' + lab

    def summ(rs):
        g = [r['rel_gap'] for r in rs]
        return dict(n=len(rs), min=q(g, 0.0), med=q(g, 0.5), max=q(g, 1.0),
                    n_above=sum(1 for x in g if x >= G.GAP_MIN))

    v = dict(batch='C1-MODELGAP', kind='RECONSTRUCTION-RESIDUAL', new_runs=0,
             grade='EXPLORATORY', n_rows=len(rows),
             gates=[dict(name=n, pass_=bool(o), detail=str(d)) for n, o, d in gate_rows],
             gates_pass=f'{sum(1 for _, o, _ in gate_rows if o)}/{len(gate_rows)}',
             blocked_by=blocked, threshold=G.GAP_MIN,
             primary=dict(stat='rel_gap >= 0.10 (HOS crowded)', w=w, n=len(crow)),
             strata=dict(hos_crowded=summ(crow), hos_sparse=summ(spar),
                         linear_control=summ(ctrl)),
             mesh_verdicts={},
             per_episode={r['episode']: dict(group=r['group'], crowded=r['crowded'],
                                             max_abs_delta=r['max_abs_delta'],
                                             sigma=r['sigma'], rel_gap=r['rel_gap'],
                                             mesh=r.get('mesh'))
                          for r in ok},
             label=lab, label_msg=msg,
             disclosures=[
                 '⛔CLAUDE.md 의 BLOCKER 를 철회하지 않는다 — 그 진술은 **선형 코퍼스**에 대해 참이다(G-CTRL 이 전건 0.0 으로 재확인).',
                 '⛔이 배치는 **거리**를 재지 **충분성**을 재지 않는다 — "모델 불일치가 충분하다" 로 읽지 마라.',
                 '⛔정확도 주장 아님 · 추정기 런 0회.',
                 '⛔HOSGATE 와 혼동 금지 — 그것은 정확도의 개입 측정, 이것은 참값과 모델류의 기하적 거리.',
                 '⚠blind 아님 — 동결 전에 HOS 6 에피소드를 보았다(그 확인이 내 열 규약 오류를 잡았다).',
                 '⚠mesh_verdict 가 시드별로 갈린다 — 행에 기록했다.',
                 '⛔실측 코퍼스에는 적용 불가(성분표가 없다).',
             ])
    for r in ok:
        k = str(r.get('mesh'))
        v['mesh_verdicts'][k] = v['mesh_verdicts'].get(k, 0) + 1
    json.dump(v, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{lab}\n  {msg}\n  gates {v["gates_pass"]}')
    for n, o, d in gate_rows:
        print(f"   [{'OK ' if o else 'FAIL'}] {n:44s} {d}")
    for k, s in v['strata'].items():
        print(f"  {k:16s} n={s['n']:3d}  rel_gap min {s['min']:.4f} · med {s['med']:.4f} · "
              f"max {s['max']:.4f}  (≥{G.GAP_MIN}: {s['n_above']})")
    print('  mesh:', v['mesh_verdicts'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
