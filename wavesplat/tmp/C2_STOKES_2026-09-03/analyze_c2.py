"""`C2-STOKES` 분석기 — ⛔**결과 전에 작성·동결**."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_c2 as G                                                 # noqa: E402
RES = os.path.join(HERE, 'results_c2.jsonl'); META = os.path.join(HERE, 'meta_c2.json')
OUT = os.path.join(HERE, 'verdict_c2.json')


def q(a, p):
    a = sorted(x for x in a if x == x)
    return float('nan') if not a else a[min(len(a)-1, max(0, int(round(p*(len(a)-1)))))]


def main():
    rows = [json.loads(l) for l in open(RES, encoding='utf-8') if l.strip()]
    meta = json.load(open(META, encoding='utf-8')) if os.path.exists(META) else {}
    ok = [r for r in rows if 'error' not in r]
    crow = [r for r in ok if r['group'] == 'hos' and r['crowded']]
    spar = [r for r in ok if r['group'] == 'hos' and not r['crowded']]
    ctrl = [r for r in ok if r['group'] == 'linear']
    w = sum(1 for r in crow if r['rms_st2'] < r['rms_lin'])
    gate_rows = G.gates(rows, meta); blocked = G.blocked_by(gate_rows)
    lab, msg = (('BLOCKED', '차단: ' + ','.join(blocked)) if blocked else G.label(w))
    lab = 'C2-STOKES :: ' + lab

    def summ(rs):
        rr = [r['rms_st2'] / r['rms_lin'] for r in rs if r['rms_lin'] > 0]
        return dict(n=len(rs), better=sum(1 for r in rs if r['rms_st2'] < r['rms_lin']),
                    lin_med=q([r['rms_lin'] for r in rs], .5),
                    st2_med=q([r['rms_st2'] for r in rs], .5),
                    ratio_min=q(rr, 0), ratio_med=q(rr, .5), ratio_max=q(rr, 1))
    v = dict(batch='C2-STOKES', kind='THEORY-PREDICTION-NO-FIT', new_runs=0,
             grade='CONFIRMATORY', n_rows=len(rows),
             gates=[dict(name=n, pass_=bool(o), detail=str(d)) for n, o, d in gate_rows],
             gates_pass=f'{sum(1 for _, o, _ in gate_rows if o)}/{len(gate_rows)}',
             blocked_by=blocked, thresholds=dict(yes=G.THRESH_YES, partial=G.THRESH_PARTIAL),
             primary=dict(stat='rms_st2 < rms_lin (HOS crowded)', w=w, n=len(crow)),
             strata=dict(hos_crowded=summ(crow), hos_sparse=summ(spar),
                         linear_control=summ(ctrl)),
             mesh_split={}, per_episode={r['episode']: dict(
                 rms_lin=r['rms_lin'], rms_st2=r['rms_st2'], depth=r['depth'],
                 mesh=r.get('mesh'), group=r['group'], crowded=r['crowded']) for r in ok},
             prediction=dict(written_before_results='GAP-IS-SECOND-ORDER',
                             stated_probability=0.60),
             label=lab, label_msg=msg,
             disclosures=[
                 '⛔C1·C1c 판정 갱신 아님 — 이 배치는 그 이탈의 **정체**를 묻는다.',
                 '⛔배포 기본값 변경 아님 — use_stokes2 를 켜자는 제안이 아니다(별도 사전등록).',
                 '⛔정확도 주장 아님 · 추정기 런 0회 · 실측 적용 불가(성분표 없음).',
                 '★ST2 는 **적합이 아니라 계산**이다(모수 0). G-NOFIT 이 소스로 강제한다.',
                 '★대조가 부호까지 지정한다 — 선형 코퍼스에서는 ST2 가 **더 나빠야** 한다.',
                 '⚠stride=25 표본 · mesh_verdict 갈림을 행에 기록했다.',
             ])
    for r in ok:
        k = str(r.get('mesh')); v['mesh_split'][k] = v['mesh_split'].get(k, 0) + 1
    json.dump(v, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{lab}\n  {msg}\n  gates {v["gates_pass"]}')
    for n, o, d in gate_rows:
        print(f"   [{'OK ' if o else 'FAIL'}] {n:44s} {d}")
    for k, s in v['strata'].items():
        print(f"  {k:16s} n={s['n']:3d} 개선 {s['better']:3d}  LIN {s['lin_med']:.4f} → "
              f"ST2 {s['st2_med']:.4f}  비 {s['ratio_min']:.3f}/{s['ratio_med']:.3f}/{s['ratio_max']:.3f}")
    print('  mesh:', v['mesh_split'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
