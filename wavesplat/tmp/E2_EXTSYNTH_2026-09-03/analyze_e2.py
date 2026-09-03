"""`E2-EXTSYNTH` 분석기 — ⛔**결과 전에 작성·동결**."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_e2 as G                                                 # noqa: E402
RES = os.path.join(HERE, 'results_e2.jsonl'); META = os.path.join(HERE, 'meta_e2.json')
OUT = os.path.join(HERE, 'verdict_e2.json')


def _med(a):
    a = sorted(x for x in a if x == x)
    return float('nan') if not a else (a[len(a)//2] if len(a) % 2 else 0.5*(a[len(a)//2-1]+a[len(a)//2]))


def main():
    rows = [json.loads(l) for l in open(RES, encoding='utf-8') if l.strip()]
    meta = json.load(open(META, encoding='utf-8'))
    ok = [r for r in rows if 'error' not in r]
    crow = [r for r in ok if r['crowded']]
    dep_w = sum(1 for r in crow if r['cnn_deploy'] < r['zero'])
    rungs = {}
    for m in G.LADDER:
        k = f'cnn_{m}'
        rungs[m] = dict(better=sum(1 for r in crow if r.get(k) is not None and r[k] < r['zero']),
                        n=sum(1 for r in crow if r.get(k) is not None),
                        med=_med([r[k] for r in crow if r.get(k) is not None]))
    passing = [m for m in G.LADDER if rungs[m]['better'] >= G.SELF_MIN]
    gate_rows = G.gates(rows, meta); blocked = G.blocked_by(gate_rows)
    lab, msg = (('BLOCKED', '차단: ' + ','.join(blocked)) if blocked
                else G.label(dep_w, bool(passing), passing[0] if passing else None))
    lab = 'E2-EXTSYNTH :: ' + lab
    v = dict(batch='E2-EXTSYNTH', n_rows=len(rows), grade='CONFIRMATORY',
             external=dict(repo='gitlab.com/fibe/wassfast', commit=G.EXT_COMMIT,
                           code_modified=False),
             gates=[dict(name=n, pass_=bool(o), detail=str(d)) for n, o, d in gate_rows],
             gates_pass=f'{sum(1 for _, o, _ in gate_rows if o)}/{len(gate_rows)}',
             blocked_by=blocked, threshold=G.SELF_MIN,
             deploy=dict(better_than_zero=dep_w, n=len(crow),
                         p_one_sided=G.sign_test_one_sided(dep_w, len(crow)) if crow else None,
                         med=_med([r['cnn_deploy'] for r in crow]),
                         med_zero=_med([r['zero'] for r in crow]),
                         med_n_obs=_med([r['n_obs_deploy'] for r in crow])),
             ladder=rungs, passing_rungs=passing,
             per_episode={r['episode']: {k: r[k] for k in r if k.startswith('cnn_')
                                         or k in ('zero', 'n_obs_deploy', 'dir_sign')}
                          for r in ok},
             prediction=dict(written_before_results='SYNTH-EXTERNAL-NEEDS-DENSITY',
                             stated_probability=0.65),
             label=lab, label_msg=msg,
             disclosures=[
                 '⛔SHIP 과의 우열 주장 아님 — 이 배치는 SHIP 을 돌리지 않는다.',
                 '⛔B-22·B-16·B-12 판정 갱신 아님.',
                 '⛔★사다리의 상위 칸은 **참값 표본**이다 — 그 밀도의 관측을 줄 수 있는 센서가 있다는 주장이 아니다.',
                 '⛔비인과성(저들 입력 t±1) · 최근접 되샘플 · 저들 학습 분포 미확인 — E1 에서 상속.',
                 '⚠나우캐스트(lead 0) · 단일 프레임 위치(창 중앙).',
             ])
    json.dump(v, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{lab}\n  {msg}\n  gates {v["gates_pass"]}')
    for n, o, d in gate_rows:
        print(f"   [{'OK ' if o else 'FAIL'}] {n:46s} {d}")
    d = v['deploy']
    print(f"  배포 밀도(저장 반환 {d['med_n_obs']:.0f}점/3프레임): {d['better_than_zero']}/{d['n']} "
          f"· CNN 중앙 {d['med']:.3f} vs ZERO 중앙 {d['med_zero']:.3f}")
    for m in G.LADDER:
        print(f"    사다리 {m:5d} 점/프레임(참값 표본) → {rungs[m]['better']}/{rungs[m]['n']} "
              f"중앙 {rungs[m]['med']:.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
