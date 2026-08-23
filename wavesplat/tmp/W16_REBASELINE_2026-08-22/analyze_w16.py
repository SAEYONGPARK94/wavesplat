"""w16 동결 분석기 — ⚠**결과 후에 쓰였다**(2026-08-23, RV-2 절차 감사 대응).

동결 `gates_w16` 의 게이트·라벨 함수를 **저장된 자료에 그대로** 돌려 `verdict_w16.json` 을 만든다.
코드는 손대지 않는다. 판정이 손이 아니라 **코드에서** 나왔다는 증거를 남기는 것이 목적이다.
"""
import json, os, sys, inspect
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_w16 as G

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
ok = [r for r in rows if 'error' not in r]
gates = G.gates(rows)
out = {'batch': 'w16'.upper(), 'n_rows': len(rows), 'n_ok': len(ok),
       'gates': [[g[0], bool(g[1]), str(g[2])] for g in gates],
       'gates_pass': sum(1 for g in gates if g[1]), 'gates_total': len(gates),
       '_amended': '2026-08-23: 동결 gates 로 사후 재도출 (분석기 부재 결함 수리)'}
pc, ps = G.primary(ok,'crowded'), G.primary(ok,'sparse')
res = {'crowded': pc, 'sparse': ps, 'all': G.primary(ok)}
lab = G.label(pc, ps)
out['results'] = res
out['label'] = lab[0] if isinstance(lab,(tuple,list)) else str(lab)
out['label_text'] = lab[1] if isinstance(lab,(tuple,list)) and len(lab)>1 else ''
for n_, p_, d_ in gates: print(f'  [{"PASS" if p_ else "FAIL"}] {n_}: {d_}')
open(os.path.join(HERE, 'verdict_w16.json'), 'w').write(json.dumps(out, indent=1, ensure_ascii=False))
print(f'\nLABEL {out["label"]} — {out["label_text"]}')
