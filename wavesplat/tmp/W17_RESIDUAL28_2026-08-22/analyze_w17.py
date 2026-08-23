"""W17 동결 분석기 — ⚠**결과 후에 쓰였다**(2026-08-23). 이 파일 자체가 결함의 증거다.

`RESULT.md` 는 *"게이트 전건 PASS · `verdict_w17.json`"* 이라 적었으나 그 파일에는
`gates` 도 `label` 도 없었다(키 = `base`/`arms`/`share` 뿐). RV-2 감사가 잡았다.
⟹ 라벨은 **동결 `gates_w17.label()` 을 저장된 자료에 그대로 돌려 재도출**한다.
코드는 손대지 않았고, 이 스크립트는 **읽고 부르기만** 한다.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_w17 as G

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
# ★차용 팔은 W9/W10 의 결과 파일에 산다 (gates_w17.BORROWED 가 그렇게 선언한다).
# 이 병합 없이는 G-2 가 0/16 로 FAIL 하고 비도 nan 이 된다 — RESULT.md 의
# "게이트 전건 PASS" 는 **병합을 전제한 문장**이었는데 그 전제가 어디에도 안 적혔다.
for _arm, _d in G.BORROWED.items():
    _f = os.path.abspath(os.path.join(HERE, _d, 'results.jsonl'))
    rows += [r for r in (json.loads(l) for l in open(_f)) if r.get('cell') == _arm]
gates = G.gates(rows)
ok = [r for r in rows if 'error' not in r]
res = {a: G.ratio_vs_ref(ok, a) for a in G.ARMS}
v = json.load(open(os.path.join(HERE, 'verdict_w17.json')))
base = G.ratio_vs_ref(ok, 'ORA-WPHI-SI-NOSM')
assert abs(base['ratio'] - v['base']) < 1e-12, (base['ratio'], v['base'])  # 손복사 금지: 원본 대조
lab = G.label(res, base['ratio'])
v['gates'] = [[g[0], bool(g[1]), g[2]] for g in gates]
v['gates_pass'] = sum(1 for g in gates if g[1]); v['gates_total'] = len(gates)
v['label'] = lab[0]; v['label_text'] = lab[1]
v['_amended'] = '2026-08-23: gates/label 을 동결 gates_w17 로 사후 재도출 (AMENDMENT_2.md)'
open(os.path.join(HERE, 'verdict_w17.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
for n_, p_, d_ in gates: print(f'  [{"PASS" if p_ else "FAIL"}] {n_}: {d_}')
print(f'\nLABEL {lab[0]} — {lab[1]}')
