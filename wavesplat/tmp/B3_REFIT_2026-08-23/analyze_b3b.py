"""B-3b 동결 분석기 — **결과 파일보다 먼저 쓰였다.** 라벨은 `gates_b3b.label()` 이 낸다."""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b3b as G

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_b3b.jsonl'))]
rows += [r for r in (json.loads(l) for l in open(os.path.join(HERE, 'results_b3.jsonl')))
         if r.get('cell') in G.REUSED]          # ★재사용 팔은 B-3 파일에서 (gates_b3b.REUSED 선언)
ok = [r for r in rows if 'error' not in r]
gates = G.gates(rows)
order = ['REFIT1', 'REFIT2', 'REFIT5', 'REFIT10']
res = {a: G.paired(ok, a, G.REF) for a in order}
lab = G.label(res)
wall = {a: st.median([r['wall_s'] for r in ok if r['cell'] == a])
        for a in ['FROZEN'] + order if any(r['cell'] == a for r in ok)}
calls = {a: sorted({r.get('refit_calls') for r in ok if r['cell'] == a}) for a in ['FROZEN'] + order}
v = dict(batch='B-3b', gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         paired_vs_frozen=res, wall_median_s=wall, refit_calls=calls,
         label=lab[0], label_text=lab[1])
open(os.path.join(HERE, 'verdict_b3b.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n=== 용량-반응 (vs FROZEN, {G.FIELD}) ===")
print(f"  {'팔':9s} {'개선':>7s} {'상대중앙':>9s} {'p':>9s} {'재적합':>10s} {'벽시계':>7s}")
for a in order:
    r = res[a]
    print(f"  {a:9s} {r['better']:2d}/{r['n']:<3d} {r['rel_median_pct']:+8.2f}% "
          f"{r['p_one_sided']:9.6f} {str(calls[a]):>10s} {wall.get(a,float('nan')):6.1f}s")
print(f"  {'FROZEN':9s} {'—':>7s} {'—':>9s} {'—':>9s} {str(calls['FROZEN']):>10s} {wall.get('FROZEN',float('nan')):6.1f}s")
print(f"\n{lab[0]} — {lab[1]}")
