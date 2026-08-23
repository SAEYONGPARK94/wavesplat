"""B-3 동결 분석기 — **결과 파일보다 먼저 쓰였다.** 라벨은 `gates_b3.label()` 이 낸다."""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b3 as G

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_b3.jsonl'))]
ok = [r for r in rows if 'error' not in r]
gates = G.gates(rows)
p = G.paired(ok, 'REFIT2', 'FROZEN')
lab = G.label(p)

# lead 별 분해 (부차, 사전 미선언 ⟹ 탐색적으로만 읽는다)
leads = sorted({float(k) for r in ok for k in (r.get('per_lead') or {})})
by_lead = {}
for l_ in leads:
    a = {G.key(r): (r.get('per_lead') or {}).get(str(l_)) for r in ok if r['cell'] == 'REFIT2'}
    b = {G.key(r): (r.get('per_lead') or {}).get(str(l_)) for r in ok if r['cell'] == 'FROZEN'}
    ks = [k for k in set(a) & set(b) if a[k] is not None and b[k] is not None]
    if ks:
        by_lead[l_] = {'n': len(ks), 'better': sum(1 for k in ks if a[k] < b[k]),
                       'rel_median_pct': st.median([(a[k]-b[k])/b[k]*100 for k in ks])}
wall = {a: st.median([r['wall_s'] for r in ok if r['cell'] == a]) for a in G.ARM_ORDER}
v = dict(batch='B-3', n_pairs=G.N_PAIRS,
         gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         paired=p, by_lead=by_lead, wall_median_s=wall,
         label=lab[0], label_text=lab[1])
open(os.path.join(HERE, 'verdict_b3.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n=== PRIMARY 짝 (REFIT2 < FROZEN, {G.FIELD}) ===")
print(f"  {p['better']}/{p['n']}  동점 {p['ties']}  상대중앙 {p['rel_median_pct']:+.2f}%  "
      f"p={p['p_one_sided']:.5f}   중앙 {p['median_arm']:.4f} vs {p['median_ref']:.4f}")
print('\n=== lead 별 (탐색적) ===')
for l_, d in by_lead.items():
    print(f"  lead {l_:5.1f} s: {d['better']:2d}/{d['n']}  {d['rel_median_pct']:+7.2f}%")
print(f"\n벽시계 중앙: " + ' · '.join(f'{a} {w:.1f} s' for a, w in wall.items()))
print(f"\n{lab[0]} — {lab[1]}")
