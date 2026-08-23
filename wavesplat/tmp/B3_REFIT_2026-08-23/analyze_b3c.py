"""B-3c 동결 분석기 — **결과 파일보다 먼저 쓰였다.**"""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b3c as G
import gates_b3 as B3

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_b3c.jsonl'))]
ok = [r for r in rows if 'error' not in r]
gates = G.gates(rows)
p = G.paired(ok, 'REFIT2', 'FROZEN')
# ★B-3 의 크기를 손복사하지 않고 원본에서 재계산해 대조한다
b3 = [r for r in (json.loads(l) for l in open(os.path.join(HERE, 'results_b3.jsonl')))
      if 'error' not in r]
b3p = B3.paired(b3, 'REFIT2', 'FROZEN')
assert abs(b3p['rel_median_pct'] - G.B3_REL_MEDIAN) < 0.01, (b3p['rel_median_pct'], G.B3_REL_MEDIAN)
lab = G.label_c(p, b3p['rel_median_pct'])
leads = sorted({float(k) for r in ok for k in (r.get('per_lead') or {})})
by_lead = {}
for l_ in leads:
    a = {G.key(r): (r.get('per_lead') or {}).get(str(l_)) for r in ok if r['cell'] == 'REFIT2'}
    b = {G.key(r): (r.get('per_lead') or {}).get(str(l_)) for r in ok if r['cell'] == 'FROZEN'}
    ks = [k for k in set(a) & set(b) if a[k] is not None and b[k] is not None]
    if ks:
        by_lead[l_] = {'n': len(ks), 'better': sum(1 for k in ks if a[k] < b[k]),
                       'rel_median_pct': st.median([(a[k]-b[k])/b[k]*100 for k in ks])}
v = dict(batch='B-3c', windows=G.WINDOWS,
         gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         paired=p, b3_paired=b3p, by_lead=by_lead, label=lab[0], label_text=lab[1])
open(os.path.join(HERE, 'verdict_b3c.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n=== PRIMARY (창 {G.WINDOWS}, 신선) ===")
print(f"  {p['better']}/{p['n']}  상대중앙 {p['rel_median_pct']:+.2f}%  p={p['p_one_sided']:.7f}")
print(f"  (B-3, 창 [1,3,5]:  {b3p['better']}/{b3p['n']}  {b3p['rel_median_pct']:+.2f}%)")
print('\n=== lead 별 ===')
for l_, d in by_lead.items():
    print(f"  lead {l_:5.1f} s: {d['better']:2d}/{d['n']}  {d['rel_median_pct']:+7.2f}%")
print(f"\n{lab[0]} — {lab[1]}")
