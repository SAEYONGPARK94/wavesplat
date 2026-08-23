"""B-4 동결 분석기 — **결과 파일보다 먼저 쓰였다.**"""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b4 as G

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_b4.jsonl'))]
rows += [r for r in (json.loads(l) for l in open(os.path.join(HERE, 'results_b3.jsonl')))
         if r.get('cell') in G.REUSED]
ok = [r for r in rows if 'error' not in r]
gates = G.gates(rows)
pf = G.paired(ok, 'FROZEN', 'PERSIST')     # FROZEN 이 더 작으면(=나으면) better
pr = G.paired(ok, 'REFIT2', 'PERSIST')
lab = G.label(pf, pr)
leads = sorted({float(k) for r in ok for k in (r.get('per_lead') or {})})
by_lead, cross = {}, None
for l_ in leads:
    d = {}
    for a in ('FROZEN', 'REFIT2', 'PERSIST'):
        v = {G.key(r): (r.get('per_lead') or {}).get(str(l_)) for r in ok if r['cell'] == a}
        d[a] = {k: x for k, x in v.items() if x is not None}
    ks = sorted(set(d['FROZEN']) & set(d['PERSIST']))
    if not ks: continue
    by_lead[l_] = {'n': len(ks),
                   'frozen_better': sum(1 for k in ks if d['FROZEN'][k] < d['PERSIST'][k]),
                   'refit_better': sum(1 for k in ks if k in d['REFIT2'] and d['REFIT2'][k] < d['PERSIST'][k]),
                   'med_frozen': st.median(d['FROZEN'][k] for k in ks),
                   'med_persist': st.median(d['PERSIST'][k] for k in ks),
                   'med_refit': st.median(d['REFIT2'][k] for k in ks if k in d['REFIT2'])}
    if cross is None and l_ > 0 and by_lead[l_]['frozen_better'] > len(ks) / 2:
        cross = l_
v = dict(batch='B-4', gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         frozen_vs_persist=pf, refit_vs_persist=pr, by_lead=by_lead,
         crossover_lead_s=cross, label=lab[0], label_text=lab[1])
open(os.path.join(HERE, 'verdict_b4.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n=== PRIMARY ===")
print(f"  FROZEN > PERSIST : {pf['better']}/{pf['n']}  {pf['rel_median_pct']:+.2f}%  p={pf['p_one_sided']:.6f}")
print(f"  REFIT2 > PERSIST : {pr['better']}/{pr['n']}  {pr['rel_median_pct']:+.2f}%  p={pr['p_one_sided']:.6f}")
print(f"\n=== lead 별 중앙 (σ 정규화) — 교차점 = {cross} s ===")
print(f"  {'lead':>5s} {'PERSIST':>8s} {'FROZEN':>8s} {'REFIT2':>8s}   {'FRO>PER':>8s} {'REF>PER':>8s}")
for l_, d in by_lead.items():
    print(f"  {l_:5.1f} {d['med_persist']:8.4f} {d['med_frozen']:8.4f} {d['med_refit']:8.4f}   "
          f"{d['frozen_better']:3d}/{d['n']:<4d} {d['refit_better']:3d}/{d['n']:<4d}")
print(f"\n{lab[0]} — {lab[1]}")
