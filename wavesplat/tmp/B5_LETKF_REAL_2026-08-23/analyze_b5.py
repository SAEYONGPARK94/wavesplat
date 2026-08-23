"""B-5 동결 분석기 — **결과 파일보다 먼저 쓰였다.** `G-8` 자기검정이 우선한다."""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b5 as G
sys.path.insert(0, G.B2)
from zero_ref import zero_ref_perlead     # ★상수 금지: 자료에서 산출한 기준

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_b5.jsonl'))]
rows += [r for r in (json.loads(l) for l in open(os.path.join(G.B3, 'results_b3.jsonl')))
         if r.get('cell') in G.REUSED]
ok = [r for r in rows if 'error' not in r]
z0 = {rec: st.median(zero_ref_perlead(rec, w)[0.0] for w in G.WINDOWS) for rec in G.RECORDS}
gates = G.gates(rows, z0)
st_w, st_n = G.selftest(rows, z0)
passed = st_w >= G.SELFTEST_MIN

pool = ok + G.best_da(rows)
p1 = G.paired(pool, 'REFIT2', 'LETKF-BEST-DA')
p2 = G.paired(pool, 'FROZEN', 'LETKF-FC50')
lab = G.label(passed, p1, p2, st_w, st_n)

leads = sorted({float(k) for r in pool for k in (r.get('per_lead') or {})})
by_lead = {}
for l_ in leads:
    d = {}
    for a in ('REFIT2', 'FROZEN', 'LETKF-BEST-DA', 'LETKF-FC50'):
        v = {G.key(r): (r.get('per_lead') or {}).get(str(l_)) for r in pool if r['cell'] == a}
        d[a] = {k: x for k, x in v.items() if x is not None}
    if d['LETKF-BEST-DA']:
        by_lead[l_] = {a: (st.median(d[a].values()) if d[a] else None) for a in d}
wall = {a: st.median([r['wall_s'] for r in pool if r['cell'] == a])
        for a in ('FROZEN', 'REFIT2') + tuple(G.ARM_ORDER) if any(r['cell'] == a for r in pool)}
v = dict(batch='B-5', gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         selftest=[st_w, st_n, G.SELFTEST_MIN], selftest_passed=passed,
         zero_ref_lead0={k: round(x, 4) for k, x in z0.items()},
         primary=p1 if passed else None, secondary=p2 if passed else None,
         by_lead=by_lead, wall_median_s=wall, label=lab[0], label_text=lab[1])
open(os.path.join(HERE, 'verdict_b5.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n★G-8 자기검정: {st_w}/{st_n} 레코드 (문턱 {G.SELFTEST_MIN}) ⟹ "
      f"{'통과 — 비교를 보고한다' if passed else '실패 — ⛔비교를 보고하지 않는다'}")
if passed:
    print(f"\n=== PRIMARY ===\n  REFIT2 < LETKF-BEST-DA : {p1['better']}/{p1['n']}  "
          f"{p1['rel_median_pct']:+.2f}%  p={p1['p_one_sided']:.6f}")
    print(f"  (부차) FROZEN < LETKF-FC50 : {p2['better']}/{p2['n']}  "
          f"{p2['rel_median_pct']:+.2f}%  p={p2['p_one_sided']:.6f}")
    print(f"\n=== lead 별 중앙 ===\n  {'lead':>5s} {'REFIT2':>9s} {'FROZEN':>9s} {'LETKF-DA':>9s} {'LETKF-FC':>9s}")
    for l_, d in by_lead.items():
        f_ = lambda x: f'{x:9.4f}' if x is not None else '        -'
        print(f"  {l_:5.1f} {f_(d['REFIT2'])} {f_(d['FROZEN'])} {f_(d['LETKF-BEST-DA'])} {f_(d['LETKF-FC50'])}")
print(f"\n벽시계 중앙: " + ' · '.join(f'{a} {w:.1f}s' for a, w in wall.items()))
print(f"\n{lab[0]} — {lab[1]}")
