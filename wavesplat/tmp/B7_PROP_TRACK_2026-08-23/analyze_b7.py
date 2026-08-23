"""B-7 동결 분석기 — ★**결과 파일보다 먼저 쓰였다.**"""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b7 as G
sys.path.insert(0, G.B2)
from zero_ref import zero_ref_perlead

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_b7.jsonl'))]
rows += [r for r in (json.loads(l) for l in open(os.path.join(G.B3, 'results_b3.jsonl')))
         if r.get('cell') in G.REUSED]
ok = [r for r in rows if 'error' not in r]
zero = {f'{rec}|{w}': {str(k): v for k, v in zero_ref_perlead(rec, w).items()}
        for rec in G.RECORDS for w in G.WINDOWS}
gates = G.gates(rows)
prim = G.beats_zero(ok, 'PROP-TRACK', zero, *G.LEAD_NEAR)
sec_a = G.beats_zero(ok, 'PROP-TRACK', zero, *G.LEAD_FAR)
ship_near = G.beats_zero(ok, 'SHIP-TRACK', zero, *G.LEAD_NEAR)
ship_far = G.beats_zero(ok, 'SHIP-TRACK', zero, *G.LEAD_FAR)
sec_b = G.paired(ok, 'PROP-TRACK', 'SHIP-TRACK')
ts_ok = all((r.get('track_steps') or 0) > 0 for r in ok if r['cell'] == 'PROP-TRACK')
lab = G.label(prim, prim['n'], sec_a, sec_b, ts_ok)
v = dict(batch='B-7', gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         family_alpha=G.FAMILY_ALPHA,
         prop_near=prim, prop_far=sec_a, ship_near=ship_near, ship_far=ship_far,
         prop_vs_ship=sec_b, track_steps_ok=ts_ok, label=lab[0], label_text=lab[1])
open(os.path.join(HERE, 'verdict_b7.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n=== 참 제로필드 대비 (문턱 24 명목 / 26 가족단위 α={G.FAMILY_ALPHA:.5f}) ===")
for nm, d in (('PROP lead 1–3', prim), ('PROP lead 4–14', sec_a),
              ('SHIP lead 1–3', ship_near), ('SHIP lead 4–14', ship_far)):
    print(f"  {nm:16s} {d['better']:2d}/{d['n']:<3d} p={d['p_one_sided']:.6f}")
print(f"\n=== 짝: PROP vs SHIP (fc_mean) ===")
print(f"  {sec_b['better']}/{sec_b['n']}  상대중앙 {sec_b['rel_median_pct']:+.2f}%  "
      f"p={sec_b['p_one_sided']:.6f}   중앙 {sec_b['median_arm']:.4f} vs {sec_b['median_ref']:.4f}")
print(f"\n{lab[0]} — {lab[1]}")
