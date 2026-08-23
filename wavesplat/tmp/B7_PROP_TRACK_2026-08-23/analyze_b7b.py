"""B-7b 동결 분석기 — ★**결과 파일보다 먼저 쓰였다.**"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b7b as G
sys.path.insert(0, G.B2)
from zero_ref import zero_ref_perlead

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_b7b.jsonl'))]
ok = [r for r in rows if 'error' not in r]
zero = {f'{rec}|{w}': {str(k): v for k, v in zero_ref_perlead(rec, w).items()}
        for rec in G.RECORDS for w in G.WINDOWS}
gates = G.gates(rows)
p = G.paired(ok, 'SHIP-B7B', 'PROP-B7B')
prop_near = G.beats_zero(ok, 'PROP-B7B', zero, *G.LEAD_NEAR)
prop_far = G.beats_zero(ok, 'PROP-B7B', zero, *G.LEAD_FAR)
ship_near = G.beats_zero(ok, 'SHIP-B7B', zero, *G.LEAD_NEAR)
ship_far = G.beats_zero(ok, 'SHIP-B7B', zero, *G.LEAD_FAR)
# ★재현성 대조: B-3 의 FROZEN 을 손복사하지 않고 원본에서 재계산한다
b3 = [r for r in (json.loads(l) for l in open(os.path.join(G.B3, 'results_b3.jsonl')))
      if r.get('cell') == 'FROZEN' and 'error' not in r]
b3_far = G.beats_zero([dict(r, cell='FROZEN') for r in b3], 'FROZEN', zero, *G.LEAD_FAR)
assert b3_far['better'] == G.FROZEN_REF_FAR, (b3_far['better'], G.FROZEN_REF_FAR)
repro_ok = abs(ship_far['better'] - b3_far['better']) <= 2
lab = G.label(p, ship_far, repro_ok)
v = dict(batch='B-7b', gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         family_alpha=G.FAMILY_ALPHA, ship_vs_prop=p,
         prop_near=prop_near, prop_far=prop_far, ship_near=ship_near, ship_far=ship_far,
         b3_frozen_far=b3_far, reproduces_b3=repro_ok,
         label=lab[0], label_text=lab[1])
open(os.path.join(HERE, 'verdict_b7b.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n=== PRIMARY 짝 (SHIP < PROP, fc_mean) ===")
print(f"  {p['better']}/{p['n']}  {p['rel_median_pct']:+.2f}%  p={p['p_one_sided']:.6f}")
print(f"  (참고: B-7 의 오염된 값 33/36 · 교차배치 대체값 23/36)")
print(f"\n=== 참 제로필드 대비 ===")
for nm, d in (('PROP lead 1–3', prop_near), ('PROP lead 4–14', prop_far),
              ('SHIP lead 1–3', ship_near), ('SHIP lead 4–14', ship_far)):
    print(f"  {nm:16s} {d['better']:2d}/{d['n']:<3d} p={d['p_one_sided']:.6f}")
print(f"\n재현성: 이 배치 SHIP lead 4–14 = {ship_far['better']}/36 vs "
      f"B-3 FROZEN {b3_far['better']}/36 ⟹ {'일치' if repro_ok else '⛔불일치'}")
print(f"\n{lab[0]} — {lab[1]}")
