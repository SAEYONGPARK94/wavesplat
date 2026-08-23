"""B-8 동결 분석기 — ★**결과 파일보다 먼저 쓰였다.**"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b8 as G
sys.path.insert(0, G.B2)
from zero_ref import zero_ref_perlead

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_b8.jsonl'))]
rows += [r for r in (json.loads(l) for l in open(os.path.join(G.B7, 'results_b7.jsonl')))
         if r.get('cell') in G.REUSED]
ok = [r for r in rows if 'error' not in r]
zero = {f'{rec}|{w}': {str(k): v for k, v in zero_ref_perlead(rec, w).items()}
        for rec in G.RECORDS for w in G.WINDOWS}
gates = G.gates(rows)
v2k_near = G.beats_zero(ok, 'V2K-TRACK', zero, *G.LEAD_NEAR)
v2k_far = G.beats_zero(ok, 'V2K-TRACK', zero, *G.LEAD_FAR)
prop_near = G.beats_zero(ok, 'PROP-TRACK', zero, *G.LEAD_NEAR)
prop_far = G.beats_zero(ok, 'PROP-TRACK', zero, *G.LEAD_FAR)
ship_near = G.beats_zero(ok, 'SHIP-TRACK', zero, *G.LEAD_NEAR)
# ★손복사 금지: B-7 의 기준선을 원본 verdict 에서 대조한다
vb7 = json.load(open(os.path.join(G.B7, 'verdict_b7.json')))
assert prop_near['better'] == vb7['prop_near']['better'] == G.PROP_BASELINE, (
    prop_near['better'], vb7['prop_near']['better'], G.PROP_BASELINE)
pair_vp = G.paired(ok, 'V2K-TRACK', 'PROP-TRACK')
pair_vs = G.paired(ok, 'V2K-TRACK', 'SHIP-TRACK')
ts_ok = all((r.get('track_steps') or 0) > 0 for r in ok if r['cell'] == 'V2K-TRACK')
lab = G.label(v2k_near, prop_near, pair_vp, ts_ok)
v = dict(batch='B-8', gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         family_alpha=G.FAMILY_ALPHA,
         v2k_near=v2k_near, v2k_far=v2k_far, prop_near=prop_near, prop_far=prop_far,
         ship_near=ship_near, v2k_vs_prop=pair_vp, v2k_vs_ship=pair_vs,
         label=lab[0], label_text=lab[1])
open(os.path.join(HERE, 'verdict_b8.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n=== 참 제로필드 대비 (문턱 24/26, 가족단위 α={G.FAMILY_ALPHA:.5f}) ===")
for nm, d in (('V2K  lead 1–3', v2k_near), ('V2K  lead 4–14', v2k_far),
              ('PROP lead 1–3', prop_near), ('PROP lead 4–14', prop_far),
              ('SHIP lead 1–3', ship_near)):
    print(f"  {nm:16s} {d['better']:2d}/{d['n']:<3d} p={d['p_one_sided']:.6f}")
print('\n=== 짝 ===')
for nm, d in (('V2K < PROP', pair_vp), ('V2K < SHIP', pair_vs)):
    print(f"  {nm:12s} {d['better']:2d}/{d['n']:<3d} {d['rel_median_pct']:+7.2f}%  p={d['p_one_sided']:.6f}")
print(f"\n{lab[0]} — {lab[1]}")
