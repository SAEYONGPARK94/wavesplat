"""B-10 동결 분석기 — ★**결과 파일보다 먼저 쓰였다.**"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b10 as G
sys.path.insert(0, G.B2)
from zero_ref import zero_ref_perlead

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_b10.jsonl'))]
rows += [r for r in (json.loads(l) for l in open(os.path.join(G.B7, 'results_b7.jsonl')))
         if r.get('cell') == 'PROP-TRACK']
rows += [r for r in (json.loads(l) for l in open(os.path.join(G.B8, 'results_b8.jsonl')))
         if r.get('cell') == 'V2K-TRACK']
ok = [r for r in rows if 'error' not in r and r.get('win') in G.WINDOWS]
zero = {f'{rec}|{w}': {str(k): v for k, v in zero_ref_perlead(rec, w).items()}
        for rec in G.RECORDS for w in G.WINDOWS}
gates = G.gates(rows)
v2 = G.beats_zero(ok, 'V2-TRACK', zero, *G.LEAD_NEAR)
v2_far = G.beats_zero(ok, 'V2-TRACK', zero, *G.LEAD_FAR)
v2k = G.beats_zero(ok, 'V2K-TRACK', zero, *G.LEAD_NEAR)
prop = G.beats_zero(ok, 'PROP-TRACK', zero, *G.LEAD_NEAR)
pair_vk = G.paired(ok, 'V2-TRACK', 'V2K-TRACK')
pair_vp = G.paired(ok, 'V2-TRACK', 'PROP-TRACK')
ts_ok = all((r.get('track_steps') or 0) > 0 for r in ok if r['cell'] == 'V2-TRACK')
lab = G.label(v2, v2k, prop, pair_vk, pair_vp, ts_ok)
import statistics as st
wall = {a: round(st.median([r['wall_s'] for r in ok if r['cell'] == a]), 1)
        for a in ('V2-TRACK', 'V2K-TRACK', 'PROP-TRACK') if any(r['cell'] == a for r in ok)}
v = dict(batch='B-10', gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         family_alpha=G.FAMILY_ALPHA, thresholds={'nominal': G.NOMINAL, 'noskill': G.NOSKILL},
         v2_near=v2, v2_far=v2_far, v2k_near=v2k, prop_near=prop,
         v2_vs_v2k=pair_vk, v2_vs_prop=pair_vp, wall_median_s=wall,
         label=lab[0], label_text=lab[1])
open(os.path.join(HERE, 'verdict_b10.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n=== 참 제로필드 대비 (창 {G.WINDOWS}, n=36; 명목 ≥{G.NOMINAL} · 무기술 ≤{G.NOSKILL}) ===")
for nm, d in (('V2   lead 1–3', v2), ('V2   lead 4–14', v2_far),
              ('V2K  lead 1–3', v2k), ('PROP lead 1–3', prop)):
    print(f"  {nm:16s} {d['better']:2d}/{d['n']:<3d} p={d['p_one_sided']:.6f}")
print('\n=== 짝 ===')
for nm, d in (('V2 < V2K', pair_vk), ('V2 < PROP', pair_vp)):
    print(f"  {nm:12s} {d['better']:2d}/{d['n']:<3d} {d['rel_median_pct']:+7.2f}%  p={d['p_one_sided']:.6f}")
print(f"\n벽시계 중앙(s): {wall}")
print(f"\n{lab[0]} — {lab[1]}")
