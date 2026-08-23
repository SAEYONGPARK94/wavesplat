"""B-8b 동결 분석기 — ★**결과 파일보다 먼저 쓰였다.**"""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b8b as G
sys.path.insert(0, G.B2)
from zero_ref import zero_ref_perlead

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_b8b.jsonl'))]
rows += [r for r in (json.loads(l) for l in open(os.path.join(HERE, 'results_b8.jsonl')))
         if r.get('cell') == 'V2K-TRACK']
rows += [r for r in (json.loads(l) for l in open(os.path.join(G.B7, 'results_b7.jsonl')))
         if r.get('cell') == 'PROP-TRACK']
ok = [r for r in rows if 'error' not in r]
zero = {f'{rec}|{w}': {str(k): v for k, v in zero_ref_perlead(rec, w).items()}
        for rec in G.RECORDS for w in G.ALL_WINDOWS}
gates = G.gates(rows)


def rec_level(arm, lo, hi):
    """레코드 안 창 6개의 '이겼나'를 중앙으로 접는다 — 창 비독립성에 보수적."""
    per = {}
    for r in ok:
        if r['cell'] != arm: continue
        pl = r.get('per_lead') or {}
        z = zero[f"{r['record']}|{r['win']}"]
        a = [v for k, v in pl.items() if lo <= float(k) <= hi]
        b = [z[k] for k in pl if lo <= float(k) <= hi]
        if a: per.setdefault(r['record'], []).append(st.mean(a) < st.mean(b))
    w = sum(1 for v in per.values() if sum(v) > len(v) / 2)
    return {'arm': arm, 'n': len(per), 'better': w,
            'p_one_sided': G.sign_test_one_sided(w, len(per)) if per else None}


v2k_w = G.beats_zero(ok, 'V2K-TRACK', zero, *G.LEAD_NEAR)
v2k_f = G.beats_zero(ok, 'V2K-TRACK', zero, *G.LEAD_FAR)
prop_w = G.beats_zero(ok, 'PROP-TRACK', zero, *G.LEAD_NEAR)
prop_f = G.beats_zero(ok, 'PROP-TRACK', zero, *G.LEAD_FAR)
v2k_r = rec_level('V2K-TRACK', *G.LEAD_NEAR)
prop_r = rec_level('PROP-TRACK', *G.LEAD_NEAR)
pair = G.paired(ok, 'V2K-TRACK', 'PROP-TRACK')
ts_ok = all((r.get('track_steps') or 0) > 0 for r in ok)
lab = G.label(v2k_w, v2k_r, prop_w, ts_ok)
v = dict(batch='B-8b', gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         thresholds={'window': [G.NOSKILL, G.NOMINAL, G.CONFIRM],
                     'record': [G.REC_NOSKILL, G.REC_NOMINAL, G.REC_CONFIRM]},
         family_alpha=G.FAMILY_ALPHA,
         v2k_window_near=v2k_w, v2k_window_far=v2k_f, v2k_record_near=v2k_r,
         prop_window_near=prop_w, prop_window_far=prop_f, prop_record_near=prop_r,
         v2k_vs_prop=pair, label=lab[0], label_text=lab[1])
open(os.path.join(HERE, 'verdict_b8b.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n=== 통합 창 6개 (문턱: 무기술 ≤{G.NOSKILL} · 명목 ≥{G.NOMINAL} · 확증 ≥{G.CONFIRM} /72) ===")
for nm, d in (('V2K  lead 1–3', v2k_w), ('V2K  lead 4–14', v2k_f),
              ('PROP lead 1–3', prop_w), ('PROP lead 4–14', prop_f)):
    print(f"  {nm:16s} {d['better']:2d}/{d['n']:<3d} p={d['p_one_sided']:.7f}")
print(f"\n=== 레코드 수준 (문턱: 무기술 ≤{G.REC_NOSKILL} · 명목 ≥{G.REC_NOMINAL} /12) ===")
for nm, d in (('V2K  lead 1–3', v2k_r), ('PROP lead 1–3', prop_r)):
    print(f"  {nm:16s} {d['better']:2d}/{d['n']:<3d} p={d['p_one_sided']:.6f}")
print(f"\n짝 V2K < PROP: {pair['better']}/{pair['n']}  {pair['rel_median_pct']:+.2f}%  p={pair['p_one_sided']:.6f}")
print(f"\n{lab[0]} — {lab[1]}")
