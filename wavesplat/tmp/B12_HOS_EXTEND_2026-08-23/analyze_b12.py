"""B-12 동결 분석기 — ★**결과 파일보다 먼저 쓰였다.**"""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b12 as G

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_b12.jsonl'))]
ok = [r for r in rows if 'error' not in r]
ship = G.med_by_episode(ok, 'SHIP-NOW')
spec = G.med_by_episode(ok, 'SPEC25')
zero = G.med_by_episode(ok, 'SPEC25', field='zero_nRMSE')
gates = G.gates(rows, ship, spec, zero)
prim = G.paired(ship, spec)
neg = G.paired(spec, zero)
sz = G.paired(ship, zero)
# ★신규 시드만 (기존 16개가 결과를 끌고 가는지)
def sub(d, seeds):
    return {k: v for k, v in d.items() if int(k.split('seed')[1].split('_')[0]) in seeds}
new_only = G.paired(sub(ship, set(G.NEW_SEEDS)), sub(spec, set(G.NEW_SEEDS)))
old_only = G.paired(sub(ship, set(G.OLD_SEEDS)), sub(spec, set(G.OLD_SEEDS)))
lab = G.label(prim, neg, new_only)
v = dict(batch='B-12', n_episodes=len(ship), family_alpha=G.FAMILY_ALPHA,
         thresholds={'nominal': G.NOMINAL, 'confirm': G.CONFIRM, 'weak': G.WEAK},
         gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         ship_vs_spec=prim, spec_vs_zero=neg, ship_vs_zero=sz,
         new_seeds_only=new_only, old_seeds_only=old_only,
         medians={k: (st.median(d.values()) if d else None)
                  for k, d in (('SHIP-NOW', ship), ('SPEC25', spec), ('ZERO', zero))},
         label=lab[0], label_text=lab[1])
open(os.path.join(HERE, 'verdict_b12.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n=== 혼잡 HOS {len(ship)} 에피소드 (median-of-3) ===")
print(f"  중앙: SHIP {v['medians']['SHIP-NOW']:.4f} · SPEC25 {v['medians']['SPEC25']:.4f} "
      f"· ZERO {v['medians']['ZERO']:.4f}")
print(f"  SHIP < SPEC25 : {prim['better']}/{prim['n']}  {prim['rel_median_pct']:+.2f}%  p={prim['p_one_sided']:.6f}")
print(f"  SHIP < ZERO   : {sz['better']}/{sz['n']}")
print(f"  ★SPEC25 < ZERO(음성대조): {neg['better']}/{neg['n']}  (문턱 {G.NEG_MIN})")
print(f"\n  신규 시드만: {new_only['better']}/{new_only['n']}  {new_only['rel_median_pct']:+.2f}%")
print(f"  기존 시드만: {old_only['better']}/{old_only['n']}  {old_only['rel_median_pct']:+.2f}%  (B-11 = 14/16)")
print(f"\n{lab[0]} — {lab[1]}")
