"""B-11 동결 분석기 — ★**결과 파일보다 먼저 쓰였다.**"""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b11 as G

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_b11.jsonl'))]
ok = [r for r in rows if 'error' not in r]
ship = G.med_by_episode(ok, 'SHIP-NOW')
spec = G.med_by_episode(ok, 'SPEC25')
zero = G.med_by_episode(ok, 'SPEC25', field='zero_nRMSE')   # 무기술 기준은 기준선 행에서
gates = G.gates(rows, ship, spec, zero)
prim = G.paired(ship, spec)
neg = G.paired(spec, zero)                                   # 음성 대조: 기준선이 무기술을 이기나
ship_vs_zero = G.paired(ship, zero)
lab = G.label(prim, neg)
v = dict(batch='B-11', n_episodes=len(ship), family_alpha=G.FAMILY_ALPHA,
         gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         ship_vs_spec=prim, spec_vs_zero=neg, ship_vs_zero=ship_vs_zero,
         medians={'SHIP-NOW': st.median(ship.values()) if ship else None,
                  'SPEC25': st.median(spec.values()) if spec else None,
                  'ZERO': st.median(zero.values()) if zero else None},
         label=lab[0], label_text=lab[1])
open(os.path.join(HERE, 'verdict_b11.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n=== 혼잡 HOS 16 에피소드 (median-of-3) ===")
print(f"  중앙 nRMSE: SHIP {v['medians']['SHIP-NOW']:.4f} · SPEC25 {v['medians']['SPEC25']:.4f} "
      f"· ZERO {v['medians']['ZERO']:.4f}")
print(f"  SHIP < SPEC25 : {prim['better']}/{prim['n']}  {prim['rel_median_pct']:+.2f}%  p={prim['p_one_sided']:.5f}")
print(f"  SHIP < ZERO   : {ship_vs_zero['better']}/{ship_vs_zero['n']}  {ship_vs_zero['rel_median_pct']:+.2f}%")
print(f"  ★SPEC25 < ZERO(음성대조): {neg['better']}/{neg['n']}  (문턱 {G.NEG_MIN})")
print(f"\n{lab[0]} — {lab[1]}")
