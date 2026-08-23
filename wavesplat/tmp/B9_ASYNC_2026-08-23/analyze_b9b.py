"""B-9 동결 분석기 — ★**결과 파일보다 먼저 쓰였다.**"""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b9b as G

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_b9b.jsonl'))]
rows += [r for r in (json.loads(l) for l in open(os.path.join(G.B3, 'results_b3.jsonl')))
         if r.get('cell') in G.REUSED]
ok = [r for r in rows if 'error' not in r]
gates = G.gates(rows)
# 레코드 단위로 접어 "최악 프레임이 dt 안" 을 센다 (창 3개 중 최악을 취한다 = 보수적)
per = {}
for r in ok:
    if r['cell'] != 'ASYNC5': continue
    per.setdefault(r['record'], []).append((r['worst_update_s'], r['dt_frame']))
n_ok = sum(1 for v in per.values() if max(w for w, _ in v) < v[0][1])
sy = {}
for r in ok:
    if r['cell'] != 'SYNC5': continue
    sy.setdefault(r['record'], []).append((r['worst_update_s'], r['dt_frame']))
neg_ok = bool(sy) and all(max(w for w, _ in v) > v[0][1] for v in sy.values())
acc = G.paired(ok, 'ASYNC5', 'FROZEN')
pair_as = G.paired(ok, 'ASYNC5', 'SYNC5')
lab = G.label(n_ok, len(per), acc, pair_as, neg_ok)
a = [r for r in ok if r['cell'] == 'ASYNC5']
v = dict(batch='B-9b', gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         n_records_frame_ok=n_ok, n_records=len(per), negative_control_ok=neg_ok,
         async_vs_frozen=acc, async_vs_sync=pair_as,
         worst_update_ms={'ASYNC5': round(st.median([r['worst_update_s'] for r in a]) * 1e3, 3),
                          'SYNC5': round(st.median([r['worst_update_s'] for r in ok
                                                    if r['cell'] == 'SYNC5']) * 1e3, 3)},
         kicks=sorted({r['async_kicks'] for r in a}),
         applied=sorted({r['async_applied'] for r in a}),
         skipped=sorted({r['async_skipped'] for r in a}),
         label=lab[0], label_text=lab[1])
open(os.path.join(HERE, 'verdict_b9b.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n=== 예산 (레코드 단위, 창 3개 중 최악) ===")
print(f"  ASYNC5 최악 프레임 < dt : {n_ok}/{len(per)}   (중앙 최악 {v['worst_update_ms']['ASYNC5']} ms)")
print(f"  SYNC5  최악 프레임 > dt : {sum(1 for x in sy.values() if max(w for w,_ in x) > x[0][1])}/{len(sy)}"
      f"   (중앙 최악 {v['worst_update_ms']['SYNC5']} ms)  ← 음성 대조")
print(f"\n=== 정확도 ===")
print(f"  ASYNC5 < FROZEN : {acc['better']}/{acc['n']}  {acc['rel_median_pct']:+.2f}%  p={acc['p_one_sided']:.7f}")
print(f"  ASYNC5 < SYNC5  : {pair_as['better']}/{pair_as['n']}  {pair_as['rel_median_pct']:+.2f}%  p={pair_as['p_one_sided']:.6f}")
print(f"\n배경 적합: kick {v['kicks']} · applied {v['applied']} · skipped {v['skipped']}")
print(f"\n{lab[0]} — {lab[1]}")
