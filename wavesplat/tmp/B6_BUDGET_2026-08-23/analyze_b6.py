"""B-6 동결 분석기 — **결과 파일보다 먼저 쓰였다.**"""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b6 as G

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_b6.jsonl'))]
ok = [r for r in rows if 'error' not in r]
gates = G.gates(rows)
ratios = [r['ratio'] for r in ok]
frame_ok = all(r['frame_cost'] < r['dt_frame'] for r in ok)
worst_frame = max(r['frame_cost'] for r in ok)
lab = G.label(st.median(ratios), max(ratios), frame_ok, worst_frame)
v = dict(batch='B-6', tau=G.TAU,
         gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         ratio_median=st.median(ratios), ratio_max=max(ratios),
         t_refit_median=st.median([r['t_refit'] for r in ok]),
         t_refit_max=max(r['t_refit'] for r in ok),
         t_init_median=st.median([r['t_init'] for r in ok]),
         t_upd_median=st.median([r['t_upd'] for r in ok]),
         t_pred_median=st.median([r['t_pred'] for r in ok]),
         frame_budget_ok=frame_ok, frame_cost_max=worst_frame,
         dt_frame_min=min(r['dt_frame'] for r in ok),
         per_record={r['record']: {k: round(r[k], 5) for k in
                                   ('t_init', 't_refit', 't_upd', 't_pred', 'ratio',
                                    'frame_cost', 'dt_frame')} for r in ok},
         label=lab[0], label_text=lab[1])
open(os.path.join(HERE, 'verdict_b6.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n{'레코드':8s} {'init(s)':>8s} {'refit(s)':>9s} {'upd(s)':>9s} {'pred(s)':>8s} "
      f"{'refit/τ':>8s} {'프레임비용':>10s} {'dt':>7s}")
for r in ok:
    flag = '' if r['frame_cost'] < r['dt_frame'] else '  ⛔초과'
    print(f"{r['record']:8s} {r['t_init']:8.3f} {r['t_refit']:9.4f} {r['t_upd']:9.5f} "
          f"{r['t_pred']:8.4f} {r['ratio']:8.3f} {r['frame_cost']:10.5f} {r['dt_frame']:7.3f}{flag}")
print(f"\n중앙: refit {v['t_refit_median']:.4f} s · init {v['t_init_median']:.3f} s · "
      f"upd {v['t_upd_median']:.5f} s · pred {v['t_pred_median']:.4f} s")
print(f"\n{lab[0]} — {lab[1]}")
