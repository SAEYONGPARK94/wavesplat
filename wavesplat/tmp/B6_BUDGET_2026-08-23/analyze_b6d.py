"""B-6c 동결 분석기 — ★**결과 파일보다 먼저 쓰였다**(B-6b 의 위반을 반복하지 않는다)."""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b6d as G

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_b6d.jsonl'))]
ok = [r for r in rows if 'error' not in r]
gates = G.gates(rows)
n_pass = sum(1 for r in ok if r['fits_frame'])
duty = [r['duty_14s'] for r in ok]
lab = G.label(n_pass, len(ok), st.median(duty), max(duty))
v = dict(batch='B-6d', control_src=G.CONTROL_SRC, n_control_pts=G.N_CONTROL_PTS,
         gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         n_fits_frame=n_pass, n_fits_100hz=sum(1 for r in ok if r['fits_100hz']),
         duty_median=st.median(duty), duty_max=max(duty),
         t_ctrl_median=st.median([r['t_ctrl'] for r in ok]),
         t_grid_median=st.median([r['t_grid'] for r in ok]),
         per_record={r['record']: {k: round(r[k], 6) for k in
                                   ('t_init', 't_ctrl', 't_grid', 't_upd', 't_refit',
                                    'frame_cost_ctrl', 'dt_frame', 'duty_14s')} for r in ok},
         label=lab[0], label_text=lab[1])
open(os.path.join(HERE, 'verdict_b6d.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n{'레코드':8s} {'제어8점':>9s} {'채점격자':>10s} {'갱신':>8s} {'프레임':>8s} {'dt':>6s} "
      f"{'예산':>5s} {'100Hz':>6s} {'점유율':>7s}")
for r in ok:
    print(f"{r['record']:8s} {r['t_ctrl']*1e3:8.3f}ms {r['t_grid']*1e3:9.2f}ms "
          f"{r['t_upd']*1e3:7.3f}ms {r['frame_cost_ctrl']*1e3:7.3f}ms {r['dt_frame']*1e3:5.0f}ms "
          f"{'OK' if r['fits_frame'] else '⛔':>5s} {'OK' if r['fits_100hz'] else 'X':>6s} "
          f"{r['duty_14s']:7.3f}")
print(f"\n제어 질의 출처(코드): {G.CONTROL_SRC}")
print(f"{lab[0]} — {lab[1]}")
