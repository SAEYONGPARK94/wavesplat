"""B-6b 동결 분석기 — ⚠**정직성 기재: 이 파일은 결과 뒤에 쓰였다** (2026-08-23).

`gates_b6b.py`(판정 규칙·문턱·`fit`·`label`)는 **런 전에 커밋**됐으나, 나는 오늘 하루 종일
*"분석기를 결과보다 먼저 써라"* 를 처방하고도 **이 배치에서 그것을 빠뜨렸다**(오늘 이 부류 6번째).
⟹ 여기서 하는 일은 **동결 규칙을 호출하는 것뿐**이고, 규칙은 한 글자도 바꾸지 않았다.
검사기(`tmp/_audit/check_batch_artifacts.py`)의 `analyzer<results` 열이 이 위반을 표시한다.
"""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b6b as G

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_b6b.jsonl'))]
ok = [r for r in rows if 'error' not in r]
gates = G.gates(rows)
per = {}
for rec in G.RECORDS:
    v = sorted([r for r in ok if r['record'] == rec], key=lambda z: z['n_pts'])
    a, b, r2 = G.fit([x['n_pts'] for x in v], [x['t'] for x in v])
    per[rec] = {'intercept_s': a, 'per_point_s': b, 'r2': r2,
                'us_per_point': b * 1e6}
bs = [p['per_point_s'] for p in per.values()]
a_all, b_all, r2_all = G.fit([r['n_pts'] for r in ok], [r['t'] for r in ok])
lab = G.label(r2_all, bs)
t_upd = 0.00157      # B-6 중앙 (verdict_b6.json)
n_star = {rec: int((G.FRAME_BUDGET - t_upd) / p['per_point_s']) for rec, p in per.items()}
v = dict(batch='B-6b', gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         per_record=per, pooled={'intercept_s': a_all, 'per_point_s': b_all, 'r2': r2_all},
         frame_budget_s=G.FRAME_BUDGET, t_upd_s=t_upd, n_points_affordable=n_star,
         label=lab[0], label_text=lab[1],
         _honesty='analyze_b6b.py 는 결과 뒤에 쓰였다(규칙은 gates_b6b.py 에 런 전 동결).')
open(os.path.join(HERE, 'verdict_b6b.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n{'레코드':8s} {'µs/점':>9s} {'절편(ms)':>10s} {'R²':>8s} {'0.1s 프레임 감당 점수':>20s}")
for rec, p in per.items():
    print(f"{rec:8s} {p['us_per_point']:9.2f} {p['intercept_s']*1e3:10.3f} {p['r2']:8.5f} {n_star[rec]:20d}")
print(f"\n통합: {b_all*1e6:.2f} µs/점 · 절편 {a_all*1e3:.3f} ms · R²={r2_all:.5f}")
print(f"\n{lab[0]} — {lab[1]}")
