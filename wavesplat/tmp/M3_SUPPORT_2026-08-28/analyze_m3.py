"""M3 동결 분석기 — ★**결과 파일보다 먼저 쓰였다.** 판정은 전부 `gates_m3.py` 가 낸다.

    python3 tmp/M3_SUPPORT_2026-08-28/analyze_m3.py
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_m3 as G                                           # noqa: E402

rows = []
for p in sorted(glob.glob(os.path.join(HERE, 'results_m3_*.jsonl'))):
    if 'smoke' in os.path.basename(p):
        continue
    for line in open(p):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
ok = [r for r in rows if 'error' not in r]
# ★0행 가드 — ⛔결과 0행에서 verdict 을 쓰지 않는다 (B-19 §10.4 항목 6 · B-26 규약).
if not ok:
    print(f'⛔결과 0행({len(rows)}행 중 유효 0) — 러너를 먼저 돌려라. verdict 을 쓰지 않는다.')
    sys.exit(1)

gates = list(G.gates_runfree()) + list(G.gates_data()) + list(G.gates_run(rows))
pc1 = {}
p1 = os.path.join(HERE, 'pc1_m3.json')
if os.path.exists(p1):
    pc1 = json.load(open(p1))
gates += list(G.gates_pc1(pc1))
pc2 = {}
p2 = os.path.join(HERE, 'pc2_m3.json')
if os.path.exists(p2):
    pc2 = json.load(open(p2))
gates += list(G.gates_pc2(pc2))

agg = G.aggregate(ok)
labels, per = {}, {}
for c in G.CELL_ORDER:
    d = agg[c]
    lab, msg = G.cell_label(d['prim'], d['neg'], d['sz'])
    labels[c] = lab
    per[c] = dict(
        label=lab, label_msg=msg, reading=G.READING[lab],
        coords=G.CELLS[c],
        ship_vs_spec=d['prim'], spec_vs_zero=d['neg'], ship_vs_zero=d['sz'],
        medians={k: (st.median(d[k].values()) if d[k] else None)
                 for k in ('ship', 'spec', 'zero')})

# ★배치 무효 조건 — PC-1/PC-2 둘 중 하나라도 죽으면 셀 해석 금지 (앵커 셀 대체물)
pc_ok = (G.gates_pc1(pc1)[0][1] and G.gates_pc2(pc2)[0][1])
dec_key, dec_text = G.decompose(labels)

primary = per[G.PRIMARY_CELL]
label = f'SUPPORT :: {G.PRIMARY_CELL} = {primary["label"]}'
if not pc_ok:
    label = ('BATCH-VOID (양성대조 PC-1/PC-2 실패 — 생성기 또는 러너가 동결 계보와 '
             '다르다) — ⛔셀 해석 금지')

v = dict(batch='M3', n_episodes=G.N_EP, n_per_cell=G.N_PER_CELL,
         thresholds=G.threshold_table(),
         seeds_consumed=list(G.SEEDS_CONSUMED),
         gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         cells=per, primary_cell=G.PRIMARY_CELL, pc_ok=bool(pc_ok),
         decomposition={'key': list(dec_key), 'text': dec_text},
         pc1=pc1, pc2={'n_rows': len(pc2.get('rows', []))},
         label=label, label_text=primary['reading'])
open(os.path.join(HERE, 'verdict_m3.json'), 'w').write(
    json.dumps(v, indent=1, ensure_ascii=False))

print('=== 게이트 ===')
for n_, q_, d_ in gates:
    print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f'\n게이트 {v["gates_pass"]}/{v["gates_total"]}')
print(f'\n=== 셀별 (n={G.N_PER_CELL}, median-of-3, 문턱 명목 {G.NOMINAL} · '
      f'확증 {G.CONFIRM} · 역전 {G.WEAK}) ===')
for c in G.CELL_ORDER:
    p = per[c]
    pr = p['ship_vs_spec']
    print(f"  {c:7s} {p['label']:28s} {pr['better']}/{pr['n']} "
          f"{(pr['rel_median_pct'] or 0):+7.2f}% p={(pr['p_one_sided'] or 1):.6f} | "
          f"neg {p['spec_vs_zero']['better']}/{p['spec_vs_zero']['n']} "
          f"sz {p['ship_vs_zero']['better']}/{p['ship_vs_zero']['n']}")
print(f'\n분해: {dec_key} — {dec_text}')
print(f'\n라벨: {label}')
