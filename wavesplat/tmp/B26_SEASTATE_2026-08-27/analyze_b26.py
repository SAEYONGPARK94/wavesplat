"""B-26 동결 분석기 — ★**결과 파일보다 먼저 쓰였다.** 판정은 전부 `gates_b26.py` 가 낸다.

    python3 tmp/B26_SEASTATE_2026-08-27/analyze_b26.py
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_b26 as G                                          # noqa: E402

rows = []
for p in sorted(glob.glob(os.path.join(HERE, 'results_b26*.jsonl'))):
    if 'smoke' in os.path.basename(p):
        continue
    for line in open(p):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
ok = [r for r in rows if 'error' not in r]
# ★0행 가드 — ⛔**결과 0행에서 `verdict_b26.json` 을 쓰지 않는다.**
#   검사기(`tmp/_audit/check_batch_artifacts.py` M-4/V3)가 *"결과 없이 라벨이 붙었다"*
#   로 잡는 바로 그 부류이고, `B-19 §10.4` 항목 6 이 금지한다.
#   `analyze_b21.py:38-41`·`analyze_b25.py:39-41` 과 같은 형태.
if not ok:
    print(f'⛔결과 0행({len(rows)}행 중 유효 0) — 러너를 먼저 돌려라. '
          'verdict 을 쓰지 않는다.')
    sys.exit(1)

gates = list(G.gates_runfree()) + list(G.gates_data()) + list(G.gates_run(rows))
pc1_path = os.path.join(HERE, 'pc1_b26.json')
pc1 = json.load(open(pc1_path)) if os.path.exists(pc1_path) else {}
gates += list(G.gates_pc1(pc1))

agg = G.aggregate(ok)
labels, per = {}, {}
for c in G.CORNER_ORDER:
    d = agg[c]
    lab, msg = G.corner_label(d['prim'], d['neg'])
    labels[c] = lab
    per[c] = dict(
        label=lab, label_msg=msg, reading=G.READING[lab],
        Hs=G.CORNERS[c]['Hs'], Tp=G.CORNERS[c]['Tp'],
        hs_level=G.CORNERS[c]['hs'], tp_level=G.CORNERS[c]['tp'],
        ship_vs_spec=d['prim'], spec_vs_zero=d['neg'], ship_vs_zero=d['sz'],
        medians={k: (st.median(d[k].values()) if d[k] else None)
                 for k in ('ship', 'spec', 'zero')})

anchor_ok = labels[G.ANCHOR_CORNER].startswith('HOLDS')
dec_key, dec_text = G.decompose(labels)
cross = G.cross_test(agg) if all(agg[c]['prim']['n'] for c in G.CORNER_ORDER) else None

primary = per[G.PRIMARY_CORNER]
label = f'SEASTATE :: {G.PRIMARY_CORNER} = {primary["label"]}'
if not anchor_ok:
    label = (f'BATCH-VOID (내부 기준 칸 {G.ANCHOR_CORNER} 가 죽었다: '
             f'{labels[G.ANCHOR_CORNER]}) — ⛔다른 칸의 해석 금지')

v = dict(batch='B-26', n_episodes=G.N_EP, n_per_corner=G.N_PER_CORNER,
         thresholds=G.threshold_table(),
         seeds_consumed=list(G.SEEDS_CONSUMED),
         gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         corners=per, anchor_corner=G.ANCHOR_CORNER, anchor_ok=bool(anchor_ok),
         decomposition={'key': list(dec_key), 'text': dec_text},
         cross_test=cross, pc1=pc1,
         label=label, label_text=primary['reading'])
open(os.path.join(HERE, 'verdict_b26.json'), 'w').write(
    json.dumps(v, indent=1, ensure_ascii=False))

print('=== 게이트 ===')
for n_, q_, d_ in gates:
    print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f'\n게이트 {v["gates_pass"]}/{v["gates_total"]}')
print(f'\n=== 코너별 (n={G.N_PER_CORNER}, median-of-3, 문턱 명목 {G.NOMINAL} · '
      f'확증 {G.CONFIRM} · 파탄 ≤{G.WEAK} · 음성대조 ≥{G.NEG_MIN}) ===')
for c in G.CORNER_ORDER:
    d = per[c]
    p = d['ship_vs_spec']
    mark = '★' if c == G.PRIMARY_CORNER else (' ' if c != G.ANCHOR_CORNER else '·')
    print(f'{mark}{c:9s} Hs={d["Hs"]:.3f} Tp={d["Tp"]:.2f} | '
          f'SHIP<SPEC25 {p["better"]:2d}/{p["n"]:2d} '
          f'{(p["rel_median_pct"] or 0):+7.2f}% p={(p["p_one_sided"] or 1):.6f} | '
          f'음성대조 {d["spec_vs_zero"]["better"]:2d}/{d["spec_vs_zero"]["n"]:2d} | '
          f'SHIP<ZERO {d["ship_vs_zero"]["better"]:2d}/{d["ship_vs_zero"]["n"]:2d} | '
          f'{d["label"]}')
print(f'\n2×2 분해 (LOWH, SHORT, REAL) = {dec_key} ⟹ {dec_text}')
if cross:
    print(f'사전 지정 대비 {cross["contrast"]}: MW p={cross["mw_p"]:.4f} · '
          f'Fisher p={cross["fisher_p"]:.4f} · 승수 {cross["wins"]} '
          f'⟹ {"갈린다" if cross["differs"] else "갈리지 않는다 (⛔`해상상태가 무관하다` 로 읽지 마라)"}')
print(f'\n{label}\n{primary["reading"]}')
