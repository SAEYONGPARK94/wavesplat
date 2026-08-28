"""M2 동결 분석기 — ★**결과 파일보다 먼저 쓰였다.** 판정은 전부 `gates_m2.py` 가 낸다.

    python3 tmp/M2_LOCTUNE_2026-08-28/analyze_m2.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_m2 as G                                            # noqa: E402

rows = []
for p in sorted(glob.glob(os.path.join(HERE, 'results_m2*.jsonl'))):
    if 'smoke' in os.path.basename(p):
        continue
    for line in open(p):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
ok = [r for r in rows if 'error' not in r]
# ★0행 가드 — ⛔결과 0행에서 `verdict_m2.json` 을 쓰지 않는다.
if not ok:
    print(f'⛔결과 0행({len(rows)}행 중 유효 0) — 러너를 먼저 돌려라. verdict 을 쓰지 않는다.')
    sys.exit(1)

frozen = G.frozen_rows()
gates = (list(G.gates_runfree()) + list(G.gates_data())
         + list(G.gates_run(rows, frozen)))
agg = G.aggregate(ok, frozen)

per = {}
for c in G.CORNERS_M2:
    d = agg[c]
    lab = G.verdict_label(d)
    d2p = {a: v['p_one_sided'] for a, v in d['d2'].items()}
    per[c] = dict(d, verdict=lab, reading=G.READING[lab], d2_p=d2p)
# 코너 간 Holm — D-2 는 **기술적**이다(라벨을 바꾸지 않는다)
best_p = {c: min([v['p_one_sided'] for v in per[c]['d2'].values()] or [1.0])
          for c in G.CORNERS_M2}
holm = G.holm(best_p)
for c in G.CORNERS_M2:
    per[c]['d2_holm_sig'] = holm[c]

prim = per[G.PRIMARY_CORNER]
disc = G.disclosures(agg)
label = f'LOCTUNE :: {G.PRIMARY_CORNER} = {prim["verdict"]}'

v = dict(batch='M2', question='B-26 짧은 주기 코너 마진 중 기준선 미조율 몫',
         direction=G.DIRECTION,
         corners=list(G.CORNERS_M2), n_per_corner=G.N_PER_CORNER,
         thresholds=G.threshold_table(),
         ladder=list(G.FROZEN_LADDER),
         candidates={c: list(G.candidates(c)) for c in G.CORNERS_M2},
         loc_star={c: G.loc_star(c) for c in G.CORNERS_M2},
         sweep_corpus=G.sweep_corpus_lambda(),
         s17_rules=G.s17_rules(),
         gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         results=per,
         disclosures={k: G.DISCLOSURE[k] for k in disc},
         label=label, label_text=prim['reading'])
open(os.path.join(HERE, 'verdict_m2.json'), 'w').write(
    json.dumps(v, indent=1, ensure_ascii=False, default=str))

print('=== 게이트 ===')
for n_, q_, d_ in gates:
    print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f'\n게이트 {v["gates_pass"]}/{v["gates_total"]}\n')

print(f'=== 팔별 (n={G.N_PER_CORNER}, median-of-3) ===')
hdr = ('코너', '팔', 'loc', '중앙 nRMSE', 'W(SHIP<a)', 'med(rel)%',
       'a<ZERO', 'skill%', 'zeroLoss%', 'elig(B26/S17)')
print('%-9s %-7s %6s %11s %10s %10s %7s %8s %10s %s' % hdr)
for c in G.CORNERS_M2:
    d = agg[c]
    for a in G.ARM_ORDER:
        if a not in d['stats']:
            continue
        s, e = d['stats'][a], d['elig'][a]
        mark = '★' if a == d['best'] else ' '
        print('%-9s %s%-6s %6.2f %11.6f %6d/%-3d %+9.2f %4d/%-2d %+7.2f %9.2f  %s/%s'
              % (c, mark, a, G.ARM_LOC[a], s['median'], s['W'], s['n'],
                 s['rel'], s['a_lt_zero'], s['a_lt_zero_n'], (s['skill'] or 0) * 100,
                 (s['zero_loss'] or 0) * 100, 'O' if e['b26'] else 'X',
                 'O' if e['s17'] else 'X'))
print()
for c in G.CORNERS_M2:
    d = per[c]
    print(f'{c:9s} loc*={d["loc_star"]:.3f} 후보={d["candidates"]} '
          f'BEST={d["best"]} | B-26 라벨 {d["b26_label_frozen"]} → {d["b26_label_new"]} '
          f'({d["b26_label_new_msg"]}) | 인용크기 '
          f'{(d["cited_rel_pct"] if d["cited_rel_pct"] is not None else float("nan")):+.2f}% '
          f'({d["cited_from"]}) | {d["verdict"]}'
          + ('  [D-2 Holm SIG]' if d['d2_holm_sig'] else ''))
    for a, x in d['d2'].items():
        print(f'    D-2 {a} < SPEC25: {x["better"]}/{x["n"]} '
              f'{(x["rel_median_pct"] or 0):+.2f}% p={(x["p_one_sided"] or 1):.6f} '
              f'(문턱 {G.D2_THRESH})')
    m = d['mde']
    if m:
        print(f'    MDE: 라벨이 CONFIRMATORY 아래로 가려면 기준선 '
              f'**{m["pct_below_confirm"]:.2f}%** 개선 필요 · NOMINAL 아래 '
              f'{m["pct_below_nominal"]:.2f}% · BREAKS 까지 {m["pct_to_weak"]:.2f}%')
print()
for k in disc:
    print(f'  {G.DISCLOSURE[k]}')
print(f'\n{label}\n{prim["reading"]}')
