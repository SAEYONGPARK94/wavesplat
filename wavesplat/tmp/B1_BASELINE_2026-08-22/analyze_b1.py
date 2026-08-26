"""B-1 동결 분석기 — ⚠**결과 뒤에 작성**(2026-08-23). 사유는 `gates_b1.py` 독스트링에.

새 런 0회. 기존 배치의 원자료에서 `RESULT.md` 의 수를 **재도출**한다.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_b1 as G

ship_rows = [json.loads(l) for l in open(os.path.join(G.W16, 'results.jsonl')) if l.strip()]
base_rows = [json.loads(l) for l in open(os.path.join(G.S16, 'results.jsonl')) if l.strip()]
ship = G.med_by_episode(ship_rows, G.SHIP_ARM)
base = G.med_by_episode(base_rows, G.BASE_ARM)
n_ship = sum(1 for r in ship_rows if G._arm_of(r) == G.SHIP_ARM and 'error' not in r)
n_base = sum(1 for r in base_rows if G._arm_of(r) == G.BASE_ARM and 'error' not in r)
gates = G.gates(ship, base, n_ship, n_base)
p = G.paired(ship, base)
lab = G.label(p)
# ★2026-08-26 (V4 F-7): 이 배치는 **자기 결과가 없는 재집계 배치**다(`runs=0`).
# 검사기가 2026-08-26 에 *"결과 행이 있어야 인용 가능"* 으로 엄격해지면서 이 배치가
# **오탐으로 인용 불가**가 됐다 — 저장소의 살아있는 헤드라인이 그것이다.
# ⛔느슨하게 되돌리지 않고 **계보를 선언**한다: 재집계 원천을 저장소 상대경로로 적고,
# 검사기가 그 파일들이 **존재하고 행이 있는지** 직접 확인한다(엄격함 유지).
# ⚠판정에 쓰이는 값은 하나도 바뀌지 않는다 — `source_results` 는 **출처 메타데이터**다.
_SRC = [os.path.relpath(os.path.join(x, 'results.jsonl'),
                        os.path.abspath(os.path.join(HERE, '..', '..')))
        for x in (G.W16, G.S16)]
v = dict(batch='B-1', runs=0, source_results=_SRC,
         ship_arm=G.SHIP_ARM, baseline_arm=G.BASE_ARM,
         gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         crowded=p, threshold=G.THRESHOLD,
         label=lab[0], label_text=lab[1],
         _honesty='gates_b1.py / analyze_b1.py 는 결과 뒤에 작성됐다(2026-08-23). '
                  '규칙은 RESULT.md 와 B1_DESIGN.md 에 이미 적힌 것을 코드로 옮긴 것이다.')
open(os.path.join(HERE, 'verdict_b1.json'), 'w').write(json.dumps(v, indent=1, ensure_ascii=False))
print('=== 게이트 ===')
for n_, q_, d_ in gates: print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f"\n=== 혼잡 층 (에피소드당 시드 median-of-3) ===")
print(f"  SHIP < SPEC25 : {p['better']}/{p['n']}  상대중앙 "
      f"{p['rel_median_pct'] if p['rel_median_pct'] is None else round(p['rel_median_pct'],2)}%  "
      f"단측 p={p['p_one_sided']:.3e}")
print(f"  (RESULT.md 문언: 62/64 · −38.58% · 양측 p=2.3e-16)")
print(f"\n{lab[0]} — {lab[1]}")
