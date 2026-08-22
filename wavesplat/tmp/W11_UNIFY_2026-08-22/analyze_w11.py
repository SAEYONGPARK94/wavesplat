"""W11 분석기 — 결과보다 먼저 작성. 규칙은 gates_w11 단일출처."""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_w11 as G  # noqa: E402
rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
ok = [r for r in rows if 'error' not in r]
print('=' * 76); print('W11 — 라우터 제거(학습 경로 통일)의 비용'); print('=' * 76)
print('\n## 게이트')
for gid, good, msg in G.gates(rows):
    print(f'  [{"PASS" if good else "FAIL"}] {gid}: {msg}')
print('\n## 기술통계')
for arm in G.ARM_ORDER:
    v = [r[G.FIELD] for r in ok if r['cell'] == arm]
    print(f'  {arm:14s} nRMSE 중앙 {st.median(v):.5f}  (n={len(v)})')
print('\n## PRIMARY — ROUTED 가 FORCE-LEARNED 보다 좋은가 (H1 = 라우터가 제 값을 한다)')
p = G.primary(ok)
print(f'  전체   ROUTED 승 {p["routed_better"]}/{p["n"]}  (문턱 {G.THRESHOLD})  '
      f'통일 비용 중앙 {p["force_cost_pct_median"]:+.2f}%  p={p["p_one_sided"]:.5f}')
for s_ in ('crowded', 'sparse'):
    q = G.primary(ok, s_)
    print(f'  {s_:8s} ROUTED 승 {q["routed_better"]}/{q["n"]}  '
          f'통일 비용 중앙 {q["force_cost_pct_median"]:+.2f}%  p={q["p_one_sided"]:.5f}')
lab, meaning = G.label(p)
print('\n' + '=' * 76); print(f'## 판정 = {lab}'); print(f'##   {meaning}'); print('=' * 76)
json.dump({'label': lab, 'meaning': meaning, 'primary': p,
           'by_stratum': {s_: G.primary(ok, s_) for s_ in ('crowded', 'sparse')},
           'gates': [{'id': g, 'ok': o, 'msg': m} for g, o, m in G.gates(rows)]},
          open(os.path.join(HERE, 'verdict_w11.json'), 'w'), ensure_ascii=False, indent=2)
print('verdict_w11.json 기록')
