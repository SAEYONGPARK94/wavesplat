"""W13 분석기 — 결과보다 먼저 작성. 규칙은 gates_w14 단일출처."""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_w14 as G  # noqa: E402
rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
ok = [r for r in rows if 'error' not in r]
print('=' * 76); print('W14 — 혼잡에서 학습 vs 해석적 정면 비교 (같은 코퍼스)'); print('=' * 76)
print('\n## 게이트')
for gid, good, msg in G.gates(rows):
    print(f'  [{"PASS" if good else "FAIL"}] {gid}: {msg}')
print('\n## 기술통계 (제로필드 ≈ 0.25)')
for arm in G.ARM_ORDER:
    v = [r[G.FIELD] for r in ok if r['cell'] == arm]
    w = [r['wall_s'] for r in ok if r['cell'] == arm]
    print(f'  {arm:9s} nRMSE 중앙 {st.median(v):.5f}   런당 중앙 {st.median(w):6.1f}s   n={len(v)}')
p = G.primary(ok); p2 = G.primary(ok, 15); p5 = G.primary(ok, 30)
print('\n## PRIMARY — 학습이 해석적보다 나은가 (혼잡 n=64)')
for nm, q in (('전체', p), ('130comp', p2), ('5comp', p5)):
    print(f'  {nm:6s} 학습 승 {q["learned_better"]:2d}/{q["n"]:2d}  상대 중앙 {q["rel_median_pct"]:+8.2f}%  '
          f'p={q["p_one_sided"]:.5f}   제로필드 이김 학습 {q["learned_beats_zero"]}/{q["n"]} · '
          f'해석적 {q["analytic_beats_zero"]}/{q["n"]}')
lab, meaning = G.label(p, p2, p5)
print('\n' + '=' * 76); print(f'## 판정 = {lab}'); print(f'##   {meaning}'); print('=' * 76)
json.dump({'label': lab, 'meaning': meaning, 'primary': p, 'by_ncomp': {'2': p2, '5': p5},
           'gates': [{'id': g, 'ok': o, 'msg': m} for g, o, m in G.gates(rows)]},
          open(os.path.join(HERE, 'verdict_w14.json'), 'w'), ensure_ascii=False, indent=2)
print('verdict_w14.json 기록')
