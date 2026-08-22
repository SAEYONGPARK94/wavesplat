"""W10 분석기 — 결과보다 먼저 작성. 규칙은 gates_w10 단일출처."""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_w10 as G  # noqa: E402

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
w9 = os.path.join(HERE, G.W9_DIR, 'results.jsonl')
rows += [r for r in (json.loads(l) for l in open(w9)) if r.get('cell') in G.W9_CELLS]
ok = [r for r in rows if 'error' not in r]

print('=' * 76); print('W10 — 창-안 벽의 남은 절반: 2x2 (척도불변 × λ_smooth)'); print('=' * 76)
print('\n## 게이트'); abort = False
for gid, good, msg in G.gates(rows):
    print(f'  [{"PASS" if good else "FAIL"}] {gid}: {msg}')
    if not good and 'ABORT' in gid: abort = True
if abort:
    print('\n⛔ ABORT — 판정하지 않는다.'); sys.exit(0)

print(f'\n## 2×2 (창 안 비, vs 진실 0-스텝, 혼잡 n={G.N_EPISODES})')
cells = [('smooth=1e-4 · SI off  (출하)', 'ORA-WPHI-SHIP'),
         ('smooth=1e-4 · SI on ', 'ORA-WPHI-SI'),
         ('smooth=0    · SI off', 'ORA-WPHI-NOSM'),
         ('smooth=0    · SI on ', 'ORA-WPHI-SI-NOSM')]
res = {}
for lab, arm in cells:
    v = G.ratio_vs_ref(ok, arm); res[arm] = v
    print(f'  {lab:28s} ratio {v["ratio"]:.4f}  나쁨 {v["above"]:2d}/{v["n"]}  '
          f'p={v["p_one_sided"]:.5f}   초과분 설명 {G.share(v["ratio"]):+.0%}')

lab, meaning = G.label(res['ORA-WPHI-NOSM'], res['ORA-WPHI-SI-NOSM'])
print('\n' + '=' * 76); print(f'## 판정 = {lab}'); print(f'##   {meaning}'); print('=' * 76)
json.dump({'label': lab, 'meaning': meaning,
           'cells': {a: res[a] for _, a in cells},
           'share': {a: G.share(res[a]['ratio']) for _, a in cells},
           'gates': [{'id': g, 'ok': o, 'msg': m} for g, o, m in G.gates(rows)]},
          open(os.path.join(HERE, 'verdict_w10.json'), 'w'), ensure_ascii=False, indent=2)
print('verdict_w10.json 기록')
