"""W10 개정 분석기 — `G-3` 의 기준값을 **W9 원본에서 재계산**한다(손으로 옮기지 않는다).

`AMENDMENT_1.md` 참조. 동결본 `gates_w10.py` 는 건드리지 않았다;
여기서는 `R_BASE` 만 자료에서 되살리고 나머지 규칙은 전부 그대로 import 한다.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_w10 as G  # noqa: E402

w9p = os.path.join(HERE, G.W9_DIR, 'results.jsonl')
rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
rows += [r for r in (json.loads(l) for l in open(w9p)) if r.get('cell') in G.W9_CELLS]
ok = [r for r in rows if 'error' not in r]

# ★기준값을 자료에서 재계산 — 이것이 개정의 전부다
G.R_BASE = G.ratio_vs_ref(ok, 'ORA-WPHI-SHIP')['ratio']
G.BASE_EXCESS = G.R_BASE - 1.0

print('=' * 76); print('W10 (개정) — 창-안 벽의 남은 절반: 2x2'); print('=' * 76)
print(f'\n기준 R_BASE 재계산 = {G.R_BASE:.10f}  (동결본의 손복사 1.5163 과 3.2e-5 차)')
print('\n## 게이트')
for gid, good, msg in G.gates(rows):
    print(f'  [{"PASS" if good else "FAIL"}] {gid}: {msg}')

print(f'\n## 2×2 (창 안 비, vs 진실 0-스텝, 혼잡 n={G.N_EPISODES})')
cells = [('smooth=1e-4 · SI off (출하)', 'ORA-WPHI-SHIP'),
         ('smooth=1e-4 · SI on ', 'ORA-WPHI-SI'),
         ('smooth=0    · SI off', 'ORA-WPHI-NOSM'),
         ('smooth=0    · SI on ', 'ORA-WPHI-SI-NOSM')]
res = {}
for lab, arm in cells:
    v = G.ratio_vs_ref(ok, arm); res[arm] = v
    print(f'  {lab:27s} ratio {v["ratio"]:.4f}  나쁨 {v["above"]:2d}/{v["n"]}  '
          f'p={v["p_one_sided"]:.5f}   초과분 설명 {G.share(v["ratio"]):+.0%}')

lab, meaning = G.label(res['ORA-WPHI-NOSM'], res['ORA-WPHI-SI-NOSM'])
print('\n' + '=' * 76); print(f'## 판정 = {lab}'); print(f'##   {meaning}'); print('=' * 76)
_g = G.gates(rows)   # ★2026-08-23: 게이트 결과가 verdict 에 안 들어가 있었다(RV-2 감사)
json.dump({'label': lab, 'meaning': meaning, 'R_BASE_recomputed': G.R_BASE,
           'gates': [[str(a), bool(b), str(c)] for a, b, c in _g],
           'gates_pass': sum(1 for a, b, c in _g if b), 'gates_total': len(_g),
           'cells': {a: res[a] for _, a in cells},
           'share': {a: G.share(res[a]['ratio']) for _, a in cells}},
          open(os.path.join(HERE, 'verdict_w10.json'), 'w'), ensure_ascii=False, indent=2)
print('verdict_w10.json 기록')
