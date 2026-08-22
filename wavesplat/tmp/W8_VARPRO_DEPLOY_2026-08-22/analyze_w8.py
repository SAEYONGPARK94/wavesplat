"""W8 분석기 — 결과보다 먼저 작성됨. 규칙은 gates_w8 단일출처."""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_w8 as G  # noqa: E402

rows = [json.loads(l) for l in open(os.environ.get('W8_OUT', os.path.join(HERE, 'results.jsonl')))]
ok = [r for r in rows if 'error' not in r]
print('=' * 74); print('W8 — VARPRO 배포 검증'); print('=' * 74)
print('\n## 게이트')
for gid, good, msg in G.gates(rows):
    print(f'  [{"PASS" if good else "FAIL"}] {gid}: {msg}')
print(f'\n## 기술통계 ({G.REGIME}, n={G.N_EPISODES})')
for a in G.ARM_ORDER:
    v = [r[G.FIELD] for r in ok if r['cell'] == a]
    t = [r['init_time_s'] for r in ok if r['cell'] == a]
    print(f'  {a:8s} nRMSE 중앙 {st.median(v):.5f}   초기화 중앙 {st.median(t):6.2f}s')
fid = G.paired(ok, 'PKG-VP', 'MP-VP')
eff = G.paired(ok, 'PKG-VP', 'PRE')
mp_eff = G.paired(ok, 'MP-VP', 'PRE')
print('\n## PRIMARY — 이식 충실도 (PKG-VP vs MP-VP)')
print(f'  상대차 ≤{G.FIDELITY_TOL:g} 인 짝: {fid["identical"]}/{fid["n"]} '
      f'(문턱 {G.FIDELITY_THRESHOLD})   rel median {fid["rel_median_pct"]:+.6f}%')
print('\n## SECONDARY — 배포 전후 (vs PRE)')
for nm, d in (('PKG-VP', eff), ('MP-VP', mp_eff)):
    print(f'  {nm:8s} lower {d["lower"]}/{d["n"]}  rel median {d["rel_median_pct"]:+.2f}%  '
          f'p={d["p_one_sided"]:.5f}')
lab, meaning = G.label(fid, eff)
print('\n' + '=' * 74); print(f'## 판정 = {lab}'); print(f'##   {meaning}'); print('=' * 74)
json.dump({'label': lab, 'meaning': meaning, 'fidelity': fid, 'effect': eff,
           'mp_effect': mp_eff,
           'gates': [{'id': g, 'ok': o, 'msg': m} for g, o, m in G.gates(rows)]},
          open(os.path.join(HERE, 'verdict_w8.json'), 'w'), ensure_ascii=False, indent=2)
print('verdict_w8.json 기록')
