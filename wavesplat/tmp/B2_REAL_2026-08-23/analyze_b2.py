"""B-2 분석기 — 규칙은 gates_b2 단일출처. ★verdict_b2.json 을 반드시 남긴다."""
import json, os, sys, statistics as st
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import gates_b2 as G  # noqa: E402
rows=[json.loads(l) for l in open(os.path.join(HERE,'results.jsonl'))]
ok=[r for r in rows if 'error' not in r]
print('='*74); print('B-2 — 실측에서 제안 표현이 무엇을 하는가'); print('='*74)
print('\n## 게이트')
gs=G.gates(rows)
for gid,g,m in gs: print(f'  [{"PASS" if g else "FAIL"}] {gid}: {m[:80]}')
print(f'\n## PRIMARY — PROP 이 제로필드(1.0)를 이기는가 (n={len(G.RECORDS)}, 문턱 {G.THRESHOLD})')
prop=G.summarize(ok,'PROP'); ship=G.summarize(ok,'SHIP')
for nm,s in (('PROP',prop),('SHIP',ship)):
    print(f'  {nm:5s} 이김 {s["beats_zero"]:2d}/{s["n"]}  중앙 {s["median"]:.4f}  단측 p={s["p_one_sided"]:.5f}')
print('\n## 층별 (결과 전 선언)')
for nm,sub in (('대형 박스',G.LARGE_BOX),('소형 박스(BS)',G.SMALL_BOX)):
    p=G.summarize(ok,'PROP',subset=sub); s=G.summarize(ok,'SHIP',subset=sub)
    print(f'  {nm:14s} PROP {p["beats_zero"]}/{p["n"]} (중앙 {p["median"]:.4f}) · '
          f'SHIP {s["beats_zero"]}/{s["n"]} (중앙 {s["median"]:.4f})')
pv=G.paired(ok,'PROP','SHIP')
print(f'\n## 부차 — PROP vs SHIP: PROP 더 좋음 {pv["better"]}/{pv["n"]}  '
      f'상대 중앙 {pv["rel_median_pct"]:+.2f}%  p={pv["p_one_sided"]:.5f}')
lab,mean=G.label(prop,ship,pv)
print('\n'+'='*74); print(f'## 판정 = {lab}'); print(f'##   {mean}'); print('='*74)
json.dump({'label':lab,'meaning':mean,'primary':prop,'ship':ship,'prop_vs_ship':pv,
           'by_stratum':{'large':{a:G.summarize(ok,a,subset=G.LARGE_BOX) for a in G.ARM_ORDER},
                         'small':{a:G.summarize(ok,a,subset=G.SMALL_BOX) for a in G.ARM_ORDER}},
           'gates':[{'id':g,'ok':o,'msg':m} for g,o,m in gs]},
          open(os.path.join(HERE,'verdict_b2.json'),'w'), ensure_ascii=False, indent=2)
print('\n★verdict_b2.json 기록 — 동결 분석기 출력이다')
