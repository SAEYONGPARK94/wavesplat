"""B-2f 동결 분석기 — **결과 파일보다 먼저 쓰였다**. 라벨은 `run_b2f` 독스트링과 축자 일치.

⛔철회 #5 의 처방: **`verdict_b2f.json` 이 없으면 판정을 보고하지 마라.**
"""
import json, os, sys, statistics as st
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import gates_b2 as G

BASE='ADAPT-ON'; TEST='RX-COMBINED'; ABL='ADAPT-OFF'
NOMINAL=10; CONFIRM=11; FAMILY_ALPHA=0.05/6   # ★여섯 번째 열람 (결과 전 기재)


def main():
    rows=[json.loads(l) for l in open(os.path.join(HERE,'results_b2f.jsonl'))]
    ok=[r for r in rows if 'error' not in r]
    exp=len(G.RECORDS)*3
    gates=[('G-0 런 수', len(ok)==exp, f'{len(ok)}/{exp}'),
           ('G-0b 에러 0', len(rows)==len(ok), f'{len(rows)-len(ok)}')]
    # ★G-1 각 팔의 설정이 실제로 적용됐나 (되읽기, 의도표 아님)
    for arm,ad,mx,si in ((BASE,True,30,False),(ABL,False,30,False),(TEST,True,10,True)):
        a=[r for r in ok if r['cell']==arm]
        gates.append((f'G-1 {arm} 설정 되읽기',
                      bool(a) and all(r['adaptive_readback']==ad and r['max_n_readback']==mx
                                      and r['grow_si_readback']==si for r in a),
                      f'n={len(a)} adapt/maxn/si'))
    # ★G-2 개입 계수기 — ADAPT 절제가 실제로 무언가를 껐나
    on=[r for r in ok if r['cell']==BASE]
    fired=sum(1 for r in on if r['apg_calls']>0)
    changed=sum(1 for r in on if r['apg_changed']>0)
    gates.append(('G-2 ADAPT-ON 에서 prune/grow 호출>0', fired>0, f'{fired}/{len(on)} 호출, {changed}/{len(on)} 성분변화'))
    gates.append(('G-3 ADAPT-OFF 에서 호출 0', all(r['apg_calls']==0 for r in ok if r['cell']==ABL),
                  str([r['apg_calls'] for r in ok if r['cell']==ABL])[:60]))
    gates.append(('G-4 12 레코드 전건', len({r['record'] for r in ok})==len(G.RECORDS), ''))
    gates.append(('G-5 VARPRO 전건 발화', all(r['varpro_fired'] for r in ok),
                  f'{sum(1 for r in ok if r["varpro_fired"])}/{len(ok)}'))

    summ={a: G.summarize(ok,a) for a in (BASE,ABL,TEST)}
    pair={a: G.paired(ok,a,BASE) for a in (ABL,TEST)}
    # 짝 비트동점 (field_hash 동일)
    fh={a:{r['record']:r['field_hash'] for r in ok if r['cell']==a} for a in (BASE,ABL,TEST)}
    ties={a: sum(1 for k in fh[BASE] if k in fh[a] and fh[a][k]==fh[BASE][k]) for a in (ABL,TEST)}

    t=summ[TEST]; w=t['beats_zero']
    if changed==0:
        prim=('ADAPT-NO-INTERVENTION',
              'ADAPT-ON 에서 prune/grow 가 성분 수를 한 번도 바꾸지 않았다 '
              f'({changed}/{len(on)}) ⟹ 절제가 원리적으로 무의미하다. 자료가 아니라 설계가 답을 막았다.')
    elif w>=CONFIRM:
        prim=('COMBINED-WORKS (CONFIRMATORY)',
              f'결합 처방이 {w}/12 (p={t["p_one_sided"]:.4f}) — 가족단위 α={FAMILY_ALPHA:.4f} 도 통과.')
    elif w>=NOMINAL:
        prim=('COMBINED-WORKS (EXPLORATORY)',
              f'결합 처방이 {w}/12 (p={t["p_one_sided"]:.4f}) — 명목 통과이나 가족단위 '
              f'α={FAMILY_ALPHA:.4f} 미달. ⚠확증 아님.')
    elif w>=8:
        prim=('PARTIAL', f'결합 처방 {w}/12 — 기준선 {summ[BASE]["beats_zero"]}/12 대비 개선하나 문턱 미달.')
    else:
        prim=('NO-EFFECT', f'결합 처방 {w}/12 ≤ 7 — 처방이 듣지 않는다.')

    a=summ[ABL]; b=summ[BASE]
    if ties[ABL]>=10: sec=('ADAPT-NO-INTERVENTION', f'짝 비트동점 {ties[ABL]}/12.')
    elif a['beats_zero']>b['beats_zero']: sec=('ADAPT-HURTS',
        f'절제가 더 낫다 {a["beats_zero"]} vs {b["beats_zero"]}/12 ⟹ 배포된 적응 기구가 실측에서 순손해.')
    elif a['beats_zero']<b['beats_zero']: sec=('ADAPT-HELPS',
        f'절제가 더 나쁘다 {a["beats_zero"]} vs {b["beats_zero"]}/12.')
    else: sec=('ADAPT-NULL', f'부호로 동수 {a["beats_zero"]}/12.')

    v=dict(batch='B-2f', n_records=len(G.RECORDS), noise=0.02,
           gates=[[g[0],bool(g[1]),g[2]] for g in gates],
           gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
           summary=summ, paired=pair, bit_ties=ties,
           apg_fired=fired, apg_changed=changed,
           primary_label=prim[0], primary_text=prim[1],
           secondary_label=sec[0], secondary_text=sec[1])
    open(os.path.join(HERE,'verdict_b2f.json'),'w').write(json.dumps(v,indent=1,ensure_ascii=False))
    print('=== 게이트 ===')
    for n_,p_,d_ in gates: print(f'  [{"PASS" if p_ else "FAIL"}] {n_}: {d_}')
    print('\n=== 팔 ===')
    for k_,s_ in summ.items():
        print(f'  {k_:12s} n={s_["n"]} beats_zero={s_["beats_zero"]}/12 '
              f'med={s_["median"]:.4f} p={s_["p_one_sided"]:.4f}')
    print('\n=== 짝(vs ADAPT-ON) ===')
    for k_,p_ in pair.items():
        print(f'  {k_:12s} better={p_["better"]}/{p_["n"]} rel_med={p_["rel_median_pct"]:+.2f}% '
              f'p={p_["p_one_sided"]:.4f} bit_ties={ties[k_]}')
    print(f'\nPRIMARY   {prim[0]} — {prim[1]}')
    print(f'SECONDARY {sec[0]} — {sec[1]}')


if __name__=='__main__': main()
