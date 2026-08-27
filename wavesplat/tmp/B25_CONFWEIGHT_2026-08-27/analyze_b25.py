r"""B-25 분석기 — ★**결과 0행에서 작성**(저장소 규율: 분석기를 결과보다 먼저 쓴다).

⛔이 파일에 판정 논리가 **없다**. 게이트·문턱·비교·라벨은 전부 `gates_b25.py` 에 있고
여기서는 **호출만** 한다 ⟹ 러너·자가진단·분석기가 같은 단일출처를 본다.

사용:
    python3 tmp/B25_CONFWEIGHT_2026-08-27/analyze_b25.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT):
    sys.path.insert(0, _p)
os.chdir(ROOT)
import gates_b25 as G                                        # noqa: E402


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'results_b25*.jsonl'))):
        if os.path.basename(p).startswith('SMOKE'):
            continue
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


def main():
    rows = load()
    if not rows:
        print('⛔결과 파일이 없다. 러너를 먼저 돌려라.')
        sys.exit(1)
    agg = G.aggregate(rows)
    gates = G.gates(rows, agg)
    cmpx = G.comparisons(agg['val'], agg['good'])
    negs = G.negatives(agg['val'], agg['zero'], agg['good'])
    label, tier, prim = G.label_primary(cmpx, negs, agg['n_used'])
    hi_n, hi_c, lo_n, lo_c = G.thresholds(agg['n_used'] or G.N_EP)

    print('═' * 78)
    print('B-25 — 관측 신뢰가중(`use_confidence`)의 절제')
    print('═' * 78)
    n_pass = sum(1 for _, o, _ in gates if o)
    print(f'\n▌게이트 {n_pass}/{len(gates)}')
    for n, ok, d in gates:
        print(('  PASS ' if ok else '  ⛔FAIL ') + n + ' | ' + d[:110])

    print(f'\n▌표본: 유효 에피소드 {agg["n_used"]}/{G.N_EP} · 행 {len(agg["ok"])}/{len(rows)}')
    print(f'  문턱(n={agg["n_used"]}, 양측): NOMINAL >={hi_n} 또는 <={lo_n} · '
          f'CONFIRM >={hi_c} 또는 <={lo_c}')

    print('\n▌음성 대조 (각 팔 < 참 제로필드)')
    for a in G.ARM_ORDER:
        c = negs[a]
        print(f'  {a:9s} {c["better"]:3d}/{c["n"]:<3d} p={c["p_one_sided"]} '
              f'rel {c["rel_median_pct"]}')

    print('\n▌짝 비교')
    lbl = {'P': '★PRIMARY  SHIP < FLATCONF  (정보만, 1요인)',
           'S1': ' SECONDARY SHIP < NOCONF    (배포 스위치, 2요인)',
           'S2': ' SECONDARY FLATCONF < NOCONF(순수 질량/λ 효과)'}
    for k in ('P', 'S1', 'S2'):
        c = cmpx[k]
        print(f'  {lbl[k]}')
        print(f'      {c["better"]}/{c["n"]} (동점 {c["n_tie"]}) · p={c["p_one_sided"]} · '
              f'rel median {c["rel_median_pct"]}')
        print(f'      median a={c["median_a"]} b={c["median_b"]}')

    print(f'\n▌PRIMARY 라벨: **{label}** ({tier})')
    print(G.ACTIONS[label])
    print('\n▌⛔널을 읽을 때 반드시 함께 인용할 것:')
    print('  ' + G.mde_note(agg['n_used'] or G.N_EP))

    verdict = dict(
        batch='B-25', question='use_confidence (GT 표면법선 유래 관측가중) 절제',
        label=label, tier=tier,
        gates=[[n, ok, d] for n, ok, d in gates],
        gates_pass=n_pass, gates_total=len(gates),
        n_used=agg['n_used'], n_rows=len(rows), n_ok=len(agg['ok']),
        thresholds=dict(nominal_hi=hi_n, confirm_hi=hi_c,
                        nominal_lo=lo_n, confirm_lo=lo_c),
        comparisons=cmpx, negatives=negs,
        mde_note=G.mde_note(agg['n_used'] or G.N_EP),
        action=G.ACTIONS[label],
        all_gates_pass=(n_pass == len(gates)))
    out = os.path.join(HERE, 'verdict_b25.json')
    with open(out, 'w') as fh:
        json.dump(verdict, fh, ensure_ascii=False, indent=1, default=str)
    print(f'\n→ {out}')
    if n_pass != len(gates):
        print('⛔게이트가 전부 통과하지 않았다 ⟹ 판정을 인용하지 마라.')


if __name__ == '__main__':
    main()
