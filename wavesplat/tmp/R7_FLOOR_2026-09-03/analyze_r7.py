"""`R7-FLOOR` 분석기 — ⛔**결과 전에 작성·동결**. 판정은 여기서만 나온다."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_r7 as G                                                 # noqa: E402

RES = os.path.join(HERE, 'results_r7.jsonl')
META = os.path.join(HERE, 'meta_r7.json')
OUT = os.path.join(HERE, 'verdict_r7.json')


def main():
    rows = [json.loads(l) for l in open(RES, encoding='utf-8') if l.strip()]
    meta = json.load(open(META, encoding='utf-8')) if os.path.exists(META) else {}
    ok = sorted([r for r in rows if 'error' not in r], key=lambda r: r['record'])

    floor = [r['floor'] for r in ok]
    ratio = [r['ratio_ship'] for r in ok]
    zero = [r['zero_new'] for r in ok]
    ratio_p = [r['ratio_prop'] for r in ok]
    abs_ship = [r['ship_nowcast'] for r in ok]

    mde = G.mde_rho(floor) if len(floor) == G.N else float('nan')
    meta = dict(meta); meta['mde'] = mde

    rho, p = G.perm_p_one_sided(floor, ratio) if len(ok) == G.N else (float('nan'),) * 2
    c1_rho, c1_p = G.perm_p_one_sided(floor, zero) if len(ok) == G.N else (float('nan'),) * 2
    c2_rho, c2_p = G.perm_p_one_sided(floor, ratio_p) if len(ok) == G.N else (float('nan'),) * 2
    c3_rho, c3_p = G.perm_p_one_sided(floor, abs_ship) if len(ok) == G.N else (float('nan'),) * 2

    gate_rows = G.gates(rows, meta)
    blocked = G.blocked_by(gate_rows)
    if blocked:
        lab, msg = 'R7-FLOOR :: BLOCKED', '차단 게이트 실패: ' + ','.join(blocked)
    else:
        lab, msg = G.label(rho, p, c1_p)
        lab = 'R7-FLOOR :: ' + lab

    v = dict(
        batch='R7-FLOOR', kind='REANALYSIS-OF-FROZEN-DATA', new_runs=0,
        n_rows=len(rows), grade='EXPLORATORY',
        gates=[dict(name=n, pass_=bool(o), detail=str(d)) for n, o, d in gate_rows],
        gates_pass=f'{sum(1 for _, o, _ in gate_rows if o)}/{len(gate_rows)}',
        blocked_by=blocked,
        alpha=G.ALPHA, mde_rho=mde,
        primary=dict(stat='spearman(floor, ratio_ship)', rho=rho, p_one_sided=p,
                     n=len(ok), mde=mde, reaches_mde=bool(rho == rho and rho >= mde)),
        controls=dict(
            C1=dict(stat='spearman(floor, zero_new)', rho=c1_rho, p_one_sided=c1_p,
                    must_be='p > alpha (아니면 FLOOR-CONFOUNDED)'),
            C2=dict(stat='spearman(floor, ratio_prop)', rho=c2_rho, p_one_sided=c2_p),
            C3=dict(stat='spearman(floor, ship_nowcast_abs)', rho=c3_rho, p_one_sided=c3_p)),
        per_record={r['record']: {k: r[k] for k in
                                  ('floor', 'ratio_ship', 'ratio_prop', 'zero_new',
                                   'ship_nowcast', 'prop_nowcast', 'sg',
                                   'hf_excess_rms_mm', 'Tp_gauge')} for r in ok},
        label=lab, label_msg=msg,
        disclosures=[
            '⛔blind 아님 — PREREG §1 이 7×6 값 노출을 축자로 공시한다(G-EXPOSE 가 대조).',
            '⛔CONFIRMATORY 청구 안 함 · 등급 EXPLORATORY.',
            '⛔floor 는 상한이다(R2b 축자) — 게이지 자기잡음·공간 비균질 포함.',
            '⚠시간 스펙트럼 → 공간 한 프레임 전이는 정상성·에르고딕성 가정이다.',
            '⛔사이트 1개(BS) · 레코드 7 · 창 1 · 추정기 시드 42 — R1b 상속.',
            '⛔보정된 nRMSE 를 계산하지 않는다(오차 독립 미측정, PREREG §5).',
            '⛔R1b·R2b 의 라벨은 갱신되지 않는다.',
        ])
    json.dump(v, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{lab}\n  {msg}')
    print(f"  gates {v['gates_pass']}  MDE={mde:.4f}")
    for n, o, d in gate_rows:
        print(f"   [{'OK ' if o else 'FAIL'}] {n:52s} {d}")
    print(f'  PRIMARY rho={rho:.4f} p={p:.5f}')
    print(f'  C-1 zero rho={c1_rho:.4f} p={c1_p:.5f}   C-2 prop rho={c2_rho:.4f} p={c2_p:.5f}'
          f'   C-3 abs rho={c3_rho:.4f} p={c3_p:.5f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
