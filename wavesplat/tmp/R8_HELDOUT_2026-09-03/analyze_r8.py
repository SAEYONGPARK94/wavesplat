"""`R8-HELDOUT` 분석기 — ⛔**결과 전에 작성·동결**. 판정은 여기서만 나온다."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_r8 as G                                                 # noqa: E402

RES = os.path.join(HERE, 'results_r8.jsonl')
META = os.path.join(HERE, 'meta_r8.json')
OUT = os.path.join(HERE, 'verdict_r8.json')


def main():
    rows = [json.loads(l) for l in open(RES, encoding='utf-8') if l.strip()]
    meta = json.load(open(META, encoding='utf-8')) if os.path.exists(META) else {}
    ok = [r for r in rows if 'error' not in r]
    ship = sorted([r for r in ok if r['cell'] == 'SHIP'], key=lambda r: r['record'])
    prop = sorted([r for r in ok if r['cell'] == 'PROP'], key=lambda r: r['record'])

    w = sum(1 for r in ship if r['ratio_out'] > r['ratio_in'])
    p = G.sign_test_one_sided(w, G.N) if ship else float('nan')
    w_prop = sum(1 for r in prop if r['ratio_out'] > r['ratio_in'])
    big = [r for r in ship if r['record'] in G.SEC_LARGE]
    w_big = sum(1 for r in big if r['ratio_out'] > r['ratio_in'])
    p_big = G.sign_test_one_sided(w_big, len(big)) if big else float('nan')

    gate_rows = G.gates(rows, meta)
    blocked = G.blocked_by(gate_rows)
    if blocked:
        lab, msg = 'BLOCKED', '차단 게이트 실패: ' + ','.join(blocked)
    else:
        lab, msg = G.label(w)
    lab = 'R8-HELDOUT :: ' + lab

    def summ(rs):
        return dict(n=len(rs),
                    better=sum(1 for r in rs if r['ratio_out'] > r['ratio_in']),
                    ratio_in_med=_med([r['ratio_in'] for r in rs]),
                    ratio_out_med=_med([r['ratio_out'] for r in rs]),
                    skill_in=sum(1 for r in rs if r['ratio_in'] < 1.0),
                    skill_out=sum(1 for r in rs if r['ratio_out'] < 1.0))

    v = dict(
        batch='R8-HELDOUT', n_rows=len(rows), grade='CONFIRMATORY',
        gates=[dict(name=n, pass_=bool(o), detail=str(d)) for n, o, d in gate_rows],
        gates_pass=f'{sum(1 for _, o, _ in gate_rows if o)}/{len(gate_rows)}',
        blocked_by=blocked,
        thresholds=dict(confirm=G.THRESH_CONFIRM, nominal=G.THRESH_NOMINAL,
                        reverse=G.THRESH_REVERSE, n=G.N),
        # ⚠결과 후 수리(순전한 크래시, 판정 로직 불변): `summ()` 도 'n' 을 내서 중복이었다.
        primary=dict(stat='ratio_out > ratio_in (SHIP)', w=w, n=G.N,
                     p_one_sided=p,
                     **{k: v for k, v in summ(ship).items() if k != 'n'}),
        C_PROP=dict(stat='same on PROP (⛔not primary)', w=w_prop, **summ(prop)),
        SEC_LARGE=dict(stat='same on box>=40m records (검정력 층)', w=w_big,
                       n=len(big), p_one_sided=p_big,
                       threshold=G.SEC_LARGE_THRESH,
                       records=list(G.SEC_LARGE),
                       reaches=bool(big and w_big >= G.SEC_LARGE_THRESH),
                       **{k: v for k, v in summ(big).items() if k != 'n'}),
        gaps={r['record']: dict(gap_in_lam=r['gap_in_lam'], gap_out_lam=r['gap_out_lam'],
                                lam=r['lam'], x_split=r['x_split'],
                                npx_in=r['npx_in'], npx_out=r['npx_out'])
              for r in ship},
        zero_levels={r['record']: dict(zero_in=r['zero_in'], zero_out=r['zero_out'],
                                       zero_ratio=r['zero_out'] / r['zero_in'])
                     for r in ship},
        per_record={r['record']: {k: r[k] for k in
                                  ('ratio_in', 'ratio_out', 'rmse_in', 'rmse_out',
                                   'zero_in', 'zero_out')} for r in ship},
        label=lab, label_msg=msg,
        disclosures=[
            '★IN 과 OUT 은 같은 런이다 — 개구·관측수·시드·초기화가 비트 동일하고 채점 영역만 다르다.',
            '⛔기제 지목 금지 — 외삽 불능/파수 분해능/위상 표류를 이 자료로 분리 불가.',
            '⚠단일 창 WIN=3 · 단일 추정기 시드 42 · 단일 분할(x 중앙 세로선).',
            '⚠분할선과 파향이 이루는 각은 레코드마다 다르다(행에 기록).',
            '⛔두 영역은 다른 자료다 — 절대오차 비교 금지, 비만 쓴다.',
            '⛔PROP 은 실측 무기술이므로 PRIMARY 에 없다(C-PROP 별도 보고).',
            '⛔R1b·B-2 판정 갱신 아님 · ⛔합성 결과로 옮기지 마라.',
            '⛔완전 blind 아님 — 스모크에서 BS11c 1 레코드 2 셀(24 중)을 보았다(PREREG §5).',
            '⚠G-SPLIT 문턱은 결과 전에 20→5 로 내렸다(질의격자 해상도 사유) ⟹ 현 코퍼스에서 발화하지 않는다. 검정력은 SEC-LARGE 가 진다.',
        ])
    json.dump(v, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{lab}\n  {msg}')
    print(f"  gates {v['gates_pass']}")
    for n, o, d in gate_rows:
        print(f"   [{'OK ' if o else 'FAIL'}] {n:56s} {d}")
    print(f'  PRIMARY  SHIP  w={w}/{G.N}  p={p:.5f}')
    print(f"    ratio_in med={v['primary']['ratio_in_med']:.4f}  "
          f"ratio_out med={v['primary']['ratio_out_med']:.4f}  "
          f"skill in={v['primary']['skill_in']}/{G.N} out={v['primary']['skill_out']}/{G.N}")
    print(f"  SEC-LARGE w={w_big}/{len(big)} p={p_big:.5f} (문턱 {G.SEC_LARGE_THRESH}) "
          f"skill in={v['SEC_LARGE']['skill_in']} out={v['SEC_LARGE']['skill_out']}")
    print(f"  C-PROP   w={w_prop}/{G.N}  skill in={v['C_PROP']['skill_in']} "
          f"out={v['C_PROP']['skill_out']}")
    return 0


def _med(a):
    a = sorted(x for x in a if x == x)
    if not a:
        return float('nan')
    n = len(a)
    return a[n // 2] if n % 2 else 0.5 * (a[n // 2 - 1] + a[n // 2])


if __name__ == '__main__':
    sys.exit(main())
