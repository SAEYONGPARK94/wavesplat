"""B-21 분석기 — ★**게이트·문턱·라벨을 하나도 여기서 만들지 않는다.**

`gates_b21` 의 `gates()`·`aggregate()`·`cmp_pair()`·`cmp_zero()`·`sign_tost()`·
`label_*()` 를 **호출만** 한다. (초판 `B-19` 가 게이트를 분석기 안에 두어 런 전
자가진단이 원리적으로 불가능했던 결함을 되풀이하지 않는다.)

⛔**결과 0행이면 verdict 을 쓰지 않고 exit 1** (`B-19` §10.4 항목 6).

    python3 tmp/B21_MEDIATION_2026-08-27/analyze_b21.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)
import gates_b21 as G                                        # noqa: E402

OUT = os.path.join(HERE, 'verdict_b21.json')


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, 'results_b21*.jsonl'))):
        if os.path.basename(f).startswith('SMOKE'):
            continue
        with open(f) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def main():
    rows = load()
    ok = [r for r in rows if 'error' not in r]
    if not ok:
        print('⛔결과 0행 ⟹ verdict 을 쓰지 않는다.', flush=True)
        sys.exit(1)

    agg = G.aggregate(rows)
    gate_rows = G.gates(rows, agg)
    print('── 게이트 ' + '─' * 62)
    for n, c, d in gate_rows:
        print(f'{"PASS" if c else "FAIL"}  {n}\n        {d}')
    n_pass = sum(1 for _n, c, _d in gate_rows if c)
    print(f'\n게이트 {n_pass}/{len(gate_rows)} PASS\n')

    val = agg['val']
    verdict = dict(batch='B-21', n_rows=len(rows), n_ok=len(ok),
                   gates=[dict(name=n, passed=c, detail=d) for n, c, d in gate_rows],
                   n_gate_pass=n_pass, n_gates=len(gate_rows),
                   family_alpha=G.FAMILY_ALPHA, n_claims=G.N_CLAIMS,
                   tost_delta_pct=G.TOST_DELTA_PCT,
                   tost_delta_conf_pct=G.TOST_DELTA_CONF_PCT,
                   arms=G.ALL_ARMS, strata={})

    for st in G.STRATA:
        good = agg['good'][st]
        neg = G.negative_ok(agg, st)
        prim = G.cmp_pair(val, good, st, 'KGRID1', G.BASELINE_ARM)
        dose = G.cmp_pair(val, good, st, 'KGRID2', G.BASELINE_ARM)
        conf = G.cmp_pair(val, good, st, 'FITSWAP', 'SHIP')
        abl = G.cmp_pair(val, good, st, 'SHIP', 'KGRID1')      # 참고: B-16/B-19 의 양
        tost_p = G.sign_tost(prim['diffs_pct'], G.TOST_DELTA_PCT)
        tost_c = G.sign_tost(conf['diffs_pct'], G.TOST_DELTA_CONF_PCT)
        share = G.attribution_share(val, good, st)

        lab_p = G.label_primary(st, prim, tost_p, neg)
        lab_s = G.label_dose(st, dose, neg)
        lab_c = G.label_confound(st, conf, tost_c, share)

        def _strip(d):
            return {k: v for k, v in d.items() if k != 'diffs_pct'}

        verdict['strata'][st] = dict(
            n_used=agg['n_used'][st], thresholds=G.thresholds(agg['n_used'][st]),
            primary=_strip(prim), primary_tost=tost_p, primary_label=lab_p[0],
            primary_note=lab_p[1],
            dose=_strip(dose), dose_label=lab_s[0], dose_note=lab_s[1],
            confound=_strip(conf), confound_tost=tost_c, confound_label=lab_c[0],
            confound_note=lab_c[1], attribution_share=share,
            ablation_ref=_strip(abl),
            negative=dict(n1=neg['n1'], n2=neg['n2'], n3=neg['n3'],
                          detail={k: _strip(v) for k, v in neg['detail'].items()}))
        print(f'── [{st}] n_used={agg["n_used"][st]} ' + '─' * 45)
        print(f'  P  KGRID1 vs SPEC25 : {prim["better"]}/{prim["n"]} '
              f'p={prim["p_one_sided"]} rel={prim["rel_median_pct"]} '
              f'TOST(Δ={G.TOST_DELTA_PCT}%) {"PASS" if tost_p["passed"] else "fail"} '
              f'⟹ ★{lab_p[0]}')
        print(f'  S  KGRID2 vs SPEC25 : {dose["better"]}/{dose["n"]} ⟹ {lab_s[0]}')
        print(f'  C  FITSWAP vs SHIP  : {conf["better"]}/{conf["n"]} '
              f'rel={conf["rel_median_pct"]} '
              f'TOST(Δ={G.TOST_DELTA_CONF_PCT}%) {"PASS" if tost_c["passed"] else "fail"} '
              f'⟹ {lab_c[0]}')
        print(f'  A  재적합 몫(기술적, ⛔검정 아님) share={share["share"]}')
        print(f'  N  n1(SHIP<SPEC25) {neg["n1"]} · n2(SPEC25<ZERO) {neg["n2"]} · '
              f'n3(SHIP<ZERO) {neg["n3"]}')
        print(f'  참고 SHIP vs KGRID1 : {abl["better"]}/{abl["n"]} '
              f'rel={abl["rel_median_pct"]}')

    verdict['label'] = ' | '.join(
        f'{st}:{verdict["strata"][st]["primary_label"]}' for st in G.STRATA)
    verdict['gates_all_pass'] = bool(n_pass == len(gate_rows))
    with open(OUT, 'w') as fh:
        json.dump(verdict, fh, indent=1, ensure_ascii=False, default=float)
    print(f'\n★label = {verdict["label"]}\n→ {OUT}')


if __name__ == '__main__':
    main()
