"""B-20b 분석기 — ★**게이트·문턱·라벨을 하나도 여기서 만들지 않는다.**

`gates_b20b` 의 `analysis()`·`gates()`·`label_b20b()`·`thresholds()`·
`followup_triggers()`·`sparse_crosscheck()` 를 **호출만** 한다.
상속 게이트는 `gates_b21.gates()` 를 그대로 부른다(중복 구현 0).

⛔**신규 런이 0개다.** 입력은 `tmp/B21_MEDIATION_2026-08-27/results_b21_*.jsonl`
   **동결 자료**이고 이 배치는 거기에 **쓰지 않는다**.
⛔결과 0행이면 verdict 을 쓰지 않고 exit 1.

    python3 tmp/B20b_QUANT_CROWDED_2026-08-27/analyze_b20b.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)
import gates_b20b as G                                          # noqa: E402
import gates_b21 as G21                                         # noqa: E402

OUT = os.path.join(HERE, 'verdict_b20b.json')
PAIRS = os.path.join(HERE, 'pairs_b20b.jsonl')     # ⚠파생 산출물 — ⛔`results*` 가 아니다


def main():
    # ★★**인자 방벽 (2026-08-27 사고 뒤 신설)** — ⛔이 분석기는 인자를 받지 않는다.
    #   실제 사고: 누군가 `analyze_b20b.py --selftest` 를 쳤는데 이 파일에 `sys.argv` 가
    #   **한 번도 없어서** 플래그가 조용히 무시되고 **진짜 재분석이 돌았다**
    #   (`verdict_b20b.json`·`pairs_b20b.jsonl` 생성 → 즉시 삭제, PREREG §7.6 에 공시).
    #   ⟹ blind 배치에서 *"아무 일도 안 하는 플래그"* 는 **`{}` 로 팔을 정의하는 것과 같은
    #   부류의 결함**이다(조용한 no-op). 산문 경고 대신 **기계**로 막는다.
    #   ⚠자가진단은 이 파일이 아니라 `gates_b20b.selftest()` 다(합성 픽스처, 동결 자료 미열람).
    if len(sys.argv) > 1:
        print(f'⛔이 분석기는 인자를 받지 않는다 (받은 것: {sys.argv[1:]}).\n'
              '  · 진짜 분석  : python3 tmp/B20b_QUANT_CROWDED_2026-08-27/analyze_b20b.py\n'
              "  · 자가진단   : python3 -c \"import sys;sys.path.insert(0,"
              "'tmp/B20b_QUANT_CROWDED_2026-08-27');import gates_b20b as G;print(G.selftest()[0])\"\n"
              '  ⚠인자를 무시하고 돌면 **blind 가 사고로 깨진다** — 그래서 돌지 않는다.',
              flush=True)
        sys.exit(2)
    rows = G.load_frozen_rows()
    ok = [r for r in rows if 'error' not in r]
    if not ok:
        print('⛔동결 결과 0행 ⟹ verdict 을 쓰지 않는다.', flush=True)
        sys.exit(1)

    res = G.analysis(rows)
    inherited = G21.gates(rows, res['agg'])            # ★상속 게이트 (단일출처)
    gate_rows = G.gates(rows, res, inherited)
    label, text = G.label_b20b(res, gate_rows)
    trig = G.followup_triggers(label, res, gate_rows)
    sc = G.sparse_crosscheck(res)

    print('── 상속 게이트 (B-21) ' + '─' * 50)
    for n, c, d in inherited:
        print(f'{"PASS" if c else "FAIL"}  {n}')
    print(f'  상속 {sum(1 for _n, c, _d in inherited if c)}/{len(inherited)}\n')
    print('── B-20b 게이트 ' + '─' * 56)
    for n, c, d in gate_rows:
        print(f'{"PASS" if c else "FAIL"}  {n}\n        {d}')
    n_pass = sum(1 for _n, c, _d in gate_rows if c)
    print(f'\n게이트 {n_pass}/{len(gate_rows)} PASS · 하드 전건 {G.hard_pass(gate_rows)}\n')

    prim, m = res['primary'], res['magnitudes']
    nominal, confirm, not_earn = G.thresholds(prim['n'])
    print(f'── PRIMARY  {G.ARM_BASE} < {G.ARM_TREAT}  [{G.STRATUM}] ' + '─' * 24)
    print(f'  {prim["better"]}/{prim["n"]} (진 {prim["worse"]}, 동점 {prim["n_tie"]}) '
          f'p={prim["p_one_sided"]} · 상대 중앙 {prim["rel_median_pct"]}%')
    print(f'  문턱 NOMINAL {nominal} · CONFIRM {confirm} · NOT_EARN {not_earn} '
          f'(연역 하한 {G.DEDUCIBLE_LOWER_BOUND})')
    print(f'  크기: MAG_CLEAN {m["mag_clean"]} · MAG_MAIN {m["mag_main"]} · '
          f'shift {m["shift_pp"]} pp · 밴드 ±{m["pp_band"]} ⟹ '
          f'{"인용값 유지" if m["magnitude_unmoved"] else "⛔MAG_CLEAN 으로 교체"}')
    for k in ('n1', 'n1b', 'n2'):
        c = res['neg'][k]
        print(f'  음성 {k}: {c["better"]}/{c["n"]} rel={c["rel_median_pct"]}')
    print(f'  희소 교차확인(보고전용): {sc["here"]["better"]}/{sc["here"]["n"]} vs '
          f'B-20 {sc["b20_primary_b"]["better"] if sc["b20_primary_b"] else None}/16')
    print(f'\n★label = {label}\n  {text}')
    print(f'★후속 `B-20c`(288런) 발화 = {trig["fire"]}  {trig}')

    eps = res['agg']['good'][G.STRATUM]
    val = res['agg']['val']
    with open(PAIRS, 'w') as fh:
        for e in eps:
            a = val.get((G.STRATUM, G.ARM_BASE, e))
            b = val.get((G.STRATUM, G.ARM_TREAT, e))
            fh.write(json.dumps(dict(
                derived=True, source='tmp/B21_MEDIATION_2026-08-27/results_b21_*.jsonl',
                stratum=G.STRATUM, episode=e, base_arm=G.ARM_BASE, treat_arm=G.ARM_TREAT,
                base_median_of_3=a, treat_median_of_3=b,
                rel_pct=((a - b) / b * 100.0 if (a is not None and b) else None),
                base_better=(a is not None and b is not None and a < b)),
                ensure_ascii=False) + '\n')

    verdict = dict(
        batch='B-20b', kind='REANALYSIS-OF-FROZEN-DATA', new_runs=0,
        source=dict(batch='B-21', results=G.B21_RESULTS, verdict=G.B21_VERDICT,
                    n_rows=len(rows), n_ok=len(ok)),
        frozen=G.frozen_check(), blind_evidence=G.blind_evidence(),
        stratum=G.STRATUM, arms=dict(base=G.ARM_BASE, treat=G.ARM_TREAT, ship=G.ARM_SHIP),
        arm_defs={G.ARM_BASE: dict(k_snap_dk=G21.FITSWAP_DK),
                  G.ARM_TREAT: dict(k_snap_dk=G21.DK1),
                  G.ARM_SHIP: dict(k_snap_dk=None)},
        alpha=G.ALPHA, family_alpha=G.FAMILY_ALPHA, n_claims=G.N_CLAIMS,
        thresholds=dict(n=prim['n'], nominal=nominal, confirm=confirm,
                        not_earn=not_earn, deducible_lower_bound=G.DEDUCIBLE_LOWER_BOUND),
        n_used=len(eps),
        inherited_gates=[dict(name=n, passed=c, detail=d) for n, c, d in inherited],
        gates=[dict(name=n, passed=c, detail=d) for n, c, d in gate_rows],
        n_gate_pass=n_pass, n_gates=len(gate_rows), hard_pass=G.hard_pass(gate_rows),
        primary={k: v for k, v in prim.items() if k != 'diffs_pct'},
        magnitudes=m,
        negative={k: {kk: vv for kk, vv in v.items() if kk != 'diffs_pct'}
                  for k, v in res['neg'].items()},
        positive_control_repro={k: {kk: vv for kk, vv in v.items() if kk != 'diffs_pct'}
                                for k, v in res['repro'].items()},
        sparse_crosscheck_report_only={
            'here': {k: v for k, v in sc['here'].items() if k != 'diffs_pct'},
            'b20_primary_b': sc['b20_primary_b'],
            'same_direction': sc['same_direction']},
        counters_absent=list(G.ABSENT_COUNTERS),
        counters_present=list(G.PRESENT_SNAP_KEYS),
        followup=trig, label=label, label_text=text)
    with open(OUT, 'w') as fh:
        json.dump(verdict, fh, indent=1, ensure_ascii=False, default=float)
    print(f'\n→ {OUT}\n→ {PAIRS}')


if __name__ == '__main__':
    main()
