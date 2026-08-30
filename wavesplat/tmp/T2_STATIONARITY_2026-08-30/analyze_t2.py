"""T2-STAT 분석기 — ⛔**판정은 `gates_t2.analyze()` 가 한다.** 여기는 입출력뿐이다.

    python3 tmp/T2_STATIONARITY_2026-08-30/analyze_t2.py            # -> verdict_t2stat.json
    python3 tmp/T2_STATIONARITY_2026-08-30/analyze_t2.py smoke      # 스모크 행으로 게이트만

⛔게이트가 하나라도 실패하면 `label` 은 `None` 이고, **그 상태로 닫는다**(선례 `M-8`).
결과를 보고 게이트를 고치지 않는다.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gates_t2 as G                                                # noqa: E402

RESULTS = os.path.join(HERE, 'results_t2.jsonl')
SMOKE = os.path.join(HERE, 'smoke_t2.jsonl')
VERDICT = os.path.join(HERE, 'verdict_t2stat.json')


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == 'smoke'
    path = SMOKE if smoke else RESULTS
    rows = load(path)
    print(f'{os.path.basename(path)}: {len(rows)} 행'
          f'{" ⚠스모크(판정 아님)" if smoke else ""}')

    if smoke:
        # ⚠스모크는 셀이 1개다 ⟹ 표본 게이트는 당연히 실패한다. **발화 여부만** 본다.
        for n, p, d in G.gates(rows):
            print(f'  [{"PASS" if p else "FAIL"}] {n}: {d}')
        for r in rows:
            if 'error' in r:
                print(f'  ⛔ERROR {r["arm"]}: {r["error"]}')
                continue
            print(f'  {r["arm"]:12s} fc={r["fc_mean"]:.4f} lead0={r["lead0"]:.4f} '
                  f'refit={r["refit_calls"]} reinit={r["reinit_calls"]} N={r["n_atoms"]} '
                  f'anchor={r["anchor_bitmatch"]} win_n={r["refit_window_n"]} '
                  f'{r["wall_s"]:.1f}s')
        return

    v = G.analyze(rows)
    json.dump(v, open(VERDICT, 'w'), indent=1, ensure_ascii=False)

    print('\n=== 게이트 ===')
    for g in v['gates']:
        print(f'  [{"PASS" if g["pass"] else "FAIL"}] {g["name"]}: {g["detail"]}')
    npass = sum(1 for g in v['gates'] if g['pass'])
    print(f'  ⟹ {npass}/{len(v["gates"])}')

    print('\n=== 판정 ===')
    t, m = v['thresholds'], v['mde']
    print(f"  문턱 n={t['n']} CONFIRM={t['CONFIRM']} NOMINAL={t['NOMINAL']} "
          f"NOT_EARN={t['NOT_EARN']} · MDE q80={m['q80']}")
    p, s = v['primary'], v['secondary_a']
    print(f"  PRIMARY   PROP-REFIT2 vs 참 제로필드: {p['better']}/{p['n']} "
          f"비={p['ratio_median']} p={p['p_one_sided']:.5f}")
    print(f"  SEC-A     PROP-REFIT2 < PROP-FROZEN : {s['better']}/{s['n']} "
          f"rel={s['rel_median_pct']}% p={s['p_one_sided']:.5f}")
    b = v['secondary_b']
    print(f"  SEC-B ⚠탐색 상호작용 g_prop<g_ship: {b['prop_more']}/{b['n']} "
          f"(g_prop={b['g_prop_median']}, g_ship={b['g_ship_median']})")
    print('\n  나우캐스트(서술 전용, ⛔판정 아님):')
    for a, d in v['descriptive_nowcast'].items():
        print(f'    {a:12s} {d["better"]}/{d["n"]} 비={d["ratio_median"]}')

    if v['blocked_by']:
        print(f'\n⛔라벨 없이 닫힘 — 실패 게이트 {len(v["blocked_by"])}건: {v["blocked_by"]}')
    else:
        print(f'\n★{v["label"]}\n   {v["why"]}')
        print(f'\n★{v["label_secondary"]}\n   {v["why_secondary"]}')
    print(f'\n방향: {v["direction"]}')
    print(f'-> {VERDICT}')


if __name__ == '__main__':
    main()
