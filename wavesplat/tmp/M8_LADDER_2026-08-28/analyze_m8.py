"""M-8 분석기 — ⛔**여기에 문턱·팔·라벨이 없다.** 전부 `gates_m8.py` 를 부른다.

    python3 tmp/M8_LADDER_2026-08-28/analyze_m8.py          # → verdict_m8.json

★**0행 가드**: 결과 파일이 없거나 기대 런 수에 못 미치면 **판정하지 않고 exit 1**.
★**차단/범위축소 분리**: `gates_m8.SCOPING` 밖의 게이트가 하나라도 FAIL 이면 판정 없음.
  ⚠단 **`G-END` 만 실패한 경우는 다르다** — 그것은 사전등록된 라벨
  **`LADDER-VOID`** 그 자체다(`PREREG §8.2`). 그때는 라벨을 적되 사슬은 계산하지 않는다.
★**커버리지 출력**: 각 게이트가 실제로 몇 행/짝을 보았는지 찍는다(`G-COV` 가 0 을 막는다).
★**척도 셋 병기**(`F1`): 짝 승수 · 절대 nRMSE · 참 제로필드 대비를 **함께** 낸다.
⛔`label` 이 `null` 로 끝나면 **결과 파일을 열지 마라**(`M5` 의 교훈).
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT,
           os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_m8 as G                                          # noqa: E402

OUT = os.path.join(HERE, 'verdict_m8.json')


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'results_m8_*.jsonl'))):
        rows += G._load_jsonl(p)
    return rows


def _dump(obj):
    with open(OUT, 'w') as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)


def main():
    rows = load()
    ok = [r for r in rows if 'error' not in r]
    if not rows:
        print('⛔결과 0행 — 판정하지 않는다.')
        return 1
    lad_ok = [r for r in ok if r.get('arm') in G.RUNGS]
    need = len(G.RUNGS) * len(G.RECORDS) * len(G.EST_SEEDS)
    if len(lad_ok) < need:
        print(f'⛔사다리 런 부족 {len(lad_ok)}/{need} (전체 {len(ok)}/{G.N_RUNS}) '
              '— 판정하지 않는다.')
        return 1

    grows = G.gates(rows)
    print('=== 게이트 ===')
    for gid, good, msg in grows:
        head = gid.split()[0]
        tag = 'PASS' if good else ('FAIL/SCOPE' if head in G.SCOPING else 'FAIL')
        print(f'  [{tag:10s}] {gid}: {msg[:110]}   (커버리지 {G.COVERAGE.get(head)})')
    blocked = G.blocking_failures(grows)
    scoped = [g for g, o, _ in grows if not o and g.split()[0] in G.SCOPING]

    # ★`G-END` 만 실패 = 사전등록된 라벨 `LADDER-VOID`. 그 외 차단 실패 = **판정 없음**.
    hard = [b for b in blocked if b.split()[0] != 'G-END']
    if hard:
        print(f'\n⛔차단 게이트 FAIL {hard} — 판정하지 않는다.')
        print('   ' + G.READING['S_EMPTY_ORDER'])
        _dump(dict(batch=G.BATCH, label=None, primary_label=None, chain_label=None,
                   blocked=hard, scoped_out=scoped, n_rows=len(ok),
                   coverage=dict(G.COVERAGE),
                   gates=[[g, o, m] for g, o, m in grows],
                   reading=G.READING))
        return 1

    v = G.verdict(rows, gate_rows=grows)
    v.update(label=v['chain_label'], scoped_out=scoped,
             n_rows=len(ok), n_error=len(rows) - len(ok),
             coverage=dict(G.COVERAGE),
             gates=[[g, o, m] for g, o, m in grows])
    if v['chain'] is None:           # ★LADDER-VOID — 결과를 열지 않는다
        _dump(v)
        e = v['endpoint']
        print(f'\n=== 판정 ===\n  {v["chain_label"]}\n  {v["chain_why"]}')
        print(f'  끝점 {e["a"]} < {e["b"]}: {e["worse"]}/{e["n"]} p={e["p_worse"]} '
              f'(문턱 {G.THRESHOLD_PRIMARY}/{G.N_PRIMARY})')
        print(f'  {v["primary_why"]}')
        print(f'\n-> {OUT}')
        return 1
    v.update(secondary=dict(
                 primary_leadmean=G.paired(rows, 'BANK-B30', 'MMSE-PROP',
                                           G.FIELD_SECOND),
                 chain_leadmean=G.chain(rows, G.FIELD_SECOND),
                 endpoint_leadmean=G.paired(rows, G.ENDPOINTS[0], G.ENDPOINTS[1],
                                            G.FIELD_SECOND),
                 dose_L2=[G.paired(rows, 'BANK-K1', a) for a in G.DOSE_ARMS[1:]],
                 band=G.paired(rows, 'MMSE-PROP', 'MMSE-PROP-BAND'),
                 snap_on_prop=G.paired(rows, 'PROP', 'PROP-SNAP'),
                 snap_on_prop_ls=G.paired(rows, 'PROP-LS', 'PROP-LS-SNAP'),
                 by_box=dict(
                     large=[G.paired([r for r in rows
                                      if r.get('record') in G.LARGE_BOX],
                                     a, b) for _l, a, b, _w in G.LINKS],
                     small=[G.paired([r for r in rows
                                      if r.get('record') in G.SMALL_BOX],
                                     a, b) for _l, a, b, _w in G.LINKS])))
    _dump(v)

    print(f'\n=== 판정 ===\n  PRIMARY  {v["primary_label"]}\n    {v["primary_why"]}')
    p = v['primary']
    print(f'    L3 {p["a"]} vs {p["b"]}: better {p["better"]}/{p["n"]} '
          f'(p={p["p_better"]})  worse {p["worse"]}/{p["n"]} (p={p["p_worse"]})  '
          f'rel median {p["rel_median_pct"]}%')
    print(f'\n  사슬     {v["chain_label"]}\n    {v["chain_why"]}')
    print(f'    Holm 생존 = {v["holm_survivors"]}')
    for d, h in zip(v['chain'], sorted(v['holm'], key=lambda x: G.LINK_IDS.index(
            x['link']))):
        print(f'    {d["link"]}  {d["a"]:>13s} → {d["b"]:<15s} '
              f'worse {d["worse"]:2d}/{d["n"]:2d}  p={d["p_worse"]:.6f}  '
              f'α_Holm={h["alpha"]:.6f}  기각={h["rejected"]}  '
              f'rel={None if d["rel_median_pct"] is None else round(d["rel_median_pct"], 2)}%'
              f'   [{d["why"]}]')
    e = v['endpoint']
    print(f'\n  끝점     {e["a"]} < {e["b"]}: {e["worse"]}/{e["n"]} p={e["p_worse"]}')

    print('\n=== ★크기 분해 (⛔검정 아님 · 척도 셋 병기 의무) ===')
    m = v['magnitude']
    for r_ in m['per_rung']:
        print(f'    R{r_.get("rung")}  {r_["arm"]:15s} n={r_["n"]:2d}  '
              f'절대 nRMSE 중앙={r_.get("abs_nrmse_median")}  '
              f'참 제로필드 대비={r_.get("vs_true_zero_median")}')
    for l_ in m['per_link']:
        print(f'    {l_["link"]}  Δlog={l_["delta_log"]}  승수={l_["mult_median"]}  '
              f'rel={l_["rel_median_pct"]}%')
    print(f'    ΣΔ_k = {m["sum_delta_log"]}   vs  끝점 Δlog = {m["endpoint_delta_log"]}'
          '   ⚠둘은 **반드시 같지 않다**(레코드별 로그비의 중앙값은 가법이 아니다) — '
          '차이 자체가 분해의 비가법성 공시다.')
    print('    ' + m['caveat'])

    print('\n=== 시드 산포 (⛔판정 아님 · 판정문에 병기) ===')
    for sp in v['seed_spread']:
        print(f'    {sp["arm"]:15s} CV 중앙={sp["cv_median_pct"]}%  '
              f'최대={sp["cv_max_pct"]}%  (n={sp["n_records"]})')

    if scoped:
        print(f'\n  ⚠범위축소: {scoped}')
        for g in scoped:
            print(f'     → {G.SCOPING[g.split()[0]]}')
    print('\n=== ⛔주장하지 않는 것 (결과 전 공시) ===')
    for line in G.READING['NOT_CLAIMED']:
        print('  ' + line)
    print('  ' + G.READING['RNG_DISCLOSURE'])
    print(f'\n-> {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
