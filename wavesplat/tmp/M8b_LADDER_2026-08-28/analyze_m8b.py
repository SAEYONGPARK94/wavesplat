"""M-8b 분석기 — ⛔**여기에 문턱·팔·라벨이 없다.** 전부 `gates_m8b` → `gates_m8` 을 부른다.

    python3 tmp/M8b_LADDER_2026-08-28/analyze_m8b.py        # → verdict_m8b.json

★**신규 판정 런 0회** — 읽는 것은 `M8` 동결 432행(`source_results` 로 계보를 선언한다)과
이 배치가 만든 앵커 재현 24행뿐이다.
★**0행 가드**: 원천이 없거나 사다리 런이 부족하면 **판정하지 않고 exit 1**.
★**차단/범위축소 분리**: `SCOPING`(승계) 밖의 게이트가 하나라도 FAIL 이면 판정 없음.
  ⚠단 `G-END` 만 실패한 경우는 사전등록된 라벨 그 자체다(`gates_m8.chain_label`).
⛔`label` 이 `null` 로 끝나면 **결과 파일을 열지 마라**(`M5` 의 교훈).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, os.path.join(ROOT, 'tmp', 'M8_LADDER_2026-08-28'), ROOT,
           os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_m8b as G                                         # noqa: E402
import gates_m8 as G8                                         # noqa: E402

OUT = os.path.join(HERE, 'verdict_m8b.json')


def load():
    return G.source_rows()


def _source_decl():
    """★재집계 계보 선언 — `tmp/_audit/check_batch_artifacts.py` 가 **실재·행 유무를
    직접 확인**한다(선언만으로는 통과하지 못한다)."""
    import glob
    base = os.path.dirname(HERE)                      # …/tmp
    out = [os.path.relpath(p, base)
           for p in sorted(glob.glob(G.SOURCE_GLOB))]
    out.append(os.path.relpath(G.ANCHOR_JSONL, base))
    return out


def _dump(obj):
    with open(OUT, 'w') as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)


def main():
    rows = load()
    ok = [r for r in rows if 'error' not in r]
    if not rows:
        print('⛔원천 0행 — 판정하지 않는다.')
        return 1
    lad_ok = [r for r in ok if r.get('arm') in G8.RUNGS]
    need = len(G8.RUNGS) * len(G8.RECORDS) * len(G8.EST_SEEDS)
    if len(lad_ok) < need:
        print(f'⛔사다리 런 부족 {len(lad_ok)}/{need} (전체 {len(ok)}/{G8.N_RUNS}) '
              '— 판정하지 않는다.')
        return 1

    grows = G.gates(rows)
    print(f'=== 게이트 ({len(grows)}개 · 교체 {G.REPLACED_GATE} → {list(G.NEW_GATES)}) ===')
    for gid, good, msg in grows:
        head = gid.split()[0]
        tag = 'PASS' if good else ('FAIL/SCOPE' if head in G.SCOPING else 'FAIL')
        print(f'  [{tag:10s}] {gid}: {msg[:110]}   (커버리지 {G.COVERAGE.get(head)})')
    blocked = G.blocking_failures(grows)
    scoped = [g for g, o, _ in grows if not o and g.split()[0] in G.SCOPING]

    hard = [b for b in blocked if b.split()[0] != 'G-END']
    if hard:
        print(f'\n⛔차단 게이트 FAIL {hard} — 판정하지 않는다.')
        print('   ' + G8.READING['S_EMPTY_ORDER'])
        _dump(dict(batch=G.BATCH, source_batch=G.SOURCE_BATCH,
                   source_results=_source_decl(),
                   label=None, primary_label=None, chain_label=None,
                   blocked=hard, scoped_out=scoped, n_rows=len(ok),
                   coverage=dict(G.COVERAGE),
                   gates=[[g, o, m] for g, o, m in grows], reading=G8.READING))
        return 1

    v = G.verdict(rows, gate_rows=grows)
    v.update(label=v['chain_label'], scoped_out=scoped,
             source_results=_source_decl(),
             n_rows=len(ok), n_error=len(rows) - len(ok),
             coverage=dict(G.COVERAGE),
             gates=[[g, o, m] for g, o, m in grows])
    if v['chain'] is None:                 # 사전등록된 「판정 없음」 분기
        _dump(v)
        e = v['endpoint']
        print(f'\n=== 판정 ===\n  {v["chain_label"]}\n  {v["chain_why"]}')
        print(f'  끝점 {e["a"]} < {e["b"]}: {e["worse"]}/{e["n"]} p={e["p_worse"]} '
              f'(문턱 {G8.THRESHOLD_PRIMARY}/{G8.N_PRIMARY})')
        print(f'  {v["primary_why"]}')
        print(f'\n-> {OUT}')
        return 1

    v.update(secondary=dict(
                 primary_leadmean=G8.paired(rows, 'BANK-B30', 'MMSE-PROP',
                                            G8.FIELD_SECOND),
                 chain_leadmean=G8.chain(rows, G8.FIELD_SECOND),
                 endpoint_leadmean=G8.paired(rows, G8.ENDPOINTS[0], G8.ENDPOINTS[1],
                                             G8.FIELD_SECOND),
                 dose_L2=[G8.paired(rows, 'BANK-K1', a) for a in G8.DOSE_ARMS[1:]],
                 band=G8.paired(rows, 'MMSE-PROP', 'MMSE-PROP-BAND'),
                 snap_on_prop=G8.paired(rows, 'PROP', 'PROP-SNAP'),
                 snap_on_prop_ls=G8.paired(rows, 'PROP-LS', 'PROP-LS-SNAP'),
                 by_box=dict(
                     large=[G8.paired([r for r in rows
                                       if r.get('record') in G8.LARGE_BOX],
                                      a, b) for _l, a, b, _w in G8.LINKS],
                     small=[G8.paired([r for r in rows
                                       if r.get('record') in G8.SMALL_BOX],
                                      a, b) for _l, a, b, _w in G8.LINKS])))
    _dump(v)

    print(f'\n=== 판정 ===\n  PRIMARY  {v["primary_label"]}\n    {v["primary_why"]}')
    p = v['primary']
    print(f'    L3 {p["a"]} vs {p["b"]}: better {p["better"]}/{p["n"]} '
          f'(p={p["p_better"]})  worse {p["worse"]}/{p["n"]} (p={p["p_worse"]})  '
          f'rel median {p["rel_median_pct"]}%')
    print(f'\n  사슬     {v["chain_label"]}\n    {v["chain_why"]}')
    print(f'    Holm 생존 = {v["holm_survivors"]}')
    for d, h in zip(v['chain'], sorted(v['holm'],
                                       key=lambda x: G8.LINK_IDS.index(x['link']))):
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
          '   ⚠둘은 **반드시 같지 않다**(레코드별 로그비의 중앙값은 가법이 아니다).')
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
    for line in G8.READING['NOT_CLAIMED']:
        print('  ' + line)
    print('  ' + G8.READING['RNG_DISCLOSURE'])
    print('  ⚠M8b 추가 공시 = `PREREG.md §5`(부분 노출 · 앵커 검증의 층 편향 · 동결의 한계)')
    print(f'\n-> {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
