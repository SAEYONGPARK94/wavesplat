"""M-5 분석기 — ⛔**여기에 문턱·라벨이 없다.** 전부 `gates_m5.py` 를 부른다.

    python3 tmp/M5_WINDOW_2026-08-28/analyze_m5.py            # → verdict_m5.json

★**0행 가드**: 결과 파일이 없거나 기대 런 수에 못 미치면 **판정을 내지 않고 종료한다**
(부분 자료로 라벨이 나오는 것이 이 저장소가 반복해서 낸 사고다).
★**차단/범위축소 분리**: `gates_m5.SCOPING` 밖의 게이트가 하나라도 FAIL 이면 판정 없음.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_m5 as G                                          # noqa: E402


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'results_m5_*.jsonl'))):
        rows += G._load_jsonl(p)
    return rows


def main():
    rows = load()
    ok = [r for r in rows if 'error' not in r]
    if not rows:
        print('⛔결과 0행 — 판정하지 않는다.'); return 2
    if len(ok) < G.N_RUNS:
        print(f'⛔런 부족 {len(ok)}/{G.N_RUNS} — 판정하지 않는다.'); return 2

    grows = G.gates(rows)
    print('=== 게이트 ===')
    for gid, good, msg in grows:
        tag = 'PASS' if good else ('FAIL/SCOPE' if gid.split()[0] in G.SCOPING
                                   else 'FAIL')
        print(f'  [{tag}] {gid}: {msg[:110]}')
    blocked = G.blocking_failures(grows)
    if blocked:
        print(f'\n⛔차단 게이트 FAIL {blocked} — 판정하지 않는다.')
        json.dump({'label': None, 'blocked': blocked,
                   'gates': [[g, o, m] for g, o, m in grows]},
                  open(os.path.join(HERE, 'verdict_m5.json'), 'w'), indent=1)
        return 1

    (lab, why), prim, disc, sec, dose = G._verdict_from(rows)
    scoped = [g for g in [gid for gid, o, _ in grows if not o]
              if g.split()[0] in G.SCOPING]

    extra = dict(
        primary_leadmean=G.paired(rows, G.PRIMARY_LEVEL, G.REF_LEVEL, G.PRIMARY_ARM,
                                  field='leadmean', wins=[G.PRIMARY_WIN]),
        sec8_prop=G.paired(rows, 'SEC8', G.REF_LEVEL, G.PRIMARY_ARM,
                           wins=[G.PRIMARY_WIN]),
        sec8_ship=G.paired(rows, 'SEC8', G.REF_LEVEL, 'SHIP', wins=[G.PRIMARY_WIN]),
        orig2_prop=G.paired(rows, 'ORIG2', G.REF_LEVEL, G.PRIMARY_ARM,
                            wins=[G.PRIMARY_WIN]),
        dose_ship=G.dose_curve(rows, 'SHIP', wins=[G.PRIMARY_WIN]),
        dose_prop_36=G.dose_curve(rows, G.PRIMARY_ARM, wins=G.WINDOWS))

    verdict = dict(batch=G.BATCH, direction=G.DIRECTION,
                   label=lab, why=why,
                   primary=prim, discriminator=disc, secondary_n36=sec,
                   dose_response=dose, extra=extra,
                   scoped_out=scoped, n_rows=len(ok),
                   gates=[[g, o, m] for g, o, m in grows])
    json.dump(verdict, open(os.path.join(HERE, 'verdict_m5.json'), 'w'), indent=1)
    print(f'\n=== 판정 ===\n  {lab}\n  {why}')
    print(f'  PRIMARY  {prim}')
    print(f'  판별자   {disc}')
    print(f'  n=36     {sec}   ⛔승격 근거 아님')
    for d in dose:
        print(f'  용량     {d["level"]}: {d["better"]}/{d["n"]} '
              f'rel={d["rel_median_pct"]}')
    if scoped:
        print(f'  ⚠범위축소: {scoped} → {[G.SCOPING[g.split()[0]] for g in scoped]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
