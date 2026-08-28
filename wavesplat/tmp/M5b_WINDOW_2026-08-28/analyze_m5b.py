"""M-5b 분석기 — ⛔**여기에 문턱·라벨이 없다.** 전부 `gates_m5b.py` 를 부른다.

    python3 tmp/M5b_WINDOW_2026-08-28/analyze_m5b.py          # → verdict_m5b.json

★**0행 가드**: 결과 파일이 없거나 기대 런 수에 못 미치면 **판정을 내지 않고 종료한다**.
★**차단/범위축소 분리**: `gates_m5b.SCOPING` 밖의 게이트가 하나라도 FAIL 이면 판정 없음.
★**커버리지 출력**: 각 게이트가 실제로 몇 행/짝을 보았는지 찍는다(`G-9` 가 0 을 막는다).
★**밀도 공변 공시**: 판정문 옆에 `density_disclosure` 표를 반드시 함께 낸다.
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

import gates_m5b as G                                         # noqa: E402


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'results_m5b_*.jsonl'))):
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
        print(f'  [{tag}] {gid}: {msg[:110]}   (커버리지 {G.COVERAGE.get(gid.split()[0])})')
    blocked = G.blocking_failures(grows)
    if blocked:
        print(f'\n⛔차단 게이트 FAIL {blocked} — 판정하지 않는다.')
        json.dump({'label': None, 'blocked': blocked,
                   'coverage': dict(G.COVERAGE),
                   'gates': [[g, o, m] for g, o, m in grows]},
                  open(os.path.join(HERE, 'verdict_m5b.json'), 'w'), indent=1)
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
                   supersedes=G.PRIOR_VOID_BATCH,
                   cap_mode=G.CAP_MODE,
                   label=lab, why=why,
                   primary=prim, discriminator=disc, secondary_n36=sec,
                   dose_response=dose, extra=extra,
                   density_disclosure=G.density_disclosure(rows),
                   density_caveat=G.DENSITY_CAVEAT,
                   scoped_out=scoped, n_rows=len(ok),
                   coverage=dict(G.COVERAGE),
                   gates=[[g, o, m] for g, o, m in grows])
    json.dump(verdict, open(os.path.join(HERE, 'verdict_m5b.json'), 'w'), indent=1)
    print(f'\n=== 판정 ===\n  {lab}\n  {why}')
    print(f'  PRIMARY  {prim}')
    print(f'  판별자   {disc}')
    print(f'  n=36     {sec}   ⛔승격 근거 아님')
    for d in dose:
        print(f'  용량     {d["level"]}: {d["better"]}/{d["n"]} '
              f'rel={d["rel_median_pct"]}')
    print('\n=== ★밀도 공변 공시 (⛔판정에 쓰지 않는다 · 인용 시 반드시 동반) ===')
    for d in verdict['density_disclosure']:
        print(f'  {d["level"]:7s} W={d["W_s"]}s ({d["W_over_Tp"]}Tp) '
              f'밀도={d["dens_per_s"]}/s  마지막1s={d["dens_last_1s"]}  '
              f'마지막0.2Tp={d["dens_last_02Tp"]}')
    print('  ' + G.DENSITY_CAVEAT)
    if scoped:
        print(f'  ⚠범위축소: {scoped} → {[G.SCOPING[g.split()[0]] for g in scoped]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
