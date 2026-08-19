#!/usr/bin/env python3
"""S13 analyzer — follows PREREG.md §4 exactly. Nothing here was chosen after seeing data.

Frozen: paired sign test on avg_nRMSE, ONE-SIDED (SHIP lower), alpha=0.025,
Holm-2 over the two PRIMARY strata, threshold >= 41/64. All three effect-size
conventions printed, verdict by SIGN only. ZERO is an IUT necessary condition
(outside the Holm family).
"""
import collections
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
ALPHA = 0.025
THRESH = 41
N_PER = 64


def table():
    d = collections.defaultdict(dict)
    for r in ROWS:
        if 'error' in r:
            continue
        d[r['episode']][r['arm']] = r
        d[r['episode']]['_meta'] = r
    return d


def sign_one_sided(a, b):
    """H1: a < b. Returns (wins, n, p_one_sided)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (a != b)
    w = int((a[m] < b[m]).sum())
    n = int(m.sum())
    return w, n, float(stats.binomtest(w, n, 0.5, alternative='greater').pvalue)


def effects(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rel = (a - b) / b
    return (100 * np.median(rel),
            100 * (np.median(a) - np.median(b)) / np.median(b),
            100 * (a.mean() - b.mean()) / b.mean())


def holm(ps):
    order = np.argsort(ps)
    out = np.empty(len(ps))
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(ps) - i) * ps[j])
        out[j] = min(1.0, run)
    return out


def report(label, a, b, na, nb):
    w, n, p = sign_one_sided(a, b)
    mr, rm, rmn = effects(a, b)
    print(f'  {label}')
    print(f'    {na} < {nb}: {w}/{n}   (문턱 {THRESH}/{N_PER})   단측 p={p:.4g}')
    print(f'    크기 3규약 의무병기: med(rel)={mr:+.1f}%  rel(med)={rm:+.1f}%  rel(mean)={rmn:+.1f}%')
    return dict(wins=w, n=n, p=p, med_rel=mr, rel_med=rm, rel_mean=rmn,
                passes_threshold=bool(w >= THRESH))


def main():
    D = table()
    strata = {'crowded': [], 'sparse': []}
    for k, v in D.items():
        if 'SHIP' in v and 'LETKF_PLUS' in v:
            strata[v['_meta']['stratum']].append(k)
    for s in strata:
        strata[s].sort()

    out = {}
    print('=' * 78)
    print('S13 PRIMARY — 출하 구성 vs 튜닝된 LETKF (라우터 out-of-sample)')
    print('사전등록 PREREG.md §4 동결: 단측 α=0.025, Holm-2, 문턱 41/64, 판정은 부호')
    print('=' * 78)
    ps, names = [], []
    for s in ('crowded', 'sparse'):
        ks = strata[s]
        ship = [D[k]['SHIP']['avg_nRMSE'] for k in ks]
        lplus = [D[k]['LETKF_PLUS']['avg_nRMSE'] for k in ks]
        zero = [D[k]['LETKF_PLUS']['zero_nRMSE'] for k in ks]
        print(f'\n[{s}]  n={len(ks)}')
        r = report(f'PRIMARY  SHIP vs LETKF+', ship, lplus, 'SHIP', 'LETKF+')
        rz = report(f'필요조건 SHIP vs ZERO  (IUT, Holm 족 밖)', ship, zero, 'SHIP', 'ZERO')
        out[s] = dict(primary=r, zero=rz)
        ps.append(r['p'])
        names.append(s)

    adj = holm(ps)
    print('\n' + '-' * 78)
    print('Holm-2 보정 (PRIMARY 2개)')
    for nm, p0, pa in zip(names, ps, adj):
        w = out[nm]['primary']['wins']
        zpass = out[nm]['zero']['passes_threshold']
        verdict = ('PASS' if (out[nm]['primary']['passes_threshold'] and pa < ALPHA and zpass)
                   else 'FAIL')
        print(f'  {nm:8s} {w}/{N_PER}  p={p0:.4g} -> p_adj={pa:.4g}   '
              f'ZERO필요조건={"통과" if zpass else "미달"}   ⟹ {verdict}')
        out[nm]['primary']['p_holm'] = float(pa)
        out[nm]['verdict'] = verdict

    # ---------------- SECONDARY ----------------
    print('\n' + '=' * 78)
    print('SECONDARY (PRIMARY 에 미기여)')
    print('=' * 78)

    print('\n[S-2e] 사전노출 24 에피소드 제외 민감도')
    for s in ('crowded', 'sparse'):
        ks = [k for k in strata[s] if not D[k]['_meta']['pre_exposed']]
        ship = [D[k]['SHIP']['avg_nRMSE'] for k in ks]
        lplus = [D[k]['LETKF_PLUS']['avg_nRMSE'] for k in ks]
        w, n, p = sign_one_sided(ship, lplus)
        print(f'  {s:8s} 미노출 n={n}: {w}/{n}  단측 p={p:.4g}')
        out[s]['unexposed'] = dict(wins=w, n=n, p=p)

    print('\n[S-2a] 근거리 고리 `0_5` (사전 지정 1개) — SHIP 절대 RMSE, LETKF+ 는 셸 미기록')
    for s in ('crowded', 'sparse'):
        ks = strata[s]
        v = [D[k]['SHIP']['shell_rmse'].get('0_5') for k in ks]
        v = [x for x in v if x is not None and np.isfinite(x)]
        print(f'  {s:8s} n={len(v)}  중앙 {np.median(v):.4f} m' if v else f'  {s}: 값 없음')

    print('\n[셀별 분해 — 기술통계]')
    by = collections.defaultdict(list)
    for k in strata['crowded'] + strata['sparse']:
        by[D[k]['_meta']['ncomp']].append(k)
    for c in sorted(by):
        ks = by[c]
        ship = np.array([D[k]['SHIP']['avg_nRMSE'] for k in ks])
        lp = np.array([D[k]['LETKF_PLUS']['avg_nRMSE'] for k in ks])
        print(f'  ncomp={c:2d} n={len(ks):2d}  SHIP<LETKF+ {int((ship < lp).sum()):2d}/{len(ks)}  '
              f'med(rel)={100*np.median((ship-lp)/lp):+7.1f}%')

    print('\n[라우터 경로 — G-1 기록]')
    rt = collections.Counter((D[k]['_meta']['stratum'], D[k]['SHIP'].get('route'))
                             for k in strata['crowded'] + strata['sparse'])
    for kk, vv in sorted(rt.items(), key=lambda x: str(x[0])):
        print(f'  {kk[0]:8s} route={kk[1]}: {vv}')

    json.dump(out, open(os.path.join(HERE, 'verdict_primary.json'), 'w'), indent=1)
    print(f'\n-> verdict_primary.json')


if __name__ == '__main__':
    main()
