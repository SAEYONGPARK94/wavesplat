#!/usr/bin/env python3
"""S13b analyzer -- follows PREREG.md exactly. Nothing here was chosen after seeing data.

Frozen (PREREG section 4): per-episode median of the three estimator seeds {42,43,44},
paired sign test vs the REUSED S13 LETKF+ column, ONE-SIDED (SHIP lower), alpha=0.025,
threshold >= 41/64, PRIMARY = crowded only (family size 1 -> no Holm). All three
effect-size conventions printed; verdict by SIGN only.
"""
import collections
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
S13_JSONL = os.path.join(ROOT, 'tmp', 'S13_oos_2026-08-12', 'results.jsonl')
S13B_JSONL = os.path.join(HERE, 'results_s13b.jsonl')
ALPHA = 0.025
THRESH = 41
N_PER = 64
SEEDS = (42, 43, 44)


def crit(n, alpha=ALPHA):
    """Smallest w with P(W >= w | p=0.5) <= alpha -- the frozen recomputation rule."""
    for w in range(n + 1):
        if stats.binom.sf(w - 1, n, 0.5) <= alpha:
            return w
    return n + 1


def load():
    ship, lplus, zero, meta, route = {}, {}, {}, {}, {}
    for line in open(S13_JSONL):
        r = json.loads(line)
        if 'error' in r:
            continue
        if r['arm'] == 'SHIP':
            ship.setdefault(r['episode'], {})[42] = r['avg_nRMSE']
            route.setdefault(r['episode'], {})[42] = r.get('route')
            meta[r['episode']] = r
        elif r['arm'] == 'LETKF_PLUS':
            lplus[r['episode']] = r['avg_nRMSE']
            zero[r['episode']] = r['zero_nRMSE']
    errs = []
    for line in open(S13B_JSONL):
        r = json.loads(line)
        if 'error' in r:
            errs.append(r)
            continue
        s = r['est_seed']
        ship.setdefault(r['episode'], {})[s] = r['avg_nRMSE']
        route.setdefault(r['episode'], {})[s] = r.get('route')
    return ship, lplus, zero, meta, route, errs


def sign_one_sided(a, b):
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


def report(label, a, b, na, nb, thresh):
    w, n, p = sign_one_sided(a, b)
    mr, rm, rmn = effects(a, b)
    print(f'  {label}')
    print(f'    {na} < {nb}: {w}/{n}   (문턱 {thresh}/{n})   단측 p={p:.4g}')
    print(f'    크기 3규약: med(rel)={mr:+.1f}%  rel(med)={rm:+.1f}%  rel(mean)={rmn:+.1f}%')
    return dict(wins=w, n=n, p=p, med_rel=mr, rel_med=rm, rel_mean=rmn,
                passes=bool(w >= thresh))


def main():
    ship, lplus, zero, meta, route, errs = load()
    out = {'errors': [dict(episode=e['episode'], arm=e.get('arm'),
                           error=e['error'], attempt=e.get('attempt')) for e in errs]}

    complete = {k for k, v in ship.items() if all(s in v for s in SEEDS)}
    dropped = sorted(set(meta) - complete)
    strata = collections.defaultdict(list)
    for k in sorted(complete):
        strata[meta[k]['stratum']].append(k)

    print('=' * 78)
    print('S13b PRIMARY — S13 판정의 추정기-시드 안정성 (median-of-3: 42/43/44)')
    print('사전등록 PREREG.md 동결: 단측 α=0.025, 문턱 41/64, PRIMARY=혼잡 1개, 판정은 부호')
    print('=' * 78)
    print(f'\n에러 런 {len(errs)}개 · median-of-3 미완성 에피소드 {len(dropped)}개 {dropped[:6]}')

    med = {k: float(np.median([ship[k][s] for s in SEEDS])) for k in complete}
    out['n_complete'] = {s: len(v) for s, v in strata.items()}

    # ── PRIMARY (crowded) + SECONDARY S-2s (sparse), same protocol ────────────
    for stratum, tag in (('crowded', 'PRIMARY'), ('sparse', 'SECONDARY S-2s')):
        ks = strata[stratum]
        n = len(ks)
        th = THRESH if n == N_PER else crit(n)
        if n != N_PER:
            print(f'\n⚠ {stratum}: n={n} ≠ 64 ⟹ 문턱 재계산 규칙 발동 → {th}/{n}')
        a = [med[k] for k in ks]
        b = [lplus[k] for k in ks]
        z = [zero[k] for k in ks]
        print(f'\n[{stratum}]  {tag}   n={n}')
        r = report('SHIP_med3 vs LETKF+', a, b, 'SHIP_med3', 'LETKF+', th)
        rz = report('필요조건 SHIP_med3 vs ZERO (IUT)', a, z, 'SHIP_med3', 'ZERO', th)
        verdict = ('PASS' if (r['passes'] and r['p'] < ALPHA and rz['passes'])
                   else 'FAIL')
        if len(dropped) > 5:
            verdict = 'UNDECIDED'
        print(f'    ⟹ {tag} 판정 = {verdict}   (문턱 {th}/{n}, ZERO '
              f'{"통과" if rz["passes"] else "미달"})')
        out[stratum] = dict(primary=r, zero=rz, verdict=verdict, thresh=th, n=n)

    # ── S-2p per-seed win ratios (NOT used for the verdict) ───────────────────
    print('\n' + '=' * 78)
    print('SECONDARY (PRIMARY 에 미기여)')
    print('=' * 78)
    print('\n[S-2p] 시드별 개별 승패비 vs LETKF+ — ⚠판정에 쓰지 않는다')
    print(f'  {"층":8s} {"seed42":>18s} {"seed43":>18s} {"seed44":>18s} {"med3":>18s}')
    for stratum in ('crowded', 'sparse'):
        ks = strata[stratum]
        b = [lplus[k] for k in ks]
        cells = []
        for s in SEEDS:
            w, n, p = sign_one_sided([ship[k][s] for k in ks], b)
            mr = effects([ship[k][s] for k in ks], b)[0]
            cells.append(f'{w}/{n} {mr:+.1f}%')
            out.setdefault('per_seed', {}).setdefault(stratum, {})[s] = dict(
                wins=w, n=n, p=p, med_rel=mr)
        w, n, p = sign_one_sided([med[k] for k in ks], b)
        mr = effects([med[k] for k in ks], b)[0]
        cells.append(f'{w}/{n} {mr:+.1f}%')
        print(f'  {stratum:8s} ' + ' '.join(f'{c:>18s}' for c in cells))

    # ── S-2v seed-to-seed dispersion ──────────────────────────────────────────
    print('\n[S-2v] 시드 간 변동계수 CV = 100·sd(3,ddof=1)/mean(3)   '
          '(저장소 기록 53% = 2026-07-23 트래커수정 *이전* 상한)')
    for stratum in ('crowded', 'sparse'):
        ks = strata[stratum]
        cv = np.array([100 * np.std([ship[k][s] for s in SEEDS], ddof=1)
                       / np.mean([ship[k][s] for s in SEEDS]) for k in ks])
        print(f'  {stratum:8s} n={len(ks)}  중앙 {np.median(cv):5.1f}%  '
              f'평균 {cv.mean():5.1f}%  p90 {np.percentile(cv, 90):5.1f}%  '
              f'최대 {cv.max():5.1f}%  (>10%: {int((cv>10).sum())}, >53%: {int((cv>53).sum())})')
        out.setdefault('cv', {})[stratum] = dict(
            median=float(np.median(cv)), mean=float(cv.mean()),
            p90=float(np.percentile(cv, 90)), max=float(cv.max()),
            n_gt10=int((cv > 10).sum()), n_gt53=int((cv > 53).sum()))
    by = collections.defaultdict(list)
    for k in complete:
        cv = 100 * np.std([ship[k][s] for s in SEEDS], ddof=1) / np.mean(
            [ship[k][s] for s in SEEDS])
        by[meta[k]['ncomp']].append(cv)
    print('  성분수별 CV 중앙: ' + '  '.join(
        f'{c}comp {np.median(v):.1f}%' for c, v in sorted(by.items())))

    # ── PREREG section-5 case 5: is seed 42 systematically lucky? ─────────────
    print('\n[PREREG §5-5] seed=42 가 체계적으로 유리한 추첨이었나 '
          '(2026-07-23 은 n=4 에서 1·1·2·8위)')
    for stratum in ('crowded', 'sparse'):
        ks = strata[stratum]
        rk = {s: [] for s in SEEDS}
        for k in ks:
            order = sorted(SEEDS, key=lambda s: ship[k][s])   # 1 = best (lowest nRMSE)
            for i, s in enumerate(order, 1):
                rk[s].append(i)
        print(f'  {stratum:8s} ' + '  '.join(
            f'seed{s}: 평균순위 {np.mean(rk[s]):.2f} 최선 {rk[s].count(1)}/{len(ks)}'
            for s in SEEDS))
        out.setdefault('rank42', {})[stratum] = {
            str(s): dict(mean_rank=float(np.mean(rk[s])), n_best=int(rk[s].count(1)))
            for s in SEEDS}

    # ── S-2e pre-exposure sensitivity ─────────────────────────────────────────
    print('\n[S-2e] 사전노출 24 에피소드 제외 민감도 (median-of-3)')
    for stratum in ('crowded', 'sparse'):
        ks = [k for k in strata[stratum] if not meta[k]['pre_exposed']]
        w, n, p = sign_one_sided([med[k] for k in ks], [lplus[k] for k in ks])
        print(f'  {stratum:8s} 미노출 n={n}: {w}/{n}  문턱 {crit(n)}/{n}  단측 p={p:.4g}  '
              f'{"통과" if w >= crit(n) else "미달"}')
        out.setdefault('unexposed', {})[stratum] = dict(wins=w, n=n, p=p, thresh=crit(n))

    # ── S-2f router stability ─────────────────────────────────────────────────
    print('\n[S-2f] 라우터 경로 시드 안정성')
    flips = collections.Counter()
    for k in complete:
        rs = tuple(route[k][s] for s in SEEDS)
        flips[(meta[k]['stratum'], rs)] += 1
    n_unstable = sum(v for (st, rs), v in flips.items() if len(set(rs)) > 1)
    for kk, vv in sorted(flips.items(), key=lambda x: str(x[0])):
        print(f'  {kk[0]:8s} routes(42,43,44)={kk[1]}: {vv}')
    print(f'  ⟹ 시드에 따라 경로가 바뀐 에피소드 {n_unstable}/{len(complete)}')
    out['route_unstable'] = int(n_unstable)

    # ── cell breakdown (descriptive) ──────────────────────────────────────────
    print('\n[셀별 분해 — 기술통계, median-of-3]')
    cells = collections.defaultdict(list)
    for k in complete:
        cells[meta[k]['ncomp']].append(k)
    for c in sorted(cells):
        ks = cells[c]
        a = np.array([med[k] for k in ks])
        b = np.array([lplus[k] for k in ks])
        s42 = np.array([ship[k][42] for k in ks])
        print(f'  ncomp={c:2d} n={len(ks):2d}  med3<LETKF+ {int((a<b).sum()):2d}/{len(ks)}  '
              f'(S13 seed42 {int((s42<b).sum()):2d}/{len(ks)})  '
              f'med(rel)={100*np.median((a-b)/b):+7.1f}%  '
              f'(S13 {100*np.median((s42-b)/b):+7.1f}%)')

    json.dump(out, open(os.path.join(HERE, 'verdict_s13b.json'), 'w'), indent=1)
    print('\n-> verdict_s13b.json')


if __name__ == '__main__':
    main()
