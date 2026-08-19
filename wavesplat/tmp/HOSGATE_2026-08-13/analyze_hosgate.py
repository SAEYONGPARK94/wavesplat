#!/usr/bin/env python3
"""HOSGATE 분석기 — PREREG.md §4~§7 을 그대로 구현. 결과를 본 뒤 수정하지 않는다.

    python3 tmp/HOSGATE_2026-08-13/analyze_hosgate.py [results.jsonl]

동결된 것: 짝 부호검정, **단측** (H1: SHIP < LETKF_PLUS), α=0.025, PRIMARY 1개(=Holm 불필요),
n=16 문턱 **13/16**, 동점은 제외하고 줄어든 n 의 임계값을 사전표에서 읽는다.
판정은 **부호로만**. 크기는 3규약 의무 병기. ZERO 는 IUT 필요조건(Holm 족 밖).
"""
import collections
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.025
# PREREG §4 동결표 (정확 이항 단측 α=0.025 임계값). n<13 이면 판정하지 않는다.
CRIT = {13: 11, 14: 12, 15: 12, 16: 13}
CRIT12 = {10: 9, 11: 10, 12: 10}          # S-1b (BF4x2 제외 12쌍) 용
N_PRIMARY = 16


def crit_for(n, table=None):
    t = table or CRIT
    return t.get(n)


def sign_one_sided(a, b):
    """H1: a < b.  동점은 제외 (PREREG §4)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (a != b)
    w, n = int((a[m] < b[m]).sum()), int(m.sum())
    p = float(stats.binomtest(w, n, 0.5, alternative='greater').pvalue) if n else float('nan')
    return w, n, p, int((np.isfinite(a) & np.isfinite(b) & (a == b)).sum())


def effects(a, b):
    """rel = (a-b)/b. 3규약 의무 병기 (PREREG §6)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rel = (a - b) / b
    return (100 * float(np.median(rel)),
            100 * float((np.median(a) - np.median(b)) / np.median(b)),
            100 * float((a.mean() - b.mean()) / b.mean()))


def report(label, a, b, na, nb, table=None, indent='  '):
    w, n, p, ties = sign_one_sided(a, b)
    c = crit_for(n, table)
    mr, rm, rmn = effects(a, b)
    passes = (c is not None and w >= c)
    print(f'{indent}{label}')
    print(f'{indent}  {na} < {nb}: {w}/{n}   문턱 {c if c else "판정불가(n<13)"}   '
          f'단측 p={p:.4g}   동점 {ties}')
    print(f'{indent}  크기 3규약: med(rel)={mr:+.1f}%  rel(med)={rm:+.1f}%  rel(mean)={rmn:+.1f}%')
    return dict(wins=w, n=n, p=p, ties=ties, crit=c, passes=bool(passes),
                med_rel=mr, rel_med=rm, rel_mean=rmn)


def load(path):
    rows = [json.loads(l) for l in open(path)]
    D = collections.defaultdict(dict)
    for r in rows:
        if 'error' in r:
            continue
        D[(r['gt'], r['pair'])][r['arm']] = r
        D[(r['gt'], r['pair'])]['_meta'] = r
    return D


def cell(D, gt, stratum, exclude_bf4x2=False):
    ks = [k for k, v in D.items()
          if k[0] == gt and v['_meta']['stratum'] == stratum
          and {'SHIP', 'LETKF_PLUS', 'ZERO'} <= set(v)]
    if exclude_bf4x2:
        ks = [k for k in ks if not (D[k]['_meta']['bf'] == 'BF4'
                                    and D[k]['_meta']['ncomp'] == 2)]
    return sorted(ks)


def vals(D, ks, arm):
    return [D[k][arm]['avg_nRMSE'] for k in ks]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results.jsonl')
    D = load(path)
    out = {'source': os.path.basename(path)}

    print('=' * 80)
    print('HOSGATE — S13 혼잡 우위가 HOS M=3 비선형 GT 에서도 서는가')
    print('PREREG.md §4 동결: 단측 α=0.025, PRIMARY 1개, 문턱 13/16, 판정은 부호')
    print('=' * 80)

    # ── PRIMARY: HOS 혼잡 ────────────────────────────────────────────────
    res = {}
    for gt in ('HOS', 'LIN'):
        ks = cell(D, gt, 'crowded')
        if not ks:
            print(f'\n[{gt} 혼잡] 행 없음')
            res[gt] = None
            continue
        ship, lp, zero = vals(D, ks, 'SHIP'), vals(D, ks, 'LETKF_PLUS'), vals(D, ks, 'ZERO')
        tag = 'PRIMARY' if gt == 'HOS' else '귀속 가드 (PREREG §5)'
        print(f'\n[{gt} 혼잡]  n={len(ks)}   {tag}')
        r = report('SHIP vs LETKF_PLUS', ship, lp, 'SHIP', 'LETKF+')
        n1 = report('필요조건 N-1  SHIP vs ZERO', ship, zero, 'SHIP', 'ZERO')
        n2 = report('필요조건 N-2  LETKF+ vs ZERO', lp, zero, 'LETKF+', 'ZERO')
        res[gt] = dict(primary=r, N1=n1, N2=n2, n=len(ks))

    # ── §5 판정표 ────────────────────────────────────────────────────────
    print('\n' + '-' * 80)
    print('PREREG §5 판정표')
    verdict = 'NO-DATA'
    if res.get('HOS') and res.get('LIN'):
        h, l = res['HOS'], res['LIN']
        nec = h['N1']['passes'] and h['N2']['passes']
        if not nec:
            verdict = 'INCONCLUSIVE-DEGENERATE'
        elif h['primary']['passes'] and l['primary']['passes']:
            verdict = 'ROBUST'
        elif (not h['primary']['passes']) and l['primary']['passes']:
            verdict = 'REPRESENTATION-DEPENDENT'
        elif (not h['primary']['passes']) and (not l['primary']['passes']):
            verdict = 'UNDECIDED-CORPUS'
        else:
            verdict = 'ANOMALY'
        print(f'  HOS 혼잡 {h["primary"]["wins"]}/{h["primary"]["n"]} '
              f'(문턱 {h["primary"]["crit"]}, p={h["primary"]["p"]:.4g})   '
              f'LIN 혼잡 {l["primary"]["wins"]}/{l["primary"]["n"]} '
              f'(p={l["primary"]["p"]:.4g})')
        print(f'  필요조건 N-1 {"통과" if h["N1"]["passes"] else "미달"} · '
              f'N-2 {"통과" if h["N2"]["passes"] else "미달"}')
    print(f'  ⟹ **판정 = {verdict}**')
    meaning = {
        'ROBUST': 'S13 헤드라인은 선형 표현에 의존하지 않는다',
        'REPRESENTATION-DEPENDENT': '★사망 조건 발동 — S13 헤드라인은 선형 표현에 의존한다',
        'UNDECIDED-CORPUS': '선형에서도 재현 실패 ⟹ 비선형성에 귀속 불가 (코퍼스/표본)',
        'ANOMALY': '조사 대상. 어떤 헤드라인도 쓰지 않는다',
        'INCONCLUSIVE-DEGENERATE': '퇴화 런 (한 팔이 제로필드를 못 이김)',
        'NO-DATA': '행 부족',
    }[verdict]
    print(f'     {meaning}')
    print('  ⚠검정력 한계 (PREREG §4): π=0.875 -> 0.870, π=0.80 -> 0.598.')
    print('     "문턱 미달" 은 "효과 0" 이 아니다.')
    out['verdict'] = verdict
    out['crowded'] = res

    # ── SECONDARY ────────────────────────────────────────────────────────
    print('\n' + '=' * 80)
    print('SECONDARY (PRIMARY 에 미기여, α 배분 없음)')
    print('=' * 80)

    print('\n[S-1] 희소 16 쌍')
    out['sparse'] = {}
    for gt in ('HOS', 'LIN'):
        ks = cell(D, gt, 'sparse')
        if not ks:
            continue
        print(f'  {gt}  n={len(ks)}')
        out['sparse'][gt] = report('SHIP vs LETKF_PLUS', vals(D, ks, 'SHIP'),
                                   vals(D, ks, 'LETKF_PLUS'), 'SHIP', 'LETKF+',
                                   indent='    ')

    print('\n[S-1b] 희소 — BF4x2 제외 12 쌍 (PREREG §2 사전 공개 결함: 2k 절단 99.8%)')
    out['sparse_ex'] = {}
    for gt in ('HOS', 'LIN'):
        ks = cell(D, gt, 'sparse', exclude_bf4x2=True)
        if not ks:
            continue
        print(f'  {gt}  n={len(ks)}')
        out['sparse_ex'][gt] = report('SHIP vs LETKF_PLUS', vals(D, ks, 'SHIP'),
                                      vals(D, ks, 'LETKF_PLUS'), 'SHIP', 'LETKF+',
                                      table=CRIT12, indent='    ')

    print('\n[S-2] 짝 델타 — 비선형화가 누구를 더 아프게 하는가  (HOS − LIN, 양수=악화)')
    out['S2'] = {}
    for stratum in ('crowded', 'sparse'):
        kh = {k[1] for k in cell(D, 'HOS', stratum)}
        kl = {k[1] for k in cell(D, 'LIN', stratum)}
        common = sorted(kh & kl)
        if not common:
            continue
        line = {}
        for arm in ('SHIP', 'LETKF_PLUS', 'ZERO'):
            d = np.array([D[('HOS', p)][arm]['avg_nRMSE'] - D[('LIN', p)][arm]['avg_nRMSE']
                          for p in common], float)
            rel = np.array([100 * (D[('HOS', p)][arm]['avg_nRMSE']
                                   / D[('LIN', p)][arm]['avg_nRMSE'] - 1) for p in common])
            line[arm] = dict(med_abs=float(np.median(d)), med_rel=float(np.median(rel)),
                             worse=int((d > 0).sum()), n=len(common))
            print(f'  {stratum:8s} {arm:11s} n={len(common):2d}  '
                  f'중앙 Δ={np.median(d):+.5f}  중앙 상대={np.median(rel):+.1f}%  '
                  f'악화 {int((d>0).sum())}/{len(common)}')
        out['S2'][stratum] = line

    print('\n[S-3] 근거리 고리 `0_5` 셸 RMSE (SHIP 만 기록)')
    out['S3'] = {}
    for gt in ('HOS', 'LIN'):
        for stratum in ('crowded', 'sparse'):
            ks = cell(D, gt, stratum)
            v = [D[k]['SHIP'].get('shell_rmse', {}).get('0_5') for k in ks]
            v = [x for x in v if x is not None and np.isfinite(x)]
            if v:
                print(f'  {gt} {stratum:8s} n={len(v):2d}  중앙 {np.median(v):.4f} m')
                out['S3'][f'{gt}_{stratum}'] = float(np.median(v))

    print('\n[S-4] 라우터 경로 / n_gaussians')
    for gt in ('HOS', 'LIN'):
        for stratum in ('crowded', 'sparse'):
            ks = cell(D, gt, stratum)
            if not ks:
                continue
            rt = collections.Counter(D[k]['SHIP'].get('route') for k in ks)
            ng = [D[k]['SHIP'].get('n_gaussians') for k in ks]
            print(f'  {gt} {stratum:8s} route={dict(rt)}  n_gaussians 중앙={np.median(ng):.0f}')

    print('\n[S-5] 셀별 분해 (기술통계, 검정 없음)')
    for gt in ('HOS', 'LIN'):
        by = collections.defaultdict(list)
        for k in cell(D, gt, 'crowded') + cell(D, gt, 'sparse'):
            by[(D[k]['_meta']['bf'], D[k]['_meta']['ncomp'])].append(k)
        for c in sorted(by):
            ks = by[c]
            s = np.array(vals(D, ks, 'SHIP'))
            lp = np.array(vals(D, ks, 'LETKF_PLUS'))
            print(f'  {gt} {c[0]}x{c[1]:2d}comp n={len(ks)}  SHIP<LETKF+ {int((s<lp).sum())}/{len(ks)}'
                  f'  med(rel)={100*np.median((s-lp)/lp):+7.1f}%')

    # ── S-6 (동결 후 추가, 2026-08-13 스모크 직후 — SMOKE.md §사전노출 참조) ──
    # PRIMARY 논리는 **손대지 않았다**. 이 블록은 기술통계일 뿐이다.
    # 추가 이유: 스모크 2쌍(BF4_15comp_seed1581, BF4_2comp_seed1585)의 SHIP/LETKF+ 값을
    # 내가 전량 실행 전에 봤다(=사전노출). 감추는 것보다 빼고도 재는 게 정직하다.
    print('\n[S-6] 사전노출 2쌍 제외 민감도 (동결 후 추가, 기술통계 — PRIMARY 미변경)')
    EXPOSED = {'BF4_15comp_seed1581_uniform_ra_rf_ds', 'BF4_2comp_seed1585_uniform_ra_rf_ds'}
    out['S6'] = {}
    for gt in ('HOS', 'LIN'):
        for stratum in ('crowded', 'sparse'):
            ks = [k for k in cell(D, gt, stratum) if k[1] not in EXPOSED]
            if not ks:
                continue
            w, n, p, _ = sign_one_sided(vals(D, ks, 'SHIP'), vals(D, ks, 'LETKF_PLUS'))
            print(f'  {gt} {stratum:8s} 미노출 n={n:2d}: {w}/{n}  단측 p={p:.4g}')
            out['S6'][f'{gt}_{stratum}'] = dict(wins=w, n=n, p=p)

    vp = os.path.join(HERE, 'verdict.json' if 'smoke' not in path else 'verdict_smoke.json')
    json.dump(out, open(vp, 'w'), indent=1)
    print(f'\n-> {os.path.basename(vp)}')


if __name__ == '__main__':
    main()
