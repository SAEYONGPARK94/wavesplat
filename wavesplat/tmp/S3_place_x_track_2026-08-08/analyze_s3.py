"""S3 채점기 — `PREREG.md` §3/§4 의 기계적 실행.

⛔결과를 읽기 전에 동결된다(md5 를 `PREREG_frozen_md5.txt` 에 기록).
⛔`analyze_s1/s2.py` 를 복사하지 않았다. §4 의 11개 의무 항목을 세어가며 새로 쓴다.
마지막에 **자체 채점**을 찍는다 — 하나라도 빠지면 무결성 주장을 인용 불가로 표시한다.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('S3_OUT', os.path.join(HERE, 'results.jsonl'))
S1 = os.path.abspath(os.path.join(HERE, '..', 'S1_indep_sparse_2026-08-08'))
S2 = os.path.abspath(os.path.join(HERE, '..', 'S2_control_metric_2026-08-08'))

METRIC = 'nRMSE'
AUTO, ANT = 'AUTO', 'AUTO-NOTRACK'
OK, ONT = 'ORACLE-K', 'ORACLE-NOTRACK'
DUP = 'AUTO-DUP'
SIGN_THRESHOLD = 13                    # PREREG §3.4

# (셀, 기준선, 라벨). PRIMARY 는 첫 항목(혼잡).
CONTRASTS = [(ONT, ANT, 'ORACLE-NOTRACK − AUTO-NOTRACK  ★PRIMARY'),
             (ANT, AUTO, 'AUTO-NOTRACK   − AUTO'),
             (OK, AUTO, 'ORACLE-K       − AUTO'),
             (ONT, OK, 'ORACLE-NOTRACK − ORACLE-K')]
PRIMARY = (ONT, ANT)
SECONDARY = [c for c in CONTRASTS if (c[0], c[1]) != PRIMARY]

REQUIRED = ['1 AUTO vs AUTO-DUP', '2 cfg 유일(오라클 예외 확인)', '3 480/480·에러0',
            '4 게이트 재수록', '5 절대·상대Δ·MDE·동점', '6 시드별+mean-of-3',
            '7 2x2 수준표', '8 판정 문자열', '9 H2 상호작용(짝별)',
            '10 H3 decorr/shrink', '11 S1·S2 재현성']
_done = set()


def did(x):
    _done.add(x)


def sign_test(d):
    d = np.asarray([x for x in d if np.isfinite(x)], float)
    ties = int(np.sum(d == 0.0))
    d = d[d != 0.0]
    n = d.size
    neg = int(np.sum(d < 0))
    if n == 0:
        return dict(n=0, neg=0, pos=0, ties=ties, p=float('nan'))
    return dict(n=n, neg=neg, pos=n - neg, ties=ties,
                p=float(binomtest(neg, n, 0.5, alternative='two-sided').pvalue))


def holm(pairs):
    v = sorted([x for x in pairs if np.isfinite(x[1])], key=lambda z: z[1])
    m, out, run = len(v), {}, 0.0
    for i, (k, p) in enumerate(v):
        adj = min(1.0, max(run, (m - i) * p))
        run = adj
        out[k] = adj
    for k, _ in pairs:
        out.setdefault(k, float('nan'))
    return out


def regime_of(ep):
    return 'crowded' if int(re.search(r'_(\d+)comp', ep).group(1)) >= 15 else 'sparse'


def med_table(path, cells, metric=METRIC):
    V = defaultdict(dict)
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' in r or r['cell'] not in cells:
            continue
        v = r.get(metric)
        if isinstance(v, (int, float)) and np.isfinite(v):
            V[(r['cell'], r['episode'])][r['seed']] = float(v)
    return V, {k: float(np.median(list(v.values()))) for k, v in V.items() if v}


def main():
    recs, errs = [], []
    for line in open(RES):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        (errs if 'error' in r else recs).append(r)
    eps = sorted({r['episode'] for r in recs})
    cells = sorted({r['cell'] for r in recs})
    seeds = sorted({r['seed'] for r in recs})
    reg = {e: regime_of(e) for e in eps}
    mde = json.load(open(os.path.join(S1, 'MDE_prespec.json')))['by_regime'][METRIC]

    V = defaultdict(dict)
    aux = defaultdict(dict)
    for r in recs:
        V[(r['cell'], r['episode'])][r['seed']] = r[METRIC]
        for k in ('decorr_mean', 'shrink_mean', 'm_mean'):
            if isinstance(r.get(k), (int, float)) and np.isfinite(r[k]):
                aux[(k, r['cell'], r['episode'])][r['seed']] = float(r[k])
    M = {k: float(np.median(list(v.values()))) for k, v in V.items() if v}
    A = {k: float(np.median(list(v.values()))) for k, v in aux.items() if v}

    print('=' * 104)
    print('S3 채점 — PREREG.md §3/§4 의 기계적 실행   (배치 × 추적기 2x2)')
    print('=' * 104)
    print(f'레코드 {len(recs)} · 에러 {len(errs)} · 셀 {cells} · 에피소드 {len(eps)} · 시드 {seeds}')
    expect = len(cells) * len(eps) * len(seeds)
    got = len({(r['cell'], r['episode'], r['seed']) for r in recs})
    print(f'[§4.3] 조합 {got}/{expect}' + ('  ✅' if got == expect and not errs else '  ⛔'))
    for e in errs[:3]:
        print('   에러:', e.get('job_id'), str(e.get('error'))[:110])
    did('3 480/480·에러0')

    dif = [abs(V[(AUTO, e)][s] - V[(DUP, e)][s]) for e in eps for s in seeds
           if s in V[(AUTO, e)] and s in V[(DUP, e)]]
    print(f'[§4.1] {AUTO} vs {DUP}: {len(dif)} 쌍 worst |Δ| = '
          f'{max(dif) if dif else float("nan"):.3e}'
          + ('  ✅' if dif and max(dif) == 0.0 else '  ⛔'))
    did('1 AUTO vs AUTO-DUP')

    cf = defaultdict(set)
    ng = defaultdict(set)
    for r in recs:
        c = r.get('cfg') or {}
        cf[r['cell']].add(json.dumps(c, sort_keys=True))
        ng[r['cell']].add((json.dumps(c.get('n_gaussians')),
                           re.search(r'_(\d+)comp', r['episode']).group(1)))
    print('[§4.2] 셀당 cfg 유일:')
    for c in cells:
        n = len(cf[c])
        if n == 1:
            print(f'      {c:16s} 1 ✅')
        else:
            pairs = sorted(ng[c])
            one2one = len({a for a, _ in pairs}) == len({b for _, b in pairs}) == n
            print(f'      {c:16s} {n} ' + ('✅ n_gaussians=[N_gt] 와 성분수 1:1 (설계상 정상)'
                                           if one2one else '⛔ 원인 불명'))
    did('2 cfg 유일(오라클 예외 확인)')

    # ---------------- 2x2 수준표 ----------------
    print('\n' + '=' * 104)
    print('§4.7  2x2 수준표 (median-of-3 의 레짐 중앙, nRMSE 무차원)')
    print('=' * 104)
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        lv = lambda c: float(np.median([M[(c, e)] for e in E if (c, e) in M]))
        print(f'\n  {rg.upper():9s}      {"추적기 ON":>12s}{"추적기 OFF":>12s}{"차(off−on)":>12s}')
        for pl, c_on, c_off in (('배치 자동(출하)', AUTO, ANT), ('배치 오라클(묶음)', OK, ONT)):
            print(f'  {pl:<18s}{lv(c_on):>12.4f}{lv(c_off):>12.4f}{lv(c_off)-lv(c_on):>+12.4f}')
        print(f'  {"차(오라클−자동)":<18s}{lv(OK)-lv(AUTO):>+12.4f}{lv(ONT)-lv(ANT):>+12.4f}')
    did('7 2x2 수준표')

    # ---------------- 대조표 ----------------
    print('\n' + '=' * 104)
    print('§3 짝 Δ = (셀 − 기준선), 에피소드당 median-of-3, 레짐 안에서 Δ 의 중앙')
    print('=' * 104)
    R_ = {}
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        m = mde[rg]
        print(f'\n### {rg.upper()} (n={len(E)})   mde_agg={m["mde_agg"]:.5f}')
        print(f'{"대조":<38s}{"절대Δ":>10s}{"상대Δ":>10s}{"부호(-)":>10s}{"동점":>5s}'
              f'{"p":>10s}{"Holm":>8s}{"|Δ|/MDE":>9s}')
        ps = []
        for cell, base, lab in CONTRASTS:
            da = [M[(cell, e)] - M[(base, e)] for e in E if (cell, e) in M and (base, e) in M]
            dr = [(M[(cell, e)] - M[(base, e)]) / M[(base, e)] for e in E
                  if (cell, e) in M and (base, e) in M and M[(base, e)]]
            st = sign_test(da)
            R_[(rg, cell, base)] = dict(medA=float(np.median(da)) if da else np.nan,
                                        medR=float(np.nanmedian(dr)) if dr else np.nan,
                                        mde=m['mde_agg'], **st)
            if (cell, base) != PRIMARY:
                ps.append(((rg, cell, base), st['p']))
        # H2 상호작용도 Holm-4 족에 넣는다 (PREREG §3.9)
        inter = [(M[(ONT, e)] - M[(ANT, e)]) - (M[(OK, e)] - M[(AUTO, e)]) for e in E
                 if all((c, e) in M for c in (ONT, ANT, OK, AUTO))]
        sti = sign_test(inter)
        ps.append(((rg, 'INTERACTION', ''), sti['p']))
        H = holm(ps)
        for cell, base, lab in CONTRASTS:
            r_ = R_[(rg, cell, base)]
            hp = ('  (PRIMARY)' if (cell, base) == PRIMARY
                  else f'{H[(rg, cell, base)]:>8.3f}')
            print(f'{lab:<38s}{r_["medA"]:>+10.4f}{r_["medR"]*100:>+9.1f}%'
                  f'{r_["neg"]:>7d}/{r_["n"]:<3d}{r_["ties"]:>5d}{r_["p"]:>10.2e}{hp}'
                  f'{abs(r_["medA"])/r_["mde"]:>9.2f}')
        R_[(rg, 'INTERACTION')] = dict(med=float(np.median(inter)) if inter else np.nan,
                                       holm=H[(rg, 'INTERACTION', '')], **sti)
        print(f'{"H2 짝별 상호작용 (2x2)":<38s}'
              f'{R_[(rg,"INTERACTION")]["med"]:>+10.4f}{"—":>10s}'
              f'{sti["neg"]:>7d}/{sti["n"]:<3d}{sti["ties"]:>5d}{sti["p"]:>10.2e}'
              f'{R_[(rg,"INTERACTION")]["holm"]:>8.3f}'
              f'{abs(R_[(rg,"INTERACTION")]["med"])/m["mde_agg"]:>9.2f}')
    did('5 절대·상대Δ·MDE·동점')
    did('9 H2 상호작용(짝별)')

    # ---------------- 판정 ----------------
    print('\n' + '=' * 104)
    print('판정 (PREREG §1) — PRIMARY = 혼잡 ORACLE-NOTRACK − AUTO-NOTRACK, 문턱 13/16')
    print('=' * 104)
    p = R_[('crowded', ONT, ANT)]
    if p['medA'] < 0 and p['neg'] >= SIGN_THRESHOLD:
        v = ('★MASKED — 추적기를 끄자 배치 이득이 드러났다 ⟹ 배치 여유는 있었고 '
             '추적기가 가리고 있었다. 배치 축 재개.')
    elif p['medA'] > 0 and p['pos'] >= SIGN_THRESHOLD:
        v = 'ORACLE-HARMFUL — 오라클 묶음이 추적기 없이는 오히려 해롭다.'
    else:
        v = ('PLACEMENT-EXHAUSTED — 추적기를 꺼도 배치가 벌지 못한다 ⟹ 출하 팔에서 배치는 '
             '진짜 소진. 남은 오차는 배치도 추적기도 아닌 곳에 있다.')
    print(f'  **{v}**')
    print(f'      Δ {p["medA"]:+.4f} · 부호 {p["neg"]}/{p["n"]} (필요 {SIGN_THRESHOLD}) · '
          f'p={p["p"]:.3e} · |Δ|/MDE={abs(p["medA"])/p["mde"]:.2f}')
    print('  ⛔ORACLE-* 는 **3요인 묶음**(오라클 k + n_gaussians=[N_gt] + 1단 900스텝)이고 '
          '**출하 불가 팔**이다. "배치 효과"라는 단일 원인 이름 금지.')
    print('  ⛔이 판정을 희소로 일반화하지 않는다 (희소엔 고칠 k 결손이 사실상 없다).')
    did('8 판정 문자열')

    # ---------------- H3 오차의 성질 ----------------
    print('\n' + '=' * 104)
    print('H3 (부지표, 판정 아님) — 남은 오차는 위상 어긋남인가 진폭 축소인가')
    print('   m = (1−ρ²) + (√VR−ρ)²  =  decorr + shrink   (프레임별로 형성 후 평균)')
    print('=' * 104)
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        print(f'\n  {rg.upper():9s}{"decorr":>10s}{"shrink":>10s}{"m":>10s}{"decorr 비중":>12s}')
        for c in (AUTO, ANT, OK, ONT):
            g = lambda k: float(np.median([A[(k, c, e)] for e in E if (k, c, e) in A]))
            d, s, mm = g('decorr_mean'), g('shrink_mean'), g('m_mean')
            print(f'  {c:<16s}{d:>10.4f}{s:>10.4f}{mm:>10.4f}{d/max(d+s,1e-12)*100:>11.1f}%')
    did('10 H3 decorr/shrink')

    # ---------------- 시드별 ----------------
    print('\n' + '=' * 104)
    print('[§4.6] 시드별 단독 재집계 + mean-of-3')
    print('=' * 104)
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        for cell, base, lab in CONTRASTS:
            parts = []
            for sd in seeds:
                d = [V[(cell, e)][sd] - V[(base, e)][sd] for e in E
                     if sd in V[(cell, e)] and sd in V[(base, e)]]
                st = sign_test(d)
                parts.append(f'{sd}:{np.median(d):+.4f}({st["neg"]}/{st["n"]})')
            dm = [float(np.mean(list(V[(cell, e)].values()))
                        - np.mean(list(V[(base, e)].values()))) for e in E
                  if V[(cell, e)] and V[(base, e)]]
            stm = sign_test(dm)
            print(f'  {rg[:4]}|{lab[:30]:<31s} ' + ' '.join(parts)
                  + f'  mean3:{np.median(dm):+.4f}({stm["neg"]}/{stm["n"]})')
    did('6 시드별+mean-of-3')

    # ---------------- S1·S2 재현성 ----------------
    print('\n' + '=' * 104)
    print('[§4.11] S1·S2 재현성 — 같은 32ep 의 AUTO 를 세 배치에서 (시드만 다름, 짝비교만)')
    print('=' * 104)
    for nm, path, cs in (('S1', os.path.join(S1, 'results.jsonl'), {AUTO}),
                         ('S2', os.path.join(S2, 'results.jsonl'), {AUTO, ANT})):
        if not os.path.exists(path):
            print(f'  {nm}: 파일 없음')
            continue
        _, Mo = med_table(path, cs)
        for rg in ('crowded', 'sparse'):
            E = [e for e in eps if reg[e] == rg and (AUTO, e) in Mo and (AUTO, e) in M]
            d = np.array([M[(AUTO, e)] - Mo[(AUTO, e)] for e in E])
            a = np.array([M[(AUTO, e)] for e in E])
            b = np.array([Mo[(AUTO, e)] for e in E])
            print(f'  {nm} vs S3  AUTO {rg:8s}: 짝차 중앙 {np.median(d):+.4f}  '
                  f'r={np.corrcoef(a, b)[0, 1]:.3f}  (n={len(E)})')
        if ANT in cs:
            for rg in ('crowded', 'sparse'):
                E = [e for e in eps if reg[e] == rg and (ANT, e) in Mo and (ANT, e) in M]
                d = np.array([(M[(ANT, e)] - M[(AUTO, e)]) - (Mo[(ANT, e)] - Mo[(AUTO, e)])
                              for e in E])
                print(f'  {nm} vs S3  (NOTRACK−AUTO) 효과 {rg:8s}: 효과차 중앙 {np.median(d):+.4f}')
    did('11 S1·S2 재현성')

    gate = os.path.join(S1, 'gate.txt')
    print('\n' + '=' * 104)
    print('[§4.4] 실행 전 게이트 재수록 (S1 gate.txt — 같은 32 에피소드)')
    print('=' * 104)
    if os.path.exists(gate):
        print(open(gate).read().strip()[-700:])
        did('4 게이트 재수록')
    else:
        print('  ⛔ gate.txt 없음')

    print('\n' + '=' * 104)
    print(f'[§4 자체 채점] 의무 {len(REQUIRED)} 항 중 **{len(_done)}개 실행**')
    for n in REQUIRED:
        print(f'   {"✅" if n in _done else "⛔ 미실행"}  {n}')
    print('  ' + ('✅ 전항 실행됨.' if len(_done) == len(REQUIRED)
                  else '⛔ **§4 위반 — 이 배치의 무결성 주장은 인용 불가.**'))
    print('=' * 104)


if __name__ == '__main__':
    main()
