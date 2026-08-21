"""S2 채점기 — `PREREG.md` §3/§4 의 기계적 실행.

⛔결과를 읽기 전에 동결된다(md5 를 `PREREG_frozen_md5.txt` 에 기록).
⛔`analyze_s1.py` 를 복사하지 않았다. §4 의 10개 의무 항목을 하나씩 세어가며 새로 쓴다.

⚠**사전등록 이후 추가한 보고 항목 1개 (판정 규칙 아님, 배치 채점 전에 추가)**:
스모크 2런에서 alpha_wave nRMSE 수준이 **1.0 근처**로 나왔다. nRMSE = 1.0 은
**"항상 0 을 내놓는 예측기"** 와 같은 성능이다. 따라서 Δ 만 보고하면
"덜 쓸모없다"를 "낫다"로 읽을 위험이 있다 ⟹ **무기술 널(=1.0) 대비 수준을 의무 출력**한다.
이건 대조·검정·문턱을 바꾸지 않는다. 순수 보고 추가다.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('S2_OUT', os.path.join(HERE, 'results.jsonl'))
S1 = os.path.abspath(os.path.join(HERE, '..', 'S1_indep_sparse_2026-08-08'))
GATE = os.path.join(S1, 'gate.txt')
S1_MDE = os.path.join(S1, 'MDE_prespec.json')

TREAT, CTRL, DUP = 'AUTO', 'AUTO-NOTRACK', 'AUTO-DUP'
LEADS = ('0.1', '0.2', '0.3', '0.5')
PRIMARY_LEAD = '0.2'                 # STUDY_GUIDE_2026-07-06 튜닝값
SIGN_THRESHOLD = 13                  # n=16, Holm-2 (alpha=0.025)
MDE_MULT = 2.1279493432578005
MEDIAN_SE = 1.2533141373155001
D2_N3 = 1.693

REQUIRED = ['1 AUTO vs AUTO-DUP', '2 cfg 유일', '3 288/288·에러0', '4 게이트 재수록',
            '5 절대·상대Δ·MDE·동점', '6 시드별+mean-of-3', '7 M1·M2 나란히',
            '8 리드 4개 전부', '9 판정 문자열', '10 스텐실 검증', '+무기술 널 대비 수준']
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
    valid = sorted([x for x in pairs if np.isfinite(x[1])], key=lambda x: x[1])
    m, out, run = len(valid), {}, 0.0
    for i, (k, p) in enumerate(valid):
        adj = min(1.0, max(run, (m - i) * p))
        run = adj
        out[k] = adj
    for k, _ in pairs:
        out.setdefault(k, float('nan'))
    return out


def regime_of(ep):
    n = int(re.search(r'_(\d+)comp', ep).group(1))
    return 'crowded' if n >= 15 else 'sparse'


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
    s1mde = json.load(open(S1_MDE))['by_regime']['nRMSE']

    # 값 테이블: eta 와 리드별 alpha (norm='episode' = PRIMARY, 'lead' = 강건성)
    V = defaultdict(dict)          # (metric, cell, ep) -> {seed: val}
    for r in recs:
        V[('eta', r['cell'], r['episode'])][r['seed']] = r['nRMSE']
        for L in LEADS:
            a = (r.get('alpha_episode') or {}).get(L)
            b = (r.get('alpha_per_lead') or {}).get(L)
            if a is not None:
                V[(f'a{L}', r['cell'], r['episode'])][r['seed']] = a
            if b is not None:
                V[(f'aL{L}', r['cell'], r['episode'])][r['seed']] = b
    M = {k: float(np.median(list(v.values()))) for k, v in V.items() if v}

    print('=' * 100)
    print('S2 채점 — PREREG.md §3/§4 의 기계적 실행')
    print('=' * 100)
    print(f'레코드 {len(recs)} · 에러 {len(errs)} · 셀 {cells} · 에피소드 {len(eps)} · 시드 {seeds}')
    expect = len(cells) * len(eps) * len(seeds)
    got = len({(r['cell'], r['episode'], r['seed']) for r in recs})
    print(f'[§4.3] 조합 {got}/{expect}' + ('  ✅' if got == expect and not errs else '  ⛔'))
    for e in errs[:3]:
        print('   에러:', e.get('job_id'), str(e.get('error'))[:110])
    did('3 288/288·에러0')

    dif = [abs(V[('eta', TREAT, e)][s] - V[('eta', DUP, e)][s])
           for e in eps for s in seeds
           if s in V[('eta', TREAT, e)] and s in V[('eta', DUP, e)]]
    dup_ok = bool(dif) and max(dif) == 0.0
    print(f'[§4.1] {TREAT} vs {DUP}: {len(dif)} 쌍 worst |Δ| = {max(dif) if dif else float("nan"):.3e}'
          + ('  ✅' if dup_ok else '  ⛔'))
    did('1 AUTO vs AUTO-DUP')

    cf = defaultdict(set)
    for r in recs:
        cf[r['cell']].add(json.dumps(r.get('cfg'), sort_keys=True))
    bad = {c: len(v) for c, v in cf.items() if len(v) != 1}
    print(f'[§4.2] 셀당 cfg 유일: ' + ('✅ 전부 1' if not bad else f'⚠ {bad}'))
    did('2 cfg 유일')

    st = [r['alpha_stencil_err'] for r in recs if r.get('alpha_stencil_err') is not None]
    print(f'[§4.10] 스텐실 검증 — GT 해석미분 대비 GT 스텐실 오차 = '
          f'{np.median(st):.4f} (중앙, GT std 대비)'
          f'   ⟹ ±2 m 스텐실이 기울기를 {np.median(st)*100:.1f}% 만큼 평활한다.')
    print('        ★양 팔과 GT 에 **동일 스텐실**을 걸었으므로 대조는 공정하나, '
          '**절대 nRMSE 는 그만큼 부풀려져 있다.**')
    did('10 스텐실 검증')

    # ---------------- 수준 + 무기술 널 ----------------
    print('\n' + '=' * 100)
    print('수준 (median-of-3 의 레짐 중앙)   ⛔alpha 와 eta 는 서로 다른 양이다 — 값 비교 금지')
    print('   ★alpha_wave nRMSE = 1.0 은 "항상 0 을 내놓는 예측기"와 동일 성능(무기술 널)')
    print('=' * 100)
    print(f'{"레짐":<9s}{"셀":<15s}{"eta nRMSE":>11s}' + ''.join(f'{"a@"+L:>10s}' for L in LEADS)
          + f'{"널대비@0.2":>12s}')
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        for c in (TREAT, CTRL):
            lv = lambda m: float(np.median([M[(m, c, e)] for e in E if (m, c, e) in M]))
            a02 = lv(f'a{PRIMARY_LEAD}')
            print(f'{rg:<9s}{c:<15s}{lv("eta"):>11.4f}'
                  + ''.join(f'{lv("a"+L):>10.4f}' for L in LEADS)
                  + f'{"기술없음" if a02 >= 1.0 else f"{(1-a02)*100:>7.1f}%":>12s}')
    did('+무기술 널 대비 수준')

    # ---------------- 대조표 ----------------
    print('\n' + '=' * 100)
    print(f'§3 짝 Δ = ({TREAT} − {CTRL}), 에피소드당 median-of-3, 레짐 안에서 Δ 의 중앙')
    print('   Δ<0 = 추적기가 그 지표를 **개선**한다 / Δ>0 = 악화')
    print('=' * 100)
    RES_ = {}
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        print(f'\n### {rg.upper()}  (n={len(E)})')
        print(f'{"지표":<12s}{"절대Δ":>10s}{"상대Δ":>10s}{"부호(-)":>10s}{"동점":>5s}'
              f'{"p":>10s}{"MDE_agg":>10s}{"|Δ|/MDE":>9s}  MDE출처')
        for m, label in [('eta', 'eta(전역)')] + [(f'a{L}', f'alpha@{L}s') for L in LEADS]:
            da, dr = [], []
            for e in E:
                if (m, TREAT, e) in M and (m, CTRL, e) in M:
                    b = M[(m, CTRL, e)]
                    da.append(M[(m, TREAT, e)] - b)
                    dr.append((M[(m, TREAT, e)] - b) / b if b else np.nan)
            s = sign_test(da)
            medA = float(np.median(da)) if da else np.nan
            medR = float(np.nanmedian(dr)) if dr else np.nan
            if m == 'eta':
                mde_agg, src = s1mde[rg]['mde_agg'], 'S1 사전동결'
            else:
                rng = [max(V[(m, CTRL, e)].values()) - min(V[(m, CTRL, e)].values())
                       for e in E if len(V[(m, CTRL, e)]) >= 2]
                sd = (float(np.median(rng)) / D2_N3) * np.sqrt(2) * 1.16 / np.sqrt(3)
                mde_agg = MDE_MULT * MEDIAN_SE * sd / np.sqrt(len(E))
                src = '⚠배치내(사전동결 아님)'
            RES_[(rg, m)] = dict(medA=medA, medR=medR, mde=mde_agg, **s)
            print(f'{label:<12s}{medA:>+10.4f}{medR*100:>+9.1f}%{s["neg"]:>7d}/{s["n"]:<3d}'
                  f'{s["ties"]:>5d}{s["p"]:>10.2e}{mde_agg:>10.4f}'
                  f'{abs(medA)/mde_agg:>9.2f}  {src}')
    did('5 절대·상대Δ·MDE·동점')
    did('7 M1·M2 나란히')
    did('8 리드 4개 전부')

    # ---------------- 판정 ----------------
    print('\n' + '=' * 100)
    print('판정 (PREREG §2.1) — 문턱 13/16, PRIMARY = alpha@0.2s, Holm-2 로 레짐 2개 묶음')
    print('=' * 100)
    H = holm([(rg, RES_[(rg, f'a{PRIMARY_LEAD}')]['p']) for rg in ('crowded', 'sparse')])
    verdicts = {}
    for rg in ('crowded', 'sparse'):
        eta, a = RES_[(rg, 'eta')], RES_[(rg, f'a{PRIMARY_LEAD}')]
        m2_repro = (eta['medA'] > 0) and (eta['pos'] >= SIGN_THRESHOLD)
        m1_helps = (a['medA'] < 0) and (a['neg'] >= SIGN_THRESHOLD) and (H[rg] <= 0.025)
        m1_hurts = (a['medA'] > 0) and (a['pos'] >= SIGN_THRESHOLD) and (H[rg] <= 0.025)
        if not m2_repro:
            v = (f'INVALID — 레거시 지표가 추적기 벌점을 재현하지 못했다 '
                 f'(eta Δ {eta["medA"]:+.4f}, 양수 {eta["pos"]}/{eta["n"]})')
        elif m1_helps:
            v = ('★METRIC-ARTIFACT — 같은 런에서 전역 eta 는 추적기를 벌하고 '
                 'alpha_wave 는 보상한다 ⟹ "추적기가 해롭다"는 **제품이 쓰지 않는 양을 채점한 결과**')
        elif m1_hurts:
            v = 'ROBUST — 제어 지표에서도 추적기가 해롭다. 재프레이밍은 죽는다.'
        else:
            v = (f'ROBUST(널) — 제어 지표에서 차이가 문턱 미달 '
                 f'(alpha Δ {a["medA"]:+.4f}, 부호 {a["neg"]}/{a["n"]}, Holm-2 p={H[rg]:.3f}) '
                 f'⟹ 추적기가 제어 지표를 개선한다는 근거 없음')
        verdicts[rg] = v
        print(f'  [{rg}] **{v}**')
        print(f'        eta Δ {eta["medA"]:+.4f} ({eta["pos"]}/{eta["n"]} 악화) · '
              f'alpha@0.2 Δ {a["medA"]:+.4f} ({a["neg"]}/{a["n"]} 개선) · Holm-2 p={H[rg]:.4f}')
    did('9 판정 문자열')
    print('\n  ⛔"METRIC-ARTIFACT" 는 **"추적기가 좋다"가 아니다.** '
          '지금까지의 채점 대상이 제품과 다르다는 뜻이다 (PREREG §2.1).')
    print('  ⛔요구 지평 0.1–0.5 s 는 **FF 제어 산출물** 기준이다. 상황인지·조종 판단이면 '
          '지평이 길어지고 E-D 의 먼바다 손상이 다시 지배한다.')

    # ---------------- 강건성: 정규화 규약 ----------------
    print('\n' + '=' * 100)
    print("정규화 규약 강건성 — norm='episode'(PRIMARY) vs norm='lead'")
    print('=' * 100)
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        row = []
        for L in LEADS:
            d = [M[(f'aL{L}', TREAT, e)] - M[(f'aL{L}', CTRL, e)] for e in E
                 if (f'aL{L}', TREAT, e) in M and (f'aL{L}', CTRL, e) in M]
            s = sign_test(d)
            row.append(f"@{L}:{np.median(d):+.4f}({s['neg']}/{s['n']})")
        print(f'  {rg:<9s} ' + '  '.join(row))

    # ---------------- 시드별 ----------------
    print('\n' + '=' * 100)
    print('[§4.6] 시드별 단독 재집계 + mean-of-3')
    print('=' * 100)
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        for m, lab in (('eta', 'eta'), (f'a{PRIMARY_LEAD}', 'alpha@0.2')):
            parts = []
            for sd in seeds:
                d = [V[(m, TREAT, e)][sd] - V[(m, CTRL, e)][sd] for e in E
                     if sd in V[(m, TREAT, e)] and sd in V[(m, CTRL, e)]]
                s = sign_test(d)
                parts.append(f"{sd}:{np.median(d):+.4f}({s['neg']}/{s['n']},p={s['p']:.3f})")
            dm = [float(np.mean(list(V[(m, TREAT, e)].values()))
                        - np.mean(list(V[(m, CTRL, e)].values()))) for e in E
                  if V[(m, TREAT, e)] and V[(m, CTRL, e)]]
            sm = sign_test(dm)
            print(f'  {rg[:4]}|{lab:<10s} ' + ' '.join(parts)
                  + f"  mean3:{np.median(dm):+.4f}({sm['neg']}/{sm['n']},p={sm['p']:.3f})")
    did('6 시드별+mean-of-3')

    print('\n' + '=' * 100)
    print('[§4.4] 실행 전 게이트 재수록 (S1 gate.txt — 같은 32 에피소드)')
    print('=' * 100)
    if os.path.exists(GATE):
        print(open(GATE).read().strip()[-900:])
        did('4 게이트 재수록')
    else:
        print('  ⛔ S1 gate.txt 없음')

    print('\n' + '=' * 100)
    print(f'[§4 자체 채점] 의무 {len(REQUIRED)} 항 중 **{len(_done)}개 실행**')
    for n in REQUIRED:
        print(f'   {"✅" if n in _done else "⛔ 미실행"}  {n}')
    print('  ' + ('✅ 전항 실행됨.' if len(_done) == len(REQUIRED)
                  else '⛔ **§4 위반 — 이 배치의 무결성 주장은 인용 불가.**'))
    print('=' * 100)


if __name__ == '__main__':
    main()
