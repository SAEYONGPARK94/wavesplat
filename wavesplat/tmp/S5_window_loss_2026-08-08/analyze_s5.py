"""S5 채점기 — `PREREG.md` §3/§4 의 기계적 실행.

⛔**결과를 읽기 전에** 작성·동결한다(md5 를 `PREREG_frozen_md5.txt` 에 기록).
⛔이전 채점기를 복사하지 않았다. §4 의 11항 + PRIMARY-2(H5) + H6 을 세어가며 새로 쓴다.
"""
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('S5_OUT', os.path.join(HERE, 'results.jsonl'))
S1 = os.path.abspath(os.path.join(HERE, '..', 'S1_indep_sparse_2026-08-08'))
S4 = os.path.abspath(os.path.join(HERE, '..', 'S4_phase_oracle_2026-08-08'))

O0, ONT, ONO = 'ORA-WPHI-0STEP', 'ORA-WPHI-NT', 'ORA-WPHI-NOREG'
A, ANO, AL1, ADUP = 'AUTO-NT', 'AUTO-NT-NOREG', 'AUTO-NT-NOL1', 'AUTO-NT-DUP'
TH = 13

REQUIRED = ['1 DUP 일치', '2 cfg 유일', '3 672/672·에러0', '4 0STEP nRMSE≈0',
            '5 window_n_obs 동일', '6 게이트 재수록', '7 대조표', '8 시드별+mean-of-3',
            '9 판정 문자열(H1)', '9b 판정 문자열(H5 배포)', '10 H3 창 안팎 분해',
            '11 S4 재현성', 'H6 어느 항인가']
_done = set()


def did(x):
    _done.add(x)


def st_(d):
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


def rg_(ep):
    return 'crowded' if int(re.search(r'_(\d+)comp', ep).group(1)) >= 15 else 'sparse'


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
    reg = {e: rg_(e) for e in eps}

    METRICS = ('window_resid', 'all_resid', 'nRMSE', 'avg_obs_loss',
               'decorr_mean', 'shrink_mean')
    V = defaultdict(dict)
    for r in recs:
        for m in METRICS:
            v = r.get(m)
            if isinstance(v, (int, float)) and np.isfinite(v):
                V[(m, r['cell'], r['episode'])][r['seed']] = float(v)
    M = {k: float(np.median(list(v.values()))) for k, v in V.items() if v}

    print('=' * 108)
    print('S5 채점 — PREREG.md §3/§4 의 기계적 실행  (학습창 잔차 + 정규화항)')
    print('=' * 108)
    print(f'레코드 {len(recs)} · 에러 {len(errs)} · 셀 {cells} · 에피소드 {len(eps)} · 시드 {seeds}')
    exp = len(cells) * len(eps) * len(seeds)
    got = len({(r['cell'], r['episode'], r['seed']) for r in recs})
    print(f'[§4.3] 조합 {got}/{exp}' + ('  ✅' if got == exp and not errs else '  ⛔'))
    for e in errs[:3]:
        print('   에러:', e.get('job_id'), str(e.get('error'))[:110])
    did('3 672/672·에러0')

    dif = [abs(V[('nRMSE', A, e)][s] - V[('nRMSE', ADUP, e)][s]) for e in eps for s in seeds
           if s in V[('nRMSE', A, e)] and s in V[('nRMSE', ADUP, e)]]
    print(f'[§4.1] {A} vs {ADUP}: {len(dif)} 쌍 worst |Δ| = '
          f'{max(dif) if dif else float("nan"):.3e}'
          + ('  ✅ 누출 0' if dif and max(dif) == 0.0 else '  ⛔'))
    did('1 DUP 일치')

    cf, ng = defaultdict(set), defaultdict(set)
    for r in recs:
        c = r.get('cfg') or {}
        cf[r['cell']].add(json.dumps(c, sort_keys=True))
        ng[r['cell']].add((json.dumps(c.get('n_gaussians')),
                           re.search(r'_(\d+)comp', r['episode']).group(1)))
    print('[§4.2] 셀당 cfg 유일:')
    for c in cells:
        n = len(cf[c])
        ok = n == 1 or (len({a for a, _ in ng[c]}) == len({b for _, b in ng[c]}) == n)
        print(f'      {c:16s} {n} ' + ('✅' if ok else '⛔ 원인 불명')
              + ('' if n == 1 else '  (n_gaussians=[N_gt] 1:1)'))
    did('2 cfg 유일')

    z = [M[('nRMSE', O0, e)] for e in eps if ('nRMSE', O0, e) in M]
    print(f'[§4.4] {O0} nRMSE max = {max(z):.2e}'
          + ('  ✅' if max(z) < 1e-6 else '  ⛔ 규약 불일치 — 배치 무효'))
    print('        ⛔이 0 은 동어반복(GT 가 모델식과 같은 식). "표현 충분" 결론 금지.')
    did('4 0STEP nRMSE≈0')

    won = defaultdict(set)
    for r in recs:
        won[r['episode']].add(int(r.get('window_n_obs', -1)))
    bad = {e: v for e, v in won.items() if len(v) != 1}
    print(f'[§4.5] window_n_obs 셀 간 동일: ' +
          ('✅ 전 에피소드 단일값 (같은 창을 쟀다)' if not bad else f'⛔ {list(bad)[:3]}'))
    did('5 window_n_obs 동일')

    # ---------------- 수준표 ----------------
    print('\n' + '=' * 108)
    print('수준 (median-of-3 의 레짐 중앙)')
    print('=' * 108)
    order = [O0, ONT, ONO, A, ANO, AL1]
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        print(f'\n  {rg.upper():9s}{"window":>10s}{"all_frame":>11s}{"밖−안":>9s}'
              f'{"nRMSE":>9s}{"obs_loss":>11s}{"decorr":>9s}{"shrink":>9s}')
        for c in order:
            g = lambda m: float(np.median([M[(m, c, e)] for e in E if (m, c, e) in M]))
            w, al = g('window_resid'), g('all_resid')
            print(f'  {c:<16s}{w:>10.5f}{al:>11.5f}{al - w:>9.5f}{g("nRMSE"):>9.4f}'
                  f'{g("avg_obs_loss"):>11.6f}{g("decorr_mean"):>9.4f}{g("shrink_mean"):>9.4f}')
    print('\n  [§4.10 H3] "밖−안" = 창 밖 열화분. 창 안 열화는 window 열이 진실 대비 얼마인지로 읽는다.')
    did('7 대조표')
    did('10 H3 창 안팎 분해')

    # ---------------- 대조 ----------------
    CON = [(ONT, O0, 'window_resid', 'ORA-WPHI-NT   − 0STEP     ★H1'),
           (ONO, O0, 'window_resid', 'ORA-WPHI-NOREG− 0STEP     (H2 회복)'),
           (ONO, ONT, 'nRMSE', 'ORA-WPHI-NOREG− ORA-WPHI-NT'),
           (ANO, A, 'nRMSE', 'AUTO-NT-NOREG − AUTO-NT   ★H5 배포'),
           (AL1, A, 'nRMSE', 'AUTO-NT-NOL1  − AUTO-NT   (H6 L1)'),
           (ANO, AL1, 'nRMSE', 'AUTO-NT-NOREG − AUTO-NT-NOL1 (H6 smooth)')]
    print('\n' + '=' * 108)
    print('§3 짝 Δ (에피소드당 median-of-3, 레짐 안 Δ 중앙, 양측 정확 부호검정, 문턱 13/16)')
    print('=' * 108)
    R_ = {}
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        print(f'\n### {rg.upper()} (n={len(E)})')
        print(f'{"대조":<40s}{"지표":<14s}{"절대Δ":>11s}{"상대Δ":>10s}{"부호(-)":>10s}'
              f'{"동점":>5s}{"p":>10s}{"Holm":>8s}')
        ps = []
        for cell, base, met, lab in CON:
            da = [M[(met, cell, e)] - M[(met, base, e)] for e in E
                  if (met, cell, e) in M and (met, base, e) in M]
            dr = [(M[(met, cell, e)] - M[(met, base, e)]) / M[(met, base, e)] for e in E
                  if (met, cell, e) in M and (met, base, e) in M and M[(met, base, e)]]
            s = st_(da)
            R_[(rg, cell, base)] = dict(medA=float(np.median(da)) if da else np.nan,
                                        medR=float(np.nanmedian(dr)) if dr else np.nan,
                                        met=met, **s)
            if (cell, base) not in ((ONT, O0), (ANO, A)):
                ps.append(((rg, cell, base), s['p']))
        H = holm(ps)
        for cell, base, met, lab in CON:
            r_ = R_[(rg, cell, base)]
            hp = ('(PRIMARY)' if (cell, base) in ((ONT, O0), (ANO, A))
                  else f'{H[(rg, cell, base)]:.3f}')
            print(f'{lab:<40s}{met:<14s}{r_["medA"]:>+11.5f}{r_["medR"]*100:>+9.1f}%'
                  f'{r_["neg"]:>7d}/{r_["n"]:<3d}{r_["ties"]:>5d}{r_["p"]:>10.2e}{hp:>8s}')
    did('8 시드별+mean-of-3')  # (아래에서 실제 출력)

    # ---------------- H1 판정 ----------------
    print('\n' + '=' * 108)
    print('판정 1 (H1) — 혼잡, window_resid, ORA-WPHI-NT − ORA-WPHI-0STEP')
    print('=' * 108)
    h1 = R_[('crowded', ONT, O0)]
    h2 = R_[('crowded', ONO, O0)]
    worse_in_window = h1['medA'] > 0 and h1['pos'] >= TH
    recovered = np.isfinite(h2['medA']) and abs(h2['medA']) < abs(h1['medA']) / 2.0
    if worse_in_window and recovered:
        v1 = ('★REGULARISER — 학습창에서도 진실보다 나쁘고, **정규화항을 끄면 회복된다** '
              '⟹ 정규화항이 해를 진실에서 밀어낸다.')
    elif worse_in_window:
        v1 = ('★OPTIMISER-CANNOT-FIT — 학습창에서도 나쁜데 정규화를 꺼도 회복되지 않는다 '
              '⟹ 데이터항 자체를 못 내린다.')
    else:
        v1 = ('★WINDOW-OVERFIT — 학습창에서는 진실보다 나쁘지 않다 '
              f'(Δ {h1["medA"]:+.5f}, 양수 {h1["pos"]}/{h1["n"]}) ⟹ 창은 맞추고 밖에서 무너진다.')
    print(f'  **{v1}**')
    print(f'      H1 Δ {h1["medA"]:+.5f} ({h1["medR"]*100:+.1f}%) · 양수 {h1["pos"]}/{h1["n"]} '
          f'· p={h1["p"]:.2e}')
    print(f'      H2 회복: NOREG−0STEP Δ {h2["medA"]:+.5f} · 기준(|H1|/2 = '
          f'{abs(h1["medA"])/2:.5f}) {"충족 ✅" if recovered else "미충족 ⛔"}')
    print('  ⛔`window_resid` 는 **무가중**이다. "자기 목적함수를 못 내린다" 로 인용 금지 — '
          '허용: "학습창의 무가중 데이터 적합에서도 진실보다 나쁘다".')
    did('9 판정 문자열(H1)')

    # ---------------- H5 판정 (배포) ----------------
    print('\n' + '=' * 108)
    print('★판정 2 (H5, PRIMARY-2 배포) — 혼잡, nRMSE, AUTO-NT-NOREG − AUTO-NT')
    print('=' * 108)
    h5 = R_[('crowded', ANO, A)]
    if h5['medA'] < 0 and h5['neg'] >= TH:
        v2 = '★DEPLOYABLE — 출하 팔에서도 정규화 제거가 이긴다.'
    elif h5['medA'] > 0 and h5['pos'] >= TH:
        v2 = 'ORACLE-ONLY-HARMFUL — 오라클에서만 좋고 배포에선 해롭다.'
    else:
        v2 = ('ORACLE-ONLY-NULL — 문턱 미달 ⟹ **배포에 대해서는 아무것도 말하지 못한다.**')
    print(f'  **{v2}**')
    print(f'      Δ {h5["medA"]:+.5f} ({h5["medR"]*100:+.1f}%) · 음수 {h5["neg"]}/{h5["n"]} '
          f'(필요 {TH}) · p={h5["p"]:.2e}')
    s5 = R_[('sparse', ANO, A)]
    print(f'      (희소, 일반화 금지) Δ {s5["medA"]:+.5f} · 음수 {s5["neg"]}/{s5["n"]} '
          f'· p={s5["p"]:.2e}')
    print('  ⛔H1 이 REGULARISER 여도 H5 가 문턱 미달이면 **배포 주장 금지** (S1 G1P1 전례).')
    did('9b 판정 문자열(H5 배포)')

    # ---------------- H6 ----------------
    print('\n' + '=' * 108)
    print('H6 — 어느 항인가 (⛔둘 다 끈 팔에 단일 원인 이름 금지)')
    print('=' * 108)
    for rg in ('crowded', 'sparse'):
        a = R_[(rg, AL1, A)]
        b = R_[(rg, ANO, AL1)]
        print(f'  {rg:9s} L1 제거분 {a["medA"]:+.5f} ({a["neg"]}/{a["n"]}, p={a["p"]:.3f})   '
              f'smooth 추가제거분 {b["medA"]:+.5f} ({b["neg"]}/{b["n"]}, p={b["p"]:.3f})')
    did('H6 어느 항인가')

    # ---------------- 시드별 ----------------
    print('\n' + '=' * 108)
    print('[§4.8] 시드별 단독 재집계 + mean-of-3 (두 PRIMARY)')
    print('=' * 108)
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        for cell, base, met, lab in ((ONT, O0, 'window_resid', 'H1'), (ANO, A, 'nRMSE', 'H5')):
            parts = []
            for sd in seeds:
                d = [V[(met, cell, e)][sd] - V[(met, base, e)][sd] for e in E
                     if sd in V[(met, cell, e)] and sd in V[(met, base, e)]]
                s = st_(d)
                parts.append(f'{sd}:{np.median(d):+.5f}({s["neg"]}/{s["n"]})')
            dm = [float(np.mean(list(V[(met, cell, e)].values()))
                        - np.mean(list(V[(met, base, e)].values()))) for e in E
                  if V[(met, cell, e)] and V[(met, base, e)]]
            sm = st_(dm)
            print(f'  {rg[:4]}|{lab:<4s} ' + ' '.join(parts)
                  + f'  mean3:{np.median(dm):+.5f}({sm["neg"]}/{sm["n"]})')

    # ---------------- S4 재현성 ----------------
    print('\n' + '=' * 108)
    print('[§4.11] S4 재현성 — ORA-WPHI-NT (같은 32ep, 시드만 다름, 짝비교)')
    print('=' * 108)
    p4 = os.path.join(S4, 'results.jsonl')
    if os.path.exists(p4):
        W = defaultdict(dict)
        for line in open(p4):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r and r['cell'] == ONT:
                W[r['episode']][r['seed']] = r['nRMSE']
        Mo = {e: float(np.median(list(v.values()))) for e, v in W.items() if v}
        for rg in ('crowded', 'sparse'):
            E = [e for e in eps if reg[e] == rg and e in Mo and ('nRMSE', ONT, e) in M]
            if not E:
                continue
            d = np.array([M[('nRMSE', ONT, e)] - Mo[e] for e in E])
            a_ = np.array([M[('nRMSE', ONT, e)] for e in E])
            b_ = np.array([Mo[e] for e in E])
            print(f'   {rg:9s} 짝차 중앙 {np.median(d):+.5f}  r={np.corrcoef(a_, b_)[0, 1]:.3f}'
                  f'  (n={len(E)})')
        did('11 S4 재현성')
    else:
        print('   ⛔ S4 results.jsonl 없음')

    gate = os.path.join(S1, 'gate.txt')
    print('\n' + '=' * 108)
    print('[§4.6] 실행 전 게이트 재수록 (S1 gate.txt — 같은 32 에피소드)')
    print('=' * 108)
    if os.path.exists(gate):
        print(open(gate).read().strip()[-500:])
        did('6 게이트 재수록')
    else:
        print('   ⛔ gate.txt 없음')

    print('\n' + '=' * 108)
    print(f'[§4 자체 채점] 의무 {len(REQUIRED)} 항 중 **{len(_done)}개 실행**')
    for n in REQUIRED:
        print(f'   {"✅" if n in _done else "⛔ 미실행"}  {n}')
    print('  ' + ('✅ 전항 실행됨.' if len(_done) == len(REQUIRED)
                  else '⛔ **§4 위반 — 무결성 주장 인용 불가.**'))
    print('=' * 108)


if __name__ == '__main__':
    main()
