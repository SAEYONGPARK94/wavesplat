"""S4 채점기 — `PREREG.md` §3/§4 의 기계적 실행.

⛔**결과를 읽기 전에** 작성·동결한다(md5 를 `PREREG_frozen_md5.txt` 에 기록).
⛔`analyze_s1/s2/s3.py` 를 복사하지 않았다. §4 의 11개 의무 항목을 세어가며 새로 쓴다.
마지막에 자체 채점을 찍고, 하나라도 빠지면 무결성 주장을 인용 불가로 표시한다.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('S4_OUT', os.path.join(HERE, 'results.jsonl'))
S1 = os.path.abspath(os.path.join(HERE, '..', 'S1_indep_sparse_2026-08-08'))
S3 = os.path.abspath(os.path.join(HERE, '..', 'S3_place_x_track_2026-08-08'))

A_NT, A_DUP = 'AUTO-NT', 'AUTO-NT-DUP'
OK_NT, OW_NT, OW_0 = 'ORA-K-NT', 'ORA-WPHI-NT', 'ORA-WPHI-0STEP'
PRIMARY_METRIC = 'avg_obs_loss'
SIGN_THRESHOLD = 13
MDE_MULT = 2.1279493432578005
MEDIAN_SE = 1.2533141373155001
D2_N3 = 1.693

# (셀, 기준선, 라벨) — 첫 항목이 PRIMARY
CONTRASTS = [(OW_NT, OW_0, 'ORA-WPHI-NT   − ORA-WPHI-0STEP  ★PRIMARY'),
             (OW_NT, OK_NT, 'ORA-WPHI-NT   − ORA-K-NT        (H4 위상오라클)'),
             (OW_NT, A_NT, 'ORA-WPHI-NT   − AUTO-NT         (H5)'),
             (OK_NT, A_NT, 'ORA-K-NT      − AUTO-NT'),
             (OW_0, A_NT, 'ORA-WPHI-0STEP− AUTO-NT         (진실 vs 배포)')]
PRIMARY = (OW_NT, OW_0)

REQUIRED = ['1 DUP 일치', '2 cfg 유일', '3 480/480·에러0', '4 결정론(nRMSE_recheck)',
            '5 게이트 재수록', '6 대조표 절대·상대·MDE·동점', '7 시드별+mean-of-3',
            '8 판정 문자열', '9 0STEP nRMSE=0 확인', '10 관측쪽 모델불일치 크기',
            '11 S3 재현성']
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
    s1mde = json.load(open(os.path.join(S1, 'MDE_prespec.json')))['by_regime']['nRMSE']

    METRICS = ('avg_obs_loss', 'avg_obs_resid_rms', 'nRMSE', 'decorr_mean', 'shrink_mean')
    V = defaultdict(dict)
    for r in recs:
        for m in METRICS + ('nRMSE_recheck',):
            v = r.get(m)
            if isinstance(v, (int, float)) and np.isfinite(v):
                V[(m, r['cell'], r['episode'])][r['seed']] = float(v)
    M = {k: float(np.median(list(v.values()))) for k, v in V.items() if v}

    print('=' * 106)
    print('S4 채점 — PREREG.md §3/§4 의 기계적 실행   (진실에서 출발시키면 어떻게 되는가)')
    print('=' * 106)
    print(f'레코드 {len(recs)} · 에러 {len(errs)} · 셀 {cells} · 에피소드 {len(eps)} · 시드 {seeds}')
    expect = len(cells) * len(eps) * len(seeds)
    got = len({(r['cell'], r['episode'], r['seed']) for r in recs})
    print(f'[§4.3] 조합 {got}/{expect}' + ('  ✅' if got == expect and not errs else '  ⛔'))
    for e in errs[:3]:
        print('   에러:', e.get('job_id'), str(e.get('error'))[:110])
    ofe = [r for r in recs if 'obs_fit_error' in r]
    if ofe:
        print(f'   ⛔ obs_fit 실패 {len(ofe)}건: {ofe[0]["obs_fit_error"][:120]}')
    did('3 480/480·에러0')

    dif = [abs(V[('nRMSE', A_NT, e)][s] - V[('nRMSE', A_DUP, e)][s])
           for e in eps for s in seeds
           if s in V[('nRMSE', A_NT, e)] and s in V[('nRMSE', A_DUP, e)]]
    print(f'[§4.1] {A_NT} vs {A_DUP}: {len(dif)} 쌍 worst |Δ| = '
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
        if n == 1:
            print(f'      {c:16s} 1 ✅')
        else:
            pr = sorted(ng[c])
            ok = len({a for a, _ in pr}) == len({b for _, b in pr}) == n
            print(f'      {c:16s} {n} ' + ('✅ n_gaussians=[N_gt] 와 성분수 1:1 (정상)'
                                           if ok else '⛔ 원인 불명'))
    did('2 cfg 유일')

    bad = [(r['cell'], r['episode'], r['seed'], r['nRMSE'], r['nRMSE_recheck'])
           for r in recs if 'nRMSE_recheck' in r
           and abs(r['nRMSE'] - r['nRMSE_recheck']) > 0]
    print(f'[§4.4] 결정론 (nRMSE vs nRMSE_recheck): 불일치 {len(bad)}/{len(recs)}'
          + ('  ✅ 두 실행 완전 동일' if not bad else f'  ⛔ 예: {bad[0]}'))
    did('4 결정론(nRMSE_recheck)')

    z = [M[('nRMSE', OW_0, e)] for e in eps if ('nRMSE', OW_0, e) in M]
    print(f'[§4.9] {OW_0} 의 nRMSE: max = {max(z):.2e} (0 이어야 함 — 아니면 규약 불일치)'
          + ('  ✅' if max(z) < 1e-6 else '  ⛔ 배치 무효'))
    print('        ⛔이 0 은 **동어반복**이다 (gt_wave_field = Σ A cos(...) = 모델식). '
          '"표현 충분" 결론 금지.')
    did('9 0STEP nRMSE=0 확인')

    print('\n' + '=' * 106)
    print('[§4.10] 관측 쪽 모델 불일치의 크기 — 진실 자신이 LiDAR 반환을 얼마나 못 맞추는가')
    print('   (Gerstner 수평변위 Q=0.5 + 센서 잡음. 이건 어떤 추정기도 넘을 수 없는 바닥이다)')
    print('=' * 106)
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        f = lambda m: float(np.median([M[(m, OW_0, e)] for e in E if (m, OW_0, e) in M]))
        print(f'   {rg:9s} obs_loss {f("avg_obs_loss"):.6f}   '
              f'obs_resid_rms {f("avg_obs_resid_rms"):.5f} m')
    did('10 관측쪽 모델불일치 크기')

    # ---------------- 수준표 ----------------
    print('\n' + '=' * 106)
    print('수준 (median-of-3 의 레짐 중앙)   ⛔ORA-* 는 출하 불가 연구 팔·묶음')
    print('=' * 106)
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        print(f'\n  {rg.upper():9s}{"nRMSE":>10s}{"obs_loss":>12s}{"obs_resid":>11s}'
              f'{"decorr":>10s}{"shrink":>10s}')
        for c in (A_NT, OK_NT, OW_NT, OW_0):
            g = lambda m: float(np.median([M[(m, c, e)] for e in E if (m, c, e) in M]))
            print(f'  {c:<16s}{g("nRMSE"):>10.4f}{g("avg_obs_loss"):>12.6f}'
                  f'{g("avg_obs_resid_rms"):>11.5f}{g("decorr_mean"):>10.4f}'
                  f'{g("shrink_mean"):>10.4f}')

    # ---------------- 대조표 ----------------
    def mde_of(metric, rg, E, base_cell):
        if metric == 'nRMSE':
            return s1mde[rg]['mde_agg'], 'S1 사전동결'
        rng = [max(V[(metric, base_cell, e)].values()) - min(V[(metric, base_cell, e)].values())
               for e in E if len(V[(metric, base_cell, e)]) >= 2]
        if not rng:
            return float('nan'), '산출 불가'
        sd = (float(np.median(rng)) / D2_N3) * np.sqrt(2) * 1.16 / np.sqrt(3)
        return MDE_MULT * MEDIAN_SE * sd / np.sqrt(len(E)), '⚠배치내'

    print('\n' + '=' * 106)
    print(f'§3 짝 Δ = (셀 − 기준선), 에피소드당 median-of-3, 레짐 안에서 Δ 의 중앙')
    print(f'   주 판정 지표 = {PRIMARY_METRIC} · 부 = obs_resid_rms, nRMSE')
    print('=' * 106)
    R_ = {}
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        print(f'\n### {rg.upper()} (n={len(E)})')
        ps = []
        for cell, base, lab in CONTRASTS:
            for metric in (PRIMARY_METRIC, 'avg_obs_resid_rms', 'nRMSE'):
                da = [M[(metric, cell, e)] - M[(metric, base, e)] for e in E
                      if (metric, cell, e) in M and (metric, base, e) in M]
                dr = [(M[(metric, cell, e)] - M[(metric, base, e)]) / M[(metric, base, e)]
                      for e in E if (metric, cell, e) in M and (metric, base, e) in M
                      and M[(metric, base, e)]]
                st = sign_test(da)
                mde, src = mde_of(metric, rg, E, base)
                R_[(rg, cell, base, metric)] = dict(
                    medA=float(np.median(da)) if da else np.nan,
                    medR=float(np.nanmedian(dr)) if dr else np.nan,
                    mde=mde, src=src, **st)
                if metric == PRIMARY_METRIC and (cell, base) != PRIMARY:
                    ps.append(((rg, cell, base), st['p']))
        H = holm(ps)
        print(f'{"대조":<44s}{"지표":<18s}{"절대Δ":>11s}{"상대Δ":>9s}'
              f'{"부호(-)":>9s}{"p":>10s}{"Holm":>8s}{"|Δ|/MDE":>9s}')
        for cell, base, lab in CONTRASTS:
            for metric, mlab in ((PRIMARY_METRIC, 'obs_loss'),
                                 ('avg_obs_resid_rms', 'obs_resid'), ('nRMSE', 'nRMSE')):
                r_ = R_[(rg, cell, base, metric)]
                hp = ('(PRIMARY)' if (cell, base) == PRIMARY and metric == PRIMARY_METRIC
                      else (f'{H[(rg, cell, base)]:.3f}' if metric == PRIMARY_METRIC else '—'))
                print(f'{(lab if metric == PRIMARY_METRIC else ""):<44s}{mlab:<18s}'
                      f'{r_["medA"]:>+11.5f}{r_["medR"]*100:>+8.1f}%'
                      f'{r_["neg"]:>6d}/{r_["n"]:<3d}{r_["p"]:>10.2e}{hp:>8s}'
                      f'{abs(r_["medA"])/r_["mde"] if np.isfinite(r_["mde"]) else float("nan"):>9.2f}')
    did('6 대조표 절대·상대·MDE·동점')

    # ---------------- 판정 ----------------
    print('\n' + '=' * 106)
    print('판정 (PREREG §1) — PRIMARY = 혼잡 ORA-WPHI-NT − ORA-WPHI-0STEP, 지표 avg_obs_loss')
    print('=' * 106)
    p = R_[('crowded', OW_NT, OW_0, PRIMARY_METRIC)]
    q = R_[('crowded', OW_NT, OW_0, 'avg_obs_resid_rms')]
    if p['medA'] > 0 and p['pos'] >= SIGN_THRESHOLD:
        v = ('★OPTIMIZER-FAILURE — 진실에서 출발시켜도 옵티마이저가 **자기 목적함수 기준으로 '
             '더 나쁜 곳**으로 간다 ⟹ 식별 문제도 표현 문제도 아니고 **최적화/일반화 실패**다.')
    elif p['medA'] < 0 and p['neg'] >= SIGN_THRESHOLD:
        v = ('OBJECTIVE-MISSPEC — 최적화 결과가 관측을 **진실보다 더 잘** 맞춘다 ⟹ '
             '관측만으로는 진실이 최적이 아니다(식별/목적함수 오설정).')
    else:
        v = (f'UNDECIDED — 문턱 미달 (Δ {p["medA"]:+.6f}, 부호 {p["neg"]}/{p["n"]}, '
             f'필요 {SIGN_THRESHOLD})')
    print(f'  **{v}**')
    print(f'      obs_loss  Δ {p["medA"]:+.6f} ({p["medR"]*100:+.1f}%) · 부호 {p["neg"]}/{p["n"]}'
          f' · p={p["p"]:.2e} · |Δ|/MDE={abs(p["medA"])/p["mde"]:.2f} [{p["src"]}]')
    print(f'      obs_resid Δ {q["medA"]:+.5f} m ({q["medR"]*100:+.1f}%) · 부호 {q["neg"]}/{q["n"]}'
          f' · p={q["p"]:.2e}')
    print('  ⛔이 판정은 **어느 이분법인지**만 가른다. "최적화만 고치면 된다"는 처방이 아니다.')
    print('  ⛔ORA-* 는 출하 불가 연구 팔이고 (N_gt + 1단 900스텝)이 함께 붙은 묶음이다.')
    did('8 판정 문자열')

    # ---------------- 시드별 ----------------
    print('\n' + '=' * 106)
    print('[§4.7] 시드별 단독 재집계 + mean-of-3 (PRIMARY 대조, 두 지표)')
    print('=' * 106)
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        for metric, mlab in ((PRIMARY_METRIC, 'obs_loss'), ('nRMSE', 'nRMSE')):
            parts = []
            for sd in seeds:
                d = [V[(metric, OW_NT, e)][sd] - V[(metric, OW_0, e)][sd] for e in E
                     if sd in V[(metric, OW_NT, e)] and sd in V[(metric, OW_0, e)]]
                st = sign_test(d)
                parts.append(f'{sd}:{np.median(d):+.5f}({st["neg"]}/{st["n"]})')
            dm = [float(np.mean(list(V[(metric, OW_NT, e)].values()))
                        - np.mean(list(V[(metric, OW_0, e)].values()))) for e in E
                  if V[(metric, OW_NT, e)] and V[(metric, OW_0, e)]]
            stm = sign_test(dm)
            print(f'  {rg[:4]}|{mlab:<10s} ' + ' '.join(parts)
                  + f'  mean3:{np.median(dm):+.5f}({stm["neg"]}/{stm["n"]})')
    did('7 시드별+mean-of-3')

    # ---------------- S3 재현성 ----------------
    print('\n' + '=' * 106)
    print('[§4.11] S3 재현성 — ORA-K-NT ↔ S3 ORACLE-NOTRACK (같은 32ep, 시드만 다름, 짝비교)')
    print('=' * 106)
    p3 = os.path.join(S3, 'results.jsonl')
    if os.path.exists(p3):
        W = defaultdict(dict)
        for line in open(p3):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' in r or r['cell'] != 'ORACLE-NOTRACK':
                continue
            W[r['episode']][r['seed']] = r['nRMSE']
        Mo = {e: float(np.median(list(v.values()))) for e, v in W.items() if v}
        for rg in ('crowded', 'sparse'):
            E = [e for e in eps if reg[e] == rg and e in Mo and ('nRMSE', OK_NT, e) in M]
            d = np.array([M[('nRMSE', OK_NT, e)] - Mo[e] for e in E])
            a = np.array([M[('nRMSE', OK_NT, e)] for e in E])
            b = np.array([Mo[e] for e in E])
            print(f'   {rg:9s} 짝차 중앙 {np.median(d):+.4f}  r={np.corrcoef(a, b)[0, 1]:.3f}'
                  f'  (n={len(E)})')
        did('11 S3 재현성')
    else:
        print('   ⛔ S3 results.jsonl 없음')

    gate = os.path.join(S1, 'gate.txt')
    print('\n' + '=' * 106)
    print('[§4.5] 실행 전 게이트 재수록 (S1 gate.txt — 같은 32 에피소드)')
    print('=' * 106)
    if os.path.exists(gate):
        print(open(gate).read().strip()[-600:])
        did('5 게이트 재수록')
    else:
        print('   ⛔ gate.txt 없음')

    print('\n' + '=' * 106)
    print(f'[§4 자체 채점] 의무 {len(REQUIRED)} 항 중 **{len(_done)}개 실행**')
    for n in REQUIRED:
        print(f'   {"✅" if n in _done else "⛔ 미실행"}  {n}')
    print('  ' + ('✅ 전항 실행됨.' if len(_done) == len(REQUIRED)
                  else '⛔ **§4 위반 — 이 배치의 무결성 주장은 인용 불가.**'))
    print('=' * 106)


if __name__ == '__main__':
    main()
