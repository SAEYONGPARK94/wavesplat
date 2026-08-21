"""S6 채점기 — `PREREG.md` §3/§4 의 기계적 실행. ⛔결과를 읽기 전에 동결한다."""
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('S6_OUT', os.path.join(HERE, 'results.jsonl'))
S1 = os.path.abspath(os.path.join(HERE, '..', 'S1_indep_sparse_2026-08-08'))
S5 = os.path.abspath(os.path.join(HERE, '..', 'S5_window_loss_2026-08-08'))

SHIP, BASE, DUP = 'AUTO', 'AUTO-NT', 'AUTO-NT-DUP'
SWEEP = [('AUTO-NT-L1-3e4', 3e-4), ('AUTO-NT-L1-1e4', 1e-4), ('AUTO-NT-L1-0', 0.0)]
TH = 13
REQUIRED = ['1 DUP 일치', '2 cfg 유일+lambda_l1 확인', '3 576/576·에러0', '4 게이트 재수록',
            '5 대조표', '6 시드별+mean-of-3', '7 용량-반응표+단조성',
            '8 판정 문자열(DOSE-FOUND/NO-DOSE)', '9 H3 출하 기본값 대비', '10 S5 재현성']
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


def med_of(path, cell, metric='nRMSE'):
    V = defaultdict(dict)
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' not in r and r['cell'] == cell and np.isfinite(r.get(metric, np.nan)):
            V[r['episode']][r['seed']] = float(r[metric])
    return {e: float(np.median(list(v.values()))) for e, v in V.items() if v}


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
    mde = json.load(open(os.path.join(S1, 'MDE_prespec.json')))['by_regime']['nRMSE']

    V = defaultdict(dict)
    for r in recs:
        for m in ('nRMSE', 'window_resid', 'all_resid', 'decorr_mean', 'shrink_mean'):
            v = r.get(m)
            if isinstance(v, (int, float)) and np.isfinite(v):
                V[(m, r['cell'], r['episode'])][r['seed']] = float(v)
    M = {k: float(np.median(list(v.values()))) for k, v in V.items() if v}

    print('=' * 104)
    print('S6 채점 — lambda_l1 용량-반응 스윕 (출하 팔)')
    print('=' * 104)
    print(f'레코드 {len(recs)} · 에러 {len(errs)} · 셀 {cells} · 에피소드 {len(eps)} · 시드 {seeds}')
    exp = len(cells) * len(eps) * len(seeds)
    got = len({(r['cell'], r['episode'], r['seed']) for r in recs})
    print(f'[§4.3] 조합 {got}/{exp}' + ('  ✅' if got == exp and not errs else '  ⛔'))
    for e in errs[:3]:
        print('   에러:', e.get('job_id'), str(e.get('error'))[:110])
    did('3 576/576·에러0')

    dif = [abs(V[('nRMSE', BASE, e)][s] - V[('nRMSE', DUP, e)][s]) for e in eps for s in seeds
           if s in V[('nRMSE', BASE, e)] and s in V[('nRMSE', DUP, e)]]
    print(f'[§4.1] {BASE} vs {DUP}: {len(dif)} 쌍 worst |Δ| = '
          f'{max(dif) if dif else float("nan"):.3e}'
          + ('  ✅ 누출 0' if dif and max(dif) == 0.0 else '  ⛔'))
    did('1 DUP 일치')

    print('[§4.2] 셀당 cfg 유일 + lambda_l1 실제값:')
    cf = defaultdict(set)
    lam = defaultdict(set)
    trk = defaultdict(set)
    for r in recs:
        c = r.get('cfg') or {}
        cf[r['cell']].add(json.dumps(c, sort_keys=True))
        lam[r['cell']].add(c.get('lambda_l1', 0.001))
        trk[r['cell']].add(c.get('tracking_lr', 0.005))
    want = dict(SWEEP)
    want[BASE] = 1e-3
    want[DUP] = 1e-3
    want[SHIP] = 1e-3
    for c in cells:
        ok_cfg = len(cf[c]) == 1
        ok_lam = len(lam[c]) == 1 and abs(list(lam[c])[0] - want.get(c, -1)) < 1e-12
        print(f'      {c:16s} cfg {len(cf[c])} {"✅" if ok_cfg else "⛔"}  '
              f'lambda_l1={list(lam[c])[0]:<8g} {"✅" if ok_lam else "⛔ 의도와 다름"}  '
              f'tracking_lr={list(trk[c])[0]}')
    did('2 cfg 유일+lambda_l1 확인')

    # ---------------- 용량-반응 ----------------
    print('\n' + '=' * 104)
    print('§4.7 용량-반응 — lambda_l1 을 내릴 때 수준(중앙)이 어떻게 변하는가')
    print('=' * 104)
    ladder = [(BASE, 1e-3)] + SWEEP
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        print(f'\n  {rg.upper():9s}{"lambda_l1":>11s}{"nRMSE":>10s}{"window":>10s}'
              f'{"all_resid":>11s}{"decorr":>9s}{"shrink":>9s}')
        lv = []
        for c, l in ladder:
            g = lambda m: float(np.median([M[(m, c, e)] for e in E if (m, c, e) in M]))
            v = g('nRMSE')
            lv.append(v)
            print(f'  {c:<16s}{l:>11g}{v:>10.4f}{g("window_resid"):>10.5f}'
                  f'{g("all_resid"):>11.5f}{g("decorr_mean"):>9.4f}{g("shrink_mean"):>9.4f}')
        mono = all(lv[i + 1] <= lv[i] for i in range(len(lv) - 1))
        print(f'   ⟹ 단조 감소? {"예 ✅" if mono else "아니오 ⛔ (비단조)"}  '
              f'(보고용 — 판정 규칙 아님)')
        s = float(np.median([M[('nRMSE', SHIP, e)] for e in E if ('nRMSE', SHIP, e) in M]))
        print(f'   (참고) 실제 출하 기본값 AUTO(추적기 ON, λ=1e-3) = {s:.4f}')
    did('7 용량-반응표+단조성')

    # ---------------- PRIMARY 대조 ----------------
    print('\n' + '=' * 104)
    print('§3 PRIMARY — 각 λ 수준 vs AUTO-NT(λ=1e-3), 혼잡 nRMSE, Holm-3, 문턱 13/16')
    print('=' * 104)
    R_ = {}
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        ps = []
        for c, l in SWEEP:
            da = [M[('nRMSE', c, e)] - M[('nRMSE', BASE, e)] for e in E
                  if ('nRMSE', c, e) in M and ('nRMSE', BASE, e) in M]
            dr = [(M[('nRMSE', c, e)] - M[('nRMSE', BASE, e)]) / M[('nRMSE', BASE, e)]
                  for e in E if ('nRMSE', c, e) in M and ('nRMSE', BASE, e) in M]
            s = st_(da)
            R_[(rg, c)] = dict(medA=float(np.median(da)) if da else np.nan,
                               medR=float(np.nanmedian(dr)) if dr else np.nan, lam=l, **s)
            ps.append(((rg, c), s['p']))
        H = holm(ps)
        print(f'\n### {rg.upper()} (n={len(E)})   mde_agg={mde[rg]["mde_agg"]:.5f}')
        print(f'{"셀":<18s}{"lambda_l1":>10s}{"절대Δ":>11s}{"상대Δ":>9s}{"부호(-)":>10s}'
              f'{"p":>10s}{"Holm":>8s}{"|Δ|/MDE":>9s}{"문턱":>7s}')
        for c, l in SWEEP:
            r_ = R_[(rg, c)]
            r_['holm'] = H[(rg, c)]
            passed = r_['medA'] < 0 and r_['neg'] >= TH and r_['holm'] <= 0.05
            print(f'{c:<18s}{l:>10g}{r_["medA"]:>+11.5f}{r_["medR"]*100:>+8.1f}%'
                  f'{r_["neg"]:>7d}/{r_["n"]:<3d}{r_["p"]:>10.2e}{r_["holm"]:>8.3f}'
                  f'{abs(r_["medA"])/mde[rg]["mde_agg"]:>9.2f}{"통과 ✅" if passed else "미달":>7s}')
    did('5 대조표')

    # ---------------- 판정 ----------------
    print('\n' + '=' * 104)
    print('판정 (PREREG §1)')
    print('=' * 104)
    winners = [(c, R_[('crowded', c)]) for c, _ in SWEEP
               if R_[('crowded', c)]['medA'] < 0 and R_[('crowded', c)]['neg'] >= TH
               and R_[('crowded', c)]['holm'] <= 0.05]
    if winners:
        winners.sort(key=lambda z: z[1]['holm'])
        cand, cr = winners[0]
        print(f'  **★DOSE-FOUND — 배포 후보 = `{cand}` (lambda_l1={cr["lam"]:g})**')
        print(f'      혼잡 Δ {cr["medA"]:+.5f} ({cr["medR"]*100:+.1f}%) · 음수 {cr["neg"]}/{cr["n"]}'
              f' · Holm p={cr["holm"]:.4f}')
        if len(winners) > 1:
            print('      (동시 통과) ' + ', '.join(f'{c}(Holm {r["holm"]:.3f})'
                                                for c, r in winners[1:]))
    else:
        cand = None
        print('  **NO-DOSE — 어떤 λ 수준도 문턱(13/16 + Holm≤0.05)을 넘지 못했다.**')
        print('      ⟹ S5 의 −11.8% 는 배포로 이어지지 않는다. 그렇게 보고한다.')
    did('8 판정 문자열(DOSE-FOUND/NO-DOSE)')

    # ---------------- H3 출하 기본값 대비 ----------------
    print('\n' + '=' * 104)
    print('H3 — 실제 출하 기본값 `AUTO`(추적기 ON, λ=1e-3) 대비   ⛔이걸 못 넘으면 "배포 개선" 금지')
    print('=' * 104)
    for rg in ('crowded', 'sparse'):
        E = [e for e in eps if reg[e] == rg]
        for c, l in [(BASE, 1e-3)] + SWEEP:
            da = [M[('nRMSE', c, e)] - M[('nRMSE', SHIP, e)] for e in E
                  if ('nRMSE', c, e) in M and ('nRMSE', SHIP, e) in M]
            s = st_(da)
            mark = ''
            if cand and c == cand:
                mark = '  ← 후보'
            print(f'  {rg[:4]}|{c:<18s} λ={l:<8g} Δ {np.median(da):+.5f} '
                  f'음수 {s["neg"]}/{s["n"]} p={s["p"]:.2e}{mark}')
    print('  ⛔`AUTO` 는 추적기 ON 이고 스윕 팔은 전부 off 다 ⟹ 이 대조는 **노브 2개**(λ + 추적기)다.')
    did('9 H3 출하 기본값 대비')

    # ---------------- 시드별 ----------------
    print('\n' + '=' * 104)
    print('§4.6 시드별 단독 재집계 + mean-of-3 (혼잡, vs AUTO-NT)')
    print('=' * 104)
    E = [e for e in eps if reg[e] == 'crowded']
    for c, l in SWEEP:
        parts = []
        for sd in seeds:
            d = [V[('nRMSE', c, e)][sd] - V[('nRMSE', BASE, e)][sd] for e in E
                 if sd in V[('nRMSE', c, e)] and sd in V[('nRMSE', BASE, e)]]
            s = st_(d)
            parts.append(f'{sd}:{np.median(d):+.5f}({s["neg"]}/{s["n"]})')
        dm = [float(np.mean(list(V[('nRMSE', c, e)].values()))
                    - np.mean(list(V[('nRMSE', BASE, e)].values()))) for e in E
              if V[('nRMSE', c, e)] and V[('nRMSE', BASE, e)]]
        sm = st_(dm)
        print(f'  {c:<18s} ' + ' '.join(parts)
              + f'  mean3:{np.median(dm):+.5f}({sm["neg"]}/{sm["n"]})')
    did('6 시드별+mean-of-3')

    # ---------------- S5 재현성 ----------------
    print('\n' + '=' * 104)
    print('§4.10 S5 재현성 — AUTO-NT (같은 32ep, 시드만 다름, 짝비교)')
    print('=' * 104)
    p5 = os.path.join(S5, 'results.jsonl')
    if os.path.exists(p5):
        Mo = med_of(p5, BASE)
        for rg in ('crowded', 'sparse'):
            E2 = [e for e in eps if reg[e] == rg and e in Mo and ('nRMSE', BASE, e) in M]
            d = np.array([M[('nRMSE', BASE, e)] - Mo[e] for e in E2])
            a = np.array([M[('nRMSE', BASE, e)] for e in E2])
            b = np.array([Mo[e] for e in E2])
            print(f'   {rg:9s} 짝차 중앙 {np.median(d):+.5f}  r={np.corrcoef(a, b)[0, 1]:.3f}'
                  f'  (n={len(E2)})')
        did('10 S5 재현성')
    else:
        print('   ⛔ S5 results.jsonl 없음')

    gate = os.path.join(S1, 'gate.txt')
    print('\n' + '=' * 104)
    print('§4.4 실행 전 게이트 재수록')
    print('=' * 104)
    if os.path.exists(gate):
        print(open(gate).read().strip()[-400:])
        did('4 게이트 재수록')
    else:
        print('   ⛔ gate.txt 없음')

    print('\n' + '=' * 104)
    print(f'[§4 자체 채점] 의무 {len(REQUIRED)} 항 중 **{len(_done)}개 실행**')
    for n in REQUIRED:
        print(f'   {"✅" if n in _done else "⛔ 미실행"}  {n}')
    print('  ' + ('✅ 전항 실행됨.' if len(_done) == len(REQUIRED)
                  else '⛔ **§4 위반 — 무결성 주장 인용 불가.**'))
    print('=' * 104)


if __name__ == '__main__':
    main()
