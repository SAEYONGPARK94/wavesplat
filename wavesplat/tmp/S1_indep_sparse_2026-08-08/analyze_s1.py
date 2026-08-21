"""S1 채점기 — `PREREG.md` + `AMENDMENT_1.md` 의 기계적 실행.

⛔결과가 생기기 전에 동결된다(md5 를 `PREREG_frozen_md5.txt` 에 기록).
⛔`analyze_eb/ec/ed.py` 를 **복사하지 않았다**. E-C 는 복사 때문에 사전등록에 없는 지표를
   PRIMARY 자리에 남겼고, E-D 는 그 복사본을 고쳐 쓰다 같은 계열의 누락(셸 MDE)을 재발시켰다.

AMENDMENT 1 F7 이 요구하는 출력 항목을 `REQUIRED` 에 이름으로 박아 두고, 마지막에 **실제로
몇 개를 print 했는지 자체 채점**한다. 하나라도 빠지면 "§5 위반 — 무결성 주장 인용 불가"를 찍는다.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('S1_OUT', os.path.join(HERE, 'results.jsonl'))
MDE_FILE = os.path.join(HERE, 'MDE_prespec.json')
GATE_TXT = os.path.join(HERE, 'gate.txt')

METRIC = 'nRMSE'                 # PREREG §3.9
CONTROL = 'X-BASE'
DUP = 'X-BASE2'
N_PERM = 20000                   # AMENDMENT 1 F4
PERM_SEED = 20260808             # AMENDMENT 1 F4 (동결)
D2_HALF = 0.0351                 # PREREG §1.1 D2 — 원 크기(-0.0701)의 50%
SIGN_THRESHOLD = 13              # AMENDMENT 1 F2 — Holm-2 (alpha=0.025) 통과에 필요한 부호수

# (셀, 기준선). AMENDMENT 1 F5 로 출하-팔 대조가 추가됐다.
CONTRASTS = [('G1', CONTROL), ('P1', CONTROL), ('G1P1', CONTROL),
             ('AUTO', CONTROL), ('ORACLE-K', CONTROL), ('AUTO-G1P1', 'AUTO')]
PRIMARY_KEY = ('G1P1', CONTROL)          # PRIMARY 족 = 이 대조의 혼잡 + 희소 (Holm-2)

REQUIRED = [
    'D1 판정', 'D2 판정', 'D3 판정', 'D4 판정',
    '대조표(절대+상대+레짐별 MDE+동점)', '시드별 재집계', 'mean-of-3 대조',
    'X-BASE vs X-BASE2', 'cfg 유일성', '조합 완전성', '게이트 재수록',
    'H3 순열검정', 'H5 분모 가드', 'H4 짝별 상호작용',
]
_printed = set()


def did(name):
    _printed.add(name)


# ------------------------------------------------------------------ helpers
def regime_of(ep):
    m = re.search(r'_(\d+)comp', ep)
    n = int(m.group(1)) if m else -1
    return 'crowded' if n >= 15 else 'sparse'


def sign_test(deltas):
    """PREREG §3.4/§3.5 — 양측 정확 부호검정. 정확히 0 인 짝은 제외하고 n 을 줄인다."""
    d = np.asarray([x for x in deltas if np.isfinite(x)], float)
    ties = int(np.sum(d == 0.0))
    d = d[d != 0.0]
    n = d.size
    neg = int(np.sum(d < 0))
    if n == 0:
        return dict(n=0, neg=0, ties=ties, p=float('nan'), floor_p=float('nan'))
    return dict(n=n, neg=neg, ties=ties,
                p=float(binomtest(neg, n, 0.5, alternative='two-sided').pvalue),
                floor_p=float(min(1.0, 2.0 / 2 ** n)))


def holm(pairs):
    """pairs = [(key, p)] -> {key: (p_adj, reject@0.05)}"""
    valid = sorted([x for x in pairs if np.isfinite(x[1])], key=lambda x: x[1])
    m = len(valid)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(valid):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = (adj, adj <= 0.05)
    for k, _ in pairs:
        out.setdefault(k, (float('nan'), False))
    return out


def perm_test(a, b, n_perm=N_PERM, seed=PERM_SEED):
    """레짐 라벨 순열검정, 통계량 = median(a) - median(b), 양측."""
    a = np.asarray([x for x in a if np.isfinite(x)], float)
    b = np.asarray([x for x in b if np.isfinite(x)], float)
    if a.size == 0 or b.size == 0:
        return float('nan'), float('nan')
    obs = float(np.median(a) - np.median(b))
    pool = np.concatenate([a, b])
    na = a.size
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(np.median(pool[:na]) - np.median(pool[na:])) >= abs(obs) - 1e-15:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def load():
    if not os.path.exists(RES):
        sys.exit(f'없음: {RES}')
    recs, errs = [], []
    for line in open(RES):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        (errs if 'error' in r else recs).append(r)
    return recs, errs


# ------------------------------------------------------------------ main
def main():
    recs, errs = load()
    mde = json.load(open(MDE_FILE))['by_regime'][METRIC]
    episodes = sorted({r['episode'] for r in recs})
    cells = sorted({r['cell'] for r in recs})
    seeds = sorted({r['seed'] for r in recs})
    reg = {e: regime_of(e) for e in episodes}

    byseed = defaultdict(dict)
    for r in recs:
        v = r.get(METRIC)
        if isinstance(v, (int, float)) and np.isfinite(v):
            byseed[(r['cell'], r['episode'])][r['seed']] = float(v)
    med = {k: float(np.median(list(v.values()))) for k, v in byseed.items() if v}

    print('=' * 96)
    print('S1 채점 — PREREG.md + AMENDMENT_1.md 의 기계적 실행')
    print('=' * 96)
    print(f'레코드 {len(recs)} · 에러 {len(errs)} · 셀 {len(cells)} · 에피소드 {len(episodes)} '
          f'· 시드 {seeds}')
    print(f'주 지표 = {METRIC} (무차원). ⛔미터 지표와 같은 표에 섞지 않는다 (PREREG §3.10).')
    for rg in ('crowded', 'sparse'):
        m = mde[rg]
        print(f'  MDE[{rg:7s}] 기준선 {m["baseline_median_XBASE"]:.4f} · '
              f'sigma_d {m["sigma_d"]:.5f} · mde_ep {m["mde_ep"]:.5f} · '
              f'mde_agg(n=16) {m["mde_agg"]:.5f}')
    print('  ⛔관측/MDE 는 **같은 레짐 안에서만** 비교한다 (AMENDMENT 1 F1).')

    # ---------------- 조합 완전성 ----------------
    expect = len(cells) * len(episodes) * len(seeds)
    got = len({(r['cell'], r['episode'], r['seed']) for r in recs})
    print(f'\n[§5.3] 조합 {got}/{expect} · 에러 {len(errs)}'
          + ('  ✅' if got == expect and not errs else '  ⛔'))
    for e in errs[:5]:
        print('   에러:', e.get('job_id'), str(e.get('error'))[:110])
    did('조합 완전성')

    # ---------------- 워커 누출 (D4) ----------------
    diffs = [abs(byseed[(CONTROL, e)][s] - byseed[(DUP, e)][s])
             for e in episodes for s in seeds
             if s in byseed.get((CONTROL, e), {}) and s in byseed.get((DUP, e), {})]
    worst = max(diffs) if diffs else float('nan')
    d4_ok = bool(diffs) and worst == 0.0
    print(f'[§5.1] X-BASE vs X-BASE2: {len(diffs)} 쌍, worst |Δ| = {worst:.3e}'
          + ('  ✅ 누출 0' if d4_ok else '  ⛔ 불일치'))
    did('X-BASE vs X-BASE2')

    cfgs = defaultdict(set)
    for r in recs:
        if isinstance(r.get('cfg'), dict):
            cfgs[r['cell']].add(json.dumps(r['cfg'], sort_keys=True))
    bad_cfg = {c: len(v) for c, v in cfgs.items() if len(v) != 1}
    print(f'[§5.2] 셀당 cfg 유일: ' + ('✅ 전부 1' if not bad_cfg else f'⛔ {bad_cfg}'))
    did('cfg 유일성')

    # ---------------- 대조표 ----------------
    print('\n' + '=' * 96)
    print('§3 + F3 — 짝 Δ = (셀 − 기준선), 에피소드당 median-of-3, 레짐 안에서 Δ 의 중앙')
    print('   절대 = nRMSE 차 · 상대 = 에피소드별 (셀−기준)/기준 의 중앙 (⛔중앙들의 비 아님)')
    print('=' * 96)
    R = {}
    for rg in ('crowded', 'sparse'):
        eps = [e for e in episodes if reg[e] == rg]
        m = mde[rg]
        for cell, base in CONTRASTS:
            da, dr = [], []
            for e in eps:
                if (cell, e) in med and (base, e) in med:
                    b = med[(base, e)]
                    da.append(med[(cell, e)] - b)
                    dr.append((med[(cell, e)] - b) / b if b else float('nan'))
            st = sign_test(da)
            R[(rg, cell, base)] = dict(
                da=da, dr=dr, st=st,
                medA=float(np.median(da)) if da else float('nan'),
                medR=float(np.nanmedian(dr)) if dr else float('nan'),
                lvl=float(np.median([med[(cell, e)] for e in eps if (cell, e) in med]))
                if any((cell, e) in med for e in eps) else float('nan'),
                mde_ep=m['mde_ep'], mde_agg=m['mde_agg'])

    # Holm: PRIMARY 족(2) 과 SECONDARY 족(레짐당 5) 을 분리 (AMENDMENT 1 F2)
    fam_p = [((rg,) + PRIMARY_KEY, R[(rg,) + PRIMARY_KEY]['st']['p']) for rg in ('crowded', 'sparse')]
    H = holm(fam_p)
    fam_label = {k: 'P2' for k, _ in fam_p}
    for rg in ('crowded', 'sparse'):
        sec = [((rg, c, b), R[(rg, c, b)]['st']['p']) for c, b in CONTRASTS
               if (c, b) != PRIMARY_KEY]
        H.update(holm(sec))
        for k, _ in sec:
            fam_label[k] = 'S5'

    hdr = (f'{"대조":<20s} {"절대Δ":>9s} {"상대Δ":>9s} {"부호(-)":>9s} {"동점":>4s} '
           f'{"p":>9s} {"Holm":>8s} {"족":>3s} {"|Δ|/mde_agg":>11s} {"수준":>8s}')
    for rg in ('crowded', 'sparse'):
        eps = [e for e in episodes if reg[e] == rg]
        print(f'\n### {rg.upper()}  (n={len(eps)})   mde_ep={mde[rg]["mde_ep"]:.5f}  '
              f'mde_agg={mde[rg]["mde_agg"]:.5f}')
        print(hdr)
        for cell, base in CONTRASTS:
            r_ = R[(rg, cell, base)]
            st = r_['st']
            hp, _ = H[(rg, cell, base)]
            print(f'{cell + " - " + base:<20s} {r_["medA"]:>+9.4f} '
                  f'{r_["medR"] * 100:>+8.1f}% {st["neg"]:>5d}/{st["n"]:<3d} {st["ties"]:>4d} '
                  f'{st["p"]:>9.2e} {hp:>8.3f} {fam_label[(rg, cell, base)]:>3s} '
                  f'{abs(r_["medA"]) / r_["mde_agg"]:>11.2f} {r_["lvl"]:>8.4f}')
        b = float(np.median([med[(CONTROL, e)] for e in eps if (CONTROL, e) in med]))
        print(f'{"X-BASE (대조)":<20s} {"—":>9s} {"—":>9s} {"—":>9s} {"—":>4s} {"—":>9s} '
              f'{"—":>8s} {"—":>3s} {"—":>11s} {b:>8.4f}   ⛔출하 팔 아님(v2_konly)')
    did('대조표(절대+상대+레짐별 MDE+동점)')

    # ---------------- D1 / D2 ----------------
    print('\n' + '=' * 96)
    print('판정 — AMENDMENT 1 F2 로 문턱 정정됨 (12/16 양측 p = 0.0768 이므로 13/16 이 필요)')
    print('=' * 96)
    c = R[('crowded',) + PRIMARY_KEY]
    hp_c = H[('crowded',) + PRIMARY_KEY][0]
    if c['st']['neg'] >= SIGN_THRESHOLD and c['medA'] < 0 and hp_c <= 0.025:
        d1 = 'SUPPORTED — 부호 재현'
    else:
        d1 = (f'NOT REPLICATED — 부호 {c["st"]["neg"]}/{c["st"]["n"]} '
              f'(필요 {SIGN_THRESHOLD}/16, Holm-2 p={hp_c:.3f}) ⟹ -0.0701 은 신선 독립 표본에서 재현되지 않았다')
    print(f'  D1 [PRIMARY H1, 혼잡 G1P1−X-BASE]: **{d1}**')
    print(f'      절대 Δ 중앙 {c["medA"]:+.4f} (원 −0.0701) · 상대 {c["medR"]*100:+.1f}% (원 −24.8%) '
          f'· |Δ|/mde_agg = {abs(c["medA"])/c["mde_agg"]:.2f}')
    did('D1 판정')

    if c['st']['neg'] >= SIGN_THRESHOLD and c['medA'] < 0:
        d2 = ('REPLICATED (부호+크기)' if abs(c['medA']) >= D2_HALF
              else f'SIGN-ONLY — 크기 미재현 (|Δ| {abs(c["medA"]):.4f} < {D2_HALF})')
    else:
        d2 = '미충족 — D1 이 먼저 실패했으므로 크기 판정은 하지 않는다'
    print(f'  D2 [크기 재현]: **{d2}**')
    did('D2 판정')

    # ---------------- D3 (H3) ----------------
    s = R[('sparse',) + PRIMARY_KEY]
    obs_a, p_a = perm_test(c['da'], s['da'])
    obs_r, p_r = perm_test(c['dr'], s['dr'])
    try:
        mwu = mannwhitneyu(c['da'], s['da'], alternative='two-sided').pvalue
    except ValueError:
        mwu = float('nan')
    d3_ok = (s['st']['neg'] <= 8) and (p_a <= 0.05)
    both = all(R[(rg,) + PRIMARY_KEY]['st']['neg'] >= SIGN_THRESHOLD
               and R[(rg,) + PRIMARY_KEY]['medA'] < 0 for rg in ('crowded', 'sparse'))
    if both:
        d3 = '★반증 — 양 레짐 모두 이긴다 ⟹ B2 의 "혼잡 전용" 판정이 반증된다'
    elif d3_ok:
        d3 = '지지 — 혼잡 전용'
    else:
        d3 = (f'미충족 — 희소 부호 {s["st"]["neg"]}/{s["st"]["n"]} (필요 ≤8), '
              f'순열 p={p_a:.4f} (필요 ≤0.05)')
    print(f'\n  D3 [H3 레짐 상호작용]: **{d3}**')
    print(f'      순열검정(20000, seed {PERM_SEED}, 양측): 절대 통계량 {obs_a:+.4f} p={p_a:.4f} · '
          f'상대 {obs_r*100:+.1f}%p p={p_r:.4f}')
    print(f'      (보고용, 판정 아님) Mann-Whitney U p={mwu:.4f}')
    print(f'      ⛔레짐 축은 성분수 단독이 아니다 — MIN_DK 억제 발동 여부·사다리 대 진실 N·'
          f'기준선 18.5배가 함께 움직인다 (AMENDMENT 1 F8).')
    did('D3 판정')
    did('H3 순열검정')

    print(f'\n  D4 [워커 누출 프로브]: **{"PASS — 배치 유효" if d4_ok else "FAIL — 배치 전체 무효"}**')
    did('D4 판정')

    # ---------------- H5 회수율 (분모 가드) ----------------
    print('\n' + '=' * 96)
    print('H5 — 도달가능 천장 회수율   ⛔ORACLE-K 는 3요인 묶음(k+N_gt+스케줄), "k 효과" 아님')
    print('=' * 96)
    for rg in ('crowded', 'sparse'):
        eps = [e for e in episodes if reg[e] == rg]
        guard = mde[rg]['mde_ep']
        ratios, npass, ntot = [], 0, 0
        for e in eps:
            if all((k, e) in med for k in ('G1P1', 'ORACLE-K', CONTROL)):
                ntot += 1
                den = med[('ORACLE-K', e)] - med[(CONTROL, e)]
                if den < 0 and abs(den) >= guard:
                    npass += 1
                    ratios.append((med[('G1P1', e)] - med[(CONTROL, e)]) / den)
        if npass < max(1, ntot // 2):
            print(f'  {rg:<8s} **회수율 미정의** — 분모 가드 통과 {npass}/{ntot} '
                  f'(가드: 분모<0 이고 |분모| ≥ mde_ep {guard:.5f})')
        else:
            print(f'  {rg:<8s} ep별 비의 중앙 = {np.median(ratios) * 100:6.1f}%   '
                  f'가드 통과 {npass}/{ntot}   (구 혼잡값 73.6%, 같은 규약)')
    did('H5 분모 가드')

    # ---------------- H4 상호작용 (짝별만) ----------------
    print('\n' + '=' * 96)
    print('H4 — G1/P1 짝별 상호작용  ⛔"중앙들의 차" 규약 금지(구 −0.0117 을 만든 산물)')
    print('=' * 96)
    for rg in ('crowded', 'sparse'):
        eps = [e for e in episodes if reg[e] == rg]
        inter = []
        for e in eps:
            if all((k, e) in med for k in ('G1', 'P1', 'G1P1', CONTROL)):
                b = med[(CONTROL, e)]
                inter.append((med[('G1P1', e)] - b) - (med[('G1', e)] - b) - (med[('P1', e)] - b))
        if inter:
            st = sign_test(inter)
            print(f'  {rg:<8s} 중앙 {np.median(inter):+.4f}  음수 {st["neg"]}/{st["n"]}  '
                  f'p={st["p"]:.3f}  |Δ|/mde_agg={abs(np.median(inter))/mde[rg]["mde_agg"]:.2f}')
    did('H4 짝별 상호작용')

    # ---------------- 시드별 재집계 ----------------
    print('\n' + '=' * 96)
    print('[§5.6] 시드별 단독 재집계 + mean-of-3 — median-of-3 규약이 판정을 만들었는가')
    print('=' * 96)
    for rg in ('crowded', 'sparse'):
        eps = [e for e in episodes if reg[e] == rg]
        for cell, base in CONTRASTS:
            parts = []
            for sd in seeds:
                d = [byseed[(cell, e)][sd] - byseed[(base, e)][sd] for e in eps
                     if sd in byseed.get((cell, e), {}) and sd in byseed.get((base, e), {})]
                st = sign_test(d)
                parts.append(f'{sd}:{np.median(d):+.4f}({st["neg"]}/{st["n"]},p={st["p"]:.3f})')
            dm = [float(np.mean(list(byseed[(cell, e)].values()))
                        - np.mean(list(byseed[(base, e)].values()))) for e in eps
                  if byseed.get((cell, e)) and byseed.get((base, e))]
            stm = sign_test(dm)
            print(f'  {rg[:4]}|{cell + "-" + base:<18s} ' + ' '.join(parts)
                  + f'  mean3:{np.median(dm):+.4f}({stm["neg"]}/{stm["n"]},p={stm["p"]:.3f})')
    did('시드별 재집계')
    did('mean-of-3 대조')

    # ---------------- 게이트 재수록 ----------------
    print('\n' + '=' * 96)
    print('[§5.4] 실행 전 게이트 재수록')
    print('=' * 96)
    if os.path.exists(GATE_TXT):
        print(open(GATE_TXT).read().strip()[-1800:])
        did('게이트 재수록')
    else:
        print('  ⛔ gate.txt 없음 — §4 게이트가 기록되지 않았다')

    # ---------------- 자체 채점 ----------------
    print('\n' + '=' * 96)
    print(f'[F7 자체 채점] 의무 출력 {len(REQUIRED)} 항 중 **{len(_printed)}개 실행**')
    for name in REQUIRED:
        print(f'   {"✅" if name in _printed else "⛔ 미실행"}  {name}')
    if len(_printed) < len(REQUIRED):
        print('  ⛔ **§5 위반 — 이 배치의 무결성 주장은 인용 불가.**')
    else:
        print('  ✅ 의무 출력 전항 실행됨.')
    print('=' * 96)


if __name__ == '__main__':
    main()
