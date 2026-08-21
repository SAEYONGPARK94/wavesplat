"""S11 NEUTRAL 감사 — 독립 재도출 + 규약 민감도 지도.

⛔`null_corrected.py` 를 import 하지도, 읽고 베끼지도 않는다. 원자료(jsonl)에서 처음부터 짠다.
⛔`analyze_s9.py` 도 §1-§4 재도출에는 쓰지 않는다 (§5 채점기 시험에서만 별도로 호출).

사용법:  python3 neutral_rederive.py [섹션...]
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict
from math import comb, sqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.abspath(os.path.join(HERE, '..'))
S9 = os.path.join(TMP, 'S9_lr_x_l1_2026-08-11', 'results.jsonl')
X2 = os.path.join(TMP, 'X2_no_learned_2026-08-11', 'results.jsonl')
S8 = os.path.join(TMP, 'S8_deploy_confirm_2026-08-10', 'results.jsonl')


# ── 적재 ─────────────────────────────────────────────────────────────────────
def regime_from_path(ep):
    """⛔레코드의 `regime` 필드를 믿지 않는다 — 경로에서 독립 유도(S8 AMENDMENT_1 전례)."""
    if '_2comp_' in ep or '_5comp_' in ep:
        return 'sparse'
    if '_15comp_' in ep or '_30comp_' in ep:
        return 'crowded'
    raise ValueError(ep)


def load(paths=(S9, X2)):
    """cell -> episode -> seed -> record"""
    D = defaultdict(lambda: defaultdict(dict))
    reg, err = {}, 0
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if 'error' in r:
                err += 1
                continue
            ep = r['episode']
            D[r['cell']][ep][r['seed']] = r
            rg = regime_from_path(ep)
            assert r['regime'] == rg, (ep, r['regime'], rg)
            reg[ep] = rg
    return D, reg, err


# ── 집계 ─────────────────────────────────────────────────────────────────────
def _finite(vals):
    return [v for v in vals if v is not None and np.isfinite(v)]


def get(r, key):
    if key.startswith('shell:'):
        return (r.get('shell_rmse') or {}).get(key[6:])
    return r.get(key)


# ★AUTO-SEED-NULL 은 추정 시드 1578-1580 을 쓴다(run_s9.py:NULL_SEEDS).
# 단일-시드 규약에서 이 대응을 안 걸면 널 팔이 통째로 사라져 SEL_null ≡ AUTO 가 된다.
SEED_MAP = {'AUTO-SEED-NULL': {1575: 1578, 1576: 1579, 1577: 1580}}


def ep_val(D, cell, ep, key, agg='median', seed=None):
    """에피소드당 스칼라. agg ∈ {median, mean, seed}."""
    recs = D.get(cell, {}).get(ep)
    if not recs:
        return None
    if agg == 'seed':
        r = recs.get(SEED_MAP.get(cell, {}).get(seed, seed))
        if r is None:
            return None
        v = get(r, key)
        return v if (v is not None and np.isfinite(v)) else None
    v = _finite([get(r, key) for r in recs.values()])
    if not v:
        return None
    return st.median(v) if agg == 'median' else float(np.mean(v))


# ── 검정 ─────────────────────────────────────────────────────────────────────
def p_two_sided(k, n):
    if n == 0:
        return float('nan')
    return sum(comb(n, i) for i in range(n + 1)
               if abs(i - n / 2.0) >= abs(k - n / 2.0)) / 2.0 ** n


def sign_test(deltas, tie='exclude'):
    """deltas = [(ep, d, rel)]. tie ∈ {exclude, half}"""
    d = [x[1] for x in deltas]
    rel = [x[2] for x in deltas]
    imp = sum(1 for x in d if x < 0)
    wor = sum(1 for x in d if x > 0)
    tie_n = len(d) - imp - wor
    if tie == 'exclude':
        n_eff, k = imp + wor, imp
        p = p_two_sided(max(imp, wor), n_eff) if n_eff else float('nan')
    else:                                  # 동점 절반씩 배분 (반올림 = 보수적 내림)
        n_eff = len(d)
        k = imp + tie_n // 2
        p = p_two_sided(max(k, n_eff - k), n_eff) if n_eff else float('nan')
    relf = _finite(rel)
    return dict(imp=imp, wor=wor, tie=tie_n, n=len(d), n_eff=n_eff, k=k, p=p,
                med_abs=st.median(d) if d else float('nan'),
                med_rel=st.median(relf) if relf else float('nan'),
                med_rel_notie=st.median([r for x, r in zip(d, rel)
                                         if x != 0 and np.isfinite(r)]) or float('nan')
                if any(x != 0 for x in d) else float('nan'),
                mean_rel=float(np.mean(relf)) if relf else float('nan'),
                mean_abs=float(np.mean(d)) if d else float('nan'),
                max_rel=max(relf, default=float('nan')))


def wilcoxon(deltas):
    """정확 부호순위 검정(작은 n) / 정규근사(큰 n). 동점 Δ 는 제외(Wilcoxon 표준)."""
    d = [x[1] for x in deltas if x[1] != 0]
    n = len(d)
    if n == 0:
        return dict(n=0, W=float('nan'), z=float('nan'), p=float('nan'))
    a = np.abs(np.array(d))
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos = float(sum(ranks[i] for i in range(n) if d[i] > 0))
    neg = float(sum(ranks[i] for i in range(n) if d[i] < 0))
    W = min(pos, neg)
    mu = n * (n + 1) / 4.0
    sd = sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu) / sd if sd else float('nan')
    from math import erf
    p = 2 * 0.5 * (1 + erf(-abs(z) / sqrt(2)))
    return dict(n=n, W=W, z=z, p=min(1.0, p), pos=pos, neg=neg)


# ── 짝 만들기 ────────────────────────────────────────────────────────────────
def eps_of(reg, rg):
    return sorted(e for e, g in reg.items() if g == rg)


def paired(D, reg, rg, tval, bval, key='nRMSE', agg='median', seed=None):
    """tval/bval = 에피소드->값 함수 또는 셀 이름."""
    def f(x):
        if callable(x):
            return x
        return lambda ep: ep_val(D, x, ep, key, agg, seed)
    tf, bf = f(tval), f(bval)
    out = []
    for ep in eps_of(reg, rg):
        a, b = tf(ep), bf(ep)
        if a is None or b is None:
            continue
        out.append((ep, a - b, (a - b) / b if b else float('nan')))
    return out


def selector(D, arms, signal, agg='median', seed=None):
    """min-signal 선택기. 반환 = (pick_fn(ep)->cell, value_fn(key)->fn)"""
    def pick(ep):
        best, bc = None, None
        for c in arms:
            v = ep_val(D, c, ep, signal, agg, seed)
            if v is None:
                continue
            if best is None or v < best:
                best, bc = v, c
        return bc
    def value(key):
        return lambda ep: (ep_val(D, pick(ep), ep, key, agg, seed)
                           if pick(ep) else None)
    return pick, value


def oracle(D, arms, key='nRMSE', agg='median', seed=None):
    def value(ep):
        v = _finite([ep_val(D, c, ep, key, agg, seed) for c in arms])
        return min(v) if v else None
    return value


def fmt(s):
    return (f"{s['imp']:2d}/{s['wor']:2d}/{s['tie']:2d}  n_eff={s['n_eff']:2d}  "
            f"p={s['p']:.4g}  med_rel={s['med_rel']*100:+.2f}%  "
            f"med_rel(동점제외)={s['med_rel_notie']*100:+.2f}%  mean_rel={s['mean_rel']*100:+.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
def sec1(D, reg):
    print('\n' + '=' * 92)
    print('§1.2 재도출 — SEL_real{AUTO,V2K} vs SEL_null{AUTO,AUTO-SEED-NULL}, 출력 nRMSE')
    print('=' * 92)
    for sig in ('window_resid', 'all_resid'):
        for rg in ('sparse', 'crowded'):
            _, vr = selector(D, ['AUTO', 'V2K'], sig)
            _, vn = selector(D, ['AUTO', 'AUTO-SEED-NULL'], sig)
            d = paired(D, reg, rg, vr('nRMSE'), vn('nRMSE'))
            s = sign_test(d)
            w = wilcoxon(d)
            print(f'  {sig:12s} {rg:8s} {fmt(s)}  |  Wilcoxon z={w["z"]:+.3f} p={w["p"]:.4g}')
    print('\n  [부수] 각 선택기 vs 자기 공유팔 AUTO')
    for lab, arms in (('SEL_real{AUTO,V2K}', ['AUTO', 'V2K']),
                      ('SEL_null{AUTO,SEEDNULL}', ['AUTO', 'AUTO-SEED-NULL'])):
        for rg in ('sparse', 'crowded'):
            _, v = selector(D, arms, 'window_resid')
            s = sign_test(paired(D, reg, rg, v('nRMSE'), 'AUTO'))
            print(f'    {lab:26s} {rg:8s} {fmt(s)}')
    print('\n  [§1.3 전환 승률] window_resid, 희소 — 전환한 에피소드에서 이겼는가')
    for lab, arms, base in (('처치 SEL_real', ['AUTO', 'V2K'], 'AUTO'),
                            ('널  SEL_null', ['AUTO', 'AUTO-SEED-NULL'], 'AUTO')):
        pick, v = selector(D, arms, 'window_resid')
        win = lose = 0
        for ep in eps_of(reg, 'sparse'):
            if pick(ep) == base:
                continue
            a, b = v('nRMSE')(ep), ep_val(D, base, ep, 'nRMSE')
            if a is None or b is None or a == b:
                continue
            win += a < b
            lose += a > b
        print(f'    {lab}: 전환 {win+lose}건 중 승 {win} = {win/(win+lose):.3f}')


def sec2(D, reg):
    print('\n' + '=' * 92)
    print('§2 재도출 — 도달 수준 (에피소드 중앙 nRMSE 의 레짐 중앙)')
    print('=' * 92)
    _, sel = selector(D, ['AUTO-L1', 'V2K'], 'window_resid')
    _, seln = selector(D, ['AUTO', 'AUTO-SEED-NULL'], 'window_resid')
    orc = oracle(D, ['AUTO-L1', 'V2K'])
    rows = [('AUTO', lambda ep: ep_val(D, 'AUTO', ep, 'nRMSE')),
            ('AUTO-L1', lambda ep: ep_val(D, 'AUTO-L1', ep, 'nRMSE')),
            ('V2K', lambda ep: ep_val(D, 'V2K', ep, 'nRMSE')),
            ('SEL{AUTO-L1,V2K}', sel('nRMSE')),
            ('SEL-NULL{AUTO,SEEDNULL}', seln('nRMSE')),
            ('오라클{AUTO-L1,V2K}', orc)]
    print(f'  {"":26s} {"sparse":>10s} {"crowded":>10s}')
    for lab, f in rows:
        vs = []
        for rg in ('sparse', 'crowded'):
            v = _finite([f(e) for e in eps_of(reg, rg)])
            vs.append(st.median(v))
        print(f'  {lab:26s} {vs[0]:10.5f} {vs[1]:10.5f}')
    print('\n  §F 팔 대조 V2K − AUTO-L1')
    for rg in ('sparse', 'crowded'):
        d = paired(D, reg, rg, 'V2K', 'AUTO-L1')
        s = sign_test(d)
        print(f'    {rg:8s} {fmt(s)}')
    print('\n  선택기 vs "항상 V2K" (상한 대조, 배포 불가)')
    for rg in ('sparse', 'crowded'):
        s = sign_test(paired(D, reg, rg, sel('nRMSE'), 'V2K'))
        print(f'    {rg:8s} {fmt(s)}')


def router_label(D, ep, mode='majority'):
    recs = D['AUTO-L1'].get(ep, {})
    v = [r.get('auto_selected') for r in recs.values()]
    if not v:
        return None
    if mode == 'all':
        return v[0] if len(set(v)) == 1 else 'mixed'
    return max(set(v), key=v.count)


def sec4(D, reg):
    print('\n' + '=' * 92)
    print('§4-A 재도출 — 희소 라우트 층화, V2K vs AUTO-L1')
    print('=' * 92)
    for mode in ('majority', 'all'):
        print(f'  [라우트 라벨 규약 = {mode}]')
        strata = defaultdict(list)
        for ep in eps_of(reg, 'sparse'):
            strata[router_label(D, ep, mode)].append(ep)
        for lab in sorted(strata, key=lambda x: -len(strata[x])):
            eps = set(strata[lab])
            d = [x for x in paired(D, reg, 'sparse', 'V2K', 'AUTO-L1') if x[0] in eps]
            s = sign_test(d)
            ml1 = st.median(_finite([ep_val(D, 'AUTO-L1', e, 'nRMSE') for e in eps]))
            mv2 = st.median(_finite([ep_val(D, 'V2K', e, 'nRMSE') for e in eps]))
            print(f'    {lab:10s} n={len(eps):3d}  {fmt(s)}  AUTO-L1중앙={ml1:.5f} V2K중앙={mv2:.5f}')
    print('\n  §4-C 사전 검정력 근거 (I900 ≡ V2K 가정 시)')
    for lab, eps in (('희소 전체', set(eps_of(reg, 'sparse'))),
                     ('희소 learned 라우트', {e for e in eps_of(reg, 'sparse')
                                              if router_label(D, e) == 'learned'})):
        d = [x for x in paired(D, reg, 'sparse', 'V2K', 'AUTO-L1') if x[0] in eps]
        s, w = sign_test(d), wilcoxon(d)
        print(f'    {lab:22s} n={len(d):3d} {s["imp"]}/{s["wor"]} p={s["p"]:.4g} '
              f'Wilcoxon z={w["z"]:+.3f} p={w["p"]:.4g} med_rel={s["med_rel"]*100:+.2f}% '
              f'mean_rel={s["mean_rel"]*100:+.1f}%')


# ── ★★규약 민감도 지도 ────────────────────────────────────────────────────────
def sec_sens(D, reg):
    print('\n' + '=' * 92)
    print('★★규약 민감도 — §1.2 결론(희소 SEL_real<SEL_null, 혼잡 부호 반대)이 살아남는가')
    print('=' * 92)
    ROWS = []

    def run(axis, conv, sig, agg, seed, tie, outkey, stat):
        _, vr = selector(D, ['AUTO', 'V2K'], sig, agg, seed)
        _, vn = selector(D, ['AUTO', 'AUTO-SEED-NULL'], sig, agg, seed)
        out = {}
        for rg in ('sparse', 'crowded'):
            d = paired(D, reg, rg, vr(outkey), vn(outkey), key=outkey, agg=agg, seed=seed)
            s = sign_test(d, tie=tie)
            if stat == 'sign':
                sig_ = s['p'] <= 0.05
                direc = -1 if s['imp'] > s['wor'] else (1 if s['wor'] > s['imp'] else 0)
                val = f"{s['imp']}/{s['wor']}/{s['tie']} p={s['p']:.3g}"
            elif stat == 'wilcoxon':
                w = wilcoxon(d)
                sig_ = w['p'] <= 0.05
                direc = -1 if w['neg'] > w['pos'] else (1 if w['pos'] > w['neg'] else 0)
                val = f"z={w['z']:+.2f} p={w['p']:.3g}"
            elif stat == 'med_rel':
                sig_ = None
                direc = -1 if s['med_rel'] < 0 else (1 if s['med_rel'] > 0 else 0)
                val = f"{s['med_rel']*100:+.2f}%"
            elif stat == 'mean_paired':
                sig_ = None
                direc = -1 if s['mean_abs'] < 0 else (1 if s['mean_abs'] > 0 else 0)
                if np.sign(s['mean_abs']) != np.sign(s['mean_rel']):
                    direc = 0                       # ★절대/상대 부호 불일치
                val = f"abs {s['mean_abs']:+.5f} / rel {s['mean_rel']*100:+.1f}%"
            elif stat == 'sign_rel':
                imp = sum(1 for _e, _d, r in d if np.isfinite(r) and r < 0)
                wor = sum(1 for _e, _d, r in d if np.isfinite(r) and r > 0)
                pr = p_two_sided(max(imp, wor), imp + wor)
                sig_ = pr <= 0.05
                direc = -1 if imp > wor else (1 if wor > imp else 0)
                val = f"{imp}/{wor} p={pr:.3g}"
            out[rg] = (direc, sig_, val)
        ROWS.append((axis, conv, out))

    # 기준
    run('기준', 'med3·window_resid·nRMSE·부호·동점제외', 'window_resid', 'median', None, 'exclude', 'nRMSE', 'sign')
    run('기준', '〃 상대Δ 로 부호검정', 'window_resid', 'median', None, 'exclude', 'nRMSE', 'sign_rel')
    # 1) 시드 집계
    run('시드집계', '평균-of-3', 'window_resid', 'mean', None, 'exclude', 'nRMSE', 'sign')
    for sd in (1575, 1576, 1577):
        run('시드집계', f'단일시드 {sd} (배포조건)', 'window_resid', 'seed', sd, 'exclude', 'nRMSE', 'sign')
    # 2) 통계량
    run('통계량', 'Wilcoxon 부호순위', 'window_resid', 'median', None, 'exclude', 'nRMSE', 'wilcoxon')
    run('통계량', 'med_rel (S9 동결 크기)', 'window_resid', 'median', None, 'exclude', 'nRMSE', 'med_rel')
    run('통계량', '짝-평균', 'window_resid', 'median', None, 'exclude', 'nRMSE', 'mean_paired')
    run('통계량', '꼬리 nRMSE_p95', 'window_resid', 'median', None, 'exclude', 'nRMSE_p95', 'sign')
    # 3) 선택 통계량
    run('선택신호', 'all_resid', 'all_resid', 'median', None, 'exclude', 'nRMSE', 'sign')
    run('선택신호', 'avg_obs_loss (⛔λ 포함)', 'avg_obs_loss', 'median', None, 'exclude', 'nRMSE', 'sign')
    # 4) 동점 처리
    run('동점처리', '절반씩 배분', 'window_resid', 'median', None, 'half', 'nRMSE', 'sign')

    print(f'  {"축":10s} {"규약":30s} | {"희소":>34s} | {"혼잡":>34s}')
    for axis, conv, out in ROWS:
        cells = []
        for rg in ('sparse', 'crowded'):
            direc, sig_, val = out[rg]
            arrow = {-1: 'real승', 1: 'null승', 0: '무'}[direc]
            mark = '' if sig_ is None else ('*' if sig_ else ' ')
            cells.append(f'{arrow:6s}{mark} {val:>26s}')
        print(f'  {axis:10s} {conv:30s} | {cells[0]} | {cells[1]}')
    print('  (* = p<=0.05 · real승 = SEL_real 이 SEL_null 보다 낮은 nRMSE)')

    # 시드-쌍 전개 (에피소드×시드 를 단위로)
    print('\n  [시드-쌍 전개] 단위 = (에피소드, 시드쌍). ⛔S9 §3.3 이 금지한 규약 — 대조용만.')
    pairs = list(zip((1575, 1576, 1577), (1578, 1579, 1580)))
    for rg in ('sparse', 'crowded'):
        imp = wor = tie_ = 0
        for ts, ns in pairs:
            _, vr = selector(D, ['AUTO', 'V2K'], 'window_resid', 'seed', ts)
            # 널 팔: AUTO(ts) vs AUTO-SEED-NULL(ns) — 시드가 다르므로 직접 조립
            for ep in eps_of(reg, rg):
                a = ep_val(D, 'AUTO', ep, 'window_resid', 'seed', ts)
                b = ep_val(D, 'AUTO-SEED-NULL', ep, 'window_resid', 'seed', ns)
                if a is None or b is None:
                    continue
                nv = (ep_val(D, 'AUTO', ep, 'nRMSE', 'seed', ts) if a <= b
                      else ep_val(D, 'AUTO-SEED-NULL', ep, 'nRMSE', 'seed', ns))
                rv = vr('nRMSE')(ep)
                if rv is None or nv is None:
                    continue
                d = rv - nv
                imp += d < 0
                wor += d > 0
                tie_ += d == 0
        print(f'    {rg:8s} {imp}/{wor}/{tie_}  p={p_two_sided(max(imp,wor), imp+wor):.4g}')


# ── 시드 사용 스캔 ───────────────────────────────────────────────────────────
def sec_seeds():
    print('\n' + '=' * 92)
    print('§5-4 추정 시드 사용 전수 스캔 (tmp/**/*.jsonl)')
    print('=' * 92)
    used = set()
    import glob
    for p in glob.glob(os.path.join(TMP, '**', '*.jsonl'), recursive=True):
        try:
            for line in open(p):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                s = r.get('seed')
                if isinstance(s, int):
                    used.add(s)
        except Exception:
            pass
    hi = sorted(x for x in used if x >= 1500)
    print(f'  jsonl 에서 관측된 추정 시드 >=1500: {hi[0]}..{hi[-1]}, 총 {len(hi)}개')
    miss = [x for x in range(hi[0], hi[-1] + 1) if x not in used]
    print(f'  구간 내 미사용: {miss}')
    for s in (1581, 1582, 1583):
        print(f'  seed {s}: {"⛔사용됨" if s in used else "✅미사용"}')


SECTIONS = {'1': sec1, '2': sec2, '4': sec4, 'sens': sec_sens}

if __name__ == '__main__':
    D, reg, err = load()
    print(f'적재: 셀 {sorted(D)} · 에피소드 {len(reg)} · 에러 {err}')
    print(f'레짐: sparse {len(eps_of(reg,"sparse"))} · crowded {len(eps_of(reg,"crowded"))}')
    want = sys.argv[1:] or ['1', '2', '4', 'sens', 'seeds']
    for w in want:
        if w == 'seeds':
            sec_seeds()
        else:
            SECTIONS[w](D, reg)
