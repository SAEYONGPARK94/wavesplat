"""E-B2K result audit (fable, round 4) -- full independent re-derivation.

Written from scratch against the RAW JSONs only. No imports from e60k_* or
e15_* code. Own wild-bootstrap implementation (different rng seed + larger MC
than the judge), exact Rademacher sign-flip enumeration, LOO with full frozen
bar, per-seed splits, weighted (cell-level) robustness reads, two-sided
variants, real-leg recomputation, cache identity, boundary-margin checks.
"""
import hashlib
import itertools
import json

import numpy as np
from scipy import stats

SEEDS = (0, 1, 2)
ARENA = 'results/direction3/e60k_hos_arena.json'
E15 = 'results/direction3/e15_hos_multidraw.json'
REAL = 'results/direction3/e60k_real_arena.json'
JUDG = 'results/direction3/e60k_judgment.json'

# The ruling's pinned failure set (R3), typed in from the ruling document.
PINNED = {
    'Tp9_Hs4': [3], 'Tp10_Hs3': [4], 'Tp10_Hs4': [2], 'Tp10_Hs5': [2, 4],
    'Tp11_Hs2': [1], 'Tp11_Hs3': [1, 2], 'Tp11_Hs4': [0, 1, 2, 4],
    'Tp11_Hs5': [0, 2, 3], 'Tp12_Hs5': [3], 'Tp13_Hs4': [0], 'Tp13_Hs5': [3]}

WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5),
                 np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def tstat(m):
    k = len(m)
    return m.mean() / (m.std(ddof=1) / np.sqrt(k)), k


def wild_p_own(m, n_mc=400_000, seed=12345, two_sided=False):
    """Null-imposed Webb wild cluster bootstrap-t. Own implementation,
    own rng seed (12345, judge used 0), 2x the MC draws."""
    t_obs, k = tstat(m)
    w = np.random.default_rng(seed).choice(WEBB, size=(n_mc, k))
    yb = m[None, :] * w
    tb = yb.mean(1) / (yb.std(1, ddof=1) / np.sqrt(k))
    if two_sided:
        return float((np.abs(tb) >= abs(t_obs)).mean())
    return float((tb >= t_obs).mean())


def signflip_exact(m, two_sided=False):
    """EXACT Rademacher sign-flip over all 2^k assignments (valid under
    per-cluster sign symmetry; heteroskedasticity-robust)."""
    t_obs, k = tstat(m)
    s = np.array(list(itertools.product([-1.0, 1.0], repeat=k)))
    yb = m[None, :] * s
    tb = yb.mean(1) / (yb.std(1, ddof=1) / np.sqrt(k))
    if two_sided:
        return float((np.abs(tb) >= abs(t_obs) - 1e-12).mean())
    return float((tb >= t_obs - 1e-12).mean())


def contrast(rows, a, b, failure):
    by = {}
    for r in rows:
        if failure and r['hand'] >= 0.5:
            continue
        d = np.mean([r[f'{a}_s{s}'] - r[f'{b}_s{s}'] for s in SEEDS])
        by.setdefault(r['sea'], []).append(d)
    keys = sorted(by)
    m = np.array([np.mean(by[k]) for k in keys])
    return keys, m, by


def report(tag, m):
    t, k = tstat(m)
    p1 = stats.t.sf(t, k - 1)
    wp1 = wild_p_own(m)
    wp2 = wild_p_own(m, two_sided=True)
    ex1 = signflip_exact(m) if k <= 20 else float('nan')
    ex2 = signflip_exact(m, two_sided=True) if k <= 20 else float('nan')
    print(f'{tag:12s} k {k:2d} mean {m.mean():+.4f} t {t:+.3f} p1 {p1:.4f} '
          f'wild1 {wp1:.4f} wild2 {wp2:.4f} exact1 {ex1:.6f} exact2 {ex2:.6f} '
          f'pos {(m > 0).sum()}/{k}')
    return dict(k=k, mean=float(m.mean()), t=float(t), p1=float(p1),
                wild1=wp1, wild2=wp2, exact1=ex1, exact2=ex2)


def main():
    arena = json.load(open(ARENA))
    rows = arena['rows']
    e15 = json.load(open(E15))['rows']

    print('=== 1. ARENA INTEGRITY ===')
    seas = sorted({r['sea'] for r in rows})
    print(f'rows {len(rows)}  seas {len(seas)}  '
          f'draws/sea {sorted({sum(1 for r in rows if r["sea"] == s) for s in seas})}')
    e15h = {(r['sea'], r['draw']): r['hand'] for r in e15}
    mism = [(r['sea'], r['draw']) for r in rows
            if e15h.get((r['sea'], r['draw'])) != r['hand']]
    print(f'hand vs E-15 cache: EXACT-EQUAL mismatches = {len(mism)}')
    e15_draws = sorted({r['draw'] for r in e15})
    print(f'E-15 cache draws available: {e15_draws} '
          f'(arena uses 0-4 only: unused draws = {[d for d in e15_draws if d > 4]})')
    sha = hashlib.sha256(open(E15, 'rb').read()).hexdigest()
    print(f'E-15 cache sha256[:16] = {sha[:16]}  (ruling pinned 6673898b9568a6a3)')

    fail = {}
    for r in rows:
        if r['hand'] < 0.5:
            fail.setdefault(r['sea'], []).append(r['draw'])
    fail = {s: sorted(v) for s, v in fail.items()}
    ok = fail == {s: sorted(v) for s, v in PINNED.items()}
    ncell = sum(len(v) for v in fail.values())
    print(f'realized failure set: {len(fail)} seas / {ncell} cells; '
          f'matches ruling pin exactly: {ok}')
    singles = sum(1 for v in fail.values() if len(v) == 1)
    print(f'single-draw clusters: {singles}/{len(fail)}  '
          f'sizes {sorted((len(v) for v in fail.values()), reverse=True)}')
    hands = np.array([r['hand'] for r in rows])
    below = hands[hands < 0.5].max()
    above = hands[hands >= 0.5].min()
    print(f'threshold margins: max-below {below:.4f}  min-above {above:.4f}')

    print('\n=== 2. PRIMARY STATISTICS (re-derived) ===')
    out = {}
    for a, b in (('T', 'L'), ('T', 'N'), ('L', 'N')):
        for failure, lab in ((True, 'fail'), (False, 'all')):
            keys, m, _ = contrast(rows, a, b, failure)
            out[f'{a}{b}_{lab}'] = report(f'{a}-{b} {lab}', m)

    print('\n=== 3. CROSS-CHECK vs judge output ===')
    jd = json.load(open(JUDG))
    for key in ('TL_fail', 'TN_fail', 'LN_fail', 'TL_all'):
        mine, theirs = out[key], jd['hos'][key]
        d = max(abs(mine['mean'] - theirs['mean']), abs(mine['t'] - theirs['t']),
                abs(mine['p1'] - theirs['p']))
        print(f'{key:8s} max|diff(mean,t,p)| = {d:.2e}  '
              f'judge wild {theirs["wild_p"]:.4f} vs mine {mine["wild1"]:.4f}')

    print('\n=== 4. LEAVE-ONE-SEA-OUT (T-L failure), full frozen bar ===')
    keys, m, by = contrast(rows, 'T', 'L', True)
    for i, kk in enumerate(keys):
        mm = np.delete(m, i)
        t, k = tstat(mm)
        p1 = stats.t.sf(t, k - 1)
        wp = wild_p_own(mm)
        ex = signflip_exact(mm)
        bar = mm.mean() > 0.02 and p1 < 0.05 and wp < 0.05
        print(f'drop {kk:10s} mean {mm.mean():+.4f} t {t:+.2f} p1 {p1:.4f} '
              f'wild1 {wp:.4f} exact1 {ex:.6f}  frozen-bar {"PASS" if bar else "FAIL"}')

    print('\n=== 5. Tp13_Hs5 CELL DETAIL (the +0.61 single draw) ===')
    for r in rows:
        if r['sea'] == 'Tp13_Hs5' and r['hand'] < 0.5:
            print(f'draw {r["draw"]}  hand {r["hand"]:+.4f}')
            for arm in ('T', 'L', 'N'):
                v = [r[f'{arm}_s{s}'] for s in SEEDS]
                print(f'  {arm}: {[round(x, 4) for x in v]}  mean {np.mean(v):+.4f}')
    # other draws of the same sea for context
    print('non-failure draws of Tp13_Hs5 (hand, T-L seedmean):')
    for r in rows:
        if r['sea'] == 'Tp13_Hs5' and r['hand'] >= 0.5:
            d = np.mean([r[f'T_s{s}'] - r[f'L_s{s}'] for s in SEEDS])
            print(f'  draw {r["draw"]} hand {r["hand"]:+.4f}  T-L {d:+.4f}')

    print('\n=== 6. ROBUSTNESS READS (report-only, NOT part of the frozen bar) ===')
    # (a) per-seed split
    for s in SEEDS:
        by_s = {}
        for r in rows:
            if r['hand'] >= 0.5:
                continue
            by_s.setdefault(r['sea'], []).append(r[f'T_s{s}'] - r[f'L_s{s}'])
        ms = np.array([np.mean(by_s[k]) for k in sorted(by_s)])
        t, k = tstat(ms)
        print(f'seed {s} only : mean {ms.mean():+.4f} t {t:+.2f} '
              f'p1 {stats.t.sf(t, k - 1):.4f} exact1 {signflip_exact(ms):.6f}')
    # (b) cell-weighted (n=18 cells, sea-clustered CR t via cluster means of
    #     cell values is what the primary does; here: pooled cells, cluster
    #     bootstrap on seas) -- report the pooled cell mean
    cells = [np.mean([r[f'T_s{s}'] - r[f'L_s{s}'] for s in SEEDS])
             for r in rows if r['hand'] < 0.5]
    print(f'pooled 18-cell mean (unclustered, report-only): {np.mean(cells):+.4f}')
    # (c) median-based
    print(f'median of 11 cluster means: {np.median(m):+.4f}')
    # (d) trimmed: drop Tp13_Hs5 AND Tp9_Hs4 (best & worst)
    mt = np.array([v for kk, v in zip(keys, m) if kk not in ('Tp13_Hs5', 'Tp9_Hs4')])
    t, k = tstat(mt)
    print(f'drop best+worst: mean {mt.mean():+.4f} t {t:+.2f} '
          f'p1 {stats.t.sf(t, k - 1):.4f} exact1 {signflip_exact(mt):.6f}')

    print('\n=== 7. REAL LEG (k=5, non-confirmatory) re-derived ===')
    rrows = json.load(open(REAL))['rows']
    recs = sorted({r['record'] for r in rrows})
    print(f'records {recs}  rows {len(rrows)}  '
          f'hand_source {sorted({r.get("hand_source", "?") for r in rrows})}')
    assert 'BS11a' not in recs, 'BS11a rows exist?!'
    for a, b in (('T', 'L'), ('T', 'N'), ('L', 'N')):
        for failure, lab in ((True, 'fail'), (False, 'all')):
            by = {}
            for r in rrows:
                if failure and r['hand'] >= 0.5:
                    continue
                d = np.mean([r[f'{a}_s{s}'] - r[f'{b}_s{s}'] for s in SEEDS])
                by.setdefault(r['site_day'], []).append(d)
            mR = np.array([np.mean(by[k]) for k in sorted(by)])
            t, k = tstat(mR)
            print(f'{a}-{b} {lab:4s} k {k} mean {mR.mean():+.4f} t {t:+.2f} '
                  f'p1 {stats.t.sf(t, k - 1):.4f} exact1 {signflip_exact(mR):.6f} '
                  f'pos {(mR > 0).sum()}/{k}')
    # cond-2 arithmetic
    by = {}
    for r in rrows:
        if r['hand'] >= 0.5:
            continue
        by.setdefault(r['site_day'], []).append(
            np.mean([r[f'T_s{s}'] - r[f'L_s{s}'] for s in SEEDS]))
    mR = np.array([np.mean(by[k]) for k in sorted(by)])
    print(f'cond2: mean {mR.mean():+.4f} >= +0.05: {mR.mean() >= 0.05}  '
          f'pos {(mR > 0).sum()}/5 (bar >=4): {(mR > 0).sum() >= 4}')

    print('\n=== 8. NLL TIE BRANCH + VOID (from manipulation JSON) ===')
    man = json.load(open('results/direction3/e60k_manipulation.json'))
    sN = man['N']['mean']
    tn_mean = out['TN_fail']['mean']
    print(f'T-N fail mean {tn_mean:+.4f} (branch needs < +0.02: {tn_mean < 0.02})')
    print(f'smult_N {sN:.3f} in [2.5,4.0]: {2.5 <= sN <= 4.0}')
    print(f'across-seed stds: T {man["T"]["std"]:.3f} L {man["L"]["std"]:.3f} '
          f'N {man["N"]["std"]:.3f}  (VOID bar 0.5) void={man["void"]}')

    print('\n=== 9. MDE at realized cluster sd ===')
    sd = m.std(ddof=1)
    mde = stats.t.ppf(0.95, 10) * sd / np.sqrt(11)
    print(f'TL_fail realized sd {sd:.4f}  MDE(one-sided, a=.05) {mde:+.4f}  '
          f'observed {m.mean():+.4f}')
    sd_loo = np.delete(m, list(keys).index('Tp13_Hs5')).std(ddof=1)
    print(f'sd without Tp13_Hs5: {sd_loo:.4f}  '
          f'MDE(k=10) {stats.t.ppf(0.95, 9) * sd_loo / np.sqrt(10):+.4f}')


if __name__ == '__main__':
    main()
