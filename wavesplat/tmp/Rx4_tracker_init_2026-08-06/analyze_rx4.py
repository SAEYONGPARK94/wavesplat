"""Rx-4 scoring -- implements tmp/Rx4_tracker_init_2026-08-06/PREREG.md sec.4-6 verbatim.
Reads results.jsonl + nulls.json only."""
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl')) if l.strip()]
ok = [r for r in rows if 'error' not in r]
print(f'rows={len(rows)} ok={len(ok)} errors={len(rows)-len(ok)}')
for r in rows:
    if 'error' in r:
        print('  ERR', r['job_id'], r['error'][:160])

# 2026-08-07: nulls.json (frozen) covered seed{0,1}x{uniform,slope}, but block C is
# seed{0,1,2,3}xuniform -> 8 episodes had no null.  nulls_full.json = the frozen 24
# reproduced BIT-EXACTLY (worst |diff| 0.000e+00) + the 8 missing, same code path.
# See nulls_missing.py / nulls_missing.log.  nulls.json itself is NOT modified.
NULL = {r['episode']: r['zero_null_tracking_nRMSE_mean']
        for r in json.load(open(os.path.join(HERE, 'nulls_full.json')))}
CROWDED16 = [f'episode_BF{bf}_{nc}comp_seed{es}_uniform.h5'
             for bf in (4, 5) for nc in (15, 30) for es in (0, 1, 2, 3)]
CROWDED4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
SPARSE8 = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
           'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
           'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
           'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5']
CELLS = ['AUTO', 'AUTO-DUP', 'AUTO-NOTRACK', 'T-PHIONLY', 'T-WONLY', 'T-EARLY20', 'T-K4REF']
CFG_GROUPS = {'BF4_15': CROWDED16[0:4], 'BF4_30': CROWDED16[4:8],
              'BF5_15': CROWDED16[8:12], 'BF5_30': CROWDED16[12:16]}

IDX = {}
for r in ok:
    IDX.setdefault((r['cell'], r['episode']), []).append(r)


def med(cell, ep, key):
    v = [x[key] for x in IDX.get((cell, ep), []) if x.get(key) is not None]
    v = [x for x in v if x == x]
    return st.median(v) if v else None


def sign_p(k, n):
    """two-sided exact sign test; k = wins, ties already excluded from k but kept in n"""
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, lo + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------- sec.6 integrity
print('\n' + '=' * 78)
print('=== sec.6.2  PHASE-1 INVARIANCE (init_nrmse / auto_selected / auto_n / k / N) ===')
bad = 0
eps_all = sorted({r['episode'] for r in ok})
seeds_all = sorted({r['seed'] for r in ok})
for e in eps_all:
    for s in seeds_all:
        vals = {}
        for c in CELLS:
            rr = [x for x in ok if x['cell'] == c and x['episode'] == e and x['seed'] == s]
            if rr:
                vals[c] = (round(rr[0]['init_nrmse'], 12), rr[0]['auto_selected'],
                           rr[0]['auto_n_selected'], rr[0]['ensemble_k'], rr[0]['N'])
        if len(set(vals.values())) > 1:
            bad += 1
            print(f'  MISMATCH {e[8:-3]} s{s}:')
            for c, v in vals.items():
                print(f'     {c:14s} {v}')
print(f'  mismatches = {bad}   -> {"PASS" if bad == 0 else "FAIL (affected cells VOID)"}')

print('\n=== sec.6.3  WORKER-LEAK PROBE: AUTO-DUP vs AUTO, bit-identical? ===')
nb = nc = 0
KEYS = ['nRMSE', 'Pearson_rho', 'Variance_Ratio', 'absVR_framewise', 'm_mean',
        'decorr_mean', 'shrink_mean', 'init_nrmse', 'N', 'Skill_Score']
for e in eps_all:
    for s in seeds_all:
        a = [x for x in ok if x['cell'] == 'AUTO' and x['episode'] == e and x['seed'] == s]
        d = [x for x in ok if x['cell'] == 'AUTO-DUP' and x['episode'] == e and x['seed'] == s]
        if not a or not d:
            continue
        for k in KEYS:
            nc += 1
            if a[0][k] != d[0][k]:
                nb += 1
                print(f'  DIFF {e[8:-3]} s{s} {k}: {a[0][k]} vs {d[0][k]}')
print(f'  comparisons={nc}  mismatches={nb} -> {"PASS" if nb == 0 else "FAIL"}')


# ---------------------------------------------------------------- sec.4 null rules
def null_report(block, name):
    print(f'\n=== sec.4 NULL RULES on {name} ===')
    excl = []
    print(f"  {'episode':26s} {'null':>7s} {'AUTO':>8s}  excluded?")
    for e in block:
        a = med('AUTO', e, 'nRMSE')
        n = NULL[e]
        ex = a is not None and a >= n
        if ex:
            excl.append(e)
        print(f'  {e[8:-3]:26s} {n:7.4f} {a:8.4f}  {"EXCLUDED (AUTO >= null)" if ex else ""}')
    print(f'  -> excluded {len(excl)}/{len(block)}: {[e[8:-3] for e in excl]}')
    nullmed = st.median([NULL[e] for e in block])
    print(f'  block null median = {nullmed:.4f}')
    print(f'  (b) cell-level NULL-FAILING check (medOfMed nRMSE >= block null median):')
    failing = []
    for c in CELLS:
        v = [med(c, e, 'nRMSE') for e in block]
        v = [x for x in v if x is not None]
        if not v:
            continue
        mm = st.median(v)
        f = mm >= nullmed
        if f:
            failing.append(c)
        print(f'     {c:14s} medOfMed={mm:.4f} {"<-- NULL-FAILING" if f else ""}')
    return excl, nullmed, failing


def score(block, name, ref, cells=None, tag=''):
    cells = cells or [c for c in CELLS if c != ref]
    n = len(block)
    print(f'\n--- {name} | REF = {ref} | n={n} {tag} ---')
    print(f"  {'cell':14s} {'medOfMed':>9s} {'medΔ':>9s} {'Δrel%':>8s} {'wins':>7s} "
          f"{'p(sign)':>9s}")
    out = {}
    for c in cells:
        d, rel = [], []
        for e in block:
            a, b = med(c, e, 'nRMSE'), med(ref, e, 'nRMSE')
            if a is None or b is None:
                continue
            d.append(a - b)
            auto = med('AUTO', e, 'nRMSE')
            rel.append((a - b) / auto if auto else float('nan'))
        if not d:
            continue
        wins = sum(1 for x in d if x < 0)
        lvl = st.median([med(c, e, 'nRMSE') for e in block])
        out[c] = dict(med=st.median(d), wins=wins, n=len(d), p=sign_p(wins, len(d)),
                      lvl=lvl, deltas=d)
        print(f'  {c:14s} {lvl:9.4f} {st.median(d):+9.4f} '
              f'{st.median(rel)*100:+7.1f}% {wins:3d}/{len(d):<3d} '
              f'{sign_p(wins, len(d)):9.4f}')
    return out


for block, name in ((CROWDED16, 'BLOCK C (CROWDED16)'), (SPARSE8, 'BLOCK S (SPARSE8)')):
    print('\n' + '=' * 78)
    excl, nullmed, failing = null_report(block, name)
    kept = [e for e in block if e not in excl]
    print('\n' + '=' * 78)
    print(f'=== {name} -- PRIMARY (null-excluded, n={len(kept)}) ===')
    vsauto = score(kept, name, 'AUTO')
    vsfloor = score(kept, name, 'AUTO-NOTRACK')
    print(f'\n  recovery R = med(Δcell-AUTO)/med(Δnotrack-AUTO)   [DESCRIPTIVE ONLY]')
    base = vsauto.get('AUTO-NOTRACK', {}).get('med')
    for c, v in vsauto.items():
        if c in ('AUTO-DUP',) or base in (None, 0):
            continue
        print(f'     {c:14s} R = {100*v["med"]/base:6.1f}% of the NOTRACK floor')
    if excl:
        print(f'\n  --- transparency: SAME SCORING on the FULL block (n={len(block)}, '
              f'no exclusion) ---')
        score(block, name + ' [unexcluded]', 'AUTO')
        score(block, name + ' [unexcluded]', 'AUTO-NOTRACK')

    if block is CROWDED16:
        print('\n' + '=' * 78)
        print('=== BLOCK C -- CONFIG-LEVEL FOLD (n=4).  floor p=0.125 => NO "significant" ===')
        for how in ('fold-delta', 'fold-level'):
            print(f'  [{how}]')
            print(f"  {'cell':14s} {'vs AUTO medΔ':>13s} {'wins':>6s} | "
                  f"{'vs FLOOR medΔ':>14s} {'wins':>6s}")
            for c in CELLS:
                if c == 'AUTO':
                    continue
                row = []
                for ref in ('AUTO', 'AUTO-NOTRACK'):
                    per = []
                    for g, eps in CFG_GROUPS.items():
                        use = [e for e in eps if e in kept]
                        if not use:
                            continue
                        if how == 'fold-delta':
                            per.append(st.median([med(c, e, 'nRMSE') - med(ref, e, 'nRMSE')
                                                  for e in use]))
                        else:
                            per.append(st.median([med(c, e, 'nRMSE') for e in use])
                                       - st.median([med(ref, e, 'nRMSE') for e in use]))
                    row.append((st.median(per), sum(1 for x in per if x < 0), len(per)))
                print(f'  {c:14s} {row[0][0]:+13.4f} {row[0][1]:3d}/{row[0][2]:<2d} | '
                      f'{row[1][0]:+14.4f} {row[1][1]:3d}/{row[1][2]:<2d}')

        print('\n' + '=' * 78)
        print('=== BLOCK C -- PANEL (sec.4.4 directions), REF=AUTO, null-excluded ===')
        for key, good in (('Pearson_rho', +1), ('absVR_framewise', -1), ('m_mean', -1),
                          ('decorr_mean', -1), ('shrink_mean', -1)):
            print(f'  [{key}]  (good = {"Δ>0" if good > 0 else "Δ<0"})')
            for c in CELLS:
                if c == 'AUTO':
                    continue
                d = [med(c, e, key) - med('AUTO', e, key) for e in kept
                     if med(c, e, key) is not None and med('AUTO', e, key) is not None]
                if not d:
                    continue
                wins = sum(1 for x in d if x * good > 0)
                print(f'     {c:14s} medΔ={st.median(d):+8.4f} wins={wins:3d}/{len(d):<3d} '
                      f'p={sign_p(wins, len(d)):.4f}')

        print('\n' + '=' * 78)
        print('=== sec.5 VERDICT (frozen rules) ===')
        need = 13
        A = vsauto.get('T-PHIONLY')
        B = vsfloor.get('T-PHIONLY')
        if A and B:
            nA = A['n']
            need = min(k for k in range(nA + 1) if sign_p(k, nA) <= 0.05 and k > nA / 2)
            condA = A['med'] < 0 and A['wins'] >= need
            floor_wins = B['n'] - B['wins']          # NOTRACK better than T-PHIONLY
            condB = floor_wins < need
            strong = B['med'] < 0 and B['wins'] >= need
            v = ('STRONG PASS' if condA and strong else
                 'PASS' if condA and condB else
                 'PARTIAL (floor still better -> intervention pointless)' if condA else 'FAIL')
            print(f'  n={nA}, wins needed for two-sided p<=0.05 : {need}')
            print(f'  A (beats shipped tracker): medΔ={A["med"]:+.4f} wins={A["wins"]}/{nA} '
                  f'p={A["p"]:.4f} -> {condA}')
            print(f'  B (reaches the floor)    : medΔ(vs NOTRACK)={B["med"]:+.4f} '
                  f'wins={B["wins"]}/{B["n"]} (NOTRACK wins {floor_wins}) '
                  f'p={B["p"]:.4f} -> {condB}')
            print(f'  PRIMARY VERDICT (T-PHIONLY, BLOCK C): **{v}**')
            if 'T-PHIONLY' in failing:
                print('  !! T-PHIONLY is NULL-FAILING on this block -> VOID')
        print('\n  --- D1 (diagnosis falsification test) ---')
        p_, w_ = vsauto.get('T-PHIONLY'), vsauto.get('T-WONLY')
        if p_ and w_:
            print(f'     diagnosis predicted the AMPLITUDE channel carries the damage,')
            print(f'     i.e. |Δ(T-PHIONLY-AUTO)| > |Δ(T-WONLY-AUTO)|')
            print(f'     measured: T-PHIONLY {p_["med"]:+.4f}  vs  T-WONLY {w_["med"]:+.4f}')
            print(f'     -> D1 {"SUPPORTED" if abs(p_["med"]) > abs(w_["med"]) else "REFUTED"}')
            dd = [a - b for a, b in zip(p_['deltas'], w_['deltas'])]
            wn = sum(1 for x in dd if x < 0)
            print(f'     paired head-to-head T-PHIONLY vs T-WONLY: medΔ={st.median(dd):+.4f} '
                  f'wins={wn}/{len(dd)} p={sign_p(wn, len(dd)):.4f}')
        e_ = vsauto.get('T-EARLY20')
        if e_ and base:
            print(f'\n  --- D2 (monotone accumulation) ---')
            print(f'     T-EARLY20 Δ={e_["med"]:+.4f}; AUTO=0.0000, floor={base:+.4f}')
            print(f'     between? {"YES" if base <= e_["med"] <= 0 else "NO"}  '
                  f'(R={100*e_["med"]/base:.1f}%)')
