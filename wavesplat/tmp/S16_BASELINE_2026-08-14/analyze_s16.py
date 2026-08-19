#!/usr/bin/env python3
"""S16 analyzer -- follows PREREG.md (frozen) exactly.  Nothing here was chosen after
seeing data; the comparator-selection rule is PREREG section 1.1, frozen before any run.

    python3 tmp/S16_BASELINE_2026-08-14/analyze_s16.py

Refuses to print a verdict if ANY row is missing a required field (PREREG section 2.1,
discipline 2 -- S13 and HOSGATE both died there).
"""
import collections
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_s16 as R                                   # noqa: E402

ALPHA = 0.025
THRESH = 41
N_PER = 64
PRIMARY_STRATUM = 'crowded'
POOL = ('SPEC25', 'SPEC50', 'SPEC100')                # PREREG 1.1
POOL_ALL = ('SPEC25', 'SPEC50', 'SPEC100', 'WHITE50', 'WHITER50')
SEC_SHELL = 'obs:150_inf'                             # PREREG 9, the ONE registered shell
# AMENDMENT_1 B2: R-2 is FIXED to SPEC50 vs WHITER50 -- same loc_radius, same code path,
# same RNG stream, same thinning subset -- so `loc` is no longer a free factor in the
# contrast.  `SPEC_best vs WHITER50` (whose loc may differ from 50) is demoted to
# EXPLORATORY under E-3.  R-2 remains a TWO-factor contrast (spectral structure AND
# spread 0.1 -> Hs/4); that pair is the repair as PREREG section 4 defines it, and R-2 is
# labelled accordingly -- it must NOT be read as "spectral structure alone".
R2_ARM = 'SPEC50'


# ── loading + the field-existence gate ────────────────────────────────────────
def base_med(by_seed):
    """PREREG G-6 escalation (fired -- see AMENDMENT_1 / NOTES 19:38): the baseline is
    aggregated as a median over the filter seeds, for the scalar AND for every shell,
    exactly as `ship_med3` does for the shipping arm."""
    seeds = sorted(by_seed)
    rows = [by_seed[s] for s in seeds]
    out = dict(rows[0])
    out['avg_nRMSE'] = float(np.median([r['avg_nRMSE'] for r in rows]))
    out['zero_nRMSE'] = float(np.median([r['zero_nRMSE'] for r in rows]))
    for fld in ('shell_rmse', 'zero_shell', 'shell_frac'):
        agg = {}
        for k in R.SHELL_KEYS:
            v = [(r.get(fld) or {}).get(k) for r in rows]
            v = [x for x in v if x is not None and np.isfinite(x)]
            agg[k] = float(np.median(v)) if v else np.nan
        out[fld] = agg
    out['filter_seeds'] = seeds
    out['n_seeds'] = len(seeds)
    out['per_seed'] = {str(s): by_seed[s]['avg_nRMSE'] for s in seeds}
    return out


def load():
    path = os.path.join(HERE, 'results.jsonl')
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    errs = [r for r in rows if 'error' in r]
    good = [r for r in rows if 'error' not in r]
    raw = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in good:
        raw[r['episode']][r['arm']][int(r.get('filter_seed', 42))] = r
    D = collections.defaultdict(dict)
    for ep, arms in raw.items():
        for a, by_seed in arms.items():
            D[ep][a] = base_med(by_seed)
    ship = R.import_ship()
    for k, v in ship.items():
        D[k]['SHIP'] = dict(ship_med3(v), episode=k, arm='SHIP',
                            stratum=None, effective_config={'seed': sorted(v)})
    return D, good, errs, ship


def ship_med3(v):
    return R.ship_med3(v)


def field_gate(good, ship):
    """PREREG G-3: every arm must carry every required field.  Hard stop."""
    bad = []
    for r in good:
        miss = R.check_row(r)
        if miss:
            bad.append((r['episode'], r['arm'], miss[:6]))
    for k, v in ship.items():
        for s, r in v.items():
            miss = [f for f in R.REQUIRED_SHIP if not r.get(f)]
            miss += ['shell_rmse[%s]' % j for j in R.SHELL_KEYS
                     if j.split(':', 1)[1] not in (r.get('shell_rmse') or {})]
            if miss:
                bad.append((k, 'SHIP@%s' % s, miss[:6]))
    print('=' * 78)
    print(f'FIELD GATE (PREREG 2.1): rows={len(good)} SHIP rows={sum(len(v) for v in ship.values())} '
          f'violations={len(bad)}')
    for b in bad[:10]:
        print('   ', b)
    if bad:
        print('\n*** VERDICT WITHHELD -- a required field is missing. '
              'This is the failure that killed S13 S-2a and HOSGATE. ***')
        sys.exit(2)
    print('    PASS')
    return True


# ── statistics (PREREG section 5) ─────────────────────────────────────────────
def sign_one_sided(a, b):
    """H1: a < b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (a != b)
    w = int((a[m] < b[m]).sum())
    n = int(m.sum())
    p = float(stats.binomtest(w, n, 0.5, alternative='greater').pvalue) if n else 1.0
    return w, n, p


def effects(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rel = (a - b) / b
    return (100 * np.median(rel),
            100 * (np.median(a) - np.median(b)) / np.median(b),
            100 * (a.mean() - b.mean()) / b.mean())


def holm(ps):
    order = np.argsort(ps)
    out = np.empty(len(ps))
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(ps) - i) * ps[j])
        out[j] = min(1.0, run)
    return out


def report(label, a, b, na, nb):
    w, n, p = sign_one_sided(a, b)
    mr, rm, rmn = effects(a, b)
    print(f'  {label}')
    # CON W5 (AMENDMENT_1): the fixed threshold 41 is kept even if ties/NaNs shrink n
    # (conservative), but then the printed "41/64" misstates the realised alpha, so the
    # exact alpha AT THE REALISED n is printed alongside it.
    a_n = (float(stats.binom.sf(THRESH - 1, n, 0.5)) if n else float('nan'))
    print(f'    {na} < {nb}: {w}/{n}  (threshold {THRESH}/{N_PER}, exact alpha at n={n}: '
          f'{a_n:.6g})  one-sided p={p:.4g}')
    print(f'    size, 3 conventions (mandatory): med(rel)={mr:+.1f}%  '
          f'rel(med)={rm:+.1f}%  rel(mean)={rmn:+.1f}%')
    return dict(wins=w, n=n, p=p, med_rel=mr, rel_med=rm, rel_mean=rmn,
                passes=bool(w >= THRESH))


# ── main ──────────────────────────────────────────────────────────────────────
def keys_of(D, stratum, arms):
    out = []
    for k, v in D.items():
        if 'SHIP' not in v or any(a not in v for a in arms):
            continue
        st = next((v[a]['stratum'] for a in arms if v[a].get('stratum')), None)
        if st == stratum:
            out.append(k)
    return sorted(out)


def col(D, ks, arm, shell=None):
    if shell is None:
        return [D[k][arm]['avg_nRMSE'] for k in ks]
    return [(D[k][arm].get('shell_rmse') or {}).get(shell, np.nan) for k in ks]


def zcol(D, ks, arm, shell=None):
    if shell is None:
        return [D[k][arm]['zero_nRMSE'] for k in ks]
    return [(D[k][arm].get('zero_shell') or {}).get(shell, np.nan) for k in ks]


def main():
    D, good, errs, ship = load()
    print(f'rows={len(good)}  errors={len(errs)}  episodes={len(D)}')
    # G-8 (AMENDMENT_1, neutral audit A9): PREREG requires a SINGLE code_md5; the old
    # analyzer only PRINTED the distinct set and never judged it.
    digs = sorted({r['code_md5'] for r in good})
    nfil = sorted({r.get('code_files_n') for r in good})
    g8 = (len(digs) <= 1)
    print(f'G-8 code_md5 uniqueness: distinct={len(digs)} {digs} files_n={nfil} '
          f'-> {"PASS" if g8 else "FAIL"}')
    if not g8:
        print('*** G-8 FAIL: the rows were not produced by one code state. Any verdict '
              'below must be adjudicated by hand in RESULT.md before it is quoted. ***')
    field_gate(good, ship)

    froz = open(os.path.join(HERE, 'FROZEN_md5.txt')).read().split()
    out = {'prereg_md5': froz[0], 'frozen_md5_txt': froz,
           'G8_code_md5_unique': g8, 'code_md5': digs}
    # an arm counts as present only if it covers EVERY episode that has a SHIP row
    n_ep = len([k for k in D if 'SHIP' in D[k]])
    cnt = collections.Counter()
    seedn = collections.defaultdict(collections.Counter)
    for k, v in D.items():
        for a, r in v.items():
            if a == 'SHIP':
                continue
            cnt[a] += 1
            seedn[a][r['n_seeds']] += 1
    have = sorted(a for a, c in cnt.items() if c == n_ep)
    print(f'arm coverage: { {a: c for a, c in sorted(cnt.items())} } of {n_ep} episodes')
    print(f'baseline filter seeds per arm (PREREG G-6 escalation -> median-of-3 required): '
          f'{ {a: dict(s) for a, s in sorted(seedn.items())} }')
    full_seeds = {a: all(n == len(R.BASELINE_SEEDS) for n in seedn[a]) for a in cnt}
    prov = [a for a in have if not full_seeds.get(a)]
    if prov:
        print(f'*** PROVISIONAL: {prov} do not carry all {len(R.BASELINE_SEEDS)} filter seeds. '
              f'Every verdict below is PROVISIONAL and must NOT be quoted as the batch result. ***')
    out_prov = bool(prov)
    print(f'arms present: {have}')
    missing_pool = [a for a in POOL if a not in have]
    if missing_pool:
        print(f'*** PRIMARY pool incomplete {missing_pool} -- phase 1 not finished. ***')
        sys.exit(3)

    ks = keys_of(D, PRIMARY_STRATUM, POOL)
    print(f'\nPRIMARY stratum={PRIMARY_STRATUM} n={len(ks)}')
    if len(ks) != N_PER:
        print(f'*** n != {N_PER}; PRIMARY not evaluable yet. ***')
        sys.exit(3)
    shipv = col(D, ks, 'SHIP')

    # --- PREREG 1.1: frozen comparator selection = the STRONGEST baseline ---
    print('\n' + '=' * 78)
    print('COMPARATOR SELECTION (PREREG 1.1, frozen before any run):')
    print('  r* = argmin_a #{SHIP < a}   i.e. the baseline SHIP beats LEAST OFTEN')
    cand = {}
    for a in POOL:
        w, n, p = sign_one_sided(shipv, col(D, ks, a))
        mr = effects(shipv, col(D, ks, a))[0]
        cand[a] = dict(wins=w, n=n, p=p, med_rel=mr,
                       loc=float(D[ks[0]][a]['loc_radius']))
        print(f'  {a:9s} loc={cand[a]["loc"]:5.1f}  SHIP wins {w}/{n}  med(rel)={mr:+.1f}%')
    best = sorted(POOL, key=lambda a: (cand[a]['wins'], -cand[a]['med_rel'], cand[a]['loc']))[0]
    print(f'  -> r* = {best}   (tie-break 1: larger med(rel); tie-break 2: smaller loc)')
    out['candidates'] = cand
    out['r_star'] = best

    if all(a in have for a in POOL_ALL):
        c2 = {a: sign_one_sided(shipv, col(D, ks, a))[0] for a in POOL_ALL}
        best_all = sorted(POOL_ALL, key=lambda a: (c2[a], a))[0]
        print(f'  secondary comparator over all arms: {c2}  -> r*_all = {best_all}')
        out['r_star_all'] = best_all

    # --- repair gates R-1, R-2 (PREREG 7.1) ---
    print('\n' + '=' * 78)
    print('REPAIR GATES (PREREG 7.1) -- PRIMARY is UNDECIDED unless both pass')
    r1 = report('R-1 necessary condition: repaired baseline beats the ZERO field',
                col(D, ks, best), zcol(D, ks, best), best, 'ZERO')
    if 'WHITE50' in have:
        wl = sum(1 for k in ks if not (D[k]['WHITE50']['avg_nRMSE']
                                       < D[k]['WHITE50']['zero_nRMSE']))
        bl = sum(1 for k in ks if not (D[k][best]['avg_nRMSE'] < D[k][best]['zero_nRMSE']))
        print(f'    zero-field losses: {best}={bl}  WHITE50={wl}  '
              f'(PREREG 0.1 measured WHITE50 = 0 in this stratum)')
        r1['losses_best'], r1['losses_white'] = bl, wl
        r1['passes'] = bool(r1['passes'] and bl <= wl)
    r2 = None
    if 'WHITER50' in have:
        # AMENDMENT_1 B2 -- fixed arm, loc-matched.  Two factors remain (structure+spread).
        r2 = report(f'R-2 repair effectiveness (AMENDMENT_1: FIXED arm {R2_ARM}, loc-matched):'
                    f'\n      repaired background (JONSWAP structure AND Hs/4 spread)'
                    f' beats the stock defect background (white, 0.1 m)',
                    col(D, ks, R2_ARM), col(D, ks, 'WHITER50'), R2_ARM, 'WHITER50')
        r2['arm'] = R2_ARM
        if best != R2_ARM:
            print(f'    note: r* = {best} != {R2_ARM}; the r*-based version is EXPLORATORY '
                  f'(see E-3) because loc would then be an extra factor.')
    else:
        print('  R-2 not evaluable -- WHITER50 (phase 2) missing.')
    out['R1'], out['R2'] = r1, r2
    gates_ok = bool(r1['passes'] and r2 and r2['passes'])
    print(f'  => repair gates {"PASS" if gates_ok else "FAIL/INCOMPLETE"}')

    # --- PRIMARY ---
    print('\n' + '=' * 78)
    print('PRIMARY -- SHIP(median-of-3) vs the strongest repaired baseline')
    pr = report(f'PRIMARY  SHIP < {best}', shipv, col(D, ks, best), 'SHIP', best)
    # AMENDMENT_1 (neutral audit B1): this necessary condition is NOT in the frozen
    # PREREG; it is declared in AMENDMENT_1 before the full run.  It is unfavourable to
    # us -- failing it routes to FAIL (headline dead), not UNDECIDED -- and it is moot in
    # practice (S13 raw data: SHIP_med3 < ZERO in 63/64 crowded episodes).
    zr = report('necessary condition  SHIP < ZERO (IUT, outside any family; AMENDMENT_1)',
                shipv, zcol(D, ks, best), 'SHIP', 'ZERO')
    verdict = ('UNDECIDED (repair gate)' if not gates_ok
               else ('PASS' if (pr['passes'] and zr['passes']) else 'FAIL'))
    if out_prov:
        verdict = 'PROVISIONAL:' + verdict
    print(f'\n  VERDICT = {verdict}   (threshold {THRESH}/{N_PER}, exact alpha 0.016383)')
    if verdict == 'FAIL':
        print('  => PREREG 8.1 fires: the S13 margin is a product of the background-error'
              '\n     covariance defect.  Accuracy headline DEAD.  Scope clause required:'
              '\n     "isotropic JONSWAP, oracle (Hs,Tp), loc in {25,50,100}, ne=600, max_obs=40".')
    out['PRIMARY'] = dict(pr, zero=zr, verdict=verdict, comparator=best)
    out['provisional'] = out_prov
    out['baseline_seed_coverage'] = {a: dict(s) for a, s in seedn.items()}

    # --- SECONDARY, Holm-2 (PREREG 9) ---
    print('\n' + '=' * 78)
    print(f'SECONDARY (Holm-2, one-sided alpha={ALPHA}) -- registered shell {SEC_SHELL} only')
    s1a = report(f'S-1a  {best} < ZERO at {SEC_SHELL}  (does the repair fix the far field?)',
                 col(D, ks, best, SEC_SHELL), zcol(D, ks, best, SEC_SHELL), best, 'ZERO')
    s1b = report(f'S-1b  SHIP < {best} at {SEC_SHELL}',
                 col(D, ks, 'SHIP', SEC_SHELL), col(D, ks, best, SEC_SHELL), 'SHIP', best)
    adj = holm([s1a['p'], s1b['p']])
    for nm, r, pa in (('S-1a', s1a, adj[0]), ('S-1b', s1b, adj[1])):
        r['p_holm'] = float(pa)
        print(f'  {nm}: {r["wins"]}/{r["n"]}  p={r["p"]:.4g} -> p_adj={pa:.4g}  '
              f'{"PASS" if (r["passes"] and pa < ALPHA) else "FAIL"}')
    out['SECONDARY'] = {'S-1a': s1a, 'S-1b': s1b}

    # --- AMENDMENT_1 B3: PREREG 8-4 and 8-5 can fire together and command opposite
    # things about the same number.  Frozen priority: 8-5 BEATS 8-4. -----------------
    print('\n' + '=' * 78)
    print('REPORTING RULE (AMENDMENT_1 B3 -- frozen before the full run)')
    s1a_pass = bool(s1a['passes'] and s1a['p_holm'] < ALPHA)
    fires_84 = (verdict == 'PASS')
    fires_85 = (not s1a_pass)
    print(f'  PREREG 8-4 fires (PRIMARY PASS -> re-report the size): {fires_84}')
    print(f'  PREREG 8-5 fires (S-1a FAIL -> global nRMSE banned as a headline): {fires_85}')
    if fires_84 and fires_85:
        print('  => 8-5 WINS.  The headline size must be reported on the near-observation'
              '\n     shells (obs:0_5 ... obs:75_100, area ~60.8%); the GLOBAL med(rel) goes'
              '\n     into an appendix table only, never into a headline sentence.')
        near = [s for s in R.SHELL_KEYS if s not in ('obs:100_150', 'obs:150_inf')]
        av, bv = [], []
        for k in ks:
            a = [(D[k]['SHIP'].get('shell_rmse') or {}).get(s, np.nan) for s in near]
            b = [(D[k][best].get('shell_rmse') or {}).get(s, np.nan) for s in near]
            f = [(D[k][best].get('shell_frac') or {}).get(s, np.nan) for s in near]
            f = np.array(f, float)
            w = f / np.nansum(f) if np.nansum(f) > 0 else np.full(len(near), np.nan)
            av.append(float(np.nansum(np.array(a, float) * w)))
            bv.append(float(np.nansum(np.array(b, float) * w)))
        nr = report('  HEADLINE (8-5 scope): SHIP vs %s, area-weighted over obs<100 m' % best,
                    av, bv, 'SHIP', best)
        out['HEADLINE_near'] = nr
    elif fires_85:
        print('  => 8-5 governs on its own: global nRMSE is not a headline for this batch.')
    out['reporting_rule'] = dict(fires_8_4=fires_84, fires_8_5=fires_85,
                                 priority='8-5 beats 8-4')

    # --- exploratory (labelled) ---
    print('\n' + '=' * 78)
    print('EXPLORATORY (not pre-registered -- label as such in any report)')
    print(f'\n[E-1] shells, {PRIMARY_STRATUM} medians (m) and paired signs')
    hdr = ['shell', 'area%', 'SHIP', best, 'WHITER50', 'WHITE50', 'ZERO',
           'S<best', 'best<ZERO']
    print('  ' + ' '.join(f'{h:>10s}' for h in hdr))
    e1 = {}
    for sh in R.SHELL_KEYS:
        area = np.nanmean([(D[k][best].get('shell_frac') or {}).get(sh, np.nan) for k in ks])
        vals = {}
        for a in ('SHIP', best, 'WHITER50', 'WHITE50'):
            vals[a] = (np.nanmedian(col(D, ks, a, sh)) if a in have or a == 'SHIP'
                       else np.nan)
        z = np.nanmedian(zcol(D, ks, best, sh))
        w1 = sign_one_sided(col(D, ks, 'SHIP', sh), col(D, ks, best, sh))
        w2 = sign_one_sided(col(D, ks, best, sh), zcol(D, ks, best, sh))
        row = [sh, 100 * area, vals['SHIP'], vals[best], vals.get('WHITER50', np.nan),
               vals.get('WHITE50', np.nan), z, f'{w1[0]}/{w1[1]}', f'{w2[0]}/{w2[1]}']
        print('  ' + ' '.join(f'{v:>10.4f}' if isinstance(v, float) else f'{v:>10s}'
                              for v in row))
        e1[sh] = dict(area=float(area), ship=float(vals['SHIP']), best=float(vals[best]),
                      zero=float(z), ship_lt_best=w1[:2], best_lt_zero=w2[:2])
    out['E1_shells'] = e1

    print('\n[E-2] localisation dose-response')
    for a in POOL:
        print(f'  loc={cand[a]["loc"]:5.1f}  SHIP wins {cand[a]["wins"]}/{cand[a]["n"]}  '
              f'med(rel)={cand[a]["med_rel"]:+.1f}%')

    if 'WHITER50' in have and 'WHITE50' in have:
        print('\n[E-3] repair size and the RNG/inflation-placement confound')
        report('  SPEC_best vs WHITER50 -- EXPLORATORY (AMENDMENT_1 B2: if r* != SPEC50 this'
               '\n  bundles loc with structure+spread; the registered R-2 is the SPEC50 row above)',
               col(D, ks, best), col(D, ks, 'WHITER50'), best, 'WHITER50')
        report('  WHITER50 vs WHITE50 (code-path difference only)',
               col(D, ks, 'WHITER50'), col(D, ks, 'WHITE50'), 'WHITER50', 'WHITE50')

    print('\n[E-4] sparse stratum mirror -- DESCRIPTIVE ONLY '
          '(S13 caveat 1: sparse advantage is a representation artifact)')
    ksp = keys_of(D, 'sparse', POOL)
    if len(ksp) == N_PER:
        sv = col(D, ksp, 'SHIP')
        for a in POOL:
            w, n, p = sign_one_sided(sv, col(D, ksp, a))
            print(f'  {a:9s} SHIP wins {w}/{n}  med(rel)={effects(sv, col(D, ksp, a))[0]:+.1f}%')

    print('\n[E-5] cell decomposition')
    by = collections.defaultdict(list)
    for k in ks:
        by[(D[k][best]['bf'], D[k][best]['ncomp'])].append(k)
    for c in sorted(by):
        kk = by[c]
        w, n, p = sign_one_sided(col(D, kk, 'SHIP'), col(D, kk, best))
        print(f'  {c[0]} {c[1]:2d}comp n={len(kk):2d}  SHIP wins {w}/{n}  '
              f'med(rel)={effects(col(D, kk, "SHIP"), col(D, kk, best))[0]:+.1f}%')

    json.dump(out, open(os.path.join(HERE, 'verdict_s16.json'), 'w'), indent=1, default=str)
    print('\n-> verdict_s16.json')


if __name__ == '__main__':
    main()
