"""W3_SOLVER analyzer -- implements PREREG.md + AMENDMENT_1.md.

Reads `results.jsonl` ONLY (never smoke_*.jsonl).  Runs every gate BEFORE any hypothesis is
printed; a DCC failure (PREREG section 6) exits with SCHEMA-FAIL and prints no verdict.

    python3 tmp/W3_SOLVER_2026-08-15/analyze_w3.py            # verdict
    python3 tmp/W3_SOLVER_2026-08-15/analyze_w3.py --dryrun   # gates only, no hypothesis
"""
import collections
import json
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import run_w3 as R                                            # noqa: E402

RES = os.path.join(HERE, 'results.jsonl')
VERDICT = os.path.join(HERE, 'verdict_w3.json')

ALPHA = 0.025                       # one-sided, frozen
THRESH = {'crowded': (32, 23), 'sparse': (16, 13)}   # (n, k*) -- PREREG section 4
M_HOLM = 4                          # AMENDMENT_2 B-2: FIXED at the four pre-registered
#                                     SECONDARY.  Never len(sec) -- a NOT-RUN arm must
#                                     not loosen the correction.

PRIMARY = ('P-1', 'ORA-K-NOREG', 'ORA-K@L3')
SECONDARY = [('S-1', 'ORA-K-NOREG', 'ORA-K@L4'),
             ('S-2', 'ORA-K-NOREG', 'SHIP'),
             ('S-3', 'SHIP-FITPHI', 'SHIP'),
             ('S-4', 'KONLY-FITPHI', 'KONLY')]

# AMENDMENT_2 D-1: PREREG section 8's verdict table, in code.  (P-1 pass, S-2 pass) -> name.
VERDICT_TABLE = {(True, True): 'LAMBDA-ARTIFACT',
                 (True, False): 'LAMBDA-HURTS-ONLY',
                 (False, True): 'NOT-A-LAMBDA-ARTIFACT',
                 (False, False): 'NOT-A-LAMBDA-ARTIFACT'}


def tail(k, n, p=0.5):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def min_wins(n, alpha):
    """AMENDMENT_2 B-3: smallest win count whose exact one-sided p clears `alpha`.

    At n=32 the Holm ladder makes the two smallest-p SECONDARY need 24/32, not the
    23/32 printed in PREREG section 4.2 (p(23)=0.010031 > alpha/3 = 0.008333).
    """
    for k in range(0, n + 1):
        if tail(k, n) <= alpha:
            return k
    return n + 1                    # unreachable at this alpha


# ── load: effective row set = last SUCCESSFUL row per job_id (AMENDMENT A-1) ──
def load():
    if not os.path.exists(RES):
        print('results.jsonl absent -- nothing to analyse.')
        sys.exit(2)
    raw = []
    for line in open(RES):
        try:
            raw.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    errs = [r for r in raw if 'error' in r]
    ok = collections.OrderedDict()
    dup = []
    for r in raw:
        if 'error' in r:
            continue
        j = r['job_id']
        if j in ok and abs(ok[j]['avg_nRMSE'] - r['avg_nRMSE']) > 0:
            dup.append((j, ok[j]['avg_nRMSE'], r['avg_nRMSE']))
        ok[j] = r
    return list(ok.values()), errs, dup


def _self_md5():
    import hashlib
    out = {}
    for f in ('analyze_w3.py', 'run_w3.py', 'PREREG.md', 'AMENDMENT_1.md',
              'AMENDMENT_2.md'):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            out[f] = hashlib.md5(open(p, 'rb').read()).hexdigest()
    return out


def gates(rows, errs, dup):
    ok = True
    print('=' * 78)
    print('W3_SOLVER gates  (PREREG + AMENDMENT_1 + AMENDMENT_2; md5 in FROZEN_md5.txt)')
    print('=' * 78)
    # AUDIT_NEUTRAL U-2 / AUDIT_CON W-6: the verdict path is outside the run
    # fingerprint, so it prints its own md5 -- compare against FROZEN_md5.txt.
    for f, d in _self_md5().items():
        print('     md5 %-16s %s' % (f, d))

    # G-1 sample
    planned = {j['job_id'] for j in R.build_jobs()}
    have = {r['job_id'] for r in rows}
    missing = sorted(planned - have)
    g1 = not missing
    print('G-1  sample      : planned=%d present=%d missing=%d -> %s'
          % (len(planned), len(have), len(missing), 'PASS' if g1 else 'FAIL'))
    if missing:
        print('     first missing:', missing[:6])
    ok &= g1

    # G-2 DCC schema
    viol = [(r['job_id'], R.missing_fields(r)) for r in rows]
    viol = [v for v in viol if v[1]]
    g2 = not viol
    print('G-2  DCC schema  : violations=%d -> %s' % (len(viol), 'PASS' if g2 else 'FAIL'))
    for j, m in viol[:5]:
        print('     %s missing %s' % (j, m[:6]))
    if not g2:
        print('\nSCHEMA-FAIL -- no death condition and no verdict is printed.')
        sys.exit(3)

    # G-3 lambda manipulation check (object readback)
    bad = [(r['job_id'], r['lambda_l1_intent'], r['lambda_l1_effective']) for r in rows
           if R.ARMS[r['arm']]['kind'] == 'est'
           and float(r['lambda_l1_effective']) != float(r['lambda_l1_intent'])]
    ens = [(r['job_id'], r['effective_config'].get('member_lambda_l1')) for r in rows
           if R.ARMS[r['arm']]['kind'] == 'est'
           and r['effective_config'].get('member_lambda_l1')
           and len(set(r['effective_config']['member_lambda_l1'])) > 1]
    g3 = not bad and not ens
    print('G-3  lambda      : mismatches=%d ensemble_disagree=%d -> %s'
          % (len(bad), len(ens), 'PASS' if g3 else 'FAIL'))
    for b in bad[:5]:
        print('     ', b)
    ok &= g3

    # G-4 FITPHI manipulation check (atom-level)
    fit = {}
    for arm in R.ARMS:
        if R.ARMS[arm]['kind'] != 'est':
            continue
        for st in ('crowded', 'sparse'):
            v = [r for r in rows if r['arm'] == arm and r['stratum'] == st]
            if not v:
                continue
            hit = sum(1 for r in v if r['fitphi_atoms'] > 0)
            fit[(arm, st)] = (hit, len(v), float(np.mean([r['fitphi_atoms'] for r in v])))
    g4_off = all(h == 0 for (a, s), (h, n, m) in fit.items()
                 if R.ARMS[a]['fitphi'] == 0)
    # AMENDMENT_2 W-2: PREREG section 2.2 ("absence >= 50% -> UNDECIDED") and section 7's
    # table ("intervention >= 50% -> pass") disagree at exactly 16/32.  Resolved in the
    # conservative direction: an arm that intervenes on exactly half its rows is DEAD.
    dead = {(a, s) for (a, s), (h, n, m) in fit.items()
            if R.ARMS[a]['fitphi'] == 1 and h <= 0.5 * n}
    print('G-4  FITPHI      : OFF-arms-clean=%s ; ON arms below 50%% intervention: %s'
          % (g4_off, sorted(dead) if dead else 'none'))
    for k in sorted(fit):
        print('     %-14s %-8s intervened %2d/%2d  mean_atoms %.1f'
              % (k[0], k[1], fit[k][0], fit[k][1], fit[k][2]))
    ok &= g4_off

    # G-5 leak probe
    a = {(r['episode'], r['est_seed']): r['avg_nRMSE']
         for r in rows if r['arm'] == 'ORA-K@L3'}
    b = {(r['episode'], r['est_seed']): r['avg_nRMSE']
         for r in rows if r['arm'] == 'ORA-K@L3-DUP'}
    common = sorted(set(a) & set(b))
    dmax = max((abs(a[k] - b[k]) for k in common), default=float('nan'))
    g5 = bool(common) and dmax == 0.0
    print('G-5  leak probe  : pairs=%d max|delta|=%.3e -> %s'
          % (len(common), dmax, 'PASS' if g5 else 'FAIL'))
    ok &= g5

    # G-5b GPU-path reproducibility probe (AMENDMENT_2 B-7).  `SHIP` is the only arm
    # that reaches the learned proposal, which ignores cfg['device'] and runs on CUDA
    # (learned_init.py:130-131); nothing in PREREG probed that path.  Hard gate, same
    # form as G-5: byte-identical config must give byte-identical nRMSE.
    a2 = {(r['episode'], r['est_seed']): r['avg_nRMSE']
          for r in rows if r['arm'] == 'SHIP'}
    b2 = {(r['episode'], r['est_seed']): r['avg_nRMSE']
          for r in rows if r['arm'] == 'SHIP-DUP'}
    common2 = sorted(set(a2) & set(b2))
    dmax2 = max((abs(a2[k] - b2[k]) for k in common2), default=float('nan'))
    g5b = bool(common2) and dmax2 == 0.0
    print('G-5b GPU repro   : pairs=%d max|delta|=%.3e -> %s'
          % (len(common2), dmax2, 'PASS' if g5b else 'FAIL'))
    ok &= g5b

    # G-6 seed spread (pseudo-replication check, PREREG 3.3)
    print('G-6  seed spread :')
    for arm in R.ARM_ORDER:
        v = [r for r in rows if r['arm'] == arm]
        if not v or R.ARMS[arm]['kind'] != 'est':
            continue
        by = collections.defaultdict(list)
        for r in v:
            by[r['episode']].append(r['avg_nRMSE'])
        sp = [max(x) - min(x) for x in by.values() if len(x) > 1]
        det = bool(sp) and max(sp) == 0.0
        print('     %-14s max_spread=%.3e median_spread=%.3e %s'
              % (arm, max(sp) if sp else float('nan'),
                 float(np.median(sp)) if sp else float('nan'),
                 '<- DETERMINISTIC: median-of-3 is PSEUDO-REPLICATION' if det else ''))

    # G-7 fingerprint
    dg = set(r['code_md5'] for r in rows)
    g7 = len(dg) == 1
    print('G-7  fingerprint : distinct code_md5=%d -> %s' % (len(dg), 'PASS' if g7 else 'FAIL'))
    ok &= g7

    # G-8 ZERO identical across arms
    z = collections.defaultdict(set)
    for r in rows:
        z[r['episode']].add(round(r['zero_nRMSE'], 12))
    g8 = all(len(v) == 1 for v in z.values())
    print('G-8  ZERO stable : episodes=%d all-single-valued=%s -> %s'
          % (len(z), g8, 'PASS' if g8 else 'FAIL'))
    ok &= g8

    # G-9 seed ledger
    used = set(r['est_seed'] for r in rows if r['est_seed'] is not None)
    g9 = used == set(R.SEEDS)
    print('G-9  seed ledger : consumed=%s expected=%s -> %s'
          % (sorted(used), list(R.SEEDS), 'PASS' if g9 else 'FAIL'))
    ok &= g9

    # G-10 (AMENDMENT A-1)
    reasons = collections.Counter(e['error'].split('(')[0][:40] for e in errs)
    g10 = not missing and not dup
    print('G-10 errors      : transient error rows=%d (%s) unrecovered=%d '
          'nondeterministic_retries=%d -> %s'
          % (len(errs), dict(reasons), len(missing), len(dup), 'PASS' if g10 else 'FAIL'))
    for d in dup[:5]:
        print('     NONDETERMINISM', d)
    ok &= g10

    # G-11 CVX self-check
    cv = [r for r in rows if r['arm'] == 'ORA-K-CVX']
    bad_rank = [r['job_id'] for r in cv if r['cvx_rank'] != 2 * r['n_gt']]
    worse = [r['job_id'] for r in cv
             if r['cvx_window_resid_rms'] > r['cvx_truth_window_resid_rms']]
    print('G-11 CVX self    : rows=%d rank_deficient=%d worse_than_truth=%d '
          '(report-only)' % (len(cv), len(bad_rank), len(worse)))

    # G-12 oracle k drift (AMENDMENT_2 B-6a, report-only).  ORA-K* keep learning k;
    # ORA-K-CVX freezes it.  If the drift is ~0 that undeclared factor is empirically
    # removed from the CVX comparison; if it is large the CVX gap is a 4-factor bundle
    # and must be labelled as one.
    print('G-12 oracle k    : (report-only; ORA-K* learn k, ORA-K-CVX freezes it)')
    for arm in R.ARM_ORDER:
        v = [r for r in rows if r['arm'] == arm and r.get('k_drift_valid')]
        allv = [r for r in rows if r['arm'] == arm
                and R.ARMS[arm]['kind'] == 'est' and R.ARMS[arm].get('oracle')]
        if not allv:
            continue
        if v:
            print('     %-14s valid %3d/%3d  median rms %.4e  max %.4e  '
                  'median rel %.3e'
                  % (arm, len(v), len(allv),
                     float(np.median([r['k_drift_rms'] for r in v])),
                     float(np.max([r['k_drift_max'] for r in v])),
                     float(np.median([r['k_drift_rel_median'] for r in v]))))
        else:
            note = collections.Counter(str(r.get('k_drift_note'))[:44] for r in allv)
            print('     %-14s valid 0/%d  %s' % (arm, len(allv), dict(note)))

    print('-' * 78)
    print('GATES %s' % ('PASS' if ok else 'FAIL'))
    return ok, fit, dead


# ── aggregation: median-of-3 over seeds, THEN episode-level sign test ─────────
def per_episode(rows, arm, stratum):
    by = collections.defaultdict(list)
    for r in rows:
        if r['arm'] == arm and r['stratum'] == stratum:
            by[r['episode']].append(r['avg_nRMSE'])
    return {k: float(np.median(v)) for k, v in by.items()}


def shells_per_episode(rows, arm, stratum):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r['arm'] == arm and r['stratum'] == stratum:
            for k in R.SHELL_KEYS:
                by[r['episode']][k].append(r['shell_rmse'][k])
    return {ep: {k: float(np.median(v)) for k, v in d.items()} for ep, d in by.items()}


def sign_test(rows, treat, ctrl, stratum):
    t = per_episode(rows, treat, stratum)
    c = per_episode(rows, ctrl, stratum)
    eps = sorted(set(t) & set(c))
    if not eps:
        return None
    d = np.array([t[e] - c[e] for e in eps])
    n = len(eps)
    wins = int((d < 0).sum())
    ties = int((d == 0).sum())
    p = tail(wins, n)
    n_exp, kstar = THRESH[stratum]
    tv = np.array([t[e] for e in eps])
    cv = np.array([c[e] for e in eps])
    return dict(treat=treat, ctrl=ctrl, stratum=stratum, n=n, n_expected=n_exp,
                wins=wins, ties=ties, threshold=kstar, p_exact=p,
                pass_=(n == n_exp and wins >= kstar),
                # PREREG section 5: the three effect-size protocols, all mandatory
                sizeA_median_abs=float(np.median(d)),
                sizeB_median_rel_pct=float(np.median(d / cv) * 100.0),
                sizeC_pooled_rel_pct=float((tv.mean() - cv.mean()) / cv.mean() * 100.0),
                treat_median=float(np.median(tv)), ctrl_median=float(np.median(cv)),
                episodes=eps)


def corpus_split(rows, treat, ctrl, stratum='crowded'):
    """AMENDMENT_2 D-2: PREREG section 3.1 mandates the 16/16 corpus split as narrative.

    The pooled 32 remains THE pre-registered judgement; this is descriptive only.
    """
    cmap = {r['episode']: r['corpus'] for r in rows}
    t = per_episode(rows, treat, stratum)
    c = per_episode(rows, ctrl, stratum)
    out = collections.OrderedDict()
    for e in sorted(set(t) & set(c)):
        k = cmap.get(e, '?')
        w, n, ds = out.get(k, (0, 0, []))
        d = t[e] - c[e]
        out[k] = (w + int(d < 0), n + 1, ds + [d])
    return collections.OrderedDict(
        (k, (w, n, float(np.median(ds)))) for k, (w, n, ds) in sorted(out.items()))


def zero_condition(rows, stratum):
    z = per_episode(rows, 'ZERO', stratum)
    out = {}
    for arm in R.ARM_ORDER:
        if arm == 'ZERO':
            continue
        a = per_episode(rows, arm, stratum)
        eps = sorted(set(a) & set(z))
        if not eps:
            continue
        w = sum(1 for e in eps if a[e] < z[e])
        out[arm] = (w, len(eps), w > 0.5 * len(eps))
    return out


def main():
    dry = '--dryrun' in sys.argv
    rows, errs, dup = load()
    ok, fit, dead = gates(rows, errs, dup)
    print()

    # degeneracy first -- it conditions how every result may be worded
    degen = {}
    for st in ('crowded', 'sparse'):
        zc = zero_condition(rows, st)
        degen[st] = {a for a, (w, n, good) in zc.items() if not good}
        print('ZERO necessary condition [%s] (arm beats the flat sea on how many episodes)'
              % st)
        for a, (w, n, good) in zc.items():
            print('   %-14s %2d/%2d  %s' % (a, w, n, '' if good else '<- DEGENERATE'))
        print()

    if not ok:
        print('GATES FAILED -- no verdict is printed (PREREG section 7).')
        sys.exit(4)
    if dry:
        print('--dryrun: gates only.')
        return

    res = {}
    print('=' * 78)
    print('PRIMARY (one only)')
    print('=' * 78)
    r = sign_test(rows, PRIMARY[1], PRIMARY[2], 'crowded')
    res['P-1'] = r
    tag = '' if PRIMARY[2] not in degen['crowded'] else '  (DEGENERATE-CONTROL)'
    print('P-1  %s < %s  [crowded]%s' % (PRIMARY[1], PRIMARY[2], tag))
    print('     %d/%d wins (threshold %d/%d, exact one-sided p=%.3e, ties=%d)  -> %s'
          % (r['wins'], r['n'], r['threshold'], r['n_expected'], r['p_exact'],
             r['ties'], 'PASS' if r['pass_'] else 'FAIL = NOT-A-LAMBDA-ARTIFACT'))
    print('     size A median abs  %+.5f | B median rel %+.2f%% | C pooled rel %+.2f%%'
          % (r['sizeA_median_abs'], r['sizeB_median_rel_pct'], r['sizeC_pooled_rel_pct']))
    r['corpus_split'] = {k: v for k, v in corpus_split(
        rows, PRIMARY[1], PRIMARY[2]).items()}
    print('     corpus split (descriptive; the pooled 32 is the frozen judgement): '
          + ' | '.join('%s %d/%d medA %+.5f' % (k.replace('data_', ''), w, n, a)
                       for k, (w, n, a) in r['corpus_split'].items()))

    print()
    print('=' * 78)
    print('SECONDARY (labelled SECONDARY; Holm-4 over the four crowded contrasts)')
    print('=' * 78)
    sec = {}
    for sid, t, c in SECONDARY:
        rr = sign_test(rows, t, c, 'crowded')
        if rr is None:
            print('%-4s %-14s < %-14s  NOT-RUN' % (sid, t, c))
            continue
        if (t, 'crowded') in dead:
            rr['pass_'] = False
            rr['verdict'] = 'UNDECIDED-NO-INTERVENTION'
        # AMENDMENT_2 B-4: PREREG section 4.3 bans a superiority sentence about ANY
        # degenerate arm, not only the PRIMARY control.  Tag both sides.
        rr['degenerate_arms'] = sorted({a for a in (t, c) if a in degen['crowded']})
        rr['corpus_split'] = {k: v for k, v in corpus_split(rows, t, c).items()}
        sec[sid] = rr
    order = sorted(sec.items(), key=lambda kv: kv[1]['p_exact'])
    m = M_HOLM                      # AMENDMENT_2 B-2: fixed, never len(order)
    alive = True                    # AMENDMENT_2 B-2: Holm STEP-DOWN -- stop at the
    #                                 first hypothesis that is not rejected.
    for i, (sid, rr) in enumerate(order):
        adj = ALPHA / (m - i)
        kmin = min_wins(rr['n'], adj)          # AMENDMENT_2 B-3: effective threshold
        step = bool(rr['p_exact'] <= adj and rr['wins'] >= rr['threshold'])
        alive = alive and step
        rr['holm_rank'] = i + 1
        rr['holm_alpha'] = adj
        rr['holm_effective_threshold'] = kmin
        rr['holm_step_reject'] = step
        rr['holm_pass'] = bool(alive) and rr.get('verdict') != 'UNDECIDED-NO-INTERVENTION'
        dtag = ('  (DEGENERATE-ARM: %s)' % ','.join(rr['degenerate_arms'])
                if rr['degenerate_arms'] else '')
        state = rr.get('verdict') or ('PASS' if rr['holm_pass'] else
                                      ('FAIL' if step else 'FAIL(step-down stop)'))
        print('%-4s %-14s < %-14s  %2d/%-2d ties=%d p=%.3e  holm_a=%.5f '
              'eff_thresh=%d/%d -> %s%s'
              % (sid, rr['treat'], rr['ctrl'], rr['wins'], rr['n'], rr['ties'],
                 rr['p_exact'], adj, kmin, rr['n'], state, dtag))
        print('       size A %+.5f | B %+.2f%% | C %+.2f%%   corpus %s'
              % (rr['sizeA_median_abs'], rr['sizeB_median_rel_pct'],
                 rr['sizeC_pooled_rel_pct'],
                 ' | '.join('%s %d/%d' % (k.replace('data_', ''), w, n)
                            for k, (w, n, a) in rr['corpus_split'].items())))
    res['secondary'] = sec

    # ── PREREG section 8 verdict table, computed (AMENDMENT_2 D-1) ──────────────
    s2 = sec.get('S-2')
    p1_pass = bool(r['pass_'])
    s2_pass = bool(s2 and s2.get('holm_pass'))
    name = VERDICT_TABLE[(p1_pass, s2_pass)]
    ctag = ' (DEGENERATE-CONTROL)' if PRIMARY[2] in degen['crowded'] else ''
    print()
    print('=' * 78)
    print('VERDICT (PREREG section 8) : %s%s' % (name, ctag))
    print('=' * 78)
    print('   P-1 %s | S-2 %s (Holm-4 step-down%s)'
          % ('PASS' if p1_pass else 'FAIL',
             'PASS' if s2_pass else 'FAIL',
             '' if s2 else '; S-2 NOT-RUN'))
    if ctag:
        print('   AMENDMENT_1 A-3: the control is worse than the flat sea, so the only')
        print('   permitted sentence is "lambda=1e-3 makes the oracle arm worse than')
        print('   predicting a flat sea, and lambda=0 undoes that damage".')
    res['verdict_name'] = name
    res['verdict_degenerate_control'] = bool(ctag)

    print()
    print('sparse stratum (DESCRIPTIVE, not confirmatory; n=16 threshold 13/16)')
    for sid, t, c in [PRIMARY] + SECONDARY:
        rr = sign_test(rows, t, c, 'sparse')
        if rr:
            res.setdefault('sparse', {})[sid] = rr
            print('   %-4s %-14s < %-14s  %2d/%-2d p=%.3e  A %+.5f  B %+.2f%%'
                  % (sid, t, c, rr['wins'], rr['n'], rr['p_exact'],
                     rr['sizeA_median_abs'], rr['sizeB_median_rel_pct']))

    print()
    print('=' * 78)
    print('UPPER BOUND  ORA-K-CVX  (closed-form lstsq, ZERO gradient steps -- not a test)')
    print('=' * 78)
    for st in ('crowded', 'sparse'):
        cvx = per_episode(rows, 'ORA-K-CVX', st)
        if not cvx:
            continue
        print(' [%s] CVX median nRMSE = %.5f   (ZERO = %.5f)'
              % (st, np.median(list(cvx.values())),
                 np.median(list(per_episode(rows, 'ZERO', st).values()))))
        for arm in R.ARM_ORDER:
            if arm in ('ORA-K-CVX', 'ZERO'):
                continue
            a = per_episode(rows, arm, st)
            eps = sorted(set(a) & set(cvx))
            if not eps:
                continue
            gap = np.array([a[e] - cvx[e] for e in eps])
            ratio = np.array([a[e] / cvx[e] for e in eps if cvx[e] > 0])
            print('    %-14s median %.5f   gap_to_CVX %+.5f   ratio x%.1f'
                  % (arm, np.median([a[e] for e in eps]), np.median(gap),
                     np.median(ratio) if ratio.size else float('nan')))
    res['cvx'] = {st: {a: per_episode(rows, a, st) for a in ('ORA-K-CVX', 'ZERO')}
                  for st in ('crowded', 'sparse')}

    print()
    print('=' * 78)
    print('SHELL DECOMPOSITION (mandatory: a global win must be split near/far)')
    print('=' * 78)
    for sid, t, c in [PRIMARY] + SECONDARY:
        st = 'crowded'
        ts = shells_per_episode(rows, t, st)
        cs = shells_per_episode(rows, c, st)
        eps = sorted(set(ts) & set(cs))
        if not eps:
            continue
        print(' %s  %s - %s  [%s]' % (sid, t, c, st))
        line = []
        for k in R.SHELL_KEYS:
            d = np.array([ts[e][k] - cs[e][k] for e in eps])
            d = d[np.isfinite(d)]
            if not d.size:
                line.append('%s n/a' % k)
                continue
            line.append('%s %+.4f(%d/%d)' % (k.split(':')[1], float(np.median(d)),
                                             int((d < 0).sum()), d.size))
        print('   ' + '  '.join(line))

    with open(VERDICT, 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    print('\nwrote %s' % VERDICT)


if __name__ == '__main__':
    main()
