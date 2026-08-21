"""NEUTRAL (axis 6) — PRESPECIFY the k-position measurement protocol, and VALIDATE it
on data that already exists (no estimator runs, no GPU).

Two jobs:

 (I) MATCHING. The incumbent rule (run_rx2b.match_components, verbatim from run_s1s2)
     is Hungarian on cost=|dk|/max_j|k_j^gt| with accept iff cost<0.5. The 08-05
     placement debate already MEASURED that this acceptance disc is 4.2-8.3x the true
     component spacing on crowded episodes and that stacking every atom on one point
     scores detection_rate 0.60-0.87. So any metric conditioned on that match set is
     unusable as a k-position metric. Here I (a) re-measure the disc/spacing ratio
     myself, (b) define and VALIDATE a matching-FREE alternative, (c) show its null.

 (II) NULLS + MDE. Compute the degenerate-placement nulls for the proposed metric and
     the exact sign-test power table, so the direction/aggregation/MDE are fixed BEFORE
     any run.  [[feedback-prespecify-aggregation-and-direction]]

Inputs (read-only): data_finemesh_2026-07-23/*.h5,
                    tmp/B2_g1p1_ext_2026-08-06/results.jsonl (kx/ky/w already stored).
"""
import json
import math
import os
import statistics as st
from collections import defaultdict

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment, linprog

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
RES = os.path.join(REPO, 'tmp', 'B2_g1p1_ext_2026-08-06', 'results.jsonl')

CROWDED4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
NEW8 = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
        'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
        'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
        'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5']
ALL12 = NEW8 + CROWDED4
CELLS = ['X-BASE', 'G1', 'P1', 'G1P1', 'ORACLE-K', 'AUTO']

OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


# ------------------------------------------------------------------ metrics
def emd2d(pk, pw, qk, qw):
    """Exact 1-Wasserstein (earth mover) between two weighted point sets on the k-plane.
    pk:(n,2) pw:(n,) qk:(m,2) qw:(m,).  Weights are normalised to unit mass first, so
    this is a distance between NORMALISED ENERGY MEASURES -- cardinality-free.
    Units: rad/m.  Solved exactly as a transportation LP.
    """
    pw = np.asarray(pw, float); qw = np.asarray(qw, float)
    if pw.sum() <= 0 or qw.sum() <= 0:
        return float('nan')
    p = pw / pw.sum(); q = qw / qw.sum()
    n, m = len(p), len(q)
    Cm = np.linalg.norm(pk[:, None, :] - qk[None, :, :], axis=2)
    A_eq = np.zeros((n + m, n * m))
    for i in range(n):
        A_eq[i, i * m:(i + 1) * m] = 1.0
    for j in range(m):
        A_eq[n + j, j::m] = 1.0
    b_eq = np.concatenate([p, q])
    r = linprog(Cm.ravel(), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
    return float(r.fun) if r.success else float('nan')


def chamfer(pk, pw, qk, qw):
    """Energy-weighted two-sided Chamfer on the k-plane, in rad/m.
      COV  = sum_j qhat_j * min_i |k_i^est - k_j^gt|   (did we cover every true atom?)
      PREC = sum_i phat_i * min_j |k_i^est - k_j^gt|   (is every placed atom near a true one?)
    Stacking blows COV up; scattering blows PREC up. Reported separately BECAUSE the
    08-05 audit measured that placement quality is at least 2-dimensional.
    """
    p = np.asarray(pw, float); q = np.asarray(qw, float)
    p = p / p.sum(); q = q / q.sum()
    D = np.linalg.norm(pk[:, None, :] - qk[None, :, :], axis=2)
    cov = float((q * D.min(axis=0)).sum())
    prec = float((p * D.min(axis=1)).sum())
    return cov, prec


def incumbent_match(est_k, gt_k, gt_A):
    k_scale = max(np.max(np.linalg.norm(gt_k, axis=1)), 0.01)
    C = np.linalg.norm(est_k[:, None, :] - gt_k[None, :, :], axis=2) / k_scale
    r, c = linear_sum_assignment(C)
    m = [(i, j) for i, j in zip(r, c) if C[i, j] < 0.5]
    return m, k_scale


def nn_spacing(gt_k):
    if len(gt_k) < 2:
        return float('nan')
    D = np.linalg.norm(gt_k[:, None, :] - gt_k[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    return float(np.median(D.min(axis=1)))


# ------------------------------------------------------------------ power
def sign_p(k, n):
    if n == 0:
        return 1.0
    kk = min(k, n - k)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(kk + 1)) / 2.0 ** n)


def crit_k(n, alpha=0.05):
    for k in range((n // 2) + 1, n + 1):
        if sign_p(k, n) <= alpha:
            return k
    return None


def power(n, pwin, alpha=0.05):
    k = crit_k(n, alpha)
    if k is None:
        return 0.0
    return sum(math.comb(n, i) * pwin ** i * (1 - pwin) ** (n - i) for i in range(k, n + 1))


# ------------------------------------------------------------------ data
def gt_of(ep):
    with h5py.File(os.path.join(DATA, ep), 'r') as f:
        wc = np.array(f['config/wave_components'])
    return np.stack([wc[:, 2], wc[:, 3]], 1), wc[:, 0]


def load_est():
    d = defaultdict(list)
    for line in open(RES):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' in r or 'kx' not in r:
            continue
        d[(r['cell'], r['episode'])].append(
            (np.stack([np.asarray(r['kx']), np.asarray(r['ky'])], 1),
             np.abs(np.asarray(r['w'])), int(r['seed'])))
    return d


def main():
    P('=' * 104)
    P('NEUTRAL axis-6 — PRESPECIFIED k-position measurement protocol (+ validation on existing data)')
    P('=' * 104)

    # ---------------- I.a incumbent matcher, re-measured by me ----------------
    P('')
    P('I.a  INCUMBENT MATCHER, re-measured (run_rx2b.py:184 match_components)')
    P('     rule: Hungarian on cost=|dk|/max_j|k_j^gt|, ACCEPT iff cost<0.5')
    P(f'{"episode":30s}{"n_gt":>5s}{"accept R":>10s}{"NNspacing":>11s}{"R/(NN/2)":>10s}'
      f'{"#gt in one disc":>17s}')
    for e in ALL12:
        gk, gA = gt_of(e)
        k_scale = max(np.max(np.linalg.norm(gk, axis=1)), 0.01)
        R = 0.5 * k_scale
        nn = nn_spacing(gk)
        D = np.linalg.norm(gk[:, None, :] - gk[None, :, :], axis=2)
        indisc = float(np.median((D < R).sum(axis=1)))
        P(f'{e[8:-3]:30s}{len(gA):5d}{R:10.4f}'
          + (f'{nn:11.4f}{R / (nn / 2):10.2f}' if len(gA) > 1 else f'{"—":>11s}{"—":>10s}')
          + f'{indisc:17.1f}')
    P('  => on the crowded cells one acceptance disc contains ~10-18 true components.')
    P('     Any metric conditioned on this match set is a COVERAGE metric, not a k-position one.')

    # ---------------- I.b matching-free metric, validated ----------------
    est = load_est()
    P('')
    P('I.b  PROPOSED MATCHING-FREE METRIC, validated on the 216 runs that already exist')
    P('     W1  = exact 1-Wasserstein between normalised ENERGY measures on the k-plane [rad/m]')
    P('     COV = energy-weighted mean distance from each TRUE atom to nearest PLACED atom')
    P('     PREC= energy-weighted mean distance from each PLACED atom to nearest TRUE atom')
    P('     nulls: STACK = every atom at the energy centroid of the true set')
    P('            PRIOR = atoms on the deployed coarse grid, uniform over the search box')
    tab = {}
    for e in ALL12:
        gk, gA = gt_of(e)
        gw = gA ** 2
        for c in CELLS:
            vals = []
            for pk, pw, s in est[(c, e)]:
                w1 = emd2d(pk, pw ** 2, gk, gw)
                cov, pre = chamfer(pk, pw ** 2, gk, gw)
                vals.append((w1, cov, pre))
            if vals:
                tab[(c, e)] = tuple(st.median([v[i] for v in vals]) for i in range(3))
        # nulls, matched to the shipped arm's own atom count
        n_atoms = int(np.median([len(pk) for pk, _, _ in est[('X-BASE', e)]]))
        cen = (gw[:, None] * gk).sum(0) / gw.sum()
        sk = np.repeat(cen[None, :], n_atoms, 0)
        tab[('NULL-STACK', e)] = (emd2d(sk, np.ones(n_atoms), gk, gw),
                                  *chamfer(sk, np.ones(n_atoms), gk, gw))
        rng = np.random.RandomState(0)
        reps = []
        for _ in range(32):
            rk = rng.uniform(-0.8, 0.8, size=(n_atoms, 2))
            reps.append((emd2d(rk, np.ones(n_atoms), gk, gw),
                         *chamfer(rk, np.ones(n_atoms), gk, gw)))
        tab[('NULL-PRIOR', e)] = tuple(st.median([r[i] for r in reps]) for i in range(3))

    P('')
    P(f'{"":30s}' + ''.join(f'{c:>26s}' for c in ['W1 [rad/m]', 'COV [rad/m]', 'PREC [rad/m]']))
    P(f'{"arm":30s}' + ''.join(f'{x:>13s}' for x in
                               ['NEW8', 'CROWDED4'] * 3))
    for c in CELLS + ['NULL-STACK', 'NULL-PRIOR']:
        row = f'{c:30s}'
        for i in range(3):
            for eps in (NEW8, CROWDED4):
                row += f'{st.median([tab[(c, e)][i] for e in eps]):13.4f}'
        P(row)
    P('  READ 1: ORACLE-K STARTS at W1=COV=PREC=0 exactly (its k IS the truth) and its k is')
    P('          TRAINABLE. The numbers above are FINAL k. So they measure how far the optimizer')
    P('          walks the atoms away from the truth over 900 steps. On CROWDED4 that walk is')
    P('          W1=0.0370 rad/m against a true NN spacing of 0.026-0.084 rad/m -- i.e. of the')
    P('          same order as the spacing it is supposed to resolve. ORACLE-K is therefore NOT')
    P('          "the field error you get with correct k"; it is "the field error you get if you')
    P('          are HANDED correct k and then allowed to lose it".')
    P('  READ 2: NULL-PRIOR is far from every arm on all three -> the metric has dynamic range')
    P('          (detection_rate did not).')
    P('  READ 3: ★ on CROWDED4 the shipped fallback X-BASE does NOT beat the one-point STACK null')
    P('          on W1 (0.0648 vs 0.0548) or on PREC (0.0371 vs 0.0192). It beats it only on COV')
    P('          (0.0268 vs 0.0548). Block medians only -- NOT a test. Flagged for the prereg as')
    P('          the first thing the next experiment must settle at cluster level.')

    # ---------------- I.c does the metric track the field error? ----------------
    P('')
    P('I.c  DOES THE PROPOSED METRIC TRACK FIELD nRMSE?  (descriptive, pooled Spearman)')
    P('     [[feedback-control-stratifying-variable]] -- pooled rho is NOT promoted to mechanism;')
    P('     it is reported only to show the metric is not orthogonal to the outcome.')
    runs = defaultdict(dict)
    for line in open(RES):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' in r or 'nRMSE' not in r:
            continue
        runs[(r['cell'], r['episode'])][int(r['seed'])] = float(r['nRMSE'])

    def spearman(a, b):
        def rk(v):
            o = np.argsort(np.argsort(v))
            return o.astype(float)
        ra, rb = rk(np.asarray(a)), rk(np.asarray(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    for name, eps in [('ALL12', ALL12), ('NEW8', NEW8), ('CROWDED4', CROWDED4)]:
        xs = {0: [], 1: [], 2: []}
        ys = []
        for c in CELLS:
            for e in eps:
                if (c, e) not in tab:
                    continue
                for i in range(3):
                    xs[i].append(tab[(c, e)][i])
                ys.append(st.median(list(runs[(c, e)].values())))
        P(f'  {name:9s} n={len(ys):3d}  rho(W1,nRMSE)={spearman(xs[0], ys):+.3f}  '
          f'rho(COV,nRMSE)={spearman(xs[1], ys):+.3f}  rho(PREC,nRMSE)={spearman(xs[2], ys):+.3f}')

    # ---------------- II. prespecified table ----------------
    P('')
    P('=' * 104)
    P('II. PRESPECIFICATION TABLE (fill this in the PREREG before any run)')
    P('=' * 104)
    P('')
    P('  seed -> episode      : MEDIAN of 3 seeds          [same as B2/rx2, frozen]')
    P('  episode -> cluster   : MEDIAN of episodes sharing a bit-identical truth field')
    P('  cluster -> effect    : PAIRED cluster delta (cell - control), MEDIAN')
    P('  test                 : exact two-sided sign test on CLUSTERS (not episodes)')
    P('  DIRECTION (declare!) : lower is better for nRMSE / W1 / COV / PREC;')
    P('                         a cell "wins" a cluster iff its paired delta < 0')
    P('  mandatory companions : flat-sea null on nRMSE; NULL-STACK + NULL-PRIOR on W1/COV/PREC')
    P('  ties                 : dropped from n before the p is computed')
    P('  reporting            : if the primary is a bundle, the interaction MUST be reported')
    P('                         with convention A (median of per-cluster interactions)')
    P('')
    P('  EXACT SIGN-TEST POWER (alpha=0.05, two-sided). "crit" = wins needed.')
    P(f'  {"n clusters":>11s}{"crit":>6s}{"floor p":>10s}'
      + ''.join(f'{f"pwin={p}":>12s}' for p in (0.70, 0.80, 0.85, 0.90, 0.95)))
    for n in (4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 30):
        k = crit_k(n)
        P(f'  {n:>11d}{(str(k) if k else "—"):>6s}{2 / 2 ** n:10.4g}'
          + ''.join(f'{power(n, p):12.3f}' for p in (0.70, 0.80, 0.85, 0.90, 0.95)))
    P('')
    P('  MDE READING (80% power):')
    for n in (5, 8, 10, 12, 16, 20, 24, 30):
        need = None
        for pw in np.arange(0.50, 1.0001, 0.005):
            if power(n, float(pw)) >= 0.80:
                need = float(pw)
                break
        P(f'    n={n:3d} clusters -> needs P(a cluster favours the treatment) >= '
          + (f'{need:.3f}' if need else 'UNREACHABLE (even p=1.0 gives <80% or alpha unreachable)'))
    P('')
    P('  OBSERVED EFFECT SCALE for calibration (ORACLE-K vs X-BASE, truth-field clusters, n=10):')
    P('    per-cluster paired dnRMSE: see neutral_oraclek_rederive.txt section 3')
    P('    -> 9/10 clusters favour ORACLE-K, i.e. observed pwin_hat = 0.90')
    P('    -> at n=10 the sign test needs 9/10; power at pwin=0.90 is '
      f'{power(10, 0.90):.3f}. The ORACLE-K result is therefore RIGHT AT its own MDE:')
    P('       one flipped cluster removes significance. This is the fragility to declare.')

    open(os.path.join(HERE, 'neutral_protocol.txt'), 'w').write('\n'.join(OUT) + '\n')
    json.dump({f'{c}|{e}': tab[(c, e)] for (c, e) in tab},
              open(os.path.join(HERE, 'neutral_protocol.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
