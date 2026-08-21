"""NEUTRAL (axis 6 = k-placement) — independent re-derivation of the ORACLE-K claim.

Written from scratch. Does NOT import analyze_b2.py, does NOT read analysis.txt.
Reads only:
  tmp/B2_g1p1_ext_2026-08-06/results.jsonl   (216 raw runs)
  data_finemesh_2026-07-23/*.h5              (generator params + truth fields)

Produces:
  (1) episode-unit paired sign test for every cell vs X-BASE
  (2) cluster-unit sign test at two clusterings I derive myself (truth-field, phase)
  (3) exact attainable-p floors
  (4) robustness of the headline to the seed->episode collapse rule
"""
import hashlib
import json
import math
import os
import statistics as st
from collections import defaultdict
from itertools import combinations

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RES = os.path.join(REPO, 'tmp', 'B2_g1p1_ext_2026-08-06', 'results.jsonl')
RX2 = os.path.join(REPO, 'tmp', 'rx2_init_2026-08-05', 'results.jsonl')
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')

CROWDED4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
NEW8 = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
        'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
        'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
        'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5']
ALL12 = NEW8 + CROWDED4
CELLS = ['X-BASE', 'G1', 'P1', 'G1P1', 'ORACLE-K', 'AUTO', 'ORACLE']

OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


# ---------------------------------------------------------------- statistics
def sign_p_two_sided(k, n):
    """Exact two-sided sign test. k = #wins among n non-tied units."""
    if n == 0:
        return 1.0
    kk = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(kk + 1))
    return min(1.0, 2.0 * tail / 2.0 ** n)


def floor_p(n):
    return min(1.0, 2.0 / 2.0 ** n) if n > 0 else 1.0


def min_wins_for_alpha(n, alpha=0.05):
    """Smallest k (>= n/2) whose two-sided sign-test p <= alpha; None if unreachable."""
    for k in range(n, n // 2, -1):
        if sign_p_two_sided(k, n) > alpha:
            return k + 1 if k + 1 <= n else None
    return (n // 2) + 1 if sign_p_two_sided(n, n) <= alpha else None


def best_k(n, alpha=0.05):
    """Smallest number of wins out of n reaching alpha, or None."""
    for k in range((n // 2) + 1, n + 1):
        if sign_p_two_sided(k, n) <= alpha:
            return k
    return None


def wilcoxon_signed_rank_exact(d):
    """Exact two-sided Wilcoxon signed-rank on nonzero d (small n only)."""
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0 or n > 16:
        return None, n
    a = sorted((abs(x), i) for i, x in enumerate(d))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[j + 1][0] == a[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for q in range(i, j + 1):
            ranks[a[q][1]] = r
        i = j + 1
    wpos = sum(ranks[i] for i in range(n) if d[i] > 0)
    tot = sum(ranks)
    obs = min(wpos, tot - wpos)
    # enumerate all 2^n sign assignments
    cnt = 0
    for mask in range(1 << n):
        s = sum(ranks[i] for i in range(n) if mask >> i & 1)
        if min(s, tot - s) <= obs + 1e-9:
            cnt += 1
    return min(1.0, cnt / 2.0 ** n), n


# ---------------------------------------------------------------- data
def load_runs(path):
    d = defaultdict(dict)
    n_err = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            n_err += 1
            continue
        if 'error' in r or 'nRMSE' not in r:
            n_err += 1
            continue
        d[(r['cell'], r['episode'])][int(r['seed'])] = float(r['nRMSE'])
    return d, n_err


def load_specs(path):
    d = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' in r or 'spec' not in r:
            continue
        d[(r['cell'], r['episode'])].append(r['spec'])
    return d


def md5(v):
    return hashlib.md5(np.ascontiguousarray(v, dtype='<f8').tobytes()).hexdigest()


def clusterings():
    """Derive my own cluster labels from the HDF5 generator params + truth fields."""
    field, phase, kvec, amp = {}, {}, {}, {}
    for ep in ALL12:
        with h5py.File(os.path.join(DATA, ep), 'r') as f:
            wc = np.array(f['config/wave_components'])
            fld = np.array(f['frames/gt_wave_field'])
        field[ep] = md5(fld)
        phase[ep] = md5(wc[:, 4])
        kvec[ep] = md5(np.concatenate([wc[:, 2], wc[:, 3]]))
        amp[ep] = md5(wc[:, 0])
    return {'truth-field': field, 'phase-phi': phase, 'k-vector': kvec, 'amplitude': amp}


# ---------------------------------------------------------------- main
def main():
    P('=' * 100)
    P('NEUTRAL axis-6 (k placement) — independent re-derivation of "ORACLE-K -0.0105, 11/12, p=0.0063"')
    P('=' * 100)
    runs, n_err = load_runs(RES)
    P(f'source: {RES}')
    P(f'parsed runs: {sum(len(v) for v in runs.values())}   skipped/error lines: {n_err}')

    cover = {(c, e): len(runs[(c, e)]) for c in CELLS for e in ALL12}
    bad = {k: v for k, v in cover.items() if v != 3}
    P(f'cells x episodes with != 3 seeds: {len(bad)}'
      + ('' if not bad else f'   -> {sorted(set(k[0] for k in bad))}'))
    seeds = sorted({s for v in runs.values() for s in v})
    P(f'seeds present: {seeds}')

    present = [c for c in CELLS if all(cover.get((c, e), 0) == 3 for e in ALL12)]
    P(f'cells with full 12x3 coverage: {present}')

    def med(c, e):
        return st.median(list(runs[(c, e)].values()))

    def mean(c, e):
        return st.mean(list(runs[(c, e)].values()))

    # ---------------- 1. episode-unit ----------------
    P('')
    P('=' * 100)
    P('1. EPISODE-UNIT paired sign test vs X-BASE  (seed->episode = median-of-3, frozen rule)')
    P('=' * 100)
    for blockname, eps in [('ALL12', ALL12), ('NEW8', NEW8), ('CROWDED4', CROWDED4)]:
        n = len(eps)
        P(f'\n--- {blockname} (n={n}, attainable floor p={floor_p(n):.4g}, '
          f'min wins for a<=0.05: {best_k(n)}) ---')
        P(f'{"cell":10s}{"medOfMed":>10s}{"pairedD_med":>13s}{"wins":>9s}{"signP":>9s}'
          f'{"wilcoxP":>9s}{"mean d":>10s}')
        for c in present:
            lvl = st.median([med(c, e) for e in eps])
            if c == 'X-BASE':
                P(f'{c:10s}{lvl:10.4f}{"—":>13s}{"—":>9s}{"—":>9s}{"—":>9s}{"—":>10s}')
                continue
            ds = [med(c, e) - med('X-BASE', e) for e in eps]
            nz = [x for x in ds if x != 0]
            w = sum(1 for x in nz if x < 0)
            wp, _ = wilcoxon_signed_rank_exact(ds)
            P(f'{c:10s}{lvl:10.4f}{st.median(ds):+13.4f}{w:>5d}/{len(nz):<3d}'
              f'{sign_p_two_sided(w, len(nz)):9.4f}'
              f'{(wp if wp is not None else float("nan")):9.4f}{st.mean(ds):+10.4f}')

    # per-episode ORACLE-K deltas, printed so the sign pattern is auditable
    P('')
    P('per-episode ORACLE-K - X-BASE  (median-of-3):')
    for e in ALL12:
        d = med('ORACLE-K', e) - med('X-BASE', e)
        P(f'   {e[8:-3]:32s} XB={med("X-BASE", e):7.4f}  OK={med("ORACLE-K", e):7.4f}  '
          f'd={d:+8.4f}  {"win" if d < 0 else "LOSS"}')

    # ---------------- 2. robustness to the collapse rule ----------------
    P('')
    P('=' * 100)
    P('2. ROBUSTNESS of the ORACLE-K headline to the seed->episode collapse rule (ALL12)')
    P('=' * 100)
    for label, agg in [('median-of-3 (frozen)', med), ('mean-of-3', mean),
                       ('best-of-3 (min)', lambda c, e: min(runs[(c, e)].values())),
                       ('worst-of-3 (max)', lambda c, e: max(runs[(c, e)].values()))]:
        ds = [agg('ORACLE-K', e) - agg('X-BASE', e) for e in ALL12]
        w = sum(1 for x in ds if x < 0)
        P(f'  {label:24s} pairedD_med {st.median(ds):+8.4f}  wins {w}/12  '
          f'signP {sign_p_two_sided(w, 12):.4f}')
    # seed-as-unit (36 pairs) -- NOT the frozen rule; shown to expose pseudo-replication
    ds36 = [runs[('ORACLE-K', e)][s] - runs[('X-BASE', e)][s] for e in ALL12 for s in seeds]
    w36 = sum(1 for x in ds36 if x < 0)
    P(f'  {"seed-as-unit (36 pairs)":24s} pairedD_med {st.median(ds36):+8.4f}  wins {w36}/36  '
      f'signP {sign_p_two_sided(w36, 36):.4g}   <-- PSEUDO-REPLICATION, not a valid unit')

    # ---------------- 3. cluster-unit ----------------
    P('')
    P('=' * 100)
    P('3. CLUSTER-UNIT sign test — clusters derived by me from the HDF5, not from any prior file')
    P('=' * 100)
    cl = clusterings()
    for cname, lab in cl.items():
        groups = defaultdict(list)
        for e in ALL12:
            groups[lab[e]].append(e)
        C = len(groups)
        P(f'\n--- clustering "{cname}": {C} clusters over 12 episodes '
          f'(floor p={floor_p(C):.4g}, min wins for a<=0.05: {best_k(C)}) ---')
        for g in sorted(groups.values(), key=lambda v: ALL12.index(v[0])):
            P('     {' + ', '.join(e[8:-3] for e in g) + '}')
        for c in present:
            if c == 'X-BASE':
                continue
            cds = []
            for g in groups.values():
                cds.append(st.median([med(c, e) - med('X-BASE', e) for e in g]))
            nz = [x for x in cds if x != 0]
            w = sum(1 for x in nz if x < 0)
            P(f'  {c:10s} clusterD_med {st.median(cds):+8.4f}  wins {w}/{len(nz)}  '
              f'signP {sign_p_two_sided(w, len(nz)):.4f}')

    # ---------------- 4. spectral k-error descriptives ----------------
    P('')
    P('=' * 100)
    P('4. SPECTRAL k-ERROR descriptives from the same 216 runs (protocol material, descriptive only)')
    P('    matching rule in force: Hungarian on cost = |dk| / max_j|k_j^gt|, accept iff cost < 0.5')
    P('=' * 100)
    sp = load_specs(RES)
    P(f'{"cell":10s}{"det_rate":>10s}{"E_det":>9s}{"n_est/n_gt":>12s}'
      f'{"|dk|/k":>10s}{"dir_deg":>10s}{"|dA|/A":>10s}')
    for c in present:
        rows = [s for e in ALL12 for s in sp[(c, e)]]
        f = lambda key: np.nanmedian([r.get(key, np.nan) for r in rows])
        P(f'{c:10s}{f("detection_rate"):10.3f}{f("energy_detection_rate"):9.3f}'
          f'{np.nanmedian([r["n_est"] / max(r["n_gt"], 1) for r in rows]):12.2f}'
          f'{f("k_mag_rel_error_mean"):10.4f}{f("direction_error_deg_mean"):10.3f}'
          f'{f("amp_rel_error_mean"):10.3f}')
    P('  NOTE: ORACLE-K is handed n_gaussians = n_gt AND the true (kx,ky); its spec row is')
    P('        therefore not an estimate of anything. It is printed as a sanity anchor only.')

    # ---------------- 5. what ORACLE-K actually manipulates ----------------
    P('')
    P('=' * 100)
    P('5. CONFOUND LEDGER — factors that differ between X-BASE and ORACLE-K')
    P('=' * 100)
    P('  a. (kx,ky) placement : beamformed  ->  TRUE values                 [the named factor]')
    P('  b. model order N     : ladder [3,8,15] + adaptive_n  ->  exactly n_gt   [CONFOUND]')
    P('  c. stage schedule    : 3 stages [200,200,500]        ->  1 stage [900]  [CONFOUND]')
    P('  d. loss thresholds   : [0.1,0.05,None]               ->  [None]         [CONFOUND]')
    P('  (w, phi) are NOT oracle in either arm (w=0.01, phi~U(0,2pi)) -- verified in')
    P('  wavesplat/optimizer/initializer.py::initialize_oracle docstring + body.')
    P('  => "ORACLE-K measures the k-position ceiling" is a BUNDLE claim, not single-factor.')
    P('  n_gt per episode (drives factor b):')
    for e in ALL12:
        with h5py.File(os.path.join(DATA, e), 'r') as f:
            ngt = int(np.array(f['config/wave_components']).shape[0])
        ns = [runs[('X-BASE', e)] and None]
        P(f'     {e[8:-3]:32s} n_gt={ngt:3d}   X-BASE ladder=[3,8,15]+adaptive')

    with open(os.path.join(HERE, 'neutral_oraclek_rederive.txt'), 'w') as fh:
        fh.write('\n'.join(OUT) + '\n')


if __name__ == '__main__':
    main()
