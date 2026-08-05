"""Independent re-derivation of the Rx-1 headline numbers.

Deliberately does NOT import analyze_rx1. Different code path on purpose:
  * medians computed by explicit sort, not numpy
  * |VR-1| / decorr / shrink recomputed from the STORED per-frame arrays, then
    cross-checked against the fields the runner wrote
  * sign test written from scratch
Any disagreement with analyze_rx1 is a bug in one of them.
"""
import json
import os
import sys
from math import comb, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float('nan')
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def sign_p(k, n):
    kk = min(k, n - k)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(kk + 1)) / 2.0 ** n)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results2.jsonl')
    rows = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' not in r and 'cell' in r:
            rows[r['job_id']] = r
    rows = list(rows.values())
    print(f'verify: {len(rows)} runs from {os.path.basename(path)}')

    # --- 1. per-frame field audit: recompute from the raw arrays ---
    bad = 0
    checked = 0
    for r in rows:
        if 'vr_frames' not in r:
            continue
        vr, rho = r['vr_frames'], r['rho_frames']
        a = sum(abs(v - 1.0) for v in vr) / len(vr)
        d = sum(1 - x * x for x in rho) / len(rho)
        s = sum((sqrt(max(v, 0)) - x) ** 2 for v, x in zip(vr, rho)) / len(vr)
        checked += 1
        for name, mine, theirs in (('absVR1', a, r['absVR1_framemean']),
                                   ('decorr', d, r['decorr_framemean']),
                                   ('shrink', s, r['shrink_framemean'])):
            if abs(mine - theirs) > 2e-5:
                bad += 1
                print(f'  MISMATCH {r["job_id"]} {name} {mine:.6f} vs {theirs:.6f}')
    print(f'  per-frame recomputation: {checked} runs checked, {bad} mismatches '
          f'(rounding tol 2e-5 because arrays are stored at 6 dp)')

    # --- 2. mean-then-transform vs transform-then-mean, on |VR-1| ---
    print('\n  |VR-1| ORDERING (AMENDMENT 2): per-frame-then-mean vs mean-then-transform')
    byc = {}
    for r in rows:
        if 'vr_frames' not in r:
            continue
        byc.setdefault(r['cell'], []).append(r)
    for c in sorted(byc):
        rs = byc[c]
        correct = median([x['absVR1_framemean'] for x in rs])
        wrong = median([abs(x['Variance_Ratio'] - 1.0) for x in rs])
        frac_above1 = sum(sum(1 for v in x['vr_frames'] if v > 1) for x in rs) / \
            sum(len(x['vr_frames']) for x in rs)
        print(f'    {c:>10s}  per-frame {correct:.4f}   mean-then-transform {wrong:.4f}'
              f'   diff {correct - wrong:+.4f}   frames with VR>1: {100*frac_above1:5.1f}%')

    # --- 3. headline contrasts, recomputed ---
    def cellmed(cell, metric):
        out = {}
        for r in rows:
            if r['cell'] != cell:
                continue
            out.setdefault(r['episode'], []).append(r.get(metric))
        return {e: median([v for v in vs if v is not None]) for e, vs in out.items()}

    print('\n  headline contrasts (nRMSE, paired per-episode median-of-3):')
    pairs = [('K1', 'X-BASE'), ('K2', 'X-BASE'), ('K12', 'X-BASE'), ('K3', 'X-BASE'),
             ('K4', 'X-BASE'), ('K5', 'X-BASE'), ('K6', 'X-BASE'), ('K7', 'X-BASE'),
             ('K12-S1', 'X-BASE'), ('X-NOTRACK', 'X-BASE'),
             ('D-K12', 'D-BASE'), ('D-K6', 'D-BASE'), ('D-NOTRACK', 'D-BASE')]
    for c, b in pairs:
        A, B = cellmed(c, 'nRMSE'), cellmed(b, 'nRMSE')
        eps = sorted(set(A) & set(B))
        if not eps:
            continue
        d = [A[e] - B[e] for e in eps]
        k = sum(1 for x in d if x < 0)
        print(f'    {c:>10s} vs {b:<8s} medD {median(d):+8.4f}  better {k}/{len(d)}  '
              f'p={sign_p(k, len(d)):.4f}   ' + ' '.join(f'{x:+.4f}' for x in d))

    # --- 4. the mechanism claim, recomputed ---
    print('\n  mechanism: share of d_m carried by the rho channel (1-rho^2):')
    for c, b in pairs:
        dm = cellmed(c, 'm_framemean'), cellmed(b, 'm_framemean')
        dd = cellmed(c, 'decorr_framemean'), cellmed(b, 'decorr_framemean')
        ds = cellmed(c, 'shrink_framemean'), cellmed(b, 'shrink_framemean')
        eps = sorted(set(dm[0]) & set(dm[1]))
        if not eps:
            continue
        M = median([dm[0][e] - dm[1][e] for e in eps])
        D = median([dd[0][e] - dd[1][e] for e in eps])
        S = median([ds[0][e] - ds[1][e] for e in eps])
        if abs(M) < 1e-9:
            continue
        print(f'    {c:>10s} vs {b:<8s} d_m {M:+.4f} = decorr {D:+.4f} ({100*D/M:5.1f}%)'
              f' + shrink {S:+.4f} ({100*S/M:5.1f}%)')

    # --- 4b. Phase-1 contamination audit (AMENDMENT 2 hygiene) ---
    print('\n  Phase-1 contamination: is init_nrmse bit-identical to the reference cell?')
    idx = {(r['cell'], r['episode'], r['seed']): r for r in rows}
    for c, b in pairs:
        same = diff = 0
        worst = 0.0
        for (cc, e, s), r in idx.items():
            if cc != c:
                continue
            ref = idx.get((b, e, s))
            if ref is None:
                continue
            if repr(r['init_nrmse']) == repr(ref['init_nrmse']):
                same += 1
            else:
                diff += 1
                worst = max(worst, abs(r['init_nrmse'] - ref['init_nrmse']))
        if same + diff:
            print(f'    {c:>10s} vs {b:<8s} identical {same:2d} / differ {diff:2d}'
                  f'   worst |d init| {worst:.5f}'
                  f'   {"CLEAN tracker-only" if diff == 0 else "PHASE-1 CONTAMINATED"}')

    # --- 5. levels of the shrink share (how big is the shrinkage axis at all) ---
    print('\n  shrinkage axis as a share of the LEVEL (per cell):')
    for c in sorted(byc):
        M = median([r['m_framemean'] for r in byc[c]])
        S = median([r['shrink_framemean'] for r in byc[c]])
        print(f'    {c:>10s}  m {M:.4f}  shrink {S:.4f}  = {100*S/M:5.2f}% '
              f'(the most a perfect scalar rescale could remove)')


if __name__ == '__main__':
    main()
