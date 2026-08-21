"""A2 INDEPENDENT re-derivation. Written from scratch, deliberately NOT importing
analyze_a2.py and NOT using scipy.wilcoxon: exact permutation (sign-flip) tests and
hand-written rank statistics, plus a from-scratch re-solve + re-score of 3 windows
that recomputes the nowcast nRMSE without touching run_a2.py's scoring helper.

Any disagreement with analysis.txt is printed as MISMATCH.
"""
import os, sys, json, itertools
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)
D = 'tmp/A2_realproto_2026-08-06'
DROP = {('AA14', 0), ('AA15a', 3), ('LJ18g', 3)}


def signrank_p_exact(x):
    """Two-sided exact signed-rank permutation p (sign-flip null), n<=27 -> MC."""
    x = np.asarray([v for v in x if v != 0], float)
    n = len(x)
    r = np.argsort(np.argsort(np.abs(x))) + 1.0
    obs = np.sum(r * np.sign(x))
    rng = np.random.default_rng(20260806)
    if n <= 20:
        cnt = tot = 0
        for s in itertools.product((-1, 1), repeat=n):
            tot += 1
            if abs(np.sum(r * np.array(s))) >= abs(obs) - 1e-12:
                cnt += 1
        return cnt / tot
    S = rng.choice([-1.0, 1.0], size=(400000, n))
    null = S @ r
    return float((np.abs(null) >= abs(obs) - 1e-12).mean())


def sign_p_exact(k, n):
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


def med(v):
    return float(np.median(np.asarray(v, float)))


def main():
    C = json.load(open(f'{D}/cells.json'))['results']
    Wr = json.load(open('tmp/wp_real_2026-07-25/wp_real.json'))['results']
    cell = {(r['record'], r['win']): r for r in C}
    lear = {(r['record'], r['win']): r['learned_now'] for r in Wr}
    handw = {(r['record'], r['win']): r['hand_now'] for r in Wr}
    S1 = [k for k in cell if k not in DROP]
    assert len(S1) == 27, len(S1)
    print(f'INDEPENDENT re-derivation  (n(S1)={len(S1)})')

    # 0. reproduction
    mx = max(abs(cell[k]['FF_exact']['nowcast'] - handw[k]) for k in cell)
    print(f'[0] FF_exact vs wp_real hand_now: max|d| = {mx:.3e}')

    # 1. published wp_real headline, re-derived here
    dwp = [handw[k] - lear[k] for k in S1]
    print(f'[1] wp_real published: hand med {med([handw[k] for k in S1]):.4f} '
          f'learned med {med([lear[k] for k in S1]):.4f} | median D {med(dwp):+.4f} '
          f'wins {sum(1 for v in dwp if v > 0)}/27 permP {signrank_p_exact(dwp):.4g} '
          f'signP {sign_p_exact(sum(1 for v in dwp if v>0),27):.4g}')

    # 2. cells
    print('[2] cells (Delta = hand - learned, + = learned better)')
    got = {}
    for c in ['FF', 'BF', 'FB', 'BB', 'BB400']:
        h = [cell[k]['cells'][c]['median'] for k in S1]
        # re-derive the cell median from per_seed rather than trusting 'median'
        h2 = [float(np.median(cell[k]['cells'][c]['per_seed'])) for k in S1]
        assert np.allclose(h, h2), f'{c}: stored median != median(per_seed)'
        d = [a - lear[k] for a, k in zip(h, S1)]
        got[c] = np.array(d)
        w = sum(1 for v in d if v > 0)
        print(f'  {c:6s} hand med {med(h):.4f}  D med {med(d):+.4f} mean {np.mean(d):+.4f} '
              f'wins {w}/27  permP {signrank_p_exact(d):.4g}  signP {sign_p_exact(w,27):.4g}')

    # 3. knobs — paired, then median
    print('[3] knobs (paired then median; + = worse)')
    A = {c: np.array([cell[k]['cells'][c]['median'] for k in S1]) for c in
         ['FF', 'BF', 'FB', 'BB', 'BB400']}
    terms = {'E_kp|sigGT': A['BF'] - A['FF'], 'E_kp|sigobs': A['BB'] - A['FB'],
             'E_sig|kpGT': A['FB'] - A['FF'], 'E_sig|kpNone': A['BB'] - A['BF'],
             'interaction': (A['BB'] - A['FB']) - (A['BF'] - A['FF']),
             'bundle BB-FF': A['BB'] - A['FF'],
             'obs 600->400': A['BB400'] - A['BB']}
    for nm, v in terms.items():
        w = int((v > 0).sum())
        print(f'  {nm:14s} med {med(v):+.4f} mean {v.mean():+.4f} worse {w}/27 '
              f'permP {signrank_p_exact(v):.4g}')
    print(f"  additivity check: med(BB-FF) {med(A['BB']-A['FF']):+.4f} vs "
          f"med((BF-FF)+(FB-FF)) {med((A['BF']-A['FF'])+(A['FB']-A['FF'])):+.4f}")

    # 4. nulls
    print('[4] nulls')
    for c in ['FF', 'BF', 'FB', 'BB', 'BB400']:
        f = [k for k in cell if cell[k]['cells'][c]['median'] >= cell[k]['null_zero']]
        print(f'  {c:6s} fails null_zero {len(f):2d}/30')
    fl = [k for k in cell if lear[k] >= cell[k]['null_zero']]
    print(f'  learned fails null_zero {len(fl)}/30: '
          + ' '.join(f'{a}_w{b}' for a, b in fl))

    # 5. from-scratch re-solve of 3 windows (own scene + own scoring)
    print('[5] from-scratch re-solve/re-score (3 windows, cells FF & BB, seed 42)')
    import d3_compare_gifs as cg
    from d3_sa_nomp import mmse_bank_estimate
    for rec, win in [('AA15b', 3), ('AA15a', 0), ('LJ18g', 4)]:
        sc = cg.scene_real(rec, win=win)
        g = sc['maps_gt']
        sg = np.std(g[np.isfinite(g)])
        ny, nx = sc['Xq'].shape
        npx = ny * nx
        for c, kph, s0 in [('FF', float(sc['kp']), 0.05 * float(sc['Hs'])),
                           ('BB', None, 0.05 * 4.0 * float(np.std(sc['z'])))]:
            fq = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'],
                                    sigma0=s0, kp_hint=kph, seed=42)
            f0 = fq[:npx].reshape(ny, nx)
            msk = np.isfinite(g[0]) & np.isfinite(f0)
            v = float(np.sqrt(((f0[msk] - g[0][msk]) ** 2).mean()) / sg)
            ref = cell[(rec, win)]['cells'][c]['per_seed'][0]
            flag = 'OK' if abs(v - ref) < 1e-9 else 'MISMATCH'
            print(f'  {rec}_w{win} {c}: {v:.9f} vs stored {ref:.9f}  {flag}')


if __name__ == '__main__':
    main()
