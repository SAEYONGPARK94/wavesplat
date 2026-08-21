"""NEUTRAL (axis 6) — CORRECTION to the bit-identity independence audit.

tmp/audit_independence_2026-08-07/audit_independence.py counts independent units by
md5 over the raw kx / ky columns. That is BIT identity. But the generator draws a
DIRECTION theta and a MAGNITUDE k separately and stores kx = k cos(theta),
ky = k sin(theta). Two episodes that share the same direction draw but differ in k
therefore produce different kx bytes -- and get counted as two independent units --
even though the only new randomness between them is the magnitude/BF ladder, which
is deterministic, not a draw.

This script re-clusters on the RECOVERED direction angles with a tolerance, and does
the same for magnitude and phase, so the sign-test cluster count can be set on
DRAW identity rather than BYTE identity.

Read-only. HDF5 generator params only. No results files, no estimator runs.
"""
import hashlib
import json
import math
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')

P1_12 = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
         'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
         'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
         'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
         'episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
         'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
CROWDED16 = [f'episode_BF{bf}_{nc}comp_seed{s}_uniform.h5'
             for bf in (4, 5) for nc in (15, 30) for s in range(4)]

OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def load(ep):
    with h5py.File(os.path.join(DATA, ep), 'r') as f:
        return np.array(f['config/wave_components'])


def cluster_tol(vecs, tol):
    """Greedy single-link clustering: same cluster iff same length AND max|diff| <= tol.
    Returns a label per input."""
    lab = [-1] * len(vecs)
    nxt = 0
    for i, v in enumerate(vecs):
        if lab[i] >= 0:
            continue
        lab[i] = nxt
        for j in range(i + 1, len(vecs)):
            if lab[j] >= 0 or len(vecs[j]) != len(v):
                continue
            if np.max(np.abs(np.asarray(vecs[j]) - np.asarray(v))) <= tol:
                lab[j] = nxt
        nxt += 1
    return lab, nxt


def floor_p(c):
    return min(1.0, 2.0 / 2 ** c) if c > 0 else 1.0


def crit(n, alpha=0.05):
    def sp(k, nn):
        kk = min(k, nn - k)
        return min(1.0, 2.0 * sum(math.comb(nn, i) for i in range(kk + 1)) / 2.0 ** nn)
    for k in range((n // 2) + 1, n + 1):
        if sp(k, n) <= alpha:
            return k
    return None


def audit(name, eps):
    P('=' * 96)
    P(f'### {name}  ({len(eps)} episodes)')
    wcs = [load(e) for e in eps]
    direc = [np.degrees(np.arctan2(w[:, 3], w[:, 2])) for w in wcs]
    kmag = [np.hypot(w[:, 2], w[:, 3]) for w in wcs]
    phi = [w[:, 4] for w in wcs]
    amp = [w[:, 0] for w in wcs]
    kxy = [np.concatenate([w[:, 2], w[:, 3]]) for w in wcs]

    def bitcount(vs):
        return len({hashlib.md5(np.ascontiguousarray(v, '<f8').tobytes()).hexdigest() for v in vs})

    P(f'{"axis":14s}{"bit-identical":>15s}{"tol 1e-6":>10s}{"tol 1e-4":>10s}{"tol 1e-2":>10s}')
    res = {}
    for label, vs in [('direction deg', direc), ('|k| rad/m', kmag),
                      ('phase rad', phi), ('amplitude m', amp), ('(kx,ky) raw', kxy)]:
        row = [bitcount(vs)]
        for tol in (1e-6, 1e-4, 1e-2):
            row.append(cluster_tol(vs, tol)[1])
        res[label] = row
        P(f'{label:14s}{row[0]:15d}{row[1]:10d}{row[2]:10d}{row[3]:10d}')

    lab, C = cluster_tol(direc, 1e-4)
    P('')
    P(f'  DIRECTION clusters at tol 1e-4 deg  ->  {C} clusters:')
    for c in range(C):
        P('     {' + ', '.join(eps[i][8:-3] for i in range(len(eps)) if lab[i] == c) + '}')
    lab2, C2 = cluster_tol(phi, 1e-4)
    P(f'  PHASE clusters at tol 1e-4 rad      ->  {C2} clusters')
    narrow = min(res[k][2] for k in res)
    P('')
    P(f'  ★ narrowest DRAW axis at tol 1e-4 = {narrow} clusters'
      f'  -> sign-test floor p = {floor_p(narrow):.4g}, wins needed for a<=0.05 = {crit(narrow)}')
    P(f'    (bit-identity audit would have said {min(res[k][0] for k in res)} clusters,'
      f' floor p = {floor_p(min(res[k][0] for k in res)):.4g})')
    return {'set': name, 'n': len(eps), 'axes': res, 'narrow_tol4': narrow,
            'floor_p_tol4': floor_p(narrow)}


def main():
    P('NEUTRAL axis-6 — DRAW-identity vs BYTE-identity independence audit')
    P('the generator stores kx = |k| cos(theta); two episodes sharing the SAME theta draw')
    P('but a different |k| ladder differ in BYTES while sharing the randomness.')
    P('')
    rows = [audit('P1_12  (the ORACLE-K "11/12" panel)', P1_12),
            audit('CROWDED16  (Rx-4 block C)', CROWDED16)]
    P('=' * 96)
    P('SUMMARY')
    for r in rows:
        P(f'  {r["set"]:40s} n={r["n"]:2d}  draw-clusters={r["narrow_tol4"]:2d}  '
          f'floor p={r["floor_p_tol4"]:.4g}  crit={crit(r["narrow_tol4"])}')
    json.dump(rows, open(os.path.join(HERE, 'neutral_cluster_tolerance.json'), 'w'),
              indent=1, ensure_ascii=False)
    open(os.path.join(HERE, 'neutral_cluster_tolerance.txt'), 'w').write('\n'.join(OUT) + '\n')


if __name__ == '__main__':
    main()
