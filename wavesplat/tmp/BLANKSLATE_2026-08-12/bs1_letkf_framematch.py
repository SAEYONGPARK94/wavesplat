#!/usr/bin/env python3
"""BS1-c: score LETKF on the SAME frames as WaveSplat and redo the head-to-head.

WHY
---
`scripts/run_100ep_letkf_and_ws_metrics.py` scores LETKF with `for frame in
loader:` (all 301 frames, cold start included) but scores WaveSplat only on
`get_tracking_frames(tp_factor=2.0)` (t > 2*Tp).  `results/100ep_ws_letkf.json`
therefore compares the two methods on DIFFERENT frame sets.  `bs1_letkf_
perframe.py` re-ran LETKF keeping every frame, so the same episodes can now be
aggregated on either set.

PROTOCOL — frozen before looking at any output of this script
-------------------------------------------------------------
* per-episode statistic : mean nRMSE over the frames of that episode
* frame sets            : ALL   = every frame (reproduces the published file)
                          MATCH = t > 2*Tp   (WaveSplat's own convention)
* WaveSplat numbers     : taken unchanged from results/100ep_ws_letkf.json
                          (already a mean over tracking frames)
* across-episode agg    : MEDIAN, paired sign test, TWO-SIDED, alpha = 0.05
                          (two-sided: this is a protocol repair, not a
                           treatment test, so no direction is pre-specified)
* strata (pre-specified): ncomp <= 5  vs  ncomp >= 15  -- the regime boundary
                          already established by X1/X2, X3, B-3, S3, S8/S9.
                          Holm over the 2 strata.
* G-0 self-test         : the ALL-frame recompute must reproduce the LETKF
                          nRMSE in results/100ep_ws_letkf.json.  If it does
                          not, the per-frame re-run is not the same estimator
                          and NOTHING below may be cited.
* G-1 self-test         : this script's `t > 2*Tp` mask must select the same
                          frame COUNT per episode as the independent raw-HDF5
                          floor computation (floor_zero_predictor.json), and
                          reproduce its zero-field nRMSE.  Otherwise "matched"
                          is not matched.

NOT MEASURED HERE
-----------------
H-3 timing is CROSS-PROCESS: the WaveSplat ms comes from the published
results file, the LETKF ms from this re-run, on different days and different
process states.  It is a sanity check, NOT a cost comparison, and the ratio
must not be cited.  The same-process cost measurement is bs1_shipping_
default.py.  (See also the standing ban on "13.4 ms" / "132.6 ms".)

Usage:  python3 tmp/BLANKSLATE_2026-08-12/bs1_letkf_framematch.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

PF = os.path.join(HERE, 'letkf_perframe.json')
PUB = os.path.join(ROOT, 'results', '100ep_ws_letkf.json')
OUT = os.path.join(HERE, 'letkf_framematch.json')

ALPHA = 0.05


def sign_test(a, b):
    """Paired two-sided sign test.  Returns (n_a_lower, n_b_lower, p)."""
    from scipy import stats
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d) & (d != 0)]
    k = int((d < 0).sum())           # a < b  (a wins, lower nRMSE is better)
    n = int(d.size)
    p = float(stats.binomtest(k, n, 0.5, alternative='two-sided').pvalue)
    return k, n - k, n, p


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * pvals[j])
        adj[j] = min(1.0, run)
    return adj


def ncomp_of(key):
    return int(key.split('_')[1].replace('comp', ''))


def main():
    pf = json.load(open(PF))
    pub = json.load(open(PUB))
    keys = sorted(set(pf) & set(pub))
    print(f'episodes: perframe={len(pf)}  published={len(pub)}  common={len(keys)}')
    if len(keys) != 100:
        print('!! WARNING: not the full 100-episode set')

    rows = {}
    for k in keys:
        e = pf[k]
        fr = e['frames']
        t = np.array([f['t'] for f in fr], float)
        nr = np.array([f['nRMSE'] for f in fr], float)
        zr = np.array([f['zero_nRMSE'] for f in fr], float)
        ms = np.array([f['ms'] for f in fr], float)
        sel = t > 2.0 * e['Tp']
        rows[k] = dict(
            letkf_all=float(np.nanmean(nr)),
            letkf_match=float(np.nanmean(nr[sel])),
            zero_all=float(np.nanmean(zr)),
            zero_match=float(np.nanmean(zr[sel])),
            letkf_ms_median=float(np.nanmedian(ms)),
            n_all=int(nr.size), n_match=int(sel.sum()),
            Tp=e['Tp'], ncomp=ncomp_of(k),
            ws=float(pub[k]['WaveSplat']['nRMSE']),
            letkf_pub=float(pub[k]['LETKF']['nRMSE']),
        )

    # ---------------------------------------------------------------- G-0
    print('\n' + '=' * 74)
    print('G-0  self-test: does the ALL-frame recompute reproduce the published')
    print('     LETKF column of results/100ep_ws_letkf.json?')
    print('=' * 74)
    a = np.array([rows[k]['letkf_all'] for k in keys])
    b = np.array([rows[k]['letkf_pub'] for k in keys])
    dif = np.abs(a - b)
    n_close = int((dif < 1e-6).sum())
    print(f'  |recompute - published|:  max={dif.max():.3e}  median={np.median(dif):.3e}')
    print(f'  exact (<1e-6) on {n_close}/{len(keys)} episodes')
    g0 = n_close == len(keys)
    print(f'  G-0 {"PASS" if g0 else "FAIL -- results below are NOT citable"}')

    # ---------------------------------------------------------------- G-1
    fzp = os.path.join(HERE, 'floor_zero_predictor.json')
    g1 = None
    if os.path.exists(fzp):
        fz = json.load(open(fzp))
        kk = [k for k in keys if k in fz]
        nf = np.array([fz[k]['n_eval_frames'] for k in kk])
        nm = np.array([rows[k]['n_match'] for k in kk])
        zf = np.array([fz[k]['zero'] for k in kk])
        zm = np.array([rows[k]['zero_match'] for k in kk])
        same_n = int((nf == nm).sum())
        g1 = (same_n == len(kk)) and bool(np.abs(zf - zm).max() < 1e-6)
        print('\nG-1  self-test vs the independent raw-HDF5 floor computation')
        print(f'     frame count identical on {same_n}/{len(kk)} episodes')
        print(f'     |zero_floor difference| max={np.abs(zf - zm).max():.3e}')
        print(f'     G-1 {"PASS" if g1 else "FAIL -- MATCH is not matched"}')

    n_all = np.array([rows[k]['n_all'] for k in keys])
    n_ma = np.array([rows[k]['n_match'] for k in keys])
    print(f'\n  frames per episode: ALL median={np.median(n_all):.0f}  '
          f'MATCH median={np.median(n_ma):.0f}  '
          f'(MATCH drops {100*(1-np.median(n_ma)/np.median(n_all)):.0f}% of frames)')

    # ------------------------------------------------------------- headline
    print('\n' + '=' * 74)
    print('H-1  WaveSplat vs LETKF on the SAME frames (t > 2*Tp), all 100 episodes')
    print('=' * 74)
    ws = np.array([rows[k]['ws'] for k in keys])
    lm = np.array([rows[k]['letkf_match'] for k in keys])
    la = np.array([rows[k]['letkf_all'] for k in keys])
    zm = np.array([rows[k]['zero_match'] for k in keys])

    def report(label, x, y, nx, ny):
        kx, ky, n, p = sign_test(x, y)
        print(f'  {label}')
        print(f'    median {nx}={np.median(x):.4f}   {ny}={np.median(y):.4f}   '
              f'delta={np.median(x)-np.median(y):+.4f}')
        print(f'    paired sign test: {kx} wins / {ky} losses (n={n})   p={p:.4g}')
        return dict(median_a=float(np.median(x)), median_b=float(np.median(y)),
                    wins=kx, losses=ky, n=n, p=p)

    res = {}
    res['ws_vs_letkf_match'] = report('[H-1a] MATCHED frame set (the repair)',
                                      ws, lm, 'WS', 'LETKF')
    res['ws_vs_letkf_all'] = report('[H-1b] published mismatch (LETKF on all frames)',
                                    ws, la, 'WS', 'LETKF_all')
    res['ws_vs_zero_match'] = report('[H-1c] WaveSplat vs zero-field floor, matched',
                                     ws, zm, 'WS', 'ZERO')
    res['letkf_match_vs_zero'] = report('[H-1d] LETKF vs zero-field floor, matched',
                                        lm, zm, 'LETKF', 'ZERO')

    # -------------------------------------------------------------- strata
    print('\n' + '=' * 74)
    print('H-2  pre-specified strata: ncomp <= 5  vs  ncomp >= 15   (Holm over 2)')
    print('=' * 74)
    strata = {'sparse_le5': [k for k in keys if rows[k]['ncomp'] <= 5],
              'crowded_ge15': [k for k in keys if rows[k]['ncomp'] >= 15]}
    ps, names = [], []
    for name, ks in strata.items():
        x = np.array([rows[k]['ws'] for k in ks])
        y = np.array([rows[k]['letkf_match'] for k in ks])
        r = report(f'[{name}]  n={len(ks)}', x, y, 'WS', 'LETKF')
        res[f'stratum_{name}'] = r
        ps.append(r['p']); names.append(name)
    adj = holm(ps)
    print('\n  Holm-adjusted:')
    for nm, p0, pa in zip(names, ps, adj):
        print(f'    {nm:<14s} p={p0:.4g} -> p_adj={pa:.4g}  '
              f'{"significant" if pa < ALPHA else "n.s."}')
        res[f'stratum_{nm}']['p_holm'] = float(pa)

    # -------------------------------------------------------------- timing
    lms = np.array([rows[k]['letkf_ms_median'] for k in keys])
    wms = np.array([pub[k]['WaveSplat']['ms'] for k in keys], float)
    print('\n' + '=' * 74)
    print('H-3  per-frame cost -- CROSS-PROCESS, NOT CITABLE as a cost comparison')
    print('     (WS ms from the published file, LETKF ms from this re-run)')
    print('=' * 74)
    print(f'  WaveSplat {np.median(wms):8.2f} ms   LETKF {np.median(lms):8.2f} ms   '
          f'ratio {np.median(lms)/np.median(wms):.2f}x   <- sanity check only')
    res['timing'] = dict(ws_ms=float(np.median(wms)), letkf_ms=float(np.median(lms)),
                         ratio=float(np.median(lms) / np.median(wms)),
                         CITABLE=False,
                         note='cross-process; use bs1_shipping_default.py instead')

    res['G0_pass'] = bool(g0)
    res['G1_pass'] = g1
    res['n_episodes'] = len(keys)
    res['per_episode'] = rows
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\n-> {OUT}')


if __name__ == '__main__':
    main()
