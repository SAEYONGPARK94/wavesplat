#!/usr/bin/env python3
"""
BS1 blank-slate recomputation of SYNTHETIC accuracy claims.

Rule: nothing here is read from a .tex / RESULT.md conclusion.  Every number is
recomputed from raw artefacts:
  * results/*.json   (per-episode metric records written by the run scripts)
  * data/episode_*.h5 (raw ground-truth fields -> trivial-predictor floors)

Sections
  S0  geometry census of the episode set
  S1  trivial-predictor floor (zero field / best constant) from raw HDF5
  S2  WaveSplat vs LETKF, 100 episodes, paired
  S3  WaveSplat vs the floor, paired
  S4  stratification: ncomp / BF / raydrop / seed
  S5  other paired baselines found in results/*.json
  S6  timing

Aggregation convention is printed for every table.
"""
import json
import os
import sys
import math
import glob
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, 'results')
DATA = os.path.join(ROOT, 'data')


# ───────────────────────── statistics helpers ──────────────────────────────
def sign_test(a, b, alternative='two-sided'):
    """Exact paired sign test.  Returns (n_a_lower, n_effective, p).

    a, b are paired arrays of a LOWER-IS-BETTER metric.  n_a_lower counts
    episodes where a < b.  Ties are dropped (standard).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    nz = d != 0
    n = int(nz.sum())
    k = int((d[nz] < 0).sum())          # a better than b
    if n == 0:
        return k, n, 1.0
    # exact binomial
    from math import comb
    def tail_ge(x):
        return sum(comb(n, i) for i in range(x, n + 1)) / 2.0 ** n
    def tail_le(x):
        return sum(comb(n, i) for i in range(0, x + 1)) / 2.0 ** n
    if alternative == 'two-sided':
        p = min(1.0, 2 * min(tail_ge(k), tail_le(k)))
    elif alternative == 'less':      # a better
        p = tail_ge(k)
    else:
        p = tail_le(k)
    return k, n, float(p)


def wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon as w
        a = np.asarray(a, float); b = np.asarray(b, float)
        ok = np.isfinite(a) & np.isfinite(b)
        if (a[ok] - b[ok]).any():
            return float(w(a[ok], b[ok]).pvalue)
    except Exception:
        pass
    return float('nan')


def boot_ci(x, n_boot=20000, seed=0, stat=np.median):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    s = stat(x[idx], axis=1)
    return (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))


def summarise_pair(name_a, name_b, a, b, extra=''):
    """Print one row of the standard comparison table."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) == 0:
        print(f'  {name_a} vs {name_b}: no finite pairs')
        return None
    k, n, p = sign_test(a, b)
    d = a - b
    med_a, med_b = np.median(a), np.median(b)
    rel = (med_a - med_b) / med_b * 100 if med_b else float('nan')
    lo, hi = boot_ci(d)
    row = dict(a=name_a, b=name_b, n=len(a), wins=k, n_eff=n, p=p,
               p_wilcoxon=wilcoxon(a, b),
               med_a=med_a, med_b=med_b, rel_pct=rel,
               med_diff=float(np.median(d)), diff_ci=(lo, hi), extra=extra)
    print(f'  {name_a:<22s} vs {name_b:<16s} n={len(a):3d}  '
          f'wins={k:3d}/{n:<3d} p={p:.3g}  '
          f'med {med_a:.4f} vs {med_b:.4f} ({rel:+.1f}%)  '
          f'med_diff={np.median(d):+.4f} [{lo:+.4f},{hi:+.4f}] {extra}')
    return row


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(idx):
        v = (m - rank) * pvals[i]
        run = max(run, v)
        adj[i] = min(1.0, run)
    return adj


# ───────────────────────── S0 geometry census ──────────────────────────────
def parse_key(k):
    """BF4_5comp_seed0_uniform -> dict"""
    parts = k.split('_')
    return dict(bf=int(parts[0][2:]),
                ncomp=int(parts[1].replace('comp', '')),
                seed=int(parts[2].replace('seed', '')),
                raydrop=parts[3])


def s0_geometry():
    print('\n' + '=' * 78)
    print('S0  GEOMETRY CENSUS  (raw HDF5, no derived file trusted)')
    print('=' * 78)
    import h5py
    files = sorted(glob.glob(os.path.join(DATA, 'episode_*.h5')))
    rows = []
    for f in files:
        with h5py.File(f, 'r') as h:
            c = h['config']
            wc = h['config/wave_components'][:]
            kx, ky = wc[:, 2], wc[:, 3]
            theta = np.degrees(np.arctan2(ky, kx))
            # energy-weighted mean direction
            A = wc[:, 0]
            mdir = np.degrees(np.arctan2(np.sum(A ** 2 * np.sin(np.radians(theta))),
                                         np.sum(A ** 2 * np.cos(np.radians(theta)))))
            gx = h['frames/gt_grid_x'][:]
            rows.append(dict(file=os.path.basename(f),
                             Hs=float(c.attrs['Hs']), Tp=float(c.attrs['Tp']),
                             ncomp=int(c.attrs['n_components']),
                             sea=str(c.attrs['sea_state']),
                             depth=float(c.attrs['depth']),
                             mean_dir_deg=float(mdir),
                             dir_std_deg=float(np.std(theta)),
                             raydrop=str(h['raydrop_model'].attrs['model_type']),
                             n_frames=h['frames/timestamps'].shape[0],
                             gx0=float(gx[0]), gx1=float(gx[-1]),
                             n_beams=h['frames/lidar_points'].shape[1]))
    print(f'  episodes in data/ : {len(rows)}')
    md = np.array([r['mean_dir_deg'] for r in rows])
    print(f'  energy-weighted mean wave direction: '
          f'min={md.min():.2f} max={md.max():.2f} median={np.median(md):.2f} deg '
          f'(0 deg = +x = AWAY from the sensor at origin)')
    print(f'  |mean_dir| < 15 deg in {int((np.abs(md) < 15).sum())}/{len(rows)} episodes')
    ds = np.array([r['dir_std_deg'] for r in rows])
    print(f'  per-episode component direction std: median={np.median(ds):.1f} deg')
    print(f'  GT grid x-range: {rows[0]["gx0"]}..{rows[0]["gx1"]} m (sensor at x=0)')
    print(f'  frames/episode: {sorted(set(r["n_frames"] for r in rows))}')
    print(f'  lidar beams/frame (post-raydrop cap): '
          f'{sorted(set(r["n_beams"] for r in rows))}')
    return rows


# ─────────────── S1 trivial-predictor floor from raw HDF5 ──────────────────
def s1_floor(cache=None):
    """Recompute the metric floor a *content-free* predictor achieves.

    Uses exactly the shipping evaluation convention discovered in
    wavesplat/run.py: per-frame nRMSE = RMSE(pred, gt)/Hs over the FULL
    80x60 GT grid, averaged over TRACKING frames (t > 2*Tp).
    """
    print('\n' + '=' * 78)
    print('S1  TRIVIAL-PREDICTOR FLOOR  (raw HDF5 -> nRMSE of a zero field)')
    print('    convention: mean over tracking frames (t > 2*Tp) of RMSE/Hs, full grid')
    print('=' * 78)
    import h5py
    out = {}
    files = sorted(glob.glob(os.path.join(DATA, 'episode_*.h5')))
    for f in files:
        base = os.path.basename(f)[len('episode_'):-3]
        with h5py.File(f, 'r') as h:
            Hs = float(h['config'].attrs['Hs'])
            Tp = float(h['config'].attrs['Tp'])
            ts = h['frames/timestamps'][:]
            sel = ts > 2.0 * Tp
            gt = h['frames/gt_wave_field'][sel, :, :]
        # zero predictor
        nz = np.sqrt(np.mean(gt ** 2, axis=(1, 2))) / Hs
        # best per-frame constant (mean of the field) -- an oracle-constant
        mu = gt.mean(axis=(1, 2), keepdims=True)
        nc = np.sqrt(np.mean((gt - mu) ** 2, axis=(1, 2))) / Hs
        out[base] = dict(zero=float(nz.mean()), const_oracle=float(nc.mean()),
                         n_eval_frames=int(sel.sum()), Hs=Hs, Tp=Tp)
    z = np.array([v['zero'] for v in out.values()])
    print(f'  n episodes = {len(out)}')
    print(f'  zero-field nRMSE: median={np.median(z):.4f} '
          f'mean={z.mean():.4f} min={z.min():.4f} max={z.max():.4f}')
    print('  (theory: for a linear sea Hs=4*sigma -> 0.250)')
    by = defaultdict(list)
    for k, v in out.items():
        by[parse_key(k)['ncomp']].append(v['zero'])
    for nc_ in sorted(by):
        print(f'    ncomp={nc_:<3d} n={len(by[nc_]):3d}  zero-nRMSE median='
              f'{np.median(by[nc_]):.4f}')
    return out


# ─────────────────────── S2/S3/S4 the 100-episode set ──────────────────────
def s2_ws_letkf(floor):
    print('\n' + '=' * 78)
    print('S2  WaveSplat vs LETKF  -- results/100ep_ws_letkf.json')
    print('    estimator: shipping SequentialUpdater (wavesplat/run.py)')
    print('    geometry : mean wave dir ~0 deg (away from sensor), depth 50 m,')
    print('               static sensor, GT grid x in [10,250] m')
    print('    aggregation: per-episode avg_nRMSE (already a mean over tracking')
    print('               frames); across episodes -> MEDIAN + paired sign test')
    print('=' * 78)
    d = json.load(open(os.path.join(RES, '100ep_ws_letkf.json')))
    keys = sorted(d)
    ws = np.array([d[k]['WaveSplat']['nRMSE'] for k in keys])
    le = np.array([d[k]['LETKF']['nRMSE'] for k in keys])
    zr = np.array([floor[k]['zero'] for k in keys])
    meta = [parse_key(k) for k in keys]

    print('\n  [S2.1] head-to-head, all 100 episodes pooled')
    rows = []
    rows.append(summarise_pair('WaveSplat', 'LETKF', ws, le))
    rows.append(summarise_pair('WaveSplat', 'ZERO-FIELD', ws, zr))
    rows.append(summarise_pair('LETKF', 'ZERO-FIELD', le, zr))

    # secondary metrics: do the winners actually reconstruct anything?
    print('\n  [S2.2] reconstruction-quality panel (medians over 100 episodes)')
    for m in ['rho', 'VR', 'SS', 'Dir_R2', 'Hs_err']:
        a = np.array([d[k]['WaveSplat'].get(m, np.nan) for k in keys], float)
        b = np.array([d[k]['LETKF'].get(m, np.nan) for k in keys], float)
        print(f'    {m:<8s} WS median={np.nanmedian(a):+.4f}   '
              f'LETKF median={np.nanmedian(b):+.4f}')
    print('    (rho = Pearson corr with the true field; SS = skill score vs zero-')
    print('     field climatology -> SS<=0 means NO skill over predicting flat sea)')
    ss = np.array([d[k]['WaveSplat']['SS'] for k in keys])
    print(f'    WaveSplat SS>0 in {int((ss > 0).sum())}/100 episodes')
    ssl = np.array([d[k]['LETKF']['SS'] for k in keys])
    print(f'    LETKF     SS>0 in {int((ssl > 0).sum())}/100 episodes')
    return d, keys, ws, le, zr, meta, rows


def s4_strat(keys, ws, le, zr, meta):
    print('\n' + '=' * 78)
    print('S4  STRATIFICATION -- where is the boundary?')
    print('    aggregation: within-cell MEDIAN, paired exact sign test, Holm across')
    print('                 the cells of each factor')
    print('=' * 78)
    for factor in ['ncomp', 'bf', 'raydrop', 'seed']:
        print(f'\n  --- by {factor} ---')
        levels = sorted(set(m[factor] for m in meta), key=str)
        ps_l, ps_z = [], []
        cells = []
        for lv in levels:
            idx = [i for i, m in enumerate(meta) if m[factor] == lv]
            a, b, z = ws[idx], le[idx], zr[idx]
            kl, nl, pl = sign_test(a, b)
            kz, nz, pz = sign_test(a, z)
            ps_l.append(pl); ps_z.append(pz)
            cells.append((lv, len(idx), np.median(a), np.median(b), np.median(z),
                          kl, nl, pl, kz, nz, pz))
        adj_l = holm(np.array(ps_l)); adj_z = holm(np.array(ps_z))
        print(f'    {factor:<8s} n   WS_med  LET_med ZERO_med | WS<LET   p_holm |'
              f' WS<ZERO   p_holm')
        for (lv, n_, ma, mb, mz, kl, nl, pl, kz, nz, pz), al, az in zip(cells, adj_l, adj_z):
            print(f'    {str(lv):<8s} {n_:<3d} {ma:.4f}  {mb:.4f}  {mz:.4f} |'
                  f' {kl:2d}/{nl:<3d} {al:8.3g} | {kz:2d}/{nz:<3d} {az:8.3g}')


def s6_timing(d, keys, meta):
    print('\n' + '=' * 78)
    print('S6  TIMING -- results/100ep_ws_letkf.json ms field (per tracking frame)')
    print('    hardware: NOT recorded inside the JSON.  See report for provenance.')
    print('=' * 78)
    wm = np.array([d[k]['WaveSplat']['ms'] for k in keys])
    lm = np.array([d[k]['LETKF']['ms'] for k in keys])
    print(f'  WaveSplat ms/frame: median={np.median(wm):.2f} '
          f'IQR=[{np.percentile(wm,25):.2f},{np.percentile(wm,75):.2f}] '
          f'max={wm.max():.2f}')
    print(f'  LETKF     ms/frame: median={np.median(lm):.2f} '
          f'IQR=[{np.percentile(lm,25):.2f},{np.percentile(lm,75):.2f}] '
          f'max={lm.max():.2f}')
    r = lm / wm
    print(f'  ratio LETKF/WS per-episode: median={np.median(r):.2f}x '
          f'[{np.percentile(r,25):.2f},{np.percentile(r,75):.2f}]')
    k, n, p = sign_test(wm, lm)
    print(f'  paired sign test WS faster: {k}/{n}, p={p:.3g}')
    print('\n  by ncomp (adaptive-N cost scaling):')
    for nc_ in sorted(set(m['ncomp'] for m in meta)):
        idx = [i for i, m in enumerate(meta) if m['ncomp'] == nc_]
        Ns = [d[keys[i]]['WaveSplat'].get('N', np.nan) for i in idx]
        print(f'    ncomp={nc_:<3d} WS ms median={np.median(wm[idx]):6.2f}  '
              f'LETKF ms median={np.median(lm[idx]):7.2f}  '
              f'WS N median={np.nanmedian(np.array(Ns,float)):.1f}')


# ─────────────────────── S5 other paired baselines ─────────────────────────
def s5_other(floor):
    print('\n' + '=' * 78)
    print('S5  OTHER PAIRED BASELINES in results/*.json')
    print('=' * 78)
    handlers = [
        ('gpr_comparison.json', _h_generic),
        ('p1_comparison.json', _h_generic),
        ('etkf_comparison.json', _h_generic),
        ('strong_baseline_comparison.json', _h_generic),
        ('b1_full_comparison.json', _h_generic),
        ('multiseed_comparison.json', _h_generic),
        ('pino_vs_wavesplat_comprehensive.json', _h_generic),
        ('raydrop_comparison.json', _h_generic),
        ('full_analysis.json', _h_generic),
        ('12ep_full_analysis.json', _h_generic),
    ]
    for fn, fn_h in handlers:
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            print(f'  [missing] {fn}')
            continue
        print(f'\n  --- {fn} ---')
        try:
            fn_h(p, floor)
        except Exception as e:
            print(f'    [handler error] {type(e).__name__}: {e}')


def _walk_numbers(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_numbers(v, f'{path}.{k}' if path else k)
    elif isinstance(obj, list):
        yield path, obj
    else:
        yield path, obj


def _h_generic(p, floor):
    d = json.load(open(p))
    # try to find dict-of-episode -> dict-of-method -> metric
    if not isinstance(d, dict):
        print(f'    top-level {type(d).__name__}, len={len(d)}')
        return
    ep_like = [k for k in d if isinstance(d[k], dict)]
    # detect method x metric layout
    methods = defaultdict(int)
    for k in ep_like:
        for m, v in d[k].items():
            if isinstance(v, dict) and any(
                    mm in v for mm in ('nRMSE', 'nrmse', 'rmse', 'RMSE')):
                methods[m] += 1
    if methods:
        print(f'    episode-keyed, n={len(ep_like)}, methods={dict(methods)}')
        names = sorted(methods, key=lambda m: -methods[m])
        vals = {}
        for m in names:
            v = []
            for k in ep_like:
                e = d[k].get(m, {})
                v.append(e.get('nRMSE', e.get('nrmse', e.get('rmse', np.nan)))
                         if isinstance(e, dict) else np.nan)
            vals[m] = np.array(v, float)
        base = names[0]
        for m in names[1:]:
            summarise_pair(base, m, vals[base], vals[m])
        return
    # aggregate-only?
    keys = list(d)[:12]
    print(f'    keys[:12]={keys}')
    print(f'    -> no per-episode method x metric layout detected by this handler')


def main():
    rows = s0_geometry()
    floor = s1_floor()
    json.dump(floor, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'floor_zero_predictor.json'), 'w'), indent=1)
    d, keys, ws, le, zr, meta, _ = s2_ws_letkf(floor)
    s4_strat(keys, ws, le, zr, meta)
    s6_timing(d, keys, meta)
    s5_other(floor)


if __name__ == '__main__':
    main()
