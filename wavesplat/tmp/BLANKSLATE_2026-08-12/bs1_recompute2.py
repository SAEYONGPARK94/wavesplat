#!/usr/bin/env python3
"""BS1 part 2:
  S7  ncomp x BF cross-tab on the 100-episode set (the boundary)
  S8  all remaining paired baselines in results/*.json, incl. avg_nRMSE layouts
  S9  independence audit of the synthetic episode set (raw HDF5)
"""
import json, os, glob, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, 'results')
DATA = os.path.join(ROOT, 'data')
sys.path.insert(0, HERE)
from bs1_recompute import sign_test, summarise_pair, boot_ci, parse_key, holm  # noqa


def load_floor():
    return json.load(open(os.path.join(HERE, 'floor_zero_predictor.json')))


# ────────────────────────── S7 cross-tab ───────────────────────────────────
def s7_crosstab():
    print('\n' + '=' * 78)
    print('S7  ncomp x BF CROSS-TAB -- results/100ep_ws_letkf.json + raw-HDF5 floor')
    print('    estimator: shipping SequentialUpdater, init_method="v2_konly"')
    print('               (NOT the shipping default "auto" -- hardcoded at')
    print('                scripts/run_100ep_letkf_and_ws_metrics.py:39)')
    print('    aggregation: cell MEDIAN of per-episode avg_nRMSE; exact sign test')
    print('=' * 78)
    d = json.load(open(os.path.join(RES, '100ep_ws_letkf.json')))
    fl = load_floor()
    keys = sorted(d)
    cells = defaultdict(list)
    for k in keys:
        m = parse_key(k)
        cells[(m['bf'], m['ncomp'])].append(k)
    print(f'\n  {"cell":<14s} {"n":>3s} {"WS":>7s} {"LETKF":>7s} {"ZERO":>7s} '
          f'{"WS<ZERO":>9s} {"p":>9s}  {"WS vs ZERO %":>13s}')
    rows = []
    for (bf, nc) in sorted(cells):
        ks = cells[(bf, nc)]
        ws = np.array([d[k]['WaveSplat']['nRMSE'] for k in ks])
        le = np.array([d[k]['LETKF']['nRMSE'] for k in ks])
        zr = np.array([fl[k]['zero'] for k in ks])
        kk, nn, pp = sign_test(ws, zr)
        rel = (np.median(ws) - np.median(zr)) / np.median(zr) * 100
        print(f'  BF{bf}_{nc}comp{"":<4s} {len(ks):>3d} {np.median(ws):7.4f} '
              f'{np.median(le):7.4f} {np.median(zr):7.4f} {kk:4d}/{nn:<4d} '
              f'{pp:9.4f} {rel:+12.1f}%')
        rows.append((bf, nc, len(ks), np.median(ws), np.median(le),
                     np.median(zr), kk, nn, pp, rel))
    # Holm over the 10 cells
    adj = holm(np.array([r[8] for r in rows]))
    print('\n  Holm-adjusted across the 10 cells (WS vs ZERO):')
    for r, a in zip(rows, adj):
        verdict = ('WS BEATS flat sea' if r[6] > r[7] / 2 and a < 0.05 else
                   'WS LOSES to flat sea' if r[6] < r[7] / 2 and a < 0.05 else
                   'null')
        print(f'    BF{r[0]}_{r[1]}comp  p_holm={a:.4g}  -> {verdict}')

    # the boundary: pool sparse (ncomp<=5) vs crowded (ncomp>=15)
    print('\n  [S7.2] the boundary, pooled')
    for label, sel in [('SPARSE ncomp<=5', lambda m: m['ncomp'] <= 5),
                       ('CROWDED ncomp>=15', lambda m: m['ncomp'] >= 15)]:
        ks = [k for k in keys if sel(parse_key(k))]
        ws = np.array([d[k]['WaveSplat']['nRMSE'] for k in ks])
        le = np.array([d[k]['LETKF']['nRMSE'] for k in ks])
        zr = np.array([fl[k]['zero'] for k in ks])
        print(f'\n   {label}  (n={len(ks)})')
        summarise_pair('WaveSplat', 'LETKF', ws, le)
        summarise_pair('WaveSplat', 'ZERO-FIELD', ws, zr)
        summarise_pair('LETKF', 'ZERO-FIELD', le, zr)
        ss = np.array([d[k]['WaveSplat']['SS'] for k in ks])
        rho = np.array([d[k]['WaveSplat']['rho'] for k in ks])
        print(f'      WS Skill_Score>0 in {int((ss>0).sum())}/{len(ks)}; '
              f'median rho={np.median(rho):.3f}')

    # BF2 anomaly: sparse but losing
    print('\n  [S7.3] BF2_1comp anomaly (sparse yet WS loses)')
    ks = [k for k in keys if parse_key(k)['bf'] == 2]
    ws = np.array([d[k]['WaveSplat']['nRMSE'] for k in ks])
    zr = np.array([fl[k]['zero'] for k in ks])
    print(f'    Hs=0.3 m (BF2).  WS median {np.median(ws):.4f} vs floor '
          f'{np.median(zr):.4f}. per-episode WS: '
          f'{np.round(np.sort(ws),3).tolist()}')


# ─────────────────── S8 remaining paired baselines ─────────────────────────
def _nrm(e):
    if not isinstance(e, dict):
        return np.nan
    for k in ('nRMSE', 'avg_nRMSE', 'nrmse', 'rmse', 'RMSE'):
        if k in e and e[k] is not None:
            return float(e[k])
    return np.nan


def _ms(e):
    if not isinstance(e, dict):
        return np.nan
    for k in ('ms', 'avg_inference_ms', 'elapsed_ms'):
        if k in e and e[k] is not None:
            return float(e[k])
    return np.nan


def s8_others():
    print('\n' + '=' * 78)
    print('S8  ALL OTHER PAIRED BASELINE FILES in results/')
    print('    aggregation: per-episode value as stored; across episodes MEDIAN +')
    print('                 exact paired sign test.  n is tiny in most of these.')
    print('=' * 78)
    fl = load_floor()

    def floor_for(k):
        # try to map an episode key to the raw-HDF5 floor
        if k in fl:
            return fl[k]['zero']
        # aggregate keys like 'BF4_5comp' -> median over its seeds/raydrops
        v = [fl[q]['zero'] for q in fl if q.startswith(k + '_')]
        return float(np.median(v)) if v else np.nan

    files = sorted(glob.glob(os.path.join(RES, '*.json')))
    for p in files:
        fn = os.path.basename(p)
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        # detect episode -> method -> metric
        methods = defaultdict(int)
        eps = []
        for k, v in d.items():
            if not isinstance(v, dict):
                continue
            sub = [m for m, e in v.items()
                   if isinstance(e, dict) and not np.isnan(_nrm(e))]
            if len(sub) >= 2:
                eps.append(k)
                for m in sub:
                    methods[m] += 1
        if len(eps) < 2 or len(methods) < 2:
            continue
        names = [m for m in methods if methods[m] == len(eps)]
        if len(names) < 2:
            names = sorted(methods, key=lambda m: -methods[m])[:4]
        print(f'\n  --- {fn}   n_episodes={len(eps)}  methods={names}')
        vals = {m: np.array([_nrm(d[k].get(m, {})) for k in eps]) for m in names}
        zf = np.array([floor_for(k) for k in eps])
        # pick the WaveSplat-ish arm as reference
        ref = next((m for m in names if 'ave' in m or m.startswith('WS')), names[0])
        for m in names:
            if m == ref:
                continue
            summarise_pair(ref, m, vals[ref], vals[m])
        if np.isfinite(zf).sum() >= 2:
            for m in names:
                summarise_pair(m, 'ZERO-FIELD', vals[m], zf)
        else:
            print('    [floor unavailable for these episode keys]')


# ─────────────────── S9 independence audit (raw HDF5) ──────────────────────
def s9_independence():
    print('\n' + '=' * 78)
    print('S9  INDEPENDENCE AUDIT of the synthetic episode set (raw HDF5)')
    print('    question: how many statistically distinct seas do the 100 episodes')
    print('              actually contain?  This caps every p-value above.')
    print('=' * 78)
    import h5py
    sig = defaultdict(list)
    full = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(DATA, 'episode_*.h5'))):
        base = os.path.basename(f)[len('episode_'):-3]
        with h5py.File(f, 'r') as h:
            wc = h['config/wave_components'][:]
        A = np.round(wc[:, 0], 10)
        om = np.round(wc[:, 1], 10)
        kx = np.round(wc[:, 2], 10)
        ky = np.round(wc[:, 3], 10)
        ph = np.round(wc[:, 4], 10)
        sig[(A.tobytes(), om.tobytes())].append(base)
        full[(A.tobytes(), om.tobytes(), kx.tobytes(), ky.tobytes(),
              ph.tobytes())].append(base)
    print(f'  episodes scanned                    : '
          f'{sum(len(v) for v in sig.values())}')
    print(f'  DISTINCT (amplitude, omega) vectors : {len(sig)}')
    print(f'  DISTINCT full component sets        : {len(full)}')
    print('  -> amplitude+omega are a deterministic function of (Hs,Tp,ncomp);')
    print('     seeds re-draw only phases and directions.  Episodes sharing an')
    print('     (A,omega) vector are NOT independent draws of a sea spectrum.')
    sizes = sorted((len(v) for v in sig.values()), reverse=True)
    print(f'  cluster sizes: {sizes}')
    # uniform/slope pairs share everything but the raydrop mask
    dup = [v for v in full.values() if len(v) > 1]
    print(f'  full-component groups with >1 episode: {len(dup)} '
          f'(these differ ONLY in the raydrop mask, i.e. the SAME sea)')
    if dup:
        print(f'    example: {sorted(dup[0])}')


if __name__ == '__main__':
    s7_crosstab()
    s9_independence()
    s8_others()
