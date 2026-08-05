"""INDEPENDENT re-derivation of the CDIP support claim (2026-08-04 audit).

Written from scratch against the HDF5 raw data; the only thing taken from the
2026-08-03 run is the CDIP per-record Hs/Tp/spread list (network fetch), which
is separately spot-checked against a fresh THREDDS pull in redo_cdip_fetch.py.
"""
import json
import os

import h5py
import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
DATA = os.path.join(ROOT, 'data_finemesh_2026-07-23')
RAW = os.path.join(ROOT, 'tmp/cdip_support_2026-08-03/cdip_support_raw.json')
FMIN, FMAX = 0.04, 0.50


def sigma1_deg(a1, b1):
    r1 = np.sqrt(np.clip(a1 ** 2 + b1 ** 2, 0, 1))
    return np.degrees(np.sqrt(np.clip(2.0 * (1.0 - r1), 0, None)))


# ---------------------------------------------------------------- synthetic
rows = []
for fn in sorted(os.listdir(DATA)):
    if not fn.endswith('.h5'):
        continue
    with h5py.File(os.path.join(DATA, fn), 'r') as f:
        wc = f['config/wave_components'][:]
        att = dict(f['config'].attrs)
    A, om, kx, ky = wc[:, 0], wc[:, 1], wc[:, 2], wc[:, 3]
    fr = om / (2 * np.pi)
    th = np.arctan2(ky, kx)
    E = 0.5 * A ** 2
    band = (fr >= FMIN) & (fr <= FMAX)
    e = E[band].sum()
    a1 = (E[band] * np.cos(th[band])).sum() / e
    b1 = (E[band] * np.sin(th[band])).sum() / e
    # circular std of the component directions themselves (a 2nd view)
    rows.append(dict(episode=fn, n=int(att['n_components']), Hs=float(att['Hs']),
                     Tp=float(att['Tp']), sea=str(att['sea_state']),
                     n_in_band=int(band.sum()),
                     sigma1=float(sigma1_deg(np.array([a1]), np.array([b1]))[0]),
                     dir_std_deg=float(np.degrees(np.std(th[band]))) if band.sum() > 1 else 0.0))
S = {k: np.array([r[k] for r in rows]) for k in ('n', 'Hs', 'Tp', 'sigma1', 'n_in_band', 'dir_std_deg')}

# ---------------------------------------------------------------- real
d = json.load(open(RAW))
R = {k: np.concatenate([np.asarray(r[k], float) for r in d['real']])
     for k in ('Hs', 'Tp', 'spread_band_deg', 'spread_total_deg')}
Rsig = R['spread_total_deg']

out = {}
out['n_synth'] = len(rows)
out['n_real'] = int(Rsig.size)
out['n_stations'] = len(d['real'])
out['real_sigma1_min'] = float(np.nanmin(Rsig))
out['real_sigma1_p05'] = float(np.nanpercentile(Rsig, 5))
out['real_sigma1_median'] = float(np.nanmedian(Rsig))
out['synth_below_real_min'] = int((S['sigma1'] < np.nanmin(Rsig)).sum())
out['synth_below_real_min_by_n'] = {int(n): int(((S['n'] == n) & (S['sigma1'] < np.nanmin(Rsig))).sum())
                                    for n in sorted(set(S['n'].tolist()))}
out['synth_count_by_n'] = {int(n): int((S['n'] == n).sum()) for n in sorted(set(S['n'].tolist()))}
out['synth_sigma1_median_by_n'] = {int(n): float(np.median(S['sigma1'][S['n'] == n]))
                                   for n in sorted(set(S['n'].tolist()))}
# percentile of the 30-comp median inside real
m30 = float(np.median(S['sigma1'][S['n'] == 30]))
out['synth_n30_median_deg'] = m30
out['synth_n30_median_pct_in_real'] = float((Rsig < m30).mean() * 100)

# Hs/Tp matched subset of real
lo_h, hi_h = S['Hs'].min(), S['Hs'].max()
lo_t, hi_t = S['Tp'].min(), S['Tp'].max()
msk = (R['Hs'] >= lo_h) & (R['Hs'] <= hi_h) & (R['Tp'] >= lo_t) & (R['Tp'] <= hi_t)
out['synth_Hs_box'] = [float(lo_h), float(hi_h)]
out['synth_Tp_box'] = [float(lo_t), float(hi_t)]
out['real_median_sigma1_all'] = float(np.nanmedian(Rsig))
out['real_median_sigma1_HsTp_matched'] = float(np.nanmedian(Rsig[msk]))
out['real_n_HsTp_matched'] = int(msk.sum())
out['real_frac_inside_Hs_box'] = float(((R['Hs'] >= lo_h) & (R['Hs'] <= hi_h)).mean())
out['real_frac_inside_Tp_box'] = float(((R['Tp'] >= lo_t) & (R['Tp'] <= hi_t)).mean())
out['real_Tp_median'] = float(np.nanmedian(R['Tp']))
out['real_Tp_frac_above_synth_Tp_max'] = float((R['Tp'] > hi_t).mean())

# ------------------------------------------- discrete-sampling bias, measured
# The generator ALWAYS draws directions ~ N(mu, 30 deg) (wave_generator.py:222,
# never overridden by pipeline.py:76).  So the *population* spread is fixed;
# only the finite-N estimate varies.  Measure that estimator's bias directly by
# replaying the generator's own component construction.
import sys
sys.path.insert(0, ROOT)
from wavesim.wave_generator import jonswap_components  # noqa: E402

SIGMA_POP = np.degrees(np.sqrt(2 * (1 - np.exp(-np.deg2rad(30.0) ** 2 / 2))))
out['population_sigma1_of_generator_deg'] = float(SIGMA_POP)
out['population_sigma1_pct_in_real'] = float((Rsig < SIGMA_POP).mean() * 100)

boot = {}
for n in (1, 2, 5, 15, 30):
    vals = []
    for s in range(400):
        c = jonswap_components(Hs=1.0, Tp=7.0, n_components=n, depth=50.0,
                               seed=10_000 + s)
        A = np.array([x.A for x in c]); om = np.array([x.omega for x in c])
        th = np.arctan2([x.ky for x in c], [x.kx for x in c])
        fr = om / (2 * np.pi); E = 0.5 * A ** 2
        b = (fr >= FMIN) & (fr <= FMAX)
        if not b.any():
            continue
        e = E[b].sum()
        a1 = (E[b] * np.cos(th[b])).sum() / e
        b1 = (E[b] * np.sin(th[b])).sum() / e
        vals.append(float(sigma1_deg(np.array([a1]), np.array([b1]))[0]))
    v = np.array(vals)
    boot[n] = dict(n_draws=len(v), median=float(np.median(v)),
                   p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
                   frac_below_real_min=float((v < np.nanmin(Rsig)).mean()))
out['finiteN_estimator_replay'] = boot

print(json.dumps(out, indent=2))
json.dump(out, open(os.path.join(ROOT, 'tmp/audit_2026-08-04/cdip_redo.json'), 'w'), indent=2)
