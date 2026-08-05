"""Reverse-engineer WHAT procedure produced measurement B.

Target numbers to reproduce (CON, depth 2):
    N=1 0.00 (100.0% below 12.18)  N=2 17.22 (37.2%)  N=5 29.17 (5.6%)
    N=15 33.31 (0.3%)              N=30 34.41 (0.1%)   real original 35.35

Hypothesis H_B: B never touched the real D(theta,f).  It took the per-record
SCALAR sigma1 (spread_total_deg, the only directional quantity stored in
cdip_support_raw.json) and re-injected it as the *population* spread parameter
of our generator's own rule  directions ~ N(mu, sigma),  then recomputed the
finite-N sample sigma1 with JONSWAP energy weights.  If so, B is a finite-N
ESTIMATOR-BIAS replay, not a discretisation of a real directional spectrum.
"""
import json
import os
import sys

import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
OUT = os.path.join(ROOT, 'tmp/depth3_ceiling_2026-08-05')
RAW = os.path.join(ROOT, 'tmp/cdip_support_2026-08-03/cdip_support_raw.json')
FMIN, FMAX = 0.04, 0.50
TARGET = {1: 0.00, 2: 17.22, 5: 29.17, 15: 33.31, 30: 34.41}
TARGET_FRAC = {1: 1.000, 2: 0.372, 5: 0.056, 15: 0.003, 30: 0.001}


def sigma1_deg(a1, b1):
    r1 = np.sqrt(np.clip(a1 ** 2 + b1 ** 2, 0, 1))
    return np.degrees(np.sqrt(np.clip(2.0 * (1.0 - r1), 0, None)))


def jonswap_weights(n, Tp, gamma=3.3, band=True):
    """Exactly the energy weights our generator produces (wave_generator.py)."""
    wp = 2 * np.pi / Tp
    w = np.linspace(0.5 * wp, 3.0 * wp, n + 1)
    wc = 0.5 * (w[:-1] + w[1:])
    dw = np.diff(w)
    sig = np.where(wc <= wp, 0.07, 0.09)
    aj = np.exp(-0.5 * ((wc / wp - 1) / sig) ** 2)
    S = (0.3125 * wp ** 4 / wc ** 5 * np.exp(-1.25 * (wp / wc) ** 4) * gamma ** aj)
    E = S * dw                      # proportional to A^2/2
    fr = wc / (2 * np.pi)
    if band:
        m = (fr >= FMIN) & (fr <= FMAX)
        if not m.any():
            m = np.ones_like(fr, bool)
        E = np.where(m, E, 0.0)
    return E / E.sum()


d = json.load(open(RAW))
Rsig = np.concatenate([np.asarray(r['spread_total_deg'], float) for r in d['real']])
Rtp = np.concatenate([np.asarray(r['Tp'], float) for r in d['real']])
ok = np.isfinite(Rsig) & np.isfinite(Rtp) & (Rtp > 0)
Rsig, Rtp = Rsig[ok], Rtp[ok]
print(f'records = {Rsig.size}   median sigma1 = {np.median(Rsig):.4f}')

rng = np.random.default_rng(20260805)
NS = (1, 2, 5, 15, 30)

results = {}
for mode in ('sigma_as_std', 'invert_r1', 'sigma_as_std_equalw', 'invert_r1_equalw'):
    invert = mode.startswith('invert')
    equalw = mode.endswith('equalw')
    med, frac = {}, {}
    for n in NS:
        # weights depend on Tp only through the band mask; cache per unique Tp is
        # overkill -- just compute per record (cheap: n<=30)
        vals = np.empty(Rsig.size)
        for i in range(Rsig.size):
            s_deg = Rsig[i]
            if invert:
                r1 = max(0.0, 1.0 - np.deg2rad(s_deg) ** 2 / 2.0)
                s_rad = np.sqrt(max(-2.0 * np.log(max(r1, 1e-12)), 0.0))
            else:
                s_rad = np.deg2rad(s_deg)
            th = rng.normal(0.0, s_rad, n)
            E = np.full(n, 1.0 / n) if equalw else jonswap_weights(n, Rtp[i])
            a1 = (E * np.cos(th)).sum() / E.sum()
            b1 = (E * np.sin(th)).sum() / E.sum()
            vals[i] = sigma1_deg(np.array([a1]), np.array([b1]))[0]
        med[n] = float(np.median(vals))
        frac[n] = float((vals < 12.180936037597037).mean())
    results[mode] = {'median': med, 'frac_below_1218': frac}
    err = np.mean([abs(med[n] - TARGET[n]) for n in NS])
    print(f'{mode:24s} med={[round(med[n],2) for n in NS]}  '
          f'frac={[round(frac[n],3) for n in NS]}  MAE_vs_B={err:.3f}')

json.dump({'target_B': TARGET, 'target_B_frac': TARGET_FRAC,
           'n_records': int(Rsig.size),
           'real_median_sigma1_total': float(np.median(Rsig)),
           'replays': results},
          open(os.path.join(OUT, 'reverse_engineer_B.json'), 'w'), indent=2)
