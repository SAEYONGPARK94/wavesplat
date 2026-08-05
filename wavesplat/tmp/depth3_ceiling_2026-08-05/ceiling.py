"""Depth-3: resolve A-vs-B, then measure the architecture ceiling directly.

Our generator's rule (wavesim/wave_generator.py:161-222):
    omegas = linspace(0.5*wp, 3*wp, N+1);  one component per bin;
    directions ~ N(mu, spread) drawn i.i.d., ONE direction per bin.
=> within-bin directional spread is structurally ZERO for every N.

Applying that rule faithfully to a real record means: keep the real energy in
each bin, but collapse the real within-bin directional distribution D(theta|f)
to a single direction (its mean).  The N -> infinity limit of that is the
"across-frequency" term.  That is the ceiling this script measures.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FMIN, FMAX = 0.04, 0.50
STATIONS = ['028p1', '094p1', '142p1', '194p1', '243p1', '278p1']
NS = (1, 2, 5, 15, 30, 60)
REAL_MIN_SPREAD = 12.180936037597037


def s1(r1):
    return np.degrees(np.sqrt(np.clip(2.0 * (1.0 - np.clip(r1, 0, 1)), 0, None)))


def load(st):
    z = np.load(os.path.join(HERE, f'spec_{st}.npz'))
    f, E, a1, b1 = z['f'], z['E'].copy(), z['a1'].copy(), z['b1'].copy()
    Hs, Tp, fl = z['Hs'], z['Tp'], z['flag']
    for arr in (E, a1, b1):
        arr[arr <= -999] = np.nan
    Hs = np.where(Hs <= -999, np.nan, Hs)
    Tp = np.where(Tp <= -999, np.nan, Tp)
    good = (fl == 1) & np.isfinite(Hs) & np.isfinite(Tp) & (Tp > 0)
    return f, E[good], a1[good], b1[good], Hs[good], Tp[good]


rows = []
for st in STATIONS:
    f, E, a1, b1, Hs, Tp = load(st)
    band = (f >= FMIN) & (f <= FMAX)
    r1f = np.sqrt(np.clip(a1 ** 2 + b1 ** 2, 0, 1))
    th = np.arctan2(b1, a1)
    W = np.where(band[None, :] & np.isfinite(E) & (E > 0) & np.isfinite(r1f), E, 0.0)

    for i in range(E.shape[0]):
        w = W[i]
        tot = w.sum()
        if tot <= 0:
            continue
        w = w / tot
        rr, tt = np.nan_to_num(r1f[i]), np.nan_to_num(th[i])

        # (1) FULL continuous representation  -- what CDIP's own spectrum says
        A1 = (w * rr * np.cos(tt)).sum()
        B1 = (w * rr * np.sin(tt)).sum()
        r_tot = np.hypot(A1, B1)

        # (2) ACROSS-frequency: collapse every bin to its mean direction (our
        #     architecture's N->infinity limit), keep the real bin energies
        Ax = (w * np.cos(tt)).sum()
        Bx = (w * np.sin(tt)).sum()
        r_across = np.hypot(Ax, Bx)

        # (3) WITHIN-frequency: energy-weighted mean resultant length inside bins
        r_within = (w * rr).sum()

        # (4) top-N energy bins, deltaized  (measurement A's ladder)
        order = np.argsort(-w)
        topn = {}
        for n in NS:
            idx = order[:n]
            ww = w[idx]
            if ww.sum() <= 0:
                topn[n] = np.nan
                continue
            ww = ww / ww.sum()
            topn[n] = float(s1(np.hypot((ww * np.cos(tt[idx])).sum(),
                                        (ww * np.sin(tt[idx])).sum())))

        # (5) OUR GENERATOR'S OWN BIN GRID applied to this record:
        #     N bins tiling [0.5*wp, 3*wp]; per bin aggregate the real moments,
        #     collapse to the bin's mean direction, weight by real bin energy.
        wp = 2 * np.pi / Tp[i]
        gen = {}
        for n in NS:
            edges = np.linspace(0.5 * wp, 3.0 * wp, n + 1) / (2 * np.pi)   # Hz
            cx = cy = 0.0
            acc = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (f >= lo) & (f < hi) & (w > 0)
                if not m.any():
                    continue
                e = w[m].sum()
                # bin mean direction from the real moments inside the bin
                bx = (w[m] * rr[m] * np.cos(tt[m])).sum()
                by = (w[m] * rr[m] * np.sin(tt[m])).sum()
                if bx == 0 and by == 0:
                    continue
                ang = np.arctan2(by, bx)
                acc.append((e, ang))
            if not acc:
                gen[n] = np.nan
                continue
            ee = np.array([a[0] for a in acc]); aa = np.array([a[1] for a in acc])
            ee = ee / ee.sum()
            gen[n] = float(s1(np.hypot((ee * np.cos(aa)).sum(), (ee * np.sin(aa)).sum())))

        rows.append(dict(station=st, Hs=float(Hs[i]), Tp=float(Tp[i]),
                         s_total=float(s1(r_tot)), s_across=float(s1(r_across)),
                         s_within_avg=float((w * s1(rr)).sum()),
                         r_tot=float(r_tot), r_across=float(r_across),
                         r_within=float(r_within),
                         topn={str(k): v for k, v in topn.items()},
                         gen={str(k): v for k, v in gen.items()}))

print(f'records = {len(rows)}')
S = {k: np.array([r[k] for r in rows]) for k in
     ('s_total', 's_across', 's_within_avg', 'r_tot', 'r_across', 'r_within')}
ST = np.array([r['station'] for r in rows])

out = {'n_records': len(rows), 'stations': STATIONS,
       'band_hz': [FMIN, FMAX], 'real_min_spread_deg': REAL_MIN_SPREAD}


def med(x, m=None):
    return float(np.nanmedian(x if m is None else x[m]))


# ---- headline medians (all 6 stations, and 028p1 alone for A-comparability)
for tag, m in (('all6', np.ones(len(rows), bool)), ('028p1', ST == '028p1')):
    out[tag] = {
        'n': int(m.sum()),
        'sigma1_total_deg': med(S['s_total'], m),
        'sigma1_across_deg__ARCHITECTURE_CEILING': med(S['s_across'], m),
        'sigma1_within_avg_deg': med(S['s_within_avg'], m),
        'gap_total_minus_across_deg': med(S['s_total'], m) - med(S['s_across'], m),
        'r_tot': med(S['r_tot'], m), 'r_across': med(S['r_across'], m),
        'r_within': med(S['r_within'], m),
        'topn_median': {str(n): med(np.array([r['topn'][str(n)] for r in rows]), m) for n in NS},
        'gen_grid_median': {str(n): med(np.array([r['gen'][str(n)] for r in rows]), m) for n in NS},
        'frac_across_below_real_min': float((S['s_across'][m] < REAL_MIN_SPREAD).mean()),
        'frac_total_below_real_min': float((S['s_total'][m] < REAL_MIN_SPREAD).mean()),
    }

# ---- factorisation r_tot ?= r_within * r_across
prod = S['r_within'] * S['r_across']
out['factorisation'] = {
    'median_r_tot': float(np.nanmedian(S['r_tot'])),
    'median_r_within_times_r_across': float(np.nanmedian(prod)),
    'median_abs_residual': float(np.nanmedian(np.abs(S['r_tot'] - prod))),
    'p95_abs_residual': float(np.nanpercentile(np.abs(S['r_tot'] - prod), 95)),
    'median_rel_residual_pct': float(np.nanmedian(np.abs(S['r_tot'] - prod) / S['r_tot']) * 100),
    'note': 'exact iff r1(f) is constant across f; residual quantifies the coupling',
}

# ---- per-station table
out['per_station'] = {
    st: {'n': int((ST == st).sum()),
         'sigma1_total': med(S['s_total'], ST == st),
         'sigma1_across_CEILING': med(S['s_across'], ST == st),
         'sigma1_within_avg': med(S['s_within_avg'], ST == st),
         'frac_records_across_ge_total': float((S['s_across'][ST == st] >= S['s_total'][ST == st]).mean())}
    for st in STATIONS}

# ---- how often does the ceiling actually bind?
out['ceiling_binds'] = {
    'frac_records_across_lt_total': float((S['s_across'] < S['s_total']).mean()),
    'median_shortfall_deg': float(np.nanmedian(S['s_total'] - S['s_across'])),
    'p05_shortfall_deg': float(np.nanpercentile(S['s_total'] - S['s_across'], 5)),
    'p95_shortfall_deg': float(np.nanpercentile(S['s_total'] - S['s_across'], 95)),
}

json.dump(out, open(os.path.join(HERE, 'ceiling.json'), 'w'), indent=2)
np.savez_compressed(os.path.join(HERE, 'per_record.npz'),
                    station=ST, **{k: v for k, v in S.items()},
                    **{f'gen{n}': np.array([r['gen'][str(n)] for r in rows]) for n in NS},
                    **{f'top{n}': np.array([r['topn'][str(n)] for r in rows]) for n in NS})
print(json.dumps({k: out[k] for k in ('all6', '028p1', 'factorisation', 'ceiling_binds')}, indent=2))
