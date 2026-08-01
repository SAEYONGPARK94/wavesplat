"""Physical inspection of the FRF lidar wave gauges (2019-10, 5 cross-shore gauges).

Question this answers, before any claim is made:
  1. What fraction of the nominal 3.5 Hz samples actually exist (not NaN)?
  2. How many hourly windows are usable, and by what definition?
  3. Are the 5 gauges simultaneous (shared clock / overlapping good windows)?
  4. Does the elevation signal look like real waves (spectrum, Hs consistency)?
  5. Is the array geometry what the catalog claims (collinear? spacing?)
Nothing here is a claim about the WaveSplat program; it is instrument triage.
"""
import json
import numpy as np
import h5py

GAUGES = ['080', '090', '100', '110', '140']
F = 'lidarWaveGauge{}_201910.nc'


def attrs(h, name):
    try:
        return {k: (v.decode() if isinstance(v, bytes) else v)
                for k, v in h[name].attrs.items()}
    except Exception:
        return {}


def scaled(d):
    """Apply CF scale_factor/add_offset/_FillValue -> float array with NaN."""
    v = np.asarray(d[...]).astype(float)
    a = d.attrs

    def s(key, dflt):
        if key not in a:
            return dflt
        x = np.asarray(a[key]).ravel()
        return float(x[0]) if x.size else dflt

    fv = s('_FillValue', None)
    mv = s('missing_value', None)
    for m in (fv, mv):
        if m is not None:
            v[v == m] = np.nan
    return v * s('scale_factor', 1.0) + s('add_offset', 0.0)


out = {}
per_gauge = {}
for g in GAUGES:
    with h5py.File(F.format(g), 'r') as h:
        keys = list(h.keys())
        t = scaled(h['time'])
        ts = scaled(h['tsTime'])
        W = scaled(h['waterLevelTimeSeries'])          # [hour, sample]
        hs = scaled(h['waveHs'])
        x = float(np.asarray(h['xFRF'][...]).ravel()[0])
        y = float(np.asarray(h['yFRF'][...]).ravel()[0])
        pmiss = scaled(h['percentTimeSeriesMissing']) if 'percentTimeSeriesMissing' in h else None
        gap = scaled(h['medianTimeSeriesGap']) if 'medianTimeSeriesGap' in h else None
        finite = np.isfinite(W)
        frac = finite.mean(axis=1)                     # per-hour valid fraction
        dt_ts = float(np.median(np.diff(ts))) if ts.size > 1 else np.nan
        per_gauge[g] = dict(
            xFRF=x, yFRF=y, n_hours=int(W.shape[0]), n_samp=int(W.shape[1]),
            ts_dt_s=round(dt_ts, 4), ts_rate_hz=round(1.0 / dt_ts, 3) if dt_ts else None,
            overall_valid_frac=round(float(finite.mean()), 4),
            hours_valid_ge50=int((frac >= 0.50).sum()),
            hours_valid_ge80=int((frac >= 0.80).sum()),
            hours_valid_ge95=int((frac >= 0.95).sum()),
            hs_median=round(float(np.nanmedian(hs)), 3),
            hs_p05_p95=[round(float(np.nanpercentile(hs, 5)), 3),
                        round(float(np.nanpercentile(hs, 95)), 3)],
            median_gap_s=round(float(np.nanmedian(gap)), 3) if gap is not None else None,
            keys=keys,
        )
        out.setdefault('_frac', {})[g] = frac
        out.setdefault('_W', {})[g] = W
        out.setdefault('_t', {})[g] = t
        out.setdefault('_hs', {})[g] = hs

# ---- geometry
geom = {g: (per_gauge[g]['xFRF'], per_gauge[g]['yFRF']) for g in GAUGES}
ys = np.array([geom[g][1] for g in GAUGES])
xs = np.array([geom[g][0] for g in GAUGES])
collinear = bool(np.ptp(ys) < 1.0)

# ---- simultaneity: identical hour stamps? joint good hours?
t_ref = out['_t'][GAUGES[0]]
same_clock = all(np.array_equal(out['_t'][g], t_ref) for g in GAUGES)
FR = np.vstack([out['_frac'][g] for g in GAUGES])       # [gauge, hour]
joint = {f'all5_ge{k}': int((FR >= k / 100).all(axis=0).sum()) for k in (50, 80, 95)}

# ---- does it look like waves? spectrum of the best hour on gauge 100
W100 = out['_W']['100']
f100 = out['_frac']['100']
best = int(np.argmax(f100))
w = W100[best].copy()
good = np.isfinite(w)
sig = {}
if good.sum() > 1024:
    idx = np.arange(w.size)
    w[~good] = np.interp(idx[~good], idx[good], w[good])   # gap-fill for FFT only
    w = w - w.mean()
    dt = per_gauge['100']['ts_dt_s']
    n = 1 << int(np.floor(np.log2(w.size)))
    seg = w[:n] * np.hanning(n)
    P = np.abs(np.fft.rfft(seg)) ** 2
    fr = np.fft.rfftfreq(n, dt)
    band = (fr > 0.03) & (fr < 0.5)
    fp = float(fr[band][np.argmax(P[band])])
    m0 = float(np.trapezoid(P[band], fr[band]) / (n * np.sum(np.hanning(n) ** 2) / n ** 2) / n)
    sig = dict(best_hour_index=best, best_hour_valid_frac=round(float(f100[best]), 4),
               peak_freq_hz=round(fp, 4), peak_period_s=round(1 / fp, 2),
               std_elev_m=round(float(np.nanstd(W100[best])), 4),
               hs_4std_m=round(4 * float(np.nanstd(W100[best])), 3),
               hs_reported_m=round(float(out['_hs']['100'][best]), 3))

# ---- cross-gauge coherence on the jointly-best hour (real waves propagate)
allgood = (FR >= 0.80).all(axis=0)
lagcorr = {}
if allgood.any():
    hbest = int(np.argmax(FR.min(axis=0) * allgood))
    ref = out['_W']['100'][hbest]
    for g in GAUGES:
        v = out['_W'][g][hbest]
        m = np.isfinite(ref) & np.isfinite(v)
        if m.sum() > 512:
            a = ref[m] - ref[m].mean()
            b = v[m] - v[m].mean()
            lagcorr[g] = round(float((a * b).mean() / (a.std() * b.std())), 3)
    lagcorr['_hour_index'] = hbest

res = dict(per_gauge=per_gauge, geometry=dict(xFRF=list(xs), yFRF=list(ys),
           collinear_in_y=collinear, x_span_m=float(np.ptp(xs))),
           same_hour_clock=same_clock, joint_good_hours=joint,
           spectrum_check=sig, zero_lag_corr_vs_x100=lagcorr)
print(json.dumps(res, indent=1))
json.dump(res, open('inspection.json', 'w'), indent=1)
