"""Place the measured WASS records on the CDIP distribution (25,426 sea states)."""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CDIP = os.path.join(ROOT, 'tmp/cdip_support_2026-08-03/cdip_support_raw.json')


def load_cdip():
    d = json.load(open(CDIP))
    g = lambda k: np.concatenate([np.array(s[k], float) for s in d['real']])
    out = {k: g(k) for k in ('Hs', 'Tp', 'spread_band_deg', 'spread_total_deg')}
    out['n_stations'] = len(d['real'])
    return out


def pct(arr, v):
    a = arr[np.isfinite(arr)]
    return float(100.0 * (a < v).mean()) if np.isfinite(v) else np.nan


def synth_cdip():
    d = json.load(open(CDIP))
    s = d['synthetic']
    return {k: np.array([r[k] for r in s], float)
            for k in ('Hs', 'Tp', 'spread_band_deg', 'spread_total_deg')}


def load_generic_calib():
    """Generic response table indexed by L/lambda (square grid, no mask)."""
    c = json.load(open(os.path.join(HERE, 'calibration.json')))['main']
    tab = {}
    for r in c:
        tab.setdefault(r['L_over_lam'], []).append(
            (r['sigma_true_deg'], r['sigma_meas_total']))
    for k in tab:
        tab[k] = np.array(sorted(tab[k]))
    return tab


def invert_generic(tab, L_over_lam, measured):
    """Invert at an arbitrary L/lambda by interpolating between table rows."""
    ks = np.array(sorted(tab))
    r = float(np.clip(L_over_lam, ks[0], ks[-1]))
    i = int(np.searchsorted(ks, r, 'right') - 1)
    i = min(max(i, 0), len(ks) - 2)
    lo, hi = ks[i], ks[i + 1]
    w = 0.0 if hi == lo else (np.log(r) - np.log(lo)) / (np.log(hi) - np.log(lo))
    A, B = tab[lo], tab[hi]
    xs = A[:, 0]
    ys = (1 - w) * A[:, 1] + w * B[:, 1]
    if not np.all(np.diff(ys) > 0):
        return np.nan
    if measured <= ys[0]:
        return 0.0
    if measured >= ys[-1]:
        return float(xs[-1])
    return float(np.interp(measured, ys, xs))


def band_corrected(r, tab):
    """Per-frequency aperture correction, then CDIP's energy weighting."""
    sp = r.get('spec')
    if not sp:
        return np.nan, np.nan
    f = np.array(sp['f']); sig = np.array(sp['sigma1_deg'])
    S = np.array(sp['S']); kr = np.array(sp['ridge_k'])
    L = min(r['Lx_used_m'], r['Ly_used_m'])
    ok = np.isfinite(sig) & np.isfinite(kr) & (kr > 0) & (S > 0)
    if not ok.any():
        return np.nan, np.nan
    lam = 2 * np.pi / kr[ok]
    corr = np.array([invert_generic(tab, L / l, s)
                     for l, s in zip(lam, sig[ok])])
    good = np.isfinite(corr)
    if not good.any():
        return np.nan, np.nan
    w = S[ok][good]
    c = corr[good]
    r_lam = (L / lam)[good]
    return (float((w * c).sum() / w.sum()),
            float((w * r_lam).sum() / w.sum()))


def main(fn='records_all.json'):
    C = load_cdip()
    S = synth_cdip()
    R = json.load(open(os.path.join(HERE, fn)))['records']
    TAB = load_generic_calib()
    rows = []
    for r in R:
        sh = r['shell']
        row = dict(site=r['site'], file=r['file'],
                   Hs=r['Hs_4sigma_full'], Hs_rob=r['Hs_4sigma_robust'],
                   Tp=sh['Tp'], sig_band=sh['spread_band_deg'],
                   sig_total=sh['spread_total_deg'],
                   sig_total_allk=r['allk']['spread_total_deg'],
                   floor=r.get('floor_deg', np.nan),
                   sig_true=r.get('sigma_true_est_total', np.nan),
                   ci=r.get('sigma_true_ci', [np.nan, np.nan]),
                   status=r.get('invert_status_total', '-'),
                   L_lam=r.get('aperture_lambda', np.nan),
                   L_lam_max=r.get('aperture_lambda_max', np.nan),
                   onshell=r['onshell_frac_med'],
                   Lx=r['Lx_used_m'], Ly=r['Ly_used_m'],
                   lam_p=r.get('lambda_p_m', np.nan),
                   k_kdeep=r.get('ridge_over_deep_at_fp', np.nan))
        bc, lo_lam_w = band_corrected(r, TAB)
        row['sig_band_corr'] = bc
        row['L_lam_ew'] = lo_lam_w
        row['p_sig_band_corr'] = pct(C['spread_band_deg'], bc)
        row['p_Hs'] = pct(C['Hs'], row['Hs'])
        row['p_Tp'] = pct(C['Tp'], row['Tp'])
        row['p_sig_raw'] = pct(C['spread_total_deg'], row['sig_total'])
        row['p_sig_band_raw'] = pct(C['spread_band_deg'], row['sig_band'])
        row['p_sig_true'] = pct(C['spread_total_deg'], row['sig_true'])
        row['p_sig_true_lo'] = pct(C['spread_total_deg'], row['ci'][0])
        row['p_sig_true_hi'] = pct(C['spread_total_deg'], row['ci'][1])
        rows.append(row)
    rows.sort(key=lambda x: (x['site'], x['file']))

    def band(a, ps=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
        a = a[np.isfinite(a)]
        return {str(p): float(np.percentile(a, p)) for p in ps}

    print('=== CDIP reference (n=%d, %d stations) ===' % (len(C['Hs']), C['n_stations']))
    for k in ('Hs', 'Tp', 'spread_band_deg', 'spread_total_deg'):
        b = band(C[k])
        print('  %-17s min %.2f  p05 %.2f  p25 %.2f  med %.2f  p75 %.2f  p95 %.2f  max %.2f'
              % (k, b['0'], b['5'], b['25'], b['50'], b['75'], b['95'], b['100']))
    print('=== our SYNTHETIC generator (n=%d) ===' % len(S['Hs']))
    for k in ('Hs', 'Tp', 'spread_total_deg'):
        b = band(S[k])
        print('  %-17s min %.2f  med %.2f  max %.2f   -> CDIP pct of max = %.1f'
              % (k, b['0'], b['50'], b['100'], pct(C[k], b['100'])))

    print()
    hdr = ('%-8s %-30s %6s %5s %6s %6s %6s %6s %-16s %5s %5s | %5s %5s %5s %5s' %
           ('site', 'record', 'Hs', 'Tp', 'sigRAW', 'floor', 'sigTRU', 'allk',
            'CI_true', 'L/lam', 'onsh', 'pHs', 'pTp', 'pRAW', 'pTRU'))
    print(hdr); print('-' * len(hdr))
    for x in rows:
        print('%-8s %-30s %6.2f %5.2f %6.1f %6.1f %6.1f %6.1f [%5.1f,%5.1f] %5.2f %5.2f | %5.1f %5.1f %5.1f %5.1f  %s'
              % (x['site'], x['file'][:30], x['Hs'], x['Tp'], x['sig_total'],
                 x['floor'], x['sig_true'], x['sig_total_allk'],
                 x['ci'][0], x['ci'][1], x['L_lam'], x['onshell'],
                 x['p_Hs'], x['p_Tp'], x['p_sig_raw'], x['p_sig_true'], x['status']))
    print()
    hdr2 = ('%-8s %-30s %6s %6s %6s | %6s %6s %6s %6s' %
            ('site', 'record', 'sigBND', 'sigBNDc', 'L/lamEW',
             'pBNDc', 'kbar/k', 'Hsrob', 'depth'))
    print(hdr2); print('-' * len(hdr2))
    for x, r in zip(rows, sorted(R, key=lambda y: (y['site'], y['file']))):
        print('%-8s %-30s %6.1f %6.1f %6.2f | %6.1f %6.2f %6.2f %6.0f'
              % (x['site'], x['file'][:30], x['sig_band'], x['sig_band_corr'],
                 x['L_lam_ew'], x['p_sig_band_corr'],
                 r.get('kbar_over_ktheory_at_fp', float('nan')),
                 x['Hs_rob'], r.get('depth_m', float('nan')) or float('nan')))

    print('\n=== ANSWER: where do the real records sit on the CDIP spread axis? ===')
    good = [x for x in rows if np.isfinite(x['sig_true'])]
    cd = C['spread_total_deg']
    print('  CDIP min = %.2f deg   p05 = %.2f   med = %.2f' %
          (np.nanmin(cd), np.percentile(cd, 5), np.median(cd)))
    print('  records with a usable directional measurement : %d / %d' % (len(good), len(rows)))
    print('  corrected sigma_1_total range  : %.1f .. %.1f deg  (median %.1f)'
          % (min(x['sig_true'] for x in good), max(x['sig_true'] for x in good),
             np.median([x['sig_true'] for x in good])))
    print('  CDIP percentile range          : %.1f .. %.1f  (median %.1f)'
          % (min(x['p_sig_true'] for x in good), max(x['p_sig_true'] for x in good),
             np.median([x['p_sig_true'] for x in good])))
    print('  records BELOW the CDIP minimum : %d  (raw, i.e. upper bound: %d)'
          % (sum(x['sig_true'] < np.nanmin(cd) for x in good),
             sum(x['sig_total'] < np.nanmin(cd) for x in rows)))
    print('  synthetic episodes below CDIP min: %d / %d'
          % (int((S['spread_total_deg'] < np.nanmin(cd)).sum()), len(S['spread_total_deg'])))
    print('  Tp percentiles of real records : %.1f .. %.1f (median %.1f); synthetic cap 9 s = p%.1f'
          % (min(x['p_Tp'] for x in rows), max(x['p_Tp'] for x in rows),
             np.median([x['p_Tp'] for x in rows]), pct(C['Tp'], 9.0)))
    print('  Hs percentiles of real records : %.1f .. %.1f (median %.1f); synthetic cap 2 m = p%.1f'
          % (min(x['p_Hs'] for x in rows), max(x['p_Hs'] for x in rows),
             np.median([x['p_Hs'] for x in rows]), pct(C['Hs'], 2.0)))
    json.dump({'cdip_n': len(C['Hs']), 'rows': rows},
              open(os.path.join(HERE, fn.replace('records', 'report')), 'w'),
              indent=1, default=float)


if __name__ == '__main__':
    main(*sys.argv[1:])
