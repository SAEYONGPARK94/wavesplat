"""Measure sigma_1 / Hs / Tp for every WASS record in data/real_bergamasco/.

Light by construction: NW windows x NT frames per record, spatial decimation to
<= NMAX points per axis, single-threaded FFT.  Nothing is parallelised.
"""
import os, re, sys, json, glob, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spread_est import (analyse_block, moments_from_E, band_summary, FMIN, FMAX, G)
import synth

ROOT = '/home/vexit/wavesplat/wavesplat'
OUT = os.path.join(ROOT, 'tmp/realspread_2026-08-05')
NT = 1024          # frames per window
NW = 3             # windows per record
NMAX = 160         # max spatial samples per axis after decimation
DXMAX = 1e9        # decimate ONLY to respect NMAX -- never throw away aperture
FULLMASK = os.environ.get('FULLMASK', '0') == '1'   # use whole grid, mask as taper
# Site depths, as already registered in tmp/record_wavelength.json for this dataset.
DEPTH = {'AA_2014': 17.0, 'AA_2015': 17.0, 'BS_2011': 30.0, 'BS_2013': 30.0,
         'LJ_2018': 45.0, 'YS_2017': 50.0}


def largest_rect(mask):
    """Largest all-True axis-aligned rectangle. -> (i0,i1,j0,j1)."""
    m = mask.astype(np.int32)
    n, w = m.shape
    h = np.zeros(w, np.int32)
    best = (0, 0, 0, 0, 0)
    for i in range(n):
        h = np.where(m[i] > 0, h + 1, 0)
        stack = []
        for j in range(w + 1):
            cur = h[j] if j < w else 0
            start = j
            while stack and stack[-1][1] >= cur:
                s, ht = stack.pop()
                area = ht * (j - s)
                if area > best[0]:
                    best = (area, i - ht + 1, i + 1, s, j)
                start = s
            stack.append((start, cur))
    return best[1:]


def read_decimated(z, t0, nt, i0, i1, j0, j1, si, sj, scale, fill):
    """Read frames [t0,t0+nt) cropped+decimated, in 128-frame chunks."""
    idx_i = np.arange(i0, i1, si)
    idx_j = np.arange(j0, j1, sj)
    out = np.empty((nt, len(idx_i), len(idx_j)), np.float32)
    step = 128
    for a in range(0, nt, step):
        b = min(a + step, nt)
        blk = z[t0 + a:t0 + b, i0:i1:si, j0:j1:sj]
        bad = blk == fill
        blk = blk.astype(np.float32) * np.float32(scale)
        if bad.any():
            blk[bad] = np.nan
        out[a:b] = blk
    return out, idx_i, idx_j


def do_record(path, log):
    t_start = time.time()
    r = {'path': path, 'site': path.split('/')[-2], 'file': os.path.basename(path)}
    with h5py.File(path, 'r') as f:
        z = f['Z']
        scale = float(np.ravel(z.attrs['scale_factor'])[0])
        fill = int(np.ravel(z.attrs['_FillValue'])[0])
        nt_all, nX, nY = z.shape
        t = f['time'][:]
        dt = float(np.median(np.diff(t)))
        Xv = f['_nc4_non_coord_X'][:].astype(np.float64) * float(np.ravel(f['_nc4_non_coord_X'].attrs['scale_factor'])[0]) + float(np.ravel(f['_nc4_non_coord_X'].attrs['add_offset'])[0])
        Yv = f['Y'][:].astype(np.float64) * float(np.ravel(f['Y'].attrs['scale_factor'])[0]) + float(np.ravel(f['Y'].attrs['add_offset'])[0])
        x = Xv[:, 0]
        y = Yv[0, :]
        dx0 = float(np.median(np.diff(x)))
        dy0 = float(np.median(np.diff(y)))
        mask = f['mask_Z'][:] == 1

        if FULLMASK:
            ii_v = np.where(mask.any(axis=1))[0]
            jj_v = np.where(mask.any(axis=0))[0]
            i0, i1, j0, j1 = int(ii_v[0]), int(ii_v[-1] + 1), int(jj_v[0]), int(jj_v[-1] + 1)
        else:
            i0, i1, j0, j1 = largest_rect(mask)
        si = max(1, int(np.ceil((i1 - i0) / NMAX)))
        sj = max(1, int(np.ceil((j1 - j0) / NMAX)))
        dx, dy = dx0 * si, dy0 * sj
        r.update(nt_all=nt_all, dt=dt, fs=1 / dt, dur_s=float(t[-1] - t[0]),
                 grid_shape=[nX, nY], dx0=dx0, dy0=dy0,
                 mask_frac=float(mask.mean()),
                 rect=[int(i0), int(i1), int(j0), int(j1)],
                 rect_Lx_m=float((i1 - i0 - 1) * dx0), rect_Ly_m=float((j1 - j0 - 1) * dy0),
                 decim=[int(si), int(sj)], dx=dx, dy=dy)

        nt = min(NT, nt_all)
        starts = np.linspace(0, nt_all - nt, NW).astype(int) if nt_all > nt else np.array([0])
        starts = sorted(set(starts.tolist()))
        Eacc = None
        hs_var, hs_n = 0.0, 0
        nan_frac = []
        for s0 in starts:
            blk, ii, jj = read_decimated(z, s0, nt, i0, i1, j0, j1, si, sj, scale, fill)
            nf_bad = float(np.isnan(blk).mean())
            nan_frac.append(nf_bad)
            if nf_bad > 0:
                blk = np.nan_to_num(blk, nan=0.0)
            v = blk - blk.mean(axis=0, keepdims=True)
            hs_var += float((v ** 2).sum()); hs_n += v.size
            ff, kx, ky, E = analyse_block(blk, dt, dx, dy)
            Eacc = E if Eacc is None else Eacc + E
            del blk, v, E
        Eacc /= len(starts)
        r['n_windows'] = len(starts)
        r['window_s'] = nt * dt
        r['df'] = float(ff[1] - ff[0])
        r['nan_frac_in_rect'] = float(np.mean(nan_frac))
        r['grid_used'] = [len(ii), len(jj)]
        r['Lx_used_m'] = float((len(ii) - 1) * dx)
        r['Ly_used_m'] = float((len(jj) - 1) * dy)

        # ---- Hs from the FULL valid mask (not just the analysis rectangle) --
        ts = np.linspace(0, nt_all - 1, min(400, nt_all)).astype(int)
        acc = []
        for k in ts:
            fr = z[k].astype(np.float32)
            fr[fr == fill] = np.nan
            acc.append(fr[mask] * scale)
        A = np.array(acc, np.float32)
        A = A - np.nanmean(A, axis=0, keepdims=True)   # per-point mean removal
        v = A[np.isfinite(A)]
        r['Hs_4sigma_full'] = float(4.0 * v.std())
        # robust: MAD-based sigma, insensitive to reconstruction outliers
        r['Hs_4sigma_robust'] = float(4.0 * 1.4826 * np.median(np.abs(v - np.median(v))))
        q = np.percentile(v, [0.1, 1, 50, 99, 99.9])
        r['z_pct'] = [float(x) for x in q]
        r['z_absmax'] = float(np.abs(v).max())
        r['Hs_4sigma_rect'] = float(4.0 * np.sqrt(hs_var / hs_n))
        r['mask_frac_in_rect'] = float(mask[i0:i1, j0:j1].mean())
        mgrid = mask[i0:i1:si, j0:j1:sj]
        r['mask_frac_used'] = float(mgrid.mean())
        del A, acc, v

    depth = DEPTH.get(r['site'])
    Lmin = min(r['Lx_used_m'], r['Ly_used_m'])
    r['depth_m'] = depth
    for tag, shell in (('shell', True), ('allk', False)):
        a1, b1, S, ridge, onsh = moments_from_E(ff, kx, ky, Eacc, use_shell=shell,
                                                depth=depth, Lmin=Lmin)
        bs = band_summary(ff, a1, b1, S)
        r[tag] = bs
        if tag == 'shell':
            m = (ff >= FMIN) & (ff <= FMAX) & np.isfinite(a1)
            r['onshell_frac_med'] = float(np.nanmedian(onsh[m])) if m.any() else np.nan
            fp = bs.get('fp', np.nan)
            if np.isfinite(fp) and fp > 0:
                ip = int(np.argmin(abs(ff - fp)))
                kr = ridge[ip]
                kdeep = (2 * np.pi * fp) ** 2 / G
                r['ridge_over_deep_at_fp'] = float(kr / kdeep) if np.isfinite(kr) else np.nan
                r['lambda_p_m'] = float(2 * np.pi / kr) if np.isfinite(kr) and kr > 0 else np.nan
                # diagnostic: does the measured energy actually sit on that shell?
                KXn = kx[:, None]; KYn = ky[None, :]
                Kn = np.hypot(KXn, KYn)
                Ep = Eacc[ip].copy(); Ep[0, 0] = 0.0
                r['kbar_over_ktheory_at_fp'] = float((Ep * Kn).sum() / max(Ep.sum(), 1e-30) / kr)
            r['spec'] = {'f': ff[m].tolist(),
                         'sigma1_deg': np.degrees(np.sqrt(np.clip(2 * (1 - np.sqrt(np.clip(a1[m] ** 2 + b1[m] ** 2, 0, 1))), 0, None))).tolist(),
                         'S': S[m].tolist(), 'onshell': onsh[m].tolist(),
                         'ridge_k': ridge[m].tolist()}
    if np.isfinite(r.get('lambda_p_m', np.nan)):
        L = min(r['Lx_used_m'], r['Ly_used_m'])
        r['aperture_lambda'] = float(L / r['lambda_p_m'])
        r['aperture_lambda_max'] = float(max(r['Lx_used_m'], r['Ly_used_m']) / r['lambda_p_m'])
        # ---- per-record instrument response on THIS grid/mask/lambda_p -----
        try:
            resp = synth.response(len(ii), len(jj), dx, dy, nt, dt,
                                  r['lambda_p_m'], r['shell']['fp'],
                                  mask=(mgrid if mgrid.mean() < 1.0 else None), n_real=3,
                                  depth=depth, Lmin=Lmin)
            r['response'] = resp
            for key, tag in (('sigma_meas_total', 'total'), ('sigma_meas_band', 'band')):
                est, st = synth.invert(resp, r['shell']['spread_total_deg' if tag == 'total'
                                                       else 'spread_band_deg'], key=key)
                r[f'sigma_true_est_{tag}'] = est
                r[f'invert_status_{tag}'] = st
            r['floor_deg'] = float(resp[0]['sigma_meas_total'])
            # sampling uncertainty: propagate the realization scatter of the
            # response through the same inversion
            xs = np.array([e['sigma_true'] for e in resp])
            ys = np.array([e['sigma_meas_total'] for e in resp])
            sds = np.array([e['real_sd'] for e in resp])
            meas = r['shell']['spread_total_deg']
            sd_here = float(np.interp(np.clip(meas, ys[0], ys[-1]), ys, sds))
            lo, _ = synth.invert(resp, meas - sd_here)
            hi, _ = synth.invert(resp, meas + sd_here)
            r['sigma_true_ci'] = [lo, hi]
            r['resp_sd_at_meas'] = sd_here
        except Exception as e:
            r['response_error'] = f'{type(e).__name__}: {e}'
    r['secs'] = round(time.time() - t_start, 1)
    log(f"  {r['file'][:34]:34s} Hs={r['Hs_4sigma_full']:6.3f} Tp={r['shell']['Tp']:5.2f} "
        f"sig_tot={r['shell']['spread_total_deg']:6.2f} floor={r.get('floor_deg', float('nan')):6.2f} "
        f"true_est={r.get('sigma_true_est_total', float('nan')):6.2f} [{r.get('invert_status_total','-')}] "
        f"L/lam={r.get('aperture_lambda', float('nan')):5.2f} onsh={r['onshell_frac_med']:.2f} "
        f"({r['secs']}s)")
    return r


def main():
    files = sorted(glob.glob(os.path.join(ROOT, 'data/real_bergamasco/*/*.nc')))
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    if only:
        files = [p for p in files if any(o in p for o in only)]
    res, fails = [], []
    logf = open(os.path.join(OUT, 'measure.log'), 'a')

    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()

    log(f'=== {time.strftime("%F %T")} NT={NT} NW={NW} NMAX={NMAX} n={len(files)}')
    for p in files:
        try:
            res.append(do_record(p, log))
        except Exception as e:
            import traceback; traceback.print_exc()
            fails.append({'path': p, 'error': f'{type(e).__name__}: {e}'})
            log(f'  FAIL {p}: {type(e).__name__}: {e}')
    tag = re.sub(r'[^A-Za-z0-9]+', '-', '_'.join(only)) if only else 'all'
    tag += ('_fullmask' if FULLMASK else '')
    with open(os.path.join(OUT, f'records_{tag}.json'), 'w') as f:
        json.dump({'params': dict(NT=NT, NW=NW, NMAX=NMAX, DXMAX=DXMAX,
                                  band_hz=[FMIN, FMAX]),
                   'records': res, 'failures': fails}, f, indent=1)
    log(f'done ok={len(res)} fail={len(fails)}')


if __name__ == '__main__':
    main()
