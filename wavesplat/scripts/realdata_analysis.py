"""
realdata_analysis.py
Analyze Guimaraes/Bergamasco 2020 stereo surface-elevation records (dense GT) with
OUR estimator (iterative RELAX), to test whether nowcast reconstruction works on
LONG-CRESTED / high-aperture-ratio seas vs the short-crested BS/AA records.

Key discriminator measured per record: directional spread + aperture-per-wavelength.
Reads NetCDF4 (=HDF5) via h5py (no netCDF4/xarray dependency).

Usage: python3 scripts/realdata_analysis.py <path_to_Surfaces.nc> [--frames 50]
"""
import os, sys, json, argparse, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
import h5py
from scipy.interpolate import griddata
from d26_relax_iterative import iterative_relax

G = 9.81


# ---------------------------------------------------------------- nc loading
def _unpack(dset, sl):
    """Apply netCDF scale_factor/add_offset/_FillValue packing to an int-packed var."""
    raw = dset[sl]
    a = dset.attrs
    fill = a['_FillValue'][0] if '_FillValue' in a else None
    sf = float(a['scale_factor'][0]) if 'scale_factor' in a else 1.0
    off = float(a['add_offset'][0]) if 'add_offset' in a else 0.0
    out = raw.astype(np.float32) * sf + off
    if fill is not None:
        out[raw == fill] = np.nan
    return out


def load_surface_nc(path, tmax_frames=None, frames=None):
    """Load Guimaraes/WASS 'Surfaces' netCDF (int16-packed Z + 2D X/Y grids) via h5py.

    `frames` = explicit frame indices to read, for sampling across a long record.
    These files are int16-packed and 1.2-11 GB on disk, so unpacking the whole series
    to float32 needs 2x that in RAM (the 11 GB LJ Grid2 record asks for 20.2 GiB and
    dies).  `tmax_frames` keeps the legacy 'first N frames' behaviour.
    """
    f = h5py.File(path, 'r')
    is_wass = ('Z' in f and 'mask_Z' in f) or \
              (b'WASS' in bytes(f.attrs.get('type', b'')) if 'type' in f.attrs else False)
    if not is_wass:
        f.close()
        raise RuntimeError("Not a recognized WASS 'Surfaces' file (need Z + mask_Z).")

    zd = f['Z']                                   # (T, n0, n1) int16, packed
    T_all, n0, n1 = zd.shape
    if frames is not None:
        sel = np.unique(np.asarray(frames, int))  # h5py fancy index must be increasing
        if sel.size and (sel[0] < 0 or sel[-1] >= T_all):
            f.close()
            raise IndexError(f'frames out of range for T={T_all}')
    else:
        sel = np.arange(min(T_all, tmax_frames) if tmax_frames else T_all)
    T = int(sel.size)
    # contiguous prefix -> plain slice (h5py fancy indexing is much slower)
    Z = _unpack(zd, sel if frames is not None else np.s_[:T])   # (T,n0,n1) m, NaN=fill
    # 2D coordinate grids: _nc4_non_coord_X varies along axis0, Y varies along axis1
    Xg2 = _unpack(f['_nc4_non_coord_X'], np.s_[:]) if '_nc4_non_coord_X' in f else None
    Yg2 = _unpack(f['Y'], np.s_[:]) if 'Y' in f and f['Y'].ndim == 2 else None
    ax0 = Xg2[:, 0] if Xg2 is not None else np.arange(n0) * 0.1   # coord along axis0 (rows)
    ax1 = Yg2[0, :] if Yg2 is not None else np.arange(n1) * 0.1   # coord along axis1 (cols)
    # mask (valid grid points, no NaN over series)
    if 'mask_Z' in f:
        mask = _unpack(f['mask_Z'], np.s_[:]); Z[:, mask == 0] = np.nan
    # fps comes from the native cadence, not from the (possibly strided) selection
    t_all = f['time'][:].astype(float) if 'time' in f else np.arange(T_all) * 0.1
    t = t_all[sel]
    dt = float(np.median(np.diff(t_all))) if T_all > 1 else None
    # label axis1 as "x", axis0 as "y" (isotropic dispersion → labeling is a convention)
    gx, gy = np.asarray(ax1, float), np.asarray(ax0, float)   # Nx=n1, Ny=n0
    depth_meta = None
    for k, v in f.attrs.items():
        if any(w in str(k).lower() for w in ('depth', 'bathy')):
            try: depth_meta = float(np.atleast_1d(v).ravel()[0])
            except Exception: pass
    meta = dict(format='WASS', shape=[int(T), int(n0), int(n1)],
                dx=float(np.median(np.diff(gx))), dy=float(np.median(np.diff(gy))),
                patch=[float(gx.max()-gx.min()), float(gy.max()-gy.min())],
                fps=float(1.0/dt) if dt else None,
                depth_meta=depth_meta, frames_total=int(T_all),
                frames_idx=sel.tolist() if frames is not None else None)
    f.close()
    return Z, gx, gy, t, meta


# ---------------------------------------------------------------- characterization
def directional_spread(snapshot, dx, dy):
    """Energy-weighted directional spread (deg) + peak wavelength + peak dir via 2D FFT."""
    s = np.nan_to_num(snapshot - np.nanmean(snapshot))
    F = np.fft.fftshift(np.fft.fft2(s))
    P = np.abs(F) ** 2
    ny, nx = s.shape
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy)) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    m = K > 1e-6
    # peak
    ip = np.argmax(P * m)
    kpx, kpy = KX.ravel()[ip], KY.ravel()[ip]
    kp = np.hypot(kpx, kpy); lam_p = 2 * np.pi / kp if kp > 0 else np.nan
    peak_dir = np.degrees(np.arctan2(kpy, kpx))
    # directional spread: energy-weighted circular std of angle, weighting by P, near-band
    band = m & (K > 0.4 * kp) & (K < 2.5 * kp)
    th = np.arctan2(KY[band], KX[band]); w = P[band]
    # fold direction to [-90,90] (wave axis) to avoid +/- ambiguity of real FFT
    C = np.sum(w * np.cos(2 * th)) / np.sum(w); S = np.sum(w * np.sin(2 * th)) / np.sum(w)
    R = np.hypot(C, S)
    spread = np.degrees(0.5 * np.sqrt(max(-2 * np.log(max(R, 1e-9)), 0.0)))
    return spread, lam_p, peak_dir, kp


def peak_period(Z, t):
    """Tp from temporal spectrum at valid central points."""
    from scipy import signal
    T, Ny, Nx = Z.shape
    dt = float(np.median(np.diff(t)))
    yy, xx = Ny // 2, Nx // 2
    ts = Z[:, yy, xx]
    if np.any(np.isnan(ts)):
        # pick a fully-valid column
        valid = ~np.any(np.isnan(Z.reshape(T, -1)), axis=0)
        if valid.any():
            idx = np.argmax(valid); ts = Z.reshape(T, -1)[:, idx]
    ts = np.nan_to_num(ts - np.nanmean(ts))
    fr, ps = signal.welch(ts, fs=1.0 / dt, nperseg=min(256, T))
    if len(ps) > 2 and ps[1:].max() > 0:
        return 1.0 / max(fr[np.argmax(ps[1:]) + 1], 1e-6), dt
    return np.nan, dt


def lowpass_field(fld, dx, dy, k_cut):
    F = np.fft.fftshift(np.fft.fft2(np.nan_to_num(fld)))
    ny, nx = fld.shape
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy)) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky); K = np.hypot(KX, KY)
    F[K > k_cut] = 0
    return np.fft.ifft2(np.fft.ifftshift(F)).real


# ---------------------------------------------------------------- analysis
def analyze(path, tmax_frames=80, n_per_frame=80, seed=0, depth=None):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    Z, x, y, t, meta = load_surface_nc(path, tmax_frames=tmax_frames)
    # resolve water depth: explicit arg > nc metadata > default (open sea) with warning
    if depth is None:
        if meta.get('depth_meta'):
            depth = float(meta['depth_meta']); print(f"  [depth from nc metadata: {depth:.1f} m]")
        else:
            depth = 60.0; print(f"  [WARNING: depth not in nc; assuming {depth:.0f} m — pass --depth for the site value]")
    T, Ny, Nx = Z.shape
    dx = float(np.median(np.diff(x))); dy = float(np.median(np.diff(y)))
    Lx = float(x.max() - x.min()); Ly = float(y.max() - y.min())
    dt = float(np.median(np.diff(t)))
    # use a central snapshot with fewest NaNs for characterization
    nanfrac = np.mean(np.isnan(Z.reshape(T, -1)), axis=1)
    i_char = int(np.argmin(nanfrac))
    Hs = 4.0 * float(np.nanstd(Z[i_char]))
    spread, lam_p, pdir, kp = directional_spread(Z[i_char], dx, dy)
    Tp, _ = peak_period(Z, t)
    ap_x = Lx / lam_p; ap_y = Ly / lam_p

    print(f"\n{'='*66}\nRECORD: {os.path.basename(path)}")
    print(f"  grid {Nx}x{Ny}  dx={dx:.3f} dy={dy:.3f} m  patch {Lx:.1f}x{Ly:.1f} m")
    print(f"  fps={1/dt:.1f}Hz  frames_used={T}  nan% (char frame)={nanfrac[i_char]*100:.1f}")
    print(f"  Hs={Hs:.2f} m  Tp={Tp:.1f} s  lam_p={lam_p:.1f} m")
    print(f"  ** directional spread={spread:.0f}°  peak_dir={pdir:.0f}°  "
          f"aperture/lam = {ap_x:.1f} x {ap_y:.1f} **")

    # --- reconstruction window: ~4 peak periods ending at i_star ---
    n_back = int(round(4.0 * Tp / dt)) if np.isfinite(Tp) else 40
    n_back = max(8, min(n_back, T - 2))
    i_star = T - 2
    gt = Z[i_star]
    valid_gt = ~np.isnan(gt)
    Hs_w = 4.0 * float(np.nanstd(gt))
    Xg, Yg = np.meshgrid(x, y)

    # sparse observations across the window (mimic sparse LiDAR sampling of the field)
    Ps, zs = [], []
    for i in range(i_star - n_back, i_star + 1):
        fr = Z[i]; vm = ~np.isnan(fr)
        idx = np.flatnonzero(vm.ravel())
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, min(n_per_frame, len(idx)), replace=False)
        ppx = Xg.ravel()[pick]; ppy = Yg.ravel()[pick]; ppz = fr.ravel()[pick]
        Ps.append(np.column_stack([ppx, ppy, np.full(len(pick), i * dt)]))
        zs.append(ppz)
    P = np.vstack(Ps).astype(np.float32); zobs = np.concatenate(zs).astype(np.float32)
    n_obs = len(zobs)

    # depth resolved above (nc metadata / arg / warned default); check kd regime
    kd = kp * depth
    print(f"  depth={depth:.0f} m  kd={kd:.2f} ({'deep' if kd>3 else 'INTERMEDIATE — depth matters'})")
    target = {'gx': x, 'gy': y, 't': float(i_star * dt), 'gt': np.nan_to_num(gt), 'Hs': Hs_w}

    def score(fld):
        f = fld[valid_gt]; g = gt[valid_gt]
        nr = float(np.sqrt(np.mean((f - g) ** 2)) / Hs_w)
        cr = float(np.corrcoef(f, g)[0, 1])
        return nr, cr

    # (1) our estimator
    fld, N = iterative_relax(P, zobs, depth, target, 3)
    nr_ws, cr_ws = score(fld)
    # (2) naive interpolation from the last-frame sparse points
    lastP = Ps[-1]; lastz = zs[-1]
    naive = griddata(lastP[:, :2], lastz, (Xg, Yg), method='linear')
    naive = np.where(np.isnan(naive), np.nanmean(lastz), naive)
    nr_nv, cr_nv = score(naive)
    # (3) band-limited scoring (remove stereo noise / bound harmonics, |k|<1.5 kp)
    k_cut = 1.5 * kp
    gt_bl = lowpass_field(np.nan_to_num(gt), dx, dy, k_cut)
    ws_bl = lowpass_field(fld, dx, dy, k_cut)
    Hs_bl = 4 * gt_bl[valid_gt].std()
    nr_ws_bl = float(np.sqrt(np.mean((ws_bl[valid_gt] - gt_bl[valid_gt]) ** 2)) / max(Hs_bl, 1e-6))
    cr_ws_bl = float(np.corrcoef(ws_bl[valid_gt], gt_bl[valid_gt])[0, 1])

    print(f"  --- nowcast reconstruction ({n_obs} obs over {n_back+1} frames, N_comp={N}) ---")
    print(f"  WaveSplat  nRMSE {nr_ws:.3f}  corr {cr_ws:.3f}   | band-limited nRMSE {nr_ws_bl:.3f} corr {cr_ws_bl:.3f}")
    print(f"  naive interp nRMSE {nr_nv:.3f}  corr {cr_nv:.3f}")
    verdict = "WS beats naive" if nr_ws < nr_nv else "naive wins (info-limit signature)"
    print(f"  => {verdict}   ({time.time()-t0:.0f}s)")

    res = dict(record=os.path.basename(path), meta=meta, Hs=Hs, Tp=Tp, lam_p=lam_p,
               spread_deg=spread, peak_dir=pdir, aperture_x=ap_x, aperture_y=ap_y,
               n_obs=int(n_obs), n_comp=int(N), nrmse_ws=nr_ws, corr_ws=cr_ws,
               nrmse_ws_bandlimited=nr_ws_bl, corr_ws_bandlimited=cr_ws_bl,
               nrmse_naive=nr_nv, corr_naive=cr_nv, verdict=verdict,
               n_back=n_back, dt=dt)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--frames', type=int, default=80)
    ap.add_argument('--nper', type=int, default=80)
    ap.add_argument('--depth', type=float, default=None, help='water depth [m]; auto from nc if omitted')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    res = analyze(args.path, tmax_frames=args.frames, n_per_frame=args.nper, depth=args.depth)
    outp = args.out or ('results/realdata_' + os.path.splitext(os.path.basename(args.path))[0] + '.json')
    os.makedirs('results', exist_ok=True)
    json.dump(res, open(outp, 'w'), indent=2)
    print(f"saved {outp}")
