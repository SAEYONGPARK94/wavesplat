"""C2 independent: temporal kp estimator + the three validations.

(i)   Hs / Tp vs Guimaraes 2020 Sci Data 7:145 published values
(ii)  LJ Grid1 (40 m) vs Grid2 (250 m): same sea, same timestamp
(iii) AA15a bimodality -- and the DECISIVE counter-test to the Jacobian /
      directional-integration objection:

      A frame-AVERAGED, Hann-TAPERED 2D periodogram has EXACTLY the same
      Jacobian (F(k,th) = E(f,th)*cg/k) and the same directional structure as
      the single-frame one. The ONLY difference is estimator variance.
      So if frame-averaging MOVES the 2D peak from k=0.243 down to ~0.10,
      then the single-frame peak was VARIANCE, not physics -- and the
      Jacobian/directional objection cannot rescue it, because that objection
      applies identically to both spectra.
"""
import sys, os, json
import numpy as np
from scipy import signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from d3_sa_multirecord import MRecord
from realdata_analysis import directional_spread

G = 9.81

REC = {
    'BS11a': ('data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc', 30.0, 1),
    'BS11b': ('data/real_bergamasco/BS_2011/Surfaces_20111001_161800_short.nc', 30.0, 1),
    'BS13a': ('data/real_bergamasco/BS_2013/Surfaces_20130925_121501_short.nc', 30.0, 1),
    'AA14':  ('data/real_bergamasco/AA_2014/Surfaces_20140327_091000.nc', 17.0, 1),
    'AA15a': ('data/real_bergamasco/AA_2015/Surfaces_20150305_103500.nc', 17.0, 1),
    'AA15b': ('data/real_bergamasco/AA_2015/Surfaces_20150515_090000.nc', 17.0, 1),
    'YS17':  ('data/real_bergamasco/YS_2017/Surfaces_20170513_050000.nc', 50.0, 2),
    'LJ18n': ('data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid1_short.nc', 45.0, 2),
    'LJ18g': ('data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid2_short.nc', 45.0, 2),
}


def k_from_omega(om, d):
    """Invert om^2 = g k tanh(k d) by bisection."""
    lo, hi = 1e-5, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = G * mid * np.tanh(mid * d) - om ** 2
        if f > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def temporal_kp(rec, depth, nframes=3000, pix_stride=12):
    """Multi-pixel Welch PSD over time -> peak f -> dispersion -> kp."""
    n = min(nframes, rec.T_all)
    i0 = (rec.T_all - n) // 2
    Z = rec.window(i0, n)                    # (n, ny, nx), NaN outside valid
    Zs = Z[:, ::pix_stride, ::pix_stride]
    ny, nx = Zs.shape[1], Zs.shape[2]
    ts = Zs.reshape(n, -1)
    good = np.isfinite(ts).all(axis=0)
    ts = ts[:, good]
    if ts.shape[1] < 5:
        return None
    ts = ts - ts.mean(axis=0, keepdims=True)
    fs = 1.0 / rec.dt
    nper = min(1024, n)
    fr, P = signal.welch(ts, fs=fs, nperseg=nper, axis=0)
    Pm = P.mean(axis=1)                      # average PSD over pixels
    band = (fr > 0.03) & (fr < 1.0)
    ip = np.argmax(np.where(band, Pm, -1))
    fp = float(fr[ip])
    om = 2 * np.pi * fp
    kp = k_from_omega(om, depth)
    Hs = 4.0 * float(np.nanstd(Z))
    return dict(fp=fp, Tp=1.0 / fp, kp=kp, lam=2 * np.pi / kp, Hs=Hs,
                fr=fr, Pm=Pm, npix=int(ts.shape[1]), nframes=n)


def spatial_2d_avg(rec, nframes=200, step=5):
    """Frame-AVERAGED, Hann-tapered 2D periodogram -> 2D argmax kp + radial S(k)."""
    idx = np.arange(0, min(nframes * step, rec.T_all - 1), step)
    ny, nx = rec.n0, rec.n1
    wy = np.hanning(ny)[:, None]
    wx = np.hanning(nx)[None, :]
    W = wy * wx
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=rec.dx)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=rec.dy)) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    acc = np.zeros((ny, nx))
    cnt = 0
    for i in idx:
        Z = rec.window(int(i), 1)[0]
        s = np.nan_to_num(Z - np.nanmean(Z))
        F = np.fft.fftshift(np.fft.fft2(s * W))
        acc += np.abs(F) ** 2
        cnt += 1
    acc /= cnt
    m = K > 1e-6
    ipk = np.argmax(np.where(m, acc, -1))
    kp2d = float(K.ravel()[ipk])
    return kp2d, K, acc, cnt


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    res = {}
    for name, (path, depth, stride) in REC.items():
        if which != 'all' and name != which:
            continue
        rec = MRecord(path, stride=stride)
        t = temporal_kp(rec, depth)
        # single-frame spatial kp, same frame as characterize would use
        Zw = rec.window(rec.T_all // 2, 1)
        _, lam_s, _, kp_s = directional_spread(np.nan_to_num(Zw[0]), rec.dx, rec.dy)
        kp2d, K, acc, cnt = spatial_2d_avg(rec)
        row = dict(rec=name, depth=depth, dt=round(rec.dt, 4), T_all=int(rec.T_all),
                   Hs=round(t['Hs'], 3), fp=round(t['fp'], 4), Tp=round(t['Tp'], 2),
                   kp_temporal=round(t['kp'], 5), lam_temporal=round(t['lam'], 1),
                   kp_spatial_1frame=round(float(kp_s), 5),
                   lam_spatial_1frame=round(float(lam_s), 1),
                   kp_spatial_2Davg=round(kp2d, 5),
                   lam_spatial_2Davg=round(2 * np.pi / kp2d, 1),
                   npix=t['npix'], nframes_welch=t['nframes'], nframes_2davg=cnt)
        res[name] = row
        print(f"{name:6s} Hs={row['Hs']:6.2f} fp={row['fp']:.4f} Tp={row['Tp']:5.2f}s "
              f"| kp_temporal={row['kp_temporal']:.4f} (lam {row['lam_temporal']:6.1f} m) "
              f"| kp_spat_1frame={row['kp_spatial_1frame']:.4f} (lam {row['lam_spatial_1frame']:5.1f}) "
              f"| kp_spat_2Davg={row['kp_spatial_2Davg']:.4f} (lam {row['lam_spatial_2Davg']:6.1f})",
              flush=True)
        if name == 'AA15a':
            np.savez(os.path.join(os.path.dirname(__file__), 'verify_aa15a_spec.npz'),
                     fr=t['fr'], Pm=t['Pm'], K=K, acc=acc)
        rec.f.close()
    json.dump(res, open(os.path.join(os.path.dirname(__file__),
                                     f'verify_c2_temporal_{which}.json'), 'w'), indent=1)
