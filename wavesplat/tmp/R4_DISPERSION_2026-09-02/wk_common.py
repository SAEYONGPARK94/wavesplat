import os, numpy as np
G = 9.81

def load_agulhas(p, i0, nf, ys, xs):
    from scipy.io import netcdf_file
    f = netcdf_file(p, 'r', mmap=True); V = f.variables
    Z = np.asarray(V['Z'][i0:i0+nf, ys, xs]).astype(np.float32)
    t = np.asarray(V['time'][:]).ravel().astype(float)[i0:i0+nf]
    gx = np.asarray(V['X_grid'][0, xs]).astype(float)/1000.0
    gy = np.asarray(V['Y_grid'][ys, 0]).astype(float)/1000.0
    f.close()
    return np.where(Z == 0, np.nan, Z)/1000.0, gx, gy, t

def load_berg(p, i0, nf, ys, xs):
    import h5py, sys
    sys.path.insert(0, os.path.join('/home/vexit/wavesplat/wavesplat', 'scripts'))
    from realdata_analysis import _unpack
    f = h5py.File(p, 'r')
    Z = _unpack(f['Z'], np.s_[i0:i0+nf, ys, xs])
    X2 = _unpack(f['_nc4_non_coord_X'], np.s_[:])   # varies axis0
    Y2 = _unpack(f['Y'], np.s_[:])                  # varies axis1
    gy = np.asarray(X2[ys, 0], float)               # along axis0
    gx = np.asarray(Y2[0, xs], float)               # along axis1
    mask = _unpack(f['mask_Z'], np.s_[:])[ys, xs] > 0
    t = f['time'][:].astype(float)[i0:i0+nf]
    f.close()
    Z[:, ~mask] = np.nan
    return Z, gx, gy, t

def wk_report(Z, gx, gy, t, label, detr='none', krings=None):
    nf = Z.shape[0]
    dt = float(np.median(np.diff(t)))
    dx = float(abs(np.median(np.diff(gx)))); dy = float(abs(np.median(np.diff(gy))))
    mu = np.nanmean(Z, axis=0)
    Zf = np.nan_to_num(np.where(np.isfinite(Z), Z, mu[None]) - mu[None])
    ny, nx = Zf.shape[1], Zf.shape[2]
    if detr == 'plane':
        X2, Y2 = np.meshgrid(gx, gy)
        A = np.column_stack([np.ones(X2.size), X2.ravel(), Y2.ravel()])
        S = Zf.reshape(nf, -1)
        c, *_ = np.linalg.lstsq(A, S.T, rcond=None)
        Zf = (S - (A @ c).T).reshape(Zf.shape)
    sig = float(Zf.std())
    W = np.hanning(nf)[:,None,None]*np.hanning(ny)[None,:,None]*np.hanning(nx)[None,None,:]
    P = np.abs(np.fft.fftn(Zf*W))**2
    om = 2*np.pi*np.fft.fftfreq(nf, dt); dom = om[1]-om[0]
    ky = 2*np.pi*np.fft.fftfreq(ny, dy); kx = 2*np.pi*np.fft.fftfreq(nx, dx)
    KX, KY = np.meshgrid(kx, ky); K = np.hypot(KX, KY)
    kmin = max(abs(kx[1]), abs(ky[1]))
    print(f"-- {label} [detr={detr}] {nf}x{ny}x{nx} dt={dt:.3f} dx={dx:.2f} dy={dy:.2f} "
          f"sigma={sig:.3f} Hs={4*sig:.2f} | k_fund={kmin:.4f} (lam_max={2*np.pi/kmin:.0f} m) "
          f"k_nyq={np.pi/max(dx,dy):.3f}")
    pos = om > 0
    if krings is None:
        krings = [0.06,0.08,0.10,0.13,0.16,0.20,0.26,0.34,0.45,0.60]
    print(f"   {'|k|':>7} {'lam':>6} {'om_pk':>7} {'om_lin':>7} {'ratio':>6} {'Efrac':>7} {'sheetfrac':>9}")
    for kr in krings:
        m = (K > kr*0.85) & (K <= kr*1.18)
        if m.sum() < 8: continue
        spec = P[:, m].sum(axis=1)
        sp = spec.copy(); sp[~pos] = 0
        j = np.argmax(sp); omp = om[j]; oml = np.sqrt(G*kr)
        # energy within +-3 om bins of the linear sheet (both signs), over ring energy
        kb = K[m]; jl = np.round(np.sqrt(G*kb)/dom).astype(int)
        Pm = P[:, m]; idx = np.arange(m.sum()); e = 0.0
        for d in range(-3, 4):
            e += Pm[(jl+d) % nf, idx].sum() + Pm[(-jl+d) % nf, idx].sum()
        print(f"   {kr:7.3f} {2*np.pi/kr:6.1f} {omp:7.4f} {oml:7.4f} {omp/oml:6.3f} "
              f"{spec.sum()/P.sum():7.4f} {e/spec.sum():9.4f}")
