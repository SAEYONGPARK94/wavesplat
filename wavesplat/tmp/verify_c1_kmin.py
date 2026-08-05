"""C1 independent re-derivation: true FFT wavenumber floor vs pipeline kp.

Independent of the prior agent's probe. Key point he may have got wrong:
on a 2D grid the SMALLEST NON-ZERO |k| that argmax can return is
  k_min = min(dkx, dky) = 2*pi / max(Lx, Ly)
NOT 2*pi/L for a single L. Check both.
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from d3_sa_multirecord import MRecord, RECORDS
from realdata_analysis import directional_spread

G = 9.81

out = []
for name, cfg in RECORDS.items():
    rec = MRecord(cfg['path'], stride=cfg.get('stride', 1))
    nyv, nxv = rec.n0, rec.n1          # axis0 -> gy, axis1 -> gx
    dx, dy = rec.dx, rec.dy
    # FFT periods: N*d  (fftfreq uses n and d)
    Lx_fft = nxv * dx
    Ly_fft = nyv * dy
    dkx = 2 * np.pi / Lx_fft
    dky = 2 * np.pi / Ly_fft
    k_min_true = min(dkx, dky)          # smallest attainable non-zero |k|
    # extent (edge to edge) as the prior agent likely used it
    Lx_ext = float(rec.gx.max() - rec.gx.min())
    Ly_ext = float(rec.gy.max() - rec.gy.min())

    # replicate characterize(): kp from the LAST frame of a window
    nf = int(round(cfg['T'] / rec.dt))
    kps = []
    for w in range(cfg['nw']):
        i0 = w * nf
        if i0 + nf > rec.T_all:
            break
        Z = rec.window(i0, nf)
        gt = Z[-1]
        spread, lam_p, pdir, kp = directional_spread(np.nan_to_num(gt), rec.dx, rec.dy)
        kps.append(kp)
    kp_med = float(np.median(kps))
    out.append(dict(rec=name, n0=nyv, n1=nxv, dx=round(dx, 4), dy=round(dy, 4),
                    Lx_fft=round(Lx_fft, 2), Ly_fft=round(Ly_fft, 2),
                    Lx_ext=round(Lx_ext, 2), Ly_ext=round(Ly_ext, 2),
                    dkx=round(dkx, 5), dky=round(dky, 5),
                    k_min_true=round(k_min_true, 5),
                    kp_pipe_med=round(kp_med, 5),
                    kp_all=[round(k, 5) for k in kps],
                    ratio_kp_over_kmin=round(kp_med / k_min_true, 4)))
    print(f"{name:6s} grid {nyv}x{nxv} d=({dx:.3f},{dy:.3f}) "
          f"Lfft=({Lx_fft:.1f},{Ly_fft:.1f}) dk=({dkx:.4f},{dky:.4f}) "
          f"k_min={k_min_true:.4f} kp_pipe={kp_med:.4f} "
          f"ratio={kp_med/k_min_true:.3f}", flush=True)

json.dump(out, open(os.path.join(os.path.dirname(__file__), 'verify_c1_kmin.json'), 'w'), indent=1)
print("\nwrote tmp/verify_c1_kmin.json")
