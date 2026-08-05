"""C2(iii) DECISIVE: is the single-frame 2D spatial peak physics or variance?

Fix vs my first attempt: MRecord.window() subtracts the TEMPORAL MEAN over the
window, so window(i,1) returns identically ZERO. Must use a multi-frame window
and then take frames out of it -- exactly as characterize() does (Zwin[-1]).

Test design (this is the whole point):
  single-frame periodogram  and  frame-AVERAGED tapered periodogram
carry the SAME Jacobian F(k,th) = E(f,th)*cg/k and the SAME directional
structure. They differ ONLY in estimator variance (chi2_2 per bin -> chi2_2M).
So:
  * if the 2D peak stays at k~0.24 after averaging -> the peak is PHYSICS
    (a real second wind-sea peak / a Jacobian effect) -> prior agent REFUTED.
  * if the 2D peak moves down to the temporal kp (~0.10) -> the single-frame
    peak was VARIANCE -> the Jacobian objection cannot rescue it, because the
    Jacobian is present in BOTH spectra.
Also report the radially-integrated S(k) peak (the directionally-integrated
spatial spectrum -- removes the "directional spread" objection entirely).
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from d3_sa_multirecord import MRecord
from realdata_analysis import directional_spread

G = 9.81
REC = {
    'AA15a': ('data/real_bergamasco/AA_2015/Surfaces_20150305_103500.nc', 17.0, 1, 580),
    'AA14':  ('data/real_bergamasco/AA_2014/Surfaces_20140327_091000.nc', 17.0, 1, 470),
    'AA15b': ('data/real_bergamasco/AA_2015/Surfaces_20150515_090000.nc', 17.0, 1, 510),
    'YS17':  ('data/real_bergamasco/YS_2017/Surfaces_20170513_050000.nc', 50.0, 2, 610),
    'LJ18g': ('data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid2_short.nc', 45.0, 2, 650),
    'BS11a': ('data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc', 30.0, 1, 300),
}

out = {}
for name, (path, depth, stride, nf) in REC.items():
    rec = MRecord(path, stride=stride)
    nf = min(nf, rec.T_all - 400)
    i0 = 200
    Z = rec.window(i0, nf)                       # temporal-mean removed (pipeline does this)

    # --- 1) single-frame, NO taper: exactly characterize()
    _, lam1, _, kp1 = directional_spread(np.nan_to_num(Z[-1]), rec.dx, rec.dy)

    ny, nx = rec.n0, rec.n1
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=rec.dx)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=rec.dy)) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    m = K > 1e-6
    W = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]

    # --- 2) frame-averaged, Hann-tapered periodogram (same Jacobian, less variance)
    acc = np.zeros((ny, nx))
    used = 0
    for j in range(0, nf, 3):
        s = np.nan_to_num(Z[j])
        F = np.fft.fftshift(np.fft.fft2(s * W))
        acc += np.abs(F) ** 2
        used += 1
    acc /= used
    ipk = int(np.argmax(np.where(m, acc, -1)))
    kp_avg = float(K.ravel()[ipk])

    # --- 3) radially integrated S(k): kills the directional-spread objection
    kb = np.arange(0.0, min(2.0, K.max()), max(kx[1] - kx[0], ky[1] - ky[0]) * 0.5)
    Sk = np.zeros(len(kb) - 1)
    for i in range(len(kb) - 1):
        sel = m & (K >= kb[i]) & (K < kb[i + 1])
        Sk[i] = acc[sel].mean() * (0.5 * (kb[i] + kb[i + 1])) if sel.any() else 0.0
    kc = 0.5 * (kb[:-1] + kb[1:])
    kp_rad = float(kc[int(np.argmax(Sk))])

    # --- 4) single-frame argmax spread across many frames (variance diagnostic)
    kps = []
    for j in range(0, nf, 20):
        _, _, _, kk = directional_spread(np.nan_to_num(Z[j]), rec.dx, rec.dy)
        kps.append(float(kk))
    kps = np.array(kps)

    out[name] = dict(
        kp_1frame_characterize=round(float(kp1), 4),
        kp_2Davg_tapered=round(kp_avg, 4),
        kp_radial_Sk=round(kp_rad, 4),
        kp_1frame_median=round(float(np.median(kps)), 4),
        kp_1frame_p10=round(float(np.percentile(kps, 10)), 4),
        kp_1frame_p90=round(float(np.percentile(kps, 90)), 4),
        kp_1frame_IQRratio=round(float(np.percentile(kps, 75) / np.percentile(kps, 25)), 3),
        n_frames_probed=len(kps))
    print(f"{name:6s} kp: 1frame(charact)={kp1:.4f} | 2D-avg-taper={kp_avg:.4f} | "
          f"radial S(k)={kp_rad:.4f} | 1frame med={np.median(kps):.4f} "
          f"[p10 {np.percentile(kps,10):.4f}, p90 {np.percentile(kps,90):.4f}]", flush=True)
    rec.f.close()

json.dump(out, open(os.path.join(os.path.dirname(__file__), 'verify_c2b_spatial.json'), 'w'), indent=1)
print("\nwrote tmp/verify_c2b_spatial.json")
