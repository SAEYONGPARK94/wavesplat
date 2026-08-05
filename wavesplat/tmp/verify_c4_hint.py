"""C4: which kp does the net EXPECT, and is E-12b's design isolating the threat?

d3_prior_net_data.py:132  "kp estimate exactly as at inference time
                           (two-stage beamform zoom)"  -> kp_e from SPARSE OBS.
d3_prior_net_eval.py:71   if kp_hint is not None: kp = kp_hint   -> DENSE spatial FFT.

So the net is TRAINED on beamform-zoom-kp (sparse) and EVALUATED on
spatial-FFT-kp (dense). That is a train/test mismatch that exists even if the
spatial kp were PERFECT. Consequence: E-12b swaps spatial->temporal, i.e. one
non-training-consistent hint for ANOTHER. The training-consistent (and
sensor-realisable, cf E-11) arm is kp_hint=None. It is missing from E-12b.

Here: compute all three kp on real records, on a matched ~600-obs sparse budget.
"""
import sys, os, json
import numpy as np
from scipy import signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from d3_sa_multirecord import MRecord
from realdata_analysis import directional_spread
from d3_sa_nomp import beamform_top

G = 9.81
FAN = {
    'YS17':  ('data/real_bergamasco/YS_2017/Surfaces_20170513_050000.nc', 50.0, 2),
    'AA15a': ('data/real_bergamasco/AA_2015/Surfaces_20150305_103500.nc', 17.0, 1),
    'AA14':  ('data/real_bergamasco/AA_2014/Surfaces_20140327_091000.nc', 17.0, 1),
    'AA15b': ('data/real_bergamasco/AA_2015/Surfaces_20150515_090000.nc', 17.0, 1),
    'LJ18g': ('data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid2_short.nc', 45.0, 2),
}
W_MEM = 40.0      # fan causal memory [s]
N_OBS = 600       # E-9 sparse observation budget


def k_of_f(f, d):
    om = 2 * np.pi * f
    lo, hi = 1e-5, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if G * mid * np.tanh(mid * d) - om ** 2 > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


rows = []
for name, (path, depth, stride) in FAN.items():
    rec = MRecord(path, stride=stride)
    nfw = int(round(W_MEM / rec.dt))
    i0 = rec.T_all // 2
    Z = rec.window(i0 - nfw + 1, nfw)

    # (a) spatial FFT kp -- what the pipeline feeds as kp_hint  (dense field!)
    _, _, _, kp_sp = directional_spread(np.nan_to_num(Z[-1]), rec.dx, rec.dy)

    # (b) temporal kp -- his "truth"
    Zs = Z[:, ::8, ::8].reshape(nfw, -1)
    good = np.isfinite(Zs).all(axis=0)
    ts = Zs[:, good] - Zs[:, good].mean(axis=0, keepdims=True)
    fr, P = signal.welch(ts, fs=1.0 / rec.dt, nperseg=min(512, nfw), axis=0)
    Pm = P.mean(axis=1)
    band = (fr > 0.03) & (fr < 1.0)
    fp = float(fr[np.argmax(np.where(band, Pm, -1))])
    kp_t = k_of_f(fp, depth)

    # (c) beamform-zoom kp on a 600-obs SPARSE budget -- what the net was TRAINED on
    rng = np.random.default_rng(0)
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    vy, vx = np.where(rec.valid)
    kps_bf = []
    for trial in range(3):
        fi = rng.integers(0, nfw, N_OBS)
        pi = rng.integers(0, len(vy), N_OBS)
        zz = Z[fi, vy[pi], vx[pi]]
        ok = np.isfinite(zz)
        Pobs = np.column_stack([Xg[vy[pi], vx[pi]], Yg[vy[pi], vx[pi]],
                                rec.t[i0 - nfw + 1 + fi] - rec.t[i0 - nfw + 1]])[ok]
        zo = zz[ok]
        bx, by = beamform_top(Pobs.astype(np.float32), zo.astype(np.float32),
                              depth, klim=2.0, n=120)
        k1 = max(float(np.hypot(bx, by)), 1e-3)
        bx, by = beamform_top(Pobs.astype(np.float32), zo.astype(np.float32),
                              depth, klim=3.5 * k1, n=120)
        kps_bf.append(max(float(np.hypot(bx, by)), 1e-3))
    kp_bf = float(np.median(kps_bf))

    rows.append(dict(rec=name, kp_spatial_dense=round(float(kp_sp), 4),
                     kp_temporal=round(kp_t, 4), kp_beamform_sparse=round(kp_bf, 4),
                     sp_over_t=round(float(kp_sp) / kp_t, 3),
                     bf_over_t=round(kp_bf / kp_t, 3),
                     sp_over_bf=round(float(kp_sp) / kp_bf, 3)))
    r = rows[-1]
    print(f"{name:6s} kp_spatial(dense,fed as hint)={kp_sp:.4f}  "
          f"kp_temporal(his truth)={kp_t:.4f}  "
          f"kp_beamform(600 sparse, TRAINING recipe)={kp_bf:.4f}   | "
          f"sp/t={r['sp_over_t']:.2f}x  bf/t={r['bf_over_t']:.2f}x  "
          f"sp/bf={r['sp_over_bf']:.2f}x", flush=True)
    rec.f.close()

json.dump(rows, open(os.path.join(os.path.dirname(__file__), 'verify_c4_hint.json'), 'w'), indent=1)
print("\nsp/bf != 1 => the hint the net RECEIVES differs from the hint it was TRAINED with,")
print("regardless of which of the two is 'correct'. E-12b (spatial vs temporal) does not")
print("test that axis at all; only kp_hint=None does.")
