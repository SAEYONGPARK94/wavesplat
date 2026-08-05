"""C1, faithful: EXACT pipeline window starts (d3_sa_multirecord.py:222)
starts = np.linspace(200, T_all - nf - 10, n_win).
Express every per-window kp as a multiple of the TRUE 2D FFT floor
k_min = min(dkx, dky) = 2*pi / max(Nx*dx, Ny*dy).
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from d3_sa_multirecord import MRecord, RECORDS
from realdata_analysis import directional_spread

out = []
for name, cfg in RECORDS.items():
    rec = MRecord(cfg['path'], stride=cfg.get('stride', 1))
    nf = int(round(cfg['T'] / rec.dt))
    n_win = cfg['nw']
    starts = np.linspace(200, rec.T_all - nf - 10, n_win).astype(int)

    dkx = 2 * np.pi / (rec.n1 * rec.dx)   # kx uses nx = n1, d = dx
    dky = 2 * np.pi / (rec.n0 * rec.dy)   # ky uses ny = n0, d = dy
    k_min = min(dkx, dky)

    kps = []
    for i0 in starts:
        Z = rec.window(int(i0), nf)
        _, lam_p, _, kp = directional_spread(np.nan_to_num(Z[-1]), rec.dx, rec.dy)
        kps.append(float(kp))
    mult = [round(k / k_min, 3) for k in kps]
    kp_med = float(np.median(kps))
    n_at_floor = sum(1 for m in mult if abs(m - 1.0) < 0.02)
    out.append(dict(rec=name, k_min_true=round(k_min, 5),
                    dkx=round(dkx, 5), dky=round(dky, 5),
                    starts=[int(s) for s in starts],
                    kp_win=[round(k, 5) for k in kps],
                    kp_over_kmin=mult,
                    kp_med=round(kp_med, 5),
                    med_over_kmin=round(kp_med / k_min, 3),
                    n_windows_exactly_at_floor=n_at_floor,
                    n_win=len(kps)))
    print(f"{name:6s} k_min={k_min:.4f}  kp/k_min per window = {mult}"
          f"   median={kp_med:.4f} ({kp_med/k_min:.2f}x floor)"
          f"  [{n_at_floor}/{len(kps)} windows EXACTLY at floor]", flush=True)

json.dump(out, open(os.path.join(os.path.dirname(__file__), 'verify_c1_exact.json'), 'w'), indent=1)
print("\nwrote tmp/verify_c1_exact.json")
