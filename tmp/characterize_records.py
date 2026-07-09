"""Characterize all Bergamasco records for the multi-record SA collapse design.

For each record: grid extent, dt, duration, Hs, lam_p, kp, Tp, cp, T_coh,
patch-diagonal / lam_p, and the implied scaled parameters (fp_diam ~ 0.18 lam_p,
T_win ~ 10 Tp). Depth per site from Guimaraes 2020.
"""
import os
import sys
import json

import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

from d3_sa_gonogo import WassRecord, characterize
from d3_sa_uspeed import estimate_tcoh

SITE_DEPTH = {'AA': 17.0, 'BS': 30.0, 'YS': 50.0, 'LJ': 45.0}

records = []
base = 'data/real_bergamasco'
for site in sorted(os.listdir(base)):
    for fn in sorted(os.listdir(os.path.join(base, site))):
        if fn.endswith('.nc'):
            records.append((site, os.path.join(base, site, fn)))

out = []
for site, path in records:
    depth = SITE_DEPTH[site[:2]]
    try:
        rec = WassRecord(path)
        dur = float(rec.t[-1] - rec.t[0])
        ext_x = float(rec.gx.max() - rec.gx.min())
        ext_y = float(rec.gy.max() - rec.gy.min())
        diag = float(np.hypot(ext_x, ext_y))
        # characterization window: min(60 s, half the record), from the middle
        nf = min(int(round(60.0 / rec.dt)), rec.T_all // 2)
        i0 = (rec.T_all - nf) // 2
        Zwin = rec.window(i0, nf)
        valid_frac = float(np.isfinite(Zwin[-1]).mean())
        Hs, lam_p, kp, pdir, spread, Tp, cp = characterize(Zwin, rec, depth)
        t_coh, capped = estimate_tcoh(Zwin, rec.dt, np.random.default_rng(0))
        row = dict(site=site, record=os.path.basename(path), depth=depth,
                   n_frames=rec.T_all, dt=round(rec.dt, 4), dur_s=round(dur, 1),
                   ext_x=round(ext_x, 1), ext_y=round(ext_y, 1), diag=round(diag, 1),
                   dx=round(rec.dx, 3), dy=round(rec.dy, 3),
                   valid_frac=round(valid_frac, 3),
                   Hs=round(Hs, 3), lam_p=round(lam_p, 1), kp=round(kp, 4),
                   Tp=round(Tp, 2), cp=round(cp, 2), peak_dir=round(pdir, 1),
                   spread=round(spread, 1),
                   t_coh=round(t_coh, 2), t_coh_capped=bool(capped),
                   diag_over_lam=round(diag / lam_p, 3),
                   tcoh_over_Tp=round(t_coh / Tp, 3),
                   fp_scaled=round(0.18 * lam_p, 2), T_win_scaled=round(10 * Tp, 1))
        rec.f.close()
    except Exception as e:
        row = dict(site=site, record=os.path.basename(path), error=repr(e))
    out.append(row)
    print(json.dumps(row))

with open('/home/vexit/wavesplat/tmp/record_characterization.json', 'w') as f:
    json.dump(out, f, indent=1)
print('\nsaved -> tmp/record_characterization.json')
