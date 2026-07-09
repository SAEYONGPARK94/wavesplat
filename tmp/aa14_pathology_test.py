"""AA14 CV-order-collapse pathology test: default vs robust relax_cv settings.
Window i0=200 (B_r0.43 collapsed to N=4 in 2/3 reps, swept -0.17 vs A +0.76)."""
import os
import sys

import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

from d3_sa_multirecord import MRecord, valid_aware_track, RECORDS
from d3_sa_gonogo import build_obs, region_masks, score_regions, characterize
from d3_sa_uspeed import centered_segment, resolve_prop_sign
from realdata_analysis import lowpass_field
from estimator_improved import iterative_relax_cv

cfg = RECORDS['AA14']
rec = MRecord(cfg['path'], stride=1)
i0, T, fp, depth = 200, cfg['T'], cfg['fp'], cfg['depth']
r_fp = fp / 2.0
nf = int(round(T / rec.dt))
Zwin = rec.window(i0, nf)
Hs, lam_p, kp, pdir, spread, Tp, cp = characterize(Zwin, rec, depth)
gt_f = np.nan_to_num(Zwin[-1])
k_cut = 1.5 * kp
gt_bl = lowpass_field(gt_f, rec.dx, rec.dy, k_cut)
t_off = float(rec.t[i0])
t_star = float(rec.t[i0 + nf - 1]) - t_off
target = {'gx': rec.gx, 'gy': rec.gy, 't': t_star, 'gt': gt_f, 'Hs': Hs}

prop_dir, _, _ = resolve_prop_sign(Zwin, rec.dx, rec.dy, kp, pdir, rec.dt)
P0, P1, ang = valid_aware_track(rec, r_fp, pdir)
pvec = np.array([np.cos(np.radians(prop_dir)), np.sin(np.radians(prop_dir))])
if float((P1 - P0) @ pvec) < 0:
    P0, P1 = P1, P0
masks = region_masks(rec, r_fp, 0.6 * fp, P0, P1)
U = 0.43 * cp
L_seg = min(U * T, float(np.linalg.norm(P1 - P0)))
seg = centered_segment(P0, P1, L_seg)

for setting, kw in [('default(p2,m1)', dict(patience=2, min_comp=1)),
                    ('robust(p5,m8)', dict(patience=5, min_comp=8))]:
    print(f"--- {setting}")
    for cond in ('A', 'B43'):
        res = []
        for rep in range(3):
            seed = 63062 + 997 * (0 if cond == 'A' else 2) + 13 * rep
            rng = np.random.default_rng(seed)
            if cond == 'A':
                P, z, _, _ = build_obs(Zwin, rec, i0, 400, r_fp, 'fixed', rng)
            else:
                P, z, _, _ = build_obs(Zwin, rec, i0, 400, r_fp, 'moving', rng,
                                       U=U, track=seg)
            P[:, 2] -= t_off
            fld, N, klim = iterative_relax_cv(P.copy(), z.copy(), depth, target,
                                              n_outer=2, seed=seed, **kw)
            fld_bl = lowpass_field(fld, rec.dx, rec.dy, k_cut)
            sc = score_regions(fld, gt_f, Hs, masks, fld_bl, gt_bl)
            res.append((N, sc['swept']['corr_bl'], sc['common_unobs']['corr_bl']))
        Ns = [r[0] for r in res]
        print(f"  {cond}: N={Ns} swept={np.mean([r[1] for r in res]):+.3f} "
              f"unobs={np.mean([r[2] for r in res]):+.3f}")
