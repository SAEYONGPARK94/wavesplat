"""Bit-identity + timing benchmark before the full Doppler run.

(1) Verify mmse_bank_estimate default (Uc=None) == Uc=(0,0) bit-identical on
    one synthetic scene and one real window (analytic arm only).
(2) Time: one full-Pq solve (final-arm cost) and one obs-only solve
    (U-search candidate cost) on a real window, to budget the grid search.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
# importing d3_compare_gifs chdir's to the wavesplat pkg dir + wires sys.path
import d3_compare_gifs as cg
from d3_sa_nomp import mmse_bank_estimate

# ---------------------------------------------------------- (1a) synthetic
print("=== (1a) synthetic scene: Uc=None vs Uc=(0,0) ===", flush=True)
sc = cg.scene_sim('linear', seed=0)
s0 = 0.05 * sc['Hs']
f_none = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'],
                            sigma0=s0, kp_hint=sc['kp'], seed=sc['seed'])
f_zero = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'],
                            sigma0=s0, kp_hint=sc['kp'], seed=sc['seed'],
                            Uc=(0.0, 0.0))
print(f"  identical: {np.array_equal(f_none, f_zero)}  "
      f"max|diff|={np.max(np.abs(f_none - f_zero)):.3e}", flush=True)

# ---------------------------------------------------------- (1b) real window
print("=== (1b) real window YS17 w0: Uc=None vs Uc=(0,0) ===", flush=True)
sc = cg.scene_real('YS17', win=0)
s0 = 0.05 * sc['Hs']
t0 = time.time()
f_none = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'],
                            sigma0=s0, kp_hint=sc['kp'], seed=sc['seed'])
t_full = time.time() - t0
f_zero = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'],
                            sigma0=s0, kp_hint=sc['kp'], seed=sc['seed'],
                            Uc=(0.0, 0.0))
print(f"  identical: {np.array_equal(f_none, f_zero)}  "
      f"max|diff|={np.max(np.abs(f_none - f_zero)):.3e}", flush=True)
print(f"  full-Pq solve time: {t_full:.2f}s  (nobs={len(sc['z'])}, "
      f"nq={len(sc['Pq'])})", flush=True)

# ------------------------------------------------- (2) obs-only search cost
print("=== (2) obs-only U-search candidate cost (YS17 w0) ===", flush=True)
gate = {}
Pq_tiny = sc['P'][:1]           # 1 query -> render cost negligible
t0 = time.time()
_ = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], Pq_tiny,
                       sigma0=s0, kp_hint=sc['kp'], seed=sc['seed'],
                       diag_out=gate, Uc=(0.3, 0.1))
t_srch = time.time() - t0
print(f"  obs-only solve time: {t_srch:.3f}s  vr={gate.get('vr'):.4f} "
      f"vr_rel={gate.get('vr_rel'):.4f}", flush=True)

# project grid-search budget
for step, label in ((0.3, 'coarse 0.3'), (0.2, 'coarse 0.2'), (0.1, 'fine 0.1')):
    ax = np.arange(-1.5, 1.5 + 1e-9, step)
    UX, UY = np.meshgrid(ax, ax)
    npts = int(((UX**2 + UY**2) <= 1.5**2 + 1e-9).sum())
    tot = npts * 30 * t_srch
    print(f"  {label}: {npts} U-pts/win -> {npts*30} solves -> "
          f"{tot/60:.1f} min", flush=True)
