"""Independent audit: raycast mesh under-rendering of stored episodes.

Read-only. Derives mesh spacing from configs, verifies wave_components
column order physically (dispersion check), and tabulates points-per-
wavelength + variance fraction below the lambda=6*dx adequacy threshold.
"""
import glob
import os
import sys
from collections import defaultdict

import h5py
import numpy as np

G = 9.81
DATA = '/home/vexit/wavesplat/wavesplat/data'

# ── 1. Mesh spacing from configs (linspace, endpoints inclusive) ──────────
def spacing(x_range, y_range, nx, ny):
    dx = (x_range[1] - x_range[0]) / (nx - 1)
    dy = (y_range[1] - y_range[0]) / (ny - 1)
    return dx, dy

XR, YR = (10.0, 250.0), (-100.0, 100.0)
dx_def, dy_def = spacing(XR, YR, 40, 30)     # default.yaml mesh
dx_srv, dy_srv = spacing(XR, YR, 80, 60)     # server.yaml mesh
print('default.yaml mesh 40x30 :  dx = %.4f m, dy = %.4f m' % (dx_def, dy_def))
print('server.yaml  mesh 80x60 :  dx = %.4f m, dy = %.4f m' % (dx_srv, dy_srv))
print('GT grid shapes: default gt 80x60 -> field (60,80); server gt 120x90 -> field (90,120)')
print()

# ── 2. Single-component BF2 fixedT sanity check ───────────────────────────
def disp_omega(k, d):
    return np.sqrt(G * k * np.tanh(k * d))

fp = os.path.join(DATA, 'episode_BF2_1comp_seed0_fixedT.h5')
with h5py.File(fp, 'r') as f:
    comp = f['config/wave_components'][:]
    at = dict(f['config'].attrs)
    gt_shape = f['frames/gt_wave_field'].shape
    npts = f['frames/lidar_points'].shape
print('=== episode_BF2_1comp_seed0_fixedT.h5 ===')
print('attrs:', {k: (v if not isinstance(v, bytes) else v.decode()) for k, v in at.items()})
print('gt_wave_field shape:', gt_shape, ' lidar_points shape:', npts)
print('wave_components shape:', comp.shape)
A, om, kx, ky, phi, Q = comp[0]
kmag = np.hypot(kx, ky)
lam = 2 * np.pi / kmag
d = float(at['depth'])
Tp = float(at['Tp'])
om_p = 2 * np.pi / Tp
print('col-order-as-[A,om,kx,ky,phi,Q]: A=%.4f om=%.4f kx=%.4f ky=%.4f phi=%.4f Q=%.4f'
      % (A, om, kx, ky, phi, Q))
print('|k| = %.5f  ->  lambda = %.3f m' % (kmag, lam))
print('dispersion check: omega = %.5f  vs  sqrt(gk tanh(kd)) = %.5f  (rel err %.2e)'
      % (om, disp_omega(kmag, d), abs(om - disp_omega(kmag, d)) / om))
print('omega/omega_p = %.4f  (expected 1.75 from linspace(0.5wp,3wp,2) bin center)'
      % (om / om_p))
lam_at_Tp = 2 * np.pi / (om_p**2 / G / np.tanh((om_p**2 / G) * d))  # approx deep
print('for reference, deep-water lambda at Tp=%.1fs would be %.2f m' % (Tp, G * Tp**2 / (2*np.pi)))
print('Hs check: 4*sqrt(sum A^2/2) = %.4f  vs attr Hs = %.4f' % (4*np.sqrt(np.sum(comp[:,0]**2/2)), at['Hs']))
print()

# also swap-hypothesis check: if cols were [A, om, kx, ky, ...] wrong, does
# treating col1 as something else satisfy dispersion better? (col order proof)
alt = np.sqrt(G * np.hypot(comp[0,2], comp[0,3]) * np.tanh(np.hypot(comp[0,2], comp[0,3]) * d))
print('(dispersion consistency is the physical proof of column order)')
print()

# ── 3. Full sweep over all episodes ───────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA, 'episode_*.h5')))
skipped_lfs, skipped_err = [], []
groups = defaultdict(list)
disp_worst = 0.0

for fp in files:
    # LFS pointer files are tiny text files
    try:
        with open(fp, 'rb') as fh:
            magic = fh.read(8)
        if not magic.startswith(b'\x89HDF'):
            skipped_lfs.append(fp)
            continue
        with h5py.File(fp, 'r') as f:
            comp = f['config/wave_components'][:]
            at = dict(f['config'].attrs)
            gt_shape = f['frames/gt_wave_field'].shape
    except Exception as e:
        skipped_err.append((fp, str(e)))
        continue

    A = comp[:, 0]; om = comp[:, 1]
    kmag = np.hypot(comp[:, 2], comp[:, 3])
    d = float(at.get('depth', 50.0))
    relerr = np.max(np.abs(om - disp_omega(kmag, d)) / om)
    disp_worst = max(disp_worst, relerr)
    lam = 2 * np.pi / kmag
    var = A**2
    ipk = int(np.argmax(var))
    lam_pk = lam[ipk]

    # infer generating config from GT field shape
    if gt_shape[1:] == (60, 80):
        cfg_tag, dx, dy = 'default(40x30)', dx_def, dy_def
    elif gt_shape[1:] == (90, 120):
        cfg_tag, dx, dy = 'server(80x60)', dx_srv, dy_srv
    else:
        cfg_tag, dx, dy = 'UNKNOWN' + str(gt_shape), dx_def, dy_def

    base = os.path.basename(fp)
    # group key: sea state, ncomp, variant suffix
    parts = base.replace('episode_', '').replace('.h5', '').split('_')
    sea, ncomp_s = parts[0], parts[1]
    suffix = parts[3] if len(parts) > 3 else '?'
    key = (sea, ncomp_s, suffix, cfg_tag)

    frac6 = float(var[lam < 6 * dx].sum() / var.sum())
    frac2 = float(var[lam < 2 * dx].sum() / var.sum())
    groups[key].append(dict(lam_pk=lam_pk, ppl_x=lam_pk / dx, ppl_y=lam_pk / dy,
                            frac6=frac6, frac2=frac2, lam_min=lam.min(),
                            Tp=float(at['Tp'])))

print('=== per-group summary (mean over seeds) ===')
hdr = ('%-6s %-7s %-8s %-15s %2s  %6s %6s %6s  %7s %7s %8s  %s'
       % ('sea', 'ncomp', 'variant', 'cfg', 'n', 'lamPk', 'pp/x', 'pp/y',
          'var<6dx', 'var<2dx', 'lam_min', 'verdict'))
print(hdr)
for key in sorted(groups):
    g = groups[key]
    lam_pk = np.mean([x['lam_pk'] for x in g])
    pplx = np.mean([x['ppl_x'] for x in g])
    pply = np.mean([x['ppl_y'] for x in g])
    f6 = np.mean([x['frac6'] for x in g])
    f2 = np.mean([x['frac2'] for x in g])
    lmin = np.mean([x['lam_min'] for x in g])
    ppl_min = min(pplx, pply)
    verdict = ('UNDER(<2)' if ppl_min < 2 else
               'BORDER(2-6)' if ppl_min < 6 else 'OK(>=6)')
    print('%-6s %-7s %-8s %-15s %2d  %6.1f %6.1f %6.1f  %6.1f%% %6.1f%% %7.2fm  %s'
          % (key[0], key[1], key[2], key[3], len(g), lam_pk, pplx, pply,
             100*f6, 100*f2, lmin, verdict))

print()
print('worst dispersion rel-err across all readable comps: %.2e' % disp_worst)
print('readable: %d, skipped (LFS pointer): %d, skipped (other error): %d'
      % (sum(len(v) for v in groups.values()), len(skipped_lfs), len(skipped_err)))
for fp, e in skipped_err[:5]:
    print('  ERR', os.path.basename(fp), e[:80])

# ── 5. HOS check ──────────────────────────────────────────────────────────
print()
print('=== HOS (d3_f1_hos_door1 / d3_h7_hos_arena) ===')
MESH_DX = 2.0
TP_HOS, D_HOS, HS_HOS = 8.0, 50.0, 1.2
om_p = 2 * np.pi / TP_HOS
# solve dispersion for k at omega_p
from scipy.optimize import brentq
k_p = brentq(lambda k: G * k * np.tanh(k * D_HOS) - om_p**2, 1e-6, 10)
lam_p = 2 * np.pi / k_p
print('MESH_DX = %.1f m, Tp = %.1f s, depth = %.0f m' % (MESH_DX, TP_HOS, D_HOS))
print('peak: k=%.5f, lambda=%.2f m -> %.1f pts/lambda' % (k_p, lam_p, lam_p / MESH_DX))
# mesh spans check (np.arange, step = MESH_DX exact)
gx = np.arange(0.0, 260.0 + MESH_DX, MESH_DX)
gy = np.arange(-240.0, 240.0 + MESH_DX, MESH_DX)
print('arena mesh: %d x %d nodes, dx=dy=%.3f m exactly' % (len(gx), len(gy), gx[1]-gx[0]))
# variance fraction below 6*dx = 12 m for the linear component sea (NCOMP=60)
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat')
from wavesim.wave_generator import jonswap_components
cs = jonswap_components(HS_HOS, TP_HOS, n_components=60, depth=D_HOS, seed=0,
                        direction_deg=0.0, direction_spread_deg=30.0)
Ac = np.array([c.A for c in cs]); kc = np.hypot([c.kx for c in cs], [c.ky for c in cs])
lamc = 2*np.pi/kc
v = Ac**2
print('linear NCOMP=60 sea: var frac lambda<12m (=6dx): %.3f%%; lambda_min=%.2f m'
      % (100*v[lamc < 6*MESH_DX].sum()/v.sum(), lamc.min()))
# analytic JONSWAP integral fraction (continuous, 0.5wp..3wp band)
def jons(om):
    sig = np.where(om <= om_p, 0.07, 0.09)
    a = np.exp(-0.5*((om/om_p-1)/sig)**2)
    return 0.3125*HS_HOS**2*om_p**4/om**5*np.exp(-1.25*(om_p/om)**4)*3.3**a
oms = np.linspace(0.5*om_p, 3.0*om_p, 20001)
S = jons(oms)
# omega at lambda=12m
k12 = 2*np.pi/12.0
om12 = disp_omega(k12, D_HOS)
frac = np.trapezoid(S[oms > om12], oms[oms > om12]) / np.trapezoid(S, oms)
print('analytic JONSWAP band [0.5,3]wp: var frac above om(lambda=12m)=%.3f rad/s: %.3f%%'
      % (om12, 100*frac))
