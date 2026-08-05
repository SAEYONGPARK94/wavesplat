"""e29_fable_axis_forensics.py — settle the ONE remaining dispute:
is A's stratification axis `det` (oracle_ls2, top-32 of a 200-component sea)
a construct-valid measure of OBSERVATION under-determination?

Mirrors e28_gennet_A.build_sea EXACTLY (same rng draws -> same seas; hashabs
is sum(ord)%97, deterministic) but returns diagnostics build_sea discards:
  - the observation-side informativeness Opus proposes as the fix
    (E-24 density axis n/(kp^2 Ax Ay), aperture in wavelengths, U)
  - `det` recomputed at K = 32/64/128/200 modes  <- tests Opus's fix option 1
    (N_OBS=400 => 200 modes = 400 unknowns = exactly determined; predict the
     "ceiling" DROPS as K grows, which would disqualify his own fix)
No training. Inference/regeneration only.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e28_gennet_A as A   # chdirs to repo root

from wavesim.wave_generator import jonswap_components
from wavesplat.model.tnse import fit_amplitudes_phases
from d18_relax_estimator import beamform_top
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, corr,
                                     N_OBS, DEPTH)

KS = [32, 64, 128, 200]


def rebuild(regime, si):
    """byte-for-byte the same draw sequence as A.build_sea."""
    R = A.REGIMES[regime]
    rng = np.random.default_rng(7000 + 31 * si + hashabs(regime))
    Tp = 7.0
    Hs = rng.uniform(*R['hs'])
    c = jonswap_components(Hs=Hs, Tp=Tp, n_components=200, depth=DEPTH,
                           gamma=3.3, direction_deg=rng.uniform(0, 360),
                           direction_spread_deg=30.0, seed=4000 + si)
    kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c])
    Am = np.array([x.A for x in c]); ph = np.array([x.phi for x in c])
    om = omega_of(kx, ky, DEPTH)
    lam = dispersion_lam(Tp, DEPTH); cp = lam / Tp; T = 10 * Tp
    fp = 0.18 * lam; r_fp = fp / 2; side = 2.0 * lam
    gx = np.linspace(0, side, 96); gy = np.linspace(0, side, 96)
    Xg, Yg = np.meshgrid(gx, gy)
    mu = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
    Xs, Ys = Xg[mu], Yg[mu]
    take = np.random.default_rng(0).choice(len(Xs), min(1200, len(Xs)),
                                           replace=False)
    Xs, Ys = Xs[take], Ys[take]
    gt_u = render_modes(kx, ky, Am, ph, DEPTH, Xs, Ys, T)
    Ucp = rng.uniform(*R['U'])
    U = Ucp * cp
    nfrac = rng.uniform(*R['nf'])
    P = sample_obs('fixed' if U < 1e-6 else 'moving', U, lam, cp, T, gx, gy,
                   r_fp, rng)
    psi = (kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2]
           - om[None] * P[:, 2:3] + ph[None])
    z = (Am[None] * np.cos(psi)).sum(1) + rng.normal(0, nfrac * Hs, N_OBS)
    bw_lvl = rng.uniform(*R['bw'])
    if bw_lvl > 1e-4:
        K = min(12, len(Am)); top = np.argsort(Am)[-K:]
        bkx, bky = 2 * kx[top], 2 * ky[top]
        bom, bph = 2 * om[top], 2 * ph[top]
        bA = 0.5 * np.hypot(kx[top], ky[top]) * Am[top] ** 2
        psib = (bkx[None] * P[:, 0:1] + bky[None] * P[:, 1:2]
                - bom[None] * P[:, 2:3] + bph[None])
        zb = (bA[None] * np.cos(psib)).sum(1)
        sc = (np.sqrt(bw_lvl) * (Hs / 4)) / max(np.std(zb), 1e-9)
        z = z + sc * zb
        psib_q = (bkx[None] * Xs[:, None] + bky[None] * Ys[:, None]
                  - bom[None] * T + bph[None])
        gt_u = gt_u + sc * (bA[None] * np.cos(psib_q)).sum(1)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          DEPTH, klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          DEPTH, klim=3.5 * kp, n=120)
    kp = max(np.hypot(bx, by), 1e-3)

    # ---- det at several truncation orders (Opus fix option 1) ----
    obs = np.column_stack([P, z]).astype(np.float32)
    order_all = np.argsort(Am)[::-1]
    dets = {}
    for K in KS:
        o = order_all[:K]
        try:
            A_l, ph_l = fit_amplitudes_phases(kx[o].astype(np.float32),
                                              ky[o].astype(np.float32), obs,
                                              DEPTH, lambda_reg=1.0)
            f_or = render_modes(kx[o], ky[o], np.asarray(A_l, float),
                                np.asarray(ph_l, float), DEPTH, Xs, Ys, T)
            dets[K] = float(corr(f_or, gt_u))
        except Exception as e:
            dets[K] = float('nan')

    # ---- observation-side informativeness (Opus fix option 2) ----
    ax = P[:, 0].max() - P[:, 0].min()
    ay = P[:, 1].max() - P[:, 1].min()
    dens = len(z) / (kp ** 2 * ax * ay / (2 * np.pi) ** 2 + 1e-9)
    # what fraction of field variance the top-32 modes even CONTAIN
    conc32 = float((Am[order_all[:32]] ** 2).sum() / (Am ** 2).sum())
    return dict(si=si, Hs=float(Hs), U_cp=float(Ucp), nfrac=float(nfrac),
                kp=float(kp), lam=float(lam),
                ap_x_wl=float(kp * ax / (2 * np.pi)),
                ap_y_wl=float(kp * ay / (2 * np.pi)),
                dens=float(dens), conc32=conc32,
                **{f'det{K}': dets[K] for K in KS})


rows = [rebuild('spread', si) for si in range(10)]
J = json.load(open('results/direction3/e28_gennet_loro.json'))
for r, p in zip(rows, J['pooled']):
    r['cl'], r['cur'], r['gn'] = p['cl'], p['cur'], p['gn']
    r['det_recorded'] = p['det']

print('si  Hs   U/cp  ap_x[wl] ap_y[wl]  dens  conc32 | det32(rec) d32   d64   d128  d200 |'
      '  cl     cur')
for r in rows:
    print(f"{r['si']:2d} {r['Hs']:.2f} {r['U_cp']:5.2f} {r['ap_x_wl']:7.2f} "
          f"{r['ap_y_wl']:7.2f} {r['dens']:6.1f} {r['conc32']:.3f} | "
          f"{r['det_recorded']:+.3f} {r['det32']:+.3f} {r['det64']:+.3f} "
          f"{r['det128']:+.3f} {r['det200']:+.3f} | {r['cl']:+.3f} {r['cur']:+.3f}")

json.dump(rows, open('results/direction3/e29_fable_axis_forensics.json', 'w'),
          indent=1, default=float)

g = lambda k: np.array([r[k] for r in rows])
print('\n--- reproduction check: det32 vs recorded det ---')
print(f"  max|det32 - recorded| = {np.abs(g('det32')-g('det_recorded')).max():.2e}")


def C(a, b):
    from scipy import stats
    r = np.corrcoef(a, b)[0, 1]
    n = len(a)
    t = r * np.sqrt((n - 2) / max(1 - r ** 2, 1e-12))
    return r, 2 * stats.t.sf(abs(t), n - 2)


print('\n--- Q1: does the DESIGNED aperture axis even vary in A?s eval regime? ---')
for k in ['ap_x_wl', 'ap_y_wl', 'dens', 'lam']:
    v = g(k)
    print(f"  {k:9s} min {v.min():8.2f} max {v.max():8.2f} "
          f"CV {v.std()/abs(v.mean()):.3f}")

print('\n--- Q2: is `det` related to OBSERVATION informativeness? ---')
for k in ['dens', 'ap_x_wl', 'U_cp', 'conc32']:
    r, p = C(g('det_recorded'), g(k))
    print(f"  corr(det, {k:8s}) = {r:+.3f}  p={p:.3f}")

print('\n--- Q3: Opus fix option 1 -- does adding modes RAISE the ceiling? ---')
for K in KS:
    d = g(f'det{K}')
    print(f"  K={K:3d}: mean det {d.mean():+.3f} | arms exceeding it: "
          f"cur {int((g('cur')>d).sum())}/10  cl {int((g('cl')>d).sum())}/10  "
          f"mean(cur-det) {(g('cur')-d).mean():+.3f}")

print('\n--- Q4: do the terciles even AGREE across K? ---')
for K in KS:
    o = np.argsort(g(f'det{K}'))
    print(f"  K={K:3d} bottom-3 seas = {sorted(g('si')[o[:3]].astype(int))}")
