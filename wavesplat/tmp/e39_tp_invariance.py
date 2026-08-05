"""e39 — verify the load-bearing claim "the kp-normalized bank cancels Tp
analytically" (probe claimed ens arm: Tp11 +0.3699 vs Tp7 +0.3694).

Rebuild the arena at Tp in {7, 9, 11} with EVERYTHING scaled as build_sea_c
scales it (T=10*Tp, side=2*lam, fp=0.18*lam, U in cp units), n=400, and run
the constant hand prior (ens, smult=3) and the oracle realization (real).
If the normalized bank cancels Tp, ens and the headroom real-ens should be
statistically flat across Tp. DEPTH=50 m fixed => kp*d changes (76 m -> ~180 m
wavelength), so exact invariance is NOT guaranteed analytically; measure it.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (REGIMES, deposit_grid, two_stage_kp, corr, DEPTH,
                          _MASK, NGRID, KREL)
from e29_gate0_ladder import solve_with_grid
from e30_conditionality_probe import radial_of, ens_grid_norm
from d3_sa_nomp import beamform_power
from d18_relax_estimator import beamform_top
from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_achievability_gap import (dispersion_lam, masks_common_unobs,
                                     hashabs)
from e38_aperture_sweep import sample_obs_n

OUT = 'results/direction3/e39_tp_invariance.json'


def build_sea_tp(si, Tp):
    R = REGIMES['spread']
    rng = np.random.default_rng(7000 + 31 * si + hashabs('spread'))
    Hs = rng.uniform(*R['hs'])
    c = jonswap_components(Hs=Hs, Tp=Tp, n_components=200, depth=DEPTH,
                           gamma=3.3, direction_deg=rng.uniform(0, 360),
                           direction_spread_deg=30.0, seed=4000 + si)
    kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c])
    Am = np.array([x.A for x in c]); ph = np.array([x.phi for x in c])
    om = omega_of(kx, ky, DEPTH)
    lam = dispersion_lam(Tp, DEPTH)
    cp = lam / Tp
    T = 10 * Tp
    fp = 0.18 * lam
    r_fp = fp / 2
    side = 2.0 * lam
    gx = np.linspace(0, side, 96); gy = np.linspace(0, side, 96)
    Xg, Yg = np.meshgrid(gx, gy)
    mu = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
    Xs, Ys = Xg[mu], Yg[mu]
    take = np.random.default_rng(0).choice(len(Xs), min(1200, len(Xs)), False)
    Xs, Ys = Xs[take], Ys[take]
    gt_u = render_modes(kx, ky, Am, ph, DEPTH, Xs, Ys, T)
    Pq = np.column_stack([Xs, Ys, np.full(len(Xs), T)])
    U = rng.uniform(*R['U']) * cp
    nfrac = rng.uniform(*R['nf'])
    P = sample_obs_n('fixed' if U < 1e-6 else 'moving', U, T, gx, gy, r_fp,
                     rng, 400)
    psi = (kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2]
           - om[None] * P[:, 2:3] + ph[None])
    z = (Am[None] * np.cos(psi)).sum(1) + rng.normal(0, nfrac * Hs, 400)
    bw_lvl = rng.uniform(*R['bw'])
    if bw_lvl > 1e-4:
        K = min(12, len(Am)); top = np.argsort(Am)[-K:]
        bkx, bky = 2 * kx[top], 2 * ky[top]
        bom, bph = 2 * om[top], 2 * ph[top]
        bA = 0.5 * np.hypot(kx[top], ky[top]) * Am[top] ** 2
        psib = (bkx[None] * P[:, 0:1] + bky[None] * P[:, 1:2]
                - bom[None] * P[:, 2:3] + bph[None])
        zb = (bA[None] * np.cos(psib)).sum(1)
        bw_sc = (np.sqrt(bw_lvl) * (Hs / 4)) / max(np.std(zb), 1e-9)
        z = z + bw_sc * zb
        psib_q = (bkx[None] * Xs[:, None] + bky[None] * Ys[:, None]
                  - bom[None] * T + bph[None])
        gt_u = gt_u + bw_sc * (bA[None] * np.cos(psib_q)).sum(1)
    kp = two_stage_kp(P, z)
    return dict(P=P, z=z, Pq=Pq, gt_u=gt_u, kp=kp, sigma0=float(nfrac * Hs),
                m_kx=kx, m_ky=ky, m_A=Am, m_ph=ph)


def main(n_seas=16, tps=(7.0, 9.0, 11.0)):
    out = {}
    for Tp in tps:
        kn, en = radial_of(Tp, 3.3)
        rows = []
        for si in range(n_seas):
            S = build_sea_tp(si, Tp)
            bx, by = beamform_top(S['P'].astype(np.float32),
                                  S['z'].astype(np.float32),
                                  DEPTH, klim=3.5 * S['kp'], n=120)
            mu_hat = np.arctan2(by, bx)
            r_ens = solve_with_grid(S, ens_grid_norm(kn, en, mu_hat,
                                                     S['kp'], 30.0))
            r_real = solve_with_grid(S, deposit_grid(S['m_kx'], S['m_ky'],
                                                     S['m_A'] ** 2 / 2,
                                                     S['kp']))
            rows.append(dict(si=si, ens=r_ens, real=r_real))
        e = np.array([r['ens'] for r in rows])
        r_ = np.array([r['real'] for r in rows])
        h = r_ - e
        se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
        print(f'Tp={Tp:g}: ens {e.mean():+.4f}±{se(e):.4f}  '
              f'real {r_.mean():+.4f}  H {h.mean():+.4f}±{se(h):.4f}',
              flush=True)
        out[f'{Tp:g}'] = rows
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
