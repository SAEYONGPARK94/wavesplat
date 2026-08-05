"""e31 — IS THE +0.182 REALIZATION PRIZE PHYSICAL, OR n_components DISCRETIZATION?

wave_generator.jonswap_components: amplitudes A_n and wavenumbers k_n are
DETERMINISTIC (JONSWAP on fixed omega_centers, renormalized to Hs). ONLY
phases and directions are random. So a "sea" = N modes pinned to fixed
(k_n, A_n) rings, each at a random angle theta_n ~ N(mu, spread).

=> the gate-0 ladder's rung4 ("the actual 200 draws") = knowing N angles.
   Its +0.182 over rung2 = the gain from knowing WHICH cells of the 1904-atom
   bank the N spikes landed in.

CRITICAL: with N=200 spikes over ~1904 band cells, the realized gamma is
SPARSE/SPIKY and differs enormously from the smooth ensemble spectrum. That
difference IS the prize. But a real sea is not 200 delta functions: as
N -> infinity the realized deposit converges to the smooth ensemble spectrum
and rung4 -> rung2, i.e. THE PRIZE VANISHES.

So: is rung4-rung2 a physical information gap, or finite-N discretization?
Sweep N in {200, 800, 3200} on the SAME seas (identical rng stream => same
Hs, mean direction, U, noise draw, obs geometry; only the mode decomposition
of the same underlying spectrum changes) and watch rung4-rung2.

  If rung4-rung2 SHRINKS toward 0 with N -> the prize is a simulator artifact.
  If it SURVIVES                        -> the prize is real realization info.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (deposit_grid, two_stage_kp, corr, DEVICE, DEPTH,
                          _MASK, REGIMES, NGRID, KREL)
from e29_gate0_ladder import solve_with_grid, SMULT
from e30_conditionality_probe import radial_of, ens_grid_norm
from wavesim.wave_generator import jonswap_components
from d18_relax_estimator import beamform_top
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, N_OBS)

OUT = 'results/direction3/e31_ncomp_scaling.json'


def build_sea_n(regime, si, ncomp):
    """EXACT copy of e29_gennet_C.build_sea_c with n_components parametrized.
    The rng stream is untouched by ncomp => Hs, mean dir, U, nfrac, obs
    geometry and the noise draw are IDENTICAL across ncomp."""
    R = REGIMES[regime]
    rng = np.random.default_rng(7000 + 31 * si + hashabs(regime))
    Tp = 7.0
    Hs = rng.uniform(*R['hs'])
    c = jonswap_components(Hs=Hs, Tp=Tp, n_components=ncomp, depth=DEPTH,
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
    gx = np.linspace(0, side, 96)
    gy = np.linspace(0, side, 96)
    Xg, Yg = np.meshgrid(gx, gy)
    mu = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
    Xs, Ys = Xg[mu], Yg[mu]
    take = np.random.default_rng(0).choice(len(Xs), min(1200, len(Xs)),
                                           replace=False)
    Xs, Ys = Xs[take], Ys[take]
    gt_u = render_modes(kx, ky, Am, ph, DEPTH, Xs, Ys, T)
    Pq = np.column_stack([Xs, Ys, np.full(len(Xs), T)])
    U = rng.uniform(*R['U']) * cp
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
        bw_sc = (np.sqrt(bw_lvl) * (Hs / 4)) / max(np.std(zb), 1e-9)
        z = z + bw_sc * zb
        psib_q = (bkx[None] * Xs[:, None] + bky[None] * Ys[:, None]
                  - bom[None] * T + bph[None])
        gt_u = gt_u + bw_sc * (bA[None] * np.cos(psib_q)).sum(1)
    kp = two_stage_kp(P, z)
    return dict(P=P, z=z, Pq=Pq, gt_u=gt_u, kp=kp, sigma0=float(nfrac * Hs),
                Hs=float(Hs), m_kx=kx, m_ky=ky, m_A=Am, m_ph=ph)


def main(n_seas=16, ncomps=(200, 800, 3200)):
    # fine ensemble radial (best approx to the CONTINUOUS jonswap spectrum)
    kn, en = radial_of(7.0, 3.3)
    out = {}
    for NC in ncomps:
        rows = []
        print(f'\n===== n_components = {NC} =====', flush=True)
        for si in range(n_seas):
            S = build_sea_n('spread', si, NC)
            bx, by = beamform_top(S['P'].astype(np.float32),
                                  S['z'].astype(np.float32),
                                  DEPTH, klim=3.5 * S['kp'], n=120)
            mu_hat = np.arctan2(by, bx)
            r2 = solve_with_grid(S, ens_grid_norm(kn, en, mu_hat, S['kp'], 30.0))
            gf = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, S['kp'])
            r4 = solve_with_grid(S, gf)
            # sparsity of the realized deposit: how many band cells are lit
            lit = int((gf[_MASK] > 1e-12).sum())
            rows.append(dict(si=si, ens=r2, real=r4, d=r4 - r2,
                             lit=lit, nb=int(_MASK.sum())))
            print(f'  sea{si:2d} ens {r2:+.3f} | real {r4:+.3f} | d {r4-r2:+.3f} '
                  f'| lit {lit}/{int(_MASK.sum())}', flush=True)
        d = np.array([r['d'] for r in rows])
        e = np.array([r['ens'] for r in rows])
        r = np.array([r['real'] for r in rows])
        lit = np.mean([x['lit'] for x in rows])
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f'  --> N={NC}: ens {e.mean():+.4f} | real {r.mean():+.4f} | '
              f'PRIZE {d.mean():+.4f} +- {se:.4f} (pos {int((d>0).sum())}/{len(d)}) '
              f'| lit cells {lit:.0f}/{int(_MASK.sum())}')
        out[str(NC)] = dict(rows=rows, ens=float(e.mean()), real=float(r.mean()),
                            prize=float(d.mean()), se=float(se), lit=float(lit))

    print('\n\n================ PRIZE vs n_components ================')
    print(f'{"N":>6} {"lit/1904":>9} {"ens":>8} {"real":>8} {"PRIZE":>9} {"se":>7}')
    for NC in ncomps:
        o = out[str(NC)]
        print(f'{NC:6d} {o["lit"]:9.0f} {o["ens"]:+8.4f} {o["real"]:+8.4f} '
              f'{o["prize"]:+9.4f} {o["se"]:7.4f}')
    p0 = out[str(ncomps[0])]['prize']; p1 = out[str(ncomps[-1])]['prize']
    print(f'\n  prize {ncomps[0]} -> {ncomps[-1]}: {p0:+.4f} -> {p1:+.4f} '
          f'({100*(p1-p0)/abs(p0):+.0f}%)')
    print('  -> ARTIFACT (prize dies with N)' if p1 < 0.5 * p0
          else '  -> the prize SURVIVES densification = real realization info')
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
