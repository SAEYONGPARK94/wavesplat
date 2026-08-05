"""E-30: is the +0.182 realization headroom EXTRACTABLE at 400 obs?

Logic (the decisive move E-29 never made):
  rung4 (true realization gamma) wins +0.182. But an estimator can only ever
  produce gamma at the resolution its aperture supports. So blur the TRUE
  gamma with a kernel of width sigma and re-solve: oracle@sigma is an UPPER
  BOUND on ANY estimator (learned or not) whose effective k-resolution is
  sigma -- it has the true energies, perfectly placed, merely band-limited.
  Then measure the ACTUAL beamforming PSF width from the real geometry.
  If oracle@PSF_width collapses to the ens_blind (hand-prior) level, the
  realization prize lives entirely BELOW the resolution floor => no
  deconvolution / no net / no amount of data can reach it. If it survives,
  sharpening is live and the direction is OPEN.

Also: per-sea aperture (U), PSF width, and the 180-deg flip diagnosis.
"""
import os, sys, json
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (build_sea_c, deposit_grid, bank_atoms, design_t,
                          corr, DEVICE, DEPTH, _MASK, REGIMES, bank_power_np,
                          dispersion_lam)
from e29_gate0_ladder import universal_radial, ensemble_grid, solve_with_grid, SMULT
from wavesim.wave_generator import jonswap_components
from d18_relax_estimator import beamform_top
from d3_sa_achievability_gap import hashabs
from d3_prior_net_train import NGRID, KREL

OUT = 'results/direction3/e30_resolution_kill.json'
CELL = 2 * KREL / (NGRID - 1)          # bank cell size in units of kp = 0.1


def psf_width_cells(S):
    """Empirical beamforming PSF on the bank: inject ONE unit mode at
    k=kp*(cos mu,sin mu) with the sea's true mean direction, render at the
    sea's actual P (same geometry, no noise), beamform at every bank atom.
    Returns (half-power equivalent radius in cells, equiv-area cells)."""
    P, kp = S['P'], S['kp']
    mu = S['mu_true']
    k0x, k0y = kp * np.cos(mu), kp * np.sin(mu)
    om0 = float(np.sqrt(9.81 * kp * np.tanh(kp * DEPTH)))
    r = np.cos(k0x * P[:, 0] + k0y * P[:, 1] - om0 * P[:, 2])
    pw = bank_power_np(P.astype(np.float64), r.astype(np.float64), kp)
    pw = pw[:len(pw) // 2]                       # free arm only
    grid = np.zeros((NGRID, NGRID))
    grid[_MASK] = pw
    # half-power area -> equivalent radius (in cells)
    n_half = int((grid >= 0.5 * grid.max()).sum())
    r_half = np.sqrt(max(n_half, 1) / np.pi)
    # equivalent-area (integral / peak) -> robust to sidelobes
    n_eq = float(grid.sum() / max(grid.max(), 1e-12))
    r_eq = np.sqrt(n_eq / np.pi)
    return float(r_half), float(r_eq), float(n_half), n_eq


def main(n_seas=16):
    kmag_u, en_u = universal_radial()
    ref = json.load(open('results/direction3/e29_gate0_ceiling.json'))['rows']
    lad = {r['si']: r for r in
           json.load(open('results/direction3/e29_gate0_ladder.json'))['rows']}
    R = REGIMES['spread']
    BLURS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        cur = ref[si]['cur']
        # replicate rng to recover true mean direction AND U (aperture)
        rng = np.random.default_rng(7000 + 31 * si + hashabs('spread'))
        _hs = rng.uniform(*R['hs'])
        mu_true = np.deg2rad(rng.uniform(0, 360))
        lam = dispersion_lam(7.0, DEPTH)
        cp = lam / 7.0
        U = rng.uniform(*R['U']) * cp
        S['mu_true'] = mu_true
        # aperture actually swept (ping-pong along diagonal, T=70s)
        r_fp = 0.18 * lam / 2
        L = np.sqrt(2) * (2 * lam - 2 * r_fp)
        s = np.mod(U * 70.0, 2 * L)
        ap = min(U * 70.0, L) if U * 70.0 <= L else L   # max along-track extent
        ap = max(ap, 2 * r_fp)                          # footprint floor
        rh, re, nh, neq = psf_width_cells(S)
        # oracle gamma blurred to resolution sigma  (free arm only, smult=3)
        gf = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, S['kp'])
        ob = {}
        for b in BLURS:
            g = gaussian_filter(gf, b) if b > 0 else gf
            ob[f'{b:g}'] = solve_with_grid(S, g)
        rows.append(dict(si=si, cur=cur, U=float(U), ap_lam=float(ap / lam),
                         psf_r_half=rh, psf_r_eq=re, psf_n_half=nh,
                         psf_n_eq=neq, oracle_blur=ob,
                         ens_blind=lad[si]['ens_blind'],
                         ens_oracle=lad[si]['ens_oracle'],
                         real_free=lad[si]['real_free'],
                         dir_err=lad[si]['dir_err_deg']))
        print(f"sea{si:2d} U {U:5.2f} ap {ap/lam:4.2f}L psf_r {rh:4.1f}/{re:4.1f}cell "
              f"| cur {cur:+.3f} blind {lad[si]['ens_blind']:+.3f} | oracle@blur "
              + " ".join(f"{b:g}:{ob[f'{b:g}']:+.3f}" for b in BLURS), flush=True)

    a = lambda k: np.array([r[k] for r in rows], float)
    cur = a('cur')
    print(f"\n=== E-30 oracle-blur ladder (n={len(rows)}, smult=3 fixed) ===")
    print(f"  cell = {CELL:.3f} kp;  mean PSF r_half = {a('psf_r_half').mean():.2f} cells "
          f"= {a('psf_r_half').mean()*CELL:.3f} kp")
    print(f"  mean PSF r_eq   = {a('psf_r_eq').mean():.2f} cells "
          f"= {a('psf_r_eq').mean()*CELL:.3f} kp")
    print(f"  ens_blind vs cur : {(a('ens_blind')-cur).mean():+.4f}")
    for b in BLURS:
        o = np.array([r['oracle_blur'][f'{b:g}'] for r in rows])
        d = o - cur
        print(f"  oracle @ blur {b:>3g} cells (= {b*CELL:.2f} kp): {o.mean():+.4f}  "
              f"vs cur {d.mean():+.4f}  (양수 {int((d>0).sum())}/{len(d)})")
    json.dump(dict(rows=rows, blurs=BLURS, cell_kp=CELL), open(OUT, 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
