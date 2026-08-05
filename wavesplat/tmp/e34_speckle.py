"""e34 — IS THE REALIZATION INFO IN THE BEAMFORM MAP AT ALL? (no estimator)

e32 showed every data-derived prior LOSES to the constant. Two readings:
  (i)  the map HAS realization info, my extractions were just crude   -> path lives
  (ii) the map has NO realization info beyond the ensemble             -> closed

Settle it WITHOUT any estimator, by direct correlation on the bank cells.

MECHANISM UNDER TEST (speckle). The beamform map is a ONE-LOOK periodogram:
  bf(k) = |sum_n A_n e^{i phi_n} PSF(k - k_n)|^2
The modes' random phases interfere inside each PSF footprint, so bf's per-cell
fluctuation is Rayleigh SPECKLE whose std equals its mean -- a 100%-noise
estimate of the realized energy. Aperture sets PSF WIDTH (resolution); it does
NOT reduce speckle contrast, which is set by the number of independent LOOKS
(=1 for a single window). That is the classic inconsistency of the periodogram:
var = mean^2, and you can only beat it by averaging looks -- which estimates
the ENSEMBLE and destroys the realization info you wanted.

MEASURE, per sea, over the 1904 band cells:
  r_ens   = spearman(bf, gamma_ens)     -- how much of the map is just ensemble
  r_true  = spearman(bf, gamma_true)    -- how much the map tracks realization
  PARTIAL = spearman(bf | gamma_ens, gamma_true | gamma_ens)
            ^ THE number: realization info in the map BEYOND the ensemble.
  ceiling = spearman(gamma_ens, gamma_true)  -- how much the ensemble alone gives

MULTILOOK sweep: split the track into L sub-apertures and average their
periodograms. Speckle variance falls as 1/L; PSF broadens by L. If the partial
correlation PEAKS at some L>1, speckle is the binding constraint and a
multilook prior is the surviving move. If PARTIAL ~ 0 for every L, the map
carries no realization info and the direction is closed.
"""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (build_sea_c, deposit_grid, DEPTH, _MASK, NGRID, KREL)
from e30_conditionality_probe import radial_of, ens_grid_norm
from d3_sa_nomp import beamform_power
from d18_relax_estimator import beamform_top

OUT = 'results/direction3/e34_speckle.json'
LOOKS = [1, 2, 4, 8]


def partial_spearman(x, y, z):
    """spearman(x,y | z) via rank residuals."""
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    bx = np.linalg.lstsq(Z, rx, rcond=None)[0]
    by = np.linalg.lstsq(Z, ry, rcond=None)[0]
    ex, ey = rx - Z @ bx, ry - Z @ by
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ex, ey)[0, 1])


def multilook_map(P, z, kp, L):
    """Average the periodograms of L contiguous time sub-apertures."""
    order = np.argsort(P[:, 2])
    acc = np.zeros(NGRID * NGRID)
    seg = np.array_split(order, L)
    for idx in seg:
        _, _, p = beamform_power(P[idx], z[idx], DEPTH, KREL * kp, NGRID)
        acc += p / max(p.max(), 1e-12)
    return (acc / L).reshape(NGRID, NGRID)


def main(n_seas=16):
    kn, en = radial_of(7.0, 3.3)
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        kp = S['kp']
        bx, by = beamform_top(S['P'].astype(np.float32), S['z'].astype(np.float32),
                              DEPTH, klim=3.5 * kp, n=120)
        mu_hat = np.arctan2(by, bx)
        g_ens = ens_grid_norm(kn, en, mu_hat, kp, 30.0)[_MASK]
        g_true = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, kp)[_MASK]
        r = dict(si=si)
        r['ceiling_ens_true'] = float(spearmanr(g_ens, g_true)[0])
        for L in LOOKS:
            bf = multilook_map(S['P'], S['z'], kp, L)[_MASK]
            r[f'r_ens_L{L}'] = float(spearmanr(bf, g_ens)[0])
            r[f'r_true_L{L}'] = float(spearmanr(bf, g_true)[0])
            r[f'partial_L{L}'] = partial_spearman(bf, g_true, g_ens)
        rows.append(r)
        msg = ' | '.join(f'L{L} part {r[f"partial_L{L}"]:+.3f}' for L in LOOKS)
        print(f'sea{si:2d} ens~true {r["ceiling_ens_true"]:+.3f} || {msg}', flush=True)

    a = lambda k: np.array([r[k] for r in rows], float)
    print(f'\n=== n={n_seas} : information in the beamform map about the REALIZATION ===')
    print(f'  ceiling spearman(gamma_ens, gamma_true) = {a("ceiling_ens_true").mean():+.4f}'
          f'   <- what the ENSEMBLE alone already explains')
    print(f'\n{"looks":>6} {"r(bf,ens)":>11} {"r(bf,true)":>11} {"PARTIAL":>10} {"se":>7} {"t":>6}')
    out = {}
    for L in LOOKS:
        p = a(f'partial_L{L}')
        se = p.std(ddof=1) / np.sqrt(len(p))
        print(f'{L:6d} {a(f"r_ens_L{L}").mean():+11.4f} {a(f"r_true_L{L}").mean():+11.4f} '
              f'{p.mean():+10.4f} {se:7.4f} {p.mean()/se:+6.2f}')
        out[f'L{L}'] = dict(partial=float(p.mean()), se=float(se),
                            r_ens=float(a(f'r_ens_L{L}').mean()),
                            r_true=float(a(f'r_true_L{L}').mean()))
    best = max(LOOKS, key=lambda L: out[f'L{L}']['partial'])
    print(f'\n  best looks = L={best}, partial = {out[f"L{best}"]["partial"]:+.4f}')
    print('  -> map carries realization info beyond the ensemble'
          if out[f'L{best}']['partial'] > 0.15 else
          '  -> NO usable realization info beyond the ensemble at any L => CLOSED')
    json.dump(dict(rows=rows, summary=out,
                   ceiling=float(a('ceiling_ens_true').mean())), open(OUT, 'w'), indent=1)
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
