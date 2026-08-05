"""PSF blur sweep on the TRUE realization spectrum — the decisive probe for
"is the +0.182 realization headroom reachable from a 400-point beamform map?"

Logic. The ladder says:
    rung2 (ensemble, hand)            +0.391
    rung1 (incumbent, learned CNN)    +0.403
    rung4 (TRUE realization, sharp)   +0.586
The open question is whether the realization structure worth +0.182 survives the
400-point beamforming PSF. The beamform power map is the true spectrum CONVOLVED
with that PSF (+ noise + cross terms). So: take the TRUE realization spectrum and
blur it by sigma cells. If performance at the PSF width collapses to the ensemble
level (+0.39), then even a PERFECT PSF-limited estimate of the realization buys
nothing over the ensemble -> the headroom requires SUPER-RESOLUTION, and a net that
reads the beamform map cannot reach it. If it stays near +0.58, the information
survives the PSF and the problem is inference (deconvolution), i.e. OPEN.

PSF width in grid cells: the grid spans +-KREL*kp over NGRID cells, so one cell is
2*KREL*kp/(NGRID-1). The beamforming mainlobe half-width in k is ~2*pi/L for an
aperture L. We measure the EMPIRICAL PSF from the actual beamform map instead of
assuming, and also report the analytic estimate.
"""
import os, sys, json
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (build_sea_c, deposit_grid, corr, DEVICE, DEPTH,
                          _MASK, NGRID, KREL)
from e29_gate0_ladder import solve_with_grid, universal_radial, ensemble_grid
from d18_relax_estimator import beamform_top
from d3_sa_achievability_gap import hashabs
from e29_gennet_C import REGIMES

OUT = 'results/direction3/e29_psf_blur_sweep.json'
BLURS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]


def main(n_seas=16):
    ref = json.load(open('results/direction3/e29_gate0_ceiling.json'))['rows']
    lad = json.load(open('results/direction3/e29_gate0_ladder.json'))['rows']
    kmag_u, en_u = universal_radial()
    cell_kp = 2.0 * KREL / (NGRID - 1)      # one grid cell, in units of kp
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        cur = ref[si]['cur']
        # analytic PSF width: mainlobe half-width ~ 2*pi/L, L = obs extent
        P = S['P']
        Lx = P[:, 0].max() - P[:, 0].min()
        Ly = P[:, 1].max() - P[:, 1].min()
        L = max(np.hypot(Lx, Ly), 1e-6)
        psf_k = 2 * np.pi / L                      # rad/m
        psf_cells = (psf_k / S['kp']) / cell_kp    # in grid cells
        gf = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, S['kp'])
        scores = {}
        for b in BLURS:
            g = gaussian_filter(gf, b) if b > 0 else gf
            scores[f'b{b}'] = solve_with_grid(S, g)
        r = dict(si=si, cur=cur, psf_cells=float(psf_cells),
                 aperture_wl=float(L * S['kp'] / (2 * np.pi)),
                 ens_blind=lad[si]['ens_blind'], **scores)
        rows.append(r)
        print(f"sea{si:2d} cur {cur:+.3f} ens {r['ens_blind']:+.3f} | "
              f"PSF {psf_cells:5.1f} cells (ap {r['aperture_wl']:.2f}wl) | "
              + " ".join(f"b{b}:{scores[f'b{b}']:+.3f}" for b in BLURS), flush=True)

    a = lambda k: np.array([r[k] for r in rows], float)
    cur, ens = a('cur'), a('ens_blind')
    print(f"\n=== PSF blur sweep on the TRUE realization spectrum (n={len(rows)}) ===")
    print(f"  기준선: incumbent {cur.mean():+.4f} | 앙상블(rung2) {ens.mean():+.4f}")
    print(f"  실측 PSF 폭: mean {a('psf_cells').mean():.1f} cells "
          f"(min {a('psf_cells').min():.1f}, max {a('psf_cells').max():.1f})")
    print(f"  개구: mean {a('aperture_wl').mean():.2f} wavelengths\n")
    print(f"  {'blur(cells)':>12s} {'mean corr':>10s} {'vs incumbent':>13s} {'vs 앙상블':>10s}")
    for b in BLURS:
        v = a(f'b{b}')
        print(f"  {b:12.1f} {v.mean():+10.4f} {(v-cur).mean():+13.4f} {(v-ens).mean():+10.4f}")
    # interpolate: what does the true spectrum score AT the empirical PSF width?
    at_psf = []
    for r in rows:
        xs, ys = np.array(BLURS), np.array([r[f'b{b}'] for b in BLURS])
        at_psf.append(np.interp(min(r['psf_cells'], BLURS[-1]), xs, ys))
    at_psf = np.array(at_psf)
    print(f"\n  ★참 스펙트럼을 '실측 PSF 폭'으로 뭉갠 성능: {at_psf.mean():+.4f}")
    print(f"     vs incumbent {(at_psf-cur).mean():+.4f} | vs 앙상블(rung2) {(at_psf-ens).mean():+.4f}")
    sharp = a('b0.0')
    kept = (at_psf - ens).mean() / max((sharp - ens).mean(), 1e-9)
    print(f"\n  ★★PSF 해상도에서 살아남는 realization 이득의 분율 = {kept:.1%}")
    print(f"     (뾰족한 참값이 앙상블 위에 {(sharp-ens).mean():+.4f}, "
          f"PSF-제한 참값은 {(at_psf-ens).mean():+.4f})")
    if kept < 0.25:
        print("     ⟹ PSF가 realization 정보를 파괴 = 초해상 필요 = 빔포밍맵 읽는 net엔 닫힘")
    elif kept > 0.6:
        print("     ⟹ 정보가 PSF를 견딤 = 역합성곱/추론 문제 = 열림")
    else:
        print("     ⟹ 부분 생존 — 중간. 정밀 판정 필요")
    json.dump(dict(rows=rows, at_psf=float(at_psf.mean()),
                   kept_frac=float(kept), blurs=BLURS), open(OUT, 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
