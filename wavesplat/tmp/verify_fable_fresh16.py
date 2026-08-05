"""Fresh-sea replication (si=16..31, never used in the e29/e30 arc).
Arms at matched solver (free arm, smult=3): cur(full incumbent path),
own(net gamma), mean(constant template FROM SEAS 0-15, out-of-sample),
ens_blind, ens_oracle, real_free."""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (build_sea_c, deposit_grid, bank_atoms, design_t,
                          corr, DEVICE, DEPTH, _MASK, REGIMES,
                          incumbent_field)
from e29_gate0_ladder import universal_radial, ensemble_grid, solve_with_grid
from e30_conditionality_probe import rotate_map, net_gamma_map
from d18_relax_estimator import beamform_top
from d3_sa_achievability_gap import hashabs

OUT = 'results/direction3/verify_fable_fresh16.json'


def main(si_lo=16, si_hi=32):
    kmag_u, en_u = universal_radial()
    R = REGIMES['spread']

    # constant template from the ORIGINAL 16 seas (out-of-sample for si>=16)
    maps0, mus0 = [], []
    for si in range(16):
        S = build_sea_c('spread', si)
        Y = net_gamma_map(S)
        bx, by = beamform_top(S['P'].astype(np.float32),
                              S['z'].astype(np.float32),
                              DEPTH, klim=3.5 * S['kp'], n=120)
        maps0.append(Y); mus0.append(np.arctan2(by, bx))
    mean_aligned = np.stack([rotate_map(maps0[i], -mus0[i])
                             for i in range(16)]).mean(0)
    print('template from seas 0-15 done', flush=True)

    rows = []
    for si in range(si_lo, si_hi):
        S = build_sea_c('spread', si)
        rng = np.random.default_rng(7000 + 31 * si + hashabs('spread'))
        _ = rng.uniform(*R['hs'])
        mu_true = np.deg2rad(rng.uniform(0, 360))
        Y = net_gamma_map(S)
        bx, by = beamform_top(S['P'].astype(np.float32),
                              S['z'].astype(np.float32),
                              DEPTH, klim=3.5 * S['kp'], n=120)
        mu = np.arctan2(by, bx)
        g_own = np.maximum(10.0 ** Y, 1e-8)
        g_mean = np.maximum(10.0 ** rotate_map(mean_aligned, mu), 1e-8)
        g_ensb = ensemble_grid(kmag_u, en_u, mu, S['kp'])
        g_enso = ensemble_grid(kmag_u, en_u, mu_true, S['kp'])
        g_real = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, S['kp'])
        f = incumbent_field(S['P'], S['z'], S['Pq'], S['kp'], S['sigma0'])
        r = dict(si=si,
                 cur=float(corr(f, S['gt_u'])),
                 own=solve_with_grid(S, g_own),
                 mean=solve_with_grid(S, g_mean),
                 ensb=solve_with_grid(S, g_ensb),
                 enso=solve_with_grid(S, g_enso),
                 real=solve_with_grid(S, g_real),
                 derr=float(np.rad2deg(abs(np.angle(np.exp(1j * (mu - mu_true)))))))
        rows.append(r)
        print(f"sea{si} cur {r['cur']:+.3f} own {r['own']:+.3f} mean {r['mean']:+.3f} "
              f"ensb {r['ensb']:+.3f} enso {r['enso']:+.3f} real {r['real']:+.3f} "
              f"derr {r['derr']:5.1f}", flush=True)

    a = lambda k: np.array([r[k] for r in rows], float)

    def tst(d, lab):
        m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {lab:18s} {m:+.4f} +- {se:.4f} t {m/se:+.2f} pos {(d>0).sum()}/{len(d)}")

    print(f"\n=== FRESH seas {si_lo}..{si_hi-1} (n={len(rows)}) ===")
    print(f"  cur {a('cur').mean():+.4f} own {a('own').mean():+.4f} mean {a('mean').mean():+.4f} "
          f"ensb {a('ensb').mean():+.4f} enso {a('enso').mean():+.4f} real {a('real').mean():+.4f}")
    tst(a('mean') - a('own'), 'B-A (mean-own)')
    tst(a('ensb') - a('cur'), 'ens_blind - cur')
    tst(a('enso') - a('cur'), 'ens_oracle - cur')
    tst(a('real') - a('cur'), 'real_free - cur')
    tst(a('own') - a('ensb'), 'own - ensb')
    json.dump(rows, open(OUT, 'w'), indent=1, default=float)
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()
