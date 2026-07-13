"""E-6c: what actually splits BF2's seeds -- the wavenumber bank, not the sensor.

Registered in docs/E2_MESH_AUDIT_prereg_2026-07-13.md §E-6 as the next test after
the sampling hypothesis was rejected: "시드별 식별 오차 |Δk|/k 를 rho와 대조 --
이봉이 식별 실패로 설명되는지."

E-6 rejected sampling: BF2's five seeds split into two healthy (flat to 200 m)
and three collapsed, under identical observation conditions, and pouring 48x more
points on them changes nothing.  The remaining suspect is the estimator's own
wavenumber bank: it is a discrete grid, and a sea whose true k falls between grid
points cannot be represented no matter how well it is observed.

Prediction: |Δk|/k to the nearest bank node ranks the seeds exactly as rho does.
"""
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_e6_sampling_test import FINE, SEEDS, fit, load, rho_in  # noqa: E402
from d3_g0b_selection_aware import BAND, DEV                    # noqa: E402
from d3_gpu_mmse import build_bank                              # noqa: E402

OUT = 'results/direction3/e6c_bank_quantisation.json'
NAME = 'episode_BF2_1comp_seed{}_fixedT.h5'


def main():
    rows = []
    for s in SEEDS:
        p = os.path.join(FINE, NAME.format(s))
        ep = load(p, np.random.default_rng(11 + s), 2000)
        bank, a = fit(ep)
        fan = rho_in(ep, bank, a, 15.0, 200.0)

        with h5py.File(p, 'r') as h:
            c = h['config/wave_components'][:]
            i = int(np.argmax(c[:, 0]))
            kx_t, ky_t = float(c[i, 2]), float(c[i, 3])

        kxb, kyb, _ = build_bank(ep['kp'], BAND, ep['kp'] / 10.0, 50.0, DEV,
                                 torch.float32)
        kxb, kyb = kxb.cpu().numpy(), kyb.cpu().numpy()
        dk = float(np.hypot(kxb - kx_t, kyb - ky_t).min()
                   / np.hypot(kx_t, ky_t))
        rows.append(dict(seed=s, dk_over_k=dk, fan=fan,
                         healthy=bool(fan > 0.6)))
        print(f'  seed{s}: |dk|/k = {dk:.4f}   fan rho = {fan:+.3f}   '
              f'{"healthy" if fan > 0.6 else "collapsed"}', flush=True)

    dk = np.array([r['dk_over_k'] for r in rows])
    fan = np.array([r['fan'] for r in rows])
    # Spearman = Pearson on ranks (scipy-free)
    rk_d = np.argsort(np.argsort(dk))
    rk_f = np.argsort(np.argsort(-fan))
    rho_s = float(np.corrcoef(rk_d, rk_f)[0, 1])

    res = dict(
        date='2026-07-13',
        prereg='docs/E2_MESH_AUDIT_prereg_2026-07-13.md',
        question='does bank quantisation explain the BF2 bimodality that the '
                 'sampling hypothesis (E-6) failed to explain?',
        seeds=rows,
        spearman_dk_vs_fan=rho_s,
        rank_order_by_dk=[int(rows[i]['seed']) for i in np.argsort(dk)],
        fan_in_that_order=[float(rows[i]['fan']) for i in np.argsort(dk)],
        verdict=('bank quantisation ranks the seeds exactly as performance does'
                 if abs(rho_s) > 0.99 else 'no clean rank correspondence'))
    json.dump(res, open(OUT, 'w'), indent=1)

    print(f'\n{"=" * 62}\n[E-6c] BF2 bimodality vs wavenumber-bank quantisation')
    print(f'  seeds ordered by |dk|/k : {res["rank_order_by_dk"]}')
    print('  their fan rho           : '
          + ', '.join(f'{v:+.2f}' for v in res['fan_in_that_order']))
    print(f'  Spearman(|dk|/k, -rho)  : {rho_s:+.2f}  (n=5; exact order by '
          f'chance = 1/120)')
    print(f'  -> {res["verdict"]}')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
