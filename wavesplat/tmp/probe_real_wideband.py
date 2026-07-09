"""Does the band lever transfer to the real records?

Synthetic seas with out-of-band energy gain 28-34% in-fan from widening the
bank from [0.35, 2.5] kp to [0.15, 5.0] kp, and the whole gain is the band
(dual_band contributes nothing; tmp/probe_band_ablation.py). If the same knob
moves the real records, the storm's problem was the ruler after all — just not
in the way C1' argued.

Two records, two observation sets, two bands. The oracle here is the same
suspect stereo GT that made C1's verdict circular, so it is reported for
comparison only, not as evidence.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

from d3_sa_nomp import mmse_bank_estimate
import d3_static_field as SF
import d3_static_limit as SL

BANDS = {'base': (0.35, 2.5), 'wide': (0.15, 5.0)}
CONFIGS = ['fan_noisy', 'all_dense']


def main():
    for rec in ('YS17', 'LJ18g'):
        sc = SF.scene_real(rec)
        Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
        eta = sc['make_eta']()
        gt = eta(Xq.ravel(), Yq.ravel(), sc['t0']).reshape(Xq.shape)
        Pq = np.column_stack([Xq.ravel(), Yq.ravel(),
                              np.full(Xq.size, sc['t0'])])
        cx, cy = sc['center']
        inside, R = SF.fan_mask(Xq, Yq, cx, cy, sc['hvec'])
        beyond = (R > SF.R_FAN[1]) & ~inside
        blind = ~inside & ~beyond
        sg = float(np.nanstd(gt))
        print(f"\n=== {rec}  (Hs {sc['Hs']:.1f} m, kp {sc['kp']:.4f}, "
              f"lam_p {2*np.pi/sc['kp']:.0f} m) ===")
        print(f"{'band':<6} {'config':<11} {'in_fan':>8} {'blind':>8} "
              f"{'beyond':>8}")
        for bname, band in BANDS.items():
            for cfg in CONFIGS:
                rng = np.random.default_rng(11)
                P, Z, noise = SL.resample(sc, cfg, rng)
                fq = mmse_bank_estimate(P, Z, sc['depth'], Pq,
                                        sigma0=0.05 * sc['Hs'],
                                        kp_hint=sc['kp'], seed=sc['seed'],
                                        band=band)
                pr = fq.reshape(Xq.shape)

                def nr(m):
                    mm = m & np.isfinite(pr) & np.isfinite(gt)
                    return float(np.sqrt(np.mean((pr[mm] - gt[mm]) ** 2)) / sg)
                print(f"{bname:<6} {cfg:<11} {nr(inside):>8.3f} "
                      f"{nr(blind):>8.3f} {nr(beyond):>8.3f}", flush=True)


if __name__ == '__main__':
    sys.exit(main())
