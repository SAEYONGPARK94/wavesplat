"""Which of the two changes bought the gain?

d3_storm_wall --wide flipped TWO knobs at once: it widened the bank band from
[0.35, 2.5] kp to [0.15, 5.0] kp AND switched on dual_band. The in-fan error
fell 32% on the storm and 33% on the bimodal sea, and the bimodal blind wedge
went 1.076 -> 0.448. Attributing that to "band coverage" is not allowed until
the knobs are turned separately.

  base        band (0.35, 2.5), dual off
  band_only   band (0.15, 5.0), dual off
  dual_only   band (0.35, 2.5), dual on
  both        band (0.15, 5.0), dual on      (= the --wide run)

Reported on the two configs that carry the argument: the deployed fan, and the
oracle (where the model class binds because information no longer does).
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
import d3_storm_wall as SW

SEEDS = 3
VARIANTS = {
    'base':      ((0.35, 2.5), False),
    'band_only': ((0.15, 5.0), False),
    'dual_only': ((0.35, 2.5), True),
    'both':      ((0.15, 5.0), True),
}
CONFIGS = ['fan_noisy', 'all_dense']


def main():
    gx = np.arange(SF.DOM_X[0], SF.DOM_X[1] + 0.1, SF.DX)
    gy = np.arange(SF.DOM_Y[0], SF.DOM_Y[1] + 0.1, SF.DY)
    Xq, Yq = np.meshgrid(gx, gy)
    t0 = SF.W_OBS
    Pq = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(Xq.size, t0)])

    for sea in ('storm', 'bimodal'):
        spec = SW.SEAS[sea]
        print(f"\n=== {sea} ===")
        print(f"{'variant':<11} {'config':<11} {'in_fan':>14} {'blind_near':>14}")
        for vname, (band, dual) in VARIANTS.items():
            for cfg in CONFIGS:
                infan, blind = [], []
                for sd in range(SEEDS):
                    eta, kp, Hs = SW.make_sea(spec, 1000 + 37 * sd)
                    gt = eta(Xq.ravel(), Yq.ravel(), t0).reshape(Xq.shape)
                    rng = np.random.default_rng(11 + sd)
                    P, Z = SW.observations(eta, cfg, rng, t0)
                    fq = mmse_bank_estimate(P, Z, spec['depth'], Pq,
                                            sigma0=0.05 * Hs, kp_hint=kp,
                                            seed=sd, band=band,
                                            dual_band=dual)
                    s = SW.score(gt, fq.reshape(Xq.shape), Xq, Yq)
                    infan.append(s['in_fan']); blind.append(s['blind_near'])
                print(f"{vname:<11} {cfg:<11} "
                      f"{np.mean(infan):.3f}±{np.std(infan):.3f}  "
                      f"{np.mean(blind):.3f}±{np.std(blind):.3f}", flush=True)


if __name__ == '__main__':
    sys.exit(main())
