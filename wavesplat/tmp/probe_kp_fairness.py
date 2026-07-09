"""Adversarial check: was the physics estimator given information the network
never got?

COMPLEX_BEAMFORM_FIELD reports MMSE 0.505 vs the best learned model 0.623
in-fan and calls the 23% gap "the ceiling of the input representation". But
d3_complex_beamform_field.evaluate calls

    mmse_bank_estimate(..., kp_hint=kp, ...)

with kp derived from the scene's true Tp, while the network sees only c(k).
The peak wavenumber places the whole bank. If MMSE loses most of its advantage
without it, the 23% was privilege, not representation.

Runs MMSE with and without kp_hint on the same scenes, same seeds.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

from d3_sa_nomp import mmse_bank_estimate
import d3_complex_beamform_field as CBF

N = 32


def main():
    rng = np.random.default_rng(777)          # same stream as evaluate()
    rows = {'with_kp': {'in': [], 'out': []}, 'no_kp': {'in': [], 'out': []}}
    for i in range(N):
        P, Z, F, Hs, kp, t0, (Xq, Yq) = CBF.scene(rng)
        inside, outside = CBF.fan_masks(Xq, Yq)
        sg = F.std()
        Pq = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(Xq.size, t0)])
        for tag, hint in (('with_kp', kp), ('no_kp', None)):
            fq = mmse_bank_estimate(P, Z, CBF.DEPTH, Pq, sigma0=0.05 * Hs,
                                    kp_hint=hint, seed=0)
            pr = fq.reshape(CBF.NG, CBF.NG)
            for key, m in (('in', inside), ('out', outside)):
                rows[tag][key].append(
                    float(np.sqrt(np.mean((pr[m] - F[m]) ** 2)) / sg))
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{N}", flush=True)

    print(f"\n{'MMSE variant':<14} {'in-fan':>16} {'out-of-fan':>16}")
    print('-' * 50)
    for tag in ('with_kp', 'no_kp'):
        a = np.array(rows[tag]['in']); b = np.array(rows[tag]['out'])
        print(f"{tag:<14} {a.mean():.3f}±{a.std()/np.sqrt(N):.3f}   "
              f"{b.mean():.3f}±{b.std()/np.sqrt(N):.3f}")
    d = np.array(rows['no_kp']['in']) - np.array(rows['with_kp']['in'])
    print(f"\npaired penalty of losing kp_hint (in-fan): "
          f"{d.mean():+.3f} ± {d.std()/np.sqrt(N):.3f} (s.e.m.)")
    print("learned best (wiener2, no kp): 0.623   "
          f"| MMSE without kp: {np.mean(rows['no_kp']['in']):.3f}")


if __name__ == '__main__':
    sys.exit(main())
