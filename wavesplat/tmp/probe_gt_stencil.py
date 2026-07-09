"""P0 — is the GT of alpha_wave noise-dominated by its own finite differences?

Every number in REQUIRED_HORIZON_persona_debate rests on a GT slope and w that
we manufactured with a +-2 m / +-0.1 s stencil on top of grid-interpolated
(HOS) or stereo-measured (real) surfaces. If moving the stencil moves the GT
by as much as the estimator error, then the comparison was measuring the
stencil, not the estimators.

Test: recompute GT alpha_wave with several stencils and score each against the
reference stencil (2 m, 0.1 s), normalized by the SAME reference std used in
the main experiment. Disagreement of order the estimator nRMSE (0.42 for YS17,
0.92 for LJ18g) would invalidate those numbers.

No solves — GT only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

import d3_required_horizon as RH

REF = (2.0, 0.1)
# real stereo frames are 0.1 s apart, so h_t < 0.1 is not available there
STENCILS_SIM = [(1.0, 0.1), (2.0, 0.1), (4.0, 0.1), (2.0, 0.05), (2.0, 0.2)]
STENCILS_REAL = [(1.0, 0.1), (2.0, 0.1), (4.0, 0.1), (2.0, 0.2)]

# estimator error at 0-0.5s, for scale comparison (required_horizon.json)
EST_ERR = {'linear': 0.666, 'hos': 0.330, 'real_YS17': 0.424,
           'real_LJ18g': 0.916}


def gt_alpha(case, rep, hx, ht, record='YS17'):
    RH.H_X, RH.H_T = hx, ht
    sc = RH.build_sim(case, rep) if case in ('linear', 'hos') \
        else RH.build_real(record, rep)
    gt, ref = RH.gt_series(sc, [RH.LEADS, RH.REF_LEADS])
    return gt, np.nanstd(ref, axis=0)


def main():
    cases = [('linear', 3, None), ('hos', 2, None),
             ('real', 3, 'YS17'), ('real', 3, 'LJ18g')]
    print('GT alpha_wave 스텐실 민감도 — 기준 (2 m, 0.1 s) 대비 불일치 nRMSE')
    print('(추정기 오차와 같은 규모면 = 우리는 스텐실을 재고 있었던 것)\n')

    for case, reps, record in cases:
        key = case if record is None else f'real_{record}'
        stens = STENCILS_REAL if case == 'real' else STENCILS_SIM
        print(f"=== {key}  (추정기 0-0.5s alpha nRMSE = {EST_ERR[key]:.3f}) ===")
        rows = {s: [] for s in stens if s != REF}
        for rep in range(reps):
            ref_gt, norm = gt_alpha(case, rep, *REF, record=record or 'YS17')
            ai = RH.QNAMES.index('alpha_wave')
            for s in stens:
                if s == REF:
                    continue
                gt, _ = gt_alpha(case, rep, *s, record=record or 'YS17')
                d = gt[:, ai] - ref_gt[:, ai]
                ok = np.isfinite(d)
                rows[s].append(float(np.sqrt(np.mean(d[ok] ** 2)) / norm[ai]))
        for s, v in rows.items():
            m, sd = np.mean(v), np.std(v)
            flag = '  <-- 추정기 오차급!' if m > 0.5 * EST_ERR[key] else ''
            print(f"  stencil ({s[0]:.2f} m, {s[1]:.2f} s): "
                  f"불일치 {m:.3f}±{sd:.3f}{flag}")
        print()
    RH.H_X, RH.H_T = REF


if __name__ == '__main__':
    sys.exit(main())
