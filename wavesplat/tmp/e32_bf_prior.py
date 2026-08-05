"""e32 — CAN A DATA-DERIVED PRIOR BEAT THE CONSTANT? (the decision-relevant test)

e30 established: the incumbent's per-sea conditioning is worth -0.013 +- 0.024
(t=-0.53). Functionally it is a CONSTANT prior + a direction read-off. So the
gate-0 ladder's "rung2 ~= rung1" is a tautology (two constants tie), and the
closure argument has no evidence for its key step.

But that only shows the net does not condition. It does NOT show conditioning
is POSSIBLE. The net's mean collapse might be CORRECT: maybe the beamform map
carries nothing about gamma beyond the direction, and the net found that out.

DECISIVE, ZERO-LEARNING TEST. The beamform power map IS a PSF-blurred
periodogram of the realization. If ANY simple function of it beats the
constant prior, per-sea conditioning demonstrably pays and the net is leaving
money on the table. If nothing does, the collapse was right.

Arms (ladder arena: free arm, smult=3 fixed, no GT):
  ens        : hand ensemble at mu_hat                     (the constant)
  bf^p       : gamma = (full-array beamform power)^p       (pure data prior)
  ens*bf^p   : ensemble MODULATED by the data map          (= what a good
               conditional net SHOULD emit: prior x evidence)
  real       : the oracle realized gamma                   (ceiling ref)
Sweep p. p=0 reduces to flat/ens, so p>0 winning = conditioning pays.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (build_sea_c, deposit_grid, corr, DEPTH, _MASK,
                          NGRID, KREL)
from e29_gate0_ladder import solve_with_grid
from e30_conditionality_probe import radial_of, ens_grid_norm
from d3_sa_nomp import beamform_power
from d18_relax_estimator import beamform_top

OUT = 'results/direction3/e32_bf_prior.json'
PS = [0.5, 1.0, 2.0, 3.0, 4.0]


def main(n_seas=16):
    kn, en = radial_of(7.0, 3.3)
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        kp = S['kp']
        bx, by = beamform_top(S['P'].astype(np.float32), S['z'].astype(np.float32),
                              DEPTH, klim=3.5 * kp, n=120)
        mu_hat = np.arctan2(by, bx)
        # full-array beamform power map on the SAME 51x51 normalized bank
        _, _, pf = beamform_power(S['P'], S['z'], DEPTH, KREL * kp, NGRID)
        pf = (pf / max(pf.max(), 1e-12)).reshape(NGRID, NGRID)

        g_ens = ens_grid_norm(kn, en, mu_hat, kp, 30.0)
        g_real = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, kp)

        r = dict(si=si)
        r['ens'] = solve_with_grid(S, g_ens)
        r['real'] = solve_with_grid(S, g_real)
        for p in PS:
            r[f'bf{p}'] = solve_with_grid(S, np.maximum(pf, 1e-8) ** p)
            r[f'ensbf{p}'] = solve_with_grid(S, g_ens * np.maximum(pf, 1e-8) ** p)
        rows.append(r)
        msg = ' | '.join(f'bf{p} {r[f"bf{p}"]:+.3f}' for p in PS)
        msg2 = ' | '.join(f'eb{p} {r[f"ensbf{p}"]:+.3f}' for p in PS)
        print(f'sea{si:2d} ens {r["ens"]:+.3f} real {r["real"]:+.3f} || {msg} || {msg2}',
              flush=True)

    a = lambda k: np.array([r[k] for r in rows], float)
    ens = a('ens')
    print(f'\n=== n={n_seas}, baseline ens (hand constant) = {ens.mean():+.4f} ===')
    print(f'{"arm":>12} {"mean":>9} {"vs ens":>9} {"se":>7} {"t":>6} {"pos":>6}')

    def show(k, lab):
        v = a(k); d = v - ens
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se > 0 else 0
        print(f'{lab:>12} {v.mean():+9.4f} {d.mean():+9.4f} {se:7.4f} {t:+6.2f} '
              f'{int((d>0).sum()):3d}/{len(d)}')
        return float(d.mean()), float(se)

    out = {}
    for p in PS:
        out[f'bf{p}'] = show(f'bf{p}', f'bf^{p}')
    print()
    for p in PS:
        out[f'ensbf{p}'] = show(f'ensbf{p}', f'ens*bf^{p}')
    print()
    out['real'] = show('real', 'real(oracle)')

    best = max([(out[f'ensbf{p}'][0], f'ens*bf^{p}') for p in PS] +
               [(out[f'bf{p}'][0], f'bf^{p}') for p in PS])
    print(f'\n  BEST data-derived arm: {best[1]} at {best[0]:+.4f} over the constant')
    print(f'  oracle realization prize: {out["real"][0]:+.4f}')
    if out['real'][0] > 0:
        print(f'  -> a zero-learning data prior recovers '
              f'{100*best[0]/out["real"][0]:.0f}% of the oracle prize')
    json.dump(dict(rows=rows, summary=out), open(OUT, 'w'), indent=1)
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
