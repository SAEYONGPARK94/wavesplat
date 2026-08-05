"""Manipulation check for the S4 arm: realized within / across / total sigma_1.

Component construction only -- no wave field is simulated, no episode is read.
Estimator is the one used on the CDIP records in tmp/depth3_ceiling_2026-08-05
(generator_side.py / CON_ceiling_math.py), reproduced here so the arm and the
measured records are scored by identical code:

  sigma_1 = sqrt(2 * (1 - r_1))                                     [radians]
  r1_total  = | sum_i w_i exp(i theta_i) |            over band-passed comps
  r1_within = energy-weighted mean over bins of |sum_{i in bin} w exp(i th)|
  r1_across = r1_total / r1_within                    (circular decomposition)

Run:  python3 tmp/s4_within_spread_2026-08-05/manip_check.py
"""
import json
import os
import sys

import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
HERE = os.path.join(ROOT, 'tmp/s4_within_spread_2026-08-05')
sys.path.insert(0, ROOT)
from wavesim.wave_generator import jonswap_components, check_q_constraint  # noqa: E402

FMIN, FMAX = 0.04, 0.50          # same band mask as the CDIP-side estimator
SEA = dict(BF4=dict(Hs=1.2, Tp=7.0))
SEEDS = [0, 1, 2, 3, 4]
DEPTH = 50.0
SPREAD = 30.0                    # unchanged from every prior episode


def s1deg(r1):
    r1 = min(max(float(r1), 0.0), 1.0)
    return float(np.degrees(np.sqrt(max(2.0 * (1.0 - r1), 0.0))))


def decompose(comps):
    A = np.array([c.A for c in comps])
    om = np.array([c.omega for c in comps])
    th = np.arctan2([c.ky for c in comps], [c.kx for c in comps])
    fr = om / (2 * np.pi)
    E = 0.5 * A ** 2
    m = (fr >= FMIN) & (fr <= FMAX)
    A, om, th, E = A[m], om[m], th[m], E[m]
    w = E / E.sum()
    r1_tot = float(np.hypot((w * np.cos(th)).sum(), (w * np.sin(th)).sum()))

    # Bin by exact omega -- the generator places components on discrete bin
    # centres, so identical omega IS "same frequency bin" with no tolerance
    # choice to make.
    r1_within, Ew = 0.0, 0.0
    uniq = np.unique(om)
    for o in uniq:
        b = om == o
        wb = E[b] / E[b].sum()
        r1_b = float(np.hypot((wb * np.cos(th[b])).sum(), (wb * np.sin(th[b])).sum()))
        r1_within += E[b].sum() * r1_b
        Ew += E[b].sum()
    r1_within /= Ew
    r1_across = r1_tot / r1_within if r1_within > 0 else np.nan
    return dict(n_comp=int(A.size), n_freq_bins=int(uniq.size),
                Hs=float(4.0 * np.sqrt((A ** 2 / 2.0).sum())),
                sigma1_within_deg=s1deg(r1_within),
                sigma1_across_deg=s1deg(r1_across),
                sigma1_total_deg=s1deg(r1_tot),
                r1_within=r1_within, r1_across=r1_across, r1_total=r1_tot,
                decomp_resid_pct=float(abs(r1_within * r1_across - r1_tot)
                                       / r1_tot * 100))


# Arms of the pre-registered design.  (label, n_components, dirs_per_freq)
ARMS = [('A_M1_N30', 30, 1), ('B_M3_N30', 30, 3), ('C_M1_N10', 10, 1)]

# Measured CDIP reference (tmp/depth3_ceiling_2026-08-05/ceiling.json, all6)
REAL = dict(within=29.28, across=21.78, total=35.35)

out = {'note': 'component construction only; no wave field simulated',
       'band_hz': [FMIN, FMAX], 'direction_spread_deg': SPREAD,
       'real_cdip_all6_deg': REAL, 'arms': {}}

for label, ncomp, M in ARMS:
    rows = []
    for sd in SEEDS:
        c = jonswap_components(Hs=SEA['BF4']['Hs'], Tp=SEA['BF4']['Tp'],
                               n_components=ncomp, depth=DEPTH,
                               direction_spread_deg=SPREAD, seed=sd,
                               dirs_per_freq=M)
        ok, q = check_q_constraint(c)
        r = decompose(c)
        r.update(seed=sd, q_ok=bool(ok), q_sum=float(q))
        rows.append(r)
    out['arms'][label] = dict(n_components=ncomp, dirs_per_freq=M, per_seed=rows,
                              median={k: float(np.median([r[k] for r in rows]))
                                      for k in ('sigma1_within_deg', 'sigma1_across_deg',
                                                'sigma1_total_deg', 'Hs', 'decomp_resid_pct')})
    m = out['arms'][label]['median']
    print(f"{label:10s} N={ncomp:3d} M={M}  bins={rows[0]['n_freq_bins']:3d}  "
          f"within={m['sigma1_within_deg']:6.2f}  across={m['sigma1_across_deg']:6.2f}  "
          f"total={m['sigma1_total_deg']:6.2f}  Hs={m['Hs']:.4f}  "
          f"Qmax={max(r['q_sum'] for r in rows):.3f}  "
          f"resid={m['decomp_resid_pct']:.3f}%")

print(f"\nCDIP measured (all6): within={REAL['within']}  across={REAL['across']}  "
      f"total={REAL['total']}")
json.dump(out, open(os.path.join(HERE, 'manip_check.json'), 'w'), indent=2)
