"""NEUTRAL probe: is "within-frequency spread == 0 for M=1" a property of the
generator, or of the BINNING OPERATOR used to score it?

The manipulation check bins the generator by EXACT omega equality (zero-width
bins).  The CDIP reference quantity is computed from directional moments inside
FINITE-WIDTH frequency bands (0.005 Hz below ~0.09 Hz, 0.010 Hz above; 38 bands
inside this generator's [0.5, 3]*omega_p range).  Those are different operators.

P5  Apply the CDIP band operator to M=1 generator output at increasing N.
P6  Where does within/across sit as a function of M alone (total is one knob)?

Component construction only.  No wave field, no episode read.
"""
import json
import os
import sys

import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
HERE = os.path.join(ROOT, 'tmp/identifiability_debate_2026-08-05')
sys.path.insert(0, ROOT)
from wavesim.wave_generator import jonswap_components  # noqa: E402

CDIP_F = np.load(os.path.join(
    ROOT, 'tmp/depth3_ceiling_2026-08-05/spec_028p1.npz'))['f']
# band edges from the CDIP centre grid
edges = np.concatenate([[CDIP_F[0] - (CDIP_F[1] - CDIP_F[0]) / 2],
                        0.5 * (CDIP_F[:-1] + CDIP_F[1:]),
                        [CDIP_F[-1] + (CDIP_F[-1] - CDIP_F[-2]) / 2]])
FMIN, FMAX = 0.04, 0.50
SEEDS = list(range(8))


def s1deg(r1):
    r1 = min(max(float(r1), 0.0), 1.0)
    return float(np.degrees(np.sqrt(max(2.0 * (1.0 - r1), 0.0))))


def decompose(comps, band_edges):
    A = np.array([c.A for c in comps])
    om = np.array([c.omega for c in comps])
    th = np.arctan2([c.ky for c in comps], [c.kx for c in comps])
    fr = om / (2 * np.pi)
    E = 0.5 * A ** 2
    m = (fr >= FMIN) & (fr <= FMAX)
    fr, th, E = fr[m], th[m], E[m]
    w = E / E.sum()
    r1_tot = float(np.hypot((w * np.cos(th)).sum(), (w * np.sin(th)).sum()))
    if band_edges is None:                       # exact-omega (zero width)
        key = om[m]
    else:
        key = np.digitize(fr, band_edges)
    r1_within, Ew, occ = 0.0, 0.0, []
    for o in np.unique(key):
        b = key == o
        occ.append(int(b.sum()))
        wb = E[b] / E[b].sum()
        r1_b = float(np.hypot((wb * np.cos(th[b])).sum(), (wb * np.sin(th[b])).sum()))
        r1_within += E[b].sum() * r1_b
        Ew += E[b].sum()
    r1_within /= Ew
    r1_ac = r1_tot / r1_within if r1_within > 0 else np.nan
    return dict(n_bins=len(occ), max_occupancy=max(occ),
                frac_bins_multi=float(np.mean([o > 1 for o in occ])),
                within=s1deg(r1_within), across=s1deg(r1_ac), total=s1deg(r1_tot))


out = {'cdip_band_edges_n': int(edges.size),
       'cdip_bands_in_generator_range': int(((CDIP_F >= 0.0714) & (CDIP_F <= 0.4286)).sum()),
       'note': 'component construction only; no wave field simulated'}

print('== P5  M=1 generator scored with the CDIP FINITE-WIDTH band operator')
print('   (the same M=1 output scored with exact-omega bins gives within=0 by construction)')
rows = []
for N in (30, 60, 120, 200, 400):
    ex, cd = [], []
    for sd in SEEDS:
        c = jonswap_components(Hs=1.2, Tp=7.0, n_components=N, depth=50.0,
                               direction_spread_deg=30.0, seed=sd, dirs_per_freq=1)
        ex.append(decompose(c, None))
        cd.append(decompose(c, edges))
    r = dict(N=N, M=1,
             exact_within=float(np.median([x['within'] for x in ex])),
             cdip_within=float(np.median([x['within'] for x in cd])),
             cdip_across=float(np.median([x['across'] for x in cd])),
             cdip_total=float(np.median([x['total'] for x in cd])),
             cdip_max_occupancy=int(np.median([x['max_occupancy'] for x in cd])),
             cdip_frac_bins_multi=float(np.median([x['frac_bins_multi'] for x in cd])))
    rows.append(r)
    print(f"   N={N:4d} M=1 | exact-omega within={r['exact_within']:5.2f}  ||  "
          f"CDIP-band within={r['cdip_within']:6.2f}  across={r['cdip_across']:6.2f}  "
          f"total={r['cdip_total']:6.2f}  (max {r['cdip_max_occupancy']} comps/band, "
          f"{100*r['cdip_frac_bins_multi']:.0f}% bands multi)")
out['P5_operator'] = rows

print('\n== P6  within/across split as a function of M alone (N=30 fixed, exact-omega bins)')
rows6 = []
for M in (1, 2, 3, 5, 6, 10, 15):
    rr = []
    for sd in SEEDS:
        c = jonswap_components(Hs=1.2, Tp=7.0, n_components=30, depth=50.0,
                               direction_spread_deg=30.0, seed=sd, dirs_per_freq=M)
        rr.append(decompose(c, None))
    r = dict(M=M, n_freq=30 // M,
             within=float(np.median([x['within'] for x in rr])),
             across=float(np.median([x['across'] for x in rr])),
             total=float(np.median([x['total'] for x in rr])))
    rows6.append(r)
    print(f"   M={M:2d} n_freq={r['n_freq']:2d} | within={r['within']:6.2f}  "
          f"across={r['across']:6.2f}  total={r['total']:6.2f}")
out['P6_M_sweep'] = rows6
# asymptotic sigma_1 of the direction law itself
sig = np.deg2rad(30.0)
r1_law = float(np.exp(-0.5 * sig ** 2))
out['direction_law_sigma1_deg'] = s1deg(r1_law)
print(f"\n   sigma_1 of the direction LAW itself (spread=30 deg) = "
      f"{s1deg(r1_law):.2f} deg  <- the ceiling that within and across share")
print(f"   CDIP measured within median = 29.28 deg")

json.dump(out, open(os.path.join(HERE, 'NEUTRAL_operator.json'), 'w'), indent=2)
print(f"\nwrote {os.path.join(HERE,'NEUTRAL_operator.json')}")
