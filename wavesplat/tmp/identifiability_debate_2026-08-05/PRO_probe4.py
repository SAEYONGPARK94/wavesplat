"""PROBE 4 -- escape (c) energy redistribution, and calibration of the
recommended contrast.  Component construction only.
"""
import json
import os
import sys

import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
HERE = os.path.join(ROOT, 'tmp/identifiability_debate_2026-08-05')
sys.path.insert(0, ROOT)
from wavesim.wave_generator import jonswap_components  # noqa

HS, TP, DEPTH, SPREAD = 1.2, 7.0, 50.0, 30.0
SEEDS = [0, 1, 2, 3, 4]
FMIN, FMAX = 0.04, 0.50


def s1deg(r1):
    r1 = min(max(float(r1), 0.0), 1.0)
    return float(np.degrees(np.sqrt(max(2.0 * (1.0 - r1), 0.0))))


def decompose_arrays(A, om, th):
    fr = om / (2 * np.pi)
    E = 0.5 * A ** 2
    m = (fr >= FMIN) & (fr <= FMAX)
    A, om, th, E = A[m], om[m], th[m], E[m]
    w = E / E.sum()
    r1_tot = float(np.hypot((w * np.cos(th)).sum(), (w * np.sin(th)).sum()))
    r1_within, Ew = 0.0, 0.0
    for o in np.unique(om):
        b = om == o
        wb = E[b] / E[b].sum()
        r1_within += E[b].sum() * float(np.hypot((wb * np.cos(th[b])).sum(),
                                                 (wb * np.sin(th[b])).sum()))
        Ew += E[b].sum()
    r1_within /= Ew
    return dict(within=s1deg(r1_within),
                across=s1deg(r1_tot / r1_within if r1_within > 0 else np.nan),
                total=s1deg(r1_tot),
                Hs=float(4.0 * np.sqrt((A ** 2 / 2.0).sum())))


def hier(N, M, delta, seed, spread_deg=SPREAD):
    base = jonswap_components(Hs=HS, Tp=TP, n_components=N, depth=DEPTH,
                              direction_spread_deg=spread_deg, seed=seed,
                              dirs_per_freq=M)
    A = np.array([c.A for c in base]); om = np.array([c.omega for c in base])
    n_freq = N // M
    r = np.random.default_rng(10_000 + seed)
    b = r.normal(0., 1., n_freq); w = r.normal(0., 1., N)
    sig = np.deg2rad(spread_deg); sw = delta * sig
    sb = np.sqrt(max(sig ** 2 - sw ** 2, 0.))
    bo = np.repeat(np.arange(n_freq), M)[:N]
    return A, om, sb * b[bo] + sw * w


out = {}

# ============ ESCAPE (c): can bin-energy reweighting restore arm A's across? ==
# Arm B (N=30, M=3) has across = 15.30 deg; arm A has 24.18 deg.  Bin energies
# are the only free weights left once (N, M, grid, directions) are fixed.
# Question: what does the 1-D spectrum have to become to buy the 8.88 deg back?
print('=== PROBE 4a: escape (c) -- reachable across under bin re-weighting ===')
p4a = []
for seed in SEEDS:
    c = jonswap_components(Hs=HS, Tp=TP, n_components=30, depth=DEPTH,
                           direction_spread_deg=SPREAD, seed=seed, dirs_per_freq=3)
    A = np.array([x.A for x in c]); om = np.array([x.omega for x in c])
    th = np.array([np.arctan2(x.ky, x.kx) for x in c])
    uniq = np.unique(om)
    # bin mean directions (energy weighted, original energies)
    E = 0.5 * A ** 2
    bin_dir, bin_E = [], []
    for o in uniq:
        b = om == o
        wb = E[b] / E[b].sum()
        bin_dir.append(np.arctan2((wb * np.sin(th[b])).sum(),
                                  (wb * np.cos(th[b])).sum()))
        bin_E.append(E[b].sum())
    bin_dir = np.array(bin_dir); bin_E = np.array(bin_E)
    # across r1 under arbitrary bin weights p on the simplex.  Minimising r1
    # (= maximising across spread) is achieved by loading the two most opposed
    # bins; the exact optimum over the simplex is found by a coarse search.
    best = None
    for i in range(len(uniq)):
        for j in range(len(uniq)):
            for a in np.linspace(0, 1, 101):
                p = np.zeros(len(uniq)); p[i] += a; p[j] += 1 - a
                r1 = np.hypot((p * np.cos(bin_dir)).sum(), (p * np.sin(bin_dir)).sum())
                if best is None or r1 < best[0]:
                    best = (r1, i, j, a)
    orig = np.hypot((bin_E / bin_E.sum() * np.cos(bin_dir)).sum(),
                    (bin_E / bin_E.sum() * np.sin(bin_dir)).sum())
    p4a.append(dict(seed=seed, across_orig_deg=s1deg(orig),
                    across_max_deg=s1deg(best[0]),
                    bins_used=[int(best[1]), int(best[2])],
                    frac=float(best[3]),
                    Tp_orig=float(2 * np.pi / om[np.argmax(A)]),
                    Tp_reweighted=float(2 * np.pi / uniq[best[1] if best[3] > .5
                                                         else best[2]])))
    print(f"  seed {seed}: across {s1deg(orig):6.2f} -> max {s1deg(best[0]):6.2f} deg "
          f"but only by loading bins {best[1]},{best[2]} "
          f"(Tp {p4a[-1]['Tp_orig']:.2f}s -> {p4a[-1]['Tp_reweighted']:.2f}s, "
          f"{len(uniq)-2} of {len(uniq)} bins set to zero energy)")
out['probe4a_escape_c'] = p4a
print(f"  arm A across = 24.18 deg (target).  median reachable max = "
      f"{np.median([r['across_max_deg'] for r in p4a]):.2f} deg")

# ============ calibration of the recommended contrast =======================
# delta_lo vs delta_hi at N=30, n_freq=10, M=3, with the knob rescaled so that
# total sigma_1 matches between the two arms.
print('\n=== PROBE 4b: calibrated LO/HI pair (rank-saturated, total matched) ===')


def med_dec(delta, spread):
    rs = [decompose_arrays(*hier(30, 3, delta, s, spread_deg=spread)) for s in SEEDS]
    return {k: float(np.median([r[k] for r in rs])) for k in rs[0]}


lo = med_dec(0.40, SPREAD)
target = lo['total']
best = None
for knob in np.linspace(24.0, 32.0, 81):
    hi = med_dec(1.0, knob)
    if best is None or abs(hi['total'] - target) < abs(best[1]['total'] - target):
        best = (knob, hi)
knob, hi = best
print(f"  LO: delta=0.40 knob={SPREAD:.2f}  within={lo['within']:6.2f} "
      f"across={lo['across']:6.2f} total={lo['total']:6.2f} Hs={lo['Hs']:.4f}")
print(f"  HI: delta=1.00 knob={knob:.2f}  within={hi['within']:6.2f} "
      f"across={hi['across']:6.2f} total={hi['total']:6.2f} Hs={hi['Hs']:.4f}")
print(f"  --> within contrast = {hi['within']-lo['within']:+.2f} deg at "
      f"total matched to {abs(hi['total']-lo['total']):.3f} deg")
out['probe4b_calibrated'] = dict(lo=dict(delta=0.40, knob=SPREAD, **lo),
                                 hi=dict(delta=1.00, knob=float(knob), **hi))

json.dump(out, open(os.path.join(HERE, 'PRO_probe4.json'), 'w'), indent=2)
print('\nwrote PRO_probe4.json')
