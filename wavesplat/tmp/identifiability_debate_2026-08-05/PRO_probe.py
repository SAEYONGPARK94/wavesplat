"""PRO probes for proposition H (within-frequency spread identifiability).

No wave run.  Component construction + linear algebra on the ALREADY-STORED
sensor sample positions of the S4 arm.  Production code untouched.

Probe 1  algebra    : is within = g(M) alone in the shipped i.i.d. sampler?
Probe 2  escape b'  : does a HIERARCHICAL direction sampler move within at
                      fixed (N, n_freq, M, total sigma_1)?   <- kills or saves H
Probe 3  operator   : effective DoF of the observation operator on the REAL
                      stored (x, y, t) samples, as a function of the same knob.
                      Within-frequency components share omega -> separable only
                      by the spatial aperture; across-frequency ones also by t.
"""
import json
import os
import sys

import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
HERE = os.path.join(ROOT, 'tmp/identifiability_debate_2026-08-05')
sys.path.insert(0, ROOT)
from wavesim.wave_generator import jonswap_components, wavenumber_from_omega  # noqa

FMIN, FMAX = 0.04, 0.50
HS, TP, DEPTH, SPREAD = 1.2, 7.0, 50.0, 30.0
SEEDS = [0, 1, 2, 3, 4]
G = 9.81


# ---------------------------------------------------------------- decomposition
# byte-for-byte the estimator used in manip_check.py / the CDIP side
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
    uniq = np.unique(om)
    for o in uniq:
        b = om == o
        wb = E[b] / E[b].sum()
        r1_b = float(np.hypot((wb * np.cos(th[b])).sum(), (wb * np.sin(th[b])).sum()))
        r1_within += E[b].sum() * r1_b
        Ew += E[b].sum()
    r1_within /= Ew
    r1_across = r1_tot / r1_within if r1_within > 0 else np.nan
    return dict(n_freq_bins=int(uniq.size),
                within=s1deg(r1_within), across=s1deg(r1_across),
                total=s1deg(r1_tot),
                Hs=float(4.0 * np.sqrt((A ** 2 / 2.0).sum())))


def decompose(comps):
    A = np.array([c.A for c in comps])
    om = np.array([c.omega for c in comps])
    th = np.arctan2([c.ky for c in comps], [c.kx for c in comps])
    return decompose_arrays(A, om, th)


out = {}

# =============================================================== PROBE 1
# Shipped sampler: directions ~ iid N(mu, sigma).  Claim to test:
# within depends on M (and sigma) only, NOT on N.  If true, then at fixed
# sigma the map (N, n_freq) -> within factors through the ratio M = N/n_freq
# and P(within | N, n_freq) is a point mass  ==> positivity fails.
print('=== PROBE 1: is within = g(M) alone (shipped iid sampler)? ===')
p1 = []
for N in (10, 12, 30, 60, 90, 120):
    for M in (1, 2, 3, 5, 6):
        if M > N or N % M:
            continue
        rows = [decompose(jonswap_components(
            Hs=HS, Tp=TP, n_components=N, depth=DEPTH,
            direction_spread_deg=SPREAD, seed=s, dirs_per_freq=M))
            for s in SEEDS]
        med = {k: float(np.median([r[k] for r in rows])) for k in
               ('within', 'across', 'total', 'Hs')}
        med.update(N=N, M=M, n_freq=N // M, n_bins=rows[0]['n_freq_bins'])
        p1.append(med)
        print(f"  N={N:3d} M={M} n_freq={N//M:3d}  within={med['within']:6.2f}"
              f"  across={med['across']:6.2f}  total={med['total']:6.2f}")
out['probe1_shipped_sampler'] = p1
# within grouped by M, spread across N
byM = {}
for r in p1:
    byM.setdefault(r['M'], []).append(r['within'])
print('  within by M (range over all N tested):')
for M in sorted(byM):
    v = byM[M]
    print(f"    M={M}: n={len(v)}  min={min(v):.2f} max={max(v):.2f} "
          f"ptp={max(v)-min(v):.2f}")
out['probe1_within_by_M'] = {str(k): dict(n=len(v), min=min(v), max=max(v),
                                          ptp=max(v) - min(v))
                             for k, v in byM.items()}

# =============================================================== PROBE 2
# Escape b-prime: hierarchical sampler.  theta_i = mu + sig_b*b_{bin(i)}
#                                              + sig_w*w_i,
# with sig_b^2 + sig_w^2 = sigma_total^2 held fixed.  N, n_freq, M, Hs, the
# frequency grid, the amplitudes and the phases are ALL held fixed; only the
# partition of a fixed angular variance between the two levels moves.
# Paired across arms at the bit level: b and w draws are reused, only rescaled.
print('\n=== PROBE 2: hierarchical sampler at fixed (N, n_freq, M) ===')


def hier_components(N, M, delta, seed, spread_deg=SPREAD):
    """Return (A, omega, theta) with the shipped amplitude/frequency ladder but
    a two-level direction draw.  delta = sig_w / sigma_total in [0, 1]."""
    base = jonswap_components(Hs=HS, Tp=TP, n_components=N, depth=DEPTH,
                              direction_spread_deg=spread_deg, seed=seed,
                              dirs_per_freq=M)
    A = np.array([c.A for c in base])
    om = np.array([c.omega for c in base])
    n_freq = N // M
    # independent RNG for the two levels, fixed across the delta sweep
    r = np.random.default_rng(10_000 + seed)
    b = r.normal(0.0, 1.0, n_freq)          # bin means (standard normal)
    w = r.normal(0.0, 1.0, N)               # within-bin offsets
    sig = np.deg2rad(spread_deg)
    sig_w = delta * sig
    sig_b = np.sqrt(max(sig ** 2 - sig_w ** 2, 0.0))
    bin_of = np.repeat(np.arange(n_freq), M)[:N]
    th = sig_b * b[bin_of] + sig_w * w
    return A, om, th


p2 = []
N, M = 30, 3
for delta in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    rows = []
    for s in SEEDS:
        A, om, th = hier_components(N, M, delta, s)
        rows.append(decompose_arrays(A, om, th))
    med = {k: float(np.median([r[k] for r in rows])) for k in
           ('within', 'across', 'total', 'Hs')}
    med.update(delta=delta, N=N, M=M, n_freq=N // M)
    p2.append(med)
    print(f"  delta={delta:.1f}  within={med['within']:6.2f}  "
          f"across={med['across']:6.2f}  total={med['total']:6.2f}  "
          f"Hs={med['Hs']:.4f}  (N={N}, n_freq={N//M}, M={M} all fixed)")
out['probe2_hierarchical'] = p2

# Can the hierarchical sampler reach the CDIP measured within (29.28 deg)
# while holding total at the CDIP measured total (35.35 deg)?
print('\n  -- reach test at CDIP total (35.35 deg knob) --')
p2b = []
for M in (2, 3, 5, 6):
    for delta in (0.7, 0.8, 0.9, 1.0):
        rows = [decompose_arrays(*hier_components(60, M, delta, s,
                                                  spread_deg=35.35))
                for s in SEEDS]
        med = {k: float(np.median([r[k] for r in rows])) for k in
               ('within', 'across', 'total')}
        med.update(N=60, M=M, delta=delta, n_freq=60 // M)
        p2b.append(med)
        print(f"    N=60 M={M} n_freq={60//M:2d} delta={delta:.1f}  "
              f"within={med['within']:6.2f} across={med['across']:6.2f} "
              f"total={med['total']:6.2f}")
out['probe2b_reach'] = p2b
print('    CDIP measured: within=29.28 across=21.78 total=35.35')

json.dump(out, open(os.path.join(HERE, 'PRO_probe12.json'), 'w'), indent=2)
print('\nwrote PRO_probe12.json')
