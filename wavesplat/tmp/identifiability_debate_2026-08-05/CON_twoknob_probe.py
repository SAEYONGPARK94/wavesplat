"""CON: is there a contrast that moves WITHIN-frequency spread with N, n_freq, M,
per-atom amplitude, per-bin energy and TOTAL spread all held fixed?

Standalone re-implementation of the direction draw only.  Production code is NOT
touched.  Everything except the `directions` line is taken from
wavesim.wave_generator unchanged.
"""
import json, os, sys
import numpy as np
ROOT = '/home/vexit/wavesplat/wavesplat'
sys.path.insert(0, ROOT)
from wavesim.wave_generator import jonswap_components

FMIN, FMAX = 0.04, 0.50
Hs, Tp, DEPTH = 1.2, 7.0, 50.0
SEEDS = list(range(20))          # 20 seeds: this is component algebra, not a wave run


def s1deg(r1):
    r1 = min(max(float(r1), 0.0), 1.0)
    return float(np.degrees(np.sqrt(max(2.0 * (1.0 - r1), 0.0))))


def decompose(A, om, th):
    fr = om / (2 * np.pi); E = 0.5 * A ** 2
    m = (fr >= FMIN) & (fr <= FMAX)
    A, om, th, E = A[m], om[m], th[m], E[m]
    w = E / E.sum()
    r1_tot = float(np.hypot((w*np.cos(th)).sum(), (w*np.sin(th)).sum()))
    r1_w, Ew = 0.0, 0.0
    for o in np.unique(om):
        b = om == o
        wb = E[b] / E[b].sum()
        r1_w += E[b].sum() * float(np.hypot((wb*np.cos(th[b])).sum(), (wb*np.sin(th[b])).sum()))
        Ew += E[b].sum()
    r1_w /= Ew
    return dict(within=s1deg(r1_w), across=s1deg(r1_tot / r1_w if r1_w > 0 else np.nan),
                total=s1deg(r1_tot), nbins=int(np.unique(om).size))


def two_knob(N, M, sig_within_deg, sig_across_deg, seed, mean_deg=0.0):
    """theta_ij = mu + sig_a * a_i + sig_w * w_ij, with a_i, w_ij ~ N(0,1) drawn
    ONCE per seed and REUSED for every (sig_w, sig_a) -> arms are bit-coupled."""
    base = jonswap_components(Hs=Hs, Tp=Tp, n_components=N, depth=DEPTH,
                              direction_spread_deg=30.0, seed=seed, dirs_per_freq=M)
    A = np.array([c.A for c in base]); om = np.array([c.omega for c in base])
    n_freq = N // M
    rng = np.random.default_rng(10_000 + seed)      # separate stream, fixed per seed
    a_std = rng.standard_normal(n_freq)
    w_std = rng.standard_normal(N)
    bin_idx = np.repeat(np.arange(n_freq), M)
    th = (np.deg2rad(mean_deg) + np.deg2rad(sig_across_deg) * a_std[bin_idx]
          + np.deg2rad(sig_within_deg) * w_std)
    return A, om, th


print('=== LADDER 1 (what HEAD can already do): fixed n_freq=30, vary N ===')
for N, M in [(30, 1), (60, 2), (90, 3), (150, 5), (300, 10)]:
    r = [decompose(*(lambda c: (np.array([x.A for x in c]), np.array([x.omega for x in c]),
                                np.arctan2([x.ky for x in c], [x.kx for x in c])))(
            jonswap_components(Hs=Hs, Tp=Tp, n_components=N, depth=DEPTH,
                               direction_spread_deg=30.0, seed=s, dirs_per_freq=M)))
         for s in SEEDS]
    print(f'  N={N:3d} M={M:2d} n_freq={N//M}  within={np.median([x["within"] for x in r]):6.2f}  '
          f'across={np.median([x["across"] for x in r]):6.2f}  total={np.median([x["total"] for x in r]):6.2f}')

print('\n=== LADDER 2 (needs the patch): N=90, n_freq=30, M=3 FIXED, sweep sig_w with sig_a^2+sig_w^2=const ===')
TOT = 30.0
rows = []
for sw in [0.0, 5.0, 10.0, 15.0, 20.0, 24.0, 27.0, 30.0]:
    sa = float(np.sqrt(max(TOT**2 - sw**2, 0.0)))
    r = [decompose(*two_knob(90, 3, sw, sa, s)) for s in SEEDS]
    row = dict(sig_w=sw, sig_a=round(sa, 2),
               within=float(np.median([x['within'] for x in r])),
               across=float(np.median([x['across'] for x in r])),
               total=float(np.median([x['total'] for x in r])),
               total_iqr=float(np.percentile([x['total'] for x in r], 75)
                               - np.percentile([x['total'] for x in r], 25)))
    rows.append(row)
    print(f"  sig_w={sw:5.1f} sig_a={sa:5.2f} -> within={row['within']:6.2f}  "
          f"across={row['across']:6.2f}  total={row['total']:6.2f} (IQR {row['total_iqr']:.2f})")

print('\n  amplitude / omega arrays identical across the whole ladder?')
A0, om0, _ = two_knob(90, 3, 0.0, 30.0, 0)
A1, om1, _ = two_knob(90, 3, 27.0, 13.08, 0)
print('   A bit-identical:', bool(np.array_equal(A0, A1)),
      '| omega bit-identical:', bool(np.array_equal(om0, om1)))

print('\n=== LADDER 3: reach the MEASURED level (within 29.28, across 21.78, total 35.35) ===')
for sw, sa in [(30, 22), (32, 24), (34, 26), (36, 26), (38, 28), (40, 30)]:
    r = [decompose(*two_knob(90, 3, sw, sa, s)) for s in SEEDS]
    print(f'  sig_w={sw:4.1f} sig_a={sa:4.1f} -> within={np.median([x["within"] for x in r]):6.2f}  '
          f'across={np.median([x["across"] for x in r]):6.2f}  total={np.median([x["total"] for x in r]):6.2f}')

print('\n=== calibration: what single-knob spread makes N=90,M=3 match A total sigma1 = 24.18? ===')
for kn in [24.0, 25.0, 26.0, 27.0, 28.0, 30.0]:
    r = []
    for s in SEEDS:
        c = jonswap_components(Hs=Hs, Tp=Tp, n_components=90, depth=DEPTH,
                               direction_spread_deg=kn, seed=s, dirs_per_freq=3)
        r.append(decompose(np.array([x.A for x in c]), np.array([x.omega for x in c]),
                           np.arctan2([x.ky for x in c], [x.kx for x in c])))
    print(f'  knob={kn:5.1f} -> within={np.median([x["within"] for x in r]):6.2f} '
          f'across={np.median([x["across"] for x in r]):6.2f} total={np.median([x["total"] for x in r]):6.2f}')

json.dump(rows, open(os.path.join(ROOT, 'tmp/identifiability_debate_2026-08-05/CON_twoknob_probe.json'), 'w'), indent=1)
