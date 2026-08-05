"""CON: does escape hatch (b) exist?  Component construction only -- no wave run.

Question 1: is (N=30,M=1) vs (N=90,M=3) really "the same frequency grid"?
Question 2: what else moves when you take it?  (per-atom amplitude, Q, lambda_min)
Question 3: is an explicit n_freq knob even needed, or does HEAD already do it?
"""
import json, os, sys
import numpy as np
ROOT = '/home/vexit/wavesplat/wavesplat'
sys.path.insert(0, ROOT)
from wavesim.wave_generator import (jonswap_components, check_q_constraint,
                                    wavenumber_from_omega)

FMIN, FMAX = 0.04, 0.50
Hs, Tp, DEPTH, SPREAD = 1.2, 7.0, 50.0, 30.0
SEEDS = [0, 1, 2, 3, 4]


def s1deg(r1):
    r1 = min(max(float(r1), 0.0), 1.0)
    return float(np.degrees(np.sqrt(max(2.0 * (1.0 - r1), 0.0))))


def decompose(comps):
    A = np.array([c.A for c in comps]); om = np.array([c.omega for c in comps])
    th = np.arctan2([c.ky for c in comps], [c.kx for c in comps])
    fr = om / (2 * np.pi); E = 0.5 * A ** 2
    m = (fr >= FMIN) & (fr <= FMAX)
    A, om, th, E = A[m], om[m], th[m], E[m]
    w = E / E.sum()
    r1_tot = float(np.hypot((w * np.cos(th)).sum(), (w * np.sin(th)).sum()))
    r1_within, Ew = 0.0, 0.0
    for o in np.unique(om):
        b = om == o
        wb = E[b] / E[b].sum()
        r1_within += E[b].sum() * float(np.hypot((wb*np.cos(th[b])).sum(), (wb*np.sin(th[b])).sum()))
        Ew += E[b].sum()
    r1_within /= Ew
    r1_across = r1_tot / r1_within if r1_within > 0 else np.nan
    return dict(within=s1deg(r1_within), across=s1deg(r1_across), total=s1deg(r1_tot),
                nbins=int(np.unique(om).size))


CELLS = [('A  N30 M1', 30, 1), ('B  N30 M3', 30, 3), ('C  N10 M1', 10, 1),
         ('D  N90 M1', 90, 1), ('E  N90 M3', 90, 3), ('F  N90 M9', 90, 9),
         ('G  N60 M2', 60, 2), ('H  N300 M10', 300, 10)]

out = {}
for label, N, M in CELLS:
    rows = []
    for sd in SEEDS:
        c = jonswap_components(Hs=Hs, Tp=Tp, n_components=N, depth=DEPTH,
                               direction_spread_deg=SPREAD, seed=sd, dirs_per_freq=M)
        A = np.array([x.A for x in c]); om = np.array([x.omega for x in c])
        ok, q = check_q_constraint(c)
        k = np.array([x.k for x in c])
        r = decompose(c)
        r.update(seed=sd, N=N, M=M, n_freq=N // M,
                 Hs_check=float(4*np.sqrt((A**2/2).sum())),
                 var_tot=float((A**2/2).sum()),
                 A_max=float(A.max()), A_median=float(np.median(A)),
                 A_rms=float(np.sqrt((A**2).mean())),
                 q_sum=float(q), q_ok=bool(ok),
                 Qmean=float(np.mean([x.Q for x in c])),
                 lambda_min=float(2*np.pi/k.max()),
                 omega_max_center=float(om.max()), omega_min_center=float(om.min()),
                 n_uniq_omega=int(np.unique(om).size))
        rows.append(r)
    out[label] = rows
    m = rows[0]
    print(f"{label:12s} N={N:3d} M={M:2d} n_freq={N//M:3d} uniq_om={m['n_uniq_omega']:3d} "
          f"lam_min={m['lambda_min']:7.3f} A_rms={m['A_rms']:.5f} A_max={m['A_max']:.5f} "
          f"var={m['var_tot']:.5f} qsum={m['q_sum']:.4f} "
          f"within={np.median([r['within'] for r in rows]):6.2f} "
          f"across={np.median([r['across'] for r in rows]):6.2f} "
          f"total={np.median([r['total'] for r in rows]):6.2f}")

# --- grid identity test: A (N30 M1) vs E (N90 M3) -------------------------
print('\n--- GRID IDENTITY  A(N30,M1)  vs  E(N90,M3) ---')
for sd in [0]:
    ca = jonswap_components(Hs=Hs, Tp=Tp, n_components=30, depth=DEPTH,
                            direction_spread_deg=SPREAD, seed=sd, dirs_per_freq=1)
    ce = jonswap_components(Hs=Hs, Tp=Tp, n_components=90, depth=DEPTH,
                            direction_spread_deg=SPREAD, seed=sd, dirs_per_freq=3)
    oa = np.unique([c.omega for c in ca]); oe = np.unique([c.omega for c in ce])
    print('n_uniq A/E:', oa.size, oe.size, ' arrays exactly equal:',
          bool(np.array_equal(oa, oe)), ' max|d|:', float(np.abs(oa-oe).max()) if oa.size==oe.size else 'n/a')
    ka = np.unique([c.k for c in ca]); ke = np.unique([c.k for c in ce])
    print('k arrays exactly equal:', bool(np.array_equal(ka, ke)))
    # per-bin energy
    Ea = np.array([0.5*c.A**2 for c in ca]); Ee = np.array([0.5*c.A**2 for c in ce])
    oma = np.array([c.omega for c in ca]); ome = np.array([c.omega for c in ce])
    pa = np.array([Ea[oma == o].sum() for o in oa]); pe = np.array([Ee[ome == o].sum() for o in oe])
    print('per-bin energy max rel diff:', float(np.abs(pa-pe).max()/pa.max()))
    print('per-atom A ratio (E/A) median:', float(np.median([c.A for c in ce])/np.median([c.A for c in ca])),
          ' 1/sqrt(3) =', 1/np.sqrt(3))
    print('A_rms ratio E/A:', float(np.sqrt(np.mean([c.A**2 for c in ce]))/np.sqrt(np.mean([c.A**2 for c in ca]))),
          ' 1/sqrt(3) =', 1/np.sqrt(3))
    print('Qw sum (Q*w*A) A/E:', check_q_constraint(ca)[1], check_q_constraint(ce)[1])
    print('lambda_min A/E:', 2*np.pi/max(c.k for c in ca), 2*np.pi/max(c.k for c in ce))

# --- per-atom amplitude vs N, at matched n_freq or not --------------------
print('\n--- per-atom A_rms  vs  N  (does it depend only on N?) ---')
for N, M in [(30,1),(90,3),(90,1),(90,9),(300,10),(60,2),(30,3),(10,1)]:
    c = jonswap_components(Hs=Hs, Tp=Tp, n_components=N, depth=DEPTH,
                           direction_spread_deg=SPREAD, seed=0, dirs_per_freq=M)
    A = np.array([x.A for x in c])
    print(f'  N={N:3d} M={M:2d} n_freq={N//M:3d}  A_rms={np.sqrt((A**2).mean()):.6f}  '
          f'A_rms*sqrt(N)={np.sqrt((A**2).mean())*np.sqrt(N):.6f}  A_max={A.max():.6f}')

json.dump({k: v for k, v in out.items()},
          open(os.path.join(ROOT, 'tmp/identifiability_debate_2026-08-05/CON_escape_check.json'), 'w'), indent=1)
