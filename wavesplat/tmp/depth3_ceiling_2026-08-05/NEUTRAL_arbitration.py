"""NEUTRAL depth-3 arbitration of Proposition F (the "ceiling" claim).

Question under arbitration:
  F: "our generator's one-direction-per-frequency-bin structure has a ceiling that no
      component count N can pass -- an architecture constraint, not a budget problem."

Two conflicting measurements exist:
  A (neutral, 08-04, tmp/audit_2026-08-04/cdip_discretize.json): real 028p1 records
    collapsed to one mean-direction per energy bin -> N=30 24.24 deg, all-bins 25.17 deg.
  B (con, 08-05, DEPTH2_VERDICT.md): 25,426 CDIP records discretized into N components
    "preserving self-spread" -> N=30 34.41 deg (97.3% of the continuous 35.35 deg).

This script does NOT pick a winner.  It establishes what each procedure can and cannot
be about, using only (i) the generator source, (ii) the already-fetched CDIP raw file,
(iii) direction-only replays of the generator's own component construction.
NO wave field is rendered.  NO episode is generated.  Nothing in paper/ is written.
"""
import json
import os
import sys

import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
sys.path.insert(0, ROOT)
from wavesim.wave_generator import jonswap_components  # noqa: E402

RAW = os.path.join(ROOT, 'tmp/cdip_support_2026-08-03/cdip_support_raw.json')
OUT = os.path.join(ROOT, 'tmp/depth3_ceiling_2026-08-05/NEUTRAL_arbitration.json')
FMIN, FMAX = 0.04, 0.50          # sea-swell band, both arms
DF_ANALYSIS = 0.005              # the analysis bandwidth the arm-identical estimator uses
SEA = {'BF2': (0.3, 4.0), 'BF3': (0.6, 5.5), 'BF4': (1.2, 7.0), 'BF5': (2.0, 9.0)}

res = {}


def sigma1_deg(a1, b1):
    r1 = np.sqrt(np.clip(a1 ** 2 + b1 ** 2, 0, 1))
    return np.degrees(np.sqrt(np.clip(2.0 * (1.0 - r1), 0, None)))


def pop_sigma1_deg(s_deg):
    """Population sigma_1 of a wrapped-Gaussian direction law with std s_deg.
    r1 = E[cos(theta-mu)] = exp(-s^2/2);  sigma_1 = sqrt(2(1-r1))."""
    s = np.deg2rad(s_deg)
    return float(np.degrees(np.sqrt(2 * (1 - np.exp(-s ** 2 / 2)))))


# ===================================================================== PART 1
# What the real sea puts WHERE.  The already-fetched file carries, per record,
#   spread_band_deg  = energy-weighted mean of sigma_1(f)   -> WITHIN-band
#   spread_total_deg = sigma_1 of the band-integrated a1/b1 -> TOTAL
# To leading order sigma_tot^2 ~= <sigma_within^2> + sigma_between^2, so the
# between-band term is recoverable.  Using (mean sigma_w)^2 in place of
# <sigma_w^2> makes the recovered between-term an UPPER BOUND.
d = json.load(open(RAW))
W = np.concatenate([np.asarray(r['spread_band_deg'], float) for r in d['real']])
T = np.concatenate([np.asarray(r['spread_total_deg'], float) for r in d['real']])
ok = np.isfinite(W) & np.isfinite(T)
W, T = W[ok], T[ok]
BET = np.sqrt(np.clip(T ** 2 - W ** 2, 0, None))     # upper bound on between-band

GEN_S = 30.0
gen_between = pop_sigma1_deg(GEN_S)                  # our generator's between-bin scatter
res['P1_real_decomposition'] = {
    'n_records': int(W.size),
    'median_total_deg': float(np.median(T)),
    'median_within_band_deg': float(np.median(W)),
    'median_between_band_deg_UPPERBOUND': float(np.median(BET)),
    'generator_between_bin_deg_at_s30_population': gen_between,
    'generator_within_band_deg_by_construction': 0.0,
    'frac_real_records_whose_between_band_is_BELOW_our_between_bin':
        float((BET < gen_between).mean()),
    'note': ('between-band term is an upper bound (Jensen); the fraction below is '
             'therefore a LOWER bound on how often we out-scatter the ocean '
             'on the between-bin axis'),
}

# ===================================================================== PART 2
# Does the bin COUNT grow with N, or is it fixed?  wave_generator.py:165
#   omegas = linspace(0.5*wp, 3*wp, n_components+1)   -> exactly N bins.
# So bin width shrinks as 1/N.  "one direction per bin" is therefore a
# RESOLUTION-RELATIVE statement, not a scale-free one.  Compute the N at which
# the generator's own bins become narrower than the fixed analysis band.
cross = {}
for name, (Hs, Tp) in SEA.items():
    f_lo, f_hi = 0.5 / Tp, 3.0 / Tp
    width = f_hi - f_lo
    cross[name] = {
        'Tp_s': Tp,
        'generator_band_hz': [round(f_lo, 4), round(f_hi, 4)],
        'generator_band_width_hz': round(width, 4),
        'df_generator_at_N30_hz': round(width / 30, 5),
        'analysis_df_hz': DF_ANALYSIS,
        'components_per_analysis_band_at_N30': round(DF_ANALYSIS / (width / 30), 3),
        'N_for_1_component_per_analysis_band': int(np.ceil(width / DF_ANALYSIS)),
        'N_for_10_components_per_analysis_band': int(np.ceil(10 * width / DF_ANALYSIS)),
    }
res['P2_bin_count_scales_with_N'] = {
    'source': 'wave_generator.py:165  linspace(omega_min, omega_max, n_components+1)',
    'n_bins': 'EQUALS n_components -- bin width shrinks as 1/N',
    'per_sea_state': cross,
}

# ===================================================================== PART 3
# Direction-only replay of the generator.  Measure, as a function of N:
#   (a) total sigma_1 over the sea-swell band  (what A and B both report)
#   (b) within-analysis-band sigma_1 at 0.005 Hz (what the architecture is said to forbid)
# Directions come from the generator's own rng call; no field is rendered.
def replay(N, s_deg, Tp, Hs, n_seed, stratified=False):
    tot, wit, occ = [], [], []
    for si in range(n_seed):
        c = jonswap_components(Hs=Hs, Tp=Tp, n_components=N, depth=50.0,
                               direction_spread_deg=s_deg, seed=20_000 + si)
        A = np.array([x.A for x in c])
        om = np.array([x.omega for x in c])
        th = np.arctan2([x.ky for x in c], [x.kx for x in c])
        if stratified:
            # same energies, directions replaced by the N-point stratified (quantile)
            # sample of the SAME N(mu, s) law -- isolates the sampling scheme from
            # the population.  Diagnostic only; not a proposed change.
            from scipy.special import erfinv
            u = (np.arange(N) + 0.5) / N
            th = np.deg2rad(s_deg) * np.sqrt(2) * erfinv(2 * u - 1)
            rng = np.random.default_rng(20_000 + si)
            th = rng.permutation(th)
        fr = om / (2 * np.pi)
        E = 0.5 * A ** 2
        b = (fr >= FMIN) & (fr <= FMAX)
        if not b.any() or E[b].sum() <= 0:
            continue
        e = E[b].sum()
        a1 = (E[b] * np.cos(th[b])).sum() / e
        b1 = (E[b] * np.sin(th[b])).sum() / e
        tot.append(float(sigma1_deg(np.array([a1]), np.array([b1]))[0]))
        # within-analysis-band, exactly as cdip_support.py:synth_records does it
        edges = np.arange(FMIN, FMAX + 1e-9, DF_ANALYSIS)
        idx = np.digitize(fr[b], edges) - 1
        sb, wb, nb = [], [], []
        for j in np.unique(idx):
            m = idx == j
            ee = E[b][m].sum()
            if ee <= 0:
                continue
            aa = (E[b][m] * np.cos(th[b][m])).sum() / ee
            bb = (E[b][m] * np.sin(th[b][m])).sum() / ee
            sb.append(float(sigma1_deg(np.array([aa]), np.array([bb]))[0]))
            wb.append(ee)
            nb.append(int(m.sum()))
        sb, wb = np.array(sb), np.array(wb)
        wit.append(float((sb * wb).sum() / wb.sum()))
        occ.append(float(np.average(nb, weights=wb)))
    return (float(np.median(tot)), float(np.median(wit)), float(np.median(occ)),
            len(tot))


ladder = {}
for N in (1, 2, 5, 15, 30, 71, 150, 400):
    ns = 200 if N <= 30 else (80 if N <= 150 else 40)
    t, w, o, n = replay(N, GEN_S, Tp=7.0, Hs=1.2, n_seed=ns)
    ladder[N] = {'n_draws': n, 'total_sigma1_deg': round(t, 3),
                 'within_analysis_band_sigma1_deg': round(w, 3),
                 'energy_wtd_components_per_occupied_band': round(o, 3)}
res['P3_generator_replay_BF4_s30'] = {
    'population_ceiling_deg': gen_between,
    'analysis_df_hz': DF_ANALYSIS,
    'ladder': ladder,
}

# ===================================================================== PART 4
# Is the ceiling a property of the ARCHITECTURE or of the PARAMETER?  Sweep s.
sweep = {}
for s in (10.0, 30.0, 37.2, 60.0, 90.0):
    t30, w30, _, _ = replay(30, s, Tp=7.0, Hs=1.2, n_seed=200)
    t400, w400, _, _ = replay(400, s, Tp=7.0, Hs=1.2, n_seed=25)
    sweep[s] = {'population_deg': round(pop_sigma1_deg(s), 3),
                'N30_total_deg': round(t30, 3), 'N30_within_deg': round(w30, 3),
                'N400_total_deg': round(t400, 3), 'N400_within_deg': round(w400, 3)}
res['P4_spread_parameter_sweep'] = {
    'sup_over_s_deg': float(np.degrees(np.sqrt(2.0))),
    'note': 'sigma_1 -> sqrt(2) rad = 81.03 deg as s -> inf; no architectural bound below that',
    'sweep': sweep,
}

# ===================================================================== PART 5
# Why do A and B disagree by ~10 deg at N=30?  Test whether it is the SAMPLING
# SCHEME (iid draw vs stratified/quantile placement) rather than the population.
strat = {}
for N in (2, 5, 15, 30):
    ti, wi, _, _ = replay(N, GEN_S, Tp=7.0, Hs=1.2, n_seed=200, stratified=False)
    ts, ws, _, _ = replay(N, GEN_S, Tp=7.0, Hs=1.2, n_seed=200, stratified=True)
    strat[N] = {'iid_total_deg': round(ti, 3), 'stratified_total_deg': round(ts, 3),
                'population_deg': round(gen_between, 3),
                'iid_recovery_pct': round(100 * ti / gen_between, 1),
                'stratified_recovery_pct': round(100 * ts / gen_between, 1)}
res['P5_sampling_scheme_vs_population'] = strat

# ===================================================================== PART 6
# Side-by-side of the three ladders, all at N=30, so the reader can see that the
# three numbers are three different estimands.
res['P6_three_ladders'] = {
    'A_real_028p1_delta_per_energy_bin': {1: 0.0, 2: 3.507, 5: 11.344, 15: 21.331,
                                          30: 24.238, 'all_bins': 25.173},
    'B_real_25426_self_spread_preserving': {1: 0.0, 2: 17.22, 5: 29.17, 30: 34.41},
    'OURS_measured_100ep_by_N': {1: 8.5e-07, 2: 5.150, 5: 17.523, 15: 17.291, 30: 24.179},
    'OURS_replay_400draws_by_N': {k: v['total_sigma1_deg'] for k, v in ladder.items()},
    'A_vs_OURS_gap_at_N30_deg': round(24.238 - 24.179, 3),
    'B_vs_OURS_gap_at_N30_deg': round(34.41 - 24.179, 3),
}

json.dump(res, open(OUT, 'w'), indent=2)
print(json.dumps(res, indent=2))
